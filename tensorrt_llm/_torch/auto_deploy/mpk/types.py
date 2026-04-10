# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Types for the Gemma4MoE MPK translator."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

SUPPORTED_SOURCE_OP_FAMILIES = (
    "auto_deploy::triton_paged_prepare_metadata",
    "auto_deploy::flashinfer_rms_norm",
    "auto_deploy::torch_linear_simple",
    "auto_deploy::flashinfer_rope",
    "auto_deploy::triton_paged_mha_with_cache",
    "auto_deploy::triton_fused_topk_softmax",
    "auto_deploy::trtllm_moe_fused",
    "auto_deploy::gather_tokens",
)

UNSUPPORTED_SOURCE_PATTERNS = (
    "mlir_fused_",
    "auto_deploy::mlir_",
)


class GemmaLayerSchema(str, Enum):
    """Gemma layer schema variants visible in the no-MLIR-fusion dumps."""

    REGULAR = "regular"
    FINAL = "final"


class GemmaLoweringStatus(str, Enum):
    """Status of a planned MPK lowering step."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    GAP = "gap"


@dataclass(frozen=True)
class GemmaNodeRef:
    """Stable node reference for analysis and plan emission."""

    name: str
    op: str
    target: str
    layer_index: Optional[int] = None


@dataclass
class GemmaLayerInfo:
    """Recovered per-layer structure from the FX graph."""

    layer_index: int
    schema: GemmaLayerSchema
    anchors: Dict[str, GemmaNodeRef] = field(default_factory=dict)
    hidden_size: Optional[int] = None
    q_heads: Optional[int] = None
    kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    router_top_k: Optional[int] = None
    moe_is_gated_mlp: Optional[bool] = None
    moe_act_fn: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "schema": self.schema.value,
            "anchors": {name: asdict(ref) for name, ref in self.anchors.items()},
            "hidden_size": self.hidden_size,
            "q_heads": self.q_heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "router_top_k": self.router_top_k,
            "moe_is_gated_mlp": self.moe_is_gated_mlp,
            "moe_act_fn": self.moe_act_fn,
        }


@dataclass
class GemmaBufferSpec:
    """Named buffer or attached tensor in the dry-run MPK plan."""

    name: str
    kind: str
    shape_expr: str
    dtype: str
    scope: str
    layer_index: Optional[int] = None
    source: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "shape_expr": self.shape_expr,
            "dtype": self.dtype,
            "scope": self.scope,
            "layer_index": self.layer_index,
            "source": self.source,
            "notes": list(self.notes),
        }


@dataclass
class GemmaBufferPlan:
    """Concrete dry-run view of graph-level and per-layer MPK buffers."""

    allocation_policy: str
    graph_inputs: List[GemmaBufferSpec] = field(default_factory=list)
    metadata_buffers: List[GemmaBufferSpec] = field(default_factory=list)
    cache_buffers: List[GemmaBufferSpec] = field(default_factory=list)
    graph_outputs: List[GemmaBufferSpec] = field(default_factory=list)
    layer_buffers: Dict[int, List[GemmaBufferSpec]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocation_policy": self.allocation_policy,
            "graph_inputs": [buf.to_dict() for buf in self.graph_inputs],
            "metadata_buffers": [buf.to_dict() for buf in self.metadata_buffers],
            "cache_buffers": [buf.to_dict() for buf in self.cache_buffers],
            "graph_outputs": [buf.to_dict() for buf in self.graph_outputs],
            "layer_buffers": {
                str(layer_idx): [buf.to_dict() for buf in buffers]
                for layer_idx, buffers in self.layer_buffers.items()
            },
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "allocation_policy": self.allocation_policy,
            "num_graph_inputs": len(self.graph_inputs),
            "num_metadata_buffers": len(self.metadata_buffers),
            "num_cache_buffers": len(self.cache_buffers),
            "num_graph_outputs": len(self.graph_outputs),
            "num_layer_buffers": sum(len(buffers) for buffers in self.layer_buffers.values()),
        }


@dataclass
class GemmaCanonicalOp:
    """Canonical semantic op emitted by the Gemma dry-run lowering plan."""

    name: str
    inputs: List[str]
    outputs: List[str]
    anchors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "anchors": list(self.anchors),
            "notes": list(self.notes),
        }


@dataclass
class GemmaMpkStep:
    """Single planned MPK method emission step."""

    name: str
    mpk_method: Optional[str]
    inputs: List[str]
    outputs: List[str]
    status: GemmaLoweringStatus
    params: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mpk_method": self.mpk_method,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "status": self.status.value,
            "params": dict(self.params),
            "notes": list(self.notes),
        }


@dataclass
class GemmaLayerLoweringPlan:
    """Canonical and MPK-lowering plan for one Gemma layer."""

    layer_index: int
    schema: GemmaLayerSchema
    input_buffer: str
    output_buffer: str
    canonical_ops: List[GemmaCanonicalOp] = field(default_factory=list)
    mpk_steps: List[GemmaMpkStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "schema": self.schema.value,
            "input_buffer": self.input_buffer,
            "output_buffer": self.output_buffer,
            "canonical_ops": [op.to_dict() for op in self.canonical_ops],
            "mpk_steps": [step.to_dict() for step in self.mpk_steps],
            "notes": list(self.notes),
        }

    def summary(self) -> Dict[str, Any]:
        num_gaps = sum(step.status == GemmaLoweringStatus.GAP for step in self.mpk_steps)
        num_partial = sum(step.status == GemmaLoweringStatus.PARTIAL for step in self.mpk_steps)
        return {
            "layer_index": self.layer_index,
            "schema": self.schema.value,
            "num_canonical_ops": len(self.canonical_ops),
            "num_mpk_steps": len(self.mpk_steps),
            "num_gaps": num_gaps,
            "num_partial": num_partial,
        }


@dataclass
class GemmaGraphInfo:
    """Graph-level analysis result for Gemma4MoE decode."""

    graph_name: str
    supported_source_ops: List[str]
    placeholder_names: List[str]
    cache_placeholder_names: List[str]
    metadata_prep: GemmaNodeRef
    metadata_outputs: List[GemmaNodeRef]
    layer_infos: List[GemmaLayerInfo]
    final_tail: Dict[str, Optional[GemmaNodeRef]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "supported_source_ops": list(self.supported_source_ops),
            "placeholder_names": list(self.placeholder_names),
            "cache_placeholder_names": list(self.cache_placeholder_names),
            "metadata_prep": asdict(self.metadata_prep),
            "metadata_outputs": [asdict(ref) for ref in self.metadata_outputs],
            "layer_infos": [layer.to_dict() for layer in self.layer_infos],
            "final_tail": {
                name: asdict(ref) if ref is not None else None
                for name, ref in self.final_tail.items()
            },
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "graph_name": self.graph_name,
            "num_layers": len(self.layer_infos),
            "layer_indices": [layer.layer_index for layer in self.layer_infos],
            "schemas": {layer.layer_index: layer.schema.value for layer in self.layer_infos},
            "metadata_prep": self.metadata_prep.name,
            "num_cache_placeholders": len(self.cache_placeholder_names),
        }


@dataclass
class GemmaMpkTranslationPlan:
    """Current translation plan emitted by the MPK translator.

    The initial implementation is intentionally analysis-first.  It captures the
    supported source contract and the recovered Gemma graph structure.  Later
    phases can extend this with buffer plans and concrete MPK lowering steps.
    """

    graph_info: GemmaGraphInfo
    buffer_plan: Optional[GemmaBufferPlan] = None
    layer_lowerings: List[GemmaLayerLoweringPlan] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_info": self.graph_info.to_dict(),
            "buffer_plan": self.buffer_plan.to_dict() if self.buffer_plan is not None else None,
            "layer_lowerings": [layer.to_dict() for layer in self.layer_lowerings],
            "notes": list(self.notes),
        }

    def summary(self) -> Dict[str, Any]:
        num_gaps = 0
        num_partial = 0
        for layer in self.layer_lowerings:
            layer_summary = layer.summary()
            num_gaps += layer_summary["num_gaps"]
            num_partial += layer_summary["num_partial"]
        return {
            "graph": self.graph_info.summary(),
            "buffer_plan": self.buffer_plan.summary() if self.buffer_plan is not None else None,
            "num_layer_lowerings": len(self.layer_lowerings),
            "num_gap_steps": num_gaps,
            "num_partial_steps": num_partial,
        }
