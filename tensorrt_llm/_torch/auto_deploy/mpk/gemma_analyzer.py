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

"""Gemma4MoE graph analyzer for the initial MPK translator."""

from __future__ import annotations

import operator
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set

from torch.fx import GraphModule, Node

from ..utils.node_utils import extract_op_args
from .types import (
    SUPPORTED_SOURCE_OP_FAMILIES,
    UNSUPPORTED_SOURCE_PATTERNS,
    GemmaGraphInfo,
    GemmaLayerInfo,
    GemmaLayerSchema,
    GemmaNodeRef,
)

_LAYER_RE = re.compile(r"model_language_model_layers_(\d+)_")
_CACHE_PLACEHOLDER_RE = re.compile(r"r(\d+)_kv_cache$")

_PASSTHROUGH_CALL_METHODS = {
    "view",
    "reshape",
    "permute",
    "transpose",
    "contiguous",
    "to",
    "type_as",
    "expand",
    "unsqueeze",
    "squeeze",
}
_PASSTHROUGH_CALL_FUNCTION_TARGETS = {
    "flashinfer_rope",
    "split_output",
}

_RMS_NORM_TARGETS = ("flashinfer_rms_norm", "torch_rmsnorm")


def _iter_node_inputs(node: Node) -> Iterable[Node]:
    for arg in node.args:
        if isinstance(arg, Node):
            yield arg
        elif isinstance(arg, (tuple, list)):
            for item in arg:
                if isinstance(item, Node):
                    yield item
    for arg in node.kwargs.values():
        if isinstance(arg, Node):
            yield arg
        elif isinstance(arg, (tuple, list)):
            for item in arg:
                if isinstance(item, Node):
                    yield item


def _qualified_target_name(node: Node) -> str:
    target = node.target
    if hasattr(target, "name"):
        return target.name()
    if hasattr(target, "__qualname__"):
        return target.__qualname__
    return str(target)


def _is_rms_norm_node(node: Node) -> bool:
    return node.op == "call_function" and any(
        target in _qualified_target_name(node) for target in _RMS_NORM_TARGETS
    )


def _extract_layer_idx_from_text(text: str) -> Optional[int]:
    match = _LAYER_RE.search(text)
    return int(match.group(1)) if match else None


def _extract_layer_idx_from_node(node: Node) -> Optional[int]:
    candidates = [node.name, _qualified_target_name(node)]
    if isinstance(node.target, str):
        candidates.append(node.target)
    for candidate in candidates:
        layer_idx = _extract_layer_idx_from_text(candidate)
        if layer_idx is not None:
            return layer_idx
    return None


def _node_ref(node: Node, layer_index: Optional[int] = None) -> GemmaNodeRef:
    return GemmaNodeRef(
        name=node.name,
        op=node.op,
        target=_qualified_target_name(node),
        layer_index=layer_index,
    )


def _find_single_op(gm: GraphModule, op_substr: str) -> Node:
    matches = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and op_substr in _qualified_target_name(node)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one '{op_substr}' node, found {len(matches)}.")
    return matches[0]


def _find_ancestor(
    node: Node,
    predicate,
    *,
    stop_nodes: Optional[Set[Node]] = None,
    max_depth: int = 32,
    _seen: Optional[Set[Node]] = None,
) -> Optional[Node]:
    if max_depth < 0:
        return None
    if _seen is None:
        _seen = set()
    if node in _seen:
        return None
    _seen.add(node)

    for input_node in _iter_node_inputs(node):
        if stop_nodes and input_node in stop_nodes:
            continue
        if predicate(input_node):
            return input_node

        is_passthrough = (
            input_node.op == "call_method" and input_node.target in _PASSTHROUGH_CALL_METHODS
        ) or (
            input_node.op == "call_function"
            and (
                input_node.target is operator.getitem
                or any(
                    target in _qualified_target_name(input_node)
                    for target in _PASSTHROUGH_CALL_FUNCTION_TARGETS
                )
            )
        )

        if is_passthrough:
            found = _find_ancestor(
                input_node,
                predicate,
                stop_nodes=stop_nodes,
                max_depth=max_depth - 1,
                _seen=_seen,
            )
            if found is not None:
                return found
    return None


def _extract_shape_last_dim(node: Node) -> Optional[int]:
    meta_val = node.meta.get("val") if hasattr(node, "meta") else None
    shape = getattr(meta_val, "shape", None)
    if not shape:
        return None
    try:
        return int(shape[-1])
    except Exception:
        return None


class GemmaGraphAnalyzer:
    """Analyzer for the supported Gemma4MoE no-MLIR-fusion graph surface."""

    def __init__(self, *, require_no_mlir_fusion: bool = True):
        self.require_no_mlir_fusion = require_no_mlir_fusion

    def _find_ancestor(
        self,
        node: Node,
        predicate,
        *,
        stop_nodes: Optional[Set[Node]] = None,
        max_depth: int = 32,
    ) -> Optional[Node]:
        """Instance wrapper around the shared recursive ancestor search."""

        return _find_ancestor(
            node,
            predicate,
            stop_nodes=stop_nodes,
            max_depth=max_depth,
        )

    def analyze(self, gm: GraphModule) -> GemmaGraphInfo:
        self._validate_supported_surface(gm)

        placeholder_names = [node.target for node in gm.graph.nodes if node.op == "placeholder"]
        cache_placeholder_names = [
            name
            for name in placeholder_names
            if isinstance(name, str) and _CACHE_PLACEHOLDER_RE.match(name)
        ]

        metadata_prep = _find_single_op(gm, "triton_paged_prepare_metadata")
        metadata_outputs = self._collect_getitem_users(metadata_prep)
        layers = self._collect_layers(gm)
        final_tail = self._collect_final_tail(gm)

        return GemmaGraphInfo(
            graph_name=getattr(gm, "_get_name", lambda: type(gm).__name__)(),
            supported_source_ops=list(SUPPORTED_SOURCE_OP_FAMILIES),
            placeholder_names=[str(name) for name in placeholder_names],
            cache_placeholder_names=cache_placeholder_names,
            metadata_prep=_node_ref(metadata_prep),
            metadata_outputs=[_node_ref(node) for node in metadata_outputs],
            layer_infos=layers,
            final_tail=final_tail,
        )

    def _validate_supported_surface(self, gm: GraphModule) -> None:
        unsupported_matches: List[str] = []
        for node in gm.graph.nodes:
            node_text = f"{node.name}:{_qualified_target_name(node)}"
            if self.require_no_mlir_fusion and any(
                pattern in node_text for pattern in UNSUPPORTED_SOURCE_PATTERNS
            ):
                unsupported_matches.append(node_text)

        if unsupported_matches:
            examples = ", ".join(unsupported_matches[:5])
            raise ValueError(
                "Gemma MPK analyzer only supports the no-MLIR-fusion graph surface. "
                f"Found unsupported nodes: {examples}"
            )

    def _collect_getitem_users(self, source_node: Node) -> List[Node]:
        outputs: List[Node] = []
        for user in source_node.users:
            if user.op == "call_function" and user.target is operator.getitem:
                outputs.append(user)
        outputs.sort(key=lambda node: node.name)
        return outputs

    def _collect_layers(self, gm: GraphModule) -> List[GemmaLayerInfo]:
        attention_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and "triton_paged_mha_with_cache" in _qualified_target_name(node)
        ]
        moe_nodes = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function" and "trtllm_moe_fused" in _qualified_target_name(node)
        ]

        if len(attention_nodes) != len(moe_nodes):
            raise ValueError(
                "Expected one cached-attention node and one MoE node per layer, got "
                f"{len(attention_nodes)} attention nodes and {len(moe_nodes)} MoE nodes."
            )

        max_layer_idx = len(attention_nodes) - 1
        layers: List[GemmaLayerInfo] = []
        for attention_node in attention_nodes:
            layer_idx = self._extract_layer_idx_from_attention_node(attention_node)
            layer_info = self._analyze_layer(attention_node, layer_idx, max_layer_idx)
            layers.append(layer_info)

        layers.sort(key=lambda layer: layer.layer_index)
        expected = list(range(len(layers)))
        actual = [layer.layer_index for layer in layers]
        if actual != expected:
            raise ValueError(f"Expected contiguous layer indices {expected}, got {actual}.")
        return layers

    def _extract_layer_idx_from_attention_node(self, attention_node: Node) -> int:
        kv_cache = extract_op_args(attention_node, "kv_cache")[0]
        if not isinstance(kv_cache, Node) or kv_cache.op != "placeholder":
            raise ValueError(
                f"Attention node {attention_node.name} is missing a KV cache placeholder."
            )
        match = _CACHE_PLACEHOLDER_RE.match(str(kv_cache.target))
        if match is None:
            raise ValueError(
                f"Attention node {attention_node.name} has unsupported KV cache target {kv_cache.target}."
            )
        return int(match.group(1))

    def _analyze_layer(
        self,
        attention_node: Node,
        layer_idx: int,
        max_layer_idx: int,
    ) -> GemmaLayerInfo:
        schema = GemmaLayerSchema.FINAL if layer_idx == max_layer_idx else GemmaLayerSchema.REGULAR
        layer_info = GemmaLayerInfo(layer_index=layer_idx, schema=schema)
        layer_info.anchors["cached_attention"] = _node_ref(attention_node, layer_idx)

        q_input, k_input, v_input, kv_cache = extract_op_args(
            attention_node, "q", "k", "v", "kv_cache"
        )
        for name, node in (("q_norm", q_input), ("k_norm", k_input), ("v_norm", v_input)):
            if not isinstance(node, Node):
                raise ValueError(f"Layer {layer_idx} missing expected node for {name}.")
            source = self._find_rms_norm_source(node, layer_idx)
            layer_info.anchors[name] = _node_ref(source, layer_idx)

        rope_node = self._find_rope_source(q_input, layer_idx)
        layer_info.anchors["rope"] = _node_ref(rope_node, layer_idx)

        qkv_linear = self._find_linear_source(
            layer_info.anchors["q_norm"].name, gm_node=attention_node
        )
        layer_info.anchors["qkv_linear"] = _node_ref(qkv_linear, layer_idx)

        o_proj = self._find_first_user_with_target(attention_node, "torch_linear_simple")
        if o_proj is None:
            raise ValueError(f"Layer {layer_idx} missing attention output projection.")
        layer_info.anchors["o_proj"] = _node_ref(o_proj, layer_idx)

        moe_node = self._find_moe_for_layer(attention_node.graph.nodes, layer_idx)
        layer_info.anchors["moe_fused"] = _node_ref(moe_node, layer_idx)

        topk_node = self._find_ancestor(
            moe_node,
            lambda node: node.op == "call_function"
            and "triton_fused_topk_softmax" in _qualified_target_name(node),
        )
        if topk_node is None:
            raise ValueError(f"Layer {layer_idx} missing triton_fused_topk_softmax.")
        layer_info.anchors["topk"] = _node_ref(topk_node, layer_idx)

        router_proj = self._find_ancestor(
            topk_node,
            lambda node: node.op == "call_function"
            and "torch_linear_simple" in _qualified_target_name(node),
        )
        if router_proj is None:
            raise ValueError(f"Layer {layer_idx} missing router projection.")
        layer_info.anchors["router_proj"] = _node_ref(router_proj, layer_idx)

        ffn_down = self._find_unique_layer_node(
            attention_node.graph.nodes,
            layer_idx,
            name_substr="mlp_down_proj",
            target_substr="torch_linear_simple",
        )
        if ffn_down is None:
            raise ValueError(f"Layer {layer_idx} missing FFN down projection.")
        layer_info.anchors["ffn_down"] = _node_ref(ffn_down, layer_idx)

        ffn_gate_up = self._find_upstream_call_function(
            ffn_down,
            "torch_linear_simple",
            exclude={ffn_down},
        )
        if ffn_gate_up is None:
            raise ValueError(f"Layer {layer_idx} missing FFN gate/up projection.")
        layer_info.anchors["ffn_gate_up"] = _node_ref(ffn_gate_up, layer_idx)

        q_meta = (
            getattr(q_input, "meta", {}).get("val", None) if isinstance(q_input, Node) else None
        )
        k_meta = (
            getattr(k_input, "meta", {}).get("val", None) if isinstance(k_input, Node) else None
        )
        if q_meta is not None and len(getattr(q_meta, "shape", ())) >= 4:
            layer_info.q_heads = int(q_meta.shape[2])
            layer_info.head_dim = int(q_meta.shape[3])
        if k_meta is not None and len(getattr(k_meta, "shape", ())) >= 4:
            layer_info.kv_heads = int(k_meta.shape[2])

        output_dim = _extract_shape_last_dim(o_proj)
        if output_dim is not None:
            layer_info.hidden_size = output_dim

        try:
            layer_info.router_top_k = int(extract_op_args(topk_node, "k")[0])
        except Exception:
            layer_info.router_top_k = None

        try:
            layer_info.moe_is_gated_mlp = bool(extract_op_args(moe_node, "is_gated_mlp")[0])
        except Exception:
            layer_info.moe_is_gated_mlp = None

        try:
            act_fn = extract_op_args(moe_node, "act_fn")[0]
            layer_info.moe_act_fn = int(act_fn) if act_fn is not None else None
        except Exception:
            layer_info.moe_act_fn = None

        if isinstance(kv_cache, Node):
            layer_info.anchors["kv_cache"] = _node_ref(kv_cache, layer_idx)

        return layer_info

    def _find_rms_norm_source(self, node: Node, layer_idx: int) -> Node:
        # q/k arrive at cached attention through getitem(flashinfer_rope, idx).
        # A generic ancestor walk will hit q_norm first for both branches, so
        # resolve the rope output index explicitly when possible.
        if node.op == "call_function" and node.target is operator.getitem and len(node.args) >= 2:
            rope_node = node.args[0]
            rope_output_idx = node.args[1]
            if (
                isinstance(rope_node, Node)
                and rope_node.op == "call_function"
                and "flashinfer_rope" in _qualified_target_name(rope_node)
                and rope_output_idx in (0, 1)
            ):
                rope_input = rope_node.args[int(rope_output_idx)]
                if isinstance(rope_input, Node):
                    if _is_rms_norm_node(rope_input):
                        return rope_input
                    rms_node = self._find_ancestor(
                        rope_input,
                        lambda candidate: _is_rms_norm_node(candidate),
                    )
                    if rms_node is not None:
                        return rms_node

        if _is_rms_norm_node(node):
            return node

        rms_node = self._find_ancestor(
            node,
            lambda candidate: _is_rms_norm_node(candidate)
            and _extract_layer_idx_from_node(candidate) == layer_idx,
        )
        if rms_node is not None:
            return rms_node

        # Some late compile-stage Gemma graphs preserve the correct local
        # RMSNorm ancestry but drop layer-identifying text from the norm node
        # names themselves (for example: flashinfer_rms_norm_1). In that case
        # we still want the nearest RMSNorm on the q/k/v path.
        rms_node = self._find_ancestor(
            node,
            lambda candidate: _is_rms_norm_node(candidate),
        )
        if rms_node is None:
            raise ValueError(f"Layer {layer_idx} missing RMSNorm ancestor for node {node.name}.")
        return rms_node

    def _find_rope_source(self, node: Node, layer_idx: int) -> Node:
        rope_node = self._find_ancestor(
            node,
            lambda candidate: candidate.op == "call_function"
            and "flashinfer_rope" in _qualified_target_name(candidate),
        )
        if rope_node is None:
            raise ValueError(f"Layer {layer_idx} missing flashinfer_rope for node {node.name}.")
        return rope_node

    def _find_linear_source(self, source_name: str, *, gm_node: Node) -> Node:
        source_node = next(node for node in gm_node.graph.nodes if node.name == source_name)
        queue: List[Node] = [source_node]
        seen: Set[Node] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            for input_node in _iter_node_inputs(current):
                if input_node in seen:
                    continue
                if (
                    input_node.op == "call_function"
                    and "torch_linear_simple" in _qualified_target_name(input_node)
                ):
                    return input_node
                queue.append(input_node)

        raise ValueError(f"Could not find torch_linear_simple ancestor for node {source_name}.")

    def _find_first_user_with_target(self, node: Node, target_substr: str) -> Optional[Node]:
        queue: List[Node] = list(node.users)
        seen: Set[Node] = set()
        while queue:
            user = queue.pop(0)
            if user in seen:
                continue
            seen.add(user)
            if user.op == "call_function" and target_substr in _qualified_target_name(user):
                return user
            queue.extend(user.users)
        return None

    def _find_unique_layer_node(
        self,
        nodes: Sequence[Node],
        layer_idx: int,
        *,
        name_substr: str,
        target_substr: str,
    ) -> Optional[Node]:
        matches = [
            node
            for node in nodes
            if node.op == "call_function"
            and name_substr in node.name
            and f"model_language_model_layers_{layer_idx}_" in node.name
            and target_substr in _qualified_target_name(node)
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _find_upstream_call_function(
        self,
        start_node: Node,
        target_substr: str,
        *,
        exclude: Optional[Set[Node]] = None,
    ) -> Optional[Node]:
        queue: List[Node] = [start_node]
        seen: Set[Node] = set()
        excluded = exclude or set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            for input_node in _iter_node_inputs(current):
                if input_node in seen or input_node in excluded:
                    continue
                if input_node.op == "call_function" and target_substr in _qualified_target_name(
                    input_node
                ):
                    return input_node
                queue.append(input_node)
        return None

    def _find_moe_for_layer(self, nodes: Sequence[Node], layer_idx: int) -> Node:
        matches: List[Node] = []
        for node in nodes:
            if node.op != "call_function" or "trtllm_moe_fused" not in _qualified_target_name(node):
                continue
            for input_node in _iter_node_inputs(node):
                node_layer_idx = _extract_layer_idx_from_node(input_node)
                if node_layer_idx == layer_idx:
                    matches.append(node)
                    break
                if input_node.op == "get_attr" and str(input_node.target).endswith(f"_{layer_idx}"):
                    matches.append(node)
                    break
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) != 1:
            raise ValueError(
                f"Expected exactly one MoE node for layer {layer_idx}, found {len(unique_matches)}."
            )
        return unique_matches[0]

    def _collect_final_tail(self, gm: GraphModule) -> Dict[str, Optional[GemmaNodeRef]]:
        tail: Dict[str, Optional[GemmaNodeRef]] = {
            "gather_tokens": None,
        }
        for node in gm.graph.nodes:
            if node.op == "call_function" and "gather_tokens" in _qualified_target_name(node):
                tail["gather_tokens"] = _node_ref(node)
                break
        return tail
