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

"""Runtime wrapper scaffold for future AutoDeploy -> MPK integration."""

from __future__ import annotations

import copy
import os
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
from torch.fx import GraphModule

from ..custom_ops.attention_interface import BatchInfo


class GemmaMpkRuntimeWrapper(nn.Module):
    """Wrapper that owns the Gemma MPK runtime boundary.

    The wrapper is intentionally policy-aware:
    - generate-only batches use the Mirage-backed runtime
    - non-generate-only batches use the original AutoDeploy GraphModule

    This keeps Mirage focused on decode-only acceleration while preserving the
    existing AutoDeploy path for prefill and mixed steps.
    """

    def __init__(
        self,
        *,
        mpk_callable: Optional[Callable[..., Any]] = None,
        original_model: Optional[nn.Module] = None,
        translation_plan: Optional[Dict[str, Any]] = None,
        input_names: Optional[Sequence[str]] = None,
        batch_info_input_name: str = "batch_info_host",
    ) -> None:
        super().__init__()
        self.mpk_callable = mpk_callable
        self.original_model = original_model
        self.translation_plan = translation_plan or {}
        self.input_names = tuple(input_names or ())
        self.batch_info_input_name = batch_info_input_name
        self._compare_decode_logits_once_done = False

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, dict):
            return model_output["logits"]
        if isinstance(model_output, (tuple, list)):
            return model_output[0]
        return model_output

    def _clone_bound_inputs_for_reference(self, bound_inputs: Dict[str, Any]) -> Dict[str, Any]:
        cloned_inputs: Dict[str, Any] = {}
        for name, value in bound_inputs.items():
            if torch.is_tensor(value):
                cloned_inputs[name] = value.clone()
            else:
                cloned_inputs[name] = value
        return cloned_inputs

    def _build_intermediate_graphmodule(
        self,
        node_name_map: Dict[str, str | Sequence[str]],
    ) -> Optional[GraphModule]:
        if self.original_model is None:
            return None

        gm = copy.deepcopy(self.original_model)
        original_graph = gm.graph
        original_nodes = {node.name: node for node in original_graph.nodes}
        target_nodes = {}
        for alias, node_name_or_candidates in node_name_map.items():
            candidates = (
                tuple(node_name_or_candidates)
                if isinstance(node_name_or_candidates, (tuple, list))
                else (node_name_or_candidates,)
            )
            node = None
            for node_name in candidates:
                node = original_nodes.get(node_name)
                if node is not None:
                    break
            if node is None:
                return None
            target_nodes[alias] = node

        new_graph = torch.fx.Graph()
        env: Dict[torch.fx.Node, torch.fx.Node] = {}
        for node in original_graph.nodes:
            if node.op == "output":
                new_graph.output({alias: env[target] for alias, target in target_nodes.items()})
                break
            env[node] = new_graph.node_copy(node, lambda n: env[n])

        intermediate_gm = GraphModule(gm, new_graph)
        intermediate_gm.recompile()
        return intermediate_gm

    def _maybe_compare_layer0_debug_tensors(
        self,
        *,
        mpk_output: Any,
        reference_inputs: Dict[str, Any],
    ) -> None:
        if os.getenv("AD_MPK_COMPARE_LAYER0_DEBUG", "0") != "1":
            return
        if self.original_model is None or not isinstance(mpk_output, dict):
            return

        debug_tensors = mpk_output.get("_debug_tensors")
        if not isinstance(debug_tensors, dict):
            return

        node_name_map = {
            "layer_0_qkv_packed": "torch_linear_simple_default",
            "layer_0_q_norm": "flashinfer_rms_norm_1",
            "layer_0_k_norm": "flashinfer_rms_norm_2",
            "layer_0_v_norm": "flashinfer_rms_norm_3",
            "layer_0_q_rope": ("getitem_180", "getitem_30"),
            "layer_0_k_rope": ("getitem_181", "getitem_31"),
            "layer_0_attn_out": "model_language_model_layers_0_self_attn_reshape",
            "layer_0_o_proj": "model_language_model_layers_0_self_attn_o_proj_torch_linear_simple_3",
            "layer_0_post_attn": "model_language_model_layers_0_add",
            "layer_0_ffn_down": "model_language_model_layers_0_mlp_down_proj_torch_linear_simple_6",
            "layer_0_ffn_norm": "flashinfer_rms_norm_6",
            "layer_0_router_in": "model_language_model_layers_0_router_mul_4",
            "layer_0_router_logits": "model_language_model_layers_0_router_proj_torch_linear_simple_7",
            "layer_0_router_topk_weight": ("getitem_20", "getitem_120"),
            "layer_0_router_topk_indices": ("getitem_21", "getitem_121"),
            "layer_0_moe_in": "flashinfer_rms_norm_7",
            "layer_0_moe_out": "model_language_model_layers_0_reshape_2",
            "layer_0_moe_norm": "flashinfer_rms_norm_8",
            "layer_0_ffn_moe_add": "model_language_model_layers_0_add_2",
            "layer_0_ffn_moe_norm": "flashinfer_rms_norm_9",
            "layer_0_hidden": "model_language_model_layers_0_mul_5",
        }
        intermediate_gm = self._build_intermediate_graphmodule(node_name_map)
        if intermediate_gm is None:
            print("[AD_MPK_COMPARE_LAYER0] intermediate graph build skipped", flush=True)
            return

        ref_outputs = intermediate_gm(**reference_inputs)
        for alias, ref_tensor in ref_outputs.items():
            mpk_tensor = debug_tensors.get(alias)
            if mpk_tensor is None:
                continue
            diff = (mpk_tensor.float() - ref_tensor.float()).abs()
            print(
                "[AD_MPK_COMPARE_LAYER0] "
                f"{alias} shape={tuple(ref_tensor.shape)} "
                f"max_abs={float(diff.max().item()):.6f} "
                f"mean_abs={float(diff.mean().item()):.6f}",
                flush=True,
            )

    def _maybe_compare_decode_logits(
        self,
        *,
        mpk_output: Any,
        reference_inputs: Dict[str, Any],
    ) -> None:
        if os.getenv("AD_MPK_COMPARE_DECODE_LOGITS", "0") != "1":
            return
        if self._compare_decode_logits_once_done:
            return
        if self.original_model is None:
            return

        ref_output = self.original_model(**reference_inputs)
        mpk_logits = self._extract_logits(mpk_output).float()
        ref_logits = self._extract_logits(ref_output).float()
        logits_abs = (mpk_logits - ref_logits).abs()
        topk = min(5, int(mpk_logits.shape[-1]))
        mpk_topk = torch.topk(mpk_logits.reshape(-1, mpk_logits.shape[-1]), k=topk, dim=-1).indices
        ref_topk = torch.topk(ref_logits.reshape(-1, ref_logits.shape[-1]), k=topk, dim=-1).indices
        top1_match = bool(torch.equal(mpk_topk[:, :1], ref_topk[:, :1]))
        top5_overlap = int(
            sum(
                len(set(mpk_row.tolist()) & set(ref_row.tolist()))
                for mpk_row, ref_row in zip(mpk_topk, ref_topk)
            )
        )
        print(
            "[AD_MPK_COMPARE] "
            f"logits_shape={tuple(mpk_logits.shape)} "
            f"max_abs={float(logits_abs.max().item()):.6f} "
            f"mean_abs={float(logits_abs.mean().item()):.6f} "
            f"top1_match={top1_match} "
            f"top{topk}_overlap={top5_overlap}",
            flush=True,
        )
        self._compare_decode_logits_once_done = True

    def _bind_inputs(self, args, kwargs) -> Dict[str, Any]:
        if len(args) > len(self.input_names):
            raise RuntimeError(
                "GemmaMpkRuntimeWrapper received more positional inputs than placeholder names."
            )

        bound_inputs = dict(kwargs)
        for name, value in zip(self.input_names, args):
            if name in bound_inputs:
                raise RuntimeError(f"GemmaMpkRuntimeWrapper received duplicate input for '{name}'.")
            bound_inputs[name] = value
        return bound_inputs

    def _is_generate_only(self, bound_inputs: Dict[str, Any]) -> bool:
        batch_info_host = bound_inputs.get(self.batch_info_input_name)
        if batch_info_host is None:
            raise RuntimeError(
                "GemmaMpkRuntimeWrapper requires 'batch_info_host' to decide whether "
                "to route to Mirage decode or the original AutoDeploy graph."
            )
        return BatchInfo(batch_info_host).is_generate_only()

    def forward(self, *args, **kwargs):
        bound_inputs = self._bind_inputs(args, kwargs)
        if self._is_generate_only(bound_inputs):
            if self.mpk_callable is not None:
                reference_inputs = None
                if (
                    os.getenv("AD_MPK_COMPARE_DECODE_LOGITS", "0") == "1"
                    and not self._compare_decode_logits_once_done
                    and self.original_model is not None
                ):
                    reference_inputs = self._clone_bound_inputs_for_reference(bound_inputs)
                mpk_output = self.mpk_callable(*args, **kwargs)
                if reference_inputs is not None:
                    self._maybe_compare_decode_logits(
                        mpk_output=mpk_output, reference_inputs=reference_inputs
                    )
                    self._maybe_compare_layer0_debug_tensors(
                        mpk_output=mpk_output, reference_inputs=reference_inputs
                    )
                return mpk_output
            raise RuntimeError(
                "GemmaMpkRuntimeWrapper was invoked on a generate-only batch without a live MPK "
                "callable. The Gemma MPK path no longer supports eager fallback for decode."
            )

        if self.original_model is not None:
            return self.original_model(**bound_inputs)

        raise RuntimeError(
            "GemmaMpkRuntimeWrapper received a non-generate-only batch without an original "
            "AutoDeploy model to execute."
        )

    def extra_repr(self) -> str:
        mode = "mpk" if self.mpk_callable is not None else "missing_mpk_callable"
        has_plan = bool(self.translation_plan)
        original = self.original_model is not None
        return (
            f"mode={mode}, has_plan={has_plan}, has_original_model={original}, "
            f"num_inputs={len(self.input_names)}"
        )
