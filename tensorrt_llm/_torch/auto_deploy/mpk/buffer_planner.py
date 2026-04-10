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

"""Dry-run MPK buffer planning for Gemma4MoE decode graphs."""

from __future__ import annotations

from .types import GemmaBufferPlan, GemmaBufferSpec, GemmaGraphInfo, GemmaLayerInfo

_GRAPH_INPUT_KINDS = {
    "input_ids": ("tokens", "[batch, step_tokens]", "int32"),
    "position_ids": ("positions", "[batch, step_tokens]", "int64"),
    "token_gather_indices": ("gather_indices", "[num_selected_tokens]", "int32"),
    "batch_info_host": ("batch_metadata_host", "[6]", "int32"),
    "cu_seqlen_host": ("sequence_offsets_host", "[batch + 1]", "int32"),
    "cu_num_pages": ("page_counts", "[batch]", "int32"),
    "cu_num_pages_host": ("page_counts_host", "[batch]", "int32"),
    "cache_loc": ("cache_locations", "[num_tokens]", "int32"),
    "last_page_len": ("last_page_lengths", "[batch]", "int32"),
    "last_page_len_host": ("last_page_lengths_host", "[batch]", "int32"),
    "seq_len_with_cache_host": ("seq_len_with_cache_host", "[batch]", "int32"),
    "cu_seqlen": ("sequence_offsets", "[batch + 1]", "int32"),
    "seq_len_with_cache": ("seq_len_with_cache", "[batch]", "int32"),
}


class GemmaBufferPlanner:
    """Create a conservative, per-layer-dedicated dry-run MPK buffer plan."""

    def build(self, graph_info: GemmaGraphInfo) -> GemmaBufferPlan:
        plan = GemmaBufferPlan(allocation_policy="conservative_per_layer")

        for placeholder_name in graph_info.placeholder_names:
            if placeholder_name in graph_info.cache_placeholder_names:
                continue
            plan.graph_inputs.append(self._build_graph_input(placeholder_name))

        for metadata_idx, metadata_ref in enumerate(graph_info.metadata_outputs):
            plan.metadata_buffers.append(
                GemmaBufferSpec(
                    name=f"paged_metadata_{metadata_idx}",
                    kind="metadata",
                    shape_expr="[dynamic]",
                    dtype="unknown",
                    scope="graph",
                    source=metadata_ref.name,
                    notes=[
                        "Output of auto_deploy::triton_paged_prepare_metadata.",
                        "Semantic role is intentionally left generic until runtime contract is finalized.",
                    ],
                )
            )

        for layer_info in graph_info.layer_infos:
            plan.cache_buffers.append(self._build_cache_buffer(layer_info))
            plan.layer_buffers[layer_info.layer_index] = self._build_layer_buffers(layer_info)

        hidden_size = self._infer_hidden_size(graph_info)
        plan.graph_outputs.append(
            GemmaBufferSpec(
                name="decode_hidden_out",
                kind="graph_output",
                shape_expr=f"[batch, step_tokens, {hidden_size}]",
                dtype="bfloat16",
                scope="graph",
                source="final_hidden_state",
                notes=[
                    "Current dry-run translator keeps the final LM-head / gather tail outside MPK."
                ],
            )
        )
        if graph_info.final_tail.get("gather_tokens") is not None:
            plan.graph_outputs.append(
                GemmaBufferSpec(
                    name="gathered_hidden_out",
                    kind="graph_output",
                    shape_expr=f"[num_selected_tokens, {hidden_size}]",
                    dtype="bfloat16",
                    scope="graph",
                    source=graph_info.final_tail["gather_tokens"].name,
                    notes=["Optional eager tail input when token gathering remains outside MPK."],
                )
            )

        return plan

    def _build_graph_input(self, placeholder_name: str) -> GemmaBufferSpec:
        kind, shape_expr, dtype = _GRAPH_INPUT_KINDS.get(
            placeholder_name,
            ("graph_input", "[dynamic]", "unknown"),
        )
        return GemmaBufferSpec(
            name=placeholder_name,
            kind=kind,
            shape_expr=shape_expr,
            dtype=dtype,
            scope="graph",
            source=placeholder_name,
        )

    def _build_cache_buffer(self, layer_info: GemmaLayerInfo) -> GemmaBufferSpec:
        kv_heads = layer_info.kv_heads if layer_info.kv_heads is not None else "kv_heads"
        head_dim = layer_info.head_dim if layer_info.head_dim is not None else "head_dim"
        return GemmaBufferSpec(
            name=f"layer_{layer_info.layer_index}_kv_cache",
            kind="kv_cache",
            shape_expr=f"[max_num_pages, page_size, {kv_heads}, {head_dim}]",
            dtype="bfloat16",
            scope="layer",
            layer_index=layer_info.layer_index,
            source=layer_info.anchors["kv_cache"].name
            if "kv_cache" in layer_info.anchors
            else None,
        )

    def _build_layer_buffers(self, layer_info: GemmaLayerInfo) -> list[GemmaBufferSpec]:
        hidden_size = (
            layer_info.hidden_size if layer_info.hidden_size is not None else "hidden_size"
        )
        q_heads = layer_info.q_heads if layer_info.q_heads is not None else "q_heads"
        kv_heads = layer_info.kv_heads if layer_info.kv_heads is not None else "kv_heads"
        head_dim = layer_info.head_dim if layer_info.head_dim is not None else "head_dim"
        router_top_k = layer_info.router_top_k if layer_info.router_top_k is not None else "top_k"
        layer_idx = layer_info.layer_index
        scope = "layer"

        return [
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_hidden_in",
                kind="activation",
                shape_expr=f"[batch, step_tokens, {hidden_size}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                notes=["Feeds the attention input layernorm / QKV projection."],
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_qkv_packed",
                kind="temporary",
                shape_expr=(
                    "[num_tokens, "
                    f"{(q_heads if isinstance(q_heads, int) else q_heads)}*{head_dim} + "
                    f"2*{kv_heads}*{head_dim}]"
                ),
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["qkv_linear"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_attn_out",
                kind="temporary",
                shape_expr=f"[num_tokens, {q_heads}*{head_dim}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["cached_attention"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_post_attn_residual",
                kind="activation",
                shape_expr=f"[batch, step_tokens, {hidden_size}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["o_proj"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_ffn_gate_up",
                kind="temporary",
                shape_expr="[num_tokens, 2 * intermediate_size]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["ffn_gate_up"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_ffn_down",
                kind="temporary",
                shape_expr=f"[batch, step_tokens, {hidden_size}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["ffn_down"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_router_logits",
                kind="temporary",
                shape_expr="[num_tokens, num_experts]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["router_proj"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_router_weights",
                kind="temporary",
                shape_expr=f"[num_tokens, {router_top_k}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["topk"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_router_indices",
                kind="temporary",
                shape_expr=f"[num_tokens, {router_top_k}]",
                dtype="int32",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["topk"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_router_masks",
                kind="temporary",
                shape_expr="[num_experts + 1]",
                dtype="int32",
                scope=scope,
                layer_index=layer_idx,
                notes=[
                    "Required by MPK MoE tasks; not materialized explicitly in current FX graph."
                ],
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_moe_input",
                kind="temporary",
                shape_expr=f"[num_tokens, {hidden_size}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                source=layer_info.anchors["moe_fused"].name,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_moe_w13_out",
                kind="temporary",
                shape_expr=f"[num_tokens, {router_top_k}, 2 * expert_intermediate_size]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_moe_act_out",
                kind="temporary",
                shape_expr=f"[num_tokens, {router_top_k}, expert_intermediate_size]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_moe_w2_out",
                kind="temporary",
                shape_expr=f"[num_tokens, {router_top_k}, {hidden_size}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
            ),
            GemmaBufferSpec(
                name=f"layer_{layer_idx}_hidden_out",
                kind="activation",
                shape_expr=f"[batch, step_tokens, {hidden_size}]",
                dtype="bfloat16",
                scope=scope,
                layer_index=layer_idx,
                notes=["Output activation forwarded to the next layer or to the eager tail."],
            ),
        ]

    def _infer_hidden_size(self, graph_info: GemmaGraphInfo) -> str:
        if graph_info.layer_infos and graph_info.layer_infos[0].hidden_size is not None:
            return str(graph_info.layer_infos[0].hidden_size)
        return "hidden_size"
