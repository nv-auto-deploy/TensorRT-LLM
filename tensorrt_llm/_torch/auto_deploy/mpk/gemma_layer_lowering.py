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

"""Build dry-run canonical and MPK lowering plans for Gemma4MoE layers."""

from __future__ import annotations

from .types import (
    GemmaCanonicalOp,
    GemmaLayerInfo,
    GemmaLayerLoweringPlan,
    GemmaLayerSchema,
    GemmaLoweringStatus,
    GemmaMpkStep,
)

_ACT_FN_NAMES = {
    0: "relu",
    1: "gelu",
    2: "silu",
    3: "swiglu",
    4: "geglu",
    5: "relu2",
}


class GemmaLayerLoweringPlanner:
    """Translate one analyzed Gemma layer into a dry-run MPK step sequence."""

    def build(self, layer_info: GemmaLayerInfo) -> GemmaLayerLoweringPlan:
        layer_idx = layer_info.layer_index
        plan = GemmaLayerLoweringPlan(
            layer_index=layer_idx,
            schema=layer_info.schema,
            input_buffer=f"layer_{layer_idx}_hidden_in",
            output_buffer=f"layer_{layer_idx}_hidden_out",
        )
        plan.canonical_ops.extend(self._build_canonical_ops(layer_info))
        plan.mpk_steps.extend(self._build_mpk_steps(layer_info))
        if layer_info.schema == GemmaLayerSchema.FINAL:
            plan.notes.append(
                "Final-layer schema kept explicit for future tail/lm-head integration."
            )
        return plan

    def _build_canonical_ops(self, layer_info: GemmaLayerInfo) -> list[GemmaCanonicalOp]:
        layer_idx = layer_info.layer_index
        return [
            GemmaCanonicalOp(
                name="attn_input_rmsnorm_qkv_proj",
                inputs=[f"layer_{layer_idx}_hidden_in"],
                outputs=[f"layer_{layer_idx}_qkv_packed"],
                anchors=["qkv_linear", "q_norm", "k_norm", "v_norm"],
                notes=["Covers input RMSNorm followed by fused QKV projection."],
            ),
            GemmaCanonicalOp(
                name="paged_cached_attention",
                inputs=[
                    f"layer_{layer_idx}_qkv_packed",
                    f"layer_{layer_idx}_kv_cache",
                    "paged_metadata_0",
                    "paged_metadata_1",
                ],
                outputs=[f"layer_{layer_idx}_attn_out"],
                anchors=["rope", "cached_attention"],
                notes=["Consumes explicit KV cache plus paged-attention metadata."],
            ),
            GemmaCanonicalOp(
                name="attn_out_proj_residual",
                inputs=[f"layer_{layer_idx}_attn_out", f"layer_{layer_idx}_hidden_in"],
                outputs=[f"layer_{layer_idx}_post_attn_residual"],
                anchors=["o_proj"],
            ),
            GemmaCanonicalOp(
                name="dense_ffn_gelu_gate",
                inputs=[f"layer_{layer_idx}_post_attn_residual"],
                outputs=[f"layer_{layer_idx}_ffn_down"],
                anchors=["ffn_gate_up", "ffn_down"],
                notes=["Observed graph pattern is gelu(gate) * up followed by down projection."],
            ),
            GemmaCanonicalOp(
                name="router_topk",
                inputs=[f"layer_{layer_idx}_post_attn_residual"],
                outputs=[
                    f"layer_{layer_idx}_router_logits",
                    f"layer_{layer_idx}_router_weights",
                    f"layer_{layer_idx}_router_indices",
                    f"layer_{layer_idx}_router_masks",
                ],
                anchors=["router_proj", "topk"],
            ),
            GemmaCanonicalOp(
                name="moe_expert_compute",
                inputs=[
                    f"layer_{layer_idx}_post_attn_residual",
                    f"layer_{layer_idx}_router_weights",
                    f"layer_{layer_idx}_router_indices",
                    f"layer_{layer_idx}_router_masks",
                ],
                outputs=[f"layer_{layer_idx}_hidden_out"],
                anchors=["moe_fused"],
                notes=["Dry-run lowering expands this into multiple MPK MoE tasks."],
            ),
        ]

    def _build_mpk_steps(self, layer_info: GemmaLayerInfo) -> list[GemmaMpkStep]:
        layer_idx = layer_info.layer_index
        mpk_steps = [
            GemmaMpkStep(
                name="attn_rmsnorm_linear",
                mpk_method="rmsnorm_linear_layer",
                inputs=[f"layer_{layer_idx}_hidden_in"],
                outputs=[f"layer_{layer_idx}_qkv_packed"],
                status=GemmaLoweringStatus.SUPPORTED,
                params={"anchor": layer_info.anchors["qkv_linear"].name},
                notes=["Direct match for input RMSNorm + fused QKV projection."],
            ),
            GemmaMpkStep(
                name="paged_attention",
                mpk_method="paged_attention_layer",
                inputs=[
                    f"layer_{layer_idx}_qkv_packed",
                    f"layer_{layer_idx}_kv_cache",
                    "paged_metadata_0",
                    "paged_metadata_1",
                ],
                outputs=[f"layer_{layer_idx}_attn_out"],
                status=GemmaLoweringStatus.PARTIAL,
                params={
                    "num_q_heads": layer_info.q_heads,
                    "num_kv_heads": layer_info.kv_heads,
                    "head_dim": layer_info.head_dim,
                },
                notes=[
                    "MPK paged_attention_layer directly matches the paged-cache decode core.",
                    (
                        "Gemma graph applies an explicit V RMSNorm, but MPK attention tasks expose "
                        "only q_norm/k_norm inputs."
                    ),
                ],
            ),
            GemmaMpkStep(
                name="attn_out_proj",
                mpk_method="linear_with_residual_layer",
                inputs=[f"layer_{layer_idx}_attn_out", f"layer_{layer_idx}_hidden_in"],
                outputs=[f"layer_{layer_idx}_post_attn_residual"],
                status=GemmaLoweringStatus.PARTIAL,
                params={"anchor": layer_info.anchors["o_proj"].name},
                notes=[
                    "Projection plus residual combine maps closely to linear_with_residual_layer.",
                    "Post-attention Gemma RMSNorm remains an explicit follow-on concern.",
                ],
            ),
            GemmaMpkStep(
                name="dense_ffn_gate_up",
                mpk_method="rmsnorm_linear_layer",
                inputs=[f"layer_{layer_idx}_post_attn_residual"],
                outputs=[f"layer_{layer_idx}_ffn_gate_up"],
                status=GemmaLoweringStatus.SUPPORTED,
                params={"anchor": layer_info.anchors["ffn_gate_up"].name},
                notes=["Pre-FFN RMSNorm + fused gate/up projection is a direct match."],
            ),
            GemmaMpkStep(
                name="dense_ffn_activation",
                mpk_method=None,
                inputs=[f"layer_{layer_idx}_ffn_gate_up"],
                outputs=[f"layer_{layer_idx}_ffn_down"],
                status=GemmaLoweringStatus.GAP,
                params={"observed_activation": "gelu_mul"},
                notes=[
                    "Observed FX graph uses gelu(gate) * up before the down projection.",
                    "MPK currently exposes silu_mul_layer but no matching gelu_mul_layer task.",
                ],
            ),
            GemmaMpkStep(
                name="router_projection",
                mpk_method="linear_layer",
                inputs=[f"layer_{layer_idx}_post_attn_residual"],
                outputs=[f"layer_{layer_idx}_router_logits"],
                status=GemmaLoweringStatus.PARTIAL,
                params={"anchor": layer_info.anchors["router_proj"].name},
                notes=[
                    "Router projection itself is a linear task.",
                    "Gemma-specific router prep arithmetic still needs explicit lowering before this point.",
                ],
            ),
            GemmaMpkStep(
                name="router_topk_softmax",
                mpk_method="moe_topk_softmax_routing_layer",
                inputs=[f"layer_{layer_idx}_router_logits"],
                outputs=[
                    f"layer_{layer_idx}_router_weights",
                    f"layer_{layer_idx}_router_indices",
                    f"layer_{layer_idx}_router_masks",
                ],
                status=GemmaLoweringStatus.SUPPORTED,
                params={"top_k": layer_info.router_top_k},
                notes=["Direct match to MPK’s routing task shape."],
            ),
        ]

        moe_act_name = _ACT_FN_NAMES.get(layer_info.moe_act_fn, "unknown")
        moe_supported = (
            moe_act_name in {"silu", "swiglu"} if layer_info.moe_act_fn is not None else False
        )
        moe_status = GemmaLoweringStatus.SUPPORTED if moe_supported else GemmaLoweringStatus.GAP
        moe_notes = [
            "Dry-run lowering expands fused MoE into MPK’s staged expert pipeline.",
        ]
        if not moe_supported:
            moe_notes.append(
                (
                    "Current MPK MoE task vocabulary is SiLU-oriented; "
                    "Gemma MoE activation needs confirmation or a new task."
                )
            )

        mpk_steps.extend(
            [
                GemmaMpkStep(
                    name="moe_w13_linear",
                    mpk_method="moe_w13_linear_layer",
                    inputs=[
                        f"layer_{layer_idx}_post_attn_residual",
                        f"layer_{layer_idx}_router_indices",
                        f"layer_{layer_idx}_router_masks",
                    ],
                    outputs=[f"layer_{layer_idx}_moe_w13_out"],
                    status=moe_status,
                    params={
                        "is_gated_mlp": layer_info.moe_is_gated_mlp,
                        "act_fn": moe_act_name,
                    },
                    notes=list(moe_notes),
                ),
                GemmaMpkStep(
                    name="moe_activation",
                    mpk_method="moe_silu_mul_layer" if moe_supported else None,
                    inputs=[f"layer_{layer_idx}_moe_w13_out"],
                    outputs=[f"layer_{layer_idx}_moe_act_out"],
                    status=moe_status,
                    params={"act_fn": moe_act_name},
                    notes=list(moe_notes),
                ),
                GemmaMpkStep(
                    name="moe_w2_linear",
                    mpk_method="moe_w2_linear_layer" if moe_supported else None,
                    inputs=[
                        f"layer_{layer_idx}_moe_act_out",
                        f"layer_{layer_idx}_router_indices",
                        f"layer_{layer_idx}_router_masks",
                    ],
                    outputs=[f"layer_{layer_idx}_moe_w2_out"],
                    status=moe_status,
                    params={"act_fn": moe_act_name},
                    notes=list(moe_notes),
                ),
                GemmaMpkStep(
                    name="moe_reduce",
                    mpk_method="moe_mul_sum_add_layer" if moe_supported else None,
                    inputs=[
                        f"layer_{layer_idx}_moe_w2_out",
                        f"layer_{layer_idx}_router_weights",
                        f"layer_{layer_idx}_ffn_down",
                    ],
                    outputs=[f"layer_{layer_idx}_hidden_out"],
                    status=moe_status,
                    params={"act_fn": moe_act_name},
                    notes=list(moe_notes),
                ),
            ]
        )
        return mpk_steps
