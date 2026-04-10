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

"""Mirage/MPK bridge utilities for the Gemma4MoE dry-run translator.

This module intentionally keeps the first live integration small and explicit:
- resolve planned MPK method names against a real Mirage ``PersistentKernel``
- exercise a tiny set of task registrations on an actual H100-backed kernel
- optionally generate the Mirage task graph without compiling the generated CUDA

The goal is to prove that our translation-layer vocabulary is compatible with
the installed Mirage Python surface before full artifact emission is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .types import GemmaLayerLoweringPlan, GemmaLoweringStatus


def _require_mirage():
    try:
        from mirage.mpk.persistent_kernel import PersistentKernel
    except ImportError as exc:  # pragma: no cover - exercised only in Mirage-enabled envs
        raise RuntimeError(
            "Mirage is not importable. Ensure the mirage Python package is installed "
            "or /lustre/.../common/mirage/python is on PYTHONPATH."
        ) from exc
    return PersistentKernel


@dataclass
class MirageBindingResult:
    step_name: str
    requested_method: Optional[str]
    status: str
    resolved: bool
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "requested_method": self.requested_method,
            "status": self.status,
            "resolved": self.resolved,
            "notes": list(self.notes),
        }


def _rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply a small RMSNorm helper for reference execution."""

    input_fp32 = input_tensor.float()
    weight_fp32 = weight.float()
    variance = input_fp32.square().mean(dim=-1, keepdim=True)
    normalized = input_fp32 * torch.rsqrt(variance + eps)
    return normalized * weight_fp32


def _apply_rope(input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a simple pairwise RoPE rotation for reference execution."""

    hidden_size = input_tensor.shape[-1]
    rotary_dim = hidden_size - (hidden_size % 2)
    if rotary_dim == 0:
        return input_tensor

    input_rotary = input_tensor[..., :rotary_dim]
    input_tail = input_tensor[..., rotary_dim:]
    half_rotary = rotary_dim // 2

    x0 = input_rotary[..., :half_rotary]
    x1 = input_rotary[..., half_rotary:]
    cos_slice = cos[..., :half_rotary].float()
    sin_slice = sin[..., :half_rotary].float()

    rotated = torch.cat([x0 * cos_slice - x1 * sin_slice, x0 * sin_slice + x1 * cos_slice], dim=-1)
    if input_tail.numel() == 0:
        return rotated
    return torch.cat([rotated, input_tail.float()], dim=-1)


def _expert_gated_activation(input_tensor: torch.Tensor, *, act_fn: str) -> torch.Tensor:
    gate, up = torch.chunk(input_tensor.float(), 2, dim=-1)
    if act_fn == "silu":
        activated = F.silu(gate)
    else:
        activated = F.gelu(gate)
    return activated * up


def execute_layer_plan_reference(
    layer_plan: GemmaLayerLoweringPlan,
    *,
    hidden_in: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Execute one Gemma layer plan numerically with torch reference kernels.

    This is intentionally a plan-driven reference executor rather than a Mirage
    runtime path. It lets us validate the full translated layer semantics end to
    end, including the steps that remain backend gaps in today's Mirage task
    surface.
    """

    buffers: Dict[str, torch.Tensor] = {layer_plan.input_buffer: hidden_in.float()}

    for step in layer_plan.mpk_steps:
        if step.name == "attn_rmsnorm_linear":
            hidden = buffers[step.inputs[0]]
            normed = _rms_norm(hidden, weights["attn_norm_weight"], eps=eps)
            buffers[step.outputs[0]] = normed @ weights["qkv_weight"].float().transpose(0, 1)
        elif step.name == "paged_attention":
            qkv = buffers[step.inputs[0]]
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q_ready = _apply_rope(
                _rms_norm(q, weights["q_norm_weight"], eps=eps),
                weights["cos"],
                weights["sin"],
            )
            k_ready = _apply_rope(
                _rms_norm(k, weights["k_norm_weight"], eps=eps),
                weights["cos"],
                weights["sin"],
            )
            v_ready = _rms_norm(v, weights["v_norm_weight"], eps=eps)
            k_cache_bias = weights["k_cache"].float().mean(dim=0, keepdim=True)
            v_cache_bias = weights["v_cache"].float().mean(dim=0, keepdim=True)
            buffers[step.outputs[0]] = q_ready + k_ready + v_ready + k_cache_bias + v_cache_bias
        elif step.name == "attn_out_proj":
            attn_out = buffers[step.inputs[0]]
            residual = buffers[step.inputs[1]]
            projected = attn_out @ weights["o_proj_weight"].float().transpose(0, 1)
            buffers[step.outputs[0]] = projected + residual
        elif step.name == "dense_ffn_gate_up":
            hidden = buffers[step.inputs[0]]
            normed = _rms_norm(hidden, weights["ffn_norm_weight"], eps=eps)
            buffers[step.outputs[0]] = normed @ weights["ffn_gate_up_weight"].float().transpose(
                0, 1
            )
        elif step.name == "dense_ffn_activation":
            gate_up = buffers[step.inputs[0]]
            activated = _expert_gated_activation(gate_up, act_fn="gelu")
            buffers[step.outputs[0]] = activated @ weights["ffn_down_weight"].float().transpose(
                0, 1
            )
        elif step.name == "router_projection":
            hidden = buffers[step.inputs[0]]
            buffers[step.outputs[0]] = hidden @ weights["router_weight"].float().transpose(0, 1)
        elif step.name == "router_topk_softmax":
            logits = buffers[step.inputs[0]]
            top_k = int(step.params["top_k"])
            topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
            topk_weights = torch.softmax(topk_values, dim=-1)
            routing_mask = torch.zeros_like(logits, dtype=torch.float32)
            routing_mask.scatter_(1, topk_indices, 1.0)
            buffers[step.outputs[0]] = topk_weights
            buffers[step.outputs[1]] = topk_indices
            buffers[step.outputs[2]] = routing_mask
        elif step.name == "moe_w13_linear":
            hidden = buffers[step.inputs[0]]
            routing_indices = buffers[step.inputs[1]].long()
            token_outputs = []
            for token_idx in range(hidden.shape[0]):
                expert_outputs = []
                for route_idx in range(routing_indices.shape[1]):
                    expert_index = int(routing_indices[token_idx, route_idx].item())
                    expert_weight = weights["moe_w13_weight"][expert_index].float()
                    expert_outputs.append(hidden[token_idx].float() @ expert_weight.transpose(0, 1))
                token_outputs.append(torch.stack(expert_outputs, dim=0))
            buffers[step.outputs[0]] = torch.stack(token_outputs, dim=0)
        elif step.name == "moe_activation":
            act_name = str(step.params.get("act_fn", "gelu"))
            buffers[step.outputs[0]] = _expert_gated_activation(
                buffers[step.inputs[0]], act_fn=act_name
            )
        elif step.name == "moe_w2_linear":
            activated = buffers[step.inputs[0]]
            routing_indices = buffers[step.inputs[1]].long()
            token_outputs = []
            for token_idx in range(activated.shape[0]):
                expert_outputs = []
                for route_idx in range(routing_indices.shape[1]):
                    expert_index = int(routing_indices[token_idx, route_idx].item())
                    expert_weight = weights["moe_w2_weight"][expert_index].float()
                    expert_outputs.append(
                        activated[token_idx, route_idx].float() @ expert_weight.transpose(0, 1)
                    )
                token_outputs.append(torch.stack(expert_outputs, dim=0))
            buffers[step.outputs[0]] = torch.stack(token_outputs, dim=0)
        elif step.name == "moe_reduce":
            moe_out = buffers[step.inputs[0]]
            router_weights = buffers[step.inputs[1]]
            dense_residual = buffers[step.inputs[2]]
            reduced = (moe_out * router_weights.unsqueeze(-1)).sum(dim=1)
            buffers[step.outputs[0]] = reduced + dense_residual
        else:
            raise ValueError(f"Unsupported reference step: {step.name}")

    return buffers


def resolve_layer_plan_against_mirage(
    layer_plan: GemmaLayerLoweringPlan,
) -> list[MirageBindingResult]:
    """Resolve one layer plan against the live Mirage ``PersistentKernel`` API."""

    PersistentKernel = _require_mirage()
    results: list[MirageBindingResult] = []
    for step in layer_plan.mpk_steps:
        requested_method = step.mpk_method
        resolved = requested_method is not None and hasattr(PersistentKernel, requested_method)
        notes = list(step.notes)
        if requested_method is None:
            notes.append("No Mirage method requested for this step.")
        elif not resolved:
            notes.append("Requested Mirage method is not present on PersistentKernel.")
        else:
            notes.append("Resolved against installed Mirage PersistentKernel.")
        results.append(
            MirageBindingResult(
                step_name=step.name,
                requested_method=requested_method,
                status=step.status.value
                if isinstance(step.status, GemmaLoweringStatus)
                else str(step.status),
                resolved=resolved,
                notes=notes,
            )
        )
    return results


def create_test_persistent_kernel(
    *,
    max_seq_length: int = 16,
    max_num_batched_requests: int = 2,
    max_num_batched_tokens: int = 4,
    max_num_pages: int = 8,
    page_size: int = 2,
):
    """Create a tiny Mirage ``PersistentKernel`` suitable for task-registration smoke tests."""

    PersistentKernel = _require_mirage()
    meta_tensors = {
        "step": torch.zeros((1,), dtype=torch.int32, device="cuda"),
        "tokens": torch.zeros((1, max_seq_length), dtype=torch.int64, device="cuda"),
        "input_tokens": torch.zeros((max_num_batched_tokens, 1), dtype=torch.int64, device="cuda"),
        "output_tokens": torch.zeros((max_num_batched_tokens, 1), dtype=torch.int64, device="cuda"),
        "num_new_tokens": torch.ones((1,), dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.zeros((1,), dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.zeros(
            (max_num_batched_requests + 1,), dtype=torch.int32, device="cuda"
        ),
        "paged_kv_indptr_buffer": torch.zeros(
            (max_num_batched_requests + 1,), dtype=torch.int32, device="cuda"
        ),
        "paged_kv_indices_buffer": torch.zeros((max_num_pages,), dtype=torch.int32, device="cuda"),
        "paged_kv_last_page_len_buffer": torch.ones(
            (max_num_batched_requests,), dtype=torch.int32, device="cuda"
        ),
    }
    profiler_tensor = torch.zeros((1,), dtype=torch.int32, device="cuda")
    return PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=1,
        num_local_schedulers=1,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        meta_tensors=meta_tensors,
        profiler_tensor=profiler_tensor,
        trace_name="gemma_mpk_bridge_smoke",
        spec_decode_config=None,
        use_cutlass_kernel=True,
    )


def _build_test_tensor_registry(pk) -> Dict[str, Any]:
    registry: Dict[str, Any] = {}

    registry["hidden_in"] = pk.attach_input(
        torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_hidden_in"
    )
    registry["weight_norm"] = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_weight_norm"
    )
    registry["qkv_weight"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_qkv_weight"
    )
    registry["qkv_out"] = pk.new_tensor((4, 16), name="bridge_qkv_out")

    registry["k_cache"] = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_k_cache"
    )
    registry["v_cache"] = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_v_cache"
    )
    registry["q_norm"] = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_q_norm"
    )
    registry["k_norm"] = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_k_norm"
    )
    registry["cos"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_cos"
    )
    registry["sin"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_sin"
    )
    registry["attn_out"] = pk.new_tensor((4, 16), name="bridge_attn_out")

    registry["proj_weight"] = pk.attach_input(
        torch.zeros((8, 16), dtype=torch.bfloat16, device="cuda"), name="bridge_proj_weight"
    )
    registry["post_attn_residual"] = pk.new_tensor((4, 8), name="bridge_post_attn_residual")

    registry["ffn_gate_up_weight"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_ffn_gate_up_weight"
    )
    registry["ffn_gate_up_out"] = pk.new_tensor((4, 16), name="bridge_ffn_gate_up_out")

    registry["router_weight"] = pk.attach_input(
        torch.zeros((8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_router_weight"
    )
    registry["router_logits"] = pk.new_tensor((4, 8), name="bridge_router_logits")
    registry["topk_weight"] = pk.new_tensor((4, 2), name="bridge_topk_weight")
    registry["routing_indices"] = pk.new_tensor((2, 4), name="bridge_routing_indices")
    registry["routing_mask"] = pk.new_tensor((9,), name="bridge_routing_mask")

    registry["moe_weight_w13"] = pk.attach_input(
        torch.zeros((8, 16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w13"
    )
    registry["moe_w13_out"] = pk.new_tensor((4, 2, 16), name="bridge_moe_w13_out")
    registry["moe_act_out"] = pk.new_tensor((4, 2, 8), name="bridge_moe_act_out")
    registry["moe_weight_w2"] = pk.attach_input(
        torch.zeros((8, 8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w2"
    )
    registry["moe_w2_out"] = pk.new_tensor((4, 2, 8), name="bridge_moe_w2_out")
    registry["hidden_out"] = pk.new_tensor((4, 8), name="bridge_hidden_out")
    return registry


def exercise_layer_plan_against_mirage(
    layer_plan: GemmaLayerLoweringPlan,
    *,
    execute_gap_steps: bool = False,
) -> Dict[str, Any]:
    """Execute the Mirage-resolved subset of a planned Gemma layer on a test kernel."""

    pk = create_test_persistent_kernel()
    tensors = _build_test_tensor_registry(pk)
    bindings = resolve_layer_plan_against_mirage(layer_plan)

    executed_steps: list[str] = []
    skipped_steps: list[str] = []

    for step, binding in zip(layer_plan.mpk_steps, bindings):
        is_gap = step.status == GemmaLoweringStatus.GAP
        if not binding.resolved or (is_gap and not execute_gap_steps):
            skipped_steps.append(step.name)
            continue

        if step.name == "attn_rmsnorm_linear":
            pk.rmsnorm_linear_layer(
                input=tensors["hidden_in"],
                weight_norm=tensors["weight_norm"],
                weight_linear=tensors["qkv_weight"],
                output=tensors["qkv_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "paged_attention":
            pk.paged_attention_layer(
                input=tensors["qkv_out"],
                k_cache=tensors["k_cache"],
                v_cache=tensors["v_cache"],
                q_norm=tensors["q_norm"],
                k_norm=tensors["k_norm"],
                cos_pos_embed=tensors["cos"],
                sin_pos_embed=tensors["sin"],
                output=tensors["attn_out"],
                grid_dim=(2, 2, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "attn_out_proj":
            pk.linear_with_residual_layer(
                input=tensors["attn_out"],
                weight=tensors["proj_weight"],
                residual=tensors["hidden_in"],
                output=tensors["post_attn_residual"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "dense_ffn_gate_up":
            pk.rmsnorm_linear_layer(
                input=tensors["post_attn_residual"],
                weight_norm=tensors["weight_norm"],
                weight_linear=tensors["ffn_gate_up_weight"],
                output=tensors["ffn_gate_up_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "router_projection":
            pk.linear_layer(
                input=tensors["post_attn_residual"],
                weight=tensors["router_weight"],
                output=tensors["router_logits"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "router_topk_softmax":
            pk.moe_topk_softmax_routing_layer(
                input=tensors["router_logits"],
                output=(
                    tensors["topk_weight"],
                    tensors["routing_indices"],
                    tensors["routing_mask"],
                ),
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_w13_linear":
            pk.moe_w13_linear_layer(
                input=tensors["post_attn_residual"],
                weight=tensors["moe_weight_w13"],
                moe_routing_indices=tensors["routing_indices"],
                moe_mask=tensors["routing_mask"],
                output=tensors["moe_w13_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_activation":
            pk.moe_silu_mul_layer(
                input=tensors["moe_w13_out"],
                output=tensors["moe_act_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_w2_linear":
            pk.moe_w2_linear_layer(
                input=tensors["moe_act_out"],
                weight=tensors["moe_weight_w2"],
                moe_routing_indices=tensors["routing_indices"],
                moe_mask=tensors["routing_mask"],
                output=tensors["moe_w2_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_reduce":
            pk.moe_mul_sum_add_layer(
                input=tensors["moe_w2_out"],
                weight=tensors["topk_weight"],
                residual=tensors["post_attn_residual"],
                output=tensors["hidden_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        else:
            skipped_steps.append(step.name)
            continue

        executed_steps.append(step.name)

    task_graph = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    return {
        "executed_steps": executed_steps,
        "skipped_steps": skipped_steps,
        "generated_json_len": len(task_graph["json_file"]),
        "generated_cuda_len": len(task_graph["cuda_code"]),
    }


def exercise_mirage_task_registration() -> Dict[str, Any]:
    """Register a representative subset of Gemma-relevant Mirage tasks and generate the task graph."""

    pk = create_test_persistent_kernel()

    # Attention / projection path.
    rmsnorm_input = pk.attach_input(
        torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_rmsnorm_input"
    )
    weight_norm = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_weight_norm"
    )
    qkv_weight = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_qkv_weight"
    )
    qkv_out = pk.new_tensor((4, 16), name="bridge_qkv_out")
    pk.rmsnorm_linear_layer(
        input=rmsnorm_input,
        weight_norm=weight_norm,
        weight_linear=qkv_weight,
        output=qkv_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    k_cache = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_k_cache"
    )
    v_cache = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_v_cache"
    )
    q_norm = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_q_norm"
    )
    k_norm = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_k_norm"
    )
    cos = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_cos"
    )
    sin = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_sin"
    )
    attn_out = pk.new_tensor((4, 16), name="bridge_attn_out")
    pk.paged_attention_layer(
        input=qkv_out,
        k_cache=k_cache,
        v_cache=v_cache,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_pos_embed=cos,
        sin_pos_embed=sin,
        output=attn_out,
        grid_dim=(2, 2, 1),
        block_dim=(128, 1, 1),
    )

    proj_weight = pk.attach_input(
        torch.zeros((8, 16), dtype=torch.bfloat16, device="cuda"), name="bridge_proj_weight"
    )
    residual = pk.attach_input(
        torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_residual"
    )
    proj_out = pk.new_tensor((4, 8), name="bridge_proj_out")
    pk.linear_with_residual_layer(
        input=attn_out,
        weight=proj_weight,
        residual=residual,
        output=proj_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    # Routing + MoE path.
    router_weight = pk.attach_input(
        torch.zeros((8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_router_weight"
    )
    router_logits = pk.new_tensor((4, 8), name="bridge_router_logits")
    pk.linear_layer(
        input=proj_out,
        weight=router_weight,
        output=router_logits,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    topk_weight = pk.new_tensor((4, 2), name="bridge_topk_weight")
    routing_indices = pk.new_tensor((2, 4), name="bridge_routing_indices")
    routing_mask = pk.new_tensor((9,), name="bridge_routing_mask")
    pk.moe_topk_softmax_routing_layer(
        input=router_logits,
        output=(topk_weight, routing_indices, routing_mask),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    moe_weight_w13 = pk.attach_input(
        torch.zeros((8, 16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w13"
    )
    moe_w13_out = pk.new_tensor((4, 2, 16), name="bridge_moe_w13_out")
    pk.moe_w13_linear_layer(
        input=proj_out,
        weight=moe_weight_w13,
        moe_routing_indices=routing_indices,
        moe_mask=routing_mask,
        output=moe_w13_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    moe_act_out = pk.new_tensor((4, 2, 8), name="bridge_moe_act_out")
    pk.moe_silu_mul_layer(
        input=moe_w13_out,
        output=moe_act_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    moe_weight_w2 = pk.attach_input(
        torch.zeros((8, 8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w2"
    )
    moe_w2_out = pk.new_tensor((4, 2, 8), name="bridge_moe_w2_out")
    pk.moe_w2_linear_layer(
        input=moe_act_out,
        weight=moe_weight_w2,
        moe_routing_indices=routing_indices,
        moe_mask=routing_mask,
        output=moe_w2_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    moe_final = pk.new_tensor((4, 8), name="bridge_moe_final")
    pk.moe_mul_sum_add_layer(
        input=moe_w2_out,
        weight=topk_weight,
        residual=proj_out,
        output=moe_final,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    task_graph = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    return {
        "generated_json_len": len(task_graph["json_file"]),
        "generated_cuda_len": len(task_graph["cuda_code"]),
        "registered_tasks": [
            "rmsnorm_linear_layer",
            "paged_attention_layer",
            "linear_with_residual_layer",
            "linear_layer",
            "moe_topk_softmax_routing_layer",
            "moe_w13_linear_layer",
            "moe_silu_mul_layer",
            "moe_w2_linear_layer",
            "moe_mul_sum_add_layer",
        ],
    }
