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

"""Triton kernels and custom op registration for dense MoE with fused GEMM + activation.

The dense MoE computation is:
    1. gate_up = bmm(hidden, gate_up_w) + gate_up_b
    2. Interleaved split: gate = gate_up[..., ::2], up = gate_up[..., 1::2]
    3. Clamp: gate = clamp(gate, max=limit), up = clamp(up, min=-limit, max=limit)
    4. GLU: glu = gate * sigmoid(gate * alpha)
    5. Fused multiply: act_out = (up + 1) * glu
    6. down_out = bmm(act_out, down_w) + down_b
    7. Weighted sum over experts

The Triton kernel fuses steps 2-5 (interleaved split, clamp, GLU, multiply) into a
single kernel to avoid multiple passes over intermediate memory.

A second Triton kernel handles the routing-weighted summation (step 7).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _fused_glu_activation_kernel(
    gate_up_ptr,
    output_ptr,
    stride_gate_up_row,
    stride_out_row,
    alpha_val,
    limit_val,
    I_SIZE: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    """Fused interleaved-split + clamp + GLU activation kernel.

    Reads gate_up tensor with interleaved [g0,u0,g1,u1,...] layout of shape [..., 2*I].
    Loads a contiguous 2*BLOCK_I chunk and splits gate/up via stride-2 indexing in registers.
    Computes the fused activation and writes output of shape [..., I].

    For each element i in [0, I):
        gate = clamp(gate_up[..., 2*i], max=limit)
        up   = clamp(gate_up[..., 2*i+1], min=-limit, max=limit)
        glu  = gate * sigmoid(gate * alpha)
        out  = (up + 1) * glu

    Grid: (num_rows, cdiv(I_SIZE, BLOCK_I)) — 2D grid over rows and I-blocks.
    """
    row_idx = tl.program_id(0)
    i_block_idx = tl.program_id(1)
    col_offsets = i_block_idx * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = col_offsets < I_SIZE

    # Stride-2 indexing into interleaved layout: gate at even, up at odd
    gate_offsets = col_offsets * 2
    up_offsets = col_offsets * 2 + 1

    gate_up_row_ptr = gate_up_ptr + row_idx * stride_gate_up_row
    gate_vals = tl.load(gate_up_row_ptr + gate_offsets, mask=mask, other=0.0)
    up_vals = tl.load(gate_up_row_ptr + up_offsets, mask=mask, other=0.0)

    # Compute in native dtype (no fp32 upcast) — Triton's sigmoid promotes
    # to fp32 internally when needed, so numerical stability is preserved.
    gate_f = tl.minimum(gate_vals, limit_val)
    up_f = tl.maximum(tl.minimum(up_vals, limit_val), -limit_val)

    # GLU: glu = gate * sigmoid(gate * alpha)
    glu = gate_f * tl.sigmoid(gate_f * alpha_val)

    # Fused multiply: (up + 1) * glu
    result = (up_f + 1.0) * glu

    # Store result — offset output by I-block
    out_row_ptr = output_ptr + row_idx * stride_out_row
    tl.store(out_row_ptr + col_offsets, result, mask=mask)


@triton.jit
def _weighted_expert_sum_kernel(
    expert_out_ptr,
    routing_weights_ptr,
    output_ptr,
    stride_expert_out_e,
    stride_expert_out_t,
    stride_expert_out_h,
    stride_routing_t,
    stride_routing_e,
    stride_out_t,
    stride_out_h,
    num_experts: tl.constexpr,
    H_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Weighted summation over experts for each token (dense fallback).

    For each token t and hidden dim h:
        output[t, h] = sum_e(routing_weights[t, e] * expert_out[e, t, h])

    Grid: (num_tokens, cdiv(H_SIZE, BLOCK_H)) — 2D grid over tokens and H-blocks.
    """
    token_idx = tl.program_id(0)
    h_block_idx = tl.program_id(1)
    col_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = col_offsets < H_SIZE

    # Accumulate weighted expert outputs in float32
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for e in range(num_experts):
        # Load routing weight for this token-expert pair
        w = tl.load(routing_weights_ptr + token_idx * stride_routing_t + e * stride_routing_e)
        w_f = w.to(tl.float32)

        # Skip zero-weight experts (common with top-k routing: e.g., top-4 of 128)
        # This avoids expensive global loads for experts that don't contribute.
        if w_f != 0.0:
            # Load expert output for this token
            expert_row_ptr = (
                expert_out_ptr + e * stride_expert_out_e + token_idx * stride_expert_out_t
            )
            expert_vals = tl.load(
                expert_row_ptr + col_offsets * stride_expert_out_h, mask=mask, other=0.0
            )

            acc += w_f * expert_vals.to(tl.float32)

    # Store result
    out_ptr = output_ptr + token_idx * stride_out_t
    tl.store(out_ptr + col_offsets * stride_out_h, acc.to(OUTPUT_DTYPE), mask=mask)


@triton.jit
def _weighted_expert_gather_kernel(
    expert_out_ptr,
    topk_indices_ptr,
    topk_weights_ptr,
    output_ptr,
    stride_expert_out_e,
    stride_expert_out_t,
    stride_expert_out_h,
    stride_idx_t,
    stride_idx_k,
    stride_w_t,
    stride_w_k,
    stride_out_t,
    stride_out_h,
    TOP_K: tl.constexpr,
    H_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    """Gather-based weighted expert sum using top-k indices.

    Instead of looping over all E experts and checking for zero weights,
    this kernel directly gathers only the TOP_K active experts per token.
    Much faster when TOP_K << num_experts (e.g., 4 out of 128).

    Grid: (num_tokens, cdiv(H_SIZE, BLOCK_H))
    """
    token_idx = tl.program_id(0)
    h_block_idx = tl.program_id(1)
    col_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = col_offsets < H_SIZE

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for k in range(TOP_K):
        # Load expert index and routing weight for this top-k slot
        e = tl.load(topk_indices_ptr + token_idx * stride_idx_t + k * stride_idx_k)
        w = tl.load(topk_weights_ptr + token_idx * stride_w_t + k * stride_w_k)
        w_f = w.to(tl.float32)

        # Gather expert output for this token
        expert_row_ptr = expert_out_ptr + e * stride_expert_out_e + token_idx * stride_expert_out_t
        expert_vals = tl.load(
            expert_row_ptr + col_offsets * stride_expert_out_h, mask=mask, other=0.0
        )

        acc += w_f * expert_vals.to(tl.float32)

    out_ptr = output_ptr + token_idx * stride_out_t
    tl.store(out_ptr + col_offsets * stride_out_h, acc.to(OUTPUT_DTYPE), mask=mask)


def _moe_dense_mlp_triton(
    hidden_states: Tensor,
    routing_weights: Tensor,
    gate_up_w: Tensor,
    gate_up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> Tensor:
    """Python launcher for the Triton-accelerated dense MoE.

    Uses torch.bmm for the GEMM components and Triton kernels for:
      - Fused interleaved-split + clamp + GLU activation
      - Routing-weighted expert summation

    Args:
        hidden_states: Input tensor [B, S, H] or [B*S, H].
        routing_weights: Dense routing weights [B*S, E].
        gate_up_w: Fused gate+up weight [E, H, 2I].
        gate_up_b: Fused gate+up bias [E, 2I].
        down_w: Down projection weight [E, I, H].
        down_b: Down projection bias [E, H].
        alpha: Scaling factor for sigmoid in GLU.
        limit: Clamp limit for gate and up projections.

    Returns:
        Output tensor with the same shape as hidden_states.
    """
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    hidden_flat = hidden_states.reshape(-1, hidden_size)  # (T, H)
    num_tokens = hidden_flat.shape[0]
    num_experts = routing_weights.shape[1]
    intermediate_size = gate_up_w.shape[2] // 2  # 2I -> I

    # Step 1: Replicate tokens across experts and compute gate_up projection (BMM)
    # hidden_rep: [E, T, H]
    hidden_rep = hidden_flat.unsqueeze(0).expand(num_experts, -1, -1)
    # gate_up: [E, T, 2I]
    gate_up = torch.bmm(hidden_rep, gate_up_w) + gate_up_b[:, None, :]

    # Step 2-5: Fused interleaved-split + clamp + GLU activation (Triton kernel)
    # The kernel reads the interleaved layout directly (stride-2 indexing).
    # BMM output is already contiguous — no copy needed.

    # Output: [E, T, I]
    act_out = torch.empty(
        num_experts,
        num_tokens,
        intermediate_size,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    total_rows = num_experts * num_tokens

    # Adaptive BLOCK_I: use smaller blocks with 2D grid when total_rows is small
    # (low parallelism) to increase occupancy. For large total_rows, use full-width
    # blocks to minimize grid overhead. num_warps=16 is best across all configs.
    if total_rows <= 128:
        BLOCK_I = 1024
    elif total_rows <= 1024:
        BLOCK_I = 2048  # medium: 2 blocks per row
    else:
        BLOCK_I = triton.next_power_of_2(intermediate_size)
    num_i_blocks = triton.cdiv(intermediate_size, BLOCK_I)
    # num_warps=8 is optimal for large T (BW-bound); 16 for small T (latency-bound)
    k1_num_warps = 16 if total_rows <= 1024 else 8

    grid = (total_rows, num_i_blocks)
    _fused_glu_activation_kernel[grid](
        gate_up,
        act_out,
        gate_up.stride(-2),
        act_out.stride(-2),
        float(alpha),
        float(limit),
        I_SIZE=intermediate_size,
        BLOCK_I=BLOCK_I,
        num_warps=k1_num_warps,
        num_stages=2,
    )

    # Step 6: Down projection (BMM)
    # next_states: [E, T, H]
    next_states = torch.bmm(act_out, down_w) + down_b[:, None, :]

    # Step 7: Routing-weighted summation over experts (Triton kernel)
    # BMM output is already contiguous — no copy needed.
    output = torch.empty(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device
    )

    # Adaptive BLOCK_H: smaller blocks for more parallelism at low T/E,
    # larger blocks to reduce grid overhead at high T.
    if num_tokens <= 32 and num_experts <= 32:
        BLOCK_H = 256  # small model, low T: maximize parallelism
    elif num_tokens <= 128:
        BLOCK_H = 1024  # medium T: good balance
    else:
        BLOCK_H = triton.next_power_of_2(hidden_size)  # high T: 1D grid
    num_h_blocks = triton.cdiv(hidden_size, BLOCK_H)
    grid_sum = (num_tokens, num_h_blocks)

    output_dtype = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[hidden_states.dtype]

    # Use gather kernel when routing is sparse and T is large enough.
    # For small T, the torch.topk extraction overhead outweighs the kernel savings.
    # For large T, the gather kernel's savings scale with T while topk is O(T*E).
    has_zeros = (routing_weights[0] == 0).any().item()
    use_gather = has_zeros and num_tokens >= 64

    if use_gather:
        # Sparse routing: count active experts and extract top-k indices
        max_active = (routing_weights[0] != 0).sum().item()
        topk_weights, topk_indices = torch.topk(routing_weights, k=max_active, dim=1)
        topk_indices = topk_indices.to(torch.int32)
        # Gather kernel uses BLOCK_H from the same adaptive logic above
        _weighted_expert_gather_kernel[grid_sum](
            next_states,
            topk_indices,
            topk_weights,
            output,
            stride_expert_out_e=next_states.stride(0),
            stride_expert_out_t=next_states.stride(1),
            stride_expert_out_h=next_states.stride(2),
            stride_idx_t=topk_indices.stride(0),
            stride_idx_k=topk_indices.stride(1),
            stride_w_t=topk_weights.stride(0),
            stride_w_k=topk_weights.stride(1),
            stride_out_t=output.stride(0),
            stride_out_h=output.stride(1),
            TOP_K=max_active,
            H_SIZE=hidden_size,
            BLOCK_H=BLOCK_H,
            OUTPUT_DTYPE=output_dtype,
            num_warps=8,
            num_stages=2,
        )
    else:
        # Dense routing: use the loop-based kernel with zero-skip
        _weighted_expert_sum_kernel[grid_sum](
            next_states,
            routing_weights,
            output,
            stride_expert_out_e=next_states.stride(0),
            stride_expert_out_t=next_states.stride(1),
            stride_expert_out_h=next_states.stride(2),
            stride_routing_t=routing_weights.stride(0),
            stride_routing_e=routing_weights.stride(1),
            stride_out_t=output.stride(0),
            stride_out_h=output.stride(1),
            num_experts=num_experts,
            H_SIZE=hidden_size,
            BLOCK_H=BLOCK_H,
            OUTPUT_DTYPE=output_dtype,
            num_warps=16,
            num_stages=2,
        )

    return output.reshape(*leading_shape, hidden_size)


@torch.library.custom_op("auto_deploy::triton_moe_dense_mlp", mutates_args=())
def triton_moe_dense_mlp(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    """Triton-accelerated dense MoE custom op (GPT-OSS style).

    Matches the signature and semantics of auto_deploy::torch_moe_dense_mlp but
    uses Triton kernels for the fused GLU activation and weighted expert summation.
    """
    return _moe_dense_mlp_triton(
        hidden_states, routing_weights, gate_up_w, gate_up_b, down_w, down_b, alpha, limit
    )


@triton_moe_dense_mlp.register_fake
def _triton_moe_dense_mlp_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
