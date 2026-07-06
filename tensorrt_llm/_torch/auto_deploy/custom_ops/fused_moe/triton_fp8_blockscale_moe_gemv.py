# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton fp8-blockscale grouped-GEMV chain for total-tokens==1 MoE decode.

At batch=1 decode the SM90 finegrained-FP8 (DeepSeek block-scale) MoE runs a full
grouped-GEMM pipeline (act quant + build-expert-maps + expand + two deep_gemm
swapAB GEMMs + activation + finalize) to process at most ``top_k`` single-row
GEMVs. The TMA/WGMMA tiles of the grouped GEMM read expert weights at a fraction
of the streaming ceiling when M==1, and the routing glue kernels cost several
extra launches per layer.

This module replaces that pipeline for ``num_tokens == 1`` with two Triton
kernels that stream each selected local expert's weights exactly once:

1. ``_fp8_bs_gate_up_swiglu_kernel`` — for every top-k slot whose (global)
   expert lives on this rank, a dual GEMV over the stacked FC1 weight
   ``[E, 2*I, H]`` (rows ``[0, I)`` = w3/up, rows ``[I, 2*I)`` = w1/gate, the
   ``cat([w3, w1], dim=1)`` layout produced by ``_stack_finegrained_fp8_moe_weights``)
   with per-128x128-block scale dequant, fused with the SwiGLU epilogue
   ``act[i] = silu(gate[i]) * up[i]``. Non-local slots exit without writing.
2. ``_fp8_bs_down_reduce_kernel`` — the FC2 GEMV over ``[E, H, I]`` for each
   local slot, scaled by the fp32 routing weight and accumulated across slots
   into the final ``[1, H]`` bf16 output (zeros when no slot is local, so the
   EP all_reduce after this op still sums correctly).

Numerics: bf16 activations are multiplied with exactly-dequantized fp8 weights
(weight fp8 value times its fp32 block scale) and accumulated in fp32 per
128-wide k-block before the block scale is applied — the same promotion
structure as the deep_gemm reference, minus the dynamic fp8 activation
quantization (strictly less quantization error). Expert locality follows the
same GLOBAL-id convention as the cpp path: a slot contributes iff
``local_expert_offset <= expert_id < local_expert_offset + num_local_experts``.

All launches use fixed, shape-resolved configs (no runtime autotuning) and no
host-device syncs, so the chain is safe under CUDA-graph capture; expert
selection is read on-device from ``selected_experts`` at replay time.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Scale granularity of the FineGrainedFP8 / DeepSeek block-scale format.
_SCALE_BLOCK = 128


@triton.jit
def _fp8_bs_gate_up_swiglu_kernel(
    x_ptr,  # [H] bf16 activations
    se_ptr,  # [TOP_K] int32 GLOBAL expert ids
    w_ptr,  # [E, 2*I, H] fp8e4m3 stacked FC1 (rows [0,I)=up/w3, [I,2I)=gate/w1)
    s_ptr,  # [E, 2*I/128, H/128] fp32 FC1 block scales
    act_ptr,  # [TOP_K, I] bf16 output (written only for local slots)
    local_expert_offset,
    num_local_experts,
    stride_w_e,  # 2*I*H
    stride_s_e,  # (2*I/128) * (H/128)
    stride_s_row,  # H/128
    INTER: tl.constexpr,
    K: tl.constexpr,  # hidden size H; K % 128 == 0
    BLOCK_N: tl.constexpr,  # rows of I per program (I % BLOCK_N == 0)
    NKB: tl.constexpr,  # 128-wide k-blocks per slab ((K/128) % NKB == 0)
):
    """Fused gate+up fp8-blockscale dual GEMV with SwiGLU epilogue for one slot.

    ``act[slot, n] = silu(sum_k deq_w[I+n, k] * x[k]) * sum_k deq_w[n, k] * x[k]``
    where ``deq_w[n, k] = w[n, k] * scale[n // 128, k // 128]`` and each 128-wide
    k-block partial sum is accumulated in fp32 before its block scale is applied.
    """
    pid = tl.program_id(0)
    n_blocks = INTER // BLOCK_N
    k_slot = pid // n_blocks
    nb = pid % n_blocks

    e = tl.load(se_ptr + k_slot)
    e_local = e - local_expert_offset
    if (e_local < 0) | (e_local >= num_local_experts):
        return

    offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)  # rows within [0, I)
    w_base = w_ptr + e_local.to(tl.int64) * stride_w_e
    s_base = s_ptr + e_local.to(tl.int64) * stride_s_e
    s_row_up = offs_n // 128
    s_row_gate = (offs_n + INTER) // 128

    acc_up = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc_gate = tl.zeros([BLOCK_N], dtype=tl.float32)
    for kb0 in range(0, K // 128, NKB):
        offs_kb = kb0 + tl.arange(0, NKB)
        offs_k = offs_kb[:, None] * 128 + tl.arange(0, 128)[None, :]
        x = tl.load(x_ptr + offs_k).to(tl.float32)  # [NKB, 128]
        # Expert weights are read exactly once per decode step; evict_first keeps
        # the streamed rows out of the L2 working set (idea_0022 lever).
        w_up = tl.load(
            w_base + offs_n[:, None, None] * K + offs_k[None, :, :],
            eviction_policy="evict_first",
        ).to(tl.float32)  # [BLOCK_N, NKB, 128]
        p_up = tl.sum(w_up * x[None, :, :], axis=2)  # [BLOCK_N, NKB]
        s_up = tl.load(s_base + s_row_up[:, None] * stride_s_row + offs_kb[None, :])
        acc_up += tl.sum(p_up * s_up, axis=1)

        w_gate = tl.load(
            w_base + (offs_n[:, None, None] + INTER) * K + offs_k[None, :, :],
            eviction_policy="evict_first",
        ).to(tl.float32)
        p_gate = tl.sum(w_gate * x[None, :, :], axis=2)
        s_gate = tl.load(s_base + s_row_gate[:, None] * stride_s_row + offs_kb[None, :])
        acc_gate += tl.sum(p_gate * s_gate, axis=1)

    act = acc_gate * tl.sigmoid(acc_gate) * acc_up
    tl.store(act_ptr + k_slot * INTER + offs_n, act.to(tl.bfloat16))


@triton.jit
def _fp8_bs_down_reduce_kernel(
    act_ptr,  # [TOP_K, I] bf16 SwiGLU outputs (valid only for local slots)
    se_ptr,  # [TOP_K] int32 GLOBAL expert ids
    rw_ptr,  # [TOP_K] fp32 routing weights
    w_ptr,  # [E, H, I] fp8e4m3 FC2
    s_ptr,  # [E, H/128, I/128] fp32 FC2 block scales
    out_ptr,  # [H] bf16 final output
    local_expert_offset,
    num_local_experts,
    stride_w_e,  # H*I
    stride_s_e,  # (H/128) * (I/128)
    stride_s_row,  # I/128
    INTER: tl.constexpr,  # FC2 reduction dim; I % 128 == 0
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,  # rows of H per program (H % BLOCK_N == 0)
    NKB: tl.constexpr,  # pow2 slab of 128-wide k-blocks, NKB >= I/128 (masked)
):
    """FC2 fp8-blockscale GEMV per local slot + routing-weight reduce.

    ``out[n] = sum_{local slots k} rw[k] * sum_i deq_w2[e_k][n, i] * act[k, i]``.
    Always stores (zeros when no slot is local) so the output is well-defined for
    the EP all_reduce that follows this op.
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    s_row = offs_n // 128

    offs_kb = tl.arange(0, NKB)
    kb_mask = offs_kb < INTER // 128
    offs_k = offs_kb[:, None] * 128 + tl.arange(0, 128)[None, :]

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k_slot in range(0, TOP_K):
        e = tl.load(se_ptr + k_slot)
        e_local = e - local_expert_offset
        if (e_local >= 0) & (e_local < num_local_experts):
            rw = tl.load(rw_ptr + k_slot)
            w_base = w_ptr + e_local.to(tl.int64) * stride_w_e
            s_base = s_ptr + e_local.to(tl.int64) * stride_s_e
            a = tl.load(act_ptr + k_slot * INTER + offs_k, mask=kb_mask[:, None], other=0.0).to(
                tl.float32
            )  # [NKB, 128]
            w = tl.load(
                w_base + offs_n[:, None, None] * INTER + offs_k[None, :, :],
                mask=kb_mask[None, :, None],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)  # [BLOCK_N, NKB, 128]
            p = tl.sum(w * a[None, :, :], axis=2)  # [BLOCK_N, NKB]
            s = tl.load(
                s_base + s_row[:, None] * stride_s_row + offs_kb[None, :],
                mask=kb_mask[None, :],
                other=0.0,
            )
            acc += rw * tl.sum(p * s, axis=1)
    tl.store(out_ptr + offs_n, acc.to(tl.bfloat16))


# (BLOCK_N, NKB, num_warps, num_stages) for the gate_up kernel, keyed by (I, H).
# Measured on H100 under CUDA-graph replay with L2-busted weights (Step-3.7-Flash
# per-rank EP8 routed expert shape); generic fallback below.
_GATE_UP_CONFIG_TABLE = {
    (1280, 4096): (4, 16, 8, 3),
}
# (BLOCK_N, NKB, num_warps, num_stages) for the down+reduce kernel, keyed by (H, I).
_DOWN_CONFIG_TABLE = {
    (4096, 1280): (16, 16, 8, 3),
}


def _gate_up_config(inter: int, hidden: int) -> Tuple[int, int, int, int]:
    cfg = _GATE_UP_CONFIG_TABLE.get((inter, hidden))
    if cfg is not None:
        return cfg
    # One row pair per CTA maximizes CTA count; slabs of up to 16 k-blocks keep
    # the register footprint of the two weight tiles bounded.
    nkb_total = hidden // _SCALE_BLOCK
    nkb = 16
    while nkb > 1 and nkb_total % nkb != 0:
        nkb //= 2
    return 1, nkb, 8, 3


def _down_config(hidden: int, inter: int) -> Tuple[int, int, int, int]:
    cfg = _DOWN_CONFIG_TABLE.get((hidden, inter))
    if cfg is not None:
        return cfg
    return 4, triton.next_power_of_2(inter // _SCALE_BLOCK), 8, 2


def can_use_fp8_blockscale_moe_gemv(
    x2d: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
) -> bool:
    """Whether the M==1 decode Triton fp8-blockscale GEMV chain can serve this call."""
    if not (x2d.is_cuda and x2d.dtype == torch.bfloat16 and x2d.shape[0] == 1):
        return False
    if fc1_expert_weights.ndim != 3 or fc2_expert_weights.ndim != 3:
        return False
    num_experts, two_inter, hidden = fc1_expert_weights.shape
    inter = two_inter // 2
    if inter % _SCALE_BLOCK != 0 or hidden % _SCALE_BLOCK != 0:
        return False
    block_n1, nkb1, _, _ = _gate_up_config(inter, hidden)
    block_n2, nkb2, _, _ = _down_config(hidden, inter)
    if (
        inter % block_n1 != 0
        or (hidden // _SCALE_BLOCK) % nkb1 != 0
        or hidden % block_n2 != 0
        or nkb2 < inter // _SCALE_BLOCK
    ):
        return False
    return (
        x2d.is_contiguous()
        and fc1_expert_weights.dtype == torch.float8_e4m3fn
        and fc2_expert_weights.dtype == torch.float8_e4m3fn
        and fc1_expert_weights.is_contiguous()
        and fc2_expert_weights.is_contiguous()
        and two_inter == 2 * inter
        and hidden == x2d.shape[1]
        and hidden % _SCALE_BLOCK == 0
        and inter % _SCALE_BLOCK == 0
        and fc2_expert_weights.shape == (num_experts, hidden, inter)
        and fc1_weight_scale.dtype == torch.float32
        and fc2_weight_scale.dtype == torch.float32
        and fc1_weight_scale.is_contiguous()
        and fc2_weight_scale.is_contiguous()
        and fc1_weight_scale.shape
        == (num_experts, two_inter // _SCALE_BLOCK, hidden // _SCALE_BLOCK)
        and fc2_weight_scale.shape == (num_experts, hidden // _SCALE_BLOCK, inter // _SCALE_BLOCK)
        and selected_experts.dtype == torch.int32
        and selected_experts.is_contiguous()
        and selected_experts.numel() == selected_experts.shape[-1]
        and routing_weights.dtype == torch.float32
        and routing_weights.is_contiguous()
        and routing_weights.shape == selected_experts.shape
    )


def fp8_blockscale_moe_gemv(
    x2d: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc1_weight_scale: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc2_weight_scale: torch.Tensor,
    local_expert_offset: int,
) -> torch.Tensor:
    """Run the two-kernel fp8-blockscale MoE GEMV chain for one decode token.

    Callers must have validated the inputs with ``can_use_fp8_blockscale_moe_gemv``.
    Returns the ``[1, H]`` bf16 combined routed-expert output (pre all_reduce).
    """
    num_local_experts, two_inter, hidden = fc1_expert_weights.shape
    inter = two_inter // 2
    top_k = selected_experts.shape[-1]

    act = torch.empty((top_k, inter), dtype=torch.bfloat16, device=x2d.device)
    out = torch.empty((1, hidden), dtype=torch.bfloat16, device=x2d.device)

    block_n1, nkb1, warps1, stages1 = _gate_up_config(inter, hidden)
    _fp8_bs_gate_up_swiglu_kernel[(top_k * (inter // block_n1),)](
        x2d,
        selected_experts,
        fc1_expert_weights,
        fc1_weight_scale,
        act,
        local_expert_offset,
        num_local_experts,
        fc1_expert_weights.stride(0),
        fc1_weight_scale.stride(0),
        fc1_weight_scale.stride(1),
        INTER=inter,
        K=hidden,
        BLOCK_N=block_n1,
        NKB=nkb1,
        num_warps=warps1,
        num_stages=stages1,
    )

    block_n2, nkb2, warps2, stages2 = _down_config(hidden, inter)
    _fp8_bs_down_reduce_kernel[(hidden // block_n2,)](
        act,
        selected_experts,
        routing_weights,
        fc2_expert_weights,
        fc2_weight_scale,
        out,
        local_expert_offset,
        num_local_experts,
        fc2_expert_weights.stride(0),
        fc2_weight_scale.stride(0),
        fc2_weight_scale.stride(1),
        INTER=inter,
        TOP_K=top_k,
        BLOCK_N=block_n2,
        NKB=nkb2,
        num_warps=warps2,
        num_stages=stages2,
    )
    return out
