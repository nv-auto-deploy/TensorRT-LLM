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

"""Fused Gemma4 post-layer ops: RMSNorm + residual_add [+ layer_scalar].

These kernels eliminate 2-3 separate kernel launches per decoder layer by
fusing the post-attention and post-feedforward normalization, residual add,
and optional layer scalar multiply into single Triton kernels.

Savings at c=1 (batch=1): ~2µs saved per fused occurrence × ~12 occurrences
per decode step = ~24µs = ~1% of 2273µs baseline TPOT.

The fused kernels are:
  - gemma4_post_norm_add: norm(x, w) + residual  (post-attention pattern)
  - gemma4_post_norm_add_scale: (norm(x, w) + residual) * scalar  (post-ffn pattern)
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = rms_norm(x[row], weight, eps) + residual[row]"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # RMSNorm: normalize x then apply learned weight
    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = normed + residual
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_scale_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = (rms_norm(x[row], weight, eps) + residual[row]) * scalar"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_norm_add_and_pre_ff_norm_kernel(
    attn_out_ptr,
    residual_ptr,
    w_post_attn_ptr,
    w_pre_ff_ptr,
    out_hs_ptr,
    out_pre_ff_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused post-attention norm and pre-feedforward norm sharing one kernel launch.

    Computes two sequential RMSNorms in a single kernel:
      out_hs[row]     = rms_norm(attn_out[row], w_post_attn, eps) + residual[row]
      out_pre_ff[row] = rms_norm(out_hs[row],   w_pre_ff,    eps)

    The intermediate out_hs stays in registers between the two norms, avoiding
    a round-trip through global memory and eliminating one kernel launch per layer.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    attn_out = tl.load(attn_out_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w_post_attn_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w2 = tl.load(w_pre_ff_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # Step 1: post-attention RMSNorm + residual add → hidden_states
    var1 = tl.sum(attn_out * attn_out) / H
    inv_rms1 = tl.rsqrt(var1 + eps)
    hs = attn_out * inv_rms1 * w1 + residual

    # Step 2: pre-feedforward RMSNorm of hidden_states (in registers — no extra load)
    var2 = tl.sum(hs * hs) / H
    inv_rms2 = tl.rsqrt(var2 + eps)
    pre_ff = hs * inv_rms2 * w2

    tl.store(out_hs_ptr + row * H + offs, hs.to(tl.bfloat16), mask=mask)
    tl.store(out_pre_ff_ptr + row * H + offs, pre_ff.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_add_norm_add_scale_kernel(
    a_ptr,
    b_ptr,
    residual_ptr,
    weight_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = (rms_norm(a[row]+b[row], weight, eps) + residual[row]) * scalar.

    Combines the dense+MoE element-wise add with the subsequent post-feedforward
    norm+residual+scale, saving 1 kernel launch per MoE layer (30 layers = 30 savings).
    Safe to fuse: a and b are already synchronized (multi_stream_moe join point).
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a = tl.load(a_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    x = a + b  # dense MLP + MoE combine
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Custom op registration for torch.export / FakeTensor tracing
# ---------------------------------------------------------------------------

# Threshold: use Triton fused kernel for T ≤ this value; use flashinfer for T > this.
# At small T (≤ 8) each kernel launch costs ~200-300ns in a CUDA graph; fusing
# 2 launches into 1 saves ~250ns × 2 sites × 30 layers ≈ 15µs at c=1.
# At large T (> 8) flashinfer's vectorised RMSNorm + elementwise add outperform
# the Triton kernel (which wastes 31% of elements due to BLOCK_H=4096 > H=2816).
_TRITON_T_THRESHOLD = 8


@torch.library.custom_op("auto_deploy::gemma4_post_norm_add", mutates_args=(), device_types="cuda")
def gemma4_post_norm_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Adaptive post-attention: rms_norm(x, weight, eps) + residual.

    Dispatches to a fused Triton kernel for small T (≤ _TRITON_T_THRESHOLD) where
    reducing kernel-launch count in the CUDA graph pays off, and falls back to
    flashinfer RMSNorm + elementwise add for larger T where flashinfer's vectorised
    kernel achieves higher bandwidth utilisation.
    """
    import flashinfer

    H = x.shape[-1]
    out = torch.empty_like(x)
    x_2d = x.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = x_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_kernel[(T_flat,)](x_2d, residual_2d, weight, out_2d, H=H, eps=eps)
    else:
        torch.add(flashinfer.norm.rmsnorm(x_2d, weight, eps), residual_2d, out=out_2d)

    return out


@gemma4_post_norm_add.register_fake
def _gemma4_post_norm_add_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_norm_add_scale", mutates_args=(), device_types="cuda"
)
def gemma4_post_norm_add_scale(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """Adaptive post-feedforward: (rms_norm(x, weight, eps) + residual) * scalar.

    Same adaptive dispatch as gemma4_post_norm_add: Triton for T ≤ threshold
    (saves 2 kernel launches in CUDA graph), flashinfer + elementwise for T > threshold.
    """
    import flashinfer

    H = x.shape[-1]
    out = torch.empty_like(x)
    x_2d = x.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = x_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_scale_kernel[(T_flat,)](
            x_2d, residual_2d, weight, scalar.view(-1), out_2d, H=H, eps=eps
        )
    else:
        torch.mul(
            flashinfer.norm.rmsnorm(x_2d, weight, eps) + residual_2d,
            scalar,
            out=out_2d,
        )

    return out


@gemma4_post_norm_add_scale.register_fake
def _gemma4_post_norm_add_scale_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_add_norm_add_scale", mutates_args=(), device_types="cuda"
)
def gemma4_post_add_norm_add_scale(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """Fused MoE combine + post-norm: (rms_norm(a+b, weight, eps) + residual) * scalar.

    Merges the dense+MoE element-wise add with post_norm_add_scale into one kernel,
    saving 1 launch per MoE layer (30 layers × 1 = 30 launches at T≤8).
    Safe: a and b are already synchronized at the multi_stream_moe join point.
    """
    import flashinfer

    H = a.shape[-1]
    out = torch.empty_like(a)
    a_2d = a.view(-1, H)
    b_2d = b.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = a_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_add_norm_add_scale_kernel[(T_flat,)](
            a_2d, b_2d, residual_2d, weight, scalar.view(-1), out_2d, H=H, eps=eps
        )
    else:
        x_2d = a_2d + b_2d
        torch.mul(
            flashinfer.norm.rmsnorm(x_2d, weight, eps) + residual_2d,
            scalar,
            out=out_2d,
        )

    return out


@gemma4_post_add_norm_add_scale.register_fake
def _gemma4_post_add_norm_add_scale_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(a)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_norm_add_and_pre_ff_norm", mutates_args=(), device_types="cuda"
)
def gemma4_post_norm_add_and_pre_ff_norm(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused post-attention norm + pre-feedforward norm: saves 1 kernel launch per layer.

    Returns a packed tensor of shape (2, *attn_out.shape) where:
      packed[0] = rms_norm(attn_out, post_attn_weight, eps) + residual  (hidden_states)
      packed[1] = rms_norm(packed[0], pre_ff_weight, eps)               (pre-feedforward input)

    Dispatches to the fused Triton kernel for T ≤ _TRITON_T_THRESHOLD (decode path),
    saving one round-trip through global memory and one kernel launch per decoder layer.
    Falls back to two flashinfer rmsnorm calls for large T (prefill path).
    """
    import flashinfer

    H = attn_out.shape[-1]
    T_flat = attn_out.numel() // H
    attn_2d = attn_out.reshape(T_flat, H)
    residual_2d = residual.reshape(T_flat, H)

    # packed_flat: (2, T_flat, H) — packed[0]=hidden_states, packed[1]=pre_ff_in
    packed_flat = torch.empty(2, T_flat, H, dtype=torch.bfloat16, device=attn_out.device)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_norm_add_and_pre_ff_norm_kernel[(T_flat,)](
            attn_2d,
            residual_2d,
            post_attn_weight,
            pre_ff_weight,
            packed_flat[0],
            packed_flat[1],
            H=H,
            eps=eps,
        )
    else:
        # Use out= to write hs directly into packed_flat[0], avoiding one extra allocation
        torch.add(
            flashinfer.norm.rmsnorm(attn_2d, post_attn_weight, eps),
            residual_2d,
            out=packed_flat[0],
        )
        packed_flat[1].copy_(flashinfer.norm.rmsnorm(packed_flat[0], pre_ff_weight, eps))

    return packed_flat.reshape(2, *attn_out.shape)


@gemma4_post_norm_add_and_pre_ff_norm.register_fake
def _gemma4_post_norm_add_and_pre_ff_norm_fake(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post_attn_weight: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty(2, *attn_out.shape, dtype=torch.bfloat16, device=attn_out.device)


# ---------------------------------------------------------------------------
# Fused Q/K/V RMSNorm (iter51)
# ---------------------------------------------------------------------------
# Gemma4 attention applies separate q_norm, k_norm, v_norm kernels after the
# QKV projection.  At c=1 (decode) each is a distinct CUDA-graph node; fusing
# all three into one saves 2 graph-node replays per layer (×30 layers = 60
# fewer replays ≈ 240-360 µs improvement on the full 30-layer model).
#
# The fused kernel processes Hq + Hk + Hk rows (one per attention head) in a
# single launch.  Rows [0, Hq) use q_norm weight; [Hq, Hq+Hk) use k_norm
# weight; [Hq+Hk, Hq+2·Hk) use v_norm weight.  Output is a flat packed
# tensor so no tuple-return issues arise with CUDA-graph capture.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 512}, num_warps=2),
        triton.Config({"BLOCK_H": 512}, num_warps=4),
        triton.Config({"BLOCK_H": 512}, num_warps=8),
        triton.Config({"BLOCK_H": 1024}, num_warps=4),
        triton.Config({"BLOCK_H": 1024}, num_warps=8),
    ],
    key=["H"],
)
@triton.jit
def _qkv_norm_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,  # flat packed output [Hq+Hk+Hk, H]
    q_w_ptr,
    k_w_ptr,
    v_w_ptr,
    Hq,  # number of Q-head rows  (T * num_q_heads)
    Hk,  # number of K-head rows  (T * num_kv_heads)
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused RMSNorm for Q, K, V heads in a single kernel launch.

    Each block handles one head row of H elements.  The first Hq blocks
    normalise Q heads (weight = q_w), the next Hk blocks normalise K heads
    (weight = k_w), and the final Hk blocks normalise V heads (weight = v_w).
    Outputs are written contiguously into the packed out_ptr buffer.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    Hqk = Hq + Hk
    is_q = row < Hq
    is_k = (row >= Hq) & (row < Hqk)

    # Safe local row indices (clamp to 0 in inactive regions to avoid
    # out-of-bounds address computation in predicated loads).
    q_local = tl.where(is_q, row, 0)
    k_local = tl.where(is_k, row - Hq, 0)
    v_local = tl.where(~is_q & ~is_k, row - Hqk, 0)

    # Predicated loads: only one branch produces non-zero data per block.
    x_q = tl.load(q_ptr + q_local * H + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    x_k = tl.load(k_ptr + k_local * H + offs, mask=mask & is_k, other=0.0).to(tl.float32)
    x_v = tl.load(v_ptr + v_local * H + offs, mask=mask & ~is_q & ~is_k, other=0.0).to(tl.float32)
    x = x_q + x_k + x_v  # exactly one term is non-zero

    w_q = tl.load(q_w_ptr + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    w_k = tl.load(k_w_ptr + offs, mask=mask & is_k, other=0.0).to(tl.float32)
    # v_norm uses weight=ones (no learned scale); default other=1.0 applies the
    # identity scale for inactive regions, but only the v rows write output.
    w_v = tl.load(v_w_ptr + offs, mask=mask & ~is_q & ~is_k, other=1.0).to(tl.float32)
    w = w_q + w_k + w_v

    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    out = (x * inv_rms * w).to(tl.bfloat16)

    tl.store(out_ptr + row * H + offs, out, mask=mask)


@torch.library.custom_op("auto_deploy::gemma4_qkv_norm", mutates_args=(), device_types="cuda")
def gemma4_qkv_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused Q/K/V RMSNorm: saves 2 kernel launches per attention layer.

    Inputs (all 2-D, [total_q_heads, head_dim] etc.):
      q  [T·Hq, D]  k  [T·Hk, D]  v  [T·Hk, D]
    Output (packed):
      [T·(Hq + 2·Hk), D]  where rows 0..T·Hq-1 are q_normed,
                           rows T·Hq..T·(Hq+Hk)-1 are k_normed,
                           rows T·(Hq+Hk)..end are v_normed.

    Dispatches to the fused Triton kernel for T ≤ _TRITON_T_THRESHOLD (decode
    path).  Falls back to three separate flashinfer calls for large T.
    """
    import flashinfer

    Tq, D = q.shape
    Tk = k.shape[0]

    out = torch.empty(Tq + 2 * Tk, D, dtype=q.dtype, device=q.device)

    # T (number of tokens) = Tq / num_q_heads — but we guard on TOTAL rows.
    # For decode (c=1) Tq = num_heads ≤ 32; for prefill Tq ≫ threshold.
    if Tq <= _TRITON_T_THRESHOLD * 32:  # ≤ 8 tokens × 32 heads heuristic
        _qkv_norm_kernel[(Tq + 2 * Tk,)](q, k, v, out, q_w, k_w, v_w, Hq=Tq, Hk=Tk, H=D, eps=eps)
    else:
        flashinfer.norm.rmsnorm(q, q_w, eps, out=out[:Tq])
        flashinfer.norm.rmsnorm(k, k_w, eps, out=out[Tq : Tq + Tk])
        flashinfer.norm.rmsnorm(v, v_w, eps, out=out[Tq + Tk :])

    return out


@gemma4_qkv_norm.register_fake
def _gemma4_qkv_norm_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    Tq, D = q.shape
    Tk = k.shape[0]
    return torch.empty(Tq + 2 * Tk, D, dtype=q.dtype, device=q.device)
