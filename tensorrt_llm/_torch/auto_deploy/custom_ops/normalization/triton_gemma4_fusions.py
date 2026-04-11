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

from typing import Tuple

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
# Standalone device-side benchmarks (CUDA events) show FlashInfer 1.4-2× faster
# than Triton for all T, but this is dominated by serial scheduling overhead that
# does NOT exist in CUDA graphs. In-graph, Triton's single kernel per fusion site
# beats 2-3 FlashInfer kernels due to lower graph dispatch overhead.
# Verified: threshold=0 (always FlashInfer) caused +1.4%/+0.7% regression vs
# threshold=8 at c=1/c=256 in the CUDA-graph decode path (iter92 vs iter82 baseline).
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
        # Write rmsnorm directly to out (no norm_out temp), then add residual in-place.
        # Saves 1 T×H intermediate allocation and 2T×H memory traffic vs 2-step approach.
        flashinfer.norm.rmsnorm(x_2d, weight, eps, out=out_2d)
        out_2d.add_(residual_2d)

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
        # Write rmsnorm directly to out (no norm_out temp), then add+scale in-place.
        # Eliminates 2 intermediate T×H allocations and 2T×H memory traffic vs 3-step approach.
        flashinfer.norm.rmsnorm(x_2d, weight, eps, out=out_2d)
        out_2d.add_(residual_2d)
        out_2d.mul_(scalar)

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
        # fused_add_rmsnorm(a, b): 1 kernel instead of 2 (separate add + rmsnorm).
        # Modifies in-place: b_2d ← a+b, a_2d ← rmsnorm(a+b).
        # Both a and b are consumed here and not used downstream — safe to mutate.
        flashinfer.norm.fused_add_rmsnorm(a_2d, b_2d, weight, eps)
        torch.add(a_2d, residual_2d, out=out_2d)
        out_2d.mul_(scalar)

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
        # Write rmsnorm directly to packed_flat[0], add residual in-place, then
        # write second rmsnorm directly to packed_flat[1].
        # Saves 2 T×H intermediate allocations and 3T×H memory traffic vs prior 4-op approach.
        flashinfer.norm.rmsnorm(attn_2d, post_attn_weight, eps, out=packed_flat[0])
        packed_flat[0].add_(residual_2d)
        flashinfer.norm.rmsnorm(packed_flat[0], pre_ff_weight, eps, out=packed_flat[1])

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
# Fused Q/K/V RMSNorm — replaces 3 separate torch_rmsnorm launches with 1
# ---------------------------------------------------------------------------

# Use Triton fused kernel when total QKV head-rows ≤ this value.
# At BS=8, local (N_H=16, N_KV=8): N_Q=128, N_KV=64, total=256.
# Matches _TRITON_T_THRESHOLD=8 spirit but expressed in total rows.
_QKV_TRITON_THRESHOLD = 256


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 512}, num_warps=4),
        triton.Config({"BLOCK_H": 512}, num_warps=8),
        triton.Config({"BLOCK_H": 512}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _qkv_norm_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    w_q_ptr,
    w_k_ptr,
    w_v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    N_Q,
    N_KV,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Single-kernel fused RMSNorm for Q, K, V head tensors.

    Grid: (N_Q + 2*N_KV,) — one program instance per head row.
    Programs [0, N_Q)         process q rows with w_q weights.
    Programs [N_Q, N_Q+N_KV)  process k rows with w_k weights.
    Programs [N_Q+N_KV, ...)   process v rows with w_v weights.

    Inactive tensor loads are predicated (mask=False → other=0.0 → no mem access).
    Because the masks are mutually exclusive, x = xq + xk + xv equals the active
    tensor's data for every thread. Same logic applies to the weight vector.
    OOB pointer arithmetic (negative kv_row / v_row for q/k blocks) is safe because
    CUDA predicated loads/stores never dereference the address when the mask is False.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    is_q = row < N_Q
    is_k = (row >= N_Q) & (row < N_Q + N_KV)
    is_v = ~is_q & ~is_k  # row >= N_Q + N_KV

    kv_row = row - N_Q  # index into k; may be negative for q-blocks (predicated out)
    v_row = row - N_Q - N_KV  # index into v; may be negative for q/k-blocks (predicated out)

    # Load input — exactly one of the three is non-zero for a given block
    xq = tl.load(q_ptr + row * H + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    xk = tl.load(k_ptr + kv_row * H + offs, mask=mask & is_k, other=0.0).to(tl.float32)
    xv = tl.load(v_ptr + v_row * H + offs, mask=mask & is_v, other=0.0).to(tl.float32)
    x = xq + xk + xv

    # Load weight — one of w_q, w_k, w_v; inactive tensors contribute 0 via other=0.0
    wq = tl.load(w_q_ptr + offs, mask=mask & is_q, other=0.0).to(tl.float32)
    wk = tl.load(w_k_ptr + offs, mask=mask & is_k, other=0.0).to(tl.float32)
    wv = tl.load(w_v_ptr + offs, mask=mask & is_v, other=0.0).to(tl.float32)
    w = wq + wk + wv

    # RMSNorm: normalize x, then scale by the learned weight
    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = (x * inv_rms * w).to(tl.bfloat16)

    # Store to the correct output tensor (predicated)
    tl.store(q_out_ptr + row * H + offs, normed, mask=mask & is_q)
    tl.store(k_out_ptr + kv_row * H + offs, normed, mask=mask & is_k)
    tl.store(v_out_ptr + v_row * H + offs, normed, mask=mask & is_v)


@torch.library.custom_op("auto_deploy::gemma4_qkv_norm", mutates_args=(), device_types="cuda")
def gemma4_qkv_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused per-head RMSNorm for Q, K, V: one Triton kernel instead of 3.

    Replaces three separate torch_rmsnorm calls (one per Q/K/V head group) with a
    single fused Triton kernel, saving 2 kernel-launch overheads per decoder layer.
    At 30 layers the savings are ~30µs at c=1 (30 × ~1µs avoided CG dispatch).

    Small T (total QKV rows ≤ _QKV_TRITON_THRESHOLD): fused Triton kernel.
    Large T: three separate flashinfer.norm.rmsnorm calls (bandwidth-optimal).

    K=V sharing (global layers where v = k before normalization): safe — the kernel
    reads k_ptr and v_ptr independently (both point to the same pre-norm data) and
    writes to separate output buffers using distinct weights (w_k vs w_v).
    """
    import flashinfer as _flashinfer

    H = q.shape[-1]
    q_2d = q.reshape(-1, H)
    k_2d = k.reshape(-1, H)
    v_2d = v.reshape(-1, H)
    N_Q = q_2d.shape[0]
    N_KV = k_2d.shape[0]

    q_out = torch.empty_like(q_2d)
    k_out = torch.empty_like(k_2d)
    v_out = torch.empty_like(v_2d)

    total_rows = N_Q + 2 * N_KV
    if total_rows <= _QKV_TRITON_THRESHOLD:
        _qkv_norm_kernel[(total_rows,)](
            q_2d,
            k_2d,
            v_2d,
            w_q,
            w_k,
            w_v,
            q_out,
            k_out,
            v_out,
            N_Q=N_Q,
            N_KV=N_KV,
            H=H,
            eps=eps,
        )
    else:
        _flashinfer.norm.rmsnorm(q_2d, w_q, eps, out=q_out)
        _flashinfer.norm.rmsnorm(k_2d, w_k, eps, out=k_out)
        _flashinfer.norm.rmsnorm(v_2d, w_v, eps, out=v_out)

    return q_out.reshape(q.shape), k_out.reshape(k.shape), v_out.reshape(v.shape)


@gemma4_qkv_norm.register_fake
def _gemma4_qkv_norm_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
