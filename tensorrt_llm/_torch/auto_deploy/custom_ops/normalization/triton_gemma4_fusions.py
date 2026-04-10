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
    """Fused post-attention norm and pre-feedforward norm in one kernel launch.

    Computes two sequential ops sharing one kernel:
      out_hs[row]     = rms_norm(attn_out[row], w_post_attn, eps) + residual[row]
      out_pre_ff[row] = rms_norm(out_hs[row], w_pre_ff, eps)   [dense MLP input]

    out_hs stays in registers between steps, eliminating one memory round-trip.
    Eliminates 1 kernel launch per MoE layer vs baseline.
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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_moe_norm_add_norm_add_scale_kernel(
    a_ptr,
    b_raw_ptr,
    residual_ptr,
    w_moe_ln_ptr,
    weight_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = (rms_norm(a[row] + rms_norm(b_raw[row], w_moe_ln, eps), weight, eps) + residual[row]) * scalar.

    Extends post_add_norm_add_scale by also fusing post_feedforward_layernorm_2 (applied to
    the raw MoE output b_raw) into the same kernel.  Two sequential RMSNorm reductions
    are computed in registers before any stores:
      Step 1: b_normed = rms_norm(b_raw, w_moe_ln, eps)   [post_feedforward_layernorm_2]
      Step 2: x = a + b_normed
      Step 3: out = (rms_norm(x, weight, eps) + residual) * scalar

    Eliminates 1 kernel launch per MoE layer (N_MoE layers × ~14µs ≈ ~84µs ≈ 3.9%).
    Two reductions have the same register footprint as post_norm_add_and_pre_ff_norm.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a = tl.load(a_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b_raw = tl.load(b_raw_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w_moe = tl.load(w_moe_ln_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # Step 1: post_feedforward_layernorm_2 on raw MoE output (stays in registers)
    var_moe = tl.sum(b_raw * b_raw) / H
    inv_rms_moe = tl.rsqrt(var_moe + eps)
    b_normed = b_raw * inv_rms_moe * w_moe

    # Step 2: dense + normed-MoE combine
    x = a + b_normed

    # Step 3: post_feedforward_layernorm + residual add + layer scalar
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
    "auto_deploy::gemma4_post_moe_norm_add_norm_add_scale", mutates_args=(), device_types="cuda"
)
def gemma4_post_moe_norm_add_norm_add_scale(
    a: torch.Tensor,
    b_raw: torch.Tensor,
    residual: torch.Tensor,
    w_moe_ln: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """Fused MoE post-norm + combine + post-norm + scale: saves 1 kernel launch per MoE layer.

    Computes: (rms_norm(a + rms_norm(b_raw, w_moe_ln, eps), weight, eps) + residual) * scalar

    Fuses post_feedforward_layernorm_2 (on raw MoE output b_raw) into the combine kernel,
    eliminating 1 separate kernel launch per MoE layer.  Two sequential RMSNorm reductions
    share the same kernel, same register file — same pattern as post_norm_add_and_pre_ff_norm.

    Dispatches to the fused Triton kernel for T ≤ _TRITON_T_THRESHOLD (decode path).
    Falls back to flashinfer rmsnorm + elementwise ops for large T (prefill path).
    """
    import flashinfer

    H = a.shape[-1]
    out = torch.empty_like(a)
    a_2d = a.view(-1, H)
    b_raw_2d = b_raw.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = a_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_moe_norm_add_norm_add_scale_kernel[(T_flat,)](
            a_2d, b_raw_2d, residual_2d, w_moe_ln, weight, scalar.view(-1), out_2d, H=H, eps=eps
        )
    else:
        b_normed_2d = flashinfer.norm.rmsnorm(b_raw_2d, w_moe_ln, eps)
        x_2d = a_2d + b_normed_2d
        torch.mul(
            flashinfer.norm.rmsnorm(x_2d, weight, eps) + residual_2d,
            scalar,
            out=out_2d,
        )

    return out


@gemma4_post_moe_norm_add_norm_add_scale.register_fake
def _gemma4_post_moe_norm_add_norm_add_scale_fake(
    a: torch.Tensor,
    b_raw: torch.Tensor,
    residual: torch.Tensor,
    w_moe_ln: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(a)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 4096}, num_warps=4),
        triton.Config({"BLOCK_H": 4096}, num_warps=8),
        triton.Config({"BLOCK_H": 4096}, num_warps=16),
    ],
    key=["H"],
)
@triton.jit
def _post_dense_moe_norm_add_norm_add_scale_kernel(
    a_raw_ptr,
    b_raw_ptr,
    residual_ptr,
    w_dense_ln_ptr,
    w_moe_ln_ptr,
    weight_ptr,
    scalar_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: 3-reduction combine kernel that also applies post_feedforward_layernorm_1 and _2.

    Computes: (rms_norm(rms_norm(a_raw, w_dense_ln) + rms_norm(b_raw, w_moe_ln), weight) + residual) * scalar

    Three sequential RMSNorm reductions all computed in the same register file:
      Step 1: a_normed = rms_norm(a_raw, w_dense_ln)   [post_feedforward_layernorm_1]
      Step 2: b_normed = rms_norm(b_raw, w_moe_ln)     [post_feedforward_layernorm_2]
      Step 3: x = a_normed + b_normed
      Step 4: out = (rms_norm(x, weight) + residual) * scalar

    Eliminates post_feedforward_layernorm_1 + post_feedforward_layernorm_2 from the graph
    (2 kernel launches per MoE layer → ~28µs / layer, ~168µs total for 6 MoE layers).
    H100 register budget: 3 weight arrays + 3 data arrays = 6 × 32 fp32/thread ≈ 192 regs,
    comfortably within the 256-register-per-thread limit.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a_raw = tl.load(a_raw_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b_raw = tl.load(b_raw_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w_dense = tl.load(w_dense_ln_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w_moe = tl.load(w_moe_ln_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)

    # Step 1: post_feedforward_layernorm_1 on dense MLP output
    var_dense = tl.sum(a_raw * a_raw) / H
    inv_rms_dense = tl.rsqrt(var_dense + eps)
    a_normed = a_raw * inv_rms_dense * w_dense

    # Step 2: post_feedforward_layernorm_2 on raw MoE output
    var_moe = tl.sum(b_raw * b_raw) / H
    inv_rms_moe = tl.rsqrt(var_moe + eps)
    b_normed = b_raw * inv_rms_moe * w_moe

    # Step 3: combine
    x = a_normed + b_normed

    # Step 4: post_feedforward_layernorm + residual add + layer scalar
    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)
    normed = x * inv_rms * weight

    out = (normed + residual) * scalar
    tl.store(out_ptr + row * H + offs, out.to(tl.bfloat16), mask=mask)


@torch.library.custom_op(
    "auto_deploy::gemma4_post_dense_moe_norm_add_norm_add_scale",
    mutates_args=(),
    device_types="cuda",
)
def gemma4_post_dense_moe_norm_add_norm_add_scale(
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    residual: torch.Tensor,
    w_dense_ln: torch.Tensor,
    w_moe_ln: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    """Fused 3-reduction combine: applies post_ff_ln_1 + post_ff_ln_2 + combine + post_ff_ln + scale.

    Saves 2 kernel launches per MoE layer vs iter47 baseline (post_ff_ln_1 + post_ff_ln_2).

    Dispatches to the fused Triton kernel for T ≤ _TRITON_T_THRESHOLD (decode path).
    Falls back to flashinfer rmsnorm + elementwise ops for large T (prefill path).
    """
    import flashinfer

    H = a_raw.shape[-1]
    out = torch.empty_like(a_raw)
    a_raw_2d = a_raw.view(-1, H)
    b_raw_2d = b_raw.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = a_raw_2d.shape[0]
    out_2d = out.view(-1, H)

    if T_flat <= _TRITON_T_THRESHOLD:
        _post_dense_moe_norm_add_norm_add_scale_kernel[(T_flat,)](
            a_raw_2d,
            b_raw_2d,
            residual_2d,
            w_dense_ln,
            w_moe_ln,
            weight,
            scalar.view(-1),
            out_2d,
            H=H,
            eps=eps,
        )
    else:
        a_normed_2d = flashinfer.norm.rmsnorm(a_raw_2d, w_dense_ln, eps)
        b_normed_2d = flashinfer.norm.rmsnorm(b_raw_2d, w_moe_ln, eps)
        x_2d = a_normed_2d + b_normed_2d
        torch.mul(
            flashinfer.norm.rmsnorm(x_2d, weight, eps) + residual_2d,
            scalar,
            out=out_2d,
        )

    return out


@gemma4_post_dense_moe_norm_add_norm_add_scale.register_fake
def _gemma4_post_dense_moe_norm_add_norm_add_scale_fake(
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    residual: torch.Tensor,
    w_dense_ln: torch.Tensor,
    w_moe_ln: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scalar: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(a_raw)


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
    """Fused post-attention norm + pre-feedforward norm: saves 1 kernel launch per MoE layer.

    Returns a packed tensor of shape (2, *attn_out.shape) where:
      packed[0] = rms_norm(attn_out, post_attn_weight, eps) + residual  (hidden_states)
      packed[1] = rms_norm(packed[0], pre_ff_weight, eps)               (dense MLP input)

    packed[0] stays in registers between steps, eliminating one memory round-trip.

    Dispatches to the fused Triton kernel for T ≤ _TRITON_T_THRESHOLD (decode path).
    Falls back to two flashinfer rmsnorm calls for large T (prefill path).
    """
    import flashinfer

    H = attn_out.shape[-1]
    T_flat = attn_out.numel() // H
    attn_2d = attn_out.reshape(T_flat, H)
    residual_2d = residual.reshape(T_flat, H)

    # packed_flat: (2, T_flat, H)
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
