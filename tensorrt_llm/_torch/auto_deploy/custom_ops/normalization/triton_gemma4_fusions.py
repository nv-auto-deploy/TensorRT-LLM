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
