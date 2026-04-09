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


@torch.library.custom_op("auto_deploy::gemma4_post_norm_add", mutates_args=(), device_types="cuda")
def gemma4_post_norm_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused post-attention: rms_norm(x, weight, eps) + residual.

    Replaces two separate kernel calls (flashinfer RMSNorm + elementwise add)
    with a single fused Triton kernel, saving ~1.5-2µs per occurrence.
    """
    H = x.shape[-1]
    out = torch.empty_like(x)
    # Reshape to 2D [T_flat, H] for row-parallel kernel
    x_2d = x.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = x_2d.shape[0]
    out_2d = out.view(-1, H)

    _post_norm_add_kernel[(T_flat,)](
        x_2d,
        residual_2d,
        weight,
        out_2d,
        H=H,
        eps=eps,
    )
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
    """Fused post-feedforward: (rms_norm(x, weight, eps) + residual) * scalar.

    Replaces three separate kernel calls (RMSNorm + residual add + scalar
    multiply) with a single fused Triton kernel, saving ~2.5-3µs per
    occurrence per layer.
    """
    H = x.shape[-1]
    out = torch.empty_like(x)
    x_2d = x.view(-1, H)
    residual_2d = residual.view(-1, H)
    T_flat = x_2d.shape[0]
    out_2d = out.view(-1, H)

    # scalar is a 1-element tensor [1]; load its pointer into the kernel
    scalar_flat = scalar.view(-1)

    _post_norm_add_scale_kernel[(T_flat,)](
        x_2d,
        residual_2d,
        weight,
        scalar_flat,
        out_2d,
        H=H,
        eps=eps,
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
