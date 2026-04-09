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

"""Fused Gemma4 per-layer normalization kernels.

Eliminates small kernel launches by fusing adjacent norm, add, and scale ops
into single Triton kernels. Each MoE layer has ~10 norm-adjacent kernel calls
that can be reduced to ~4.

Fused kernels:
  - gemma4_post_norm_add: norm(x, w) + residual  (post-attention)
  - gemma4_post_norm_add_scale: (norm(x, w) + residual) * s  (post-ffn)
  - gemma4_dual_norm: norm(x, w1), norm(x, w2)  (pre-ffn dual — same input)
  - gemma4_norm_add2: norm(a, wa) + norm(b, wb)  (post-moe add — different inputs)
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


# ---------------------------------------------------------------------------
# Dual-output norm: same input, two weight vectors (iter37 F1)
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
def _dual_norm_kernel(
    x_ptr,
    w1_ptr,
    w2_ptr,
    out1_ptr,
    out2_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out1[row]=rms_norm(x,w1), out2[row]=rms_norm(x,w2) sharing variance."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(x_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # Shared RMS factor — compute variance only once
    var = tl.sum(x * x) / H
    inv_rms = tl.rsqrt(var + eps)

    tl.store(out1_ptr + row * H + offs, (x * inv_rms * w1).to(tl.bfloat16), mask=mask)
    tl.store(out2_ptr + row * H + offs, (x * inv_rms * w2).to(tl.bfloat16), mask=mask)


@torch.library.custom_op("auto_deploy::gemma4_dual_norm", mutates_args=(), device_types="cuda")
def gemma4_dual_norm(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused dual-output RMSNorm: returns (norm(x,w1), norm(x,w2)).

    Replaces two separate norm calls on the same input with one kernel,
    sharing the variance computation and halving the x memory load.
    Used for pre_feedforward_layernorm and pre_feedforward_layernorm_2 which
    both normalize the same hidden_states tensor.
    """
    H = x.shape[-1]
    x_2d = x.view(-1, H)
    T_flat = x_2d.shape[0]
    out1 = torch.empty_like(x)
    out2 = torch.empty_like(x)

    _dual_norm_kernel[(T_flat,)](
        x_2d,
        weight1,
        weight2,
        out1.view(-1, H),
        out2.view(-1, H),
        H=H,
        eps=eps,
    )
    return out1, out2


@gemma4_dual_norm.register_fake
def _gemma4_dual_norm_fake(
    x: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


# ---------------------------------------------------------------------------
# norm_add2: norm(a, wa) + norm(b, wb)  (post-MoE combine, iter37 F2)
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
def _norm_add2_kernel(
    a_ptr,
    b_ptr,
    wa_ptr,
    wb_ptr,
    out_ptr,
    H: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused: out[row] = rms_norm(a[row], wa) + rms_norm(b[row], wb)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    a = tl.load(a_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    wa = tl.load(wa_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    wb = tl.load(wb_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    # RMSNorm a
    var_a = tl.sum(a * a) / H
    normed_a = a * tl.rsqrt(var_a + eps) * wa

    # RMSNorm b
    var_b = tl.sum(b * b) / H
    normed_b = b * tl.rsqrt(var_b + eps) * wb

    tl.store(out_ptr + row * H + offs, (normed_a + normed_b).to(tl.bfloat16), mask=mask)


@torch.library.custom_op("auto_deploy::gemma4_norm_add2", mutates_args=(), device_types="cuda")
def gemma4_norm_add2(
    a: torch.Tensor,
    b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused: rms_norm(a, wa) + rms_norm(b, wb).

    Replaces post_feedforward_layernorm_1(hs_dense) + post_feedforward_layernorm_2(hs_moe)
    + elementwise add with a single Triton kernel (3 kernels -> 1), saving ~4µs per layer.
    """
    H = a.shape[-1]
    out = torch.empty_like(a)
    a_2d = a.view(-1, H)
    b_2d = b.view(-1, H)
    T_flat = a_2d.shape[0]

    _norm_add2_kernel[(T_flat,)](
        a_2d,
        b_2d,
        weight_a,
        weight_b,
        out.view(-1, H),
        H=H,
        eps=eps,
    )
    return out


@gemma4_norm_add2.register_fake
def _gemma4_norm_add2_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(a)
