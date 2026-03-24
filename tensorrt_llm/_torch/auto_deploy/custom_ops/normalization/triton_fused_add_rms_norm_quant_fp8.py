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

"""Triton RMSNorm + FP8 quantization custom ops.

v2 kernel improvements over the original single-tile approach:
  - Two-pass streaming: pass1 accumulates x² with no register pressure;
    pass2 normalizes+quantizes with num_stages=2 prefetch hiding HBM latency.
  - BLOCK_N=128 (divides Nemotron hidden_size=2688 exactly: 2688=128×21)
    vs. prior next_power_of_2(2688)=4096 which wasted 35% of compute.
  - tl.rsqrt (hardware instruction) instead of 1/tl.sqrt.
  - 1/scale precomputed → multiply instead of divide per element.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl
from torch import Tensor

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)

# Tuned for Nemotron hidden_size=2688=128×21: zero waste, clean loop.
# Fall back to 256 for hidden sizes that are multiples of 256 (e.g. 7168=256×28).
_BLOCK_N = 128
_NUM_WARPS = 4


@triton.jit
def rms_norm_quant_fp8_kernel(
    x_ptr,
    w_ptr,
    out_bf16_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_cols,
    eps: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Two-pass streaming RMSNorm + FP8 quant.

    Pass 1: accumulate x² in tiles → compute rrms (no large register footprint).
    Pass 2: prefetch next tile (num_stages=2) while writing normed BF16 + FP8.
    """
    row = tl.program_id(0)
    row_off = row * n_cols

    # ---- Pass 1: variance accumulation ----
    acc = tl.zeros([1], dtype=tl.float32)
    for col_start in tl.range(0, n_cols, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x)

    rrms = tl.rsqrt(acc / n_cols + eps)  # hardware rsqrt: faster than 1/sqrt

    # Pre-invert scale: turn per-element division into multiplication
    inv_scale = 1.0 / tl.load(scale_ptr)

    # ---- Pass 2: normalize + gamma + quantize (num_stages=2 prefetches ahead) ----
    for col_start in tl.range(0, n_cols, BLOCK_N, num_stages=2):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_off + cols, mask=mask)  # L2 hit from pass1
        w = tl.load(w_ptr + cols, mask=mask)
        normed = x.to(tl.float32) * rrms * w.to(tl.float32)
        # BF16 output for the residual connection
        tl.store(out_bf16_ptr + row_off + cols, normed.to(tl.bfloat16), mask=mask)
        # FP8 output for the downstream linear layer
        q = tl.clamp(normed * inv_scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + row_off + cols, q, mask=mask)


def rms_norm_quant_fp8(
    hidden_states: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor]:
    assert hidden_states.shape[-1] == weight.numel(), "hidden size must match weight size"

    orig_shape = hidden_states.shape
    feat_size = weight.shape[0]
    hidden_states_flat = hidden_states.reshape(-1, feat_size)
    seq_len = hidden_states_flat.shape[0]

    out_bf16 = torch.empty_like(hidden_states_flat)
    out_fp8 = torch.empty(
        hidden_states_flat.shape,
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )

    grid = (seq_len,)
    rms_norm_quant_fp8_kernel[grid](
        hidden_states_flat,
        weight,
        out_bf16,
        out_fp8,
        scale,
        n_cols=feat_size,
        eps=eps,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    return out_bf16.reshape(orig_shape), out_fp8.reshape(orig_shape)


@torch.library.custom_op("auto_deploy::triton_rms_norm_quant_fp8", mutates_args=())
def triton_rms_norm_quant_fp8(
    input: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return rms_norm_quant_fp8(input, weight, eps, scale)


@triton_rms_norm_quant_fp8.register_fake
def _rms_norm_quant_fp8_fake(
    input: torch.Tensor, weight: torch.Tensor, eps: float, scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    bf16_out = torch.empty_like(input)
    fp8_out = torch.empty(input.shape, dtype=torch.float8_e4m3fn, device=input.device)
    return bf16_out, fp8_out


@triton.jit
def fused_add_rms_norm_quant_fp8_kernel(
    x_ptr,
    residual_ptr,
    w_ptr,
    out_bf16_ptr,
    out_fp8_ptr,
    out_add_ptr,
    scale_ptr,
    n_cols,
    eps: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Two-pass: residual add → variance → normalize+gamma+quant.

    Pass 1: stream x+residual tiles to compute RMS (stores add_out to HBM).
    Pass 2: re-stream add_out (L2 hot) to normalize, write BF16 + FP8.
    """
    row = tl.program_id(0)
    row_off = row * n_cols

    # ---- Pass 1: residual add + variance accumulation ----
    acc = tl.zeros([1], dtype=tl.float32)
    for col_start in tl.range(0, n_cols, BLOCK_N):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
        r = tl.load(residual_ptr + row_off + cols, mask=mask, other=0.0)
        add = x + r
        tl.store(out_add_ptr + row_off + cols, add, mask=mask)  # for next residual
        add_f32 = add.to(tl.float32)
        acc += tl.sum(add_f32 * add_f32)

    rrms = tl.rsqrt(acc / n_cols + eps)
    inv_scale = 1.0 / tl.load(scale_ptr)

    # ---- Pass 2: normalize + gamma + quantize (add_out hot in L2/L1) ----
    for col_start in tl.range(0, n_cols, BLOCK_N, num_stages=2):
        cols = col_start + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        add = tl.load(out_add_ptr + row_off + cols, mask=mask)  # L2 hit
        w = tl.load(w_ptr + cols, mask=mask)
        normed = add.to(tl.float32) * rrms * w.to(tl.float32)
        tl.store(out_bf16_ptr + row_off + cols, normed.to(tl.bfloat16), mask=mask)
        q = tl.clamp(normed * inv_scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + row_off + cols, q, mask=mask)


def fused_add_rms_norm_quant_fp8(
    x: Tensor, residual: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    assert x.shape == residual.shape, "x and residual must have identical shape"
    assert x.shape[-1] == weight.numel(), "x hidden size must match weight size"

    orig_shape = x.shape
    feat_size = weight.shape[0]
    x_flat = x.reshape(-1, feat_size)
    residual_flat = residual.reshape(-1, feat_size)
    seq_len = x_flat.shape[0]

    out_bf16 = torch.empty_like(x_flat)
    out_fp8 = torch.empty(x_flat.shape, dtype=torch.float8_e4m3fn, device=x.device)
    out_add = torch.empty_like(x_flat)

    grid = (seq_len,)
    fused_add_rms_norm_quant_fp8_kernel[grid](
        x_flat,
        residual_flat,
        weight,
        out_bf16,
        out_fp8,
        out_add,
        scale,
        n_cols=feat_size,
        eps=eps,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        BLOCK_N=_BLOCK_N,
        num_warps=_NUM_WARPS,
    )

    return out_bf16.reshape(orig_shape), out_fp8.reshape(orig_shape), out_add.reshape(orig_shape)


@torch.library.custom_op("auto_deploy::triton_fused_add_rms_norm_quant_fp8", mutates_args=())
def triton_fused_add_rms_norm_quant_fp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_add_rms_norm_quant_fp8(x, residual, weight, eps, scale)


@triton_fused_add_rms_norm_quant_fp8.register_fake
def _fused_add_rms_norm_quant_fp8_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bf16_out = torch.empty_like(x)
    fp8_out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    add_out = torch.empty_like(x)
    return bf16_out, fp8_out, add_out
