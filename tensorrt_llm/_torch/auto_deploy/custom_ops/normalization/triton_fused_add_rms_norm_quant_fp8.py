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

v3 (iter31): single-pass kernel for small batch (seq_len <= _SINGLE_PASS_THRESHOLD):
  - Loads all n_cols elements into registers at once (BLOCK_N=4096, 32 warps).
  - No L2 re-read in pass2 — norm factor computed in-register, normalize inline.
  - Saves one HBM read per row vs two-pass; fewer loop iterations (1 vs 21).
  - Dispatched when seq_len <= 4 (decode path); two-pass used for larger batches.

v4 (iter33c): adaptive FlashInfer+Triton dispatch for small/medium batch:
  - For seq_len <= _RMN_ADAPTIVE_THRESHOLD (32): FlashInfer rmsnorm + _fp8_quant_only_kernel.
    Key insight: out_bf16 is ALWAYS dead code after fuse_rmsnorm_quant_fp8 transform
    (transform only fires when ALL terminal consumers are FP8 linears → bf16_node has
    zero users → eliminated by DCE). So only FP8 output and out_add need to be correct.
    Fix vs iter33b: replaced rmsnorm_quant (needs scale.item() → D2H sync → CUDA graph crash)
    with rmsnorm (no scale, CUDA graph OK) + _fp8_quant_only_kernel (reads scale as tensor
    pointer via tl.load, no .item() call, fully CUDA graph compatible).
    Non-fused: rmsnorm + fp8_quant_only (2 calls). Fused: add + rmsnorm + fp8_quant_only (3 ops).
  - For seq_len > 32: use Triton two-pass fused kernel (fewer launches at large batch).
  - Imports are lazy to avoid CUDA context init ordering issues.

v5 (iter40): fix dispatch ordering bug in fused_add_rms_norm_quant_fp8:
  - The _SINGLE_PASS_THRESHOLD check (seq_len<=4 → 1 Triton kernel) was dead code because
    _RMN_ADAPTIVE_THRESHOLD=32 fired first for ALL seq_len<=32 (including seq_len=1).
    Result: decode c=1 used 3 kernels (add+flashinfer_rmsnorm+fp8_quant) instead of 1.
  - Fix: check _SINGLE_PASS_THRESHOLD first so seq_len in {1,2,3,4} uses the single-pass
    Triton kernel (1 kernel). Combined with fuse_rmsnorm_quant_fp8 transform enabled, this
    replaces 2 kernels (flashinfer_fused_add_rms_norm + scaleMatrixPerTensorVec) with 1
    Triton kernel per matched FP8 linear, saving ~2.6μs per match at decode c=1.
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

# Single-pass kernel config for small batches (seq_len <= threshold).
# BLOCK_N_B1=4096 covers hidden_size=2688 (next_power_of_2); 32 warps for SM utilization.
_BLOCK_N_B1 = 4096
_NUM_WARPS_B1 = 32
_SINGLE_PASS_THRESHOLD = 4  # use single-pass for seq_len in {1,2,3,4}

# Adaptive threshold: for seq_len <= this value use FlashInfer CUDA kernels instead of Triton.
# FlashInfer is faster than Triton at small batch (decode path); Triton fused wins at large batch
# (fewer kernel launches than 2x FlashInfer). Matches the adaptive SSM dispatch pattern (iter28b).
_RMN_ADAPTIVE_THRESHOLD = 32

# Block size for the pure-FP8-quant kernel (flat over all elements).
# 256 elements × 2 bytes = 512 bytes per block; 8 warps for SM occupancy.
_FP8_QUANT_ONLY_BLOCK = 256
_FP8_QUANT_ONLY_WARPS = 8


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


@triton.jit
def rms_norm_quant_fp8_kernel_b1(
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
    """Single-pass RMSNorm + FP8 quant for small batch (seq_len <= 4).

    Loads all n_cols elements into registers at once (BLOCK_N=4096 covers 2688).
    No L2 re-read: norm factor computed in-register, write BF16+FP8 immediately.
    Reduces loop iterations from 21 (BLOCK_N=128) to 1, with 32 warps for utilization.
    """
    row = tl.program_id(0)
    row_off = row * n_cols
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    # Load all elements once (no second load needed)
    x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)

    # Compute RMS norm in-register (x already loaded, no L2 re-read)
    x_f32 = x.to(tl.float32)
    rrms = tl.rsqrt(tl.sum(x_f32 * x_f32) / n_cols + eps)
    inv_scale = 1.0 / tl.load(scale_ptr)

    # Normalize + quantize in one shot (data already in registers)
    normed = x_f32 * rrms * w.to(tl.float32)
    tl.store(out_bf16_ptr + row_off + cols, normed.to(tl.bfloat16), mask=mask)
    q = tl.clamp(normed * inv_scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + row_off + cols, q, mask=mask)


@triton.jit
def fused_add_rms_norm_quant_fp8_kernel_b1(
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
    """Single-pass residual-add + RMSNorm + FP8 quant for small batch (seq_len <= 4).

    All data loaded once into registers; residual add + norm + quant in one pass.
    """
    row = tl.program_id(0)
    row_off = row * n_cols
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    x = tl.load(x_ptr + row_off + cols, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row_off + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)

    add = x + r
    tl.store(out_add_ptr + row_off + cols, add, mask=mask)

    add_f32 = add.to(tl.float32)
    rrms = tl.rsqrt(tl.sum(add_f32 * add_f32) / n_cols + eps)
    inv_scale = 1.0 / tl.load(scale_ptr)

    normed = add_f32 * rrms * w.to(tl.float32)
    tl.store(out_bf16_ptr + row_off + cols, normed.to(tl.bfloat16), mask=mask)
    q = tl.clamp(normed * inv_scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + row_off + cols, q, mask=mask)


@triton.jit
def _fp8_quant_only_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    n_elems,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Pure FP8 quantization kernel: reads scale as a tensor pointer (CUDA graph compatible).

    No .item() call needed — scale is loaded inside the kernel via tl.load(scale_ptr).
    Used in the adaptive small-batch path after FlashInfer rmsnorm has already run.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elems
    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    inv_scale = 1.0 / tl.load(scale_ptr)
    q = tl.clamp(x * inv_scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_ptr + off, q, mask=mask)


def _fp8_quant_tensor(normed_flat: Tensor, scale: Tensor) -> Tensor:
    """Quantize a BF16 flat tensor to FP8 using a tensor scale (CUDA graph compatible)."""
    n_elems = normed_flat.numel()
    out_fp8 = torch.empty(normed_flat.shape, dtype=torch.float8_e4m3fn, device=normed_flat.device)
    grid = (triton.cdiv(n_elems, _FP8_QUANT_ONLY_BLOCK),)
    _fp8_quant_only_kernel[grid](
        normed_flat,
        out_fp8,
        scale,
        n_elems,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        BLOCK=_FP8_QUANT_ONLY_BLOCK,
        num_warps=_FP8_QUANT_ONLY_WARPS,
    )
    return out_fp8


def rms_norm_quant_fp8(
    hidden_states: Tensor, weight: Tensor, eps: float, scale: Tensor
) -> Tuple[Tensor, Tensor]:
    assert hidden_states.shape[-1] == weight.numel(), "hidden size must match weight size"

    orig_shape = hidden_states.shape
    feat_size = weight.shape[0]
    hidden_states_flat = hidden_states.reshape(-1, feat_size)
    seq_len = hidden_states_flat.shape[0]

    if seq_len <= _RMN_ADAPTIVE_THRESHOLD:
        # Small/medium batch: FlashInfer rmsnorm + _fp8_quant_only_kernel.
        # NOTE: iter43 added mixed-consumer support, so bf16_node may have non-FP8 users
        # (e.g. MoE router). Return normed_flat as out_bf16 — already computed, zero extra cost.
        # We split into 2 kernels to avoid scale.item() (which triggers a D2H sync forbidden during
        # CUDA graph capture): flashinfer.norm.rmsnorm needs no scale (CUDA graph OK), then
        # _fp8_quant_only_kernel reads scale via tl.load(scale_ptr) with no .item() call.
        # Lazy import to avoid CUDA context init ordering issues (same pattern as adaptive SSM).
        import flashinfer.norm

        normed_flat = flashinfer.norm.rmsnorm(hidden_states_flat, weight, eps)
        out_fp8 = _fp8_quant_tensor(normed_flat, scale)
        return normed_flat.reshape(orig_shape), out_fp8.reshape(orig_shape)

    out_bf16 = torch.empty_like(hidden_states_flat)
    out_fp8 = torch.empty(
        hidden_states_flat.shape,
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )

    grid = (seq_len,)
    if seq_len <= _SINGLE_PASS_THRESHOLD:
        # Single-pass: all elements in registers, no L2 re-read
        rms_norm_quant_fp8_kernel_b1[grid](
            hidden_states_flat,
            weight,
            out_bf16,
            out_fp8,
            scale,
            n_cols=feat_size,
            eps=eps,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            BLOCK_N=_BLOCK_N_B1,
            num_warps=_NUM_WARPS_B1,
        )
    else:
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
    if seq_len <= _SINGLE_PASS_THRESHOLD:
        # Very small batch (seq_len in {1,2,3,4}): single-pass Triton kernel (1 kernel total).
        # All hidden_size elements loaded into registers once; add + norm + quant in one pass.
        # This path is fastest for decode (c=1) — replaces the old 3-kernel adaptive path that
        # was unreachable here (it ran for seq_len <= 32 before this check, making this dead).
        # Fix (iter40): moved _SINGLE_PASS_THRESHOLD check before _RMN_ADAPTIVE_THRESHOLD so
        # seq_len=1 uses 1 Triton kernel instead of 3 kernels (add + flashinfer_rmsnorm + quant).
        fused_add_rms_norm_quant_fp8_kernel_b1[grid](
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
            BLOCK_N=_BLOCK_N_B1,
            num_warps=_NUM_WARPS_B1,
        )
    elif seq_len <= _RMN_ADAPTIVE_THRESHOLD:
        # Medium batch (5 <= seq_len <= 32): add + FlashInfer rmsnorm + fp8_quant (3 kernels).
        # NOTE: iter43 added mixed-consumer support, so bf16_node may have non-FP8 users
        # (e.g. MoE router). Return normed_flat as out_bf16 — already computed, zero extra cost.
        # out_add IS used (wired to next layer's residual input) so must be correct.
        # Split to avoid scale.item() CUDA graph crash (same fix as non-fused path above):
        # elementwise add (1 kernel) + flashinfer.norm.rmsnorm (no scale, CUDA graph OK) +
        # _fp8_quant_only_kernel (reads scale via tl.load, CUDA graph OK) = 3 kernels total.
        import flashinfer.norm

        out_add_flat = x_flat + residual_flat  # elementwise add → out_add = x + residual
        normed_flat = flashinfer.norm.rmsnorm(out_add_flat, weight, eps)
        out_fp8_med = _fp8_quant_tensor(normed_flat, scale)
        return (
            normed_flat.reshape(
                orig_shape
            ),  # iter46 fix: was out_bf16 (uninitialized) — MoE router needs correct bf16
            out_fp8_med.reshape(orig_shape),
            out_add_flat.reshape(orig_shape),
        )
    else:
        # Large batch (seq_len > 32): 2-pass Triton kernel (1 kernel total).
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
