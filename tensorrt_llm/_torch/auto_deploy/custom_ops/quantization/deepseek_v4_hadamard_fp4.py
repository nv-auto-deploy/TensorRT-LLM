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

"""Fused Hadamard rotate + fake-FP4 activation quant for DeepSeek-V4.

Replaces ``fake_fp4_act_quant(hadamard_rotate(x), block_size)`` — log2(dim)
butterfly stages of reshape/add/sub/cat plus the quant's
abs/amax/log2/ceil/exp2/clamp/where ladder — with a single Triton kernel that
keeps the whole row in fp32 registers. Bit-identical to the eager reference
(``test_deepseek_v4_hadamard_fp4.py``), including the reference's bf16
round-trip between rotate and quant.

The butterfly is unrolled via ``if DIM >= ...`` constexpr guards: Triton loop
variables are runtime values, which would make the per-stage reshape shapes
non-constexpr.
"""

import torch
import triton
import triton.language as tl

# FP4 (e2m1) fake-quant constants from the reference recipe.
_FP4_MAX = 6.0
_FP4_MIN = 6.0 * 2.0**-126


@triton.jit
def _hadamard_stage(x, ROWS: tl.constexpr, DIM: tl.constexpr, G: tl.constexpr, W: tl.constexpr):
    """One butterfly stage, pair-stride ``W``, on a ``[ROWS, DIM]`` tile: view as
    ``[ROWS * G, 2, W]``; pair-axis index 0 becomes ``l + r``, index 1 ``l - r``."""
    a = tl.reshape(x, (ROWS * G, 2, W))
    sw = tl.flip(a, 1)
    lower = (tl.arange(0, 2) == 0)[None, :, None]  # [1, 2, 1]
    a = tl.where(lower, a + sw, sw - a)
    return tl.reshape(a, (ROWS, DIM))


@triton.jit
def _hadamard_fp4_kernel(
    x_ptr,  # [R, DIM] contiguous input
    out_ptr,  # [R, DIM] contiguous output
    R,
    BLOCK_R: tl.constexpr,  # rows per program
    HAS_TAIL: tl.constexpr,  # last program has out-of-range rows to mask off
    DIM: tl.constexpr,  # power-of-two row width (== Hadamard dim)
    BLOCK_SIZE: tl.constexpr,  # fp4 quant group size (DIM % BLOCK_SIZE == 0)
    NB: tl.constexpr,  # DIM // BLOCK_SIZE
    INV_SQRT_DIM: tl.constexpr,  # DIM ** -0.5
    FP4_MAX: tl.constexpr,
    FP4_MIN: tl.constexpr,
):
    # int64 rows: offsets overflow int32 once x.numel() >= 2**31.
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    offs = rows[:, None] * DIM + tl.arange(0, DIM)[None, :]  # [BLOCK_R, DIM]
    if HAS_TAIL:
        mask = rows[:, None] < R
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        x = tl.load(x_ptr + offs).to(tl.float32)

    # Walsh-Hadamard butterfly, fp32, unrolled: stage s has pair-stride 2**s.
    if DIM >= 2:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 2, 1)
    if DIM >= 4:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 4, 2)
    if DIM >= 8:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 8, 4)
    if DIM >= 16:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 16, 8)
    if DIM >= 32:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 32, 16)
    if DIM >= 64:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 64, 32)
    if DIM >= 128:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 128, 64)
    if DIM >= 256:
        x = _hadamard_stage(x, BLOCK_R, DIM, DIM // 256, 128)
    tl.static_assert(DIM <= 256, "deepseek_v4_hadamard_fp4 supports DIM <= 256")
    x = x * INV_SQRT_DIM

    # Replicate the reference's dtype round-trip between rotate and quant.
    x = x.to(out_ptr.dtype.element_ty).to(tl.float32)

    # Block fake-FP4 quant.
    xb = tl.reshape(x, (BLOCK_R * NB, BLOCK_SIZE))
    amax = tl.max(tl.abs(xb), axis=1, keep_dims=True)  # [BLOCK_R * NB, 1]
    # ceil_pow2_scale(amax, FP4_MAX, FP4_MIN)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, FP4_MIN) / FP4_MAX)))
    n = tl.minimum(tl.maximum(xb / scale, -FP4_MAX), FP4_MAX)  # clamp(.,-6,6)
    an = tl.abs(n)
    q = tl.zeros_like(an)
    q = tl.where(an > 0.25, 0.5, q)
    q = tl.where(an > 0.75, 1.0, q)
    q = tl.where(an > 1.25, 1.5, q)
    q = tl.where(an > 1.75, 2.0, q)
    q = tl.where(an > 2.5, 3.0, q)
    q = tl.where(an > 3.5, 4.0, q)
    q = tl.where(an > 5.0, 6.0, q)
    sign = tl.where(n > 0, 1.0, 0.0) - tl.where(n < 0, 1.0, 0.0)
    res = (q * sign) * scale

    res = tl.reshape(res, (BLOCK_R, DIM)).to(out_ptr.dtype.element_ty)
    if HAS_TAIL:
        tl.store(out_ptr + offs, res, mask=mask)
    else:
        tl.store(out_ptr + offs, res)


# Microbenched: below this row count BLOCK_R=1 wins (decode is latency-bound);
# at/above it BLOCK_R=2 wins (-5% at R=1024 to -35% at R=8000). BLOCK_R>2
# over-subscribes registers at num_warps=1.
_BLOCKED_ROW_THRESHOLD = 1024
_BLOCK_R = 2


@torch.library.custom_op("auto_deploy::deepseek_v4_hadamard_fp4", mutates_args=())
def deepseek_v4_hadamard_fp4(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Fused Hadamard rotate + fake-FP4 block-quant round-trip.

    ``x`` has shape ``[..., dim]`` with ``dim`` a power of two (<= 256) and a
    multiple of ``block_size``. The last dim must be contiguous. Returns a tensor
    of the same shape and dtype as ``x``.
    """
    dim = x.shape[-1]
    assert dim > 1 and (dim & (dim - 1)) == 0 and dim <= 256, (
        "hadamard dim must be a power of two in (1, 256]"
    )
    assert dim % block_size == 0, "dim must be a multiple of block_size"
    assert x.stride(-1) == 1, "last dim of x must be contiguous"

    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    R = x.numel() // dim
    if R == 0:
        return out

    x2 = x.reshape(R, dim)
    out2 = out.reshape(R, dim)

    block_r = _BLOCK_R if R >= _BLOCKED_ROW_THRESHOLD else 1
    # num_warps=1: one warp covers a DIM<=256 row with intra-warp reductions;
    # extra warps only add smem barriers. Loop-free kernel, so num_stages is inert.
    _hadamard_fp4_kernel[(triton.cdiv(R, block_r),)](
        x2,
        out2,
        R,
        BLOCK_R=block_r,
        HAS_TAIL=(R % block_r != 0),
        DIM=dim,
        BLOCK_SIZE=block_size,
        NB=dim // block_size,
        INV_SQRT_DIM=float(dim**-0.5),
        FP4_MAX=_FP4_MAX,
        FP4_MIN=_FP4_MIN,
        num_warps=1,
        num_stages=1,
    )
    return out


@deepseek_v4_hadamard_fp4.register_fake
def _deepseek_v4_hadamard_fp4_fake(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)
