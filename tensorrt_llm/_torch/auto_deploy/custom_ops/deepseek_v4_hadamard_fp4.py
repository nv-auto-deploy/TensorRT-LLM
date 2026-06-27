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

"""Fused Hadamard-rotate + fake-FP4 activation-quant custom op for DeepSeek-V4.

The DeepSeek-V4 indexer quantizes its query (and the rotated compressor output)
with ``fake_fp4_act_quant(hadamard_rotate(x), block_size=32)`` at two sites in
``modeling_deepseek_v4.py``. In eager/decomposed form that composite is a swarm
of tiny launch-bound kernels per call:

* ``hadamard_rotate`` is a ``log2(dim)``-stage Walsh-Hadamard butterfly. Each
  stage emits an ``add``, a ``sub`` and a ``cat`` (``CatArrayBatchedCopy``), plus
  a leading ``.float()`` and a trailing ``* dim**-0.5`` / ``.to(dtype)`` — ~15
  ``elementwise`` + ~9 ``copy_cast`` kernels for ``dim=128`` (7 stages).
* ``fake_fp4_act_quant`` is a ~15-op chain: ``.float()``, per-block ``abs``/
  ``amax`` (a ``reduce``), ``ceil_pow2_scale`` (clamp/div/log2/ceil/exp2),
  ``clamp``, ``abs``, a 7-level ``torch.where`` ladder, ``sign``, two ``mul`` and
  a final ``.to(dtype)`` — ~18 ``elementwise`` + 1 ``reduction`` + 2 ``copy_cast``.

This op collapses the whole ~30-40-kernel chain into a *single* Triton kernel:
one program per ``dim``-length row loads the row, runs the Hadamard butterfly and
the FP4 quant entirely in registers (fp32), and stores the result. The fp32 math
mirrors the reference exactly (including the intermediate ``bf16`` round-trip the
reference incurs between ``hadamard_rotate``'s ``.to(dtype)`` and
``fake_fp4_act_quant``'s ``.float()``), so the result is bit-identical.

Triton (3.x) loop induction variables are runtime tensors, so the butterfly is
unrolled with compile-time ``if DIM >= 2**(s+1)`` guards calling a ``constexpr``
stage helper — that keeps the per-stage reshape shapes ``constexpr[int]``.
"""

import torch
import triton
import triton.language as tl

# FP4 (e2m1) quant constants — must match
# ``utils/quantization_utils.fake_fp4_act_quant`` / ``ceil_pow2_scale``.
_FP4_MAX = 6.0
_FP4_MIN = 6.0 * 2.0**-126


@triton.jit
def _hadamard_stage(x, DIM: tl.constexpr, G: tl.constexpr, W: tl.constexpr):
    """One Walsh-Hadamard butterfly stage with pair-stride ``W`` on a [DIM] row.

    View the row as ``[G, 2, W]`` (``G = DIM // (2*W)``); the lower half
    (pair-axis index 0) becomes ``left + right`` and the upper half becomes
    ``left - right``. ``tl.flip`` along the pair axis swaps the two halves so a
    single ``where`` produces both.
    """
    a = tl.reshape(x, (G, 2, W))
    sw = tl.flip(a, 1)
    lower = (tl.arange(0, 2) == 0)[None, :, None]  # [1, 2, 1]
    a = tl.where(lower, a + sw, sw - a)
    return tl.reshape(a, (DIM,))


@triton.jit
def _hadamard_fp4_kernel(
    x_ptr,  # [R, DIM] contiguous input
    out_ptr,  # [R, DIM] contiguous output
    R,
    DIM: tl.constexpr,  # power-of-two row width (== Hadamard dim)
    BLOCK_SIZE: tl.constexpr,  # fp4 quant group size (DIM % BLOCK_SIZE == 0)
    NB: tl.constexpr,  # DIM // BLOCK_SIZE
    INV_SQRT_DIM: tl.constexpr,  # DIM ** -0.5
    FP4_MAX: tl.constexpr,
    FP4_MIN: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= R:
        return

    offs = tl.arange(0, DIM)
    x = tl.load(x_ptr + row * DIM + offs).to(tl.float32)  # [DIM]

    # --- Walsh-Hadamard butterfly (Sylvester order), fp32, unrolled stages ---
    # Stage s has pair-stride W = 2**s, s = 0 .. log2(DIM)-1.
    if DIM >= 2:
        x = _hadamard_stage(x, DIM, DIM // 2, 1)
    if DIM >= 4:
        x = _hadamard_stage(x, DIM, DIM // 4, 2)
    if DIM >= 8:
        x = _hadamard_stage(x, DIM, DIM // 8, 4)
    if DIM >= 16:
        x = _hadamard_stage(x, DIM, DIM // 16, 8)
    if DIM >= 32:
        x = _hadamard_stage(x, DIM, DIM // 32, 16)
    if DIM >= 64:
        x = _hadamard_stage(x, DIM, DIM // 64, 32)
    if DIM >= 128:
        x = _hadamard_stage(x, DIM, DIM // 128, 64)
    if DIM >= 256:
        x = _hadamard_stage(x, DIM, DIM // 256, 128)
    tl.static_assert(DIM <= 256, "deepseek_v4_hadamard_fp4 supports DIM <= 256")
    x = x * INV_SQRT_DIM

    # Reference: hadamard_rotate casts the rotated row to the input dtype (bf16)
    # and fake_fp4_act_quant immediately casts it back to fp32 — replicate that
    # round-trip so the fused result is bit-identical.
    x = x.to(out_ptr.dtype.element_ty).to(tl.float32)

    # --- block fake-FP4 quant ---
    xb = tl.reshape(x, (NB, BLOCK_SIZE))  # [NB, BLOCK_SIZE]
    amax = tl.max(tl.abs(xb), axis=1, keep_dims=True)  # [NB, 1]
    # ceil_pow2_scale(amax, FP4_MAX, FP4_MIN)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, FP4_MIN) / FP4_MAX)))  # [NB, 1]
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
    res = (q * sign) * scale  # [NB, BLOCK_SIZE]

    res = tl.reshape(res, (DIM,))
    tl.store(out_ptr + row * DIM + offs, res.to(out_ptr.dtype.element_ty))


@triton.jit
def _hadamard_stage_blocked(
    x, BR: tl.constexpr, DIM: tl.constexpr, G: tl.constexpr, W: tl.constexpr
):
    """``_hadamard_stage`` over a ``[BR, DIM]`` tile (butterfly along the DIM axis).

    Identical math/associativity to the 1-row helper -- the row axis ``BR`` is just
    a batched outer dimension -- so the result is bit-identical row-for-row.
    """
    a = tl.reshape(x, (BR, G, 2, W))
    sw = tl.flip(a, 2)
    lower = (tl.arange(0, 2) == 0)[None, None, :, None]  # [1, 1, 2, 1]
    a = tl.where(lower, a + sw, sw - a)
    return tl.reshape(a, (BR, DIM))


@triton.jit
def _hadamard_fp4_kernel_blocked(
    x_ptr,  # [R, DIM] contiguous input
    out_ptr,  # [R, DIM] contiguous output
    R,
    BLOCK_R: tl.constexpr,  # rows handled per program
    DIM: tl.constexpr,  # power-of-two row width (== Hadamard dim)
    BLOCK_SIZE: tl.constexpr,  # fp4 quant group size (DIM % BLOCK_SIZE == 0)
    NB: tl.constexpr,  # DIM // BLOCK_SIZE
    INV_SQRT_DIM: tl.constexpr,  # DIM ** -0.5
    FP4_MAX: tl.constexpr,
    FP4_MIN: tl.constexpr,
):
    """Row-blocked variant of ``_hadamard_fp4_kernel``.

    The 1-row kernel launches ``R`` single-warp CTAs, each doing a tiny amount of
    work; at large ``R`` (prefill: indexer-q is ``B*S*index_n_heads_local`` rows)
    that is dominated by fixed per-CTA scheduling overhead. Handling ``BLOCK_R``
    rows per program (a ``[BLOCK_R, DIM]`` register tile, butterfly batched over the
    row axis) amortizes that overhead and lets the warp issue wider work. The math
    is the same per row, so the result is bit-identical to the 1-row kernel; only
    the element-to-thread mapping (layout) differs.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)  # [BLOCK_R]
    mask = rows < R
    offs = tl.arange(0, DIM)  # [DIM]
    ptr = x_ptr + rows[:, None] * DIM + offs[None, :]  # [BLOCK_R, DIM]
    x = tl.load(ptr, mask=mask[:, None], other=0.0).to(tl.float32)

    # --- Walsh-Hadamard butterfly (Sylvester order), fp32, unrolled stages ---
    if DIM >= 2:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 2, 1)
    if DIM >= 4:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 4, 2)
    if DIM >= 8:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 8, 4)
    if DIM >= 16:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 16, 8)
    if DIM >= 32:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 32, 16)
    if DIM >= 64:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 64, 32)
    if DIM >= 128:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 128, 64)
    if DIM >= 256:
        x = _hadamard_stage_blocked(x, BLOCK_R, DIM, DIM // 256, 128)
    tl.static_assert(DIM <= 256, "deepseek_v4_hadamard_fp4 supports DIM <= 256")
    x = x * INV_SQRT_DIM

    # bf16 round-trip mirror (see 1-row kernel).
    x = x.to(out_ptr.dtype.element_ty).to(tl.float32)

    # --- block fake-FP4 quant (reduce along the BLOCK_SIZE axis) ---
    xb = tl.reshape(x, (BLOCK_R, NB, BLOCK_SIZE))  # [BLOCK_R, NB, BLOCK_SIZE]
    amax = tl.max(tl.abs(xb), axis=2, keep_dims=True)  # [BLOCK_R, NB, 1]
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, FP4_MIN) / FP4_MAX)))
    n = tl.minimum(tl.maximum(xb / scale, -FP4_MAX), FP4_MAX)
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
    res = (q * sign) * scale  # [BLOCK_R, NB, BLOCK_SIZE]

    res = tl.reshape(res, (BLOCK_R, DIM))
    tl.store(
        out_ptr + rows[:, None] * DIM + offs[None, :],
        res.to(out_ptr.dtype.element_ty),
        mask=mask[:, None],
    )


# Row-blocking only pays off once enough rows exist to amortize the extra per-program
# masking/index arithmetic. Below this many rows the minimal 1-warp 1-row kernel is
# fastest (decode: indexer-q is R=8, latency-bound); at/above it the BLOCK_R=2 tile
# wins (R=1024 -5%, R=2048 -11%, R=4096 -26%, R=8000 -35%). BLOCK_R=2 beats 4/8/16
# (which over-subscribe registers at num_warps=1). Microbench: bench/bench_hadamard_fp4.py.
_BLOCKED_ROW_THRESHOLD = 1024
_BLOCK_R = 2


@torch.library.custom_op("auto_deploy::deepseek_v4_hadamard_fp4", mutates_args=())
def deepseek_v4_hadamard_fp4(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Fused ``fake_fp4_act_quant(hadamard_rotate(x), block_size)``.

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

    out = torch.empty_like(x)
    R = x.numel() // dim
    if R == 0:
        return out

    x2 = x.reshape(R, dim)
    out2 = out.reshape(R, dim)

    # Occupancy (idea_0046): one warp per program. Each program handles a
    # single DIM<=256 row (or BLOCK_R such rows below), so a single warp covers
    # it with intra-warp shuffles for the FP4 block reductions; extra warps add
    # cross-warp smem barriers. num_stages=1 is a zero-cost guard: the kernel is
    # loop-free so there is nothing to pipeline (nw*/default == nw*/s1, inert).
    if R >= _BLOCKED_ROW_THRESHOLD:
        # Layout (idea_0049): at large R the 1-row kernel's R single-warp CTAs are
        # dominated by per-CTA scheduling overhead. A [BLOCK_R, DIM] tile per
        # program amortizes it -- bit-identical, but the element-to-thread mapping
        # changes. Big win on the prefill indexer-q path (R = B*S*n_heads_local).
        _hadamard_fp4_kernel_blocked[(triton.cdiv(R, _BLOCK_R),)](
            x2,
            out2,
            R,
            BLOCK_R=_BLOCK_R,
            DIM=dim,
            BLOCK_SIZE=block_size,
            NB=dim // block_size,
            INV_SQRT_DIM=float(dim**-0.5),
            FP4_MAX=_FP4_MAX,
            FP4_MIN=_FP4_MIN,
            num_warps=1,
            num_stages=1,
        )
    else:
        # Decode / small-R: the minimal 1-row kernel is fastest (R=8 indexer-q is
        # latency-bound -- extra masking/index arithmetic regresses it ~11%).
        _hadamard_fp4_kernel[(R,)](
            x2,
            out2,
            R,
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
    return torch.empty_like(x)
