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

"""Custom op for fused DeepSeek-V4 hyper-connection (HC) composition.

The HC ``_hc_pre`` step in ``modeling_deepseek_v4.py`` splits a per-token mix
vector into ``pre`` / ``post`` gates and a doubly-stochastic ``comb`` matrix.
The ``comb`` matrix is produced by a softmax followed by a ``sinkhorn_iters``
(=20) loop of alternating row / column normalizations over a tiny
``[N, hc_mult, hc_mult] = [N, 4, 4]`` tensor.

In eager / decomposed form that loop emits ~40 grid=1 ``reduce`` kernels plus a
``softmax`` per call, run twice per layer per step. Each grid=1 launch pays a
per-kernel GPU execution floor that is *not* hidden by CUDA graphs (the floor is
GPU-side, not host launch latency). This op collapses the whole chain
(sigmoid + softmax + the 20-iteration sinkhorn loop) into a *single* Triton
kernel that keeps the identical fp32 math entirely in registers — one launch per
HC call instead of ~40.

The kernel name deliberately avoids the ``reduction`` op-type regex (no
``reduce`` / ``sum`` / ``softmax`` / ``norm`` substrings) so the collapsed work
leaves the ``reduction`` bucket entirely.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Registers auto_deploy::deepseek_v4_hc_combine_rmsnorm for the prefill fallback
# of ``deepseek_v4_hc_pre_mix_combine`` when this module is imported standalone
# (the package ``__init__`` auto-imports every sibling anyway). One-way import:
# ``deepseek_v4_hc_pre_norm`` does not import this module.
from . import deepseek_v4_hc_pre_norm as _hc_pre_norm  # noqa: F401


@triton.jit
def _hc_composition_kernel(
    mixes_ptr,  # [N, MIX_HC] fp32
    scale_ptr,  # [3] fp32
    base_ptr,  # [MIX_HC] fp32
    pre_ptr,  # [N, HM] fp32 (out)
    post_ptr,  # [N, HM] fp32 (out)
    comb_ptr,  # [N, HM*HM] fp32 (out)
    N,
    eps,
    MIX_HC: tl.constexpr,
    HM: tl.constexpr,  # hc_mult
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    SINKHORN_ITERS: tl.constexpr,
):
    """One program per token row. Computes pre / post / comb in registers.

    Mirrors ``_hc_split_sinkhorn`` exactly in fp32:
      pre  = sigmoid(mixes[:HM]      * scale[0] + base[:HM])      + eps
      post = 2 * sigmoid(mixes[HM:2HM] * scale[1] + base[HM:2HM])
      comb = (softmax(mixes[2HM:] * scale[2] + base[2HM:], dim=-1) + eps)
             then col-normalize, then (SINKHORN_ITERS-1) x (row-norm, col-norm)
    """
    row = tl.program_id(0)
    if row >= N:
        return
    base_row = row * MIX_HC

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    d = tl.arange(0, BM)
    dmask = d < HM

    # pre = sigmoid(logits) + eps
    pre_logits = tl.load(mixes_ptr + base_row + d, mask=dmask, other=0.0) * s0 + tl.load(
        base_ptr + d, mask=dmask, other=0.0
    )
    pre = tl.sigmoid(pre_logits) + eps
    tl.store(pre_ptr + row * HM + d, pre, mask=dmask)

    # post = 2 * sigmoid(logits)
    post_logits = tl.load(mixes_ptr + base_row + HM + d, mask=dmask, other=0.0) * s1 + tl.load(
        base_ptr + HM + d, mask=dmask, other=0.0
    )
    post = 2.0 * tl.sigmoid(post_logits)
    tl.store(post_ptr + row * HM + d, post, mask=dmask)

    # comb logits, laid out as [i (dim=-2), j (dim=-1)]
    i = tl.arange(0, BM)[:, None]
    j = tl.arange(0, BM)[None, :]
    m2 = (i < HM) & (j < HM)
    cflat = i * HM + j
    comb_logits = tl.load(mixes_ptr + base_row + 2 * HM + cflat, mask=m2, other=0.0) * s2 + tl.load(
        base_ptr + 2 * HM + cflat, mask=m2, other=0.0
    )

    # softmax over dim=-1 (j / axis=1), then + eps (valid entries only)
    neg_inf = float("-inf")
    logits_sm = tl.where(m2, comb_logits, neg_inf)
    mx = tl.max(logits_sm, axis=1)[:, None]
    e = tl.exp(logits_sm - mx)
    sden = tl.sum(e, axis=1)[:, None]
    comb = tl.where(m2, e / sden + eps, 0.0)

    # initial col-normalize: sum over dim=-2 (i / axis=0)
    cs = tl.sum(comb, axis=0)[None, :]
    comb = tl.where(m2, comb / (cs + eps), 0.0)

    for _ in range(SINKHORN_ITERS - 1):
        rs = tl.sum(comb, axis=1)[:, None]
        comb = tl.where(m2, comb / (rs + eps), 0.0)
        cs = tl.sum(comb, axis=0)[None, :]
        comb = tl.where(m2, comb / (cs + eps), 0.0)

    tl.store(comb_ptr + row * (HM * HM) + cflat, comb, mask=m2)


@torch.library.custom_op("auto_deploy::hc_split_sinkhorn", mutates_args=())
def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused HC split + sinkhorn. Drop-in for ``_hc_split_sinkhorn``.

    Args:
        mixes:          [..., MIX_HC] fp32, MIX_HC == (2 + hc_mult) * hc_mult
        hc_scale:       [3] fp32
        hc_base:        [MIX_HC] fp32
        hc_mult:        comb matrix side (4 for DeepSeek-V4-Flash)
        sinkhorn_iters: number of sinkhorn normalization rounds
        eps:            stabilization epsilon

    Returns:
        pre:  [..., hc_mult]            fp32
        post: [..., hc_mult]            fp32
        comb: [..., hc_mult, hc_mult]   fp32
    """
    lead = list(mixes.shape[:-1])
    mix_hc = mixes.shape[-1]
    n = 1
    for s in lead:
        n *= s

    mixes_f = mixes.reshape(n, mix_hc).contiguous().float()
    scale_f = hc_scale.contiguous().float()
    base_f = hc_base.contiguous().float()

    pre = torch.empty(n, hc_mult, device=mixes.device, dtype=torch.float32)
    post = torch.empty(n, hc_mult, device=mixes.device, dtype=torch.float32)
    comb = torch.empty(n, hc_mult * hc_mult, device=mixes.device, dtype=torch.float32)

    bm = triton.next_power_of_2(hc_mult)
    grid = (n,)
    _hc_composition_kernel[grid](
        mixes_f,
        scale_f,
        base_f,
        pre,
        post,
        comb,
        n,
        eps,
        MIX_HC=mix_hc,
        HM=hc_mult,
        BM=bm,
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=1,
    )

    return (
        pre.reshape(*lead, hc_mult),
        post.reshape(*lead, hc_mult),
        comb.reshape(*lead, hc_mult, hc_mult),
    )


@hc_split_sinkhorn.register_fake
def _hc_split_sinkhorn_fake(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = list(mixes.shape[:-1])
    pre = mixes.new_empty((*lead, hc_mult), dtype=torch.float32)
    post = mixes.new_empty((*lead, hc_mult), dtype=torch.float32)
    comb = mixes.new_empty((*lead, hc_mult, hc_mult), dtype=torch.float32)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Fused HC-pre front: RMS statistic + hc_fn GEMV + ordered scaling + sinkhorn
# ---------------------------------------------------------------------------
#
# The remaining eager front of ``_hc_pre``::
#
#     flat  = x.flatten(2).float()                                    # HBM cast
#     rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
#     mixes = _linear(flat, hc_fn) * rsqrt                            # cublas GEMV
#
# emits ~6 kernels per call (bf16->fp32 cast, square, mean-reduce, +eps,
# rsqrt, GEMV, broadcast mul) and materializes two ``[N, hc_mult*H]`` fp32
# intermediates. ``deepseek_v4_hc_pre_mix`` collapses the whole front + the
# sinkhorn composition into two launches:
#
#   * ``_hc_fn_partials_kernel`` (grid ``(n, split)``): each program owns one
#     D-chunk of one row, loads the input in its native dtype (bf16->fp32
#     conversion is exact), and emits per-chunk partials for the square-sum
#     statistic and all ``MIX_HC`` hc_fn dot products — one pass over the
#     input, ``flat`` fp32 is never materialized.
#   * ``_hc_partials_composition_kernel`` (grid ``(n,)``): reduces the
#     partials in a fixed (deterministic) tree order, applies
#     ``rstd = rsqrt(sum/D + norm_eps)`` and the ordered scaling
#     ``(mix * rstd) * hc_scale + hc_base`` (same fp32 op order as the eager
#     chain), then runs the identical sigmoid/softmax/sinkhorn math as
#     ``_hc_composition_kernel``.
#
# The split-D partial layout ``[n, MIX_HC + 1, split]`` (dots first, square-sum
# last) is internal to this op. Reduction order differs from cublas/torch, so
# ``mixes`` matches the eager chain to ~1 ULP (fp32), not bit-exactly; the
# sinkhorn math downstream of ``mixes`` is unchanged. For large token counts
# (prefill) the op falls back to the eager torch front + the existing
# composition kernel, where the cublas GEMM is faster than the split-D GEMV.

_HC_PRE_MIX_FUSED_N_MAX = 64


@triton.jit
def _hc_fn_partials_kernel(
    x_ptr,  # [N, D] input (any float dtype; converted to fp32 in-register)
    fn_ptr,  # [MIX_HC, D] fp32
    part_ptr,  # [N, MIX_HC + 1, SPLIT] fp32 out
    N,
    D,
    SPLIT,
    MIX_HC: tl.constexpr,
    KBLOCK: tl.constexpr,  # next_power_of_2(MIX_HC)
    CHUNK: tl.constexpr,  # elements per split slot (power of 2)
):
    """One program per (row, D-chunk): square-sum + MIX_HC dot partials."""
    row = tl.program_id(0)
    s = tl.program_id(1)
    if row >= N:
        return

    offs = s * CHUNK + tl.arange(0, CHUNK)
    cmask = offs < D
    x = tl.load(x_ptr + row * D + offs, mask=cmask, other=0.0).to(tl.float32)

    # Per-chunk square-sum partial for the RMS statistic.
    sq = tl.sum(x * x, axis=0)

    # Per-chunk partial dot products for all MIX_HC hc_fn rows at once.
    k = tl.arange(0, KBLOCK)
    kmask = k < MIX_HC
    w = tl.load(
        fn_ptr + k[:, None] * D + offs[None, :],
        mask=kmask[:, None] & cmask[None, :],
        other=0.0,
    )
    part = tl.sum(w * x[None, :], axis=1)

    out_base = part_ptr + row * ((MIX_HC + 1) * SPLIT)
    tl.store(out_base + k * SPLIT + s, part, mask=kmask)
    tl.store(out_base + MIX_HC * SPLIT + s, sq)


@triton.jit
def _hc_partials_composition_kernel(
    part_ptr,  # [N, MIX_HC + 1, SPLIT] fp32
    scale_ptr,  # [3] fp32
    base_ptr,  # [MIX_HC] fp32
    pre_ptr,  # [N, HM] fp32 (out)
    post_ptr,  # [N, HM] fp32 (out)
    comb_ptr,  # [N, HM*HM] fp32 (out)
    N,
    D,
    SPLIT,
    norm_eps,
    eps,
    MIX_HC: tl.constexpr,
    HM: tl.constexpr,  # hc_mult
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    SBLOCK: tl.constexpr,  # next_power_of_2(SPLIT)
    SINKHORN_ITERS: tl.constexpr,
):
    """One program per token row: reduce partials -> mixes -> pre/post/comb.

    The post-``mixes`` math mirrors ``_hc_composition_kernel`` exactly; only the
    ``mixes`` values come from the in-kernel partial reduction + RMS scaling
    instead of a pre-materialized tensor.
    """
    row = tl.program_id(0)
    if row >= N:
        return
    pbase = part_ptr + row * ((MIX_HC + 1) * SPLIT)

    s = tl.arange(0, SBLOCK)
    smask = s < SPLIT

    # rstd = rsqrt(mean(x^2) + norm_eps), fixed-order tree reduce over SPLIT.
    sq = tl.load(pbase + MIX_HC * SPLIT + s, mask=smask, other=0.0)
    rstd = tl.rsqrt(tl.sum(sq, axis=0) / D + norm_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    d = tl.arange(0, BM)
    dmask = d < HM

    # pre = sigmoid((mix * rstd) * scale[0] + base) + eps
    pre_part = tl.load(
        pbase + d[:, None] * SPLIT + s[None, :],
        mask=dmask[:, None] & smask[None, :],
        other=0.0,
    )
    pre_logits = (tl.sum(pre_part, axis=1) * rstd) * s0 + tl.load(
        base_ptr + d, mask=dmask, other=0.0
    )
    pre = tl.sigmoid(pre_logits) + eps
    tl.store(pre_ptr + row * HM + d, pre, mask=dmask)

    # post = 2 * sigmoid((mix * rstd) * scale[1] + base)
    post_part = tl.load(
        pbase + (HM + d)[:, None] * SPLIT + s[None, :],
        mask=dmask[:, None] & smask[None, :],
        other=0.0,
    )
    post_logits = (tl.sum(post_part, axis=1) * rstd) * s1 + tl.load(
        base_ptr + HM + d, mask=dmask, other=0.0
    )
    post = 2.0 * tl.sigmoid(post_logits)
    tl.store(post_ptr + row * HM + d, post, mask=dmask)

    # comb logits, laid out as [i (dim=-2), j (dim=-1)]
    i = tl.arange(0, BM)[:, None]
    j = tl.arange(0, BM)[None, :]
    m2 = (i < HM) & (j < HM)
    cflat = i * HM + j
    comb_part = tl.load(
        pbase + (2 * HM + cflat)[:, :, None] * SPLIT + s[None, None, :],
        mask=m2[:, :, None] & smask[None, None, :],
        other=0.0,
    )
    comb_logits = (tl.sum(comb_part, axis=2) * rstd) * s2 + tl.load(
        base_ptr + 2 * HM + cflat, mask=m2, other=0.0
    )

    # softmax over dim=-1 (j / axis=1), then + eps (valid entries only)
    neg_inf = float("-inf")
    logits_sm = tl.where(m2, comb_logits, neg_inf)
    mx = tl.max(logits_sm, axis=1)[:, None]
    e = tl.exp(logits_sm - mx)
    sden = tl.sum(e, axis=1)[:, None]
    comb = tl.where(m2, e / sden + eps, 0.0)

    # initial col-normalize: sum over dim=-2 (i / axis=0)
    cs = tl.sum(comb, axis=0)[None, :]
    comb = tl.where(m2, comb / (cs + eps), 0.0)

    for _ in range(SINKHORN_ITERS - 1):
        rs = tl.sum(comb, axis=1)[:, None]
        comb = tl.where(m2, comb / (rs + eps), 0.0)
        cs = tl.sum(comb, axis=0)[None, :]
        comb = tl.where(m2, comb / (cs + eps), 0.0)

    tl.store(comb_ptr + row * (HM * HM) + cflat, comb, mask=m2)


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_pre_mix", mutates_args=())
def deepseek_v4_hc_pre_mix(
    flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused HC-pre front. Drop-in for the eager ``_hc_pre`` mix chain.

    Computes, per leading (token) index::

        f = flat.float()  # exact for bf16
        rstd = rsqrt(mean(f ^ 2) + norm_eps)
        mixes = (f @ hc_fn.T) * rstd
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)

    Args:
        flat:           [..., D] input hidden states (bf16/fp16/fp32),
                        D == hc_mult * hidden_size
        hc_fn:          [MIX_HC, D] fp32, MIX_HC == (2 + hc_mult) * hc_mult
        hc_scale:       [3] fp32
        hc_base:        [MIX_HC] fp32
        hc_mult:        comb matrix side (4 for DeepSeek-V4-Flash)
        sinkhorn_iters: number of sinkhorn normalization rounds
        eps:            sinkhorn stabilization epsilon
        norm_eps:       RMS statistic epsilon

    Returns:
        pre:  [..., hc_mult]            fp32
        post: [..., hc_mult]            fp32
        comb: [..., hc_mult, hc_mult]   fp32
    """
    lead = list(flat.shape[:-1])
    dim = flat.shape[-1]
    n = 1
    for sz in lead:
        n *= sz
    mix_hc = hc_fn.shape[0]
    assert hc_fn.shape[-1] == dim, "hc_fn last dim must match flat last dim"

    if n > _HC_PRE_MIX_FUSED_N_MAX:
        # Prefill-sized token counts: the eager cublas GEMM front beats the
        # split-D GEMV. Same math, routed through the existing fused
        # composition kernel.
        flat_f = flat.reshape(n, dim).float()
        rsqrt = torch.rsqrt(flat_f.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat_f, hc_fn) * rsqrt
        pre, post, comb = torch.ops.auto_deploy.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        return (
            pre.reshape(*lead, hc_mult),
            post.reshape(*lead, hc_mult),
            comb.reshape(*lead, hc_mult, hc_mult),
        )

    flat_c = flat.reshape(n, dim).contiguous()
    fn_f = hc_fn.contiguous().float()
    scale_f = hc_scale.contiguous().float()
    base_f = hc_base.contiguous().float()

    pre = torch.empty(n, hc_mult, device=flat.device, dtype=torch.float32)
    post = torch.empty(n, hc_mult, device=flat.device, dtype=torch.float32)
    comb = torch.empty(n, hc_mult * hc_mult, device=flat.device, dtype=torch.float32)
    if n == 0:
        return (
            pre.reshape(*lead, hc_mult),
            post.reshape(*lead, hc_mult),
            comb.reshape(*lead, hc_mult, hc_mult),
        )

    # ~128 chunks for the model shape (D = 16384 -> CHUNK = 128); small D used
    # in tests degrades gracefully to fewer/masked chunks.
    chunk = max(64, triton.next_power_of_2((dim + 127) // 128))
    split = (dim + chunk - 1) // chunk

    partials = torch.empty(n, mix_hc + 1, split, device=flat.device, dtype=torch.float32)
    _hc_fn_partials_kernel[(n, split)](
        flat_c,
        fn_f,
        partials,
        n,
        dim,
        split,
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        num_warps=4,
    )

    _hc_partials_composition_kernel[(n,)](
        partials,
        scale_f,
        base_f,
        pre,
        post,
        comb,
        n,
        dim,
        split,
        norm_eps,
        eps,
        MIX_HC=mix_hc,
        HM=hc_mult,
        BM=triton.next_power_of_2(hc_mult),
        SBLOCK=triton.next_power_of_2(split),
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=2,
    )

    return (
        pre.reshape(*lead, hc_mult),
        post.reshape(*lead, hc_mult),
        comb.reshape(*lead, hc_mult, hc_mult),
    )


@deepseek_v4_hc_pre_mix.register_fake
def _deepseek_v4_hc_pre_mix_fake(
    flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = list(flat.shape[:-1])
    pre = flat.new_empty((*lead, hc_mult), dtype=torch.float32)
    post = flat.new_empty((*lead, hc_mult), dtype=torch.float32)
    comb = flat.new_empty((*lead, hc_mult, hc_mult), dtype=torch.float32)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Fully fused HC-pre: composition + weighted-combine + block RMSNorm
# ---------------------------------------------------------------------------
#
# ``deepseek_v4_hc_pre_mix`` still hands ``pre`` back through HBM so the
# separate ``deepseek_v4_hc_combine_rmsnorm`` launch can consume it — three
# kernels per HC call at decode. But ``pre`` is only ever fed to that combine,
# and it is ready *before* the serial sinkhorn loop (it needs the partial
# reduction and ``rstd`` only). ``deepseek_v4_hc_pre_mix_combine`` therefore
# merges the composition and the combine into ONE kernel per row (two launches
# per HC call total):
#
#   * ``_hc_fn_partials_kernel`` — unchanged split-D front (a single CTA cannot
#     stream the [MIX_HC, D] fp32 ``hc_fn`` weight competitively).
#   * ``_hc_pre_composition_combine_kernel`` — reduces the partials, computes
#     ``pre``/``post`` and the comb logits, runs the weighted combine +
#     RMSNorm, then the sinkhorn loop. ``pre`` never leaves registers.
#
# The block ORDER inside the fused kernel is deliberate: all small partial
# loads and logit reductions run first, then the wide ``flat`` loads + combine,
# and the register-only sinkhorn loop last. With the comb logits already in
# registers the sinkhorn chain has no memory dependency, so ptxas overlaps its
# ~40-step serial ALU chain with the outstanding ``flat`` load latency
# (measured on B200: 6.56us fused vs 7.74us for the landed two-kernel
# sequence at the decode shape, amortized in a CUDA graph).
#
# Bit-exactness (torch.equal vs the landed two-kernel path) requires
# ``num_warps=2``: the composition reductions change bits with warp count
# (their tile-reduce tree is layout-dependent), while the combine part is
# warp-count-invariant, so the fused kernel adopts the composition kernel's
# ``num_warps=2`` and reproduces both kernels' bits. ``maxnreg=240`` only
# relaxes ptxas scheduling (measured fastest 192..255 cap sweep); it does not
# change results.

_HC_PRE_MIX_COMBINE_NUM_WARPS = 2
_HC_PRE_MIX_COMBINE_MAXNREG = 240


@triton.jit
def _hc_pre_composition_combine_kernel(
    part_ptr,  # [N, MIX_HC + 1, SPLIT] fp32
    scale_ptr,  # [3] fp32
    base_ptr,  # [MIX_HC] fp32
    flat_ptr,  # [N, HM * H] input (any float dtype; converted in-register)
    weight_ptr,  # [H] fp32 (RMSNorm weight)
    y_ptr,  # [N, H] out_dtype (out)
    post_ptr,  # [N, HM] fp32 (out)
    comb_ptr,  # [N, HM*HM] fp32 (out)
    N,
    D,
    SPLIT,
    H,
    norm_eps,
    eps,
    rms_eps,
    MIX_HC: tl.constexpr,
    HM: tl.constexpr,  # hc_mult
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    SBLOCK: tl.constexpr,  # next_power_of_2(SPLIT)
    BLOCK_H: tl.constexpr,  # next_power_of_2(H)
    SINKHORN_ITERS: tl.constexpr,
):
    """One program per token row: partials -> pre/post/comb logits -> y -> comb.

    The composition math mirrors ``_hc_partials_composition_kernel`` and the
    combine + RMSNorm math mirrors ``_hc_weighted_combine_kernel`` exactly
    (same tile shapes and op order, so at ``num_warps=2`` the outputs are
    bit-identical to the two-kernel sequence). ``pre`` is consumed from
    registers via an exact one-hot extraction instead of an HBM round-trip.
    """
    row = tl.program_id(0)
    if row >= N:
        return
    pbase = part_ptr + row * ((MIX_HC + 1) * SPLIT)

    s = tl.arange(0, SBLOCK)
    smask = s < SPLIT

    # rstd = rsqrt(mean(x^2) + norm_eps), fixed-order tree reduce over SPLIT.
    sq = tl.load(pbase + MIX_HC * SPLIT + s, mask=smask, other=0.0)
    rstd = tl.rsqrt(tl.sum(sq, axis=0) / D + norm_eps)

    s0 = tl.load(scale_ptr + 0)
    s1 = tl.load(scale_ptr + 1)
    s2 = tl.load(scale_ptr + 2)

    d = tl.arange(0, BM)
    dmask = d < HM

    # pre = sigmoid((mix * rstd) * scale[0] + base) + eps — kept in registers.
    pre_part = tl.load(
        pbase + d[:, None] * SPLIT + s[None, :],
        mask=dmask[:, None] & smask[None, :],
        other=0.0,
    )
    pre_logits = (tl.sum(pre_part, axis=1) * rstd) * s0 + tl.load(
        base_ptr + d, mask=dmask, other=0.0
    )
    pre = tl.sigmoid(pre_logits) + eps

    # post = 2 * sigmoid((mix * rstd) * scale[1] + base)
    post_part = tl.load(
        pbase + (HM + d)[:, None] * SPLIT + s[None, :],
        mask=dmask[:, None] & smask[None, :],
        other=0.0,
    )
    post_logits = (tl.sum(post_part, axis=1) * rstd) * s1 + tl.load(
        base_ptr + HM + d, mask=dmask, other=0.0
    )
    post = 2.0 * tl.sigmoid(post_logits)
    tl.store(post_ptr + row * HM + d, post, mask=dmask)

    # comb logits, laid out as [i (dim=-2), j (dim=-1)] — computed BEFORE the
    # combine so the sinkhorn loop below is register-only and can overlap the
    # flat load latency.
    i = tl.arange(0, BM)[:, None]
    j = tl.arange(0, BM)[None, :]
    m2 = (i < HM) & (j < HM)
    cflat = i * HM + j
    comb_part = tl.load(
        pbase + (2 * HM + cflat)[:, :, None] * SPLIT + s[None, None, :],
        mask=m2[:, :, None] & smask[None, None, :],
        other=0.0,
    )
    comb_logits = (tl.sum(comb_part, axis=2) * rstd) * s2 + tl.load(
        base_ptr + 2 * HM + cflat, mask=m2, other=0.0
    )

    # --- weighted combine + RMSNorm (mirrors _hc_weighted_combine_kernel) ---
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    flat_row = flat_ptr + row * (HM * H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for m in tl.static_range(HM):
        # Exact scalar extraction of pre[m] from the register tile (sum of a
        # one-hot selection reproduces the element bit-for-bit).
        p = tl.sum(tl.where(d == m, pre, 0.0), axis=0)
        f = tl.load(flat_row + m * H + h, mask=hmask, other=0.0).to(tl.float32)
        acc += p * f
    y = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(y * y, axis=0) / H
    y_rstd = tl.rsqrt(var + rms_eps)
    normed = (y * y_rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(weight_ptr + h, mask=hmask, other=0.0)
    out = w * normed
    tl.store(y_ptr + row * H + h, out.to(y_ptr.dtype.element_ty), mask=hmask)

    # --- sinkhorn (register-only; mirrors _hc_partials_composition_kernel) ---
    neg_inf = float("-inf")
    logits_sm = tl.where(m2, comb_logits, neg_inf)
    mx = tl.max(logits_sm, axis=1)[:, None]
    e = tl.exp(logits_sm - mx)
    sden = tl.sum(e, axis=1)[:, None]
    comb = tl.where(m2, e / sden + eps, 0.0)

    cs = tl.sum(comb, axis=0)[None, :]
    comb = tl.where(m2, comb / (cs + eps), 0.0)

    for _ in range(SINKHORN_ITERS - 1):
        rs = tl.sum(comb, axis=1)[:, None]
        comb = tl.where(m2, comb / (rs + eps), 0.0)
        cs = tl.sum(comb, axis=0)[None, :]
        comb = tl.where(m2, comb / (cs + eps), 0.0)

    tl.store(comb_ptr + row * (HM * HM) + cflat, comb, mask=m2)


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_pre_mix_combine", mutates_args=())
def deepseek_v4_hc_pre_mix_combine(
    flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
    rms_eps: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused HC-pre. Drop-in for ``deepseek_v4_hc_pre_mix`` + the combine op.

    Computes, per leading (token) index::

        pre, post, comb = deepseek_v4_hc_pre_mix(flat, hc_fn, ...)
        y = deepseek_v4_hc_combine_rmsnorm(pre, flat, norm_weight, rms_eps, ...)

    but on the decode path the composition and the combine run inside one
    kernel and ``pre`` never touches HBM. Outputs are bit-identical to the
    two-op sequence on every path.

    Args:
        flat:           [..., D] input hidden states (bf16/fp16/fp32),
                        D == hc_mult * H
        hc_fn:          [MIX_HC, D] fp32, MIX_HC == (2 + hc_mult) * hc_mult
        hc_scale:       [3] fp32
        hc_base:        [MIX_HC] fp32
        norm_weight:    [H] fp32 RMSNorm weight (attn_norm / ffn_norm)
        hc_mult:        comb matrix side (4 for DeepSeek-V4-Flash)
        sinkhorn_iters: number of sinkhorn normalization rounds
        eps:            sinkhorn stabilization epsilon
        norm_eps:       RMS statistic epsilon (mix scaling)
        rms_eps:        RMSNorm epsilon (y normalization)
        out_dtype:      dtype of the returned ``y`` (the residual dtype)

    Returns:
        y:    [..., H]                  out_dtype == rmsnorm(sum_m pre*flat)
        post: [..., hc_mult]            fp32
        comb: [..., hc_mult, hc_mult]   fp32
    """
    lead = list(flat.shape[:-1])
    dim = flat.shape[-1]
    n = 1
    for sz in lead:
        n *= sz
    mix_hc = hc_fn.shape[0]
    hidden = norm_weight.shape[0]
    assert hc_fn.shape[-1] == dim, "hc_fn last dim must match flat last dim"
    assert dim == hc_mult * hidden, "flat last dim must equal hc_mult * H"

    if n > _HC_PRE_MIX_FUSED_N_MAX:
        # Prefill-sized token counts: identical to the landed two-op path
        # (eager cublas front + composition kernel + combine kernel), so this
        # branch is bit-exact vs current behavior by construction.
        flat_f = flat.reshape(n, dim).float()
        rsqrt = torch.rsqrt(flat_f.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat_f, hc_fn) * rsqrt
        pre, post, comb = torch.ops.auto_deploy.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        y = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
            pre, flat.reshape(n, dim), norm_weight, rms_eps, hc_mult, out_dtype
        )
        return (
            y.reshape(*lead, hidden),
            post.reshape(*lead, hc_mult),
            comb.reshape(*lead, hc_mult, hc_mult),
        )

    flat_c = flat.reshape(n, dim).contiguous()
    fn_f = hc_fn.contiguous().float()
    scale_f = hc_scale.contiguous().float()
    base_f = hc_base.contiguous().float()
    weight_f = norm_weight.contiguous().float()

    y = torch.empty(n, hidden, device=flat.device, dtype=out_dtype)
    post = torch.empty(n, hc_mult, device=flat.device, dtype=torch.float32)
    comb = torch.empty(n, hc_mult * hc_mult, device=flat.device, dtype=torch.float32)
    if n == 0:
        return (
            y.reshape(*lead, hidden),
            post.reshape(*lead, hc_mult),
            comb.reshape(*lead, hc_mult, hc_mult),
        )

    # Same split-D layout as deepseek_v4_hc_pre_mix (~128 chunks at D = 16384).
    chunk = max(64, triton.next_power_of_2((dim + 127) // 128))
    split = (dim + chunk - 1) // chunk

    partials = torch.empty(n, mix_hc + 1, split, device=flat.device, dtype=torch.float32)
    _hc_fn_partials_kernel[(n, split)](
        flat_c,
        fn_f,
        partials,
        n,
        dim,
        split,
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        num_warps=4,
    )

    _hc_pre_composition_combine_kernel[(n,)](
        partials,
        scale_f,
        base_f,
        flat_c,
        weight_f,
        y,
        post,
        comb,
        n,
        dim,
        split,
        hidden,
        norm_eps,
        eps,
        rms_eps,
        MIX_HC=mix_hc,
        HM=hc_mult,
        BM=triton.next_power_of_2(hc_mult),
        SBLOCK=triton.next_power_of_2(split),
        BLOCK_H=triton.next_power_of_2(hidden),
        SINKHORN_ITERS=sinkhorn_iters,
        num_warps=_HC_PRE_MIX_COMBINE_NUM_WARPS,
        num_stages=2,
        maxnreg=_HC_PRE_MIX_COMBINE_MAXNREG,
    )

    return (
        y.reshape(*lead, hidden),
        post.reshape(*lead, hc_mult),
        comb.reshape(*lead, hc_mult, hc_mult),
    )


@deepseek_v4_hc_pre_mix_combine.register_fake
def _deepseek_v4_hc_pre_mix_combine_fake(
    flat: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
    rms_eps: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = list(flat.shape[:-1])
    hidden = norm_weight.shape[0]
    y = flat.new_empty((*lead, hidden), dtype=out_dtype)
    post = flat.new_empty((*lead, hc_mult), dtype=torch.float32)
    comb = flat.new_empty((*lead, hc_mult, hc_mult), dtype=torch.float32)
    return y, post, comb
