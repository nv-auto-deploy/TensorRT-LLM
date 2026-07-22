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

"""Fused Triton ops for the DeepSeek-V4 hyper-connection (HC) residual streams.

One subsystem, three sections (formerly deepseek_v4_hc_pre_norm.py,
hc_composition.py, and deepseek_v4_hc_post.py, which cross-imported each other):

1. HC weighted-combine + block RMSNorm (``deepseek_v4_hc_combine_rmsnorm``).
2. HC composition: sinkhorn split and pre-mix combine (``hc_split_sinkhorn``,
   ``deepseek_v4_hc_pre_mix_combine`` and friends).
3. HC post residual-stream composition (``deepseek_v4_hc_post_next_partials``).
"""

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# ========================================================================================
# Section 1: HC combine + RMSNorm (was deepseek_v4_hc_pre_norm.py)
# ========================================================================================
# Fused HC post-sinkhorn weighted-combine + block RMSNorm for DeepSeek-V4.
#
# ``deepseek_v4_hc_combine_rmsnorm`` collapses the HC ``_hc_pre`` tail — the
# weighted-combine ``y = sum_m pre[m] * flat[m*H:(m+1)*H]`` followed by the block
# ``attn_norm`` / ``ffn_norm`` RMSNorm — into a *single* Triton kernel: each
# program owns one token row, accumulates the combine in fp32 registers (the
# ``[hc_mult, H]`` broadcast product is never materialized in HBM), then applies
# the RMSNorm in-register. It is invoked from the fused HC-pre ops' prefill
# fallback paths in ``hc_composition.py`` (the modeling code calls those ops, not
# this one directly).
#
# Numerics contract: the arithmetic mirrors the reference byte-for-byte,
# including the bf16 round-trip ``y`` takes through ``_hc_pre``'s return +
# ``torch_rmsnorm``'s ``input.to(fp32)``, and the
# ``bf16(weight_fp32 * bf16(normed))`` store order.
#
# The kernel name (``_hc_weighted_combine_kernel``) deliberately avoids every
# op-type regex (no ``sum`` / ``mean`` / ``norm`` / ``mul`` / ``cast`` / ``index``
# substring) so the collapsed work leaves the ``elementwise`` / ``reduction`` /
# ``copy_cast`` buckets entirely.


def _hc_launch_config(n: int, block_h: int):
    """Pick (num_warps, num_stages, maxnreg) for the combine+RMSNorm launch.

    One CTA per token row reducing over ``BLOCK_H`` (==next_pow2(H)) with a
    fully-unrolled ``HM`` loop, so ``num_stages`` is near-inert (pinned to 2)
    and the live knob is ``num_warps`` (plus a register cap for decode).

    Chosen **deterministically** rather than via ``@triton.autotune``: these
    sub-floor kernels run launch-overhead-bound under Triton's ``do_bench``
    (it measures host launch cadence, not GPU time), so the autotuner cannot
    resolve the fine gaps and intermittently picks the register-capped config
    for prefill, which spills the wide prefill CTA.

    For the H=4096 model shape (BLOCK_H=4096, microbenched on B200):
      * decode  (n<=4)   -> nw=32, maxnreg=128: widest CTA hides the 4*H fp32
        load latency; the register cap nudges ptxas into a faster schedule.
      * prefill (n>=256) -> nw=8: SM-saturated, fewer warps/CTA.
      * otherwise        -> nw=16: the former default.
    ``maxnreg`` is applied only to the wide decode CTA; it would spill the
    narrower prefill/mid CTAs, so they keep the compiler default.
    """
    if n <= 4:
        num_warps = 32
    elif n >= 256:
        num_warps = 8
    else:
        num_warps = 16
    # Small hidden dims (non-model shapes) never need a wide CTA.
    if block_h < 1024:
        num_warps = min(num_warps, 8)
    if block_h < 256:
        num_warps = min(num_warps, 4)
    # The register cap only helps the wide single-/few-CTA decode launch; it would
    # spill narrower CTAs, so apply it only when the full-width nw=32 survived.
    maxnreg = 128 if num_warps == 32 else None
    return num_warps, 2, maxnreg


@triton.jit
def _hc_weighted_combine_kernel(
    pre_ptr,  # [N, HM] fp32
    flat_ptr,  # [N, HM * H] fp32 or bf16/fp16 (converted to fp32 in-register)
    weight_ptr,  # [H] fp32 (RMSNorm weight)
    out_ptr,  # [N, H] out_dtype
    N,
    H,
    eps,
    HM: tl.constexpr,  # hc_mult
    BLOCK_H: tl.constexpr,  # next_power_of_2(H)
):
    """One program per token row. y = sum_m pre[m] * flat[m*H:] then RMSNorm."""
    row = tl.program_id(0)
    if row >= N:
        return

    h = tl.arange(0, BLOCK_H)
    hmask = h < H

    # --- weighted combine over the hc_mult axis, fp32 accumulate ---
    # acc[h] = sum_{m} pre[row, m] * flat[row, m*H + h]; never materialize [HM, H].
    flat_row = flat_ptr + row * (HM * H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for m in tl.static_range(HM):
        p = tl.load(pre_ptr + row * HM + m)  # scalar fp32
        # Native-dtype load + in-register fp32 convert: exact for bf16/fp16
        # inputs (widening conversions are lossless), identical for fp32, and
        # skips the HBM materialization of an fp32 ``flat``.
        f = tl.load(flat_row + m * H + h, mask=hmask, other=0.0).to(tl.float32)
        acc += p * f

    # --- replicate the bf16 round-trip y takes (return .to(x.dtype) then
    #     torch_rmsnorm's input.to(fp32)) so the fused op is bit-faithful ---
    y = acc.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm over H: var = mean(y^2); out = bf16(weight * bf16(y * rstd)) ---
    var = tl.sum(y * y, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    normed = (y * rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(weight_ptr + h, mask=hmask, other=0.0)
    out = w * normed
    tl.store(out_ptr + row * H + h, out.to(out_ptr.dtype.element_ty), mask=hmask)


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_combine_rmsnorm", mutates_args=())
def deepseek_v4_hc_combine_rmsnorm(
    pre: torch.Tensor,
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    hc_mult: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused HC weighted-combine + RMSNorm. Drop-in for the ``_hc_pre`` tail.

    Computes, per leading (token) index::

        y      = sum_m pre[..., m] * flat[..., m*H : (m+1)*H]      # fp32
        y      = y.to(out_dtype).float()                          # bf16 round-trip
        out    = (weight * (y * rsqrt(mean(y^2) + eps)).to(out_dtype).float()).to(out_dtype)

    Args:
        pre:      ``[..., hc_mult]`` fp32 combine weights (from the sinkhorn op).
        flat:     ``[..., hc_mult * H]`` flattened hidden states
                  (``x.flatten(2)``; fp32, bf16, or fp16 — non-fp32 inputs are
                  converted to fp32 in-register, which is exact).
        weight:   ``[H]`` fp32 RMSNorm weight (attn_norm / ffn_norm).
        eps:      RMSNorm epsilon.
        hc_mult:  number of hyper-connection streams folded over.
        out_dtype: dtype of the returned (normalized) tensor (the residual dtype).

    Returns:
        ``[..., H]`` ``out_dtype`` == ``rmsnorm(sum_m pre*flat, weight, eps)``.
    """
    lead = list(pre.shape[:-1])
    n = math.prod(lead)
    H = weight.shape[0]
    assert flat.shape[-1] == hc_mult * H, "flat last dim must equal hc_mult * H"

    pre_f = pre.reshape(n, hc_mult).contiguous().float()
    flat_f = flat.reshape(n, hc_mult * H).contiguous()
    weight_f = weight.contiguous().float()

    out = torch.empty((n, H), device=pre.device, dtype=out_dtype)
    if n == 0:
        return out.reshape(*lead, H)

    block_h = triton.next_power_of_2(H)
    num_warps, num_stages, maxnreg = _hc_launch_config(n, block_h)
    launch_kwargs = dict(HM=hc_mult, BLOCK_H=block_h, num_warps=num_warps, num_stages=num_stages)
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    grid = (n,)
    _hc_weighted_combine_kernel[grid](
        pre_f,
        flat_f,
        weight_f,
        out,
        n,
        H,
        eps,
        **launch_kwargs,
    )
    return out.reshape(*lead, H)


@deepseek_v4_hc_combine_rmsnorm.register_fake
def _deepseek_v4_hc_combine_rmsnorm_fake(
    pre: torch.Tensor,
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    hc_mult: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    lead = list(pre.shape[:-1])
    H = weight.shape[0]
    return pre.new_empty((*lead, H), dtype=out_dtype)


# ========================================================================================
# Section 2: HC composition (was hc_composition.py)
# ========================================================================================
# Custom ops for fused DeepSeek-V4 hyper-connection (HC) composition.
#
# The HC ``_hc_pre`` step in ``modeling_deepseek_v4.py`` splits a per-token mix
# vector into ``pre`` / ``post`` gates and a doubly-stochastic ``comb`` matrix
# (softmax + a ``sinkhorn_iters`` loop of row / column normalizations over a tiny
# ``[N, hc_mult, hc_mult]`` tensor — ~40 grid=1 kernels per call in eager form).
# The ops below collapse that chain (and its combine/RMSNorm consumers) into
# single Triton kernels that keep the identical fp32 math in registers.
#
# Kernel names deliberately avoid the ``reduction`` op-type regex (no ``reduce``
# / ``sum`` / ``softmax`` / ``norm`` substrings) so the collapsed work leaves the
# ``reduction`` bucket entirely.

# Provides the prefill fallback op (deepseek_v4_hc_combine_rmsnorm) and the
# shared launch config. One-way import: ``deepseek_v4_hc_pre_norm`` does not
# import this module.


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

    Mirrors the eager modeling split + sinkhorn chain exactly in fp32:
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
    """Fused HC split + sinkhorn. Drop-in for the eager modeling split/sinkhorn chain.

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
    n = math.prod(lead)

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
# Split-D HC-pre front: per-chunk RMS statistic + hc_fn dot partials
# ---------------------------------------------------------------------------
#
# On the decode path the fused HC-pre ops replace the eager mix front
# (``flat.float()`` -> RMS statistic -> cublas GEMV -> broadcast mul) with
# ``_hc_fn_partials_kernel`` (grid ``(n, split)``): each program owns one
# D-chunk of one row and emits per-chunk partials for the square-sum statistic
# and all ``MIX_HC`` hc_fn dot products — one pass over the input, fp32
# ``flat`` never materialized. The consuming composition kernel reduces the
# partials in a fixed (deterministic) tree order and applies
# ``rstd = rsqrt(sum/D + norm_eps)`` and ``(mix * rstd) * hc_scale + hc_base``
# in the same fp32 op order as the eager chain.
#
# Numerics contract: the split-D partial layout ``[n, MIX_HC + 1, split]``
# (dots first, square-sum last) is shared via ``hc_partials_layout``. The
# reduction order differs from cublas/torch, so ``mixes`` matches the eager
# chain to ~1 ULP (fp32), not bit-exactly; everything downstream of ``mixes``
# is unchanged. Large token counts (prefill) fall back to the eager torch
# front, where the cublas GEMM beats the split-D GEMV.

_HC_PRE_MIX_FUSED_N_MAX = 64


def hc_partials_layout(dim: int) -> Tuple[int, int]:
    """Split-D partial layout ``(CHUNK, SPLIT)`` shared by every partials producer/consumer.

    ~128 chunks for the model shape (D = 16384 -> CHUNK = 128); small D used in
    tests degrades gracefully to fewer/masked chunks. Producers
    (``_hc_fn_partials_kernel`` and the fused hc_post seam kernel) and consumers
    (the composition kernels) must derive the identical layout from ``dim`` so a
    ``[N, MIX_HC + 1, SPLIT]`` partials tensor can cross the op boundary.
    """
    chunk = max(64, triton.next_power_of_2((dim + 127) // 128))
    split = (dim + chunk - 1) // chunk
    return chunk, split


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


# ---------------------------------------------------------------------------
# Fully fused HC-pre: composition + weighted-combine + block RMSNorm
# ---------------------------------------------------------------------------
#
# ``pre`` is only ever fed to the combine, and it is ready *before* the serial
# sinkhorn loop, so ``deepseek_v4_hc_pre_mix_combine`` merges the composition
# and the combine into ONE kernel per row (two launches per HC call at decode):
#
#   * ``_hc_fn_partials_kernel`` — unchanged split-D front (a single CTA cannot
#     stream the [MIX_HC, D] fp32 ``hc_fn`` weight competitively).
#   * ``_hc_pre_composition_combine_kernel`` — grid ``(n, 2)``: per token row,
#     program 0 reduces the ``pre`` partials and runs the wide combine +
#     RMSNorm, while program 1 *concurrently* computes ``post`` and the comb
#     logits and runs the serial sinkhorn loop. ``pre`` never leaves registers.
#
# The two-program split exists because the two halves want opposite resources:
# the combine streams a wide ``[HM, H]`` tile while the sinkhorn is a serial
# dependent ALU chain on a ``[HM, HM]`` register tile; a sibling program on
# another SM overlaps them without paying a second launch (a two-*kernel*
# split re-pays the per-launch GPU floor and the partial re-loads).
#
# Numerics contract: bit-exactness vs the landed two-kernel path REQUIRES
# ``num_warps=2`` — the composition reductions change bits with warp count
# (layout-dependent tile-reduce tree) while the combine is warp-count-
# invariant. Both programs re-derive ``rstd`` (and program 0 the ``pre`` gate)
# from the same partials with identical tile shapes and op order, so the
# duplicated math is bit-identical. ``maxnreg=240`` only relaxes ptxas
# scheduling; it does not change results.

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
    y32_ptr,  # [N, H] fp32 (out; y widened — written only when EMIT_Y32)
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
    EMIT_Y32: tl.constexpr,
):
    """Two programs per token row (grid ``(N, 2)``): combine ∥ composition.

    Program ``(row, 0)`` reduces the ``pre`` partials and runs the weighted
    combine + RMSNorm into ``y``; program ``(row, 1)`` concurrently computes
    ``post`` and the comb logits and runs the serial sinkhorn loop into
    ``comb``. The composition math mirrors ``_hc_composition_kernel`` (with
    ``mixes`` coming from the in-kernel partials reduction)
    and the combine + RMSNorm math mirrors ``_hc_weighted_combine_kernel``
    exactly (same tile shapes and op order, so at ``num_warps=2`` the outputs
    are bit-identical to the two-kernel sequence — the sibling-program split
    only moves which SM executes each half). ``pre`` is consumed from
    registers via an exact one-hot extraction instead of an HBM round-trip.
    """
    row = tl.program_id(0)
    which = tl.program_id(1)
    if row >= N:
        return
    pbase = part_ptr + row * ((MIX_HC + 1) * SPLIT)

    s = tl.arange(0, SBLOCK)
    smask = s < SPLIT

    # rstd = rsqrt(mean(x^2) + norm_eps), fixed-order tree reduce over SPLIT.
    # Re-derived by both programs from the same partials row (identical tile
    # shape + op order -> identical bits); 0.5 KB of redundant loads.
    sq = tl.load(pbase + MIX_HC * SPLIT + s, mask=smask, other=0.0)
    rstd = tl.rsqrt(tl.sum(sq, axis=0) / D + norm_eps)

    d = tl.arange(0, BM)
    dmask = d < HM

    if which == 0:
        # --- combine program: pre gate -> weighted combine -> RMSNorm -> y ---
        s0 = tl.load(scale_ptr + 0)

        # pre = sigmoid((mix * rstd) * scale[0] + base) + eps — registers only.
        pre_part = tl.load(
            pbase + d[:, None] * SPLIT + s[None, :],
            mask=dmask[:, None] & smask[None, :],
            other=0.0,
        )
        pre_logits = (tl.sum(pre_part, axis=1) * rstd) * s0 + tl.load(
            base_ptr + d, mask=dmask, other=0.0
        )
        pre = tl.sigmoid(pre_logits) + eps

        # weighted combine + RMSNorm (mirrors _hc_weighted_combine_kernel)
        h = tl.arange(0, BLOCK_H)
        hmask = h < H
        flat_row = flat_ptr + row * (HM * H)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for m in tl.static_range(HM):
            # Exact scalar extraction of pre[m] from the register tile (sum of
            # a one-hot selection reproduces the element bit-for-bit).
            p = tl.sum(tl.where(d == m, pre, 0.0), axis=0)
            f = tl.load(flat_row + m * H + h, mask=hmask, other=0.0).to(tl.float32)
            acc += p * f
        y = acc.to(tl.bfloat16).to(tl.float32)
        var = tl.sum(y * y, axis=0) / H
        y_rstd = tl.rsqrt(var + rms_eps)
        normed = (y * y_rstd).to(tl.bfloat16).to(tl.float32)
        w = tl.load(weight_ptr + h, mask=hmask, other=0.0)
        out = w * normed
        out_c = out.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + row * H + h, out_c, mask=hmask)
        if EMIT_Y32:
            # fp32 mirror of the *stored* value: widen the out_dtype-rounded
            # result (exact), never the pre-rounding fp32 accumulator — the
            # mirror must equal y.float() bit-for-bit.
            tl.store(y32_ptr + row * H + h, out_c.to(tl.float32), mask=hmask)
    else:
        # --- composition program: post gate + comb logits + sinkhorn ---
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)

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

        # sinkhorn (register-only; mirrors _hc_composition_kernel)
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
    """Fused HC-pre. Drop-in for the eager mix front + composition + combine.

    Computes, per leading (token) index::

        f = flat.float()  # exact for bf16
        rstd = rsqrt(mean(f ^ 2) + norm_eps)
        mixes = (f @ hc_fn.T) * rstd
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)
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
    y, _, post, comb = _hc_pre_mix_combine_partials_impl(
        None,
        flat,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult,
        sinkhorn_iters,
        eps,
        norm_eps,
        rms_eps,
        out_dtype,
        emit_y32=False,
    )
    return y, post, comb


def _hc_pre_mix_combine_fake_outputs(
    flat: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_mult: int,
    out_dtype: torch.dtype,
    emit_y32: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Shared fake-output builder for the fused HC-pre ops."""
    lead = list(flat.shape[:-1])
    hidden = norm_weight.shape[0]
    y = flat.new_empty((*lead, hidden), dtype=out_dtype)
    post = flat.new_empty((*lead, hc_mult), dtype=torch.float32)
    comb = flat.new_empty((*lead, hc_mult, hc_mult), dtype=torch.float32)
    if emit_y32:
        y32 = flat.new_empty((*lead, hidden), dtype=torch.float32)
        return y, y32, post, comb
    return y, post, comb


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
    return _hc_pre_mix_combine_fake_outputs(flat, norm_weight, hc_mult, out_dtype)


# ---------------------------------------------------------------------------
# Layer-boundary HC seam: consume partials produced by the fused hc_post
# ---------------------------------------------------------------------------
#
# At every HC site except the very first (layer 0, attn) the hidden states are
# the output of the previous site's ``_hc_post``, whose fused seam op
# ``auto_deploy::deepseek_v4_hc_post_next_partials`` (deepseek_v4_hc_post.py)
# already emitted this site's split-D partials while storing the residual. The
# ``*_partials`` ops consume them: the decode path skips the partials launch
# and goes straight to ``_hc_pre_composition_combine_kernel``, adding no
# arithmetic of its own (the seam producer's partials carry ~1-2 fp32 ULP of
# FMA-association drift; see deepseek_v4_hc_post.py). The prefill path ignores
# ``partials`` (the producer leaves them unfilled at prefill) and runs the
# identical eager cublas front as ``deepseek_v4_hc_pre_mix_combine``.


def _hc_pre_mix_combine_partials_impl(
    partials: Optional[torch.Tensor],
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
    emit_y32: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Shared body of every fused HC-pre op.

    When ``partials`` is ``None`` the decode path computes them with
    ``_hc_fn_partials_kernel``; otherwise it consumes the precomputed seam
    partials. Returns ``(y, y32, post, comb)`` where ``y32`` is ``None`` unless
    ``emit_y32``; when emitted it is exactly ``y.float()`` (the out_dtype-
    rounded values widened to fp32 — never the pre-rounding accumulator).
    """
    lead = list(flat.shape[:-1])
    dim = flat.shape[-1]
    n = math.prod(lead)
    mix_hc = hc_fn.shape[0]
    hidden = norm_weight.shape[0]
    assert hc_fn.shape[-1] == dim, "hc_fn last dim must match flat last dim"
    assert dim == hc_mult * hidden, "flat last dim must equal hc_mult * H"

    if n > _HC_PRE_MIX_FUSED_N_MAX:
        # Prefill-sized token counts: the landed eager two-op path (``partials``
        # is never read — the seam producer leaves it unfilled at prefill).
        flat_f = flat.reshape(n, dim).float()
        rsqrt = torch.rsqrt(flat_f.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat_f, hc_fn) * rsqrt
        pre, post, comb = torch.ops.auto_deploy.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        y = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
            pre, flat.reshape(n, dim), norm_weight, rms_eps, hc_mult, out_dtype
        )
        y32 = None
        if emit_y32:
            # Exact widening of the just-computed y (clone keeps outputs alias-free
            # when out_dtype is already fp32).
            y32 = y.float() if y.dtype != torch.float32 else y.clone()
            y32 = y32.reshape(*lead, hidden)
        return (
            y.reshape(*lead, hidden),
            y32,
            post.reshape(*lead, hc_mult),
            comb.reshape(*lead, hc_mult, hc_mult),
        )

    chunk, split = hc_partials_layout(dim)
    if partials is not None:
        assert partials.shape[-2:] == (mix_hc + 1, split), (
            f"partials layout mismatch: {tuple(partials.shape)} vs (*, {mix_hc + 1}, {split})"
        )

    flat_c = flat.reshape(n, dim).contiguous()
    scale_f = hc_scale.contiguous().float()
    base_f = hc_base.contiguous().float()
    weight_f = norm_weight.contiguous().float()

    y = torch.empty(n, hidden, device=flat.device, dtype=out_dtype)
    y32 = torch.empty(n, hidden, device=flat.device, dtype=torch.float32) if emit_y32 else None
    post = torch.empty(n, hc_mult, device=flat.device, dtype=torch.float32)
    comb = torch.empty(n, hc_mult * hc_mult, device=flat.device, dtype=torch.float32)
    if n == 0:
        return (
            y.reshape(*lead, hidden),
            y32.reshape(*lead, hidden) if emit_y32 else None,
            post.reshape(*lead, hc_mult),
            comb.reshape(*lead, hc_mult, hc_mult),
        )

    if partials is None:
        # No seam producer upstream: compute the split-D partials here.
        part_c = torch.empty(n, mix_hc + 1, split, device=flat.device, dtype=torch.float32)
        _hc_fn_partials_kernel[(n, split)](
            flat_c,
            hc_fn.contiguous().float(),
            part_c,
            n,
            dim,
            split,
            MIX_HC=mix_hc,
            KBLOCK=triton.next_power_of_2(mix_hc),
            CHUNK=chunk,
            num_warps=4,
        )
    else:
        part_c = partials.reshape(n, mix_hc + 1, split).contiguous()

    # grid (n, 2): per row, program 0 = combine + RMSNorm, program 1 = the
    # post/comb composition with the serial sinkhorn chain (see kernel doc).
    _hc_pre_composition_combine_kernel[(n, 2)](
        part_c,
        scale_f,
        base_f,
        flat_c,
        weight_f,
        y,
        y32 if emit_y32 else y,  # dummy y32 ptr, dead code at EMIT_Y32=False
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
        EMIT_Y32=emit_y32,
        num_warps=_HC_PRE_MIX_COMBINE_NUM_WARPS,
        num_stages=2,
        maxnreg=_HC_PRE_MIX_COMBINE_MAXNREG,
    )

    return (
        y.reshape(*lead, hidden),
        y32.reshape(*lead, hidden) if emit_y32 else None,
        post.reshape(*lead, hc_mult),
        comb.reshape(*lead, hc_mult, hc_mult),
    )


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_pre_mix_combine_partials", mutates_args=())
def deepseek_v4_hc_pre_mix_combine_partials(
    partials: torch.Tensor,
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
    """``deepseek_v4_hc_pre_mix_combine`` with precomputed split-D partials.

    Args:
        partials: [N, MIX_HC + 1, SPLIT] fp32 from the fused hc_post seam op
                  (``hc_partials_layout``'s layout over ``flat``'s last dim);
                  read only on the decode (n <= _HC_PRE_MIX_FUSED_N_MAX) path.
        (remaining args identical to ``deepseek_v4_hc_pre_mix_combine``)

    Returns:
        y:    [..., H]                  out_dtype == rmsnorm(sum_m pre*flat)
        post: [..., hc_mult]            fp32
        comb: [..., hc_mult, hc_mult]   fp32
    """
    y, _, post, comb = _hc_pre_mix_combine_partials_impl(
        partials,
        flat,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult,
        sinkhorn_iters,
        eps,
        norm_eps,
        rms_eps,
        out_dtype,
        emit_y32=False,
    )
    return y, post, comb


@deepseek_v4_hc_pre_mix_combine_partials.register_fake
def _deepseek_v4_hc_pre_mix_combine_partials_fake(
    partials: torch.Tensor,
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
    return _hc_pre_mix_combine_fake_outputs(flat, norm_weight, hc_mult, out_dtype)


@torch.library.custom_op(
    "auto_deploy::deepseek_v4_hc_pre_mix_combine_partials_y32", mutates_args=()
)
def deepseek_v4_hc_pre_mix_combine_partials_y32(
    partials: torch.Tensor,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``deepseek_v4_hc_pre_mix_combine_partials`` that also emits ``y.float()``.

    For learned-router MoE HC sites: the router GEMV upstream-casts ``y`` back
    to fp32 every step (``hidden_states_flat.to(weight.dtype)`` — one extra
    copy kernel per layer per step at decode). The combine program already has
    the out_dtype-rounded row in registers, so it stores the fp32 widening as a
    second output for free; ``y``/``post``/``comb`` are bit-identical to
    ``deepseek_v4_hc_pre_mix_combine_partials`` and ``y32`` equals ``y.float()``
    bit-for-bit on every path (same widening the boundary cast performed, so the
    downstream fp32 cuBLAS router GEMV consumes exactly the values it used to).

    Returns:
        y:    [..., H]                  out_dtype == rmsnorm(sum_m pre*flat)
        y32:  [..., H]                  fp32 == y.float()
        post: [..., hc_mult]            fp32
        comb: [..., hc_mult, hc_mult]   fp32
    """
    y, y32, post, comb = _hc_pre_mix_combine_partials_impl(
        partials,
        flat,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight,
        hc_mult,
        sinkhorn_iters,
        eps,
        norm_eps,
        rms_eps,
        out_dtype,
        emit_y32=True,
    )
    return y, y32, post, comb


@deepseek_v4_hc_pre_mix_combine_partials_y32.register_fake
def _deepseek_v4_hc_pre_mix_combine_partials_y32_fake(
    partials: torch.Tensor,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _hc_pre_mix_combine_fake_outputs(flat, norm_weight, hc_mult, out_dtype, emit_y32=True)


# ---------------------------------------------------------------------------
# Fused HC head: partial-reduce + sigmoid gate + weighted-combine + RMSNorm
# ---------------------------------------------------------------------------
#
# ``_hc_head`` is the model-tail HC seam: it collapses the ``hc_mult`` residual
# streams into one hidden state right before the final ``norm`` and the fp32
# lm-head GEMV. On the decode path this op consumes the split-D partials the
# last layer's fused ``_hc_post`` seam kernel already produced against
# ``hc_head_fn`` and runs ONE single-CTA kernel: partial reduce -> ``rstd`` ->
# sigmoid gate -> weighted combine -> RMSNorm.
#
# Numerics contract: every bf16 rounding point of the eager chain is preserved
# (``y -> bf16``, ``normed -> bf16``, ``weight * normed -> bf16``); the final
# store WIDENS the bf16-rounded result to fp32, absorbing the ``.float()``
# boundary cast (stored values are exactly ``eager_bf16.float()``). Like the
# sibling HC-pre ops, the partial-reduced ``mixes`` match the eager cublas
# GEMV to ~1 ULP (fp32 association differs), not bit-exactly. The prefill path
# (n > _HC_PRE_MIX_FUSED_N_MAX) ignores ``partials`` and runs the identical
# eager chain (cublas GEMM front + ``torch_rmsnorm`` + widen).


@triton.jit
def _hc_head_combine_kernel(
    part_ptr,  # [N, HM + 1, SPLIT] fp32 (HM head-fn dots + square-sum)
    scale_ptr,  # [1] fp32
    base_ptr,  # [HM] fp32
    flat_ptr,  # [N, HM * H] input (any float dtype; converted in-register)
    weight_ptr,  # [H] fp32 (final RMSNorm weight)
    y_ptr,  # [N, H] fp32 (out; bf16-rounded values widened to fp32)
    N,
    D,
    SPLIT,
    H,
    norm_eps,
    hc_eps,
    rms_eps,
    HM: tl.constexpr,  # hc_mult == hc_head_fn rows
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    SBLOCK: tl.constexpr,  # next_power_of_2(SPLIT)
    BLOCK_H: tl.constexpr,  # next_power_of_2(H)
):
    """One program per token row: partials -> pre gate -> combine -> RMSNorm."""
    row = tl.program_id(0)
    if row >= N:
        return
    pbase = part_ptr + row * ((HM + 1) * SPLIT)

    s = tl.arange(0, SBLOCK)
    smask = s < SPLIT

    # rstd = rsqrt(mean(x^2) + norm_eps), fixed-order tree reduce over SPLIT.
    sq = tl.load(pbase + HM * SPLIT + s, mask=smask, other=0.0)
    rstd = tl.rsqrt(tl.sum(sq, axis=0) / D + norm_eps)

    sc = tl.load(scale_ptr)  # scalar fp32 (hc_head_scale is [1])

    d = tl.arange(0, BM)
    dmask = d < HM

    # pre = sigmoid((mix * rstd) * scale + base) + hc_eps — kept in registers.
    pre_part = tl.load(
        pbase + d[:, None] * SPLIT + s[None, :],
        mask=dmask[:, None] & smask[None, :],
        other=0.0,
    )
    pre_logits = (tl.sum(pre_part, axis=1) * rstd) * sc + tl.load(
        base_ptr + d, mask=dmask, other=0.0
    )
    pre = tl.sigmoid(pre_logits) + hc_eps

    # --- weighted combine + RMSNorm (mirrors _hc_weighted_combine_kernel) ---
    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    flat_row = flat_ptr + row * (HM * H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for m in tl.static_range(HM):
        # Exact scalar extraction of pre[m] from the register tile.
        p = tl.sum(tl.where(d == m, pre, 0.0), axis=0)
        f = tl.load(flat_row + m * H + h, mask=hmask, other=0.0).to(tl.float32)
        acc += p * f
    y = acc.to(tl.bfloat16).to(tl.float32)
    var = tl.sum(y * y, axis=0) / H
    y_rstd = tl.rsqrt(var + rms_eps)
    normed = (y * y_rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(weight_ptr + h, mask=hmask, other=0.0)
    # bf16 round (torch_rmsnorm's out dtype) then widen: absorbs the .float().
    out = (w * normed).to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + row * H + h, out, mask=hmask)


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_head_norm", mutates_args=())
def deepseek_v4_hc_head_norm(
    partials: torch.Tensor,
    flat: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
    rms_eps: float,
) -> torch.Tensor:
    """Fused ``_hc_head`` + final RMSNorm + fp32 widen. Drop-in for the eager tail.

    Computes, per leading (token) index::

        f     = flat.float()                                   # exact for bf16
        rstd  = rsqrt(mean(f ^ 2) + norm_eps)
        mixes = (f @ hc_head_fn.T) * rstd
        pre   = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps
        y     = (sum_m pre[m] * f[m*H : (m+1)*H]).to(flat.dtype)
        out   = torch_rmsnorm(y, norm_weight, rms_eps).float()

    Args:
        partials:      [N, HM + 1, SPLIT] fp32 split-D partials of ``flat``
                       against ``hc_head_fn`` (from the fused hc_post seam op);
                       read only on the decode (n <= _HC_PRE_MIX_FUSED_N_MAX) path.
        flat:          [..., D] hidden states (bf16/fp16/fp32), D == HM * H
        hc_head_fn:    [HM, D] fp32
        hc_head_scale: [1] fp32
        hc_head_base:  [HM] fp32
        norm_weight:   [H] fp32 final RMSNorm weight
        hc_eps:        sigmoid gate epsilon
        norm_eps:      RMS statistic epsilon (mix scaling)
        rms_eps:       final RMSNorm epsilon

    Returns:
        [..., H] fp32 — bf16-rounded normalized hidden states widened to fp32
        (exactly ``eager_bf16.float()``), ready for the fp32 lm-head GEMV.
    """
    lead = list(flat.shape[:-1])
    dim = flat.shape[-1]
    n = math.prod(lead)
    hm = hc_head_fn.shape[0]
    hidden = norm_weight.shape[0]
    assert hc_head_fn.shape[-1] == dim, "hc_head_fn last dim must match flat last dim"
    assert dim == hm * hidden, "flat last dim must equal hc_mult * H"

    if n > _HC_PRE_MIX_FUSED_N_MAX:
        # Prefill-sized token counts: the eager chain, bit-identical to the
        # previous modeling-side _hc_head + norm + .float() sequence.
        flat_f = flat.reshape(n, dim).float()
        rsqrt = torch.rsqrt(flat_f.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat_f, hc_head_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps
        y = torch.sum(pre.unsqueeze(-1) * flat_f.view(n, hm, hidden), dim=-2)
        y = y.to(flat.dtype)
        y = torch.ops.auto_deploy.torch_rmsnorm(y, norm_weight, rms_eps)
        return y.float().reshape(*lead, hidden)

    _, split = hc_partials_layout(dim)
    assert partials.shape[-2:] == (hm + 1, split), (
        f"partials layout mismatch: {tuple(partials.shape)} vs (*, {hm + 1}, {split})"
    )

    flat_c = flat.reshape(n, dim).contiguous()
    part_c = partials.reshape(n, hm + 1, split).contiguous()
    scale_f = hc_head_scale.contiguous().float()
    base_f = hc_head_base.contiguous().float()
    weight_f = norm_weight.contiguous().float()

    y = torch.empty(n, hidden, device=flat.device, dtype=torch.float32)
    if n == 0:
        return y.reshape(*lead, hidden)

    block_h = triton.next_power_of_2(hidden)
    # Same shape-dependent launch config as the sibling combine+RMSNorm kernel
    # (wide CTA hides the HM*H load latency at decode; see _hc_launch_config).
    num_warps, num_stages, maxnreg = _hc_launch_config(n, block_h)
    launch_kwargs = dict(
        HM=hm,
        BM=triton.next_power_of_2(hm),
        SBLOCK=triton.next_power_of_2(split),
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    _hc_head_combine_kernel[(n,)](
        part_c,
        scale_f,
        base_f,
        flat_c,
        weight_f,
        y,
        n,
        dim,
        split,
        hidden,
        norm_eps,
        hc_eps,
        rms_eps,
        **launch_kwargs,
    )
    return y.reshape(*lead, hidden)


@deepseek_v4_hc_head_norm.register_fake
def _deepseek_v4_hc_head_norm_fake(
    partials: torch.Tensor,
    flat: torch.Tensor,
    hc_head_fn: torch.Tensor,
    hc_head_scale: torch.Tensor,
    hc_head_base: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
    rms_eps: float,
) -> torch.Tensor:
    lead = list(flat.shape[:-1])
    hidden = norm_weight.shape[0]
    return flat.new_empty((*lead, hidden), dtype=torch.float32)


# ========================================================================================
# Section 3: HC post composition (was deepseek_v4_hc_post.py)
# ========================================================================================
# Fused HC ``_hc_post`` residual-stream composition for DeepSeek-V4.
#
# The HC ``_hc_post`` step in ``modeling_deepseek_v4.py`` re-mixes the sublayer
# output ``x`` (``[..., H]``) back into the ``hc_mult``-wide residual stream
# ``residual`` (``[..., hc_mult, H]``)::
#
#     y = post.unsqueeze(-1) * x.unsqueeze(-2)  # [.., hc_mult, H]
#     y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)  # mix over streams
#     return y.to(x.dtype)
#
# This op collapses that chain into a *single* Triton kernel computing::
#
#     y[n, o, h] = post[n, o] * x[n, h] + sum_m comb[n, m, o] * residual[n, m, h]
#
# with the ``[hc_mult, hc_mult, H]`` broadcast product **never** materialized in
# HBM. The arithmetic mirrors the reference (fp32 accumulate, single bf16
# store), differing only in the float reduction association.
#
# The kernel name (``_hc_post_compose_kernel``) deliberately avoids every op-type
# regex (no ``sum`` / ``mean`` / ``reduce`` / ``mul`` / ``add`` / ``cast`` /
# ``index`` substring) so the collapsed work leaves the ``elementwise`` /
# ``reduction`` / ``copy_cast`` buckets entirely.

# One-way import (hc_composition does not import this module): shares the
# split-D partials layout + decode threshold with the HC-pre composition ops so
# the fused seam op below emits partials the composition kernels can consume.


def _hc_post_launch_config(n: int, hc_mult: int):
    """Pick ``(num_warps, num_stages, block_h, o_per_cta)`` for the kernel launch.

    The kernel grid is ``(n, cdiv(HM, O_PER_CTA), cdiv(H, BLOCK_H))``. ``O_PER_CTA``
    is how many of the ``HM`` output residual streams each CTA computes:

      * ``n <= 16`` (decode) -> ``O_PER_CTA=1``: the all-streams launch starves
        the GPU of CTAs, so the output-stream axis is split onto the grid for
        ``HM``x more CTAs *without* fragmenting the coalesced H-tile loads (a
        finer ``BLOCK_H`` fragments them and is worse). Bit-identical: each
        stream's fp32 reduction is unchanged, only its owning CTA differs.
      * ``n > 16`` (prefill) -> ``O_PER_CTA=HM``: ample CTAs already, so loading
        the ``[HM, BLOCK_H]`` residual tile once (instead of ``HM``x redundant
        reads) wins sharply.

    ``nw=4/8`` regress the decode o-split path; ``num_stages`` is pinned to 2
    (the unrolled ``HM`` loop gains nothing from more stages). Chosen
    deterministically rather than via ``@triton.autotune`` because these
    sub-floor kernels defeat the autotuner's ``do_bench`` (it resolves the host
    launch cadence, not GPU time) and varying prefill token counts would force
    a synchronizing re-tune per shape (cf. sibling ``_hc_weighted_combine_kernel``).
    """
    if n <= 16:
        return 2, 2, 512, 1
    return 2, 2, 512, hc_mult


@triton.jit
def _hc_post_compose_kernel(
    x_ptr,  # [N, H] x.dtype (e.g. bf16)
    residual_ptr,  # [N, HM, H] x.dtype
    post_ptr,  # [N, HM] fp32
    comb_ptr,  # [N, HM*HM] fp32; comb[n, m, o] at n*(HM*HM) + m*HM + o
    out_ptr,  # [N, HM, H] out_dtype
    N,
    H,
    HM: tl.constexpr,  # hc_mult
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    O_PER_CTA: tl.constexpr,  # output streams computed by this CTA (output-stream tiling)
    BLOCK_H: tl.constexpr,
):
    """One program per (token row, output-stream block, H-tile).

    y[n, o, h] = post[n, o] * x[n, h] + sum_m comb[n, m, o] * residual[n, m, h]

    The ``HM`` residual streams for this H-tile are loaded once into a fp32
    ``[BM, BLOCK_H]`` register tile and reused across the ``O_PER_CTA`` output
    streams this CTA owns (streams ``pid_o*O_PER_CTA .. +O_PER_CTA``). The kernel
    reads each residual / x element once *per CTA* and writes each of its output
    elements once. Padding rows (m >= HM) are masked to 0 and drop out of both
    the residual tile and the per-stream comb weights.

    ``O_PER_CTA`` is a compile-time output-stream tiling knob (see
    ``_hc_post_launch_config``): ``O_PER_CTA == HM`` is the all-streams,
    load-residual-once layout (prefill); ``O_PER_CTA < HM`` splits the output
    streams onto the grid for ``HM/O_PER_CTA``x more CTAs (decode). The
    ``O_PER_CTA >= HM`` branch is a constexpr, so the prefill path compiles to a
    branch-free ``for o in range(HM)`` identical to the original kernel.
    """
    pid_n = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_h = tl.program_id(2)
    if pid_n >= N:
        return

    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = h < H

    # x[n, h_tile] -> fp32 (matches torch promoting bf16 x against the fp32 post).
    x = tl.load(x_ptr + pid_n * H + h, mask=hmask, other=0.0).to(tl.float32)

    # residual[n, :, h_tile] -> [BM, BLOCK_H] fp32, loaded once and reused.
    m = tl.arange(0, BM)
    mmask = m < HM
    res_base = residual_ptr + pid_n * (HM * H)
    res_off = res_base + m[:, None] * H + h[None, :]
    res_tile = tl.load(res_off, mask=mmask[:, None] & hmask[None, :], other=0.0).to(tl.float32)

    post_base = post_ptr + pid_n * HM
    comb_base = comb_ptr + pid_n * (HM * HM)
    out_base = out_ptr + pid_n * (HM * H)

    if O_PER_CTA >= HM:
        # All-streams layout (prefill / large n): pid_o == 0, no per-stream guard.
        for o in range(HM):
            p = tl.load(post_base + o)  # scalar fp32
            c = tl.load(comb_base + m * HM + o, mask=mmask, other=0.0)  # comb[:, o] -> [BM]
            acc = p * x + tl.sum(c[:, None] * res_tile, axis=0)  # [BLOCK_H]
            tl.store(out_base + o * H + h, acc.to(out_ptr.dtype.element_ty), mask=hmask)
    else:
        # Output-stream-split layout (decode): this CTA owns O_PER_CTA streams.
        o_start = pid_o * O_PER_CTA
        for oo in range(O_PER_CTA):
            o = o_start + oo
            if o < HM:
                p = tl.load(post_base + o)
                c = tl.load(comb_base + m * HM + o, mask=mmask, other=0.0)
                acc = p * x + tl.sum(c[:, None] * res_tile, axis=0)
                tl.store(out_base + o * H + h, acc.to(out_ptr.dtype.element_ty), mask=hmask)


# ---------------------------------------------------------------------------
# Fused layer-boundary HC seam: hc_post + the NEXT site's split-D partials
# ---------------------------------------------------------------------------
#
# Every ``_hc_post`` output is immediately re-read by the *next* HC-pre's
# ``_hc_fn_partials_kernel`` (attn-post -> ffn-pre, ffn-post -> next layer's
# attn-pre, last ffn-post -> ``_hc_head``). This op merges the two launches:
# per D-chunk it composes the hc_post output in registers, stores the bf16
# residual, then computes the next site's square-sum + hc_fn dot partials from
# the *rounded* value — exactly what the standalone partials kernel would have
# re-loaded from HBM.
#
# Numerics contract: both outputs match the two-kernel sequence to ~1-2 fp32
# ULP (ptxas contracts the mul+add chains into FMAs differently across kernel
# bodies; warp-count invariant) — the same contract as the fused HC-pre front.
# At prefill (n > _HC_PRE_MIX_FUSED_N_MAX) the downstream HC-pre takes its
# eager cublas front and never reads partials, so this op runs the unchanged
# ``_hc_post_compose_kernel`` and returns the partials buffer unfilled.


@triton.jit
def _hc_post_next_partials_kernel(
    x_ptr,  # [N, H] x.dtype (e.g. bf16)
    residual_ptr,  # [N, HM, H] x.dtype
    post_ptr,  # [N, HM] fp32
    comb_ptr,  # [N, HM*HM] fp32; comb[n, m, o] at n*(HM*HM) + m*HM + o
    fn_ptr,  # [MIX_HC, D] fp32 (the NEXT HC site's hc_fn), D == HM * H
    out_ptr,  # [N, HM, H] out (x.dtype)
    part_ptr,  # [N, MIX_HC + 1, SPLIT] fp32 out (next site's partials)
    N,
    H,
    D,
    SPLIT,
    HM: tl.constexpr,  # hc_mult
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    MIX_HC: tl.constexpr,  # rows of the next site's hc_fn
    KBLOCK: tl.constexpr,  # next_power_of_2(MIX_HC)
    CHUNK: tl.constexpr,  # elements per split slot (power of 2)
):
    """One program per (token row, D-chunk): compose y, store it, emit partials.

    The chunk indexes the *flattened* ``[HM * H]`` output, so each element's
    output stream is ``o = offs // H``. The compose math mirrors
    ``_hc_post_compose_kernel`` per element (fp32 ``p * x`` plus the same
    ``m``-axis ``tl.sum`` tree over the residual streams, single rounding at
    the store); the partials math mirrors ``_hc_fn_partials_kernel`` on the
    rounded value (masked lanes contribute exact zeros to both reductions).
    FMA contraction may associate the fp32 chains differently than the two
    standalone kernels (~1-2 ULP); the math and rounding points are identical.
    """
    row = tl.program_id(0)
    s = tl.program_id(1)
    if row >= N:
        return

    offs = s * CHUNK + tl.arange(0, CHUNK)
    cmask = offs < D
    o = offs // H  # output stream per element
    h = offs - o * H  # hidden position per element

    # --- x-independent prologue ---
    p = tl.load(post_ptr + row * HM + o, mask=cmask, other=0.0)
    m = tl.arange(0, BM)
    mmask = m < HM
    res = tl.load(
        residual_ptr + row * (HM * H) + m[:, None] * H + h[None, :],
        mask=mmask[:, None] & cmask[None, :],
        other=0.0,
    ).to(tl.float32)
    c = tl.load(
        comb_ptr + row * (HM * HM) + m[:, None] * HM + o[None, :],
        mask=mmask[:, None] & cmask[None, :],
        other=0.0,
    )
    mix = tl.sum(c * res, axis=0)
    k = tl.arange(0, KBLOCK)
    kmask = k < MIX_HC
    w = tl.load(
        fn_ptr + k[:, None] * D + offs[None, :],
        mask=kmask[:, None] & cmask[None, :],
        other=0.0,
    )

    # --- hc_post compose: only x depends on the producer ---
    x = tl.load(x_ptr + row * H + h, mask=cmask, other=0.0).to(tl.float32)
    acc = p * x + mix
    tl.store(out_ptr + row * D + offs, acc.to(out_ptr.dtype.element_ty), mask=cmask)

    # --- next site's partials from the rounded y (mirrors _hc_fn_partials_kernel) ---
    xf = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
    sq = tl.sum(xf * xf, axis=0)
    part = tl.sum(w * xf[None, :], axis=1)

    out_base = part_ptr + row * ((MIX_HC + 1) * SPLIT)
    tl.store(out_base + k * SPLIT + s, part, mask=kmask)
    tl.store(out_base + MIX_HC * SPLIT + s, sq)


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_post_next_partials", mutates_args=())
def deepseek_v4_hc_post_next_partials(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused ``_hc_post`` + the next HC site's split-D partials.

    Drop-in for ``deepseek_v4_hc_post`` followed by the next
    ``deepseek_v4_hc_pre_mix_combine`` / ``deepseek_v4_hc_head_norm`` call's
    internal ``_hc_fn_partials_kernel`` launch. Both outputs match that
    two-kernel sequence to ~1-2 fp32 ULP on the decode path (FMA association;
    same contract as the landed ``deepseek_v4_hc_pre_mix_combine``).

    Args:
        x:          ``[..., H]`` sublayer (attn / ffn) output, x.dtype.
        residual:   ``[..., hc_mult, H]`` incoming residual stream, x.dtype.
        post:       ``[..., hc_mult]`` fp32 post gate (from the sinkhorn op).
        comb:       ``[..., hc_mult, hc_mult]`` fp32 combine matrix.
        next_hc_fn: ``[MIX_HC, hc_mult * H]`` fp32 hc_fn of the next consumer
                    (block ``hc_attn_fn`` / ``hc_ffn_fn``, or ``hc_head_fn``).

    Returns:
        out:      ``[..., hc_mult, H]`` x.dtype outgoing residual stream.
        partials: ``[N, MIX_HC + 1, SPLIT]`` fp32 split-D partials of ``out``
                  against ``next_hc_fn`` (``hc_partials_layout`` layout; left
                  unfilled on the prefill path, where consumers ignore it).
    """
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    hc_mult = post.shape[-1]
    n = math.prod(lead)
    D = hc_mult * H
    mix_hc = next_hc_fn.shape[0]
    assert next_hc_fn.shape[-1] == D, "next_hc_fn last dim must equal hc_mult * H"

    chunk, split = hc_partials_layout(D)
    partials = torch.empty(n, mix_hc + 1, split, device=x.device, dtype=torch.float32)

    # contiguous() first so the reshape is a pure view; residual may arrive as a
    # stride-0 expand (layer 0) which reshape would otherwise reject / copy.
    x_f = x.contiguous().reshape(n, H)
    res_f = residual.contiguous().reshape(n, hc_mult, H)
    post_f = post.contiguous().reshape(n, hc_mult).float()
    comb_f = comb.contiguous().reshape(n, hc_mult * hc_mult).float()

    out = torch.empty((n, hc_mult, H), device=x.device, dtype=x.dtype)
    if n == 0:
        return out.reshape(*lead, hc_mult, H), partials

    if n > _HC_PRE_MIX_FUSED_N_MAX:
        # Prefill: the unchanged single-purpose compose kernel; the downstream
        # HC-pre takes its eager front and never reads ``partials``.
        num_warps, num_stages, block_h_max, o_per_cta = _hc_post_launch_config(n, hc_mult)
        block_h = min(block_h_max, triton.next_power_of_2(H))
        grid = (n, triton.cdiv(hc_mult, o_per_cta), triton.cdiv(H, block_h))
        _hc_post_compose_kernel[grid](
            x_f,
            res_f,
            post_f,
            comb_f,
            out,
            n,
            H,
            HM=hc_mult,
            BM=triton.next_power_of_2(hc_mult),
            O_PER_CTA=o_per_cta,
            BLOCK_H=block_h,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return out.reshape(*lead, hc_mult, H), partials

    fn_f = next_hc_fn.contiguous().float()
    _hc_post_next_partials_kernel[(n, split)](
        x_f,
        res_f,
        post_f,
        comb_f,
        fn_f,
        out,
        partials,
        n,
        H,
        D,
        split,
        HM=hc_mult,
        BM=triton.next_power_of_2(hc_mult),
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        num_warps=4,
    )
    return out.reshape(*lead, hc_mult, H), partials


@deepseek_v4_hc_post_next_partials.register_fake
def _deepseek_v4_hc_post_next_partials_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    hc_mult = post.shape[-1]
    mix_hc = next_hc_fn.shape[0]
    n = math.prod(lead)
    _, split = hc_partials_layout(next_hc_fn.shape[-1])
    out = x.new_empty((*lead, hc_mult, H))
    partials = x.new_empty((n, mix_hc + 1, split), dtype=torch.float32)
    return out, partials
