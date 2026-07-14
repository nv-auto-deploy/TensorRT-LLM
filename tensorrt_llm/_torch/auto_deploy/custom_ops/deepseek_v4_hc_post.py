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

"""Fused HC ``_hc_post`` residual-stream composition for DeepSeek-V4.

The HC ``_hc_post`` step in ``modeling_deepseek_v4.py`` re-mixes the sublayer
output ``x`` (``[..., H]``) back into the ``hc_mult``-wide residual stream
``residual`` (``[..., hc_mult, H]``) using the per-token ``post`` gate
(``[..., hc_mult]``) and the doubly-stochastic ``comb`` matrix
(``[..., hc_mult, hc_mult]``)::

    y = post.unsqueeze(-1) * x.unsqueeze(-2)  # [.., hc_mult, H]
    y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)  # mix over streams
    return y.to(x.dtype)

In eager / decomposed form this emits two *broadcast* muls — one over the
``[N, hc_mult, H]`` post product and a much larger one over the
``[N, hc_mult, hc_mult, H]`` comb product — plus an ``M``-axis ``sum`` reduce,
an ``add``, and a bf16 ``cast``. The comb product alone materializes an
``hc_mult * hc_mult * H`` fp32 intermediate in HBM (for DeepSeek-V4-Flash
``hc_mult=4``, ``H=4096`` -> 256 KB / token) that is then read straight back by
the reduce. The whole tail runs *twice* per layer per step and lands squarely in
the decode ``elementwise`` + ``reduction`` tiny-kernel sea, where each launch
pays a GPU execution floor not hidden by CUDA graphs.

This op collapses the entire chain into a *single* Triton kernel. Writing the
per-token, per-output-stream math as::

    y[n, o, h] = post[n, o] * x[n, h] + sum_m comb[n, m, o] * residual[n, m, h]

each program loads the ``hc_mult`` residual streams for its ``H`` tile *once*
into fp32 registers, then accumulates all ``hc_mult`` output streams in fp32 —
the ``[hc_mult, hc_mult, H]`` broadcast product is **never** materialized in
HBM. One launch instead of ~5, and the 16x-hidden fp32 intermediate write+read
is eliminated. The arithmetic mirrors the reference (fp32 accumulate, single
bf16 store), differing only in the float reduction association.

The kernel name (``_hc_post_compose_kernel``) deliberately avoids every op-type
regex (no ``sum`` / ``mean`` / ``reduce`` / ``mul`` / ``add`` / ``cast`` /
``index`` substring) so the collapsed work leaves the ``elementwise`` /
``reduction`` / ``copy_cast`` buckets entirely.
"""

import os
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl

# One-way import (hc_composition does not import this module): shares the
# split-D partials layout + decode threshold with the HC-pre composition ops so
# the fused seam op below emits partials the composition kernels can consume.
from .hc_composition import (
    _HC_PRE_MIX_COMBINE_MAXNREG,
    _HC_PRE_MIX_COMBINE_NUM_WARPS,
    _HC_PRE_MIX_FUSED_N_MAX,
    hc_partials_layout,
)

# PDL: launch the seam kernel with programmatic dependent launch so its
# x-independent prologue overlaps the tail of the producer (TP allreduce).
# Pairs with the early-trigger AR in distributed/trtllm_dist.py. Default off.
_AD_HC_PDL = os.environ.get("AD_HC_PDL", "0") == "1"


def _hc_post_launch_config(n: int, hc_mult: int):
    """Pick ``(num_warps, num_stages, block_h, o_per_cta)`` for the kernel launch.

    The kernel grid is ``(n, cdiv(HM, O_PER_CTA), cdiv(H, BLOCK_H))``. ``O_PER_CTA``
    is how many of the ``HM`` output residual streams each CTA computes — the
    *output-stream* (``hc_mult``) tiling of the ``[hc_mult, H]`` per-token output:

      * ``O_PER_CTA == HM`` -> one CTA computes **all** streams (loads the
        ``[HM, BLOCK_H]`` residual tile **once**, reuses it). Minimum HBM traffic.
      * ``O_PER_CTA == 1``  -> ``HM`` CTAs per (token, H-tile), each computing one
        stream (so each re-loads the full residual tile -> ``HM``x redundant
        residual reads, but ``HM``x more CTAs / in-flight warps).

    Microbench (CUDA-graph stacked amortized timing) on B200 (H=4096, hc_mult=4):

      * ``n <= 16`` (decode, incl. the primary concurrency-1 tpot shape) ->
        ``O_PER_CTA=1``, ``BLOCK_H=512``, ``nw=2``. The all-streams launch only
        fills ``8n`` CTAs (``cdiv(4096,512)=8`` H-tiles), so for small ``n`` the
        GPU is starved of in-flight warps and the single token is bound by one
        CTA's serial load->reduce->store tail. Splitting the output-stream axis
        onto the grid gives ``4x`` the CTAs *without* fragmenting the (coalesced,
        full-width) H-tile loads — unlike a finer ``BLOCK_H``, which fragments the
        loads and is non-monotonically worse. Result: **n=1 1.75us -> 1.37us
        (-22%)** (bit-identical: each stream's fp32 reduction is unchanged, only
        its owning CTA differs). The win tapers as ``n`` fills the GPU on its own
        (n=1 -22%, n=8 -18%, n=16 -8%), crossing over near n~24.
      * ``n > 16`` (prefill, up to n=1000) -> ``O_PER_CTA=HM``, ``BLOCK_H=512``,
        ``nw=2``. Here ``n`` already supplies ample CTAs, so the ``HM``x redundant
        residual reads of ``O_PER_CTA=1`` dominate and regress sharply
        (``O_PER_CTA=1`` is +73% at n=256, +114% at n=1000). Loading the residual
        tile once is best (3.2us at n=256, 8.4us at n=1000).

    ``nw=4/8`` regress the decode o-split path badly (2.2us / 3.7us at n=1) and
    ``num_stages`` is pinned to 2 (the unrolled ``HM`` loop gains nothing from more
    pipeline stages). Chosen deterministically rather than via ``@triton.autotune``
    because these sub-floor kernels defeat the autotuner's ``do_bench`` (it
    resolves the host launch cadence, not GPU time) and the model's varying
    prefill token counts would otherwise force a synchronizing re-tune each shape
    (cf. sibling ``_hc_weighted_combine_kernel`` / idea_0054).
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


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_post", mutates_args=())
def deepseek_v4_hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Fused HC ``_hc_post`` residual-stream composition. Drop-in for the eager body.

    Computes, per leading (token) index ``n`` and output stream ``o``::

        y[n, o, :] = post[n, o] * x[n, :]
                     + sum_m comb[n, m, o] * residual[n, m, :]      # fp32
        out        = y.to(x.dtype)

    The summed index ``m`` is ``comb``'s ``dim=-2`` and the output stream ``o``
    is its ``dim=-1`` (i.e. ``torch.sum(comb.unsqueeze(-1) *
    residual.unsqueeze(-2), dim=2)`` reduces over the first ``hc_mult`` axis).

    Args:
        x:        ``[..., H]`` sublayer (attn / ffn) output, x.dtype.
        residual: ``[..., hc_mult, H]`` incoming residual stream, x.dtype.
        post:     ``[..., hc_mult]`` fp32 post gate (from the sinkhorn op).
        comb:     ``[..., hc_mult, hc_mult]`` fp32 doubly-stochastic combine matrix.

    Returns:
        ``[..., hc_mult, H]`` x.dtype outgoing residual stream.
    """
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    hc_mult = post.shape[-1]
    n = 1
    for s in lead:
        n *= s

    # contiguous() first so the reshape is a pure view; residual may arrive as a
    # stride-0 expand (layer 0) which reshape would otherwise reject / copy.
    x_f = x.contiguous().reshape(n, H)
    res_f = residual.contiguous().reshape(n, hc_mult, H)
    post_f = post.contiguous().reshape(n, hc_mult).float()
    comb_f = comb.contiguous().reshape(n, hc_mult * hc_mult).float()

    out = torch.empty((n, hc_mult, H), device=x.device, dtype=x.dtype)
    if n == 0:
        return out.reshape(*lead, hc_mult, H)

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
    return out.reshape(*lead, hc_mult, H)


@deepseek_v4_hc_post.register_fake
def _deepseek_v4_hc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    hc_mult = post.shape[-1]
    return x.new_empty((*lead, hc_mult, H))


# ---------------------------------------------------------------------------
# Fused layer-boundary HC seam: hc_post + the NEXT site's split-D partials
# ---------------------------------------------------------------------------
#
# Every ``_hc_post`` output is immediately re-read by the *next* HC-pre's
# ``_hc_fn_partials_kernel`` (attn-post -> ffn-pre, ffn-post -> next layer's
# attn-pre, last ffn-post -> ``_hc_head``). At decode that is two back-to-back
# launches over the same ``[N, HM * H]`` tensor. This op merges them: the fused
# kernel adopts the partials kernel's ``(N, SPLIT)`` grid / CHUNK layout /
# ``num_warps`` and, per D-chunk, composes the hc_post output in registers
# (identical per-element math and ``m``-axis reduce as
# ``_hc_post_compose_kernel``), stores the bf16 residual, then computes the
# next site's square-sum + hc_fn dot partials from the *rounded* value —
# exactly what the standalone partials kernel would have re-loaded from HBM.
# Both outputs match the two-kernel sequence to ~1-2 fp32 ULP (ptxas contracts
# the mul+add chains into FMAs differently across kernel bodies; measured 1 /
# 16384 output elements and ~45 / 3200 partials at the model shape, warp-count
# invariant) — the same numerics contract as the landed
# ``deepseek_v4_hc_pre_mix`` front. One launch and the ``[N, HM * H]`` HBM
# re-read are removed per seam.
#
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
    LAUNCH_PDL: tl.constexpr,  # PDL: prologue overlaps the producer (AR) tail
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

    # --- x-independent prologue (overlaps the producer AR under PDL) ---
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
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + row * H + h, mask=cmask, other=0.0).to(tl.float32)
    if LAUNCH_PDL:
        tl.extra.cuda.gdc_launch_dependents()
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
    same contract as the landed ``deepseek_v4_hc_pre_mix``).

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
    n = 1
    for s in lead:
        n *= s
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
        LAUNCH_PDL=_AD_HC_PDL,
        num_warps=4,
        launch_pdl=_AD_HC_PDL,
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
    n = 1
    for s in lead:
        n *= s
    _, split = hc_partials_layout(next_hc_fn.shape[-1])
    out = x.new_empty((*lead, hc_mult, H))
    partials = x.new_empty((n, mix_hc + 1, split), dtype=torch.float32)
    return out, partials


# ---------------------------------------------------------------------------
# Fully fused HC seam pair: hc_post + next partials + next site's HC-pre
# ---------------------------------------------------------------------------
#
# At decode, every ``deepseek_v4_hc_post_next_partials`` launch is immediately
# followed by the ``deepseek_v4_hc_pre_mix_combine_partials[_y32]`` launch that
# consumes its partials AND its composed residual stream (measured gap a few
# hundred ns; both are launch-floor-bound). The pair op below runs both in ONE
# kernel: phase 1 is the seam producer verbatim on its ``(n, SPLIT)`` grid
# (keeping the PDL prologue / gdc_wait / gdc_launch_dependents structure, so
# the AR -> seam overlap survives); a device-scope release/acquire spin barrier
# then lets programs 0 and 1 of each row run the two HC-pre phase-2 programs
# verbatim, reading the partials back through L2. The barrier is safe: at most
# ``2n <= 128`` programs spin (fused path requires n <= _HC_PRE_MIX_FUSED_N_MAX)
# while far more chunk programs are co-resident, so arrivals always progress.
#
# num_warps follows the composition kernel (=2). ``out`` is bit-identical to
# the pair; partials/post/comb/y carry the usual ~1-2 fp32 ULP FMA-contraction
# drift across kernel bodies (same contract as this file's seam op vs its
# two-kernel predecessor). The prefill path (n > _HC_PRE_MIX_FUSED_N_MAX) runs
# the two ops' identical prefill branches back to back — bit-identical.
#
# MEASURED VERDICT (B200, n=1, CUDA-graph amortized): the merge LOSES to the
# pair — the launch boundary it removes is worth only ~0.5 us/site while the
# cheapest arrival barrier costs ~1.2 us/site and phase 2 still serializes
# behind phase 1 (pair 7.6 us vs merged 8.3-10.5 us/site). The op and its
# ``fuse_hc_seam_pair`` transform therefore stay DEFAULT OFF; the shipped
# alternative is the PDL dependent-launch of the consumer kernels
# (hc_composition.py), which overlaps their ramp + weight prologue with this
# producer instead (~0.9 us/site measured).

_HC_SEAM_PAIR_NUM_WARPS = _HC_PRE_MIX_COMBINE_NUM_WARPS
_HC_SEAM_PAIR_MAXNREG = _HC_PRE_MIX_COMBINE_MAXNREG

# Persistent per-device barrier state [2, MAX_N] int32 ([0]=arrivals, [1]=done).
# The kernel self-resets its row to zero before exiting; successive seam calls
# are stream/graph-ordered, so one shared buffer serves eager + graph replay.
_hc_seam_barrier_buffers: Dict[int, torch.Tensor] = {}


def _hc_seam_barrier_state(device: torch.device) -> torch.Tensor:
    key = device.index if device.index is not None else torch.cuda.current_device()
    buf = _hc_seam_barrier_buffers.get(key)
    if buf is None:
        # First decode-path call allocates (warmup runs before any CUDA-graph
        # capture, so the buffer never lives in a capture memory pool).
        buf = torch.zeros(2, _HC_PRE_MIX_FUSED_N_MAX, device=device, dtype=torch.int32)
        _hc_seam_barrier_buffers[key] = buf
    return buf


@triton.jit
def _hc_seam_pair_kernel(
    x_ptr,  # [N, H] x.dtype (e.g. bf16)
    residual_ptr,  # [N, HM, H] x.dtype
    post_ptr,  # [N, HM] fp32
    comb_ptr,  # [N, HM*HM] fp32; comb[n, m, o] at n*(HM*HM) + m*HM + o
    fn_ptr,  # [MIX_HC, D] fp32 (the NEXT HC site's hc_fn), D == HM * H
    scale_ptr,  # [3] fp32 (next site's hc_scale)
    base_ptr,  # [MIX_HC] fp32 (next site's hc_base)
    weight_ptr,  # [H] fp32 (next site's RMSNorm weight)
    out_ptr,  # [N, HM, H] out (x.dtype)
    part_ptr,  # [N, MIX_HC + 1, SPLIT] fp32 (cross-program handoff via L2)
    y_ptr,  # [N, H] out_dtype (out)
    y32_ptr,  # [N, H] fp32 (out; written only when EMIT_Y32)
    post_next_ptr,  # [N, HM] fp32 (out)
    comb_next_ptr,  # [N, HM*HM] fp32 (out)
    counter_ptr,  # [>=N] int32, zero on entry (self-reset arrival counter)
    done_ptr,  # [>=N] int32, zero on entry (self-reset completion counter)
    N,
    H,
    D,
    SPLIT,
    norm_eps,
    eps,
    rms_eps,
    HM: tl.constexpr,  # hc_mult
    BM: tl.constexpr,  # next_power_of_2(hc_mult)
    MIX_HC: tl.constexpr,  # rows of the next site's hc_fn
    KBLOCK: tl.constexpr,  # next_power_of_2(MIX_HC)
    CHUNK: tl.constexpr,  # elements per split slot (power of 2)
    SBLOCK: tl.constexpr,  # next_power_of_2(SPLIT)
    BLOCK_H: tl.constexpr,  # next_power_of_2(H)
    SINKHORN_ITERS: tl.constexpr,
    EMIT_Y32: tl.constexpr,
    LAUNCH_PDL: tl.constexpr,  # PDL: phase-1 prologue overlaps the producer (AR) tail
):
    """Grid ``(N, max(SPLIT, 2))``: seam phase 1 in every program, HC-pre phase 2
    in programs 0/1 of each row after an in-kernel arrival barrier.

    Phase 1 mirrors ``_hc_post_next_partials_kernel`` (same PDL points); phase 2
    mirrors the two programs of ``_hc_pre_composition_combine_kernel``. Programs
    with ``s >= 2`` exit at the barrier; programs 0/1 spin on the row's arrival
    counter (release/acquire, gpu scope) so all ``out``/partials stores are
    visible, then the last finisher re-zeros the row's barrier state.
    """
    row = tl.program_id(0)
    s = tl.program_id(1)
    if row >= N:
        return
    pbase = part_ptr + row * ((MIX_HC + 1) * SPLIT)

    # --- phase 1: the seam producer, verbatim (grid programs beyond SPLIT skip) ---
    if s < SPLIT:
        offs = s * CHUNK + tl.arange(0, CHUNK)
        cmask = offs < D
        o = offs // H  # output stream per element
        h = offs - o * H  # hidden position per element

        # x-independent prologue (overlaps the producer AR under PDL)
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

        # hc_post compose: only x depends on the producer
        if LAUNCH_PDL:
            tl.extra.cuda.gdc_wait()
        x = tl.load(x_ptr + row * H + h, mask=cmask, other=0.0).to(tl.float32)
        if LAUNCH_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        acc = p * x + mix
        tl.store(out_ptr + row * D + offs, acc.to(out_ptr.dtype.element_ty), mask=cmask)

        # next site's partials from the rounded y (mirrors _hc_fn_partials_kernel)
        xf = acc.to(out_ptr.dtype.element_ty).to(tl.float32)
        sq = tl.sum(xf * xf, axis=0)
        part = tl.sum(w * xf[None, :], axis=1)
        tl.store(pbase + k * SPLIT + s, part, mask=kmask)
        tl.store(pbase + MIX_HC * SPLIT + s, sq)

    # --- arrival barrier: chunk programs exit, programs 0/1 spin ---
    tl.atomic_add(counter_ptr + row, 1, sem="release")
    if s >= 2:
        return
    target = tl.num_programs(1)
    cnt = tl.atomic_add(counter_ptr + row, 0, sem="acquire")
    while cnt < target:
        cnt = tl.atomic_add(counter_ptr + row, 0, sem="acquire")

    # --- phase 2: the HC-pre composition/combine programs, verbatim ---
    sb = tl.arange(0, SBLOCK)
    smask = sb < SPLIT

    # rstd re-derived by both programs from the same partials row (identical
    # tile shape + op order -> identical bits); see the composition kernel.
    sqp = tl.load(pbase + MIX_HC * SPLIT + sb, mask=smask, other=0.0)
    rstd = tl.rsqrt(tl.sum(sqp, axis=0) / D + norm_eps)

    d = tl.arange(0, BM)
    dmask = d < HM

    if s == 0:
        # --- combine program: pre gate -> weighted combine -> RMSNorm -> y ---
        s0 = tl.load(scale_ptr + 0)

        pre_part = tl.load(
            pbase + d[:, None] * SPLIT + sb[None, :],
            mask=dmask[:, None] & smask[None, :],
            other=0.0,
        )
        pre_logits = (tl.sum(pre_part, axis=1) * rstd) * s0 + tl.load(
            base_ptr + d, mask=dmask, other=0.0
        )
        pre = tl.sigmoid(pre_logits) + eps

        # weighted combine + RMSNorm over the out tensor phase 1 just stored
        hh = tl.arange(0, BLOCK_H)
        hmask = hh < H
        flat_row = out_ptr + row * D
        acc2 = tl.zeros([BLOCK_H], dtype=tl.float32)
        for mm in tl.static_range(HM):
            # Exact scalar extraction of pre[mm] from the register tile.
            pm = tl.sum(tl.where(d == mm, pre, 0.0), axis=0)
            f = tl.load(flat_row + mm * H + hh, mask=hmask, other=0.0).to(tl.float32)
            acc2 += pm * f
        yv = acc2.to(tl.bfloat16).to(tl.float32)
        var = tl.sum(yv * yv, axis=0) / H
        y_rstd = tl.rsqrt(var + rms_eps)
        normed = (yv * y_rstd).to(tl.bfloat16).to(tl.float32)
        wn = tl.load(weight_ptr + hh, mask=hmask, other=0.0)
        outv = wn * normed
        out_c = outv.to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + row * H + hh, out_c, mask=hmask)
        if EMIT_Y32:
            # fp32 mirror of the *stored* value (exact widening of the rounded y).
            tl.store(y32_ptr + row * H + hh, out_c.to(tl.float32), mask=hmask)
    else:
        # --- composition program: post gate + comb logits + sinkhorn ---
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)

        post_part = tl.load(
            pbase + (HM + d)[:, None] * SPLIT + sb[None, :],
            mask=dmask[:, None] & smask[None, :],
            other=0.0,
        )
        post_logits = (tl.sum(post_part, axis=1) * rstd) * s1 + tl.load(
            base_ptr + HM + d, mask=dmask, other=0.0
        )
        post_v = 2.0 * tl.sigmoid(post_logits)
        tl.store(post_next_ptr + row * HM + d, post_v, mask=dmask)

        i = tl.arange(0, BM)[:, None]
        j = tl.arange(0, BM)[None, :]
        m2 = (i < HM) & (j < HM)
        cflat = i * HM + j
        comb_part = tl.load(
            pbase + (2 * HM + cflat)[:, :, None] * SPLIT + sb[None, None, :],
            mask=m2[:, :, None] & smask[None, None, :],
            other=0.0,
        )
        comb_logits = (tl.sum(comb_part, axis=2) * rstd) * s2 + tl.load(
            base_ptr + 2 * HM + cflat, mask=m2, other=0.0
        )

        neg_inf = float("-inf")
        logits_sm = tl.where(m2, comb_logits, neg_inf)
        mx = tl.max(logits_sm, axis=1)[:, None]
        e = tl.exp(logits_sm - mx)
        sden = tl.sum(e, axis=1)[:, None]
        comb_v = tl.where(m2, e / sden + eps, 0.0)

        cs = tl.sum(comb_v, axis=0)[None, :]
        comb_v = tl.where(m2, comb_v / (cs + eps), 0.0)

        for _ in range(SINKHORN_ITERS - 1):
            rs = tl.sum(comb_v, axis=1)[:, None]
            comb_v = tl.where(m2, comb_v / (rs + eps), 0.0)
            cs = tl.sum(comb_v, axis=0)[None, :]
            comb_v = tl.where(m2, comb_v / (cs + eps), 0.0)

        tl.store(comb_next_ptr + row * (HM * HM) + cflat, comb_v, mask=m2)

    # --- self-reset: the last of the two finishers re-zeros the row's state ---
    old = tl.atomic_add(done_ptr + row, 1, sem="acq_rel")
    if old == 1:
        tl.atomic_xchg(counter_ptr + row, 0)
        tl.atomic_xchg(done_ptr + row, 0)


def _hc_post_pre_combine_impl(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
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
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Shared body of the fused seam-pair ops. Returns ``(out, y, y32, post, comb)``
    with ``y32`` ``None`` unless ``emit_y32``."""
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    hidden = norm_weight.shape[0]
    n = 1
    for sz in lead:
        n *= sz
    D = hc_mult * H
    mix_hc = next_hc_fn.shape[0]
    assert next_hc_fn.shape[-1] == D, "next_hc_fn last dim must equal hc_mult * H"
    assert hidden == H, "norm_weight must match x's hidden size"

    # contiguous() first so the reshape is a pure view; residual may arrive as a
    # stride-0 expand (layer 0) which reshape would otherwise reject / copy.
    x_f = x.contiguous().reshape(n, H)
    res_f = residual.contiguous().reshape(n, hc_mult, H)
    post_f = post.contiguous().reshape(n, hc_mult).float()
    comb_f = comb.contiguous().reshape(n, hc_mult * hc_mult).float()

    out = torch.empty((n, hc_mult, H), device=x.device, dtype=x.dtype)
    if n == 0:
        y = torch.empty(n, hidden, device=x.device, dtype=out_dtype)
        y32 = torch.empty(n, hidden, device=x.device, dtype=torch.float32) if emit_y32 else None
        post_n = torch.empty(n, hc_mult, device=x.device, dtype=torch.float32)
        comb_n = torch.empty(n, hc_mult * hc_mult, device=x.device, dtype=torch.float32)
        return (
            out.reshape(*lead, hc_mult, H),
            y.reshape(*lead, hidden),
            y32.reshape(*lead, hidden) if emit_y32 else None,
            post_n.reshape(*lead, hc_mult),
            comb_n.reshape(*lead, hc_mult, hc_mult),
        )

    if n > _HC_PRE_MIX_FUSED_N_MAX:
        # Prefill: the two ops' identical prefill branches back to back
        # (compose kernel + eager cublas front) — bit-identical by construction.
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
        flat = out.reshape(n, D)
        flat_f = flat.float()
        rsqrt = torch.rsqrt(flat_f.square().mean(-1, keepdim=True) + norm_eps)
        mixes = torch.nn.functional.linear(flat_f, next_hc_fn) * rsqrt
        pre, post_n, comb_n = torch.ops.auto_deploy.hc_split_sinkhorn(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        y = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
            pre, flat, norm_weight, rms_eps, hc_mult, out_dtype
        )
        y32 = None
        if emit_y32:
            y32 = y.float() if y.dtype != torch.float32 else y.clone()
        return (
            out.reshape(*lead, hc_mult, H),
            y.reshape(*lead, hidden),
            y32.reshape(*lead, hidden) if emit_y32 else None,
            post_n.reshape(*lead, hc_mult),
            comb_n.reshape(*lead, hc_mult, hc_mult),
        )

    fn_f = next_hc_fn.contiguous().float()
    scale_f = hc_scale.contiguous().float()
    base_f = hc_base.contiguous().float()
    weight_f = norm_weight.contiguous().float()

    chunk, split = hc_partials_layout(D)
    partials = torch.empty(n, mix_hc + 1, split, device=x.device, dtype=torch.float32)
    y = torch.empty(n, hidden, device=x.device, dtype=out_dtype)
    y32 = torch.empty(n, hidden, device=x.device, dtype=torch.float32) if emit_y32 else None
    post_n = torch.empty(n, hc_mult, device=x.device, dtype=torch.float32)
    comb_n = torch.empty(n, hc_mult * hc_mult, device=x.device, dtype=torch.float32)

    barrier = _hc_seam_barrier_state(x.device)
    # Grid always has the two phase-2 programs even when SPLIT == 1 (tiny D).
    grid_s = max(split, 2)
    _hc_seam_pair_kernel[(n, grid_s)](
        x_f,
        res_f,
        post_f,
        comb_f,
        fn_f,
        scale_f,
        base_f,
        weight_f,
        out,
        partials,
        y,
        y32 if emit_y32 else y,  # dummy y32 ptr, dead code at EMIT_Y32=False
        post_n,
        comb_n,
        barrier[0],
        barrier[1],
        n,
        H,
        D,
        split,
        norm_eps,
        eps,
        rms_eps,
        HM=hc_mult,
        BM=triton.next_power_of_2(hc_mult),
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        SBLOCK=triton.next_power_of_2(split),
        BLOCK_H=triton.next_power_of_2(hidden),
        SINKHORN_ITERS=sinkhorn_iters,
        EMIT_Y32=emit_y32,
        LAUNCH_PDL=_AD_HC_PDL,
        num_warps=_HC_SEAM_PAIR_NUM_WARPS,
        num_stages=2,
        maxnreg=_HC_SEAM_PAIR_MAXNREG,
        launch_pdl=_AD_HC_PDL,
    )
    return (
        out.reshape(*lead, hc_mult, H),
        y.reshape(*lead, hidden),
        y32.reshape(*lead, hidden) if emit_y32 else None,
        post_n.reshape(*lead, hc_mult),
        comb_n.reshape(*lead, hc_mult, hc_mult),
    )


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_post_pre_combine", mutates_args=())
def deepseek_v4_hc_post_pre_combine(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
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
    """Fused HC seam pair. Drop-in for ``deepseek_v4_hc_post_next_partials``
    followed by ``deepseek_v4_hc_pre_mix_combine_partials`` on its outputs.

    Args mirror the pair: ``x``/``residual``/``post``/``comb`` are the seam
    producer's, ``next_hc_fn``/``hc_scale``/``hc_base``/``norm_weight`` and the
    scalars are the consuming HC site's. On the decode path ``out`` is
    bit-identical to the pair; ``y``/``post``/``comb`` carry ~1-2 fp32 ULP of
    FMA-contraction drift across kernel bodies (the file's standing seam
    contract; measured y flips 2e-5 of elements). Prefill is bit-identical.

    Returns:
        out:  [..., hc_mult, H] x.dtype outgoing residual stream.
        y:    [..., H] out_dtype next block input (combine + RMSNorm).
        post: [..., hc_mult] fp32 next site's post gate.
        comb: [..., hc_mult, hc_mult] fp32 next site's combine matrix.
    """
    out, y, _, post_n, comb_n = _hc_post_pre_combine_impl(
        x,
        residual,
        post,
        comb,
        next_hc_fn,
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
    return out, y, post_n, comb_n


@deepseek_v4_hc_post_pre_combine.register_fake
def _deepseek_v4_hc_post_pre_combine_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
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
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    out = x.new_empty((*lead, hc_mult, H))
    y = x.new_empty((*lead, H), dtype=out_dtype)
    post_n = x.new_empty((*lead, hc_mult), dtype=torch.float32)
    comb_n = x.new_empty((*lead, hc_mult, hc_mult), dtype=torch.float32)
    return out, y, post_n, comb_n


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_post_pre_combine_y32", mutates_args=())
def deepseek_v4_hc_post_pre_combine_y32(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
    rms_eps: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``deepseek_v4_hc_post_pre_combine`` that also emits ``y32 == y.float()``
    (drop-in for the ``_y32`` consumer at learned-router MoE sites).

    Returns:
        out, y, y32, post, comb — as the base op plus the exact fp32 widening
        of the just-rounded ``y``.
    """
    out, y, y32, post_n, comb_n = _hc_post_pre_combine_impl(
        x,
        residual,
        post,
        comb,
        next_hc_fn,
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
    return out, y, y32, post_n, comb_n


@deepseek_v4_hc_post_pre_combine_y32.register_fake
def _deepseek_v4_hc_post_pre_combine_y32_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    next_hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
    rms_eps: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    out = x.new_empty((*lead, hc_mult, H))
    y = x.new_empty((*lead, H), dtype=out_dtype)
    y32 = x.new_empty((*lead, H), dtype=torch.float32)
    post_n = x.new_empty((*lead, hc_mult), dtype=torch.float32)
    comb_n = x.new_empty((*lead, hc_mult, hc_mult), dtype=torch.float32)
    return out, y, y32, post_n, comb_n
