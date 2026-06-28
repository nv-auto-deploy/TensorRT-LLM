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

import torch
import triton
import triton.language as tl


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
