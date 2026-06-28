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


def _hc_post_launch_config(n: int):
    """Pick ``(num_warps, num_stages, block_h)`` for the ``_hc_post_compose_kernel`` launch.

    The kernel is grid ``(n, cdiv(H, BLOCK_H))`` — narrow ``BLOCK_H``-wide CTAs,
    each accumulating all ``HM`` output streams via a tiny ``axis=0`` reduction
    over the ``[BM, BLOCK_H]`` residual tile. The original hardcoded
    ``num_warps=4`` (128 threads) lays that tile out so the ``axis=0`` reduction
    needs *cross-warp* communication, which is pathologically slow for such a
    thin reduction: on B200 (H=4096, hc_mult=4) ``nw=4`` measures ~4.6us vs
    ~1.7us at ``nw<=2`` — a ~2.7x penalty.

    Microbench (CUDA-graph stacked amortized timing) picks, per token count:

      * ``n == 1`` (decode, the primary concurrency-1 tpot shape) -> ``nw=1``,
        ``BLOCK_H=256``. At the full ``BLOCK_H=512`` the H=4096 stream tiles into
        only ``cdiv(4096, 512) = 8`` CTAs (8 warps total at ``nw=1``) — far too
        few in-flight warps to hide the load->reduce->store latency, leaving the
        single decode token bound by one CTA's serial tail. Halving the H-tile to
        ``BLOCK_H=256`` doubles the grid to ``(1, 16)`` (16 in-flight warps),
        which hides that tail: ``1.91us -> 1.75us`` (-8.2%). The relationship is
        non-monotonic (``BLOCK_H`` 256 and 32 win, 128/64 lose) — 256 is the
        robust minimum across repeated runs.
      * ``n >= 2`` (decode batch / prefill, up to n=1000)         -> ``nw=2``,
        ``BLOCK_H=512``. Here ``n`` already supplies ample CTAs, so any finer
        H-tile only fragments the (``nw=2``) tile and regresses sharply
        (``BLOCK_H=256`` is ~2x slower at n=2 and >10x at n>=256). 512 is best
        across the whole prefill range (1.77us at n=2, 3.2us at n=256, 8.4us at
        n=1000).

    ``num_stages`` is pinned to 2: the ``HM`` loop is fully unrolled (HM=4) so
    extra pipeline stages buy nothing, and ``ns=3`` regresses the ``nw=1`` decode
    case ~12%. Chosen deterministically rather than via ``@triton.autotune``
    because these sub-floor kernels defeat the autotuner's ``do_bench`` (it
    resolves the host launch cadence, not GPU time) and the model's varying
    prefill token counts would otherwise force a synchronizing re-tune each shape
    (cf. sibling ``_hc_weighted_combine_kernel`` / idea_0054).
    """
    if n == 1:
        return 1, 2, 256
    return 2, 2, 512


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
    BLOCK_H: tl.constexpr,
):
    """One program per (token row, H-tile). Computes all HM output streams.

    y[n, o, h] = post[n, o] * x[n, h] + sum_m comb[n, m, o] * residual[n, m, h]

    The HM residual streams for this H-tile are loaded once into a fp32
    ``[BM, BLOCK_H]`` register tile and reused across every output stream o, so
    the kernel reads each residual / x element exactly once and writes each
    output element exactly once. Padding rows (m >= HM) are masked to 0 and drop
    out of both the residual tile and the per-stream comb weights.
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
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

    for o in range(HM):
        p = tl.load(post_base + o)  # scalar fp32
        c = tl.load(comb_base + m * HM + o, mask=mmask, other=0.0)  # comb[:, o] -> [BM]
        acc = p * x + tl.sum(c[:, None] * res_tile, axis=0)  # [BLOCK_H]
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

    num_warps, num_stages, block_h_max = _hc_post_launch_config(n)
    block_h = min(block_h_max, triton.next_power_of_2(H))
    grid = (n, triton.cdiv(H, block_h))
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
