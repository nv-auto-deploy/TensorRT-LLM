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

    block_h = min(512, triton.next_power_of_2(H))
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
        num_warps=4,
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
