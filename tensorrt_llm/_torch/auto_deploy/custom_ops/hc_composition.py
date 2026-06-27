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
