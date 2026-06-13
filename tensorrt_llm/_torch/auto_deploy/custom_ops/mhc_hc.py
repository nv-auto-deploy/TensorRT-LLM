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

"""Functional AutoDeploy ops wrapping the fused mHC hyper-connection kernels.

Replaces the eager hyper-connection swarm in the DeepSeek-V4 custom modeling.
The trtllm mhc kernels are in-place/void; these ops allocate outputs, run the
pre-mapping (gemm+sqrsum -> big_fuse: RMS+sigmoid+Sinkhorn-in-warp+pre-mix, one
kernel) and the post-mapping, and return clean functional tensors so AD's FX
export sees proper op outputs. register_fake gives FakeTensor-shaped meta outputs.
big_fuse runs the full Sinkhorn (sinkhorn_iters) in-kernel via warp-shuffle, so it
eliminates the ~80-kernel/sublayer eager Sinkhorn swarm (the conc1 ITL lever).

Pre-mapping (per token, residual x = [M, n, hidden]):
  mixes[M, mix_hc], sqrsum[M]              = gemm_sqrsum_fma(x_flat, fn)  # fn=[mix_hc, n*hidden]
  post[M,n], comb[M,n,n], layer_input[M,h] = big_fuse(mixes, sqrsum, x, scale, base, ...)
Post-mapping:
  out[M, n, hidden] = post * x + comb.T @ residual
"""

from typing import Tuple

import torch


def _split_sinkhorn(
    mixes_scaled: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """post[..., n] and comb[..., n, n] from rstd-scaled GEMM mixes.

    Mirrors the eager _hc_split_sinkhorn; only post/comb are returned since the
    pre weighted-sum (layer_input) is produced by the head_apply kernel.
    """
    post_logits = mixes_scaled[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    comb_logits = mixes_scaled[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]

    post = 2.0 * torch.sigmoid(post_logits)
    comb = comb_logits.view(*comb_logits.shape[:-1], hc_mult, hc_mult)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return post, comb


@torch.library.custom_op("auto_deploy::mhc_hc_pre", mutates_args=())
def mhc_hc_pre(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mHC pre-mapping (non-autotuned FMA path).

    Args:
        x:    [..., n, hidden] residual streams (bf16 expected).
        fn:   [mix_hc, n*hidden] fp32 mix weight, mix_hc = (2+n)*n.
        scale:[3] fp32 (pre/post/comb logit scales).
        base: [mix_hc] fp32.
        hc_mult: n (number of hyper-connection streams).
        sinkhorn_iters / norm_eps / eps: hc hyperparameters.

    Returns:
        post:        [..., n] fp32
        comb:        [..., n, n] fp32
        layer_input: [..., hidden] bf16 (weighted sum over the n streams)
    """
    n = hc_mult
    outer = x.shape[:-2]
    hidden = x.shape[-1]
    hc_dim = n * hidden
    mix_hc = (2 + n) * n

    x3 = x.reshape(-1, n, hidden).contiguous()
    M = x3.shape[0]
    x_bf16 = x3.to(torch.bfloat16) if x3.dtype != torch.bfloat16 else x3
    x_flat = x_bf16.reshape(M, hc_dim)

    fn_f = fn.to(torch.float32).contiguous()
    scale_f = scale.to(torch.float32).contiguous()
    base_f = base.to(torch.float32).contiguous()

    # gemm + fused sqrsum: mixes = x_flat @ fn.T, sqrsum = sum_k x_flat^2
    mixes = torch.empty((M, mix_hc), dtype=torch.float32, device=x.device)
    sqrsum = torch.empty((M,), dtype=torch.float32, device=x.device)
    torch.ops.trtllm.mhc_gemm_sqrsum_fma(x_flat, fn_f, mixes, sqrsum, M, mix_hc, hc_dim, 0, 0)

    # big_fuse: RMS-norm + sigmoid gates + Sinkhorn (in-warp __shfl) + pre-mix in ONE kernel.
    # Replaces the head_apply + eager Sinkhorn loop -> kills the ~80-kernel/sublayer swarm.
    post = torch.empty((M, n), dtype=torch.float32, device=x.device)
    comb = torch.empty((M, n * n), dtype=torch.float32, device=x.device)
    layer_input = torch.empty((M, hidden), dtype=torch.bfloat16, device=x.device)
    torch.ops.trtllm.mhc_big_fuse(
        mixes.contiguous(),        # y_acc  [M, mix_hc]
        sqrsum.contiguous(),       # r_acc  [M]
        x_bf16.contiguous(),       # residual [M, n, hidden]
        scale_f,                   # hc_scale [3]
        base_f,                    # hc_base  [mix_hc]
        post,                      # out post_mix [M, n]
        comb,                      # out comb_mix [M, n*n]
        layer_input,               # out layer_input [M, hidden]
        M,
        hc_dim,                    # K
        hidden,
        float(norm_eps),           # rms_eps
        float(eps),                # hc_pre_eps
        float(eps),                # hc_sinkhorn_eps
        2.0,                       # hc_post_mult_value (eager: post = 2.0 * sigmoid)
        int(sinkhorn_iters),       # sinkhorn_repeat
        1,                         # num_splits (FMA gemm, no split-K)
        0,                         # block_size (0 = auto / fallback tactic)
    )

    post = post.view(*outer, n)
    comb = comb.view(*outer, n, n)
    layer_input = layer_input.view(*outer, hidden)
    return post, comb, layer_input


@mhc_hc_pre.register_fake
def _mhc_hc_pre_fake(
    x: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    norm_eps: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = hc_mult
    outer = x.shape[:-2]
    hidden = x.shape[-1]
    post = x.new_empty((*outer, n), dtype=torch.float32)
    comb = x.new_empty((*outer, n, n), dtype=torch.float32)
    layer_input = x.new_empty((*outer, hidden), dtype=torch.bfloat16)
    return post, comb, layer_input


@torch.library.custom_op("auto_deploy::mhc_hc_post", mutates_args=())
def mhc_hc_post(
    residual: torch.Tensor,
    x: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Fused mHC post-mapping: out = post * x + comb.T @ residual.

    Args:
        residual: [..., n, hidden] bf16
        x:        [..., hidden]    bf16 (sublayer output)
        post:     [..., n]         fp32
        comb:     [..., n, n]      fp32
    Returns:
        out:      [..., n, hidden] (dtype of x)
    """
    n = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]

    residual3 = residual.reshape(-1, n, hidden).to(torch.bfloat16).contiguous()
    B = residual3.shape[0]
    x2 = x.reshape(B, hidden).to(torch.bfloat16).contiguous()
    post2 = post.reshape(B, n).to(torch.float32).contiguous()
    comb3 = comb.reshape(B, n, n).to(torch.float32).contiguous()

    out = torch.empty((B, n, hidden), dtype=torch.bfloat16, device=x.device)
    torch.ops.trtllm.mhc_post_mapping(residual3, x2, post2, comb3, out, B, hidden)
    return out.view(*outer, n, hidden).to(x.dtype)


@mhc_hc_post.register_fake
def _mhc_hc_post_fake(
    residual: torch.Tensor,
    x: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    n = residual.shape[-2]
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    return x.new_empty((*outer, n, hidden), dtype=x.dtype)
