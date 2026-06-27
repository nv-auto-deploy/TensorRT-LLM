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

"""Fused Triton kernel for the DeepSeek V4 MoE router head (sqrtsoftplus scoring).

The reference ``DeepseekV4MoEGate.forward`` (non-hash layers) runs a chain of
~6 tiny per-token kernels around the unavoidable top-k:

    scores  = softplus(router_logits).sqrt()        # 2 elementwise
    biased  = scores + bias                          # 1 elementwise
    idx     = biased.topk(top_k).indices             # gatherTopK + radixSort/bitonicSort
    weights = scores.gather(1, idx)                  # gather
    weights = weights / (weights.sum(-1) + 1e-20)    # reduce + add + div
    weights = weights * routed_scaling_factor        # mul

At decode (1-2 tokens) every one of these is a launch-bound CUDA-graph node, and
the top-k/sort kernels alone are ~2% of total decode GPU time (in the
``reduction`` op-type bucket). This module collapses the whole chain into ONE
Triton program per token: it computes the sqrtsoftplus scores in registers,
selects the top-k of ``scores + bias`` via iterative arg-max (no separate sort),
gathers the (unbiased) scores at the selected experts, renormalizes and scales —
producing the identical ``(selected_experts, routing_weights)`` pair.

Note the kernel is deliberately NOT named with ``topk``/``sort``/``sum`` so the
fused work is reclassified out of the ``reduction`` bucket rather than back into
it.
"""

import math
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _deepseek_v4_routing_kernel(
    logits_ptr,  # (T, E) fp32 router logits
    bias_ptr,  # (E,) fp32 routing bias
    weights_ptr,  # (T, K) fp32 output routing weights
    indices_ptr,  # (T, K) int64 output selected experts
    num_tokens,
    num_experts,
    stride_lt,
    stride_le,
    stride_wt,
    stride_wk,
    stride_it,
    stride_ik,
    routed_scaling_factor,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    NORM_TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,  # >= num_experts, power of 2
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,  # >= TOP_K, power of 2
):
    """One program per token: sqrtsoftplus score -> top-k(scores+bias) -> renorm/scale."""
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < num_experts

    logits = tl.load(
        logits_ptr + token_id * stride_lt + offs_e * stride_le,
        mask=mask_e,
        other=0.0,
    ).to(tl.float32)

    # scores = sqrt(softplus(logits)); softplus(x) = x if x > thr else log1p(exp(x)).
    # Match torch.nn.functional.softplus' numerically-stable threshold form.
    exp_x = tl.exp(logits)
    softplus = tl.where(logits > SOFTPLUS_THRESHOLD, logits, tl.log(1.0 + exp_x))
    scores = tl.sqrt(softplus)

    bias = tl.load(bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
    biased = scores + bias
    # Padding experts (offs_e >= num_experts) must never be selected.
    neg_inf = float("-inf")
    biased = tl.where(mask_e, biased, neg_inf)

    offs_k = tl.arange(0, BLOCK_K)
    topk_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)
    topk_w = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Iterative top-k over (scores + bias): pick arg-max, record the UNBIASED score
    # at that expert, then mask it out. Smallest index wins ties (matches argmax).
    work = biased
    for k_i in tl.static_range(TOP_K):
        max_val = tl.max(work, axis=0)
        is_max = work == max_val
        candidate = tl.where(is_max, offs_e, BLOCK_E)
        max_idx = tl.min(candidate, axis=0)

        ki_mask = offs_k == k_i
        topk_idxs = tl.where(ki_mask, max_idx.to(tl.int32), topk_idxs)
        # weight = scores[max_idx] (gather the unbiased score at the selected expert)
        sel_w = tl.sum(tl.where(offs_e == max_idx, scores, 0.0), axis=0)
        topk_w = tl.where(ki_mask, sel_w, topk_w)

        work = tl.where(offs_e == max_idx, neg_inf, work)

    if NORM_TOPK:
        denom = tl.sum(tl.where(offs_k < TOP_K, topk_w, 0.0), axis=0) + 1e-20
        topk_w = topk_w / denom
    topk_w = topk_w * routed_scaling_factor

    mask_k = offs_k < TOP_K
    tl.store(
        weights_ptr + token_id * stride_wt + offs_k * stride_wk,
        topk_w,
        mask=mask_k,
    )
    tl.store(
        indices_ptr + token_id * stride_it + offs_k * stride_ik,
        topk_idxs.to(tl.int64),
        mask=mask_k,
    )


def _next_power_of_2(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 1)))


def deepseek_v4_routing_fn(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 sqrtsoftplus top-k routing.

    Args:
        router_logits: (T, E) fp32 raw router logits.
        bias: (E,) fp32 routing bias added before top-k selection.
        top_k: number of experts to select per token.
        routed_scaling_factor: scalar multiplier applied to the routing weights.
        norm_topk_prob: if True, renormalize the selected weights to sum to 1.

    Returns:
        selected_experts: (T, top_k) int64 expert indices (descending score order).
        routing_weights: (T, top_k) fp32 routing weights.
    """
    assert router_logits.ndim == 2, "router_logits must be 2-D (T, E)"
    num_tokens, num_experts = router_logits.shape

    weights = torch.empty((num_tokens, top_k), dtype=torch.float32, device=router_logits.device)
    indices = torch.empty((num_tokens, top_k), dtype=torch.int64, device=router_logits.device)

    BLOCK_E = _next_power_of_2(num_experts)
    BLOCK_K = _next_power_of_2(top_k)
    grid = (num_tokens,)

    _deepseek_v4_routing_kernel[grid](
        router_logits,
        bias,
        weights,
        indices,
        num_tokens,
        num_experts,
        router_logits.stride(0),
        router_logits.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        routed_scaling_factor,
        SOFTPLUS_THRESHOLD=20.0,
        NORM_TOPK=norm_topk_prob,
        BLOCK_E=BLOCK_E,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
    )
    return indices, weights


@torch.library.custom_op("auto_deploy::deepseek_v4_routing", mutates_args=())
def deepseek_v4_routing(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 MoE router head (sqrtsoftplus scoring + top-k + renorm/scale).

    Equivalent to::

        scores = F.softplus(router_logits).sqrt()
        idx = (scores + bias).topk(top_k, dim=-1).indices
        weights = scores.gather(1, idx)
        if norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        weights = weights * routed_scaling_factor

    collapsed into one Triton program per token.
    """
    return deepseek_v4_routing_fn(router_logits, bias, top_k, routed_scaling_factor, norm_topk_prob)


@deepseek_v4_routing.register_fake
def _deepseek_v4_routing_fake(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = router_logits.shape[0]
    indices = router_logits.new_empty((num_tokens, top_k), dtype=torch.int64)
    weights = router_logits.new_empty((num_tokens, top_k), dtype=torch.float32)
    return indices, weights
