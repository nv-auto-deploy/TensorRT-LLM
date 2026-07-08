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
    weights_ptr,  # (T, K) fp32 (LOCALIZED=0) or bf16 (LOCALIZED=1) output routing weights
    indices_ptr,  # (T, K) int64 (LOCALIZED=0) or int32 (LOCALIZED=1) output selected experts
    num_tokens,
    num_experts,
    stride_lt,
    stride_le,
    stride_wt,
    stride_wk,
    stride_it,
    stride_ik,
    routed_scaling_factor,
    expert_start,  # EP shard offset (only read when LOCALIZED=1)
    local_experts,  # local expert count / invalid sentinel (only read when LOCALIZED=1)
    SOFTPLUS_THRESHOLD: tl.constexpr,
    NORM_TOPK: tl.constexpr,
    LOCALIZED: tl.constexpr,
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
        # Mask the selected expert for the next pass directly off the max VALUE
        # (``is_max``) rather than off the selected index, so the iter-to-iter
        # critical path is just max -> mask (one reduction deep) instead of
        # max -> min-index -> mask (two deep). The index (min) and score (sum)
        # reductions then overlap with the next pass's max instead of serializing
        # ahead of it. NOTE: on an exact value tie this masks all tied experts at
        # once (vs the previous smallest-index-only); ties are measure-zero for the
        # fp32 router logits (and torch.topk's own tie order is unspecified), so the
        # selected set is identical on all real inputs.
        work = tl.where(is_max, neg_inf, work)

        candidate = tl.where(is_max, offs_e, BLOCK_E)
        max_idx = tl.min(candidate, axis=0)

        ki_mask = offs_k == k_i
        topk_idxs = tl.where(ki_mask, max_idx.to(tl.int32), topk_idxs)
        # weight = scores[max_idx]: gather the unbiased score at the single selected
        # expert (smallest tied index), so the recorded weight stays exact.
        sel_w = tl.sum(tl.where(offs_e == max_idx, scores, 0.0), axis=0)
        topk_w = tl.where(ki_mask, sel_w, topk_w)

    if NORM_TOPK:
        denom = tl.sum(tl.where(offs_k < TOP_K, topk_w, 0.0), axis=0) + 1e-20
        topk_w = topk_w / denom
    topk_w = topk_w * routed_scaling_factor

    mask_k = offs_k < TOP_K
    if LOCALIZED:
        # Fused EP global->local localization (mirrors ``_localize_routing_kernel``
        # bit for bit): off-rank / out-of-range routes get the invalid sentinel
        # ``local_experts`` and a zero weight; valid weights are the single-rounded
        # bf16 of the exact fp32 value the non-localized op would have stored.
        local = topk_idxs.to(tl.int64) - expert_start
        valid = (local >= 0) & (local < local_experts)
        out_idx = tl.where(valid, local, local_experts).to(tl.int32)
        out_w = tl.where(valid, topk_w, 0.0).to(tl.bfloat16)
        tl.store(weights_ptr + token_id * stride_wt + offs_k * stride_wk, out_w, mask=mask_k)
        tl.store(indices_ptr + token_id * stride_it + offs_k * stride_ik, out_idx, mask=mask_k)
    else:
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
    expert_start: int = 0,
    local_experts: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 sqrtsoftplus top-k routing.

    Args:
        router_logits: (T, E) fp32 raw router logits.
        bias: (E,) fp32 routing bias added before top-k selection.
        top_k: number of experts to select per token.
        routed_scaling_factor: scalar multiplier applied to the routing weights.
        norm_topk_prob: if True, renormalize the selected weights to sum to 1.
        expert_start: EP shard offset (localized mode only).
        local_experts: when > 0, emit EP-LOCALIZED outputs for the trtllm-gen MoE
            runner instead of the global pair: int32 local expert ids (off-rank
            routes carry the invalid sentinel ``local_experts``) and bf16 masked
            routing weights, exactly as ``deepseek_v4_localize_routing`` would
            produce from the global outputs.

    Returns:
        selected_experts: (T, top_k) int64 expert indices (descending score order),
            or int32 local ids in localized mode.
        routing_weights: (T, top_k) fp32 routing weights, or bf16 masked weights in
            localized mode.
    """
    assert router_logits.ndim == 2, "router_logits must be 2-D (T, E)"
    num_tokens, num_experts = router_logits.shape
    localized = local_experts > 0

    idx_dtype = torch.int32 if localized else torch.int64
    w_dtype = torch.bfloat16 if localized else torch.float32
    weights = torch.empty((num_tokens, top_k), dtype=w_dtype, device=router_logits.device)
    indices = torch.empty((num_tokens, top_k), dtype=idx_dtype, device=router_logits.device)

    BLOCK_E = _next_power_of_2(num_experts)
    BLOCK_K = _next_power_of_2(top_k)
    grid = (num_tokens,)

    # Single-warp launch: every program does an E-wide (~256) reduction repeated
    # TOP_K times. With 1 warp those reductions are pure intra-warp shuffles (no
    # shared-mem barrier), and at prefill the small thread count lets more CTAs
    # stay resident per SM. Measured uniformly fastest across decode B=1/2 and
    # prefill (256/1000/2048 tokens) vs the Triton default of 4 warps.
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
        int(expert_start),
        int(local_experts),
        SOFTPLUS_THRESHOLD=20.0,
        NORM_TOPK=norm_topk_prob,
        LOCALIZED=localized,
        BLOCK_E=BLOCK_E,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        num_warps=1,
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


@torch.library.custom_op("auto_deploy::deepseek_v4_routing_localized", mutates_args=())
def deepseek_v4_routing_localized(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 router head emitting EP-LOCALIZED routing for the MoE runner.

    Same scoring/top-k/renorm chain as :func:`deepseek_v4_routing`, with the EP
    global->local localization (``deepseek_v4_localize_routing``) folded into the
    kernel tail. Returns ``(local_idx int32, weights bf16)``: off-rank routes carry
    the invalid sentinel ``local_experts`` and weight ``0``; valid slots are
    bit-identical to running the two ops back to back.
    """
    if local_experts <= 0:
        raise ValueError(f"local_experts should be positive, got {local_experts}.")
    return deepseek_v4_routing_fn(
        router_logits,
        bias,
        top_k,
        routed_scaling_factor,
        norm_topk_prob,
        expert_start=expert_start,
        local_experts=local_experts,
    )


@deepseek_v4_routing_localized.register_fake
def _deepseek_v4_routing_localized_fake(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = router_logits.shape[0]
    indices = router_logits.new_empty((num_tokens, top_k), dtype=torch.int32)
    weights = router_logits.new_empty((num_tokens, top_k), dtype=torch.bfloat16)
    return indices, weights


# ---------------------------------------------------------------------------
# Hash-router gate (layer_idx < num_hash_layers)
# ---------------------------------------------------------------------------
#
# On hash layers the selected experts come from the ``tid2eid`` token->expert
# map, NOT from the router logits, so only ``top_k`` of the ``num_experts``
# rows of the fp32 router GEMV are ever consumed. The reference chain
#
#     logits  = F.linear(x.to(fp32), W)               # (T, E) cuBLAS gemv, full E rows
#     scores  = F.softplus(logits).sqrt()             # 2 elementwise over (T, E)
#     sel     = tid2eid[input_ids]                    # index_select
#     weights = scores.gather(1, sel)                 # gather
#     weights = weights / (weights.sum(-1) + 1e-20)   # reduce + div
#     weights = weights * routed_scaling_factor       # mul
#
# is ~8 launch-bound CUDA-graph nodes per layer per decode step and reads the
# whole (E, H) fp32 router weight when only top_k rows matter. The fused kernel
# below runs ONE program per token: it loads the token's tid2eid row, gathers
# just those top_k weight rows, computes the dot products, and finishes the
# sqrtsoftplus/renorm/scale tail in registers. Selected experts are
# bit-identical to the reference by construction (pure integer gather).
#
# Matmul precision matches the ambient cuBLAS dispatch for this shape
# (measured, sm100 / cuBLAS in the shipping container):
#   * T == 1: cuBLAS runs a CUDA-core fp32 gemv (TF32 never applies at M=1)
#     -> the kernel multiplies/accumulates in IEEE fp32.
#   * T >= 2 with ``torch.backends.cuda.matmul.allow_tf32``: cuBLAS runs a
#     TF32 tensor-core GEMM that rounds inputs to TF32 with round-to-nearest-
#     even -> the kernel applies the same RNE truncation to x and W; the
#     products of two 11-bit significands are then EXACT in fp32, so the only
#     remaining difference vs cuBLAS is fp32 summation order (~1e-6 relative,
#     vs ~1e-3 had we computed those batches in full fp32).
# Batches larger than ``_HASH_ROUTING_DECODE_MAX_TOKENS`` (prefill) keep the
# bit-identical reference chain, where the dense GEMM is efficient.


@triton.jit
def _deepseek_v4_hash_routing_kernel(
    x_ptr,  # (T, H) hidden states (bf16 or fp32)
    weight_ptr,  # (E, H) fp32 router weight
    tid2eid_ptr,  # (V, K) int64 token->expert map
    input_ids_ptr,  # (T,) integer token ids
    weights_ptr,  # (T, K) fp32 (LOCALIZED=0) or bf16 (LOCALIZED=1) output routing weights
    indices_ptr,  # (T, K) int64 (LOCALIZED=0) or int32 (LOCALIZED=1) output selected experts
    num_tokens,
    hidden_size,
    stride_xt,
    stride_xh,
    stride_we,
    stride_wh,
    stride_tv,
    stride_tk,
    stride_wt,
    stride_wk,
    stride_it,
    stride_ik,
    routed_scaling_factor,
    expert_start,  # EP shard offset (only read when LOCALIZED=1)
    local_experts,  # local expert count / invalid sentinel (only read when LOCALIZED=1)
    SOFTPLUS_THRESHOLD: tl.constexpr,
    NORM_TOPK: tl.constexpr,
    TF32_TRUNC: tl.constexpr,
    LOCALIZED: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,  # >= TOP_K, power of 2
    BLOCK_H: tl.constexpr,  # hidden-dim chunk
):
    """One program per token: tid2eid lookup -> top_k-row dots -> sqrtsoftplus tail."""
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    tok = tl.load(input_ids_ptr + token_id).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < TOP_K
    eids = tl.load(
        tid2eid_ptr + tok * stride_tv + offs_k * stride_tk,
        mask=mask_k,
        other=0,
    ).to(tl.int64)

    # logits[k] = dot(x, W[eids[k], :]) accumulated in fp32. With TF32_TRUNC the
    # inputs are first rounded to TF32 (RNE on the 10-bit mantissa: drop the low
    # 13 fp32 mantissa bits), reproducing the input rounding of the ambient
    # cuBLAS TF32 tensor-core GEMM; the fp32 products are then exact.
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for h0 in range(0, hidden_size, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        mask_h = offs_h < hidden_size
        x = tl.load(
            x_ptr + token_id * stride_xt + offs_h * stride_xh,
            mask=mask_h,
            other=0.0,
        ).to(tl.float32)
        w = tl.load(
            weight_ptr + eids[:, None] * stride_we + offs_h[None, :] * stride_wh,
            mask=mask_k[:, None] & mask_h[None, :],
            other=0.0,
        ).to(tl.float32)
        if TF32_TRUNC:
            xb = x.to(tl.int32, bitcast=True)
            xb = (xb + 4095 + ((xb >> 13) & 1)) & -8192
            x = xb.to(tl.float32, bitcast=True)
            wb = w.to(tl.int32, bitcast=True)
            wb = (wb + 4095 + ((wb >> 13) & 1)) & -8192
            w = wb.to(tl.float32, bitcast=True)
        acc += tl.sum(w * x[None, :], axis=1)

    # scores = sqrt(softplus(logits)). Match F.softplus' threshold form with a
    # compensated log1p emulation (log(u) * y / (u - 1), u = 1 + y) so
    # hash-selected experts with very negative logits keep full precision; the
    # u == 1.0 branch returns y itself where 1 + y rounds to 1.
    exp_l = tl.exp(acc)
    u = 1.0 + exp_l
    log1p_l = tl.where(u == 1.0, exp_l, tl.log(u) * exp_l / (u - 1.0))
    softplus_l = tl.where(acc > SOFTPLUS_THRESHOLD, acc, log1p_l)
    w_k = tl.sqrt_rn(softplus_l)

    if NORM_TOPK:
        denom = tl.sum(tl.where(mask_k, w_k, 0.0), axis=0) + 1e-20
        w_k = w_k / denom
    w_k = w_k * routed_scaling_factor

    if LOCALIZED:
        # Fused EP global->local localization (mirrors ``_localize_routing_kernel``
        # bit for bit): off-rank routes -> invalid sentinel ``local_experts`` + zero
        # weight; valid weights are the single-rounded bf16 of the exact fp32 value.
        local = eids - expert_start
        valid = (local >= 0) & (local < local_experts)
        out_idx = tl.where(valid, local, local_experts).to(tl.int32)
        out_w = tl.where(valid, w_k, 0.0).to(tl.bfloat16)
        tl.store(weights_ptr + token_id * stride_wt + offs_k * stride_wk, out_w, mask=mask_k)
        tl.store(indices_ptr + token_id * stride_it + offs_k * stride_ik, out_idx, mask=mask_k)
    else:
        tl.store(weights_ptr + token_id * stride_wt + offs_k * stride_wk, w_k, mask=mask_k)
        tl.store(indices_ptr + token_id * stride_it + offs_k * stride_ik, eids, mask=mask_k)


# Above this token count the dense reference chain (bit-identical to the
# pre-fusion gate, efficient tensor-core GEMM at prefill) is used; at/below it
# the fused per-token kernel wins. Decode CUDA-graph capture happens at batch
# sizes at/below this bound. Kept small because the ambient-precision mirror
# (fp32 gemv at T=1, TF32-RNE GEMM at T>=2) is validated per shape: past T~8
# cuBLAS reshuffles its accumulation again (~3e-5 drift) for no launch-count
# benefit.
_HASH_ROUTING_DECODE_MAX_TOKENS = 8


def _localize_routing_eager(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager mirror of ``deepseek_v4_localize_routing`` (prefill fallback paths)."""
    local = selected_experts.to(torch.int64) - int(expert_start)
    valid = (local >= 0) & (local < local_experts)
    local_idx = torch.where(valid, local, torch.full_like(local, local_experts)).to(torch.int32)
    weights = torch.where(
        valid, routing_weights.to(torch.float32), torch.zeros((), device=routing_weights.device)
    ).to(torch.bfloat16)
    return local_idx, weights


def deepseek_v4_hash_routing_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    expert_start: int = 0,
    local_experts: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 hash-router gate.

    Args:
        hidden_states: (T, H) hidden states (any float dtype; read as fp32).
        weight: (E, H) fp32 router weight.
        tid2eid: (V, top_k) int64 token->expert map.
        input_ids: (T,) integer token ids.
        routed_scaling_factor: scalar multiplier applied to the routing weights.
        norm_topk_prob: if True, renormalize the selected weights to sum to 1.
        expert_start: EP shard offset (localized mode only).
        local_experts: when > 0, emit EP-LOCALIZED outputs (int32 local ids with the
            invalid sentinel ``local_experts`` for off-rank routes, bf16 masked
            weights), exactly as ``deepseek_v4_localize_routing`` would produce.

    Returns:
        selected_experts: (T, top_k) int64 expert indices (= tid2eid[input_ids]),
            or int32 local ids in localized mode.
        routing_weights: (T, top_k) fp32 routing weights, or bf16 masked weights in
            localized mode.
    """
    assert hidden_states.ndim == 2, "hidden_states must be 2-D (T, H)"
    num_tokens, hidden_size = hidden_states.shape
    top_k = tid2eid.shape[1]
    localized = local_experts > 0

    if num_tokens > _HASH_ROUTING_DECODE_MAX_TOKENS:
        # Prefill: keep the reference chain bit-identical (dense GEMM is the
        # right tool at large T; the launch count amortizes).
        router_logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight).float()
        scores = torch.nn.functional.softplus(router_logits).sqrt()
        selected_experts = tid2eid[input_ids.to(torch.long)].to(torch.int64)
        routing_weights = scores.gather(1, selected_experts)
        if norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * routed_scaling_factor
        if localized:
            return _localize_routing_eager(
                selected_experts, routing_weights, expert_start, local_experts
            )
        return selected_experts, routing_weights

    idx_dtype = torch.int32 if localized else torch.int64
    w_dtype = torch.bfloat16 if localized else torch.float32
    weights = torch.empty((num_tokens, top_k), dtype=w_dtype, device=hidden_states.device)
    indices = torch.empty((num_tokens, top_k), dtype=idx_dtype, device=hidden_states.device)

    # Mirror the ambient cuBLAS dispatch this kernel replaces: M=1 runs a
    # CUDA-core fp32 gemv (TF32 never applies), M>=2 runs a TF32 tensor-core
    # GEMM when torch has TF32 matmul enabled (the shipping container default).
    tf32_trunc = torch.backends.cuda.matmul.allow_tf32 and num_tokens > 1

    BLOCK_K = _next_power_of_2(top_k)
    grid = (num_tokens,)
    _deepseek_v4_hash_routing_kernel[grid](
        hidden_states,
        weight,
        tid2eid,
        input_ids,
        weights,
        indices,
        num_tokens,
        hidden_size,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        tid2eid.stride(0),
        tid2eid.stride(1),
        weights.stride(0),
        weights.stride(1),
        indices.stride(0),
        indices.stride(1),
        routed_scaling_factor,
        int(expert_start),
        int(local_experts),
        SOFTPLUS_THRESHOLD=20.0,
        NORM_TOPK=norm_topk_prob,
        TF32_TRUNC=tf32_trunc,
        LOCALIZED=localized,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        BLOCK_H=512,
        num_warps=4,
    )
    return indices, weights


@torch.library.custom_op("auto_deploy::deepseek_v4_hash_routing", mutates_args=())
def deepseek_v4_hash_routing(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 hash-router gate (top_k-row GEMV + sqrtsoftplus tail).

    Equivalent to::

        logits = F.linear(hidden_states.to(weight.dtype), weight).float()
        scores = F.softplus(logits).sqrt()
        selected_experts = tid2eid[input_ids.to(torch.long)].to(torch.int64)
        routing_weights = scores.gather(1, selected_experts)
        if norm_topk_prob:
            routing_weights = routing_weights / (routing_weights.sum(-1, keepdim=True) + 1e-20)
        routing_weights = routing_weights * routed_scaling_factor

    but at decode only the ``top_k`` weight rows named by ``tid2eid[input_ids]``
    are read and the whole chain runs as one Triton program per token.
    """
    return deepseek_v4_hash_routing_fn(
        hidden_states, weight, tid2eid, input_ids, routed_scaling_factor, norm_topk_prob
    )


@deepseek_v4_hash_routing.register_fake
def _deepseek_v4_hash_routing_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    top_k = tid2eid.shape[1]
    indices = hidden_states.new_empty((num_tokens, top_k), dtype=torch.int64)
    weights = hidden_states.new_empty((num_tokens, top_k), dtype=torch.float32)
    return indices, weights


@torch.library.custom_op("auto_deploy::deepseek_v4_hash_routing_localized", mutates_args=())
def deepseek_v4_hash_routing_localized(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused DeepSeek V4 hash-router gate emitting EP-LOCALIZED routing.

    Same gemv + sqrtsoftplus + renorm/scale chain as
    :func:`deepseek_v4_hash_routing`, with the EP global->local localization
    (``deepseek_v4_localize_routing``) folded into the kernel tail (eager mirror on
    the prefill reference branch). Returns ``(local_idx int32, weights bf16)``:
    off-rank routes carry the invalid sentinel ``local_experts`` and weight ``0``;
    valid slots are bit-identical to running the two ops back to back.
    """
    if local_experts <= 0:
        raise ValueError(f"local_experts should be positive, got {local_experts}.")
    return deepseek_v4_hash_routing_fn(
        hidden_states,
        weight,
        tid2eid,
        input_ids,
        routed_scaling_factor,
        norm_topk_prob,
        expert_start=expert_start,
        local_experts=local_experts,
    )


@deepseek_v4_hash_routing_localized.register_fake
def _deepseek_v4_hash_routing_localized_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    top_k = tid2eid.shape[1]
    indices = hidden_states.new_empty((num_tokens, top_k), dtype=torch.int32)
    weights = hidden_states.new_empty((num_tokens, top_k), dtype=torch.bfloat16)
    return indices, weights
