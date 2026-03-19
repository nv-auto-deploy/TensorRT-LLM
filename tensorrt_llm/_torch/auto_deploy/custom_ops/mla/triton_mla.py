# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton-based MLA (Multi-head Latent Attention) backend with unpaged cache.

This module provides:
- triton_cached_mla_with_cache: Triton-optimized cached MLA with KV cache
- TritonMLAAttention: Attention descriptor for Triton MLA backend

Both prefill and decode paths use Triton-accelerated attention with weight
absorption, eliminating Python loops entirely. A single shared kernel handles
both phases via per-token KV length metadata for causal masking.

MLA Cache Layout (same as torch_mla backend):
    mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    - compressed_kv_cached = mla_cache[:, :, :kv_lora_rank]  (zero-copy slice)
    - kpe_cached = mla_cache[:, :, kv_lora_rank:]  (zero-copy slice)
"""

import math
from typing import List, Optional

import torch
import triton
import triton.language as tl
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)


@triton.jit
def _mla_attention_kernel(
    # Tensor pointers
    q_absorbed_ptr,  # [num_tokens, N, kv_lora_rank]
    q_pe_ptr,  # [num_tokens, N, qk_rope_head_dim]
    mla_cache_ptr,  # [max_batch, max_seq, cache_dim]
    token_slot_ptr,  # [num_tokens] - cache slot per token
    token_kv_len_ptr,  # [num_tokens] - KV length per token (causal boundary)
    out_ptr,  # [num_tokens, N, kv_lora_rank] (float32)
    # Constexpr parameters
    SCALE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    CACHE_DIM: tl.constexpr,
    KV_BLOCK: tl.constexpr,  # next_power_of_2(kv_lora_rank)
    PE_BLOCK: tl.constexpr,  # next_power_of_2(qk_rope_head_dim)
    SEQ_BLOCK: tl.constexpr,  # sequence tile size
):
    """MLA attention kernel with online softmax (shared by prefill and decode).

    Each program processes one (token, head) pair. Iterates over the cached
    KV sequence using online softmax to compute the softmax-weighted
    sum of compressed_kv values.

    Grid: (num_tokens, N_HEADS)

    This kernel implements the core MLA attention in compressed space:
        score[t] = (q_absorbed . compressed_kv[t] + q_pe . kpe[t]) * scale
        attn_weights = softmax(scores[:kv_len])
        weighted_kv = sum(attn_weights[t] * compressed_kv[t])

    The weight absorption (q_absorbed = q_nope @ w_k_nope^T) and the final
    value projection (out = weighted_kv @ w_v^T) are done outside this kernel.
    Per-token kv_len provides causal masking for prefill tokens.
    """
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    slot_idx = tl.load(token_slot_ptr + token_id)
    kv_len = tl.load(token_kv_len_ptr + token_id)

    # Load absorbed query for this head: [KV_BLOCK]
    q_abs_base = token_id * N_HEADS * KV_LORA_RANK + head_id * KV_LORA_RANK
    kv_offsets = tl.arange(0, KV_BLOCK)
    kv_mask = kv_offsets < KV_LORA_RANK
    q_abs = tl.load(q_absorbed_ptr + q_abs_base + kv_offsets, mask=kv_mask, other=0.0).to(
        tl.float32
    )

    # Load q_pe for this head: [PE_BLOCK]
    q_pe_base = token_id * N_HEADS * QK_ROPE_HEAD_DIM + head_id * QK_ROPE_HEAD_DIM
    pe_offsets = tl.arange(0, PE_BLOCK)
    pe_mask = pe_offsets < QK_ROPE_HEAD_DIM
    q_pe = tl.load(q_pe_ptr + q_pe_base + pe_offsets, mask=pe_mask, other=0.0).to(tl.float32)

    # Initialize online softmax accumulators
    m_i = float("-inf")  # running max score
    l_i = 0.0  # running sum of exp(score - max)
    acc = tl.zeros([KV_BLOCK], dtype=tl.float32)  # running weighted sum of compressed_kv

    cache_batch_base = slot_idx * MAX_SEQ_LEN * CACHE_DIM
    num_blocks = (kv_len + SEQ_BLOCK - 1) // SEQ_BLOCK

    for block_id in range(0, num_blocks):
        block_start = block_id * SEQ_BLOCK
        seq_offsets = block_start + tl.arange(0, SEQ_BLOCK)
        seq_mask = seq_offsets < kv_len

        # Load compressed_kv for this block: [SEQ_BLOCK, KV_BLOCK]
        ckv_ptrs = (
            mla_cache_ptr
            + cache_batch_base
            + seq_offsets[:, None] * CACHE_DIM
            + kv_offsets[None, :]
        )
        ckv = tl.load(ckv_ptrs, mask=seq_mask[:, None] & kv_mask[None, :], other=0.0).to(tl.float32)

        # Load kpe for this block: [SEQ_BLOCK, PE_BLOCK]
        kpe_ptrs = (
            mla_cache_ptr
            + cache_batch_base
            + seq_offsets[:, None] * CACHE_DIM
            + KV_LORA_RANK
            + pe_offsets[None, :]
        )
        kpe = tl.load(kpe_ptrs, mask=seq_mask[:, None] & pe_mask[None, :], other=0.0).to(tl.float32)

        # Compute attention scores: [SEQ_BLOCK]
        # score_nope = q_absorbed . compressed_kv  (dot product over kv_lora_rank)
        # score_pe = q_pe . kpe  (dot product over qk_rope_head_dim)
        scores_nope = tl.sum(q_abs[None, :] * ckv, axis=1)
        scores_pe = tl.sum(q_pe[None, :] * kpe, axis=1)
        scores = (scores_nope + scores_pe) * SCALE
        scores = tl.where(seq_mask, scores, float("-inf"))

        # Online softmax update
        m_ij = tl.max(scores)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        # Ensure masked positions contribute 0 (guards against -inf - (-inf) = nan)
        p = tl.where(seq_mask, p, 0.0)

        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * ckv, axis=0)
        m_i = m_new

    # Normalize by softmax denominator
    safe_l_i = tl.maximum(l_i, 1e-38)
    acc = acc / safe_l_i

    # Store weighted_kv result
    out_base = token_id * N_HEADS * KV_LORA_RANK + head_id * KV_LORA_RANK
    tl.store(out_ptr + out_base + kv_offsets, acc, mask=kv_mask)


def _triton_mla_decode(
    q_nope: torch.Tensor,  # [B, 1, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, 1, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, 1, kv_lora_rank]
    kpe: torch.Tensor,  # [B, 1, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, cache_dim]
    slot_idx: torch.Tensor,  # [num_decode]
    input_pos: torch.Tensor,  # [num_decode]
    scale: float,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,  # [B, N, v_head_dim]
) -> None:
    """Triton-accelerated MLA decode with weight absorption.

    Steps:
    1. Cache update (PyTorch advanced indexing)
    2. Weight absorption: q_absorbed = q_nope @ w_k_nope^T (PyTorch einsum)
    3. Triton kernel: fused attention scoring + online softmax + weighted sum
    4. Value projection: out = weighted_kv @ w_v^T (PyTorch einsum)
    """
    b = q_nope.shape[0]
    qk_rope_head_dim = q_pe.shape[3]

    # Reshape kv_b_proj_weight to extract w_k_nope and w_v
    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]

    # Step 1: Update cache with new token
    compressed_kv_flat = compressed_kv.squeeze(1)  # [B, kv_lora_rank]
    kpe_flat = kpe.squeeze(1).squeeze(1)  # [B, qk_rope_head_dim]

    cache_dtype = mla_cache.dtype
    if compressed_kv_flat.dtype != cache_dtype:
        compressed_kv_for_cache = compressed_kv_flat.to(cache_dtype)
    else:
        compressed_kv_for_cache = compressed_kv_flat
    if kpe_flat.dtype != cache_dtype:
        kpe_for_cache = kpe_flat.to(cache_dtype)
    else:
        kpe_for_cache = kpe_flat

    # Batch scatter into cache (no Python loops)
    mla_cache[slot_idx.long(), input_pos.long(), :kv_lora_rank] = compressed_kv_for_cache
    mla_cache[slot_idx.long(), input_pos.long(), kv_lora_rank:] = kpe_for_cache

    # Step 2: Weight absorption
    q_nope_2d = q_nope.squeeze(1)  # [B, N, qk_nope_head_dim]
    q_absorbed = torch.einsum("bnd,ndk->bnk", q_nope_2d, w_k_nope).contiguous()
    # [B, N, kv_lora_rank]

    q_pe_2d = q_pe.squeeze(1).contiguous()  # [B, N, qk_rope_head_dim]

    # Step 3: Triton kernel for attention computation
    weighted_kv = torch.empty(b, num_heads, kv_lora_rank, device=q_nope.device, dtype=torch.float32)

    max_seq_len = mla_cache.shape[1]
    cache_dim = mla_cache.shape[2]
    kv_block = triton.next_power_of_2(kv_lora_rank)
    pe_block = triton.next_power_of_2(qk_rope_head_dim)

    # Per-token KV length: decode attends to positions [0, input_pos]
    kv_len = (input_pos + 1).to(torch.int32)

    grid = (b, num_heads)
    _mla_attention_kernel[grid](
        q_absorbed,
        q_pe_2d,
        mla_cache,
        slot_idx,
        kv_len,
        weighted_kv,
        SCALE=scale,
        MAX_SEQ_LEN=max_seq_len,
        N_HEADS=num_heads,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        CACHE_DIM=cache_dim,
        KV_BLOCK=kv_block,
        PE_BLOCK=pe_block,
        SEQ_BLOCK=8,
        num_warps=2,
        num_stages=2,
    )

    # Step 4: Value projection
    weighted_kv = weighted_kv.to(q_nope.dtype)
    attn_out = torch.einsum("bnk,nvk->bnv", weighted_kv, w_v)  # [B, N, v_head_dim]

    out.copy_(attn_out)


def _triton_mla_prefill(
    q_nope: torch.Tensor,  # [total_padded, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [total_padded, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [total_padded, kv_lora_rank]
    kpe: torch.Tensor,  # [total_padded, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    input_pos: torch.Tensor,  # [num_seq] - starting cache position per sequence
    slot_idx: torch.Tensor,  # [num_seq] - cache slot per sequence
    seq_len: torch.Tensor,  # [num_seq] - token count per sequence
    seq_start: torch.Tensor,  # [num_seq] - start index in flattened tensor
    scale: float,
    kv_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,  # [total_padded, N, v_head_dim]
) -> None:
    """Triton-accelerated MLA prefill with weight absorption.

    Uses the same attention kernel as decode but with per-token causal masking.
    Replaces the PyTorch reference (_torch_mla_context_with_expansion) which had
    Python loops over sequences.

    Steps:
    1. Vectorized cache update (no Python loops)
    2. Weight absorption: q_absorbed = q_nope @ W_kn^T (PyTorch einsum)
    3. Triton kernel: fused attention scoring + online softmax + weighted sum
    4. Value projection: out = weighted_kv @ W_v^T (PyTorch einsum)
    """
    device = q_nope.device
    qk_rope_head_dim = q_pe.shape[2]
    num_seq = seq_len.shape[0]
    seq_lengths = seq_len.long()
    seq_start = seq_start.long()
    total_tokens = seq_lengths.sum().item()

    if total_tokens == 0:
        out.zero_()
        return

    # =====================================================================
    # Step 1: Vectorized cache update (no Python loops)
    # =====================================================================
    # Build per-token metadata from per-sequence metadata
    token_slots = slot_idx.long().repeat_interleave(seq_lengths)  # [total_tokens]
    base_positions = input_pos.long().repeat_interleave(seq_lengths)  # [total_tokens]

    # Within-sequence offsets: [0,1,...,sl0-1, 0,1,...,sl1-1, ...]
    cum_lengths = torch.zeros(num_seq + 1, device=device, dtype=torch.long)
    cum_lengths[1:] = seq_lengths.cumsum(0)
    base_in_dense = cum_lengths[:-1].repeat_interleave(seq_lengths)
    within_offsets = torch.arange(total_tokens, device=device) - base_in_dense

    # Index into flattened/padded input tensors using true per-sequence starts.
    token_input_idx = seq_start.repeat_interleave(seq_lengths) + within_offsets  # [total_tokens]
    token_cache_pos = base_positions + within_offsets  # [total_tokens]

    # Vectorized cache write
    kpe_flat = kpe.index_select(0, token_input_idx).squeeze(1)  # [total_tokens, qk_rope_head_dim]
    ckv_actual = compressed_kv.index_select(0, token_input_idx)  # [total_tokens, kv_lora_rank]

    cache_dtype = mla_cache.dtype
    ckv_for_cache = ckv_actual.to(cache_dtype) if ckv_actual.dtype != cache_dtype else ckv_actual
    kpe_for_cache = kpe_flat.to(cache_dtype) if kpe_flat.dtype != cache_dtype else kpe_flat

    mla_cache[token_slots, token_cache_pos, :kv_lora_rank] = ckv_for_cache
    mla_cache[token_slots, token_cache_pos, kv_lora_rank:] = kpe_for_cache

    # =====================================================================
    # Step 2: Weight absorption
    # =====================================================================
    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]

    q_nope_actual = q_nope.index_select(0, token_input_idx)  # [total_tokens, N, qk_nope_head_dim]
    q_pe_actual = q_pe.index_select(
        0, token_input_idx
    ).contiguous()  # [total_tokens, N, qk_rope_head_dim]

    # q_absorbed: [total_tokens, N, kv_lora_rank]
    # Use fp32 for absorption to minimize bf16 reduction error over qk_nope_head_dim
    q_absorbed = torch.einsum("tnd,ndk->tnk", q_nope_actual.float(), w_k_nope.float()).contiguous()

    # =====================================================================
    # Step 3: Triton attention kernel
    # =====================================================================
    # Per-token KV length for causal masking: token at cache position p attends to [0, p]
    token_kv_len = (token_cache_pos + 1).to(torch.int32)  # [total_tokens]

    weighted_kv = torch.empty(
        total_tokens, num_heads, kv_lora_rank, device=device, dtype=torch.float32
    )

    max_seq_len = mla_cache.shape[1]
    cache_dim = mla_cache.shape[2]
    kv_block = triton.next_power_of_2(kv_lora_rank)
    pe_block = triton.next_power_of_2(qk_rope_head_dim)

    grid = (total_tokens, num_heads)
    _mla_attention_kernel[grid](
        q_absorbed,
        q_pe_actual,
        mla_cache,
        token_slots,
        token_kv_len,
        weighted_kv,
        SCALE=scale,
        MAX_SEQ_LEN=max_seq_len,
        N_HEADS=num_heads,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        CACHE_DIM=cache_dim,
        KV_BLOCK=kv_block,
        PE_BLOCK=pe_block,
        SEQ_BLOCK=16,  # Larger blocks for prefill (longer sequences)
        num_warps=4,
        num_stages=2,
    )

    # =====================================================================
    # Step 4: Value projection (fp32 to minimize bf16 reduction error over kv_lora_rank)
    # =====================================================================
    attn_out = torch.einsum("tnk,nvk->tnv", weighted_kv, w_v.float()).to(
        q_nope.dtype
    )  # [total_tokens, N, v_head_dim]

    out.zero_()
    out.index_copy_(0, token_input_idx, attn_out)


@torch.library.custom_op("auto_deploy::triton_cached_mla_with_cache", mutates_args=())
def triton_cached_mla_with_cache(
    # 5 tensor args (get_num_qkv_args = 5)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    # Standard metadata
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # Cache (unpaged, same layout as torch_mla)
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    # Constants
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
) -> torch.Tensor:
    """Triton backend MLA with compressed cache.

    Both prefill and decode use Triton-accelerated attention with weight
    absorption and online softmax. No Python loops in either path.

    Cache Layout:
        mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
        - compressed_kv = mla_cache[:, :, :kv_lora_rank]  (zero-copy slice)
        - kpe = mla_cache[:, :, kv_lora_rank:]  (zero-copy slice)
    """
    # Get dimensions
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Infer v_head_dim from kv_b_proj_weight
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    # Get cleaned up metadata
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    input_pos = input_pos[:num_seq]
    slot_idx = slot_idx[:num_seq]
    seq_start = cu_seqlen[:num_seq]

    # Set scale
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Define output shape: [B, S, N, v_head_dim]
    output_shape = (b, s, num_heads, v_head_dim)

    if s == 1:
        # =================================================================
        # Generate phase: Use Triton-accelerated decode with absorption
        # =================================================================
        y = q_nope.new_empty(b, num_heads, v_head_dim).contiguous()

        _triton_mla_decode(
            q_nope,
            q_pe,
            compressed_kv,
            kpe,
            kv_b_proj_weight,
            mla_cache,
            slot_idx,
            input_pos,
            scale,
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )

        return y.unsqueeze(1)  # [B, 1, N, v_head_dim]
    else:
        # =================================================================
        # Context phase: Triton attention with absorption (no Python loops)
        # =================================================================
        bs_view = (b * s,)

        q_nope_flat = q_nope.contiguous().view(*bs_view, num_heads, qk_nope_head_dim)
        q_pe_flat = q_pe.contiguous().view(*bs_view, num_heads, qk_rope_head_dim)
        compressed_kv_flat = compressed_kv.contiguous().view(*bs_view, kv_lora_rank)
        kpe_flat = kpe.contiguous().view(*bs_view, 1, qk_rope_head_dim)

        y = q_nope.new_empty(*bs_view, num_heads, v_head_dim).contiguous()

        _triton_mla_prefill(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            kv_b_proj_weight,
            mla_cache,
            input_pos,
            slot_idx,
            seq_len,
            seq_start,
            scale,
            kv_lora_rank,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )

        return y.view(*output_shape)


@triton_cached_mla_with_cache.register_fake
def triton_cached_mla_with_cache_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    mla_cache: torch.Tensor,
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    return q_nope.new_empty(
        q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
    ).contiguous()


@AttentionRegistry.register("triton_mla")
class TritonMLAAttention(AttentionDescriptor):
    """Attention descriptor for Triton-based MLA with unpaged cache.

    Uses the same cache layout as torch_mla:
        mla_cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]

    Both prefill and decode use a shared Triton kernel with weight absorption
    + online softmax. Per-token KV length metadata provides causal masking.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 5  # q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.triton_cached_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Get cache initializers using unpaged MLA cache layout."""
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]

        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]

        model_dtype = compressed_kv_fake.dtype
        cache_dtype = cls.resolve_cache_dtype(cache_config.dtype, model_dtype)

        return {
            "mla_cache": UnpagedResourceHandler(
                kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        scale = source_attn_node.kwargs.get("scale", None)
        return [scale, kv_lora_rank]
