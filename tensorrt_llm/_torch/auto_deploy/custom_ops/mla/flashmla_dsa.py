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

"""FlashMLA-based DSA (DeepSeek Sparse Attention) backend with paged KV caches.

Provides:
- flash_mla_dsa_with_cache: cached DSA op using FlashMLA kernels with paged caches
- FlashMLADSAAttention: AttentionDescriptor registered as "flashmla_dsa"

Cache layout (paged, managed by MLAPagedResourceHandler):
    mla_cache:     [num_blocks, page_size, 1, kv_lora_rank + qk_rope_head_dim]
    index_k_cache: [num_blocks, page_size, index_head_dim]

Decode:  flash_mla_with_kvcache (sparse, causal=False, no Python loops)
Prefill: flash_mla_sparse_fwd (flat ragged, no Python loops) when kv_lora_rank==512;
         falls back to flash_mla_with_kvcache(causal=True) + Q-padding otherwise.
"""

import math
from typing import List, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    MLAPagedResourceHandler,
    ResourceHandlerDict,
)

# ---------------------------------------------------------------------------
# Vectorized paged-cache helpers (no Python loops over sequences)
# ---------------------------------------------------------------------------


def _write_tokens_to_paged_caches(
    compressed_kv: torch.Tensor,  # [total_tokens, kv_lora_rank]
    kpe: torch.Tensor,  # [total_tokens, 1, qk_rope_head_dim]
    index_k: torch.Tensor,  # [total_tokens, index_head_dim]
    mla_cache: torch.Tensor,  # [num_blocks, page_size, 1, kv_lora_rank + rope_dim]
    index_k_cache: torch.Tensor,  # [num_blocks, page_size, index_head_dim]
    seq_len: torch.Tensor,  # [B] int32 — tokens per seq in this batch
    input_pos: torch.Tensor,  # [B] int32 — absolute start position in cache per seq
    cu_seqlen: torch.Tensor,  # [B+1] int32 — cumulative token offsets in flat tensors
    cache_loc: torch.Tensor,  # [total_pages] int32 — page index array
    cu_num_pages: torch.Tensor,  # [B+1] int32 — cumulative page counts per seq
) -> None:
    """Write all tokens to paged caches using vectorized scatter (no Python loops)."""
    B = seq_len.shape[0]
    total_tokens = int(cu_seqlen[-1].item())
    page_size = mla_cache.shape[1]
    device = mla_cache.device

    # seq_ids[t] = which sequence token t belongs to
    seq_ids = torch.repeat_interleave(torch.arange(B, device=device), seq_len)  # [total_tokens]

    # local_pos[t] = position within sequence (0-indexed from the batch's q start)
    seq_start = cu_seqlen[:-1]  # [B]
    local_pos = torch.arange(total_tokens, device=device) - seq_start[seq_ids]

    # abs_pos[t] = absolute position in cache for token t
    abs_pos = input_pos[seq_ids] + local_pos  # [total_tokens]

    # Map to (page_idx, page_offset) via page table
    page_k = abs_pos // page_size  # [total_tokens]
    page_off = abs_pos % page_size  # [total_tokens]
    gather_base = cu_num_pages[seq_ids] + page_k  # [total_tokens]
    page_indices = cache_loc[gather_base]  # [total_tokens]

    # Combine ckv + kpe into mla_cache entry
    kpe_sq = kpe.squeeze(1)  # [total_tokens, rope_dim]
    flat_kv = torch.cat([compressed_kv, kpe_sq], dim=-1)  # [total_tokens, lora+rope]
    mla_cache[page_indices, page_off, 0, :] = flat_kv.to(mla_cache.dtype)
    index_k_cache[page_indices, page_off, :] = index_k.to(index_k_cache.dtype)


def _gather_index_k_dense(
    index_k_cache: torch.Tensor,  # [num_blocks, page_size, D_idx]
    cache_loc: torch.Tensor,  # [total_pages] int32
    cu_num_pages: torch.Tensor,  # [B+1] int32
    cache_seqlens: torch.Tensor,  # [B] int32 — total cached lengths per seq
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather index_k into dense [B, max_T, D_idx] without Python loops.

    Returns:
        dense_index_k: [B, max_T, D_idx] (padded with zeros)
        valid_mask:    [B, max_T] bool (True = valid KV position)
    """
    B = cache_seqlens.shape[0]
    max_T = int(cache_seqlens.max().item())
    page_size = index_k_cache.shape[1]
    device = index_k_cache.device

    t_range = torch.arange(max_T, device=device)  # [max_T]
    page_k = t_range.unsqueeze(0) // page_size  # [1, max_T]
    page_off = t_range.unsqueeze(0) % page_size  # [1, max_T]

    # Clamp to avoid out-of-bounds on padded positions
    base = cu_num_pages[:-1].unsqueeze(1)  # [B, 1]
    gather_idx = (base + page_k).clamp(0, cache_loc.shape[0] - 1)  # [B, max_T]
    page_indices = cache_loc[gather_idx]  # [B, max_T]

    dense_index_k = index_k_cache[page_indices, page_off, :]  # [B, max_T, D_idx]

    # Mask out positions beyond actual cached length
    valid_mask = t_range.unsqueeze(0) < cache_seqlens.unsqueeze(1)  # [B, max_T]
    return dense_index_k, valid_mask


def _gather_mla_kv_dense(
    mla_cache: torch.Tensor,  # [num_blocks, page_size, 1, D_qk]
    cache_loc: torch.Tensor,  # [total_pages] int32
    cu_num_pages: torch.Tensor,  # [B+1] int32
    cache_seqlens: torch.Tensor,  # [B] int32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather all cached KV tokens into flat dense tensors without Python loops.

    Returns:
        dense_kv:        [total_kv_tokens, 1, D_qk] for flash_mla_sparse_fwd
        kv_cu_seqlens:   [B+1] int32 cumulative KV seqlens
    """
    B = cache_seqlens.shape[0]
    total_kv = int(cache_seqlens.sum().item())
    page_size = mla_cache.shape[1]
    device = mla_cache.device

    kv_seq_ids = torch.repeat_interleave(torch.arange(B, device=device), cache_seqlens)
    kv_cu_seqlens = torch.cat([cache_seqlens.new_zeros(1), cache_seqlens.cumsum(0)])
    kv_local_pos = torch.arange(total_kv, device=device) - kv_cu_seqlens[kv_seq_ids]

    page_k = kv_local_pos // page_size
    page_off = kv_local_pos % page_size
    gather_base = cu_num_pages[kv_seq_ids] + page_k
    page_indices = cache_loc[gather_base]

    dense_kv = mla_cache[page_indices, page_off, :, :]  # [total_kv, 1, D_qk]
    return dense_kv, kv_cu_seqlens


def _build_block_table(
    cache_loc: torch.Tensor,  # [total_pages] int32
    cu_num_pages: torch.Tensor,  # [B+1] int32
    B: int,
    device: torch.device,
) -> torch.Tensor:
    """Build block_table [B, max_pages] from cache_loc and cu_num_pages (no Python loop)."""
    page_counts = cu_num_pages[1:] - cu_num_pages[:-1]  # [B]
    max_pages = int(page_counts.max().item())
    block_table = cache_loc.new_zeros(B, max_pages)
    col_idx = torch.arange(max_pages, device=device).unsqueeze(0)  # [1, max_pages]
    valid = col_idx < page_counts.unsqueeze(1)  # [B, max_pages]
    flat_src_idx = cu_num_pages[:-1].unsqueeze(1) + col_idx  # [B, max_pages]
    flat_src_idx = flat_src_idx.clamp(0, cache_loc.shape[0] - 1)
    block_table[valid] = cache_loc[flat_src_idx[valid]]
    return block_table.to(torch.int32)


def _compute_batched_index_topk(
    index_q: torch.Tensor,  # [B, S, H, D_idx]
    dense_index_k: torch.Tensor,  # [B, max_T, D_idx]
    index_weights: torch.Tensor,  # [B, S, H]
    valid_mask: torch.Tensor,  # [B, max_T] bool
    index_topk: int,
    index_softmax_scale: float,
    is_causal: bool,
    cache_seqlens: torch.Tensor,  # [B] int32
) -> torch.Tensor:
    """Returns sparse_indices [B, S, topk] int32 — no Python loops."""
    S = index_q.shape[1]
    max_T = dense_index_k.shape[1]

    # [B, S, H, max_T]
    per_head_scores = torch.einsum(
        "bshd,btd->bsht",
        index_q.float(),
        dense_index_k.float(),
    )
    # [B, S, max_T]
    index_score = torch.einsum("bsht,bsh->bst", per_head_scores, index_weights.float())
    index_score = index_score * index_softmax_scale

    # Mask invalid (padded) KV positions
    index_score.masked_fill_(~valid_mask.unsqueeze(1), float("-inf"))

    # Causal mask for prefill
    if is_causal and S > 1:
        s_range = torch.arange(S, device=index_q.device)  # [S]
        t_range = torch.arange(max_T, device=index_q.device)  # [max_T]
        # query s can attend to KV positions t where t < cache_seqlens[b] - S + s + 1
        causal_limit = cache_seqlens.unsqueeze(1).long() - S + s_range.unsqueeze(0)  # [B, S]
        future_mask = t_range.unsqueeze(0).unsqueeze(0) >= causal_limit.unsqueeze(
            2
        )  # [B, S, max_T]
        index_score.masked_fill_(future_mask, float("-inf"))

    effective_topk = min(index_topk, max_T)
    topk_indices = index_score.topk(effective_topk, dim=-1).indices  # [B, S, topk]
    return topk_indices.to(torch.int32)


# ---------------------------------------------------------------------------
# Custom op: flash_mla_dsa_with_cache
# ---------------------------------------------------------------------------


def _is_sparse_mla_supported() -> bool:
    """Check if sparse FlashMLA is supported on the current device (SM100+ / Blackwell)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


@torch.library.custom_op("auto_deploy::flash_mla_dsa_with_cache", mutates_args=())
def flash_mla_dsa_with_cache(
    # 8 QKV tensor args (matches get_num_qkv_args = 8)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N*(qk_nope+v), kv_lora_rank]
    index_q: torch.Tensor,  # [B, S, H, D_idx]
    index_k: torch.Tensor,  # [B, S, D_idx]
    index_weights: torch.Tensor,  # [B, S, H]
    # Standard paged metadata
    batch_info_host: torch.Tensor,  # [3] int host: [num_prefill, num_prefill_tokens, num_decode]
    seq_len: torch.Tensor,  # [B] int32 — token counts per seq in this batch
    input_pos: torch.Tensor,  # [B] int32 — position offset into cache per seq
    cu_seqlen: torch.Tensor,  # [B+1] int32 — cumulative token offsets in flat tensors
    cache_loc: torch.Tensor,  # [total_pages] int32 — page index array
    cu_num_pages: torch.Tensor,  # [B+1] int32 — cumulative page counts per seq
    last_page_len: torch.Tensor,  # [B] int32 — valid tokens in last page
    # Paged caches
    mla_cache: torch.Tensor,  # [num_blocks, page_size, 1, kv_lora_rank + rope_dim]
    index_k_cache: torch.Tensor,  # [num_blocks, page_size, index_head_dim]
    # Constants
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
    index_topk: int = 64,
) -> torch.Tensor:
    """FlashMLA-based DSA with paged KV caches.

    Decode  (S==1):
      - SM100+ (Blackwell): flash_mla_with_kvcache with sparse indices
      - SM90  (Hopper):    flash_mla_with_kvcache dense (no indices); sparsity not supported
    Prefill (S>1):
      - SM100+ and kv_lora_rank==512: flash_mla_sparse_fwd (flat ragged, no Q-padding)
      - Otherwise: flash_mla_with_kvcache(causal=True) + Q-padding
    """
    from tensorrt_llm.flash_mla.flash_mla_interface import (
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )

    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    index_head_dim = index_q.shape[-1]
    index_softmax_scale = 1.0 / math.sqrt(index_head_dim)

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len_active = seq_len[:num_seq]
    input_pos_active = input_pos[:num_seq]
    cu_seqlen_active = cu_seqlen[: num_seq + 1]

    # Extract MLA weight matrices
    w = kv_b_proj_weight.view(num_heads, kv_head_dim, kv_lora_rank)
    w_kn = w[:, :qk_nope_head_dim, :]  # [N, nope, lora]
    w_v = w[:, qk_nope_head_dim:, :]  # [N, v, lora]

    device = q_nope.device
    use_sparse = _is_sparse_mla_supported()

    # Flatten inputs to [total_tokens, ...] for cache-write helper
    total_tokens = int(cu_seqlen_active[-1].item())
    compressed_kv_flat = compressed_kv.reshape(total_tokens, kv_lora_rank)
    kpe_flat = kpe.reshape(total_tokens, 1, qk_rope_head_dim)
    index_k_flat = index_k.reshape(total_tokens, index_head_dim)

    # Write new tokens to paged caches
    _write_tokens_to_paged_caches(
        compressed_kv_flat,
        kpe_flat,
        index_k_flat,
        mla_cache,
        index_k_cache,
        seq_len_active,
        input_pos_active,
        cu_seqlen_active,
        cache_loc,
        cu_num_pages,
    )

    # cache_seqlens = total tokens in cache after write
    cache_seqlens = (input_pos_active + seq_len_active).to(torch.int32)

    if s == 1:
        # ---------------------------------------------------------------
        # DECODE path (all sequences, causal=False)
        # ---------------------------------------------------------------
        B = num_decode

        # Absorb Q: q_absorbed = einsum("bsnd,ndk->bsnk", q_nope, w_kn)
        q_nope_b = q_nope.reshape(B, 1, num_heads, qk_nope_head_dim)
        q_absorbed = torch.einsum("bsnd,ndk->bsnk", q_nope_b.float(), w_kn.float()).to(
            q_nope.dtype
        )  # [B, 1, N, lora]
        q_full = torch.cat([q_absorbed, q_pe.reshape(B, 1, num_heads, qk_rope_head_dim)], dim=-1)
        # [B, 1, N, lora+rope]

        block_table = _build_block_table(cache_loc, cu_num_pages, B, device)

        if use_sparse:
            # Sparse decode: compute Indexer top-k and pass to FlashMLA
            dense_index_k, valid_mask = _gather_index_k_dense(
                index_k_cache, cache_loc, cu_num_pages, cache_seqlens
            )
            index_q_b = index_q.reshape(B, 1, index_q.shape[-2], index_head_dim)
            index_weights_b = index_weights.reshape(B, 1, index_weights.shape[-1])
            sparse_indices = _compute_batched_index_topk(
                index_q_b,
                dense_index_k,
                index_weights_b,
                valid_mask,
                index_topk,
                index_softmax_scale,
                is_causal=False,
                cache_seqlens=cache_seqlens,
            )  # [B, 1, topk]

            tile_meta, num_splits = get_mla_metadata(
                cache_seqlens,
                1 * num_heads,
                num_heads_k=1,
                num_heads_q=num_heads,
                topk=index_topk,
            )
            out_latent, _ = flash_mla_with_kvcache(
                q_full,
                mla_cache,
                block_table,
                cache_seqlens,
                head_dim_v=kv_lora_rank,
                tile_scheduler_metadata=tile_meta,
                num_splits=num_splits,
                softmax_scale=scale,
                causal=False,
                indices=sparse_indices,
            )  # [B, 1, N, lora]
        else:
            # Dense decode fallback for SM90 (sparsity not available)
            tile_meta, num_splits = get_mla_metadata(
                cache_seqlens,
                1 * num_heads,
                num_heads_k=1,
                num_heads_q=num_heads,
                topk=None,
            )
            out_latent, _ = flash_mla_with_kvcache(
                q_full,
                mla_cache,
                block_table,
                cache_seqlens,
                head_dim_v=kv_lora_rank,
                tile_scheduler_metadata=tile_meta,
                num_splits=num_splits,
                softmax_scale=scale,
                causal=False,
            )  # [B, 1, N, lora]

        out_latent = out_latent.float()
        out = torch.einsum("bsnk,nvk->bsnv", out_latent, w_v.float()).to(q_nope.dtype)
        return out  # [B, 1, N, v_head_dim]

    else:
        # ---------------------------------------------------------------
        # PREFILL path
        # ---------------------------------------------------------------
        B = num_prefill

        if use_sparse and kv_lora_rank == 512:
            # Use flash_mla_sparse_fwd (flat ragged, no Q-padding)
            # Gather dense KV from paged cache
            dense_kv, kv_cu_seqlens = _gather_mla_kv_dense(
                mla_cache, cache_loc, cu_num_pages, cache_seqlens
            )  # dense_kv: [total_kv, 1, lora+rope]

            # Gather index_k for batched Indexer (padded)
            dense_index_k, valid_mask = _gather_index_k_dense(
                index_k_cache, cache_loc, cu_num_pages, cache_seqlens
            )  # [B, max_T, D_idx]

            # Pad query-side tensors to [B, max_S, ...] for batched Indexer
            max_S = int(seq_len_active.max().item())

            # Build padded index_q and index_weights [B, max_S, ...]
            index_q_flat = index_q.reshape(total_tokens, index_q.shape[-2], index_head_dim)
            index_w_flat = index_weights.reshape(total_tokens, index_weights.shape[-1])

            index_q_padded = index_q_flat.new_zeros(B, max_S, index_q.shape[-2], index_head_dim)
            index_w_padded = index_w_flat.new_zeros(B, max_S, index_weights.shape[-1])

            for i in range(B):
                sl = int(seq_len_active[i].item())
                ss = int(cu_seqlen_active[i].item())
                index_q_padded[i, :sl] = index_q_flat[ss : ss + sl]
                index_w_padded[i, :sl] = index_w_flat[ss : ss + sl]

            # Batched Indexer top-k [B, max_S, topk]
            local_indices = _compute_batched_index_topk(
                index_q_padded,
                dense_index_k,
                index_w_padded,
                valid_mask,
                index_topk,
                index_softmax_scale,
                is_causal=True,
                cache_seqlens=cache_seqlens,
            )  # [B, max_S, topk]

            # Convert local KV indices to global flat KV indices
            total_kv = int(dense_kv.shape[0])

            # q_seq_ids[t] = batch index, q_local_pos[t] = within-seq query pos
            q_seq_ids = torch.repeat_interleave(
                torch.arange(B, device=device), seq_len_active
            )  # [total_tokens]
            q_local_pos = (
                torch.arange(total_tokens, device=device) - cu_seqlen_active[:-1][q_seq_ids]
            )  # [total_tokens]

            # global_indices[t, :] = kv_cu_seqlens[b] + local_indices[b, s, :]
            global_indices = local_indices[q_seq_ids, q_local_pos, :]  # [total_tokens, topk]
            global_indices = global_indices.long() + kv_cu_seqlens[q_seq_ids].unsqueeze(1)
            # Mark out-of-range as -1 (flash_mla_sparse_fwd treats -1 as invalid)
            global_indices[global_indices >= total_kv] = -1
            indices_flat = global_indices.unsqueeze(1).to(torch.int32)  # [total_tokens, 1, topk]

            # Absorb Q: q_absorbed = einsum("tnd,ndk->tnk", q_nope_flat, w_kn)
            q_nope_flat = q_nope.reshape(total_tokens, num_heads, qk_nope_head_dim)
            q_absorbed = torch.einsum("tnd,ndk->tnk", q_nope_flat.float(), w_kn.float()).to(
                q_nope.dtype
            )  # [total_tokens, N, lora]
            q_pe_flat = q_pe.reshape(total_tokens, num_heads, qk_rope_head_dim)
            q_flat = torch.cat([q_absorbed, q_pe_flat], dim=-1)  # [total_tokens, N, lora+rope]

            # flash_mla_sparse_fwd expects bfloat16
            compute_dtype = torch.bfloat16
            q_fwd = q_flat.to(compute_dtype)
            kv_fwd = dense_kv.squeeze(1).to(compute_dtype)  # [total_kv, lora+rope]
            # flash_mla_sparse_fwd: q=[s_q,h_q,d_qk], kv=[s_kv,h_kv,d_qk]
            # h_kv=1 for MLA; indices=[s_q, h_kv, topk]

            out_latent_flat, _, _ = flash_mla_sparse_fwd(
                q_fwd,
                kv_fwd.unsqueeze(1),  # [total_kv, 1, lora+rope] as expected
                indices_flat,
                scale,
                d_v=kv_lora_rank,
            )  # [total_tokens, N, lora]

            out_latent_flat = out_latent_flat.float()
            out_flat = torch.einsum("tnk,nvk->tnv", out_latent_flat, w_v.float()).to(
                q_nope.dtype
            )  # [total_tokens, N, v_head_dim]

            return out_flat.reshape(b, s, num_heads, v_head_dim)

        else:
            # Fallback: flash_mla_with_kvcache(causal=True) with Q-padding
            # Used on SM90 or when kv_lora_rank != 512
            max_S = int(seq_len_active.max().item())

            # Build padded queries [B, max_S, N, lora+rope]
            q_nope_flat = q_nope.reshape(total_tokens, num_heads, qk_nope_head_dim)
            q_pe_flat_r = q_pe.reshape(total_tokens, num_heads, qk_rope_head_dim)

            q_absorbed_flat = torch.einsum("tnd,ndk->tnk", q_nope_flat.float(), w_kn.float()).to(
                q_nope.dtype
            )
            q_full_flat = torch.cat(
                [q_absorbed_flat, q_pe_flat_r], dim=-1
            )  # [total_tokens, N, lora+rope]

            # Pad to [B, max_S, N, lora+rope]
            q_padded = q_full_flat.new_zeros(B, max_S, num_heads, kv_lora_rank + qk_rope_head_dim)
            for i in range(B):
                sl = int(seq_len_active[i].item())
                ss = int(cu_seqlen_active[i].item())
                q_padded[i, :sl] = q_full_flat[ss : ss + sl]

            block_table = _build_block_table(cache_loc, cu_num_pages, B, device)

            tile_meta, num_splits = get_mla_metadata(
                cache_seqlens,
                max_S * num_heads,
                num_heads_k=1,
                num_heads_q=num_heads,
                topk=None,
            )

            out_padded, _ = flash_mla_with_kvcache(
                q_padded,
                mla_cache,
                block_table,
                cache_seqlens,
                head_dim_v=kv_lora_rank,
                tile_scheduler_metadata=tile_meta,
                num_splits=num_splits,
                softmax_scale=scale,
                causal=True,
            )  # [B, max_S, N, lora]

            # Unpad and project
            out_flat_list = []
            for i in range(B):
                sl = int(seq_len_active[i].item())
                out_i = out_padded[i, :sl]  # [sl, N, lora]
                out_flat_list.append(out_i)
            out_latent_flat = torch.cat(out_flat_list, dim=0).float()  # [total_tokens, N, lora]
            out_flat = torch.einsum("tnk,nvk->tnv", out_latent_flat, w_v.float()).to(q_nope.dtype)

            return out_flat.reshape(b, s, num_heads, v_head_dim)


@flash_mla_dsa_with_cache.register_fake
def _(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_pages: torch.Tensor,
    last_page_len: torch.Tensor,
    mla_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
    index_topk: int = 64,
) -> torch.Tensor:
    """Fake impl for torch.export / graph tracing."""
    B = q_nope.shape[0]
    S = q_nope.shape[1]
    N = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // N
    v_head_dim = kv_head_dim - qk_nope_head_dim
    return q_nope.new_empty(B, S, N, v_head_dim)


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


@AttentionRegistry.register("flashmla_dsa")
class FlashMLADSAAttention(AttentionDescriptor):
    """Attention descriptor for FlashMLA-based DSA with paged caches.

    Source op: torch_dsa (same as TorchBackendDSAAttention)
    Cached op: flash_mla_dsa_with_cache

    Cache layout (paged via MLAPagedResourceHandler):
        mla_cache:     [num_blocks, page_size, 1, kv_lora_rank + qk_rope_head_dim]
        index_k_cache: [num_blocks, page_size, index_head_dim]
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight,
        # index_q, index_k, index_weights
        return 8

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_dsa

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flash_mla_dsa_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return [
            "batch_info_host",
            "seq_len",
            "input_pos",
            "cu_seqlen",
            "cache_loc",
            "cu_num_pages",
            "last_page_len",
        ]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Initialize paged mla_cache and index_k_cache."""
        # torch_dsa args: q_nope[0], q_pe[1], compressed_kv[2], kpe[3],
        #                 kv_b_proj_weight[4], index_q[5], index_k[6], index_weights[7]
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]
        index_q_fake = source_attn_node.args[5].meta["val"]

        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]
        index_head_dim = index_q_fake.shape[-1]

        model_dtype = compressed_kv_fake.dtype
        cache_dtype = cls.resolve_cache_dtype(cache_config.dtype, model_dtype)

        return {
            "mla_cache": MLAPagedResourceHandler(
                1,
                kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
            ),
            "index_k_cache": MLAPagedResourceHandler(
                index_head_dim,
                dtype=cache_dtype,
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Return [scale, kv_lora_rank, index_topk] constants."""
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kv_lora_rank = compressed_kv_fake.shape[-1]
        scale = source_attn_node.kwargs.get("scale", None)
        index_topk = source_attn_node.kwargs.get("index_topk", 64)
        return [scale, kv_lora_rank, index_topk]
