# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Torch backend for DeepSeek Sparse Attention (DSA) with KV cache.

Provides:
- torch_cached_dsa_with_cache: cached DSA op managing two caches:
    mla_cache:     [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]  (same as MLA)
    index_k_cache: [max_batch, max_seq, index_head_dim]                   (Indexer keys)
- TorchBackendDSAAttention: AttentionDescriptor registered as "torch_dsa"

Cache layout:
    mla_cache:      identical to FlashInfer MLA cache layout
    index_k_cache:  [max_batch, max_seq, index_head_dim]

Prefill:  expand compressed_kv -> full K/V, compute index scores vs cached index_k
Generate: weight absorption for MLA part + index sparse mask from cached index_k
"""

import math
from typing import List, Optional

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
    ResourceHandlerDict,
    UnpagedResourceHandler,
)
from .torch_backend_mla import _update_mla_cache


def _update_index_k_cache(
    index_k: torch.Tensor,  # [total_tokens, index_head_dim]
    index_k_cache: torch.Tensor,  # [max_batch, max_seq, index_head_dim]
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_start: torch.Tensor,
) -> None:
    """Update the Indexer key cache with new token keys."""
    cache_dtype = index_k_cache.dtype
    if index_k.dtype != cache_dtype:
        index_k = index_k.to(cache_dtype)

    for idx in range(seq_len.shape[0]):
        start = seq_start[idx].item()
        length = seq_len[idx].item()
        cache_idx = slot_idx[idx].item()
        pos = input_pos[idx].item()
        index_k_cache[cache_idx, pos : pos + length] = index_k[start : start + length]


def _compute_seq_index_score(
    index_q_seq: torch.Tensor,  # [S_q, H, D_idx]
    index_k_cached: torch.Tensor,  # [T, D_idx]
    index_weights_seq: torch.Tensor,  # [S_q, H]
    index_softmax_scale: float,
    index_topk: int,
    causal_diagonal: int,
) -> torch.Tensor:
    """Compute DSA index mask for a single sequence.

    Returns index_mask [S_q, T] with 0.0 at top-k positions and -inf elsewhere.
    causal_diagonal: passed to torch.triu to enforce causality (use kv_seq_len - seq_len + 1).
    """
    s_q = index_q_seq.shape[0]
    t = index_k_cached.shape[0]

    # per_head_scores: [S_q, H, T]
    per_head_scores = torch.einsum(
        "shd,td->sht",
        index_q_seq.float(),
        index_k_cached.float(),
    )

    # weighted sum over heads: [S_q, T]
    index_score = torch.einsum("sht,sh->st", per_head_scores, index_weights_seq.float())
    index_score = index_score * index_softmax_scale

    # Causal mask: upper-triangular positions get -inf
    if causal_diagonal <= t:
        causal_mask = torch.triu(
            torch.ones(s_q, t, device=index_q_seq.device, dtype=torch.bool),
            diagonal=causal_diagonal,
        )
        index_score.masked_fill_(causal_mask, float("-inf"))

    effective_topk = min(index_topk, t)
    topk_indices = index_score.topk(effective_topk, dim=-1).indices  # [S_q, topk]

    index_mask = index_score.new_full(index_score.shape, float("-inf"))
    index_mask.scatter_(-1, topk_indices, 0.0)  # [S_q, T]
    return index_mask


def _torch_dsa_generate_with_absorption(
    q_nope: torch.Tensor,  # [B, 1, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, 1, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, 1, kv_lora_rank]
    kpe: torch.Tensor,  # [B, 1, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N*(qk_nope+v), kv_lora_rank]
    index_q: torch.Tensor,  # [B, 1, H, D_idx]
    index_k: torch.Tensor,  # [B, 1, D_idx]
    index_weights: torch.Tensor,  # [B, 1, H]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    index_k_cache: torch.Tensor,  # [max_batch, max_seq, index_head_dim]
    slot_idx: torch.Tensor,
    input_pos: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    index_topk: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,
) -> None:
    """Generate-phase DSA with MLA weight absorption + sparse index masking."""
    b = q_nope.shape[0]
    index_head_dim = index_q.shape[-1]
    index_softmax_scale = 1.0 / math.sqrt(index_head_dim)

    # Extract MLA weight components
    weight_reshaped = kv_b_proj_weight.view(num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
    w_k_nope = weight_reshaped[:, :qk_nope_head_dim, :]  # [N, qk_nope_head_dim, kv_lora_rank]
    w_v = weight_reshaped[:, qk_nope_head_dim:, :]  # [N, v_head_dim, kv_lora_rank]

    # Update both caches with current tokens
    compressed_kv_flat = compressed_kv.squeeze(1)  # [B, kv_lora_rank]
    kpe_flat = kpe.squeeze(1).squeeze(1)  # [B, qk_rope_head_dim]
    index_k_flat = index_k.squeeze(1)  # [B, D_idx]

    cache_dtype = mla_cache.dtype
    idx_cache_dtype = index_k_cache.dtype

    for i in range(b):
        cache_idx = slot_idx[i].item()
        pos = input_pos[i].item()

        ckv = compressed_kv_flat[i]
        kpe_i = kpe_flat[i]
        if ckv.dtype != cache_dtype:
            ckv = ckv.to(cache_dtype)
        if kpe_i.dtype != cache_dtype:
            kpe_i = kpe_i.to(cache_dtype)
        mla_cache[cache_idx, pos, :kv_lora_rank] = ckv
        mla_cache[cache_idx, pos, kv_lora_rank:] = kpe_i

        ik = index_k_flat[i]
        if ik.dtype != idx_cache_dtype:
            ik = ik.to(idx_cache_dtype)
        index_k_cache[cache_idx, pos] = ik

    # Compute attention per sequence
    compute_dtype = q_nope.dtype
    for i in range(b):
        cache_idx = slot_idx[i].item()
        pos = input_pos[i].item()

        q_nope_i = q_nope[i, 0]  # [N, qk_nope_head_dim]
        q_pe_i = q_pe[i, 0]  # [N, qk_rope_head_dim]

        # MLA cached data
        cached_data = mla_cache[cache_idx, : pos + 1]  # [T, kv_lora_rank + qk_rope_head_dim]
        compressed_kv_cached = cached_data[:, :kv_lora_rank]
        kpe_cached = cached_data[:, kv_lora_rank:]
        if compressed_kv_cached.dtype != compute_dtype:
            compressed_kv_cached = compressed_kv_cached.to(compute_dtype)
        if kpe_cached.dtype != compute_dtype:
            kpe_cached = kpe_cached.to(compute_dtype)

        # Index key cache
        index_k_cached = index_k_cache[cache_idx, : pos + 1]  # [T, D_idx]
        if index_k_cached.dtype != compute_dtype:
            index_k_cached = index_k_cached.to(compute_dtype)

        # --- DSA index mask ---
        index_q_i = index_q[i, 0]  # [H, D_idx]
        index_w_i = index_weights[i, 0]  # [H]
        index_mask = _compute_seq_index_score(
            index_q_i.unsqueeze(0),
            index_k_cached,
            index_w_i.unsqueeze(0),
            index_softmax_scale,
            index_topk,
            causal_diagonal=pos + 2,  # all positions valid (pos+1 is current, no future)
        )  # [1, T]
        index_mask = index_mask.squeeze(0)  # [T]

        # --- MLA weight absorption ---
        # q_absorbed: [N, kv_lora_rank]
        q_absorbed = torch.einsum("nd,ndk->nk", q_nope_i, w_k_nope)

        scores_nope = torch.matmul(q_absorbed.float(), compressed_kv_cached.float().t())  # [N, T]
        scores_pe = torch.matmul(q_pe_i.float(), kpe_cached.float().t())  # [N, T]
        attn_scores = (scores_nope + scores_pe) * scale  # [N, T]

        # Add DSA sparse mask (broadcast [T] -> [N, T])
        attn_scores = attn_scores + index_mask.unsqueeze(0)

        attn_weights = torch.softmax(attn_scores, dim=-1).to(compute_dtype)  # [N, T]

        # Weighted sum over compressed_kv, then project to v_head_dim
        weighted_kv = torch.matmul(attn_weights, compressed_kv_cached)  # [N, kv_lora_rank]
        attn_out = torch.einsum("nk,nvk->nv", weighted_kv, w_v)  # [N, v_head_dim]

        out[i] = attn_out


def _torch_dsa_context_with_expansion(
    q_nope: torch.Tensor,  # [total_tokens, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [total_tokens, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [total_tokens, kv_lora_rank]
    kpe: torch.Tensor,  # [total_tokens, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N*(qk_nope+v), kv_lora_rank]
    index_q: torch.Tensor,  # [total_tokens, H, D_idx]
    index_k: torch.Tensor,  # [total_tokens, D_idx]
    index_weights: torch.Tensor,  # [total_tokens, H]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    index_k_cache: torch.Tensor,  # [max_batch, max_seq, index_head_dim]
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    scale: float,
    kv_lora_rank: int,
    index_topk: int,
    num_heads: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    out: torch.Tensor,
) -> None:
    """Context-phase DSA: kv_b_proj expansion + sparse index masking."""
    index_head_dim = index_q.shape[-1]
    index_softmax_scale = 1.0 / math.sqrt(index_head_dim)

    kpe_flat = kpe.squeeze(1)  # [total_tokens, qk_rope_head_dim]

    # Update MLA cache
    _update_mla_cache(
        compressed_kv,
        kpe_flat,
        mla_cache,
        seq_len,
        input_pos,
        slot_idx,
        seq_start,
        kv_lora_rank,
    )

    # Update index_k cache
    _update_index_k_cache(
        index_k,
        index_k_cache,
        seq_len,
        input_pos,
        slot_idx,
        seq_start,
    )

    compute_dtype = q_nope.dtype
    attn_outputs = []

    for idx in range(seq_len.shape[0]):
        seq_len_i = seq_len[idx].item()
        input_pos_i = input_pos[idx].item()
        slot_idx_i = slot_idx[idx].item()
        seq_start_i = seq_start[idx].item()

        if seq_len_i == 0:
            continue

        kv_seq_len = input_pos_i + seq_len_i

        # Gather query tokens for this sequence
        q_nope_seq = q_nope[seq_start_i : seq_start_i + seq_len_i]  # [S, N, nope]
        q_pe_seq = q_pe[seq_start_i : seq_start_i + seq_len_i]  # [S, N, rope]

        # Get cached MLA data
        cached_data = mla_cache[slot_idx_i, :kv_seq_len]  # [T, kv_lora_rank + qk_rope_head_dim]
        compressed_kv_cached = cached_data[:, :kv_lora_rank]
        kpe_cached = cached_data[:, kv_lora_rank:]
        if compressed_kv_cached.dtype != compute_dtype:
            compressed_kv_cached = compressed_kv_cached.to(compute_dtype)
        if kpe_cached.dtype != compute_dtype:
            kpe_cached = kpe_cached.to(compute_dtype)

        # Get cached index keys
        index_k_cached = index_k_cache[slot_idx_i, :kv_seq_len]  # [T, D_idx]
        if index_k_cached.dtype != compute_dtype:
            index_k_cached = index_k_cached.to(compute_dtype)

        # --- DSA index mask ---
        index_q_seq = index_q[seq_start_i : seq_start_i + seq_len_i]  # [S, H, D_idx]
        index_w_seq = index_weights[seq_start_i : seq_start_i + seq_len_i]  # [S, H]
        index_mask = _compute_seq_index_score(
            index_q_seq,
            index_k_cached,
            index_w_seq,
            index_softmax_scale,
            index_topk,
            causal_diagonal=kv_seq_len - seq_len_i + 1,
        )  # [S, T]

        # --- Expand compressed_kv and compute attention ---
        kv_expanded = torch.matmul(compressed_kv_cached, kv_b_proj_weight.t())
        kv_expanded = kv_expanded.view(kv_seq_len, num_heads, qk_nope_head_dim + v_head_dim)
        k_nope_expanded = kv_expanded[:, :, :qk_nope_head_dim]  # [T, N, nope]
        v_expanded = kv_expanded[:, :, qk_nope_head_dim:]  # [T, N, v]

        kpe_expanded = kpe_cached.unsqueeze(1).expand(-1, num_heads, -1)  # [T, N, rope]

        query_full = torch.cat([q_nope_seq, q_pe_seq], dim=-1)  # [S, N, qk_head_dim]
        key_full = torch.cat([k_nope_expanded, kpe_expanded], dim=-1)  # [T, N, qk_head_dim]

        # Transpose to [1, N, S/T, D] for batched matmul
        query_t = query_full.transpose(0, 1).unsqueeze(0)  # [1, N, S, D]
        key_t = key_full.transpose(0, 1).unsqueeze(0)  # [1, N, T, D]

        attn_scores = (
            torch.matmul(query_t.float(), key_t.float().transpose(-2, -1)) * scale
        )  # [1, N, S, T] in fp32

        # Causal mask (upper-triangular relative to kv_seq_len)
        causal_mask = torch.triu(
            torch.ones(seq_len_i, kv_seq_len, device=q_nope.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len_i + 1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # DSA sparse mask: [S, T] -> [1, 1, S, T]
        attn_scores = attn_scores + index_mask.unsqueeze(0).unsqueeze(0)

        attn_weights = torch.softmax(attn_scores, dim=-1).to(compute_dtype)  # [1, N, S, T]

        v_t = v_expanded.transpose(0, 1).unsqueeze(0)  # [1, N, T, v]
        attn_out = torch.matmul(attn_weights, v_t)  # [1, N, S, v]
        attn_out = attn_out[0].transpose(0, 1)  # [S, N, v]

        attn_outputs.append(attn_out)

    if len(attn_outputs) == 0:
        out.zero_()
    elif len(attn_outputs) == 1:
        out.copy_(attn_outputs[0])
    else:
        out.copy_(torch.cat(attn_outputs, dim=0))


@torch.library.custom_op("auto_deploy::torch_cached_dsa_with_cache", mutates_args=())
def torch_backend_dsa_with_cache(
    # 8 tensor args (get_num_qkv_args = 8)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim]
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim]
    kv_b_proj_weight: torch.Tensor,  # [N*(qk_nope+v), kv_lora_rank]
    index_q: torch.Tensor,  # [B, S, H, D_idx]
    index_k: torch.Tensor,  # [B, S, D_idx]
    index_weights: torch.Tensor,  # [B, S, H]
    # Standard metadata
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # Caches
    mla_cache: torch.Tensor,  # [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
    index_k_cache: torch.Tensor,  # [max_batch, max_seq, index_head_dim]
    # Constants
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
    index_topk: int = 64,
) -> torch.Tensor:
    """Torch backend DSA with KV cache and Indexer key cache.

    Prefill:  expand compressed_kv, compute index mask vs cached index_k
    Generate: weight absorption for MLA + index sparse mask
    """
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    input_pos = input_pos[:num_seq]
    slot_idx = slot_idx[:num_seq]
    seq_start = cu_seqlen[:num_seq]

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    output_shape = (b, s, num_heads, v_head_dim)

    if s == 1:
        # Generate phase
        y = q_nope.new_empty(b, num_heads, v_head_dim).contiguous()
        _torch_dsa_generate_with_absorption(
            q_nope,
            q_pe,
            compressed_kv,
            kpe,
            kv_b_proj_weight,
            index_q,
            index_k,
            index_weights,
            mla_cache,
            index_k_cache,
            slot_idx,
            input_pos,
            scale,
            kv_lora_rank,
            index_topk,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )
        return y.unsqueeze(1)  # [B, 1, N, v_head_dim]
    else:
        # Prefill / context phase
        bs_view = (b * s,)
        q_nope_flat = q_nope.contiguous().view(*bs_view, num_heads, qk_nope_head_dim)
        q_pe_flat = q_pe.contiguous().view(*bs_view, num_heads, qk_rope_head_dim)
        compressed_kv_flat = compressed_kv.contiguous().view(*bs_view, kv_lora_rank)
        kpe_flat = kpe.contiguous().view(*bs_view, 1, qk_rope_head_dim)

        index_n_heads = index_q.shape[2]
        index_head_dim = index_q.shape[3]
        index_q_flat = index_q.contiguous().view(*bs_view, index_n_heads, index_head_dim)
        index_k_flat = index_k.contiguous().view(*bs_view, index_head_dim)
        index_weights_flat = index_weights.contiguous().view(*bs_view, index_n_heads)

        y = q_nope.new_empty(*bs_view, num_heads, v_head_dim).contiguous()
        _torch_dsa_context_with_expansion(
            q_nope_flat,
            q_pe_flat,
            compressed_kv_flat,
            kpe_flat,
            kv_b_proj_weight,
            index_q_flat,
            index_k_flat,
            index_weights_flat,
            mla_cache,
            index_k_cache,
            input_pos,
            slot_idx,
            seq_len,
            seq_start,
            scale,
            kv_lora_rank,
            index_topk,
            num_heads,
            qk_nope_head_dim,
            v_head_dim,
            y,
        )
        return y.view(*output_shape)


@torch_backend_dsa_with_cache.register_fake
def torch_backend_dsa_with_cache_fake(
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
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    mla_cache: torch.Tensor,
    index_k_cache: torch.Tensor,
    scale: Optional[float] = None,
    kv_lora_rank: int = 512,
    index_topk: int = 64,
) -> torch.Tensor:
    """Fake implementation for torch_backend_dsa_with_cache."""
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[-1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim
    return q_nope.new_empty(
        q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
    ).contiguous()


@AttentionRegistry.register("torch_dsa")
class TorchBackendDSAAttention(AttentionDescriptor):
    """Attention descriptor for DeepSeek Sparse Attention (DSA).

    Uses torch_dsa as the source op and torch_cached_dsa_with_cache as
    the cached op, managing two caches:
      - mla_cache:     identical layout to MLA cache
      - index_k_cache: Indexer key cache [max_batch, max_seq, index_head_dim]
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
        return torch.ops.auto_deploy.torch_cached_dsa_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Initialize mla_cache and index_k_cache."""
        # torch_dsa args: q_nope[0], q_pe[1], compressed_kv[2], kpe[3], kv_b_proj_weight[4],
        #                 index_q[5], index_k[6], index_weights[7]
        compressed_kv_fake = source_attn_node.args[2].meta["val"]
        kpe_fake = source_attn_node.args[3].meta["val"]
        index_q_fake = source_attn_node.args[5].meta["val"]

        kv_lora_rank = compressed_kv_fake.shape[-1]
        qk_rope_head_dim = kpe_fake.shape[-1]
        index_head_dim = index_q_fake.shape[-1]

        model_dtype = compressed_kv_fake.dtype
        cache_dtype = cls.resolve_cache_dtype(cache_config.dtype, model_dtype)

        return {
            "mla_cache": UnpagedResourceHandler(
                kv_lora_rank + qk_rope_head_dim,
                dtype=cache_dtype,
            ),
            "index_k_cache": UnpagedResourceHandler(
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
