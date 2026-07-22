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

"""Unit tests for the DeepSeek V4 sparse-attention custom ops and cache transform."""

from __future__ import annotations

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import Dim
from torch.fx import Graph

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (
    deepseek_v4_sparse_attention as dsv4_sparse,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    DeepSeekV4SparseAttention,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import (
    BatchInfo,
    PagedResourceHandler,
    SequenceInfo,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4Indexer,
)
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
    InsertCachedAttentionConfig,
    InsertCachedDeepSeekV4SparseAttention,
    _InsertCachedOperator,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="compressed sparse-attention paths require CUDA (Triton fused ops)",
)


def _page_meta(
    seq_lens: list[int],
    input_positions: list[int],
    slot_indices: list[int],
    tokens_per_block: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_num_pages = [0]
    cache_loc = []
    next_page_id = 0
    for seq_len, input_pos, slot_idx in zip(seq_lens, input_positions, slot_indices, strict=True):
        total_len = input_pos + seq_len
        if tokens_per_block is None:
            seq_pages = [slot_idx]
        else:
            num_pages = max((max(total_len, 1) + tokens_per_block - 1) // tokens_per_block, 1)
            seq_pages = list(range(next_page_id, next_page_id + num_pages))
            next_page_id += num_pages
        cache_loc.extend(seq_pages)
        cu_num_pages.append(cu_num_pages[-1] + len(seq_pages))

    return (
        torch.tensor(cu_num_pages, dtype=torch.int32),
        torch.tensor(cache_loc, dtype=torch.int32),
    )


def _context_meta(seq_len: int, tokens_per_block: int | None = None, input_pos: int = 0):
    batch_info_host = BatchInfo()
    batch_info_host.update([1, seq_len, 0, 0, 0, 0])
    cu_num_pages, cache_loc = _page_meta([seq_len], [input_pos], [0], tokens_per_block)
    return (
        batch_info_host.serialize(),
        torch.tensor([seq_len], dtype=torch.int32),
        torch.tensor([input_pos], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, seq_len], dtype=torch.int32),
        cu_num_pages,
        cache_loc,
    )


def _multi_context_meta(seq_lens: list[int], tokens_per_block: int | None = None):
    total_tokens = sum(seq_lens)
    cu_seqlen = [0]
    for seq_len in seq_lens:
        cu_seqlen.append(cu_seqlen[-1] + seq_len)

    batch_info_host = BatchInfo()
    batch_info_host.update([len(seq_lens), total_tokens, 0, 0, 0, 0])
    slot_indices = list(range(len(seq_lens)))
    cu_num_pages, cache_loc = _page_meta(
        seq_lens, [0] * len(seq_lens), slot_indices, tokens_per_block
    )
    return (
        batch_info_host.serialize(),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.zeros(len(seq_lens), dtype=torch.int32),
        torch.tensor(slot_indices, dtype=torch.int64),
        torch.tensor(cu_seqlen, dtype=torch.int32),
        cu_num_pages,
        cache_loc,
    )


def _decode_meta(input_pos: int, tokens_per_block: int | None = None):
    batch_info_host = BatchInfo()
    batch_info_host.update([0, 0, 0, 0, 1, 1])
    cu_num_pages, cache_loc = _page_meta([1], [input_pos], [0], tokens_per_block)
    return (
        batch_info_host.serialize(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([input_pos], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int32),
        cu_num_pages,
        cache_loc,
    )


def _multi_decode_meta(input_positions: list[int], tokens_per_block: int | None = None):
    seq_lens = [1] * len(input_positions)
    cu_seqlen = list(range(len(input_positions) + 1))
    batch_info_host = BatchInfo()
    batch_info_host.update([0, 0, 0, 0, len(input_positions), len(input_positions)])
    slot_indices = list(range(len(input_positions)))
    cu_num_pages, cache_loc = _page_meta(seq_lens, input_positions, slot_indices, tokens_per_block)
    return (
        batch_info_host.serialize(),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(input_positions, dtype=torch.int32),
        torch.tensor(slot_indices, dtype=torch.int64),
        torch.tensor(cu_seqlen, dtype=torch.int32),
        cu_num_pages,
        cache_loc,
    )


def _cuda_decode_meta(input_pos: int, slot_idx: int = 0, tokens_per_block: int | None = None):
    (
        batch_info_host,
        seq_len,
        input_pos_tensor,
        slot_idx_tensor,
        cu_seqlen,
        cu_num_pages,
        cache_loc,
    ) = _decode_meta(input_pos, tokens_per_block=tokens_per_block)
    input_pos_tensor.fill_(input_pos)
    slot_idx_tensor.fill_(slot_idx)
    if tokens_per_block is None:
        cache_loc.fill_(slot_idx)
    return (
        batch_info_host,
        seq_len.cuda(),
        input_pos_tensor.cuda(),
        slot_idx_tensor.cuda(),
        cu_seqlen.cuda(),
        cu_num_pages.cuda(),
        cache_loc.cuda(),
    )


def _standard_metadata(base_meta: tuple[torch.Tensor, ...], device: torch.device):
    """The 10 standard metadata tensors (device tensors + host mirrors) of the cached op."""
    batch_info_host, seq_len, input_pos, slot_idx, cu_seqlen, cu_num_pages, cache_loc = base_meta
    return (
        batch_info_host,
        input_pos.to(device),
        slot_idx.to(device),
        cu_num_pages.to(device),
        cache_loc.to(device),
        seq_len.cpu(),
        input_pos.cpu(),
        cu_seqlen.cpu(),
        cu_num_pages.cpu(),
        cache_loc.cpu(),
    )


def _prepare_extra_metadata(
    base_meta: tuple[torch.Tensor, ...],
    swa_cache: torch.Tensor,
    *,
    window_size: int | None = None,
    compress_ratio: int = 0,
    max_compressed_len: int | None = None,
    position_ids: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The 23 hoisted metadata tensors, produced via the production prepare op."""
    _, _, input_pos, _, _, cu_num_pages, cache_loc = base_meta
    device = swa_cache.device
    input_pos = input_pos.to(device)
    cu_num_pages = cu_num_pages.to(device).contiguous()
    cache_loc = cache_loc.to(device).contiguous()
    position_ids = input_pos if position_ids is None else position_ids.to(device)
    overlap_m = max_compressed_len if compress_ratio == 4 else None
    dense_m = max_compressed_len if compress_ratio == 128 else None
    return torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr(
        input_pos,
        position_ids,
        cu_num_pages,
        cache_loc,
        int(swa_cache.shape[1]),
        max(int(overlap_m or 1), 1),
        max(int(dense_m or 1), 1),
        max(int(window_size or 1), 1),
    )


def _op_metadata(
    base_meta: tuple[torch.Tensor, ...],
    swa_cache: torch.Tensor,
    *,
    window_size: int | None = None,
    compress_ratio: int = 0,
    max_compressed_len: int | None = None,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Full 33-tensor metadata bundle (10 standard + 23 hoisted) for the cached op."""
    extra = _prepare_extra_metadata(
        base_meta,
        swa_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        position_ids=position_ids,
    )
    return (*_standard_metadata(base_meta, swa_cache.device), *extra)


def _sparse_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """fp32 reference of the documented semantics (sink, masking of negative/oor, duplicates)."""
    batch_size, seq_len, num_heads, _ = q.shape
    kv_rows = kv.shape[1]
    batch_idx = torch.arange(batch_size, device=q.device).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, seq_len, topk_idxs.shape[-1])

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    valid = (topk_idxs >= 0) & (topk_idxs < kv_rows)
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0, max=max(kv_rows - 1, 0))
    selected_kv = kv[batch_idx, gather_idxs].to(compute_dtype)
    logits = torch.matmul(q.to(compute_dtype), selected_kv.transpose(-1, -2))
    logits = logits * softmax_scale
    logits = logits.masked_fill((~valid).unsqueeze(2), float("-inf"))

    sink_logits = attn_sink.to(dtype=compute_dtype).view(1, 1, num_heads, 1)
    sink_logits = sink_logits.expand(batch_size, seq_len, num_heads, 1)
    weights = torch.softmax(torch.cat([logits, sink_logits], dim=-1), dim=-1)
    output = torch.matmul(weights[..., :-1], selected_kv)
    return output.to(q.dtype)


def _empty_sparse_attention_tensors(q: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, ...]:
    del kv
    return (
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(0, 0),
        q.new_empty(0),
        q.new_empty(0, 0),
        q.new_empty(0, 0),
        q.new_empty(q.shape[0], q.shape[1]),
        q.new_empty(q.shape[0], q.shape[1], 0, 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(q.shape[0], q.shape[1], 0),
        q.new_empty(0, 0),
        q.new_empty(0),
    )


def _run_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int | None = None,
    topk_is_placeholder: bool = False,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        *_empty_sparse_attention_tensors(q, kv),
        softmax_scale,
        window_size=window_size,
        compress_ratio=0,
        topk_is_placeholder=topk_is_placeholder,
    )


def _run_cached_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    metadata: tuple[torch.Tensor, ...],
    swa_cache: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int | None = None,
    compress_ratio: int = 0,
    max_compressed_len: int | None = None,
    rope_dim: int | None = None,
    mhc_cache: torch.Tensor | None = None,
    topk_is_placeholder: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if mhc_cache is None:
        mhc_cache = swa_cache.new_empty(swa_cache.shape)
    compressor_kv_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    compressor_gate_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    indexer_compressor_kv_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    indexer_compressor_gate_cache = q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    full_metadata = _op_metadata(
        metadata,
        swa_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
    )
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        *_empty_sparse_attention_tensors(q, kv),
        *full_metadata,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        1e-6,
        rope_dim,
        topk_is_placeholder,
        out=out,
    )


def _run_sparse_attention_with_compressor(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor: DeepseekV4Compressor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int | None = None,
    compress_ratio: int = 0,
    indexer_q: torch.Tensor | None = None,
    indexer_weights: torch.Tensor | None = None,
    indexer_compressor_kv: torch.Tensor | None = None,
    indexer_compressor_gate: torch.Tensor | None = None,
    indexer_compressor_ape: torch.Tensor | None = None,
    indexer_compressor_norm_weight: torch.Tensor | None = None,
    topk_is_placeholder: bool = False,
) -> torch.Tensor:
    indexer_q = indexer_q if indexer_q is not None else q.new_empty(q.shape[0], q.shape[1], 0, 0)
    indexer_weights = (
        indexer_weights if indexer_weights is not None else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_kv = (
        indexer_compressor_kv
        if indexer_compressor_kv is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_gate = (
        indexer_compressor_gate
        if indexer_compressor_gate is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_ape = (
        indexer_compressor_ape if indexer_compressor_ape is not None else q.new_empty(0, 0)
    )
    indexer_compressor_norm_weight = (
        indexer_compressor_norm_weight
        if indexer_compressor_norm_weight is not None
        else q.new_empty(0)
    )
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor.ape,
        compressor.norm.weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        softmax_scale,
        False,
        "mha_sparse",
        0,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        kv.shape[-1],
        compressor.rope_head_dim,
        compressor.norm.eps,
        topk_is_placeholder,
    )


def _run_cached_sparse_attention_with_compressor(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor: DeepseekV4Compressor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: tuple[torch.Tensor, ...],
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    softmax_scale: float = 1.0,
    window_size: int = 4,
    compress_ratio: int = 4,
    indexer_q: torch.Tensor | None = None,
    indexer_weights: torch.Tensor | None = None,
    indexer_compressor_kv: torch.Tensor | None = None,
    indexer_compressor_gate: torch.Tensor | None = None,
    indexer_compressor_ape: torch.Tensor | None = None,
    indexer_compressor_norm_weight: torch.Tensor | None = None,
    indexer_compressor_kv_cache: torch.Tensor | None = None,
    indexer_compressor_gate_cache: torch.Tensor | None = None,
    topk_is_placeholder: bool = False,
) -> torch.Tensor:
    indexer_q = indexer_q if indexer_q is not None else q.new_empty(q.shape[0], q.shape[1], 0, 0)
    indexer_weights = (
        indexer_weights if indexer_weights is not None else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_kv = (
        indexer_compressor_kv
        if indexer_compressor_kv is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_gate = (
        indexer_compressor_gate
        if indexer_compressor_gate is not None
        else q.new_empty(q.shape[0], q.shape[1], 0)
    )
    indexer_compressor_ape = (
        indexer_compressor_ape if indexer_compressor_ape is not None else q.new_empty(0, 0)
    )
    indexer_compressor_norm_weight = (
        indexer_compressor_norm_weight
        if indexer_compressor_norm_weight is not None
        else q.new_empty(0)
    )
    indexer_compressor_kv_cache = (
        indexer_compressor_kv_cache
        if indexer_compressor_kv_cache is not None
        else q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    )
    indexer_compressor_gate_cache = (
        indexer_compressor_gate_cache
        if indexer_compressor_gate_cache is not None
        else q.new_empty(swa_cache.shape[0], swa_cache.shape[1], 0)
    )
    full_metadata = _op_metadata(
        metadata,
        swa_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=compressor.max_compressed_len,
        position_ids=position_ids,
    )
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor.ape,
        compressor.norm.weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        *full_metadata,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        compressor.norm.eps,
        compressor.rope_head_dim,
        topk_is_placeholder,
    )


def _rope_tables(max_seq_len: int, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_dim == 0:
        return torch.empty(max_seq_len, 0), torch.empty(max_seq_len, 0)
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.linspace(0.05, 0.25, rope_dim // 2, dtype=torch.float32).unsqueeze(0)
    angles = positions * freqs
    return angles.cos(), angles.sin()


def _compressor_case(
    compress_ratio: int,
    seq_len: int,
    *,
    compressed_capacity_tokens: int | None = None,
    batch_size: int = 1,
    head_dim: int = 8,
    device: str | torch.device = "cuda",
):
    hidden_size = 16
    rope_dim = 4
    capacity = compressed_capacity_tokens or seq_len
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity,
        ad_rope_cache_len=max(capacity, seq_len, 1),
    )
    compressor = DeepseekV4Compressor(config, compress_ratio, head_dim).eval().to(device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    compressor_kv, compressor_gate = compressor.project(hidden_states)
    cos_table, sin_table = (t.to(device) for t in _rope_tables(max(capacity, seq_len, 1), rope_dim))
    position_ids = (
        torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1).contiguous()
    )
    compressed_kv = compressor(hidden_states, cos_table, sin_table, position_ids)
    return (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    )


def _indexer_case(
    compress_ratio: int,
    total_len: int,
    *,
    index_topk: int = 2,
    index_n_heads: int = 1,
    index_head_dim: int = 32,
) -> tuple[DeepseekV4Indexer, torch.Tensor, torch.Tensor]:
    # index_head_dim must be a multiple of the hadamard-fp4 block (32) and
    # index_topk >= 2 (the fused top-k select kernel has no single-slot config).
    config = DeepseekV4Config(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=8,
        q_lora_rank=8,
        qk_rope_head_dim=4,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=total_len,
        ad_rope_cache_len=total_len,
    )
    indexer = DeepseekV4Indexer(config, compress_ratio).eval().cuda()
    hidden_states = torch.randn(1, total_len, config.hidden_size, device="cuda")
    q_lora = torch.randn(1, total_len, config.q_lora_rank, device="cuda")
    return indexer, hidden_states, q_lora


def _visible_source_topk(
    query_len: int,
    input_pos: int,
    kv_rows: int,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    max_select = window_size + max_compressed_len
    for token_offset in range(query_len):
        query_pos = input_pos + token_offset
        local_start = max(0, query_pos - window_size + 1)
        selected = list(range(local_start, query_pos + 1))
        visible_compressed = min((query_pos + 1) // compress_ratio, max_compressed_len)
        selected.extend(kv_rows + row_idx for row_idx in range(visible_compressed))
        selected.extend([-1] * (max_select - len(selected)))
        rows.append(selected)
    return torch.tensor([rows], dtype=torch.int64, device=device)


def _make_sparse_attention_caches(
    max_seq_len: int,
    head_dim: int,
    compressor_state_dim: int,
    fill_value: float = 0.0,
    *,
    num_slots: int = 1,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((num_slots, max_seq_len, head_dim), fill_value, device=device),
        torch.full((num_slots, max_seq_len, head_dim), fill_value, device=device),
        torch.full((num_slots, max_seq_len, compressor_state_dim), fill_value, device=device),
        torch.full((num_slots, max_seq_len, compressor_state_dim), fill_value, device=device),
    )


def _make_paged_sparse_attention_caches(
    max_seq_len: int,
    tokens_per_block: int,
    head_dim: int,
    compressor_state_dim: int,
    fill_value: float = 0.0,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pages = max((max_seq_len + tokens_per_block - 1) // tokens_per_block, 1)
    return (
        torch.full((num_pages, tokens_per_block, head_dim), fill_value, device=device),
        torch.full((num_pages, tokens_per_block, head_dim), fill_value, device=device),
        torch.full((num_pages, tokens_per_block, compressor_state_dim), fill_value, device=device),
        torch.full((num_pages, tokens_per_block, compressor_state_dim), fill_value, device=device),
    )


def _paged_cache_row(
    cache: torch.Tensor,
    logical_pos: int,
    tokens_per_block: int,
) -> torch.Tensor:
    return cache[logical_pos // tokens_per_block, logical_pos % tokens_per_block]


def _compressed_row_from_paged_state(
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    seq_idx: int,
    row_idx: int,
    row_position_id: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    head_dim: int,
    dtype: torch.dtype,
    rotate: bool = False,
) -> torch.Tensor:
    """One-row wrapper over the production batched reconstruction helper."""
    row_idx_tensor = torch.tensor([row_idx], dtype=torch.long, device=compressor_kv_cache.device)
    position_id_tensor = torch.tensor(
        [row_position_id], dtype=torch.long, device=compressor_kv_cache.device
    )
    return dsv4_sparse._compressed_rows_from_paged_state(
        compressor_kv_cache,
        compressor_gate_cache,
        seq_idx,
        row_idx_tensor,
        position_id_tensor,
        cu_num_pages_host,
        cache_loc_host,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        compress_ratio,
        head_dim,
        dtype,
        rotate=rotate,
    ).squeeze(0)


def _has_resource_with_suffix(resource_names: list[str], suffix: str) -> bool:
    return any(name.endswith(suffix) for name in resource_names)


# ---------------------------------------------------------------------------
# Source op semantics (ratio 0, CPU)
# ---------------------------------------------------------------------------


def test_sink_only_all_negative_topk_yields_finite_zero_output() -> None:
    q = torch.randn(1, 2, 2, 4)
    kv = torch.full((1, 5, 4), 1_000.0)
    attn_sink = torch.tensor([-3.0, 2.0])
    topk_idxs = torch.full((1, 2, 4), -1, dtype=torch.int64)

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(q), rtol=0, atol=0)


def test_source_ratio0_matches_reference_for_mixed_patterns() -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 4, 3, 6)
    kv = torch.randn(2, 8, 6)
    attn_sink = torch.tensor([-0.5, 0.25, 1.0])
    # duplicates, negative, and out-of-range (99) selections in one grid
    topk_idxs = torch.tensor(
        [
            [[0, 1, -1, 1], [2, 99, 3, -1], [4, -1, -1, 5], [6, 0, 6, 1]],
            [[7, 6, 5, 4], [3, -1, 3, 0], [-1, -1, -1, -1], [1, 2, 2, 7]],
        ],
        dtype=torch.int64,
    )

    output = _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.375)
    expected = _sparse_attention_reference(q, kv, attn_sink, topk_idxs, softmax_scale=0.375)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="mixed sparse attention: ")


def test_source_window_placeholder_matches_explicit_selection() -> None:
    torch.manual_seed(21)
    seq_len, window_size = 6, 3
    q = torch.randn(1, seq_len, 2, 4)
    kv = torch.randn(1, seq_len, 4)
    attn_sink = torch.tensor([-0.5, 0.25])
    explicit_topk = _visible_source_topk(seq_len, 0, seq_len, window_size, 1, 0, q.device)

    output = _run_sparse_attention(
        q,
        kv,
        attn_sink,
        torch.zeros(1, seq_len, window_size, dtype=torch.int64),
        window_size=window_size,
        topk_is_placeholder=True,
    )
    expected = _sparse_attention_reference(q, kv, attn_sink, explicit_topk, 1.0)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="window placeholder: ")


def test_fake_tensor_shape_behavior() -> None:
    q = torch.randn(2, 3, 2, 4)
    kv = torch.randn(2, 6, 4)
    attn_sink = torch.randn(2)
    topk_idxs = torch.tensor(
        [
            [[0, 1], [1, 2], [2, 3]],
            [[3, 4], [4, 5], [5, -1]],
        ],
        dtype=torch.int64,
    )

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        q_fake = fake_mode.from_tensor(q)
        kv_fake = fake_mode.from_tensor(kv)
        sink_fake = fake_mode.from_tensor(attn_sink)
        topk_fake = fake_mode.from_tensor(topk_idxs)
        output = _run_sparse_attention(q_fake, kv_fake, sink_fake, topk_fake, softmax_scale=0.5)

    assert isinstance(output, FakeTensor)
    assert output.shape == q.shape
    assert output.dtype == q.dtype


def test_export_with_dynamic_batch_sequence_and_topk() -> None:
    class SparseAttentionModule(torch.nn.Module):
        def forward(
            self,
            q: torch.Tensor,
            kv: torch.Tensor,
            attn_sink: torch.Tensor,
            topk_idxs: torch.Tensor,
        ) -> torch.Tensor:
            return _run_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale=0.5)

    batch = Dim("batch", min=1, max=4)
    seq = Dim("seq", min=1, max=8)
    kv_rows = Dim("kv_rows", min=4, max=12)
    k_select = Dim("k_select", min=1, max=4)

    q = torch.randn(2, 3, 2, 4)
    kv = torch.randn(2, 6, 4)
    attn_sink = torch.randn(2)
    topk_idxs = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[5, 4], [3, 2], [1, 0]],
        ],
        dtype=torch.int64,
    )

    exported = torch.export.export(
        SparseAttentionModule(),
        (q, kv, attn_sink, topk_idxs),
        dynamic_shapes={
            "q": {0: batch, 1: seq},
            "kv": {0: batch, 1: kv_rows},
            "attn_sink": {},
            "topk_idxs": {0: batch, 1: seq, 2: k_select},
        },
    )

    target_names = {str(node.target) for node in exported.graph.nodes if node.op == "call_function"}
    assert "auto_deploy.torch_deepseek_v4_sparse_attention.default" in target_names

    q_alt = torch.randn(1, 4, 2, 4)
    kv_alt = torch.randn(1, 7, 4)
    sink_alt = torch.randn(2)
    topk_alt = torch.tensor([[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]])
    output = exported.module()(q_alt, kv_alt, sink_alt, topk_alt)
    assert output.shape == q_alt.shape


# ---------------------------------------------------------------------------
# Fused Triton attend kernel (CUDA, bf16/fp16)
# ---------------------------------------------------------------------------


def _check_fused_attend(q, kv, sink, topk, scale, tol=2e-2):
    out = dsv4_sparse._deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    ref = _sparse_attention_reference(q, kv, sink, topk, scale)
    assert torch.isfinite(out).all()
    assert_rmse_close(out, ref, rmse_ratio_tol=tol, msg="fused attend: ")
    return out


@_requires_cuda
@pytest.mark.parametrize(
    ("num_heads", "dtype"),
    [
        (1, torch.bfloat16),
        (8, torch.float16),
        (16, torch.bfloat16),
        (64, torch.bfloat16),
    ],
)
def test_fused_attend_decode_per_rank_head_counts(num_heads, dtype) -> None:
    # H<=8 exercises the small-head split-K branch; H=64 the full decode shape.
    torch.manual_seed(0)
    B, S, D, L = 1, 1, 512, 640
    q = torch.randn(B, S, num_heads, D, device="cuda", dtype=dtype)
    kv = torch.randn(B, L, D, device="cuda", dtype=dtype)
    sink = torch.randn(num_heads, device="cuda", dtype=dtype)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L).expand(B, S, L)
    assert dsv4_sparse._can_use_fused_sparse_attention(
        q.reshape(B * S, num_heads, D), kv, topk.reshape(B * S, L)
    )
    _check_fused_attend(q, kv, sink, topk, D**-0.5)


@_requires_cuda
def test_fused_attend_decode_split_k_partial_mask() -> None:
    torch.manual_seed(4)
    B, S, H, D, L = 1, 1, 8, 512, 512
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L).clone()
    topk[0, 0, ::3] = -1
    _check_fused_attend(q, kv, sink, topk, D**-0.5)


@_requires_cuda
def test_fused_attend_all_negative_topk_yields_zero() -> None:
    torch.manual_seed(3)
    B, S, H, D, L = 1, 1, 64, 512, 640
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((B, L, D), 1000.0, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((B, S, L), -1, device="cuda", dtype=torch.int64)
    out = dsv4_sparse._deepseek_v4_sparse_attention(q, kv, sink, topk, D**-0.5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


@_requires_cuda
def test_fused_attend_prefill_random_selection() -> None:
    torch.manual_seed(1)
    B, S, H, D, kv_rows, K = 1, 128, 64, 512, 512, 256
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, kv_rows, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.randint(0, kv_rows, (B, S, K), device="cuda", dtype=torch.int64)
    topk = torch.where(torch.rand(B, S, K, device="cuda") < 0.1, torch.full_like(topk, -1), topk)
    _check_fused_attend(q, kv, sink, topk, D**-0.5)


@_requires_cuda
def test_fused_attend_batched_duplicates_and_out_of_range() -> None:
    torch.manual_seed(2)
    B, S, H, D, kv_rows, K = 2, 16, 8, 512, 128, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, kv_rows, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.randint(-1, kv_rows, (B, S, K), device="cuda", dtype=torch.int64)
    topk[..., 0] = topk[..., 1]  # duplicate
    topk[..., 2] = -1
    topk[..., 3] = 9999  # out of range -> masked
    _check_fused_attend(q, kv, sink, topk, D**-0.5)


@_requires_cuda
@pytest.mark.parametrize("head_dim", [64, 576])
def test_fused_attend_head_dim_tail_masking(head_dim) -> None:
    torch.manual_seed(5)
    B, S, H, L = 1, 1, 32, 320
    q = torch.randn(B, S, H, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, L, head_dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L)
    _check_fused_attend(q, kv, sink, topk, head_dim**-0.5)


# ---------------------------------------------------------------------------
# Cached op, ratio 0 (CPU unless noted)
# ---------------------------------------------------------------------------


def test_cached_ratio0_decode_reads_and_writes_across_paged_boundary() -> None:
    tokens_per_block = 2
    q_prefill = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    kv_prefill = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_prefill = torch.zeros(1, 3, 1, dtype=torch.int64)
    swa_cache, _, _, _ = _make_paged_sparse_attention_caches(4, tokens_per_block, 2, 0)

    _run_cached_sparse_attention(
        q_prefill,
        kv_prefill,
        attn_sink,
        topk_prefill,
        _context_meta(seq_len=3, tokens_per_block=tokens_per_block),
        swa_cache,
        window_size=4,
    )

    q_decode = torch.tensor([[[[1.0, 0.5]]]])
    kv_decode = torch.tensor([[[3.0, -1.0]]])
    output = _run_cached_sparse_attention(
        q_decode,
        kv_decode,
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64),
        _decode_meta(input_pos=3, tokens_per_block=tokens_per_block),
        swa_cache,
        window_size=4,
    )

    expected_kv = torch.cat([kv_prefill, kv_decode], dim=1)
    expected_topk = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int64)
    expected = _sparse_attention_reference(q_decode, expected_kv, attn_sink, expected_topk, 1.0)

    torch.testing.assert_close(swa_cache[0], expected_kv[0, :2])
    torch.testing.assert_close(swa_cache[1], expected_kv[0, 2:4])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="paged cached local window: ")


def test_cached_ratio0_flattened_prefill_uses_per_sequence_kv_slice() -> None:
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]]]])
    kv = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [10.0, 0.0], [0.0, 20.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0], [1], [0], [1]]], dtype=torch.int64)
    swa_cache = torch.empty(2, 8, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _multi_context_meta([2, 2]),
        swa_cache,
        window_size=2,
    )

    expected_seq0 = _run_sparse_attention(q[:, :2], kv[:, :2], attn_sink, topk_idxs[:, :2])
    expected_seq1 = _run_sparse_attention(q[:, 2:], kv[:, 2:], attn_sink, topk_idxs[:, 2:])
    expected = torch.cat((expected_seq0, expected_seq1), dim=1)

    torch.testing.assert_close(swa_cache[0, :2], kv[0, :2])
    torch.testing.assert_close(swa_cache[1, :2], kv[0, 2:])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="flattened prefill: ")


def test_cached_ratio0_prefill_honors_topk_duplicates_and_mask() -> None:
    # window_size present but prefill still reads the explicit top-k values
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    kv = torch.tensor([[[2.0, 0.0], [100.0, 100.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0, 0, -1], [1, -1, 0]]], dtype=torch.int64)
    swa_cache = torch.empty(1, 8, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _context_meta(seq_len=2),
        swa_cache,
        window_size=2,
    )
    expected = _run_sparse_attention(q, kv, attn_sink, topk_idxs)

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached prefill topk: ")


def test_cached_ratio0_topk_decode_without_window_uses_cache_positions() -> None:
    q_prefill = torch.zeros(1, 1, 1, 2)
    kv_prefill = torch.tensor([[[2.0, 0.0]]])
    attn_sink = torch.tensor([-20.0])
    swa_cache = torch.empty(1, 8, 2)

    _run_cached_sparse_attention(
        q_prefill,
        kv_prefill,
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64),
        _context_meta(seq_len=1),
        swa_cache,
    )

    q_decode = torch.tensor([[[[1.0, 0.0]]]])
    kv_decode = torch.tensor([[[3.0, 0.0]]])
    topk_decode = torch.tensor([[[0, 1]]], dtype=torch.int64)

    output = _run_cached_sparse_attention(
        q_decode,
        kv_decode,
        attn_sink,
        topk_decode,
        _decode_meta(input_pos=1),
        swa_cache,
    )

    expected_kv = torch.cat((kv_prefill, kv_decode), dim=1)
    expected = _sparse_attention_reference(q_decode, expected_kv, attn_sink, topk_decode, 1.0)

    torch.testing.assert_close(swa_cache[0, :2], expected_kv[0])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached top-k decode: ")


def test_cached_ratio0_chunked_prefill_window_and_topk_modes() -> None:
    torch.manual_seed(5)
    window_size = 2
    q = torch.randn(1, 5, 1, 4)
    kv = torch.randn(1, 5, 4)
    attn_sink = torch.tensor([-0.5])

    # window mode: continuation chunk attends the local window from the cache
    swa_cache = torch.empty(1, 8, 4)
    _run_cached_sparse_attention(
        q[:, :3],
        kv[:, :3],
        attn_sink,
        torch.zeros(1, 3, 1, dtype=torch.int64),
        _context_meta(seq_len=3),
        swa_cache,
        window_size=window_size,
    )
    output = _run_cached_sparse_attention(
        q[:, 3:],
        kv[:, 3:],
        attn_sink,
        torch.zeros(1, 2, 1, dtype=torch.int64),
        _context_meta(seq_len=2, input_pos=3),
        swa_cache,
        window_size=window_size,
    )
    window_topk = torch.tensor([[[2, 3], [3, 4]]], dtype=torch.int64)
    expected = _sparse_attention_reference(q[:, 3:], kv, attn_sink, window_topk, 1.0)
    torch.testing.assert_close(swa_cache[0, :5], kv[0])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="chunked window prefill: ")

    # top-k mode: continuation chunk reads explicit global cache positions
    swa_cache = torch.empty(1, 8, 4)
    _run_cached_sparse_attention(
        q[:, :3],
        kv[:, :3],
        attn_sink,
        torch.tensor([[[0], [1], [2]]], dtype=torch.int64),
        _context_meta(seq_len=3),
        swa_cache,
    )
    topk_chunk2 = torch.tensor([[[0, 3], [2, 4]]], dtype=torch.int64)
    output = _run_cached_sparse_attention(
        q[:, 3:],
        kv[:, 3:],
        attn_sink,
        topk_chunk2,
        _context_meta(seq_len=2, input_pos=3),
        swa_cache,
    )
    expected = _sparse_attention_reference(q[:, 3:], kv, attn_sink, topk_chunk2, 1.0)
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="chunked topk prefill: ")


def test_cached_ratio0_sink_only_negative_topk_yields_zero_output() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]])
    kv = torch.tensor([[[5.0, 5.0]]])
    attn_sink = torch.tensor([3.0])
    topk_idxs = torch.full((1, 1, 3), -1, dtype=torch.int64)
    swa_cache = torch.empty(1, 4, 2)

    output = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _decode_meta(input_pos=0),
        swa_cache,
    )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(q), rtol=0, atol=0)


def test_cached_ratio0_out_buffer_returns_dummy_and_fills_output() -> None:
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[7.0, 7.0]]]])
    kv = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [100.0, 100.0]]])
    attn_sink = torch.tensor([-20.0])
    topk_idxs = torch.tensor([[[0], [1], [2]]], dtype=torch.int64)

    expected = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _context_meta(seq_len=2),
        torch.empty(1, 8, 2),
        window_size=2,
    )

    out = torch.full_like(q, 123.0)
    result = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        topk_idxs,
        _context_meta(seq_len=2),
        torch.empty(1, 8, 2),
        window_size=2,
        out=out,
    )

    assert result.numel() == 0
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(out[:, 2:], torch.zeros_like(out[:, 2:]), rtol=0, atol=0)


def test_cached_window_placeholder_initial_prefill_matches_explicit() -> None:
    torch.manual_seed(19)
    seq_len, window_size = 4, 2
    q = torch.randn(1, seq_len, 1, 4)
    kv = torch.randn(1, seq_len, 4)
    attn_sink = torch.tensor([-0.5])
    explicit_topk = _visible_source_topk(seq_len, 0, seq_len, window_size, 1, 0, q.device)

    out_placeholder = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        torch.zeros(1, seq_len, window_size, dtype=torch.int64),
        _context_meta(seq_len=seq_len),
        torch.empty(1, 8, 4),
        window_size=window_size,
        topk_is_placeholder=True,
    )
    out_explicit = _run_cached_sparse_attention(
        q,
        kv,
        attn_sink,
        explicit_topk,
        _context_meta(seq_len=seq_len),
        torch.empty(1, 8, 4),
        window_size=window_size,
    )

    torch.testing.assert_close(out_placeholder, out_explicit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_cached_ratio0_decode_cuda_graph_replay_uses_runtime_slot_and_input_pos() -> None:
    q = torch.zeros(1, 1, 1, 2, device="cuda")
    kv = torch.tensor([[[2.0, 0.0]]], device="cuda")
    attn_sink = torch.tensor([-20.0], device="cuda")
    topk_idxs = torch.tensor([[[1]]], dtype=torch.int64, device="cuda")
    metadata = _cuda_decode_meta(input_pos=1, slot_idx=0)
    swa_cache = torch.full((2, 6, 2), -99.0, device="cuda")
    mhc_cache = torch.empty_like(swa_cache)
    compressor_kv_cache = q.new_empty(2, 6, 0)
    compressor_gate_cache = q.new_empty(2, 6, 0)
    indexer_compressor_kv_cache = q.new_empty(2, 6, 0)
    indexer_compressor_gate_cache = q.new_empty(2, 6, 0)
    out = torch.empty_like(q)
    empty_sparse_args = _empty_sparse_attention_tensors(q, kv)
    # Host mirrors captured once; the device-side prepare op re-runs inside the
    # graph so hoisted metadata tracks the runtime input_pos / cache_loc.
    std_metadata = _standard_metadata(metadata, q.device)

    def run_op() -> None:
        extra_metadata = _prepare_extra_metadata(metadata, swa_cache)
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
            q,
            kv,
            attn_sink,
            topk_idxs,
            *empty_sparse_args,
            *std_metadata,
            *extra_metadata,
            swa_cache,
            mhc_cache,
            compressor_kv_cache,
            compressor_gate_cache,
            indexer_compressor_kv_cache,
            indexer_compressor_gate_cache,
            1.0,
            None,
            0,
            None,
            1e-6,
            None,
            out=out,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run_op()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_op()

    kv.copy_(torch.tensor([[[7.0, 3.0]]], device="cuda"))
    topk_idxs.copy_(torch.tensor([[[3]]], dtype=torch.int64, device="cuda"))
    metadata[2].copy_(torch.tensor([3], dtype=torch.int32, device="cuda"))
    metadata[3].copy_(torch.tensor([1], dtype=torch.int64, device="cuda"))
    metadata[6].copy_(torch.tensor([1], dtype=torch.int32, device="cuda"))
    swa_cache[1, 3].fill_(-11.0)

    graph.replay()
    torch.cuda.synchronize()

    expected = _sparse_attention_reference(
        q,
        kv,
        attn_sink,
        torch.zeros_like(topk_idxs),
        softmax_scale=1.0,
    )
    torch.testing.assert_close(swa_cache[1, 3], kv[0, 0])
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Source + cached op, compressed ratios (CUDA)
# ---------------------------------------------------------------------------


@_requires_cuda
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_source_matches_expanded_sparse_construction(compress_ratio: int) -> None:
    torch.manual_seed(123 + compress_ratio)
    seq_len = compress_ratio
    q = torch.randn(1, seq_len, 1, 8, device="cuda")
    kv = torch.randn(1, seq_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.25], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len)
    topk_idxs = torch.arange(seq_len + compressor.max_compressed_len, device="cuda")
    topk_idxs = topk_idxs.view(1, 1, -1).expand(1, seq_len, -1).to(torch.int64)

    output = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        compress_ratio=compress_ratio,
    )
    expected = _run_sparse_attention(
        q,
        torch.cat((kv, compressed_kv), dim=1),
        attn_sink,
        topk_idxs,
    )

    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg=f"source ratio-{compress_ratio}: ",
    )


@_requires_cuda
def test_source_ratio128_placeholder_matches_explicit_selection() -> None:
    torch.manual_seed(77)
    compress_ratio, seq_len, window_size = 128, 256, 4
    q = torch.randn(1, seq_len, 1, 8, device="cuda")
    kv = torch.randn(1, seq_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.25], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len, compressed_capacity_tokens=seq_len)
    explicit_topk = _visible_source_topk(
        seq_len, 0, seq_len, window_size, compress_ratio, compressor.max_compressed_len, q.device
    )
    placeholder = torch.zeros(
        1,
        seq_len,
        window_size + compressor.max_compressed_len,
        dtype=torch.int64,
        device="cuda",
    )

    output = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        placeholder,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
        topk_is_placeholder=True,
    )
    expected = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        explicit_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="ratio-128 placeholder: ")


@_requires_cuda
def test_source_ratio4_placeholder_rebuilds_learned_selection() -> None:
    torch.manual_seed(93)
    compress_ratio, seq_len, window_size = 4, 16, 4
    q = torch.randn(1, seq_len, 1, 8, device="cuda")
    kv = torch.randn(1, seq_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.25], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len)
    indexer, hidden_states, q_lora = _indexer_case(compress_ratio, seq_len)
    cos = cos_table[position_ids]
    sin = sin_table[position_ids]
    indexer_q, indexer_weights, indexer_kv, indexer_gate = indexer.project(
        hidden_states, q_lora, cos, sin
    )
    learned_idxs = indexer(
        hidden_states, q_lora, cos, sin, cos_table, sin_table, position_ids, seq_len
    )
    local_topk = _visible_source_topk(seq_len, 0, seq_len, window_size, compress_ratio, 0, q.device)
    explicit_topk = torch.cat((local_topk, learned_idxs), dim=-1)
    placeholder = torch.zeros(
        1, seq_len, window_size + indexer.index_topk, dtype=torch.int64, device="cuda"
    )
    indexer_kwargs = dict(
        indexer_q=indexer_q,
        indexer_weights=indexer_weights,
        indexer_compressor_kv=indexer_kv,
        indexer_compressor_gate=indexer_gate,
        indexer_compressor_ape=indexer.compressor.ape,
        indexer_compressor_norm_weight=indexer.compressor.norm.weight,
    )

    output = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        placeholder,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
        topk_is_placeholder=True,
        **indexer_kwargs,
    )
    expected = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        explicit_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
        **indexer_kwargs,
    )

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="ratio-4 placeholder rebuild: ")


@_requires_cuda
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_cached_compressed_prefill_matches_source(compress_ratio: int) -> None:
    torch.manual_seed(311 + compress_ratio)
    seq_len = compress_ratio
    q = torch.randn(1, seq_len, 1, 8, device="cuda")
    kv = torch.randn(1, seq_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.25], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len)
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            seq_len,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            device="cuda",
        )
    )
    topk_idxs = torch.arange(seq_len + compressor.max_compressed_len, device="cuda")
    topk_idxs = topk_idxs.view(1, 1, -1).expand(1, seq_len, -1).to(torch.int64)

    output = _run_cached_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        _context_meta(seq_len=seq_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        compress_ratio=compress_ratio,
    )
    expected = _run_sparse_attention_with_compressor(
        q,
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        compress_ratio=compress_ratio,
    )

    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg=f"cached ratio-{compress_ratio} prefill: ",
    )


@_requires_cuda
def test_cached_ratio4_placeholder_initial_prefill_matches_explicit() -> None:
    torch.manual_seed(59)
    compress_ratio, seq_len, window_size = 4, 8, 4
    q = torch.randn(1, seq_len, 1, 8, device="cuda")
    kv = torch.randn(1, seq_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.25], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len)
    explicit_topk = _visible_source_topk(
        seq_len, 0, seq_len, window_size, compress_ratio, compressor.max_compressed_len, q.device
    )
    placeholder = torch.zeros(
        1,
        seq_len,
        window_size + compressor.max_compressed_len,
        dtype=torch.int64,
        device="cuda",
    )

    outputs = []
    for topk, is_placeholder in ((placeholder, True), (explicit_topk, False)):
        swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
            _make_sparse_attention_caches(
                seq_len, kv.shape[-1], compressor_kv.shape[-1], fill_value=777.0, device="cuda"
            )
        )
        outputs.append(
            _run_cached_sparse_attention_with_compressor(
                q,
                kv,
                attn_sink,
                topk,
                compressor_kv,
                compressor_gate,
                compressor,
                cos_table,
                sin_table,
                position_ids,
                _context_meta(seq_len=seq_len),
                swa_cache,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk_is_placeholder=is_placeholder,
            )
        )

    torch.testing.assert_close(outputs[0], outputs[1])


@_requires_cuda
def test_cached_ratio128_decode_uses_offset_position_ids_for_compressed_row() -> None:
    torch.manual_seed(229)
    compress_ratio = 128
    total_len = 128
    compressed_capacity_tokens = 256
    prefill_len = total_len - 1
    position_offset = 17
    window_size = 4
    q = torch.randn(1, total_len, 1, 8, device="cuda")
    kv = torch.randn(1, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        _,
        _,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=compressed_capacity_tokens,
    )
    cos_table, sin_table = (
        t.cuda()
        for t in _rope_tables(
            position_offset + compressed_capacity_tokens,
            compressor.rope_head_dim,
        )
    )
    position_ids = position_ids + position_offset
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            compressed_capacity_tokens,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            device="cuda",
        )
    )

    topk_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64, device="cuda"),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg="cached ratio-128 offset position_ids decode: ",
    )


@_requires_cuda
def test_cached_ratio128_multi_decode_metadata_matches_source_and_writes_slots() -> None:
    torch.manual_seed(217)
    batch_size = 2
    compress_ratio = 128
    total_len = 128
    prefill_len = total_len - 1
    window_size = 4
    compressed_capacity_tokens = 256
    q = torch.randn(batch_size, total_len, 1, 8, device="cuda")
    kv = torch.randn(batch_size, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=compressed_capacity_tokens,
        batch_size=batch_size,
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            compressed_capacity_tokens,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            num_slots=batch_size,
            device="cuda",
        )
    )

    topk_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    ).expand(batch_size, -1, -1)
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _multi_context_meta([prefill_len, prefill_len]),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    torch.testing.assert_close(mhc_cache[:, 0], torch.full_like(mhc_cache[:, 0], 777.0))

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(batch_size, 1, 1, dtype=torch.int64, device="cuda"),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _multi_decode_meta([prefill_len, prefill_len]),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    ).expand(batch_size, -1, -1)
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    for slot_idx in range(batch_size):
        torch.testing.assert_close(swa_cache[slot_idx, :total_len], kv[slot_idx])
        torch.testing.assert_close(mhc_cache[slot_idx, 0], compressed_kv[slot_idx, 0])
        torch.testing.assert_close(
            mhc_cache[slot_idx, compress_ratio],
            torch.full_like(mhc_cache[slot_idx, compress_ratio], 777.0),
        )
    assert_rmse_close(
        output,
        expected,
        rmse_ratio_tol=1e-6,
        msg="cached ratio-128 multi decode: ",
    )


@_requires_cuda
def test_cached_ratio128_emits_boundary_row_and_hides_future_rows() -> None:
    torch.manual_seed(128)
    compress_ratio = 128
    total_len = 128
    prefill_len = total_len - 1
    window_size = 4
    q = torch.randn(1, total_len, 1, 8, device="cuda")
    kv = torch.randn(1, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=256,
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            256,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            device="cuda",
        )
    )

    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        _visible_source_topk(
            prefill_len,
            0,
            prefill_len,
            window_size,
            compress_ratio,
            compressor.max_compressed_len,
            q.device,
        ),
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    torch.testing.assert_close(mhc_cache[0, 0], torch.full_like(mhc_cache[0, 0], 777.0))

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64, device="cuda"),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(swa_cache[0, :total_len], kv[0])
    torch.testing.assert_close(mhc_cache[0, 0], compressed_kv[0, 0])
    torch.testing.assert_close(
        mhc_cache[0, compress_ratio],
        torch.full_like(mhc_cache[0, compress_ratio], 777.0),
    )
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached ratio-128 boundary: ")


@_requires_cuda
def test_cached_ratio4_decode_matches_source_with_learned_indexer_topk() -> None:
    torch.manual_seed(44)
    compress_ratio = 4
    prefill_len = 15
    total_len = 16
    cache_capacity = 32  # 32-token pages engage the fused initial-prefill store
    window_size = 4
    q = torch.randn(1, total_len, 1, 8, device="cuda")
    kv = torch.randn(1, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, total_len)
    indexer, hidden_states, q_lora = _indexer_case(compress_ratio, total_len)
    cos = cos_table[position_ids]
    sin = sin_table[position_ids]
    indexer_q, indexer_weights, indexer_compressor_kv, indexer_compressor_gate = indexer.project(
        hidden_states,
        q_lora,
        cos,
        sin,
    )
    compressed_idxs_prefill = indexer(
        hidden_states[:, :prefill_len],
        q_lora[:, :prefill_len],
        cos[:, :prefill_len],
        sin[:, :prefill_len],
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        prefill_len,
    )
    compressed_idxs_decode = indexer(
        hidden_states,
        q_lora,
        cos,
        sin,
        cos_table,
        sin_table,
        position_ids,
        total_len,
    )[:, prefill_len:]
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            cache_capacity,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            device="cuda",
        )
    )
    indexer_compressor_kv_cache = torch.full(
        (1, cache_capacity, indexer_compressor_kv.shape[-1]),
        777.0,
        dtype=indexer_compressor_kv.dtype,
        device="cuda",
    )
    indexer_compressor_gate_cache = torch.full_like(indexer_compressor_kv_cache, 777.0)

    local_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        0,
        q.device,
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        torch.cat((local_prefill, compressed_idxs_prefill), dim=-1),
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q[:, :prefill_len],
        indexer_weights=indexer_weights[:, :prefill_len],
        indexer_compressor_kv=indexer_compressor_kv[:, :prefill_len],
        indexer_compressor_gate=indexer_compressor_gate[:, :prefill_len],
        indexer_compressor_ape=indexer.compressor.ape,
        indexer_compressor_norm_weight=indexer.compressor.norm.weight,
        indexer_compressor_kv_cache=indexer_compressor_kv_cache,
        indexer_compressor_gate_cache=indexer_compressor_gate_cache,
    )
    torch.testing.assert_close(mhc_cache[0, 0], compressed_kv[0, 0])

    local_decode = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        0,
        q.device,
    )
    expected_topk = torch.cat((local_decode, compressed_idxs_decode), dim=-1)
    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, window_size + indexer.index_topk, dtype=torch.int64, device="cuda"),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q[:, prefill_len:],
        indexer_weights=indexer_weights[:, prefill_len:],
        indexer_compressor_kv=indexer_compressor_kv[:, prefill_len:],
        indexer_compressor_gate=indexer_compressor_gate[:, prefill_len:],
        indexer_compressor_ape=indexer.compressor.ape,
        indexer_compressor_norm_weight=indexer.compressor.norm.weight,
        indexer_compressor_kv_cache=indexer_compressor_kv_cache,
        indexer_compressor_gate_cache=indexer_compressor_gate_cache,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    all_visible_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    all_visible = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        all_visible_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    # decode at input_pos == 15 completes compressed row 3 at logical position 12
    torch.testing.assert_close(mhc_cache[0, 3 * compress_ratio], compressed_kv[0, 3])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cached ratio-4 indexer: ")
    assert not torch.allclose(output, all_visible, rtol=1e-6, atol=1e-6)


@_requires_cuda
def test_cached_ratio4_chunked_prefill_matches_source_with_indexer() -> None:
    torch.manual_seed(48)
    compress_ratio, total_len, chunk_len, window_size = 4, 16, 8, 4
    q = torch.randn(1, total_len, 1, 8, device="cuda")
    kv = torch.randn(1, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        _,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, total_len)
    indexer, hidden_states, q_lora = _indexer_case(compress_ratio, total_len)
    cos = cos_table[position_ids]
    sin = sin_table[position_ids]
    indexer_q, indexer_weights, indexer_compressor_kv, indexer_compressor_gate = indexer.project(
        hidden_states, q_lora, cos, sin
    )
    idxs_chunk1 = indexer(
        hidden_states[:, :chunk_len],
        q_lora[:, :chunk_len],
        cos[:, :chunk_len],
        sin[:, :chunk_len],
        cos_table,
        sin_table,
        position_ids[:, :chunk_len],
        chunk_len,
    )
    idxs_full = indexer(
        hidden_states, q_lora, cos, sin, cos_table, sin_table, position_ids, total_len
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            total_len, kv.shape[-1], compressor_kv.shape[-1], fill_value=777.0, device="cuda"
        )
    )
    indexer_compressor_kv_cache = torch.full(
        (1, total_len, indexer_compressor_kv.shape[-1]), 777.0, device="cuda"
    )
    indexer_compressor_gate_cache = torch.full_like(indexer_compressor_kv_cache, 777.0)
    indexer_kwargs = dict(
        indexer_compressor_ape=indexer.compressor.ape,
        indexer_compressor_norm_weight=indexer.compressor.norm.weight,
        indexer_compressor_kv_cache=indexer_compressor_kv_cache,
        indexer_compressor_gate_cache=indexer_compressor_gate_cache,
    )

    local_chunk1 = _visible_source_topk(
        chunk_len, 0, chunk_len, window_size, compress_ratio, 0, q.device
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :chunk_len],
        kv[:, :chunk_len],
        attn_sink,
        torch.cat((local_chunk1, idxs_chunk1), dim=-1),
        compressor_kv[:, :chunk_len],
        compressor_gate[:, :chunk_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :chunk_len],
        _context_meta(seq_len=chunk_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q[:, :chunk_len],
        indexer_weights=indexer_weights[:, :chunk_len],
        indexer_compressor_kv=indexer_compressor_kv[:, :chunk_len],
        indexer_compressor_gate=indexer_compressor_gate[:, :chunk_len],
        **indexer_kwargs,
    )

    output = _run_cached_sparse_attention_with_compressor(
        q[:, chunk_len:],
        kv[:, chunk_len:],
        attn_sink,
        torch.zeros(
            1,
            total_len - chunk_len,
            window_size + indexer.index_topk,
            dtype=torch.int64,
            device="cuda",
        ),
        compressor_kv[:, chunk_len:],
        compressor_gate[:, chunk_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, chunk_len:],
        _context_meta(seq_len=total_len - chunk_len, input_pos=chunk_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        indexer_q=indexer_q[:, chunk_len:],
        indexer_weights=indexer_weights[:, chunk_len:],
        indexer_compressor_kv=indexer_compressor_kv[:, chunk_len:],
        indexer_compressor_gate=indexer_compressor_gate[:, chunk_len:],
        **indexer_kwargs,
    )

    local_chunk2 = _visible_source_topk(
        total_len - chunk_len, chunk_len, total_len, window_size, compress_ratio, 0, q.device
    )
    expected_topk = torch.cat((local_chunk2, idxs_full[:, chunk_len:]), dim=-1)
    expected = _run_sparse_attention_with_compressor(
        q[:, chunk_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="ratio-4 chunked prefill: ")


@_requires_cuda
def test_cached_ratio128_chunked_prefill_matches_source() -> None:
    torch.manual_seed(52)
    compress_ratio, total_len, chunk_len, window_size = 128, 256, 128, 4
    q = torch.randn(1, total_len, 1, 8, device="cuda")
    kv = torch.randn(1, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, total_len, compressed_capacity_tokens=total_len)
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            total_len, kv.shape[-1], compressor_kv.shape[-1], fill_value=777.0, device="cuda"
        )
    )

    _run_cached_sparse_attention_with_compressor(
        q[:, :chunk_len],
        kv[:, :chunk_len],
        attn_sink,
        _visible_source_topk(
            chunk_len,
            0,
            chunk_len,
            window_size,
            compress_ratio,
            compressor.max_compressed_len,
            q.device,
        ),
        compressor_kv[:, :chunk_len],
        compressor_gate[:, :chunk_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :chunk_len],
        _context_meta(seq_len=chunk_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    output = _run_cached_sparse_attention_with_compressor(
        q[:, chunk_len:],
        kv[:, chunk_len:],
        attn_sink,
        torch.zeros(
            1,
            total_len - chunk_len,
            window_size + compressor.max_compressed_len,
            dtype=torch.int64,
            device="cuda",
        ),
        compressor_kv[:, chunk_len:],
        compressor_gate[:, chunk_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, chunk_len:],
        _context_meta(seq_len=total_len - chunk_len, input_pos=chunk_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    expected_topk = _visible_source_topk(
        total_len - chunk_len,
        chunk_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, chunk_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(mhc_cache[0, 0], compressed_kv[0, 0])
    torch.testing.assert_close(mhc_cache[0, compress_ratio], compressed_kv[0, 1])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="ratio-128 chunked prefill: ")


@_requires_cuda
def test_cached_ratio128_mhc_cache_uses_token_domain_paged_positions() -> None:
    torch.manual_seed(129)
    compress_ratio = 128
    tokens_per_block = 32  # 32-token pages also engage the fused initial-prefill store
    total_len = 256
    prefill_len = total_len - 1
    window_size = 4
    q = torch.randn(1, total_len, 1, 8, device="cuda")
    kv = torch.randn(1, total_len, 8, device="cuda")
    attn_sink = torch.tensor([-0.25], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=total_len,
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_paged_sparse_attention_caches(
            total_len,
            tokens_per_block,
            kv.shape[-1],
            compressor_kv.shape[-1],
            fill_value=777.0,
            device="cuda",
        )
    )

    topk_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len, tokens_per_block=tokens_per_block),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    torch.testing.assert_close(
        _paged_cache_row(mhc_cache, 0, tokens_per_block), compressed_kv[0, 0]
    )

    output = _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, 1, dtype=torch.int64, device="cuda"),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len, tokens_per_block=tokens_per_block),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )
    expected_topk = _visible_source_topk(
        1,
        prefill_len,
        total_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    expected = _run_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv,
        attn_sink,
        expected_topk,
        compressor_kv,
        compressor_gate,
        compressor,
        cos_table,
        sin_table,
        position_ids,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(
        _paged_cache_row(mhc_cache, compress_ratio, tokens_per_block),
        compressed_kv[0, 1],
    )
    torch.testing.assert_close(
        _paged_cache_row(swa_cache, prefill_len, tokens_per_block),
        kv[0, prefill_len],
    )
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="paged ratio-128 mhc: ")


@_requires_cuda
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_decode_fused_compressed_row_update_writes_module_rows(compress_ratio: int) -> None:
    # head_dim 68 -> nope 64 satisfies the fused row-update kernels' fp8 alignment
    torch.manual_seed(41 + compress_ratio)
    head_dim = 68
    window_size = 4
    total_len = 8 if compress_ratio == 4 else 128
    capacity = total_len if compress_ratio == 4 else 256
    prefill_len = total_len - 1
    q = torch.randn(1, total_len, 1, head_dim, device="cuda")
    kv = torch.randn(1, total_len, head_dim, device="cuda")
    attn_sink = torch.tensor([-0.5], device="cuda")
    (
        compressor_kv,
        compressor_gate,
        compressed_kv,
        compressor,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(
        compress_ratio,
        total_len,
        compressed_capacity_tokens=capacity,
        head_dim=head_dim,
    )
    swa_cache, mhc_cache, compressor_kv_cache, compressor_gate_cache = (
        _make_sparse_attention_caches(
            capacity, head_dim, compressor_kv.shape[-1], fill_value=777.0, device="cuda"
        )
    )

    topk_prefill = _visible_source_topk(
        prefill_len,
        0,
        prefill_len,
        window_size,
        compress_ratio,
        compressor.max_compressed_len,
        q.device,
    )
    _run_cached_sparse_attention_with_compressor(
        q[:, :prefill_len],
        kv[:, :prefill_len],
        attn_sink,
        topk_prefill,
        compressor_kv[:, :prefill_len],
        compressor_gate[:, :prefill_len],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, :prefill_len],
        _context_meta(seq_len=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    row_idx = prefill_len // compress_ratio
    row_logical_pos = row_idx * compress_ratio
    torch.testing.assert_close(
        mhc_cache[0, row_logical_pos],
        torch.full_like(mhc_cache[0, row_logical_pos], 777.0),
    )

    topk_width = window_size if compress_ratio == 4 else window_size + compressor.max_compressed_len
    _run_cached_sparse_attention_with_compressor(
        q[:, prefill_len:],
        kv[:, prefill_len:],
        attn_sink,
        torch.zeros(1, 1, topk_width, dtype=torch.int64, device="cuda"),
        compressor_kv[:, prefill_len:],
        compressor_gate[:, prefill_len:],
        compressor,
        cos_table,
        sin_table,
        position_ids[:, prefill_len:],
        _decode_meta(input_pos=prefill_len),
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
    )

    torch.testing.assert_close(
        mhc_cache[0, row_logical_pos],
        compressed_kv[0, row_idx],
        rtol=1e-5,
        atol=1e-5,
    )


def _build_paged_caches(
    num_seq: int,
    pages_per_seq: int,
    tokens_per_block: int,
    state_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    num_pages = num_seq * pages_per_seq
    kv_cache = torch.randn(
        num_pages, tokens_per_block, state_dim, dtype=dtype, device=device, generator=generator
    )
    gate_cache = torch.randn(
        num_pages, tokens_per_block, state_dim, dtype=dtype, device=device, generator=generator
    )
    cu_num_pages = torch.arange(
        0, (num_seq + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=device
    )
    cache_loc = torch.arange(num_pages, dtype=torch.int32, device=device)
    seq_idx = torch.arange(num_seq, dtype=torch.int64, device=device)
    return kv_cache, gate_cache, cu_num_pages, cache_loc, seq_idx


@_requires_cuda
def test_decode_ratio4_fused_index_score_select_matches_eager() -> None:
    torch.manual_seed(9)
    device = torch.device("cuda")
    max_compressed_len = 3
    index_head_dim = 32
    num_heads = 16  # H > 8 engages the fused score kernel
    state_dim = 2 * index_head_dim
    rope_dim = 4
    tokens_per_block = 4
    kv_cache, gate_cache, cu_num_pages, cache_loc, seq_idx = _build_paged_caches(
        2, max_compressed_len, tokens_per_block, state_dim, dtype=torch.float32, device=device
    )
    input_pos = torch.tensor([11, 7], dtype=torch.int32, device=device)
    position_ids = input_pos.clone()
    extra = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr(
        input_pos,
        position_ids,
        cu_num_pages,
        cache_loc,
        tokens_per_block,
        max_compressed_len,
        1,
        1,
    )
    full_page_map = (extra[5], extra[6], extra[7])
    ape = torch.randn(4, state_dim, device=device)
    norm_weight = torch.randn(index_head_dim, device=device)
    cos_table, sin_table = (t.to(device) for t in _rope_tables(16, rope_dim))
    q_index = torch.randn(2, num_heads, index_head_dim, dtype=torch.bfloat16, device=device)
    indexer_weights = torch.randn(2, num_heads, dtype=torch.bfloat16, device=device)

    def run(page_map):
        return dsv4_sparse._select_decode_ratio4_indexer_rows(
            q_index,
            indexer_weights,
            kv_cache,
            gate_cache,
            seq_idx,
            input_pos.to(torch.long),
            position_ids.to(torch.long),
            2,
            cu_num_pages,
            cache_loc,
            ape,
            norm_weight,
            cos_table,
            sin_table,
            1e-6,
            rope_dim,
            max_compressed_len,
            full_page_map=page_map,
        )

    fused_rows, fused_valid = run(full_page_map)
    eager_rows, eager_valid = run(None)

    assert torch.equal(fused_rows, eager_rows)
    assert torch.equal(fused_valid, eager_valid)


def test_cached_ratio128_cpu_decode_eager_fallback_matches_reference() -> None:
    torch.manual_seed(6)
    compress_ratio, max_compressed_len, window_size, head_dim = 128, 2, 4, 8
    capacity = compress_ratio * max_compressed_len
    input_pos = capacity - 1
    q = torch.randn(1, 1, 1, head_dim)
    kv_decode = torch.randn(1, 1, head_dim)
    attn_sink = torch.tensor([-0.5])
    swa_cache = torch.randn(1, capacity, head_dim)
    mhc_cache = torch.randn(1, capacity, head_dim)
    swa_before = swa_cache.clone()

    output = _run_cached_sparse_attention(
        q,
        kv_decode,
        attn_sink,
        torch.zeros(1, 1, window_size + max_compressed_len, dtype=torch.int64),
        _decode_meta(input_pos=input_pos),
        swa_cache,
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        rope_dim=0,
        mhc_cache=mhc_cache,
    )

    local_kv = torch.cat((swa_before[:, input_pos - window_size + 1 : input_pos], kv_decode), dim=1)
    selected_kv = torch.cat(
        (local_kv, mhc_cache[:, 0:1], mhc_cache[:, compress_ratio : compress_ratio + 1]), dim=1
    )
    expected_topk = torch.arange(window_size + max_compressed_len).view(1, 1, -1)
    expected = _sparse_attention_reference(q, selected_kv, attn_sink, expected_topk, 1.0)

    torch.testing.assert_close(swa_cache[0, input_pos], kv_decode[0, 0])
    assert_rmse_close(output, expected, rmse_ratio_tol=1e-6, msg="cpu ratio-128 decode: ")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@pytest.mark.parametrize(
    ("compress_ratio", "capture_pos", "replay_pos", "state_dim", "topk_width"),
    [
        (4, 3, 7, 16, 6),
        (128, 127, 255, 8, 6),
    ],
)
def test_cached_compressed_decode_cuda_graph_replay_updates_runtime_compressed_row(
    compress_ratio: int,
    capture_pos: int,
    replay_pos: int,
    state_dim: int,
    topk_width: int,
) -> None:
    torch.manual_seed(1900 + compress_ratio)
    max_seq_len = replay_pos + 1
    max_compressed_len = 2
    head_dim = 8
    window_size = 4
    q = torch.zeros(1, 1, 1, head_dim, device="cuda")
    kv = torch.zeros(1, 1, head_dim, device="cuda")
    attn_sink = torch.tensor([-20.0], device="cuda")
    topk_idxs = torch.zeros(1, 1, topk_width, dtype=torch.int64, device="cuda")
    metadata = _cuda_decode_meta(input_pos=capture_pos, slot_idx=0)

    compressor_kv_values = torch.randn(2, max_seq_len, state_dim, device="cuda")
    compressor_gate_values = torch.randn(2, max_seq_len, state_dim, device="cuda")
    compressor_kv = compressor_kv_values[0:1, capture_pos : capture_pos + 1].clone()
    compressor_gate = compressor_gate_values[0:1, capture_pos : capture_pos + 1].clone()
    compressor_ape = torch.zeros(compress_ratio, state_dim, device="cuda")
    compressor_norm_weight = torch.ones(head_dim, device="cuda")
    cos_table = torch.empty(max_seq_len, 0, device="cuda")
    sin_table = torch.empty(max_seq_len, 0, device="cuda")
    position_ids = torch.tensor([[capture_pos]], dtype=torch.int64, device="cuda")

    swa_cache = torch.zeros(2, max_seq_len, head_dim, device="cuda")
    mhc_cache = torch.full((2, max_seq_len, head_dim), -77.0, device="cuda")
    compressor_kv_cache = torch.zeros(2, max_seq_len, state_dim, device="cuda")
    compressor_gate_cache = torch.zeros_like(compressor_kv_cache)
    compressor_kv_cache[0, :capture_pos] = compressor_kv_values[0, :capture_pos]
    compressor_gate_cache[0, :capture_pos] = compressor_gate_values[0, :capture_pos]
    compressor_kv_cache[1, :replay_pos] = compressor_kv_values[1, :replay_pos]
    compressor_gate_cache[1, :replay_pos] = compressor_gate_values[1, :replay_pos]

    if compress_ratio == 4:
        index_head_dim = 32  # hadamard-fp4 block size
        indexer_q = torch.ones(1, 1, 1, index_head_dim, device="cuda")
        indexer_weights = torch.ones(1, 1, 1, device="cuda")
        indexer_compressor_kv = torch.randn(1, 1, 2 * index_head_dim, device="cuda")
        indexer_compressor_gate = torch.randn(1, 1, 2 * index_head_dim, device="cuda")
        indexer_compressor_ape = torch.zeros(compress_ratio, 2 * index_head_dim, device="cuda")
        indexer_compressor_norm_weight = torch.ones(index_head_dim, device="cuda")
        indexer_compressor_kv_cache = torch.zeros(2, max_seq_len, 2 * index_head_dim, device="cuda")
        indexer_compressor_gate_cache = torch.zeros_like(indexer_compressor_kv_cache)
    else:
        indexer_q = q.new_empty(1, 1, 0, 0)
        indexer_weights = q.new_empty(1, 1, 0)
        indexer_compressor_kv = q.new_empty(1, 1, 0)
        indexer_compressor_gate = q.new_empty(1, 1, 0)
        indexer_compressor_ape = q.new_empty(0, 0)
        indexer_compressor_norm_weight = q.new_empty(0)
        indexer_compressor_kv_cache = q.new_empty(2, max_seq_len, 0)
        indexer_compressor_gate_cache = q.new_empty(2, max_seq_len, 0)

    out = torch.empty_like(q)
    # Host mirrors captured once; the device-side prepare op re-runs inside the
    # graph so hoisted metadata tracks the runtime input_pos / position_ids.
    std_metadata = _standard_metadata(metadata, q.device)

    def run_op() -> None:
        extra_metadata = _prepare_extra_metadata(
            metadata,
            swa_cache,
            window_size=window_size,
            compress_ratio=compress_ratio,
            max_compressed_len=max_compressed_len,
            position_ids=position_ids,
        )
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            indexer_q,
            indexer_weights,
            indexer_compressor_kv,
            indexer_compressor_gate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            *std_metadata,
            *extra_metadata,
            swa_cache,
            mhc_cache,
            compressor_kv_cache,
            compressor_gate_cache,
            indexer_compressor_kv_cache,
            indexer_compressor_gate_cache,
            1.0,
            window_size,
            compress_ratio,
            max_compressed_len,
            1e-6,
            0,
            out=out,
        )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run_op()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_op()

    compressor_kv.copy_(compressor_kv_values[1:2, replay_pos : replay_pos + 1])
    compressor_gate.copy_(compressor_gate_values[1:2, replay_pos : replay_pos + 1])
    metadata[2].copy_(torch.tensor([replay_pos], dtype=torch.int32, device="cuda"))
    metadata[3].copy_(torch.tensor([1], dtype=torch.int64, device="cuda"))
    metadata[6].copy_(torch.tensor([1], dtype=torch.int32, device="cuda"))
    position_ids.copy_(torch.tensor([[replay_pos]], dtype=torch.int64, device="cuda"))
    row_logical_pos = compress_ratio
    mhc_cache[1, row_logical_pos].fill_(-33.0)

    graph.replay()
    torch.cuda.synchronize()

    expected_row = _compressed_row_from_paged_state(
        compressor_kv_cache,
        compressor_gate_cache,
        0,
        1,
        compress_ratio,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        1e-6,
        0,
        compress_ratio,
        head_dim,
        compressor_kv.dtype,
    )
    torch.testing.assert_close(mhc_cache[1, row_logical_pos], expected_row, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp4 quant / hadamard rotate need CUDA")
@pytest.mark.parametrize(
    ("num_decode_rows", "max_compressed_len"),
    [(1, 1), (3, 7)],
)
def test_overlap_fullrange_matches_generic_paged_gather(
    num_decode_rows: int, max_compressed_len: int
) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    compress_ratio = 4
    head_dim = 32  # hadamard-fp4 block size
    state_dim = 2 * head_dim
    rope_dim = 4
    rms_norm_eps = 1e-6

    tokens_per_block = compress_ratio
    needed_tokens = max_compressed_len * compress_ratio
    pages_per_seq = max((needed_tokens + tokens_per_block - 1) // tokens_per_block, 1)

    kv_cache, gate_cache, cu_num_pages, cache_loc, seq_idx = _build_paged_caches(
        num_decode_rows,
        pages_per_seq,
        tokens_per_block,
        state_dim,
        dtype=torch.float32,
        device=device,
    )

    table_len = max(needed_tokens + 8, 8)
    cos_table, sin_table = (t.to(device) for t in _rope_tables(table_len, rope_dim))
    ape = torch.randn(1, state_dim, dtype=torch.float32, device=device)
    norm_weight = torch.randn(head_dim, dtype=torch.float32, device=device)

    candidate_rows = torch.arange(max_compressed_len, dtype=torch.long, device=device)
    candidate_rows = candidate_rows.view(1, -1).expand(num_decode_rows, -1)
    input_pos = torch.full((num_decode_rows,), needed_tokens - 1, dtype=torch.long, device=device)
    row_position_id = input_pos.unsqueeze(1) - (
        input_pos.unsqueeze(1) - candidate_rows * compress_ratio
    )

    flat_seq_idx = seq_idx.unsqueeze(1).expand_as(candidate_rows).reshape(-1)
    flat_rows = candidate_rows.reshape(-1)
    flat_row_position_id = row_position_id.reshape(-1)

    expected = dsv4_sparse._batched_compressed_rows_from_paged_state(
        kv_cache,
        gate_cache,
        flat_seq_idx,
        flat_rows,
        flat_row_position_id,
        cu_num_pages,
        cache_loc,
        ape,
        norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        compress_ratio,
        head_dim,
        dtype,
        rotate=True,
    ).view(num_decode_rows, max_compressed_len, head_dim)

    actual = dsv4_sparse._batched_overlap_compressed_rows_fullrange(
        kv_cache,
        gate_cache,
        seq_idx,
        row_position_id,
        cu_num_pages,
        cache_loc,
        ape,
        norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        compress_ratio,
        head_dim,
        max_compressed_len,
        dtype,
        rotate=True,
    )

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_source_and_cached_op_invalid_inputs_raise() -> None:
    q = torch.randn(1, 1, 1, 2)
    kv = torch.randn(1, 1, 2)
    sink = torch.randn(1)
    topk = torch.zeros(1, 1, 1, dtype=torch.int64)
    swa = torch.empty(1, 4, 2)
    meta = _decode_meta(input_pos=0)

    with pytest.raises(TypeError, match="q must be floating point"):
        _run_sparse_attention(q.int(), kv, sink, topk)
    with pytest.raises(TypeError, match="kv must be floating point"):
        _run_sparse_attention(q, kv.int(), sink, topk)
    with pytest.raises(TypeError, match="attn_sink must be floating point"):
        _run_sparse_attention(q, kv, sink.int(), topk)
    with pytest.raises(TypeError, match="same dtype"):
        _run_sparse_attention(q, kv.half(), sink, topk)
    with pytest.raises(TypeError, match="topk_idxs must be int32 or int64"):
        _run_sparse_attention(q, kv, sink, topk.float())
    with pytest.raises(ValueError, match="kv batch dimension"):
        _run_sparse_attention(q, kv.expand(2, -1, -1), sink, topk)
    with pytest.raises(ValueError, match="topk_idxs batch dimension"):
        _run_sparse_attention(q, kv, sink, topk.expand(2, -1, -1))
    with pytest.raises(ValueError, match="topk_idxs sequence dimension"):
        _run_sparse_attention(q, kv, sink, topk.expand(-1, 2, -1))
    with pytest.raises(ValueError, match="kv head dimension"):
        _run_sparse_attention(q, torch.randn(1, 1, 3), sink, topk)
    with pytest.raises(ValueError, match="attn_sink length"):
        _run_sparse_attention(q, kv, torch.randn(2), topk)

    with pytest.raises(TypeError, match="swa_cache must be floating point"):
        _run_cached_sparse_attention(q, kv, sink, topk, meta, torch.zeros(1, 4, 2).int())
    with pytest.raises(ValueError, match="swa_cache head dimension"):
        _run_cached_sparse_attention(q, kv, sink, topk, meta, torch.empty(1, 4, 3))
    with pytest.raises(ValueError, match="window_size must be positive"):
        _run_cached_sparse_attention(q, kv, sink, topk, meta, swa, window_size=0)
    with pytest.raises(ValueError, match="window_size is required"):
        _run_cached_sparse_attention(q, kv, sink, topk, meta, swa, compress_ratio=4)
    with pytest.raises(ValueError, match="max_compressed_len must be positive"):
        _run_cached_sparse_attention(q, kv, sink, topk, meta, swa, window_size=2, compress_ratio=4)
    with pytest.raises(ValueError, match="rope_dim is required"):
        _run_cached_sparse_attention(
            q, kv, sink, topk, meta, swa, window_size=2, compress_ratio=4, max_compressed_len=2
        )
    with pytest.raises(ValueError, match="compress_ratio"):
        _run_cached_sparse_attention(q, kv, sink, topk, meta, swa, compress_ratio=2)

    with pytest.raises(ValueError, match="window_size is required to rebuild the window topk"):
        _run_sparse_attention(q, kv, sink, topk, topk_is_placeholder=True)


def _run_source_compressed_for_errors(
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    max_compressed_len: int | None = 2,
    rope_dim: int | None = 0,
    window_size: int | None = None,
    topk_is_placeholder: bool = False,
) -> torch.Tensor:
    seq_len = compressor_kv.shape[1]
    head_dim = 1
    q = torch.randn(1, seq_len, 1, head_dim)
    kv = torch.randn(1, seq_len, head_dim)
    sink = torch.randn(1)
    topk = torch.zeros(1, seq_len, 1, dtype=torch.int64)
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        sink,
        topk,
        compressor_kv,
        compressor_gate,
        torch.zeros(4, compressor_kv.shape[-1]),
        torch.ones(head_dim),
        torch.empty(8, 0),
        torch.empty(8, 0),
        torch.arange(seq_len).unsqueeze(0),
        q.new_empty(1, seq_len, 0, 0),
        q.new_empty(1, seq_len, 0),
        q.new_empty(1, seq_len, 0),
        q.new_empty(1, seq_len, 0),
        q.new_empty(0, 0),
        q.new_empty(0),
        1.0,
        False,
        "mha_sparse",
        0,
        window_size,
        4,
        max_compressed_len,
        head_dim,
        rope_dim,
        1e-6,
        topk_is_placeholder,
    )


def test_source_compressed_invalid_inputs_raise() -> None:
    compressor_kv = torch.randn(1, 4, 2)
    compressor_gate = torch.randn(1, 4, 2)

    with pytest.raises(ValueError, match="max_compressed_len is required"):
        _run_source_compressed_for_errors(compressor_kv, compressor_gate, max_compressed_len=None)
    with pytest.raises(ValueError, match="rope_dim is required"):
        _run_source_compressed_for_errors(compressor_kv, compressor_gate, rope_dim=None)
    with pytest.raises(ValueError, match="window_size is required to rebuild the compressed topk"):
        _run_source_compressed_for_errors(compressor_kv, compressor_gate, topk_is_placeholder=True)
    with pytest.raises(ValueError, match="matching shapes"):
        _run_source_compressed_for_errors(compressor_kv, torch.randn(1, 4, 4))
    with pytest.raises(ValueError, match="max_compressed_len must be positive"):
        _run_source_compressed_for_errors(compressor_kv, compressor_gate, max_compressed_len=0)
    with pytest.raises(ValueError, match="not divisible"):
        _run_source_compressed_for_errors(torch.randn(1, 4, 3), torch.randn(1, 4, 3))
    with pytest.raises(ValueError, match="exceeds compressed capacity"):
        _run_source_compressed_for_errors(
            torch.randn(1, 8, 2), torch.randn(1, 8, 2), max_compressed_len=1
        )
    with pytest.raises(ValueError, match="rope_dim must be in"):
        _run_source_compressed_for_errors(compressor_kv, compressor_gate, rope_dim=-1)


@_requires_cuda
def test_device_mismatch_inputs_raise() -> None:
    q = torch.randn(1, 1, 1, 2, device="cuda")
    kv = torch.randn(1, 1, 2, device="cuda")
    sink = torch.randn(1, device="cuda")
    topk = torch.zeros(1, 1, 1, dtype=torch.int64, device="cuda")

    with pytest.raises(ValueError, match="kv must be on"):
        _run_sparse_attention(q, kv.cpu(), sink, topk)
    with pytest.raises(ValueError, match="attn_sink must be on"):
        _run_sparse_attention(q, kv, sink.cpu(), topk)
    with pytest.raises(ValueError, match="topk_idxs must be on"):
        _run_sparse_attention(q, kv, sink, topk.cpu())
    with pytest.raises(ValueError, match="swa_cache must be on"):
        _run_cached_sparse_attention(
            q, kv, sink, topk, _cuda_decode_meta(input_pos=0), torch.empty(1, 4, 2)
        )


def test_cached_sparse_attention_rejects_short_metadata() -> None:
    q = torch.randn(1, 1, 1, 2)
    kv = torch.randn(1, 1, 2)
    attn_sink = torch.randn(1)
    topk_idxs = torch.zeros(1, 1, 1, dtype=torch.int64)
    batch_info_host = BatchInfo()
    batch_info_host.update([1, 1, 0, 0, 0, 0])
    cu_num_pages, cache_loc = _page_meta([1], [0], [0])
    metadata = (
        batch_info_host.serialize(),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int32),
        cu_num_pages,
        cache_loc,
    )

    with pytest.raises(ValueError, match="cu_seqlen_host must have at least 2 elements"):
        _run_cached_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            metadata,
            torch.empty(1, 4, 2),
        )


# ---------------------------------------------------------------------------
# Fake tensors, cache transform, descriptor, and interface plumbing
# ---------------------------------------------------------------------------


def test_cached_fake_tensor_rank_behavior() -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.zeros(1, 2, 1, dtype=torch.int64)
    metadata = _context_meta(seq_len=2)
    swa_cache = torch.empty(1, 8, 4)

    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        q_fake = fake_mode.from_tensor(q)
        metadata_fake = tuple(fake_mode.from_tensor(tensor) for tensor in metadata)
        output = _run_cached_sparse_attention(
            q_fake,
            fake_mode.from_tensor(kv),
            fake_mode.from_tensor(attn_sink),
            fake_mode.from_tensor(topk_idxs),
            metadata_fake,
            fake_mode.from_tensor(swa_cache),
            window_size=2,
        )

        assert isinstance(output, FakeTensor)
        assert output.shape == q.shape
        assert output.dtype == q.dtype

        with pytest.raises(ValueError, match="swa_cache must have rank 3"):
            _run_cached_sparse_attention(
                q_fake,
                fake_mode.from_tensor(kv),
                fake_mode.from_tensor(attn_sink),
                fake_mode.from_tensor(topk_idxs),
                metadata_fake,
                fake_mode.from_tensor(torch.empty(1, 8, 1, 4)),
                window_size=2,
            )


class _TinyDeepSeekSparseModule(torch.nn.Module):
    def __init__(self, compress_ratio: int = 0) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
    ) -> torch.Tensor:
        state_dim = kv.shape[-1] * (2 if self.compress_ratio == 4 else 1)
        compressor_kv = q.new_empty(q.shape[0], q.shape[1], state_dim)
        compressor_gate = q.new_empty(q.shape[0], q.shape[1], state_dim)
        compressor_ape = q.new_empty(self.compress_ratio, state_dim)
        compressor_norm_weight = q.new_empty(kv.shape[-1])
        cos_table = q.new_empty(8, 0)
        sin_table = q.new_empty(8, 0)
        position_ids = torch.arange(q.shape[1], device=q.device).unsqueeze(0)
        indexer_q = q.new_empty(q.shape[0], q.shape[1], 0, 0)
        indexer_weights = q.new_empty(q.shape[0], q.shape[1], 0)
        indexer_compressor_kv = q.new_empty(q.shape[0], q.shape[1], 0)
        indexer_compressor_gate = q.new_empty(q.shape[0], q.shape[1], 0)
        indexer_compressor_ape = q.new_empty(0, 0)
        indexer_compressor_norm_weight = q.new_empty(0)
        max_compressed_len = 2 if self.compress_ratio else None
        rope_dim = 0 if self.compress_ratio else None
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            indexer_q,
            indexer_weights,
            indexer_compressor_kv,
            indexer_compressor_gate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            1.0,
            False,
            "mha_sparse",
            0,
            4,
            self.compress_ratio,
            max_compressed_len,
            kv.shape[-1],
            rope_dim,
            1e-6,
        )


class _TinyDenseAttentionModule(torch.nn.Module):
    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_attention(
            qkv,
            qkv,
            qkv,
            None,
            0.0,
            True,
            1.0,
            None,
            None,
            None,
            "bsnd",
        )


def test_deepseek_sparse_cache_initializers_use_schema_names_for_source_args() -> None:
    graph = Graph()
    q_node = graph.placeholder("q")
    kv_node = graph.placeholder("kv")
    compressor_kv_node = graph.placeholder("compressor_kv")
    indexer_compressor_kv_node = graph.placeholder("indexer_compressor_kv")

    kv_node.meta["val"] = torch.empty(1, 2, 4, dtype=torch.float16)
    compressor_kv_node.meta["val"] = torch.empty(1, 2, 8, dtype=torch.float32)
    indexer_compressor_kv_node.meta["val"] = torch.empty(1, 2, 3, dtype=torch.float32)

    source_node = graph.call_function(
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default,
        args=(q_node,),
        kwargs={
            "kv": kv_node,
            "compressor_kv": compressor_kv_node,
            "indexer_compressor_kv": indexer_compressor_kv_node,
        },
    )

    handlers = DeepSeekV4SparseAttention.get_cache_initializers(source_node, KvCacheConfig())

    assert DeepSeekV4SparseAttention.get_num_qkv_args() == len(dsv4_sparse._SOURCE_TENSOR_ARG_NAMES)
    assert isinstance(handlers["swa_cache"], PagedResourceHandler)
    assert handlers["swa_cache"].token_shape == (4,)
    assert handlers["swa_cache"].dtype == torch.float16
    assert handlers["compressor_kv_cache"].token_shape == (8,)
    assert handlers["indexer_compressor_kv_cache"].token_shape == (3,)
    # PagedResourceHandler contract: equality, sizing, and paged-ness
    assert handlers["swa_cache"].is_paged
    assert handlers["swa_cache"] == PagedResourceHandler(4, dtype=torch.float16)
    assert handlers["swa_cache"] != PagedResourceHandler(8, dtype=torch.float16)
    assert handlers["swa_cache"] != handlers["compressor_kv_cache"]
    assert handlers["swa_cache"] != object()
    assert handlers["swa_cache"]._get_bytes_per_token() == 4 * 2
    assert handlers["compressor_kv_cache"]._get_bytes_per_token() == 8 * 4


@pytest.mark.parametrize("compress_ratio", [0, 4])
def test_deepseek_sparse_cache_transform_rewrites_source_op_and_adds_resource(
    compress_ratio: int,
) -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 4 if compress_ratio else 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    gm = torch_export_to_gm(
        _TinyDeepSeekSparseModule(compress_ratio=compress_ratio),
        (q, kv, attn_sink, topk_idxs),
    )
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=8,
        device="cpu",
    )

    transform = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT)
    )
    gm, info = transform._apply(gm, cm, factory=None, shared_config=SharedConfig())

    assert info.num_matches == 1
    targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    assert torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention.default not in targets
    assert torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default in targets

    placeholder_names = [node.target for node in gm.graph.nodes if node.op == "placeholder"]
    resource_names = list(cm._resource_lookup)
    for suffix in (
        "_swa_cache",
        "_mhc_cache",
        "_compressor_kv_cache",
        "_compressor_gate_cache",
    ):
        assert _has_resource_with_suffix(placeholder_names, suffix)
        assert _has_resource_with_suffix(resource_names, suffix)


def test_deepseek_sparse_cache_transform_rejects_bad_ratio_and_backend() -> None:
    q = torch.randn(1, 2, 1, 4)
    kv = torch.randn(1, 2, 4)
    attn_sink = torch.randn(1)
    topk_idxs = torch.tensor([[[0], [1]]], dtype=torch.int64)
    gm = torch_export_to_gm(
        _TinyDeepSeekSparseModule(compress_ratio=2),
        (q, kv, attn_sink, topk_idxs),
    )
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=8,
        device="cpu",
    )
    transform = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT)
    )

    with pytest.raises(RuntimeError, match="supports compress_ratio"):
        transform._apply(gm, cm, factory=None, shared_config=SharedConfig())

    mismatched = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="torch")
    )
    with pytest.raises(ValueError, match="only supports"):
        mismatched._apply(gm, cm, factory=None, shared_config=SharedConfig())


def test_dense_torch_attention_cache_insertion_remains_separate() -> None:
    qkv = torch.randn(1, 2, 1, 4)
    gm = torch_export_to_gm(_TinyDenseAttentionModule(), (qkv,))
    cm = CachedSequenceInterface(
        max_seq_len=8,
        max_batch_size=2,
        max_num_tokens=8,
        device="cpu",
    )

    deepseek_transform = InsertCachedDeepSeekV4SparseAttention(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT)
    )
    gm_after_deepseek, deepseek_info = deepseek_transform._apply(
        gm, cm, factory=None, shared_config=SharedConfig()
    )
    assert deepseek_info.num_matches == 0

    dense_transform = _InsertCachedOperator(
        InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="torch")
    )
    gm_after_dense, dense_info = dense_transform._apply(
        gm_after_deepseek, cm, factory=None, shared_config=SharedConfig()
    )

    assert dense_info.num_matches == 1
    targets = [node.target for node in gm_after_dense.graph.nodes if node.op == "call_function"]
    assert torch.ops.auto_deploy.torch_cached_attention_with_cache.default in targets
    assert (
        torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default not in targets
    )


def test_sequence_info_decode_fast_paths_and_paged_allocate() -> None:
    si = SequenceInfo(max_seq_len=16, max_batch_size=4, max_num_tokens=16)
    si.to("cpu")

    # mixed batch -> general repeat/cumsum position_ids path
    si.nest_sequences(
        input_ids=[1, 2, 3, 4, 5],
        cu_seqlen=[0, 3, 5],
        input_pos=[2, 7],
        slot_idx=[0, 1],
    )
    pos_host = si.get_arg("position_ids_host", truncate=True, unflatten=False)
    assert pos_host.tolist() == [2, 3, 4, 7, 8]

    # generate-only batch with identity gather/scatter -> rescatter fast path
    ungathered = torch.tensor([11, 12, 13], dtype=torch.int)
    si.nest_sequences(
        input_ids=[-1, -1, -1],
        cu_seqlen=[0, 1, 2, 3],
        input_pos=[5, 6, 7],
        slot_idx=[0, 1, 2],
        _gather_idx=[0, 1, 2],
        _mask_scatter_indices=[0, 1, 2],
        _ungathered_input_ids=ungathered,
    )
    assert si.get_arg("input_ids", truncate=True, unflatten=False).tolist() == [11, 12, 13]
    pos_host = si.get_arg("position_ids_host", truncate=True, unflatten=False)
    assert pos_host.tolist() == [5, 6, 7]

    handler = PagedResourceHandler(4, dtype=torch.float16)
    si.update_cache_information(num_blocks=2)
    buffer = handler.allocate(si)
    assert buffer.shape == (si.num_blocks, si.tokens_per_block, 4)
    assert buffer.dtype == torch.float16


def test_torch_attention_sink_scores_cast_to_value_dtype() -> None:
    torch.manual_seed(3)
    q = torch.randn(1, 2, 1, 4, dtype=torch.float16)
    sinks = torch.tensor([0.25], dtype=torch.float16)

    output = torch.ops.auto_deploy.torch_attention(
        q, q, q, None, 0.0, True, 1.0, sinks, None, None, "bsnd"
    )

    qt = q.transpose(1, 2).float()
    scores = torch.matmul(qt, qt.transpose(-1, -2))
    scores = scores.masked_fill(
        torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1), float("-inf")
    )
    logits_max = scores.amax(dim=-1, keepdim=True)
    probs = torch.exp(scores - logits_max)
    normalizer = probs.sum(dim=-1, keepdim=True) + torch.exp(
        sinks.float().view(1, 1, 1, 1) - logits_max
    )
    expected = torch.matmul((probs / normalizer).to(q.dtype), q.transpose(1, 2)).transpose(1, 2)

    assert output.dtype == q.dtype
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
