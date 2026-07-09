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

"""Tests for vectorized DeepSeek-V4 compressed-cache prefill updates."""

import math

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def test_cached_op_uses_host_prefill_metadata_within_schema_limit():
    """Host mirrors replace per-layer D2H reads without exceeding PyTorch's op limit."""
    metadata_args = M.DeepSeekV4SparseAttention.get_standard_metadata_args()
    assert metadata_args == [
        "batch_info_host",
        "input_pos",
        "slot_idx",
        "cu_num_pages",
        "cache_loc",
        "seq_len_host",
        "input_pos_host",
        "cu_seqlen_host",
        "cu_num_pages_host",
        "cache_loc_host",
    ]

    schema = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default._schema
    assert len(schema.arguments) == 64
    assert "last_page_len" not in {argument.name for argument in schema.arguments}
    arg_names = {argument.name for argument in schema.arguments}
    # Device-side seq_len / cu_seqlen were dropped (only their *_host mirrors are
    # read); their slots carry the hoisted long decode metadata (idea_0090).
    assert {"seq_idx_long", "input_pos_long"} <= arg_names
    assert "seq_len" not in arg_names and "cu_seqlen" not in arg_names


def _scalar_write_paged_cache_rows(
    values,
    cache,
    seq_idx,
    input_pos,
    cu_num_pages,
    cache_loc,
):
    """Reference implementation that writes one contiguous slice per physical page."""
    cursor = 0
    logical_pos = input_pos
    tokens_per_block = int(cache.shape[1])
    while cursor < values.shape[0]:
        page_id, page_offset = M._host_page_id_and_offset(
            cache,
            seq_idx,
            logical_pos,
            cu_num_pages,
            cache_loc,
        )
        write_len = min(values.shape[0] - cursor, tokens_per_block - page_offset)
        cache[page_id, page_offset : page_offset + write_len].copy_(
            values[cursor : cursor + write_len].to(cache.dtype)
        )
        cursor += write_len
        logical_pos += write_len


def _legacy_compressed_row(
    kv_cache,
    gate_cache,
    seq_idx,
    row_idx,
    row_position_id,
    cu_num_pages,
    cache_loc,
    ape,
    norm_weight,
    cos_table,
    sin_table,
    eps,
    rope_dim,
    ratio,
    head_dim,
    dtype,
):
    """Row-at-a-time implementation that predates the vectorized prefill path."""
    anchor = row_idx * ratio
    kv_rows = []
    gate_rows = []
    mode = M._compression_mode(ratio)
    if mode.overlap:
        for offset in range(ratio):
            position = anchor - ratio + offset
            if position < 0:
                kv_rows.append(torch.zeros(head_dim, dtype=dtype, device=kv_cache.device))
                gate_rows.append(
                    torch.full((head_dim,), -1.0e20, dtype=dtype, device=gate_cache.device)
                )
                continue
            kv_state = M._gather_paged_rows(
                kv_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages,
                cache_loc,
                dtype,
            ).squeeze(0)
            gate_state = M._gather_paged_rows(
                gate_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages,
                cache_loc,
                dtype,
            ).squeeze(0)
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + ape[offset, :head_dim].to(dtype))

        for offset in range(ratio):
            position = anchor + offset
            kv_state = M._gather_paged_rows(
                kv_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages,
                cache_loc,
                dtype,
            ).squeeze(0)
            gate_state = M._gather_paged_rows(
                gate_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages,
                cache_loc,
                dtype,
            ).squeeze(0)
            kv_rows.append(kv_state[head_dim : 2 * head_dim])
            gate_rows.append(
                gate_state[head_dim : 2 * head_dim] + ape[offset, head_dim : 2 * head_dim].to(dtype)
            )
    else:
        for offset in range(ratio):
            position = anchor + offset
            kv_state = M._gather_paged_rows(
                kv_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages,
                cache_loc,
                dtype,
            ).squeeze(0)
            gate_state = M._gather_paged_rows(
                gate_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages,
                cache_loc,
                dtype,
            ).squeeze(0)
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + ape[offset, :head_dim].to(dtype))

    pooled = torch.ops.auto_deploy.deepseek_v4_compress_pool(
        torch.stack(kv_rows), torch.stack(gate_rows)
    )
    pooled = M._rms_norm_ref(pooled.unsqueeze(0), norm_weight, eps).squeeze(0)
    row_position_id = max(0, min(row_position_id, cos_table.shape[0] - 1))
    return M._apply_compressed_rope_and_quantize(
        pooled.unsqueeze(0),
        cos_table[row_position_id].unsqueeze(0),
        sin_table[row_position_id].unsqueeze(0),
        rope_dim,
    ).squeeze(0)


def _legacy_update(inp, kv_cache, gate_cache, mhc_cache):
    ratio = inp["ratio"]
    M._write_paged_cache_rows(
        inp["kv_seq"],
        kv_cache,
        inp["seq_idx"],
        inp["input_pos"],
        inp["cu_num_pages"],
        inp["cache_loc"],
    )
    M._write_paged_cache_rows(
        inp["gate_seq"],
        gate_cache,
        inp["seq_idx"],
        inp["input_pos"],
        inp["cu_num_pages"],
        inp["cache_loc"],
    )

    old_completed = min(inp["input_pos"] // ratio, inp["max_compressed_len"])
    new_completed = min(
        (inp["input_pos"] + inp["kv_seq"].shape[0]) // ratio,
        inp["max_compressed_len"],
    )
    flat_position_ids = inp["position_ids"].flatten()
    first_position_id = int(flat_position_ids[0].item())
    for row_idx in range(old_completed, new_completed):
        row_token_offset = row_idx * ratio - inp["input_pos"]
        if 0 <= row_token_offset < flat_position_ids.numel():
            row_position_id = int(flat_position_ids[row_token_offset].item())
        else:
            row_position_id = first_position_id + row_token_offset
        row = _legacy_compressed_row(
            kv_cache,
            gate_cache,
            inp["seq_idx"],
            row_idx,
            row_position_id,
            inp["cu_num_pages"],
            inp["cache_loc"],
            inp["ape"],
            inp["norm_weight"],
            inp["cos_table"],
            inp["sin_table"],
            inp["eps"],
            inp["rope_dim"],
            ratio,
            inp["head_dim"],
            inp["kv_seq"].dtype,
        )
        page_id, page_offset = M._host_page_id_and_offset(
            mhc_cache,
            inp["seq_idx"],
            row_idx * ratio,
            inp["cu_num_pages"],
            inp["cache_loc"],
        )
        mhc_cache[page_id, page_offset].copy_(row.to(mhc_cache.dtype))


def _build_case(
    ratio,
    input_pos,
    seq_len,
    max_compressed_len,
    seq_idx,
    device,
    head_dim=128,
):
    torch.manual_seed(7100 + ratio + input_pos)
    rope_dim = 64
    channels = 2 if ratio == 4 else 1
    state_dim = channels * head_dim
    tokens_per_block = 7
    logical_capacity = max(input_pos + seq_len, max_compressed_len * ratio)
    pages_per_seq = math.ceil(logical_capacity / tokens_per_block)
    num_seq = 2
    total_pages = num_seq * pages_per_seq
    cu_num_pages = torch.arange(num_seq + 1, dtype=torch.long) * pages_per_seq
    cache_loc = torch.randperm(total_pages, dtype=torch.long)

    kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)
    gate_cache = torch.randn_like(kv_cache)
    mhc_cache = torch.randn(
        total_pages,
        tokens_per_block,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_seq = torch.randn(seq_len, state_dim, dtype=torch.bfloat16, device=device)
    gate_seq = torch.randn_like(kv_seq)
    ape = torch.randn(ratio, state_dim, device=device)
    norm_weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    max_position_id = 256 + max(0, seq_len - 1) * 3
    cos_table = torch.randn(max(1024, max_position_id + 1), rope_dim // 2, device=device)
    sin_table = torch.randn_like(cos_table)
    # A non-unit stride makes the in-chunk lookup distinguishable from the
    # first-id extrapolation used when a row starts before this chunk.
    position_ids = (torch.arange(seq_len, dtype=torch.long, device=device) * 3 + 256).unsqueeze(0)
    return {
        "ratio": ratio,
        "input_pos": input_pos,
        "max_compressed_len": max_compressed_len,
        "seq_idx": seq_idx,
        "head_dim": head_dim,
        "rope_dim": rope_dim,
        "eps": 1e-6,
        "cu_num_pages": cu_num_pages,
        "cache_loc": cache_loc,
        "kv_cache": kv_cache,
        "gate_cache": gate_cache,
        "mhc_cache": mhc_cache,
        "kv_seq": kv_seq,
        "gate_seq": gate_seq,
        "position_ids": position_ids,
        "ape": ape,
        "norm_weight": norm_weight,
        "cos_table": cos_table,
        "sin_table": sin_table,
    }


def _vectorized_update(
    inp,
    kv_cache,
    gate_cache,
    mhc_cache,
    precomputed_initial_rows=None,
    raw_cache_rows_already_written=False,
):
    M._update_compressed_paged_caches(
        inp["kv_seq"],
        inp["gate_seq"],
        inp["position_ids"],
        inp["ape"],
        inp["norm_weight"],
        inp["cos_table"],
        inp["sin_table"],
        inp["seq_idx"],
        inp["input_pos"],
        inp["cu_num_pages"],
        inp["cache_loc"],
        mhc_cache,
        kv_cache,
        gate_cache,
        inp["eps"],
        inp["rope_dim"],
        inp["ratio"],
        inp["max_compressed_len"],
        precomputed_initial_rows=precomputed_initial_rows,
        raw_cache_rows_already_written=raw_cache_rows_already_written,
    )


def _source_compressed_rows(inp, max_compressed_len):
    return M._build_full_compressed_kv(
        inp["kv_seq"].unsqueeze(0),
        inp["gate_seq"].unsqueeze(0),
        inp["ape"],
        inp["norm_weight"],
        inp["cos_table"],
        inp["sin_table"],
        inp["position_ids"],
        inp["eps"],
        inp["rope_dim"],
        inp["ratio"],
        max_compressed_len,
    ).squeeze(0)


@pytest.mark.parametrize(
    "tokens_per_block,input_pos,num_rows",
    [
        (4, 3, 17),
        (32, 0, 1004),
        (32, 31, 1004),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_vectorized_contiguous_paged_write_matches_page_loop(
    tokens_per_block,
    input_pos,
    num_rows,
):
    """One indexed write preserves shuffled-page layout and untouched cache bytes."""
    torch.manual_seed(8100 + tokens_per_block + input_pos)
    num_seq = 2
    pages_per_seq = math.ceil((input_pos + num_rows) / tokens_per_block) + 1
    total_pages = num_seq * pages_per_seq
    cu_num_pages = torch.arange(num_seq + 1, dtype=torch.long) * pages_per_seq
    cache_loc = torch.randperm(total_pages, dtype=torch.long)
    cache = torch.randn(
        total_pages,
        tokens_per_block,
        64,
        dtype=torch.bfloat16,
        device="cuda",
    )
    values = torch.randn(num_rows, 64, dtype=torch.bfloat16, device="cuda")
    expected = cache.clone()
    actual = cache.clone()

    _scalar_write_paged_cache_rows(
        values,
        expected,
        1,
        input_pos,
        cu_num_pages,
        cache_loc,
    )
    M._write_paged_cache_rows(
        values,
        actual,
        1,
        input_pos,
        cu_num_pages,
        cache_loc,
    )

    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    "state_dims,dtypes",
    [
        (
            (512, 1024, 1024, 256, 256),
            (
                torch.bfloat16,
                torch.float32,
                torch.float32,
                torch.float32,
                torch.float32,
            ),
        ),
        (
            (512, 512, 512),
            (torch.bfloat16, torch.float32, torch.float32),
        ),
    ],
    ids=["r4-five-caches", "r128-three-caches"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_initial_prefill_multi_cache_write_matches_reference(state_dims, dtypes):
    """The fused R4/R128 layouts preserve all prompt rows and untouched page bytes."""
    torch.manual_seed(8200 + len(state_dims))
    tokens_per_block = 32
    num_rows = 1004
    pages_per_seq = math.ceil(num_rows / tokens_per_block) + 2
    num_seq = 2
    total_pages = num_seq * pages_per_seq
    cu_num_pages = torch.arange(num_seq + 1, dtype=torch.long) * pages_per_seq
    cache_loc = torch.randperm(total_pages, dtype=torch.long)
    seq_idx = 1

    actual = [
        torch.randn(
            total_pages,
            tokens_per_block,
            state_dim,
            dtype=dtype,
            device="cuda",
        )
        for state_dim, dtype in zip(state_dims, dtypes)
    ]
    expected = [cache.clone() for cache in actual]
    values = [
        torch.randn(num_rows, state_dim, dtype=dtype, device="cuda")
        for state_dim, dtype in zip(state_dims, dtypes)
    ]

    for cache, rows in zip(expected, values):
        M._write_paged_cache_rows(
            rows,
            cache,
            seq_idx,
            0,
            cu_num_pages,
            cache_loc,
        )
    used_fused = M._try_fused_initial_prefill_cache_write(
        actual,
        values,
        seq_idx,
        cu_num_pages.cuda(),
        cache_loc.cuda(),
        cu_num_pages,
        cache_loc,
    )

    assert used_fused
    for fused_cache, reference_cache in zip(actual, expected):
        assert torch.equal(fused_cache, reference_cache)


def test_initial_prefill_multi_cache_write_fallback_does_not_mutate():
    """Unsupported page layouts return cleanly so the caller can run the reference path."""
    caches = [torch.randn(4, 7, 8), torch.randn(4, 7, 16)]
    before = [cache.clone() for cache in caches]
    values = [torch.randn(13, 8), torch.randn(13, 16)]
    cu_num_pages = torch.tensor([0, 4], dtype=torch.long)
    cache_loc = torch.arange(4, dtype=torch.long)

    used_fused = M._try_fused_initial_prefill_cache_write(
        caches,
        values,
        0,
        cu_num_pages,
        cache_loc,
        cu_num_pages,
        cache_loc,
    )

    assert not used_fused
    for cache, original in zip(caches, before):
        assert torch.equal(cache, original)


@pytest.mark.parametrize(
    "ratio,input_pos,seq_len,max_compressed_len,seq_idx",
    [
        (4, 6, 15, 6, 1),
        (128, 96, 300, 3, 0),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_vectorized_prefill_cache_update_matches_row_at_a_time_reference(
    ratio, input_pos, seq_len, max_compressed_len, seq_idx
):
    """Pin R4/R128, continuation offsets, page boundaries, and shuffled page maps."""
    inp = _build_case(ratio, input_pos, seq_len, max_compressed_len, seq_idx, "cuda")
    legacy = tuple(inp[name].clone() for name in ("kv_cache", "gate_cache", "mhc_cache"))
    vectorized = tuple(inp[name].clone() for name in ("kv_cache", "gate_cache", "mhc_cache"))

    _legacy_update(inp, *legacy)
    _vectorized_update(inp, *vectorized)

    for actual, expected in zip(vectorized, legacy):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_vectorized_prefill_position_ids_keep_legacy_chunk_boundary_rule(monkeypatch):
    """Rows before the chunk extrapolate; rows in the chunk gather their exact IDs."""
    inp = _build_case(
        4,
        input_pos=6,
        seq_len=11,
        max_compressed_len=5,
        seq_idx=0,
        device="cpu",
    )
    captured = {}

    def fake_compress(*args, **_kwargs):
        row_idx = args[3]
        captured["row_idx"] = row_idx.clone()
        captured["row_position_id"] = args[4].clone()
        head_dim = args[14]
        return torch.zeros(row_idx.numel(), head_dim, dtype=inp["kv_seq"].dtype)

    monkeypatch.setattr(M, "_compressed_rows_from_paged_state", fake_compress)
    _vectorized_update(
        inp,
        inp["kv_cache"].clone(),
        inp["gate_cache"].clone(),
        inp["mhc_cache"].clone(),
    )

    # Completed rows are 1..3. Row 1 starts two tokens before the chunk and is
    # extrapolated from first_position_id; rows 2 and 3 gather offsets 2 and 6.
    torch.testing.assert_close(captured["row_idx"], torch.tensor([1, 2, 3]))
    torch.testing.assert_close(captured["row_position_id"], torch.tensor([254, 262, 274]))


@pytest.mark.parametrize(
    "ratio,max_compressed_len",
    [
        (4, 512),
        (128, 16),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_initial_source_rows_match_paged_reconstruction_cache_layout(
    ratio,
    max_compressed_len,
):
    """Direct source rows preserve production R4/R128 paged-cache bytes exactly."""
    inp = _build_case(
        ratio,
        input_pos=0,
        seq_len=1004,
        max_compressed_len=max_compressed_len,
        seq_idx=1,
        device="cuda",
        head_dim=512,
    )
    # The cached op widens raw activation-dtype compressor rows once before any
    # prefill consumer (idea_0092). Mirror that production boundary so the direct
    # source path and paged-cache reconstruction use the same FP32 inputs.
    inp["kv_seq"] = inp["kv_seq"].float()
    inp["gate_seq"] = inp["gate_seq"].float()
    paged = tuple(inp[name].clone() for name in ("kv_cache", "gate_cache", "mhc_cache"))
    reused = tuple(inp[name].clone() for name in ("kv_cache", "gate_cache", "mhc_cache"))

    source_rows = _source_compressed_rows(inp, max_compressed_len)
    _vectorized_update(inp, *paged)
    _vectorized_update(
        inp,
        *reused,
        precomputed_initial_rows=source_rows,
    )

    for actual, expected in zip(reused, paged):
        assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_preexisting_raw_cache_rows_skip_duplicate_writes(monkeypatch):
    """The compressed-cache update writes MHC rows without rewriting fused raw rows."""
    inp = _build_case(
        4,
        input_pos=0,
        seq_len=20,
        max_compressed_len=8,
        seq_idx=0,
        device="cuda",
    )
    names = ("kv_cache", "gate_cache", "mhc_cache")
    expected = tuple(inp[name].clone() for name in names)
    actual = tuple(inp[name].clone() for name in names)
    source_rows = _source_compressed_rows(inp, inp["max_compressed_len"])

    _vectorized_update(
        inp,
        *expected,
        precomputed_initial_rows=source_rows,
    )
    M._write_paged_cache_rows(
        inp["kv_seq"],
        actual[0],
        inp["seq_idx"],
        0,
        inp["cu_num_pages"],
        inp["cache_loc"],
    )
    M._write_paged_cache_rows(
        inp["gate_seq"],
        actual[1],
        inp["seq_idx"],
        0,
        inp["cu_num_pages"],
        inp["cache_loc"],
    )

    def unexpected_raw_write(*_args, **_kwargs):
        raise AssertionError("raw cache rows were written twice")

    monkeypatch.setattr(M, "_write_paged_cache_rows", unexpected_raw_write)
    _vectorized_update(
        inp,
        *actual,
        precomputed_initial_rows=source_rows,
        raw_cache_rows_already_written=True,
    )

    for fused_cache, reference_cache in zip(actual, expected):
        assert torch.equal(fused_cache, reference_cache)


def test_preexisting_raw_cache_rows_reject_continuation():
    """A continuation cannot claim that its raw rows came from the fresh-prefill path."""
    inp = _build_case(
        4,
        input_pos=4,
        seq_len=4,
        max_compressed_len=4,
        seq_idx=0,
        device="cpu",
    )
    with pytest.raises(ValueError, match="require input_pos == 0"):
        _vectorized_update(
            inp,
            inp["kv_cache"].clone(),
            inp["gate_cache"].clone(),
            inp["mhc_cache"].clone(),
            raw_cache_rows_already_written=True,
        )


@pytest.mark.parametrize(
    "ratio,seq_len,max_compressed_len",
    [
        (4, 20, 8),
        (128, 257, 8),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_initial_attention_reused_rows_match_source_op(
    ratio,
    seq_len,
    max_compressed_len,
):
    """Reusing source-built rows preserves initial attention output exactly."""
    inp = _build_case(
        ratio,
        input_pos=0,
        seq_len=seq_len,
        max_compressed_len=max_compressed_len,
        seq_idx=0,
        device="cuda",
        head_dim=512,
    )
    window_size = 4
    batch_size = 1
    num_heads = 2
    q = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        inp["head_dim"],
        dtype=torch.bfloat16,
        device="cuda",
    )
    kv = torch.randn(
        batch_size,
        seq_len,
        inp["head_dim"],
        dtype=torch.bfloat16,
        device="cuda",
    )
    attn_sink = torch.randn(num_heads, dtype=torch.bfloat16, device="cuda")
    topk = M._build_placeholder_topk_idxs(
        window_size,
        ratio,
        batch_size,
        seq_len,
        max_compressed_len,
        q.device,
    )
    source_rows = _source_compressed_rows(inp, max_compressed_len)
    initial_kv = torch.cat((kv, source_rows.unsqueeze(0).to(kv.dtype)), dim=1)
    actual = M._deepseek_v4_sparse_attention(
        q,
        initial_kv,
        attn_sink,
        topk,
        1.0,
    )

    empty_indexer_q = q.new_empty(batch_size, seq_len, 0, 0)
    empty_indexer_weights = q.new_empty(batch_size, seq_len, 0)
    empty_indexer_state = q.new_empty(batch_size, seq_len, 0)
    expected = torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        topk,
        inp["kv_seq"].unsqueeze(0),
        inp["gate_seq"].unsqueeze(0),
        inp["ape"],
        inp["norm_weight"],
        inp["cos_table"],
        inp["sin_table"],
        inp["position_ids"],
        empty_indexer_q,
        empty_indexer_weights,
        empty_indexer_state,
        empty_indexer_state,
        q.new_empty(0, 0),
        q.new_empty(0),
        1.0,
        window_size=window_size,
        compress_ratio=ratio,
        max_compressed_len=max_compressed_len,
        rope_dim=inp["rope_dim"],
        rms_norm_eps=inp["eps"],
    )
    assert torch.equal(actual, expected)
