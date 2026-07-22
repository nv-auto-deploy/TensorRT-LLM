# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 sparse-attention gather/assemble tests.

Bit-exactness of the vectorized paged-gather helpers and the fused decode
selected-KV assembly in ``deepseek_v4_sparse_attention``.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

_requires_cuda_triton = pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()), reason="requires CUDA + triton"
)


# ---------------------------------------------------------------------------
# Frozen pre-vectorization loop references (independent of the module).
# ---------------------------------------------------------------------------
def _host_page_id_and_offset(cache, seq_idx, logical_pos, cu_num_pages_host, cache_loc_host):
    if logical_pos < 0:
        raise ValueError(f"logical_pos must be non-negative, got {logical_pos}")
    tokens_per_block = int(cache.shape[1])
    page_ordinal = logical_pos // tokens_per_block
    page_offset = logical_pos % tokens_per_block
    page_start = int(cu_num_pages_host[seq_idx].item())
    page_end = int(cu_num_pages_host[seq_idx + 1].item())
    page_table_idx = page_start + page_ordinal
    if page_table_idx >= page_end:
        raise ValueError(f"seq {seq_idx} pos {logical_pos} beyond active pages")
    return int(cache_loc_host[page_table_idx].item()), page_offset


def _host_position_is_valid(cache, seq_idx, logical_pos, cu_num_pages_host):
    if logical_pos < 0:
        return False
    page_ordinal = logical_pos // int(cache.shape[1])
    page_start = int(cu_num_pages_host[seq_idx].item())
    page_end = int(cu_num_pages_host[seq_idx + 1].item())
    return page_start + page_ordinal < page_end


def _compressed_row_from_paged_state(
    compressor_kv_cache,
    compressor_gate_cache,
    seq_idx,
    row_idx,
    row_position_id,
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
    state_dim,
    dtype,
    rotate=False,
):
    del state_dim
    row_idx_tensor = torch.tensor([row_idx], dtype=torch.long, device=compressor_kv_cache.device)
    position_id_tensor = torch.tensor(
        [row_position_id], dtype=torch.long, device=compressor_kv_cache.device
    )
    return M._compressed_rows_from_paged_state(
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


def _ref_gather_paged_rows_from_positions(
    cache, seq_idx, positions, cu_num_pages_host, cache_loc_host, dtype
):
    positions_host = positions.detach().cpu().to(torch.long).flatten()
    rows = []
    valid_rows = []
    row_width = cache.shape[-1]
    zero = cache.new_zeros(row_width)
    for logical_pos_tensor in positions_host:
        logical_pos = int(logical_pos_tensor.item())
        is_valid = _host_position_is_valid(cache, seq_idx, logical_pos, cu_num_pages_host)
        valid_rows.append(is_valid)
        if is_valid:
            page_id, page_offset = _host_page_id_and_offset(
                cache, seq_idx, logical_pos, cu_num_pages_host, cache_loc_host
            )
            rows.append(cache[page_id, page_offset].to(dtype))
        else:
            rows.append(zero.to(dtype))

    if rows:
        gathered = torch.stack(rows, dim=0)
    else:
        gathered = cache.new_empty(0, row_width, dtype=dtype)
    valid = torch.tensor(valid_rows, dtype=torch.bool, device=positions.device)
    return gathered.view(*positions.shape, row_width), valid.view(positions.shape)


def _ref_indexer_row_stack(
    indexer_compressor_kv_cache,
    indexer_compressor_gate_cache,
    seq_idx,
    visible_len,
    query_pos,
    query_position_id,
    cu_num_pages_host,
    cache_loc_host,
    indexer_compressor_ape,
    indexer_compressor_norm_weight,
    cos_table,
    sin_table,
    rms_norm_eps,
    rope_dim,
    index_head_dim,
    dtype,
):
    state_dim = int(indexer_compressor_kv_cache.shape[-1])
    return torch.stack(
        [
            _compressed_row_from_paged_state(
                indexer_compressor_kv_cache,
                indexer_compressor_gate_cache,
                seq_idx,
                row_idx,
                query_position_id - (query_pos - row_idx * 4),
                cu_num_pages_host,
                cache_loc_host,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                cos_table,
                sin_table,
                rms_norm_eps,
                rope_dim,
                4,
                index_head_dim,
                state_dim,
                dtype,
                rotate=True,
            )
            for row_idx in range(visible_len)
        ],
        dim=0,
    )


def _build_page_table(num_seq, tokens_per_block, pages_per_seq, seed, device):
    # Shuffled physical pool so an identity (non-paged) translation is detected.
    g = torch.Generator(device="cpu").manual_seed(seed)
    cu_num_pages = torch.arange(0, (num_seq + 1) * pages_per_seq, pages_per_seq, dtype=torch.long)
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, generator=g, dtype=torch.long)
    return cu_num_pages.to(device), cache_loc.to(device), total_pages


# ---------------------------------------------------------------------------
# Vectorized prefill gather (_gather_paged_rows_from_positions & friends).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tokens_per_block", [1, 32])
@pytest.mark.parametrize("pos_shape", [(0,), (7,), (3, 4)])
def test_gather_paged_rows_bit_exact(device, tokens_per_block, pos_shape):
    torch.manual_seed(hash((tokens_per_block, pos_shape)) & 0xFFFF)
    num_seq = 3
    pages_per_seq = 6
    cu_num_pages, cache_loc, total_pages = _build_page_table(
        num_seq, tokens_per_block, pages_per_seq, seed=1234, device=device
    )
    head_width = 6
    cache = torch.randn(
        total_pages, tokens_per_block, head_width, dtype=torch.float32, device=device
    )

    seq_capacity = pages_per_seq * tokens_per_block
    # Positions span valid, negative (invalid), and out-of-page (invalid) values.
    high = seq_capacity + 3
    n = 1
    for s in pos_shape:
        n *= s
    if n == 0:
        positions = torch.empty(pos_shape, dtype=torch.long, device=device)
    else:
        positions = torch.randint(-3, high, pos_shape, dtype=torch.long, device=device)

    for seq_idx in range(num_seq):
        ref_rows, ref_valid = _ref_gather_paged_rows_from_positions(
            cache, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16
        )
        new_rows, new_valid = M._gather_paged_rows_from_positions(
            cache, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16
        )
        assert new_rows.shape == ref_rows.shape, (seq_idx, ref_rows.shape, new_rows.shape)
        assert new_valid.shape == ref_valid.shape
        assert new_valid.device == ref_valid.device
        assert torch.equal(new_valid, ref_valid), f"valid mismatch seq={seq_idx}"
        assert torch.equal(new_rows, ref_rows), f"rows mismatch seq={seq_idx}"


@pytest.mark.parametrize("device", _DEVICES)
def test_gather_paged_rows_all_invalid_and_all_valid(device):
    torch.manual_seed(7)
    num_seq, tokens_per_block, pages_per_seq = 2, 4, 5
    cu_num_pages, cache_loc, total_pages = _build_page_table(
        num_seq, tokens_per_block, pages_per_seq, seed=99, device=device
    )
    cache = torch.randn(total_pages, tokens_per_block, 8, dtype=torch.float32, device=device)

    all_invalid = torch.full((6,), -1, dtype=torch.long, device=device)
    all_valid = torch.arange(pages_per_seq * tokens_per_block, dtype=torch.long, device=device)
    for positions in (all_invalid, all_valid):
        for seq_idx in range(num_seq):
            ref_rows, ref_valid = _ref_gather_paged_rows_from_positions(
                cache, seq_idx, positions, cu_num_pages, cache_loc, torch.float32
            )
            new_rows, new_valid = M._gather_paged_rows_from_positions(
                cache, seq_idx, positions, cu_num_pages, cache_loc, torch.float32
            )
            assert torch.equal(new_valid, ref_valid)
            torch.testing.assert_close(new_rows, ref_rows, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="rotate=True path uses CUDA triton ops")
@pytest.mark.parametrize("query_pos", [3, 4, 128])
def test_indexer_row_stack_bit_exact(query_pos):
    device = "cuda"
    torch.manual_seed(query_pos + 3)

    index_head_dim = 128
    state_dim = 2 * index_head_dim
    rope_dim = 64
    compress_ratio = 4
    rms_norm_eps = 1e-6

    max_compressed_len = 64
    visible_len = min((query_pos + 1) // compress_ratio, max_compressed_len)
    assert visible_len > 0

    tokens_per_block = 16
    pages_per_seq = (query_pos + 1 + tokens_per_block - 1) // tokens_per_block + 2
    num_seq = 3
    cu_num_pages, cache_loc, total_pages = _build_page_table(
        num_seq, tokens_per_block, pages_per_seq, seed=query_pos, device=device
    )

    kv_cache = torch.randn(
        total_pages, tokens_per_block, state_dim, dtype=torch.bfloat16, device=device
    )
    gate_cache = torch.randn(
        total_pages, tokens_per_block, state_dim, dtype=torch.bfloat16, device=device
    )
    ape = torch.randn(compress_ratio, state_dim, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(index_head_dim, dtype=torch.bfloat16, device=device)
    cos_table = torch.randn(4096, rope_dim // 2, dtype=torch.float32, device=device)
    sin_table = torch.randn(4096, rope_dim // 2, dtype=torch.float32, device=device)

    query_position_id = query_pos
    dtype = torch.bfloat16

    for seq_idx in range(num_seq):
        ref_k = _ref_indexer_row_stack(
            kv_cache,
            gate_cache,
            seq_idx,
            visible_len,
            query_pos,
            query_position_id,
            cu_num_pages,
            cache_loc,
            ape,
            norm_weight,
            cos_table,
            sin_table,
            rms_norm_eps,
            rope_dim,
            index_head_dim,
            dtype,
        )

        row_idx = torch.arange(visible_len, dtype=torch.long, device=device)
        seq_idx_rows = torch.full((visible_len,), seq_idx, dtype=torch.long, device=device)
        row_position_id_rows = query_position_id - (query_pos - row_idx * compress_ratio)
        new_k = M._batched_compressed_rows_from_paged_state(
            kv_cache,
            gate_cache,
            seq_idx_rows,
            row_idx,
            row_position_id_rows,
            cu_num_pages,
            cache_loc,
            ape,
            norm_weight,
            cos_table,
            sin_table,
            rms_norm_eps,
            rope_dim,
            compress_ratio,
            index_head_dim,
            dtype,
            rotate=True,
        )
        assert new_k.shape == ref_k.shape, (seq_idx, ref_k.shape, new_k.shape)
        assert torch.equal(new_k, ref_k), (
            f"index_k row-stack mismatch seq={seq_idx}, qpos={query_pos}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="rotate=True path uses CUDA triton ops")
@pytest.mark.parametrize("query_pos", [0, 3, 67])
def test_select_ratio4_indexer_rows_short_seq(query_pos):
    # query_pos=0: -1 sentinel; 3: visible_len=1 < topk (pad); 67: visible_len=17 > topk=8
    # (discriminating selection vs the loop reference).
    device = "cuda"
    torch.manual_seed(100 + query_pos)

    index_head_dim = 128
    state_dim = 2 * index_head_dim
    rope_dim = 64
    rms_norm_eps = 1e-6
    max_compressed_len = 64
    index_topk = 8

    tokens_per_block = 16
    pages_per_seq = 8
    num_seq = 2
    cu_num_pages, cache_loc, total_pages = _build_page_table(
        num_seq, tokens_per_block, pages_per_seq, seed=query_pos + 5, device=device
    )
    kv_cache = torch.randn(
        total_pages, tokens_per_block, state_dim, dtype=torch.bfloat16, device=device
    )
    gate_cache = torch.randn(
        total_pages, tokens_per_block, state_dim, dtype=torch.bfloat16, device=device
    )
    ape = torch.randn(4, state_dim, dtype=torch.bfloat16, device=device)
    norm_weight = torch.randn(index_head_dim, dtype=torch.bfloat16, device=device)
    cos_table = torch.randn(4096, rope_dim // 2, dtype=torch.float32, device=device)
    sin_table = torch.randn(4096, rope_dim // 2, dtype=torch.float32, device=device)

    n_index_heads = 4
    q_index = torch.randn(n_index_heads, index_head_dim, dtype=torch.bfloat16, device=device)
    indexer_weights = torch.randn(n_index_heads, dtype=torch.bfloat16, device=device)

    seq_idx = 0
    out = M._select_ratio4_indexer_rows(
        q_index,
        indexer_weights,
        kv_cache,
        gate_cache,
        seq_idx,
        query_pos,
        query_pos,
        index_topk,
        cu_num_pages,
        cache_loc,
        ape,
        norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        max_compressed_len,
    )
    visible_len = min((query_pos + 1) // 4, max_compressed_len)
    if visible_len <= 0:
        assert torch.equal(out, torch.full((index_topk,), -1, dtype=torch.int64, device=device))
    else:
        ref_k = _ref_indexer_row_stack(
            kv_cache,
            gate_cache,
            seq_idx,
            visible_len,
            query_pos,
            query_pos,
            cu_num_pages,
            cache_loc,
            ape,
            norm_weight,
            cos_table,
            sin_table,
            rms_norm_eps,
            rope_dim,
            index_head_dim,
            torch.bfloat16,
        )
        ref_score = torch.matmul(q_index, ref_k.transpose(-1, -2)).float()
        ref_score = (ref_score.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=0)
        topk_count = min(index_topk, visible_len)
        ref_sel = ref_score.topk(topk_count, dim=-1).indices.to(torch.int64)
        if topk_count < index_topk:
            pad = torch.full((index_topk - topk_count,), -1, dtype=ref_sel.dtype, device=device)
            ref_sel = torch.cat((ref_sel, pad), dim=0)
        assert torch.equal(out, ref_sel), f"selection mismatch qpos={query_pos}"


# ---------------------------------------------------------------------------
# Fused decode selected-KV paged assembly (_fused_assemble_selected_kv).
# ---------------------------------------------------------------------------
def _reference_assemble(
    swa_cache,
    mhc_cache,
    selected_rows,
    compressed_valid,
    input_pos,
    seq_idx,
    cu_num_pages,
    cache_loc,
    window_size,
    compress_ratio,
    dtype,
):
    """The eager gather/cat/where chain the fused kernel replaces (module helpers)."""
    local_kv, local_valid = M._decode_local_cache_rows(
        swa_cache, seq_idx, input_pos, cu_num_pages, cache_loc, window_size, dtype
    )
    compressed_positions = selected_rows.clamp(min=0) * compress_ratio
    compressed_kv, page_valid = M._decode_cache_rows_from_positions(
        mhc_cache, seq_idx, compressed_positions, cu_num_pages, cache_loc, dtype
    )
    compressed_valid_full = compressed_valid & page_valid & (selected_rows >= 0)
    selected_kv = torch.cat((local_kv, compressed_kv), dim=1)
    valid_rows = torch.cat((local_valid, compressed_valid_full), dim=1)
    rel_topk = torch.arange(selected_kv.shape[1], dtype=torch.int64, device=selected_kv.device)
    rel_topk = rel_topk.view(1, -1).expand(selected_kv.shape[0], -1)
    rel_topk = torch.where(valid_rows, rel_topk, torch.full_like(rel_topk, -1))
    return selected_kv, rel_topk


@_requires_cuda_triton
@pytest.mark.parametrize("head_dim", [8, 128, 576])
@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("topk", [0, 6])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_fused_assemble_matches_eager(head_dim, compress_ratio, topk, cache_dtype):
    torch.manual_seed(20260701 + head_dim + compress_ratio + topk)
    device = "cuda"
    dtype = torch.bfloat16
    window_size = 4
    tokens_per_block = 8

    pages_per_seq = [24, 30]
    num_seq = len(pages_per_seq)
    cu_num_pages = torch.tensor(
        [0, *torch.cumsum(torch.tensor(pages_per_seq), 0).tolist()],
        dtype=torch.long,
        device=device,
    )
    total_slots = int(cu_num_pages[-1])
    num_physical = total_slots + 11
    cache_loc = torch.randperm(num_physical, device=device)[:total_slots].to(torch.long)

    swa_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(cache_dtype)
    mhc_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(cache_dtype)

    seq_idx = torch.arange(num_seq, dtype=torch.long, device=device)
    input_pos = torch.tensor([37, 2], dtype=torch.long, device=device)

    if topk == 0:
        selected_rows = torch.empty(num_seq, 0, dtype=torch.int64, device=device)
        compressed_valid = torch.empty(num_seq, 0, dtype=torch.bool, device=device)
    else:
        # Mix of in-range rows, pad rows (-1), and beyond-page-extent rows.
        selected_rows = torch.randint(0, 40, (num_seq, topk), dtype=torch.int64, device=device)
        selected_rows[0, 0] = -1
        if topk > 3:
            selected_rows[0, 3] = -1
            selected_rows[1, 2] = 500
        compressed_valid = torch.rand(num_seq, topk, device=device) > 0.3

    ref_kv, ref_relidx = _reference_assemble(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    out_kv, out_relidx = M._fused_assemble_selected_kv(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )

    assert out_kv.shape == ref_kv.shape
    assert out_relidx.shape == ref_relidx.shape
    case = f"head_dim={head_dim} ratio={compress_ratio} topk={topk} cache_dtype={cache_dtype}"
    assert torch.equal(out_relidx, ref_relidx), f"rel_topk diverged: {case}"
    assert torch.equal(out_kv, ref_kv), f"selected_kv diverged: {case}"


@_requires_cuda_triton
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_fused_assemble_large_decode_position(compress_ratio):
    # input_pos ~1000, many pages, wide top-k: guards int64 address arithmetic.
    torch.manual_seed(70123 + compress_ratio)
    device = "cuda"
    dtype = torch.bfloat16
    head_dim, window_size, tokens_per_block = 576, 128, 64
    max_compressed_len = 512

    pages_per_seq = [24, 20]
    num_seq = len(pages_per_seq)
    cu_num_pages = torch.tensor(
        [0, *torch.cumsum(torch.tensor(pages_per_seq), 0).tolist()],
        dtype=torch.long,
        device=device,
    )
    total_slots = int(cu_num_pages[-1])
    num_physical = total_slots + 7
    cache_loc = torch.randperm(num_physical, device=device)[:total_slots].to(torch.long)
    swa_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(dtype)
    mhc_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(dtype)

    seq_idx = torch.arange(num_seq, dtype=torch.long, device=device)
    input_pos = torch.tensor([1005, 992], dtype=torch.long, device=device)

    if compress_ratio == 4:
        topk = 384
        selected_rows = torch.randint(0, 250, (num_seq, topk), dtype=torch.int64, device=device)
        selected_rows[:, ::37] = -1
        compressed_valid = torch.rand(num_seq, topk, device=device) > 0.2
    else:
        topk = max_compressed_len
        rows = torch.arange(max_compressed_len, dtype=torch.int64, device=device)
        selected_rows = rows.view(1, -1).expand(num_seq, -1).contiguous()
        visible = ((input_pos + 1) // compress_ratio).clamp(max=max_compressed_len)
        compressed_valid = selected_rows < visible.unsqueeze(1)

    ref_kv, ref_relidx = _reference_assemble(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    out_kv, out_relidx = M._fused_assemble_selected_kv(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    assert torch.equal(out_relidx, ref_relidx), f"rel_topk diverged: ratio={compress_ratio}"
    assert torch.equal(out_kv, ref_kv), f"selected_kv diverged: ratio={compress_ratio}"


@_requires_cuda_triton
@pytest.mark.parametrize(
    "window_size,tokens_per_block,max_compressed_len",
    [(128, 64, 512), (4, 8, 40)],
)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_fused_assemble_dense_in_kernel_matches_materialized(
    window_size, tokens_per_block, max_compressed_len, cache_dtype
):
    # Dense (ratio-128): selected_rows=None derives row ids + visibility in-kernel;
    # must equal both the materialized chain through the kernel and the eager reference.
    compress_ratio = 128
    torch.manual_seed(90_000 + window_size + max_compressed_len)
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = 64

    # Crosses ratio-128 row boundaries, short histories, and negative (padded) rows.
    input_pos = torch.tensor(
        [126, 127, 128, 129, 255, 256, 0, 2, -1, -130, 1005],
        dtype=torch.long,
        device=device,
    )
    num_seq = int(input_pos.shape[0])
    pages_per_seq = [(1006 + tokens_per_block) // tokens_per_block + 1] * num_seq
    cu_num_pages = torch.tensor(
        [0, *torch.cumsum(torch.tensor(pages_per_seq), 0).tolist()],
        dtype=torch.long,
        device=device,
    )
    total_slots = int(cu_num_pages[-1])
    num_physical = total_slots + 5
    cache_loc = torch.randperm(num_physical, device=device)[:total_slots].to(torch.long)
    swa_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(cache_dtype)
    mhc_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(cache_dtype)
    seq_idx = torch.arange(num_seq, dtype=torch.long, device=device)

    rows = torch.arange(max_compressed_len, dtype=torch.long, device=device)
    selected_rows = rows.view(1, -1).expand(num_seq, -1)
    compressed_len = ((input_pos + 1) // compress_ratio).clamp(max=max_compressed_len)
    compressed_valid = selected_rows < compressed_len.unsqueeze(1)

    ref_kv, ref_relidx = _reference_assemble(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    mat_kv, mat_relidx = M._fused_assemble_selected_kv(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    dense_kv, dense_relidx = M._fused_assemble_selected_kv(
        swa_cache,
        mhc_cache,
        None,
        None,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
        dense_num_rows=max_compressed_len,
    )

    assert dense_kv.shape == ref_kv.shape
    assert dense_relidx.shape == ref_relidx.shape
    case = f"W={window_size} tpb={tokens_per_block} m={max_compressed_len} dtype={cache_dtype}"
    assert torch.equal(dense_relidx, ref_relidx), f"dense rel_topk diverged from eager: {case}"
    assert torch.equal(dense_kv, ref_kv), f"dense selected_kv diverged from eager: {case}"
    assert torch.equal(dense_relidx, mat_relidx)
    assert torch.equal(dense_kv, mat_kv)


@_requires_cuda_triton
def test_fused_assemble_attend_output_matches_eager():
    # Exercises the downstream _decode_attention_from_selected consumer (bf16
    # head_dim=128 routes through the fused Triton attend) on fused-assembled inputs.
    torch.manual_seed(4242)
    device = "cuda"
    dtype = torch.bfloat16
    head_dim, window_size, tokens_per_block, topk, compress_ratio = 128, 4, 8, 12, 4
    num_heads = 8
    softmax_scale = head_dim**-0.5

    pages_per_seq = [24, 30]
    num_seq = len(pages_per_seq)
    cu_num_pages = torch.tensor(
        [0, *torch.cumsum(torch.tensor(pages_per_seq), 0).tolist()],
        dtype=torch.long,
        device=device,
    )
    total_slots = int(cu_num_pages[-1])
    num_physical = total_slots + 11
    cache_loc = torch.randperm(num_physical, device=device)[:total_slots].to(torch.long)
    swa_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(dtype)
    mhc_cache = torch.randn(num_physical, tokens_per_block, head_dim, device=device).to(dtype)

    seq_idx = torch.arange(num_seq, dtype=torch.long, device=device)
    input_pos = torch.tensor([37, 20], dtype=torch.long, device=device)
    selected_rows = torch.randint(0, 20, (num_seq, topk), dtype=torch.int64, device=device)
    selected_rows[0, 1] = -1
    compressed_valid = torch.rand(num_seq, topk, device=device) > 0.25

    q_decode = torch.randn(num_seq, num_heads, head_dim, device=device, dtype=dtype)
    attn_sink = torch.randn(num_heads, device=device, dtype=dtype)

    ref_kv, ref_relidx = _reference_assemble(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    ref_out = M._decode_attention_from_selected(
        q_decode, ref_kv, ref_relidx, attn_sink, softmax_scale
    )

    out_kv, out_relidx = M._fused_assemble_selected_kv(
        swa_cache,
        mhc_cache,
        selected_rows,
        compressed_valid,
        input_pos,
        seq_idx,
        cu_num_pages,
        cache_loc,
        window_size,
        compress_ratio,
        dtype,
    )
    fused_out = M._decode_attention_from_selected(
        q_decode, out_kv, out_relidx, attn_sink, softmax_scale
    )

    assert torch.equal(fused_out, ref_out)
    assert torch.isfinite(fused_out).all()
