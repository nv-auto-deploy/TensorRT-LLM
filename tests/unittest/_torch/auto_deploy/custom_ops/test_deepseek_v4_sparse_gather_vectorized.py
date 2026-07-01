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
"""Bit-exactness guard for the vectorized DeepSeek-V4 sparse *prefill* gather path.

The prefill gather used to run O(seq_len^2) nested Python loops with a per-position
host ``.cpu()`` sync + per-element ``.item()``, hanging the model at ISL=1000. The
two hot spots were rewritten to reuse the vectorized decode helpers:

  * ``_gather_paged_rows_from_positions`` -- fully vectorized page-id/offset gather.
  * ``_select_ratio4_indexer_rows`` -- the visible-row ``_compressed_row_from_paged_state``
    stack replaced by one ``_batched_compressed_rows_from_paged_state`` call.

This test keeps a copy of the ORIGINAL loop implementations (``_ref_*``) and asserts
the new vectorized functions are **bit-exact** (``torch.equal`` / ``atol=rtol=0``)
across several shapes, including empty positions, invalid/out-of-range positions,
and short sequences.
"""

import pytest
import torch

# Registers the op module and exposes the helper functions under test.
import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


# ---------------------------------------------------------------------------
# Reference (ORIGINAL loop) implementations, pasted verbatim from before the
# vectorization. They intentionally use the pre-change per-position host loop.
# ---------------------------------------------------------------------------
def _ref_gather_paged_rows_from_positions(
    cache: torch.Tensor,
    seq_idx: int,
    positions: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    dtype: torch.dtype,
    width=None,
):
    positions_host = positions.detach().cpu().to(torch.long).flatten()
    rows = []
    valid_rows = []
    row_width = cache.shape[-1] if width is None else width
    zero = cache.new_zeros(row_width)
    for logical_pos_tensor in positions_host:
        logical_pos = int(logical_pos_tensor.item())
        is_valid = M._host_position_is_valid(cache, seq_idx, logical_pos, cu_num_pages_host)
        valid_rows.append(is_valid)
        if is_valid:
            page_id, page_offset = M._host_page_id_and_offset(
                cache, seq_idx, logical_pos, cu_num_pages_host, cache_loc_host
            )
            row = cache[page_id, page_offset]
            if width is not None:
                row = row[..., :width]
            rows.append(row.to(dtype))
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
    """The ORIGINAL per-row stack from ``_select_ratio4_indexer_rows`` (loop form).

    Uses the still-present (unmodified) ``_compressed_row_from_paged_state`` helper.
    """
    state_dim = int(indexer_compressor_kv_cache.shape[-1])
    return torch.stack(
        [
            M._compressed_row_from_paged_state(
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


# ---------------------------------------------------------------------------
# Helpers to build small random paged caches + a valid page table.
# ---------------------------------------------------------------------------
def _build_page_table(num_seq, tokens_per_block, pages_per_seq, seed, device):
    """Random-but-valid page table. Every sequence gets ``pages_per_seq`` pages
    drawn from a shuffled pool so an identity (non-paged) translation is detected."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    cu_num_pages = torch.arange(0, (num_seq + 1) * pages_per_seq, pages_per_seq, dtype=torch.long)
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, generator=g, dtype=torch.long)
    return cu_num_pages.to(device), cache_loc.to(device), total_pages


# ---------------------------------------------------------------------------
# 1) _gather_paged_rows_from_positions bit-exactness.
# ---------------------------------------------------------------------------
_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("tokens_per_block", [1, 4, 32])
@pytest.mark.parametrize(
    "pos_shape",
    [(0,), (1,), (7,), (3, 4), (2, 5)],
)
@pytest.mark.parametrize("width", [None, 3])
def test_gather_paged_rows_bit_exact(device, tokens_per_block, pos_shape, width):
    torch.manual_seed(hash((tokens_per_block, pos_shape, width)) & 0xFFFF)
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
    # Positions spanning valid, negative (invalid), and out-of-page (invalid) values.
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
            cache, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, width=width
        )
        new_rows, new_valid = M._gather_paged_rows_from_positions(
            cache, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, width=width
        )
        assert new_rows.shape == ref_rows.shape, (seq_idx, ref_rows.shape, new_rows.shape)
        assert new_valid.shape == ref_valid.shape
        assert new_valid.device == ref_valid.device
        assert torch.equal(new_valid, ref_valid), f"valid mismatch seq={seq_idx}"
        # bfloat16 exact equality (values are copied/zeroed, never reduced).
        assert torch.equal(new_rows, ref_rows), f"rows mismatch seq={seq_idx}"


@pytest.mark.parametrize("device", _DEVICES)
def test_gather_paged_rows_all_invalid_and_all_valid(device):
    """Edge cases: every position invalid (all -1) and every position in-range."""
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


# ---------------------------------------------------------------------------
# 2) _select_ratio4_indexer_rows: batched row-stack == loop row-stack.
#    (The rotate=True compressor path uses the triton hadamard/rms ops -> CUDA.)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="rotate=True path uses CUDA triton ops")
@pytest.mark.parametrize("query_pos", [3, 4, 7, 8, 40, 128])
def test_indexer_row_stack_bit_exact(query_pos):
    """The vectorized ``_batched_compressed_rows_from_paged_state`` used by the new
    ``_select_ratio4_indexer_rows`` must produce the same ``index_k`` row-stack that the
    old ``_compressed_row_from_paged_state`` loop produced."""
    device = "cuda"
    torch.manual_seed(query_pos + 3)

    index_head_dim = 128  # DeepSeek-V4 indexer head_dim.
    state_dim = 2 * index_head_dim  # overlap indexer has 2 channels.
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

    rope_half = rope_dim // 2
    max_pos = 4096
    cos_table = torch.randn(max_pos, rope_half, dtype=torch.float32, device=device)
    sin_table = torch.randn(max_pos, rope_half, dtype=torch.float32, device=device)

    query_position_id = query_pos  # position_id == pos for a single-sequence prefill.
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
@pytest.mark.parametrize("query_pos", [0, 1, 2, 3])
def test_select_ratio4_indexer_rows_short_seq(query_pos):
    """Short-sequence guards: visible_len<=0 returns the -1 / empty sentinel, and the
    end-to-end selection matches a loop-based reference for the smallest valid window."""
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
        # No completed compressed row yet -> full -1 sentinel of length index_topk.
        assert torch.equal(out, torch.full((index_topk,), -1, dtype=torch.int64, device=device))
    else:
        # Non-trivial window: the selection must equal a loop-reference selection built
        # on the same (unchanged) index_k math.
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
