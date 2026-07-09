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
"""Byte-exactness guard for the fused decode selected-KV paged assembly (idea_0001).

``_fused_assemble_selected_kv`` folds the per-ratio-4/128-layer decode tail of
``_decode_compressed_cache_attention`` -- the local-window ``swa_cache`` gather, the
dynamic compressed ``mhc_cache`` page translation + gather, the two ``torch.cat``s
that build ``selected_kv`` / ``valid_rows`` and the ``arange``/``where`` that yields
the attend's relative row indices -- into one paged Triton kernel that reads both
caches directly and emits ``selected_kv`` + ``rel_topk``.

This test rebuilds the exact eager reference from the module's own helpers
(``_decode_local_cache_rows`` + ``_decode_cache_rows_from_positions`` + cat + where)
and asserts the fused kernel is bit-identical (``torch.equal``) across head dims,
compression ratios, top-k widths, bf16 / fp32 caches, and a mix of pad (``-1``),
out-of-range, and invalid rows -- including the clamped content of masked slots,
which the attend ignores but which must still match the gather it replaces.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


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


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused selected-KV assembly requires triton + CUDA",
)
@pytest.mark.parametrize("head_dim", [8, 16, 128, 576])
@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("topk", [0, 6, 16])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_fused_assemble_matches_eager(head_dim, compress_ratio, topk, cache_dtype):
    torch.manual_seed(20260701 + head_dim + compress_ratio + topk)
    device = "cuda"
    dtype = torch.bfloat16  # activation / selected_kv dtype (== q_decode.dtype)
    window_size = 4
    tokens_per_block = 8

    # Two decode sequences with distinct page-table extents. cu_num_pages is the
    # prefix-sum page-table offset; cache_loc maps each page-table slot to a physical
    # page id (a permutation into a larger physical pool).
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
    # input_pos chosen so the local window straddles a page boundary; pad small enough
    # that all local rows are page-valid, big enough that some are position-invalid.
    input_pos = torch.tensor([37, 2], dtype=torch.long, device=device)

    if topk == 0:
        selected_rows = torch.empty(num_seq, 0, dtype=torch.int64, device=device)
        compressed_valid = torch.empty(num_seq, 0, dtype=torch.bool, device=device)
    else:
        # Mix of in-range rows, pad rows (-1), and out-of-range rows whose paged
        # position lands past the sequence's page extent (must mask to rel_topk=-1).
        selected_rows = torch.randint(0, 40, (num_seq, topk), dtype=torch.int64, device=device)
        selected_rows[0, 0] = -1
        if topk > 3:
            selected_rows[0, 3] = -1
            selected_rows[1, 2] = 500  # position 500*ratio -> beyond page extent
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
    assert torch.equal(out_relidx, ref_relidx), (
        f"rel_topk diverged: head_dim={head_dim} ratio={compress_ratio} topk={topk} "
        f"cache_dtype={cache_dtype}"
    )
    assert torch.equal(out_kv, ref_kv), (
        f"selected_kv diverged: head_dim={head_dim} ratio={compress_ratio} topk={topk} "
        f"cache_dtype={cache_dtype}"
    )


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused selected-KV assembly requires triton + CUDA",
)
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_fused_assemble_large_decode_position(compress_ratio):
    """Real-scale regime: input_pos ~1000 (the proxy decode window) with many pages
    and wide top-k, so paged positions cross many page boundaries and physical page
    ids are large (guards the int64 address arithmetic against int32 overflow)."""
    torch.manual_seed(70123 + compress_ratio)
    device = "cuda"
    dtype = torch.bfloat16
    head_dim, window_size, tokens_per_block = 576, 128, 64
    max_compressed_len = 512

    pages_per_seq = [24, 20]  # 24*64=1536, 20*64=1280 tokens -> covers input_pos ~1000
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
        # indexer-style dynamic selection: rows in [0, visible_len), some -1 pads.
        topk = 384
        selected_rows = torch.randint(0, 250, (num_seq, topk), dtype=torch.int64, device=device)
        selected_rows[:, ::37] = -1
        compressed_valid = torch.rand(num_seq, topk, device=device) > 0.2
    else:
        # ratio-128 dense: selected_rows = arange, compressed_valid = row < visible_len.
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


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused selected-KV assembly requires triton + CUDA",
)
@pytest.mark.parametrize(
    "window_size,tokens_per_block,max_compressed_len",
    [(128, 64, 16), (128, 64, 512), (4, 8, 40)],
)
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_fused_assemble_dense_in_kernel_matches_materialized(
    window_size, tokens_per_block, max_compressed_len, cache_dtype
):
    """Dense (ratio-128) mode, idea_0090: ``selected_rows=None`` derives the row ids
    and their visibility in-kernel; the result must be bit-identical to both the
    materialized arange/floordiv/clamp/lt chain fed through the fused kernel and the
    eager reference. The input_pos sweep crosses the ratio-128 row boundaries
    (126/127/128/255/256), partial pages, short histories, and negative (padded)
    rows where torch's floor division semantics differ from truncation."""
    compress_ratio = 128
    torch.manual_seed(90_000 + window_size + max_compressed_len)
    device = "cuda"
    dtype = torch.bfloat16
    head_dim = 64

    input_pos = torch.tensor(
        [126, 127, 128, 129, 255, 256, 0, 2, -1, -130, 1005],
        dtype=torch.long,
        device=device,
    )
    num_seq = int(input_pos.shape[0])
    # Enough pages per sequence to cover input_pos ~1005 plus the current token;
    # rows past each sequence's extent must resolve page-invalid in both paths.
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

    # The materialized chain the dense specialization replaces (byte-for-byte).
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
    assert torch.equal(dense_relidx, ref_relidx), (
        f"dense rel_topk diverged from eager: W={window_size} tpb={tokens_per_block} "
        f"m={max_compressed_len} cache_dtype={cache_dtype}"
    )
    assert torch.equal(dense_kv, ref_kv), (
        f"dense selected_kv diverged from eager: W={window_size} tpb={tokens_per_block} "
        f"m={max_compressed_len} cache_dtype={cache_dtype}"
    )
    assert torch.equal(dense_relidx, mat_relidx)
    assert torch.equal(dense_kv, mat_kv)


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused selected-KV assembly requires triton + CUDA",
)
def test_fused_assemble_attend_output_matches_eager():
    """End-to-end: the fused assemble + attend must match the eager rows + attend."""
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

    # Same rows + same rel indices -> identical attend inputs -> identical output.
    assert torch.equal(out_kv, ref_kv)
    assert torch.equal(out_relidx, ref_relidx)
    assert torch.equal(fused_out, ref_out)
