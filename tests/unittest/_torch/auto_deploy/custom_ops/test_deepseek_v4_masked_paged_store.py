# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exactness check for the DeepSeek V4 validity-masked paged store.

``_masked_write_decode_cache_rows`` (idea_0035) replaces the decode ``mhc_cache``
compressed-row update's previous-row gather + ``torch.where`` + unconditional
index_put write-back with a single read-free masked store: for each decode row it
writes the freshly compressed row only when ``row_valid`` is true, and stores
nothing otherwise. This test guards that the masked store is byte-identical to the
reference read-old / where / write-back it replaces -- both for the rows it writes
(a plain copy of the cast value) and, critically, for the invalid rows it must leave
untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _HAS_TRITON,
    _masked_write_decode_cache_rows,
)


def _reference_write(cache, compressed_rows, row_valid, page_ids, page_offsets):
    """The read-old + torch.where + unconditional write-back this op replaces."""
    previous_rows = cache[page_ids, page_offsets].to(cache.dtype)
    rows_to_write = torch.where(
        row_valid.unsqueeze(-1),
        compressed_rows.to(cache.dtype),
        previous_rows,
    )
    cache[page_ids, page_offsets] = rows_to_write.to(cache.dtype)


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="masked paged store requires triton + CUDA",
)
@pytest.mark.parametrize("state_dim", [64, 128, 576, 1024])
@pytest.mark.parametrize("num_rows", [1, 2, 5])
@pytest.mark.parametrize("src_dtype", [torch.bfloat16, torch.float32])
def test_masked_paged_store_matches_read_where_writeback(state_dim, num_rows, src_dtype):
    torch.manual_seed(1234 + state_dim + num_rows)
    device = "cuda"
    cache_dtype = torch.bfloat16
    num_pages, tokens_per_block = 17, 8

    # Distinct (page_id, page_offset) per row -- mirrors the real decode where each
    # sequence writes its own logical row (no index_put aliasing).
    base = torch.randperm(num_pages * tokens_per_block, device=device)[:num_rows]
    page_ids = (base // tokens_per_block).to(torch.long)
    page_offsets = (base % tokens_per_block).to(torch.long)

    cache_ref = torch.randn(num_pages, tokens_per_block, state_dim, device=device).to(cache_dtype)
    cache_masked = cache_ref.clone()

    compressed_rows = torch.randn(num_rows, state_dim, device=device, dtype=src_dtype)

    # Exercise a mix of valid / invalid rows (the invalid ones must stay untouched),
    # plus the all-invalid and all-valid extremes.
    for row_valid in (
        torch.tensor([bool((i + num_rows) % 2) for i in range(num_rows)], device=device),
        torch.zeros(num_rows, dtype=torch.bool, device=device),
        torch.ones(num_rows, dtype=torch.bool, device=device),
    ):
        ref = cache_ref.clone()
        masked = cache_masked.clone()
        _reference_write(ref, compressed_rows, row_valid, page_ids, page_offsets)
        _masked_write_decode_cache_rows(
            masked,
            compressed_rows.to(cache_dtype),
            row_valid,
            page_ids,
            page_offsets,
        )
        assert torch.equal(masked, ref), (
            f"masked store diverged: state_dim={state_dim} num_rows={num_rows} "
            f"src_dtype={src_dtype} n_valid={int(row_valid.sum())}"
        )


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="masked paged store requires triton + CUDA",
)
def test_masked_paged_store_leaves_invalid_rows_bit_identical():
    """A fully-invalid store must not touch a single byte of the cache."""
    torch.manual_seed(7)
    device = "cuda"
    num_pages, tokens_per_block, state_dim = 9, 8, 576
    cache = torch.randn(num_pages, tokens_per_block, state_dim, device=device).to(torch.bfloat16)
    before = cache.clone()

    num_rows = 3
    base = torch.randperm(num_pages * tokens_per_block, device=device)[:num_rows]
    page_ids = (base // tokens_per_block).to(torch.long)
    page_offsets = (base % tokens_per_block).to(torch.long)
    compressed_rows = torch.randn(num_rows, state_dim, device=device, dtype=torch.bfloat16)
    row_valid = torch.zeros(num_rows, dtype=torch.bool, device=device)

    _masked_write_decode_cache_rows(cache, compressed_rows, row_valid, page_ids, page_offsets)
    assert torch.equal(cache, before)
