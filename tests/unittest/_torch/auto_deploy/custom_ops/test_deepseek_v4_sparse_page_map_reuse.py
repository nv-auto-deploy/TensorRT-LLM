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
"""Unit test for the per-layer DeepSeek-V4 sparse page-map reuse (idea_0007).

The decode op recomputes ``_decode_page_ids_and_offsets`` once per cache read /
write. Every sparse cache is a ``PagedResourceHandler`` allocated as
``[num_blocks, tokens_per_block, *token_shape]`` from one ``sequence_info``, so
``shape[1]`` (tokens_per_block) is uniform and the page map for a given
``positions`` is identical regardless of which cache it indexes. The decode path
therefore computes each distinct ``(positions)`` map once and reuses it for the
paired kv/gate reads and the MHC read+write. This test locks in that premise:

  1. ``_decode_page_ids_and_offsets`` is byte-identical across two caches that
     share ``shape[1]`` but differ in ``shape[2:]``, dtype, and contents.
  2. ``_decode_cache_rows_from_positions`` with a precomputed ``page_map`` is
     byte-identical to the recompute path, including gathering one cache with a
     map derived from another (the kv/gate-pair commoning).
"""

import pytest
import torch

# Registers the op module and exposes the helper functions under test.
import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def _build_page_table(num_seq: int, tokens_per_block: int, seed: int):
    torch.manual_seed(seed)
    max_blocks_per_seq = 2048 // tokens_per_block
    cu_num_pages = torch.arange(
        0, (num_seq + 1) * max_blocks_per_seq, max_blocks_per_seq, dtype=torch.long
    )
    total_pages = int(cu_num_pages[-1].item())
    # Shuffled page pool so an identity (non-paged) translation would be detected.
    cache_loc = torch.randperm(total_pages, dtype=torch.long)
    return cu_num_pages, cache_loc, total_pages


@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_seq", [1, 2, 3])
def test_page_map_identical_across_caches(tokens_per_block, num_seq):
    """The map depends only on shape[1] + cache_loc, not on which cache is indexed."""
    cu_num_pages, cache_loc, total_pages = _build_page_table(
        num_seq, tokens_per_block, seed=num_seq
    )
    seq_idx = torch.arange(num_seq, dtype=torch.long)

    # 2D positions ([B, ratio]) mixing valid, negative (masked), and out-of-range rows.
    positions = torch.randint(-3, 2048, (num_seq, 4), dtype=torch.long)

    # Two caches sharing num_blocks + tokens_per_block but differing in token_shape,
    # dtype, and contents -- exactly the compressor_kv_cache vs compressor_gate_cache
    # (and mhc/indexer) relationship after PagedResourceHandler.allocate.
    cache_kv = torch.randn(total_pages, tokens_per_block, 6, dtype=torch.float32)
    cache_gate = torch.randn(total_pages, tokens_per_block, 3, dtype=torch.float32)

    ids_a, off_a, valid_a = M._decode_page_ids_and_offsets(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc
    )
    ids_b, off_b, valid_b = M._decode_page_ids_and_offsets(
        cache_gate, seq_idx, positions, cu_num_pages, cache_loc
    )
    assert torch.equal(ids_a, ids_b)
    assert torch.equal(off_a, off_b)
    assert torch.equal(valid_a, valid_b)


@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_seq", [1, 2])
def test_decode_cache_rows_page_map_is_transparent(tokens_per_block, num_seq):
    """Passing a precomputed page_map yields byte-identical rows to recomputing it,
    including reusing a kv-derived map to gather the gate cache."""
    cu_num_pages, cache_loc, total_pages = _build_page_table(
        num_seq, tokens_per_block, seed=num_seq + 7
    )
    seq_idx = torch.arange(num_seq, dtype=torch.long)
    positions = torch.randint(-2, 2048, (num_seq, 4), dtype=torch.long)

    cache_kv = torch.randn(total_pages, tokens_per_block, 6, dtype=torch.float32)
    cache_gate = torch.randn(total_pages, tokens_per_block, 3, dtype=torch.float32)

    # Reference: each cache recomputes its own map (the pre-idea behavior).
    ref_kv, ref_kv_valid = M._decode_cache_rows_from_positions(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16
    )
    ref_gate, _ = M._decode_cache_rows_from_positions(
        cache_gate, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16
    )

    # New behavior: compute the map once (from cache_kv) and reuse for both gathers.
    shared_map = M._decode_page_ids_and_offsets(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc
    )
    new_kv, new_kv_valid = M._decode_cache_rows_from_positions(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, shared_map
    )
    new_gate, _ = M._decode_cache_rows_from_positions(
        cache_gate, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, shared_map
    )

    assert torch.equal(ref_kv, new_kv)
    assert torch.equal(ref_gate, new_gate)
    assert torch.equal(ref_kv_valid, new_kv_valid)

    # A map carrying valid=None (the MHC read+write commoning) still gathers correctly.
    ids, off, _ = shared_map
    rows_none_valid, valid_none = M._decode_cache_rows_from_positions(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, (ids, off, None)
    )
    assert torch.equal(rows_none_valid, ref_kv)
    assert valid_none is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
