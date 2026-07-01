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
"""Unit test for the hoisted DeepSeek-V4 current-token page-address prepare op.

``deepseek_v4_sparse_prepare_decode_page_addr`` computes the current-token
paged write address once per forward; the result is shared by every layer and
every current-token cache write. It must be bit-identical to the per-layer
``_decode_page_ids_and_offsets`` translation that it replaces.
"""

import pytest
import torch

# Registers auto_deploy::deepseek_v4_sparse_prepare_decode_page_addr
import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M  # noqa: F401


@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_seq", [1, 2, 3])
def test_prepare_decode_page_addr_matches_reference(tokens_per_block, num_seq):
    torch.manual_seed(num_seq * 100 + tokens_per_block)
    max_blocks_per_seq = 2048 // tokens_per_block

    # Per-sequence page table (cu_num_pages cumulative) with a shuffled page pool
    # so an identity translation would be detected.
    cu_num_pages = torch.arange(
        0, (num_seq + 1) * max_blocks_per_seq, max_blocks_per_seq, dtype=torch.long
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, dtype=torch.long)

    # Current-token positions for each sequence; buffer padded beyond num_seq.
    positions = torch.randint(0, 2048, (num_seq,), dtype=torch.long)
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    page_ids, page_offsets = prep(input_pos, cu_num_pages, cache_loc, tokens_per_block)

    # Reference: the per-layer translation each current-token write performs.
    cache = torch.zeros(total_pages, tokens_per_block, 4)
    seq_idx = torch.arange(num_seq, dtype=torch.long)
    ref_ids, ref_offsets, _ = M._decode_page_ids_and_offsets(
        cache, seq_idx, input_pos.reshape(-1)[:num_seq], cu_num_pages, cache_loc
    )

    assert page_ids.shape[0] == num_seq
    assert torch.equal(page_ids[:num_seq], ref_ids)
    assert torch.equal(page_offsets[:num_seq], ref_offsets)
    # Decode slices [:num_decode]; any prefix must also match.
    for nd in range(1, num_seq + 1):
        assert torch.equal(page_ids[:nd], ref_ids[:nd])
        assert torch.equal(page_offsets[:nd], ref_offsets[:nd])


@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_seq", [1, 2, 3])
def test_prepare_decode_page_addr_overlap_and_full_maps_match_reference(tokens_per_block, num_seq):
    """The hoisted ratio-4 overlap + full-range maps must be bit-identical to the
    per-layer ``_decode_page_ids_and_offsets`` translation they replace in
    ``_batched_compressed_rows_from_paged_state`` (overlap) and
    ``_batched_overlap_compressed_rows_fullrange``.
    """
    torch.manual_seed(num_seq * 131 + tokens_per_block)
    ratio = 4
    m = 37  # ratio-4 max_compressed_len (arbitrary; full range spans m*ratio positions)
    max_blocks_per_seq = 2048 // tokens_per_block

    cu_num_pages = torch.arange(
        0, (num_seq + 1) * max_blocks_per_seq, max_blocks_per_seq, dtype=torch.long
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, dtype=torch.long)

    positions = torch.randint(0, 2048, (num_seq,), dtype=torch.long)
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    outs = prep(input_pos, cu_num_pages, cache_loc, tokens_per_block, m)
    assert len(outs) == 8, "overlap request must return 2 current-token + 6 map tensors"
    (
        cur_pid,
        cur_poff,
        ovl_pid,
        ovl_poff,
        ovl_valid,
        full_pid,
        full_poff,
        full_valid,
    ) = outs

    seq_idx = torch.arange(num_seq, dtype=torch.long)
    pos_seq = input_pos.reshape(-1)[:num_seq]
    cache = torch.zeros(total_pages, tokens_per_block, 4)

    # Current-token map path stays bit-identical when overlap maps are also emitted.
    ref_cur_pid, ref_cur_poff, _ = M._decode_page_ids_and_offsets(
        cache, seq_idx, pos_seq, cu_num_pages, cache_loc
    )
    assert torch.equal(cur_pid[:num_seq], ref_cur_pid)
    assert torch.equal(cur_poff[:num_seq], ref_cur_poff)

    # Overlap band: previous block [anchor - ratio, anchor), current [anchor, anchor + ratio).
    row_idx = (pos_seq // ratio).clamp(min=0, max=m - 1)
    anchor = row_idx * ratio
    offsets = torch.arange(ratio, dtype=torch.long)
    prev_pos = anchor.unsqueeze(1) - ratio + offsets.view(1, -1)
    curr_pos = anchor.unsqueeze(1) + offsets.view(1, -1)
    ref_prev = M._decode_page_ids_and_offsets(cache, seq_idx, prev_pos, cu_num_pages, cache_loc)
    ref_curr = M._decode_page_ids_and_offsets(cache, seq_idx, curr_pos, cu_num_pages, cache_loc)
    assert torch.equal(ovl_pid[:num_seq, :ratio], ref_prev[0])
    assert torch.equal(ovl_poff[:num_seq, :ratio], ref_prev[1])
    assert torch.equal(ovl_valid[:num_seq, :ratio], ref_prev[2])
    assert torch.equal(ovl_pid[:num_seq, ratio:], ref_curr[0])
    assert torch.equal(ovl_poff[:num_seq, ratio:], ref_curr[1])
    assert torch.equal(ovl_valid[:num_seq, ratio:], ref_curr[2])

    # Full candidate range [0, m * ratio).
    full_pos = torch.arange(m * ratio, dtype=torch.long).view(1, -1).expand(num_seq, -1)
    ref_full = M._decode_page_ids_and_offsets(cache, seq_idx, full_pos, cu_num_pages, cache_loc)
    assert torch.equal(full_pid[:num_seq], ref_full[0])
    assert torch.equal(full_poff[:num_seq], ref_full[1])
    assert torch.equal(full_valid[:num_seq], ref_full[2])


def test_prepare_decode_page_addr_overlap_fake_shape():
    """Fake path returns 8 tensors with the ratio-4 map shapes when overlap requested."""
    tokens_per_block = 32
    num_seq = 2
    ratio = 4
    m = 11

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    with torch._subclasses.FakeTensorMode():
        fake_pos = torch.empty(num_seq, dtype=torch.long)
        fake_cu = torch.empty(num_seq + 1, dtype=torch.long)
        fake_loc = torch.empty(128, dtype=torch.long)
        outs = prep(fake_pos, fake_cu, fake_loc, tokens_per_block, m)
    assert len(outs) == 8
    assert outs[2].shape == (num_seq, 2 * ratio) and outs[2].dtype == torch.long
    assert outs[4].shape == (num_seq, 2 * ratio) and outs[4].dtype == torch.bool
    assert outs[5].shape == (num_seq, m * ratio) and outs[5].dtype == torch.long
    assert outs[7].shape == (num_seq, m * ratio) and outs[7].dtype == torch.bool


def test_prepare_decode_page_addr_fake_shape():
    """Fake (meta) path returns ``[num_seq]`` int64 tensors for export/cudagraph."""
    tokens_per_block = 32
    num_seq = 2

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    with torch._subclasses.FakeTensorMode():
        fake_pos = torch.empty(num_seq, dtype=torch.long)
        fake_cu = torch.empty(num_seq + 1, dtype=torch.long)
        fake_loc = torch.empty(128, dtype=torch.long)
        page_ids, page_offsets = prep(fake_pos, fake_cu, fake_loc, tokens_per_block)
    assert page_ids.dtype == torch.long and page_offsets.dtype == torch.long
    assert page_ids.shape[0] == num_seq and page_offsets.shape[0] == num_seq


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
