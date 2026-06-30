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

"""Byte-exact tests for the fused DeepSeek V4 local-window page-map kernel.

The fused kernel (``_fused_local_window_pagemap``) collapses the per-decode-step
local-window position generation + validity tests + page-address translation
(~20 tiny element-wise / gather kernels) into one Triton launch. The final
``swa_cache[page_ids, page_offsets]`` row gather is left to the caller. These
tests assert the fused addresses + combined validity mask are *bit-identical* to
the original reference chain across batch sizes, dtypes, and boundary cases.
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (
    deepseek_v4_sparse_attention as dsv4_sparse,
)


def _ref_pagemap(input_pos, seq_idx, cu_num_pages, cache_loc, window_size, tokens_per_block):
    """Reference replica of position gen + ``_decode_page_ids_and_offsets``."""
    offsets = torch.arange(window_size, dtype=torch.long, device=input_pos.device)
    positions = input_pos.unsqueeze(1) - window_size + 1 + offsets.view(1, -1)
    valid_pos = (positions >= 0) & (positions <= input_pos.unsqueeze(1))

    positions_long = positions.to(torch.long)
    safe_positions = positions_long.clamp(min=0)
    page_ordinals = safe_positions // tokens_per_block
    page_offsets = safe_positions % tokens_per_block
    seq_idx_long = seq_idx.to(torch.long)
    while seq_idx_long.dim() < positions_long.dim():
        seq_idx_long = seq_idx_long.unsqueeze(-1)
    page_start = cu_num_pages[seq_idx_long].to(torch.long)
    page_end = cu_num_pages[seq_idx_long + 1].to(torch.long)
    page_table_idx = page_start + page_ordinals
    page_valid = (positions_long >= 0) & (page_table_idx < page_end)
    safe_page_table_idx = torch.where(page_valid, page_table_idx, page_start)
    safe_page_table_idx = safe_page_table_idx.clamp(min=0, max=cache_loc.numel() - 1)
    page_ids = cache_loc[safe_page_table_idx].to(torch.long)
    valid = valid_pos & page_valid
    return page_ids, page_offsets, valid


def _make_page_table(per_seq_pages, total_pages, device, idx_dtype):
    """Build (cu_num_pages, cache_loc) for the given per-sequence page counts."""
    cu = torch.zeros(len(per_seq_pages) + 1, dtype=idx_dtype, device=device)
    cu[1:] = torch.tensor(per_seq_pages, dtype=idx_dtype, device=device).cumsum(0)
    n_slots = int(cu[-1].item())
    # cache_loc maps each page-table slot to a (shuffled) physical page id.
    perm = torch.randperm(total_pages, device=device)[:n_slots]
    cache_loc = perm.to(idx_dtype)
    return cu, cache_loc


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + Triton")
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "input_pos_list,per_seq_pages,window_size,tokens_per_block",
    [
        # decode batch=1, window fully inside one+ pages
        ([300], [8], 128, 64),
        # decode batch=2 (the captured graph also captures bs=2)
        ([300, 130], [8, 5], 128, 64),
        # input_pos < window_size -> leading positions negative -> invalid
        ([40], [4], 128, 64),
        # input_pos at a page boundary
        ([255], [4], 128, 64),
        ([256], [8], 128, 64),
        # tiny window / small tokens_per_block
        ([77, 12], [3, 2], 16, 32),
        # position at sequence start (pos 0)
        ([0], [1], 128, 64),
        # larger window
        ([1000, 500], [20, 12], 256, 64),
    ],
)
def test_fused_local_window_pagemap_byte_exact(
    input_pos_list, per_seq_pages, window_size, tokens_per_block, idx_dtype
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    n = len(input_pos_list)
    input_pos = torch.tensor(input_pos_list, dtype=torch.long, device=device)
    seq_idx = torch.arange(n, dtype=torch.long, device=device)
    total_pages = sum(per_seq_pages) + 4
    cu_num_pages, cache_loc = _make_page_table(per_seq_pages, total_pages, device, idx_dtype)

    ref_ids, ref_off, ref_valid = _ref_pagemap(
        input_pos, seq_idx, cu_num_pages, cache_loc, window_size, tokens_per_block
    )
    got_ids, got_off, got_valid = dsv4_sparse._fused_local_window_pagemap(
        input_pos, seq_idx, cu_num_pages, cache_loc, window_size, tokens_per_block
    )

    assert got_ids.dtype == torch.long and got_off.dtype == torch.long
    assert got_valid.dtype == torch.bool
    assert torch.equal(got_off, ref_off), "page_offsets differ"
    # page_ids only need to match where valid (invalid slots gather a masked row);
    # but the reference clamps to page_start so they match everywhere -- assert all.
    assert torch.equal(got_ids, ref_ids), "page_ids differ"
    assert torch.equal(got_valid, ref_valid), "validity mask differs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA + Triton")
@pytest.mark.parametrize("n", [1, 2])
def test_decode_local_cache_rows_matches_reference(n):
    """End-to-end ``_decode_local_cache_rows`` (fused) vs the torch fallback chain."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    window_size = 128
    tokens_per_block = 64
    head_dim = 32
    per_seq_pages = [8, 6][:n]
    total_pages = sum(per_seq_pages) + 4
    cu_num_pages, cache_loc = _make_page_table(per_seq_pages, total_pages, device, torch.int32)
    swa_cache = torch.randn(
        total_pages, tokens_per_block, head_dim, dtype=torch.bfloat16, device=device
    )
    input_pos = torch.tensor([300, 130][:n], dtype=torch.long, device=device)
    seq_idx = torch.arange(n, dtype=torch.long, device=device)

    rows_fused, valid_fused = dsv4_sparse._decode_local_cache_rows(
        swa_cache, seq_idx, input_pos, cu_num_pages, cache_loc, window_size, torch.bfloat16
    )

    # Reference (the original torch chain, bypassing the Triton fast path).
    ref_ids, ref_off, ref_valid = _ref_pagemap(
        input_pos, seq_idx, cu_num_pages, cache_loc, window_size, tokens_per_block
    )
    rows_ref = swa_cache[ref_ids, ref_off].to(torch.bfloat16)

    assert torch.equal(valid_fused, ref_valid)
    assert torch.equal(rows_fused, rows_ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
