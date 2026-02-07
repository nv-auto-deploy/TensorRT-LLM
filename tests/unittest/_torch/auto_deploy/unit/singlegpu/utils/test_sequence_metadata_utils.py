"""Unit tests for sequence_metadata_utils.

Tests build_cache_loc_from_full (trivial and nontrivial slicing) and
increment_position_ids (page overflow and cache_loc rebuild).
"""

from typing import List, Optional

import torch

from tensorrt_llm._torch.auto_deploy.utils.sequence_metadata_utils import (
    build_cache_loc_from_full,
    increment_position_ids,
)

# =============================================================================
# build_cache_loc_from_full
# =============================================================================


class TestBuildCacheLocFromFull:
    """Tests for build_cache_loc_from_full."""

    def test_trivial_single_sequence_use_all_pages(self):
        """Single sequence, use all allocated pages; output equals full_cache_loc."""
        full_cache_loc = [10, 20]
        cu_allocated_pages = [0, 2]
        pages_per_seq = [2]
        batch_size = 1
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        assert out.shape == (2,)
        assert out.tolist() == [10, 20]

    def test_trivial_batch_all_use_full_allocation(self):
        """Batch of two sequences, each uses all allocated pages."""
        full_cache_loc = [1, 2, 3, 4]
        cu_allocated_pages = [0, 2, 4]
        pages_per_seq = [2, 2]
        batch_size = 2
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        assert out.shape == (4,)
        assert out.tolist() == [1, 2, 3, 4]

    def test_nontrivial_single_sequence_one_of_two_pages(self):
        """Single sequence, use one of two allocated pages."""
        full_cache_loc = [10, 20]
        cu_allocated_pages = [0, 2]
        pages_per_seq = [1]
        batch_size = 1
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        assert out.shape == (1,)
        assert out.tolist() == [10]

    def test_nontrivial_batch_mixed_usage(self):
        """Batch: seq0 uses 1 of 2, seq1 uses 2 of 2."""
        full_cache_loc = [100, 101, 200, 201]
        cu_allocated_pages = [0, 2, 4]
        pages_per_seq = [1, 2]
        batch_size = 2
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        assert out.shape == (3,)
        assert out.tolist() == [100, 200, 201]

    def test_nontrivial_batch_all_use_subset(self):
        """Three sequences, each has 2 allocated but uses 1."""
        full_cache_loc = [10, 11, 20, 21, 30, 31]
        cu_allocated_pages = [0, 2, 4, 6]
        pages_per_seq = [1, 1, 1]
        batch_size = 3
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        assert out.shape == (3,)
        assert out.tolist() == [10, 20, 30]

    def test_input_as_tensors(self):
        """Same logic works when inputs are torch.Tensor instead of lists."""
        full_cache_loc = torch.tensor([10, 20], dtype=torch.int)
        cu_allocated_pages = torch.tensor([0, 2], dtype=torch.int)
        pages_per_seq = torch.tensor([1], dtype=torch.int)
        batch_size = 1
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        assert out.shape == (1,)
        assert out.tolist() == [10]

    def test_contract_cu_num_pages_segments(self):
        """Verify output segments match first pages_per_seq[i] of each block."""
        full_cache_loc = [100, 101, 200, 201, 202, 300]
        cu_allocated_pages = [0, 2, 5, 6]
        pages_per_seq = [1, 2, 1]
        batch_size = 3
        out = build_cache_loc_from_full(
            full_cache_loc, cu_allocated_pages, pages_per_seq, batch_size
        )
        cu = torch.cumsum(torch.tensor([0] + pages_per_seq), dim=0)
        assert out.numel() == sum(pages_per_seq)
        assert out[cu[0] : cu[1]].tolist() == [100]
        assert out[cu[1] : cu[2]].tolist() == [200, 201]
        assert out[cu[2] : cu[3]].tolist() == [300]


# =============================================================================
# increment_position_ids (page overflow)
# =============================================================================


def _make_named_args(
    batch_size: int,
    seq_lens: list,
    position_ids_flat: list,
    seq_len_with_cache: list,
    last_page_len: list,
    pages_per_seq: list,
    cu_num_pages: list,
    cache_loc: Optional[List[int]] = None,
    full_cache_loc: Optional[List[int]] = None,
    cu_allocated_pages: Optional[List[int]] = None,
) -> dict:
    """Build named_args for increment_position_ids. All tensors CPU."""
    cu_seqlen = [0]
    for s in seq_lens:
        cu_seqlen.append(cu_seqlen[-1] + s)
    named_args = {
        "position_ids": torch.tensor(position_ids_flat, dtype=torch.long),
        "cu_seqlen": torch.tensor(cu_seqlen, dtype=torch.int),
        "seq_len_with_cache": torch.tensor(seq_len_with_cache, dtype=torch.int),
        "last_page_len": torch.tensor(last_page_len, dtype=torch.int),
        "pages_per_seq": torch.tensor(pages_per_seq, dtype=torch.int),
        "cu_num_pages": torch.tensor(cu_num_pages, dtype=torch.int),
    }
    if cache_loc is not None:
        named_args["cache_loc"] = torch.tensor(cache_loc, dtype=torch.int)
    if full_cache_loc is not None:
        named_args["full_cache_loc"] = torch.tensor(full_cache_loc, dtype=torch.int)
    if cu_allocated_pages is not None:
        named_args["cu_allocated_pages"] = torch.tensor(cu_allocated_pages, dtype=torch.int)
    return named_args


class TestIncrementPositionIdsPageOverflow:
    """Tests for increment_position_ids when last_page_len is near page end and increment causes overflow."""

    def test_no_overflow_metadata_unchanged(self):
        """page_size=32; increment keeps all sequences within current page."""
        page_size = 32
        batch_size = 2
        seq_lens = [1, 1]
        position_ids_flat = [10, 15]
        seq_len_with_cache = [10, 15]
        last_page_len = [10, 15]
        pages_per_seq = [1, 1]
        cu_num_pages = [0, 1, 2]
        increment = torch.tensor([2, 1], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        new_swc = result["seq_len_with_cache"][:batch_size]
        assert new_swc.tolist() == [12, 16]
        old_pages = (torch.tensor(seq_len_with_cache) + page_size - 1) // page_size
        new_pages = (new_swc + page_size - 1) // page_size
        extra_pages = (new_pages - old_pages).tolist()
        assert extra_pages == [0, 0]
        assert result["pages_per_seq"][:batch_size].tolist() == [1, 1]
        assert result["cu_num_pages"][: batch_size + 1].tolist() == [0, 1, 2]

    def test_single_sequence_overflows(self):
        """One sequence at last token of page 1; increment=1 crosses to next page."""
        page_size = 32
        batch_size = 1
        seq_lens = [1]
        position_ids_flat = [31]
        seq_len_with_cache = [32]
        last_page_len = [32]
        pages_per_seq = [1]
        cu_num_pages = [0, 1]
        increment = torch.tensor([1], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        new_swc = result["seq_len_with_cache"][:batch_size]
        assert new_swc.tolist() == [33]
        new_last_page_len = result["last_page_len"][:batch_size]
        assert new_last_page_len.tolist() == [1]
        new_pages_per_seq = result["pages_per_seq"][:batch_size]
        assert new_pages_per_seq.tolist() == [2]
        new_cu_num_pages = result["cu_num_pages"][: batch_size + 1]
        assert new_cu_num_pages.tolist() == [0, 2]

    def test_single_sequence_overflows_with_cache_loc_rebuild(self):
        """Same as single overflow but with full_cache_loc/cu_allocated_pages; cache_loc rebuilt."""
        page_size = 32
        batch_size = 1
        seq_lens = [1]
        position_ids_flat = [31]
        seq_len_with_cache = [32]
        last_page_len = [32]
        pages_per_seq = [1]
        cu_num_pages = [0, 1]
        cache_loc = [100]
        full_cache_loc = [100, 101]
        cu_allocated_pages = [0, 2]
        increment = torch.tensor([1], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
            cache_loc=cache_loc,
            full_cache_loc=full_cache_loc,
            cu_allocated_pages=cu_allocated_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        new_pages_per_seq = result["pages_per_seq"][:batch_size]
        expected_cache_loc = build_cache_loc_from_full(
            full_cache_loc,
            cu_allocated_pages,
            new_pages_per_seq.tolist(),
            batch_size,
        )
        actual = result["cache_loc"]
        valid_len = new_pages_per_seq.sum().item()
        assert actual[:valid_len].tolist() == expected_cache_loc.tolist()

    def test_multiple_sequences_only_some_overflow(self):
        """batch_size=3; seq0 and seq2 cross page boundary, seq1 does not."""
        page_size = 32
        batch_size = 3
        seq_lens = [1, 1, 1]
        position_ids_flat = [30, 49, 62]
        seq_len_with_cache = [31, 50, 63]
        last_page_len = [31, 18, 31]
        pages_per_seq = [1, 2, 2]
        cu_num_pages = [0, 1, 3, 5]
        increment = torch.tensor([2, 1, 2], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        new_swc = result["seq_len_with_cache"][:batch_size]
        assert new_swc.tolist() == [33, 51, 65]
        new_pages_per_seq = result["pages_per_seq"][:batch_size]
        assert new_pages_per_seq.tolist() == [2, 2, 3]
        new_cu_num_pages = result["cu_num_pages"][: batch_size + 1]
        assert new_cu_num_pages.tolist() == [0, 2, 4, 7]
        extra_pages_expected = [1, 0, 1]
        old_pages = (torch.tensor(seq_len_with_cache) + page_size - 1) // page_size
        new_pages = (new_swc + page_size - 1) // page_size
        assert (new_pages - old_pages).tolist() == extra_pages_expected

    def test_rebuild_consistency_multi_overflow(self):
        """Overflow with full_cache_loc/cu_allocated_pages; result cache_loc matches build_cache_loc_from_full."""
        page_size = 32
        batch_size = 2
        seq_lens = [1, 1]
        position_ids_flat = [31, 63]
        seq_len_with_cache = [32, 64]
        last_page_len = [32, 32]
        pages_per_seq = [1, 2]
        cu_num_pages = [0, 1, 3]
        full_cache_loc = [10, 11, 20, 21, 22]
        cu_allocated_pages = [0, 2, 5]
        cache_loc = [10, 20, 21]
        increment = torch.tensor([1, 1], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
            cache_loc=cache_loc,
            full_cache_loc=full_cache_loc,
            cu_allocated_pages=cu_allocated_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        new_pages_per_seq = result["pages_per_seq"][:batch_size]
        expected_cache_loc = build_cache_loc_from_full(
            full_cache_loc,
            cu_allocated_pages,
            new_pages_per_seq.tolist(),
            batch_size,
        )
        actual = result["cache_loc"]
        valid_len = int(new_pages_per_seq.sum().item())
        assert torch.equal(actual[:valid_len], expected_cache_loc)


# =============================================================================
# increment_position_ids (negative increment / rewind)
# =============================================================================


class TestIncrementPositionIdsNegativeIncrement:
    """Tests for increment_position_ids with negative increments (rewind)."""

    def test_negative_increment_metadata_updated(self):
        """Negative increment (rewind): position_ids, seq_len_with_cache, last_page_len,
        pages_per_seq, cu_num_pages updated correctly."""
        page_size = 32
        batch_size = 2
        seq_lens = [1, 1]
        position_ids_flat = [4, 4]
        seq_len_with_cache = [5, 5]
        last_page_len = [(5 - 1) % page_size + 1, (5 - 1) % page_size + 1]
        pages_per_seq = [1, 1]
        cu_num_pages = [0, 1, 2]
        increment = torch.tensor([-2, 0], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        assert result["position_ids"].view(-1)[:batch_size].tolist() == [2, 4]
        new_swc = result["seq_len_with_cache"][:batch_size]
        assert new_swc.tolist() == [3, 5]
        new_last_page_len = result["last_page_len"][:batch_size]
        assert new_last_page_len.tolist() == [(3 - 1) % page_size + 1, (5 - 1) % page_size + 1]
        new_pages_per_seq = result["pages_per_seq"][:batch_size]
        assert new_pages_per_seq.tolist() == [1, 1]
        new_cu_num_pages = result["cu_num_pages"][: batch_size + 1]
        assert new_cu_num_pages.tolist() == [0, 1, 2]

    def test_negative_increment_cache_loc_rebuild(self):
        """Cache_loc is rebuilt when increment is negative (used pages decrease)."""
        page_size = 4
        batch_size = 2
        seq_lens = [1, 1]
        position_ids_flat = [7, 7]
        seq_len_with_cache = [8, 8]
        last_page_len = [(8 - 1) % page_size + 1, (8 - 1) % page_size + 1]
        pages_per_seq = [2, 2]
        cu_num_pages = [0, 2, 4]
        full_cache_loc = [10, 11, 20, 21]
        cu_allocated_pages = [0, 2, 4]
        cache_loc = [10, 11, 20, 21]
        increment = torch.tensor([-4, -4], dtype=torch.int)
        named_args = _make_named_args(
            batch_size,
            seq_lens,
            position_ids_flat,
            seq_len_with_cache,
            last_page_len,
            pages_per_seq,
            cu_num_pages,
            cache_loc=cache_loc,
            full_cache_loc=full_cache_loc,
            cu_allocated_pages=cu_allocated_pages,
        )
        result = increment_position_ids(named_args, increment, page_size)
        new_pages_per_seq = result["pages_per_seq"][:batch_size]
        assert (new_pages_per_seq >= 1).all(), "each sequence must have at least one page"
        assert new_pages_per_seq.tolist() == [1, 1]
        # After rewind: seq_len_with_cache 8->4 each, so 1 page per seq. full_cache_loc has
        # seq0 pages [10,11], seq1 pages [20,21]; we use first 1 of each -> [10, 20].
        expected_cache_loc = [10, 20]
        actual = result["cache_loc"]
        valid_len = int(result["cu_num_pages"][batch_size].item())
        assert actual[:valid_len].tolist() == expected_cache_loc
