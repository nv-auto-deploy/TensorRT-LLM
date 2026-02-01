"""Utilities for managing sequence metadata in attention operations.

This module provides helper functions for updating sequence-related tensors
such as position_ids, seq_len_with_cache, and paging metadata.
"""

from typing import Dict, Sequence, Union

import torch


def _to_tensor(
    x: Union[Sequence[int], torch.Tensor], dtype: torch.dtype = torch.int
) -> torch.Tensor:
    """Convert list or tensor to 1D tensor."""
    if isinstance(x, torch.Tensor):
        return x.flatten().to(dtype)
    return torch.tensor(x, dtype=dtype)


def build_cache_loc_from_full(
    full_cache_loc: Union[Sequence[int], torch.Tensor],
    cu_allocated_pages: Union[Sequence[int], torch.Tensor],
    pages_per_seq: Union[Sequence[int], torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    """Build compact cache_loc from full allocation and current used page counts.

    Takes the first pages_per_seq[i] entries from each sequence's block in
    full_cache_loc and concatenates them so that FlashInfer's cu_num_pages
    indexing sees the correct segment per sequence.

    Args:
        full_cache_loc: Flat list/tensor of all allocated page IDs (by sequence).
        cu_allocated_pages: Cumulative allocated pages [0, a0, a0+a1, ...], length batch_size+1.
        pages_per_seq: Current used pages per sequence, length batch_size.
        batch_size: Number of sequences.

    Returns:
        1D tensor of page IDs of length sum(pages_per_seq). Use .tolist() if a list is needed.
    """
    full_cache_loc = _to_tensor(full_cache_loc)
    cu_allocated_pages = _to_tensor(cu_allocated_pages)
    pages_per_seq = _to_tensor(pages_per_seq)

    # Only slice the valid prefix of full_cache_loc (up to cu_allocated_pages[batch_size]).
    end_valid = int(cu_allocated_pages[batch_size].item())
    assert end_valid <= full_cache_loc.numel(), (
        f"cu_allocated_pages[{batch_size}]={end_valid} > full_cache_loc.numel()={full_cache_loc.numel()}"
    )

    slices = []
    for i in range(batch_size):
        start = int(cu_allocated_pages[i].item())
        used = int(pages_per_seq[i].item())
        end_alloc = int(cu_allocated_pages[i + 1].item())
        assert used <= end_alloc - start, f"Seq {i}: used {used} > allocated {end_alloc - start}"
        # Slice only valid pages for this sequence (start + used <= end_valid).
        slices.append(full_cache_loc[start : start + used])
    return torch.cat(slices)


def increment_position_ids(
    named_args: Dict[str, torch.Tensor],
    increment: torch.Tensor,
    page_size: int,
) -> Dict[str, torch.Tensor]:
    """Increment position_ids and update related sequence metadata consistently.

    This is a pure function that takes a dictionary of tensors (like named_args)
    and returns a new dictionary with all keys preserved. Position IDs are incremented
    and related metadata (seq_len_with_cache, last_page_len, pages_per_seq, cu_num_pages)
    is updated to maintain consistency. Both device and host versions of tensors are
    updated (e.g., seq_len_with_cache and seq_len_with_cache_host).

    NOTE: We must update both GPU and host tensors because FlashInfer's plan() API for
    prefill kernels requires host-side tensors (see flashinfer_attention.py). The plan
    uses host versions to enable non-blocking invocation, triggering a non-blocking copy
    to device rather than a blocking copy to CPU. To support GPU-only updates in the
    future, kernel changes would be needed to allow calling prefill kernels for
    sampling/verification while only passing GPU-side tensors.

    NOTE: This function is not CUDA graph compatible with variable batch sizes due to
    repeat_interleave producing variable-sized outputs. To make it CUDA graph compatible,
    refactor to use max_batch_size with proper masking and fixed-size operations.

    Args:
        named_args: Dictionary of sequence metadata tensors (position_ids,
                    seq_len_with_cache, last_page_len, pages_per_seq, cu_num_pages, etc.)
                    May include both device and host versions (*_host suffix).
        increment: 1D tensor of shape [batch_size] with increment values.
                   Expected to be 0 for context requests and (accepted_tokens - 1)
                   for generation requests.
        page_size: Page size for computing last_page_len and page counts.

    Returns:
        Dictionary with all input keys preserved. Modified tensors are updated
        (both device and host versions), unmodified tensors are passed through unchanged.
    """
    # Start with a copy of all input keys (unmodified tensors pass through)
    result = dict(named_args)

    # Helper to get tensor preferring device version over host version
    def get_tensor(base_name: str) -> torch.Tensor:
        if base_name in named_args:
            return named_args[base_name]
        return named_args[f"{base_name}_host"]

    # Get reference tensors (prefer device version)
    cu_seqlen = get_tensor("cu_seqlen")
    position_ids = named_args["position_ids"]
    seq_len_with_cache = get_tensor("seq_len_with_cache")

    batch_size = len(increment)
    total_tokens = position_ids.numel()

    # Compute per-sequence token counts from cu_seqlen
    # seq_lens[i] = cu_seqlen[i+1] - cu_seqlen[i] = number of tokens for sequence i
    seq_lens = cu_seqlen[1 : batch_size + 1] - cu_seqlen[:batch_size]

    # Broadcast increment to each token position based on sequence lengths.
    # Since position_ids is flattened [seq0_tok0, seq0_tok1, ..., seq1_tok0, seq1_tok1, ...],
    # we need to expand increment from [batch_size] to [total_tokens] where each
    # increment[i] is repeated seq_lens[i] times to match the token layout.
    # Example: increment=[0,2,1], seq_lens=[3,2,4] -> broadcast_increment=[0,0,0,2,2,1,1,1,1]
    broadcast_increment = torch.repeat_interleave(increment, seq_lens)

    # Helper to update both device and host versions of a tensor
    def update_tensor(base_name: str, new_value: torch.Tensor) -> None:
        if base_name in named_args:
            result[base_name] = new_value
        host_name = f"{base_name}_host"
        if host_name in named_args:
            result[host_name] = new_value.cpu()

    # Update position_ids (no host version typically)
    new_position_ids = position_ids.clone()
    new_position_ids.view(-1)[:total_tokens] += broadcast_increment
    result["position_ids"] = new_position_ids

    # Update seq_len_with_cache (device + host)
    new_seq_len_with_cache = seq_len_with_cache.clone()
    new_seq_len_with_cache[:batch_size] += increment
    update_tensor("seq_len_with_cache", new_seq_len_with_cache)

    # Update last_page_len (device + host)
    last_page_len = get_tensor("last_page_len")
    new_last_page_len = last_page_len.clone()
    new_last_page_len[:batch_size] = (new_seq_len_with_cache[:batch_size] - 1) % page_size + 1
    update_tensor("last_page_len", new_last_page_len)

    # Compute extra pages needed per sequence
    old_pages = (seq_len_with_cache[:batch_size] + page_size - 1) // page_size
    new_pages = (new_seq_len_with_cache[:batch_size] + page_size - 1) // page_size
    extra_pages = new_pages - old_pages

    # new_pages_per_seq: always defined (for cache_loc rebuild); from buffer when present, else computed
    has_pages_per_seq = "pages_per_seq" in named_args or "pages_per_seq_host" in named_args
    new_pages_per_seq = new_pages.clone()
    if has_pages_per_seq:
        pages_per_seq = get_tensor("pages_per_seq")
        new_pages_per_seq = pages_per_seq.clone()
        new_pages_per_seq[:batch_size] += extra_pages
        update_tensor("pages_per_seq", new_pages_per_seq)

    # Update cu_num_pages (device + host)
    has_cu_num_pages = "cu_num_pages" in named_args or "cu_num_pages_host" in named_args
    if has_cu_num_pages:
        cu_num_pages = get_tensor("cu_num_pages")
        new_cu_num_pages = cu_num_pages.clone()
        # Add cumulative extra pages directly to cu_num_pages
        # (equivalent to recomputing cumsum from pages_per_seq since cumsum is linear)
        cumulative_extra_pages = torch.cumsum(extra_pages, dim=0)
        new_cu_num_pages[1 : batch_size + 1] += cumulative_extra_pages
        update_tensor("cu_num_pages", new_cu_num_pages)

    # Rebuild cache_loc from full allocation when used page counts per sequence changed.
    # Use old_pages/new_pages (computed from seq_len_with_cache) so we don't rely on optional pages_per_seq.
    has_full_allocation = "full_cache_loc" in named_args and "cu_allocated_pages" in named_args
    if has_full_allocation:
        cu_allocated_pages_tensor = named_args["cu_allocated_pages"]
        allocated_per_seq = (
            cu_allocated_pages_tensor[1 : batch_size + 1] - cu_allocated_pages_tensor[:batch_size]
        )
        used_per_seq = new_pages_per_seq[:batch_size]
        assert (used_per_seq <= allocated_per_seq).all(), (
            f"used pages_per_seq exceeds allocated: "
            f"used={used_per_seq.tolist()}, allocated={allocated_per_seq.tolist()}"
        )
    rebuild_cache_loc = has_full_allocation and (new_pages != old_pages).any().item()
    if rebuild_cache_loc:
        assert (new_pages >= 1).all(), (
            "page count must be >= 1 per sequence after increment (each sequence needs at least one page)"
        )
        full_cache_loc = named_args["full_cache_loc"]
        cu_allocated_pages = named_args["cu_allocated_pages"]
        new_cache_loc = build_cache_loc_from_full(
            full_cache_loc,
            cu_allocated_pages,
            new_pages_per_seq[:batch_size],
            batch_size,
        )
        # Ensure same dtype/device as buffer (full_cache_loc is already on target device)
        new_cache_loc = new_cache_loc.to(device=full_cache_loc.device, dtype=full_cache_loc.dtype)
        # If the existing buffer is larger than the new compact tensor, write prefix and zero the rest.
        cache_loc_buf = get_tensor("cache_loc") if "cache_loc" in named_args else None
        if cache_loc_buf is not None and cache_loc_buf.numel() > new_cache_loc.numel():
            out = cache_loc_buf.clone()
            out[: new_cache_loc.numel()].copy_(new_cache_loc)
            out[new_cache_loc.numel() :].zero_()
            update_tensor("cache_loc", out)
        else:
            update_tensor("cache_loc", new_cache_loc)

    # Zero out garbage values beyond the valid batch_size region.
    # This prevents stale data from previous (larger) batches from affecting computations.
    # Tensors are typically pre-allocated to max_batch_size, so we need to zero the unused portions.

    def zero_garbage(base_name: str, valid_len: int) -> None:
        """Zero out elements beyond valid_len for both device and host versions."""
        if base_name in result:
            tensor = result[base_name]
            if tensor.numel() > valid_len:
                tensor[valid_len:].zero_()
        host_name = f"{base_name}_host"
        if host_name in result:
            tensor = result[host_name]
            if tensor.numel() > valid_len:
                tensor[valid_len:].zero_()

    # Zero garbage in cumulative tensors (valid: batch_size + 1 elements)
    zero_garbage("cu_seqlen", batch_size + 1)
    zero_garbage("cu_num_pages", batch_size + 1)

    # Zero garbage in per-sequence tensors (valid: batch_size elements)
    zero_garbage("seq_len_with_cache", batch_size)
    zero_garbage("last_page_len", batch_size)
    zero_garbage("pages_per_seq", batch_size)

    # Zero garbage in position_ids (valid: total_tokens elements)
    if new_position_ids.numel() > total_tokens:
        new_position_ids.view(-1)[total_tokens:].zero_()

    return result
