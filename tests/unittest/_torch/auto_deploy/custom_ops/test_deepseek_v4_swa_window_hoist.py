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
"""Byte-exactness tests for the hoisted DeepSeek-V4 SWA local-window decode bundle.

``deepseek_v4_sparse_prepare_decode_page_addr`` now also emits, once per forward,
the SWA local-window page map + precombined ``rel_topk`` shared by every
window-only (``compress_ratio == 0``) layer (idea_0086). These tests pin that:

* the hoisted ``(page_ids, page_offsets, rel_topk)`` bundle is bit-identical to
  the per-layer chain it replaces (``_fused_local_window_pagemap`` /
  ``_decode_local_cache_rows`` position generation + the
  ``_decode_attention_from_rows`` ``rel_topk`` construction), including page
  boundaries, window rollover at the sequence start, and out-of-range rows;
* the Triton single-launch path matches the torch reference body for the full
  21-output contract;
* decode attention consuming the hoisted bundle is ``torch.equal`` to the
  per-layer fallback and leaves the cache bytes untouched;
* the window-only ``topk_is_placeholder`` emission is bit-identical to the
  explicit model-side ``_window_topk_idxs`` chain it replaces.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import _window_topk_idxs


def _page_table(num_seq: int, tokens_per_block: int, seed: int):
    """Shuffled per-sequence page table so an identity translation is detected."""
    torch.manual_seed(seed)
    max_pages = 2048 // tokens_per_block
    pages_per_seq = torch.randint(1, max_pages + 1, (num_seq,), dtype=torch.long)
    cu_num_pages = torch.cat([torch.zeros(1, dtype=torch.long), pages_per_seq.cumsum(0)])
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, dtype=torch.long)
    return pages_per_seq, cu_num_pages, total_pages, cache_loc


def _edge_positions(num_seq: int, pages_per_seq: torch.Tensor, tokens_per_block: int, w: int):
    """Positions cycling window rollover, page boundaries, last token, and padding."""
    positions = torch.empty(num_seq, dtype=torch.long)
    for i in range(num_seq):
        limit = int(pages_per_seq[i].item()) * tokens_per_block
        kinds = [
            0,  # window rollover: only the last slot is a valid position
            w - 1,  # window exactly fills the prefix
            tokens_per_block - 1,  # window straddles the first page boundary
            tokens_per_block,  # first position of the second page
            limit - 1,  # last allocated token
            limit + 5,  # out-of-range row (page-table bound masks the tail)
            -1,  # negative (padded) row
            int(torch.randint(0, limit, (1,)).item()),
        ]
        positions[i] = kinds[i % len(kinds)]
    return positions


def _reference_swa_bundle(
    positions: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    tokens_per_block: int,
    total_pages: int,
    w: int,
):
    """The per-layer chain the hoist replaces (fallback path of
    ``_decode_local_cache_rows`` + ``_decode_attention_from_rows``)."""
    num_seq = positions.shape[0]
    seq_idx = torch.arange(num_seq, dtype=torch.long)
    offsets = torch.arange(w, dtype=torch.long)
    w_positions = positions.unsqueeze(1) - w + 1 + offsets.view(1, -1)
    pos_valid = (w_positions >= 0) & (w_positions <= positions.unsqueeze(1))
    cache = torch.zeros(total_pages, tokens_per_block, 4)
    ref_ids, ref_offs, page_valid = M._decode_page_ids_and_offsets(
        cache, seq_idx, w_positions, cu_num_pages, cache_loc
    )
    valid = pos_valid & page_valid
    rel = offsets.view(1, -1).expand(num_seq, -1)
    ref_rel = torch.where(valid, rel, torch.full_like(rel, -1))
    return ref_ids, ref_offs, ref_rel, valid


@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("window_size", [4, 128])
@pytest.mark.parametrize("num_seq", [1, 3, 8])
def test_prepare_swa_bundle_matches_per_layer_reference(tokens_per_block, window_size, num_seq):
    """Torch body: the hoisted SWA bundle is bit-identical to the per-layer chain."""
    pages_per_seq, cu_num_pages, total_pages, cache_loc = _page_table(
        num_seq, tokens_per_block, seed=num_seq * 31 + tokens_per_block + window_size
    )
    positions = _edge_positions(num_seq, pages_per_seq, tokens_per_block, window_size)
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])
    position_ids = input_pos + 3

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    outs = prep(
        input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, 37, 5, window_size
    )
    assert len(outs) == 23, "window request must extend the contract to 23 outputs"
    swa_pid, swa_poff, swa_rel = outs[18:21]
    assert swa_pid.shape == (num_seq, window_size)
    assert swa_rel.dtype == torch.long
    # Once-per-forward long decode metadata (idea_0090): the exact arange /
    # widened input_pos every layer's decode path used to rebuild per call.
    assert torch.equal(outs[21], torch.arange(num_seq, dtype=torch.long))
    assert torch.equal(outs[22], input_pos.reshape(-1)[:num_seq].to(torch.long))

    ref_ids, ref_offs, ref_rel, _ = _reference_swa_bundle(
        positions, cu_num_pages, cache_loc, tokens_per_block, total_pages, window_size
    )
    assert torch.equal(swa_pid, ref_ids)
    assert torch.equal(swa_poff, ref_offs)
    assert torch.equal(swa_rel, ref_rel)
    # Decode slices [:num_decode]; any prefix must also match.
    for nd in range(1, num_seq + 1):
        assert torch.equal(swa_pid[:nd], ref_ids[:nd])
        assert torch.equal(swa_rel[:nd], ref_rel[:nd])

    # Without a window request the legacy 18-output contract is unchanged.
    outs18 = prep(input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, 37, 5)
    assert len(outs18) == 18
    for a, b in zip(outs18, outs[:18]):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton path")
@pytest.mark.parametrize("window_size", [5, 128])
@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_seq", [1, 3, 8])
def test_prepare_swa_bundle_triton_matches_torch(window_size, tokens_per_block, num_seq):
    """Triton single-launch path: all 21 outputs bit-identical to the torch body.

    ``window_size=5`` exercises the non-power-of-two masking of the SWA block.
    """
    pages_per_seq, cu_num_pages, total_pages, cache_loc = _page_table(
        num_seq, tokens_per_block, seed=num_seq * 1013 + tokens_per_block + window_size
    )
    positions = _edge_positions(num_seq, pages_per_seq, tokens_per_block, window_size)
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])
    position_ids = input_pos + 3

    args_cpu = (
        input_pos.to(torch.int32),
        position_ids.to(torch.long),
        cu_num_pages.to(torch.int32),
        cache_loc.to(torch.int32),
    )
    args_gpu = tuple(t.cuda() for t in args_cpu)

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    ref = prep(*args_cpu, tokens_per_block, 37, 5, window_size)
    outs = prep(*args_gpu, tokens_per_block, 37, 5, window_size)
    assert len(ref) == 23 and len(outs) == 23
    for i, (o, r) in enumerate(zip(outs, ref)):
        assert o.is_cuda, f"output {i} expected on CUDA"
        assert o.dtype == r.dtype, f"output {i} dtype {o.dtype} != {r.dtype}"
        assert o.shape == r.shape, f"output {i} shape {o.shape} != {r.shape}"
        assert torch.equal(o.cpu(), r), f"output {i} diverges from torch reference"

    # The hoisted bundle must also match the per-layer fused kernel it replaces
    # (exact window IDs at page boundaries / rollover / padded rows).
    seq_idx = torch.arange(num_seq, dtype=torch.long, device="cuda")
    f_pid, f_poff, f_valid = M._fused_local_window_pagemap(
        args_gpu[0].reshape(-1)[:num_seq],
        seq_idx,
        args_gpu[2],
        args_gpu[3],
        window_size,
        tokens_per_block,
    )
    assert torch.equal(outs[18], f_pid)
    assert torch.equal(outs[19], f_poff)
    rel = torch.arange(window_size, dtype=torch.long, device="cuda")
    rel = rel.view(1, -1).expand(num_seq, -1)
    assert torch.equal(outs[20], torch.where(f_valid, rel, torch.full_like(rel, -1)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the decode path")
@pytest.mark.parametrize("window_size", [4, 128])
def test_decode_attention_hoisted_swa_map_matches_per_layer(window_size):
    """Decode attention over the hoisted bundle equals the per-layer fallback.

    The consumption path swaps only the address/mask source; the gather and the
    attend are unchanged, so outputs must be ``torch.equal`` and the cache reads
    must leave the cache bytes untouched.
    """
    torch.manual_seed(20260708 + window_size)
    num_seq = 6
    tokens_per_block = 32
    num_heads, head_dim = 2, 64
    pages_per_seq, cu_num_pages, total_pages, cache_loc = _page_table(
        num_seq, tokens_per_block, seed=window_size
    )
    positions = _edge_positions(num_seq, pages_per_seq, tokens_per_block, window_size)
    # The decode consumption only ever sees active (non-padded) rows.
    positions = positions.clamp(min=0)

    device = torch.device("cuda")
    input_pos = positions.to(device)
    seq_idx = torch.arange(num_seq, dtype=torch.long, device=device)
    cu_gpu = cu_num_pages.to(device)
    loc_gpu = cache_loc.to(device)
    swa_cache = torch.randn(
        total_pages, tokens_per_block, head_dim, dtype=torch.bfloat16, device=device
    )
    cache_before = swa_cache.clone()
    q_decode = torch.randn(num_seq, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    attn_sink = torch.randn(num_heads, dtype=torch.bfloat16, device=device)
    topk_unused = torch.zeros(num_seq, window_size, dtype=torch.long, device=device)

    out_fallback = M._decode_topk_cache_attention(
        q_decode,
        attn_sink,
        topk_unused,
        swa_cache,
        seq_idx,
        input_pos,
        cu_gpu,
        loc_gpu,
        0.25,
        window_size,
    )

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    outs = prep(
        input_pos,
        input_pos + 3,
        cu_gpu,
        loc_gpu,
        tokens_per_block,
        37,
        5,
        window_size,
    )
    out_hoisted = M._decode_topk_cache_attention(
        q_decode,
        attn_sink,
        topk_unused,
        swa_cache,
        seq_idx,
        input_pos,
        cu_gpu,
        loc_gpu,
        0.25,
        window_size,
        swa_page_map=(outs[18], outs[19], outs[20]),
    )

    assert torch.equal(out_hoisted, out_fallback)
    assert torch.equal(swa_cache, cache_before), "decode read path must not mutate the cache"


@pytest.mark.parametrize("batch_size,seq_len,window_size", [(1, 1, 4), (1, 12, 4), (2, 9, 3)])
def test_window_only_placeholder_matches_explicit(batch_size, seq_len, window_size):
    """Width-only placeholder + rebuild equals the explicit model-side window chain.

    fp32 + head_dim < 16 keeps the attend on the deterministic reference chunk loop,
    so the two invocations are exactly comparable with ``torch.equal``.
    """
    torch.manual_seed(20260708 + seq_len)
    device = torch.device("cpu")
    num_heads, head_dim = 2, 8
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    kv = torch.randn(batch_size, seq_len, head_dim, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)

    empties = (
        q.new_empty(batch_size, seq_len, 0),  # compressor_kv
        q.new_empty(batch_size, seq_len, 0),  # compressor_gate
        q.new_empty(0, 0),  # compressor_ape
        q.new_empty(0),  # compressor_norm_weight
        q.new_empty(0, 0),  # cos_table
        q.new_empty(0, 0),  # sin_table
        q.new_empty(batch_size, seq_len),  # position_ids
        q.new_empty(batch_size, seq_len, 0, 0),  # indexer_q
        q.new_empty(batch_size, seq_len, 0),  # indexer_weights
        q.new_empty(batch_size, seq_len, 0),  # indexer_compressor_kv
        q.new_empty(batch_size, seq_len, 0),  # indexer_compressor_gate
        q.new_empty(0, 0),  # indexer_compressor_ape
        q.new_empty(0),  # indexer_compressor_norm_weight
    )

    def _run(topk_idxs, topk_is_placeholder):
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            *empties,
            0.375,
            window_size=window_size,
            compress_ratio=0,
            topk_is_placeholder=topk_is_placeholder,
        )

    explicit = _window_topk_idxs(window_size, batch_size, seq_len, device).to(torch.int64)
    out_explicit = _run(explicit, topk_is_placeholder=False)

    width_only = q.new_empty(batch_size, seq_len, window_size, dtype=torch.int64)
    out_placeholder = _run(width_only, topk_is_placeholder=True)

    assert torch.equal(out_explicit, out_placeholder)

    # The op-side rebuild must be bit-identical to the model-side chain it replaces.
    rebuilt = M._build_window_topk_idxs(window_size, batch_size, seq_len, device)
    assert torch.equal(rebuilt.to(torch.int64), explicit)


def test_prepare_swa_fake_shapes():
    """Fake (meta) path returns 23 tensors with the SWA shapes for export/cudagraph."""
    tokens_per_block = 32
    num_seq = 2
    w = 128

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    with torch._subclasses.FakeTensorMode():
        fake_pos = torch.empty(num_seq, dtype=torch.long)
        fake_pids = torch.empty(num_seq, dtype=torch.long)
        fake_cu = torch.empty(num_seq + 1, dtype=torch.long)
        fake_loc = torch.empty(128, dtype=torch.long)
        outs = prep(fake_pos, fake_pids, fake_cu, fake_loc, tokens_per_block, 11, 1, w)
    assert len(outs) == 23
    for i in (18, 19, 20):
        assert outs[i].shape == (num_seq, w)
        assert outs[i].dtype == torch.long
    for i in (21, 22):
        assert outs[i].shape == (num_seq,)
        assert outs[i].dtype == torch.long


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
