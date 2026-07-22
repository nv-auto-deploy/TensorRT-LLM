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
"""Tests for ``deepseek_v4_sparse_prepare_decode_page_addr`` and its consumers."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import _window_topk_idxs

_RATIO4 = 4
_RATIO128 = 128


def _page_table(num_seq: int, tokens_per_block: int, seed: int):
    # Shuffled page pool so an identity translation would be detected.
    torch.manual_seed(seed)
    max_pages = 2048 // tokens_per_block
    pages_per_seq = torch.randint(1, max_pages + 1, (num_seq,), dtype=torch.long)
    cu_num_pages = torch.cat([torch.zeros(1, dtype=torch.long), pages_per_seq.cumsum(0)])
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, dtype=torch.long)
    return pages_per_seq, cu_num_pages, total_pages, cache_loc


def _edge_positions(num_seq: int, pages_per_seq: torch.Tensor, tokens_per_block: int, w: int):
    # Window rollover, page boundaries, row-completion boundaries, last token,
    # out-of-range and padded rows, cycled across sequences.
    positions = torch.empty(num_seq, dtype=torch.long)
    for i in range(num_seq):
        limit = int(pages_per_seq[i].item()) * tokens_per_block
        kinds = [
            0,
            w - 1,
            tokens_per_block - 1,
            127,
            limit - 1,
            limit + 5,
            -1,
            int(torch.randint(0, limit, (1,)).item()),
        ]
        positions[i] = kinds[i % len(kinds)]
    return positions


def _scalar_page_map(positions, cu_num_pages, cache_loc, tokens_per_block):
    # Independent scalar host page translation (does NOT call the shared
    # production helper), including the invalid-row fallback values.
    positions = positions.to(torch.long)
    squeeze = positions.dim() == 1
    if squeeze:
        positions = positions.unsqueeze(1)
    ids = torch.empty_like(positions)
    offs = torch.empty_like(positions)
    valid = torch.empty(positions.shape, dtype=torch.bool)
    for s in range(positions.shape[0]):
        start = int(cu_num_pages[s].item())
        end = int(cu_num_pages[s + 1].item())
        for j in range(positions.shape[1]):
            p = int(positions[s, j].item())
            sp = max(p, 0)
            ordinal, off = divmod(sp, tokens_per_block)
            idx = start + ordinal
            ok = p >= 0 and idx < end
            safe = idx if ok else start
            safe = min(max(safe, 0), cache_loc.numel() - 1)
            ids[s, j] = int(cache_loc[safe].item())
            offs[s, j] = off
            valid[s, j] = ok
    if squeeze:
        return ids.squeeze(1), offs.squeeze(1), valid.squeeze(1)
    return ids, offs, valid


def _ref_update_metadata(input_pos, position_ids, cu_num_pages, cache_loc, tpb, ratio, m):
    # Independent per-step compressed-cache update metadata reference.
    input_pos = input_pos.to(torch.long)
    position_ids = position_ids.to(torch.long)
    old_completed = input_pos // ratio
    new_completed = (input_pos + 1) // ratio
    row_valid = (new_completed > old_completed) & (old_completed < m)
    row_idx = old_completed.clamp(min=0, max=m - 1)
    row_position_id = position_ids - (input_pos - row_idx * ratio)
    row_logical_pos = row_idx * ratio
    mhc_pid, mhc_poff, _ = _scalar_page_map(row_logical_pos, cu_num_pages, cache_loc, tpb)
    offsets = torch.arange(ratio, dtype=torch.long)
    positions = row_logical_pos.unsqueeze(1) + offsets.view(1, -1)
    pos_pid, pos_poff, _ = _scalar_page_map(positions, cu_num_pages, cache_loc, tpb)
    return row_valid, row_position_id, mhc_pid, mhc_poff, pos_pid, pos_poff


@pytest.mark.parametrize(
    "tokens_per_block,num_seq,overlap_m,dense_m,window_size",
    [
        (32, 3, 13, 5, 4),
        (64, 8, 37, 3, 128),
    ],
)
def test_prepare_full_contract_matches_independent_reference(
    tokens_per_block, num_seq, overlap_m, dense_m, window_size
):
    pages_per_seq, cu_num_pages, total_pages, cache_loc = _page_table(
        num_seq, tokens_per_block, seed=num_seq * 31 + tokens_per_block
    )
    positions = _edge_positions(num_seq, pages_per_seq, tokens_per_block, window_size)
    # Buffer padded beyond num_seq; position_ids offset to catch swapped operands.
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])
    position_ids = input_pos + 3

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    outs = prep(
        input_pos,
        position_ids,
        cu_num_pages,
        cache_loc,
        tokens_per_block,
        overlap_m,
        dense_m,
        window_size,
    )
    assert len(outs) == 23

    pos_seq = positions
    pid_seq = position_ids.reshape(-1)[:num_seq]

    # outs[0:2] current-token write address.
    cur_pid, cur_poff, _ = _scalar_page_map(pos_seq, cu_num_pages, cache_loc, tokens_per_block)
    assert torch.equal(outs[0], cur_pid)
    assert torch.equal(outs[1], cur_poff)

    # outs[2:5] ratio-4 overlap band [anchor - ratio, anchor + ratio).
    row_idx = (pos_seq // _RATIO4).clamp(min=0, max=overlap_m - 1)
    band = torch.arange(2 * _RATIO4, dtype=torch.long) - _RATIO4
    ovl_positions = (row_idx * _RATIO4).unsqueeze(1) + band.view(1, -1)
    ovl = _scalar_page_map(ovl_positions, cu_num_pages, cache_loc, tokens_per_block)
    for i in range(3):
        assert torch.equal(outs[2 + i], ovl[i]), f"overlap map output {i}"

    # outs[5:8] full candidate range [0, overlap_m * ratio).
    full_positions = torch.arange(overlap_m * _RATIO4, dtype=torch.long)
    full_positions = full_positions.view(1, -1).expand(num_seq, -1)
    full = _scalar_page_map(full_positions, cu_num_pages, cache_loc, tokens_per_block)
    for i in range(3):
        assert torch.equal(outs[5 + i], full[i]), f"full-range map output {i}"

    # outs[8:12] ratio-4 and outs[12:18] ratio-128 update metadata bundles.
    r4 = _ref_update_metadata(
        pos_seq, pid_seq, cu_num_pages, cache_loc, tokens_per_block, _RATIO4, overlap_m
    )
    r128 = _ref_update_metadata(
        pos_seq, pid_seq, cu_num_pages, cache_loc, tokens_per_block, _RATIO128, dense_m
    )
    for i in range(4):
        assert torch.equal(outs[8 + i], r4[i]), f"r4 update meta output {i}"
    for i in range(6):
        assert torch.equal(outs[12 + i], r128[i]), f"r128 update meta output {i}"

    # outs[18:21] SWA local-window bundle: page map + precombined rel_topk.
    w_offsets = torch.arange(window_size, dtype=torch.long)
    w_positions = pos_seq.unsqueeze(1) - window_size + 1 + w_offsets.view(1, -1)
    swa_pid, swa_poff, page_valid = _scalar_page_map(
        w_positions, cu_num_pages, cache_loc, tokens_per_block
    )
    swa_valid = (w_positions >= 0) & (w_positions <= pos_seq.unsqueeze(1)) & page_valid
    rel = w_offsets.view(1, -1).expand(num_seq, -1)
    ref_rel = torch.where(swa_valid, rel, torch.full_like(rel, -1))
    assert torch.equal(outs[18], swa_pid)
    assert torch.equal(outs[19], swa_poff)
    assert torch.equal(outs[20], ref_rel)

    # outs[21:23] hoisted long decode metadata.
    assert torch.equal(outs[21], torch.arange(num_seq, dtype=torch.long))
    assert torch.equal(outs[22], pos_seq.to(torch.long))

    # Reduced arities are prefixes of the full contract.
    outs2 = prep(input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block)
    assert len(outs2) == 2
    outs18 = prep(
        input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, overlap_m, dense_m
    )
    assert len(outs18) == 18
    for a, b in zip(outs2 + outs18, outs[:2] + outs[:18]):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton path")
@pytest.mark.parametrize(
    "idx_dtype,tokens_per_block,num_seq,overlap_m,dense_m,window_size",
    [
        (torch.int32, 32, 8, 512, 16, 128),  # production dtypes, large maps
        (torch.long, 64, 3, 37, 1, 5),  # int64 inputs, non-pow2 window
        (torch.int32, 64, 1, 37, 5, 128),  # single sequence
    ],
)
def test_prepare_triton_matches_torch_reference(
    idx_dtype, tokens_per_block, num_seq, overlap_m, dense_m, window_size
):
    # CPU takes the torch body, CUDA the single-launch Triton path.
    pages_per_seq, cu_num_pages, _, cache_loc = _page_table(
        num_seq, tokens_per_block, seed=num_seq * 1009 + tokens_per_block + window_size
    )
    positions = _edge_positions(num_seq, pages_per_seq, tokens_per_block, window_size)
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])
    position_ids = input_pos + 3

    args_cpu = (
        input_pos.to(idx_dtype),
        position_ids.to(torch.long),
        cu_num_pages.to(idx_dtype),
        cache_loc.to(idx_dtype),
    )
    args_gpu = tuple(t.cuda() for t in args_cpu)

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    ref = prep(*args_cpu, tokens_per_block, overlap_m, dense_m, window_size)
    outs = prep(*args_gpu, tokens_per_block, overlap_m, dense_m, window_size)
    assert len(ref) == 23 and len(outs) == 23
    for i, (o, r) in enumerate(zip(outs, ref)):
        assert o.is_cuda, f"output {i} expected on CUDA"
        assert o.dtype == r.dtype, f"output {i} dtype {o.dtype} != {r.dtype}"
        assert o.shape == r.shape, f"output {i} shape {o.shape} != {r.shape}"
        assert torch.equal(o.cpu(), r), f"output {i} diverges from torch reference"


def test_prepare_fake_shapes():
    tokens_per_block = 32
    num_seq = 2
    m = 11
    w = 128

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    with torch._subclasses.FakeTensorMode():
        fake_pos = torch.empty(num_seq, dtype=torch.long)
        fake_pids = torch.empty(num_seq, dtype=torch.long)
        fake_cu = torch.empty(num_seq + 1, dtype=torch.long)
        fake_loc = torch.empty(128, dtype=torch.long)
        outs2 = prep(fake_pos, fake_pids, fake_cu, fake_loc, tokens_per_block)
        outs18 = prep(fake_pos, fake_pids, fake_cu, fake_loc, tokens_per_block, m, 3)
        outs23 = prep(fake_pos, fake_pids, fake_cu, fake_loc, tokens_per_block, m, 3, w)

    assert len(outs2) == 2 and len(outs18) == 18 and len(outs23) == 23
    for t in outs2:
        assert t.shape == (num_seq,) and t.dtype == torch.long
    assert outs18[2].shape == (num_seq, 2 * _RATIO4) and outs18[2].dtype == torch.long
    assert outs18[4].shape == (num_seq, 2 * _RATIO4) and outs18[4].dtype == torch.bool
    assert outs18[5].shape == (num_seq, m * _RATIO4) and outs18[5].dtype == torch.long
    assert outs18[7].shape == (num_seq, m * _RATIO4) and outs18[7].dtype == torch.bool
    assert outs18[8].dtype == torch.bool and outs18[8].shape == (num_seq,)  # r4_row_valid
    assert outs18[12].dtype == torch.bool and outs18[12].shape == (num_seq,)  # r128_row_valid
    for i in (16, 17):  # r128 pos map is [num_seq, dense ratio]
        assert outs18[i].shape == (num_seq, _RATIO128) and outs18[i].dtype == torch.long
    for i in (18, 19, 20):
        assert outs23[i].shape == (num_seq, w) and outs23[i].dtype == torch.long
    for i in (21, 22):
        assert outs23[i].shape == (num_seq,) and outs23[i].dtype == torch.long


def test_page_map_reuse_across_caches():
    # One translation serves every cache sharing the page table + tokens_per_block:
    # both for map computation and for gathers fed a precomputed page_map.
    num_seq, tokens_per_block = 2, 32
    _, cu_num_pages, total_pages, cache_loc = _page_table(num_seq, tokens_per_block, seed=9)
    seq_idx = torch.arange(num_seq, dtype=torch.long)
    positions = torch.randint(-3, 2048, (num_seq, 4), dtype=torch.long)

    cache_kv = torch.randn(total_pages, tokens_per_block, 6, dtype=torch.float32)
    cache_gate = torch.randn(total_pages, tokens_per_block, 3, dtype=torch.float32)

    map_kv = M._decode_page_ids_and_offsets(cache_kv, seq_idx, positions, cu_num_pages, cache_loc)
    map_gate = M._decode_page_ids_and_offsets(
        cache_gate, seq_idx, positions, cu_num_pages, cache_loc
    )
    for a, b in zip(map_kv, map_gate):
        assert torch.equal(a, b)

    ref_kv, ref_kv_valid = M._decode_cache_rows_from_positions(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16
    )
    ref_gate, _ = M._decode_cache_rows_from_positions(
        cache_gate, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16
    )
    new_kv, new_kv_valid = M._decode_cache_rows_from_positions(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, map_kv
    )
    new_gate, _ = M._decode_cache_rows_from_positions(
        cache_gate, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, map_kv
    )
    assert torch.equal(ref_kv, new_kv)
    assert torch.equal(ref_gate, new_gate)
    assert torch.equal(ref_kv_valid, new_kv_valid)

    # valid=None page maps (the MHC read+write commoning) still gather correctly.
    ids, offs, _ = map_kv
    rows_none_valid, valid_none = M._decode_cache_rows_from_positions(
        cache_kv, seq_idx, positions, cu_num_pages, cache_loc, torch.bfloat16, (ids, offs, None)
    )
    assert torch.equal(rows_none_valid, ref_kv)
    assert valid_none is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the decode path")
@pytest.mark.parametrize("window_size", [4, 128])
def test_decode_attention_hoisted_swa_map_matches_per_layer(window_size):
    torch.manual_seed(20260708 + window_size)
    num_seq = 6
    tokens_per_block = 32
    num_heads, head_dim = 2, 64
    pages_per_seq, cu_num_pages, total_pages, cache_loc = _page_table(
        num_seq, tokens_per_block, seed=window_size
    )
    positions = _edge_positions(num_seq, pages_per_seq, tokens_per_block, window_size)
    positions = positions.clamp(min=0)  # decode only sees active rows

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
    outs = prep(input_pos, input_pos + 3, cu_gpu, loc_gpu, tokens_per_block, 37, 5, window_size)
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


@pytest.mark.parametrize("batch_size,seq_len,window_size", [(1, 1, 4), (2, 9, 3)])
def test_window_only_placeholder_matches_explicit(batch_size, seq_len, window_size):
    # fp32 + head_dim < 16 keeps the attend on the deterministic reference chunk
    # loop, so the two invocations are exactly comparable with torch.equal.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
