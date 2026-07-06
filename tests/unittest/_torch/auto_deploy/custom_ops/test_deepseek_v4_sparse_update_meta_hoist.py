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
"""Unit test for the hoisted DeepSeek-V4 compressed-cache UPDATE metadata (idea_0044).

``deepseek_v4_sparse_prepare_decode_page_addr`` now also emits, once per forward, the
per-decode-row update metadata shared by every layer of a compression ratio:
``row_valid`` (does this step complete a compressed row), the query-relative RoPE
position of that row, the ``mhc_cache`` write address, and -- for the dense ratio-128
layers -- the ``[num_seq, ratio]`` compressor read page map.  ``_update_decode_compressed_caches``
consumes it via ``update_meta`` instead of recomputing the chain per layer.

These tests pin (a) that the hoisted metadata is bit-identical to an independent
per-layer reference across page boundaries, out-of-range rows, and multi-sequence
inputs, and (b) that feeding the hoisted metadata into the update produces a
byte-identical ``mhc_cache`` to the un-hoisted per-layer path, for both ratio-4 and
ratio-128.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M

_RATIO4 = 4
_RATIO128 = 128


def _ref_update_metadata(input_pos, position_ids, seq_idx, cu_num_pages, cache_loc, tpb, ratio, m):
    """Independent reference: the original per-layer update-metadata chain.

    Deliberately does NOT call ``M._compressed_row_update_metadata`` so it is a genuine
    cross-check of the shared helper / hoisted op.
    """
    input_pos = input_pos.to(torch.long)
    position_ids = position_ids.to(torch.long)
    old_completed = input_pos // ratio
    new_completed = (input_pos + 1) // ratio
    row_valid = (new_completed > old_completed) & (old_completed < m)
    row_idx = old_completed.clamp(min=0, max=m - 1)
    row_position_id = position_ids - (input_pos - row_idx * ratio)
    row_logical_pos = row_idx * ratio
    total_pages = int(cu_num_pages[-1].item())
    dummy_cache = torch.zeros(total_pages, tpb, 4, device=cache_loc.device)
    mhc_pid, mhc_poff, _ = M._decode_page_ids_and_offsets(
        dummy_cache, seq_idx, row_logical_pos, cu_num_pages, cache_loc
    )
    offsets = torch.arange(ratio, dtype=torch.long, device=input_pos.device)
    positions = row_logical_pos.unsqueeze(1) + offsets.view(1, -1)
    pos_pid, pos_poff, _ = M._decode_page_ids_and_offsets(
        dummy_cache, seq_idx, positions, cu_num_pages, cache_loc
    )
    return row_valid, row_position_id, mhc_pid, mhc_poff, pos_pid, pos_poff


@pytest.mark.parametrize("tokens_per_block", [8, 32])
@pytest.mark.parametrize("num_seq", [1, 2, 3])
def test_prepare_update_metadata_matches_reference(tokens_per_block, num_seq):
    torch.manual_seed(num_seq * 17 + tokens_per_block)
    overlap_m = 13
    dense_m = 5
    max_blocks_per_seq = 4096 // tokens_per_block

    cu_num_pages = torch.arange(
        0, (num_seq + 1) * max_blocks_per_seq, max_blocks_per_seq, dtype=torch.long
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.randperm(total_pages, dtype=torch.long)

    # Positions include a page-boundary hit (multiple of tokens_per_block), an
    # out-of-range-ish large value and small values so validity / clamping is exercised.
    positions = torch.tensor(
        [tokens_per_block * (i + 1) + (i % 3) for i in range(num_seq)], dtype=torch.long
    )
    input_pos = torch.cat([positions, torch.zeros(4, dtype=torch.long)])
    # position_ids == input_pos on decode (seq_len == 1); mirror that here, but the
    # reference uses the general formula so a divergence would still be caught.
    position_ids = input_pos.clone()

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    outs = prep(
        input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, overlap_m, dense_m
    )
    assert len(outs) == 18, "production contract must return 2 + 6 + 10 tensors"

    seq_idx = torch.arange(num_seq, dtype=torch.long)
    pos_seq = input_pos.reshape(-1)[:num_seq]
    pid_seq = position_ids.reshape(-1)[:num_seq]

    # Ratio-4 update bundle: outs[8..11] (no pos map for the overlap path).
    r4 = _ref_update_metadata(
        pos_seq, pid_seq, seq_idx, cu_num_pages, cache_loc, tokens_per_block, _RATIO4, overlap_m
    )
    assert torch.equal(outs[8][:num_seq], r4[0]), "r4 row_valid"
    assert torch.equal(outs[9][:num_seq], r4[1]), "r4 row_position_id"
    assert torch.equal(outs[10][:num_seq], r4[2]), "r4 mhc_page_ids"
    assert torch.equal(outs[11][:num_seq], r4[3]), "r4 mhc_page_offsets"

    # Ratio-128 update bundle: outs[12..17] incl. the [num_seq, ratio] read page map.
    r128 = _ref_update_metadata(
        pos_seq, pid_seq, seq_idx, cu_num_pages, cache_loc, tokens_per_block, _RATIO128, dense_m
    )
    assert torch.equal(outs[12][:num_seq], r128[0]), "r128 row_valid"
    assert torch.equal(outs[13][:num_seq], r128[1]), "r128 row_position_id"
    assert torch.equal(outs[14][:num_seq], r128[2]), "r128 mhc_page_ids"
    assert torch.equal(outs[15][:num_seq], r128[3]), "r128 mhc_page_offsets"
    assert outs[16].shape == (num_seq, _RATIO128)
    assert torch.equal(outs[16][:num_seq], r128[4]), "r128 pos_page_ids"
    assert torch.equal(outs[17][:num_seq], r128[5]), "r128 pos_page_offsets"


def test_prepare_update_metadata_fake_shape():
    """Fake (meta) path returns the 18 tensors with the ratio-128 [num_seq, ratio] shape."""
    tokens_per_block = 32
    num_seq = 2
    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    with torch._subclasses.FakeTensorMode():
        fake_pos = torch.empty(num_seq, dtype=torch.long)
        fake_pids = torch.empty(num_seq, dtype=torch.long)
        fake_cu = torch.empty(num_seq + 1, dtype=torch.long)
        fake_loc = torch.empty(256, dtype=torch.long)
        outs = prep(fake_pos, fake_pids, fake_cu, fake_loc, tokens_per_block, 7, 3)
    assert len(outs) == 18
    assert outs[8].dtype == torch.bool and outs[8].shape == (num_seq,)  # r4_row_valid
    assert outs[12].dtype == torch.bool and outs[12].shape == (num_seq,)  # r128_row_valid
    assert outs[16].dtype == torch.long and outs[16].shape == (num_seq, _RATIO128)  # r128 pos ids
    assert outs[17].dtype == torch.long and outs[17].shape == (num_seq, _RATIO128)  # r128 pos offs


def _build_update_inputs(compress_ratio, num_rows, seed):
    """Mirror test_deepseek_v4_compressed_row_update._build_inputs, on CUDA."""
    torch.manual_seed(seed)
    dev = "cuda"
    head_dim = 512
    rope_dim = 64
    channels = 2 if compress_ratio == _RATIO4 else 1
    state_dim = channels * head_dim
    tokens_per_block = 8
    max_compressed_len = 4
    eps = 1e-6
    dtype = torch.bfloat16

    max_pos = max_compressed_len * compress_ratio + tokens_per_block
    pages_per_seq = (max_pos + tokens_per_block - 1) // tokens_per_block
    cu_num_pages = torch.tensor(
        [0, *torch.tensor([pages_per_seq] * num_rows).cumsum(0).tolist()],
        dtype=torch.long,
        device=dev,
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=dev)

    kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=dev)
    gate_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=dev)
    mhc_cache = torch.randn(total_pages, tokens_per_block, head_dim, device=dev, dtype=dtype)

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=dev)
    input_pos = torch.tensor(
        [compress_ratio - 1 + i * compress_ratio + (i % 2) for i in range(num_rows)],
        dtype=torch.long,
        device=dev,
    )
    position_ids = input_pos.clone()

    compressor_kv_decode = torch.randn(num_rows, state_dim, device=dev, dtype=dtype)
    compressor_gate_decode = torch.randn(num_rows, state_dim, device=dev, dtype=dtype)
    ape = torch.randn(compress_ratio, state_dim, device=dev)
    norm_weight = torch.randn(head_dim, device=dev, dtype=dtype)
    n_pos = int(input_pos.max().item()) + 4
    cos_table = torch.randn(n_pos, rope_dim // 2, device=dev)
    sin_table = torch.randn(n_pos, rope_dim // 2, device=dev)

    prep = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr
    overlap_m = max_compressed_len if compress_ratio == _RATIO4 else 1
    dense_m = max_compressed_len if compress_ratio == _RATIO128 else 1
    outs = prep(
        input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, overlap_m, dense_m
    )
    overlap_page_map = None
    update_meta = None
    if compress_ratio == _RATIO4:
        overlap_page_map = (outs[2][:num_rows], outs[3][:num_rows], outs[4][:num_rows])
        update_meta = (
            outs[8][:num_rows],
            outs[9][:num_rows],
            outs[10][:num_rows],
            outs[11][:num_rows],
            None,
            None,
        )
    else:
        update_meta = (
            outs[12][:num_rows],
            outs[13][:num_rows],
            outs[14][:num_rows],
            outs[15][:num_rows],
            outs[16][:num_rows],
            outs[17][:num_rows],
        )

    common = dict(
        compressor_kv_decode=compressor_kv_decode,
        compressor_gate_decode=compressor_gate_decode,
        position_ids_decode=position_ids,
        compressor_ape=ape,
        compressor_norm_weight=norm_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        seq_idx=seq_idx,
        input_pos=input_pos,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
        compressor_kv_cache=kv_cache,
        compressor_gate_cache=gate_cache,
        rms_norm_eps=eps,
        rope_dim=rope_dim,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        overlap_page_map=overlap_page_map,
    )
    return common, mhc_cache, update_meta


def _run_update(common, mhc, update_meta):
    M._update_decode_compressed_caches(
        common["compressor_kv_decode"],
        common["compressor_gate_decode"],
        common["position_ids_decode"],
        common["compressor_ape"],
        common["compressor_norm_weight"],
        common["cos_table"],
        common["sin_table"],
        common["seq_idx"],
        common["input_pos"],
        common["cu_num_pages"],
        common["cache_loc"],
        mhc,
        common["compressor_kv_cache"],
        common["compressor_gate_cache"],
        common["rms_norm_eps"],
        common["rope_dim"],
        common["compress_ratio"],
        common["max_compressed_len"],
        common["overlap_page_map"],
        update_meta,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [_RATIO4, _RATIO128])
@pytest.mark.parametrize("num_rows", [1, 3])
def test_update_with_hoisted_meta_byte_exact(compress_ratio, num_rows):
    """Feeding the hoisted update metadata must give a byte-identical mhc_cache to the
    un-hoisted per-layer path (both drive the same fused reconstruction+store)."""
    assert M._HAS_TRITON, "test requires triton"
    common, mhc_base, update_meta = _build_update_inputs(
        compress_ratio, num_rows, seed=400 + compress_ratio + num_rows
    )

    mhc_local = mhc_base.clone()
    _run_update(common, mhc_local, update_meta=None)  # per-layer metadata

    mhc_hoisted = mhc_base.clone()
    _run_update(common, mhc_hoisted, update_meta=update_meta)  # hoisted metadata

    assert torch.equal(mhc_local, mhc_hoisted), (
        f"hoisted update_meta changed the store (ratio={compress_ratio}, rows={num_rows})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
