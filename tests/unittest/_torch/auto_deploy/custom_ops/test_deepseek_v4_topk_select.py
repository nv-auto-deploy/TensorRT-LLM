# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused exact decode top-k row selection (idea_0046).

The decode ratio-4 selection tail in ``_select_decode_ratio4_indexer_rows`` runs

    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(int64), -1)
    # + -1/False pad up to index_topk when topk_count < index_topk

which costs a fat ``gatherTopK`` + ``radixSortKVInPlace`` pair plus the decomposed
``isfinite``/``where``/pad swarm per ratio-4 layer per decode step. idea_0046 replaces
it with one Triton kernel (``_dsv4_topk_select_kernel`` via ``_fused_topk_select``)
that sorts a packed (float-flip score, index) int64 key per candidate and emits the
padded ``topk_rows`` / ``topk_valid`` directly.

The contract is BYTE-exact equality with the eager tail on the same score tensor,
tie order included: torch's top-k at these shapes orders ties (equal values,
``+0.0`` == ``-0.0`` included) by ascending index, which the packed key reproduces
(``-0.0`` is canonicalized to ``+0.0`` before the bitcast).  This is pinned across
distinct scores, ``-inf`` visibility tails (partial / single-row / all-invalid
history), heavy quantized ties, mixed ``+-0.0``, non-power-of-two candidate counts,
pad (``index_topk > C``) and truncating (``index_topk < C``, distinct scores)
selections, and multi-row batches.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def _supported() -> bool:
    return torch.cuda.is_available() and M._HAS_TRITON


def _eager_topk_tail(index_score: torch.Tensor, index_topk: int, max_compressed_len: int):
    """The exact eager chain replaced by ``_fused_topk_select``."""
    topk_count = min(index_topk, max_compressed_len)
    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(torch.int64), torch.full_like(topk_rows, -1))
    if topk_count < index_topk:
        pad_shape = (index_score.shape[0], index_topk - topk_count)
        row_pad = torch.full(pad_shape, -1, dtype=torch.int64, device=index_score.device)
        valid_pad = torch.zeros(pad_shape, dtype=torch.bool, device=index_score.device)
        topk_rows = torch.cat((topk_rows, row_pad), dim=-1)
        topk_valid = torch.cat((topk_valid, valid_pad), dim=-1)
    return topk_rows, topk_valid


def _check(index_score: torch.Tensor, index_topk: int, case: str):
    c = int(index_score.shape[1])
    ref_rows, ref_valid = _eager_topk_tail(index_score, index_topk, c)
    got_rows, got_valid = M._fused_topk_select(index_score, index_topk, min(index_topk, c))
    assert got_rows.dtype == torch.int64 and got_valid.dtype == torch.bool, case
    assert got_rows.shape == ref_rows.shape and got_valid.shape == ref_valid.shape, case
    assert torch.equal(got_rows, ref_rows), f"{case}: rows mismatch"
    assert torch.equal(got_valid, ref_valid), f"{case}: valid mismatch"


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_matches_eager_tail():
    torch.manual_seed(0)
    dev = "cuda"

    # Proxy shape: distinct scores, k == C.
    s = torch.randn(1, 512, device=dev)
    _check(s, 512, "distinct k==C")

    # -inf visibility tails: partial, single visible row, all-invalid history.
    for visible in (0, 1, 3, 17, 130, 511):
        s = torch.randn(2, 512, device=dev)
        s[:, visible:] = float("-inf")
        _check(s, 512, f"visible={visible}")

    # Heavy identical-bit ties (quantized) + -inf tail, multi-row.
    for t in range(4):
        s = (torch.randn(4, 512, device=dev) * 2).round() / 2
        s[:, 100 + 90 * t :] = float("-inf")
        _check(s, 512, f"ties trial {t}")

    # Mixed +0.0 / -0.0 among distinct values (torch compares them equal).
    s = torch.randn(3, 512, device=dev) + 3.0
    s[:, ::7] = 0.0
    s[:, 3::11] = -0.0
    s[:, 470:] = float("-inf")
    _check(s, 512, "pm-zero mix")

    # Pad path: index_topk > C fills the tail with -1/False.
    s = torch.randn(3, 512, device=dev)
    s[:, 200:] = float("-inf")
    _check(s, 640, "pad k>C")

    # Truncating selection (index_topk < C) on distinct scores == sort prefix.
    s = torch.randn(4, 512, device=dev)
    s[2, 40:] = float("-inf")
    s[3, :] = float("-inf")
    _check(s, 128, "truncate k<C distinct")

    # Non-power-of-two candidate count (kernel pads the sort width).
    s = torch.randn(2, 384, device=dev)
    s[:, 300:] = float("-inf")
    _check(s, 512, "C=384 non-pow2 + pad")
    _check(torch.randn(2, 384, device=dev), 384, "C=384 k==C")

    # Negative scores (negative indexer weights make the whole row negative).
    s = -torch.rand(2, 512, device=dev) - 1.0
    s[:, 400:] = float("-inf")
    _check(s, 512, "all-negative scores")


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_inside_select_decode_gate():
    """The integration gate routes fp32 CUDA scores through the fused kernel."""
    torch.manual_seed(1)
    s = torch.randn(1, 512, device="cuda")
    s[:, 490:] = float("-inf")
    rows, valid = M._fused_topk_select(s, 512, 512)
    ref_rows, ref_valid = _eager_topk_tail(s, 512, 512)
    assert torch.equal(rows, ref_rows) and torch.equal(valid, ref_valid)
    # cudagraph-style replay stability: same inputs, repeated launches, same bytes.
    for _ in range(5):
        rows2, valid2 = M._fused_topk_select(s, 512, 512)
        assert torch.equal(rows2, rows) and torch.equal(valid2, valid)
