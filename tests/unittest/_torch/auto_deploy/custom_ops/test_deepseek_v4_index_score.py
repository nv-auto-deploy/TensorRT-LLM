# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ratio-4 lightning-indexer score (idea_0004).

The decode ratio-4 selection in ``_select_decode_ratio4_indexer_rows`` computes a
per-candidate score with

    index_score = (matmul(q_index, index_k^T).float().relu()
                   * indexer_weights.float().unsqueeze(-1)).sum(dim=1)
    index_score = index_score.masked_fill(~visible, -inf)

then feeds ``index_score`` into ``topk`` to pick the compressed rows. idea_0004 fuses
the matmul + relu + weighted head reduction + visibility mask into one Triton kernel
(``_dsv4_index_score_kernel`` via ``_fused_index_score``) so the ``[N, H, C]``
head-by-candidate score and the separate masked ``[N, C]`` tensor are never
materialized.

Byte-exact score equality is not achievable (the bf16 matmul accumulation order and the
fp32 head reduction differ from cublas/torch by ~1 ULP), but the *selection* is what
the model consumes: the downstream sparse attention indexes the selected rows and is
permutation-sensitive only through fp reduction order, so the top-k rows AND their order
must match the eager chain. This test pins exactly that -- the ``topk_rows`` /
``topk_valid`` tail is reproduced from both the fused and eager scores and required to be
identical -- across partial-visibility, all-visible, tiny-visible, and tied-score cases.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # bf16 tl.dot contraction needs Ampere+ tensor cores.
    return torch.cuda.get_device_capability()[0] >= 8


def _eager_index_score(q_index, index_k, indexer_weights, visible_len, max_compressed_len):
    """The exact eager chain replaced by ``_fused_index_score``."""
    index_score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
    index_score = (index_score.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=1)
    cand = torch.arange(max_compressed_len, device=q_index.device).view(1, -1)
    return index_score.masked_fill(cand >= visible_len.view(-1, 1), float("-inf"))


def _topk_tail(index_score, index_topk, max_compressed_len):
    """The unchanged top-k tail of ``_select_decode_ratio4_indexer_rows``."""
    topk_count = min(index_topk, max_compressed_len)
    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(torch.int64), torch.full_like(topk_rows, -1))
    return topk_rows, topk_valid


@pytest.mark.skipif(not _supported(), reason="fused index-score kernel requires Ampere+ CUDA")
@pytest.mark.parametrize("num_rows", [1, 2])
@pytest.mark.parametrize("index_head_dim", [64, 128])
@pytest.mark.parametrize("vis_frac", [1.0, 0.5, 0.51, 0.02])
def test_fused_index_score_selection_matches_eager(num_rows, index_head_dim, vis_frac):
    torch.manual_seed(20260703 + num_rows + index_head_dim + int(vis_frac * 100))
    device = "cuda"
    dtype = torch.bfloat16
    h = 64  # index_n_heads
    c = 512  # max_compressed_len
    index_topk = 512  # decode index_topk == max_compressed_len -> full sort (proxy regime)

    q_index = (torch.randn(num_rows, h, index_head_dim, device=device, dtype=dtype)) * 0.1
    index_k = (torch.randn(num_rows, c, index_head_dim, device=device, dtype=dtype)) * 0.1
    weights = torch.randn(num_rows, h, device=device, dtype=torch.float32) * (index_head_dim**-0.5)
    vlen = max(1, int(round(c * vis_frac)))
    visible_len = torch.full((num_rows,), vlen, device=device, dtype=torch.int64)

    assert M._HAS_TRITON, "test requires triton"
    fused = M._fused_index_score(q_index, index_k, weights, visible_len, c)
    eager = _eager_index_score(q_index, index_k, weights, visible_len, c)

    assert fused.shape == (num_rows, c)
    assert fused.dtype == torch.float32

    # (1) Visibility mask must be bit-exact: exactly the visible candidates are finite.
    assert torch.equal(torch.isfinite(fused), torch.isfinite(eager))
    # (2) Finite scores agree to within a small fp tolerance (~1 ULP of the reduction).
    fin = torch.isfinite(eager)
    torch.testing.assert_close(fused[fin], eager[fin], rtol=2e-4, atol=2e-4)
    # (3) The selection the model actually consumes -- topk rows + validity -- is identical.
    rows_f, valid_f = _topk_tail(fused, index_topk, c)
    rows_e, valid_e = _topk_tail(eager, index_topk, c)
    assert torch.equal(valid_f, valid_e), "top-k validity diverged"
    assert torch.equal(rows_f, rows_e), "top-k selected rows / order diverged"


@pytest.mark.skipif(not _supported(), reason="fused index-score kernel requires Ampere+ CUDA")
@pytest.mark.parametrize("index_topk", [512, 128])
def test_fused_index_score_partial_topk_matches_eager(index_topk):
    """index_topk < max_compressed_len: the genuine top-k regime (long-context)."""
    torch.manual_seed(777 + index_topk)
    device = "cuda"
    dtype = torch.bfloat16
    num_rows, h, d, c = 2, 64, 128, 512
    q_index = torch.randn(num_rows, h, d, device=device, dtype=dtype) * 0.1
    index_k = torch.randn(num_rows, c, d, device=device, dtype=dtype) * 0.1
    weights = torch.randn(num_rows, h, device=device, dtype=torch.float32) * (d**-0.5)
    visible_len = torch.tensor([c, 300], device=device, dtype=torch.int64)

    fused = M._fused_index_score(q_index, index_k, weights, visible_len, c)
    eager = _eager_index_score(q_index, index_k, weights, visible_len, c)
    rows_f, valid_f = _topk_tail(fused, index_topk, c)
    rows_e, valid_e = _topk_tail(eager, index_topk, c)
    assert torch.equal(valid_f, valid_e)
    assert torch.equal(rows_f, rows_e)


@pytest.mark.skipif(not _supported(), reason="fused index-score kernel requires Ampere+ CUDA")
def test_fused_index_score_ties_and_validity():
    """Duplicate candidate rows (exact score ties) + partial visibility.

    With identical keys the tied rows carry identical scores in both paths, so the
    validity mask and the selected multiset must match. Row order among an exact tie is
    torch.topk's (index-based) tie-break, identical for both since the tie values match.
    """
    torch.manual_seed(9090)
    device = "cuda"
    dtype = torch.bfloat16
    num_rows, h, d, c = 1, 64, 128, 512
    index_topk = 512
    q_index = torch.randn(num_rows, h, d, device=device, dtype=dtype) * 0.1
    index_k = torch.randn(num_rows, c, d, device=device, dtype=dtype) * 0.1
    # Force exact ties: make rows 10..19 identical to row 0.
    index_k[:, 10:20, :] = index_k[:, 0:1, :]
    weights = torch.randn(num_rows, h, device=device, dtype=torch.float32) * (d**-0.5)
    visible_len = torch.tensor([200], device=device, dtype=torch.int64)

    fused = M._fused_index_score(q_index, index_k, weights, visible_len, c)
    eager = _eager_index_score(q_index, index_k, weights, visible_len, c)

    assert torch.equal(torch.isfinite(fused), torch.isfinite(eager))
    rows_f, valid_f = _topk_tail(fused, index_topk, c)
    rows_e, valid_e = _topk_tail(eager, index_topk, c)
    assert torch.equal(valid_f, valid_e)
    # Selected valid multiset identical (order among exact ties is deterministic per path).
    sel_f = torch.sort(rows_f[valid_f]).values
    sel_e = torch.sort(rows_e[valid_e]).values
    assert torch.equal(sel_f, sel_e)


@pytest.mark.skipif(not _supported(), reason="fused index-score kernel requires Ampere+ CUDA")
def test_fused_index_score_empty_rows():
    """Zero decode rows -> empty [0, C] score (graph-capture / drain edge case)."""
    device = "cuda"
    dtype = torch.bfloat16
    q_index = torch.empty(0, 64, 128, device=device, dtype=dtype)
    index_k = torch.empty(0, 512, 128, device=device, dtype=dtype)
    weights = torch.empty(0, 64, device=device, dtype=torch.float32)
    visible_len = torch.empty(0, device=device, dtype=torch.int64)
    out = M._fused_index_score(q_index, index_k, weights, visible_len, 512)
    assert out.shape == (0, 512)
