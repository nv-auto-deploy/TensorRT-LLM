# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AR-removal identity for the DeepSeek V4 indexer index-score reduction.

The indexer index score is reduced over ALL index heads::

    index_score = (matmul(q_index, index_k.T).float().relu()
                   * indexer_weights.float().unsqueeze(-1)).sum(dim=<head>)

Previously the index-score projection (``DeepseekV4Indexer.wq_b`` /
``weights_proj``) was column(head)-sharded across TP ranks, so each rank held only
its head shard and produced a *partial* score; a hand-coded ``all_reduce(SUM)``
(at ``select_topk`` and inside the sparse-attention custom op) summed the partials
into the global score. We removed those all_reduces and replicated the (small)
projection on every rank instead.

This test pins the mathematical justification: the global full-head score equals
the sum of per-head-shard partial scores (i.e. exactly what the removed
``all_reduce(SUM)`` produced), and -- the only thing that matters downstream --
the top-k row selection is identical. The full sum and the stack-of-partials sum
use different fp reduction trees, so they are ``allclose`` rather than bit-equal;
the replicated path actually matches the single-GPU reference order (one
contiguous ``sum`` over all heads), so it is at least as reference-faithful as the
sharded+all_reduce path it replaces.
"""

import pytest
import torch


def _index_score_full(q_index, index_k, weights):
    """Replicated path (post-change): every rank computes the full-head score."""
    score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
    return (score.relu() * weights.float().unsqueeze(-1)).sum(dim=0)


def _index_score_sharded_then_allreduce(q_index, index_k, weights, tp_size):
    """Pre-change path: head-shard the projection, sum partials (== all_reduce SUM)."""
    num_heads = q_index.shape[0]
    assert num_heads % tp_size == 0
    heads_per_rank = num_heads // tp_size
    partials = []
    for rank in range(tp_size):
        sl = slice(rank * heads_per_rank, (rank + 1) * heads_per_rank)
        q_r, w_r = q_index[sl], weights[sl]
        score_r = torch.matmul(q_r, index_k.transpose(-1, -2)).float()
        partials.append((score_r.relu() * w_r.float().unsqueeze(-1)).sum(dim=0))
    # torch.stack(...).sum(0) is the deterministic analogue of all_reduce(SUM).
    return torch.stack(partials, dim=0).sum(dim=0)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("index_topk", [32, 256])
def test_replicated_full_score_equals_sharded_allreduce(tp_size, index_topk):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234 + tp_size)
    # DeepSeek-V4 indexer shapes: 64 index heads, 128 head dim.
    num_heads, head_dim, visible_len = 64, 128, 512
    q_index = torch.randn(num_heads, head_dim, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(visible_len, head_dim, device=device, dtype=torch.bfloat16)
    # ``weights`` is f32 and pre-scaled in the model; here just a positive vector.
    weights = torch.rand(num_heads, device=device, dtype=torch.float32) + 0.1

    full = _index_score_full(q_index, index_k, weights)
    sharded = _index_score_sharded_then_allreduce(q_index, index_k, weights, tp_size)

    # The removed all_reduce summed exactly the full-head score (up to fp order).
    torch.testing.assert_close(full, sharded, rtol=1e-4, atol=1e-3)

    # The only downstream consumer is top-k row selection -- it must be identical.
    topk_count = min(index_topk, visible_len)
    full_rows = full.topk(topk_count, dim=-1).indices
    sharded_rows = sharded.topk(topk_count, dim=-1).indices
    assert torch.equal(full_rows, sharded_rows)


def test_decode_batched_head_dim1_identity():
    """Prefill/batched variant reduces over dim=1 (``_select_decode_ratio4_indexer_rows``)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(99)
    batch, num_heads, head_dim, visible_len = 3, 64, 128, 400
    q_index = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    index_k = torch.randn(batch, visible_len, head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.rand(batch, num_heads, device=device, dtype=torch.float32) + 0.1

    def full(qi, ik, w):
        s = torch.matmul(qi, ik.transpose(-1, -2)).float()
        return (s.relu() * w.float().unsqueeze(-1)).sum(dim=1)

    tp_size, hpr = 8, num_heads // 8
    parts = []
    for r in range(tp_size):
        sl = slice(r * hpr, (r + 1) * hpr)
        s = torch.matmul(q_index[:, sl], index_k.transpose(-1, -2)).float()
        parts.append((s.relu() * weights[:, sl].float().unsqueeze(-1)).sum(dim=1))
    sharded = torch.stack(parts, dim=0).sum(dim=0)

    torch.testing.assert_close(full(q_index, index_k, weights), sharded, rtol=1e-4, atol=1e-3)
