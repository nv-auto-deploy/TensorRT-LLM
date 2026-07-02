# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ratio-4 full-range candidate-row reconstruction (idea_0075).

``_batched_overlap_compressed_rows_fullrange`` reconstructs every lightning-indexer
candidate compressed row at decode time.  idea_0075 collapses the gather / row-shift /
concat / where / pool / rmsnorm swarm of that helper into a single Triton kernel
(``_dsv4_fullrange_candidate_rows_kernel``) that emits the post-pool, post-rmsnorm rows
directly.  The kernel replicates the eager numerics -- fp32-internal softmax pool and
RMSNorm with bf16 rounding at the same points as the reference (the gather ``.to(dtype)``,
the ape add, the ``compress_pool`` output and the ``_rms_norm_ref`` output) -- so its
output must match the eager chain up to the ``rsqrt`` primitive, which bf16 rounding
absorbs.  This test pins that equivalence over the ratio-4 overlap layout, including
sequences whose page count is short of the full candidate range (previous-block
validity masking).
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def _identity_rope(pooled, cos, sin, rope_dim, rotate=False):
    """Strip the shared rope/quantize tail so we compare the pool/norm result only."""
    return pooled


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("head_dim", [8, 16, 128])
@pytest.mark.parametrize("num_rows", [1, 3])
def test_fused_fullrange_matches_eager(monkeypatch, head_dim, num_rows):
    torch.manual_seed(1234 + head_dim + num_rows)
    device = "cuda"
    dtype = torch.bfloat16
    ratio = 4
    m = 6  # max_compressed_len
    state_dim = 2 * head_dim
    tokens_per_block = 8
    eps = 1e-6
    rope_dim = min(4, head_dim)

    pos_count = m * ratio  # candidate token positions per sequence
    full_pages = (pos_count + tokens_per_block - 1) // tokens_per_block
    # Give sequence 0 one page fewer than the full range so its tail candidate
    # positions are page-invalid -> exercises the previous-block validity mask.
    page_counts = [
        full_pages - 1 if (i == 0 and num_rows > 1) else full_pages for i in range(num_rows)
    ]
    cu_num_pages = torch.tensor(
        [0, *torch.tensor(page_counts).cumsum(0).tolist()], dtype=torch.long, device=device
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=device)

    kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)
    gate_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=device)
    row_position_id = torch.zeros(num_rows, m, dtype=torch.long, device=device)
    ape = torch.randn(ratio, state_dim, device=device)
    norm_weight = torch.randn(head_dim, device=device)
    cos_table = torch.randn(pos_count + 1, rope_dim, device=device)
    sin_table = torch.randn(pos_count + 1, rope_dim, device=device)

    def run():
        return M._batched_overlap_compressed_rows_fullrange(
            kv_cache,
            gate_cache,
            seq_idx,
            row_position_id,
            cu_num_pages,
            cache_loc,
            ape,
            norm_weight,
            cos_table,
            sin_table,
            eps,
            rope_dim,
            ratio,
            head_dim,
            m,
            dtype,
            rotate=True,
            full_page_map=None,
        )

    # Compare the pool/norm result only (shared rope/quantize tail stripped).
    monkeypatch.setattr(M, "_apply_compressed_rope_and_quantize", _identity_rope)

    assert M._HAS_TRITON, "test requires triton"
    out_fused = run()
    monkeypatch.setattr(M, "_HAS_TRITON", False)
    out_eager = run()

    assert out_fused.shape == (num_rows, m, head_dim)
    assert out_fused.dtype == dtype

    exact = (out_fused == out_eager).float().mean().item()
    max_abs = (out_fused.float() - out_eager.float()).abs().max().item()
    assert exact >= 0.95, f"exact bf16 match ratio {exact} too low (max_abs={max_abs})"
    torch.testing.assert_close(out_fused, out_eager, rtol=1.6e-2, atol=1e-2)
