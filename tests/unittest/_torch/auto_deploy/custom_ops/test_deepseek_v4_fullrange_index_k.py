# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ratio-4 decode index-key production (idea_0063 remainder).

``_fused_fullrange_index_k`` collapses the decode lightning-indexer candidate front --
paged full-range reconstruction (``_dsv4_fullrange_candidate_rows_kernel``), the
cos/sin table gathers, the eager interleaved rope, the nope/pe concat and the
``deepseek_v4_hadamard_fp4`` rotate+fake-fp4 quant -- into a single kernel emitting the
final ``[B, M, head_dim]`` index keys.  This test pins:

* value equivalence of the emitted index keys against the current production chain
  (``_batched_overlap_compressed_rows_fullrange(rotate=True)`` + its rope/hadamard
  tail), allowing only the documented <=1-ULP fp32 rope FMA tolerance ahead of the
  bf16 rounding;
* exact top-k selection equivalence (ids, order, validity) of the full fused
  ``_select_decode_ratio4_indexer_rows`` path against the eager reference chain
  (eager score formula + ``torch.topk`` tail), across short/empty/full histories,
  page-short sequences, heavy score ties, shifted rope positions and non-finite
  cache garbage beyond the visible range.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def _rope_tables(num_pos: int, rope_dim: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    angles = torch.outer(torch.arange(num_pos, dtype=torch.float32), freqs)
    return angles.cos().to(device), angles.sin().to(device)


def _build_fixture(
    *,
    num_rows: int,
    head_dim: int,
    rope_dim: int,
    m: int,
    tokens_per_block: int,
    input_pos: list,
    pos_delta: int = 0,
    short_pages_row: int = -1,
    constant_caches: bool = False,
    poison_beyond_visible: bool = False,
    device: str = "cuda",
    seed: int = 0,
):
    torch.manual_seed(seed)
    ratio = 4
    state_dim = 2 * head_dim
    pos_count = m * ratio
    full_pages = (pos_count + tokens_per_block - 1) // tokens_per_block
    page_counts = [full_pages - 1 if i == short_pages_row else full_pages for i in range(num_rows)]
    cu_num_pages = torch.tensor(
        [0, *torch.tensor(page_counts).cumsum(0).tolist()], dtype=torch.long, device=device
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=device)

    if constant_caches:
        kv_cache = torch.full((total_pages, tokens_per_block, state_dim), 0.5, device=device)
        gate_cache = torch.full((total_pages, tokens_per_block, state_dim), -0.25, device=device)
    else:
        kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)
        gate_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)

    input_pos_t = torch.tensor(input_pos, dtype=torch.long, device=device)
    if poison_beyond_visible:
        # Positions beyond the written history hold garbage in real caches; the
        # visibility mask must keep it out of the selection either way.
        for b in range(num_rows):
            start_page = int(cu_num_pages[b].item())
            n_pages = int(cu_num_pages[b + 1].item()) - start_page
            written = int(input_pos_t[b].item()) + 1
            for p in range(n_pages):
                page = int(cache_loc[start_page + p].item())
                lo = p * tokens_per_block
                for t in range(tokens_per_block):
                    if lo + t >= written:
                        kv_cache[page, t] = float("inf")
                        gate_cache[page, t] = float("nan")

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=device)
    position_ids = input_pos_t + pos_delta
    ape = torch.randn(ratio, state_dim, device=device)
    norm_weight = torch.randn(head_dim, device=device)
    cos_table, sin_table = _rope_tables(pos_count + 8, rope_dim, device)

    full_positions = torch.arange(pos_count, dtype=torch.long, device=device)
    full_positions = full_positions.view(1, -1).expand(num_rows, -1)
    full_page_map = M._decode_page_ids_and_offsets(
        kv_cache, seq_idx, full_positions, cu_num_pages, cache_loc
    )
    return dict(
        kv_cache=kv_cache,
        gate_cache=gate_cache,
        seq_idx=seq_idx,
        input_pos=input_pos_t,
        position_ids=position_ids,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
        ape=ape,
        norm_weight=norm_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        full_page_map=full_page_map,
        m=m,
        ratio=ratio,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=1e-6,
    )


def _reference_index_k(fx, dtype):
    """Current production chain: fused pooled rows + eager rope + hadamard op."""
    num_rows = fx["seq_idx"].shape[0]
    candidate_rows = torch.arange(fx["m"], dtype=torch.long, device=fx["seq_idx"].device)
    candidate_rows = candidate_rows.view(1, -1).expand(num_rows, -1)
    row_position_id = fx["position_ids"].unsqueeze(1) - (
        fx["input_pos"].unsqueeze(1) - candidate_rows * fx["ratio"]
    )
    return M._batched_overlap_compressed_rows_fullrange(
        fx["kv_cache"],
        fx["gate_cache"],
        fx["seq_idx"],
        row_position_id,
        fx["cu_num_pages"],
        fx["cache_loc"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["eps"],
        fx["rope_dim"],
        fx["ratio"],
        fx["head_dim"],
        fx["m"],
        dtype,
        rotate=True,
        full_page_map=fx["full_page_map"],
    )


def _reference_select(fx, q_index, indexer_weights, index_topk, dtype):
    """Eager score formula + torch.topk tail (the pre-fusion reference chain)."""
    index_k = _reference_index_k(fx, dtype)
    num_rows = q_index.shape[0]
    candidate_rows = torch.arange(fx["m"], dtype=torch.long, device=q_index.device)
    candidate_rows = candidate_rows.view(1, -1).expand(num_rows, -1)
    visible_len = ((fx["input_pos"] + 1) // fx["ratio"]).clamp(max=fx["m"])
    index_score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
    index_score = (index_score.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=1)
    visible = candidate_rows < visible_len.unsqueeze(1)
    index_score = index_score.masked_fill(~visible, float("-inf"))
    topk_count = min(index_topk, fx["m"])
    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(torch.int64), torch.full_like(topk_rows, -1))
    if topk_count < index_topk:
        pad_shape = (num_rows, index_topk - topk_count)
        topk_rows = torch.cat(
            (topk_rows, torch.full(pad_shape, -1, dtype=torch.int64, device=q_index.device)),
            dim=-1,
        )
        topk_valid = torch.cat(
            (topk_valid, torch.zeros(pad_shape, dtype=torch.bool, device=q_index.device)),
            dim=-1,
        )
    return topk_rows, topk_valid, index_score


CASES = [
    # (name, kwargs, index_topk)
    (
        "prod_shape_mixed_history",
        dict(
            num_rows=3,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 3, 0],
            short_pages_row=0,
            seed=11,
        ),
        8,
    ),
    (
        "empty_history",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=6,
            tokens_per_block=8,
            input_pos=[0, 2],
            seed=13,
        ),
        6,
    ),
    (
        "small_pow2_head",
        dict(
            num_rows=2,
            head_dim=32,
            rope_dim=16,
            m=5,
            tokens_per_block=4,
            input_pos=[19, 7],
            seed=17,
        ),
        4,
    ),
    (
        "heavy_ties",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 31],
            constant_caches=True,
            seed=19,
        ),
        8,
    ),
    (
        "shifted_rope_positions",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[27, 15],
            pos_delta=3,
            seed=23,
        ),
        8,
    ),
    (
        "poisoned_tail",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[13, 5],
            poison_beyond_visible=True,
            seed=29,
        ),
        8,
    ),
    (
        "short_topk_pad",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 9],
            seed=31,
        ),
        3,
    ),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("name,kwargs,index_topk", CASES, ids=[c[0] for c in CASES])
def test_fused_index_k_and_selection_match_reference(name, kwargs, index_topk):
    assert M._HAS_TRITON, "test requires triton"
    dtype = torch.bfloat16
    fx = _build_fixture(**kwargs)
    torch.manual_seed(1000 + index_topk)
    num_rows, head_dim = fx["seq_idx"].shape[0], fx["head_dim"]
    num_heads = 16
    q_index = torch.randn(num_rows, num_heads, head_dim, device="cuda", dtype=dtype)
    indexer_weights = torch.randn(num_rows, num_heads, device="cuda", dtype=dtype)

    # --- index-key value equivalence against the production chain ---
    fused_k = M._fused_fullrange_index_k(
        fx["kv_cache"],
        fx["gate_cache"],
        fx["full_page_map"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["input_pos"],
        fx["position_ids"],
        fx["eps"],
        fx["ratio"],
        head_dim,
        fx["rope_dim"],
        fx["m"],
        dtype,
    )
    ref_k = _reference_index_k(fx, dtype)
    assert fused_k.shape == ref_k.shape and fused_k.dtype == ref_k.dtype
    # Compare only rows the visibility mask can ever expose (garbage tail rows are
    # masked to -inf in the score either way and may legitimately hold NaN).
    visible_len = ((fx["input_pos"] + 1) // fx["ratio"]).clamp(max=fx["m"])
    vis_mask = (
        torch.arange(fx["m"], device="cuda").view(1, -1) < visible_len.view(-1, 1)
    ).unsqueeze(-1)
    f_vis = torch.where(vis_mask, fused_k, torch.zeros_like(fused_k)).float()
    r_vis = torch.where(vis_mask, ref_k, torch.zeros_like(ref_k)).float()
    exact = (f_vis == r_vis).float().mean().item()
    assert exact >= 0.999, f"[{name}] visible index-k exact-match ratio {exact} too low"
    torch.testing.assert_close(f_vis, r_vis, rtol=1.6e-2, atol=1e-2)

    # --- exact selection equivalence (ids, order, validity) ---
    rows_fused, valid_fused = M._select_decode_ratio4_indexer_rows(
        q_index,
        indexer_weights,
        fx["kv_cache"],
        fx["gate_cache"],
        fx["seq_idx"],
        fx["input_pos"],
        fx["position_ids"],
        index_topk,
        fx["cu_num_pages"],
        fx["cache_loc"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["eps"],
        fx["rope_dim"],
        fx["m"],
        full_page_map=fx["full_page_map"],
    )
    rows_ref, valid_ref, score_ref = _reference_select(
        fx, q_index, indexer_weights, index_topk, dtype
    )
    assert torch.equal(rows_fused, rows_ref), f"[{name}] selected row ids/order differ"
    assert torch.equal(valid_fused, valid_ref), f"[{name}] selection validity differs"

    # --- masked score equivalence (finite region) ---
    score_fused = M._fused_index_score(
        q_index, fused_k, indexer_weights, fx["input_pos"], fx["m"], fx["ratio"]
    )
    finite = torch.isfinite(score_ref)
    assert torch.equal(finite, torch.isfinite(score_fused)), f"[{name}] visibility differs"
    torch.testing.assert_close(score_fused[finite], score_ref[finite], rtol=2e-2, atol=2e-2)
