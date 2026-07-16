# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness guard for the fused paged index-score fold.

``_fused_fullrange_index_score`` folds the whole eager decode indexer front --
paged candidate-row reconstruction + rope + hadamard/fake-fp4 index keys
(``_batched_overlap_compressed_rows_fullrange(rotate=True)``) followed by the
``matmul().float().relu() * w`` weighted head reduction + visibility mask --
into one launch, so the ``[B, M, head_dim]`` candidate index-key tensor is
never materialized.  The reference here is that retained pure-eager chain (the
CPU/edge fallback of ``_select_decode_ratio4_indexer_rows``): the key is
rounded to the query dtype at the same points the eager chain materializes it,
so the emitted scores must be bit-identical (``torch.equal``) to the eager
chain across histories, tie-heavy caches, shifted rope positions and poisoned
invisible tails -- except the M=512+ production scales (compiler fp32
FMA/accumulation context) and the H != 16 head counts (fp32 head-reduction
order vs the eager ``.sum(dim=1)``), where a small score tail moves by a
one-bf16-ULP flip of a single weighted head term (the tolerance the original
test documented for kernel-vs-eager comparisons).  The top-k ids, tie order
and validity consuming these scores must be identical in every case.
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


def _eager_score(fx, q_index, indexer_weights, w_scale: float = 1.0):
    """The retained pure-eager chain the fold replaces.

    Replicates the eager fallback of ``_select_decode_ratio4_indexer_rows``:
    index keys via ``_batched_overlap_compressed_rows_fullrange(rotate=True)``
    (the production eager body), then the plain-torch
    ``matmul().float().relu() * (w.float() * w_scale)`` head reduction and the
    ``candidate < visible_len`` mask.
    """
    m, ratio = fx["m"], fx["ratio"]
    num_rows = int(fx["seq_idx"].shape[0])
    candidate_rows = torch.arange(m, dtype=torch.long, device=q_index.device)
    candidate_rows = candidate_rows.view(1, -1).expand(num_rows, -1)
    row_position_id = fx["position_ids"].unsqueeze(1) - (
        fx["input_pos"].unsqueeze(1) - candidate_rows * ratio
    )
    index_k = M._batched_overlap_compressed_rows_fullrange(
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
        ratio,
        fx["head_dim"],
        m,
        q_index.dtype,
        rotate=True,
        full_page_map=fx["full_page_map"],
    )
    visible_len = ((fx["input_pos"] + 1) // ratio).clamp(max=m)
    score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
    score = (score.relu() * (indexer_weights.float() * w_scale).unsqueeze(-1)).sum(dim=1)
    visible = candidate_rows < visible_len.unsqueeze(1)
    return score.masked_fill(~visible, float("-inf"))


def _fused_score(fx, q_index, indexer_weights, w_scale: float = 1.0):
    return M._fused_fullrange_index_score(
        fx["kv_cache"],
        fx["gate_cache"],
        fx["full_page_map"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["input_pos"],
        fx["position_ids"],
        q_index,
        indexer_weights,
        fx["eps"],
        fx["ratio"],
        fx["head_dim"],
        fx["rope_dim"],
        fx["m"],
        w_scale=w_scale,
    )


CASES = [
    # (name, fixture kwargs, num_heads, exact) -- ``exact=False`` marks cases
    # in the documented kernel-vs-eager tolerance class: the M=512+ production
    # scale (compiler fp32 FMA/accumulation context) and the H != 16 head
    # counts (the kernel's H_BLOCK-lane fp32 head reduction orders differently
    # than the eager ``.sum(dim=1)``), each moving a small score tail by a
    # one-bf16-ULP flip of a single weighted head term (observed <= 4e-6
    # absolute -- the tolerance the original test documented for kernel-vs-
    # eager comparisons); the top-k ids, tie order and validity must be
    # identical either way.
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
        16,
        True,
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
        16,
        True,
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
        16,
        True,
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
        16,
        True,
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
        16,
        True,
    ),
    (
        "nonpow2_heads_h24",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 9],
            seed=31,
        ),
        24,
        False,
    ),
    (
        "full_heads_h64",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 9],
            seed=37,
        ),
        64,
        False,
    ),
    (
        "prod_scale_m512",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=512,
            tokens_per_block=64,
            input_pos=[1001, 998],
            seed=41,
        ),
        16,
        False,
    ),
    (
        # The proxy/production decode grid (M=1024): one row deep in the
        # early-exit regime (vlen 250) and one fully visible (vlen == M, no
        # early exits), pinning the visible-prefix split.
        "prod_scale_m1024_visible_split",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=1024,
            tokens_per_block=64,
            input_pos=[1001, 4095],
            seed=59,
        ),
        16,
        False,
    ),
]


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused index-score fold requires triton + CUDA",
)
@pytest.mark.parametrize("name,kwargs,num_heads,exact", CASES, ids=[c[0] for c in CASES])
def test_fused_index_score_matches_two_kernel_chain(name, kwargs, num_heads, exact):
    dtype = torch.bfloat16
    fx = _build_fixture(**kwargs)
    torch.manual_seed(2000 + num_heads)
    num_rows, head_dim = fx["seq_idx"].shape[0], fx["head_dim"]
    q_index = torch.randn(num_rows, num_heads, head_dim, device="cuda", dtype=dtype)
    indexer_weights = torch.randn(num_rows, num_heads, device="cuda", dtype=dtype)

    score_ref = _eager_score(fx, q_index, indexer_weights)
    score_fused = _fused_score(fx, q_index, indexer_weights)
    assert score_fused.shape == score_ref.shape and score_fused.dtype == score_ref.dtype
    if exact:
        assert torch.equal(score_fused, score_ref), f"[{name}] fused index score diverged"
    else:
        # Documented kernel-vs-eager tolerance: visibility must match exactly;
        # most scores bit-equal, with a small tail moved by a one-bf16-ULP flip
        # of a single weighted head term (compiler fp32 FMA / head-reduction
        # order; observed <= 4e-6 abs).  The 0.9 exact-fraction floor is only
        # meaningful at the production candidate scale (hundreds of finite
        # entries); tiny visible prefixes pin the per-entry magnitude instead.
        # The selection equality below is the load-bearing invariant.
        finite = torch.isfinite(score_ref)
        assert torch.equal(torch.isfinite(score_fused), finite), f"[{name}] visibility diverged"
        f = score_fused[finite]
        r = score_ref[finite]
        if f.numel() >= 64:
            exact_frac = (f == r).float().mean().item()
            assert exact_frac >= 0.9, f"[{name}] exact-match fraction {exact_frac} too low"
        max_abs = (f - r).abs().max().item()
        assert max_abs <= 4e-6, f"[{name}] score tail moved by {max_abs} > 4e-6"
        torch.testing.assert_close(f, r, rtol=1e-5, atol=1e-4)

    # The selection consuming these scores must be identical: ids, tie order and
    # validity, across narrow and select-all top-k widths.
    for index_topk in {min(fx["m"], 8), fx["m"]}:
        rows_ref, valid_ref = M._fused_topk_select(score_ref, index_topk, min(index_topk, fx["m"]))
        rows_fused, valid_fused = M._fused_topk_select(
            score_fused, index_topk, min(index_topk, fx["m"])
        )
        assert torch.equal(rows_fused, rows_ref), f"[{name}] top-k ids/order diverged"
        assert torch.equal(valid_fused, valid_ref), f"[{name}] top-k validity diverged"


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused index-score fold requires triton + CUDA",
)
def test_fused_index_score_fp32_weights():
    """fp32 indexer weights skip the old wrapper's cast; values must still match.

    fp32 weights shift the fp32 head-reduction rounding pattern, so the match is
    the documented kernel-vs-eager tolerance (<= 4e-6 abs tail, selection
    identical) rather than ``torch.equal``.
    """
    fx = _build_fixture(
        num_rows=2,
        head_dim=128,
        rope_dim=64,
        m=8,
        tokens_per_block=8,
        input_pos=[31, 9],
        seed=43,
    )
    torch.manual_seed(4300)
    q_index = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    indexer_weights = torch.randn(2, 16, device="cuda", dtype=torch.float32)
    score_ref = _eager_score(fx, q_index, indexer_weights)
    score_fused = _fused_score(fx, q_index, indexer_weights)
    finite = torch.isfinite(score_ref)
    assert torch.equal(torch.isfinite(score_fused), finite)
    max_abs = (score_fused[finite] - score_ref[finite]).abs().max().item()
    assert max_abs <= 4e-6, f"score tail moved by {max_abs} > 4e-6"
    rows_ref, valid_ref = M._fused_topk_select(score_ref, fx["m"], fx["m"])
    rows_fused, valid_fused = M._fused_topk_select(score_fused, fx["m"], fx["m"])
    assert torch.equal(rows_fused, rows_ref)
    assert torch.equal(valid_fused, valid_ref)


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused index-score fold requires triton + CUDA",
)
@pytest.mark.parametrize("m,input_pos", [(8, [31, 9]), (512, [1001, 998])])
def test_raw_weight_scale_fold_matches_prescaled(m, input_pos):
    """``w_scale`` in-kernel fold == the eager ``weights.float() * scale`` pre-scale.

    The kernels widen the raw model-dtype weights to fp32 and multiply by the
    scalar before the head weighting -- the identical fp32 multiply on the
    identical widened value -- so the scores must be bit-equal
    (``torch.equal``) to pre-scaling on the host, in BOTH the fused fold and
    the two-kernel chain, at toy and production candidate scales.
    """
    fx = _build_fixture(
        num_rows=2,
        head_dim=128,
        rope_dim=64,
        m=m,
        tokens_per_block=8 if m == 8 else 64,
        input_pos=input_pos,
        seed=53,
    )
    torch.manual_seed(5300)
    num_heads = 16
    q_index = torch.randn(2, num_heads, 128, device="cuda", dtype=torch.bfloat16)
    raw_weights = torch.randn(2, num_heads, device="cuda", dtype=torch.bfloat16)
    # The production scale expression (DeepseekV4Indexer: softmax_scale * H**-0.5).
    w_scale = fx["head_dim"] ** -0.5 * num_heads**-0.5
    prescaled = raw_weights.float() * w_scale

    fused_folded = _fused_score(fx, q_index, raw_weights, w_scale=w_scale)
    fused_prescaled = _fused_score(fx, q_index, prescaled)
    assert torch.equal(fused_folded, fused_prescaled), "fused w_scale fold diverged"

    eager_folded = _eager_score(fx, q_index, raw_weights, w_scale=w_scale)
    eager_prescaled = _eager_score(fx, q_index, prescaled)
    assert torch.equal(eager_folded, eager_prescaled), "eager w_scale fold diverged"


@pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()),
    reason="fused index-score fold requires triton + CUDA",
)
def test_selection_path_routes_through_fold_and_matches():
    """End-to-end ``_select_decode_ratio4_indexer_rows`` (which now routes H > 8
    through the fold) must keep the exact selection of the two-kernel chain.

    The selection helper consumes RAW model-dtype weights and folds the
    indexer pre-scale (``head_dim**-0.5 * num_heads**-0.5``) into
    the kernel, so the reference chain applies the identical ``w_scale``.
    """
    fx = _build_fixture(
        num_rows=3,
        head_dim=128,
        rope_dim=64,
        m=8,
        tokens_per_block=8,
        input_pos=[31, 3, 0],
        short_pages_row=0,
        seed=47,
    )
    torch.manual_seed(4700)
    q_index = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    indexer_weights = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    index_topk = 8

    rows, valid = M._select_decode_ratio4_indexer_rows(
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
    w_scale = fx["head_dim"] ** -0.5 * int(q_index.shape[1]) ** -0.5
    score_ref = _eager_score(fx, q_index, indexer_weights, w_scale=w_scale)
    rows_ref, valid_ref = M._fused_topk_select(score_ref, index_topk, min(index_topk, fx["m"]))
    assert torch.equal(rows, rows_ref)
    assert torch.equal(valid, valid_ref)
