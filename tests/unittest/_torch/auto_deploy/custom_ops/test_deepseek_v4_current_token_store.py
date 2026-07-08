# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exactness check for the DeepSeek V4 fused current-token multi-cache store.

``_fused_current_token_store`` (idea_0006) folds the per-cache current-token
``index_put`` scatters -- SWA kv, main-compressor kv/gate, and the ratio-4
indexer-compressor kv/gate, which all write logical position ``input_pos`` and
therefore share the hoisted ``(page_ids, page_offsets)`` address -- into one
heterogeneous paged-store launch.  The caches differ in dtype (SWA is the
activation dtype, the compressor caches are fp32) and in row width, so this test
guards that the fused store is byte-identical to the per-cache reference it
replaces (``cache[page_ids, page_offsets] = value.to(cache.dtype)``), including
that it touches only the addressed slots and leaves every other byte untouched.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _HAS_TRITON,
    _fused_current_token_store,
)


def _reference_store(caches, values, page_ids, page_offsets):
    """Per-cache index_put the fused store replaces (numel==0 values skipped)."""
    for cache, value in zip(caches, values):
        if value.numel() == 0:
            continue
        cache[page_ids, page_offsets] = value.to(cache.dtype)


def _make_page_addr(num_pages, tokens_per_block, num_rows, device):
    # Distinct (page_id, page_offset) per row -- mirrors the real decode where each
    # sequence writes its own logical row (no index_put aliasing).
    base = torch.randperm(num_pages * tokens_per_block, device=device)[:num_rows]
    page_ids = (base // tokens_per_block).to(torch.long)
    page_offsets = (base % tokens_per_block).to(torch.long)
    return page_ids, page_offsets


# (cache dtype, state_dim) per cache -- SWA (activation dtype) + fp32 compressors.
_CONFIGS = {
    "ratio4": [
        (torch.bfloat16, 128),
        (torch.float32, 256),
        (torch.float32, 256),
        (torch.float32, 192),
        (torch.float32, 192),
    ],
    "ratio128": [
        (torch.bfloat16, 128),
        (torch.float32, 256),
        (torch.float32, 256),
    ],
    "pair": [
        (torch.bfloat16, 64),
        (torch.float32, 576),
    ],
    # Compression-off layers write only the SWA kv cache; since idea_0092 this
    # single-cache arity also runs the fused kernel (N_CACHES=1) instead of index_put.
    "off": [
        (torch.bfloat16, 128),
    ],
    # A 4-cache arity has no production layer mode but completes the 1..5 slot
    # coverage of the fixed 5-pointer kernel signature.
    "quad": [
        (torch.bfloat16, 128),
        (torch.float32, 256),
        (torch.float32, 256),
        (torch.float32, 192),
    ],
    # Row width above the 1024 BLOCK_S cap exercises the multi-block (cdiv) grid.
    "bigdim": [
        (torch.bfloat16, 1152),
        (torch.float32, 1536),
    ],
}


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="fused current-token store requires triton + CUDA",
)
@pytest.mark.parametrize("config", list(_CONFIGS))
@pytest.mark.parametrize("num_rows", [1, 3, 8])
@pytest.mark.parametrize("value_dtype", [torch.bfloat16, torch.float32])
def test_fused_current_token_store_matches_per_cache(config, num_rows, value_dtype):
    torch.manual_seed(20260703 + num_rows + len(config))
    device = "cuda"
    num_pages, tokens_per_block = 19, 8
    specs = _CONFIGS[config]

    page_ids, page_offsets = _make_page_addr(num_pages, tokens_per_block, num_rows, device)

    caches_ref = [
        torch.randn(num_pages, tokens_per_block, s, device=device).to(dt) for dt, s in specs
    ]
    caches_fused = [c.clone() for c in caches_ref]
    # Values arrive in ``value_dtype`` and are cast to each cache dtype by the store
    # (bf16 kv -> fp32 compressor cast is the common decode case).
    values = [torch.randn(num_rows, s, device=device, dtype=value_dtype) for _, s in specs]

    # Dummy page-table args (unused on the fused path; only the eager fallback reads
    # them, and it uses the provided page_ids/page_offsets directly).
    seq_idx = torch.arange(num_rows, device=device)
    input_pos = torch.full((num_rows,), 1000, dtype=torch.long, device=device)
    cu_num_pages = torch.tensor([0, num_pages], dtype=torch.long, device=device)
    cache_loc = torch.arange(num_pages, dtype=torch.long, device=device)

    _reference_store(caches_ref, values, page_ids, page_offsets)
    _fused_current_token_store(
        caches_fused,
        values,
        seq_idx,
        input_pos,
        cu_num_pages,
        cache_loc,
        page_ids,
        page_offsets,
    )

    for i, (ref, fused) in enumerate(zip(caches_ref, caches_fused)):
        assert torch.equal(fused, ref), (
            f"fused store diverged: config={config} cache={i} "
            f"num_rows={num_rows} value_dtype={value_dtype}"
        )


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="fused current-token store requires triton + CUDA",
)
def test_fused_current_token_store_touches_only_addressed_slots():
    """Every byte outside the written (page_id, page_offset) slots stays identical."""
    torch.manual_seed(11)
    device = "cuda"
    num_pages, tokens_per_block, num_rows = 12, 8, 3
    specs = _CONFIGS["ratio4"]

    caches = [torch.randn(num_pages, tokens_per_block, s, device=device).to(dt) for dt, s in specs]
    before = [c.clone() for c in caches]
    page_ids, page_offsets = _make_page_addr(num_pages, tokens_per_block, num_rows, device)
    values = [torch.randn(num_rows, s, device=device, dtype=torch.bfloat16) for _, s in specs]

    seq_idx = torch.arange(num_rows, device=device)
    input_pos = torch.full((num_rows,), 1000, dtype=torch.long, device=device)
    cu_num_pages = torch.tensor([0, num_pages], dtype=torch.long, device=device)
    cache_loc = torch.arange(num_pages, dtype=torch.long, device=device)

    _fused_current_token_store(
        caches, values, seq_idx, input_pos, cu_num_pages, cache_loc, page_ids, page_offsets
    )

    for cache, orig, value in zip(caches, before, values):
        # Written slots equal the cast value; all other slots are byte-identical.
        expected = orig.clone()
        expected[page_ids, page_offsets] = value.to(cache.dtype)
        assert torch.equal(cache, expected)


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="fused current-token store requires triton + CUDA",
)
def test_fused_current_token_store_single_cache_and_empty():
    """A single cache runs the fused kernel (N_CACHES=1); empty rows are a no-op."""
    device = "cuda"
    num_pages, tokens_per_block, num_rows, state_dim = 9, 8, 4, 128
    page_ids, page_offsets = _make_page_addr(num_pages, tokens_per_block, num_rows, device)
    seq_idx = torch.arange(num_rows, device=device)
    input_pos = torch.full((num_rows,), 1000, dtype=torch.long, device=device)
    cu_num_pages = torch.tensor([0, num_pages], dtype=torch.long, device=device)
    cache_loc = torch.arange(num_pages, dtype=torch.long, device=device)

    # Single cache -> fused kernel with N_CACHES=1 (the compression-off layers' SWA
    # write since idea_0092), still byte-identical to the index_put reference.
    torch.manual_seed(5)
    cache = torch.randn(num_pages, tokens_per_block, state_dim, device=device).to(torch.bfloat16)
    ref = cache.clone()
    value = torch.randn(num_rows, state_dim, device=device, dtype=torch.bfloat16)
    ref[page_ids, page_offsets] = value.to(ref.dtype)
    _fused_current_token_store(
        [cache], [value], seq_idx, input_pos, cu_num_pages, cache_loc, page_ids, page_offsets
    )
    assert torch.equal(cache, ref)

    # An all-empty value list is a complete no-op (num_rows == 0 addresses).
    empty_page_ids = page_ids[:0]
    empty_page_offsets = page_offsets[:0]
    cache2 = torch.randn(num_pages, tokens_per_block, state_dim, device=device).to(torch.bfloat16)
    before2 = cache2.clone()
    empty_values = [torch.empty(0, state_dim, device=device, dtype=torch.bfloat16)]
    _fused_current_token_store(
        [cache2],
        empty_values,
        seq_idx[:0],
        input_pos[:0],
        cu_num_pages,
        cache_loc,
        empty_page_ids,
        empty_page_offsets,
    )
    assert torch.equal(cache2, before2)


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="fused current-token store requires triton + CUDA",
)
def test_fused_current_token_store_extreme_bf16_widening():
    """The in-kernel bf16 -> fp32 store conversion is byte-exact on extreme values.

    Since idea_0092 the wrapper no longer pre-casts values with torch ``.to``;
    the Triton store performs the widening. bf16 -> fp32 is a pure mantissa
    zero-extension, so +/-inf, NaN (payload preserved), +/-0, the bf16
    max/min normals and subnormals must all round-trip byte-identically to the
    ``value.to(torch.float32)`` reference. Rows write the page-boundary offsets
    (0 and tokens_per_block - 1) to cover page rollover addressing.
    """
    device = "cuda"
    num_pages, tokens_per_block = 4, 8
    extremes = torch.tensor(
        [
            float("inf"),
            float("-inf"),
            float("nan"),
            0.0,
            -0.0,
            3.3895313892515355e38,  # bf16 max normal
            -3.3895313892515355e38,
            1.1754943508222875e-38,  # bf16 min normal
            9.183549615799121e-41,  # bf16 subnormal
            -9.183549615799121e-41,
            1.0,
            -2.0,
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    state_dim = extremes.numel()
    # Two rows at the in-page boundaries: offset 0 of page 1 and the last slot of
    # page 2 (rollover neighbors).
    page_ids = torch.tensor([1, 2], dtype=torch.long, device=device)
    page_offsets = torch.tensor([0, tokens_per_block - 1], dtype=torch.long, device=device)
    num_rows = 2

    caches_ref = [
        torch.zeros(num_pages, tokens_per_block, state_dim, device=device, dtype=torch.bfloat16),
        torch.zeros(num_pages, tokens_per_block, state_dim, device=device, dtype=torch.float32),
    ]
    caches_fused = [c.clone() for c in caches_ref]
    values = [extremes.expand(num_rows, state_dim).contiguous() for _ in caches_ref]

    seq_idx = torch.arange(num_rows, device=device)
    input_pos = torch.full((num_rows,), 7, dtype=torch.long, device=device)
    cu_num_pages = torch.tensor([0, num_pages], dtype=torch.long, device=device)
    cache_loc = torch.arange(num_pages, dtype=torch.long, device=device)

    _reference_store(caches_ref, values, page_ids, page_offsets)
    _fused_current_token_store(
        caches_fused, values, seq_idx, input_pos, cu_num_pages, cache_loc, page_ids, page_offsets
    )
    for i, (ref, fused) in enumerate(zip(caches_ref, caches_fused)):
        # Bitwise comparison (torch.equal treats NaN != NaN).
        assert torch.equal(
            fused.view(torch.int16 if fused.dtype == torch.bfloat16 else torch.int32),
            ref.view(torch.int16 if ref.dtype == torch.bfloat16 else torch.int32),
        ), f"extreme-value widening diverged on cache {i}"


@pytest.mark.skipif(
    not (_HAS_TRITON and torch.cuda.is_available()),
    reason="fused current-token store requires triton + CUDA",
)
def test_fused_current_token_store_cuda_graph_replay():
    """The fused store is CUDA-graph capturable and replays byte-identically.

    The decode path runs under piecewise CUDA graphs, so the launch (including the
    native-dtype source pointers of idea_0092) must capture without host fallbacks
    and produce the reference bytes on every replay with refreshed inputs.
    """
    torch.manual_seed(92)
    device = "cuda"
    num_pages, tokens_per_block, num_rows = 6, 8, 2
    specs = _CONFIGS["ratio4"]

    page_ids, page_offsets = _make_page_addr(num_pages, tokens_per_block, num_rows, device)
    caches = [torch.zeros(num_pages, tokens_per_block, s, device=device).to(dt) for dt, s in specs]
    values = [torch.randn(num_rows, s, device=device, dtype=torch.bfloat16) for _, s in specs]
    seq_idx = torch.arange(num_rows, device=device)
    input_pos = torch.full((num_rows,), 3, dtype=torch.long, device=device)
    cu_num_pages = torch.tensor([0, num_pages], dtype=torch.long, device=device)
    cache_loc = torch.arange(num_pages, dtype=torch.long, device=device)

    def _run():
        _fused_current_token_store(
            caches, values, seq_idx, input_pos, cu_num_pages, cache_loc, page_ids, page_offsets
        )

    # Warm up on a side stream (Triton JIT + allocator), then capture one launch.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run()

    for it in range(3):
        for v in values:
            v.copy_(torch.randn_like(v))
        for c in caches:
            c.zero_()
        graph.replay()
        torch.cuda.synchronize()
        refs = [torch.zeros_like(c) for c in caches]
        _reference_store(refs, values, page_ids, page_offsets)
        for i, (ref, cache) in enumerate(zip(refs, caches)):
            assert torch.equal(cache, ref), f"replay {it} diverged on cache {i}"
