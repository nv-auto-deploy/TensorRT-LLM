# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Benchmark sweep for triton_paged_attention kernels.

Target model: google/gemma-4-26B-A4B-it
  head_dim=176, n_heads=16, n_kv_heads=8, HEAD_RATIO=2
  24 sliding-window layers (window=1024) + 6 full-attention layers

Usage:
  # Baseline benchmark (all shapes)
  python sweep_triton_paged_attention.py

  # Filter to specific shapes
  python sweep_triton_paged_attention.py --shapes D1,D2,D3

  # Override stage1 kernel params (for manual sweep)
  python sweep_triton_paged_attention.py --num-warps 8 --num-stages 4

  # Full parameter sweep (outputs JSON)
  python sweep_triton_paged_attention.py --sweep --sweep-out sweep_results.json

  # Benchmark only decode or context
  python sweep_triton_paged_attention.py --mode decode
  python sweep_triton_paged_attention.py --mode context
  python sweep_triton_paged_attention.py --mode gather
"""

import argparse
import json
import math

# ---------------------------------------------------------------------------
# Make TRT-LLM importable when run from repo root
# ---------------------------------------------------------------------------
import os
import sys
from typing import List, Optional, Tuple

import torch
import triton

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (  # noqa: E402
    _fast_gather_sdpa_kernel,
    _flash_decode_stage1_kernel,
    _flash_decode_stage2_kernel,
    _get_num_splits,
    _paged_context_kernel,
    triton_paged_context,
    triton_paged_decode,
    update_paged_kv_cache,
)

DTYPE = torch.bfloat16
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Shape Definitions
# ---------------------------------------------------------------------------

# Decode shapes — (id, batch, n_heads, n_kv_heads, head_dim, page_size, seq_len, sliding_window)
DECODE_SHAPES = [
    ("D1", 1, 16, 8, 176, 16, 128, 0),
    ("D2", 1, 16, 8, 176, 16, 512, 0),
    ("D3", 1, 16, 8, 176, 16, 1024, 1024),
    ("D4", 1, 16, 8, 176, 16, 2048, 0),
    ("D5", 8, 16, 8, 176, 16, 512, 0),
    ("D6", 8, 16, 8, 176, 16, 1024, 1024),
    ("D7", 32, 16, 8, 176, 16, 512, 0),
    ("D8", 64, 16, 8, 176, 16, 512, 1024),
]

# Context/prefill shapes (Triton paged path, q_len < 512)
# columns: (id, batch, n_heads, n_kv_heads, head_dim, page_size, q_len, kv_len)
CONTEXT_SHAPES = [
    ("P1", 1, 16, 8, 176, 16, 64, 64),
    ("P2", 1, 16, 8, 176, 16, 256, 256),
    ("P3", 4, 16, 8, 176, 16, 128, 128),
]

# Gather+SDPA shapes (q_len >= 512) — (id, batch, n_heads, n_kv_heads, head_dim, page_size, q_len)
GATHER_SHAPES = [
    ("G1", 1, 16, 8, 176, 16, 512),
    ("G2", 1, 16, 8, 176, 16, 2048),
    ("G3", 4, 16, 8, 176, 16, 512),
]

# ---------------------------------------------------------------------------
# Data Setup Helpers
# ---------------------------------------------------------------------------


def make_decode_inputs(
    batch: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    page_size: int,
    seq_len: int,
    sliding_window: int = 0,
    extra_blocks: int = 0,
):
    """Create all tensors needed for a decode forward pass."""
    pages_per_seq = math.ceil(seq_len / page_size)
    total_pages = batch * pages_per_seq
    num_blocks = total_pages + extra_blocks
    last_page_tokens = seq_len - (pages_per_seq - 1) * page_size

    q = torch.randn(batch, n_heads, head_dim, dtype=DTYPE, device=DEVICE)
    kv_cache = torch.randn(
        num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=DTYPE, device=DEVICE
    )

    # Sequential physical pages per batch item
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * pages_per_seq
    kv_last_page_len = torch.full((batch,), last_page_tokens, dtype=torch.int32, device=DEVICE)

    sm_scale = 1.0 / math.sqrt(head_dim)
    sw = sliding_window if sliding_window > 0 else None

    return q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale, sw


def make_context_inputs(
    batch: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_len: int,
    kv_len: int,
    sliding_window: int = 0,
):
    """Create all tensors needed for a context/prefill forward pass."""
    total_tokens = batch * q_len
    pages_per_seq = math.ceil(kv_len / page_size)
    total_pages = batch * pages_per_seq
    num_blocks = total_pages
    last_page_tokens = kv_len - (pages_per_seq - 1) * page_size

    q = torch.randn(total_tokens, n_heads, head_dim, dtype=DTYPE, device=DEVICE)
    kv_cache = torch.randn(
        num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=DTYPE, device=DEVICE
    )

    # qo_indptr: cumulative Q token counts
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * q_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=DEVICE) * pages_per_seq
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    kv_last_page_len = torch.full((batch,), last_page_tokens, dtype=torch.int32, device=DEVICE)
    seq_len_with_cache = torch.full((batch,), kv_len, dtype=torch.int32, device=DEVICE)

    sm_scale = 1.0 / math.sqrt(head_dim)
    sw = sliding_window if sliding_window > 0 else None

    return (
        q,
        kv_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_len_with_cache,
        sm_scale,
        sw,
    )


def make_kv_update_inputs(
    num_tokens: int,
    n_kv_heads: int,
    head_dim: int,
    page_size: int,
    num_blocks: int,
):
    """Inputs for _update_paged_kv_cache_kernel."""
    k = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    v = torch.randn(num_tokens, n_kv_heads, head_dim, dtype=DTYPE, device=DEVICE)
    batch_indices = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    positions = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    kv_cache = torch.randn(
        num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=DTYPE, device=DEVICE
    )
    kv_indices = torch.arange(math.ceil(num_tokens / page_size), dtype=torch.int32, device=DEVICE)
    kv_indptr = torch.tensor(
        [0, math.ceil(num_tokens / page_size)], dtype=torch.int32, device=DEVICE
    )
    return k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr


# ---------------------------------------------------------------------------
# Individual Kernel Benchmarks
# ---------------------------------------------------------------------------


def bench_decode_stage1(
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    page_size,
    seq_len,
    sw_val,
    num_warps_override=None,
    num_stages_override=None,
) -> Tuple[float, float, int]:
    """Benchmark _flash_decode_stage1_kernel. Returns (stage1_us, num_splits)."""
    q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale, sw = make_decode_inputs(
        batch, n_heads, n_kv_heads, head_dim, page_size, seq_len, sw_val or 0
    )
    head_ratio = n_heads // n_kv_heads
    head_ratio_padded = max(1, 2 ** math.ceil(math.log2(head_ratio))) if head_ratio > 1 else 1
    effective_seq = min(seq_len, sw_val) if sw_val and sw_val > 0 else seq_len
    num_splits = _get_num_splits(effective_seq, batch, n_kv_heads, page_size)
    sw_kernel = sw_val if sw_val and sw_val > 0 else 0

    partial_o = torch.empty(
        batch, n_heads, num_splits, head_dim, dtype=torch.float32, device=DEVICE
    )
    partial_lse = torch.empty(batch, n_heads, num_splits, dtype=torch.float32, device=DEVICE)

    extra_kwargs = {}
    if num_warps_override is not None:
        extra_kwargs["num_warps"] = num_warps_override
    if num_stages_override is not None:
        extra_kwargs["num_stages"] = num_stages_override

    def fn():
        _flash_decode_stage1_kernel[(batch, n_kv_heads, num_splits)](
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            partial_o,
            partial_lse,
            q.stride(0),
            q.stride(1),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            SM_SCALE=sm_scale,
            N_HEADS=n_heads,
            N_KV_HEADS=n_kv_heads,
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
            PAGE_SIZE=page_size,
            HEAD_RATIO=head_ratio,
            HEAD_RATIO_PADDED=head_ratio_padded,
            NUM_SPLITS=num_splits,
            SLIDING_WINDOW=sw_kernel,
            **extra_kwargs,
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0, num_splits


def bench_decode_stage2(
    batch,
    n_heads,
    head_dim,
    num_splits,
) -> float:
    """Benchmark _flash_decode_stage2_kernel. Returns latency_us."""
    partial_o = torch.randn(
        batch, n_heads, num_splits, head_dim, dtype=torch.float32, device=DEVICE
    )
    partial_lse = torch.randn(batch, n_heads, num_splits, dtype=torch.float32, device=DEVICE)
    output = torch.empty(batch, n_heads, head_dim, dtype=torch.float32, device=DEVICE)

    def fn():
        _flash_decode_stage2_kernel[(batch, n_heads)](
            partial_o,
            partial_lse,
            output,
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            output.stride(0),
            output.stride(1),
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
            NUM_SPLITS=num_splits,
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


def bench_decode_e2e(
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    page_size,
    seq_len,
    sw_val,
) -> float:
    """Benchmark full triton_paged_decode. Returns latency_us."""
    q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale, sw = make_decode_inputs(
        batch, n_heads, n_kv_heads, head_dim, page_size, seq_len, sw_val or 0
    )

    def fn():
        triton_paged_decode(
            q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale, sliding_window=sw
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


def bench_kv_update(
    num_tokens,
    n_kv_heads,
    head_dim,
    page_size,
) -> float:
    """Benchmark _update_paged_kv_cache_kernel. Returns latency_us."""
    num_blocks = math.ceil(num_tokens / page_size) + 1
    k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr = make_kv_update_inputs(
        num_tokens, n_kv_heads, head_dim, page_size, num_blocks
    )

    def fn():
        update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


def bench_context_e2e(
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    page_size,
    q_len,
    kv_len,
    sw_val,
) -> float:
    """Benchmark full triton_paged_context. Returns latency_us."""
    (
        q,
        kv_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_len_with_cache,
        sm_scale,
        sw,
    ) = make_context_inputs(
        batch, n_heads, n_kv_heads, head_dim, page_size, q_len, kv_len, sw_val or 0
    )

    def fn():
        triton_paged_context(
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            sm_scale,
            sliding_window=sw,
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


def bench_context_stage1_kernel(
    batch,
    n_heads,
    n_kv_heads,
    head_dim,
    page_size,
    q_len,
    kv_len,
    num_warps_override=None,
    num_stages_override=None,
    q_block_override=None,
) -> float:
    """Benchmark _paged_context_kernel directly. Returns latency_us."""
    (
        q,
        kv_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_len_with_cache,
        sm_scale,
        _sw,
    ) = make_context_inputs(batch, n_heads, n_kv_heads, head_dim, page_size, q_len, kv_len, 0)
    output = torch.empty_like(q)

    ctx_extra = {}
    if num_warps_override is not None:
        ctx_extra["num_warps"] = num_warps_override
    if num_stages_override is not None:
        ctx_extra["num_stages"] = num_stages_override

    def grid_fn(meta):
        qb = meta["Q_BLOCK"]
        num_q_blocks = (q_len + qb - 1) // qb
        return (batch, n_heads, num_q_blocks)

    def fn():
        _paged_context_kernel[grid_fn](
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            output,
            q.stride(0),
            q.stride(1),
            output.stride(0),
            output.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            SM_SCALE=sm_scale,
            N_HEADS=n_heads,
            N_KV_HEADS=n_kv_heads,
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
            PAGE_SIZE=page_size,
            SLIDING_WINDOW=0,
            **ctx_extra,
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


def bench_gather_kernel(
    batch,
    n_kv_heads,
    head_dim,
    page_size,
    q_len,
) -> float:
    """Benchmark _fast_gather_sdpa_kernel. Returns latency_us."""
    max_pages = math.ceil(q_len / page_size)
    total_pages = batch * max_pages
    num_blocks = total_pages
    kv_cache = torch.randn(
        num_blocks, 2, n_kv_heads, page_size, head_dim, dtype=DTYPE, device=DEVICE
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=DEVICE)
    max_kv_len = max_pages * page_size
    k_sdpa = torch.empty(batch, n_kv_heads, max_kv_len, head_dim, dtype=DTYPE, device=DEVICE)
    v_sdpa = torch.empty(batch, n_kv_heads, max_kv_len, head_dim, dtype=DTYPE, device=DEVICE)

    def fn():
        _fast_gather_sdpa_kernel[(total_pages, n_kv_heads)](
            kv_cache,
            kv_indices,
            k_sdpa,
            v_sdpa,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            k_sdpa.stride(0),
            k_sdpa.stride(1),
            k_sdpa.stride(2),
            MAX_PAGES=max_pages,
            N_KV_HEADS=n_kv_heads,
            PAGE_SIZE=page_size,
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=triton.next_power_of_2(head_dim),
        )

    ms = triton.testing.do_bench(fn, warmup=25, rep=100)
    return ms * 1000.0


# ---------------------------------------------------------------------------
# Main Benchmark Runners
# ---------------------------------------------------------------------------


def run_decode_benchmarks(
    shape_filter: Optional[List[str]] = None,
    num_warps_override=None,
    num_stages_override=None,
) -> List[dict]:
    results = []
    print("\n" + "=" * 110)
    print(f"{'DECODE BENCHMARKS':^110}")
    print("=" * 110)
    hdr = (
        f"{'ID':<4} {'B':>3} {'NH':>4} {'NKV':>4} {'HD':>4} {'PS':>4} {'SL':>5} {'SW':>6} "
        f"{'splits':>6} "
        f"{'s1(us)':>8} {'s2(us)':>8} {'e2e(us)':>8} {'s1%e2e':>7}"
    )
    print(hdr)
    print("-" * 110)

    for row in DECODE_SHAPES:
        sid, batch, n_heads, n_kv_heads, head_dim, page_size, seq_len, sw = row
        if shape_filter and sid not in shape_filter:
            continue

        s1_us, num_splits = bench_decode_stage1(
            batch,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size,
            seq_len,
            sw,
            num_warps_override,
            num_stages_override,
        )
        s2_us = bench_decode_stage2(batch, n_heads, head_dim, num_splits)
        e2e_us = bench_decode_e2e(batch, n_heads, n_kv_heads, head_dim, page_size, seq_len, sw)
        s1_pct = 100.0 * s1_us / e2e_us if e2e_us > 0 else 0.0

        print(
            f"{sid:<4} {batch:>3} {n_heads:>4} {n_kv_heads:>4} {head_dim:>4} {page_size:>4} "
            f"{seq_len:>5} {sw:>6} {num_splits:>6} "
            f"{s1_us:>8.2f} {s2_us:>8.2f} {e2e_us:>8.2f} {s1_pct:>6.1f}%"
        )

        results.append(
            {
                "id": sid,
                "batch": batch,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "page_size": page_size,
                "seq_len": seq_len,
                "sw": sw,
                "num_splits": num_splits,
                "stage1_us": s1_us,
                "stage2_us": s2_us,
                "e2e_us": e2e_us,
            }
        )

    return results


def run_context_benchmarks(
    shape_filter: Optional[List[str]] = None,
    num_warps_override=None,
    num_stages_override=None,
) -> List[dict]:
    results = []
    print("\n" + "=" * 100)
    print(f"{'CONTEXT/PREFILL BENCHMARKS (Triton paged path, q_len < 512)':^100}")
    print("=" * 100)
    hdr = (
        f"{'ID':<4} {'B':>3} {'NH':>4} {'NKV':>4} {'HD':>4} {'PS':>4} {'qlen':>5} {'kvlen':>6} "
        f"{'ctx(us)':>8} {'e2e(us)':>8}"
    )
    print(hdr)
    print("-" * 100)

    for row in CONTEXT_SHAPES:
        sid, batch, n_heads, n_kv_heads, head_dim, page_size, q_len, kv_len = row
        if shape_filter and sid not in shape_filter:
            continue

        ctx_us = bench_context_stage1_kernel(
            batch,
            n_heads,
            n_kv_heads,
            head_dim,
            page_size,
            q_len,
            kv_len,
            num_warps_override,
            num_stages_override,
        )
        e2e_us = bench_context_e2e(
            batch, n_heads, n_kv_heads, head_dim, page_size, q_len, kv_len, 0
        )

        print(
            f"{sid:<4} {batch:>3} {n_heads:>4} {n_kv_heads:>4} {head_dim:>4} {page_size:>4} "
            f"{q_len:>5} {kv_len:>6} {ctx_us:>8.2f} {e2e_us:>8.2f}"
        )

        results.append(
            {
                "id": sid,
                "batch": batch,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "page_size": page_size,
                "q_len": q_len,
                "kv_len": kv_len,
                "ctx_kernel_us": ctx_us,
                "e2e_us": e2e_us,
            }
        )

    return results


def run_gather_benchmarks(
    shape_filter: Optional[List[str]] = None,
) -> List[dict]:
    results = []
    print("\n" + "=" * 90)
    print(f"{'GATHER+SDPA BENCHMARKS (q_len >= 512)':^90}")
    print("=" * 90)
    hdr = (
        f"{'ID':<4} {'B':>3} {'NH':>4} {'NKV':>4} {'HD':>4} {'PS':>4} {'qlen':>5} "
        f"{'gather(us)':>10} {'e2e(us)':>9}"
    )
    print(hdr)
    print("-" * 90)

    for row in GATHER_SHAPES:
        sid, batch, n_heads, n_kv_heads, head_dim, page_size, q_len = row
        if shape_filter and sid not in shape_filter:
            continue

        gather_us = bench_gather_kernel(batch, n_kv_heads, head_dim, page_size, q_len)
        e2e_us = bench_context_e2e(batch, n_heads, n_kv_heads, head_dim, page_size, q_len, q_len, 0)

        print(
            f"{sid:<4} {batch:>3} {n_heads:>4} {n_kv_heads:>4} {head_dim:>4} {page_size:>4} "
            f"{q_len:>5} {gather_us:>10.2f} {e2e_us:>9.2f}"
        )

        results.append(
            {
                "id": sid,
                "batch": batch,
                "n_heads": n_heads,
                "n_kv_heads": n_kv_heads,
                "head_dim": head_dim,
                "page_size": page_size,
                "q_len": q_len,
                "gather_us": gather_us,
                "e2e_us": e2e_us,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------------------------------


def run_parameter_sweep(
    shape_filter: Optional[List[str]] = None,
    output_file: Optional[str] = None,
):
    """Sweep num_warps and num_stages for stage1 kernel across all decode shapes."""
    sweep_results = []

    warp_choices = [1, 2, 4, 8, 16]
    stage_choices = [1, 2, 3, 4, 5]

    total = len(DECODE_SHAPES) * len(warp_choices) * len(stage_choices)
    done = 0

    print(
        f"\nSweeping {len(warp_choices)}×{len(stage_choices)} = "
        f"{len(warp_choices) * len(stage_choices)} configs × {len(DECODE_SHAPES)} shapes "
        f"= {total} runs"
    )
    print("(This will take several minutes due to recompilation per config)\n")

    for row in DECODE_SHAPES:
        sid, batch, n_heads, n_kv_heads, head_dim, page_size, seq_len, sw = row
        if shape_filter and sid not in shape_filter:
            continue

        best_us = float("inf")
        best_cfg = None

        for nw in warp_choices:
            for ns in stage_choices:
                try:
                    # Disable autotune by passing overrides
                    s1_us, num_splits = bench_decode_stage1(
                        batch,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        page_size,
                        seq_len,
                        sw,
                        num_warps_override=nw,
                        num_stages_override=ns,
                    )
                    entry = {
                        "id": sid,
                        "batch": batch,
                        "seq_len": seq_len,
                        "sw": sw,
                        "num_warps": nw,
                        "num_stages": ns,
                        "stage1_us": s1_us,
                        "num_splits": num_splits,
                    }
                    sweep_results.append(entry)

                    if s1_us < best_us:
                        best_us = s1_us
                        best_cfg = (nw, ns)

                    done += 1
                    print(f"  [{done:4d}/{total}] {sid} nw={nw} ns={ns}: {s1_us:.2f}us", end="\r")

                except Exception as e:
                    print(f"\n  [{sid}] nw={nw} ns={ns}: FAILED ({e})")
                    done += 1

        if best_cfg:
            print(
                f"\n{sid}: best config = num_warps={best_cfg[0]}, num_stages={best_cfg[1]}, "
                f"stage1={best_us:.2f}us"
            )

    if output_file:
        with open(output_file, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep results saved to {output_file}")

    return sweep_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Triton paged attention benchmark")
    parser.add_argument(
        "--shapes",
        type=str,
        default=None,
        help="Comma-separated shape IDs to benchmark (e.g. D1,D2,D5)",
    )
    parser.add_argument(
        "--mode",
        choices=["decode", "context", "gather", "all"],
        default="all",
        help="Which benchmark group to run",
    )
    parser.add_argument(
        "--num-warps", type=int, default=None, help="Override num_warps for stage1/context kernel"
    )
    parser.add_argument(
        "--num-stages", type=int, default=None, help="Override num_stages for stage1/context kernel"
    )
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep (slow)")
    parser.add_argument(
        "--sweep-out",
        type=str,
        default="sweep_results.json",
        help="Output JSON file for sweep results",
    )
    return parser.parse_args()


def print_env():
    import triton

    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    print(f"Triton:  {triton.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype:   {DTYPE}")
    print("Bench:   triton.testing.do_bench(warmup=25, rep=100)")


if __name__ == "__main__":
    args = parse_args()
    shape_filter = [s.strip() for s in args.shapes.split(",")] if args.shapes else None

    print_env()

    if args.sweep:
        run_parameter_sweep(shape_filter=shape_filter, output_file=args.sweep_out)
    else:
        if args.mode in ("decode", "all"):
            run_decode_benchmarks(shape_filter, args.num_warps, args.num_stages)

        if args.mode in ("context", "all"):
            run_context_benchmarks(shape_filter, args.num_warps, args.num_stages)

        if args.mode in ("gather", "all"):
            run_gather_benchmarks(shape_filter)

    print("\nDone.")
