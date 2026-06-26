# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbench for the DeepSeek V4 sparse-attention attend kernel.

Benchmarks ``_deepseek_v4_sparse_attention`` (the function fused by idea_0001)
on the two shapes the real workload hits:

* decode  : num_tokens=1, num_heads=64, L(selected)=640, D=512  -> measured under
            CUDA-graph replay (faithful proxy for tpot, which runs under cudagraph)
* prefill : num_tokens=512, num_heads=64, k_select=640, kv_rows=2048, D=512 (eager)

DSV4-Flash dims: head_dim=512, num_attention_heads=64, sliding_window=128,
index_topk=512  (so decode selected rows ~= window+index_topk = 640).
"""

from __future__ import annotations

import time

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (
    deepseek_v4_sparse_attention as dsv4,
)

D = 512
NUM_HEADS = 64
SCALE = D**-0.5
DTYPE = torch.bfloat16
DEV = "cuda"


def _make_decode_inputs(L: int = 640):
    """Decode: 1 query token, 64 heads, L selected kv rows (rel_topk = arange)."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, NUM_HEADS, D, device=DEV, dtype=DTYPE)
    kv = torch.randn(1, L, D, device=DEV, dtype=DTYPE)
    attn_sink = torch.randn(NUM_HEADS, device=DEV, dtype=DTYPE)
    # identity selection (the decode path passes rel_topk = arange over selected rows)
    topk = torch.arange(L, device=DEV, dtype=torch.int64).view(1, 1, L)
    return q, kv, attn_sink, topk


def _make_prefill_inputs(num_tokens: int = 512, kv_rows: int = 2048, k_select: int = 640):
    """Prefill: num_tokens queries, each selecting k_select rows out of kv_rows."""
    torch.manual_seed(1)
    q = torch.randn(1, num_tokens, NUM_HEADS, D, device=DEV, dtype=DTYPE)
    kv = torch.randn(1, kv_rows, D, device=DEV, dtype=DTYPE)
    attn_sink = torch.randn(NUM_HEADS, device=DEV, dtype=DTYPE)
    # realistic-ish: per token a random valid subset, ~10% masked (-1)
    topk = torch.randint(0, kv_rows, (1, num_tokens, k_select), device=DEV, dtype=torch.int64)
    mask = torch.rand(1, num_tokens, k_select, device=DEV) < 0.1
    topk = torch.where(mask, torch.full_like(topk, -1), topk)
    return q, kv, attn_sink, topk


def _time_eager(fn, iters: int = 100, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def _time_graph(fn, iters: int = 200, warmup: int = 5) -> float:
    """Time fn() under CUDA-graph replay (hides launch overhead, like real serving)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    fn_attend = dsv4._deepseek_v4_sparse_attention

    # ---- decode (primary; cudagraph) ----
    q, kv, sink, topk = _make_decode_inputs()
    call = lambda: fn_attend(q, kv, sink, topk, SCALE)  # noqa: E731
    eager_dec = _time_eager(call)
    try:
        graph_dec = _time_graph(call)
    except Exception as e:  # noqa: BLE001
        graph_dec = float("nan")
        print(f"[warn] decode graph capture failed: {e}")
    print(
        f"shape=decode(B1,H64,L640,D512) microbench_ms={graph_dec:.5f} "
        f"(graph) eager_ms={eager_dec:.5f}"
    )

    # ---- prefill (eager) ----
    q, kv, sink, topk = _make_prefill_inputs()
    call = lambda: fn_attend(q, kv, sink, topk, SCALE)  # noqa: E731
    eager_pre = _time_eager(call, iters=30, warmup=5)
    print(f"shape=prefill(T512,H64,K640,kv2048,D512) microbench_ms={eager_pre:.5f} (eager)")

    # primary number = decode under cudagraph (matches tpot metric)
    primary = graph_dec if graph_dec == graph_dec else eager_dec  # nan check
    print(f"PRIMARY_microbench_ms={primary:.5f}")


if __name__ == "__main__":
    main()
