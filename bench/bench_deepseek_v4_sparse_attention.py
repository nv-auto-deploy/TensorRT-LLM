# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbench for the DeepSeek V4 sparse-attention attend kernel.

Benchmarks ``_deepseek_v4_sparse_attention`` (the function fused by idea_0001)
on the shapes the real workload hits:

* decode (PRIMARY, tpot proxy) : num_tokens=1, L(selected)=640, D=512.  The model
            has num_attention_heads=64 but the e2e config shards MLA over TP=8
            (shard_layers:[mla,moe], tp:8), so the *per-rank* decode shape the
            kernel actually sees is num_heads = 64/8 = **8**.  This is the shape
            autotuning must target; H=64 is kept only as the original (off-shape)
            reference.
* prefill : num_tokens=512, num_heads=64, k_select=640, kv_rows=2048, D=512 (eager)

PRIMARY metric = decode at H=8 timed under **amortized (stacked) CUDA-graph
replay**.  A single g.replay() per call measures the host launch cadence
(~10us) which hides the (shorter) GPU kernel time and is invariant to kernel
changes; in real serving the attend runs back-to-back *inside* the per-decode
cudagraph, so its true contribution is the serial GPU time.  We capture many
calls into one graph and divide, which both reveals sub-floor kernel wins and
matches the e2e-relevant cost.

DSV4-Flash dims: head_dim=512, num_attention_heads=64, sliding_window=128,
index_topk=512  (so decode selected rows ~= window+index_topk = 640).  TP=8.
"""

from __future__ import annotations

import time

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (
    deepseek_v4_sparse_attention as dsv4,
)

D = 512
NUM_HEADS = 64  # model total; the e2e per-rank count after TP=8 sharding is 8
TP = 8
SCALE = D**-0.5
DTYPE = torch.bfloat16
DEV = "cuda"


def _make_decode_inputs(num_heads: int, L: int = 640):
    """Decode: 1 query token, ``num_heads`` heads, L selected kv rows (rel_topk = arange)."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, num_heads, D, device=DEV, dtype=DTYPE)
    kv = torch.randn(1, L, D, device=DEV, dtype=DTYPE)
    attn_sink = torch.randn(num_heads, device=DEV, dtype=DTYPE)
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


def _time_graph_single(fn, iters: int = 500, warmup: int = 20) -> float:
    """One captured call per replay -> measures the HOST g.replay() cadence floor."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
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


def _time_graph_amortized(
    fn, reps: int = 64, iters: int = 100, warmup: int = 10, best_of: int = 3
) -> float:
    """Amortized true GPU kernel time (the e2e-relevant cost).

    Capture ``reps`` back-to-back calls into ONE graph and amortize the single
    host launch over them; this reveals sub-host-floor kernel wins the
    single-call timer cannot see.  best_of takes the min over repeats to
    suppress clock/thermal jitter on these tiny kernels.
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(best_of):
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / iters / reps * 1000.0)
    return best


def _bench_decode(num_heads: int, tag: str) -> float:
    q, kv, sink, topk = _make_decode_inputs(num_heads)
    call = lambda: dsv4._deepseek_v4_sparse_attention(q, kv, sink, topk, SCALE)  # noqa: E731
    try:
        gpu = _time_graph_amortized(call)
        host = _time_graph_single(call)
    except Exception as e:  # noqa: BLE001
        gpu = host = float("nan")
        print(f"[warn] decode graph capture failed ({tag}): {e}")
    print(
        f"shape=decode-{tag}(B1,H{num_heads},L640,D512) microbench_ms={gpu:.5f} "
        f"(amortized GPU; host-floor={host * 1000:.2f}us)"
    )
    return gpu


def main():
    # ---- decode PRIMARY: per-rank shape (H = 64 / TP = 8) ----
    primary = _bench_decode(NUM_HEADS // TP, tag="tp8")  # H=8, the real per-rank shape
    # ---- decode reference: full H=64 (original off-shape sweep target) ----
    _bench_decode(NUM_HEADS, tag="h64")

    # ---- prefill (eager) ----
    q, kv, sink, topk = _make_prefill_inputs()
    call = lambda: dsv4._deepseek_v4_sparse_attention(q, kv, sink, topk, SCALE)  # noqa: E731
    eager_pre = _time_eager(call, iters=30, warmup=5)
    print(f"shape=prefill(T512,H64,K640,kv2048,D512) microbench_ms={eager_pre:.5f} (eager)")

    # primary number = H=8 decode amortized GPU time (the real per-rank tpot shape)
    print(f"PRIMARY_microbench_ms={primary:.5f}")


if __name__ == "__main__":
    main()
