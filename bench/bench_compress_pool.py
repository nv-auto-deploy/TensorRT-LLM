# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Amortized CUDA-graph microbench for ``_dsv4_compress_pool_kernel`` autotuning.

The fused DeepSeek-V4 compressor attention-pool kernel (idea_0036) launches with
a hardcoded ``num_warps=4`` and no ``@triton.autotune``.  Decode profile
(idea_0036 trace, B200 TP8, 10-layer proxy) shows TWO distinct decode shapes:

* PRIMARY  N=1024 R=8 D=128  (indexer fullrange, grid=(1024,1)) -> 11.5us avg, 75%
* secondary N=1    R=8 D=512  (main compressor new row, grid=(1,4)) -> 3.87us avg, 25%

The PRIMARY shape is a 1024-block grid -- genuinely GPU-bound (~23x the HBM-bandwidth
floor), so occupancy IS tunable.  Single-call replay measures the host launch
cadence (~10us) which hides the kernel time and is invariant to config, so we
stack K launches into one captured graph and divide (cudagraph-microbench-amortized
-timing note).

Usage:
  PYTHONPATH=<worktree> python bench/bench_compress_pool.py            # sweep
  PYTHONPATH=<worktree> python bench/bench_compress_pool.py --prefill  # incl prefill
  PYTHONPATH=<worktree> python bench/bench_compress_pool.py --prod     # shipped op as-is
"""

import argparse
import sys

import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_compressor import (
    _dsv4_compress_pool_kernel,
    deepseek_v4_compress_pool,
)

DTYPE = torch.bfloat16
DEV = "cuda"

# (N, R, D): the shapes the real DSV4-Flash decode/prefill workload hits.
#  N = rows (B*max_compressed_len for fullrange, or B for new-row); R = ratio axis
#  (2*compress_ratio=8 for the overlap layers); D = head_dim (128 indexer / 512 main).
DECODE_SHAPES = [
    (1024, 8, 128),  # PRIMARY: indexer fullrange B=1 (max_compressed_len=1024)
    (2048, 8, 128),  # indexer fullrange B=2
    (1, 8, 512),  # main compressor new-row B=1
    (2, 8, 512),  # main compressor new-row B=2
    (4, 8, 128),  # small indexer (grid~4 secondary bucket)
]
PREFILL_SHAPES = [
    (256, 8, 512),  # main compressor context (~ISL/ratio rows), head_dim 512
    (256, 8, 128),  # indexer compressor context, index_head_dim 128
    (512, 8, 512),
]


def amortized_us(launch_fn, k_per_graph=64, n_replays=200, best_of=3):
    """Per-launch GPU time (us), amortized over a graph of k_per_graph launches."""
    for _ in range(15):
        launch_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k_per_graph):
            launch_fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(best_of):
        start.record()
        for _ in range(n_replays):
            g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) * 1000.0 / (n_replays * k_per_graph))
    return best


def make_buffers(N, R, D):
    kv = (torch.randn(N, R, D, device=DEV, dtype=DTYPE) * 0.5).contiguous()
    gate = (torch.randn(N, R, D, device=DEV, dtype=DTYPE) * 0.5).contiguous()
    out = torch.empty((N, D), device=DEV, dtype=DTYPE)
    return kv, gate, out


def bench_config(N, R, D, num_warps, num_stages, block_d):
    kv, gate, out = make_buffers(N, R, D)
    BLOCK_D = block_d if block_d is not None else min(128, triton.next_power_of_2(D))
    grid = (N, triton.cdiv(D, BLOCK_D))
    kw = {"BLOCK_R": triton.next_power_of_2(R), "BLOCK_D": BLOCK_D}
    if num_warps is not None:
        kw["num_warps"] = num_warps
    if num_stages is not None:
        kw["num_stages"] = num_stages

    def launch():
        _dsv4_compress_pool_kernel[grid](kv, gate, out, N, R, D, **kw)

    return amortized_us(launch)


def bench_prod(N, R, D):
    kv, gate, _ = make_buffers(N, R, D)

    def launch():
        deepseek_v4_compress_pool(kv, gate)

    return amortized_us(launch)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def robust_compare(shapes, configs, rounds=7):
    """Round-robin interleaved measurement to cancel thermal/boost drift."""
    samples = {lbl: {sh: [] for sh in shapes} for (lbl, _, _, _) in configs}
    for _ in range(rounds):
        for N, R, D in shapes:
            for lbl, nw, ns, bd in configs:
                us = bench_config(N, R, D, nw, ns, bd)
                samples[lbl][(N, R, D)].append(us)
    return {lbl: {sh: _median(v) for sh, v in d.items()} for lbl, d in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true", help="bench shipped op")
    ap.add_argument("--prefill", action="store_true", help="include prefill shapes")
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()

    print(
        f"# device={torch.cuda.get_device_name()} triton={triton.__version__} rounds={args.rounds}",
        flush=True,
    )

    decode = list(DECODE_SHAPES)
    shapes = decode + (PREFILL_SHAPES if args.prefill else [])

    if args.prod:
        print("# shipped deepseek_v4_compress_pool")
        tot = 0.0
        for N, R, D in shapes:
            us = bench_prod(N, R, D)
            tot += us
            print(f"N={N:<5} R={R} D={D:<4} prod_us={us:.4f}", flush=True)
        print(f"# mean_us={tot / len(shapes):.4f}")
        return 0

    # None == hardcoded num_warps=4 baseline (current ship). Include explicit nw=4
    # as a noise gauge: |median(default) - median(nw=4)| lower-bounds the noise.
    configs = [
        ("default", None, None, None),
        ("nw=1", 1, None, None),
        ("nw=2", 2, None, None),
        ("nw=4", 4, None, None),
        ("nw=8", 8, None, None),
        ("nw=1,ns=1", 1, 1, None),
        ("nw=2,ns=1", 2, 1, None),
    ]
    res = robust_compare(shapes, configs, rounds=args.rounds)

    hdr = "config".ljust(12) + "  " + "  ".join(f"{N}x{R}x{D}".rjust(12) for N, R, D in shapes)
    print(hdr)
    for lbl, _, _, _ in configs:
        row = [res[lbl][(N, R, D)] for (N, R, D) in shapes]
        print(lbl.ljust(12) + "  " + "  ".join(f"{v:12.4f}" for v in row), flush=True)

    # weighted decode score: PRIMARY shape (N=1024 D=128) gets 3x weight (75% of time)
    prim = (1024, 8, 128)
    print("\n# per-config PRIMARY (N=1024 R=8 D=128) us  [the 75%-of-decode shape]")
    base = res["default"][prim]
    for lbl, _, _, _ in configs:
        v = res[lbl][prim]
        print(f"#   {lbl.ljust(12)} {v:8.4f}us  delta={(v - base) / base * 100:+.2f}%")
    noise = abs(res["default"][prim] - res["nw=4"][prim]) / base * 100
    print(f"# PRIMARY noise floor |default-nw=4| = {noise:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
