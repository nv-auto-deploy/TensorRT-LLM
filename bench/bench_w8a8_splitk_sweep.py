# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BLOCK_SIZE_K x SPLIT_K sweep for the split-K block-FP8 decode GEMV (idea_0063).

idea_0025 added the split-K reduction path (SPLIT_K=24, BLOCK_SIZE_K pinned to the
quant group_k=128) and swept SPLIT_K only. This sweeps the 2D
(BLOCK_SIZE_K x SPLIT_K) granularity for the K=7168 M=1 decode GEMV, holding
BLOCK_SIZE_N at its tuned per-N default. BLOCK_SIZE_K (the MMA contraction tile)
is decoupled from the scale group_k: it must divide 128, and a smaller tile keeps
the atomic count fixed at SPLIT_K while raising the K-loop trip count (deeper
software pipeline of the memory-bound B-tile loads).

CUDA-graph-amortized timing (the real decode path runs under a cudagraph; a plain
wall-clock loop on an M=1 GEMV measures the ~10us host launch cadence, not GPU
time). We capture K back-to-back calls into one graph, replay R times, divide.

Usage:
    PYTHONPATH=<worktree> python bench/bench_w8a8_splitk_sweep.py [M ...]
"""

import sys

import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _SPLITK_SPLIT_K,
    _safe_act_quant,
    _splitk_block_n,
    _w8a8_block_fp8_matmul_splitk,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# DeepSeek-V4-class K=7168 decode projections that hit the split-K path (M<=4).
K = 7168
SPLITK_SHAPES = [
    (256, K),  # shared-expert gate/up-like
    (576, K),  # kv_a_proj-like (N not mult of 128)
    (1536, K),  # q_a_proj-like
    (2304, K),  # dense gate/up-like
]

# 2D sweep grid. BLOCK_SIZE_K must divide the quant block_k (128).
BLOCK_SIZE_K_GRID = [32, 64, 128]
SPLIT_K_GRID = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56]

DEFAULT_BLOCK_SIZE_K = 128  # current pinned default (== group_k)
DEFAULT_SPLIT_K = _SPLITK_SPLIT_K  # 24


def _quant_weight_block_fp8(w, block_n=128, block_k=128):
    N, Kd = w.shape
    sn, sk = triton.cdiv(N, block_n), triton.cdiv(Kd, block_k)
    scale = torch.empty(sn, sk, dtype=torch.float32, device=w.device)
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    for i in range(sn):
        for j in range(sk):
            blk = w[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k].float()
            s = (blk.abs().amax() / FP8_MAX).clamp(min=1e-12)
            scale[i, j] = s
            w_fp8[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k] = (
                (blk / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return w_fp8, scale


def _make_inputs(M, N, Kd):
    torch.manual_seed(0)
    a = torch.randn(M, Kd, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, Kd, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), 128)
    b_fp8, b_s = _quant_weight_block_fp8(b)
    return a_fp8, b_fp8, a_s, b_s


def bench_config(M, N, Kd, split_k, block_size_k, calls_per_graph=64, replays=50):
    a_fp8, b_fp8, a_s, b_s = _make_inputs(M, N, Kd)
    bn = _splitk_block_n(N)

    def fn():
        return _w8a8_block_fp8_matmul_splitk(
            a_fp8,
            b_fp8,
            a_s,
            b_s,
            128,
            128,
            torch.bfloat16,
            M,
            N,
            Kd,
            SPLIT_K=split_k,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=block_size_k,
        )

    for _ in range(15):
        fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(calls_per_graph):
            fn()
    torch.cuda.synchronize()

    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / (calls_per_graph * replays) * 1e3  # us/call


def main():
    Ms = [int(x) for x in sys.argv[1:]] or [1]
    print(f"device={torch.cuda.get_device_name()}  cap={torch.cuda.get_device_capability()}")
    print(f"default config: BLOCK_SIZE_K={DEFAULT_BLOCK_SIZE_K} SPLIT_K={DEFAULT_SPLIT_K}\n")

    for M in Ms:
        print(f"================ M={M} ================")
        default_total = 0.0
        best_per_shape_total = 0.0
        # accumulate per-(bk,sk) totals across shapes to find a single global config
        global_acc = {(bk, sk): 0.0 for bk in BLOCK_SIZE_K_GRID for sk in SPLIT_K_GRID}
        for N, Kd in SPLITK_SHAPES:
            results = {}
            for bk in BLOCK_SIZE_K_GRID:
                for sk in SPLIT_K_GRID:
                    us = bench_config(M, N, Kd, sk, bk)
                    results[(bk, sk)] = us
                    global_acc[(bk, sk)] += us
            default_us = results[(DEFAULT_BLOCK_SIZE_K, DEFAULT_SPLIT_K)]
            (best_bk, best_sk), best_us = min(results.items(), key=lambda kv: kv[1])
            default_total += default_us
            best_per_shape_total += best_us
            delta = (best_us - default_us) / default_us * 100
            print(
                f"  N={N:5d} K={Kd}  default(BK{DEFAULT_BLOCK_SIZE_K},SK{DEFAULT_SPLIT_K})="
                f"{default_us:.3f}us  best(BK{best_bk},SK{best_sk})={best_us:.3f}us  "
                f"delta={delta:+.1f}%"
            )
            # show the top-5 configs for this shape
            top = sorted(results.items(), key=lambda kv: kv[1])[:5]
            print("       top5: " + "  ".join(f"BK{bk}/SK{sk}={us:.3f}" for (bk, sk), us in top))

        # best single global config (same (bk,sk) for all shapes)
        (gbk, gsk), gtot = min(global_acc.items(), key=lambda kv: kv[1])
        gmean = gtot / len(SPLITK_SHAPES)
        dmean = default_total / len(SPLITK_SHAPES)
        bmean = best_per_shape_total / len(SPLITK_SHAPES)
        print(
            f"\n  MEAN  default={dmean:.3f}us  best-global(BK{gbk},SK{gsk})={gmean:.3f}us "
            f"({(gmean - dmean) / dmean * 100:+.1f}%)  best-per-shape={bmean:.3f}us "
            f"({(bmean - dmean) / dmean * 100:+.1f}%)"
        )
        # rank global configs
        ranked = sorted(global_acc.items(), key=lambda kv: kv[1])[:8]
        print(
            "  global top8 (mean us): "
            + "  ".join(f"BK{bk}/SK{sk}={tot / len(SPLITK_SHAPES):.3f}" for (bk, sk), tot in ranked)
        )
        print()


if __name__ == "__main__":
    main()
