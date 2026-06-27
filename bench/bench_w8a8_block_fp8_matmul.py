# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph-amortized microbench for the block-scaled FP8 GEMM kernel.

The real decode path runs this kernel under a CUDA graph (compile_backend:
torch-cudagraph), so per-launch host overhead is amortized in production. A
plain python-loop wall-clock measure on an M=1 GEMV would instead measure the
~10us host launch cadence and show real GPU-time wins as flat parity. We
therefore capture K back-to-back kernel calls into one graph, replay it R times,
and divide -- isolating GPU time per call (the e2e-faithful per-decode cost).

@triton.autotune (if present) is warmed up *eagerly* before capture so the
in-capture launches are pure cuLaunchKernel (autotune cache hits), matching how
AutoDeploy warms up before graph capture.

Usage:
    PYTHONPATH=<worktree> python bench/bench_w8a8_block_fp8_matmul.py [M ...]
"""

import sys

import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _safe_act_quant,
    _w8a8_block_fp8_matmul_triton,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# DeepSeek-V4-class MLA / dense projection (N, K) per TP8 rank (decode GEMV).
DECODE_SHAPES = [
    (1536, 7168),  # q_a_proj-like
    (3072, 1536),  # q_b_proj-like
    (576, 7168),  # kv_a_proj-like (N not mult of 128)
    (4096, 512),  # kv_b_proj-like
    (7168, 2048),  # o_proj-like
    (2304, 7168),  # dense gate/up-like
    (7168, 2304),  # dense down-like
    (256, 7168),  # shared-expert gate/up-like
]


def _quant_weight_block_fp8(w, block_n=128, block_k=128):
    N, K = w.shape
    sn, sk = triton.cdiv(N, block_n), triton.cdiv(K, block_k)
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


def _make_inputs(M, N, K):
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), 128)
    b_fp8, b_s = _quant_weight_block_fp8(b)
    return a_fp8, b_fp8, a_s, b_s


def bench_shape(M, N, K, calls_per_graph=64, replays=40):
    a_fp8, b_fp8, a_s, b_s = _make_inputs(M, N, K)

    def fn():
        return _w8a8_block_fp8_matmul_triton(
            a_fp8, b_fp8, a_s, b_s, [128, 128], output_dtype=torch.bfloat16
        )

    # Eager warmup: triggers Triton JIT + autotune selection (do_bench) outside capture.
    for _ in range(15):
        fn()
    torch.cuda.synchronize()

    # Warm the private mempool on a side stream, then capture K calls.
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
    total_ms = start.elapsed_time(end)
    return total_ms / (calls_per_graph * replays)  # ms per kernel call


def main():
    Ms = [int(x) for x in sys.argv[1:]] or [1, 2, 512]
    print(f"device={torch.cuda.get_device_name()}  cap={torch.cuda.get_device_capability()}")
    for M in Ms:
        per_call = []
        for N, K in DECODE_SHAPES:
            ms = bench_shape(M, N, K)
            per_call.append(ms)
            print(f"  M={M:4d} N={N:5d} K={K:5d}  per_call_ms={ms:.6f}  ({ms * 1e3:.3f} us)")
        mean_ms = sum(per_call) / len(per_call)
        print(
            f"M={M:4d} MEAN per_call_ms={mean_ms:.6f}  ({mean_ms * 1e3:.3f} us)  [primary metric]\n"
        )


if __name__ == "__main__":
    main()
