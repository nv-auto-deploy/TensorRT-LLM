# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Definitive within-process A/B: old fixed SPLIT_K=24 vs new per-N _splitk_split_k.

Cross-process bench comparison is contaminated by clock/thermal drift, so we A/B
both configs back-to-back in one process (interleaved, min over REPS) on the 4
K=7168 decode projection shapes. This isolates the GPU-time effect of the
idea_0063 per-N SPLIT_K schedule on the split-K decode path.
"""

import sys

import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _safe_act_quant,
    _splitk_block_n,
    _splitk_split_k,
    _w8a8_block_fp8_matmul_splitk,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
K = 7168
OLD_SPLIT_K = 24  # idea_0025 fixed default
SHAPES = [(256, K), (576, K), (1536, K), (2304, K)]


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


def bench(M, N, Kd, split_k, calls_per_graph=64, replays=50):
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
            BLOCK_SIZE_N=bn,  # BLOCK_SIZE_K=None -> _splitk_block_k=128
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
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(replays):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / (calls_per_graph * replays) * 1e3  # us/call


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"device={torch.cuda.get_device_name()}  M={M}  reps={reps}\n")
    old_sum = new_sum = 0.0
    for N, Kd in SHAPES:
        new_sk = _splitk_split_k(N)
        old = min(bench(M, N, Kd, OLD_SPLIT_K) for _ in range(reps))
        new = min(bench(M, N, Kd, new_sk) for _ in range(reps))
        old_sum += old
        new_sum += new
        print(
            f"  N={N:5d}  old(SK24)={old:.3f}us  new(SK{new_sk})={new:.3f}us  "
            f"delta={(new - old) / old * 100:+.2f}%"
        )
    om, nm = old_sum / len(SHAPES), new_sum / len(SHAPES)
    print(f"\n  MEAN  old={om:.4f}us  new={nm:.4f}us  delta={(nm - om) / om * 100:+.2f}%")
    print("  (split-K-shape mean; the metric of record for idea_0063)")


if __name__ == "__main__":
    main()
