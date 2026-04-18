#!/usr/bin/env python3
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

"""Benchmark BF16 tensor-core candidates for Gemma4 Kernel A decode shapes."""

from __future__ import annotations

import statistics

import torch

from tensorrt_llm import deep_gemm


def _bench_us(fn, warmup: int = 50, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return statistics.median(times_us)


def _print_case(name: str, m: int, k: int, n: int) -> None:
    print(f"CASE {name}: m={m} k={k} n={n}")
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b_nt = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    b_nn = b_nt.t().contiguous()
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    ref_nt = torch.nn.functional.linear(a, b_nt)
    ref_nn = a @ b_nn

    def torch_nt() -> None:
        torch.matmul(a, b_nt.t(), out=out)

    print(f"  torch matmul nt    {_bench_us(torch_nt):8.3f} us")

    variants = [
        ("cublaslt_nt", deep_gemm.cublaslt_gemm_nt, b_nt, ref_nt),
        ("bf16_nt", deep_gemm.bf16_gemm_nt, b_nt, ref_nt),
        ("cublaslt_nn", deep_gemm.cublaslt_gemm_nn, b_nn, ref_nn),
        ("bf16_nn", deep_gemm.bf16_gemm_nn, b_nn, ref_nn),
    ]
    for label, fn, b, ref in variants:

        def run() -> None:
            fn(a, b, out)

        us = _bench_us(run)
        diff = (out.float() - ref.float()).abs()
        print(
            f"  {label:14s} {us:8.3f} us  "
            f"max_diff={diff.max().item():.6f}  bit_exact={torch.equal(out, ref)}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Skipping: requires Hopper+, got {torch.cuda.get_device_name()}")
        return

    print("=" * 72)
    print("Gemma4 Megakernel: BF16 Tensor-Core Candidate Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)
    _print_case("qkv", m=1, k=2816, n=8192)
    _print_case("oproj", m=1, k=4096, n=2816)
    print("=" * 72)


if __name__ == "__main__":
    main()
