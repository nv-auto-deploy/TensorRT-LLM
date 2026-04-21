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

"""Standalone Hopper BF16 WGMMA microbenchmark for Gemma4 MoE expert shapes."""

from __future__ import annotations

import statistics

import torch
from test_wgmma_qkv_micro import _load_module

HIDDEN_SIZE = 2816
EXPERT_INTERMEDIATE = 704
FC1_ROWS = 2 * EXPERT_INTERMEDIATE
FC2_ROWS = HIDDEN_SIZE


def _bench_us(fn, warmup: int = 20, iters: int = 100) -> float:
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


def _run_shape(mod, *, name: str, num_rows: int, hidden_size: int) -> None:
    x = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(num_rows, hidden_size, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x.unsqueeze(0), w).squeeze(0).float()

    print(f"{name}: rows={num_rows} hidden={hidden_size}")
    variant_names = {
        0: "nn",
        1: "nt",
        2: "tn",
        3: "tt",
    }
    for dup_n in (8, 16):
        best_variant = None
        best_time = None
        for variant in range(4):
            checksum = torch.empty((num_rows + 63) // 64, device="cuda", dtype=torch.float32)
            mod.run_dup_into(x, w, checksum, dup_n, variant)
            total = checksum.sum().item()
            expected = ref.sum().item() * dup_n
            diff = abs(total - expected)

            def run() -> None:
                mod.run_dup_into(x, w, checksum, dup_n, variant)

            us = _bench_us(run)
            per_real_col = us / dup_n
            if best_time is None or us < best_time:
                best_time = us
                best_variant = variant
            print(
                f"  dup_n={dup_n:>2d}  variant={variant_names[variant]}  time={us:7.2f} us  "
                f"per_real_col={per_real_col:6.2f} us  checksum_diff={diff:.4f}"
            )
        assert best_variant is not None and best_time is not None
        print(
            f"  best dup_n={dup_n:>2d} variant={variant_names[best_variant]} "
            f"time={best_time:7.2f} us per_real_col={best_time / dup_n:6.2f} us"
        )


def main() -> None:
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Skipping: requires Hopper+, got {torch.cuda.get_device_name()}")
        return

    mod = _load_module()

    print("=" * 72)
    print("Gemma4 Megakernel: Hopper BF16 WGMMA MoE Microbenchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)
    _run_shape(mod, name="moe_fc1", num_rows=FC1_ROWS, hidden_size=HIDDEN_SIZE)
    _run_shape(mod, name="moe_fc2", num_rows=FC2_ROWS, hidden_size=EXPERT_INTERMEDIATE)
    print("=" * 72)


if __name__ == "__main__":
    main()
