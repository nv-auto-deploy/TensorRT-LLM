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

"""Benchmark / sweep script for relu2_quant_fp8 Triton kernel.

Nemotron-3-Nano-30B-A3B shapes:
  - moe_shared_expert_intermediate_size = 3712  (= 128*29, target dimension)
  - hidden_size = 2688
  - For decode concurrency C, each decode step processes T=C tokens per layer.
    n_elements = T * intermediate_size.

Target concurrencies: 1, 4, 16  (decode TPOT focus)
Additional prefill shapes included for reference.

Usage:
  python sweep_relu2_quant_fp8.py                   # benchmark current kernel
  python sweep_relu2_quant_fp8.py --sweep           # full param sweep, writes results.json
  python sweep_relu2_quant_fp8.py --block 128 --warps 4   # test specific config
"""

import argparse
import json
import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Shape matrix: Nemotron-3-Nano-30B, concurrency 1 / 4 / 16 decode + prefill
# ---------------------------------------------------------------------------
# intermediate_size = 3712 (shared expert, moe_shared_expert_intermediate_size)
# For concurrency C decode: T = C decode tokens, n_elements = T * 3712

INTERMEDIATE_SIZE = 3712  # moe_shared_expert_intermediate_size

SHAPES = [
    # (id, description, T, intermediate_size)
    ("D1", "c=1  decode  T=1", 1, INTERMEDIATE_SIZE),
    ("D4", "c=4  decode  T=4", 4, INTERMEDIATE_SIZE),
    ("D16", "c=16 decode  T=16", 16, INTERMEDIATE_SIZE),
    ("D32", "c=32 decode  T=32", 32, INTERMEDIATE_SIZE),
    ("P64", "prefill T=64", 64, INTERMEDIATE_SIZE),
    ("P256", "prefill T=256", 256, INTERMEDIATE_SIZE),
    ("P1K", "prefill T=1024", 1024, INTERMEDIATE_SIZE),
]

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)  # -448.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@triton.jit
def _relu2_quant_fp8_div(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Baseline: relu2 + FP8 quant using division."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = tl.maximum(xf, 0.0)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    out_fp8 = out_clamped.to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_quant_fp8_mul(
    x_ptr,
    out_fp8_ptr,
    inv_scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Iter 1: replace division with multiply-by-reciprocal (inv_scale precomputed on host)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = tl.maximum(xf, 0.0)
    relu2 = r * r
    inv_scale = tl.load(inv_scale_ptr)
    out_scaled = relu2 * inv_scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    out_fp8 = out_clamped.to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def run_div(x: torch.Tensor, scale: torch.Tensor, BLOCK: int, num_warps: int) -> torch.Tensor:
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device=x.device)
    grid = (triton.cdiv(n, BLOCK),)
    _relu2_quant_fp8_div[grid](
        x_flat,
        out,
        scale,
        n,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)


def run_mul(x: torch.Tensor, scale: torch.Tensor, BLOCK: int, num_warps: int) -> torch.Tensor:
    """Uses inv_scale = 1.0/scale passed from host."""
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device=x.device)
    inv_scale = (1.0 / scale).to(torch.float32)
    grid = (triton.cdiv(n, BLOCK),)
    _relu2_quant_fp8_mul[grid](
        x_flat,
        out,
        inv_scale,
        n,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Reference (pure PyTorch) for correctness check
# ---------------------------------------------------------------------------


def relu2_quant_fp8_ref(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    r = torch.relu(xf)
    relu2 = r * r
    out_scaled = relu2 / scale.float()
    out_clamped = out_scaled.clamp(_FP8_MIN, _FP8_MAX)
    return out_clamped.to(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def bench(fn, warmup=25, rep=100):
    """Return latency in microseconds."""
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3  # us


def make_inputs(T, intermediate_size, device="cuda"):
    x = torch.randn(T, intermediate_size, dtype=torch.bfloat16, device=device)
    scale = torch.tensor(0.05, dtype=torch.float32, device=device)
    return x, scale


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def check_correctness(BLOCK, num_warps):
    x, scale = make_inputs(16, INTERMEDIATE_SIZE)
    ref = relu2_quant_fp8_ref(x, scale)
    out_div = run_div(x, scale, BLOCK, num_warps)
    out_mul = run_mul(x, scale, BLOCK, num_warps)
    ref_f = ref.to(torch.float32)
    div_f = out_div.to(torch.float32)
    mul_f = out_mul.to(torch.float32)
    max_diff_div = (ref_f - div_f).abs().max().item()
    max_diff_mul = (ref_f - mul_f).abs().max().item()
    # FP8 e4m3 has limited precision; tolerance = 1 ULP ≈ 0.5 (for values near 448)
    ok_div = max_diff_div < 2.0
    ok_mul = max_diff_mul < 2.0
    return ok_div, ok_mul, max_diff_div, max_diff_mul


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(BLOCK: int, num_warps: int, verbose: bool = True):
    header = f"BLOCK={BLOCK:<5} num_warps={num_warps}"
    results = {}

    rows = []
    for sid, desc, T, n_size in SHAPES:
        x, scale = make_inputs(T, n_size)
        n = T * n_size

        t_div = bench(lambda: run_div(x, scale, BLOCK, num_warps))
        t_mul = bench(lambda: run_mul(x, scale, BLOCK, num_warps))

        n_blocks = math.ceil(n / BLOCK)
        results[sid] = {"T": T, "n": n, "n_blocks": n_blocks, "div_us": t_div, "mul_us": t_mul}
        rows.append((sid, desc, T, n, n_blocks, t_div, t_mul))

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Config: {header}")
        print(f"{'=' * 80}")
        cols = f"{'ID':<5} {'Description':<22} {'T':>4} {'n_elem':>7}"
        cols += f" {'n_blk':>6} {'div(us)':>9} {'mul(us)':>9} {'div/mul':>8}"
        print(cols)
        print(f"{'-' * 80}")
        for sid, desc, T, n, n_blocks, t_div, t_mul in rows:
            ratio = t_div / t_mul if t_mul > 0 else float("nan")
            print(
                f"{sid:<5} {desc:<22} {T:>4} {n:>7} {n_blocks:>6} {t_div:>9.2f} {t_mul:>9.2f} {ratio:>8.3f}"
            )

    return results


def run_sweep():
    """Full parameter sweep over BLOCK sizes and num_warps."""
    block_sizes = [64, 128, 256, 512, 1024, 2048]
    warp_configs = [2, 4, 8, 16]

    all_results = {}
    total = len(block_sizes) * len(warp_configs)
    idx = 0

    for BLOCK in block_sizes:
        for num_warps in warp_configs:
            idx += 1
            key = f"BLOCK{BLOCK}_W{num_warps}"
            print(f"\n[{idx}/{total}] {key}", flush=True)
            ok_div, ok_mul, diff_div, diff_mul = check_correctness(BLOCK, num_warps)
            div_s = "PASS" if ok_div else "FAIL"
            mul_s = "PASS" if ok_mul else "FAIL"
            print(
                f"  Correctness: div={div_s}(diff={diff_div:.3f})  mul={mul_s}(diff={diff_mul:.3f})"
            )
            res = run_benchmark(BLOCK, num_warps, verbose=True)
            all_results[key] = {
                "BLOCK": BLOCK,
                "num_warps": num_warps,
                "ok_div": ok_div,
                "ok_mul": ok_mul,
                "shapes": res,
            }

    # Write results
    out_path = __file__.replace(".py", "_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print best per shape
    print_best(all_results)
    return all_results


def print_best(all_results):
    """For each shape and kernel variant, print the best config."""
    shape_ids = [s[0] for s in SHAPES]
    variants = ["div_us", "mul_us"]

    print("\n" + "=" * 80)
    print("BEST CONFIGS PER SHAPE")
    print("=" * 80)

    for sid in shape_ids:
        print(f"\nShape {sid}:")
        for var in variants:
            best_key = None
            best_us = float("inf")
            for key, rec in all_results.items():
                if not rec.get("ok_div" if var == "div_us" else "ok_mul"):
                    continue
                us = rec["shapes"].get(sid, {}).get(var, float("inf"))
                if us < best_us:
                    best_us = us
                    best_key = key
            if best_key:
                rec = all_results[best_key]
                print(
                    f"  {var:<10}: BLOCK={rec['BLOCK']:<5} warps={rec['num_warps']} → {best_us:.2f} us  ({best_key})"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark relu2_quant_fp8 Triton kernel")
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--block", type=int, default=512, help="BLOCK size")
    parser.add_argument("--warps", type=int, default=4, help="num_warps")
    args = parser.parse_args()

    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print(f"intermediate_size (shared expert): {INTERMEDIATE_SIZE}")
    print("  3712 = 128 × 29 (exact multiple of 128)")
    print(f"  3712 mod 512 = {3712 % 512} → BLOCK=512 wastes last block")

    # Correctness check first
    print("\nChecking correctness (BLOCK=512, warps=4)...")
    ok_div, ok_mul, d1, d2 = check_correctness(512, 4)
    print(f"  div: {'PASS' if ok_div else 'FAIL'} (max_diff={d1:.4f})")
    print(f"  mul: {'PASS' if ok_mul else 'FAIL'} (max_diff={d2:.4f})")

    if args.sweep:
        run_sweep()
    else:
        run_benchmark(args.block, args.warps, verbose=True)


if __name__ == "__main__":
    main()
