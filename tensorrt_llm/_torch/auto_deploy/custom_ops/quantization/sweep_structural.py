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

"""Structural kernel variant benchmark for relu2_quant_fp8.

Tests structural alternatives beyond parameter tuning (iters 4-8):
  iter4: num_stages=2 (software pipelining)
  iter5: bf16 relu first, then fp32 upcast + square (lower register pressure)
  iter6: tl.clamp instead of manual max/min
  iter7: explicit int32 vectorized load (pack 2 bf16 as uint32)
  iter8: combined best structural changes

Usage:
  python sweep_structural.py
"""

import torch
import triton
import triton.language as tl

INTERMEDIATE_SIZE = 3712
SHAPES = [
    ("D1", "c=1  decode  T=1", 1, INTERMEDIATE_SIZE),
    ("D4", "c=4  decode  T=4", 4, INTERMEDIATE_SIZE),
    ("D16", "c=16 decode  T=16", 16, INTERMEDIATE_SIZE),
    ("D32", "c=32 decode  T=32", 32, INTERMEDIATE_SIZE),
    ("P256", "prefill T=256", 256, INTERMEDIATE_SIZE),
    ("P1K", "prefill T=1024", 1024, INTERMEDIATE_SIZE),
]

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)  # -448.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0

# Best config from param sweep (iter 3)
BEST_BLOCK = 1024
BEST_WARPS = 4


# ---------------------------------------------------------------------------
# Variant 0: current baseline (BLOCK=1024, warps=4, no pipelining)
# ---------------------------------------------------------------------------
@triton.jit
def _v0_baseline(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
    tl.store(out_fp8_ptr + offs, out_clamped.to(tl.float8e4nv), mask=mask)


# ---------------------------------------------------------------------------
# Variant 1: num_stages=2 (software pipelining hint)
# ---------------------------------------------------------------------------
@triton.jit
def _v1_stages2(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
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
    tl.store(out_fp8_ptr + offs, out_clamped.to(tl.float8e4nv), mask=mask)


# ---------------------------------------------------------------------------
# Variant 2: relu in bf16 space, then upcast for square+quant
# Avoids upcasting all elements to fp32 before the relu check.
# ---------------------------------------------------------------------------
@triton.jit
def _v2_bf16_relu(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # relu in bf16 (cheaper - narrower type, avoids early upcast)
    r_bf16 = tl.maximum(x, 0.0)
    # upcast to fp32 for squaring (bf16 overflow risk: max^2 = 65504^2 ≈ 4e9)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    tl.store(out_fp8_ptr + offs, out_clamped.to(tl.float8e4nv), mask=mask)


# ---------------------------------------------------------------------------
# Variant 3: tl.clamp instead of manual max/min
# ---------------------------------------------------------------------------
@triton.jit
def _v3_tl_clamp(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = tl.maximum(xf, 0.0)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    # tl.clamp (may compile to a single instruction on some targets)
    out_clamped = tl.clamp(out_scaled, FP8_MIN, FP8_MAX)
    tl.store(out_fp8_ptr + offs, out_clamped.to(tl.float8e4nv), mask=mask)


# ---------------------------------------------------------------------------
# Variant 4: int32 vectorized load (explicit 2×bf16 packed load)
# Each thread loads 2 bf16 values as a single uint32, then unpacks.
# BLOCK must be even. We process BLOCK/2 iterations of 2 elements each.
# Note: Triton already coalesces bf16 loads, but explicit int32 load
# forces the compiler to emit 4-byte loads (vs possible 2-byte).
# ---------------------------------------------------------------------------
@triton.jit
def _v4_int32_load(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Load 2 bf16 elements as one int32 per thread, unpack, process."""
    pid = tl.program_id(0)
    # Each program handles BLOCK bf16 elements = BLOCK/2 int32 loads
    HALF_BLOCK: tl.constexpr = BLOCK // 2
    base = pid * BLOCK
    # Reinterpret x_ptr as int32 pointer; each int32 holds 2 consecutive bf16
    x_i32_ptr = x_ptr.to(tl.pointer_type(tl.int32))
    offs_i32 = base // 2 + tl.arange(0, HALF_BLOCK)
    mask_i32 = offs_i32 * 2 < n_elements
    packed = tl.load(x_i32_ptr + offs_i32, mask=mask_i32, other=0)
    # Unpack: low 16 bits = first bf16, high 16 bits = second bf16
    lo = (packed & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    hi = ((packed >> 16) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    scale = tl.load(scale_ptr)

    def process(v):
        vf = v.to(tl.float32)
        r = tl.maximum(vf, 0.0)
        relu2 = r * r
        out_s = relu2 / scale
        out_c = tl.maximum(tl.minimum(out_s, FP8_MAX), FP8_MIN)
        return out_c.to(tl.float8e4nv)

    lo_fp8 = process(lo)
    hi_fp8 = process(hi)

    # Store: write to interleaved positions
    out_fp8_ptr_cast = out_fp8_ptr.to(tl.pointer_type(tl.float8e4nv))
    offs_lo = base + tl.arange(0, HALF_BLOCK) * 2
    offs_hi = offs_lo + 1
    mask_lo = offs_lo < n_elements
    mask_hi = offs_hi < n_elements
    tl.store(out_fp8_ptr_cast + offs_lo, lo_fp8, mask=mask_lo)
    tl.store(out_fp8_ptr_cast + offs_hi, hi_fp8, mask=mask_hi)


# ---------------------------------------------------------------------------
# Variant 5: combined bf16_relu + tl.clamp (may save 1-2 ops)
# ---------------------------------------------------------------------------
@triton.jit
def _v5_combined(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_clamped = tl.clamp(out_scaled, FP8_MIN, FP8_MAX)
    tl.store(out_fp8_ptr + offs, out_clamped.to(tl.float8e4nv), mask=mask)


# ---------------------------------------------------------------------------
# Variant 6: scale loaded before data (reorder to help latency hiding)
# ---------------------------------------------------------------------------
@triton.jit
def _v6_scale_first(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Load scale before x to allow latency hiding for scalar load."""
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)  # load scalar first
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = tl.maximum(xf, 0.0)
    relu2 = r * r
    out_scaled = relu2 / scale
    out_clamped = tl.maximum(tl.minimum(out_scaled, FP8_MAX), FP8_MIN)
    tl.store(out_fp8_ptr + offs, out_clamped.to(tl.float8e4nv), mask=mask)


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------
_FP8_CONSTS = dict(FP8_MIN=_FP8_MIN, FP8_MAX=_FP8_MAX)


def _launch(kernel, x, scale, BLOCK, num_warps, num_stages=1):
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device=x.device)
    grid = (triton.cdiv(n, BLOCK),)
    kernel[grid](
        x_flat,
        out,
        scale,
        n,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
        **_FP8_CONSTS,
    )
    return out.reshape(x.shape)


def launch_v4(x, scale, BLOCK, num_warps):
    """int32 vectorized variant has a different signature."""
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device=x.device)
    HALF_BLOCK = BLOCK // 2
    grid = (triton.cdiv(n, BLOCK),)
    _v4_int32_load[grid](
        x_flat, out, scale, n, BLOCK=HALF_BLOCK, num_warps=num_warps, **_FP8_CONSTS
    )
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------
def ref(x, scale):
    xf = x.float()
    r = torch.relu(xf)
    relu2 = r * r
    out_s = relu2 / scale.float()
    return out_s.clamp(_FP8_MIN, _FP8_MAX).to(torch.float8_e4m3fn)


def check(out, x, scale, name):
    r = ref(x, scale)
    diff = (out.float() - r.float()).abs().max().item()
    ok = diff < 2.0
    status = "PASS" if ok else f"FAIL(diff={diff:.3f})"
    print(f"  {name}: {status}")
    return ok


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def bench(fn, warmup=25, rep=100):
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep) * 1e3  # us


VARIANTS = [
    ("v0_baseline", lambda x, s: _launch(_v0_baseline, x, s, BEST_BLOCK, BEST_WARPS)),
    ("v1_stages2", lambda x, s: _launch(_v1_stages2, x, s, BEST_BLOCK, BEST_WARPS, num_stages=2)),
    ("v2_bf16relu", lambda x, s: _launch(_v2_bf16_relu, x, s, BEST_BLOCK, BEST_WARPS)),
    ("v3_tl_clamp", lambda x, s: _launch(_v3_tl_clamp, x, s, BEST_BLOCK, BEST_WARPS)),
    # v4 int32 load — may fail on some shapes, caught below
    ("v5_combined", lambda x, s: _launch(_v5_combined, x, s, BEST_BLOCK, BEST_WARPS)),
    ("v6_scale_first", lambda x, s: _launch(_v6_scale_first, x, s, BEST_BLOCK, BEST_WARPS)),
]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print(f"Config: BLOCK={BEST_BLOCK}, num_warps={BEST_WARPS}")

    # Correctness
    print("\n=== Correctness (T=16, n=3712) ===")
    x16, s = (
        torch.randn(16, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda"),
        torch.tensor(0.05, dtype=torch.float32, device="cuda"),
    )
    for name, fn in VARIANTS:
        check(fn(x16, s), x16, s, name)
    # v4 separately
    try:
        out4 = launch_v4(x16, s, BEST_BLOCK, BEST_WARPS)
        check(out4, x16, s, "v4_int32")
        v4_ok = True
    except Exception as e:
        print(f"  v4_int32: ERROR ({e})")
        v4_ok = False

    # Benchmark
    print("\n=== Latency (µs) ===")
    headers = ["Shape"] + [v[0] for v in VARIANTS]
    if v4_ok:
        headers.append("v4_int32")
    col_w = 14
    print("  ".join(f"{h:<{col_w}}" for h in headers))
    print("-" * (col_w * len(headers) + 2 * (len(headers) - 1)))

    all_results = {}
    for sid, desc, T, n_size in SHAPES:
        x, s = (
            torch.randn(T, n_size, dtype=torch.bfloat16, device="cuda"),
            torch.tensor(0.05, dtype=torch.float32, device="cuda"),
        )
        row = {sid: {}}
        timings = [f"{sid}({T}×{n_size})"]
        for name, fn in VARIANTS:
            t = bench(lambda fn=fn, x=x, s=s: fn(x, s))
            timings.append(f"{t:.2f}")
            row[sid][name] = t
        if v4_ok:
            t4 = bench(lambda x=x, s=s: launch_v4(x, s, BEST_BLOCK, BEST_WARPS))
            timings.append(f"{t4:.2f}")
            row[sid]["v4_int32"] = t4
        print("  ".join(f"{v:<{col_w}}" for v in timings))
        all_results.update(row)

    # Summary: best variant per shape
    print("\n=== Best variant per shape ===")
    for sid, timings in all_results.items():
        best = min(timings, key=timings.get)
        best_v0 = timings["v0_baseline"]
        best_t = timings[best]
        delta = (best_t - best_v0) / best_v0 * 100
        print(f"  {sid}: best={best} {best_t:.2f}µs  (v0={best_v0:.2f}µs  delta={delta:+.1f}%)")


if __name__ == "__main__":
    main()
