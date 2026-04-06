"""Test script for eviction policy sweep."""

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
INTERMEDIATE_SIZE = 3712
BLOCK = 4096


@triton.jit
def _relu2_quant_fp8_evict_last_scale(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """evict_last on scale (scalar, reused every launch — keep in L1)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr, eviction_policy="evict_last")
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_quant_fp8_evict_first_x(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """evict_first on x (streaming hint — x not reused)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy="evict_first")
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_quant_fp8_evict_both(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Both: evict_first on x + evict_last on scale."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, eviction_policy="evict_first")
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr, eviction_policy="evict_last")
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


def run(kernel, T):
    x = torch.randn(T, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")
    grid = (triton.cdiv(n, BLOCK),)

    def fn():
        kernel[grid](
            x_flat,
            out,
            scale,
            n,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )

    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


if __name__ == "__main__":
    kernels = [
        ("evict_last_scale", _relu2_quant_fp8_evict_last_scale),
        ("evict_first_x", _relu2_quant_fp8_evict_first_x),
        ("evict_both", _relu2_quant_fp8_evict_both),
    ]
    for name, kernel in kernels:
        d1 = run(kernel, 1)
        p1k = run(kernel, 1024)
        print(f"{name}: D1={d1:.2f}us  P1K={p1k:.2f}us")
