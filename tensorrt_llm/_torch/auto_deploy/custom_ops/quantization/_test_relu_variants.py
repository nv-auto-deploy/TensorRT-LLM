"""Test script for alternative relu implementations."""

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
INTERMEDIATE_SIZE = 3712
BLOCK = 4096


@triton.jit
def _relu2_tl_where(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 33: tl.where(x > 0, x, 0.0) instead of tl.maximum."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.where(x > 0, x, tl.zeros_like(x))
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_mask_multiply(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 34: relu as mask multiply: x * (x > 0).to(float32)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = xf * (xf > 0).to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_abs_trick(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 35: abs trick: relu(x) = (x + abs(x)) * 0.5."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    r = (xf + tl.abs(xf)) * 0.5
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_where_bf16_output(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 36: tl.where with explicit bf16 output before upcast."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    zero = tl.zeros_like(x).to(tl.bfloat16)
    r_bf16 = tl.where(x > zero, x, zero)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
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
        ("tl_where", _relu2_tl_where),
        ("mask_multiply", _relu2_mask_multiply),
        ("abs_trick", _relu2_abs_trick),
        ("where_bf16_output", _relu2_where_bf16_output),
    ]
    for name, kernel in kernels:
        d1 = run(kernel, 1)
        p1k = run(kernel, 1024)
        print(f"{name}: D1={d1:.2f}us  P1K={p1k:.2f}us")
