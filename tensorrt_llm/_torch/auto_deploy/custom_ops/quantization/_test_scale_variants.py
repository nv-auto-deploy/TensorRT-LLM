"""Test script for scale_first, inv_scale (in-kernel), tl.fdiv variants."""

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
INTERMEDIATE_SIZE = 3712
BLOCK = 4096


@triton.jit
def _relu2_scale_first(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 37: Load scale before x data (schedule scale load early)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    scale = tl.load(scale_ptr)  # load scale first
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_inv_scale_kernel(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 38: Compute inv_scale = 1.0/scale inside the kernel (avoid division per element)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    inv_scale = 1.0 / scale  # scalar reciprocal, computed once per block
    out_scaled = relu2 * inv_scale  # multiply instead of divide
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_tl_fdiv(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 39: Use tl.fdiv instead of / operator."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    scale = tl.load(scale_ptr)
    out_scaled = tl.fdiv(relu2, scale)
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_square_then_max(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 40: Square first, then max(result, 0) — same as relu2 since x^2 >= 0 always.
    Extra max(0) is a no-op but compiler may handle differently.
    Actually: relu2 = max(x,0)^2 = x^2 if x>0, 0 otherwise.
    This variant: square first then clamp to 0 (mathematically equivalent).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    xsq = xf * xf  # x^2 (could be negative^2 = positive)
    # relu2(x) = x^2 * (x > 0) — use tl.where to zero out negative inputs
    relu2 = tl.where(xf > 0, xsq, tl.zeros_like(xsq))
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


@triton.jit
def _relu2_scale_first_inv(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 41: scale_first + inv_scale multiply combo."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    scale = tl.load(scale_ptr)  # load scale first
    inv_scale = 1.0 / scale  # compute reciprocal while x is loading
    x = tl.load(x_ptr + offs, mask=mask)
    r_bf16 = tl.maximum(x, 0.0)
    r = r_bf16.to(tl.float32)
    relu2 = r * r
    out_scaled = relu2 * inv_scale
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
        ("scale_first", _relu2_scale_first),
        ("inv_scale_kernel", _relu2_inv_scale_kernel),
        ("tl_fdiv", _relu2_tl_fdiv),
        ("square_then_max", _relu2_square_then_max),
        ("scale_first_inv", _relu2_scale_first_inv),
    ]
    for name, kernel in kernels:
        d1 = run(kernel, 1)
        p1k = run(kernel, 1024)
        print(f"{name}: D1={d1:.2f}us  P1K={p1k:.2f}us")
