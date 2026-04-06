"""Temporary test script for num_stages sweep."""

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
INTERMEDIATE_SIZE = 3712
BLOCK = 4096


@triton.jit
def _relu2_quant_fp8_kernel(
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
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


def run(T, stages):
    x = torch.randn(T, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")
    grid = (triton.cdiv(n, BLOCK),)

    def fn():
        _relu2_quant_fp8_kernel[grid](
            x_flat,
            out,
            scale,
            n,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=stages,
        )

    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


if __name__ == "__main__":
    for stages in [1, 2, 3, 4]:
        d1 = run(1, stages)
        p1k = run(1024, stages)
        print(f"stages={stages}: D1={d1:.2f}us  P1K={p1k:.2f}us")
