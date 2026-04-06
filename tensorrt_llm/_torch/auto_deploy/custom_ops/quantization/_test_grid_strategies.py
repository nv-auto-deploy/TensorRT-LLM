"""Test script for grid strategy variants."""

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
INTERMEDIATE_SIZE = 3712
BLOCK = 4096


@triton.jit
def _relu2_stride_loop_x2(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 42: Grid-stride loop — each program processes 2 tiles (loop unroll 2).

    Grid size halved; each program does 2 BLOCK-sized tiles.
    Reduces grid launch overhead for large shapes.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    scale = tl.load(scale_ptr)

    # First tile
    offs0 = pid * BLOCK + tl.arange(0, BLOCK)
    mask0 = offs0 < n_elements
    x0 = tl.load(x_ptr + offs0, mask=mask0)
    r0 = tl.maximum(x0, 0.0).to(tl.float32)
    out0 = tl.clamp(r0 * r0 / scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs0, out0, mask=mask0)

    # Second tile (stride by num_programs)
    offs1 = (pid + num_programs) * BLOCK + tl.arange(0, BLOCK)
    mask1 = offs1 < n_elements
    x1 = tl.load(x_ptr + offs1, mask=mask1)
    r1 = tl.maximum(x1, 0.0).to(tl.float32)
    out1 = tl.clamp(r1 * r1 / scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs1, out1, mask=mask1)


@triton.jit
def _relu2_two_tiles_per_program(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 43: Process 2 consecutive tiles per program (loop unroll 2, contiguous tiles).

    Each program handles tiles at pid*2 and pid*2+1.
    Grid size = ceil(n / (2*BLOCK)).
    """
    pid = tl.program_id(0)
    scale = tl.load(scale_ptr)

    # First tile
    offs0 = (pid * 2) * BLOCK + tl.arange(0, BLOCK)
    mask0 = offs0 < n_elements
    x0 = tl.load(x_ptr + offs0, mask=mask0)
    r0 = tl.maximum(x0, 0.0).to(tl.float32)
    out0 = tl.clamp(r0 * r0 / scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs0, out0, mask=mask0)

    # Second tile
    offs1 = (pid * 2 + 1) * BLOCK + tl.arange(0, BLOCK)
    mask1 = offs1 < n_elements
    x1 = tl.load(x_ptr + offs1, mask=mask1)
    r1 = tl.maximum(x1, 0.0).to(tl.float32)
    out1 = tl.clamp(r1 * r1 / scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + offs1, out1, mask=mask1)


@triton.jit
def _relu2_persistent(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    n_tiles,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """iter 44: Persistent kernel — each SM processes tiles until all work is done.

    Grid = SM count; each program loops over all assigned tiles.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    scale = tl.load(scale_ptr)

    tile_id = pid
    while tile_id < n_tiles:
        offs = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask)
        r = tl.maximum(x, 0.0).to(tl.float32)
        out = tl.clamp(r * r / scale, FP8_MIN, FP8_MAX).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + offs, out, mask=mask)
        tile_id += num_programs


def run_stride_loop(T):
    x = torch.randn(T, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")
    n_tiles = triton.cdiv(n, BLOCK)
    # Half the grid since each program does 2 tiles
    grid_size = max(1, triton.cdiv(n_tiles, 2))
    grid = (grid_size,)

    def fn():
        _relu2_stride_loop_x2[grid](
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


def run_two_tiles(T):
    x = torch.randn(T, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")
    grid_size = max(1, triton.cdiv(n, BLOCK * 2))
    grid = (grid_size,)

    def fn():
        _relu2_two_tiles_per_program[grid](
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


def run_persistent(T, sm_count=132):
    """H100 has 132 SMs."""
    x = torch.randn(T, INTERMEDIATE_SIZE, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor(0.05, dtype=torch.float32, device="cuda")
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out = torch.empty(n, dtype=torch.float8_e4m3fn, device="cuda")
    n_tiles = triton.cdiv(n, BLOCK)
    grid = (min(sm_count, n_tiles),)

    def fn():
        _relu2_persistent[grid](
            x_flat,
            out,
            scale,
            n,
            n_tiles,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )

    return triton.testing.do_bench(fn, warmup=25, rep=100) * 1e3


if __name__ == "__main__":
    for T in [1, 1024]:
        d = run_stride_loop(T)
        print(f"stride_loop_x2   T={T:>4}: {d:.2f}us")
    for T in [1, 1024]:
        d = run_two_tiles(T)
        print(f"two_tiles_per_prog T={T:>4}: {d:.2f}us")
    for T in [1, 1024]:
        d = run_persistent(T)
        print(f"persistent        T={T:>4}: {d:.2f}us")
