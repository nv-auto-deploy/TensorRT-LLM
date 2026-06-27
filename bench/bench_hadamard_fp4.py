# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbench for ``_hadamard_fp4_kernel`` (idea_0049, kernel_layout axis).

CUDA-graph *stacked* timing (K kernel launches captured into one graph, replayed
G times, divided by K) isolates GPU time from host launch cadence -- mandatory for
this tiny launch-bound kernel (see memory: cudagraph-microbench-amortized-timing).

We time the RAW kernel launch (not the custom-op wrapper) for both the committed
baseline and inline candidate kernels, so the A/B isolates the per-thread / butterfly
LAYOUT change. Candidates are correctness-checked (torch.equal) before timing.
"""

import statistics
import sys

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_hadamard_fp4 import (
    _hadamard_fp4_kernel as BASELINE_KERNEL,
)

_FP4_MAX = 6.0
_FP4_MIN = 6.0 * 2.0**-126

# Decode shapes first (R<=128), then prefill (R>=256). DIM is always 128 in DSV4.
SHAPES = [
    (8, 128),  # indexer-q decode (B=1,S=1,H_local=8) -- the smallest, latency-bound
    (16, 128),
    (32, 128),
    (64, 128),
    (128, 128),  # compressor-rotate at decode (moderate)
    (256, 128),
    (512, 128),  # prefill / large compressor
    (1024, 128),
    (2048, 128),
    (4096, 128),
    (8000, 128),  # prefill indexer-q (B=1,S=1000,H_local=8)
]


def ref_op(x):
    return torch.ops.auto_deploy.deepseek_v4_hadamard_fp4(x, 32)


def make_baseline_launch(x2, out2, R, DIM, BS, num_warps=1, num_stages=1):
    NB = DIM // BS

    def run():
        BASELINE_KERNEL[(R,)](
            x2,
            out2,
            R,
            DIM=DIM,
            BLOCK_SIZE=BS,
            NB=NB,
            INV_SQRT_DIM=float(DIM**-0.5),
            FP4_MAX=_FP4_MAX,
            FP4_MIN=_FP4_MIN,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return run


def time_graph(run, K=64, reps=400, warmup=120):
    """Return per-call GPU time in microseconds via stacked cudagraph timing."""
    for _ in range(25):
        run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(K):
            run()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / (reps * K) * 1000.0  # us/call


def best_of(run, rounds=5, **kw):
    return min(time_graph(run, **kw) for _ in range(rounds))


def bench_launchers(launchers, dtype=torch.bfloat16, rounds=5):
    """launchers: dict name -> (kernel_fn_builder). Each builder(x2,out2,R,DIM,BS)->run."""
    torch.manual_seed(0)
    print(f"{'shape':>14} | " + " | ".join(f"{n:>10}" for n in launchers) + " | best-vs-base%")
    agg = {n: [] for n in launchers}
    for R, DIM in SHAPES:
        x = (torch.randn(R, DIM, device="cuda", dtype=dtype) * 1.3).contiguous()
        ref = ref_op(x)
        x2 = x.reshape(R, DIM)
        results = {}
        for name, builder in launchers.items():
            out = torch.empty_like(x)
            out2 = out.reshape(R, DIM)
            run = builder(x2, out2, R, DIM, 32)
            run()
            torch.cuda.synchronize()
            ok = torch.equal(out, ref)
            if not ok:
                results[name] = float("nan")
                print(
                    f"  !! {name} INCORRECT at R={R} maxdiff="
                    f"{(out.float() - ref.float()).abs().max().item():.2e}"
                )
                continue
            results[name] = best_of(run, rounds=rounds)
        base = results.get("base", float("nan"))
        for n in launchers:
            agg[n].append(results[n])
        cells = " | ".join(f"{results[n]:10.4f}" for n in launchers)
        cand_names = [n for n in launchers if n != "base"]
        deltas = ""
        if cand_names and base == base:
            b = min(results[n] for n in cand_names)
            deltas = f"{(b - base) / base * 100:+.2f}%"
        print(f"{f'({R},{DIM})':>14} | {cells} | {deltas}")
    # geomean-ish summary
    print("--- mean us/call ---")
    for n in launchers:
        vals = [v for v in agg[n] if v == v]
        print(f"  {n:>10}: {statistics.fmean(vals):.4f}")


# ---------------------------------------------------------------------------
# Candidate: row-blocked element-to-thread mapping ([BLOCK_R, DIM] tile / program)
# ---------------------------------------------------------------------------
@triton.jit
def _had_stage_2d(x, BR: tl.constexpr, DIM: tl.constexpr, G: tl.constexpr, W: tl.constexpr):
    a = tl.reshape(x, (BR, G, 2, W))
    sw = tl.flip(a, 2)
    lower = (tl.arange(0, 2) == 0)[None, None, :, None]
    a = tl.where(lower, a + sw, sw - a)
    return tl.reshape(a, (BR, DIM))


@triton.jit
def _hadamard_fp4_kernel_blocked(
    x_ptr,
    out_ptr,
    R,
    BLOCK_R: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NB: tl.constexpr,
    INV_SQRT_DIM: tl.constexpr,
    FP4_MAX: tl.constexpr,
    FP4_MIN: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows < R
    offs = tl.arange(0, DIM)
    ptr = x_ptr + rows[:, None] * DIM + offs[None, :]
    x = tl.load(ptr, mask=mask[:, None], other=0.0).to(tl.float32)

    if DIM >= 2:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 2, 1)
    if DIM >= 4:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 4, 2)
    if DIM >= 8:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 8, 4)
    if DIM >= 16:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 16, 8)
    if DIM >= 32:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 32, 16)
    if DIM >= 64:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 64, 32)
    if DIM >= 128:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 128, 64)
    if DIM >= 256:
        x = _had_stage_2d(x, BLOCK_R, DIM, DIM // 256, 128)
    tl.static_assert(DIM <= 256)
    x = x * INV_SQRT_DIM
    x = x.to(out_ptr.dtype.element_ty).to(tl.float32)

    xb = tl.reshape(x, (BLOCK_R, NB, BLOCK_SIZE))
    amax = tl.max(tl.abs(xb), axis=2, keep_dims=True)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, FP4_MIN) / FP4_MAX)))
    n = tl.minimum(tl.maximum(xb / scale, -FP4_MAX), FP4_MAX)
    an = tl.abs(n)
    q = tl.zeros_like(an)
    q = tl.where(an > 0.25, 0.5, q)
    q = tl.where(an > 0.75, 1.0, q)
    q = tl.where(an > 1.25, 1.5, q)
    q = tl.where(an > 1.75, 2.0, q)
    q = tl.where(an > 2.5, 3.0, q)
    q = tl.where(an > 3.5, 4.0, q)
    q = tl.where(an > 5.0, 6.0, q)
    sign = tl.where(n > 0, 1.0, 0.0) - tl.where(n < 0, 1.0, 0.0)
    res = (q * sign) * scale
    res = tl.reshape(res, (BLOCK_R, DIM))
    tl.store(
        out_ptr + rows[:, None] * DIM + offs[None, :],
        res.to(out_ptr.dtype.element_ty),
        mask=mask[:, None],
    )


def make_blocked_launch(BLOCK_R, num_warps=1, num_stages=1):
    def builder(x2, out2, R, DIM, BS):
        NB = DIM // BS
        grid = (triton.cdiv(R, BLOCK_R),)

        def run():
            _hadamard_fp4_kernel_blocked[grid](
                x2,
                out2,
                R,
                BLOCK_R=BLOCK_R,
                DIM=DIM,
                BLOCK_SIZE=BS,
                NB=NB,
                INV_SQRT_DIM=float(DIM**-0.5),
                FP4_MAX=_FP4_MAX,
                FP4_MIN=_FP4_MIN,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        return run

    return builder


def bench_op_vs_base(dtype=torch.bfloat16, rounds=5):
    """Time the committed custom op (adaptive dispatch) vs the original 1-row kernel."""
    torch.manual_seed(0)
    print(f"{'shape':>14} | {'base':>10} | {'op':>10} | op-vs-base%")
    bvals, ovals = [], []
    for R, DIM in SHAPES:
        x = (torch.randn(R, DIM, device="cuda", dtype=dtype) * 1.3).contiguous()
        ref = ref_op(x)
        # correctness: the op return value must equal the reference (no copy in timing)
        assert torch.equal(ref_op(x), ref)
        x2 = x.reshape(R, DIM)
        out = torch.empty_like(x).reshape(R, DIM)
        base_us = best_of(make_baseline_launch(x2, out, R, DIM, 32), rounds=rounds)
        op_us = best_of(
            lambda: torch.ops.auto_deploy.deepseek_v4_hadamard_fp4(x2, 32), rounds=rounds
        )
        bvals.append(base_us)
        ovals.append(op_us)
        print(
            f"{f'({R},{DIM})':>14} | {base_us:10.4f} | {op_us:10.4f} | {(op_us - base_us) / base_us * 100:+.2f}%"
        )
    print(
        f"--- mean us/call ---  base: {statistics.fmean(bvals):.4f}  op: {statistics.fmean(ovals):.4f} "
        f"({(statistics.fmean(ovals) - statistics.fmean(bvals)) / statistics.fmean(bvals) * 100:+.2f}%)"
    )


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "br"
    if sel == "op":
        bench_op_vs_base()
        sys.exit(0)
    if sel == "br":
        launchers = {
            "base": make_baseline_launch,
            "br1": make_blocked_launch(1),
            "br2": make_blocked_launch(2),
        }
    elif sel == "br_big":
        launchers = {
            "base": make_baseline_launch,
            "br8": make_blocked_launch(8),
            "br16": make_blocked_launch(16),
            "br32": make_blocked_launch(32),
        }
    elif sel == "br24":
        launchers = {
            "base": make_baseline_launch,
            "br2": make_blocked_launch(2),
            "br4": make_blocked_launch(4),
        }
    else:
        launchers = {"base": make_baseline_launch}
    bench_launchers(launchers)
