"""Amortized CUDA-graph microbench for ``_act_quant_kernel`` occupancy tuning.

Single-call replay measures the ~10us host launch cadence, not GPU time, so we
stack K kernel launches into one captured graph and divide (see the
cudagraph-microbench-amortized-timing note). Sweeps num_warps/num_stages over the
DeepSeek-V4 decode (M=1/2) activation-quant shapes.

Usage:
  PYTHONPATH=<worktree> python bench/bench_act_quant.py            # sweep
  PYTHONPATH=<worktree> python bench/bench_act_quant.py --prod     # bench _safe_act_quant as-shipped
"""

import argparse
import sys

import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _act_quant_kernel,
    _safe_act_quant,
)

BLOCK_SIZE = 128

# (M, K): DeepSeek-V4-Flash TP8 per-rank projection activation shapes.
#  K=7168 is the dominant/largest (o_proj-like, 56 blocks at M=1); the smaller K
#  are the q/kv projection inputs (4-16 blocks) -- even more launch-bound.
DECODE_SHAPES = [(1, 7168), (1, 2048), (1, 1536), (1, 512), (2, 7168), (2, 2048)]
PREFILL_SHAPES = [(256, 7168), (512, 2048)]


def amortized_us(launch_fn, k_per_graph=64, n_replays=300):
    """Per-launch GPU time (us), amortized over a graph of k_per_graph launches."""
    for _ in range(15):
        launch_fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k_per_graph):
            launch_fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # a couple of replay warmups
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_replays):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms * 1000.0 / (n_replays * k_per_graph)


def make_buffers(M, K):
    x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.shape[:-1], K // BLOCK_SIZE, dtype=x.dtype)
    return x, y, s


def bench_config(M, K, num_warps, num_stages, round_scale=False):
    x, y, s = make_buffers(M, K)
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    kw = {"BLOCK_SIZE": BLOCK_SIZE, "ROUND_SCALE": round_scale}
    if num_warps is not None:
        kw["num_warps"] = num_warps
    if num_stages is not None:
        kw["num_stages"] = num_stages

    def launch():
        _act_quant_kernel[grid](x, y, s, **kw)

    return amortized_us(launch)


def bench_prod(M, K):
    x, _, _ = make_buffers(M, K)

    def launch():
        _safe_act_quant(x, BLOCK_SIZE)

    return amortized_us(launch)


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def robust_compare(shapes, configs, rounds=7):
    """Round-robin interleaved measurement to cancel thermal/boost drift.

    Within one round every config is measured back-to-back (≈same clock state);
    the per-config median over `rounds` is robust to slow drift across rounds.
    `configs` is a list of (label, num_warps, num_stages).
    Returns {label: {shape: median_us}}.
    """
    samples = {lbl: {sh: [] for sh in shapes} for (lbl, _, _) in configs}
    for r in range(rounds):
        for M, K in shapes:
            for lbl, nw, ns in configs:
                us = bench_config(M, K, nw, ns)
                samples[lbl][(M, K)].append(us)
    return {lbl: {sh: _median(v) for sh, v in d.items()} for lbl, d in samples.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true", help="bench shipped _safe_act_quant")
    ap.add_argument("--prefill", action="store_true", help="include prefill shapes")
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()

    dev = torch.cuda.get_device_name()
    print(f"# device={dev} triton={triton.__version__} rounds={args.rounds}", flush=True)

    decode = list(DECODE_SHAPES)
    shapes = decode + (PREFILL_SHAPES if args.prefill else [])

    if args.prod:
        print("# _safe_act_quant (shipped launch config)")
        tot = 0.0
        for M, K in shapes:
            us = bench_prod(M, K)
            tot += us
            print(f"M={M:<4} K={K:<6} prod_us={us:.4f}", flush=True)
        print(f"# mean_us={tot / len(shapes):.4f}")
        return 0

    # None == Triton default (num_warps=4). Include explicit 4 as a noise gauge:
    # |median(None) - median(nw=4)| is a lower bound on the true measurement noise.
    configs = [
        ("default", None, None),
        ("nw=1", 1, None),
        ("nw=2", 2, None),
        ("nw=4", 4, None),
        ("nw=8", 8, None),
        ("nw=1,ns=1", 1, 1),
        ("nw=2,ns=1", 2, 1),
    ]
    res = robust_compare(shapes, configs, rounds=args.rounds)

    hdr = (
        "config".ljust(12)
        + "  "
        + "  ".join(f"{M}x{K}".rjust(9) for M, K in shapes)
        + "   dec_mean"
    )
    print(hdr)
    means = {}
    for lbl, _, _ in configs:
        row = [res[lbl][(M, K)] for (M, K) in shapes]
        dec_mean = sum(res[lbl][sh] for sh in decode) / len(decode)
        means[lbl] = dec_mean
        print(
            lbl.ljust(12) + "  " + "  ".join(f"{v:9.3f}" for v in row) + f"   {dec_mean:8.4f}",
            flush=True,
        )

    noise = abs(means["default"] - means["nw=4"]) / means["default"] * 100
    best = min(means, key=means.get)
    base = means["default"]
    print(f"# decode noise floor |default-nw=4| = {noise:.2f}%")
    print(
        f"# best decode = {best} ({means[best]:.4f}us)  vs default {base:.4f}us  "
        f"delta={(means[best] - base) / base * 100:+.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
