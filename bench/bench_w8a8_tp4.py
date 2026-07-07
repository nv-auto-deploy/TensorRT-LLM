# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4 live-shape microbench for the W8A8 block-FP8 decode kernels (idea_0040).

DeepSeek-V4-Flash at TP4/EP4 dispatches these per-rank decode (M=1) GEMMs into
``_w8a8_block_fp8_matmul_triton`` (fused shapes reflect fuse_gemms_mixed_children +
fuse_finegrained_fp8_gate_up on HEAD; counts are per decode step over 43 layers):

  split-K path (M<=4, K>=4096):
    wqa_wkv   N=1536  K=4096   x43   (fused wq_a[1024] + wkv[512], replicated)
    w1w3      N=1024  K=4096   x43   (fused shared-expert w1+w3, colwise/4)
    wo_a      2 groups x (N=1024 K=4096) into one shared fp32 acc + 1 cast  x43
  full-K autotuned path:
    wqb_idx   N=16384 K=1024   x21   (fused wq_b[8192 colwise] + indexer.wq_b[8192 repl])
    wqb       N=8192  K=1024   x22   (ratio-128/0 layers: wq_b alone)
    wo_b      N=4096  K=2048   x43   (rowwise: K=8192/4)
    w2        N=4096  K=512    x43   (rowwise: K=2048/4)

Timing: CUDA-graph stacked amortized (capture STEPS back-to-back calls, replay,
divide) -- the e2e decode path runs under a cudagraph, and a plain wall loop on a
~2us GEMV measures the ~10us host launch cadence instead of GPU time.

L2 realism: each stacked call reads a DIFFERENT weight copy (cycling >= ~128MB,
above the B200 126MB L2) because in e2e every layer's weight is L2-cold; a single
hot weight buffer would mis-rank configs on this memory-bound GEMV.

Usage:
  python bench/bench_w8a8_tp4.py baseline [--m 1]
  python bench/bench_w8a8_tp4.py sweep-splitk --site wqa_wkv [--m 1]
  python bench/bench_w8a8_tp4.py sweep-fullk --site wqb_idx [--m 1]
  python bench/bench_w8a8_tp4.py rr --site wqa_wkv --cfgs "SK16:BN128:BK128:w4:s3,SK24:BN128:BK128:w4:s3" [--rounds 8]
"""

import argparse
import json

import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _safe_act_quant,
    _splitk_block_n,
    _splitk_num_warps,
    _splitk_split_k,
    _use_splitk_decode,
    _w8a8_block_fp8_matmul_kernel,
    _w8a8_block_fp8_matmul_splitk,
    _w8a8_block_fp8_matmul_triton,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
L2_BYTES = 128 * 1024 * 1024  # cycle a bit above the B200 126MB L2
STEPS = 64  # stacked calls per graph
REPLAYS = 30

# (name, N, K, per-step count). wo_a is the grouped composite site.
SITES = {
    "wqa_wkv": (1536, 4096, 43, "splitk"),
    "w1w3": (1024, 4096, 43, "splitk"),
    "wo_a": (1024, 4096, 43, "grouped"),  # 2 groups/rank, shared acc
    "wqb_idx": (16384, 1024, 21, "fullk"),
    "wqb": (8192, 1024, 22, "fullk"),
    "wo_b": (4096, 2048, 43, "fullk"),
    "w2": (4096, 512, 43, "fullk"),
}


def _quant_weight_block_fp8(w, block_n=128, block_k=128):
    N, Kd = w.shape
    sn, sk = triton.cdiv(N, block_n), triton.cdiv(Kd, block_k)
    scale = torch.empty(sn, sk, dtype=torch.float32, device=w.device)
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    for i in range(sn):
        for j in range(sk):
            blk = w[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k].float()
            s = (blk.abs().amax() / FP8_MAX).clamp(min=1e-12)
            scale[i, j] = s
            w_fp8[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k] = (
                (blk / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return w_fp8, scale


def _n_weight_copies(N, K):
    return max(2, min(64, -(-L2_BYTES // (N * K))))


class Site:
    """Inputs for one live GEMM site: activation + cycling weight copies."""

    def __init__(self, name, M, copies=None):
        N, K, count, kind = SITES[name]
        self.name, self.M, self.N, self.K, self.count, self.kind = name, M, N, K, count, kind
        torch.manual_seed(0)
        self.n_copies = copies if copies else _n_weight_copies(N, K)
        base_w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
        w_fp8, b_s = _quant_weight_block_fp8(base_w)
        self.Bs = b_s
        # distinct allocations so consecutive stacked calls are L2-cold on weights
        self.Ws = [w_fp8.clone() for _ in range(self.n_copies)]
        if kind == "grouped":
            a = torch.randn(M, 2, K, device="cuda", dtype=torch.bfloat16) * 0.1
            qa, sa = _safe_act_quant(a.contiguous(), 128)
            self.qin = qa.reshape(M, 2, K)
            self.sin = sa.reshape(M, 2, K // 128)
            # grouped weight [2*N, K]; scale rows stacked the same way
            gw = torch.randn(2 * N, K, device="cuda", dtype=torch.bfloat16) * 0.1
            gw_fp8, gb_s = _quant_weight_block_fp8(gw)
            self.Bs = gb_s
            gcopies = copies if copies else max(2, min(64, -(-L2_BYTES // (2 * N * K))))
            self.Ws = [gw_fp8.clone() for _ in range(gcopies)]
            self.n_copies = len(self.Ws)
        else:
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
            self.qA, self.As = _safe_act_quant(a.contiguous(), 128)

    def dispatcher_call(self, i):
        """The exact call the model makes on HEAD (module heuristics)."""
        if self.kind == "grouped":
            acc = self.qin.new_zeros((self.M, 2 * self.N), dtype=torch.float32)
            w = self.Ws[i % self.n_copies].view(2, self.N, self.K)
            bs = self.Bs.view(2, self.N // 128, self.K // 128)
            for g in range(2):
                _w8a8_block_fp8_matmul_splitk(
                    self.qin[:, g, :],
                    w[g],
                    self.sin[:, g, :],
                    bs[g],
                    128,
                    128,
                    torch.float32,
                    self.M,
                    self.N,
                    self.K,
                    C_out=acc[:, g * self.N : (g + 1) * self.N],
                )
            return acc.to(torch.bfloat16)
        return _w8a8_block_fp8_matmul_triton(
            self.qA,
            self.Ws[i % self.n_copies],
            self.As,
            self.Bs,
            [128, 128],
            output_dtype=torch.bfloat16,
        )

    def splitk_call(self, i, sk, bn, bk, nw, ns):
        if self.kind == "grouped":
            acc = self.qin.new_zeros((self.M, 2 * self.N), dtype=torch.float32)
            w = self.Ws[i % self.n_copies].view(2, self.N, self.K)
            bs = self.Bs.view(2, self.N // 128, self.K // 128)
            for g in range(2):
                _w8a8_block_fp8_matmul_splitk(
                    self.qin[:, g, :],
                    w[g],
                    self.sin[:, g, :],
                    bs[g],
                    128,
                    128,
                    torch.float32,
                    self.M,
                    self.N,
                    self.K,
                    SPLIT_K=sk,
                    BLOCK_SIZE_N=bn,
                    BLOCK_SIZE_K=bk,
                    num_warps=nw,
                    num_stages=ns,
                    C_out=acc[:, g * self.N : (g + 1) * self.N],
                )
            return acc.to(torch.bfloat16)
        return _w8a8_block_fp8_matmul_splitk(
            self.qA,
            self.Ws[i % self.n_copies],
            self.As,
            self.Bs,
            128,
            128,
            torch.bfloat16,
            self.M,
            self.N,
            self.K,
            SPLIT_K=sk,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=bk,
            num_warps=nw,
            num_stages=ns,
        )

    def fullk_call(self, i, bm, bn, gm, nw, ns):
        A, W, As, Bs = self.qA, self.Ws[i % self.n_copies], self.As, self.Bs
        M, N, K = self.M, self.N, self.K
        C = A.new_empty((M, N), dtype=torch.bfloat16)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        _w8a8_block_fp8_matmul_kernel.fn[grid](
            A,
            W,
            C,
            As,
            Bs,
            M,
            N,
            K,
            128,
            128,
            A.stride(-2),
            A.stride(-1),
            W.stride(1),
            W.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=128,
            GROUP_SIZE_M=gm,
            num_warps=nw,
            num_stages=ns,
        )
        return C


class GraphTimer:
    def __init__(self, fn, steps=STEPS):
        self.steps = steps
        for _ in range(5):
            fn(0)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                fn(i)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            for i in range(steps):
                fn(i)
        torch.cuda.synchronize()
        for _ in range(3):
            self.g.replay()
        torch.cuda.synchronize()

    def time_us(self, replays=REPLAYS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            self.g.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / (self.steps * replays) * 1e3


def round_robin(timers, rounds=8, replays=REPLAYS):
    """Interleave timing across variants to cancel thermal/clock drift."""
    res = {k: [] for k in timers}
    for _ in range(rounds):
        for k, t in timers.items():
            res[k].append(t.time_us(replays))
    return {
        k: (sum(v) / len(v), min(v), (max(v) - min(v)) / (sum(v) / len(v)) * 100)
        for k, v in res.items()
    }


def mode_baseline(args):
    print(f"device={torch.cuda.get_device_name()} M={args.m}")
    total = 0.0
    per_site = {}
    for name in SITES:
        site = Site(name, args.m)
        t = GraphTimer(site.dispatcher_call)
        us = min(t.time_us() for _ in range(5))
        per_site[name] = us
        total += us * site.count
        cfg = ""
        if site.kind == "fullk":
            key = (args.m, site.N, site.K)
            bc = {tuple(k[:3]): v for k, v in _w8a8_block_fp8_matmul_kernel.cache.items()}.get(key)
            cfg = f" autotune={bc}" if bc else ""
        elif site.kind in ("splitk", "grouped"):
            sk = _splitk_split_k(site.N, site.K, args.m)
            bn = _splitk_block_n(site.N, site.K, args.m)
            nw = _splitk_num_warps(site.N, site.K, args.m)
            cfg = f" heuristic=SK{sk}/BN{bn}/w{nw}"
        print(f"  {name:8s} N={site.N:5d} K={site.K:4d} x{site.count:2d}  {us:7.3f} us{cfg}")
        del t, site
        torch.cuda.empty_cache()
    print(f"  step-weighted W8A8 total: {total / 1e3:.4f} ms/step")
    print(json.dumps({"per_site_us": per_site, "weighted_ms_per_step": total / 1e3}))


def _parse_cfg(s):
    d = {}
    for part in s.split(":"):
        if part.startswith("SK"):
            d["sk"] = int(part[2:])
        elif part.startswith("BN"):
            d["bn"] = int(part[2:])
        elif part.startswith("BK"):
            d["bk"] = int(part[2:])
        elif part.startswith("BM"):
            d["bm"] = int(part[2:])
        elif part.startswith("GM"):
            d["gm"] = int(part[2:])
        elif part.startswith("w"):
            d["nw"] = int(part[1:])
        elif part.startswith("s"):
            d["ns"] = int(part[1:])
    return d


def mode_sweep_splitk(args):
    site = Site(args.site, args.m, copies=args.copies)
    assert site.kind in ("splitk", "grouped")
    n_kblocks = site.K // 128
    print(f"site={args.site} M={args.m} N={site.N} K={site.K} ({n_kblocks} K-blocks)")
    default = (
        _splitk_split_k(site.N, site.K, args.m),
        _splitk_block_n(site.N, site.K, args.m),
        128,
        _splitk_num_warps(site.N, site.K, args.m),
        3,
    )
    results = {}
    grid = []
    for sk in [8, 12, 16, 24, 32]:
        for bn in [32, 64, 128, 256]:
            if bn > site.N:
                continue
            grid.append((sk, bn, 128, 4, 3))
    if default not in grid:
        grid.append(default)
    for cfg in grid:
        t = GraphTimer(lambda i, c=cfg: site.splitk_call(i, *c))
        results[cfg] = min(t.time_us() for _ in range(3))
        del t
    top = sorted(results.items(), key=lambda kv: kv[1])[:8]
    print(
        f"  default SK{default[0]}/BN{default[1]}/BK{default[2]}"
        f"/w{default[3]}/s{default[4]} = {results[default]:.3f} us"
    )
    for cfg, us in top:
        print(f"  SK{cfg[0]:2d}/BN{cfg[1]:3d}/BK{cfg[2]}/w{cfg[3]}/s{cfg[4]} = {us:.3f} us")
    # refine: warps/stages/BK around the top-2 (sk, bn)
    fine = {}
    for (sk, bn, _, _, _), _us in top[:2]:
        for bk in [64, 128]:
            for nw in [2, 4, 8]:
                for ns in [2, 3, 4]:
                    cfg = (sk, bn, bk, nw, ns)
                    if cfg in results:
                        fine[cfg] = results[cfg]
                        continue
                    t = GraphTimer(lambda i, c=cfg: site.splitk_call(i, *c))
                    fine[cfg] = min(t.time_us() for _ in range(3))
                    del t
    top_fine = sorted(fine.items(), key=lambda kv: kv[1])[:8]
    print("  -- fine (BK/warps/stages) --")
    for cfg, us in top_fine:
        print(f"  SK{cfg[0]:2d}/BN{cfg[1]:3d}/BK{cfg[2]:3d}/w{cfg[3]}/s{cfg[4]} = {us:.3f} us")


def mode_sweep_fullk(args):
    site = Site(args.site, args.m, copies=args.copies)
    assert site.kind == "fullk"
    print(f"site={args.site} M={args.m} N={site.N} K={site.K} ({site.K // 128} K-blocks)")
    # current dispatcher pick for reference
    t = GraphTimer(site.dispatcher_call)
    disp_us = min(t.time_us() for _ in range(3))
    key = tuple(
        k[:3] for k in _w8a8_block_fp8_matmul_kernel.cache if k[:3] == (args.m, site.N, site.K)
    )
    bc = None
    for k, v in _w8a8_block_fp8_matmul_kernel.cache.items():
        if tuple(k[:3]) == (args.m, site.N, site.K):
            bc = v
    print(f"  dispatcher (autotune {bc}) = {disp_us:.3f} us")
    del t
    results = {}
    for bn in [32, 64, 128, 256]:
        for nw in [4, 8]:
            for ns in [2, 3, 4, 5]:
                if bn > site.N:
                    continue
                cfg = (16, bn, 1, nw, ns)
                t = GraphTimer(lambda i, c=cfg: site.fullk_call(i, *c))
                results[cfg] = min(t.time_us() for _ in range(3))
                del t
    top = sorted(results.items(), key=lambda kv: kv[1])[:10]
    for cfg, us in top:
        print(f"  BM{cfg[0]}/BN{cfg[1]:3d}/GM{cfg[2]}/w{cfg[3]}/s{cfg[4]} = {us:.3f} us")
    print(f"  key={key}")


def mode_rr(args):
    """Round-robin A/B of explicit configs at one site (finalist gate)."""
    site = Site(args.site, args.m, copies=args.copies)
    timers = {}
    for cs in args.cfgs.split(","):
        d = _parse_cfg(cs)
        if site.kind in ("splitk", "grouped"):
            cfg = (d["sk"], d["bn"], d.get("bk", 128), d.get("nw", 4), d.get("ns", 3))
            timers[cs] = GraphTimer(lambda i, c=cfg: site.splitk_call(i, *c))
        else:
            cfg = (d.get("bm", 16), d["bn"], d.get("gm", 1), d.get("nw", 4), d.get("ns", 3))
            timers[cs] = GraphTimer(lambda i, c=cfg: site.fullk_call(i, *c))
    stats = round_robin(timers, rounds=args.rounds)
    for k, (mean, mn, spread) in stats.items():
        print(f"  {k:28s} mean={mean:.3f} us  min={mn:.3f}  spread={spread:.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["baseline", "sweep-splitk", "sweep-fullk", "rr"])
    p.add_argument("--site", default="wqa_wkv")
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--cfgs", default="")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--copies", type=int, default=0)
    args = p.parse_args()
    torch.cuda.init()
    assert _use_splitk_decode(1, 1536, 4096), "gate must route K=4096 decode to split-K"
    {
        "baseline": mode_baseline,
        "sweep-splitk": mode_sweep_splitk,
        "sweep-fullk": mode_sweep_fullk,
        "rr": mode_rr,
    }[args.mode](args)


if __name__ == "__main__":
    main()
