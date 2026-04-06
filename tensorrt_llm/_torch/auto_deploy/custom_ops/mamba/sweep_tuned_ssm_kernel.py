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

"""Benchmark and correctness script for tuned_ssm_kernel.py.

Usage:
    # Run default benchmark:
    python sweep_tuned_ssm_kernel.py

    # Sweep all param combinations:
    python sweep_tuned_ssm_kernel.py --sweep

    # Custom params:
    python sweep_tuned_ssm_kernel.py --block_m 32 --block_dstate 128 --warps 4 --stages 3

    # Correctness check only:
    python sweep_tuned_ssm_kernel.py --correctness
"""

import argparse
import itertools

import torch
import torch.nn.functional as F
import triton
from tuned_ssm_kernel import _tuned_ssm_update_kernel, tuned_selective_state_update

# ── shape definitions ────────────────────────────────────────────────────────

SHAPES = [
    dict(id="B33", batch=33),
    dict(id="B64", batch=64),
    dict(id="B128", batch=128),
    dict(id="B256", batch=256),
    dict(id="B384", batch=384),
]

NHEADS = 64
DIM = 64
DSTATE = 128
NGROUPS = 8

DT_CLAMP_MIN = 0.001
DT_CLAMP_MAX = 0.1


# ── tensor factory ───────────────────────────────────────────────────────────


def make_tensors(batch, device="cuda"):
    state = torch.randn(batch, NHEADS, DIM, DSTATE, dtype=torch.float32, device=device)
    x = torch.randn(batch, NHEADS, DIM, dtype=torch.bfloat16, device=device)
    dt = torch.randn(batch, NHEADS, DIM, dtype=torch.bfloat16, device=device)
    dt_bias = torch.randn(NHEADS, DIM, dtype=torch.float32, device=device) * 0.1
    A = -torch.rand(NHEADS, DIM, DSTATE, dtype=torch.float32, device=device)
    B = torch.randn(batch, NGROUPS, DSTATE, dtype=torch.bfloat16, device=device)
    C = torch.randn(batch, NGROUPS, DSTATE, dtype=torch.bfloat16, device=device)
    D = torch.randn(NHEADS, DIM, dtype=torch.float32, device=device)
    out = torch.zeros(batch, NHEADS, DIM, dtype=torch.bfloat16, device=device)
    return state, x, dt, dt_bias, A, B, C, D, out


# ── reference implementation ─────────────────────────────────────────────────


def reference_ssm_update(state, x, dt, A, B, C, D, dt_bias, dt_clamp_min=None, dt_clamp_max=None):
    """Pure-PyTorch reference for _tuned_ssm_update_kernel.

    Shapes (after the launcher's squeezes):
      state  : [batch, nheads, dim, dstate]  float32
      x      : [batch, nheads, dim]          bfloat16
      dt     : [batch, nheads, dim]          bfloat16
      dt_bias: [nheads, dim]                 float32
      A      : [nheads, dim, dstate]         float32
      B      : [batch, ngroups, dstate]      bfloat16
      C      : [batch, ngroups, dstate]      bfloat16
      D      : [nheads, dim]                 float32
    """
    batch = x.shape[0]
    nheads = x.shape[1]
    ngroups = B.shape[1]
    ratio = nheads // ngroups

    x_f = x.float()  # [batch, nheads, dim]
    dt_f = F.softplus((dt.float() + dt_bias[None, :, :]))  # [batch, nheads, dim]

    if dt_clamp_min is not None:
        dt_f = dt_f.clamp(dt_clamp_min, dt_clamp_max)

    # A: [nheads, dim, dstate] → broadcast over batch
    # dt_f: [batch, nheads, dim, 1]
    dA = torch.exp(A[None, :, :, :] * dt_f[:, :, :, None])  # [batch, nheads, dim, dstate]

    # B: [batch, ngroups, dstate] → [batch, nheads, dstate]
    B_exp = B.float()[:, :, None, :].expand(
        -1,
        -1,
        ratio,
        -1,
    )  # [batch, ngroups, ratio, dstate]
    B_exp = B_exp.reshape(batch, nheads, DSTATE)  # [batch, nheads, dstate]

    # dB = B_exp * dt_f  → [batch, nheads, dim, dstate]
    dB = B_exp[:, :, None, :] * dt_f[:, :, :, None]  # [batch, nheads, dim, dstate]

    # state update
    state_new = state * dA + dB * x_f[:, :, :, None]  # [batch, nheads, dim, dstate]

    # C: [batch, ngroups, dstate] → [batch, nheads, dstate]
    C_exp = C.float()[:, :, None, :].expand(-1, -1, ratio, -1)
    C_exp = C_exp.reshape(batch, nheads, DSTATE)  # [batch, nheads, dstate]

    # out = sum(state * C, dstate) + x * D
    out = (state_new * C_exp[:, :, None, :]).sum(-1)  # [batch, nheads, dim]
    out = out + x_f * D[None, :, :]

    return out.to(torch.bfloat16), state_new


# ── correctness check ─────────────────────────────────────────────────────────


def check_correctness(batch=64, use_clamp=True, verbose=True):
    """Compare tuned_selective_state_update against reference."""
    state, x, dt, dt_bias, A, B, C, D, out_buf = make_tensors(batch)
    state_ref = state.clone()

    # Run tuned kernel — launcher writes into out_buf (passed via out= kwarg)
    # out_buf is [batch, nheads, dim] bfloat16
    tuned_selective_state_update(
        state.clone(),
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out_buf,
    )
    out_kernel = out_buf  # [batch, nheads, dim]

    # Run reference
    clamp_min = DT_CLAMP_MIN if use_clamp else None
    clamp_max = DT_CLAMP_MAX if use_clamp else None
    out_ref, _ = reference_ssm_update(
        state_ref, x, dt, A, B, C, D, dt_bias, dt_clamp_min=clamp_min, dt_clamp_max=clamp_max
    )

    # Compare
    max_diff = (out_kernel.float() - out_ref.float()).abs().max().item()
    mean_diff = (out_kernel.float() - out_ref.float()).abs().mean().item()

    if verbose:
        clamp_str = f"[{clamp_min}, {clamp_max}]" if use_clamp else "none"
        print(f"Correctness check (batch={batch}, clamp={clamp_str}):")
        print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        if max_diff < 0.2:
            print("  PASS (within bfloat16 precision)")
        else:
            print("  FAIL (large diff — possible clamp mismatch)")

    return max_diff, mean_diff


# ── kernel-level benchmark ────────────────────────────────────────────────────


def bench_kernel(
    batch, block_m=16, block_dstate=128, warps=4, stages=3, warmup=25, rep=100, device="cuda"
):
    """Benchmark _tuned_ssm_update_kernel directly (no launcher overhead)."""
    state, x, dt, dt_bias, A, B, C, D, out_buf = make_tensors(batch, device)

    nheads_ngroups_ratio = NHEADS // NGROUPS

    def grid(META):
        return (triton.cdiv(DIM, META["BLOCK_SIZE_M"]), batch, NHEADS)

    def fn():
        _tuned_ssm_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            out_buf,
            None,  # state_batch_indices
            -1,  # pad_slot_id
            batch,
            NHEADS,
            DIM,
            DSTATE,
            nheads_ngroups_ratio,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            dt_bias.stride(0),
            dt_bias.stride(1),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            D.stride(0),
            D.stride(1),
            out_buf.stride(0),
            out_buf.stride(1),
            out_buf.stride(2),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_DSTATE=block_dstate,
            HAS_STATE_BATCH_INDICES=False,
            DT_CLAMP_MIN=DT_CLAMP_MIN,
            DT_CLAMP_MAX=DT_CLAMP_MAX,
            DSTATE_CONSTEXPR=DSTATE,
            DIM_CONSTEXPR=DIM,
            NHEADS_NGROUPS_RATIO=NHEADS // NGROUPS,
            num_warps=warps,
            num_stages=stages,
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3  # convert ms → us


# ── end-to-end benchmark ──────────────────────────────────────────────────────


def bench_e2e(batch, warmup=25, rep=100, device="cuda"):
    """Benchmark full tuned_selective_state_update launcher."""
    state, x, dt, dt_bias, A, B, C, D, _ = make_tensors(batch, device)

    def fn():
        tuned_selective_state_update(
            state.clone(),
            x,
            dt,
            A,
            B,
            C,
            D,
            dt_bias=dt_bias,
            dt_softplus=True,
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3  # us


# ── table printer ─────────────────────────────────────────────────────────────


def print_table(results, header="Results"):
    print(f"\n{'=' * 70}")
    print(f"  {header}")
    print(f"{'=' * 70}")
    print(
        f"{'ID':>6} {'batch':>6} {'B_M':>5} {'B_DS':>6} {'W':>3} {'S':>3} "
        f"{'kernel_us':>10} {'e2e_us':>10}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['id']:>6} {r['batch']:>6} {r['block_m']:>5} {r['block_dstate']:>6} "
            f"{r['warps']:>3} {r['stages']:>3} "
            f"{r['kernel_us']:>10.1f} {r.get('e2e_us', 0):>10.1f}"
        )
    print("=" * 70)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark tuned_ssm_kernel")
    parser.add_argument("--block_m", type=int, default=16)
    parser.add_argument("--block_dstate", type=int, default=128)
    parser.add_argument("--warps", type=int, default=4)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--no_e2e", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        print("=== Correctness check (no clamp, vs reference without clamp) ===")
        for shape in SHAPES:
            check_correctness(shape["batch"], use_clamp=False)
        print("\n=== Correctness check (with clamp [0.001,0.1], vs reference with clamp) ===")
        for shape in SHAPES:
            check_correctness(shape["batch"], use_clamp=True)
        return

    if args.sweep:
        block_ms = [4, 8, 16, 32, 64]
        block_dstates = [32, 64, 128]
        warpss = [1, 2, 4, 8]
        stagess = [1, 2, 3, 4]
        combos = list(itertools.product(block_ms, block_dstates, warpss, stagess))
        print(f"Sweeping {len(combos)} combos × {len(SHAPES)} shapes …")
        results = []
        for bm, bds, w, s in combos:
            for shape in SHAPES:
                try:
                    ku = bench_kernel(
                        shape["batch"], block_m=bm, block_dstate=bds, warps=w, stages=s
                    )
                    results.append(
                        dict(
                            id=shape["id"],
                            batch=shape["batch"],
                            block_m=bm,
                            block_dstate=bds,
                            warps=w,
                            stages=s,
                            kernel_us=ku,
                            e2e_us=0,
                        )
                    )
                except Exception as e:
                    print(f"  SKIP {shape['id']} bm={bm} bds={bds} w={w} s={s}: {e}")
        print_table(results, "Sweep results")
        return

    # Single config benchmark
    results = []
    for shape in SHAPES:
        ku = bench_kernel(
            shape["batch"],
            block_m=args.block_m,
            block_dstate=args.block_dstate,
            warps=args.warps,
            stages=args.stages,
        )
        e2e = 0 if args.no_e2e else bench_e2e(shape["batch"])
        results.append(
            dict(
                id=shape["id"],
                batch=shape["batch"],
                block_m=args.block_m,
                block_dstate=args.block_dstate,
                warps=args.warps,
                stages=args.stages,
                kernel_us=ku,
                e2e_us=e2e,
            )
        )

    print_table(
        results,
        f"Single config: M={args.block_m} DS={args.block_dstate} W={args.warps} S={args.stages}",
    )


if __name__ == "__main__":
    main()
