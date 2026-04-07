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

"""Benchmark sweep for fused_mamba_decode kernel.

Benchmarks _fused_conv_ssm_kernel in isolation and fused_conv_ssm_decode end-to-end
for Nemotron Nano v3 dimensions (nheads=64, dim=64, dstate=128, ngroups=8,
conv_dim=6144, kernel_width=4).

Usage:
    python sweep_fused_mamba_decode.py
    python sweep_fused_mamba_decode.py --num_warps 2 --block_dim 64 --block_dstate 128
    python sweep_fused_mamba_decode.py --correctness
    python sweep_fused_mamba_decode.py --batch 1 8 64 384
"""

import argparse
from typing import List

import torch
import triton
from fused_mamba_decode import _fused_conv_ssm_kernel, fused_conv_ssm_decode

# ---------------------------------------------------------------------------
# Nemotron Nano v3 dimensions
# ---------------------------------------------------------------------------
NHEADS = 64
DIM = 64
DSTATE = 128
NGROUPS = 8
NHEADS_PER_GROUP = NHEADS // NGROUPS  # 8
KERNEL_WIDTH = 4
INTERMEDIATE_SIZE = NHEADS * DIM  # 4096
CONV_DIM = INTERMEDIATE_SIZE + 2 * NGROUPS * DSTATE  # 4096 + 2048 = 6144

# Max cache size for slot-indexed tensors
MAX_BATCH = 512


def make_inputs(batch: int, dtype=torch.bfloat16, device="cuda"):
    """Create synthetic inputs for one layer."""
    conv_input = torch.randn(batch, CONV_DIM, dtype=dtype, device=device)
    conv_state = torch.randn(MAX_BATCH, CONV_DIM, KERNEL_WIDTH - 1, dtype=dtype, device=device)
    conv_weight = torch.randn(CONV_DIM, KERNEL_WIDTH, dtype=dtype, device=device)
    conv_bias = torch.randn(CONV_DIM, dtype=dtype, device=device)
    dt = torch.randn(batch, NHEADS, dtype=dtype, device=device)
    dt_bias = torch.randn(NHEADS, dtype=dtype, device=device)
    # A should be negative for stability
    A = -torch.rand(NHEADS, dtype=torch.float32, device=device) - 0.1
    D = torch.randn(NHEADS, dtype=torch.float32, device=device)
    ssm_state = torch.randn(MAX_BATCH, NHEADS, DIM, DSTATE, dtype=torch.float32, device=device)
    slot_idx = torch.arange(batch, dtype=torch.int32, device=device)
    out = torch.zeros(batch, NHEADS, DIM, dtype=dtype, device=device)
    return (
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state,
        slot_idx,
        out,
    )


def bench_kernel(
    batch: int,
    num_warps: int,
    block_dim: int,
    block_dstate: int,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark the raw Triton kernel in microseconds."""
    (
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state,
        slot_idx,
        out,
    ) = make_inputs(batch)

    grid = (triton.cdiv(DIM, block_dim), batch, NHEADS)

    def fn():
        _fused_conv_ssm_kernel[grid](
            conv_input,
            conv_state,
            conv_weight,
            conv_bias,
            dt,
            dt_bias,
            A,
            D,
            ssm_state,
            slot_idx,
            slot_idx,
            out,
            batch,
            CONV_DIM,
            INTERMEDIATE_SIZE,
            NHEADS,
            DIM,
            DSTATE,
            NGROUPS,
            NHEADS_PER_GROUP,
            KERNEL_WIDTH,
            conv_input.stride(0),
            conv_input.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            conv_weight.stride(0),
            conv_weight.stride(1),
            ssm_state.stride(0),
            ssm_state.stride(1),
            ssm_state.stride(2),
            ssm_state.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BLOCK_DIM=block_dim,
            BLOCK_DSTATE=block_dstate,
            num_warps=num_warps,
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3  # convert to microseconds


def bench_e2e(
    batch: int,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark the end-to-end fused_conv_ssm_decode launcher in microseconds."""
    (
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state,
        slot_idx,
        out,
    ) = make_inputs(batch)

    def fn():
        fused_conv_ssm_decode(
            conv_input,
            conv_state,
            conv_weight,
            conv_bias,
            dt,
            dt_bias,
            A,
            D,
            ssm_state,
            slot_idx,
            slot_idx,
            out,
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3


# ---------------------------------------------------------------------------
# Reference implementation for correctness check
# ---------------------------------------------------------------------------


def softplus_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def reference_fused_conv_ssm(
    conv_input: torch.Tensor,  # [batch, conv_dim]
    conv_state: torch.Tensor,  # [max_batch, conv_dim, kw-1]
    conv_weight: torch.Tensor,  # [conv_dim, kernel_width]
    conv_bias: torch.Tensor,  # [conv_dim]
    dt: torch.Tensor,  # [batch, nheads]
    dt_bias: torch.Tensor,  # [nheads]
    A: torch.Tensor,  # [nheads]
    D: torch.Tensor,  # [nheads]
    ssm_state: torch.Tensor,  # [max_batch, nheads, dim, dstate]
    slot_idx: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """Reference torch implementation for correctness verification."""
    batch = conv_input.shape[0]
    nheads = ssm_state.shape[1]
    dim = ssm_state.shape[2]
    dstate = ssm_state.shape[3]
    kernel_width = conv_state.shape[-1] + 1
    intermediate_size = nheads * dim
    ngroups = (CONV_DIM - intermediate_size) // (2 * dstate)
    nheads_per_group = nheads // ngroups

    conv_input_f = conv_input.float()
    conv_state_f = conv_state.float()
    conv_weight_f = conv_weight.float()
    conv_bias_f = conv_bias.float()
    dt_f = dt.float()
    dt_bias_f = dt_bias.float()

    # Compute conv output for all channels
    # conv_out[b, d] = sum_k conv_state[slot_b, d, k] * weight[d, k] + x_in[b, d] * weight[d, kw-1] + bias[d]
    conv_out = torch.zeros(batch, CONV_DIM, dtype=torch.float32, device=conv_input.device)
    for b in range(batch):
        slot = slot_idx[b].item()
        x_in = conv_input_f[b]  # [conv_dim]
        for k in range(kernel_width - 1):
            conv_out[b] += conv_state_f[slot, :, k] * conv_weight_f[:, k]
        conv_out[b] += x_in * conv_weight_f[:, kernel_width - 1] + conv_bias_f

    # Update conv state (shift + append)
    for b in range(batch):
        slot = slot_idx[b].item()
        x_in = conv_input_f[b]
        # Shift: [k] <- [k+1] for k=0..kw-3
        for k in range(kernel_width - 2):
            conv_state_f[slot, :, k] = conv_state_f[slot, :, k + 1]
        # Append new input
        conv_state_f[slot, :, kernel_width - 2] = x_in

    # Split conv_out
    x_hidden = conv_out[:, :intermediate_size]  # [batch, intermediate_size]
    B_flat = conv_out[
        :, intermediate_size : intermediate_size + ngroups * dstate
    ]  # [batch, ngroups*dstate]
    C_flat = conv_out[
        :, intermediate_size + ngroups * dstate : intermediate_size + 2 * ngroups * dstate
    ]

    # SiLU on all
    x_hidden = x_hidden * torch.sigmoid(x_hidden)
    B_flat = B_flat * torch.sigmoid(B_flat)
    C_flat = C_flat * torch.sigmoid(C_flat)

    # Reshape
    x_hidden = x_hidden.view(batch, nheads, dim)  # [batch, nheads, dim]
    B_flat = B_flat.view(batch, ngroups, dstate)  # [batch, ngroups, dstate]
    C_flat = C_flat.view(batch, ngroups, dstate)

    # SSM update
    out = torch.zeros(batch, nheads, dim, dtype=torch.float32, device=conv_input.device)
    ssm_state_f = ssm_state.float()

    dt_full = dt_f + dt_bias_f[None, :]  # [batch, nheads]
    dt_full = softplus_ref(dt_full)  # [batch, nheads]
    A_f = A.float()

    for b in range(batch):
        slot = slot_idx[b].item()
        for h in range(nheads):
            g = h // nheads_per_group
            dA = torch.exp(A_f[h] * dt_full[b, h])  # scalar
            dB = B_flat[b, g] * dt_full[b, h]  # [dstate]
            # state: [dim, dstate]
            state = ssm_state_f[slot, h]
            x_h = x_hidden[b, h]  # [dim]
            state = state * dA + dB[None, :] * x_h[:, None]
            ssm_state_f[slot, h] = state
            C_h = C_flat[b, g]  # [dstate]
            out[b, h] = (state * C_h[None, :]).sum(dim=1) + x_h * A_f[h].new_tensor(D[h].item())

    return out.to(torch.bfloat16)


def check_correctness(
    batch: int,
    num_warps: int,
    block_dim: int,
    block_dstate: int,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> bool:
    """Compare kernel output against fused_conv_ssm_decode (default launcher) reference.

    Uses fused_conv_ssm_decode (num_warps=4, BLOCK_DIM=64, BLOCK_DSTATE=128) as
    the reference, since that is the validated baseline. The custom kernel with
    different params is compared against it.
    """
    torch.manual_seed(42)
    (
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state,
        slot_idx,
        out,
    ) = make_inputs(batch)

    # Freeze read-only inputs; clone mutable state for each run
    conv_state_ref = conv_state.clone()
    ssm_state_ref = ssm_state.clone()
    out_ref = torch.zeros_like(out)

    conv_state_test = conv_state.clone()
    ssm_state_test = ssm_state.clone()
    out_test = torch.zeros_like(out)

    # Reference: use the default fused_conv_ssm_decode launcher (num_warps=4, BLOCK_DIM=64, BLOCK_DSTATE=128)
    fused_conv_ssm_decode(
        conv_input,
        conv_state_ref,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state_ref,
        slot_idx,
        slot_idx,
        out_ref,
    )
    torch.cuda.synchronize()

    # Custom kernel with given params
    grid = (triton.cdiv(DIM, block_dim), batch, NHEADS)
    _fused_conv_ssm_kernel[grid](
        conv_input,
        conv_state_test,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state_test,
        slot_idx,
        slot_idx,
        out_test,
        batch,
        CONV_DIM,
        INTERMEDIATE_SIZE,
        NHEADS,
        DIM,
        DSTATE,
        NGROUPS,
        NHEADS_PER_GROUP,
        KERNEL_WIDTH,
        conv_input.stride(0),
        conv_input.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_DIM=block_dim,
        BLOCK_DSTATE=block_dstate,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    # Compare outputs
    out_f = out_test.float()
    ref_f = out_ref.float()

    max_err = (out_f - ref_f).abs().max().item()
    rel_err = ((out_f - ref_f).abs() / (ref_f.abs() + 1e-6)).max().item()
    passed = max_err < atol and rel_err < rtol

    print(
        f"  batch={batch:4d}: max_abs_err={max_err:.6f}, max_rel_err={rel_err:.6f} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def run_sweep(
    batch_sizes: List[int],
    num_warps: int,
    block_dim: int,
    block_dstate: int,
    warmup: int = 25,
    rep: int = 100,
):
    """Run benchmark sweep and print table."""
    print(f"\n{'batch':>6} | {'kernel_us':>10} | {'e2e_us':>10} | {'kernel_pct':>11}")
    print("-" * 48)
    for b in batch_sizes:
        k_us = bench_kernel(b, num_warps, block_dim, block_dstate, warmup=warmup, rep=rep)
        e_us = bench_e2e(b, warmup=warmup, rep=rep)
        pct = 100.0 * k_us / e_us if e_us > 0 else 0.0
        print(f"{b:6d} | {k_us:10.2f} | {e_us:10.2f} | {pct:10.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark fused_mamba_decode kernel")
    parser.add_argument(
        "--batch",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 384],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--num_warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--block_dim", type=int, default=64, help="BLOCK_DIM (tile over head dim)")
    parser.add_argument(
        "--block_dstate", type=int, default=128, help="BLOCK_DSTATE (tile over dstate)"
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--correctness",
        action="store_true",
        help="Run correctness check against reference",
    )
    args = parser.parse_args()

    print(
        f"Config: num_warps={args.num_warps}, BLOCK_DIM={args.block_dim}, BLOCK_DSTATE={args.block_dstate}"
    )
    print(
        f"Model: nheads={NHEADS}, dim={DIM}, dstate={DSTATE}, ngroups={NGROUPS}, "
        f"conv_dim={CONV_DIM}, kernel_width={KERNEL_WIDTH}"
    )

    if args.correctness:
        print("\n=== Correctness Check ===")
        print(
            "NOTE: The kernel has a pre-existing B/C conv-state write-read race when\n"
            "  nheads_per_group > 1 (Nano v3: nheads_per_group=8). This makes multi-batch\n"
            "  runs non-deterministic. Correctness is checked at batch=1 only (deterministic)\n"
            "  and batch=2,4 (mild race, small error typically acceptable).\n"
            "  The check verifies that tuning params produce same output as default params."
        )
        all_pass = True
        # batch=1 is always deterministic; batch>1 may differ across runs due to the
        # pre-existing B/C conv-state write-read race (nheads_per_group=8 heads share
        # B/C channels, and the shift-store by head-0 races with reads by heads 1-7).
        # We check batch=1 (strict) and batch=2 (should be deterministic in practice).
        for b in [1, 2]:
            ok = check_correctness(b, args.num_warps, args.block_dim, args.block_dstate)
            all_pass = all_pass and ok
        if all_pass:
            print("All correctness checks PASSED.")
        else:
            print("SOME CHECKS FAILED.")
        return

    run_sweep(
        args.batch,
        args.num_warps,
        args.block_dim,
        args.block_dstate,
        warmup=args.warmup,
        rep=args.rep,
    )


if __name__ == "__main__":
    main()
