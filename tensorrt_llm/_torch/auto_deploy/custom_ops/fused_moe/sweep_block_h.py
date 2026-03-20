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

"""Sweep BLOCK_H for kernel 2 and BLOCK_I for kernel 1 with various grid strategies.

Also tests a coalesced-load variant of kernel 1 (load contiguous, then deinterleave).
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _this_dir = str(Path(__file__).resolve().parent)
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    import torch
    import triton
    import triton.language as tl
    from triton_moe_dense_mlp import _weighted_expert_sum_kernel

    SHAPES = [
        ("A1", 128, 1, 2880, 2880, "120B T=1"),
        ("A2", 128, 8, 2880, 2880, "120B T=8"),
        ("A3", 128, 32, 2880, 2880, "120B T=32"),
        ("A4", 128, 128, 2880, 2880, "120B T=128"),
        ("A5", 128, 512, 2880, 2880, "120B T=512"),
        ("B1", 32, 1, 2880, 2880, "20B T=1"),
        ("B2", 32, 32, 2880, 2880, "20B T=32"),
        ("B3", 32, 512, 2880, 2880, "20B T=512"),
    ]
    DTYPE = torch.bfloat16
    WU, REP = 25, 100

    # =========================================================================
    # K2 BLOCK_H sweep
    # =========================================================================
    print("=" * 80)
    print("KERNEL 2: BLOCK_H sweep (2D grid, w=16, s=2)")
    print("=" * 80)

    # tl.arange requires power-of-2 ranges; 1440/2880 are excluded
    BLOCK_H_OPTIONS = [256, 512, 1024, 2048, 4096]
    k2_best = {}

    for sid, E, T, H, inter, desc in SHAPES:
        print(f"\n--- {sid}: E={E}, T={T} ({desc}) ---")
        results = []
        for bh in BLOCK_H_OPTIONS:
            nhb = triton.cdiv(H, bh)
            eo = torch.randn(E, T, H, device="cuda", dtype=DTYPE).contiguous()
            rw = torch.randn(T, E, device="cuda", dtype=DTYPE).softmax(dim=-1)
            out = torch.empty(T, H, device="cuda", dtype=DTYPE)

            def _run(_eo=eo, _rw=rw, _out=out, _bh=bh, _nhb=nhb):
                _weighted_expert_sum_kernel[(T, _nhb)](
                    _eo,
                    _rw,
                    _out,
                    stride_expert_out_e=_eo.stride(0),
                    stride_expert_out_t=_eo.stride(1),
                    stride_expert_out_h=_eo.stride(2),
                    stride_routing_t=_rw.stride(0),
                    stride_routing_e=_rw.stride(1),
                    stride_out_t=_out.stride(0),
                    stride_out_h=_out.stride(1),
                    num_experts=E,
                    H_SIZE=H,
                    BLOCK_H=_bh,
                    num_warps=16,
                    num_stages=2,
                )

            us = triton.testing.do_bench(_run, warmup=WU, rep=REP) * 1000
            results.append((bh, nhb, us))

        results.sort(key=lambda x: x[2])
        for rank, (bh, nhb, us) in enumerate(results):
            marker = " <-- BEST" if rank == 0 else ""
            print(f"  BLOCK_H={bh:>5} ({nhb:>2} blks): {us:>8.1f} us{marker}")
        k2_best[sid] = results[0]

    # =========================================================================
    # K1: Coalesced-load variant (load contiguous 2I, then split in registers)
    # =========================================================================
    print()
    print("=" * 80)
    print("KERNEL 1: coalesced-load variant vs current stride-2 variant")
    print("=" * 80)

    @triton.jit
    def _fused_glu_activation_coalesced(
        gate_up_ptr,
        output_ptr,
        stride_gate_up_row,
        stride_out_row,
        alpha_val,
        limit_val,
        I_SIZE: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """Coalesced-load variant: load gate and up halves separately (each contiguous), then apply GLU."""
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_I)
        mask = col_offsets < I_SIZE

        # The weight matrix is stored as [gate | up] (two contiguous halves of size I_SIZE each)
        # Load gate half: offsets 0..I_SIZE-1, load up half: offsets I_SIZE..2*I_SIZE-1
        gate_up_row_ptr = gate_up_ptr + row_idx * stride_gate_up_row
        gate_vals = tl.load(gate_up_row_ptr + col_offsets, mask=mask, other=0.0)
        up_vals = tl.load(gate_up_row_ptr + I_SIZE + col_offsets, mask=mask, other=0.0)

        # Upcast + clamp + GLU
        gate_f = gate_vals.to(tl.float32)
        up_f = up_vals.to(tl.float32)
        gate_f = tl.minimum(gate_f, limit_val)
        up_f = tl.maximum(tl.minimum(up_f, limit_val), -limit_val)
        glu = gate_f * tl.sigmoid(gate_f * alpha_val)
        result = (up_f + 1.0) * glu

        out_row_ptr = output_ptr + row_idx * stride_out_row
        tl.store(out_row_ptr + col_offsets, result.to(gate_vals.dtype), mask=mask)

    @triton.jit
    def _fused_glu_activation_stride2(
        gate_up_ptr,
        output_ptr,
        stride_gate_up_row,
        stride_out_row,
        alpha_val,
        limit_val,
        I_SIZE: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """Original stride-2 variant (current implementation)."""
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_I)
        mask = col_offsets < I_SIZE
        gate_offsets = col_offsets * 2
        up_offsets = col_offsets * 2 + 1
        gate_up_row_ptr = gate_up_ptr + row_idx * stride_gate_up_row
        gate_vals = tl.load(gate_up_row_ptr + gate_offsets, mask=mask, other=0.0)
        up_vals = tl.load(gate_up_row_ptr + up_offsets, mask=mask, other=0.0)
        gate_f = gate_vals.to(tl.float32)
        up_f = up_vals.to(tl.float32)
        gate_f = tl.minimum(gate_f, limit_val)
        up_f = tl.maximum(tl.minimum(up_f, limit_val), -limit_val)
        glu = gate_f * tl.sigmoid(gate_f * alpha_val)
        result = (up_f + 1.0) * glu
        out_row_ptr = output_ptr + row_idx * stride_out_row
        tl.store(out_row_ptr + col_offsets, result.to(gate_vals.dtype), mask=mask)

    for sid, E, T, H, inter, desc in SHAPES:
        total_rows = E * T
        nw = 16 if total_rows <= 128 else 4
        block_i = triton.next_power_of_2(inter)
        gu = torch.randn(total_rows, 2 * inter, device="cuda", dtype=DTYPE)
        ao = torch.empty(total_rows, inter, device="cuda", dtype=DTYPE)

        def _run_stride2(_gu=gu, _ao=ao):
            _fused_glu_activation_stride2[(total_rows,)](
                _gu,
                _ao,
                _gu.stride(0),
                _ao.stride(0),
                1.702,
                7.0,
                I_SIZE=inter,
                BLOCK_I=block_i,
                num_warps=nw,
                num_stages=2,
            )

        def _run_coalesced(_gu=gu, _ao=ao):
            _fused_glu_activation_coalesced[(total_rows,)](
                _gu,
                _ao,
                _gu.stride(0),
                _ao.stride(0),
                1.702,
                7.0,
                I_SIZE=inter,
                BLOCK_I=block_i,
                num_warps=nw,
                num_stages=2,
            )

        us_s2 = triton.testing.do_bench(_run_stride2, warmup=WU, rep=REP) * 1000
        us_co = triton.testing.do_bench(_run_coalesced, warmup=WU, rep=REP) * 1000
        delta = (us_co - us_s2) / us_s2 * 100
        print(f"{sid}: stride2={us_s2:>8.1f} us, coalesced={us_co:>8.1f} us, delta={delta:>+6.1f}%")

    # =========================================================================
    # K1: Multi-row (persistent) variant
    # =========================================================================
    print()
    print("=" * 80)
    print("KERNEL 1: multi-row variant (process ROWS_PER_PROG rows per program)")
    print("=" * 80)

    @triton.jit
    def _fused_glu_activation_multirow(
        gate_up_ptr,
        output_ptr,
        stride_gate_up_row,
        stride_out_row,
        alpha_val,
        limit_val,
        num_rows,
        I_SIZE: tl.constexpr,
        BLOCK_I: tl.constexpr,
        ROWS_PER_PROG: tl.constexpr,
    ):
        """Process multiple rows per program to amortize launch overhead."""
        pid = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_I)
        col_mask = col_offsets < I_SIZE
        gate_offsets = col_offsets * 2
        up_offsets = col_offsets * 2 + 1

        for r in tl.static_range(ROWS_PER_PROG):
            row_idx = pid * ROWS_PER_PROG + r
            row_active = row_idx < num_rows
            gate_up_row_ptr = gate_up_ptr + row_idx * stride_gate_up_row
            mask = col_mask & row_active
            gate_vals = tl.load(gate_up_row_ptr + gate_offsets, mask=mask, other=0.0)
            up_vals = tl.load(gate_up_row_ptr + up_offsets, mask=mask, other=0.0)
            gate_f = gate_vals.to(tl.float32)
            up_f = up_vals.to(tl.float32)
            gate_f = tl.minimum(gate_f, limit_val)
            up_f = tl.maximum(tl.minimum(up_f, limit_val), -limit_val)
            glu = gate_f * tl.sigmoid(gate_f * alpha_val)
            result = (up_f + 1.0) * glu
            out_row_ptr = output_ptr + row_idx * stride_out_row
            tl.store(out_row_ptr + col_offsets, result.to(gate_vals.dtype), mask=mask)

    ROWS_OPTIONS = [1, 2, 4, 8]

    for sid, E, T, H, inter, desc in SHAPES:
        total_rows = E * T
        nw = 16 if total_rows <= 128 else 4
        block_i = triton.next_power_of_2(inter)
        gu = torch.randn(total_rows, 2 * inter, device="cuda", dtype=DTYPE)
        ao = torch.empty(total_rows, inter, device="cuda", dtype=DTYPE)

        results = []
        for rpp in ROWS_OPTIONS:
            grid_size = triton.cdiv(total_rows, rpp)

            def _run(_gu=gu, _ao=ao, _rpp=rpp, _gs=grid_size):
                _fused_glu_activation_multirow[(_gs,)](
                    _gu,
                    _ao,
                    _gu.stride(0),
                    _ao.stride(0),
                    1.702,
                    7.0,
                    total_rows,
                    I_SIZE=inter,
                    BLOCK_I=block_i,
                    ROWS_PER_PROG=_rpp,
                    num_warps=nw,
                    num_stages=2,
                )

            us = triton.testing.do_bench(_run, warmup=WU, rep=REP) * 1000
            results.append((rpp, us))

        results.sort(key=lambda x: x[1])
        best_rpp, best_us = results[0]
        print(f"{sid}: ", end="")
        for rpp, us in sorted(results, key=lambda x: x[0]):
            marker = " *" if rpp == best_rpp else ""
            print(f"rpp={rpp}: {us:.1f}{marker}  ", end="")
        print()
