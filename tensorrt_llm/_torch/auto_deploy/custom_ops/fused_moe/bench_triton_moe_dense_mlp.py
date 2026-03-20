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

"""Benchmark for triton_moe_dense_mlp Triton kernels.

Measures kernel-level latency for:
  - _fused_glu_activation_kernel  (activation fusion, steps 2-5)
  - _weighted_expert_sum_kernel   (expert weighted sum, step 7)
  - End-to-end _moe_dense_mlp_triton (all steps including torch.bmm)

Usage (run from repo root or this directory):
    python bench_triton_moe_dense_mlp.py
"""

# Guard all heavy imports behind __name__ == "__main__" so that the auto-import
# in custom_ops/__init__.py does not trigger CUDA init or missing-module errors.
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure the fused_moe package directory is on sys.path for direct script execution
    _this_dir = str(Path(__file__).resolve().parent)
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    import torch
    import triton
    import triton.language as tl
    from triton_moe_dense_mlp import (
        _fused_glu_activation_kernel,
        _moe_dense_mlp_triton,
        _weighted_expert_sum_kernel,
    )

    # -----------------------------------------------------------------------
    # Benchmark shapes: (ID, E, T, H, inter, description)
    # -----------------------------------------------------------------------
    SHAPES = [
        ("A1", 128, 1, 2880, 2880, "GPT-OSS-120B single-token decode"),
        ("A2", 128, 8, 2880, 2880, "GPT-OSS-120B small batch decode"),
        ("A3", 128, 32, 2880, 2880, "GPT-OSS-120B medium batch decode"),
        ("A4", 128, 128, 2880, 2880, "GPT-OSS-120B large batch decode"),
        ("A5", 128, 512, 2880, 2880, "GPT-OSS-120B prefill"),
        ("B1", 32, 1, 2880, 2880, "GPT-OSS-20B single-token decode"),
        ("B2", 32, 32, 2880, 2880, "GPT-OSS-20B medium batch decode"),
        ("B3", 32, 512, 2880, 2880, "GPT-OSS-20B prefill"),
    ]

    DTYPE = torch.bfloat16
    WARMUP_MS = 25
    REP_MS = 100

    def _get_k1_config(total_rows, inter):
        """Get kernel 1 launch config matching the launcher in triton_moe_dense_mlp.py."""
        if total_rows <= 128:
            block_i = 1024
            num_warps = 16
        else:
            block_i = triton.next_power_of_2(inter)
            num_warps = 4
        num_i_blocks = triton.cdiv(inter, block_i)
        return num_warps, 2, block_i, num_i_blocks

    def _get_k2_config(E, T, H):
        """Get kernel 2 launch config matching the launcher in triton_moe_dense_mlp.py."""
        if T <= 32 and E <= 32:
            block_h = 256
        elif T <= 128:
            block_h = 1024
        else:
            block_h = triton.next_power_of_2(H)
        num_h_blocks = triton.cdiv(H, block_h)
        return 16, 2, block_h, num_h_blocks

    def bench_activation_kernel(E, T, H, inter):
        """Benchmark _fused_glu_activation_kernel in isolation with coalesced layout."""
        total_rows = E * T
        gate_up = torch.randn(total_rows, 2 * inter, device="cuda", dtype=DTYPE)
        act_out = torch.empty(total_rows, inter, device="cuda", dtype=DTYPE)
        alpha = 1.702
        limit = 7.0
        nw, ns, block_i, num_i_blocks = _get_k1_config(total_rows, inter)

        def _run():
            _fused_glu_activation_kernel[(total_rows, num_i_blocks)](
                gate_up,
                act_out,
                gate_up.stride(0),
                act_out.stride(0),
                float(alpha),
                float(limit),
                I_SIZE=inter,
                BLOCK_I=block_i,
                num_warps=nw,
                num_stages=ns,
            )

        ms = triton.testing.do_bench(_run, warmup=WARMUP_MS, rep=REP_MS)
        return ms

    def bench_weighted_sum_kernel(E, T, H):
        """Benchmark _weighted_expert_sum_kernel in isolation."""
        expert_out = torch.randn(E, T, H, device="cuda", dtype=DTYPE).contiguous()
        routing_weights = torch.randn(T, E, device="cuda", dtype=DTYPE).softmax(dim=-1)
        output = torch.empty(T, H, device="cuda", dtype=DTYPE)
        nw, ns, block_h, num_h_blocks = _get_k2_config(E, T, H)

        def _run():
            _weighted_expert_sum_kernel[(T, num_h_blocks)](
                expert_out,
                routing_weights,
                output,
                stride_expert_out_e=expert_out.stride(0),
                stride_expert_out_t=expert_out.stride(1),
                stride_expert_out_h=expert_out.stride(2),
                stride_routing_t=routing_weights.stride(0),
                stride_routing_e=routing_weights.stride(1),
                stride_out_t=output.stride(0),
                stride_out_h=output.stride(1),
                num_experts=E,
                H_SIZE=H,
                BLOCK_H=block_h,
                OUTPUT_DTYPE=tl.bfloat16,
                num_warps=nw,
                num_stages=ns,
            )

        ms = triton.testing.do_bench(_run, warmup=WARMUP_MS, rep=REP_MS)
        return ms

    def bench_e2e(E, T, H, inter):
        """Benchmark end-to-end _moe_dense_mlp_triton (includes torch.bmm)."""
        hidden = torch.randn(T, H, device="cuda", dtype=DTYPE)
        routing = torch.randn(T, E, device="cuda", dtype=DTYPE).softmax(dim=-1)
        gate_up_w = torch.randn(E, H, 2 * inter, device="cuda", dtype=DTYPE) * 0.02
        gate_up_b = torch.randn(E, 2 * inter, device="cuda", dtype=DTYPE) * 0.02
        down_w = torch.randn(E, inter, H, device="cuda", dtype=DTYPE) * 0.02
        down_b = torch.randn(E, H, device="cuda", dtype=DTYPE) * 0.02

        def _run():
            _moe_dense_mlp_triton(hidden, routing, gate_up_w, gate_up_b, down_w, down_b, 1.702, 7.0)

        ms = triton.testing.do_bench(_run, warmup=WARMUP_MS, rep=REP_MS)
        return ms

    # -------------------------------------------------------------------
    # Main
    # -------------------------------------------------------------------
    device = torch.cuda.get_device_name(0)
    print(f"GPU: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Dtype: {DTYPE}")
    print(
        "K1 config: _get_k1_config (adaptive 2D grid); K2 config: _get_k2_config (adaptive 2D/1D grid)"
    )
    print()

    header = (
        f"{'ID':<4} {'E':>4} {'T':>5} {'H':>5} {'I':>5}  "
        f"{'Activ(us)':>10} {'WSum(us)':>10} {'E2E(us)':>10}  "
        f"{'Scenario'}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for shape_id, E, T, H, inter, desc in SHAPES:
        ms_act = bench_activation_kernel(E, T, H, inter)
        ms_wsum = bench_weighted_sum_kernel(E, T, H)
        ms_e2e = bench_e2e(E, T, H, inter)

        us_act = ms_act * 1000
        us_wsum = ms_wsum * 1000
        us_e2e = ms_e2e * 1000

        print(
            f"{shape_id:<4} {E:>4} {T:>5} {H:>5} {inter:>5}  "
            f"{us_act:>10.1f} {us_wsum:>10.1f} {us_e2e:>10.1f}  "
            f"{desc}"
        )
        results.append((shape_id, E, T, H, inter, us_act, us_wsum, us_e2e, desc))

    # Print summary
    print()
    print("Triton kernel share of E2E:")
    print(f"{'ID':<4} {'Activ%':>8} {'WSum%':>8} {'Triton%':>9} {'BMM%':>8}")
    print("-" * 42)
    for shape_id, E, T, H, inter, us_act, us_wsum, us_e2e, desc in results:
        triton_total = us_act + us_wsum
        pct_act = us_act / us_e2e * 100 if us_e2e > 0 else 0
        pct_wsum = us_wsum / us_e2e * 100 if us_e2e > 0 else 0
        pct_triton = triton_total / us_e2e * 100 if us_e2e > 0 else 0
        pct_bmm = 100 - pct_triton
        print(
            f"{shape_id:<4} {pct_act:>7.1f}% {pct_wsum:>7.1f}% {pct_triton:>8.1f}% {pct_bmm:>7.1f}%"
        )
