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

"""Sweep num_warps x num_stages for triton_moe_dense_mlp kernels.

Finds the best (num_warps, num_stages) per shape for each kernel.
Outputs a lookup table suitable for integration into the launcher.

Usage:
    python sweep_triton_moe_dense_mlp.py
"""

# Guard all heavy imports and executable code behind __name__ == "__main__" so
# that the auto-import in custom_ops/__init__.py does not trigger CUDA init or
# missing-module errors.
if __name__ == "__main__":
    import json
    import sys
    from itertools import product
    from pathlib import Path

    _this_dir = str(Path(__file__).resolve().parent)
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    import torch
    import triton
    from triton_moe_dense_mlp import _fused_glu_activation_kernel, _weighted_expert_sum_kernel

    # ---------------------------------------------------------------------------
    # Benchmark shapes: (ID, E, T, H, inter, description)
    # ---------------------------------------------------------------------------
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

    NUM_WARPS_OPTIONS = [1, 2, 4, 8, 16]
    NUM_STAGES_OPTIONS = [1, 2, 3, 4, 5]

    def sweep_activation_kernel(E, T, H, inter, num_warps, num_stages):
        """Benchmark _fused_glu_activation_kernel with specific config."""
        gate_up = torch.randn(E * T, 2 * inter, device="cuda", dtype=DTYPE)
        act_out = torch.empty(E * T, inter, device="cuda", dtype=DTYPE)
        block_i = triton.next_power_of_2(inter)
        alpha = 1.702
        limit = 7.0

        def _run():
            _fused_glu_activation_kernel[(E * T,)](
                gate_up,
                act_out,
                gate_up.stride(0),
                act_out.stride(0),
                float(alpha),
                float(limit),
                I_SIZE=inter,
                BLOCK_I=block_i,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        ms = triton.testing.do_bench(_run, warmup=WARMUP_MS, rep=REP_MS)
        return ms * 1000  # return in us

    def sweep_weighted_sum_kernel(E, T, H, num_warps, num_stages):
        """Benchmark _weighted_expert_sum_kernel with specific config."""
        expert_out = torch.randn(E, T, H, device="cuda", dtype=DTYPE).contiguous()
        routing_weights = torch.randn(T, E, device="cuda", dtype=DTYPE).softmax(dim=-1)
        output = torch.empty(T, H, device="cuda", dtype=DTYPE)
        block_h = triton.next_power_of_2(H)

        def _run():
            _weighted_expert_sum_kernel[(T,)](
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
                num_warps=num_warps,
                num_stages=num_stages,
            )

        ms = triton.testing.do_bench(_run, warmup=WARMUP_MS, rep=REP_MS)
        return ms * 1000  # return in us

    def main():
        device = torch.cuda.get_device_name(0)
        print(f"GPU: {device}")
        print(f"Sweeping num_warps={NUM_WARPS_OPTIONS}, num_stages={NUM_STAGES_OPTIONS}")
        print(f"Total configs per shape: {len(NUM_WARPS_OPTIONS) * len(NUM_STAGES_OPTIONS)}")
        print()

        combos = list(product(NUM_WARPS_OPTIONS, NUM_STAGES_OPTIONS))

        # -----------------------------------------------------------------------
        # Sweep Kernel 1: _fused_glu_activation_kernel
        # -----------------------------------------------------------------------
        print("=" * 80)
        print("KERNEL 1: _fused_glu_activation_kernel")
        print("=" * 80)

        k1_best = {}
        k1_all_results = {}
        for shape_id, E, T, H, inter, desc in SHAPES:
            print(f"\n--- {shape_id}: E={E}, T={T}, H={H}, I={inter} ({desc}) ---")
            best_us = float("inf")
            best_cfg = None
            results_for_shape = []

            for nw, ns in combos:
                us = sweep_activation_kernel(E, T, H, inter, nw, ns)
                results_for_shape.append((nw, ns, us))
                if us < best_us:
                    best_us = us
                    best_cfg = (nw, ns)

            # Sort by latency and print top 5
            results_for_shape.sort(key=lambda x: x[2])
            for rank, (nw, ns, us) in enumerate(results_for_shape[:5]):
                marker = " <-- BEST" if rank == 0 else ""
                print(f"  warps={nw:>2}, stages={ns}: {us:>8.1f} us{marker}")

            k1_best[shape_id] = {"warps": best_cfg[0], "stages": best_cfg[1], "us": best_us}
            k1_all_results[shape_id] = results_for_shape

        # -----------------------------------------------------------------------
        # Sweep Kernel 2: _weighted_expert_sum_kernel
        # -----------------------------------------------------------------------
        print()
        print("=" * 80)
        print("KERNEL 2: _weighted_expert_sum_kernel")
        print("=" * 80)

        k2_best = {}
        k2_all_results = {}
        for shape_id, E, T, H, inter, desc in SHAPES:
            print(f"\n--- {shape_id}: E={E}, T={T}, H={H} ({desc}) ---")
            best_us = float("inf")
            best_cfg = None
            results_for_shape = []

            for nw, ns in combos:
                us = sweep_weighted_sum_kernel(E, T, H, nw, ns)
                results_for_shape.append((nw, ns, us))
                if us < best_us:
                    best_us = us
                    best_cfg = (nw, ns)

            results_for_shape.sort(key=lambda x: x[2])
            for rank, (nw, ns, us) in enumerate(results_for_shape[:5]):
                marker = " <-- BEST" if rank == 0 else ""
                print(f"  warps={nw:>2}, stages={ns}: {us:>8.1f} us{marker}")

            k2_best[shape_id] = {"warps": best_cfg[0], "stages": best_cfg[1], "us": best_us}
            k2_all_results[shape_id] = results_for_shape

        # -----------------------------------------------------------------------
        # Summary: baseline vs best
        # -----------------------------------------------------------------------
        baseline_k1 = {"warps": 4, "stages": 3}
        baseline_k2 = {"warps": 4, "stages": 3}

        print()
        print("=" * 80)
        print("SUMMARY: Best configs vs baseline (warps=4, stages=3)")
        print("=" * 80)

        print(f"\n{'ID':<4} | {'Kernel 1':^40} | {'Kernel 2':^40}")
        print(
            f"{'':4} | {'best_cfg':>10} {'best_us':>8} {'baseline':>8} {'delta':>8} | "
            f"{'best_cfg':>10} {'best_us':>8} {'baseline':>8} {'delta':>8}"
        )
        print("-" * 100)

        for shape_id, E, T, H, inter, desc in SHAPES:
            # Kernel 1
            k1b = k1_best[shape_id]
            k1_baseline_us = next(
                us
                for nw, ns, us in k1_all_results[shape_id]
                if nw == baseline_k1["warps"] and ns == baseline_k1["stages"]
            )
            k1_delta = (k1b["us"] - k1_baseline_us) / k1_baseline_us * 100

            # Kernel 2
            k2b = k2_best[shape_id]
            k2_baseline_us = next(
                us
                for nw, ns, us in k2_all_results[shape_id]
                if nw == baseline_k2["warps"] and ns == baseline_k2["stages"]
            )
            k2_delta = (k2b["us"] - k2_baseline_us) / k2_baseline_us * 100

            print(
                f"{shape_id:<4} | "
                f"w={k1b['warps']:>2},s={k1b['stages']} {k1b['us']:>8.1f} {k1_baseline_us:>8.1f} {k1_delta:>+7.1f}% | "
                f"w={k2b['warps']:>2},s={k2b['stages']} {k2b['us']:>8.1f} {k2_baseline_us:>8.1f} {k2_delta:>+7.1f}%"
            )

        # Save full results as JSON for later analysis
        output = {
            "kernel1_best": k1_best,
            "kernel2_best": k2_best,
            "kernel1_all": {
                sid: [(nw, ns, us) for nw, ns, us in results]
                for sid, results in k1_all_results.items()
            },
            "kernel2_all": {
                sid: [(nw, ns, us) for nw, ns, us in results]
                for sid, results in k2_all_results.items()
            },
        }
        with open("sweep_results.json", "w") as f:
            json.dump(output, f, indent=2)
        print("\nFull results saved to sweep_results.json")

    main()
