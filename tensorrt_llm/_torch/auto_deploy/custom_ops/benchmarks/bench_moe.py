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

"""Benchmark Fused MoE custom op across Torch and TRT-LLM backends.

Benchmarks the BF16/FP16 gated MoE (SiLU activation) variant:
  y = sum_k routing_weight_k * ( silu(x @ w1_k.T) * (x @ w3_k.T) ) @ w2_k.T

Usage::

    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_moe
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_moe --dtype half
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_moe --num-experts 16 --top-k 4
"""

import itertools

import torch
from triton.testing import Benchmark, do_bench, perf_report

# Register all auto_deploy custom ops (side-effect imports)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

from .bench_utils import STR_DTYPE_TO_TORCH_DTYPE, filter_providers_by_dtype, make_arg_parser

# ---------------------------------------------------------------------------
# Provider dtype support for fused MoE (BF16/FP16 gated SiLU)
# ---------------------------------------------------------------------------

PROVIDER_DTYPE_SUPPORT = {
    "torch": {torch.float16, torch.bfloat16},
    "trtllm": {torch.float16, torch.bfloat16},
}

# ---------------------------------------------------------------------------
# Default benchmark configs
# ---------------------------------------------------------------------------

num_tokens_range = [1, 32, 128, 512, 1024]
hidden_size_range = [2048, 4096]
configs = list(itertools.product(num_tokens_range, hidden_size_range))


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------


def _prepare_moe_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype,
    device: str,
):
    """Create inputs for fused MoE benchmarking.

    Returns:
        x: [num_tokens, hidden_size]
        selected_experts: [num_tokens, top_k]
        routing_weights: [num_tokens, top_k]
        w3_w1_stacked: [num_experts, 2*intermediate_size, hidden_size]
        w2_stacked: [num_experts, hidden_size, intermediate_size]
    """
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Random expert selection and routing weights
    selected_experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    routing_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=torch.float32), dim=-1
    ).to(dtype)

    # Stacked weights in TRT-LLM format
    # w3_w1_stacked: [E, 2*I, H] (gate + up projections concatenated)
    w3_w1_stacked = (
        torch.randn(num_experts, 2 * intermediate_size, hidden_size, dtype=dtype, device=device)
        * 0.01
    )
    # w2_stacked: [E, H, I] (down projection)
    w2_stacked = (
        torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype, device=device) * 0.01
    )

    return x, selected_experts, routing_weights, w3_w1_stacked, w2_stacked


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


def benchmark_moe(
    num_tokens: int,
    hidden_size: int,
    provider: str,
    dtype: torch.dtype,
    num_experts: int,
    top_k: int,
    intermediate_size_multiplier: int,
):
    """Benchmark a single (num_tokens, hidden_size, provider) configuration."""
    device = "cuda"
    intermediate_size = hidden_size * intermediate_size_multiplier

    x, selected_experts, routing_weights, w3_w1_stacked, w2_stacked = _prepare_moe_inputs(
        num_tokens, hidden_size, intermediate_size, num_experts, top_k, dtype, device
    )

    if provider == "torch":
        fn = lambda: torch.ops.auto_deploy.torch_moe_fused(  # noqa: E731
            x, selected_experts, routing_weights, w3_w1_stacked, w2_stacked
        )
    elif provider == "trtllm":
        fn = lambda: torch.ops.auto_deploy.trtllm_moe_fused(  # noqa: E731
            x, selected_experts, routing_weights, w3_w1_stacked, w2_stacked, True
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, max_ms, min_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser(description="Benchmark Fused MoE custom op across backends.")
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts (default: 8).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of experts selected per token (default: 2).",
    )
    parser.add_argument(
        "--intermediate-multiplier",
        type=int,
        default=4,
        help="Intermediate size = hidden_size * multiplier (default: 4).",
    )
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    providers = filter_providers_by_dtype(PROVIDER_DTYPE_SUPPORT, dtype)
    if not providers:
        raise RuntimeError(f"No providers support dtype={args.dtype}")

    provider_names = {
        "torch": "Torch (fused)",
        "trtllm": "TRT-LLM (fused)",
    }

    report = perf_report(
        Benchmark(
            x_names=["num_tokens", "hidden_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=[provider_names[p] for p in providers],
            styles=[("blue", "-"), ("green", "-")][: len(providers)],
            ylabel="ms",
            plot_name=f"moe-fused-performance-{args.dtype}-E{args.num_experts}-K{args.top_k}",
            args={},
        )
    )

    report(
        lambda num_tokens, hidden_size, provider: benchmark_moe(
            num_tokens,
            hidden_size,
            provider,
            dtype,
            args.num_experts,
            args.top_k,
            args.intermediate_multiplier,
        )
    ).run(
        print_data=True,
        save_path=args.output_dir if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
