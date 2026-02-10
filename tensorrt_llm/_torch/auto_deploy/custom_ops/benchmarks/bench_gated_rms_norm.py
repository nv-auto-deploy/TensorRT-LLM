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

"""Benchmark Gated RMSNorm custom op across Torch and Triton backends.

Usage::

    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_gated_rms_norm
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_gated_rms_norm --dtype half
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_gated_rms_norm --group-size 64
"""

import itertools

import torch
from triton.testing import Benchmark, do_bench, perf_report

# Register all auto_deploy custom ops (side-effect imports)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

from .bench_utils import STR_DTYPE_TO_TORCH_DTYPE, filter_providers_by_dtype, make_arg_parser

# ---------------------------------------------------------------------------
# Provider dtype support for Gated RMSNorm
# ---------------------------------------------------------------------------

PROVIDER_DTYPE_SUPPORT = {
    "torch": {torch.float16, torch.bfloat16, torch.float32},
    "triton": {torch.float16, torch.bfloat16},
}

# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

num_tokens_range = [1, 32, 128, 512, 1024, 4096]
hidden_size_range = [2048, 4096, 8192]
configs = list(itertools.product(num_tokens_range, hidden_size_range))


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


def benchmark_gated_rms_norm(
    num_tokens: int,
    hidden_size: int,
    provider: str,
    dtype: torch.dtype,
    group_size: int,
):
    """Benchmark a single (num_tokens, hidden_size, provider) configuration."""
    device = "cuda"
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    w = torch.randn(hidden_size, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    # Ensure group_size divides hidden_size; fall back to hidden_size if not.
    effective_group_size = group_size if hidden_size % group_size == 0 else hidden_size

    if provider == "torch":
        fn = lambda: torch.ops.auto_deploy.torch_rmsnorm_gated(  # noqa: E731
            x, w, gate, eps, effective_group_size, False
        )
    elif provider == "triton":
        fn = lambda: torch.ops.auto_deploy.triton_rmsnorm_gated(  # noqa: E731
            x, w, gate, eps, effective_group_size, False
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, max_ms, min_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser(description="Benchmark Gated RMSNorm custom op across backends.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help=(
            "Group size for grouped normalization. 0 (default) means use hidden_size "
            "(full-dim grouping). Must divide all hidden_size values in the sweep."
        ),
    )
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    providers = filter_providers_by_dtype(PROVIDER_DTYPE_SUPPORT, dtype)
    if not providers:
        raise RuntimeError(f"No providers support dtype={args.dtype}")

    # If group_size is 0, we'll use hidden_size inside the benchmark function
    # by passing 0 and having the function use hidden_size as the default.
    group_size = args.group_size

    provider_names = {
        "torch": "Torch",
        "triton": "Triton",
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
            plot_name=f"gated-rms-norm-performance-{args.dtype}",
            args={},
        )
    )

    def _bench(num_tokens, hidden_size, provider):
        # If group_size is 0, use hidden_size (full-dim grouping)
        gs = group_size if group_size > 0 else hidden_size
        return benchmark_gated_rms_norm(num_tokens, hidden_size, provider, dtype, gs)

    report(_bench).run(
        print_data=True,
        save_path=args.output_dir if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
