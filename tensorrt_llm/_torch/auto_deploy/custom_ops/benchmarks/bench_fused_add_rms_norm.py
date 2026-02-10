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

"""Benchmark Fused Add + RMSNorm custom op across backends.

Compares the FlashInfer fused add+RMSNorm kernel against a Torch baseline
that performs the two operations separately (add then RMSNorm).

Usage::

    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_fused_add_rms_norm
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_fused_add_rms_norm --dtype half
"""

import itertools

import torch
from triton.testing import Benchmark, do_bench, perf_report

# Register all auto_deploy custom ops (side-effect imports)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

from .bench_utils import STR_DTYPE_TO_TORCH_DTYPE, filter_providers_by_dtype, make_arg_parser

# ---------------------------------------------------------------------------
# Provider dtype support for Fused Add + RMSNorm
# ---------------------------------------------------------------------------

PROVIDER_DTYPE_SUPPORT = {
    "flashinfer": {torch.float16, torch.bfloat16},
    "torch": {torch.float16, torch.bfloat16, torch.float32},
}

# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

num_tokens_range = [1, 32, 128, 512, 1024, 4096]
hidden_size_range = [2048, 4096, 8192]
configs = list(itertools.product(num_tokens_range, hidden_size_range))


# ---------------------------------------------------------------------------
# Torch baseline: separate add + RMSNorm
# ---------------------------------------------------------------------------


def torch_fused_add_rms_norm_baseline(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
):
    """Torch baseline: residual = x + residual, then x = rmsnorm(residual)."""
    residual = x + residual
    x = torch.ops.auto_deploy.torch_rmsnorm(residual, weight, eps)
    return x, residual


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


def benchmark_fused_add_rms_norm(
    num_tokens: int,
    hidden_size: int,
    provider: str,
    dtype: torch.dtype,
):
    """Benchmark a single (num_tokens, hidden_size, provider) configuration."""
    device = "cuda"
    w = torch.randn(hidden_size, dtype=dtype, device=device)
    eps = 1e-6

    if provider == "flashinfer":

        def fn():
            # Allocate fresh tensors each call since the op is in-place
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            torch.ops.auto_deploy.flashinfer_fused_add_rms_norm_inplace(x, residual, w, eps)
            return x, residual

    elif provider == "torch":

        def fn():
            x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            return torch_fused_add_rms_norm_baseline(x, residual, w, eps)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, max_ms, min_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser(description="Benchmark Fused Add + RMSNorm across backends.")
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    providers = filter_providers_by_dtype(PROVIDER_DTYPE_SUPPORT, dtype)
    if not providers:
        raise RuntimeError(f"No providers support dtype={args.dtype}")

    provider_names = {
        "flashinfer": "FlashInfer (fused)",
        "torch": "Torch (add + rmsnorm)",
    }

    report = perf_report(
        Benchmark(
            x_names=["num_tokens", "hidden_size"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=[provider_names[p] for p in providers],
            styles=[("blue", "-"), ("red", "-")][: len(providers)],
            ylabel="ms",
            plot_name=f"fused-add-rms-norm-performance-{args.dtype}",
            args={},
        )
    )

    report(
        lambda num_tokens, hidden_size, provider: benchmark_fused_add_rms_norm(
            num_tokens, hidden_size, provider, dtype
        )
    ).run(
        print_data=True,
        save_path=args.output_dir if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
