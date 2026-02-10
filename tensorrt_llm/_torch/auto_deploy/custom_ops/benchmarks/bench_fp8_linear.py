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

"""Benchmark FP8 Linear custom op across TRT-LLM and Torch backends.

Usage::

    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_fp8_linear
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_fp8_linear --dtype half
"""

import itertools

import torch
from triton.testing import Benchmark, do_bench, perf_report

# Register all auto_deploy custom ops (side-effect imports)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

from .bench_utils import STR_DTYPE_TO_TORCH_DTYPE, filter_providers_by_dtype, make_arg_parser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# ---------------------------------------------------------------------------
# Provider dtype support for FP8 Linear
# Both providers accept float16/bfloat16 inputs (they internally quantize to FP8).
# ---------------------------------------------------------------------------

PROVIDER_DTYPE_SUPPORT = {
    "trtllm": {torch.float16, torch.bfloat16},
    "torch": {torch.float16, torch.bfloat16},
}

# ---------------------------------------------------------------------------
# Benchmark configs: (num_tokens, in_features, out_features)
# Representative LLM dimensions (roughly matching Llama-style models)
# ---------------------------------------------------------------------------

num_tokens_range = [1, 32, 128, 512, 1024]
dim_pairs = [
    (4096, 4096),  # self-attn QKV / dense projection
    (4096, 11008),  # MLP gate/up projection
    (11008, 4096),  # MLP down projection
    (8192, 8192),  # larger model
]
configs = list(itertools.product(num_tokens_range, dim_pairs))
# Flatten to (num_tokens, in_features, out_features)
configs = [(n, d[0], d[1]) for n, d in configs]


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------


def _prepare_fp8_weight(out_features: int, in_features: int, device: str):
    """Create a pre-quantized FP8 weight and its scale."""
    weight_f32 = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    weight_scale = (weight_f32.abs().max() / FP8_MAX).to(torch.float32)
    weight_fp8 = (weight_f32 / weight_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return weight_fp8, weight_scale


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


def benchmark_fp8_linear(
    num_tokens: int,
    in_features: int,
    out_features: int,
    provider: str,
    dtype: torch.dtype,
):
    """Benchmark a single (num_tokens, in_features, out_features, provider) configuration."""
    device = "cuda"
    x = torch.randn(num_tokens, in_features, dtype=dtype, device=device)
    weight_fp8, weight_scale = _prepare_fp8_weight(out_features, in_features, device)
    input_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    if provider == "trtllm":
        fn = lambda: torch.ops.auto_deploy.trtllm_quant_fp8_linear(  # noqa: E731
            x, weight_fp8, None, input_scale, weight_scale
        )
    elif provider == "torch":
        fn = lambda: torch.ops.auto_deploy.torch_quant_fp8_linear(  # noqa: E731
            x, weight_fp8, None, input_scale, weight_scale
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, max_ms, min_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser(description="Benchmark FP8 Linear custom op across backends.")
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    providers = filter_providers_by_dtype(PROVIDER_DTYPE_SUPPORT, dtype)
    if not providers:
        raise RuntimeError(f"No providers support dtype={args.dtype}")

    provider_names = {
        "trtllm": "TRT-LLM (FP8)",
        "torch": "Torch (FP8)",
    }

    report = perf_report(
        Benchmark(
            x_names=["num_tokens", "in_features", "out_features"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=[provider_names[p] for p in providers],
            styles=[("blue", "-"), ("green", "-")][: len(providers)],
            ylabel="ms",
            plot_name=f"fp8-linear-performance-{args.dtype}",
            args={},
        )
    )

    report(
        lambda num_tokens, in_features, out_features, provider: benchmark_fp8_linear(
            num_tokens, in_features, out_features, provider, dtype
        )
    ).run(
        print_data=True,
        save_path=args.output_dir if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
