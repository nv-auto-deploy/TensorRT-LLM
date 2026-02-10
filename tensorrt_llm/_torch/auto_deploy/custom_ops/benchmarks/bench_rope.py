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

"""Benchmark RoPE custom op across FlashInfer, Triton, and Torch backends.

Each backend uses a different RoPE API, so the benchmark function prepares
the appropriate inputs for each provider.

Usage::

    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_rope
    python -m tensorrt_llm._torch.auto_deploy.custom_ops.benchmarks.bench_rope --dtype half
"""

import itertools

import torch
from triton.testing import Benchmark, do_bench, perf_report

# Register all auto_deploy custom ops (side-effect imports)
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

from .bench_utils import STR_DTYPE_TO_TORCH_DTYPE, filter_providers_by_dtype, make_arg_parser

# ---------------------------------------------------------------------------
# Provider dtype support for RoPE
# ---------------------------------------------------------------------------

PROVIDER_DTYPE_SUPPORT = {
    "triton": {torch.float16, torch.bfloat16},
    "flashinfer": {torch.float16, torch.bfloat16},
    "torch": {torch.float16, torch.bfloat16, torch.float32},
}

# ---------------------------------------------------------------------------
# Benchmark configs
# ---------------------------------------------------------------------------

NUM_HEADS = 32
NUM_KV_HEADS = 8  # for GQA-style models
MAX_SEQ_LEN = 8192

num_tokens_range = [1, 32, 128, 512, 1024, 4096]
head_dim_range = [64, 128, 256]
configs = list(itertools.product(num_tokens_range, head_dim_range))


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------


def _make_cos_sin(seq_len: int, head_dim: int, dtype: torch.dtype, device: str):
    """Create cos/sin tensors for Torch RoPE (HF-style)."""
    # Simulate HF-style cos/sin: shape [1, seq_len, head_dim]
    freqs = torch.randn(1, seq_len, head_dim, dtype=dtype, device=device)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def _make_cos_sin_cache(max_seq_len: int, head_dim: int, device: str):
    """Create a fused cos_sin_cache for FlashInfer RoPE.

    FlashInfer expects a float32 cache of shape [max_seq_len, head_dim] where
    the first half contains cos values and the second half contains sin values
    for head_dim//2 frequencies.
    """
    cache = torch.randn(max_seq_len, head_dim, dtype=torch.float32, device=device)
    return cache


def _make_freqs_cis(seq_len: int, head_dim: int, device: str):
    """Create interleaved freqs_cis for Triton RoPE.

    The Triton kernel expects freqs_cis of shape [max_seq_len, head_dim]
    containing interleaved cos and sin values.
    """
    freqs_cis = torch.randn(seq_len + MAX_SEQ_LEN, head_dim, dtype=torch.float32, device=device)
    return freqs_cis


# ---------------------------------------------------------------------------
# Benchmark function
# ---------------------------------------------------------------------------


def benchmark_rope(
    num_tokens: int,
    head_dim: int,
    provider: str,
    dtype: torch.dtype,
):
    """Benchmark a single (num_tokens, head_dim, provider) configuration."""
    device = "cuda"
    batch_size = 1
    seq_len = num_tokens

    if provider == "torch":
        # Torch RoPE: torch_rope_with_explicit_cos_sin
        # q, k: [B, N, S, D], cos/sin: [B, S, D]
        q = torch.randn(batch_size, NUM_HEADS, seq_len, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, NUM_KV_HEADS, seq_len, head_dim, dtype=dtype, device=device)
        cos, sin = _make_cos_sin(seq_len, head_dim, dtype, device)
        fn = lambda: torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(  # noqa: E731
            q, k, cos, sin, 1
        )

    elif provider == "triton":
        # Triton RoPE: triton_rope_with_input_pos
        # x: [B, S, N, D] (layout='bsnd'), freqs_cis: [max_seq_len, D], input_pos: [B]
        q = torch.randn(batch_size, seq_len, NUM_HEADS, head_dim, dtype=dtype, device=device)
        freqs_cis = _make_freqs_cis(seq_len, head_dim, device)
        input_pos = torch.zeros(batch_size, dtype=torch.int32, device=device)

        def fn():
            # Benchmark Q and K separately since triton_rope_with_input_pos operates on a single tensor
            q_rot = torch.ops.auto_deploy.triton_rope_with_input_pos(
                q, freqs_cis, input_pos, "bsnd"
            )
            return q_rot

    elif provider == "flashinfer":
        # FlashInfer RoPE: flashinfer_rope
        # q, k: [B, S, N, D], position_ids: [B, S], cos_sin_cache: [max_seq_len, D]
        # Note: head_dim must be a multiple of 64 for FlashInfer
        if head_dim % 64 != 0:
            # Return a dummy value for unsupported head_dim
            return 0.0, 0.0, 0.0

        q = torch.randn(batch_size, seq_len, NUM_HEADS, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, NUM_KV_HEADS, head_dim, dtype=dtype, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cos_sin_cache = _make_cos_sin_cache(MAX_SEQ_LEN, head_dim, device)

        fn = lambda: torch.ops.auto_deploy.flashinfer_rope(  # noqa: E731
            q, k, position_ids, cos_sin_cache, True
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    ms, min_ms, max_ms = do_bench(fn, quantiles=[0.5, 0.2, 0.8])
    return ms, max_ms, min_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = make_arg_parser(description="Benchmark RoPE custom op across backends.")
    args = parser.parse_args()
    dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    providers = filter_providers_by_dtype(PROVIDER_DTYPE_SUPPORT, dtype)
    if not providers:
        raise RuntimeError(f"No providers support dtype={args.dtype}")

    provider_names = {
        "triton": "Triton",
        "flashinfer": "FlashInfer",
        "torch": "Torch",
    }

    report = perf_report(
        Benchmark(
            x_names=["num_tokens", "head_dim"],
            x_vals=configs,
            line_arg="provider",
            line_vals=providers,
            line_names=[provider_names[p] for p in providers],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")][: len(providers)],
            ylabel="ms",
            plot_name=f"rope-performance-{args.dtype}",
            args={},
        )
    )

    report(
        lambda num_tokens, head_dim, provider: benchmark_rope(num_tokens, head_dim, provider, dtype)
    ).run(
        print_data=True,
        save_path=args.output_dir if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
