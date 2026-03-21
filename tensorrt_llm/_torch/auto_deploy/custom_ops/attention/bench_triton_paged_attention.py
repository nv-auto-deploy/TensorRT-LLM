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

"""Benchmark script for Triton Paged Attention kernels.

Benchmarks decode (stage1+stage2) and prefill kernels in isolation,
and compares against FlashInfer as reference.

Usage:
    python bench_triton_paged_attention.py                # all benchmarks
    python bench_triton_paged_attention.py --mode decode   # decode only
    python bench_triton_paged_attention.py --mode prefill  # prefill only
"""

import argparse
import math

import flashinfer
import torch
import triton

try:
    # When running as a standalone script from the attention directory
    from triton_paged_attention import (
        triton_paged_context,
        triton_paged_decode,
        update_paged_kv_cache,
    )
except ModuleNotFoundError:
    # When imported as part of the package
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
        triton_paged_context,
        triton_paged_decode,
        update_paged_kv_cache,
    )

PAGE_SIZE = 16
DTYPE = torch.float16
WARMUP = 25
REP = 100


# =============================================================================
# Helpers
# =============================================================================


def create_paged_kv_cache(num_blocks, n_kv_heads, head_dim):
    return torch.randn(num_blocks, 2, n_kv_heads, PAGE_SIZE, head_dim, dtype=DTYPE, device="cuda")


def make_decode_inputs(batch_size, seq_len, n_heads, n_kv_heads, head_dim):
    """Create inputs for decode benchmark."""
    num_pages_per_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = batch_size * num_pages_per_seq + 16

    q = torch.randn(batch_size, n_heads, head_dim, dtype=DTYPE, device="cuda")
    kv_cache = create_paged_kv_cache(num_blocks, n_kv_heads, head_dim)

    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * num_pages_per_seq,
        num_pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )[: batch_size + 1]
    kv_indices = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda")
    last_in_page = seq_len % PAGE_SIZE
    kv_last_page_len = torch.full(
        (batch_size,),
        last_in_page if last_in_page > 0 else PAGE_SIZE,
        dtype=torch.int32,
        device="cuda",
    )
    sm_scale = 1.0 / math.sqrt(head_dim)

    return q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale


def make_prefill_inputs(batch_size, seq_len, n_heads, n_kv_heads, head_dim):
    """Create inputs for prefill benchmark."""
    num_pages_per_seq = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = batch_size * num_pages_per_seq + 16
    total_tokens = batch_size * seq_len

    q = torch.randn(total_tokens, n_heads, head_dim, dtype=DTYPE, device="cuda")
    k = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=DTYPE, device="cuda")
    v = torch.randn(total_tokens, n_kv_heads, head_dim, dtype=DTYPE, device="cuda")

    qo_indptr = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device="cuda"
    )[: batch_size + 1]
    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * num_pages_per_seq,
        num_pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )[: batch_size + 1]
    kv_indices = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device="cuda")
    last_in_page = seq_len % PAGE_SIZE
    kv_last_page_len = torch.full(
        (batch_size,),
        last_in_page if last_in_page > 0 else PAGE_SIZE,
        dtype=torch.int32,
        device="cuda",
    )
    seq_len_with_cache = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    batch_indices = torch.repeat_interleave(
        torch.arange(batch_size, device="cuda", dtype=torch.int32), seq_len
    )
    positions = torch.tile(torch.arange(seq_len, device="cuda", dtype=torch.int32), (batch_size,))

    # Fill cache
    kv_cache = create_paged_kv_cache(num_blocks, n_kv_heads, head_dim)
    update_paged_kv_cache(k, v, batch_indices, positions, kv_cache, kv_indices, kv_indptr)

    sm_scale = 1.0 / math.sqrt(head_dim)

    return (
        q,
        kv_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        seq_len_with_cache,
        sm_scale,
    )


# =============================================================================
# Decode benchmark
# =============================================================================

DECODE_SHAPES = [
    # (ID, batch, seq_len, n_heads, n_kv_heads, head_dim)
    # Llama-8B (HEAD_RATIO=4)
    ("D1", 1, 512, 32, 8, 128),
    ("D2", 1, 2048, 32, 8, 128),
    ("D3", 1, 8192, 32, 8, 128),
    ("D4", 8, 512, 32, 8, 128),
    ("D5", 8, 2048, 32, 8, 128),
    ("D6", 8, 8192, 32, 8, 128),
    ("D7", 32, 2048, 32, 8, 128),
    ("D8", 128, 2048, 32, 8, 128),
    # Llama-70B TP=4 (HEAD_RATIO=8)
    ("D11", 1, 2048, 16, 2, 128),
    ("D12", 8, 2048, 16, 2, 128),
    # Nemotron-Nano (HEAD_RATIO=16)
    ("D13", 1, 2048, 32, 2, 128),
    ("D14", 8, 2048, 32, 2, 128),
    ("D15", 32, 2048, 32, 2, 128),
    # Qwen-7B (HEAD_RATIO=7, non-power-of-2 — may fail)
    ("D9", 1, 2048, 28, 4, 128),
    ("D10", 8, 2048, 28, 4, 128),
]


def bench_decode():
    print("\n" + "=" * 100)
    print("DECODE BENCHMARK")
    print("=" * 100)
    header = (
        f"{'ID':<5} {'batch':>5} {'seqlen':>6} {'heads':>5} {'kv_h':>4} {'hdim':>4}"
        f" | {'Triton(us)':>10} {'FI(us)':>10} {'Ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    for shape_id, batch, seq_len, n_heads, n_kv_heads, head_dim in DECODE_SHAPES:
        try:
            q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale = make_decode_inputs(
                batch, seq_len, n_heads, n_kv_heads, head_dim
            )

            # --- Triton ---
            triton_paged_decode(q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale)
            torch.cuda.synchronize()

            triton_us = (
                triton.testing.do_bench(
                    lambda: triton_paged_decode(
                        q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale
                    ),
                    warmup=WARMUP,
                    rep=REP,
                )
                * 1000
            )  # ms -> us

            # --- FlashInfer ---
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "HND", use_tensor_cores=True
            )
            wrapper.plan(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                n_heads,
                n_kv_heads,
                head_dim,
                PAGE_SIZE,
                q_data_type=DTYPE,
                kv_data_type=DTYPE,
                sm_scale=sm_scale,
            )
            wrapper.run(q, kv_cache)
            torch.cuda.synchronize()

            fi_us = (
                triton.testing.do_bench(
                    lambda: wrapper.run(q, kv_cache),
                    warmup=WARMUP,
                    rep=REP,
                )
                * 1000
            )

            ratio = triton_us / fi_us if fi_us > 0 else float("inf")
            print(
                f"{shape_id:<5} {batch:>5} {seq_len:>6} {n_heads:>5} {n_kv_heads:>4} {head_dim:>4} | "
                f"{triton_us:>10.1f} {fi_us:>10.1f} {ratio:>7.2f}x"
            )
        except Exception as e:
            print(
                f"{shape_id:<5} {batch:>5} {seq_len:>6} {n_heads:>5} {n_kv_heads:>4} {head_dim:>4} | FAILED: {e}"
            )


# =============================================================================
# Prefill benchmark
# =============================================================================

PREFILL_SHAPES = [
    # (ID, batch, seq_len, n_heads, n_kv_heads, head_dim)
    ("P1", 1, 128, 32, 8, 128),
    ("P2", 1, 512, 32, 8, 128),
    ("P3", 1, 2048, 32, 8, 128),
    ("P4", 4, 512, 32, 8, 128),
    ("P5", 4, 2048, 32, 8, 128),
    ("P6", 1, 2048, 28, 4, 128),
    ("P7", 1, 2048, 16, 2, 128),
    ("P8", 1, 2048, 32, 2, 128),
    ("P9", 4, 512, 32, 2, 128),
]


def bench_prefill():
    print("\n" + "=" * 100)
    print("PREFILL BENCHMARK")
    print("=" * 100)
    header = (
        f"{'ID':<5} {'batch':>5} {'seqlen':>6} {'heads':>5} {'kv_h':>4} {'hdim':>4}"
        f" | {'Triton(us)':>10} {'FI(us)':>10} {'Ratio':>7}"
    )
    print(header)
    print("-" * len(header))

    for shape_id, batch, seq_len, n_heads, n_kv_heads, head_dim in PREFILL_SHAPES:
        try:
            (
                q,
                kv_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                seq_len_with_cache,
                sm_scale,
            ) = make_prefill_inputs(batch, seq_len, n_heads, n_kv_heads, head_dim)

            # --- Triton ---
            triton_paged_context(
                q,
                kv_cache,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                seq_len_with_cache,
                sm_scale,
            )
            torch.cuda.synchronize()

            triton_us = (
                triton.testing.do_bench(
                    lambda: triton_paged_context(
                        q,
                        kv_cache,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        seq_len_with_cache,
                        sm_scale,
                    ),
                    warmup=WARMUP,
                    rep=REP,
                )
                * 1000
            )

            # --- FlashInfer ---
            workspace = torch.empty(320 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "HND")
            wrapper.plan(
                qo_indptr.cpu(),
                kv_indptr.cpu(),
                kv_indices,
                kv_last_page_len.cpu(),
                n_heads,
                n_kv_heads,
                head_dim,
                PAGE_SIZE,
                causal=True,
                q_data_type=DTYPE,
                kv_data_type=DTYPE,
                sm_scale=sm_scale,
                seq_lens=seq_len_with_cache.cpu(),
            )
            wrapper.run(q, kv_cache)
            torch.cuda.synchronize()

            fi_us = (
                triton.testing.do_bench(
                    lambda: wrapper.run(q, kv_cache),
                    warmup=WARMUP,
                    rep=REP,
                )
                * 1000
            )

            ratio = triton_us / fi_us if fi_us > 0 else float("inf")
            print(
                f"{shape_id:<5} {batch:>5} {seq_len:>6} {n_heads:>5} {n_kv_heads:>4} {head_dim:>4} | "
                f"{triton_us:>10.1f} {fi_us:>10.1f} {ratio:>7.2f}x"
            )
        except Exception as e:
            print(
                f"{shape_id:<5} {batch:>5} {seq_len:>6} {n_heads:>5} {n_kv_heads:>4} {head_dim:>4} | FAILED: {e}"
            )


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton Paged Attention")
    parser.add_argument(
        "--mode",
        choices=["all", "decode", "prefill"],
        default="all",
        help="Which benchmark to run",
    )
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"dtype: {DTYPE}, page_size: {PAGE_SIZE}")
    print(f"Benchmark: warmup={WARMUP}, rep={REP}")

    if args.mode in ("all", "decode"):
        bench_decode()
    if args.mode in ("all", "prefill"):
        bench_prefill()

    print("\nDone.")


if __name__ == "__main__":
    main()
