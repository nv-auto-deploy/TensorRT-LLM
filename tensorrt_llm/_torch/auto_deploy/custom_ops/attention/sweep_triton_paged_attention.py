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

"""Parameter sweep for Triton Paged Attention kernels.

Sweeps num_warps, num_stages, and block sizes for both decode and prefill
kernels, then outputs the best config per shape.

Usage:
    python sweep_triton_paged_attention.py
    python sweep_triton_paged_attention.py --mode prefill
    python sweep_triton_paged_attention.py --mode decode
"""

import argparse
import json
import math
import os

import torch
import triton

try:
    from triton_paged_attention import (
        _flash_decode_stage1_kernel,
        _flash_decode_stage2_kernel,
        _paged_context_kernel,
        update_paged_kv_cache,
    )
except ModuleNotFoundError:
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
        _flash_decode_stage1_kernel,
        _flash_decode_stage2_kernel,
        _paged_context_kernel,
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


def _get_num_splits(max_seq_len, batch_size, n_kv_heads, page_size):
    if max_seq_len <= 0:
        return 1
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    existing_parallelism = batch_size * n_kv_heads
    if existing_parallelism >= num_sms * 2:
        return 1
    target_blocks = num_sms * 4
    num_splits = max(1, (target_blocks + existing_parallelism - 1) // existing_parallelism)
    max_pages = max_seq_len // page_size
    max_splits = max(1, max_pages // 2)
    num_splits = min(num_splits, max_splits)
    if num_splits > 1:
        num_splits = 2 ** math.ceil(math.log2(num_splits))
    return min(num_splits, 128)


# =============================================================================
# Decode sweep
# =============================================================================

DECODE_SHAPES = [
    ("D1", 1, 512, 32, 8, 128),
    ("D2", 1, 2048, 32, 8, 128),
    ("D5", 8, 2048, 32, 8, 128),
    ("D8", 128, 2048, 32, 8, 128),
    ("D13", 1, 2048, 32, 2, 128),
    ("D14", 8, 2048, 32, 2, 128),
]

DECODE_CONFIGS = [(nw, ns) for nw in [2, 4, 8, 16] for ns in [1, 2, 3, 4]]


def bench_decode_with_config(
    q,
    kv_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    sm_scale,
    num_warps,
    num_stages,
):
    batch_size, n_heads, head_dim = q.shape
    _, _, n_kv_heads, page_size, _ = kv_cache.shape
    head_ratio = n_heads // n_kv_heads
    head_ratio_padded = max(1, 2 ** math.ceil(math.log2(head_ratio))) if head_ratio > 1 else 1
    max_pages = kv_indices.shape[0]
    max_seq_len = max_pages * page_size
    num_splits = _get_num_splits(max_seq_len, batch_size, n_kv_heads, page_size)
    output = torch.empty_like(q)
    partial_o = torch.empty(
        batch_size,
        n_heads,
        num_splits,
        head_dim,
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        batch_size,
        n_heads,
        num_splits,
        dtype=torch.float32,
        device=q.device,
    )

    # Override autotune: set configs to single config
    saved_configs = _flash_decode_stage1_kernel.configs
    _flash_decode_stage1_kernel.configs = [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
    ]
    # Reset cache to force recompile with new config
    _flash_decode_stage1_kernel.cache = {}

    def run():
        _flash_decode_stage1_kernel[(batch_size, n_kv_heads, num_splits)](
            q,
            kv_cache,
            kv_indices,
            kv_indptr,
            kv_last_page_len,
            partial_o,
            partial_lse,
            q.stride(0),
            q.stride(1),
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            SM_SCALE=sm_scale,
            N_HEADS=n_heads,
            N_KV_HEADS=n_kv_heads,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
            HEAD_RATIO=head_ratio,
            HEAD_RATIO_PADDED=head_ratio_padded,
            NUM_SPLITS=num_splits,
        )
        _flash_decode_stage2_kernel[(batch_size, n_heads)](
            partial_o,
            partial_lse,
            output,
            partial_o.stride(0),
            partial_o.stride(1),
            partial_o.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            output.stride(0),
            output.stride(1),
            HEAD_DIM=head_dim,
            NUM_SPLITS=num_splits,
        )

    # Warm up
    run()
    torch.cuda.synchronize()
    us = triton.testing.do_bench(run, warmup=WARMUP, rep=REP) * 1000

    # Restore
    _flash_decode_stage1_kernel.configs = saved_configs
    _flash_decode_stage1_kernel.cache = {}
    return us


def sweep_decode():
    print("\n" + "=" * 90)
    print("DECODE PARAMETER SWEEP")
    print("=" * 90)
    results = {}

    for sid, batch, seq_len, n_heads, n_kv_heads, head_dim in DECODE_SHAPES:
        print(f"\n--- {sid}: batch={batch} seq={seq_len} heads={n_heads}/{n_kv_heads} ---")
        q, kv_cache, kv_indices, kv_indptr, kv_last_page_len, sm_scale = make_decode_inputs(
            batch, seq_len, n_heads, n_kv_heads, head_dim
        )
        best_us = float("inf")
        best_cfg = None
        shape_results = []

        for nw, ns in DECODE_CONFIGS:
            try:
                us = bench_decode_with_config(
                    q,
                    kv_cache,
                    kv_indices,
                    kv_indptr,
                    kv_last_page_len,
                    sm_scale,
                    nw,
                    ns,
                )
                tag = " *BEST*" if us < best_us else ""
                if us < best_us:
                    best_us = us
                    best_cfg = (nw, ns)
                print(f"  warps={nw:2d} stages={ns}: {us:8.1f} us{tag}")
                shape_results.append({"warps": nw, "stages": ns, "us": us})
            except Exception as e:
                print(f"  warps={nw:2d} stages={ns}: FAILED ({e})")

        results[sid] = {
            "best_us": best_us,
            "best_warps": best_cfg[0] if best_cfg else None,
            "best_stages": best_cfg[1] if best_cfg else None,
            "all": shape_results,
        }
        print(f"  >>> Best: warps={best_cfg[0]} stages={best_cfg[1]} -> {best_us:.1f} us")

    return results


# =============================================================================
# Prefill sweep
# =============================================================================

PREFILL_SHAPES = [
    ("P1", 1, 128, 32, 8, 128),
    ("P3", 1, 2048, 32, 8, 128),
    ("P5", 4, 2048, 32, 8, 128),
    ("P8", 1, 2048, 32, 2, 128),
    ("P9", 4, 512, 32, 2, 128),
]

PREFILL_CONFIGS = [(qb, nw, ns) for qb in [32, 64, 128] for nw in [4, 8, 16] for ns in [2, 3, 4]]


def bench_prefill_with_config(
    q,
    kv_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_len,
    seq_len_with_cache,
    sm_scale,
    n_heads,
    n_kv_heads,
    head_dim,
    q_block,
    num_warps,
    num_stages,
):
    _, _, _, page_size, _ = kv_cache.shape
    num_seq = qo_indptr.shape[0] - 1
    output = torch.empty_like(q)

    q_lens = qo_indptr[1:] - qo_indptr[:-1]
    max_q_len = int(q_lens.max().item()) if num_seq > 0 else 0
    num_q_blocks = (max_q_len + q_block - 1) // q_block
    grid = (num_seq, n_heads, num_q_blocks)

    # Override autotune
    saved_configs = _paged_context_kernel.configs
    _paged_context_kernel.configs = [
        triton.Config({"Q_BLOCK": q_block}, num_warps=num_warps, num_stages=num_stages)
    ]
    _paged_context_kernel.cache = {}

    def run():
        _paged_context_kernel[grid](
            q,
            kv_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            seq_len_with_cache,
            output,
            q.stride(0),
            q.stride(1),
            output.stride(0),
            output.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            SM_SCALE=sm_scale,
            N_HEADS=n_heads,
            N_KV_HEADS=n_kv_heads,
            HEAD_DIM=head_dim,
            PAGE_SIZE=page_size,
        )

    run()
    torch.cuda.synchronize()
    us = triton.testing.do_bench(run, warmup=WARMUP, rep=REP) * 1000

    _paged_context_kernel.configs = saved_configs
    _paged_context_kernel.cache = {}
    return us


def sweep_prefill():
    print("\n" + "=" * 90)
    print("PREFILL PARAMETER SWEEP")
    print("=" * 90)
    results = {}

    for sid, batch, seq_len, n_heads, n_kv_heads, head_dim in PREFILL_SHAPES:
        print(f"\n--- {sid}: batch={batch} seq={seq_len} heads={n_heads}/{n_kv_heads} ---")
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

        best_us = float("inf")
        best_cfg = None
        shape_results = []

        for qb, nw, ns in PREFILL_CONFIGS:
            try:
                us = bench_prefill_with_config(
                    q,
                    kv_cache,
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    seq_len_with_cache,
                    sm_scale,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    qb,
                    nw,
                    ns,
                )
                tag = " *BEST*" if us < best_us else ""
                if us < best_us:
                    best_us = us
                    best_cfg = (qb, nw, ns)
                print(f"  Q_BLOCK={qb:3d} warps={nw:2d} stages={ns}: {us:8.1f} us{tag}")
                shape_results.append(
                    {
                        "Q_BLOCK": qb,
                        "warps": nw,
                        "stages": ns,
                        "us": us,
                    }
                )
            except Exception as e:
                print(f"  Q_BLOCK={qb:3d} warps={nw:2d} stages={ns}: FAILED ({e})")

        results[sid] = {
            "best_us": best_us,
            "best_Q_BLOCK": best_cfg[0] if best_cfg else None,
            "best_warps": best_cfg[1] if best_cfg else None,
            "best_stages": best_cfg[2] if best_cfg else None,
            "all": shape_results,
        }
        print(
            f"  >>> Best: Q_BLOCK={best_cfg[0]} warps={best_cfg[1]} "
            f"stages={best_cfg[2]} -> {best_us:.1f} us"
        )

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for Triton Paged Attention")
    parser.add_argument(
        "--mode",
        choices=["all", "decode", "prefill"],
        default="all",
    )
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"dtype: {DTYPE}, page_size: {PAGE_SIZE}")

    all_results = {}

    if args.mode in ("all", "decode"):
        all_results["decode"] = sweep_decode()

    if args.mode in ("all", "prefill"):
        all_results["prefill"] = sweep_prefill()

    # Save results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
