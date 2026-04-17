#!/usr/bin/env python3
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

"""Phase 1 test: QKV GEMV opcode correctness and performance.

Tests:
  1. GEMV-only correctness: megakernel QKV projection matches aten.linear
  2. Full opcode: GEMV + per-head RMSNorm + RoPE + KV cache write
  3. Benchmark: GEMV opcode latency vs cuBLAS
"""

from __future__ import annotations

import sys

import torch

HIDDEN_SIZE = 2816
HEAD_DIM = 256
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
Q_WIDTH = NUM_Q_HEADS * HEAD_DIM  # 4096
KV_WIDTH = NUM_KV_HEADS * HEAD_DIM  # 2048
QKV_WIDTH = Q_WIDTH + 2 * KV_WIDTH  # 8192
NUM_SMS = 132


def distribute_rows(total_rows: int, num_sms: int) -> list[tuple[int, int]]:
    """Distribute output rows across SMs as evenly as possible.

    Returns (row_start, row_end) per SM. Rows are assigned in contiguous
    chunks aligned to HEAD_DIM boundaries where possible.
    """
    rows_per_sm = total_rows // num_sms
    remainder = total_rows % num_sms
    ranges = []
    start = 0
    for i in range(num_sms):
        count = rows_per_sm + (1 if i < remainder else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def test_gemv_only():
    """Test 1: QKV GEMV matches torch.nn.functional.linear."""
    from launcher import InstructionBuilder, MegakernelLauncher

    print("Test 1: GEMV correctness (no norms/RoPE)...", end=" ", flush=True)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    # Model inputs
    attn_normed = torch.randn(1, HIDDEN_SIZE, device=device, dtype=dtype)
    qkv_weight = torch.randn(QKV_WIDTH, HIDDEN_SIZE, device=device, dtype=dtype)

    # Reference: standard matmul
    ref_qkv = torch.nn.functional.linear(attn_normed, qkv_weight).squeeze(0)

    # Megakernel setup
    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    builder = InstructionBuilder(num_sms=NUM_SMS)

    row_ranges = distribute_rows(QKV_WIDTH, NUM_SMS)
    all_sms = range(NUM_SMS)

    builder.add_gemv_qkv(all_sms, row_ranges, token_id=0)
    builder.add_done(all_sms)

    # Dummy metadata (not used for GEMV-only, but needed by the launcher)
    q_norm = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    k_norm = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    v_norm = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    cos_sin = torch.zeros(4096, HEAD_DIM, device=device, dtype=torch.float32)
    page_size = 16
    kv_cache = torch.zeros(16, 2, NUM_KV_HEADS, page_size, HEAD_DIM, device=device, dtype=dtype)
    cache_loc = torch.arange(16, dtype=torch.int32, device=device)
    cu_num_pages = torch.tensor([0, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([0], dtype=torch.int32, device=device)
    batch_idx = torch.tensor([0], dtype=torch.int32, device=device)
    last_page = torch.tensor([1], dtype=torch.int32, device=device)
    qkv_scratch = torch.zeros(1, QKV_WIDTH, device=device, dtype=dtype)
    attn_scratch = torch.zeros(1, Q_WIDTH, device=device, dtype=dtype)

    launcher.launch_gemv_qkv(
        builder,
        attn_normed,
        qkv_weight,
        q_norm,
        k_norm,
        v_norm,
        cos_sin,
        kv_cache,
        cache_loc,
        cu_num_pages,
        positions,
        batch_idx,
        last_page,
        qkv_scratch,
        attn_scratch,
        eps=1e-6,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    # Compare GEMV output (in qkv_scratch) with reference
    mk_qkv = qkv_scratch[0]
    max_diff = (mk_qkv.float() - ref_qkv.float()).abs().max().item()
    mean_diff = (mk_qkv.float() - ref_qkv.float()).abs().mean().item()

    # bf16 matmul tolerance
    passed = max_diff < 2.0  # bf16 accumulation has significant error for 2816-wide dot
    status = "PASS" if passed else "FAIL"
    print(f"{status}  (max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f})")
    if not passed:
        # Print more diagnostics
        print(f"  ref range: [{ref_qkv.min():.4f}, {ref_qkv.max():.4f}]")
        print(f"  mk range:  [{mk_qkv.min():.4f}, {mk_qkv.max():.4f}]")
    return passed


def test_benchmark_gemv():
    """Test 2: Benchmark GEMV latency vs cuBLAS."""
    from launcher import InstructionBuilder, MegakernelLauncher

    print("Test 2: GEMV benchmark...", flush=True)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    attn_normed = torch.randn(1, HIDDEN_SIZE, device=device, dtype=dtype)
    qkv_weight = torch.randn(QKV_WIDTH, HIDDEN_SIZE, device=device, dtype=dtype)

    # Megakernel setup
    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    builder = InstructionBuilder(num_sms=NUM_SMS)
    row_ranges = distribute_rows(QKV_WIDTH, NUM_SMS)
    builder.add_gemv_qkv(range(NUM_SMS), row_ranges, token_id=0)
    builder.add_done(range(NUM_SMS))

    # Dummy metadata
    q_norm = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    k_norm = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    v_norm = torch.ones(HEAD_DIM, device=device, dtype=torch.float32)
    cos_sin = torch.zeros(4096, HEAD_DIM, device=device, dtype=torch.float32)
    page_size = 16
    kv_cache = torch.zeros(16, 2, NUM_KV_HEADS, page_size, HEAD_DIM, device=device, dtype=dtype)
    cache_loc = torch.arange(16, dtype=torch.int32, device=device)
    cu_num_pages = torch.tensor([0, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([0], dtype=torch.int32, device=device)
    batch_idx = torch.tensor([0], dtype=torch.int32, device=device)
    last_page = torch.tensor([1], dtype=torch.int32, device=device)
    qkv_scratch = torch.zeros(1, QKV_WIDTH, device=device, dtype=dtype)
    attn_scratch_b = torch.zeros(1, Q_WIDTH, device=device, dtype=dtype)
    barrier_slots = torch.zeros(256, dtype=torch.int32, device=device)
    debug_output = torch.zeros(NUM_SMS, dtype=torch.int32, device=device)

    warmup = 20
    iters = 100

    # ── Benchmark megakernel GEMV ──
    for _ in range(warmup):
        barrier_slots.zero_()
        debug_output.zero_()
        launcher.launch_gemv_qkv(
            builder,
            attn_normed,
            qkv_weight,
            q_norm,
            k_norm,
            v_norm,
            cos_sin,
            kv_cache,
            cache_loc,
            cu_num_pages,
            positions,
            batch_idx,
            last_page,
            qkv_scratch,
            attn_scratch_b,
            eps=1e-6,
            page_size=page_size,
        )
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        barrier_slots.zero_()
        debug_output.zero_()
        qkv_scratch.zero_()
        start_events[i].record()
        launcher.launch_gemv_qkv(
            builder,
            attn_normed,
            qkv_weight,
            q_norm,
            k_norm,
            v_norm,
            cos_sin,
            kv_cache,
            cache_loc,
            cu_num_pages,
            positions,
            batch_idx,
            last_page,
            qkv_scratch,
            attn_scratch_b,
            eps=1e-6,
            page_size=page_size,
        )
        end_events[i].record()
    torch.cuda.synchronize()
    mk_times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events))
    mk_median = mk_times[len(mk_times) // 2]

    # ── Benchmark cuBLAS (aten.linear) ──
    for _ in range(warmup):
        torch.nn.functional.linear(attn_normed, qkv_weight)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        torch.nn.functional.linear(attn_normed, qkv_weight)
        end_events[i].record()
    torch.cuda.synchronize()
    cublas_times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events))
    cublas_median = cublas_times[len(cublas_times) // 2]

    bw_theoretical = QKV_WIDTH * HIDDEN_SIZE * 2 / (3.35e12) * 1e6  # µs
    print(f"  Megakernel GEMV:     {mk_median:>8.2f} us")
    print(f"  cuBLAS linear:       {cublas_median:>8.2f} us")
    print(f"  Theoretical (BW):    {bw_theoretical:>8.2f} us")
    print(f"  Ratio (mk/cuBLAS):   {mk_median / cublas_median:>8.2f}x")


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm matching the Triton kernel."""
    x_f32 = x.float()
    var = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def apply_rope_ref(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Reference RoPE: split head_dim in half, rotate."""
    half = q.shape[-1] // 2
    q_first, q_second = q[..., :half].float(), q[..., half:].float()
    k_first, k_second = k[..., :half].float(), k[..., half:].float()
    q_rope = torch.cat(
        [
            q_first * cos - q_second * sin,
            q_second * cos + q_first * sin,
        ],
        dim=-1,
    ).to(q.dtype)
    k_rope = torch.cat(
        [
            k_first * cos - k_second * sin,
            k_second * cos + k_first * sin,
        ],
        dim=-1,
    ).to(k.dtype)
    return q_rope, k_rope


def test_full_qkv_opcode():
    """Test 3: Full QKV opcode with real norms, RoPE, and KV cache writes."""
    from launcher import InstructionBuilder, MegakernelLauncher

    print("Test 3: Full QKV opcode (norms + RoPE + cache)...", flush=True)

    torch.manual_seed(123)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    page_size = 16
    past_length = 128  # tokens already in cache
    position = past_length  # decode position

    # Model weights
    attn_normed = torch.randn(1, HIDDEN_SIZE, device=device, dtype=dtype)
    qkv_weight = torch.randn(QKV_WIDTH, HIDDEN_SIZE, device=device, dtype=dtype) * 0.02
    q_norm_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.float32).abs() + 0.5
    k_norm_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.float32).abs() + 0.5
    v_norm_weight = torch.randn(HEAD_DIM, device=device, dtype=torch.float32).abs() + 0.5

    # RoPE cos/sin cache — stride-1 layout [max_pos, head_dim]
    max_pos = 4096
    cos_sin_cache = torch.randn(max_pos, HEAD_DIM, device=device, dtype=torch.float32) * 0.1
    half_d = HEAD_DIM // 2
    # cos = cos_sin_cache[:, :half_d], sin = cos_sin_cache[:, half_d:]
    cos_vals = cos_sin_cache[position, :half_d]  # [half_d]
    sin_vals = cos_sin_cache[position, half_d:]  # [half_d]

    # KV cache setup
    total_length = past_length + 1
    num_pages = (total_length + page_size - 1) // page_size
    kv_cache_mk = torch.zeros(
        num_pages + 4, 2, NUM_KV_HEADS, page_size, HEAD_DIM, device=device, dtype=dtype
    )
    kv_cache_ref = kv_cache_mk.clone()
    cache_loc = torch.arange(num_pages + 4, dtype=torch.int32, device=device)
    cu_num_pages = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [((total_length - 1) % page_size) + 1], dtype=torch.int32, device=device
    )
    triton_positions = torch.tensor([position], dtype=torch.int32, device=device)
    triton_batch_indices = torch.tensor([0], dtype=torch.int32, device=device)

    # ── Reference computation ──
    # 1. QKV projection
    ref_qkv = torch.nn.functional.linear(attn_normed, qkv_weight)  # [1, 8192]
    ref_q = ref_qkv[..., :Q_WIDTH].reshape(1, NUM_Q_HEADS, HEAD_DIM)
    ref_k = ref_qkv[..., Q_WIDTH : Q_WIDTH + KV_WIDTH].reshape(1, NUM_KV_HEADS, HEAD_DIM)
    ref_v = ref_qkv[..., Q_WIDTH + KV_WIDTH :].reshape(1, NUM_KV_HEADS, HEAD_DIM)

    # 2. Per-head RMSNorm
    ref_q_norm = rms_norm_ref(ref_q, q_norm_weight, eps)
    ref_k_norm = rms_norm_ref(ref_k, k_norm_weight, eps)
    ref_v_norm = rms_norm_ref(ref_v, v_norm_weight, eps)

    # 3. RoPE on Q and K
    ref_q_rope, ref_k_rope = apply_rope_ref(ref_q_norm, ref_k_norm, cos_vals, sin_vals)

    # 4. Write K and V to reference cache
    page_idx = position // page_size
    tok_in_page = position % page_size
    phys_page = cache_loc[page_idx].item()
    for h in range(NUM_KV_HEADS):
        kv_cache_ref[phys_page, 0, h, tok_in_page] = ref_k_rope[0, h]
        kv_cache_ref[phys_page, 1, h, tok_in_page] = ref_v_norm[0, h]

    # ── Megakernel computation ──
    # Instruction stream: GEMV (all SMs) → barrier → POST (32 SMs) → DONE
    qkv_scratch = torch.zeros(1, QKV_WIDTH, device=device, dtype=dtype)
    attn_scratch = torch.zeros(1, Q_WIDTH, device=device, dtype=dtype)

    TOTAL_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 32
    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    builder = InstructionBuilder(num_sms=NUM_SMS)
    all_sms = range(NUM_SMS)
    post_sms = range(TOTAL_HEADS)  # 32 SMs for post-processing

    # Phase 1a: Raw GEMV across all 132 SMs
    row_ranges = distribute_rows(QKV_WIDTH, NUM_SMS)
    builder.add_gemv_qkv(all_sms, row_ranges, token_id=0)

    # Barrier: wait for all GEMV writes to complete
    builder.add_barrier(all_sms, barrier_id=0)

    # Phase 1b: Post-processing on 32 SMs (one head per SM)
    head_ranges = [(h, h + 1) for h in range(TOTAL_HEADS)]
    builder.add_qkv_post(post_sms, head_ranges, token_id=0)

    # DONE for all SMs
    builder.add_done(all_sms)

    launcher.launch_gemv_qkv(
        builder,
        attn_normed,
        qkv_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        cos_sin_cache,
        kv_cache_mk,
        cache_loc,
        cu_num_pages,
        triton_positions,
        triton_batch_indices,
        last_page_len,
        qkv_scratch,
        attn_scratch,
        eps=eps,
        page_size=page_size,
    )
    torch.cuda.synchronize()

    # ── Compare Q (in qkv_scratch, after norms + RoPE) ──
    mk_q = qkv_scratch[0, :Q_WIDTH].reshape(NUM_Q_HEADS, HEAD_DIM)
    ref_q_flat = ref_q_rope[0]  # [16, 256]
    q_diff = (mk_q.float() - ref_q_flat.float()).abs()
    q_max_diff = q_diff.max().item()
    q_mean_diff = q_diff.mean().item()
    q_pass = q_max_diff < 0.5  # bf16 tolerance for chained ops

    # ── Compare K in KV cache ──
    mk_k = kv_cache_mk[phys_page, 0, :, tok_in_page]  # [8, 256]
    ref_k_cached = kv_cache_ref[phys_page, 0, :, tok_in_page]
    k_diff = (mk_k.float() - ref_k_cached.float()).abs()
    k_max_diff = k_diff.max().item()
    k_pass = k_max_diff < 0.5

    # ── Compare V in KV cache ──
    mk_v = kv_cache_mk[phys_page, 1, :, tok_in_page]  # [8, 256]
    ref_v_cached = kv_cache_ref[phys_page, 1, :, tok_in_page]
    v_diff = (mk_v.float() - ref_v_cached.float()).abs()
    v_max_diff = v_diff.max().item()
    v_pass = v_max_diff < 0.5

    all_pass = q_pass and k_pass and v_pass
    print(
        f"  Q: max_diff={q_max_diff:.4f} mean_diff={q_mean_diff:.6f} {'PASS' if q_pass else 'FAIL'}"
    )
    print(f"  K: max_diff={k_max_diff:.4f} {'PASS' if k_pass else 'FAIL'}")
    print(f"  V: max_diff={v_max_diff:.4f} {'PASS' if v_pass else 'FAIL'}")
    return all_pass


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)

    print("=" * 60)
    print("Gemma4 Megakernel Phase 1: QKV GEMV Tests")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Shapes: hidden={HIDDEN_SIZE}, qkv_width={QKV_WIDTH}")
    print("=" * 60)

    p1 = test_gemv_only()
    test_benchmark_gemv()
    p3 = test_full_qkv_opcode()

    all_pass = p1 and p3
    print("=" * 60)
    if all_pass:
        print("All Phase 1 tests PASSED")
    else:
        print("Phase 1 tests FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
