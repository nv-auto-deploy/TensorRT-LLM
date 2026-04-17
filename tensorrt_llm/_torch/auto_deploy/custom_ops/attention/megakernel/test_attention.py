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

"""Phase 2 test: Paged attention opcode correctness.

Tests the full QKV + Attention pipeline:
  1. GEMV → barrier → QKV_POST → barrier → PAGED_ATTN → DONE
  Compare attention output against PyTorch SDPA reference.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

HIDDEN_SIZE = 2816
HEAD_DIM = 256
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
GQA_RATIO = NUM_Q_HEADS // NUM_KV_HEADS
Q_WIDTH = NUM_Q_HEADS * HEAD_DIM
KV_WIDTH = NUM_KV_HEADS * HEAD_DIM
QKV_WIDTH = Q_WIDTH + 2 * KV_WIDTH
NUM_SMS = 132
TOTAL_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS  # 32


def distribute_rows(total_rows: int, num_sms: int) -> list[tuple[int, int]]:
    rows_per_sm = total_rows // num_sms
    remainder = total_rows % num_sms
    ranges = []
    start = 0
    for i in range(num_sms):
        count = rows_per_sm + (1 if i < remainder else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def rms_norm_ref(x, weight, eps=1e-6):
    x_f32 = x.float()
    var = (x_f32 * x_f32).mean(dim=-1, keepdim=True)
    return (x_f32 * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


def apply_rope_ref(q, k, cos, sin):
    half = q.shape[-1] // 2
    qf, qs = q[..., :half].float(), q[..., half:].float()
    kf, ks = k[..., :half].float(), k[..., half:].float()
    q_r = torch.cat([qf * cos - qs * sin, qs * cos + qf * sin], dim=-1).to(q.dtype)
    k_r = torch.cat([kf * cos - ks * sin, ks * cos + kf * sin], dim=-1).to(k.dtype)
    return q_r, k_r


def test_paged_attention():
    """Full pipeline: QKV → norms → RoPE → cache → attention vs SDPA reference."""
    from launcher import InstructionBuilder, MegakernelLauncher

    torch.manual_seed(777)
    device = "cuda"
    dtype = torch.bfloat16
    eps = 1e-6
    page_size = 16
    past_length = 64
    position = past_length
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # Model weights
    attn_normed = torch.randn(1, HIDDEN_SIZE, device=device, dtype=dtype) * 0.1
    qkv_weight = torch.randn(QKV_WIDTH, HIDDEN_SIZE, device=device, dtype=dtype) * 0.02
    q_norm_w = torch.randn(HEAD_DIM, device=device, dtype=torch.float32).abs() + 0.5
    k_norm_w = torch.randn(HEAD_DIM, device=device, dtype=torch.float32).abs() + 0.5
    v_norm_w = torch.randn(HEAD_DIM, device=device, dtype=torch.float32).abs() + 0.5
    max_pos = 4096
    cos_sin_cache = torch.randn(max_pos, HEAD_DIM, device=device, dtype=torch.float32) * 0.1
    half_d = HEAD_DIM // 2

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

    # ── Reference: build KV cache with past tokens ──
    # Fill cache with random past K/V (pre-RoPE'd, pre-normed — as if already stored)
    past_k = torch.randn(past_length, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) * 0.1
    past_v = torch.randn(past_length, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype) * 0.1
    for t in range(past_length):
        pg = t // page_size
        off = t % page_size
        phys = cache_loc[pg].item()
        for h in range(NUM_KV_HEADS):
            kv_cache_ref[phys, 0, h, off] = past_k[t, h]
            kv_cache_ref[phys, 1, h, off] = past_v[t, h]
    kv_cache_mk.copy_(kv_cache_ref)

    # ── Reference: compute current token QKV + norms + RoPE ──
    ref_qkv = F.linear(attn_normed, qkv_weight)
    ref_q = ref_qkv[..., :Q_WIDTH].reshape(1, NUM_Q_HEADS, HEAD_DIM)
    ref_k = ref_qkv[..., Q_WIDTH : Q_WIDTH + KV_WIDTH].reshape(1, NUM_KV_HEADS, HEAD_DIM)
    ref_v = ref_qkv[..., Q_WIDTH + KV_WIDTH :].reshape(1, NUM_KV_HEADS, HEAD_DIM)

    ref_q_norm = rms_norm_ref(ref_q, q_norm_w, eps)
    ref_k_norm = rms_norm_ref(ref_k, k_norm_w, eps)
    ref_v_norm = rms_norm_ref(ref_v, v_norm_w, eps)

    cos_vals = cos_sin_cache[position, :half_d]
    sin_vals = cos_sin_cache[position, half_d:]
    ref_q_rope, ref_k_rope = apply_rope_ref(ref_q_norm, ref_k_norm, cos_vals, sin_vals)

    # Write current token to ref cache
    pg_idx = position // page_size
    tok_off = position % page_size
    phys = cache_loc[pg_idx].item()
    for h in range(NUM_KV_HEADS):
        kv_cache_ref[phys, 0, h, tok_off] = ref_k_rope[0, h]
        kv_cache_ref[phys, 1, h, tok_off] = ref_v_norm[0, h]

    # ── Reference: attention via SDPA ──
    # Build full K/V sequences from cache for each KV head
    ref_attn_out = torch.zeros(1, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.float32)

    for kv_h in range(NUM_KV_HEADS):
        # Collect all K and V tokens from cache
        k_all = torch.zeros(total_length, HEAD_DIM, device=device, dtype=dtype)
        v_all = torch.zeros(total_length, HEAD_DIM, device=device, dtype=dtype)
        for t in range(total_length):
            pg = t // page_size
            off = t % page_size
            p = cache_loc[pg].item()
            k_all[t] = kv_cache_ref[p, 0, kv_h, off]
            v_all[t] = kv_cache_ref[p, 1, kv_h, off]

        # For each Q head in the GQA group
        for qi in range(GQA_RATIO):
            q_h = kv_h * GQA_RATIO + qi
            q_vec = ref_q_rope[0, q_h].float()  # [HEAD_DIM]

            # Compute attention scores
            scores = (k_all.float() @ q_vec) * scale  # [total_length]
            probs = torch.softmax(scores, dim=0)  # [total_length]
            attn = (probs.unsqueeze(1) * v_all.float()).sum(dim=0)  # [HEAD_DIM]
            ref_attn_out[0, q_h] = attn

    # ── Megakernel: GEMV → barrier → QKV_POST → barrier → PAGED_ATTN → DONE ──
    qkv_scratch = torch.zeros(1, QKV_WIDTH, device=device, dtype=dtype)
    attn_scratch = torch.zeros(1, Q_WIDTH, device=device, dtype=dtype)

    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    builder = InstructionBuilder(num_sms=NUM_SMS)
    all_sms = range(NUM_SMS)
    post_sms = range(TOTAL_HEADS)
    attn_sms = range(NUM_KV_HEADS)

    # Phase 1a: GEMV
    row_ranges = distribute_rows(QKV_WIDTH, NUM_SMS)
    builder.add_gemv_qkv(all_sms, row_ranges, token_id=0)
    builder.add_barrier(all_sms, barrier_id=0)

    # Phase 1b: QKV post-processing
    head_ranges = [(h, h + 1) for h in range(TOTAL_HEADS)]
    builder.add_qkv_post(post_sms, head_ranges, token_id=0)
    builder.add_barrier(all_sms, barrier_id=1)

    # Phase 2: Paged attention (8 SMs, one per KV head)
    builder.add_paged_attn(attn_sms, list(range(NUM_KV_HEADS)), token_id=0)

    builder.add_done(all_sms)

    launcher.launch_gemv_qkv(
        builder,
        attn_normed,
        qkv_weight,
        q_norm_w,
        k_norm_w,
        v_norm_w,
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
        attn_scale=scale,
    )
    torch.cuda.synchronize()

    # ── Compare attention output ──
    mk_attn = attn_scratch[0].reshape(NUM_Q_HEADS, HEAD_DIM).float()
    ref_attn = ref_attn_out[0].float()

    diff = (mk_attn - ref_attn).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Per-head breakdown
    print(f"  Attention output (seq={total_length}):")
    print(f"    max_diff={max_diff:.6f}  mean_diff={mean_diff:.8f}")

    per_head_max = diff.amax(dim=-1)
    for h in range(NUM_Q_HEADS):
        status = "OK" if per_head_max[h] < 0.05 else "!!"
        print(f"    Q head {h:2d}: max_diff={per_head_max[h]:.6f} {status}")

    passed = max_diff < 0.05  # bf16 attention tolerance
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def test_benchmark_attention():
    """Benchmark the attention opcode."""

    print("  Benchmark: measuring attention opcode latency...", flush=True)
    # For now, just print the SDPA baseline
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    seq_len = 129  # past + current
    scale = 1.0 / math.sqrt(HEAD_DIM)

    q = torch.randn(1, NUM_Q_HEADS, 1, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(1, NUM_Q_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(1, NUM_Q_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)

    warmup = 20
    iters = 100
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, scale=scale)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        F.scaled_dot_product_attention(q, k, v, scale=scale)
        end_events[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events))
    print(f"  PyTorch SDPA (seq={seq_len}): {times[len(times) // 2]:.2f} us")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(0)

    print("=" * 60)
    print("Gemma4 Megakernel Phase 2: Paged Attention Tests")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 60)

    passed = test_paged_attention()
    test_benchmark_attention()

    print("=" * 60)
    print(f"Phase 2: {'PASS' if passed else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
