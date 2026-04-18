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

"""Hybrid QKV tensor-core benchmark for Gemma4 Kernel A decode.

This is an upper-bound experiment:
- QKV projection runs through DeepGEMM BF16 tensor cores on the host side.
- The persistent megakernel still handles QKV_POST, attention, O-proj, and post ops.

It does not preserve the single-kernel property. The point is to measure how much
full-kernel latency is available from a clean QKV tensor-core replacement before
rewriting the persistent opcode path.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F
from test_kernel_a import (
    GQA,
    NKV,
    NQ,
    NUM_SMS,
    QKVW,
    QW,
    TH,
    D,
    H,
    dist_rows,
    rms_norm,
    rope,
    test_benchmark_kernel_a,
)

from tensorrt_llm import deep_gemm


def _bench_us(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _build_tail_instructions(total_pages: int):
    from launcher import (
        InstructionBuilder,
        choose_attention_num_partials,
        partition_attention_pages,
    )

    builder = InstructionBuilder(num_sms=NUM_SMS)
    all_sms = range(NUM_SMS)

    builder.add_qkv_post(range(TH), [(h, h + 1) for h in range(TH)], token_id=0)
    builder.add_barrier(all_sms, barrier_id=0)

    num_pp = choose_attention_num_partials(total_pages)
    use_multi = num_pp > 1
    if use_multi:
        sm_ids, kv_heads, page_ranges, partial_ids = [], [], [], []
        sm_counter = 0
        ranges = partition_attention_pages(total_pages, num_pp)
        for kv_h in range(NKV):
            for pi, (ps, pe) in enumerate(ranges):
                sm_ids.append(sm_counter)
                kv_heads.append(kv_h)
                page_ranges.append((ps, pe))
                partial_ids.append(kv_h * num_pp + pi)
                sm_counter += 1
        builder.add_paged_attn(
            sm_ids,
            kv_heads,
            token_id=0,
            page_ranges=page_ranges,
            partial_indices=partial_ids,
            is_single=False,
        )
        builder.add_barrier(all_sms, barrier_id=1)
        partial_starts = [kv_h * num_pp for kv_h in range(NKV)]
        builder.add_attn_reduce(
            range(NKV),
            list(range(NKV)),
            num_pp,
            partial_starts=partial_starts,
            token_id=0,
        )
        builder.add_barrier(all_sms, barrier_id=2)
        next_bid = 3
        max_partials = NKV * num_pp
    else:
        builder.add_paged_attn(range(NKV), list(range(NKV)), token_id=0)
        builder.add_barrier(all_sms, barrier_id=1)
        next_bid = 2
        max_partials = 0

    builder.add_gemv_oproj(all_sms, dist_rows(H, NUM_SMS), token_id=0)
    builder.add_barrier(all_sms, barrier_id=next_bid)
    builder.add_oproj_post(0, token_id=0)
    builder.add_done(all_sms)
    return builder, max_partials


def _make_setup(past_length: int, page_size: int = 16) -> dict[str, torch.Tensor | float | int]:
    from launcher import MegakernelLauncher

    torch.manual_seed(42 + past_length)
    dev, dt = "cuda", torch.bfloat16
    eps = 1e-6
    scale = 1.0 / math.sqrt(D)
    pos = past_length
    total_len = past_length + 1
    num_pages = (total_len + page_size - 1) // page_size

    attn_normed = torch.randn(1, H, device=dev, dtype=dt) * 0.1
    residual_in = torch.randn(1, H, device=dev, dtype=dt) * 0.1
    qkv_w = torch.randn(QKVW, H, device=dev, dtype=dt) * 0.02
    oproj_w = torch.randn(H, QW, device=dev, dtype=dt) * 0.02
    qnw = torch.randn(D, device=dev, dtype=torch.float32).abs() + 0.5
    knw = torch.randn(D, device=dev, dtype=torch.float32).abs() + 0.5
    vnw = torch.randn(D, device=dev, dtype=torch.float32).abs() + 0.5
    panw = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    pfnw = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    cs_cache = torch.randn(4096, D, device=dev, dtype=torch.float32) * 0.1

    kv_cache = torch.zeros(num_pages + 4, 2, NKV, page_size, D, device=dev, dtype=dt)
    cache_loc = torch.arange(num_pages + 4, dtype=torch.int32, device=dev)
    cu_np = torch.tensor([0, num_pages], dtype=torch.int32, device=dev)
    last_page = torch.tensor([((total_len - 1) % page_size) + 1], dtype=torch.int32, device=dev)
    positions = torch.tensor([pos], dtype=torch.int32, device=dev)
    batch_idx = torch.tensor([0], dtype=torch.int32, device=dev)

    past_k = torch.randn(past_length, NKV, D, device=dev, dtype=dt) * 0.1
    past_v = torch.randn(past_length, NKV, D, device=dev, dtype=dt) * 0.1
    for t in range(past_length):
        page, offset = t // page_size, t % page_size
        for h in range(NKV):
            kv_cache[cache_loc[page], 0, h, offset] = past_k[t, h]
            kv_cache[cache_loc[page], 1, h, offset] = past_v[t, h]

    qkv_scratch = torch.empty(1, QKVW, device=dev, dtype=dt)
    attn_scratch = torch.empty(1, QW, device=dev, dtype=dt)
    oproj_scratch = torch.empty(1, H, device=dev, dtype=torch.float32)
    post_out = torch.empty(1, H, device=dev, dtype=torch.float32)
    pre_out = torch.empty(1, H, device=dev, dtype=torch.float32)

    builder, max_partials = _build_tail_instructions(num_pages)
    partial_scratch = (
        torch.zeros(max_partials, NQ, D + 2, device=dev, dtype=torch.float32)
        if max_partials > 0
        else None
    )

    return {
        "launcher": MegakernelLauncher(num_sms=NUM_SMS),
        "builder": builder,
        "attn_normed": attn_normed,
        "residual_in": residual_in,
        "qkv_w": qkv_w,
        "oproj_w": oproj_w,
        "qnw": qnw,
        "knw": knw,
        "vnw": vnw,
        "panw": panw,
        "pfnw": pfnw,
        "cs_cache": cs_cache,
        "kv_cache": kv_cache,
        "cache_loc": cache_loc,
        "cu_np": cu_np,
        "last_page": last_page,
        "positions": positions,
        "batch_idx": batch_idx,
        "qkv_scratch": qkv_scratch,
        "attn_scratch": attn_scratch,
        "oproj_scratch": oproj_scratch,
        "post_out": post_out,
        "pre_out": pre_out,
        "partial_scratch": partial_scratch,
        "eps": eps,
        "page_size": page_size,
        "scale": scale,
    }


def _run_hybrid(setup: dict[str, torch.Tensor | float | int]) -> None:
    deep_gemm.bf16_gemm_nt(setup["attn_normed"], setup["qkv_w"], setup["qkv_scratch"])
    _run_tail_only(setup)


def _run_tail_only(setup: dict[str, torch.Tensor | float | int]) -> None:
    setup["launcher"].launch_gemv_qkv(
        setup["builder"],
        setup["attn_normed"],
        setup["qkv_w"],
        setup["qnw"],
        setup["knw"],
        setup["vnw"],
        setup["cs_cache"],
        setup["kv_cache"],
        setup["cache_loc"],
        setup["cu_np"],
        setup["positions"],
        setup["batch_idx"],
        setup["last_page"],
        setup["qkv_scratch"],
        setup["attn_scratch"],
        eps=setup["eps"],
        page_size=setup["page_size"],
        attn_scale=setup["scale"],
        o_proj_weight=setup["oproj_w"],
        residual=setup["residual_in"],
        post_attn_norm_weight=setup["panw"],
        pre_ffn_norm_weight=setup["pfnw"],
        o_proj_scratch=setup["oproj_scratch"],
        post_attn_out=setup["post_out"],
        pre_ffn_out=setup["pre_out"],
        partial_attn_scratch=setup["partial_scratch"],
    )


def test_hybrid_qkv_correctness(past_length: int = 128, page_size: int = 16) -> tuple[float, float]:
    setup = _make_setup(past_length, page_size)
    pos = past_length
    half_d = D // 2

    # Reference
    ref_qkv = F.linear(setup["attn_normed"], setup["qkv_w"])
    ref_q = rms_norm(ref_qkv[..., :QW].reshape(1, NQ, D), setup["qnw"], setup["eps"])
    ref_k = rms_norm(
        ref_qkv[..., QW : QW + (NKV * D)].reshape(1, NKV, D), setup["knw"], setup["eps"]
    )
    ref_v = rms_norm(ref_qkv[..., QW + (NKV * D) :].reshape(1, NKV, D), setup["vnw"], setup["eps"])
    cos_v = setup["cs_cache"][pos, :half_d]
    sin_v = setup["cs_cache"][pos, half_d:]
    ref_q, ref_k = rope(ref_q, ref_k, cos_v, sin_v)

    kvc_ref = setup["kv_cache"].clone()
    page_idx = pos // page_size
    token_offset = pos % page_size
    for h in range(NKV):
        kvc_ref[setup["cache_loc"][page_idx], 0, h, token_offset] = ref_k[0, h]
        kvc_ref[setup["cache_loc"][page_idx], 1, h, token_offset] = ref_v[0, h]

    ref_attn = torch.zeros(1, NQ, D, device="cuda", dtype=torch.float32)
    total_len = past_length + 1
    for kh in range(NKV):
        k_all = torch.zeros(total_len, D, device="cuda", dtype=torch.bfloat16)
        v_all = torch.zeros(total_len, D, device="cuda", dtype=torch.bfloat16)
        for t in range(total_len):
            page, offset = t // page_size, t % page_size
            k_all[t] = kvc_ref[setup["cache_loc"][page], 0, kh, offset]
            v_all[t] = kvc_ref[setup["cache_loc"][page], 1, kh, offset]
        for qi in range(GQA):
            qh = kh * GQA + qi
            scores = (k_all.float() @ ref_q[0, qh].float()) * setup["scale"]
            probs = torch.softmax(scores, 0)
            ref_attn[0, qh] = (probs.unsqueeze(1) * v_all.float()).sum(0)

    ref_oproj = F.linear(ref_attn.to(torch.bfloat16).reshape(1, QW), setup["oproj_w"]).float()
    ref_post = (
        setup["residual_in"].float()
        + rms_norm(ref_oproj.to(torch.bfloat16), setup["panw"], setup["eps"]).float()
    )
    ref_pre = rms_norm(ref_post.to(torch.bfloat16), setup["pfnw"], setup["eps"]).float()

    _run_hybrid(setup)
    torch.cuda.synchronize()
    post_max = (setup["post_out"] - ref_post).abs().max().item()
    pre_max = (setup["pre_out"] - ref_pre).abs().max().item()
    return post_max, pre_max


def benchmark_hybrid_qkv(past_length: int = 128, page_size: int = 16) -> tuple[float, float]:
    setup = _make_setup(past_length, page_size)

    qkv_only_out = torch.empty((1, QKVW), device="cuda", dtype=torch.bfloat16)

    def run_qkv_tc() -> None:
        deep_gemm.bf16_gemm_nt(setup["attn_normed"], setup["qkv_w"], qkv_only_out)

    deep_gemm.bf16_gemm_nt(setup["attn_normed"], setup["qkv_w"], setup["qkv_scratch"])
    hybrid_us = _bench_us(lambda: _run_hybrid(setup))
    tail_us = _bench_us(lambda: _run_tail_only(setup))
    qkv_tc_us = _bench_us(run_qkv_tc)
    return hybrid_us, qkv_tc_us, tail_us


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit(0)

    print("=" * 72)
    print("Gemma4 Megakernel: Hybrid QKV Tensor-Core Upper Bound")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)

    post_max, pre_max = test_hybrid_qkv_correctness(past_length=128)
    print(
        f"  correctness seq=129: post_attn max_diff={post_max:.4f}  pre_ffn max_diff={pre_max:.4f}"
    )

    print()
    for past in [128, 512, 1024]:
        baseline_us = test_benchmark_kernel_a(past_length=past)
        hybrid_us, qkv_tc_us, tail_us = benchmark_hybrid_qkv(past_length=past)
        print(
            f"  seq={past + 1:>4d}: baseline={baseline_us:.1f} us  "
            f"hybrid_qkv_tc={hybrid_us:.1f} us  qkv_tc_only={qkv_tc_us:.1f} us  "
            f"tail_only={tail_us:.1f} us"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
