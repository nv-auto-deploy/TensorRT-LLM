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

"""Full Kernel A end-to-end test: all opcodes assembled.

Instruction stream:
  132 SMs: GEMV_QKV      → raw QKV projection to scratch
           BARRIER 0
   32 SMs: QKV_POST      → norms + RoPE + cache write
           BARRIER 1
    8 SMs: PAGED_ATTN    → decode attention
           BARRIER 2
  132 SMs: GEMV_OPROJ    → O-projection to scratch
           BARRIER 3
    1 SM:  OPROJ_POST    → norms + residual + pre-FFN norm
           DONE

Compare post_attn_out and pre_ffn_out against decomposed reference.
Also benchmark end-to-end latency vs decomposed.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

H = 2816
D = 256
NQ = 16
NKV = 8
GQA = 2
QW = NQ * D
KVW = NKV * D
QKVW = QW + 2 * KVW
NUM_SMS = 132
TH = NQ + 2 * NKV  # 32 total heads


def dist_rows(n, sms):
    r = []
    s = 0
    for i in range(sms):
        c = n // sms + (1 if i < n % sms else 0)
        r.append((s, s + c))
        s += c
    return r


def rms_norm(x, w, eps=1e-6):
    f = x.float()
    return (f * torch.rsqrt((f * f).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def rope(q, k, cos, sin):
    h = q.shape[-1] // 2
    qf, qs = q[..., :h].float(), q[..., h:].float()
    kf, ks = k[..., :h].float(), k[..., h:].float()
    return (
        torch.cat([qf * cos - qs * sin, qs * cos + qf * sin], -1).to(q.dtype),
        torch.cat([kf * cos - ks * sin, ks * cos + kf * sin], -1).to(k.dtype),
    )


def test_full_kernel_a(past_length=128, page_size=16):
    from launcher import InstructionBuilder, MegakernelLauncher

    torch.manual_seed(42 + past_length)
    dev, dt = "cuda", torch.bfloat16
    eps = 1e-6
    scale = 1.0 / math.sqrt(D)
    pos = past_length

    # Weights
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
    half_d = D // 2

    # KV cache
    tl = past_length + 1
    np_ = (tl + page_size - 1) // page_size
    kvc_mk = torch.zeros(np_ + 4, 2, NKV, page_size, D, device=dev, dtype=dt)
    cache_loc = torch.arange(np_ + 4, dtype=torch.int32, device=dev)
    cu_np = torch.tensor([0, np_], dtype=torch.int32, device=dev)
    lpl = torch.tensor([((tl - 1) % page_size) + 1], dtype=torch.int32, device=dev)
    tpos = torch.tensor([pos], dtype=torch.int32, device=dev)
    tbat = torch.tensor([0], dtype=torch.int32, device=dev)

    # Fill past KV cache with random data
    pk = torch.randn(past_length, NKV, D, device=dev, dtype=dt) * 0.1
    pv = torch.randn(past_length, NKV, D, device=dev, dtype=dt) * 0.1
    for t in range(past_length):
        pg, off = t // page_size, t % page_size
        for h in range(NKV):
            kvc_mk[cache_loc[pg], 0, h, off] = pk[t, h]
            kvc_mk[cache_loc[pg], 1, h, off] = pv[t, h]
    kvc_ref = kvc_mk.clone()

    # ── Reference ──
    qkv = F.linear(attn_normed, qkv_w)
    rq = rms_norm(qkv[..., :QW].reshape(1, NQ, D), qnw, eps)
    rk = rms_norm(qkv[..., QW : QW + KVW].reshape(1, NKV, D), knw, eps)
    rv = rms_norm(qkv[..., QW + KVW :].reshape(1, NKV, D), vnw, eps)
    cos_v, sin_v = cs_cache[pos, :half_d], cs_cache[pos, half_d:]
    rq_r, rk_r = rope(rq, rk, cos_v, sin_v)

    pi, to = pos // page_size, pos % page_size
    for h in range(NKV):
        kvc_ref[cache_loc[pi], 0, h, to] = rk_r[0, h]
        kvc_ref[cache_loc[pi], 1, h, to] = rv[0, h]

    # Reference attention
    ref_attn = torch.zeros(1, NQ, D, device=dev, dtype=torch.float32)
    for kh in range(NKV):
        k_all = torch.zeros(tl, D, device=dev, dtype=dt)
        v_all = torch.zeros(tl, D, device=dev, dtype=dt)
        for t in range(tl):
            pg, off = t // page_size, t % page_size
            k_all[t] = kvc_ref[cache_loc[pg], 0, kh, off]
            v_all[t] = kvc_ref[cache_loc[pg], 1, kh, off]
        for qi in range(GQA):
            qh = kh * GQA + qi
            scores = (k_all.float() @ rq_r[0, qh].float()) * scale
            probs = torch.softmax(scores, 0)
            ref_attn[0, qh] = (probs.unsqueeze(1) * v_all.float()).sum(0)

    ref_oproj = F.linear(ref_attn.to(dt).reshape(1, QW), oproj_w).float()
    ref_oproj_norm = rms_norm(ref_oproj.to(dt), panw, eps).float()
    ref_post = residual_in.float() + ref_oproj_norm
    ref_pre = rms_norm(ref_post.to(dt), pfnw, eps).float()

    # ── Megakernel ──
    qkv_scr = torch.zeros(1, QKVW, device=dev, dtype=dt)
    attn_scr = torch.zeros(1, QW, device=dev, dtype=dt)
    oproj_scr = torch.zeros(1, H, device=dev, dtype=torch.float32)
    post_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    pre_out = torch.zeros(1, H, device=dev, dtype=torch.float32)

    # Multi-SM attention scheduling: use multi-SM for all sequences > 2 pages
    # Single-SM attention wastes 90% of SM bandwidth (only 2/20 warps active).
    # Multi-SM distributes pages across SMs, then reduces via LSE.
    total_pages = (tl + page_size - 1) // page_size
    if total_pages > 2:
        # Target: 2-4 pages per SM-partition
        num_partials_per_head = min((total_pages + 1) // 2, 16)
    else:
        num_partials_per_head = 1
    use_multi_sm = num_partials_per_head > 1
    total_attn_sms = NKV * num_partials_per_head
    max_partials = total_attn_sms
    partial_scr = (
        torch.zeros(max_partials, NQ, D + 2, device=dev, dtype=torch.float32)
        if use_multi_sm
        else None
    )

    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    builder = InstructionBuilder(num_sms=NUM_SMS)
    all_sms = range(NUM_SMS)

    # Phase 1a: GEMV QKV
    builder.add_gemv_qkv(all_sms, dist_rows(QKVW, NUM_SMS), token_id=0)
    builder.add_barrier(all_sms, barrier_id=0)

    # Phase 1b: QKV post
    builder.add_qkv_post(range(TH), [(h, h + 1) for h in range(TH)], token_id=0)
    builder.add_barrier(all_sms, barrier_id=1)

    # Phase 2: Attention
    if use_multi_sm:
        attn_sm_ids = []
        attn_kv_heads = []
        attn_page_ranges = []
        attn_partial_ids = []
        sm_counter = 0
        pps = (total_pages + num_partials_per_head - 1) // num_partials_per_head
        for kv_h in range(NKV):
            for pi in range(num_partials_per_head):
                ps_start = pi * pps
                ps_end = min(ps_start + pps, total_pages)
                if ps_start >= total_pages:
                    break
                attn_sm_ids.append(sm_counter)
                attn_kv_heads.append(kv_h)
                attn_page_ranges.append((ps_start, ps_end))
                attn_partial_ids.append(kv_h * num_partials_per_head + pi)
                sm_counter += 1
        builder.add_paged_attn(
            attn_sm_ids,
            attn_kv_heads,
            token_id=0,
            page_ranges=attn_page_ranges,
            partial_indices=attn_partial_ids,
            is_single=False,
        )
        builder.add_barrier(all_sms, barrier_id=2)
        partial_starts = [kv_h * num_partials_per_head for kv_h in range(NKV)]
        builder.add_attn_reduce(
            range(NKV),
            list(range(NKV)),
            num_partials_per_head,
            partial_starts=partial_starts,
            token_id=0,
        )
        builder.add_barrier(all_sms, barrier_id=3)
        next_bid = 4
    else:
        builder.add_paged_attn(range(NKV), list(range(NKV)), token_id=0)
        builder.add_barrier(all_sms, barrier_id=2)
        next_bid = 3

    # Phase 3: O-proj GEMV + OPROJ_POST
    builder.add_gemv_oproj(all_sms, dist_rows(H, NUM_SMS), token_id=0)
    builder.add_barrier(all_sms, barrier_id=next_bid)
    builder.add_oproj_post(0, token_id=0)
    builder.add_done(all_sms)

    launcher.launch_gemv_qkv(
        builder,
        attn_normed,
        qkv_w,
        qnw,
        knw,
        vnw,
        cs_cache,
        kvc_mk,
        cache_loc,
        cu_np,
        tpos,
        tbat,
        lpl,
        qkv_scr,
        attn_scr,
        eps=eps,
        page_size=page_size,
        attn_scale=scale,
        o_proj_weight=oproj_w,
        residual=residual_in,
        post_attn_norm_weight=panw,
        pre_ffn_norm_weight=pfnw,
        o_proj_scratch=oproj_scr,
        post_attn_out=post_out,
        pre_ffn_out=pre_out,
        partial_attn_scratch=partial_scr,
    )
    torch.cuda.synchronize()

    # ── Compare ──
    post_diff = (post_out[0] - ref_post[0]).abs()
    pre_diff = (pre_out[0] - ref_pre[0]).abs()
    post_max = post_diff.max().item()
    pre_max = pre_diff.max().item()

    return post_max, pre_max


def test_benchmark_kernel_a(past_length=128, page_size=16):
    """Benchmark full Kernel A megakernel vs decomposed."""
    from launcher import InstructionBuilder, MegakernelLauncher

    torch.manual_seed(42)
    dev, dt = "cuda", torch.bfloat16
    eps = 1e-6
    scale = 1.0 / math.sqrt(D)
    pos = past_length

    attn_normed = torch.randn(1, H, device=dev, dtype=dt)
    residual_in = torch.randn(1, H, device=dev, dtype=dt)
    qkv_w = torch.randn(QKVW, H, device=dev, dtype=dt) * 0.02
    oproj_w = torch.randn(H, QW, device=dev, dtype=dt) * 0.02
    qnw = torch.ones(D, device=dev, dtype=torch.float32)
    knw = torch.ones(D, device=dev, dtype=torch.float32)
    vnw = torch.ones(D, device=dev, dtype=torch.float32)
    panw = torch.ones(H, device=dev, dtype=torch.float32)
    pfnw = torch.ones(H, device=dev, dtype=torch.float32)
    cs_cache = torch.randn(4096, D, device=dev, dtype=torch.float32)

    tl = past_length + 1
    np_ = (tl + page_size - 1) // page_size
    kvc = torch.randn(np_ + 4, 2, NKV, page_size, D, device=dev, dtype=dt) * 0.1
    cache_loc = torch.arange(np_ + 4, dtype=torch.int32, device=dev)
    cu_np = torch.tensor([0, np_], dtype=torch.int32, device=dev)
    lpl = torch.tensor([((tl - 1) % page_size) + 1], dtype=torch.int32, device=dev)
    tpos = torch.tensor([pos], dtype=torch.int32, device=dev)
    tbat = torch.tensor([0], dtype=torch.int32, device=dev)

    qkv_scr = torch.zeros(1, QKVW, device=dev, dtype=dt)
    attn_scr = torch.zeros(1, QW, device=dev, dtype=dt)
    oproj_scr = torch.zeros(1, H, device=dev, dtype=torch.float32)
    post_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    pre_out = torch.zeros(1, H, device=dev, dtype=torch.float32)

    launcher = MegakernelLauncher(num_sms=NUM_SMS)

    # Multi-SM attention scheduling
    total_pages = np_
    MULTI_SM_THRESHOLD = 16
    if total_pages > MULTI_SM_THRESHOLD:
        num_pp = min((total_pages + 7) // 8, 16)
    else:
        num_pp = 1
    use_multi = num_pp > 1
    max_part = NKV * num_pp
    partial_scr = (
        torch.zeros(max_part, NQ, D + 2, device=dev, dtype=torch.float32) if use_multi else None
    )

    def build_instructions():
        b = InstructionBuilder(num_sms=NUM_SMS)
        a = range(NUM_SMS)
        b.add_gemv_qkv(a, dist_rows(QKVW, NUM_SMS), token_id=0)
        b.add_barrier(a, barrier_id=0)
        b.add_qkv_post(range(TH), [(h, h + 1) for h in range(TH)], token_id=0)
        b.add_barrier(a, barrier_id=1)
        if use_multi:
            sids, khs, prs, pids = [], [], [], []
            sc = 0
            pps = (total_pages + num_pp - 1) // num_pp
            for kh in range(NKV):
                for pi in range(num_pp):
                    ps, pe = pi * pps, min(pi * pps + pps, total_pages)
                    if ps >= total_pages:
                        break
                    sids.append(sc)
                    khs.append(kh)
                    prs.append((ps, pe))
                    pids.append(kh * num_pp + pi)
                    sc += 1
            b.add_paged_attn(
                sids, khs, token_id=0, page_ranges=prs, partial_indices=pids, is_single=False
            )
            b.add_barrier(a, barrier_id=2)
            pstarts = [kh * num_pp for kh in range(NKV)]
            b.add_attn_reduce(
                range(NKV), list(range(NKV)), num_pp, partial_starts=pstarts, token_id=0
            )
            b.add_barrier(a, barrier_id=3)
            nb = 4
        else:
            b.add_paged_attn(range(NKV), list(range(NKV)), token_id=0)
            b.add_barrier(a, barrier_id=2)
            nb = 3
        b.add_gemv_oproj(a, dist_rows(H, NUM_SMS), token_id=0)
        b.add_barrier(a, barrier_id=nb)
        b.add_oproj_post(0, token_id=0)
        b.add_done(a)
        return b

    builder = build_instructions()

    def run_mk():
        launcher.launch_gemv_qkv(
            builder,
            attn_normed,
            qkv_w,
            qnw,
            knw,
            vnw,
            cs_cache,
            kvc,
            cache_loc,
            cu_np,
            tpos,
            tbat,
            lpl,
            qkv_scr,
            attn_scr,
            eps=eps,
            page_size=page_size,
            attn_scale=scale,
            o_proj_weight=oproj_w,
            residual=residual_in,
            post_attn_norm_weight=panw,
            pre_ffn_norm_weight=pfnw,
            o_proj_scratch=oproj_scr,
            post_attn_out=post_out,
            pre_ffn_out=pre_out,
            partial_attn_scratch=partial_scr,
        )

    warmup, iters = 20, 100
    for _ in range(warmup):
        run_mk()
        torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run_mk()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


def main():
    if not torch.cuda.is_available():
        sys.exit(0)

    print("=" * 70)
    print("Gemma4 Megakernel: Full Kernel A End-to-End Test")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 70)

    all_pass = True
    for past in [64, 128, 512]:
        post_max, pre_max = test_full_kernel_a(past_length=past)
        ok = post_max < 0.1 and pre_max < 0.1
        all_pass = all_pass and ok
        print(
            f"  seq={past + 1:>4d}:  post_attn max_diff={post_max:.4f}  "
            f"pre_ffn max_diff={pre_max:.4f}  {'PASS' if ok else 'FAIL'}"
        )

    print()
    for past in [128, 512, 1024]:
        mk_us = test_benchmark_kernel_a(past_length=past)
        print(f"  seq={past + 1:>4d}:  megakernel Kernel A = {mk_us:.1f} us")

    print("=" * 70)
    print(f"Full Kernel A: {'ALL PASS' if all_pass else 'FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
