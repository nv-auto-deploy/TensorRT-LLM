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

"""Standalone dense Kernel B test.

This validates the first real single-kernel B slice:
  FFN_GATEUP -> FFN_DOWN -> B_POST

Router/top-k and MoE are intentionally omitted for this first CUDA step so we
can validate the dense path and next-layer normalized output contract.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

H = 2816
INTERMEDIATE_SIZE = 2112
NUM_SMS = 132


def dist_rows(n, sms):
    ranges = []
    start = 0
    for i in range(sms):
        count = n // sms + (1 if i < n % sms else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def rms_norm(x, w, eps=1e-6):
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * w.float()


def _bench_us(fn, warmup=20, iters=100):
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


def build_dense_b_builder(
    *,
    include_gateup: bool,
    include_down: bool,
    include_b_post: bool,
    num_sms: int,
):
    from launcher import InstructionBuilder

    builder = InstructionBuilder(num_sms=num_sms)
    all_sms = range(num_sms)

    if include_gateup:
        builder.add_ffn_gateup(
            all_sms, dist_rows(2 * INTERMEDIATE_SIZE, num_sms), token_id=0, layer_id=0
        )
    if include_gateup and (include_down or include_b_post):
        builder.add_barrier(all_sms, barrier_id=0)
    if include_down:
        builder.add_ffn_down(all_sms, dist_rows(H, num_sms), token_id=0, layer_id=0)
    if include_down and include_b_post:
        builder.add_barrier(all_sms, barrier_id=1)
    if include_b_post:
        builder.add_b_post([0], token_id=0, layer_id=0)
    builder.add_done(all_sms)
    return builder


def test_dense_kernel_b():
    from launcher import MegakernelLauncher

    torch.manual_seed(123)
    dev = "cuda"
    eps = 1e-6

    post_attn = torch.randn(1, H, device=dev, dtype=torch.float32) * 0.1
    pre_ffn = torch.randn(1, H, device=dev, dtype=torch.float32) * 0.1
    ffn_gate_up_weight = (
        torch.randn(2 * INTERMEDIATE_SIZE, H, device=dev, dtype=torch.bfloat16) * 0.02
    )
    ffn_down_weight = torch.randn(H, INTERMEDIATE_SIZE, device=dev, dtype=torch.bfloat16) * 0.02
    post_ffn1_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    post_ffn_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    layer_scalar = torch.ones(1, device=dev, dtype=torch.float32)
    next_input_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5

    gate_up = F.linear(pre_ffn.to(torch.bfloat16), ffn_gate_up_weight).float()
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.gelu(gate, approximate="tanh") * up
    dense_down = F.linear(act.to(torch.bfloat16), ffn_down_weight).float()
    dense_norm = rms_norm(dense_down, post_ffn1_norm_weight, eps)
    merged_norm = rms_norm(dense_norm, post_ffn_norm_weight, eps)
    ref_hidden = (post_attn + merged_norm) * layer_scalar.view(1, -1)
    ref_next = rms_norm(ref_hidden, next_input_norm_weight, eps)

    ffn_gate_scratch = torch.zeros(1, INTERMEDIATE_SIZE, device=dev, dtype=torch.bfloat16)
    ffn_up_scratch = torch.zeros(1, INTERMEDIATE_SIZE, device=dev, dtype=torch.bfloat16)
    ffn_down_scratch = torch.zeros(1, H, device=dev, dtype=torch.float32)
    hidden_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    next_attn_normed_out = torch.zeros(1, H, device=dev, dtype=torch.float32)

    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    builder = build_dense_b_builder(
        include_gateup=True,
        include_down=True,
        include_b_post=True,
        num_sms=NUM_SMS,
    )

    barrier_slots = torch.zeros(256, dtype=torch.int32, device=dev)
    debug_output = torch.zeros(NUM_SMS, dtype=torch.int32, device=dev)
    launcher.launch_dense_b(
        builder,
        post_attn,
        pre_ffn,
        ffn_gate_up_weight,
        ffn_down_weight,
        post_ffn1_norm_weight,
        post_ffn_norm_weight,
        layer_scalar,
        next_input_norm_weight,
        ffn_gate_scratch,
        ffn_up_scratch,
        ffn_down_scratch,
        hidden_out,
        next_attn_normed_out,
        eps=eps,
        barrier_slots=barrier_slots,
        debug_output=debug_output,
    )
    torch.cuda.synchronize()

    hidden_diff = (hidden_out - ref_hidden).abs().max().item()
    next_diff = (next_attn_normed_out - ref_next).abs().max().item()
    print(f"hidden_out max_diff={hidden_diff:.4f}")
    print(f"next_attn_normed_out max_diff={next_diff:.4f}")
    assert hidden_diff < 0.2
    assert next_diff < 0.2

    def run_kernel_b():
        launcher.launch_dense_b(
            builder,
            post_attn,
            pre_ffn,
            ffn_gate_up_weight,
            ffn_down_weight,
            post_ffn1_norm_weight,
            post_ffn_norm_weight,
            layer_scalar,
            next_input_norm_weight,
            ffn_gate_scratch,
            ffn_up_scratch,
            ffn_down_scratch,
            hidden_out,
            next_attn_normed_out,
            eps=eps,
            barrier_slots=barrier_slots,
            debug_output=debug_output,
        )

    def run_reference():
        gate_up = F.linear(pre_ffn.to(torch.bfloat16), ffn_gate_up_weight).float()
        gate, up_ = gate_up.chunk(2, dim=-1)
        act_ = F.gelu(gate, approximate="tanh") * up_
        dense_down_ = F.linear(act_.to(torch.bfloat16), ffn_down_weight).float()
        dense_norm_ = rms_norm(dense_down_, post_ffn1_norm_weight, eps)
        merged_norm_ = rms_norm(dense_norm_, post_ffn_norm_weight, eps)
        hidden_ref = (post_attn + merged_norm_) * layer_scalar.view(1, -1)
        rms_norm(hidden_ref, next_input_norm_weight, eps)

    mk_us = _bench_us(run_kernel_b)
    ref_us = _bench_us(run_reference)
    print(f"dense Kernel B = {mk_us:.1f} us")
    print(f"reference torch = {ref_us:.1f} us")

    gateup_only_builder = build_dense_b_builder(
        include_gateup=True,
        include_down=False,
        include_b_post=False,
        num_sms=NUM_SMS,
    )
    gateup_down_builder = build_dense_b_builder(
        include_gateup=True,
        include_down=True,
        include_b_post=False,
        num_sms=NUM_SMS,
    )
    b_post_only_builder = build_dense_b_builder(
        include_gateup=False,
        include_down=False,
        include_b_post=True,
        num_sms=NUM_SMS,
    )

    ref_dense_down = dense_down.contiguous()
    ffn_down_scratch.copy_(ref_dense_down)

    def run_gateup_only():
        launcher.launch_dense_b(
            gateup_only_builder,
            post_attn,
            pre_ffn,
            ffn_gate_up_weight,
            ffn_down_weight,
            post_ffn1_norm_weight,
            post_ffn_norm_weight,
            layer_scalar,
            next_input_norm_weight,
            ffn_gate_scratch,
            ffn_up_scratch,
            ffn_down_scratch,
            hidden_out,
            next_attn_normed_out,
            eps=eps,
            barrier_slots=barrier_slots,
            debug_output=debug_output,
        )

    def run_gateup_down_only():
        launcher.launch_dense_b(
            gateup_down_builder,
            post_attn,
            pre_ffn,
            ffn_gate_up_weight,
            ffn_down_weight,
            post_ffn1_norm_weight,
            post_ffn_norm_weight,
            layer_scalar,
            next_input_norm_weight,
            ffn_gate_scratch,
            ffn_up_scratch,
            ffn_down_scratch,
            hidden_out,
            next_attn_normed_out,
            eps=eps,
            barrier_slots=barrier_slots,
            debug_output=debug_output,
        )

    def run_b_post_only():
        launcher.launch_dense_b(
            b_post_only_builder,
            post_attn,
            pre_ffn,
            ffn_gate_up_weight,
            ffn_down_weight,
            post_ffn1_norm_weight,
            post_ffn_norm_weight,
            layer_scalar,
            next_input_norm_weight,
            ffn_gate_scratch,
            ffn_up_scratch,
            ffn_down_scratch,
            hidden_out,
            next_attn_normed_out,
            eps=eps,
            barrier_slots=barrier_slots,
            debug_output=debug_output,
        )

    gateup_us = _bench_us(run_gateup_only)
    gateup_down_us = _bench_us(run_gateup_down_only)
    b_post_us = _bench_us(run_b_post_only)
    print(f"phase gateup_only = {gateup_us:.1f} us")
    print(f"phase gateup_down = {gateup_down_us:.1f} us")
    print(f"phase b_post_only = {b_post_us:.1f} us")
    print(f"phase down_only ~= {gateup_down_us - gateup_us:.1f} us")
    print("PASS")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        sys.exit(0)
    print("Gemma4 Megakernel: Dense Kernel B Test")
    test_dense_kernel_b()


if __name__ == "__main__":
    main()
