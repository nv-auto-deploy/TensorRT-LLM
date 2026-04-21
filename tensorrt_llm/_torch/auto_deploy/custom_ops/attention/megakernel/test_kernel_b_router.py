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

"""Standalone router/top-k and overlap test for single-kernel Gemma4 Kernel B."""

from __future__ import annotations

import torch
import torch.nn.functional as F

H = 2816
INTERMEDIATE_SIZE = 2112
NUM_EXPERTS = 128
TOP_K = 8
NUM_SMS = 132


def dist_rows(n: int, sms: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for i in range(len(sms)):
        count = n // len(sms) + (1 if i < n % len(sms) else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt((xf * xf).mean(-1, keepdim=True) + eps) * w.float()


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


def build_router_only_builder(num_sms: int):
    from launcher import InstructionBuilder

    builder = InstructionBuilder(num_sms=num_sms)
    builder.add_router_topk([0], token_id=0, layer_id=0)
    builder.add_done(range(num_sms))
    return builder


def build_gateup_router_builder(num_sms: int, overlap: bool):
    from launcher import InstructionBuilder

    builder = InstructionBuilder(num_sms=num_sms)
    all_sms = list(range(num_sms))
    gate_sms = list(range(1, num_sms))
    gate_rows = dist_rows(2 * INTERMEDIATE_SIZE, gate_sms)

    if overlap:
        builder.add_ffn_gateup(gate_sms, gate_rows, token_id=0, layer_id=0)
        builder.add_router_topk([0], token_id=0, layer_id=0)
        builder.add_barrier(all_sms, barrier_id=0)
    else:
        builder.add_ffn_gateup(gate_sms, gate_rows, token_id=0, layer_id=0)
        builder.add_barrier(all_sms, barrier_id=0)
        builder.add_router_topk([0], token_id=0, layer_id=0)

    builder.add_done(all_sms)
    return builder


def main() -> None:
    from launcher import MegakernelLauncher

    torch.manual_seed(1234)
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
    router_proj_weight = torch.randn(NUM_EXPERTS, H, device=dev, dtype=torch.bfloat16) * 0.02
    router_scale = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    router_root_size = torch.tensor([H**-0.5], device=dev, dtype=torch.float32)
    pre_ffn2_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5

    ffn_gate_scratch = torch.zeros(1, INTERMEDIATE_SIZE, device=dev, dtype=torch.bfloat16)
    ffn_up_scratch = torch.zeros(1, INTERMEDIATE_SIZE, device=dev, dtype=torch.bfloat16)
    ffn_down_scratch = torch.zeros(1, H, device=dev, dtype=torch.float32)
    hidden_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    next_attn_normed_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    router_topk_weights = torch.zeros(1, TOP_K, device=dev, dtype=torch.float32)
    router_topk_indices = torch.zeros(1, TOP_K, device=dev, dtype=torch.int32)
    moe_input_scratch = torch.zeros(1, H, device=dev, dtype=torch.bfloat16)

    router_normed = rms_norm(post_attn, torch.ones(H, device=dev, dtype=torch.float32), eps)
    router_input = router_normed * router_root_size.view(1, 1) * router_scale.view(1, -1)
    router_logits = F.linear(router_input.to(torch.bfloat16), router_proj_weight).float()
    ref_router_weights_from_probs, ref_router_indices_from_probs = torch.topk(
        F.softmax(router_logits, dim=-1),
        k=TOP_K,
        dim=-1,
    )
    ref_router_weights_from_probs = (
        ref_router_weights_from_probs / ref_router_weights_from_probs.sum(dim=-1, keepdim=True)
    )
    ref_router_top_logits, ref_router_indices_from_logits = torch.topk(
        router_logits, k=TOP_K, dim=-1
    )
    ref_router_weights_from_logits = F.softmax(ref_router_top_logits, dim=-1)
    ref_moe_input = rms_norm(post_attn, pre_ffn2_norm_weight, eps).to(torch.bfloat16)

    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    router_builder = build_router_only_builder(NUM_SMS)
    overlap_builder = build_gateup_router_builder(NUM_SMS, overlap=True)
    sequential_builder = build_gateup_router_builder(NUM_SMS, overlap=False)
    barrier_slots = torch.zeros(256, dtype=torch.int32, device=dev)
    debug_output = torch.zeros(NUM_SMS, dtype=torch.int32, device=dev)

    def launch(builder) -> None:
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
            router_proj_weight=router_proj_weight,
            router_scale=router_scale,
            router_root_size=router_root_size,
            pre_ffn2_norm_weight=pre_ffn2_norm_weight,
            router_topk_weights=router_topk_weights,
            router_topk_indices=router_topk_indices,
            moe_input_scratch=moe_input_scratch,
            eps=eps,
            barrier_slots=barrier_slots,
            debug_output=debug_output,
        )

    launch(router_builder)
    torch.cuda.synchronize()

    weights_diff = (router_topk_weights - ref_router_weights_from_probs).abs().max().item()
    logits_weights_diff = (router_topk_weights - ref_router_weights_from_logits).abs().max().item()
    moe_input_diff = (moe_input_scratch.float() - ref_moe_input.float()).abs().max().item()
    indices_match = torch.equal(
        router_topk_indices.cpu(), ref_router_indices_from_probs.to(torch.int32).cpu()
    )
    logits_indices_match = torch.equal(
        router_topk_indices.cpu(), ref_router_indices_from_logits.to(torch.int32).cpu()
    )

    print(f"router_topk weights max_diff_vs_probs={weights_diff:.6f}")
    print(f"router_topk weights max_diff_vs_logits={logits_weights_diff:.6f}")
    print(f"router_topk indices match_probs={indices_match}")
    print(f"router_topk indices match_logits={logits_indices_match}")
    print(f"moe_input max_diff={moe_input_diff:.6f}")
    assert weights_diff < 1e-4
    assert logits_weights_diff < 1e-4
    assert indices_match
    assert logits_indices_match
    assert moe_input_diff < 5e-3

    router_us = _bench_us(lambda: launch(router_builder))
    overlap_us = _bench_us(lambda: launch(overlap_builder))
    sequential_us = _bench_us(lambda: launch(sequential_builder))

    print(f"router_only           = {router_us:.1f} us")
    print(f"gateup_plus_router_seq= {sequential_us:.1f} us")
    print(f"gateup_plus_router_ovl= {overlap_us:.1f} us")
    print(f"overlap_saved         = {sequential_us - overlap_us:.1f} us")
    print("PASS")


if __name__ == "__main__":
    main()
