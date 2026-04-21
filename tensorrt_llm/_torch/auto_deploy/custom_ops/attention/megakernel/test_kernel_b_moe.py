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

"""Standalone full Kernel B test with router + MoE in the persistent kernel."""

from __future__ import annotations

import torch
import torch.nn.functional as F

H = 2816
INTERMEDIATE_SIZE = 2112
NUM_EXPERTS = 128
TOP_K = 8
EXPERT_INTERMEDIATE_SIZE = 704
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


def reference_moe(
    moe_input: torch.Tensor,
    route_idx: torch.Tensor,
    route_w: torch.Tensor,
    moe_w13_stacked_weight: torch.Tensor,
    moe_w2_weight: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros((1, H), device=moe_input.device, dtype=torch.float32)
    for slot in range(TOP_K):
        expert = int(route_idx[0, slot].item())
        gate_up = F.linear(moe_input.to(torch.bfloat16), moe_w13_stacked_weight[expert]).float()
        gate, up = gate_up.chunk(2, dim=-1)
        act = F.gelu(gate, approximate="tanh") * up
        expert_out = F.linear(act.to(torch.bfloat16), moe_w2_weight[expert]).float()
        out += route_w[0, slot].float() * expert_out
    return out


def build_full_b_builder(
    num_sms: int,
    overlap_router: bool,
    moe_fc1_shards: int = 12,
    moe_fc2_shards: int = 16,
    merged_fc2: bool = False,
    overlap_fc1_with_down: bool = True,
    bpost_partials: int = 8,
    bpost_merge_partials: int | None = None,
    bpost_hidden_partials: int | None = None,
    bpost_next_partials: int | None = None,
):
    from launcher import InstructionBuilder

    builder = InstructionBuilder(num_sms=num_sms)
    all_sms = list(range(num_sms))
    gate_sms = list(range(1, num_sms))

    if overlap_router:
        builder.add_ffn_gateup(
            gate_sms, dist_rows(2 * INTERMEDIATE_SIZE, gate_sms), token_id=0, layer_id=0
        )
        builder.add_router_topk([0], token_id=0, layer_id=0)
    else:
        builder.add_ffn_gateup(
            gate_sms, dist_rows(2 * INTERMEDIATE_SIZE, gate_sms), token_id=0, layer_id=0
        )
        builder.add_noop([0])
    builder.add_barrier(all_sms, barrier_id=0)

    if not overlap_router:
        builder.add_router_topk([0], token_id=0, layer_id=0)
        builder.add_noop(gate_sms)
        builder.add_barrier(all_sms, barrier_id=1)
        down_barrier = 2
    else:
        down_barrier = 1

    fc1_sms = list(range(TOP_K * moe_fc1_shards))
    fc1_slots = [sm_id // moe_fc1_shards for sm_id in fc1_sms]
    fc1_row_ranges = dist_rows(2 * EXPERT_INTERMEDIATE_SIZE, list(range(moe_fc1_shards)))
    fc1_rows = [fc1_row_ranges[sm_id % moe_fc1_shards] for sm_id in fc1_sms]
    if overlap_fc1_with_down:
        down_sms = list(range(len(fc1_sms), num_sms))
        builder.add_moe_sharded(fc1_sms, fc1_slots, fc1_rows, token_id=0, mode=0)
        builder.add_ffn_down(down_sms, dist_rows(H, down_sms), token_id=0, layer_id=0)
        builder.add_barrier(all_sms, barrier_id=down_barrier)
        fc2_base_barrier = down_barrier + 1
    else:
        builder.add_ffn_down(all_sms, dist_rows(H, all_sms), token_id=0, layer_id=0)
        builder.add_barrier(all_sms, barrier_id=down_barrier)
        builder.add_moe_sharded(fc1_sms, fc1_slots, fc1_rows, token_id=0, mode=0)
        builder.add_noop(range(len(fc1_sms), num_sms))
        builder.add_barrier(all_sms, barrier_id=down_barrier + 1)
        fc2_base_barrier = down_barrier + 2

    if merged_fc2:
        fc2_sms = list(range(moe_fc2_shards))
        fc2_rows = dist_rows(H, fc2_sms)
        builder.add_moe_merged_down(fc2_sms, fc2_rows, token_id=0)
    else:
        fc2_sms = list(range(TOP_K * moe_fc2_shards))
        fc2_slots = [sm_id // moe_fc2_shards for sm_id in fc2_sms]
        fc2_row_ranges = dist_rows(H, list(range(moe_fc2_shards)))
        fc2_rows = [fc2_row_ranges[sm_id % moe_fc2_shards] for sm_id in fc2_sms]
        builder.add_moe_sharded(fc2_sms, fc2_slots, fc2_rows, token_id=0, mode=1)
    builder.add_noop(range(len(fc2_sms), num_sms))
    builder.add_barrier(all_sms, barrier_id=fc2_base_barrier)

    merge_partials = bpost_partials if bpost_merge_partials is None else bpost_merge_partials
    hidden_partials = merge_partials if bpost_hidden_partials is None else bpost_hidden_partials
    next_partials = hidden_partials if bpost_next_partials is None else bpost_next_partials

    stats_sms = list(range(bpost_partials))
    stats_rows = dist_rows(H, stats_sms)
    builder.add_b_post_stats(stats_sms, stats_rows, token_id=0, num_partials=bpost_partials)
    builder.add_noop(range(len(stats_sms), num_sms))
    builder.add_barrier(all_sms, barrier_id=fc2_base_barrier + 1)

    merge_sms = list(range(merge_partials))
    merge_rows = dist_rows(H, merge_sms)
    builder.add_b_post_merge(merge_sms, merge_rows, token_id=0, num_partials=bpost_partials)
    builder.add_noop(range(len(merge_sms), num_sms))
    builder.add_barrier(all_sms, barrier_id=fc2_base_barrier + 2)

    hidden_sms = list(range(hidden_partials))
    hidden_rows = dist_rows(H, hidden_sms)
    builder.add_b_post_hidden(hidden_sms, hidden_rows, token_id=0, num_partials=merge_partials)
    builder.add_noop(range(len(hidden_sms), num_sms))
    builder.add_barrier(all_sms, barrier_id=fc2_base_barrier + 3)

    next_sms = list(range(next_partials))
    next_rows = dist_rows(H, next_sms)
    builder.add_b_post_next_norm(next_sms, next_rows, token_id=0, num_partials=hidden_partials)
    builder.add_noop(range(len(next_sms), num_sms))
    builder.add_done(all_sms)
    return builder


def main() -> None:
    from launcher import MegakernelLauncher

    torch.manual_seed(4321)
    dev = "cuda"
    eps = 1e-6
    dt = torch.bfloat16

    post_attn = torch.randn(1, H, device=dev, dtype=torch.float32) * 0.1
    pre_ffn = torch.randn(1, H, device=dev, dtype=torch.float32) * 0.1

    ffn_gate_up_weight = torch.randn(2 * INTERMEDIATE_SIZE, H, device=dev, dtype=dt) * 0.02
    ffn_down_weight = torch.randn(H, INTERMEDIATE_SIZE, device=dev, dtype=dt) * 0.02
    router_proj_weight = torch.randn(NUM_EXPERTS, H, device=dev, dtype=dt) * 0.02
    router_scale = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    router_root_size = torch.tensor([H**-0.5], device=dev, dtype=torch.float32)
    moe_w13_stacked_weight = (
        torch.randn(NUM_EXPERTS, 2 * EXPERT_INTERMEDIATE_SIZE, H, device=dev, dtype=dt) * 0.02
    )
    moe_w2_weight = (
        torch.randn(NUM_EXPERTS, H, EXPERT_INTERMEDIATE_SIZE, device=dev, dtype=dt) * 0.02
    )

    post_ffn1_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    pre_ffn2_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    post_ffn2_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    post_ffn_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    next_input_norm_weight = torch.randn(H, device=dev, dtype=torch.float32).abs() + 0.5
    layer_scalar = torch.ones(1, device=dev, dtype=torch.float32)

    gate_up = F.linear(pre_ffn.to(dt), ffn_gate_up_weight).float()
    gate, up = gate_up.chunk(2, dim=-1)
    dense_act = F.gelu(gate, approximate="tanh") * up
    dense_down = F.linear(dense_act.to(dt), ffn_down_weight).float()
    dense_norm = rms_norm(dense_down, post_ffn1_norm_weight, eps)

    router_hidden = rms_norm(post_attn, torch.ones(H, device=dev, dtype=torch.float32), eps)
    router_input = router_hidden * router_root_size.view(1, 1) * router_scale.view(1, -1)
    router_logits = F.linear(router_input.to(dt), router_proj_weight).float()
    route_vals, route_idx = torch.topk(router_logits, k=TOP_K, dim=-1)
    route_w = F.softmax(route_vals, dim=-1)
    moe_input = rms_norm(post_attn, pre_ffn2_norm_weight, eps)
    moe_out = reference_moe(moe_input, route_idx, route_w, moe_w13_stacked_weight, moe_w2_weight)
    moe_norm = rms_norm(moe_out, post_ffn2_norm_weight, eps)
    merged_norm = rms_norm(dense_norm + moe_norm, post_ffn_norm_weight, eps)
    ref_hidden = (post_attn + merged_norm) * layer_scalar.view(1, -1)
    ref_next = rms_norm(ref_hidden, next_input_norm_weight, eps)

    ffn_gate_scratch = torch.zeros(1, INTERMEDIATE_SIZE, device=dev, dtype=dt)
    ffn_up_scratch = torch.zeros(1, INTERMEDIATE_SIZE, device=dev, dtype=dt)
    ffn_down_scratch = torch.zeros(1, H, device=dev, dtype=torch.float32)
    hidden_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    next_attn_normed_out = torch.zeros(1, H, device=dev, dtype=torch.float32)
    router_topk_weights = torch.zeros(1, TOP_K, device=dev, dtype=torch.float32)
    router_topk_indices = torch.zeros(1, TOP_K, device=dev, dtype=torch.int32)
    moe_input_scratch = torch.zeros(1, H, device=dev, dtype=dt)
    moe_gate_scratch = torch.zeros(1, TOP_K, EXPERT_INTERMEDIATE_SIZE, device=dev, dtype=dt)
    moe_up_scratch = torch.zeros(1, TOP_K, EXPERT_INTERMEDIATE_SIZE, device=dev, dtype=dt)
    moe_scratch = torch.zeros(1, H, TOP_K, device=dev, dtype=torch.float32)
    moe_merged_scratch = torch.zeros(1, H, device=dev, dtype=torch.float32)

    launcher = MegakernelLauncher(num_sms=NUM_SMS)
    overlap_builder = build_full_b_builder(
        NUM_SMS,
        overlap_router=True,
        moe_fc1_shards=12,
        moe_fc2_shards=16,
        overlap_fc1_with_down=True,
        bpost_partials=8,
    )
    sequential_builder = build_full_b_builder(
        NUM_SMS,
        overlap_router=False,
        moe_fc1_shards=12,
        moe_fc2_shards=16,
        overlap_fc1_with_down=True,
        bpost_partials=8,
    )
    barrier_slots = torch.zeros(256, dtype=torch.int32, device=dev)
    debug_output = torch.zeros(NUM_SMS * 5, dtype=torch.int32, device=dev)

    def launch(builder, use_merged_fc2: bool = False) -> None:
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
            post_ffn2_norm_weight=post_ffn2_norm_weight,
            moe_w13_stacked_weight=moe_w13_stacked_weight,
            moe_w2_weight=moe_w2_weight,
            moe_gate_scratch=moe_gate_scratch,
            moe_up_scratch=moe_up_scratch,
            moe_scratch=moe_scratch,
            moe_merged_scratch=moe_merged_scratch if use_merged_fc2 else None,
            eps=eps,
            barrier_slots=barrier_slots,
            debug_output=debug_output,
        )

    launch(overlap_builder)
    torch.cuda.synchronize()

    hidden_diff = (hidden_out - ref_hidden).abs().max().item()
    next_diff = (next_attn_normed_out - ref_next).abs().max().item()
    moe_diff = (moe_scratch.sum(dim=2) - moe_out).abs().max().item()
    print(f"hidden_out max_diff={hidden_diff:.4f}")
    print(f"next_attn_normed_out max_diff={next_diff:.4f}")
    print(f"moe_scratch max_diff={moe_diff:.4f}")
    assert hidden_diff < 0.4
    assert next_diff < 0.4
    assert moe_diff < 0.4

    overlap_us = _bench_us(lambda: launch(overlap_builder))
    sequential_us = _bench_us(lambda: launch(sequential_builder))
    print(f"full Kernel B overlap   = {overlap_us:.1f} us")
    print(f"full Kernel B sequential= {sequential_us:.1f} us")
    print(f"router overlap saved    = {sequential_us - overlap_us:.1f} us")

    sweep = [(2, 8), (4, 8), (4, 12), (4, 16), (8, 14), (8, 16), (10, 15), (12, 16), (16, 16)]
    for fc1_shards, fc2_shards in sweep:
        builder = build_full_b_builder(
            NUM_SMS,
            overlap_router=True,
            moe_fc1_shards=fc1_shards,
            moe_fc2_shards=fc2_shards,
            merged_fc2=False,
            overlap_fc1_with_down=True,
            bpost_partials=8,
        )
        sweep_us = _bench_us(lambda b=builder: launch(b), warmup=10, iters=40)
        print(f"sweep fc1={fc1_shards:2d} fc2={fc2_shards:2d} -> {sweep_us:.1f} us")
    print("PASS")


if __name__ == "__main__":
    main()
