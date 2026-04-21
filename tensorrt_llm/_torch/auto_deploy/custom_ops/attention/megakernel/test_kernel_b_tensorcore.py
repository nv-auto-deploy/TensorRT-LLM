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

"""Benchmark tensor-core candidates for Gemma4 Kernel B decode shapes."""

from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.triton_routing  # noqa: F401
from tensorrt_llm import deep_gemm

HIDDEN = 2816
DENSE_INTERMEDIATE = 2112
NUM_EXPERTS = 128
TOPK = 8
EXPERT_INTERMEDIATE = 704
EPS = 1e-6


def _bench_us(fn, warmup: int = 50, iters: int = 200) -> float:
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
    times_us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return statistics.median(times_us)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def _run_grouped_moe(
    moe_input: torch.Tensor,
    routing_indices: torch.Tensor,
    topk_weight: torch.Tensor,
    moe_w13_stacked_weight: torch.Tensor,
    moe_w2_weight: torch.Tensor,
    grouped_w13_out: torch.Tensor,
    grouped_w2_in: torch.Tensor,
    grouped_w2_out: torch.Tensor,
    masked_m: torch.Tensor,
    reduced_out: torch.Tensor,
) -> None:
    selected_w13 = moe_w13_stacked_weight[routing_indices[0]].contiguous()
    grouped_a = moe_input.expand(TOPK, -1).contiguous().view(TOPK, 1, HIDDEN)
    deep_gemm.m_grouped_bf16_gemm_nt_masked(grouped_a, selected_w13, grouped_w13_out, masked_m, 1)

    gate, up = torch.chunk(grouped_w13_out[:, 0], 2, dim=-1)
    grouped_w2_in.copy_((F.gelu(gate.float(), approximate="tanh") * up.float()).to(torch.bfloat16))

    selected_w2 = moe_w2_weight[routing_indices[0]].contiguous()
    deep_gemm.m_grouped_bf16_gemm_nt_masked(
        grouped_w2_in.view(TOPK, 1, EXPERT_INTERMEDIATE),
        selected_w2,
        grouped_w2_out,
        masked_m,
        1,
    )
    reduced_out.copy_(torch.sum(grouped_w2_out[:, 0].float() * topk_weight[0].view(TOPK, 1), dim=0))


def main() -> None:
    if not torch.cuda.is_available():
        return
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        print(f"Skipping: requires Hopper+, got {torch.cuda.get_device_name()}")
        return

    print("=" * 72)
    print("Gemma4 Megakernel: Kernel B Tensor-Core Candidate Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print("=" * 72)

    dev = "cuda"
    dt = torch.bfloat16

    post_attn = torch.randn((1, HIDDEN), device=dev, dtype=dt) * 0.1
    pre_ffn = torch.randn((1, HIDDEN), device=dev, dtype=dt) * 0.1

    ffn_gate_up_weight = torch.randn((2 * DENSE_INTERMEDIATE, HIDDEN), device=dev, dtype=dt) * 0.02
    ffn_down_weight = torch.randn((HIDDEN, DENSE_INTERMEDIATE), device=dev, dtype=dt) * 0.02
    router_proj_weight = torch.randn((NUM_EXPERTS, HIDDEN), device=dev, dtype=dt) * 0.02
    router_root_size = torch.tensor(HIDDEN**-0.5, device=dev, dtype=torch.float32)
    router_scale = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5

    moe_w13_stacked_weight = (
        torch.randn((NUM_EXPERTS, 2 * EXPERT_INTERMEDIATE, HIDDEN), device=dev, dtype=dt) * 0.02
    )
    moe_w2_weight = (
        torch.randn((NUM_EXPERTS, HIDDEN, EXPERT_INTERMEDIATE), device=dev, dtype=dt) * 0.02
    )

    post_ffn_ln1 = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5
    pre_ffn_ln2 = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5
    post_ffn_ln2 = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5
    post_ffn_ln = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5
    next_input_ln = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5
    layer_scalar = torch.randn((HIDDEN,), device=dev, dtype=torch.float32).abs() + 0.5

    dense_gate_up_out = torch.empty((1, 2 * DENSE_INTERMEDIATE), device=dev, dtype=dt)
    dense_mid_out = torch.empty((1, DENSE_INTERMEDIATE), device=dev, dtype=dt)
    dense_down_out = torch.empty((1, HIDDEN), device=dev, dtype=dt)
    router_logits = torch.empty((1, NUM_EXPERTS), device=dev, dtype=dt)

    moe_input = _rms_norm(post_attn, pre_ffn_ln2)
    routing_normed = _rms_norm(post_attn, router_root_size)
    routing_input = routing_normed * router_scale.view(1, -1)
    routing_probs = F.softmax(F.linear(routing_input.to(dt), router_proj_weight).float(), dim=-1)
    topk_weight, routing_indices = torch.topk(routing_probs, k=TOPK, dim=-1)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    fused_moe_ref = torch.ops.trtllm.fused_moe(
        moe_input.to(dt),
        routing_indices.to(torch.int32),
        topk_weight.float(),
        moe_w13_stacked_weight,
        None,
        moe_w2_weight,
        None,
        dt,
        [],
        activation_type=5,
    )[0]

    grouped_w13_out = torch.empty((TOPK, 1, 2 * EXPERT_INTERMEDIATE), device=dev, dtype=dt)
    grouped_w2_in = torch.empty((TOPK, EXPERT_INTERMEDIATE), device=dev, dtype=dt)
    grouped_w2_out = torch.empty((TOPK, 1, HIDDEN), device=dev, dtype=dt)
    grouped_reduce_out = torch.empty((HIDDEN,), device=dev, dtype=torch.float32)
    masked_m = torch.ones((TOPK,), device=dev, dtype=torch.int32)

    def dense_gateup_tc() -> None:
        deep_gemm.bf16_gemm_nt(pre_ffn, ffn_gate_up_weight, dense_gate_up_out)

    def dense_down_tc() -> None:
        deep_gemm.bf16_gemm_nt(dense_mid_out, ffn_down_weight, dense_down_out)

    def router_topk() -> None:
        torch.matmul(routing_input.to(dt), router_proj_weight.t(), out=router_logits)
        probs = F.softmax(router_logits.float(), dim=-1)
        torch.topk(probs, k=TOPK, dim=-1)

    def router_topk_fused() -> None:
        route_logits = F.linear(routing_input.to(dt), router_proj_weight).float()
        torch.ops.auto_deploy.triton_fused_topk_softmax.default(route_logits, TOPK)

    def router_topk_logit_softmax() -> None:
        route_logits = F.linear(routing_input.to(dt), router_proj_weight).float()
        topk_vals, _ = torch.topk(route_logits, k=TOPK, dim=-1)
        F.softmax(topk_vals, dim=-1)

    def fused_moe_only() -> None:
        torch.ops.trtllm.fused_moe(
            moe_input.to(dt),
            routing_indices.to(torch.int32),
            topk_weight.float(),
            moe_w13_stacked_weight,
            None,
            moe_w2_weight,
            None,
            dt,
            [],
            activation_type=5,
        )

    def grouped_moe_only() -> None:
        _run_grouped_moe(
            moe_input.to(dt),
            routing_indices,
            topk_weight.float(),
            moe_w13_stacked_weight,
            moe_w2_weight,
            grouped_w13_out,
            grouped_w2_in,
            grouped_w2_out,
            masked_m,
            grouped_reduce_out,
        )

    dense_gate_up_ref = F.linear(pre_ffn, ffn_gate_up_weight)
    dense_gate_ref, dense_up_ref = torch.chunk(dense_gate_up_ref, 2, dim=-1)
    dense_ref = F.linear(
        (F.gelu(dense_gate_ref.float(), approximate="tanh") * dense_up_ref.float()).to(dt),
        ffn_down_weight,
    )
    dense_gateup_tc()
    dense_gate, dense_up = torch.chunk(dense_gate_up_out[:, :], 2, dim=-1)
    dense_mid_out.copy_((F.gelu(dense_gate.float(), approximate="tanh") * dense_up.float()).to(dt))
    deep_gemm.bf16_gemm_nt(dense_mid_out, ffn_down_weight, dense_down_out)
    dense_diff = (dense_down_out.float() - dense_ref.float()).abs()

    grouped_moe_only()
    grouped_diff = (grouped_reduce_out - fused_moe_ref.float().reshape(-1)).abs()

    print(
        f"dense_gateup_tc       {_bench_us(dense_gateup_tc):8.3f} us"
        f"  shape=1x{HIDDEN} @ {2 * DENSE_INTERMEDIATE}x{HIDDEN}"
    )
    print(
        f"dense_down_tc         {_bench_us(dense_down_tc):8.3f} us"
        f"  shape=1x{DENSE_INTERMEDIATE} @ {HIDDEN}x{DENSE_INTERMEDIATE}"
    )
    print(
        f"router_topk           {_bench_us(router_topk):8.3f} us  experts={NUM_EXPERTS} topk={TOPK}"
    )
    print(f"router_topk_fused     {_bench_us(router_topk_fused):8.3f} us")
    print(f"router_logit_softmax  {_bench_us(router_topk_logit_softmax):8.3f} us")
    print(f"fused_moe             {_bench_us(fused_moe_only):8.3f} us")
    print(
        f"grouped_moe_tc        {_bench_us(grouped_moe_only):8.3f} us"
        f"  max_diff_vs_fused={grouped_diff.max().item():.6f}"
    )
    print(f"dense_down max_diff_vs_ref={dense_diff.max().item():.6f}")

    def kernel_b_hybrid_fused() -> None:
        deep_gemm.bf16_gemm_nt(pre_ffn, ffn_gate_up_weight, dense_gate_up_out)
        gate, up = torch.chunk(dense_gate_up_out, 2, dim=-1)
        dense_mid_out.copy_((F.gelu(gate.float(), approximate="tanh") * up.float()).to(dt))
        deep_gemm.bf16_gemm_nt(dense_mid_out, ffn_down_weight, dense_down_out)
        dense_normed = _rms_norm(dense_down_out, post_ffn_ln1)

        route_logits = F.linear(routing_input.to(dt), router_proj_weight).float()
        route_w, route_idx = torch.ops.auto_deploy.triton_fused_topk_softmax.default(
            route_logits, TOPK
        )
        moe_out = torch.ops.trtllm.fused_moe(
            moe_input.to(dt),
            route_idx.to(torch.int32),
            route_w.float(),
            moe_w13_stacked_weight,
            None,
            moe_w2_weight,
            None,
            dt,
            [],
            activation_type=5,
        )[0]
        moe_normed = _rms_norm(moe_out.reshape(post_attn.shape), post_ffn_ln2)
        hidden = (
            post_attn.float() + _rms_norm(dense_normed + moe_normed, post_ffn_ln).float()
        ) * layer_scalar
        _rms_norm(hidden.to(dt), next_input_ln)

    def kernel_b_hybrid_grouped() -> None:
        deep_gemm.bf16_gemm_nt(pre_ffn, ffn_gate_up_weight, dense_gate_up_out)
        gate, up = torch.chunk(dense_gate_up_out, 2, dim=-1)
        dense_mid_out.copy_((F.gelu(gate.float(), approximate="tanh") * up.float()).to(dt))
        deep_gemm.bf16_gemm_nt(dense_mid_out, ffn_down_weight, dense_down_out)
        dense_normed = _rms_norm(dense_down_out, post_ffn_ln1)

        _run_grouped_moe(
            moe_input.to(dt),
            routing_indices,
            topk_weight.float(),
            moe_w13_stacked_weight,
            moe_w2_weight,
            grouped_w13_out,
            grouped_w2_in,
            grouped_w2_out,
            masked_m,
            grouped_reduce_out,
        )
        moe_normed = _rms_norm(grouped_reduce_out.view(1, HIDDEN).to(dt), post_ffn_ln2)
        hidden = (
            post_attn.float() + _rms_norm(dense_normed + moe_normed, post_ffn_ln).float()
        ) * layer_scalar
        _rms_norm(hidden.to(dt), next_input_ln)

    def kernel_b_hybrid_logit_softmax() -> None:
        deep_gemm.bf16_gemm_nt(pre_ffn, ffn_gate_up_weight, dense_gate_up_out)
        gate, up = torch.chunk(dense_gate_up_out, 2, dim=-1)
        dense_mid_out.copy_((F.gelu(gate.float(), approximate="tanh") * up.float()).to(dt))
        deep_gemm.bf16_gemm_nt(dense_mid_out, ffn_down_weight, dense_down_out)
        dense_normed = _rms_norm(dense_down_out, post_ffn_ln1)

        route_logits = F.linear(routing_input.to(dt), router_proj_weight).float()
        route_vals, route_idx = torch.topk(route_logits, k=TOPK, dim=-1)
        route_w = F.softmax(route_vals, dim=-1)
        moe_out = torch.ops.trtllm.fused_moe(
            moe_input.to(dt),
            route_idx.to(torch.int32),
            route_w.float(),
            moe_w13_stacked_weight,
            None,
            moe_w2_weight,
            None,
            dt,
            [],
            activation_type=5,
        )[0]
        moe_normed = _rms_norm(moe_out.reshape(post_attn.shape), post_ffn_ln2)
        hidden = (
            post_attn.float() + _rms_norm(dense_normed + moe_normed, post_ffn_ln).float()
        ) * layer_scalar
        _rms_norm(hidden.to(dt), next_input_ln)

    print(f"kernel_b_hybrid_fused   {_bench_us(kernel_b_hybrid_fused):8.3f} us")
    print(f"kernel_b_hybrid_grouped {_bench_us(kernel_b_hybrid_grouped):8.3f} us")
    print(f"kernel_b_hybrid_logit   {_bench_us(kernel_b_hybrid_logit_softmax):8.3f} us")
    print("=" * 72)


if __name__ == "__main__":
    main()
