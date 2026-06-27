# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness check for the selected-expert MXFP4 dequant refactor.

``_run_torch_mxfp4_from_routing_core`` was changed to gather each routing slot's
packed (uint8) blocks/scales *before* the MXFP4 dequant, so the data-volume-bound
``table[blocks]`` dequant runs over ``num_tokens * top_k`` slots instead of all
local experts. Because ``_decode_mxfp4_blocks`` is elementwise over the leading
expert dim, ``decode(blocks[idx]) == decode(blocks)[idx]``, so the new path must
be bit-identical to the previous "dequant all experts, then index_select" path.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_moe import (
    _apply_swiglu,
    _decode_mxfp4_blocks,
    _run_torch_mxfp4_from_routing_core,
)


def _reference_dequant_all_then_select(
    hidden_states,
    selected_experts,
    routing_weights,
    gate_up_blocks,
    gate_up_bias,
    gate_up_scales,
    alpha,
    limit,
    down_blocks,
    down_bias,
    down_scales,
    expert_start=0,
    gate_up_order="up_gate",
    swiglu_mode="deepseek",
):
    """Previous algorithm: dequant ALL experts, then index_select per slot."""
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size).to(torch.float32)
    selected_experts = selected_experts.reshape(x.shape[0], -1).to(torch.int64)
    routing_weights = routing_weights.reshape_as(selected_experts).to(torch.float32)
    gate_up_weight = _decode_mxfp4_blocks(gate_up_blocks, gate_up_scales)
    down_weight = _decode_mxfp4_blocks(down_blocks, down_scales)

    output = torch.zeros((x.shape[0], hidden_size), device=x.device, dtype=torch.float32)
    local_experts = gate_up_weight.shape[0]
    local_expert_idx = selected_experts - int(expert_start)
    valid_route = (local_expert_idx >= 0) & (local_expert_idx < local_experts)
    local_expert_idx = local_expert_idx.clamp(0, local_experts - 1).to(torch.int64)
    x_for_bmm = x.unsqueeze(-1)
    chunk = 16
    for route_idx in range(local_expert_idx.shape[1]):
        for start in range(0, x.shape[0], chunk):
            end = min(start + chunk, x.shape[0])
            ts = slice(start, end)
            ei = local_expert_idx[ts, route_idx]
            gu = torch.bmm(gate_up_weight.index_select(0, ei), x_for_bmm[ts]).squeeze(-1)
            gu = gu + gate_up_bias.index_select(0, ei).to(torch.float32)
            inter = _apply_swiglu(gu, alpha, limit, gate_up_order, swiglu_mode)
            eo = torch.bmm(down_weight.index_select(0, ei), inter.unsqueeze(-1)).squeeze(-1)
            eo = eo + down_bias.index_select(0, ei).to(torch.float32)
            rs = routing_weights[ts, route_idx, None] * valid_route[ts, route_idx, None].to(
                torch.float32
            )
            output[ts] = output[ts] + eo * rs
    return output.reshape(*leading_shape, hidden_size).to(hidden_states.dtype)


def _make_inputs(num_tokens, top_k, num_experts, hidden, inter, expert_start, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    block = 32
    packed = block // 2
    gate_up_blocks = torch.randint(
        0, 256, (num_experts, 2 * inter, hidden // block, packed), dtype=torch.uint8, generator=g
    ).to(device)
    gate_up_scales = torch.randint(
        110, 135, (num_experts, 2 * inter, hidden // block), dtype=torch.uint8, generator=g
    ).to(device)
    down_blocks = torch.randint(
        0, 256, (num_experts, hidden, inter // block, packed), dtype=torch.uint8, generator=g
    ).to(device)
    down_scales = torch.randint(
        110, 135, (num_experts, hidden, inter // block), dtype=torch.uint8, generator=g
    ).to(device)
    gate_up_bias = torch.randn(num_experts, 2 * inter, generator=g).to(device)
    down_bias = torch.randn(num_experts, hidden, generator=g).to(device)
    hidden_states = torch.randn(num_tokens, hidden, generator=g).to(device).to(torch.bfloat16)
    # Mix in-range, out-of-range (masked), and (with expert_start) below-range slots.
    selected = torch.randint(
        expert_start - 1, expert_start + num_experts + 2, (num_tokens, top_k), generator=g
    ).to(device)
    routing_weights = torch.rand(num_tokens, top_k, generator=g).to(device)
    return (
        hidden_states,
        selected,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        down_blocks,
        down_bias,
        down_scales,
    )


# (num_tokens, top_k, num_experts) chosen to exercise both the selected (decode:
# num_tokens*top_k < num_experts) and dense (prefill: >=) code paths.
@pytest.mark.parametrize(
    "num_tokens,top_k,num_experts",
    [
        (1, 6, 32),  # selected
        (2, 6, 32),  # selected
        (1, 6, 8),  # selected (boundary)
        (8, 6, 16),  # dense
        (3, 4, 8),  # dense
        (16, 6, 8),  # dense (many tokens)
    ],
)
@pytest.mark.parametrize("expert_start", [0, 8])
def test_selected_dequant_bit_exact(num_tokens, top_k, num_experts, expert_start):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden, inter = 64, 32
    inputs = _make_inputs(
        num_tokens, top_k, num_experts, hidden, inter, expert_start, seed=1234, device=device
    )
    (
        hidden_states,
        selected,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        down_blocks,
        down_bias,
        down_scales,
    ) = inputs
    alpha, limit = 1.0, 7.0

    got = _run_torch_mxfp4_from_routing_core(
        hidden_states,
        selected,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        expert_start=expert_start,
        gate_up_order="up_gate",
        swiglu_mode="deepseek",
    )
    ref = _reference_dequant_all_then_select(
        hidden_states,
        selected,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        expert_start=expert_start,
        gate_up_order="up_gate",
        swiglu_mode="deepseek",
    )
    assert got.shape == ref.shape
    assert torch.equal(got, ref), (got - ref).abs().max().item()


if __name__ == "__main__":
    for nt, tk, ne in [(1, 6, 32), (2, 6, 32), (1, 6, 8), (8, 6, 16), (3, 4, 8), (16, 6, 8)]:
        for es in [0, 8]:
            test_selected_dequant_bit_exact(nt, tk, ne, es)
            branch = "selected" if nt * tk < ne else "dense"
            print(f"OK num_tokens={nt} top_k={tk} num_experts={ne} expert_start={es} [{branch}]")
