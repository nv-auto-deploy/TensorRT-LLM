# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for multi-stream router/experts parallelism in MoE blocks.

The transform should:
  1. Detect a fork point that feeds both ``torch_moe_router`` and
     ``torch_moe_dense_mlp`` consuming the same hidden tensor.
  2. Move the router op onto the auxiliary CUDA stream.
  3. Preserve numerical correctness.
  4. Be compatible with CUDA graph capture & replay.
  5. Skip when MoE has been fused away (e.g., MXFP4 path) — no router op left.
"""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_router import (
    _execute_router_in_aux_stream,
    _find_router_expert_pairs,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import cuda_stream_manager


class MockMoEBlock(nn.Module):
    """Minimal MoE block: torch_moe_router followed by torch_moe_dense_mlp."""

    def __init__(self, hidden: int, num_experts: int, intermediate: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.alpha = 1.0
        self.limit = 10.0
        self.router_w = nn.Parameter(torch.randn(num_experts, hidden) * 0.02)
        self.router_b = nn.Parameter(torch.zeros(num_experts))
        self.gate_up_w = nn.Parameter(torch.randn(num_experts, hidden, 2 * intermediate) * 0.02)
        self.gate_up_b = nn.Parameter(torch.zeros(num_experts, 2 * intermediate))
        self.down_w = nn.Parameter(torch.randn(num_experts, intermediate, hidden) * 0.02)
        self.down_b = nn.Parameter(torch.zeros(num_experts, hidden))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        routing = torch.ops.auto_deploy.torch_moe_router(
            hidden_states, self.router_w, self.router_b, self.top_k
        )
        out = torch.ops.auto_deploy.torch_moe_dense_mlp(
            hidden_states,
            routing,
            self.gate_up_w,
            self.gate_up_b,
            self.down_w,
            self.down_b,
            self.alpha,
            self.limit,
        )
        return out


def _build_gm(model, example_input):
    egm = torch.export.export(model, (example_input,))
    return egm.module()


def test_pattern_matching_router_experts():
    """Router followed by dense MLP both consuming the same hidden tensor matches once."""
    model = MockMoEBlock(hidden=32, num_experts=4, intermediate=16, top_k=2).eval().to("cuda")
    example_input = torch.randn(2, 4, 32, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_router_expert_pairs(gm)
    assert len(pairs) == 1, f"Expected 1 router/experts pair, got {len(pairs)}"


def test_no_match_when_router_absent():
    """A graph with no torch_moe_router op (e.g., MXFP4-fused) yields zero matches."""

    class NoRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(16, 16, bias=False)

        def forward(self, x):
            return self.lin(x)

    model = NoRouter().eval().to("cuda")
    example_input = torch.randn(4, 16, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_router_expert_pairs(gm)
    assert len(pairs) == 0, f"Expected 0 matches without a router op, got {len(pairs)}"


def test_numerical_correctness():
    """After the rewrite the GraphModule must match the original output."""
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMoEBlock(hidden=32, num_experts=4, intermediate=16, top_k=2).eval().to("cuda")
    example_input = torch.randn(2, 4, 32, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(2, 4, 32, device="cuda")
    ref = model(test_x)

    gm, num_replaced = _execute_router_in_aux_stream(gm)
    assert num_replaced == 1, f"Expected 1 replacement, got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref, atol=1e-4), (
        f"Output mismatch: max diff = {(y - ref).abs().max().item()}"
    )


def test_cuda_graph_compatibility():
    """The transformed GraphModule must work under CUDA graph capture and replay."""
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMoEBlock(hidden=32, num_experts=4, intermediate=16, top_k=2).eval().to("cuda")
    example_input = torch.randn(2, 4, 32, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(2, 4, 32, device="cuda")
    ref = model(test_x)

    gm, num_replaced = _execute_router_in_aux_stream(gm)
    assert num_replaced == 1

    static_x = torch.randn(2, 4, 32, device="cuda")
    static_y = torch.randn(2, 4, 32, device="cuda")

    for _ in range(3):
        static_y.copy_(gm(static_x))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y.copy_(gm(static_x))

    static_x.copy_(test_x)
    graph.replay()

    assert torch.allclose(static_y, ref, atol=1e-4), (
        f"CUDA graph output mismatch: max diff = {(static_y - ref).abs().max().item()}"
    )
