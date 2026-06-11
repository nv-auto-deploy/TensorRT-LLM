# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the fused Step-3.7-Flash MoE routing custom op.

The op ``auto_deploy::step3p7_fused_router_topk`` fuses the 7 separate torch
ops of Step's sigmoid + per-expert-bias top-k router into a CuteDSL kernel.
These tests validate that it is (1) numerically faithful to the
separate-op reference, (2) safe to capture/replay inside a CUDA graph (the
production execution mode), and (3) traceable by ``torch.export`` (uses
``register_fake`` and a ``torch.dtype`` argument).
"""

import pytest
import torch

# Importing the model module registers the custom op at import time.
from tensorrt_llm._torch.auto_deploy.models.custom import modeling_step3p7


def _reference_routing(router_logits, router_bias, top_k, scaling, out_dtype):
    """The exact separate-op reference replaced by the fused op."""
    probs = torch.sigmoid(router_logits)
    scores = probs + router_bias.unsqueeze(0)
    _, selected = torch.topk(scores, top_k, dim=-1)
    weights = torch.gather(probs, 1, selected)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    weights = weights * scaling
    weights = weights.to(out_dtype)
    return weights, selected


def _sorted_by_index(weights, indices):
    s = indices.sort(dim=-1)
    return weights.gather(-1, s.indices).float(), s.values


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(modeling_step3p7.cute is None, reason="requires Cutlass CuteDSL")
@pytest.mark.parametrize("num_tokens", [1, 4, 17])
@pytest.mark.parametrize("num_experts,top_k", [(288, 8)])
@pytest.mark.parametrize("bias_dtype", [torch.float32, torch.bfloat16])
def test_fused_router_topk_matches_reference(num_tokens, num_experts, top_k, bias_dtype):
    device = "cuda"
    torch.manual_seed(1234)
    # Scale logits so the top-k boundary is well separated (no near-ties that
    # could differ between torch.topk and the iterative-argmax kernel).
    router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) * 2.0
    router_bias = (torch.randn(num_experts, dtype=torch.float32, device=device) * 0.5).to(
        bias_dtype
    )
    scaling = 3.0
    out_dtype = torch.bfloat16

    ref_w, ref_i = _reference_routing(router_logits, router_bias, top_k, scaling, out_dtype)
    out_w, out_i = torch.ops.auto_deploy.step3p7_fused_router_topk(
        router_logits, router_bias, top_k, scaling, out_dtype
    )

    assert out_w.shape == (num_tokens, top_k)
    assert out_i.shape == (num_tokens, top_k)
    assert out_w.dtype == out_dtype
    assert out_i.dtype == torch.int64

    ref_ws, ref_is = _sorted_by_index(ref_w, ref_i)
    out_ws, out_is = _sorted_by_index(out_w, out_i)

    # Same set of selected experts (order within top_k is irrelevant downstream).
    assert torch.equal(ref_is.to(torch.int64), out_is.to(torch.int64)), (
        f"expert selection differs:\nref={ref_is}\nout={out_is}"
    )
    # Same renormalized + scaled weights (bf16 rounding tolerance).
    torch.testing.assert_close(out_ws, ref_ws, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(modeling_step3p7.cute is None, reason="requires Cutlass CuteDSL")
def test_fused_router_topk_cuda_graph():
    """The op must be capturable and replayable inside a CUDA graph."""
    device = "cuda"
    num_tokens, num_experts, top_k = 1, 288, 8
    scaling, out_dtype = 3.0, torch.bfloat16
    torch.manual_seed(7)

    static_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) * 2.0
    static_bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.5

    op = torch.ops.auto_deploy.step3p7_fused_router_topk
    # Eager reference for this exact input.
    eager_w, eager_i = op(static_logits, static_bias, top_k, scaling, out_dtype)

    static_w = torch.empty_like(eager_w)
    static_i = torch.empty_like(eager_i)

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            op(static_logits, static_bias, top_k, scaling, out_dtype)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        w, i = op(static_logits, static_bias, top_k, scaling, out_dtype)
        static_w.copy_(w)
        static_i.copy_(i)

    # Replay with new data copied into the captured input buffers.
    torch.manual_seed(99)
    new_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) * 2.0
    new_bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.5
    static_logits.copy_(new_logits)
    static_bias.copy_(new_bias)
    g.replay()
    torch.cuda.synchronize()

    ref_w, ref_i = _reference_routing(new_logits, new_bias, top_k, scaling, out_dtype)
    ref_ws, ref_is = _sorted_by_index(ref_w, ref_i)
    out_ws, out_is = _sorted_by_index(static_w, static_i)
    assert torch.equal(ref_is.to(torch.int64), out_is.to(torch.int64))
    torch.testing.assert_close(out_ws, ref_ws, atol=2e-2, rtol=2e-2)


def test_fused_router_topk_exports():
    """register_fake + torch.dtype arg must allow torch.export (meta path)."""

    class M(torch.nn.Module):
        def __init__(self, top_k, scaling, num_experts):
            super().__init__()
            self.top_k = top_k
            self.scaling = scaling
            self.register_buffer("bias", torch.zeros(num_experts, dtype=torch.float32))

        def forward(self, logits):
            return torch.ops.auto_deploy.step3p7_fused_router_topk(
                logits, self.bias, self.top_k, self.scaling, torch.bfloat16
            )

    m = M(8, 3.0, 288)
    ep = torch.export.export(m, (torch.randn(4, 288, dtype=torch.float32),))
    assert ep is not None
    # The fused op should appear exactly once as a single graph node.
    target = torch.ops.auto_deploy.step3p7_fused_router_topk.default
    n = sum(1 for node in ep.graph.nodes if node.op == "call_function" and node.target is target)
    assert n == 1, f"expected exactly one fused routing node, found {n}"
