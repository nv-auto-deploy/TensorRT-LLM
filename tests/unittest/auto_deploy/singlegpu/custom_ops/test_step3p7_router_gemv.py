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

"""Unit tests for the Step-3.7-Flash router gate GEMV custom op.

The op ``auto_deploy::step3p7_router_gemv`` computes the fp32 router logits from a bf16
weight read (fp32 accumulate, fp32 store) on the batch=1 decode hot path, and falls back
to the reference fp32 GEMM (upcasting both operands) for every other shape/dtype/device.
These tests validate that it is (1) numerically faithful to the fp32 reference GEMM up to
summation order, with the routing decision (top-k expert set) unchanged, (2) bit-identical
to the reference on the fallback paths, (3) safe to capture/replay inside a CUDA graph (the
production execution mode), and (4) traceable by ``torch.export``.
"""

import pytest
import torch
import torch.nn.functional as F

# Importing the model module registers the custom op at import time.
from tensorrt_llm._torch.auto_deploy.models.custom import modeling_step3p7  # noqa: F401

NUM_EXPERTS, HIDDEN = 288, 4096


def _reference_logits(hidden, weight):
    """The exact fp32 GEMM the op replaces (fp32-materialized weight)."""
    return F.linear(hidden.float(), weight.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_gemv_matches_fp32_reference():
    device = "cuda"
    torch.manual_seed(1234)
    hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)

    out = torch.ops.auto_deploy.step3p7_router_gemv(hidden, weight)
    ref = _reference_logits(hidden, weight)

    assert out.shape == (1, NUM_EXPERTS)
    assert out.dtype == torch.float32
    # bf16 x bf16 products are exact in fp32; only the summation order differs from the
    # cuBLAS fp32 GEMM, so the logits agree to fp32 round-off of a 4096-term sum.
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_gemv_preserves_topk_selection():
    """The routing decision (top-8 by sigmoid + bias) must match the fp32 reference."""
    device = "cuda"
    torch.manual_seed(42)
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device=device) * 0.5
    weight = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)

    flips = 0
    for seed in range(64):
        torch.manual_seed(1000 + seed)
        hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
        out = torch.ops.auto_deploy.step3p7_router_gemv(hidden, weight)
        ref = _reference_logits(hidden, weight)
        top_out = (torch.sigmoid(out) + bias).topk(8, dim=-1).indices.sort(-1).values
        top_ref = (torch.sigmoid(ref) + bias).topk(8, dim=-1).indices.sort(-1).values
        flips += int(not torch.equal(top_out, top_ref))
    assert flips == 0, f"top-8 expert selection flipped on {flips}/64 tokens"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [2, 17, 128])
def test_router_gemv_multi_token_fallback_bit_identical(num_tokens):
    """T > 1 (prefill / multi-token decode) must be bit-identical to the fp32 GEMM."""
    device = "cuda"
    torch.manual_seed(7)
    hidden = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)

    out = torch.ops.auto_deploy.step3p7_router_gemv(hidden, weight)
    assert out.dtype == torch.float32
    assert torch.equal(out, _reference_logits(hidden, weight))


def test_router_gemv_cpu_fallback_bit_identical():
    """Non-CUDA inputs (offline sharding-IR equivalence harness) use the reference GEMM."""
    torch.manual_seed(3)
    hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16)
    weight = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16)

    out = torch.ops.auto_deploy.step3p7_router_gemv(hidden, weight)
    assert torch.equal(out, _reference_logits(hidden, weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_gemv_fp32_weight_fallback_bit_identical():
    """An fp32 gate weight (e.g. after a blanket ``.float()``) uses the reference GEMM."""
    device = "cuda"
    torch.manual_seed(13)
    hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.float32, device=device)

    out = torch.ops.auto_deploy.step3p7_router_gemv(hidden, weight)
    assert torch.equal(out, F.linear(hidden.float(), weight))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_router_gemv_cuda_graph():
    """The op must be capturable and replayable inside a CUDA graph."""
    device = "cuda"
    torch.manual_seed(11)
    static_hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16, device=device)

    op = torch.ops.auto_deploy.step3p7_router_gemv

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            op(static_hidden, weight)
    torch.cuda.current_stream().wait_stream(s)

    static_out = torch.empty(1, NUM_EXPERTS, dtype=torch.float32, device=device)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out.copy_(op(static_hidden, weight))

    # Replay with new data copied into the captured input buffer.
    torch.manual_seed(23)
    new_hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device)
    static_hidden.copy_(new_hidden)
    g.replay()
    torch.cuda.synchronize()

    ref = _reference_logits(new_hidden, weight)
    torch.testing.assert_close(static_out, ref, atol=5e-4, rtol=1e-5)


def test_router_gemv_exports():
    """register_fake must allow torch.export (meta path) with fp32 output shape."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.zeros(NUM_EXPERTS, HIDDEN, dtype=torch.bfloat16))

        def forward(self, hidden):
            return torch.ops.auto_deploy.step3p7_router_gemv(hidden, self.weight)

    m = M()
    ep = torch.export.export(m, (torch.randn(4, HIDDEN, dtype=torch.bfloat16),))
    assert ep is not None
    target = torch.ops.auto_deploy.step3p7_router_gemv.default
    n = sum(1 for node in ep.graph.nodes if node.op == "call_function" and node.target is target)
    assert n == 1, f"expected exactly one router GEMV node, found {n}"


def test_moe_gate_weight_pinned_bf16():
    """Blanket dtype casts must not change the bf16-by-construction gate weight dtype."""
    config = modeling_step3p7.Step3p7Config(
        hidden_size=64,
        intermediate_size=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=32,
    )
    moe = modeling_step3p7.Step3p7MoE(config)
    assert moe.gate.weight.dtype == torch.bfloat16
    moe.to(torch.float32)  # e.g. an offline harness upcast
    assert moe.gate.weight.dtype == torch.bfloat16
    moe.to(torch.bfloat16)  # the HF fp8 quant reader post-process cast
    assert moe.gate.weight.dtype == torch.bfloat16
