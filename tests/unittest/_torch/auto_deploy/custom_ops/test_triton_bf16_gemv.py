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

"""Unit tests for the ``auto_deploy::triton_bf16_gemv_linear`` custom op.

The M==1 Triton path promises fp32 accumulation over bf16 inputs (cuBLAS-equivalent
up to fp32 summation order); every other path must fall back to ``aten.linear``
bit-exactly. The op must also stay CUDA-graph capturable with deterministic replay.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.linear.triton_bf16_gemv import (
    triton_bf16_gemv_head_gate_linear,
    triton_bf16_gemv_linear,
)

# Registers auto_deploy::step3p7_head_gate (the unfused reference for the fused
# head-gate GEMV, and its fallback path).
from tensorrt_llm._torch.auto_deploy.models.custom import modeling_step3p7  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# The Step-3.7-Flash per-rank TP8 decode shapes this op is deployed on.
PRODUCTION_SHAPES = [
    (1288, 4096),  # fused qkvg, full attn
    (1804, 4096),  # fused qkvg, sliding attn
    (4096, 1024),  # o_proj, full attn
    (4096, 1536),  # o_proj, sliding attn
    (2816, 4096),  # dense MLP fused gate_up
    (4096, 1408),  # dense MLP down
]

# Shapes off the config table: heuristic fallback, N-masking edges, minimal/wide K.
EDGE_SHAPES = [
    (1001, 128),
    (17, 256),
    (5120, 2048),
    (33, 8192),
    (127, 384),
]


def _make_inputs(n, k, batch_shape=(1,), dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed + n * 7 + k)
    x = (0.1 * torch.randn(*batch_shape, k, device="cuda", generator=gen, dtype=torch.float32)).to(
        dtype
    )
    w = (0.1 * torch.randn(n, k, device="cuda", generator=gen, dtype=torch.float32)).to(dtype)
    return x, w


@pytest.mark.parametrize("n,k", PRODUCTION_SHAPES + EDGE_SHAPES)
def test_m1_matches_fp32_reference(n, k):
    x, w = _make_inputs(n, k)
    y = triton_bf16_gemv_linear(x, w)
    assert y.shape == (1, n)
    assert y.dtype == torch.bfloat16
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    # Same bf16 inputs on both paths; only fp32 summation order may differ.
    torch.testing.assert_close(y, ref, rtol=2e-2, atol=2e-3)


def test_m1_3d_input_shape():
    n, k = 4096, 1024
    x, w = _make_inputs(n, k, batch_shape=(1, 1))
    y = triton_bf16_gemv_linear(x, w)
    assert y.shape == (1, 1, n)
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(y, ref, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize(
    "batch_shape",
    [(4,), (2, 3)],
)
def test_m_gt_1_falls_back_bit_exact(batch_shape):
    n, k = 1288, 4096
    x, w = _make_inputs(n, k, batch_shape=batch_shape)
    y = triton_bf16_gemv_linear(x, w)
    ref = torch.ops.aten.linear.default(x, w, None)
    assert torch.equal(y, ref)


def test_non_bf16_falls_back_bit_exact():
    n, k = 1288, 4096
    x, w = _make_inputs(n, k, dtype=torch.float16)
    y = triton_bf16_gemv_linear(x, w)
    ref = torch.ops.aten.linear.default(x, w, None)
    assert torch.equal(y, ref)


def test_k_not_multiple_of_128_falls_back_bit_exact():
    n, k = 64, 192
    x, w = _make_inputs(n, k)
    y = triton_bf16_gemv_linear(x, w)
    ref = torch.ops.aten.linear.default(x, w, None)
    assert torch.equal(y, ref)


# The Step-3.7-Flash per-rank TP8 o_proj shapes the fused head-gate GEMV serves:
# (out_features, local_heads, head_dim). K = local_heads * head_dim.
HEAD_GATE_SHAPES = [
    (4096, 8, 128),  # o_proj, full attn (K=1024, loop kernel)
    (4096, 12, 128),  # o_proj, sliding attn (K=1536, peel2 kernel)
]


def _make_head_gate_inputs(n, h, d, batch_shape=(1, 1), seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed + n + h * 31 + d)
    attn = (
        0.1 * torch.randn(*batch_shape, h, d, device="cuda", generator=gen, dtype=torch.float32)
    ).to(torch.bfloat16)
    gate = (torch.randn(*batch_shape, h, device="cuda", generator=gen, dtype=torch.float32)).to(
        torch.bfloat16
    )
    w = (0.1 * torch.randn(n, h * d, device="cuda", generator=gen, dtype=torch.float32)).to(
        torch.bfloat16
    )
    return attn, gate, w


def _unfused_head_gate_chain(attn, gate, w):
    gated = torch.ops.auto_deploy.step3p7_head_gate(attn, gate)
    flat = gated.reshape(*gated.shape[:-2], gated.shape[-2] * gated.shape[-1])
    return triton_bf16_gemv_linear(flat, w)


@pytest.mark.parametrize("n,h,d", HEAD_GATE_SHAPES)
def test_head_gate_m1_bit_exact_vs_unfused_chain(n, h, d):
    """M==1 fused path must match the unfused head_gate -> reshape -> gemv bitwise."""
    attn, gate, w = _make_head_gate_inputs(n, h, d)
    y = triton_bf16_gemv_head_gate_linear(attn, gate, w)
    assert y.shape == (1, 1, n)
    assert y.dtype == torch.bfloat16
    ref = _unfused_head_gate_chain(attn, gate, w)
    assert torch.equal(y, ref)


@pytest.mark.parametrize("n,h,d", HEAD_GATE_SHAPES)
def test_head_gate_m1_matches_fp32_reference(n, h, d):
    attn, gate, w = _make_head_gate_inputs(n, h, d)
    y = triton_bf16_gemv_head_gate_linear(attn, gate, w)
    gated = (attn.float() * gate.float().sigmoid().unsqueeze(-1)).reshape(1, 1, h * d)
    ref = (gated @ w.float().T).to(torch.bfloat16)
    torch.testing.assert_close(y, ref, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("batch_shape", [(4,), (2, 3)])
def test_head_gate_m_gt_1_falls_back_bit_exact(batch_shape):
    n, h, d = 4096, 8, 128
    attn, gate, w = _make_head_gate_inputs(n, h, d, batch_shape=batch_shape)
    y = triton_bf16_gemv_head_gate_linear(attn, gate, w)
    gated = torch.ops.auto_deploy.step3p7_head_gate(attn, gate)
    ref = torch.ops.aten.linear.default(gated.reshape(*batch_shape, h * d), w, None)
    assert y.shape == (*batch_shape, n)
    assert torch.equal(y, ref)


def test_head_gate_non_pow2_head_dim_falls_back_bit_exact():
    n, h, d = 512, 4, 96
    attn, gate, w = _make_head_gate_inputs(n, h, d)
    y = triton_bf16_gemv_head_gate_linear(attn, gate, w)
    gated = torch.ops.auto_deploy.step3p7_head_gate(attn, gate)
    ref = torch.ops.aten.linear.default(gated.reshape(1, 1, h * d), w, None)
    assert torch.equal(y, ref)


@pytest.mark.parametrize("n,h,d", HEAD_GATE_SHAPES)
def test_head_gate_cuda_graph_capture_replay_matches_eager(n, h, d):
    attn, gate, w = _make_head_gate_inputs(n, h, d)
    ref = triton_bf16_gemv_head_gate_linear(attn, gate, w).clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = triton_bf16_gemv_head_gate_linear(attn, gate, w)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = triton_bf16_gemv_head_gate_linear(attn, gate, w)

    y.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, ref)


def test_cuda_graph_capture_replay_matches_eager():
    """The op must stay capturable, and replay must be bit-identical to eager."""
    n, k = 1288, 4096
    x, w = _make_inputs(n, k)
    ref = triton_bf16_gemv_linear(x, w).clone()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            y = triton_bf16_gemv_linear(x, w)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = triton_bf16_gemv_linear(x, w)

    y.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, ref)
