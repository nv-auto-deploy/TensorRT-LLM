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

"""Unit tests for the M==1 Triton bf16 GEMV custom op and its swap transform.

``auto_deploy::triton_bf16_gemv_linear`` replaces cuBLAS for bias-free bf16
projections on the batch=1 decode hot path (fp32 accumulate, bf16 store) and
falls back to ``aten.linear`` everywhere else. The ``swap_linear_gemv``
transform retargets eligible ``torch_linear_simple`` graph nodes to the op.
Tests validate (1) numerical faithfulness to the fp32 reference on all deployed
Step-3.7-Flash per-rank shapes, (2) bit-identical fallback paths, (3) CUDA
graph capture/replay safety (the production execution mode), and (4) the
transform's node retargeting and eligibility gating.
"""

import pytest
import torch
import torch.nn.functional as F

# Importing the custom-op module registers the op at import time.
import tensorrt_llm._torch.auto_deploy.custom_ops.linear.triton_bf16_gemv  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.library.swap_linear_gemv import SwapLinearGemv

# Per-rank (TP8) decode GEMV shapes of Step-3.7-Flash covered by the config table.
PRODUCTION_SHAPES = [
    (1288, 4096),  # fused qkvg, full attention
    (1804, 4096),  # fused qkvg, sliding attention
    (4096, 1024),  # o_proj, full attention
    (4096, 1536),  # o_proj, sliding attention
    (2816, 4096),  # dense MLP fused gate_up
    (4096, 1408),  # dense MLP down
    (512, 1280),  # off-table shape -> heuristic config
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n,k", PRODUCTION_SHAPES)
def test_gemv_matches_fp32_reference(n, k):
    torch.manual_seed(1234)
    x = torch.randn(1, 1, k, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02

    out = torch.ops.auto_deploy.triton_bf16_gemv_linear(x, w)
    ref = F.linear(x.float(), w.float())
    cublas = F.linear(x, w)

    assert out.shape == (1, 1, n)
    assert out.dtype == torch.bfloat16
    # bf16 x bf16 products are exact in fp32; the Triton GEMV differs from the fp32
    # reference only by summation order + the final bf16 round, i.e. by no more than
    # cuBLAS itself (also fp32-accumulate) plus one bf16 ulp.
    tol = (cublas.float() - ref).abs().max().item() + 2**-8 * ref.abs().max().item()
    assert (out.float() - ref).abs().max().item() <= tol


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [2, 17])
def test_gemv_multi_token_fallback_bit_identical(num_tokens):
    torch.manual_seed(1234)
    x = torch.randn(1, num_tokens, 4096, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1288, 4096, dtype=torch.bfloat16, device="cuda")
    out = torch.ops.auto_deploy.triton_bf16_gemv_linear(x, w)
    assert torch.equal(out, F.linear(x, w, None))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemv_non_multiple_k_fallback_bit_identical():
    torch.manual_seed(1234)
    x = torch.randn(1, 1, 4160, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(1288, 4160, dtype=torch.bfloat16, device="cuda")  # 4160 % 128 != 0
    out = torch.ops.auto_deploy.triton_bf16_gemv_linear(x, w)
    assert torch.equal(out, F.linear(x, w, None))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemv_cuda_graph():
    """Capture/replay the M==1 Triton path inside a CUDA graph (production mode)."""
    torch.manual_seed(1234)
    n, k = 1288, 4096
    x = torch.randn(1, 1, k, dtype=torch.bfloat16, device="cuda") * 0.1
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.02

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops.auto_deploy.triton_bf16_gemv_linear(x, w)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.ops.auto_deploy.triton_bf16_gemv_linear(x, w)

    x_new = torch.randn_like(x) * 0.1
    x.copy_(x_new)
    graph.replay()
    torch.cuda.synchronize()
    ref = torch.ops.auto_deploy.triton_bf16_gemv_linear(x_new, w)
    assert torch.equal(out, ref)


class _TwoLinears(torch.nn.Module):
    """One eligible projection, one too-small projection, one with bias."""

    def __init__(self):
        super().__init__()
        self.w_big = torch.nn.Parameter(
            torch.randn(1288, 4096, dtype=torch.bfloat16) * 0.02, requires_grad=False
        )
        self.w_small = torch.nn.Parameter(
            torch.randn(320, 4096, dtype=torch.bfloat16) * 0.02, requires_grad=False
        )
        self.w_bias = torch.nn.Parameter(
            torch.randn(2048, 4096, dtype=torch.bfloat16) * 0.02, requires_grad=False
        )
        self.b = torch.nn.Parameter(torch.zeros(2048, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, x):
        big = torch.ops.auto_deploy.torch_linear_simple(x, self.w_big, None)
        small = torch.ops.auto_deploy.torch_linear_simple(x, self.w_small, None)
        biased = torch.ops.auto_deploy.torch_linear_simple(x, self.w_bias, self.b)
        return big.sum() + small.sum() + biased.sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_swap_linear_gemv_transform():
    torch.manual_seed(1234)
    mod = _TwoLinears().cuda()
    x = torch.randn(1, 1, 4096, dtype=torch.bfloat16, device="cuda") * 0.1
    gm = torch.fx.symbolic_trace(mod)
    ref = gm(x)

    transform = SwapLinearGemv.from_kwargs(stage="post_load_fusion")
    gm, info = transform._apply(gm, None, None, None)
    gm.recompile()

    targets = [n.target for n in gm.graph.nodes if n.op == "call_function"]
    assert info.num_matches == 1
    assert torch.ops.auto_deploy.triton_bf16_gemv_linear.default in targets
    # small and biased linears keep their original target (symbolic_trace records the
    # OpOverloadPacket; torch.export would record the .default overload)
    simple_targets = (
        torch.ops.auto_deploy.torch_linear_simple,
        torch.ops.auto_deploy.torch_linear_simple.default,
    )
    assert sum(t in simple_targets for t in targets) == 2

    out = gm(x)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
