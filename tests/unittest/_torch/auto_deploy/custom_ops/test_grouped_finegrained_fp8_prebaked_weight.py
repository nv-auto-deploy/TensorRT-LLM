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

"""Byte-exactness of the pre-baked BF16 weight path for grouped FineGrained FP8 linear.

The ``bake_grouped_finegrained_fp8_weight`` transform replaces the static FP8 weight of
``torch_fake_quant_grouped_finegrained_fp8_linear`` with its dequantized BF16 value at load
time. These tests prove the op's BF16-weight branch produces bit-for-bit identical output to
the original per-call FP8-dequant branch (the proxy uses random init, so byte-eq is asserted
here rather than via end-to-end output comparison).
"""

import pytest
import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _dequant_block_fp8_weight,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.bake_grouped_finegrained_fp8_weight import (
    BakeGroupedFineGrainedFP8Weight,
)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

pytestmark = [
    pytest.mark.skipif(_FP8_DTYPE is None, reason="Requires torch.float8_e4m3fn"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton act-quant"),
]

_GROUPED_OP = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear


def _make_inputs(num_groups, rank, in_features, scale_shape, batch=2, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out_rows = num_groups * rank
    w_fp8 = torch.randn(
        out_rows, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
    ).to(_FP8_DTYPE)
    scale = torch.rand(scale_shape, generator=gen, device="cuda", dtype=torch.float32) + 0.5
    x = torch.randn(
        batch, num_groups, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    return x, w_fp8, scale


def _bake(w_fp8, scale):
    out_rows, in_features = w_fp8.shape
    scale_n, scale_k = scale.shape
    block_n = -(-out_rows // scale_n)
    block_k = -(-in_features // scale_k)
    return _dequant_block_fp8_weight(w_fp8, scale, block_n, block_k, dtype=torch.bfloat16)


@pytest.mark.parametrize("input_scale_fmt", ["", "ue8m0"])
def test_prebaked_bf16_weight_matches_fp8_path_bitwise(input_scale_fmt):
    num_groups, rank, in_features = 4, 8, 128
    x, w_fp8, scale = _make_inputs(num_groups, rank, in_features, (1, 1))

    out_fp8 = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [], input_scale_fmt=input_scale_fmt)
    w_bf16 = _bake(w_fp8, scale)
    out_baked = _GROUPED_OP(x, w_bf16, None, [], [scale], [], [], input_scale_fmt=input_scale_fmt)

    assert out_fp8.dtype == out_baked.dtype == torch.bfloat16
    assert torch.equal(out_fp8, out_baked)


def test_prebaked_path_with_bias_matches_fp8_path_bitwise():
    num_groups, rank, in_features = 4, 8, 128
    x, w_fp8, scale = _make_inputs(num_groups, rank, in_features, (1, 1), seed=3)
    bias = torch.randn(num_groups * rank, device="cuda", dtype=torch.bfloat16)

    out_fp8 = _GROUPED_OP(x, w_fp8, bias, [], [scale], [], [])
    out_baked = _GROUPED_OP(x, _bake(w_fp8, scale), bias, [], [scale], [], [])

    assert torch.equal(out_fp8, out_baked)


def _build_grouped_fp8_gm(rank, w_fp8, scale):
    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.Module()
            self.attn.wo_a = nn.Module()
            self.attn.wo_a.weight = nn.Parameter(w_fp8, requires_grad=False)
            self.attn.wo_a.register_buffer("weight_scale_inv", scale)

    model = _M()
    graph = Graph()
    x = graph.placeholder("x")
    weight = graph.get_attr("attn.wo_a.weight")
    scale_attr = graph.get_attr("attn.wo_a.weight_scale_inv")
    node = graph.call_function(
        _GROUPED_OP.default,
        args=(x, weight, None, [], [scale_attr], [], []),
        kwargs={
            "tp_mode": "colwise",
            "tp_min_local_shape": rank,
            "layer_type": "mla",
            "input_scale_fmt": "",
        },
    )
    graph.output(node)
    return GraphModule(model, graph)


def test_transform_bakes_weight_to_expected_bf16_and_preserves_output():
    num_groups, rank, in_features = 4, 8, 128
    x, w_fp8, scale = _make_inputs(num_groups, rank, in_features, (1, 1), seed=7)

    out_ref = _GROUPED_OP(x, w_fp8.clone(), None, [], [scale], [], [])

    gm = _build_grouped_fp8_gm(rank, w_fp8.clone(), scale.clone())
    transform = BakeGroupedFineGrainedFP8Weight(TransformConfig(stage=Stages.POST_LOAD_FUSION))
    gm, info = transform._apply(gm, None, None, SharedConfig())

    assert info.num_matches == 1
    baked = gm.get_parameter("attn.wo_a.weight")
    assert baked.dtype == torch.bfloat16
    assert torch.equal(baked, _bake(w_fp8, scale))

    # Re-running the transform is a no-op (weight is no longer FP8).
    gm, info2 = transform._apply(gm, None, None, SharedConfig())
    assert info2.num_matches == 0

    out_baked = gm(x)
    assert torch.equal(out_ref, out_baked)
