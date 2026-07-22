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
"""Bit-exactness tests for torch_fp8_swiglu_clamp_act_quant and its fusion transform."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (register ops)
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import _safe_act_quant
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_quant import FuseFP8SwigluActQuant
from tensorrt_llm._torch.auto_deploy.transform.library.fusion import (
    FuseGemmsMixedChildren,
    FuseGemmsMixedChildrenConfig,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

torch.manual_seed(0)

BLOCK_SIZE = 128
LIMIT = 10.0  # DeepSeek-V4-Flash swiglu_limit


def _fp8_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _ref_chain(gate, up, limit, block_size, fmt, model_dtype):
    """The eager chain the transform replaces (modeling_deepseek_v4.DeepseekV4MLP)."""
    if limit is not None and limit > 0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
    hidden = (F.silu(gate.float()) * up.float()).to(model_dtype)
    return _safe_act_quant(hidden.contiguous(), block_size, fmt)


def _assert_pair_equal(got, ref, ctx):
    q, s = got
    q_ref, s_ref = ref
    assert q.dtype == q_ref.dtype == torch.float8_e4m3fn
    assert q.shape == q_ref.shape and s.shape == s_ref.shape
    assert s.dtype == s_ref.dtype
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), f"{ctx}: fp8 payload differs"
    assert torch.equal(s, s_ref), f"{ctx}: per-block scales differ"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("M", [1, 5, 64])
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
@pytest.mark.parametrize("limit", [LIMIT, None])
def test_fused_swiglu_act_quant_bit_exact(M, fmt, limit):
    device = "cuda"
    dtype = torch.bfloat16
    width = 512
    gate = torch.randn(M, width, device=device, dtype=dtype) * 6.0
    up = torch.randn(M, width, device=device, dtype=dtype) * 6.0
    if limit is not None:
        assert (gate > limit).any() and (up.abs() > limit).any(), "clamp path not exercised"

    ref = _ref_chain(gate, up, limit, BLOCK_SIZE, fmt, dtype)
    got = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, limit, BLOCK_SIZE, fmt)
    _assert_pair_equal(got, ref, f"M={M} fmt={fmt!r} limit={limit}")


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_fused_swiglu_act_quant_strided_views(fmt):
    """Production form: gate/up are the two narrow views of one merged [M, 2I] GEMM out."""
    device = "cuda"
    dtype = torch.bfloat16
    M, width = 3, 512
    gate_up = torch.randn(M, 2 * width, device=device, dtype=dtype) * 6.0
    gate = gate_up.narrow(-1, 0, width)
    up = gate_up.narrow(-1, width, width)

    ref = _ref_chain(gate, up, LIMIT, BLOCK_SIZE, fmt, dtype)
    got = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, LIMIT, BLOCK_SIZE, fmt)
    _assert_pair_equal(got, ref, f"strided fmt={fmt!r}")


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_fused_swiglu_act_quant_edge_values(fmt):
    device = "cuda"
    dtype = torch.bfloat16
    width = 256
    rows = [
        torch.zeros(width, device=device, dtype=dtype),  # all-zero block -> scale floor
        torch.full((width,), LIMIT, device=device, dtype=dtype),  # exactly at the bound
        torch.full((width,), -LIMIT, device=device, dtype=dtype),
        torch.full((width,), 1e4, device=device, dtype=dtype),
        torch.full((width,), -30.0, device=device, dtype=dtype),  # silu underflow
        torch.randn(width, device=device, dtype=dtype) * 100.0,
    ]
    gate = torch.stack(rows)
    up = torch.stack(rows[::-1])

    ref = _ref_chain(gate, up, LIMIT, BLOCK_SIZE, fmt, dtype)
    got = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, LIMIT, BLOCK_SIZE, fmt)
    _assert_pair_equal(got, ref, f"edge fmt={fmt!r}")

    gate16 = (torch.randn(4, width, device=device, dtype=torch.float16) * 6.0).contiguous()
    up16 = (torch.randn(4, width, device=device, dtype=torch.float16) * 6.0).contiguous()
    ref16 = _ref_chain(gate16, up16, LIMIT, BLOCK_SIZE, fmt, torch.float16)
    got16 = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(
        gate16, up16, LIMIT, BLOCK_SIZE, fmt
    )
    _assert_pair_equal(got16, ref16, f"fp16 fmt={fmt!r}")


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_residual_add_accepts_prequantized_input():
    """residual_add(quant(x), input_scale=[s]) matches its raw-input form exactly."""
    device = "cuda"
    M, N, K = 8, 256, 512
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    w = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
    ws = torch.rand(N // 128, K // 128, device=device, dtype=torch.float32) * 0.05 + 0.01
    residual = torch.randn(M, N, device=device, dtype=torch.bfloat16)

    for fmt in ["", "ue8m0"]:
        ref = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
            x, w, None, [], [ws], [], [], input_scale_fmt=fmt, residual=residual
        )
        q, s = _safe_act_quant(x, BLOCK_SIZE, fmt)
        got = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
            q, w, None, [s], [ws], [], [], residual=residual
        )
        assert torch.equal(got, ref), f"prequantized residual-add mismatch (fmt={fmt!r})"


class _FP8Proj(nn.Module):
    def __init__(self, n, k, device, seed):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(seed)
        w = (torch.randn(n, k, generator=g, device=device, dtype=torch.bfloat16) * 0.1).to(
            torch.float8_e4m3fn
        )
        ws = (
            torch.rand(n // 128, k // 128, generator=g, device=device, dtype=torch.float32) * 0.05
            + 0.01
        )
        self.weight = nn.Parameter(w, requires_grad=False)
        self.register_buffer("weight_scale_inv", ws)


class _SharedExpertMLP(nn.Module):
    """DeepSeek-V4 shared-expert MLP in its post-quantization graph form."""

    def __init__(self, hidden, inter, device, fmt):
        super().__init__()
        self.w1 = _FP8Proj(inter, hidden, device, seed=1)
        self.w3 = _FP8Proj(inter, hidden, device, seed=3)
        self.w2 = _FP8Proj(hidden, inter, device, seed=2)
        self.fmt = fmt
        self.limit = LIMIT

    def forward(self, x, routed):
        lin = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
        gate = lin(
            x,
            self.w1.weight,
            None,
            [],
            [self.w1.weight_scale_inv],
            [],
            [],
            input_scale_fmt=self.fmt,
        )
        up = lin(
            x,
            self.w3.weight,
            None,
            [],
            [self.w3.weight_scale_inv],
            [],
            [],
            input_scale_fmt=self.fmt,
        )
        gate = torch.clamp(gate, max=self.limit)
        up = torch.clamp(up, min=-self.limit, max=self.limit)
        hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
            hidden,
            self.w2.weight,
            None,
            [],
            [self.w2.weight_scale_inv],
            [],
            [],
            input_scale_fmt=self.fmt,
            residual=routed,
        )


def _count_ops(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_transform_fuses_dsv4_shared_expert_chain(fmt):
    device = "cuda"
    hidden, inter, M = 256, 512, 8  # M>4 + K<4096 -> deterministic base matmul kernel
    model = _SharedExpertMLP(hidden, inter, device, fmt).eval()
    x = torch.randn(M, hidden, device=device, dtype=torch.bfloat16) * 5.0  # make clamps clip
    routed = torch.randn(M, hidden, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = model(x, routed).clone()

    gm = torch_export_to_gm(model, args=(x, routed))

    # Production order: merge the sibling gate/up projections first.
    mixed_cfg = FuseGemmsMixedChildrenConfig(stage=Stages.POST_LOAD_FUSION, quantized_only=True)
    gm, info_mixed = FuseGemmsMixedChildren(mixed_cfg)._apply(gm, None, None, SharedConfig())
    assert info_mixed.num_matches == 1
    gm.recompile()
    with torch.no_grad():
        merged_ref = gm(x, routed)
        merged_ref = (
            merged_ref[0] if isinstance(merged_ref, (tuple, list)) else merged_ref
        ).clone()

    gm, info = FuseFP8SwigluActQuant(TransformConfig(stage=Stages.POST_LOAD_FUSION))._apply(
        gm, None, None, SharedConfig()
    )
    gm.recompile()
    assert info.num_matches == 1

    assert _count_ops(gm, torch.ops.aten.silu) == 0
    assert not any(
        is_op(n, (torch.ops.aten.clamp, torch.ops.aten.clamp_max, torch.ops.aten.clamp_min))
        for n in gm.graph.nodes
    )
    assert _count_ops(gm, torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant) == 1
    residual_nodes = [
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add)
    ]
    assert len(residual_nodes) == 1
    assert len(residual_nodes[0].args[3]) == 1, "rewritten down linear must consume qscale"

    with torch.no_grad():
        out = gm(x, routed)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.equal(out, merged_ref), f"fusion changed the merged graph's output (fmt={fmt!r})"
    if fmt == "":
        # Default fmt: the gate/up merge itself is lossless -> must also match eager.
        assert torch.equal(out, ref)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_transform_matches_only_wellformed_chains():
    """A=unclamped, B=gate clamped both sides, C=biased down, D=swapped mul; only D fuses."""

    class _Chains(nn.Module):
        def __init__(self, hidden, inter, device):
            super().__init__()
            for i, name in enumerate(("a", "b", "c", "d")):
                setattr(self, f"w1_{name}", _FP8Proj(inter, hidden, device, seed=10 + i))
                setattr(self, f"w3_{name}", _FP8Proj(inter, hidden, device, seed=20 + i))
                setattr(self, f"w2_{name}", _FP8Proj(hidden, inter, device, seed=30 + i))
            self.down_bias = nn.Parameter(
                torch.randn(hidden, device=device, dtype=torch.bfloat16), requires_grad=False
            )

        def _gate_up(self, x, name):
            lin = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
            w1 = getattr(self, f"w1_{name}")
            w3 = getattr(self, f"w3_{name}")
            gate = lin(x, w1.weight, None, [], [w1.weight_scale_inv], [], [])
            up = lin(x, w3.weight, None, [], [w3.weight_scale_inv], [], [])
            return gate, up

        def _down(self, h, name, routed, bias=None):
            w2 = getattr(self, f"w2_{name}")
            return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
                h, w2.weight, bias, [], [w2.weight_scale_inv], [], [], residual=routed
            )

        def forward(self, x, routed):
            g, u = self._gate_up(x, "a")
            ha = (F.silu(g.float()) * u.float()).to(x.dtype)
            ya = self._down(ha, "a", routed)

            g, u = self._gate_up(x, "b")
            g = torch.clamp(g, min=-LIMIT, max=LIMIT)
            u = torch.clamp(u, min=-LIMIT, max=LIMIT)
            hb = (F.silu(g.float()) * u.float()).to(x.dtype)
            yb = self._down(hb, "b", routed)

            g, u = self._gate_up(x, "c")
            g = torch.clamp(g, max=LIMIT)
            u = torch.clamp(u, min=-LIMIT, max=LIMIT)
            hc = (F.silu(g.float()) * u.float()).to(x.dtype)
            yc = self._down(hc, "c", routed, bias=self.down_bias)

            g, u = self._gate_up(x, "d")
            g = torch.clamp(g, max=LIMIT)
            u = torch.clamp(u, min=-LIMIT, max=LIMIT)
            hd = (u.float() * F.silu(g.float())).to(x.dtype)
            yd = self._down(hd, "d", routed)
            return ya + yb + yc + yd

    device = "cuda"
    hidden, inter = 128, 128
    model = _Chains(hidden, inter, device).eval()
    x = torch.randn(4, hidden, device=device, dtype=torch.bfloat16)
    routed = torch.randn(4, hidden, device=device, dtype=torch.bfloat16)
    gm = torch_export_to_gm(model, args=(x, routed))

    gm, info = FuseFP8SwigluActQuant(TransformConfig(stage=Stages.POST_LOAD_FUSION))._apply(
        gm, None, None, SharedConfig()
    )
    gm.recompile()

    assert info.num_matches == 1
    assert _count_ops(gm, torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant) == 1
    assert _count_ops(gm, torch.ops.aten.silu) == 3  # chains A-C keep their eager epilogues


def test_matcher_helpers_reject_malformed_nodes():
    from tensorrt_llm._torch.auto_deploy.transform.library.fuse_quant import (
        _cast_target_dtype,
        _clamp_scalar_bounds,
        _finegrained_fp8_block_k,
    )

    g = torch.fx.Graph()
    x = g.placeholder("x")
    to_kw = g.call_function(torch.ops.aten._to_copy.default, (x,), {"dtype": torch.float32})
    to_bad = g.call_function(torch.ops.aten._to_copy.default, (x,), {"dtype": "not_a_dtype"})
    to_none = g.call_function(torch.ops.aten._to_copy.default, (x,))
    assert _cast_target_dtype(to_kw) == torch.float32
    assert _cast_target_dtype(to_bad) is None
    assert _cast_target_dtype(to_none) is None

    cmax = g.call_function(torch.ops.aten.clamp_max.default, (x, 3.0))
    cmin = g.call_function(torch.ops.aten.clamp_min.default, (x, -3.0))
    assert _clamp_scalar_bounds(cmax) == (None, 3.0)
    assert _clamp_scalar_bounds(cmin) == (-3.0, None)

    root = nn.Module()
    root.register_buffer("w", torch.zeros(4, 8))
    root.register_buffer("s_good", torch.ones(1, 2))
    root.register_buffer("s_1d", torch.ones(2))
    root.register_buffer("s_empty", torch.ones(2, 0))
    g2 = torch.fx.Graph()
    w_node = g2.get_attr("w")
    s_good = g2.get_attr("s_good")
    s_1d = g2.get_attr("s_1d")
    s_empty = g2.get_attr("s_empty")
    g2.output(None)
    gm = torch.fx.GraphModule(root, g2)

    assert _finegrained_fp8_block_k(gm, w_node, [s_good]) == 4  # ceil(8 / 2)
    assert _finegrained_fp8_block_k(gm, "not_a_node", [s_good]) is None
    assert _finegrained_fp8_block_k(gm, w_node, []) is None
    assert _finegrained_fp8_block_k(gm, w_node, [s_1d]) is None
    assert _finegrained_fp8_block_k(gm, w_node, [s_empty]) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
