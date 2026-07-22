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
"""Sibling-GEMM fusion transforms: mixed-children FP8, replicated bf16, coexistence."""

import pytest
import torch
from torch.fx import GraphModule, symbolic_trace

from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fusion import (
    FuseGemmsMixedChildren,
    FuseGemmsMixedChildrenConfig,
    FuseReplicatedBf16Linear,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

fp8 = torch.float8_e4m3fn
lin_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
bf16_lin_op = torch.ops.auto_deploy.torch_linear_simple


def _fp8_supported():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9  # Hopper+ for fp8


def _make_fp8_weight(N, K, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=device, dtype=torch.bfloat16) * 0.1).to(fp8)
    ws = (  # per-128x128-block weight scale, strictly positive
        torch.rand(N // 128, K // 128, generator=g, device=device, dtype=torch.float32) * 0.05
        + 0.01
    )
    return w, ws


def _populate_getattr_val_meta(gm):
    """Mimic shape-prop: tag get_attr nodes with meta['val'] (symbolic_trace doesn't)."""
    for n in gm.graph.nodes:
        if n.op != "get_attr":
            continue
        try:
            n.meta["val"] = gm.get_parameter(n.target)
        except (AttributeError, KeyError):
            try:
                n.meta["val"] = gm.get_buffer(n.target)
            except (AttributeError, KeyError):
                pass


def _count_lin(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def _count_narrow(gm):
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and "narrow" in str(n.target))


# ---------------------------------------------------------------------------
# Quantized mixed-children FP8 fusion
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_different_shaped_concat_bit_exact_base_kernel():
    """Weight/scale concat over different-N siblings is byte-exact on the base kernel."""
    device = "cuda"
    M = 8  # M > 4 avoids the split-K decode path -> deterministic kernel
    K, Na, Nb = 256, 512, 1024
    x = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    wa, wsa = _make_fp8_weight(Na, K, device, seed=1)
    wb, wsb = _make_fp8_weight(Nb, K, device, seed=2)

    def _lin(w, ws):
        return lin_op(x, w, None, input_scale=[], weight_scale=[ws], input_zp=[], weight_zp=[])

    ya, yb = _lin(wa, wsa), _lin(wb, wsb)

    cat_w = torch.cat([wa, wb], dim=0)
    cat_s = torch.cat([wsa, wsb], dim=0)
    merged = lin_op(x, cat_w, None, input_scale=[], weight_scale=[cat_s], input_zp=[], weight_zp=[])

    assert merged.shape == (M, Na + Nb)
    assert torch.equal(merged[:, :Na], ya)
    assert torch.equal(merged[:, Na:], yb)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_different_shaped_concat_matches_splitk_decode():
    """M=1, K>=4096 (split-K atomic path): concat matches up to reduction rounding."""
    device = "cuda"
    K, Na, Nb = 4096, 512, 1024
    x = (torch.randn(1, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    wa, wsa = _make_fp8_weight(Na, K, device, seed=11)
    wb, wsb = _make_fp8_weight(Nb, K, device, seed=12)

    def _lin(w, ws):
        return lin_op(x, w, None, input_scale=[], weight_scale=[ws], input_zp=[], weight_zp=[])

    ya, yb = _lin(wa, wsa), _lin(wb, wsb)
    merged = lin_op(
        x,
        torch.cat([wa, wb], dim=0),
        None,
        input_scale=[],
        weight_scale=[torch.cat([wsa, wsb], dim=0)],
        input_zp=[],
        weight_zp=[],
    )

    torch.testing.assert_close(merged[:, :Na], ya, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(merged[:, Na:], yb, rtol=1e-2, atol=1e-2)


class _FP8Proj(torch.nn.Module):
    def __init__(self, N, K, device, seed):
        super().__init__()
        w, ws = _make_fp8_weight(N, K, device, seed)
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.register_buffer("weight_scale_inv", ws)

    def forward(self, x):
        # Positional args: the fuser rebuilds the scale/zp lists positionally.
        return lin_op(x, self.weight, None, [], [self.weight_scale_inv], [], [])


class _FP8SiblingModule(torch.nn.Module):
    def __init__(self, K, device):
        super().__init__()
        self.proj_a = _FP8Proj(512, K, device, seed=1)
        self.proj_b = _FP8Proj(1024, K, device, seed=2)
        self.proj_solo = _FP8Proj(256, K, device, seed=3)
        self.register_buffer("z", torch.zeros(1, K, device=device, dtype=torch.bfloat16))

    def forward(self, x):
        ya = self.proj_a(x)
        yb = self.proj_b(x)
        y_solo = self.proj_solo(x + self.z)  # different input node -> untouched
        return ya, yb, y_solo


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_transform_fuses_different_shaped_fp8_siblings():
    device = "cuda"
    K = 256
    x = (torch.randn(2, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    gm: GraphModule = symbolic_trace(_FP8SiblingModule(K, device))
    _populate_getattr_val_meta(gm)
    ref_a, ref_b, ref_solo = (t.clone() for t in gm(x))

    assert _count_lin(gm, lin_op) == 3

    cfg = FuseGemmsMixedChildrenConfig(stage=Stages.POST_LOAD_FUSION, quantized_only=True)
    new_gm, info = FuseGemmsMixedChildren(cfg)._apply(gm, None, None, SharedConfig())
    new_gm.recompile()

    assert info.num_matches == 1
    assert _count_lin(new_gm, lin_op) == 2  # 1 fused + 1 solo
    assert _count_narrow(new_gm) == 2

    out_a, out_b, out_solo = new_gm(x)
    torch.testing.assert_close(out_a, ref_a, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_b, ref_b, rtol=1e-2, atol=1e-2)
    assert torch.equal(out_solo, ref_solo)


# ---------------------------------------------------------------------------
# Replicated bf16 linear fusion
# ---------------------------------------------------------------------------
def _bf16_lin(x, w):
    return bf16_lin_op(x, w, None, tp_mode="none", layer_type="mla")


class _Bf16SiblingModule(torch.nn.Module):
    """3 eligible siblings + singleton + ineligible decoys (biased/sharded/fp16)."""

    def __init__(self, K, device):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(0)

        def _w(n, scale, dtype=torch.bfloat16):
            return torch.nn.Parameter(
                (torch.randn(n, K, generator=g, device=device, dtype=dtype) * scale),
                requires_grad=False,
            )

        # Very different weight magnitudes: a wrong slice->weight mapping fails loudly.
        self.w_a = _w(64, 0.02)
        self.w_b = _w(256, 0.20)
        self.w_c = _w(256, 2.00)
        self.w_other = _w(128, 0.05)
        self.w_biased = _w(32, 0.05)
        self.bias = torch.nn.Parameter(
            torch.randn(32, generator=g, device=device, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.w_sharded = _w(32, 0.05)
        self.w_fp16 = _w(32, 0.05, dtype=torch.float16)
        self.register_buffer("z", torch.zeros(1, K, device=device, dtype=torch.bfloat16))

    def forward(self, x):
        ya = _bf16_lin(x, self.w_a)
        yb = _bf16_lin(x, self.w_b)
        yc = _bf16_lin(x, self.w_c)
        y_biased = bf16_lin_op(x, self.w_biased, self.bias, tp_mode="none", layer_type="mla")
        y_sharded = bf16_lin_op(x, self.w_sharded, None, tp_mode="colwise", layer_type="mla")
        y_fp16 = bf16_lin_op(
            x.to(torch.float16), self.w_fp16, None, tp_mode="none", layer_type="mla"
        )
        y_other = _bf16_lin(x + self.z, self.w_other)
        return torch.cat([ya, yb, yc], dim=-1), y_other, y_biased, y_sharded, y_fp16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_replicated_bf16_transform_fuses_siblings_and_preserves_output():
    device = "cuda"
    K = 4096
    x = torch.randn(2, K, device=device, dtype=torch.bfloat16)

    gm: GraphModule = symbolic_trace(_Bf16SiblingModule(K, device))
    _populate_getattr_val_meta(gm)
    ref = [t.clone() for t in gm(x)]

    assert _count_lin(gm, bf16_lin_op) == 7  # 3 siblings + 1 singleton + 3 decoys

    transform = FuseReplicatedBf16Linear(TransformConfig(stage=Stages.POST_LOAD_FUSION))
    new_gm, info = transform._apply(gm, None, None, SharedConfig())
    new_gm.recompile()

    assert info.num_matches == 1
    assert _count_lin(new_gm, bf16_lin_op) == 5  # 1 fused + 1 singleton + 3 decoys
    assert _count_narrow(new_gm) == 3

    out = new_gm(x)
    torch.testing.assert_close(out[0], ref[0], rtol=1e-2, atol=1e-2)
    for got, exp in zip(out[1:], ref[1:]):  # singleton + decoys must be untouched
        assert torch.equal(got, exp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_concat_rows_matches_per_linear():
    """Op-level identity: aten.linear over row-concatenated weights == per-weight linears."""
    device = "cuda"
    K = 4096
    x = torch.randn(3, K, device=device, dtype=torch.bfloat16)
    sizes = [64, 256, 256]
    ws = [
        torch.randn(n, K, device=device, dtype=torch.bfloat16) * s
        for n, s in zip(sizes, [0.02, 0.2, 2.0])
    ]

    per = [torch.ops.aten.linear(x, w, None) for w in ws]
    fused = torch.ops.aten.linear(x, torch.cat(ws, dim=0), None)

    off = 0
    for w, y in zip(ws, per):
        n = w.shape[0]
        torch.testing.assert_close(fused.narrow(-1, off, n), y, rtol=1e-2, atol=1e-2)
        off += n


# ---------------------------------------------------------------------------
# Coexistence: both transforms on one graph must not collide on fused params
# ---------------------------------------------------------------------------
class _MixedModule(torch.nn.Module):
    def __init__(self, K, device):
        super().__init__()
        self.proj_a = _FP8Proj(512, K, device, seed=5)
        self.proj_b = _FP8Proj(1024, K, device, seed=6)
        g = torch.Generator(device=device).manual_seed(7)
        self.bf_a = torch.nn.Parameter(
            torch.randn(64, K, generator=g, device=device, dtype=torch.bfloat16) * 0.02,
            requires_grad=False,
        )
        self.bf_b = torch.nn.Parameter(
            torch.randn(256, K, generator=g, device=device, dtype=torch.bfloat16) * 0.2,
            requires_grad=False,
        )

    def forward(self, x):
        ya = self.proj_a(x)
        yb = self.proj_b(x)
        za = _bf16_lin(x, self.bf_a)
        zb = _bf16_lin(x, self.bf_b)
        return ya, yb, za, zb


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_no_param_collision_with_replicated_bf16():
    """Both fusions on one graph register distinct fused weights (suffix on collision)."""
    device = "cuda"
    K = 256
    x = (torch.randn(2, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    gm: GraphModule = symbolic_trace(_MixedModule(K, device))
    _populate_getattr_val_meta(gm)
    ref = tuple(t.clone() for t in gm(x))

    first_fp8_node = next(n for n in gm.graph.nodes if is_op(n, lin_op))
    planted = f"fused_weight_{first_fp8_node.name}"
    setattr(gm, planted, torch.zeros(1, device=device))

    # Production order: mixed_children (FP8) first, then the bf16 fusion.
    cfg = FuseGemmsMixedChildrenConfig(stage=Stages.POST_LOAD_FUSION, quantized_only=True)
    gm, info_fp8 = FuseGemmsMixedChildren(cfg)._apply(gm, None, None, SharedConfig())
    gm, info_bf16 = FuseReplicatedBf16Linear(TransformConfig(stage=Stages.POST_LOAD_FUSION))._apply(
        gm, None, None, SharedConfig()
    )
    gm.recompile()

    assert info_fp8.num_matches == 1
    assert info_bf16.num_matches == 1

    fused_names = [name for name, _ in gm.named_parameters() if name.startswith("fused_weight_")]
    assert len(fused_names) == 2
    assert len(set(fused_names)) == 2, f"fused param name collision: {fused_names}"
    assert f"{planted}_1" in fused_names, f"expected suffixed name, got {fused_names}"
    assert torch.equal(getattr(gm, planted), torch.zeros(1, device=device))

    out = gm(x)
    for got, exp in zip(out, ref):
        torch.testing.assert_close(got, exp, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
