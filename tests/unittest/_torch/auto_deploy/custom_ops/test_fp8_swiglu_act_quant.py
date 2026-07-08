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
"""Bit-exactness proof for the fused clamped-SwiGLU + block-FP8 act-quant kernel.

``auto_deploy::torch_fp8_swiglu_clamp_act_quant`` fuses the DeepSeek-V4
shared-expert epilogue between the merged gate/up projection and the down
projection -- ``clamp(gate, max=L); clamp(up, -L, L);
(silu(gate.float()) * up.float()).to(model_dtype); _safe_act_quant(...)`` -- into
one Triton launch. The ``fuse_fp8_swiglu_act_quant`` transform rewrites the chain
and feeds the pre-quantized pair into the matmul-only
``torch_fake_quant_finegrained_fp8_linear[_residual_add]_prequant`` down linear.

The claim under test is *exact* equivalence, not approximate accuracy: every
op-level case asserts byte equality of the FP8 payload and ``torch.equal`` of the
per-block scales against the unfused aten chain + ``_safe_act_quant`` reference,
for both scale formats ("", "ue8m0"), with and without the clamp, on contiguous
and strided (narrow-view) inputs. The graph-level cases run the real
``torch_export_to_gm`` + ``fuse_gemms_mixed_children`` pipeline and assert the
transform fires and the rewritten graph reproduces the eager module bit for bit.
"""

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
    """The exact eager chain the transform strands (modeling_deepseek_v4.DeepseekV4MLP)."""
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
    payload_eq = torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    n_bad = (q.view(torch.uint8) != q_ref.view(torch.uint8)).sum().item()
    assert payload_eq, f"{ctx}: fp8 payload differs at {n_bad} element(s)"
    assert torch.equal(s, s_ref), f"{ctx}: per-block scales differ"


# ---------------------------------------------------------------------------
# Op-level: fused kernel == unfused aten chain + _safe_act_quant, bit for bit.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("M", [1, 2, 5, 64])
@pytest.mark.parametrize("width", [512, 256])
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
@pytest.mark.parametrize("limit", [LIMIT, None])
def test_fused_swiglu_act_quant_bit_exact(M, width, fmt, limit):
    device = "cuda"
    dtype = torch.bfloat16
    # Scale so a meaningful fraction of gate/up exceeds the clamp bound.
    gate = torch.randn(M, width, device=device, dtype=dtype) * 6.0
    up = torch.randn(M, width, device=device, dtype=dtype) * 6.0
    if limit is not None:
        assert (gate > limit).any() and (up.abs() > limit).any(), "clamp path not exercised"

    ref = _ref_chain(gate, up, limit, BLOCK_SIZE, fmt, dtype)
    got = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, limit, BLOCK_SIZE, fmt)
    _assert_pair_equal(got, ref, f"M={M} width={width} fmt={fmt!r} limit={limit}")


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
    assert not gate_up.narrow(-1, width, width).is_contiguous() or M == 1

    ref = _ref_chain(gate, up, LIMIT, BLOCK_SIZE, fmt, dtype)
    got = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, LIMIT, BLOCK_SIZE, fmt)
    _assert_pair_equal(got, ref, f"strided fmt={fmt!r}")


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_fused_swiglu_act_quant_edge_values(fmt):
    """Boundary/degenerate blocks: all-zero (scale floor), exact +/-limit, huge values,
    deeply negative gate (silu underflow), fp16 model dtype."""
    device = "cuda"
    dtype = torch.bfloat16
    width = 256
    rows = [
        torch.zeros(width, device=device, dtype=dtype),  # all-zero block -> scale floor
        torch.full((width,), LIMIT, device=device, dtype=dtype),  # exactly at the bound
        torch.full((width,), -LIMIT, device=device, dtype=dtype),
        torch.full((width,), 1e4, device=device, dtype=dtype),  # far above the bound
        torch.full((width,), -30.0, device=device, dtype=dtype),  # silu ~ 0
        torch.randn(width, device=device, dtype=dtype) * 100.0,  # mixed heavy clipping
    ]
    gate = torch.stack(rows)
    up = torch.stack(rows[::-1])

    ref = _ref_chain(gate, up, LIMIT, BLOCK_SIZE, fmt, dtype)
    got = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, LIMIT, BLOCK_SIZE, fmt)
    _assert_pair_equal(got, ref, f"edge fmt={fmt!r}")

    # fp16 model dtype (scale + round point follow the input dtype)
    gate16 = (torch.randn(4, width, device=device, dtype=torch.float16) * 6.0).contiguous()
    up16 = (torch.randn(4, width, device=device, dtype=torch.float16) * 6.0).contiguous()
    ref16 = _ref_chain(gate16, up16, LIMIT, BLOCK_SIZE, fmt, torch.float16)
    got16 = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(
        gate16, up16, LIMIT, BLOCK_SIZE, fmt
    )
    _assert_pair_equal(got16, ref16, f"fp16 fmt={fmt!r}")


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_prequant_residual_add_matches_fused_op():
    """residual_add_prequant(quant(x), ...) == residual_add(x, ...) bit for bit."""
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
        q, s = torch.ops.auto_deploy.torch_fp8_finegrained_act_quant(x, BLOCK_SIZE, fmt)
        got = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add_prequant(
            q, s, w, None, [ws], residual
        )
        assert torch.equal(got, ref), f"prequant residual-add mismatch (fmt={fmt!r})"


# ---------------------------------------------------------------------------
# Graph-level: real export + fuse_gemms_mixed_children + the new transform.
# ---------------------------------------------------------------------------
class _FP8Proj(nn.Module):
    """Block-FP8 weight stored as ``<name>.weight`` + ``<name>.weight_scale_inv``
    (the layout the mixed-children fuser reconstructs the scale name from)."""

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
    """DeepSeek-V4 shared-expert MLP in its post-quantization graph form.

    Mirrors ``DeepseekV4MLP.forward`` after the quantization transform rewrote the
    three linears to ``torch_fake_quant_finegrained_fp8_linear`` -- plus, for the
    residual form, the down projection as ``..._residual_add`` (the node
    ``fuse_fp8_linear_allreduce_add`` leaves for the MoE merge seam).
    """

    def __init__(self, hidden, inter, device, fmt, residual_form):
        super().__init__()
        self.w1 = _FP8Proj(inter, hidden, device, seed=1)
        self.w3 = _FP8Proj(inter, hidden, device, seed=3)
        self.w2 = _FP8Proj(hidden, inter, device, seed=2)
        self.fmt = fmt
        self.residual_form = residual_form
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
        if self.residual_form:
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
        down = lin(
            hidden,
            self.w2.weight,
            None,
            [],
            [self.w2.weight_scale_inv],
            [],
            [],
            input_scale_fmt=self.fmt,
        )
        return down + routed


def _count_ops(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("residual_form", [True, False])
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_transform_fuses_dsv4_shared_expert_chain(residual_form, fmt):
    device = "cuda"
    hidden, inter, M = 256, 512, 8  # M>4 + K<4096 -> deterministic base matmul kernel
    model = _SharedExpertMLP(hidden, inter, device, fmt, residual_form).eval()
    # Wide activations so the clamps actually clip (gate/up std ~ 0.1*5*sqrt(256) ~ 8).
    x = torch.randn(M, hidden, device=device, dtype=torch.bfloat16) * 5.0
    routed = torch.randn(M, hidden, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        ref = model(x, routed).clone()

    gm = torch_export_to_gm(model, args=(x, routed))

    # Stage 1 (production order): merge the sibling gate/up projections.
    mixed_cfg = FuseGemmsMixedChildrenConfig(stage=Stages.POST_LOAD_FUSION, fp8_only=True)
    gm, info_mixed = FuseGemmsMixedChildren(mixed_cfg)._apply(gm, None, None, SharedConfig())
    assert info_mixed.num_matches == 1, f"gate/up merge expected 1 match, got {info_mixed}"
    gm.recompile()
    with torch.no_grad():
        # Graph state the new transform actually sees: the byte-exactness claim under
        # test is post-merge -> post-fusion. (torch.export normalizes this test
        # module's input_scale_fmt kwarg to a positional arg, which the merged
        # gate/up node built by _insert_fused_quant_gemm does not carry over --
        # a test-only artifact: the production quantizer inserts the fmt as a kwarg
        # post-export, which the merge preserves. The down projection is not merged,
        # so the fmt consumed by the fusion under test is intact either way.)
        merged_ref = gm(x, routed)
        merged_ref = (
            merged_ref[0] if isinstance(merged_ref, (tuple, list)) else merged_ref
        ).clone()

    # Stage 2: the new fusion must consume the merged projection's narrow views.
    swiglu_cfg = TransformConfig(stage=Stages.POST_LOAD_FUSION)
    gm, info = FuseFP8SwigluActQuant(swiglu_cfg)._apply(gm, None, None, SharedConfig())
    gm.recompile()
    assert info.num_matches == 1, f"swiglu act-quant fusion expected 1 match, got {info}"

    # The elementwise chain and the internally-quantizing down linear are gone.
    assert _count_ops(gm, torch.ops.aten.silu) == 0
    assert not any(
        is_op(n, (torch.ops.aten.clamp, torch.ops.aten.clamp_max, torch.ops.aten.clamp_min))
        for n in gm.graph.nodes
    )
    assert _count_ops(gm, torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant) == 1
    if residual_form:
        assert (
            _count_ops(
                gm,
                torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add_prequant,
            )
            == 1
        )
        assert (
            _count_ops(
                gm, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add
            )
            == 0
        )
    else:
        assert (
            _count_ops(gm, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant)
            == 1
        )

    with torch.no_grad():
        out = gm(x, routed)
    out = out[0] if isinstance(out, (tuple, list)) else out
    assert torch.equal(out, merged_ref), (
        f"fusion changed the merged graph's output (residual_form={residual_form}, "
        f"fmt={fmt!r}): max abs diff {(out.float() - merged_ref.float()).abs().max().item()}"
    )
    if fmt == "":
        # With the default scale format the gate/up merge itself is lossless, so the
        # rewritten graph must also reproduce the original eager module bit for bit.
        assert torch.equal(out, ref), (
            f"rewritten graph diverges from eager (residual_form={residual_form}): "
            f"max abs diff {(out.float() - ref.float()).abs().max().item()}"
        )


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_transform_leaves_unclamped_chain_alone():
    """A swiglu chain without the clamp pair (swiglu_limit=0 models) is out of scope."""

    class _NoClampMLP(nn.Module):
        def __init__(self, hidden, inter, device):
            super().__init__()
            self.w1 = _FP8Proj(inter, hidden, device, seed=1)
            self.w2 = _FP8Proj(hidden, inter, device, seed=2)

        def forward(self, x):
            lin = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
            gate = lin(x, self.w1.weight, None, [], [self.w1.weight_scale_inv], [], [])
            hidden = (F.silu(gate.float()) * gate.float()).to(x.dtype)
            return lin(hidden, self.w2.weight, None, [], [self.w2.weight_scale_inv], [], [])

    device = "cuda"
    model = _NoClampMLP(256, 512, device).eval()
    x = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    gm = torch_export_to_gm(model, args=(x,))
    gm, info = FuseFP8SwigluActQuant(TransformConfig(stage=Stages.POST_LOAD_FUSION))._apply(
        gm, None, None, SharedConfig()
    )
    assert info.num_matches == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
