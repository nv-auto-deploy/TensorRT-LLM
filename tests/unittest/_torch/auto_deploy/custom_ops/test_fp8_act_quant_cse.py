# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exactness for the FineGrained FP8 activation-quant CSE split (idea_0021).

``fuse_fp8_act_quant_cse`` hoists the per-block FP8 activation quant out of
``torch_fake_quant_finegrained_fp8_linear`` so sibling linears that consume the
same activation share ONE ``_act_quant_kernel`` launch instead of re-running it.
The rewrite is only legal if it is reference-exact, i.e. the split form

    qfp8, scale = torch_fp8_finegrained_act_quant(x, block_k, fmt)
    y = torch_fake_quant_finegrained_fp8_linear_prequant(qfp8, scale, w, b, [ws])

is bit-for-bit identical to the fused op for every linear. These tests guard
that invariant for both the no-bias and bias paths, default and ue8m0 scale
formats, and confirm the graph transform shares one quant node across siblings.
"""

import pytest
import torch
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_quant import FuseFP8ActQuantCSE
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


def _fp8_supported():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9  # Hopper+ for fp8


fp8 = torch.float8_e4m3fn
lin_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
act_op = torch.ops.auto_deploy.torch_fp8_finegrained_act_quant
prequant_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant


def _make_fp8_weight(N, K, device):
    """Random block-FP8 weight + per-128x128-block scale (matches checkpoint layout)."""
    w = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1).to(fp8)
    ws = torch.rand(N // 128, K // 128, device=device, dtype=torch.float32) * 0.05 + 0.01
    return w, ws


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("fmt", ["", "ue8m0"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_split_act_quant_is_byte_exact(fmt, with_bias):
    """split(act_quant -> prequant matmul) == fused finegrained fp8 linear, bit-exact."""
    device = "cuda"
    M, K, N = 4, 256, 256
    block_k = 128
    x = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    w, ws = _make_fp8_weight(N, K, device)
    bias = torch.randn(N, device=device, dtype=torch.bfloat16) if with_bias else None

    fused = lin_op(
        x,
        w,
        bias,
        input_scale=[],
        weight_scale=[ws],
        input_zp=[],
        weight_zp=[],
        input_scale_fmt=fmt,
    )

    qfp8, scale = act_op(x, block_k, fmt)
    split = prequant_op(qfp8, scale, w, bias, [ws])

    assert split.dtype == fused.dtype == torch.bfloat16
    assert torch.equal(split, fused), f"fmt={fmt} bias={with_bias}: split != fused"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_shared_quant_matches_independent_quant():
    """Two siblings sharing one quant == each quantizing independently (bit-exact)."""
    device = "cuda"
    M, K = 2, 256
    block_k = 128
    x = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
    wa, wsa = _make_fp8_weight(384, K, device)
    wb, wsb = _make_fp8_weight(128, K, device)

    ya_ref = lin_op(x, wa, None, input_scale=[], weight_scale=[wsa], input_zp=[], weight_zp=[])
    yb_ref = lin_op(x, wb, None, input_scale=[], weight_scale=[wsb], input_zp=[], weight_zp=[])

    qfp8, scale = act_op(x, block_k, "")  # ONE shared quant for both siblings
    ya = prequant_op(qfp8, scale, wa, None, [wsa])
    yb = prequant_op(qfp8, scale, wb, None, [wsb])

    assert torch.equal(ya, ya_ref)
    assert torch.equal(yb, yb_ref)


def _build_two_sibling_graph(device, with_third_singleton):
    """A tiny GraphModule: 2 (or 3) finegrained fp8 linears, two sharing input x."""
    K, Na, Nb = 256, 384, 128
    Nc = Na + Nb  # singleton output matches cat([ya, yb]) so it can be added
    wa, wsa = _make_fp8_weight(Na, K, device)
    wb, wsb = _make_fp8_weight(Nb, K, device)
    wc, wsc = _make_fp8_weight(Nc, K, device)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("wa", wa)
            self.register_buffer("wsa", wsa)
            self.register_buffer("wb", wb)
            self.register_buffer("wsb", wsb)
            self.register_buffer("wc", wc)
            self.register_buffer("wsc", wsc)
            self.register_buffer("z", torch.zeros(1, K, device=device, dtype=torch.bfloat16))

        def forward(self, x):
            # wa, wb both consume x (siblings). wc consumes a different tensor (singleton).
            ya = lin_op(
                x, self.wa, None, input_scale=[], weight_scale=[self.wsa], input_zp=[], weight_zp=[]
            )
            yb = lin_op(
                x, self.wb, None, input_scale=[], weight_scale=[self.wsb], input_zp=[], weight_zp=[]
            )
            out = torch.cat([ya, yb], dim=-1)
            if with_third_singleton:
                x2 = x + self.z
                yc = lin_op(
                    x2,
                    self.wc,
                    None,
                    input_scale=[],
                    weight_scale=[self.wsc],
                    input_zp=[],
                    weight_zp=[],
                )
                out = out + yc
            return out

    from torch.fx import symbolic_trace

    gm = symbolic_trace(M())
    return gm


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("with_third_singleton", [False, True])
def test_cse_transform_shares_and_preserves_output(with_third_singleton):
    """The transform inserts ONE shared act-quant for the sibling pair and is exact."""
    device = "cuda"
    x = (torch.randn(2, 256, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    gm: GraphModule = _build_two_sibling_graph(device, with_third_singleton)
    ref = gm(x).clone()

    n_lin_before = sum(1 for n in gm.graph.nodes if is_op(n, lin_op))
    expected_lin = 3 if with_third_singleton else 2
    assert n_lin_before == expected_lin

    transform = FuseFP8ActQuantCSE(TransformConfig(stage=Stages.POST_LOAD_FUSION))
    new_gm, info = transform._apply(gm, None, None, SharedConfig())
    new_gm.recompile()

    # Exactly one sibling group found; siblings rewritten to prequant; singleton untouched.
    assert info.num_matches == 1
    n_act = sum(1 for n in new_gm.graph.nodes if is_op(n, act_op))
    n_prequant = sum(1 for n in new_gm.graph.nodes if is_op(n, prequant_op))
    n_lin_after = sum(1 for n in new_gm.graph.nodes if is_op(n, lin_op))
    assert n_act == 1, f"expected 1 shared act-quant, got {n_act}"
    assert n_prequant == 2, f"expected 2 prequant matmuls, got {n_prequant}"
    assert n_lin_after == (1 if with_third_singleton else 0)

    out = new_gm(x)
    assert torch.equal(out, ref), "transform changed numerics"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
