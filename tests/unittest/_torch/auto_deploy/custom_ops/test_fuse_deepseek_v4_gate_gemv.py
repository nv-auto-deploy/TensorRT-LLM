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

"""Graph-rewrite tests for ``fuse_deepseek_v4_gate_gemv``.

The transform must retarget ONLY the DSV4 router-gate ``torch_linear_simple``
(fp32 x, fp32 2-D weight, no bias, sole consumer = ``deepseek_v4_routing`` /
``deepseek_v4_routing_localized``, optionally through an identity fp32 cast)
to ``auto_deploy::deepseek_v4_gate_gemv`` — in place, with args/kwargs, node
order, and every other node (incl. the MoE-side routing consumer) untouched.
"""

import pytest
import torch
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import deepseek_v4_routing  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_deepseek_v4_gate_gemv import (
    FuseDeepseekV4GateGemv,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

lin_op = torch.ops.auto_deploy.torch_linear_simple
gemv_op = torch.ops.auto_deploy.deepseek_v4_gate_gemv


class _GateModule(torch.nn.Module):
    """DSV4 gate head as exported: linear -> identity fp32 cast -> routing,
    plus a decoy fp32 linear NOT feeding a routing op (must stay untouched)."""

    def __init__(self, E, H, device, with_cast=True):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(0)
        self.weight = torch.nn.Parameter(
            torch.randn(E, H, generator=g, device=device, dtype=torch.float32) * 0.05,
            requires_grad=False,
        )
        self.bias = torch.nn.Parameter(
            torch.randn(E, generator=g, device=device, dtype=torch.float32) * 0.5,
            requires_grad=False,
        )
        self.w_decoy = torch.nn.Parameter(
            torch.randn(64, H, generator=g, device=device, dtype=torch.float32) * 0.05,
            requires_grad=False,
        )
        self.with_cast = with_cast

    def forward(self, x):
        logits = lin_op(x, self.weight, None)
        if self.with_cast:
            logits = torch.ops.aten.to.dtype(logits, torch.float32)
        sel, w = torch.ops.auto_deploy.deepseek_v4_routing(logits, self.bias, 6, 1.5, True)
        decoy = lin_op(x, self.w_decoy, None)
        return sel, w, decoy


def _apply(gm: GraphModule, x: torch.Tensor):
    from torch._subclasses.fake_tensor import FakeTensorMode

    # Real params in the traced module; the production pipeline shape-props with
    # fakes, so mirror that here to populate node.meta["val"] for the matcher.
    FakeTensorProp(gm, mode=FakeTensorMode(allow_non_fake_inputs=True)).propagate(x)
    transform = FuseDeepseekV4GateGemv(TransformConfig(stage=Stages.POST_LOAD_FUSION))
    new_gm, info = transform._apply(gm, None, None, SharedConfig())
    new_gm.recompile()
    return new_gm, info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_cast", [True, False])
def test_transform_swaps_gate_producer_in_place(with_cast):
    device = "cuda"
    E, H = 256, 4096
    x = torch.randn(1, H, device=device, dtype=torch.float32)

    gm = symbolic_trace(_GateModule(E, H, device, with_cast=with_cast))
    nodes_before = [(n.name, n.op) for n in gm.graph.nodes]
    routing_args_before = [
        (n.args[1:], n.kwargs)
        for n in gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.deepseek_v4_routing)
    ]
    ref_sel, ref_w, ref_decoy = gm(x)
    ref_sel, ref_w, ref_decoy = ref_sel.clone(), ref_w.clone(), ref_decoy.clone()

    new_gm, info = _apply(gm, x)
    assert info.num_matches == 1, f"expected 1 gate swap, got {info.num_matches}"

    # Pure in-place retarget: node names/ops/order identical; only the gate
    # linear's target changed. The routing consumer's args are untouched.
    assert [(n.name, n.op) for n in new_gm.graph.nodes] == nodes_before
    gate_nodes = [n for n in new_gm.graph.nodes if is_op(n, gemv_op)]
    assert len(gate_nodes) == 1
    assert gate_nodes[0].args[1].target == "weight"
    decoy_nodes = [n for n in new_gm.graph.nodes if is_op(n, lin_op)]
    assert len(decoy_nodes) == 1 and decoy_nodes[0].args[1].target == "w_decoy", (
        "decoy linear (no routing consumer) must not be swapped"
    )
    routing_args_after = [
        (n.args[1:], n.kwargs)
        for n in new_gm.graph.nodes
        if is_op(n, torch.ops.auto_deploy.deepseek_v4_routing)
    ]
    assert routing_args_after == routing_args_before, "routing consumer args changed"

    sel, w, decoy = new_gm(x)
    assert torch.equal(decoy, ref_decoy), "decoy output changed"
    # Random single token: selection parity expected (~1e-6 fp32 sum-order band).
    assert torch.equal(sel, ref_sel), f"selection flip after swap: {ref_sel} -> {sel}"
    torch.testing.assert_close(w, ref_w, rtol=1e-4, atol=1e-5)

    # Prefill-shaped input takes the op's cuBLAS fallback: bit-identical graph output.
    x_prefill = torch.randn(8, H, device=device, dtype=torch.float32)
    ref_out = gm(x_prefill)
    new_out = new_gm(x_prefill)
    for a, b in zip(new_out, ref_out):
        assert torch.equal(a, b), "prefill path must be bit-identical"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_transform_skips_non_matching_patterns():
    """bf16 gate weight or an extra logits consumer must NOT match."""
    device = "cuda"
    E, H = 64, 512

    class _NonMatch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w_bf16 = torch.nn.Parameter(
                torch.randn(E, H, device=device, dtype=torch.bfloat16), requires_grad=False
            )
            self.w_fp32 = torch.nn.Parameter(
                torch.randn(E, H, device=device, dtype=torch.float32), requires_grad=False
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(E, device=device, dtype=torch.float32), requires_grad=False
            )

        def forward(self, x):
            # bf16 weight (wrong dtype for the gate contract).
            y1 = lin_op(x.to(torch.bfloat16), self.w_bf16, None)
            # fp32 gate whose logits have a SECOND consumer besides routing.
            logits = lin_op(x, self.w_fp32, None)
            sel, w = torch.ops.auto_deploy.deepseek_v4_routing(logits, self.bias, 6, 1.5, True)
            return sel, w, y1, logits * 2.0

    x = torch.randn(1, H, device=device, dtype=torch.float32)
    gm = symbolic_trace(_NonMatch())
    new_gm, info = _apply(gm, x)
    assert info.num_matches == 0
    assert not any(is_op(n, gemv_op) for n in new_gm.graph.nodes)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
