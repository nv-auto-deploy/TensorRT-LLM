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
"""Equivalence proof for the replicated bf16 multi-output linear fusion (idea_0029).

``fuse_replicated_bf16_linear`` merges sibling replicated (``tp_mode="none"``),
bias-free bf16 ``torch_linear_simple`` projections that read the *same* activation
(e.g. DeepSeek-V4 sparse-layer ``attn/indexer`` compressor ``wkv``+``wgate`` and the
indexer ``weights_proj``) into one concatenated ``[sum_N, K]`` projection, then splits
the result back with ``torch.narrow`` views. Concatenating output rows leaves each
row's ``K``-reduction unchanged, so the split output equals the per-linear outputs
(cuBLASLt may pick a different tiling for the wider N, so the M=1 GEMV path matches to
rounding rather than bit-for-bit). These tests guard both the numeric identity and the
graph rewrite (right sibling group fused, singleton untouched, launch count dropped).
"""

import pytest
import torch
from torch.fx import GraphModule, symbolic_trace

from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fusion import FuseReplicatedBf16Linear
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


def _populate_getattr_val_meta(gm):
    """Tag get_attr nodes with meta['val'] (their param/buffer tensor).

    The production pipeline runs shape propagation before post_load_fusion, so the
    weight-node classifier (``is_weight_node`` -> ``has_shape``) sees a shaped
    ``meta['val']`` on every parameter get_attr. ``symbolic_trace`` does not populate
    that, so we replicate it here.
    """
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


lin_op = torch.ops.auto_deploy.torch_linear_simple


def _lin(x, w):
    # Replicated, bias-free bf16 projection -- exactly how DeepSeek-V4's compressor /
    # indexer emit their projections (tp_mode="none", layer_type="mla").
    return lin_op(x, w, None, tp_mode="none", layer_type="mla")


class _SiblingModule(torch.nn.Module):
    """hidden -> {N=64, N=256, N=256} siblings + a singleton on a different input.

    The three siblings mirror indexer.weights_proj (64) + compressor wkv/wgate (256) that
    all consume the layer's hidden_states. Per-projection weight magnitudes are made very
    different so a wrong narrow-slice -> weight mapping would fail loudly, not by rounding.
    """

    def __init__(self, K, device):
        super().__init__()
        g = torch.Generator(device=device).manual_seed(0)

        def _w(n, scale):
            return torch.nn.Parameter(
                (torch.randn(n, K, generator=g, device=device, dtype=torch.bfloat16) * scale),
                requires_grad=False,
            )

        self.w_a = _w(64, 0.02)  # weights_proj-like
        self.w_b = _w(256, 0.20)  # wkv-like (10x magnitude of w_a)
        self.w_c = _w(256, 2.00)  # wgate-like (100x magnitude of w_a)
        self.w_other = _w(128, 0.05)  # singleton on a different activation
        self.register_buffer("z", torch.zeros(1, K, device=device, dtype=torch.bfloat16))

    def forward(self, x):
        ya = _lin(x, self.w_a)
        yb = _lin(x, self.w_b)
        yc = _lin(x, self.w_c)
        # A singleton bf16 linear on a *different* input node must be left alone.
        y_other = _lin(x + self.z, self.w_other)
        return torch.cat([ya, yb, yc], dim=-1), y_other


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_transform_fuses_siblings_and_preserves_output():
    device = "cuda"
    K = 4096  # DeepSeek-V4 hidden_size
    x = torch.randn(2, K, device=device, dtype=torch.bfloat16)

    gm: GraphModule = symbolic_trace(_SiblingModule(K, device))
    # The production pipeline shape-props before post_load_fusion; the weight-node
    # classifier (is_weight_node) needs meta["val"] shapes on the get_attr nodes.
    _populate_getattr_val_meta(gm)
    ref_cat, ref_other = gm(x)
    ref_cat = ref_cat.clone()
    ref_other = ref_other.clone()

    n_lin_before = sum(1 for n in gm.graph.nodes if is_op(n, lin_op))
    assert n_lin_before == 4  # 3 siblings + 1 singleton

    transform = FuseReplicatedBf16Linear(TransformConfig(stage=Stages.POST_LOAD_FUSION))
    new_gm, info = transform._apply(gm, None, None, SharedConfig())
    new_gm.recompile()

    # Exactly one sibling group (the 3 sharing x) fused; the singleton is untouched.
    assert info.num_matches == 1, f"expected 1 fused group, got {info.num_matches}"
    n_lin_after = sum(1 for n in new_gm.graph.nodes if is_op(n, lin_op))
    assert n_lin_after == 2, (
        f"expected 2 linears after fusion (1 fused + 1 singleton), got {n_lin_after}"
    )
    n_narrow = sum(
        1 for n in new_gm.graph.nodes if n.op == "call_function" and "narrow" in str(n.target)
    )
    assert n_narrow == 3, f"expected 3 narrow splits, got {n_narrow}"

    out_cat, out_other = new_gm(x)
    # M=2 GEMV path: algebraically identical, differs only by cuBLASLt reduction rounding.
    torch.testing.assert_close(out_cat, ref_cat, rtol=1e-2, atol=1e-2)
    # The singleton must be byte-identical (untouched by the transform).
    assert torch.equal(out_other, ref_other), "singleton linear was altered"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_concat_rows_matches_per_linear():
    """Op-level identity: aten.linear over concatenated rows == per-weight linears.

    Uses aten.linear directly (no dtype/SM branch) so the row-concat identity is checked
    independent of the cublas_mm decode routing.
    """
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
