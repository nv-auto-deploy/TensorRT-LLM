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
"""Correctness + no-collision proof for the scoped ``fuse_gemms_mixed_children`` (idea_0022).

Enabling ``fuse_gemms_mixed_children`` (``fp8_only=True``) merges sibling
``torch_fake_quant_finegrained_fp8_linear`` projections that read the *same* activation
into one concatenated ``[sum_N, K]`` block-FP8 GEMM + ``torch.narrow`` views. Unlike
``fuse_finegrained_fp8_gate_up`` (which only merges the equal-shape shared-expert w1/w3),
this transform also merges the DIFFERENT-shaped DeepSeek-V4 attention groups
(``wq_a``+``wkv``, ``wq_b``+``indexer.wq_b``), so these tests guard the general
different-N concat identity.

They also guard the fused-parameter namespace: the transform registers its fused weights
under a ``mixed_children_fused_weight_{idx}`` prefix instead of the generic
``fused_weight_{idx}`` used by ``_insert_fused_gemm`` / ``fuse_replicated_bf16_linear``.
Without that unique prefix the two transforms collide — the later ``setattr`` overwrites
the earlier fused weight, so an FP8 fused node silently reads a smaller bf16 tensor and
shape-prop fails with ``start (0) + length (N) exceeds dimension size``.
"""

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
    # per-128x128-block weight scale, strictly positive (matches checkpoint layout)
    ws = (
        torch.rand(N // 128, K // 128, generator=g, device=device, dtype=torch.float32) * 0.05
        + 0.01
    )
    return w, ws


def _populate_getattr_val_meta(gm):
    """Tag get_attr nodes with meta['val'] (their param/buffer tensor).

    The production pipeline shape-props before post_load_fusion, so the weight-node
    classifier (``is_weight_node`` -> ``has_shape``) sees a shaped ``meta['val']`` on
    every parameter get_attr. ``symbolic_trace`` does not populate that.
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


# ---------------------------------------------------------------------------
# Op-level identity: concatenating DIFFERENT-shaped block-FP8 siblings.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("M", [8, 16])
def test_different_shaped_concat_bit_exact_base_kernel(M):
    """M > 4 uses the deterministic (non split-K) base kernel -> byte-exact merge, even
    when the fused siblings have different output widths (wq_a=512 + wkv=1024 style)."""
    device = "cuda"
    K, Na, Nb = 256, 512, 1024  # different N, both 128-aligned
    x = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    wa, wsa = _make_fp8_weight(Na, K, device, seed=1)
    wb, wsb = _make_fp8_weight(Nb, K, device, seed=2)

    def _lin(w, ws):
        return lin_op(x, w, None, input_scale=[], weight_scale=[ws], input_zp=[], weight_zp=[])

    ya, yb = _lin(wa, wsa), _lin(wb, wsb)

    # Fused: cat weights + cat per-block scales along the output dim, one matmul, then slice.
    cat_w = torch.cat([wa, wb], dim=0)
    cat_s = torch.cat([wsa, wsb], dim=0)
    merged = lin_op(x, cat_w, None, input_scale=[], weight_scale=[cat_s], input_zp=[], weight_zp=[])

    assert merged.shape == (M, Na + Nb)
    assert torch.equal(merged[:, :Na], ya), "first sibling slice is not bit-exact"
    assert torch.equal(merged[:, Na:], yb), "second sibling slice is not bit-exact"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_different_shaped_concat_matches_splitk_decode():
    """M=1, K>=4096 hits the split-K atomic decode path; the concat is algebraically
    identical, differing only by the split-K's own atomic-reduction rounding."""
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


# ---------------------------------------------------------------------------
# Graph-level: the transform fuses the sibling group and preserves numerics.
# ---------------------------------------------------------------------------
class _FP8Proj(torch.nn.Module):
    """One block-FP8 projection stored as ``<name>.weight`` + ``<name>.weight_scale_inv``
    (mirrors the checkpoint layout the fuser reconstructs the scale name from)."""

    def __init__(self, N, K, device, seed):
        super().__init__()
        w, ws = _make_fp8_weight(N, K, device, seed)
        self.weight = torch.nn.Parameter(w, requires_grad=False)
        self.register_buffer("weight_scale_inv", ws)

    def forward(self, x):
        # Positional args (no kwargs) to mirror how the DeepSeek-V4 graph emits the op;
        # the fuser rebuilds input_scale/weight_scale/zp positionally, so a kwarg here
        # would double-specify them.
        return lin_op(x, self.weight, None, [], [self.weight_scale_inv], [], [])


class _FP8SiblingModule(torch.nn.Module):
    """x -> {proj_a(N=512), proj_b(N=1024)} siblings (different shapes) + a singleton."""

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

    assert sum(1 for n in gm.graph.nodes if is_op(n, lin_op)) == 3

    cfg = FuseGemmsMixedChildrenConfig(stage=Stages.POST_LOAD_FUSION, fp8_only=True)
    new_gm, info = FuseGemmsMixedChildren(cfg)._apply(gm, None, None, SharedConfig())
    new_gm.recompile()

    # Exactly one sibling group (proj_a + proj_b) fused; the solo projection is untouched.
    assert info.num_matches == 1, f"expected 1 fused group, got {info.num_matches}"
    assert sum(1 for n in new_gm.graph.nodes if is_op(n, lin_op)) == 2  # 1 fused + 1 solo
    n_narrow = sum(
        1 for n in new_gm.graph.nodes if n.op == "call_function" and "narrow" in str(n.target)
    )
    assert n_narrow == 2, f"expected 2 narrow splits, got {n_narrow}"

    out_a, out_b, out_solo = new_gm(x)
    # M=2 decode GEMV: algebraically identical, differs only by kernel reduction rounding.
    torch.testing.assert_close(out_a, ref_a, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_b, ref_b, rtol=1e-2, atol=1e-2)
    assert torch.equal(out_solo, ref_solo), "singleton projection was altered"


# ---------------------------------------------------------------------------
# Regression: mixed-children FP8 fusion must not collide with the bf16 fusion.
# ---------------------------------------------------------------------------
class _MixedModule(torch.nn.Module):
    """x feeds BOTH block-FP8 siblings and replicated bf16 siblings.

    mixed_children fuses the FP8 pair (registers a fused weight), then
    fuse_replicated_bf16_linear fuses the bf16 pair. Both used to register
    ``fused_weight_0`` -> the second overwrote the first and shape-prop blew up.
    """

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
        za = bf16_lin_op(x, self.bf_a, None, tp_mode="none", layer_type="mla")
        zb = bf16_lin_op(x, self.bf_b, None, tp_mode="none", layer_type="mla")
        return ya, yb, za, zb


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_no_param_collision_with_replicated_bf16():
    device = "cuda"
    K = 256
    x = (torch.randn(2, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()

    gm: GraphModule = symbolic_trace(_MixedModule(K, device))
    _populate_getattr_val_meta(gm)
    ref = tuple(t.clone() for t in gm(x))

    # Run mixed_children (FP8) FIRST, then the bf16 fusion -- the production order.
    cfg = FuseGemmsMixedChildrenConfig(stage=Stages.POST_LOAD_FUSION, fp8_only=True)
    gm, info_fp8 = FuseGemmsMixedChildren(cfg)._apply(gm, None, None, SharedConfig())
    gm, info_bf16 = FuseReplicatedBf16Linear(TransformConfig(stage=Stages.POST_LOAD_FUSION))._apply(
        gm, None, None, SharedConfig()
    )
    gm.recompile()

    assert info_fp8.num_matches == 1, "FP8 sibling pair should fuse"
    assert info_bf16.num_matches == 1, "bf16 sibling pair should fuse"

    # The two transforms must register disjoint fused-parameter namespaces.
    fused_names = [
        n.target for n in gm.graph.nodes if n.op == "get_attr" and "fused_weight" in str(n.target)
    ]
    mixed = [n for n in fused_names if str(n).startswith("mixed_children_fused_weight")]
    generic = [n for n in fused_names if str(n).startswith("fused_weight")]
    assert mixed, "mixed_children fused weight not found under its unique prefix"
    assert generic, "bf16 fused weight not found under the generic prefix"
    assert set(mixed).isdisjoint(set(generic)), f"fused param name collision: {fused_names}"

    # And the graph must still execute correctly (the collision previously corrupted the
    # FP8 fused weight -> shape-prop / runtime error).
    out = gm(x)
    for got, exp in zip(out, ref):
        torch.testing.assert_close(got, exp, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
