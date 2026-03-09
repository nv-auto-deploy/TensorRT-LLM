# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for sharding hint kwargs on torch_linear_simple.

AD-SHARD-001: Verify that torch_linear_simple accepts TP sharding hints
(tp_mode, output_sizes, tp_min_local_shape) as keyword arguments,
that they are metadata-only (do not change numerical output), and that
they are preserved on FX graph nodes.
"""

import importlib.util
import os

import pytest
import torch

_linear_path = os.path.join(
    os.path.dirname(__file__),
    *([".."] * 8),
    "tensorrt_llm",
    "_torch",
    "auto_deploy",
    "custom_ops",
    "linear",
    "linear.py",
)
_spec = importlib.util.spec_from_file_location("_ad_linear", os.path.normpath(_linear_path))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

torch.manual_seed(42)

M, K, N = 4, 32, 64
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 8, 2, 16
Q_DIM = NUM_Q_HEADS * HEAD_DIM
KV_DIM = NUM_KV_HEADS * HEAD_DIM


def _make_inputs(m=M, k=K, n=N, bias=True):
    x = torch.randn(m, k, device="cuda")
    w = torch.randn(n, k, device="cuda")
    b = torch.randn(n, device="cuda") if bias else None
    return x, w, b


# ---------------------------------------------------------------------------
# Test 1: Op accepts each valid tp_mode value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_mode", ["none", "colwise", "rowwise"])
def test_tp_mode_accepted(tp_mode):
    x, w, b = _make_inputs()
    out = torch.ops.auto_deploy.torch_linear_simple(x, w, b, tp_mode=tp_mode)
    assert out.shape == (M, N)


# ---------------------------------------------------------------------------
# Test 2: Op accepts output_sizes
# ---------------------------------------------------------------------------
def test_fused_group_sizes_accepted():
    fused_n = Q_DIM + 2 * KV_DIM
    x, w, b = _make_inputs(n=fused_n)
    out = torch.ops.auto_deploy.torch_linear_simple(
        x,
        w,
        b,
        tp_mode="colwise",
        output_sizes=[Q_DIM, KV_DIM, KV_DIM],
    )
    assert out.shape == (M, fused_n)


# ---------------------------------------------------------------------------
# Test 3: Op accepts tp_min_local_shape
# ---------------------------------------------------------------------------
def test_min_local_shape_accepted():
    x, w, b = _make_inputs()
    out = torch.ops.auto_deploy.torch_linear_simple(
        x,
        w,
        b,
        tp_mode="colwise",
        tp_min_local_shape=HEAD_DIM,
    )
    assert out.shape == (M, N)


# ---------------------------------------------------------------------------
# Test 4: All three hint kwargs together
# ---------------------------------------------------------------------------
def test_all_hints_together():
    fused_n = Q_DIM + 2 * KV_DIM
    x, w, b = _make_inputs(n=fused_n)
    out = torch.ops.auto_deploy.torch_linear_simple(
        x,
        w,
        b,
        tp_mode="colwise",
        output_sizes=[Q_DIM, KV_DIM, KV_DIM],
        tp_min_local_shape=HEAD_DIM,
    )
    assert out.shape == (M, fused_n)


# ---------------------------------------------------------------------------
# Test 5: Hints do not change numerical output
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tp_mode", ["none", "colwise", "rowwise"])
@pytest.mark.parametrize("bias", [True, False])
def test_hints_do_not_change_output(tp_mode, bias):
    x, w, b = _make_inputs(bias=bias)
    out_baseline = torch.ops.auto_deploy.torch_linear_simple(x, w, b)
    out_hinted = torch.ops.auto_deploy.torch_linear_simple(
        x,
        w,
        b,
        tp_mode=tp_mode,
        tp_min_local_shape=HEAD_DIM,
    )
    torch.testing.assert_close(out_baseline, out_hinted, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 6: Hints are preserved on FX graph nodes
# ---------------------------------------------------------------------------
def _get_node_hint_args(node):
    """Extract sharding hints from a torch_linear_simple node.

    torch.export flattens all args positionally into node.args (not kwargs).
    Schema: (input, weight, bias, tp_mode, output_sizes, tp_min_local_shape)
    """
    return {
        "tp_mode": node.args[3] if len(node.args) > 3 else None,
        "output_sizes": node.args[4] if len(node.args) > 4 else None,
        "tp_min_local_shape": node.args[5] if len(node.args) > 5 else None,
    }


def test_hints_visible_on_graph_node():
    class HintedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(N, K))
            self.b = torch.nn.Parameter(torch.randn(N))

        def forward(self, x):
            return torch.ops.auto_deploy.torch_linear_simple(
                x,
                self.w,
                self.b,
                tp_mode="colwise",
                output_sizes=[Q_DIM, KV_DIM, KV_DIM],
                tp_min_local_shape=HEAD_DIM,
            )

    model = HintedLinear().cuda()
    gm = torch.export.export(model, (torch.randn(M, K, device="cuda"),))
    graph = gm.graph_module.graph

    found = False
    for node in graph.nodes:
        if node.op == "call_function" and "torch_linear_simple" in str(node.target):
            found = True
            hints = _get_node_hint_args(node)
            assert hints["tp_mode"] == "colwise", f"tp_mode wrong, got: {hints}"
            assert list(hints["output_sizes"]) == [Q_DIM, KV_DIM, KV_DIM]
            assert hints["tp_min_local_shape"] == HEAD_DIM
            break

    assert found, "torch_linear_simple node not found in graph"


# ---------------------------------------------------------------------------
# Test 7: Backward compatibility -- call without hints still works
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bias", [True, False])
def test_backward_compat_no_hints(bias):
    x, w, b = _make_inputs(bias=bias)
    out = torch.ops.auto_deploy.torch_linear_simple(x, w, b)
    expected = torch.ops.aten.linear(x, w, b)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
