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

"""Unit tests for the fuse_collinear_allreduce transform.

These are pure FX graph-rewrite tests (no collective execution): they verify that
``add(all_reduce(a), all_reduce(b))`` collapses to ``all_reduce(add(a, b))`` and
that the safety guards (sole-user, matching strategy, matching shape) hold. The
algebraic identity ``AR(a) + AR(b) == AR(a + b)`` is exact for any reduction over
a single shared process group; AutoDeploy's ``*_dist_all_reduce`` always reduce
over the world group, so the rewrite is value-preserving end-to-end.
"""

import torch

# Register the distributed custom ops (auto_deploy::trtllm_dist_all_reduce, ...).
import tensorrt_llm._torch.auto_deploy.custom_ops.distributed.trtllm_dist  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_AR = torch.ops.auto_deploy.trtllm_dist_all_reduce.default
_N, _H = 4, 16


def _run(gm, enabled=True):
    """Apply only the fuse_collinear_allreduce transform."""
    shared_config = SharedConfig(local_rank=0, world_size=1)
    config_cls = TransformRegistry.get_config_class("fuse_collinear_allreduce")
    config = config_cls(stage="post_load_fusion", enabled=enabled)
    transform = TransformRegistry.get("fuse_collinear_allreduce")(config)
    return transform._apply(gm, cm=None, factory=None, shared_config=shared_config)


def _meta(shape=(_N, _H), dtype=torch.bfloat16):
    return torch.empty(shape, dtype=dtype, device="meta")


def _finalize(graph, out, extra_meta=None):
    graph.output(out)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            n.meta["val"] = _meta()
        elif is_op(n, _AR) or is_op(n, torch.ops.aten.add.Tensor):
            n.meta["val"] = _meta()
        elif is_op(n, (torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype)):
            n.meta["val"] = _meta()
    if extra_meta:
        extra_meta(gm)
    return gm


def _count(gm, op):
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def _build_dual_ar_add_graph(strategy_a="NCCL", strategy_b="NCCL"):
    """add(all_reduce(a, sA), all_reduce(b, sB)) -> output."""
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    ar_a = graph.call_function(_AR, args=(a, strategy_a))
    ar_b = graph.call_function(_AR, args=(b, strategy_b))
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(ar_a, ar_b))
    return _finalize(graph, add)


def _build_dual_ar_with_noop_cast_graph(cast_op=torch.ops.aten._to_copy.default):
    """add(all_reduce(a), cast(all_reduce(b))) with a dtype-preserving cast.

    ``cast_op`` defaults to ``_to_copy`` but also covers ``aten.to.dtype`` -- the
    form the row-parallel MLP's trailing ``.to(x.dtype)`` actually lowers to in the
    real DeepSeek-V4 graph.
    """
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    ar_a = graph.call_function(_AR, args=(a, "NCCL"))
    ar_b = graph.call_function(_AR, args=(b, "NCCL"))
    if cast_op is torch.ops.aten.to.dtype:
        cast = graph.call_function(cast_op, args=(ar_b, torch.bfloat16))
    else:
        cast = graph.call_function(cast_op, args=(ar_b,), kwargs={"dtype": torch.bfloat16})
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(ar_a, cast))
    return _finalize(graph, add)


def test_fuses_dual_allreduce():
    """Two world all_reduces feeding one add collapse into a single all_reduce."""
    gm = _build_dual_ar_add_graph()
    assert _count(gm, _AR) == 2
    gm, info = _run(gm)
    assert info.num_matches == 1
    # 2 collectives -> 1; the add survives (now reducing the summed local partials).
    assert _count(gm, _AR) == 1
    assert _count(gm, torch.ops.aten.add.Tensor) == 1
    # The surviving all_reduce must consume the add (i.e. AR(a + b), not add(AR, AR)).
    ar = next(n for n in gm.graph.nodes if is_op(n, _AR))
    assert is_op(ar.args[0], torch.ops.aten.add.Tensor)


def test_fuses_through_noop_to_copy():
    """A dtype-preserving _to_copy between an all_reduce and the add is peeled."""
    gm = _build_dual_ar_with_noop_cast_graph(torch.ops.aten._to_copy.default)
    gm, info = _run(gm)
    assert info.num_matches == 1
    assert _count(gm, _AR) == 1


def test_fuses_through_noop_to_dtype():
    """aten.to.dtype (the real shared-MLP `.to(x.dtype)` form) is peeled too.

    This is the exact node that blocked the first DeepSeek-V4 build: the shared
    expert's trailing ``.to(x.dtype)`` lowers to ``aten.to.dtype``, not
    ``_to_copy``, so ``add(all_reduce, to.dtype(all_reduce))`` only fuses if the
    peeler handles both cast op forms.
    """
    gm = _build_dual_ar_with_noop_cast_graph(torch.ops.aten.to.dtype)
    gm, info = _run(gm)
    assert info.num_matches == 1
    assert _count(gm, _AR) == 1


def test_skips_when_strategy_differs():
    """Different all_reduce strategies => different collectives, do not fuse."""
    gm = _build_dual_ar_add_graph(strategy_a="NCCL", strategy_b="ONESHOT")
    gm, info = _run(gm)
    assert info.num_matches == 0
    assert _count(gm, _AR) == 2


def test_skips_when_allreduce_is_shared():
    """An all_reduce reused elsewhere must not be folded (would add a 3rd collective)."""
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    ar_a = graph.call_function(_AR, args=(a, "NCCL"))
    ar_b = graph.call_function(_AR, args=(b, "NCCL"))
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(ar_a, ar_b))
    # ar_a has a second consumer -> not sole-user -> fusion would be net +1 collective.
    extra = graph.call_function(torch.ops.aten.mul.Tensor, args=(ar_a, add))
    gm = _finalize(graph, extra)
    gm, info = _run(gm)
    assert info.num_matches == 0
    assert _count(gm, _AR) == 2


def test_skips_when_shapes_mismatch():
    """All_reduce inputs of different shapes cannot be summed before reducing."""
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    ar_a = graph.call_function(_AR, args=(a, "NCCL"))
    ar_b = graph.call_function(_AR, args=(b, "NCCL"))
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(ar_a, ar_b))

    def _set_mismatched(gm):
        for n in gm.graph.nodes:
            if n.op == "placeholder" and n.target == "b":
                n.meta["val"] = _meta(shape=(_N, _H * 2))

    graph.output(add)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            n.meta["val"] = _meta()
        elif is_op(n, _AR) or is_op(n, torch.ops.aten.add.Tensor):
            n.meta["val"] = _meta()
    _set_mismatched(gm)
    gm, info = _run(gm)
    assert info.num_matches == 0
    assert _count(gm, _AR) == 2


def test_skipped_when_disabled():
    gm = _build_dual_ar_add_graph()
    gm, info = _run(gm, enabled=False)
    assert info.skipped
    assert _count(gm, _AR) == 2
