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

"""CPU-only allreduce transform tests: collinear/fp8-residual-add rewrites + ONESHOT qualification."""

import torch
import torch.fx as fx

# Register the distributed + FP8 linear custom ops used by the rewrites.
import tensorrt_llm._torch.auto_deploy.custom_ops.distributed.trtllm_dist  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.distributed.trtllm_dist import (
    ONESHOT_SMALL_STRATEGY,
    resolve_oneshot_small_strategy,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    qualify_small_oneshot_allreduce,
    resolve_plain_allreduce_strategy,
)
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_AR = torch.ops.auto_deploy.trtllm_dist_all_reduce.default
_TORCH_AR = torch.ops.auto_deploy.torch_dist_all_reduce.default
_LINEAR = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default
_LINEAR_RESIDUAL_ADD = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add
_N, _H, _K = 4, 16, 16


def _run(gm, enabled=True, name="fuse_collinear_allreduce"):
    shared_config = SharedConfig(local_rank=0, world_size=1)
    config_cls = TransformRegistry.get_config_class(name)
    config = config_cls(stage="post_load_fusion", enabled=enabled)
    transform = TransformRegistry.get(name)(config)
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


def _build_dual_ar_add_graph(strategy_a="NCCL", strategy_b="NCCL", alpha=1):
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    ar_a = graph.call_function(_AR, args=(a, strategy_a))
    ar_b = graph.call_function(_AR, args=(b, strategy_b))
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(ar_a, ar_b), kwargs={"alpha": alpha})
    return _finalize(graph, add)


def _build_fp8_linear_add_allreduce_graph(alpha=1, linear_first=True):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    weight_scale = graph.placeholder("weight_scale")
    residual = graph.placeholder("residual")
    linear = graph.call_function(
        _LINEAR,
        args=(x, weight, None, [], [weight_scale], [], []),
    )
    add_args = (linear, residual) if linear_first else (residual, linear)
    add = graph.call_function(
        torch.ops.aten.add.Tensor,
        args=add_args,
        kwargs={"alpha": alpha},
    )
    allreduce = graph.call_function(_AR, args=(add, "NCCL"))
    graph.output(allreduce)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target == "x":
                node.meta["val"] = _meta(shape=(_N, _K))
            elif node.target == "weight":
                node.meta["val"] = _meta(shape=(_H, _K), dtype=torch.float8_e4m3fn)
            elif node.target == "weight_scale":
                node.meta["val"] = _meta(shape=(1, 1), dtype=torch.float32)
            else:
                node.meta["val"] = _meta()
        elif is_op(node, (_LINEAR, torch.ops.aten.add.Tensor, _AR)):
            node.meta["val"] = _meta()
    return gm


def _build_dual_ar_with_noop_cast_graph(cast_op=torch.ops.aten._to_copy.default):
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
    gm = _build_dual_ar_add_graph()
    assert _count(gm, _AR) == 2
    gm, info = _run(gm)
    assert info.num_matches == 1
    # 2 collectives -> 1; the surviving all_reduce must consume the add: AR(a + b).
    assert _count(gm, _AR) == 1
    assert _count(gm, torch.ops.aten.add.Tensor) == 1
    ar = next(n for n in gm.graph.nodes if is_op(n, _AR))
    assert is_op(ar.args[0], torch.ops.aten.add.Tensor)


def test_skips_scaled_dual_allreduce_add():
    # Non-unit alpha: moving the add ahead of the reductions would change its value.
    gm = _build_dual_ar_add_graph(alpha=2)
    gm, info = _run(gm)
    assert info.num_matches == 0
    assert _count(gm, _AR) == 2


def test_fp8_residual_add_fuses_only_for_unit_alpha():
    transform_name = "fuse_fp8_linear_allreduce_add"
    gm = _build_fp8_linear_add_allreduce_graph()
    gm, info = _run(gm, name=transform_name)
    assert info.num_matches == 1
    assert _count(gm, _LINEAR_RESIDUAL_ADD) == 1

    for linear_first in (True, False):
        gm = _build_fp8_linear_add_allreduce_graph(alpha=2, linear_first=linear_first)
        gm, info = _run(gm, name=transform_name)
        assert info.num_matches == 0
        assert _count(gm, _LINEAR) == 1
        assert _count(gm, _LINEAR_RESIDUAL_ADD) == 0


def test_fuses_through_noop_to_copy():
    gm = _build_dual_ar_with_noop_cast_graph(torch.ops.aten._to_copy.default)
    gm, info = _run(gm)
    assert info.num_matches == 1
    assert _count(gm, _AR) == 1


def test_fuses_through_noop_to_dtype():
    # The shared expert's trailing .to(x.dtype) lowers to aten.to.dtype, not _to_copy.
    gm = _build_dual_ar_with_noop_cast_graph(torch.ops.aten.to.dtype)
    gm, info = _run(gm)
    assert info.num_matches == 1
    assert _count(gm, _AR) == 1


def test_skips_when_strategy_differs():
    gm = _build_dual_ar_add_graph(strategy_a="NCCL", strategy_b="ONESHOT")
    gm, info = _run(gm)
    assert info.num_matches == 0
    assert _count(gm, _AR) == 2


def test_skips_when_allreduce_is_shared():
    # ar_a has a second consumer -> fusing would be net +1 collective.
    graph = torch.fx.Graph()
    a = graph.placeholder("a")
    b = graph.placeholder("b")
    ar_a = graph.call_function(_AR, args=(a, "NCCL"))
    ar_b = graph.call_function(_AR, args=(b, "NCCL"))
    add = graph.call_function(torch.ops.aten.add.Tensor, args=(ar_a, ar_b))
    extra = graph.call_function(torch.ops.aten.mul.Tensor, args=(ar_a, add))
    gm = _finalize(graph, extra)
    gm, info = _run(gm)
    assert info.num_matches == 0
    assert _count(gm, _AR) == 2


def test_skips_when_shapes_mismatch():
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


# Small-message ONESHOT allreduce qualification (static transform-time gate +
# per-call runtime numel gate). No collectives run here; multi-GPU numerics live in
# tests/unittest/auto_deploy/multigpu/custom_ops/test_small_oneshot_allreduce.py.


def _dc(world_size=4, tp_size=4, strategy="NCCL"):
    return DistConfig(
        world_size=world_size,
        rank=0,
        tp_size=tp_size,
        moe_ep_size=tp_size,
        allreduce_strategy=strategy,
    )


def _node_with_val(val):
    g = fx.Graph()
    n = g.placeholder("x")
    if val is not None:
        n.meta["val"] = val
    return n


def test_qualified_grid_upgrades_to_oneshot_small():
    strategy = qualify_small_oneshot_allreduce(_dc(), _AR, torch.bfloat16, 4096)
    assert strategy == ONESHOT_SMALL_STRATEGY


def test_explicit_non_nccl_strategy_is_preserved():
    for base in ("AUTO", "SYMM_MEM", "ONESHOT", "TWOSHOT"):
        strategy = qualify_small_oneshot_allreduce(_dc(strategy=base), _AR, torch.bfloat16, 4096)
        assert strategy == base


def test_torch_backend_keeps_nccl():
    assert qualify_small_oneshot_allreduce(_dc(), _TORCH_AR, torch.bfloat16, 4096) == "NCCL"


def test_other_topologies_keep_nccl():
    assert (
        qualify_small_oneshot_allreduce(_dc(world_size=8, tp_size=8), _AR, torch.bfloat16, 4096)
        == "NCCL"
    )
    assert (
        qualify_small_oneshot_allreduce(_dc(world_size=4, tp_size=2), _AR, torch.bfloat16, 4096)
        == "NCCL"
    )


def test_other_dtype_or_hidden_keeps_nccl():
    assert qualify_small_oneshot_allreduce(_dc(), _AR, torch.float16, 4096) == "NCCL"
    assert qualify_small_oneshot_allreduce(_dc(), _AR, torch.bfloat16, 8192) == "NCCL"
    # non-int (e.g. symbolic) last dim cannot prove a static hidden size
    assert qualify_small_oneshot_allreduce(_dc(), _AR, torch.bfloat16, None) == "NCCL"


def test_node_meta_resolution():
    qualified = _node_with_val(torch.empty(2, 1, 4096, dtype=torch.bfloat16, device="meta"))
    assert resolve_plain_allreduce_strategy(_dc(), qualified, _AR) == ONESHOT_SMALL_STRATEGY

    wrong_hidden = _node_with_val(torch.empty(2, 1, 2048, dtype=torch.bfloat16, device="meta"))
    assert resolve_plain_allreduce_strategy(_dc(), wrong_hidden, _AR) == "NCCL"

    wrong_dtype = _node_with_val(torch.empty(2, 1, 4096, dtype=torch.float32, device="meta"))
    assert resolve_plain_allreduce_strategy(_dc(), wrong_dtype, _AR) == "NCCL"

    missing_meta = _node_with_val(None)
    assert resolve_plain_allreduce_strategy(_dc(), missing_meta, _AR) == "NCCL"


def test_runtime_numel_gate():
    # one decode token at hidden 4096 -> ONESHOT; anything larger -> NCCL
    assert resolve_oneshot_small_strategy(4096) == "ONESHOT"
    assert resolve_oneshot_small_strategy(1) == "ONESHOT"
    assert resolve_oneshot_small_strategy(4097) == "NCCL"
    assert resolve_oneshot_small_strategy(2 * 4096) == "NCCL"
    assert resolve_oneshot_small_strategy(512 * 4096) == "NCCL"
