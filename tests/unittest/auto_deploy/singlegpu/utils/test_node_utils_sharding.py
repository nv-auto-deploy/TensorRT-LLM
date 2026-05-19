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

"""CPU-only tests for sharding-related FX node predicates in ``node_utils``."""

import operator

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import GraphModule

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — register custom ops
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    FineGrainedFP8WeightShardingInfo,
    ShardingSource,
    ShardingTransformConfig,
    ShardingTransformContainer,
    SplitDimension,
    detect_sharding_from_config,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding_ir import ShardableNode
from tensorrt_llm._torch.auto_deploy.utils.node_utils import (
    extract_weight_name,
    get_all_layer_subgraphs,
    get_layer_after_linear_node,
    get_weight_shape,
    identify_regions_between_residuals,
    is_any_split_op,
    is_any_view_op,
)


def _call_function_nodes(gm: GraphModule):
    return [n for n in gm.graph.nodes if n.op == "call_function"]


def test_identify_regions_uses_embedding_input_ids_seed():
    root = nn.Module()
    root.weight = nn.Parameter(torch.randn(16, 8))

    graph = fx.Graph()
    input_ids = graph.placeholder("input_ids")
    hidden_states = graph.placeholder("hidden_states")
    weight = graph.get_attr("weight")
    embedding = graph.call_function(torch.ops.aten.embedding.default, args=(weight, input_ids))
    residual = graph.call_function(torch.ops.aten.add.Tensor, args=(embedding, hidden_states))
    graph.output(residual)
    gm = GraphModule(root, graph)

    boundaries = identify_regions_between_residuals(gm)

    assert [node.name for node in boundaries] == [
        input_ids.name,
        embedding.name,
        residual.name,
        "output",
    ]


def test_identify_regions_uses_inputs_embeds_when_input_ids_is_unused():
    graph = fx.Graph()
    input_ids = graph.placeholder("input_ids")
    inputs_embeds = graph.placeholder("inputs_embeds")
    hidden_states = graph.placeholder("hidden_states")
    residual = graph.call_function(torch.ops.aten.add.Tensor, args=(inputs_embeds, hidden_states))
    graph.output(residual)
    gm = GraphModule(nn.Module(), graph)

    boundaries = identify_regions_between_residuals(gm)

    assert input_ids not in boundaries
    assert [node.name for node in boundaries] == [
        inputs_embeds.name,
        residual.name,
        "output",
    ]


def test_identify_regions_returns_minimal_boundaries_without_residual_seed():
    graph = fx.Graph()
    input_ids = graph.placeholder("input_ids")
    hidden_states = graph.placeholder("hidden_states")
    residual = graph.call_function(torch.ops.aten.add.Tensor, args=(input_ids, hidden_states))
    graph.output(residual)
    gm = GraphModule(nn.Module(), graph)

    boundaries = identify_regions_between_residuals(gm)

    assert boundaries == [input_ids, next(node for node in gm.graph.nodes if node.op == "output")]


def test_is_any_view_op_aten_view():
    class ViewModel(nn.Module):
        def forward(self, x):
            return x.view(2, 4)

    # ``symbolic_trace`` records ``Tensor.view`` as ``call_method``; ``torch.export`` lowers to
    # ``torch.ops.aten.view.default``, which ``is_any_view_op`` matches.
    exported = torch.export.export(ViewModel(), (torch.randn(8),))
    gm = exported.module()
    assert any(n.target == torch.ops.aten.view.default for n in _call_function_nodes(gm))
    assert any(is_any_view_op(n) for n in _call_function_nodes(gm)), (
        f"Expected aten view in graph, got targets: {[n.target for n in _call_function_nodes(gm)]}"
    )


def test_is_any_view_op_auto_deploy():
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(x, [2, 4]),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    graph.output(out)
    gm = GraphModule(nn.Module(), graph)
    view_nodes = [n for n in _call_function_nodes(gm) if is_any_view_op(n)]
    assert len(view_nodes) == 1
    assert is_any_view_op(view_nodes[0])


def test_is_any_view_op_negative():
    class AtenLinearOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.ops.aten.linear.default(x, self.w, self.b)

    gm = torch.fx.symbolic_trace(AtenLinearOnly())
    assert not any(is_any_view_op(n) for n in _call_function_nodes(gm)), (
        f"Unexpected view op in linear-only graph: {[n.target for n in _call_function_nodes(gm)]}"
    )


def test_get_weight_shape_returns_none_for_unregistered_weight():
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    lin = graph.call_function(torch.ops.aten.linear.default, args=(x, w, None))
    graph.output(lin)
    gm = GraphModule(nn.Module(), graph)

    lin_node = next(
        n for n in _call_function_nodes(gm) if n.target == torch.ops.aten.linear.default
    )
    assert get_weight_shape(lin_node) is None


def test_extract_weight_name_with_auxiliary_parameter_user():
    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))

    root = Shell()
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("w")
    w.meta["val"] = root.w
    graph.call_function(torch._assert, args=(w, "keep weight live"))
    lin = graph.call_function(torch.ops.aten.linear.default, args=(x, w, None))
    graph.output(lin)
    gm = GraphModule(root, graph)

    lin_node = next(
        n for n in _call_function_nodes(gm) if n.target == torch.ops.aten.linear.default
    )
    assert extract_weight_name(lin_node) == "w"


def test_config_sharding_skips_linear_without_weight_name():
    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))

    root = Shell()
    graph = fx.Graph()
    x = graph.placeholder("x")
    runtime_w = graph.placeholder("runtime_w")
    w = graph.get_attr("w")
    w.meta["val"] = root.w
    shapeable = graph.call_function(torch.ops.aten.linear.default, args=(x, w, None))
    unshapeable = graph.call_function(
        torch.ops.aten.linear.default, args=(shapeable, runtime_w, None)
    )
    graph.output(unshapeable)
    gm = GraphModule(root, graph)

    container = ShardingTransformContainer(
        config=ShardingTransformConfig(
            stage="sharding", manual_config={"tp_plan": {"lm_head": "gather"}}
        )
    )
    info = detect_sharding_from_config(gm, container, ShardingSource.MANUAL)

    assert info.num_matches == 0


def test_draft_mtp_prologue_does_not_break_manual_attention_sharding():
    hidden_size = 32

    class DraftShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def add_weight_node(graph, root, name):
        node = graph.get_attr(name)
        node.meta["val"] = root.get_parameter(name)
        return node

    root = DraftShell()
    graph = fx.Graph()
    inputs_embeds = graph.placeholder("inputs_embeds")
    inputs_embeds.meta["val"] = torch.empty(2, 4, hidden_size)
    hidden_states = graph.placeholder("hidden_states")
    hidden_states.meta["val"] = torch.empty(2, 4, hidden_size)

    cat = graph.call_function(torch.ops.aten.cat.default, args=([inputs_embeds, hidden_states], -1))
    cat.meta["val"] = torch.empty(2, 4, hidden_size * 2)
    fc = graph.call_function(
        torch.ops.aten.linear.default, args=(cat, add_weight_node(graph, root, "fc.weight"), None)
    )
    fc.meta["val"] = torch.empty(2, 4, hidden_size)
    q_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "q_proj.weight"), None),
    )
    q_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    k_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "k_proj.weight"), None),
    )
    k_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    v_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "v_proj.weight"), None),
    )
    v_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    attention = graph.call_function(
        torch.ops.auto_deploy.torch_attention.default,
        args=(q_proj, k_proj, v_proj, None, 0.0, True, "bsnd"),
    )
    attention.meta["val"] = torch.empty(2, 4, hidden_size)
    o_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(attention, add_weight_node(graph, root, "o_proj.weight"), None),
    )
    o_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    graph.output(o_proj)
    gm = GraphModule(root, graph)

    layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)
    assert [node.name for node in layer_subgraphs[0].opening_nodes] == [
        "linear_default_1",
        "linear_default_2",
        "linear_default_3",
    ]
    assert layer_subgraphs[0].terminating_node.name == "linear_default_4"
    assert "linear_default" in {node.name for node in unprocessed_linear_nodes}

    container = ShardingTransformContainer(
        config=ShardingTransformConfig(
            stage="sharding",
            rank=0,
            world_size=8,
            manual_config={
                "tp_plan": {
                    "q_proj": "colwise",
                    "k_proj": "colwise",
                    "v_proj": "colwise",
                    "o_proj": "rowwise",
                }
            },
        )
    )
    info = detect_sharding_from_config(gm, container, ShardingSource.MANUAL)

    assert info.num_matches == 4
    assert {transform.target_node for transform in container.weight_sharding_transforms} == {
        "linear_default_1",
        "linear_default_2",
        "linear_default_3",
        "linear_default_4",
    }


def test_qwen3_5_gated_attention_uses_grouped_column_sharding():
    hidden_size = 32
    head_size = 8
    num_heads = 4
    num_kv_heads = 2

    class DraftShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.q_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
            self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def add_weight_node(graph, root, name):
        node = graph.get_attr(name)
        node.meta["val"] = root.get_parameter(name)
        return node

    root = DraftShell()
    graph = fx.Graph()
    inputs_embeds = graph.placeholder("inputs_embeds")
    inputs_embeds.meta["val"] = torch.empty(2, 4, hidden_size)
    hidden_states = graph.placeholder("hidden_states")
    hidden_states.meta["val"] = torch.empty(2, 4, hidden_size)

    cat = graph.call_function(torch.ops.aten.cat.default, args=([inputs_embeds, hidden_states], -1))
    cat.meta["val"] = torch.empty(2, 4, hidden_size * 2)
    fc = graph.call_function(
        torch.ops.aten.linear.default, args=(cat, add_weight_node(graph, root, "fc.weight"), None)
    )
    fc.meta["val"] = torch.empty(2, 4, hidden_size)
    q_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "q_proj.weight"), None),
    )
    q_proj.meta["val"] = torch.empty(2, 4, hidden_size * 2)
    q_view = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(q_proj, (2, 4, num_heads, head_size * 2)),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    q_view.meta["val"] = torch.empty(2, 4, num_heads, head_size * 2)
    q_chunk = graph.call_function(torch.ops.aten.chunk.default, args=(q_view, 2, -1))
    q_chunk.meta["val"] = (torch.empty(2, 4, num_heads, head_size),) * 2
    query = graph.call_function(operator.getitem, args=(q_chunk, 0))
    query.meta["val"] = torch.empty(2, 4, num_heads, head_size)
    gate = graph.call_function(operator.getitem, args=(q_chunk, 1))
    gate.meta["val"] = torch.empty(2, 4, num_heads, head_size)
    gate_flat = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(gate, (2, 4, hidden_size)),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    gate_flat.meta["val"] = torch.empty(2, 4, hidden_size)
    k_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "k_proj.weight"), None),
    )
    k_proj.meta["val"] = torch.empty(2, 4, num_kv_heads * head_size)
    k_view = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(k_proj, (2, 4, num_kv_heads, head_size)),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    k_view.meta["val"] = torch.empty(2, 4, num_kv_heads, head_size)
    v_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "v_proj.weight"), None),
    )
    v_proj.meta["val"] = torch.empty(2, 4, num_kv_heads * head_size)
    v_view = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(v_proj, (2, 4, num_kv_heads, head_size)),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    v_view.meta["val"] = torch.empty(2, 4, num_kv_heads, head_size)
    attention = graph.call_function(
        torch.ops.auto_deploy.torch_attention.default,
        args=(query, k_view, v_view, None, 0.0, True, "bsnd"),
    )
    attention.meta["val"] = torch.empty(2, 4, num_heads, head_size)
    attention_flat = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(attention, (2, 4, hidden_size)),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    attention_flat.meta["val"] = torch.empty(2, 4, hidden_size)
    gate_sigmoid = graph.call_function(torch.ops.aten.sigmoid.default, args=(gate_flat,))
    gate_sigmoid.meta["val"] = torch.empty(2, 4, hidden_size)
    gated = graph.call_function(torch.ops.aten.mul.Tensor, args=(attention_flat, gate_sigmoid))
    gated.meta["val"] = torch.empty(2, 4, hidden_size)
    o_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(gated, add_weight_node(graph, root, "o_proj.weight"), None),
    )
    o_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    graph.output(o_proj)
    gm = GraphModule(root, graph)

    layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)

    assert layer_subgraphs[0].layer_type.name == "MHA"
    assert {node.name for node in layer_subgraphs[0].opening_nodes} == {
        "linear_default_1",
        "linear_default_2",
        "linear_default_3",
    }
    assert layer_subgraphs[0].terminating_node.name == "linear_default_4"
    assert "linear_default" in {node.name for node in unprocessed_linear_nodes}

    container = ShardingTransformContainer(
        config=ShardingTransformConfig(
            stage="sharding",
            rank=0,
            world_size=4,
            manual_config={
                "tp_plan": {
                    "q_proj": "colwise",
                    "k_proj": "colwise",
                    "v_proj": "colwise",
                    "o_proj": "rowwise",
                }
            },
        )
    )
    info = detect_sharding_from_config(gm, container, ShardingSource.MANUAL)

    assert info.num_matches == 4
    assert {transform.target_node for transform in container.weight_sharding_transforms} == {
        "linear_default_1",
        "linear_default_2",
        "linear_default_3",
        "linear_default_4",
    }
    updated_views = {
        update.target_node: update.args[1]
        for update in container.parameter_update_transforms
        if "view" in update.target_node
    }
    assert updated_views["view_default"] == (2, 4, -1, 16)
    assert updated_views["view_default_1"] == (2, 4, -1)
    assert updated_views["view_default_2"] == (2, 4, -1, 8)
    assert updated_views["view_default_3"] == (2, 4, -1, 8)
    assert updated_views["view_default_4"] == (2, 4, -1)


def test_draft_embedding_inference_prefers_mtp_prologue_over_trailing_gate():
    hidden_size = 32

    class DraftShell(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def add_weight_node(graph, root, name):
        node = graph.get_attr(name)
        node.meta["val"] = root.get_parameter(name)
        return node

    root = DraftShell()
    graph = fx.Graph()
    inputs_embeds = graph.placeholder("inputs_embeds")
    inputs_embeds.meta["val"] = torch.empty(2, 4, hidden_size)
    hidden_states = graph.placeholder("hidden_states")
    hidden_states.meta["val"] = torch.empty(2, 4, hidden_size)

    cat = graph.call_function(torch.ops.aten.cat.default, args=([inputs_embeds, hidden_states], -1))
    cat.meta["val"] = torch.empty(2, 4, hidden_size * 2)
    fc = graph.call_function(
        torch.ops.aten.linear.default, args=(cat, add_weight_node(graph, root, "fc.weight"), None)
    )
    fc.meta["val"] = torch.empty(2, 4, hidden_size)
    q_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "q_proj.weight"), None),
    )
    q_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    k_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "k_proj.weight"), None),
    )
    k_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    v_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(fc, add_weight_node(graph, root, "v_proj.weight"), None),
    )
    v_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    attention = graph.call_function(
        torch.ops.auto_deploy.torch_attention.default,
        args=(q_proj, k_proj, v_proj, None, 0.0, True, "bsnd"),
    )
    attention.meta["val"] = torch.empty(2, 4, hidden_size)
    o_proj = graph.call_function(
        torch.ops.aten.linear.default,
        args=(attention, add_weight_node(graph, root, "o_proj.weight"), None),
    )
    o_proj.meta["val"] = torch.empty(2, 4, hidden_size)
    trailing_gate = graph.call_function(
        torch.ops.aten.linear.default,
        args=(o_proj, add_weight_node(graph, root, "shared_expert_gate.weight"), None),
    )
    trailing_gate.meta["val"] = torch.empty(2, 4, 1)
    graph.output(trailing_gate)
    gm = GraphModule(root, graph)
    gm.is_draft = True

    layer_subgraphs, unprocessed_linear_nodes = get_all_layer_subgraphs(gm)

    assert len(layer_subgraphs) == 1
    assert layer_subgraphs[0].layer_type.name == "MHA"
    assert {node.name for node in layer_subgraphs[0].opening_nodes} == {
        "linear_default_1",
        "linear_default_2",
        "linear_default_3",
    }
    assert layer_subgraphs[0].terminating_node.name == "linear_default_4"
    assert {"linear_default", "linear_default_5"} <= {
        node.name for node in unprocessed_linear_nodes
    }


def test_layer_boundary_skips_linear_that_is_not_an_opening_node():
    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.prologue = nn.Parameter(torch.randn(32, 64))
            self.inner = nn.Parameter(torch.randn(64, 32))
            self.terminator = nn.Parameter(torch.randn(32, 64))

    root = Shell()
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(2, 4, 32)
    prologue_weight = graph.get_attr("prologue")
    prologue_weight.meta["val"] = root.prologue
    prologue = graph.call_function(
        torch.ops.aten.linear.default,
        args=(x, prologue_weight, None),
    )
    prologue.meta["val"] = torch.empty(2, 4, 32)
    inner_weight = graph.get_attr("inner")
    inner_weight.meta["val"] = root.inner
    inner = graph.call_function(
        torch.ops.aten.linear.default,
        args=(prologue, inner_weight, None),
    )
    inner.meta["val"] = torch.empty(2, 4, 64)
    terminator_weight = graph.get_attr("terminator")
    terminator_weight.meta["val"] = root.terminator
    terminator = graph.call_function(
        torch.ops.aten.linear.default,
        args=(inner, terminator_weight, None),
    )
    terminator.meta["val"] = torch.empty(2, 4, 32)
    graph.output(terminator)
    gm = GraphModule(root, graph)

    linear_nodes = [node for node in gm.graph.nodes if node.target == torch.ops.aten.linear.default]
    for node in linear_nodes:
        node.meta["lin_node_shape"] = get_weight_shape(node)
    terminating_indices = [-1]

    layer_subgraph = get_layer_after_linear_node(
        linear_nodes,
        terminating_indices,
        embd=32,
        residuals=[],
    )

    assert layer_subgraph.opening_nodes == []
    assert layer_subgraph.terminating_node is None
    assert terminating_indices == [-1, 0]


def test_quant_scale_load_hook_runs_after_parent_remap():
    class Child(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(512, 4096))
            self.register_buffer("weight_scale_inv", torch.empty(8, 32, dtype=torch.bfloat16))

    class Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Child()
            self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

        @staticmethod
        def _remap_checkpoint_keys(state_dict, prefix, *args):
            old_key = prefix + "old.child.weight_scale_inv"
            new_key = prefix + "child.weight_scale_inv"
            if old_key in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)

    root = Root()
    sharding_info = FineGrainedFP8WeightShardingInfo(
        target_node="child_linear",
        split_dim=SplitDimension.COLUMN,
        config=ShardingTransformConfig(stage="sharding", rank=0, world_size=2),
    )

    sharding_info.quantization_cb(
        gm=root,
        submod=root.child,
        node=None,
        weight_key="child.weight",
        weight_new_shape=torch.Size([512, 4096]),
        weight_original_shape=torch.Size([1024, 4096]),
        dim=0,
        rank=0,
        world_size=2,
    )

    assert root.child.weight_scale_inv.shape == (4, 32)
    checkpoint_scale = torch.arange(8 * 32, dtype=torch.bfloat16).reshape(8, 32)
    root.load_state_dict({"old.child.weight_scale_inv": checkpoint_scale}, strict=False)

    torch.testing.assert_close(root.child.weight_scale_inv, checkpoint_scale[:4])


def test_is_any_split_op_aten():
    class SplitModel(nn.Module):
        def forward(self, x):
            a, b = torch.split(x, [2, 2], dim=-1)
            return a + b

    exported = torch.export.export(SplitModel(), (torch.randn(2, 4),))
    gm = exported.module()
    assert any(
        n.target == torch.ops.aten.split_with_sizes.default for n in _call_function_nodes(gm)
    )
    assert any(is_any_split_op(n) for n in _call_function_nodes(gm)), (
        f"Expected split op in graph, got: {[n.target for n in _call_function_nodes(gm)]}"
    )


def test_is_any_split_op_auto_deploy():
    graph = fx.Graph()
    x = graph.placeholder("x")
    splits = graph.call_function(
        torch.ops.auto_deploy.split_with_sizes.default,
        args=(x, [2, 2], -1),
        kwargs={"enable_sharding": False, "layer_type": "unknown"},
    )
    first = graph.call_function(operator.getitem, args=(splits, 0))
    graph.output(first)
    gm = GraphModule(nn.Module(), graph)
    split_nodes = [n for n in _call_function_nodes(gm) if is_any_split_op(n)]
    assert len(split_nodes) == 1
    assert is_any_split_op(split_nodes[0])


def _minimal_graph_module_for_enable_sharding_linear():
    class Shell(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

    root = Shell()
    graph = fx.Graph()
    x = graph.placeholder("x")
    w = graph.get_attr("w")
    b = graph.get_attr("b")
    lin = graph.call_function(
        torch.ops.auto_deploy.torch_linear_simple.default,
        args=(x, w, b, "none", None, 1, "unknown"),
        kwargs={},
    )
    graph.output(lin)
    return GraphModule(root, graph)


def test_enable_sharding_node_linear():
    gm = _minimal_graph_module_for_enable_sharding_linear()
    lin_nodes = [n for n in _call_function_nodes(gm) if ShardableNode.from_node(n) is not None]
    assert len(lin_nodes) == 1
    assert ShardableNode.from_node(lin_nodes[0]) is not None


def test_enable_sharding_node_view():
    graph = fx.Graph()
    x = graph.placeholder("x")
    out = graph.call_function(
        torch.ops.auto_deploy.view.default,
        args=(x, [2, 4]),
        kwargs={"tp_scaled_dim": -1, "layer_type": "unknown"},
    )
    graph.output(out)
    gm = GraphModule(nn.Module(), graph)
    view_nodes = [n for n in _call_function_nodes(gm) if ShardableNode.from_node(n) is not None]
    assert len(view_nodes) == 1
    assert ShardableNode.from_node(view_nodes[0]) is not None


def test_enable_sharding_node_none_for_aten():
    class AtenLinearOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(4, 8))
            self.b = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.ops.aten.linear.default(x, self.w, self.b)

    gm = torch.fx.symbolic_trace(AtenLinearOnly())
    aten_linear = [n for n in _call_function_nodes(gm) if n.target == torch.ops.aten.linear.default]
    assert len(aten_linear) == 1
    assert ShardableNode.from_node(aten_linear[0]) is None
