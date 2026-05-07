# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transform for multi-stream execution of MoE routers.

In MoE blocks the router (linear + topk + softmax + scatter) and the dense
expert path both consume the post-attention layernorm output ``xn2``.  When the
two appear as separate ops in the FX graph (``torch_moe_router`` followed by
``torch_moe_dense_mlp``), the router is the lighter side and can be moved onto
the auxiliary CUDA stream so its launch overlaps with kernels that the main
stream issues just before the experts kick in.

The pattern matched here is::

    xn2 -> torch_moe_router(...)        -> routing_weights
        -> torch_moe_dense_mlp(...)     -> experts_out

After the rewrite the router runs on the aux stream and the dense MLP keeps
running on the main stream; the dense MLP's data dependency on
``routing_weights`` ensures correctness via the wait recorded at the end of
the aux variant.  This is structurally distinct from ``multi_stream_moe``
(shared expert vs routed experts) and from ``multi_stream_qkv`` (q/k/v
projection fork) — neither of those covers the router-vs-experts split.

Note: when MoE weights are MXFP4-quantized, ``quantize_mxfp4_moe`` fuses the
router and the dense MLP into a single ``triton_mxfp4_moe`` op at the
``pattern_matcher`` stage.  This transform runs at the ``compile`` stage and
finds nothing to do for those models — no router op survives the fusion.
"""

from typing import Callable, List, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import create_derived_custom_op
from ...utils.multi_stream_utils import (
    _make_aux_stream_impl,
    cuda_stream_manager,
    record_event_passthrough,
)
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# Router op variants that can be moved to the aux stream.  Two cases:
#   1. ``torch_moe_router``: a separate router op followed by a dense MLP op.
#      We pair it with its downstream expert node and move the router to aux.
#   2. ``triton_mxfp4_moe``: a single fused op that internally does the router
#      linear + topk + experts.  No separate fork is available so we move the
#      whole fused op to aux instead.  The benefit is bounded but it lets the
#      MoE kernel dispatch overlap with the residual-add / post-attention
#      layernorm that prepare its input.
_ROUTER_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_moe_router,
]

# Expert ops that consume routing_weights.  Used to identify a true
# router/experts pair (the experts node is the "anchor" we keep on main).
_EXPERT_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_moe_dense_mlp,
]

# Fused router+experts ops.  The whole op is the unit of aux-stream movement
# for these — the router is internal and cannot be split out from outside.
_FUSED_MOE_OPS: List[Callable] = [
    torch.ops.auto_deploy.triton_mxfp4_moe,
]


def _is_router(node: Node) -> bool:
    return is_op(node, _ROUTER_OPS)


def _is_expert(node: Node) -> bool:
    return is_op(node, _EXPERT_OPS)


def _is_fused_moe(node: Node) -> bool:
    return is_op(node, _FUSED_MOE_OPS)


def _find_router_expert_pairs(gm: GraphModule) -> List[Tuple[Node, Node, Node]]:
    """Return ``(fork_point, router_node, expert_node)`` triples.

    A *router/experts pair* is a router op whose output is consumed by an
    expert op that ALSO consumes the same ``hidden_states`` (the fork point).
    This is the canonical MoE shape produced by AD's modeling code.
    """
    pairs: List[Tuple[Node, Node, Node]] = []

    for router in gm.graph.nodes:
        if not _is_router(router):
            continue
        if len(router.args) < 1:
            continue
        hidden = router.args[0]
        if not isinstance(hidden, Node):
            continue

        # Find an expert user whose first arg is the same hidden tensor.
        expert: Node = None
        for user in router.users:
            if not _is_expert(user):
                continue
            if len(user.args) < 1:
                continue
            if user.args[0] is hidden:
                expert = user
                break
        if expert is None:
            continue

        pairs.append((hidden, router, expert))

    return pairs


def _find_fused_moe_nodes(gm: GraphModule) -> List[Node]:
    """Return all fused MoE op nodes in *gm* (e.g., ``triton_mxfp4_moe``)."""
    return [n for n in gm.graph.nodes if _is_fused_moe(n)]


def _create_aux_op(base_op: Callable) -> Callable:
    return create_derived_custom_op(
        base_op,
        "_aux",
        _make_aux_stream_impl,
        make_fake=lambda base: lambda *a, **kw: base(*a, **kw),
    )


def _execute_router_in_aux_stream(gm: GraphModule) -> Tuple[GraphModule, int]:
    """Reroute each matched router op onto the aux stream.

    For each ``(fork_point, router, expert)`` triple the rewriter:

    1. Inserts ``record_event_passthrough(fork_point)`` immediately before the
       expert op so the main-stream event is recorded *before* any expert
       kernels are submitted.
    2. Replaces the router's target with its ``_aux`` variant and rewires its
       hidden-state input to the ``record_event_passthrough`` output to create
       a true data dependency.

    Aux ops are created lazily — only for router op types that actually appear
    in matched positions.
    """
    pairs = _find_router_expert_pairs(gm)
    if not pairs:
        return gm, 0

    graph = gm.graph
    ops_in_graph = {router.target for _, router, _ in pairs}
    op_dict = {op: _create_aux_op(op) for op in ops_in_graph}

    num_replaced = 0

    for fork_point, router, expert in pairs:
        # Insert record_event_passthrough right before the expert op so the
        # event is recorded before the expert's kernels hit the GPU.  Anchor
        # to the expert (not the router) because the expert is the heavier
        # branch that stays on the main stream.
        with graph.inserting_before(expert):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(fork_point,),
            )

        # Replace the router with its aux-stream variant, rewiring the
        # hidden-state input to ``rec_node`` for the data dependency.
        new_args = tuple(rec_node if arg is fork_point else arg for arg in router.args)
        with graph.inserting_after(router):
            new_node = graph.call_function(
                op_dict[router.target], args=new_args, kwargs=router.kwargs
            )
        router.replace_all_uses_with(new_node)
        graph.erase_node(router)
        num_replaced += 1

    return gm, num_replaced


def _execute_fused_moe_in_aux_stream(gm: GraphModule) -> Tuple[GraphModule, int]:
    """Move each fused MoE op (e.g., ``triton_mxfp4_moe``) onto the aux stream.

    For each fused MoE node *n* with hidden-state input *h*, the rewriter:

    1. Inserts ``record_event_passthrough(h)`` directly before *n* so the
       main-stream event is recorded *before* the aux variant is dispatched.
    2. Replaces *n*'s target with its ``_aux`` variant and rewires the
       hidden-state input to the ``record_event_passthrough`` output.

    The aux variant blocks the main stream on completion (via the
    ``wait_event`` baked into ``_make_aux_stream_impl``), so the rewrite
    cannot make total wall time worse than the serial case under CUDA-graph
    capture: the kernel still runs once, just on a different stream.  The
    potential gain is launch-overlap — main can keep dispatching
    pre-MoE setup work (residual add, layernorm) while the heavy MoE kernel
    is queued on aux.  This is bounded; do not expect more than a fraction
    of a percent on cuda-graph deployments.
    """
    moe_nodes = _find_fused_moe_nodes(gm)
    if not moe_nodes:
        return gm, 0

    graph = gm.graph
    ops_in_graph = {n.target for n in moe_nodes}
    op_dict = {op: _create_aux_op(op) for op in ops_in_graph}

    num_replaced = 0

    for moe_node in moe_nodes:
        if len(moe_node.args) < 1:
            continue
        hidden = moe_node.args[0]
        if not isinstance(hidden, Node):
            continue

        with graph.inserting_before(moe_node):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(hidden,),
            )

        new_args = tuple(rec_node if arg is hidden else arg for arg in moe_node.args)
        with graph.inserting_after(moe_node):
            new_node = graph.call_function(
                op_dict[moe_node.target], args=new_args, kwargs=moe_node.kwargs
            )
        moe_node.replace_all_uses_with(new_node)
        graph.erase_node(moe_node)
        num_replaced += 1

    return gm, num_replaced


@TransformRegistry.register("multi_stream_router")
class MultiStreamRouter(BaseTransform):
    """Run MoE router on the aux stream concurrent with main-stream MoE setup."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cuda_stream_manager.add_device(torch.cuda.current_device())

        gm, n_router = _execute_router_in_aux_stream(gm)
        gm, n_fused = _execute_fused_moe_in_aux_stream(gm)
        num_matches = n_router + n_fused

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
