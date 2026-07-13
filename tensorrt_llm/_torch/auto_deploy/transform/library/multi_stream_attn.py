# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Multi-stream MLA transform: overlaps Q and KV paths on separate CUDA streams.

Applies up to two optimizations, tried in priority order:

**Pattern 0 — Full KV path overlap (unfused GEMMs)**:

When Q_a and KV_a projections are separate GEMMs (fuse_gemms_mixed_children
disabled), the ENTIRE KV path (GEMM + AllGather) is placed on the auxiliary
CUDA stream using begin/end_aux_stream_passthrough, overlapping with the
heavier Q path (GEMM + AllGather + LayerNorm + Q_b_proj) on main.

This eliminates the narrow→contiguous copies that fused GEMMs require and
gives better overlap since both KV GEMM and KV AllGather run on aux.

Match (in the original FX graph — any op in all_gather_ops()):
    fork_point → Q_a_proj → ... (Q chain)
              → KV_a_proj → <any AllGather op> → ...

Rewrite (the matched AllGather is rebuilt on the aux stream with
``workspace_id=_AUX_WORKSPACE_ID``; symm-mem strategies use a distinct
workspace via this id, NCCL strategies just ignore it):

                       fork_point (input layernorm out)
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
              main stream                    aux stream
              ───────────                    ──────────
              Q_a_proj                       begin_aux
              Q_AllGather                    KV_a_proj
              Q_LayerNorm                    KV_AllGather (workspace_id=1)
              Q_b_proj                       end_aux
                  │                               │
                  └──────────► wait_aux ◄─────────┘
                                  │
                          downstream MLA

GPU timeline:
    Main: [Q_GEMM] → [Q_AllGather] → [Q_LayerNorm] → [Q_b_proj] → [wait_aux]
    Aux:  [KV_GEMM] → [KV_AllGather (aux ws)] → done

**Pattern 1 — Projection-only overlap**:

Moves only the KV projection linear onto the auxiliary CUDA stream; the rest
of the KV chain (split, rms_norm, view) stays on main.  The aux variant is
created via _make_aux_stream_impl, which records/waits events internally
instead of using the begin/end_aux passthroughs.

                       fork_point
                            │
                  ┌─────────┴─────────────────┐
                  ▼                           ▼
              main stream                 aux stream
              ───────────                 ──────────
              record_event                wait_event(main)
              Q_a_proj                    KV_a_proj
              ...                         record_event(aux)
              Q_b_proj                        │
                  │                           │
                  └────► wait_event(aux) ◄────┘
                            │
                   (KV split / rms_norm
                    continues on main)

GPU timeline:
    Main: [record_event] → [Q_a_proj] → [...] → [Q_b_proj] → [wait_aux_event]
    Aux:                   [KV_a_proj] → done

**Pattern 1 extended (config ``extended_aux_window``)**:

Same match as pattern 1, but the aux window is built from the begin/end/wait
passthroughs so it can span multiple nodes and join late.  Applies when the
side projection's outputs are consumed only through view splits that all meet
at one common consumer (the attention op):

                fork_point
                     │
        ┌────────────┴──────────────────┐
        ▼                               ▼
    main stream                     aux stream
    ───────────                     ──────────
        │                           begin_aux ─ side_proj ─ end_aux
    Q_a_proj                            │ (views of side_proj output
    ...                                 │  stay as metadata-only nodes)
    Q_b_proj ──────────────────────► begin_aux ─ side cone ─ end_aux
    ...                                 │ (kernel chains off the last
    Q rope / KV rope                    │  Q-chain linear that feed only
        │                               │  the attention op)
        └─────────► wait_aux ◄──────────┘
                        │
                  attention op

The second window needs a second main→aux event.  The manager's single
event pair is reused safely because both windows run on the same aux
stream: the AUX record at the end of window 2 may overwrite window 1's
un-waited record, but the single ``wait_aux`` on that final record
transitively covers all earlier same-stream aux work (stream-order
domination).  MAIN records are each consumed inside the next
``begin_aux`` before any re-record, and CUDA graph capture materializes
each record/wait as its own dependency edge, so no extra event is needed.
"""

import operator
from collections import deque
from typing import Callable, List, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import create_derived_custom_op, eliminate_dead_code
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import (
    _make_aux_stream_impl,
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    record_event_passthrough,
    wait_aux_stream_passthrough,
)
from ...utils.node_utils import (
    all_gather_ops,
    all_reduce_ops,
    is_any_moe_op,
    is_fake_quantized_linear_op,
    is_finegrained_fp8_linear_op,
    is_op,
)
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

# ===========================================================================
# Shared helpers
# ===========================================================================

_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]


# Multi-stream passthroughs inserted by sibling transforms; forks that already
# carry one are skipped to avoid conflicting rewrites.
_MULTI_STREAM_OPS = [
    begin_aux_stream_passthrough,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
    record_event_passthrough,
]


def _is_multi_stream_node(node: Node) -> bool:
    return node.op == "call_function" and node.target in _MULTI_STREAM_OPS


# Distinct symm-mem workspace slot for the aux KV path. The unified
# *_dist_all_gather op routes workspace_id != 0 to a separate ProcessGroup
# (and therefore a separate symm_mem workspace), so a concurrent main-stream
# allgather on workspace_id=0 cannot clobber its buffer.
_AUX_WORKSPACE_ID = 1


def _is_linear(node: Node) -> bool:
    """Return ``True`` if *node* is any kind of linear op (regular or quantized)."""
    return (
        is_op(node, _LINEAR_OPS)
        or is_fake_quantized_linear_op(node)
        or is_finegrained_fp8_linear_op(node)
    )


def _has_downstream_linear(start: Node, max_depth: int = 3) -> bool:
    """BFS from *start* through its users and return ``True`` if a linear op is reachable.

    The search only follows *user* edges (downstream in the data-flow graph)
    and stops after *max_depth* hops.  ``start`` itself is **not** checked.
    """
    visited: set[Node] = {start}
    queue: deque[Tuple[Node, int]] = deque()

    for user in start.users:
        queue.append((user, 1))

    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)

        if _is_linear(node):
            return True

        if depth < max_depth:
            for user in node.users:
                queue.append((user, depth + 1))

    return False


def _get_output_feature_dim(node: Node) -> int:
    """Get the last dimension (output features) from a node's meta shape."""
    val = node.meta.get("val")
    if val is not None and hasattr(val, "shape") and len(val.shape) > 0:
        dim = val.shape[-1]
        return int(dim)
    return 0


def _find_downstream_node(
    start: Node, predicate: Callable[[Node], bool], max_depth: int = 2
) -> Optional[Node]:
    """BFS to find the first downstream node matching *predicate*."""
    visited: set[Node] = {start}
    queue: deque[Tuple[Node, int]] = deque()
    for u in start.users:
        queue.append((u, 1))
    while queue:
        node, depth = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if predicate(node):
            return node
        if depth < max_depth:
            for u in node.users:
                queue.append((u, depth + 1))
    return None


# ===========================================================================
# Pattern 0: Full KV path on aux stream (unfused GEMMs)
# ===========================================================================


def _find_mla_qkv_pairs(gm: GraphModule) -> List[Tuple[Node, Node, Node]]:
    """Find MLA ``(fork_point, q_linear, kv_linear)`` triples.

    Identifies fork points where exactly 2 linears share the same input.
    The linear with the larger output dimension is Q_a, the smaller is KV_a.
    """
    results: List[Tuple[Node, Node, Node]] = []
    for node in gm.graph.nodes:
        linear_users = [u for u in node.users if _is_linear(u)]
        if len(linear_users) != 2:
            continue

        sizes = [_get_output_feature_dim(lin) for lin in linear_users]
        if sizes[0] <= 0 or sizes[1] <= 0 or sizes[0] == sizes[1]:
            continue

        if sizes[0] > sizes[1]:
            q_lin, kv_lin = linear_users[0], linear_users[1]
        else:
            q_lin, kv_lin = linear_users[1], linear_users[0]

        results.append((node, q_lin, kv_lin))

    return results


def _execute_kv_path_in_aux_stream(gm: GraphModule, world_size: int) -> Tuple[GraphModule, int]:
    """Move KV projection + AllGather onto the auxiliary CUDA stream.

    When Q and KV projections are separate (unfused) GEMMs, this places the
    entire KV path on the aux stream via begin/end_aux_stream_passthrough,
    overlapping with the heavier Q path on main.  The KV AllGather is
    re-emitted with ``workspace_id=_AUX_WORKSPACE_ID`` so symm-mem strategies
    use a distinct ProcessGroup/workspace and do not conflict with the
    main-stream AllGather.

    Returns ``(gm, num_matches)``.
    """
    if world_size <= 1:
        return gm, 0

    triples = _find_mla_qkv_pairs(gm)
    if not triples:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}
    num_matches = 0

    for fork_point, q_linear, kv_linear in triples:
        kv_ag = _find_downstream_node(
            kv_linear,
            lambda n: is_op(n, all_gather_ops()),
            max_depth=2,
        )
        if kv_ag is None:
            ad_logger.warning(f"No AllGather found downstream of {kv_linear.name}, skipping")
            continue

        ag_dim = kv_ag.args[1] if len(kv_ag.args) > 1 else -1

        ad_logger.info(
            f"Multi-stream MLA pattern 0 (unfused): "
            f"Q={q_linear.name} (dim={_get_output_feature_dim(q_linear)}), "
            f"KV={kv_linear.name} (dim={_get_output_feature_dim(kv_linear)}), "
            f"KV_AG={kv_ag.name} (fork={fork_point.name})"
        )

        # --- Move KV linear's get_attr args before q_linear ---
        # FP8 linears reference weight/scale_inv get_attr nodes that may sit
        # between q_linear and kv_linear in graph order.  Moving them earlier
        # is always safe (get_attr nodes have no data-flow inputs).
        q_pos = node_order.get(q_linear, 0)
        for arg in kv_linear.args:
            if isinstance(arg, Node) and arg.op == "get_attr":
                if node_order.get(arg, -1) >= q_pos:
                    q_linear.prepend(arg)
        for arg in kv_linear.kwargs.values():
            if isinstance(arg, Node) and arg.op == "get_attr":
                if node_order.get(arg, -1) >= q_pos:
                    q_linear.prepend(arg)

        # --- Build new KV path BEFORE q_linear in graph order ---
        with graph.inserting_before(q_linear):
            begin_node = graph.call_function(
                begin_aux_stream_passthrough,
                args=(fork_point,),
            )
            begin_node.meta["val"] = fork_point.meta.get("val")

            new_kv_args = tuple(begin_node if arg is fork_point else arg for arg in kv_linear.args)
            new_kv_gemm = graph.call_function(
                kv_linear.target, args=new_kv_args, kwargs=kv_linear.kwargs
            )
            for k, v in kv_linear.meta.items():
                new_kv_gemm.meta[k] = v

            ag_sizes = kv_ag.args[2] if len(kv_ag.args) > 2 else None
            ag_strategy = kv_ag.args[3] if len(kv_ag.args) > 3 else "AUTO"
            new_kv_ag = graph.call_function(
                kv_ag.target,
                args=(new_kv_gemm, ag_dim, ag_sizes, ag_strategy, _AUX_WORKSPACE_ID),
            )
            for k, v in kv_ag.meta.items():
                new_kv_ag.meta[k] = v

            end_node = graph.call_function(
                end_aux_stream_passthrough,
                args=(new_kv_ag,),
            )
            end_node.meta["val"] = kv_ag.meta.get("val")

        # --- Insert wait_aux before the earliest consumer of old kv_ag ---
        kv_ag_users = sorted(
            list(kv_ag.users.keys()),
            key=lambda n: node_order.get(n, float("inf")),
        )
        if kv_ag_users:
            earliest_user = kv_ag_users[0]
            with graph.inserting_before(earliest_user):
                wait_node = graph.call_function(
                    wait_aux_stream_passthrough,
                    args=(end_node,),
                )
                wait_node.meta["val"] = end_node.meta.get("val")
            kv_ag.replace_all_uses_with(wait_node)
        else:
            kv_ag.replace_all_uses_with(end_node)

        num_matches += 1

    if num_matches > 0:
        eliminate_dead_code(gm)

    return gm, num_matches


# ===========================================================================
# Pattern 1: Projection overlap (fallback)
# ===========================================================================


def _find_kv_proj_linears(gm: GraphModule, max_depth: int = 3) -> List[Tuple[Node, Node]]:
    """Find (fork_point, kv_linear) pairs suitable for aux-stream execution.

    A *fork point* is a node that directly feeds two or more supported linear
    ops.  Among these linears the one that does **not** lead to another linear
    within *max_depth* BFS hops is the KV projection candidate (the lighter
    branch).

    Returns a list of ``(fork_point, kv_linear_node)`` tuples.
    """
    results: List[Tuple[Node, Node]] = []

    for node in gm.graph.nodes:
        # Collect direct linear users of this node.
        linear_users = [u for u in node.users if _is_linear(u)]
        if len(linear_users) < 2:
            continue

        # Skip forks already rewritten by another multi-stream transform and MoE
        # hidden forks (the router gate consumes a separate logits node, never this).
        if any(_is_multi_stream_node(u) or is_any_moe_op(u) for u in node.users):
            continue

        # Separate into "has downstream linear" (Q-like) and "does not" (KV-like).
        kv_candidates = [ln for ln in linear_users if not _has_downstream_linear(ln, max_depth)]
        q_candidates = [ln for ln in linear_users if _has_downstream_linear(ln, max_depth)]

        if not kv_candidates or not q_candidates:
            continue

        # Pick the KV candidate(s).  In MLA there is exactly one per fork point.
        for kv_linear in kv_candidates:
            results.append((node, kv_linear))

    return results


def _create_aux_linear_op(base_op: Callable) -> Callable:
    """Create an ``_aux`` variant of a linear op that runs on the auxiliary CUDA stream."""
    return create_derived_custom_op(
        base_op,
        "_aux",
        _make_aux_stream_impl,
        make_fake=lambda base: lambda *a, **kw: base(*a, **kw),
    )


def _rewrite_kv_proj_single_op(graph, node_order, fork_point: Node, kv_linear: Node) -> None:
    """Single-op aux rewrite for one ``(fork_point, kv_linear)`` pair.

    1. Inserts ``record_event_passthrough(fork_point)`` so the main-stream
       event is recorded *before* the Q-chain kernels are submitted.
    2. Replaces the KV linear's target with its ``_aux`` variant and wires the
       ``record_event_passthrough`` output as the hidden-state input
       (creating a true data dependency).

    The remaining KV-chain ops (split, rms_norm, view) stay on the main
    stream — they are lightweight and run after the aux wait that is built
    into the derived op.
    """
    aux_op = _create_aux_linear_op(kv_linear.target)

    # Find the Q-chain linear(s) so we can insert the event record
    # *before* the earliest Q-chain op in graph order.
    q_linears = [u for u in fork_point.users if _is_linear(u) and u is not kv_linear]
    earliest_q = min(q_linears, key=lambda n: node_order.get(n, 0))

    # Insert record_event_passthrough right before the first Q-chain
    # linear so the event is recorded before Q kernels hit the GPU.
    with graph.inserting_before(earliest_q):
        rec_node = graph.call_function(
            record_event_passthrough,
            args=(fork_point,),
        )

    # Replace KV linear with its aux-stream variant.  The hidden-state
    # input (args[0]) is rewired to ``rec_node`` to create a data
    # dependency that ensures the event is recorded first.
    new_args = tuple(rec_node if arg is fork_point else arg for arg in kv_linear.args)

    with graph.inserting_after(kv_linear):
        new_node = graph.call_function(aux_op, args=new_args, kwargs=kv_linear.kwargs)

    kv_linear.replace_all_uses_with(new_node)
    graph.erase_node(kv_linear)


def _execute_kv_proj_in_aux_stream(gm: GraphModule, max_depth: int = 3) -> Tuple[GraphModule, int]:
    """Replace KV projection linears with aux-stream variants.

    Aux-stream variants are created lazily — only for base ops that actually
    appear in the matched KV positions.
    """
    pairs = _find_kv_proj_linears(gm, max_depth)
    if not pairs:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}

    num_replaced = 0
    for fork_point, kv_linear in pairs:
        _rewrite_kv_proj_single_op(graph, node_order, fork_point, kv_linear)
        num_replaced += 1

    return gm, num_replaced


# ===========================================================================
# Pattern 1 extended: multi-node aux windows with a late join
# ===========================================================================

# Pure view-split targets a fused side projection is split back with (fusion
# emits torch.narrow; exported mocks may carry the aten equivalents).
_VIEW_SPLIT_TARGETS = (
    torch.narrow,
    torch.ops.aten.narrow.default,
    torch.ops.aten.slice.Tensor,
)

# call_function targets that never launch GPU work (metadata views / bare
# allocations).  A side cone made only of these gains nothing from a stream
# move; anything else counts as a kernel.
_NON_KERNEL_TARGETS = (
    torch.narrow,
    operator.getitem,
    torch.ops.aten.narrow.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.split_with_sizes.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.new_empty.default,
)


def _replace_input(node: Node, old: Node, new: Node) -> None:
    """Replace every use of *old* in *node*'s args/kwargs (including nested lists)."""
    node.args = torch.fx.node.map_arg(node.args, lambda a: new if a is old else a)
    node.kwargs = torch.fx.node.map_arg(node.kwargs, lambda a: new if a is old else a)


def _is_view_split(node: Node) -> bool:
    return node.op == "call_function" and node.target in _VIEW_SPLIT_TARGETS


def _find_view_join(kv_linear: Node) -> Optional[Node]:
    """Return the unique consumer behind *kv_linear*'s view-only outputs.

    Requires every user of *kv_linear* to be a pure view-split op and every
    view's user to be one common ``call_function`` node (the join consumer).
    Returns ``None`` when this shape does not hold.
    """
    if not kv_linear.users:
        return None
    join: Optional[Node] = None
    for view in kv_linear.users:
        if not _is_view_split(view) or not view.users:
            return None
        for consumer in view.users:
            if join is None:
                join = consumer
            elif consumer is not join:
                return None
    if join is None or join.op != "call_function":
        return None
    return join


def _cone_blocked(node: Node) -> bool:
    """Ops that must never migrate to the aux stream."""
    return (
        _is_linear(node)
        or _is_multi_stream_node(node)
        or is_any_moe_op(node)
        or is_op(node, all_gather_ops())
        or is_op(node, all_reduce_ops())
    )


def _collect_exclusive_cone(root: Node, join: Node, max_nodes: int = 32) -> Optional[List[Node]]:
    """Forward closure from *root* that terminates exclusively at *join*.

    Returns the cone nodes (unordered) or ``None`` if the closure escapes to
    any consumer other than *join*, contains a blocked op, or grows beyond
    *max_nodes*.
    """
    if root is join:
        return None
    cone: List[Node] = []
    seen: set = set()
    queue: List[Node] = [root]
    while queue:
        n = queue.pop()
        if n in seen:
            continue
        seen.add(n)
        if n.op not in ("call_function", "call_method") or _cone_blocked(n):
            return None
        cone.append(n)
        if len(cone) > max_nodes:
            return None
        for u in n.users:
            if u is not join:
                queue.append(u)
    return cone


def _cone_inputs_available(cone: List[Node], anchor: Node, node_order) -> bool:
    """True if every external input of *cone* is already produced at *anchor*."""
    cone_set = set(cone)
    anchor_pos = node_order.get(anchor, 0)
    for n in cone:
        for inp in n.all_input_nodes:
            if inp in cone_set or inp is anchor:
                continue
            if inp.op in ("get_attr", "placeholder"):
                continue
            if node_order.get(inp, 1 << 62) >= anchor_pos:
                return False
    return True


def _cone_weight(cone: Optional[List[Node]]) -> float:
    """Crude cost proxy (sum of output feature dims); ``None`` cones are heaviest."""
    if cone is None:
        return float("inf")
    return float(sum(_get_output_feature_dim(n) for n in cone))


def _cone_has_kernel(cone: List[Node]) -> bool:
    return any(not (n.op == "call_function" and n.target in _NON_KERNEL_TARGETS) for n in cone)


def _find_q_chain_tail(q_linear: Node, join: Node, node_order, max_depth: int) -> Node:
    """Follow the Q chain linear-to-linear; return the last linear before *join*."""
    tail = q_linear
    for _ in range(8):
        nxt = _find_downstream_node(tail, _is_linear, max_depth)
        if nxt is None or node_order.get(nxt, 1 << 62) >= node_order.get(join, 0):
            break
        tail = nxt
    return tail


def _execute_kv_proj_in_aux_stream_extended(
    gm: GraphModule, max_depth: int = 3
) -> Tuple[GraphModule, int]:
    """Pattern-1 rewrite with multi-node aux windows and a late join.

    Extends the single-op rewrite in three ways (falling back to it per fork
    when the required graph shape is absent):

    1. The side projection runs inside a begin/end aux window opened at the
       fork; its view-split outputs (metadata-only) stay in place and read the
       ``end_aux`` passthrough, so the join is no longer baked into the op.
    2. Kernel side-cones hanging off the *last* Q-chain linear that terminate
       exclusively in the join consumer (e.g. an indexer rope + quant chain
       reading a slice of a fused Q projection) migrate into a second aux
       window opened right after that linear.  The second main→aux event
       reuses the manager's single event pair: both windows share one aux
       stream, so the single wait on the final AUX record transitively
       covers window-1 work (stream-order domination); MAIN records are
       consumed inside the next ``begin_aux`` before any re-record.
    3. ``wait_aux`` is inserted immediately before the join consumer — the
       first point where the main stream actually reads aux-produced data.
    """
    pairs = _find_kv_proj_linears(gm, max_depth)
    if not pairs:
        return gm, 0

    graph = gm.graph
    num_matched = 0

    for fork_point, kv_linear in pairs:
        # Recompute per pair: earlier rewrites shift node positions.
        node_order = {n: i for i, n in enumerate(graph.nodes)}

        q_linears = [u for u in fork_point.users if _is_linear(u) and u is not kv_linear]
        earliest_q = min(q_linears, key=lambda n: node_order.get(n, 0))
        q_pos = node_order.get(earliest_q, 0)

        join = _find_view_join(kv_linear)
        # The window opens before the Q chain, so every non-attr input of the
        # side projection must already be available there.
        inputs_ready = all(
            inp is fork_point or inp.op == "get_attr" or node_order.get(inp, 1 << 62) < q_pos
            for inp in kv_linear.all_input_nodes
        )
        if join is None or not inputs_ready:
            ad_logger.info(
                f"Multi-stream MLA pattern 1: no extended-window shape at "
                f"{kv_linear.name}; using single-op aux rewrite"
            )
            _rewrite_kv_proj_single_op(graph, node_order, fork_point, kv_linear)
            num_matched += 1
            continue

        views = list(kv_linear.users)

        # ---- Window 1: side projection on aux, opened at the fork ----
        for arg in kv_linear.all_input_nodes:
            if arg.op == "get_attr" and node_order.get(arg, -1) >= q_pos:
                earliest_q.prepend(arg)
        with graph.inserting_before(earliest_q):
            begin1 = graph.call_function(begin_aux_stream_passthrough, args=(fork_point,))
            begin1.meta["val"] = fork_point.meta.get("val")
        _replace_input(kv_linear, fork_point, begin1)
        begin1.append(kv_linear)
        with graph.inserting_after(kv_linear):
            end1 = graph.call_function(end_aux_stream_passthrough, args=(kv_linear,))
            end1.meta["val"] = kv_linear.meta.get("val")
        for view in views:
            _replace_input(view, kv_linear, end1)

        # ---- Window 2: movable side cones off the last Q-chain linear ----
        tail = _find_q_chain_tail(earliest_q, join, node_order, max_depth)
        branches = list(tail.users)
        end2: Optional[Node] = None
        moved_nodes: List[Node] = []
        if len(branches) >= 2:
            cones = {b: _collect_exclusive_cone(b, join) for b in branches}
            # Keep the heaviest branch (the attention Q/KV path) on main;
            # move the lighter, self-contained kernel cones.
            kept = max(branches, key=lambda b: _cone_weight(cones[b]))
            movable = [
                b
                for b in branches
                if b is not kept
                and cones[b] is not None
                and _cone_has_kernel(cones[b])
                and _cone_inputs_available(cones[b], tail, node_order)
            ]
            if movable:
                with graph.inserting_after(tail):
                    begin2 = graph.call_function(begin_aux_stream_passthrough, args=(tail,))
                    begin2.meta["val"] = tail.meta.get("val")
                for b in movable:
                    _replace_input(b, tail, begin2)
                # Hoist the cones into a contiguous block right after begin2
                # (original relative order is topological and preserved).
                moved_nodes = sorted(
                    {n for b in movable for n in cones[b]},
                    key=lambda n: node_order.get(n, 1 << 62),
                )
                anchor = begin2
                for n in moved_nodes:
                    for inp in n.all_input_nodes:
                        if inp.op == "get_attr" and node_order.get(inp, -1) >= node_order.get(
                            tail, 0
                        ):
                            begin2.prepend(inp)
                    anchor.append(n)
                    anchor = n
                with graph.inserting_after(anchor):
                    end2 = graph.call_function(end_aux_stream_passthrough, args=(anchor,))
                    end2.meta["val"] = anchor.meta.get("val")
                for user in [u for u in list(anchor.users) if u is not end2]:
                    _replace_input(user, anchor, end2)

        # ---- Late join: main waits for aux right before the join consumer ----
        chain_src = end2 if end2 is not None else max(views, key=lambda n: node_order.get(n, 0))
        with graph.inserting_before(join):
            wait_node = graph.call_function(wait_aux_stream_passthrough, args=(chain_src,))
            wait_node.meta["val"] = chain_src.meta.get("val")
        _replace_input(join, chain_src, wait_node)

        ad_logger.info(
            f"Multi-stream MLA pattern 1 extended: side={kv_linear.name} "
            f"(views={len(views)}), join={join.name}, "
            f"second_window={[n.name for n in moved_nodes] or None}"
        )
        num_matched += 1

    return gm, num_matched


# ===========================================================================
# Transform class
# ===========================================================================


class MultiStreamMLAAttnConfig(TransformConfig):
    """Configuration for the multi-stream MLA attention transform."""

    downstream_linear_depth: int = Field(
        default=3,
        description=(
            "Max BFS depth (user hops) used to classify a fork-point linear as "
            "Q-like, i.e. another linear is reachable downstream. Fused-GEMM "
            "output splits interpose narrow+contiguous nodes, which can push "
            "the next linear one hop deeper than the unfused chain."
        ),
    )
    extended_aux_window: bool = Field(
        default=False,
        description=(
            "Pattern-1 multi-node aux window: open the window at the fork, join "
            "immediately before the common attention consumer instead of at the "
            "side projection, and migrate kernel side-cones hanging off the last "
            "Q-chain linear (e.g. an indexer rope + quant chain) onto the aux "
            "stream. Falls back to the single-op rewrite per fork when the "
            "required graph shape is absent."
        ),
    )
    decode_selection_aux: bool = Field(
        default=False,
        description=(
            "Run the decode sparse-selection chain (current-token indexer row "
            "store + index score + top-k) on the auxiliary CUDA stream inside "
            "the cached sparse-attention op, overlapping it with the "
            "main-stream cache store/update kernels; the main stream re-joins "
            "immediately before the assemble kernel that consumes the selected "
            "rows. No graph rewrite: this only flips an op-internal flag."
        ),
    )


@TransformRegistry.register("multi_stream_mla_attn")
class MultiStreamMLAAttn(BaseTransform):
    """Multi-stream Q/KV parallelism for MLA attention blocks.

    Pattern 0: Full KV path overlap for unfused Q/KV GEMMs (begin/end aux).
    Pattern 1: Overlaps KV projection linear with Q projection chain (fallback).

    Pattern 0 is tried first; if it matches (unfused graph), pattern 1 is skipped.
    If pattern 0 finds nothing (fused graph), pattern 1 runs as fallback.
    """

    config: MultiStreamMLAAttnConfig

    @classmethod
    def get_config_class(cls) -> Type[MultiStreamMLAAttnConfig]:
        return MultiStreamMLAAttnConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cuda_stream_manager.add_device(torch.cuda.current_device())

        if self.config.decode_selection_aux:
            # Op-internal aux window (no graph rewrite): flip the module flag
            # once, before warmup/capture, so eager and captured paths agree.
            from ...custom_ops.attention.deepseek_v4_sparse_attention import (
                set_decode_selection_aux,
            )

            set_decode_selection_aux(True)
            ad_logger.info("Multi-stream MLA: decode selection on aux stream enabled")

        # Pattern 0: full KV path on aux (unfused GEMMs)
        gm, n_unfused = _execute_kv_path_in_aux_stream(gm, shared_config.world_size)
        ad_logger.info(f"Multi-stream MLA pattern 0 (unfused KV path): {n_unfused} matches")

        if n_unfused > 0:
            total = n_unfused
        else:
            # Fallback: Pattern 1 (projection overlap)
            if self.config.extended_aux_window:
                gm, n_proj = _execute_kv_proj_in_aux_stream_extended(
                    gm, self.config.downstream_linear_depth
                )
            else:
                gm, n_proj = _execute_kv_proj_in_aux_stream(gm, self.config.downstream_linear_depth)
            ad_logger.info(f"Multi-stream MLA pattern 1 (projection): {n_proj} matches")
            total = n_proj

        info = TransformInfo(
            skipped=False,
            num_matches=total,
            is_clean=total == 0,
            has_valid_shapes=total == 0,
        )
        return gm, info
