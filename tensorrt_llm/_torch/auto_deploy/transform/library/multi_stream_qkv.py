"""Transform for multi-stream execution of Q/K/V projections in GQA attention.

In GPT-OSS-style attention the input layernorm output forks into THREE
independent linear projections (q_proj, k_proj, v_proj) that all merge at
the attention op:

  - **Q projection** (heaviest): hidden -> num_heads * head_dim
  - **K projection** (lighter):  hidden -> num_kv_heads * head_dim
  - **V projection** (lighter):  hidden -> num_kv_heads * head_dim

For GQA models (num_kv_heads << num_heads, e.g. GPT-OSS 64/8) the K and V
projections combined are still much lighter than Q.  This transform moves
K and V onto the auxiliary CUDA stream so they execute concurrently with
the heavier Q projection on the main stream.  The pattern is analogous to
``multi_stream_mla_attn`` but matches a 3-way (q/k/v) fork instead of the
MLA-specific 2-way fork (q_a / kv_a_with_mqa).
"""

from typing import Callable, List, Optional, Tuple

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

# Linear ops whose output we may reroute onto the aux stream.
_LINEAR_OPS: List[Callable] = [
    torch.ops.auto_deploy.torch_linear_simple,
    torch.ops.aten.linear,
]


def _is_linear(node: Node) -> bool:
    return is_op(node, _LINEAR_OPS)


def _linear_out_features(node: Node) -> Optional[int]:
    """Return the output feature dimension of a linear node, or ``None``.

    Reads the weight argument's fake-tensor shape (weight is laid out as
    ``(out_features, in_features)``).  Falls back to ``None`` when the
    weight node carries no shape metadata, in which case the caller skips
    classification for that fork.
    """
    if len(node.args) < 2:
        return None
    weight = node.args[1]
    if not isinstance(weight, Node):
        return None
    val = weight.meta.get("val", None)
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return None
    return int(val.shape[0])


def _find_qkv_fork_lighter_linears(gm: GraphModule) -> List[Tuple[Node, List[Node]]]:
    """Find ``(fork_point, [lighter_linears])`` pairs for q/k/v forks.

    A *fork point* is a node that directly feeds three or more supported
    linear ops.  The heaviest linear (largest ``out_features``) is treated
    as the Q-projection and stays on the main stream; the remaining linears
    are returned as candidates for aux-stream execution.

    Returns an empty list for forks that don't conclusively look like q/k/v
    (fewer than three direct linear users, or weight shapes unavailable).
    """
    results: List[Tuple[Node, List[Node]]] = []

    for node in gm.graph.nodes:
        linear_users = [u for u in node.users if _is_linear(u)]
        if len(linear_users) < 3:
            continue

        out_dims = [_linear_out_features(ln) for ln in linear_users]
        if any(d is None for d in out_dims):
            continue

        heaviest_idx = max(range(len(linear_users)), key=lambda i: out_dims[i])
        heaviest_dim = out_dims[heaviest_idx]
        lighter = [ln for i, ln in enumerate(linear_users) if i != heaviest_idx]
        lighter_dims = [out_dims[i] for i in range(len(linear_users)) if i != heaviest_idx]

        # Require Q strictly larger than the lighter projections (true GQA).
        # MHA (num_kv_heads == num_heads) has equal-sized q/k/v and gains
        # nothing from this rewrite — skip those to avoid pointless overhead.
        if any(d >= heaviest_dim for d in lighter_dims):
            continue

        results.append((node, lighter))

    return results


def _create_aux_op(base_op: Callable) -> Callable:
    return create_derived_custom_op(
        base_op,
        "_aux",
        _make_aux_stream_impl,
        make_fake=lambda base: lambda *a, **kw: base(*a, **kw),
    )


def _execute_qkv_lighter_in_aux_stream(gm: GraphModule) -> Tuple[GraphModule, int]:
    """Reroute the K/V projections of each q/k/v fork onto the aux stream.

    For each matched ``(fork_point, [lighter_linears])`` the rewriter:

    1. Inserts ``record_event_passthrough(fork_point)`` immediately before
       the heaviest (Q) linear, so the main-stream event is recorded *before*
       Q kernels are submitted.
    2. Replaces each lighter linear's target with its ``_aux`` variant and
       rewires its hidden-state input to ``record_event_passthrough`` to
       create a true data dependency.

    Aux ops are created lazily — only for base ops that actually appear in
    matched lighter positions.
    """
    pairs = _find_qkv_fork_lighter_linears(gm)
    if not pairs:
        return gm, 0

    graph = gm.graph
    node_order = {n: i for i, n in enumerate(graph.nodes)}

    ops_in_graph = {ln.target for _, lighters in pairs for ln in lighters}
    op_dict = {op: _create_aux_op(op) for op in ops_in_graph}

    num_replaced = 0

    for fork_point, lighters in pairs:
        lighter_set = set(lighters)
        # The heaviest (Q) linear is the only direct linear user of the fork
        # point that is NOT in the lighter set.  Use graph order to break
        # ties when there are multiple Q-like linears (shouldn't happen in
        # practice, but guards against weird IR shapes).
        heavies = [u for u in fork_point.users if _is_linear(u) and u not in lighter_set]
        if not heavies:
            continue
        earliest_heavy = min(heavies, key=lambda n: node_order.get(n, 0))

        # Insert record_event_passthrough right before the first Q-chain
        # linear so the event is recorded before Q kernels hit the GPU.
        with graph.inserting_before(earliest_heavy):
            rec_node = graph.call_function(
                record_event_passthrough,
                args=(fork_point,),
            )

        # Replace each lighter linear with its aux-stream variant, rewiring
        # the hidden-state input to ``rec_node`` for a true data dependency.
        for lin in lighters:
            new_args = tuple(rec_node if arg is fork_point else arg for arg in lin.args)
            with graph.inserting_after(lin):
                new_node = graph.call_function(
                    op_dict[lin.target], args=new_args, kwargs=lin.kwargs
                )
            lin.replace_all_uses_with(new_node)
            graph.erase_node(lin)
            num_replaced += 1

    return gm, num_replaced


@TransformRegistry.register("multi_stream_qkv")
class MultiStreamQKV(BaseTransform):
    """Multi-stream Q/(K,V) projection parallelism for GQA attention blocks."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        cuda_stream_manager.add_device(torch.cuda.current_device())

        gm, num_matches = _execute_qkv_lighter_in_aux_stream(gm)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
