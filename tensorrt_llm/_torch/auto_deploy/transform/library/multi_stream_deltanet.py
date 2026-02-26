"""Transform for multi-stream execution of CausalConv1d and GDN Gating in DeltaNet layers.

In Qwen3.5/DeltaNet-style linear attention, the linear projection output forks
into two independent computation paths that merge at the gated delta rule op:

  - **CausalConv1d path** (heavier): narrow -> causal_conv1d -> split -> reshape (q, k, v)
  - **GDN Gating path** (lighter): narrow -> sigmoid (beta) + triton_fused_gdn_gating (decay)

This transform moves the GDN Gating path onto the auxiliary CUDA stream so it
executes concurrently with the CausalConv1d path on the main stream.
"""

from typing import List, Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import (
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
)
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_causal_conv1d(node: Node) -> bool:
    """Return True if *node* is a call to a causal_conv1d wrapper function."""
    if node.op != "call_function":
        return False
    name = getattr(node.target, "__name__", "")
    return "causal_conv1d" in name


def _find_conv1d_ancestor(gdr_node: Node) -> Optional[Node]:
    """Walk backward from gdr's q input to find the causal_conv1d node.

    The q input chain is: gdr.args[0] -> reshape -> getitem -> split -> conv1d.
    """
    visited: set[Node] = set()
    queue = list(gdr_node.args[0].all_input_nodes) if isinstance(gdr_node.args[0], Node) else []
    queue.insert(0, gdr_node.args[0]) if isinstance(gdr_node.args[0], Node) else None

    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        if _is_causal_conv1d(n):
            return n
        queue.extend(n.all_input_nodes)
    return None


def _execute_gdn_gating_in_aux_stream(gm: GraphModule) -> Tuple[GraphModule, int]:
    """Move GDN Gating ops onto the aux stream, parallel with CausalConv1d on main.

    For each ``fla_cached_gated_delta_rule`` node:
      1. Identify the sigmoid (beta) and triton_fused_gdn_gating (decay) nodes
         from its inputs.
      2. Find the causal_conv1d node by walking backward from the q input.
      3. Reorder the graph so GDN gating ops come before causal_conv1d.
      4. Insert ``begin_aux_stream_passthrough`` before sigmoid to switch to aux.
      5. Insert ``end_aux_stream_passthrough`` after gdn_gating to switch back.
      6. Insert ``wait_aux_stream_passthrough`` before gdr so main waits for aux.
    """
    graph = gm.graph
    num_replaced = 0

    try:
        gdr_op = torch.ops.auto_deploy.fla_cached_gated_delta_rule
        gdn_gating_op = torch.ops.auto_deploy.triton_fused_gdn_gating
    except AttributeError:
        return gm, 0

    # Collect gdr nodes first to avoid mutating while iterating.
    gdr_nodes: List[Node] = [n for n in graph.nodes if is_op(n, gdr_op)]
    if not gdr_nodes:
        return gm, 0

    for gdr_node in gdr_nodes:
        # ---- Step 1: Identify gating path nodes. ----
        # fla_cached_gated_delta_rule args:
        #   [0] = q (reshape from conv1d)
        #   [1] = k (reshape from conv1d)
        #   [2] = v (reshape from conv1d)
        #   [3] = gating (triton_fused_gdn_gating output)
        #   [4] = beta (sigmoid output)
        #   [5+] = batch_info, caches, scale, etc.
        if len(gdr_node.args) < 5:
            ad_logger.warning(
                f"gated_delta_rule node {gdr_node.name} has fewer than 5 args; "
                "skipping multi-stream transform."
            )
            continue

        gating_node = gdr_node.args[3]
        beta_node = gdr_node.args[4]

        if not isinstance(gating_node, Node) or not isinstance(beta_node, Node):
            ad_logger.warning(
                f"gated_delta_rule node {gdr_node.name}: gating or beta input is not a Node; "
                "skipping."
            )
            continue

        if not is_op(gating_node, gdn_gating_op):
            ad_logger.warning(
                f"gated_delta_rule node {gdr_node.name}: args[3] is {gating_node.target}, "
                "expected triton_fused_gdn_gating; skipping."
            )
            continue

        if not is_op(beta_node, torch.ops.aten.sigmoid):
            ad_logger.warning(
                f"gated_delta_rule node {gdr_node.name}: args[4] is {beta_node.target}, "
                "expected aten.sigmoid; skipping."
            )
            continue

        # ---- Step 2: Find conv1d node (fork point for main stream). ----
        conv1d_node = _find_conv1d_ancestor(gdr_node)
        if conv1d_node is None:
            ad_logger.warning(
                f"Could not find causal_conv1d ancestor for {gdr_node.name}; skipping."
            )
            continue

        # ---- Step 3: Reorder graph — move sigmoid & gdn_gating before conv1d. ----
        # This allows aux stream ops to be submitted first, enabling GPU overlap
        # with the conv1d that follows on main stream.
        # Node.prepend(x) moves x to immediately before self in the graph.
        conv1d_node.prepend(gating_node)
        gating_node.prepend(beta_node)

        # ---- Step 4: Insert begin_aux before sigmoid. ----
        # Pass sigmoid's data input through begin_aux to create a data dependency.
        sigmoid_input = beta_node.args[0]  # narrow_2
        with graph.inserting_before(beta_node):
            begin_aux_node = graph.call_function(
                begin_aux_stream_passthrough,
                args=(sigmoid_input,),
            )

        # Rewire sigmoid to read from begin_aux output instead of narrow_2.
        beta_node.args = tuple(
            begin_aux_node if arg is sigmoid_input else arg for arg in beta_node.args
        )

        # ---- Step 5: Insert end_aux after gdn_gating. ----
        with graph.inserting_after(gating_node):
            end_aux_node = graph.call_function(
                end_aux_stream_passthrough,
                args=(gating_node,),
            )

        # Rewire gdr to read gating from end_aux output.
        gdr_node.args = tuple(end_aux_node if arg is gating_node else arg for arg in gdr_node.args)

        # ---- Step 6: Insert wait_aux before gdr on the main stream path. ----
        # Wrap the q input (first conv1d-path output consumed by gdr) with wait_aux
        # so the main stream waits for the aux stream before gdr executes.
        q_node = gdr_node.args[0]
        with graph.inserting_before(gdr_node):
            wait_aux_node = graph.call_function(
                wait_aux_stream_passthrough,
                args=(q_node,),
            )

        gdr_node.args = tuple(wait_aux_node if arg is q_node else arg for arg in gdr_node.args)

        num_replaced += 1
        ad_logger.info(f"Multi-stream DeltaNet: moved GDN gating to aux stream for {gdr_node.name}")

    return gm, num_replaced


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------


@TransformRegistry.register("multi_stream_deltanet")
class MultiStreamDeltaNet(BaseTransform):
    """Multi-stream CausalConv1d / GDN Gating parallelism for DeltaNet layers."""

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Ensure aux stream and events are set up for the current device.
        cuda_stream_manager.add_device(torch.cuda.current_device())

        gm, num_matches = _execute_gdn_gating_in_aux_stream(gm)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
