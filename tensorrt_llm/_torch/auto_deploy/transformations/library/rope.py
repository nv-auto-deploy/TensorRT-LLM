import operator
from typing import List

import torch
from torch.fx import GraphModule

from ...utils.logger import ad_logger
from ...utils.node_utils import bfs, identify_regions_between_residuals, is_op
from .._graph import canonicalize_graph


def _match_rotary_subpattern(add_node):
    """
    Given an aten.add.Tensor node that is expected to compute:
      output = (raw_input * unsqueeze(cos)) + (rotate_half(raw_input) * unsqueeze(sin))
    where rotate_half is implemented as:
      rotate_half(x) = cat([ -slice(x, second_half), slice(x, first_half) ], dim=-1)
    this function inspects the structure of add_node and returns a dictionary with:
       - "raw_input": the original q/k tensor,
       - "unsqueeze_cos": the unsqueeze node feeding the raw multiplication,
       - "unsqueeze_sin": the unsqueeze node feeding the rotated multiplication,
       - "add_node": the addition node itself.
    Returns None if the pattern does not match.
    """
    # Check that add_node is an add operation with two inputs.
    if not is_op(add_node, torch.ops.aten.add):
        return None
    if not (len(add_node.args) == 2):
        return None

    mul1, mul2 = add_node.args
    # Both inputs to the add should be multiplications.
    if not is_op(mul1, torch.ops.aten.mul):
        return None
    if not is_op(mul2, torch.ops.aten.mul):
        return None

    # One branch should be the raw branch and the other the rotated branch.
    # We decide by checking if one multiplication’s first argument is a cat (i.e. the rotate_half result).
    if is_op(mul1.args[0], torch.ops.aten.cat):
        mul_rot = mul1
        mul_raw = mul2
    elif is_op(mul2.args[0], torch.ops.aten.cat):
        mul_rot = mul2
        mul_raw = mul1
    else:
        return None

    # Verify that both multiplications have an unsqueeze as their second argument.
    unsqueeze_cos = mul_raw.args[1]
    unsqueeze_sin = mul_rot.args[1]
    if not is_op(unsqueeze_cos, torch.ops.aten.unsqueeze):
        return None
    if not is_op(unsqueeze_sin, torch.ops.aten.unsqueeze):
        return None

    # Check that the rotated branch is a cat of two tensors along -1.
    cat_node = mul_rot.args[0]
    if not is_op(cat_node, torch.ops.aten.cat):
        return None
    # Expecting two inputs in a list/tuple.
    cat_inputs = cat_node.args[0]
    if not (isinstance(cat_inputs, (list, tuple)) and len(cat_inputs) == 2):
        return None

    # One of the two inputs should be a negation of a slice, the other should be a slice.
    first_item, second_item = cat_inputs
    if not is_op(first_item, torch.ops.aten.neg):
        return None
    if not is_op(second_item, torch.ops.aten.slice):
        return None

    # The negation node should wrap a slice.
    neg_node = first_item
    if not (len(neg_node.args) >= 1 and is_op(neg_node.args[0], torch.ops.aten.slice)):
        return None

    # For simplicity, require that the two slice operations (the one inside neg and the one used directly)
    # are applied on the same original tensor. This original tensor is the one being rotated.
    slice_in_neg = neg_node.args[0]
    if slice_in_neg.args[0] != second_item.args[0]:
        return None

    # Finally, the raw branch should multiply the original tensor (i.e. q or k) by unsqueeze_cos.
    raw_input = mul_raw.args[0]
    # We also expect that the tensor being sliced (and negated) is the same as raw_input.
    if raw_input != slice_in_neg.args[0]:
        return None

    return {
        "raw_input": raw_input,
        "unsqueeze_cos": unsqueeze_cos,
        "unsqueeze_sin": unsqueeze_sin,
        "add_node": add_node,
    }


def match_rope(gm: GraphModule) -> GraphModule:
    """
    Identify the subgraph corresponding to rotary positional embeddings in each region.
    For each region, we search for the two branches implementing:
       (raw * unsqueeze(cos)) + (rotate_half(raw) * unsqueeze(sin))
    If exactly two such patterns are found, a heuristic is used to decide which is query and which is key.
    Then the cosine-sine cache is precomputed and the subgraph is replaced with a call to torch.ops.rope.flashinfer.
    """
    graph = gm.graph
    boundary_nodes: List[torch.fx.Node] = identify_regions_between_residuals(gm)

    for start_boundary, end_boundary in zip(boundary_nodes[:-1], boundary_nodes[1:]):
        region_matches = []
        node = start_boundary
        while node != end_boundary:
            if is_op(node, torch.ops.aten.add):
                match_info = _match_rotary_subpattern(node)
                if match_info is not None:
                    region_matches.append(match_info)
            node = node.next

        if len(region_matches) == 0:
            continue
        if len(region_matches) != 2:
            raise RuntimeError(
                f"Expected to find exactly 2 rotary embedding branches in region"
                f" between {start_boundary} and {end_boundary}, but found {len(region_matches)}."
            )
        # Use a heuristic to decide which branch is query and which is key.
        q_match = None
        k_match = None
        for cand in region_matches:
            raw_name = cand["raw_input"].name.lower() if hasattr(cand["raw_input"], "name") else ""
            if "q" in raw_name and q_match is None:
                q_match = cand
            elif "k" in raw_name and k_match is None:
                k_match = cand
        if q_match is None or k_match is None:
            # Fall back on ordering if names are ambiguous.
            q_match, k_match = region_matches[0], region_matches[1]

        q_node = q_match["raw_input"]
        k_node = k_match["raw_input"]

        cos_node = q_match["unsqueeze_cos"].args[0]
        sin_node = q_match["unsqueeze_sin"].args[0]
        # Sanity-check: ensure cos_node eventually comes from torch.ops.aten.cos
        bfs(
            cos_node,
            lambda n: is_op(n, torch.ops.aten.cos),
            attr_next="all_input_nodes",
            boundary=start_boundary,
        )
        # Sanity-check: ensure sin_node eventually comes from torch.ops.aten.sin
        bfs(
            sin_node,
            lambda n: is_op(n, torch.ops.aten.sin),
            attr_next="all_input_nodes",
            boundary=start_boundary,
        )
        # TODO: extract a concrete value and register as buffer to avoid recalculation

        # We expect the tensor shape to be either:
        #   [b, n, s, d]  if q_fake.shape[1] is an integer
        #   [b, s, n, d]  if q_fake.shape[1] is symbolic
        q_fake = q_node.meta.get("val", None)
        if q_fake is not None and len(q_fake.shape) > 2:
            if isinstance(q_fake.shape[1], int):
                need_transpose = True
                ad_logger.debug("Infer RoPE input has layout: [b, n, s, d]")
            else:
                need_transpose = False
                ad_logger.debug("Infer RoPE input has layout: [b, s, n, d]")
        else:
            ad_logger.warning("Unable to infer layout of q node. Defaulting to [b, n, s, d].")
            need_transpose = True

        with graph.inserting_before(q_match["add_node"]):
            if need_transpose:
                # Input is [b, n, s, d]; transpose to [b, s, n, d] for the custom op.
                q_for_op = graph.call_function(torch.ops.aten.transpose, args=(q_node, 1, 2))
                k_for_op = graph.call_function(torch.ops.aten.transpose, args=(k_node, 1, 2))
                q_for_op_contig = graph.call_method("contiguous", (q_for_op,))
                k_for_op_contig = graph.call_method("contiguous", (k_for_op,))
            else:
                q_for_op_contig = q_node
                k_for_op_contig = k_node
            flash_node = graph.call_function(
                torch.ops.rope.flashinfer,
                args=(q_for_op_contig, k_for_op_contig, cos_node, sin_node, True),
            )
        with graph.inserting_after(flash_node):
            raw_q = graph.call_function(operator.getitem, args=(flash_node, 0))
            raw_k = graph.call_function(operator.getitem, args=(flash_node, 1))
        if need_transpose:
            with graph.inserting_after(raw_q):
                new_q = graph.call_function(torch.ops.aten.transpose, args=(raw_q, 1, 2))
            with graph.inserting_after(raw_k):
                new_k = graph.call_function(torch.ops.aten.transpose, args=(raw_k, 1, 2))
        else:
            new_q, new_k = raw_q, raw_k

        q_match["add_node"].replace_all_uses_with(new_q)
        k_match["add_node"].replace_all_uses_with(new_k)

    gm = canonicalize_graph(gm)
    return gm
