"""Pattern matching for detecting repeat_kv pattern from Huggingface models."""

from typing import Any, Dict, List, Type

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import AttentionDescriptor
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from .._graph import canonicalize_graph, lift_to_meta


def _repeat_kv_pattern1(hidden_states, n_rep) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _repeat_kv_pattern2(hidden_states, n_rep) -> torch.Tensor:
    # using the aten op directly will force the contiguous call to be inserted
    return torch.ops.aten.contiguous.default(_repeat_kv_pattern1(hidden_states, n_rep))


def _repeat_kv_repl(hidden_states, n_rep) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_attention_repeat_kv(hidden_states, n_rep)


def match_repeat_kv(gm: GraphModule) -> None:
    """Match and replace the repeat_kv pattern in the graph.

    This is replaced with torch.ops.auto_deploy.torch_attention_repeat_kv.
    """
    graph = gm.graph
    patterns = ADPatternMatcherPass()

    # dummy shapes: can be arbitrary
    batch_size = 8
    seq_len = 16
    num_heads = 8
    hidden_size = 512
    head_dim = hidden_size // num_heads

    dummy_explicit = [
        torch.randn(batch_size, num_heads, seq_len, head_dim, device="meta", dtype=torch.float16),
        7,
    ]

    def _register(search_fn):
        register_ad_pattern(
            search_fn=search_fn,
            replace_fn=_repeat_kv_repl,
            patterns=patterns,
            dummy_args=dummy_explicit,
            op_ignore_types={
                torch.ops.aten.reshape.default: (int,),
                torch.ops.aten.expand.default: (int,),
            },
            scalar_workaround={"n_rep": dummy_explicit[1]},
        )

    _register(_repeat_kv_pattern1)
    # _register(_repeat_kv_pattern2)

    num_matches = patterns.apply(graph)
    with lift_to_meta(gm):
        gm = canonicalize_graph(gm, shape_prop=True)
    ad_logger.info(f"Found and matched {num_matches} Repeat KV patterns")


# with causal_mask, no division
def _sfdp_pattern_1(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_1(query, key, value, attention_mask, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# no causal_mask, no division
def _sfdp_pattern_2(query, key, value, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_2(query, key, value, scaling, dropout):
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# with causal_mask, with division
def _sfdp_pattern_3(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_3(query, key, value, attention_mask, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# no causal_mask, with division
def _sfdp_pattern_4(query, key, value, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_4(query, key, value, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# no causal_mask, with division, 'complex' model
def _sfdp_pattern_5(query, key, value, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = attn_weights.to(torch.float32)
    attn_weights = F.softmax(attn_weights, dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_5(query, key, value, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


# with causal_mask, with division, 'complex' model
def _sfdp_pattern_6(query, key, value, attention_mask, scaling, dropout):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) / scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = attn_weights.to(torch.float32)
    attn_weights = F.softmax(attn_weights, dim=-1).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output


def _sfdp_replacement_6(query, key, value, attention_mask, scaling, dropout):
    scaling = 1.0 / scaling
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=False,
        scale=scaling,
    )


def _get_sfdp_patterns() -> List[Dict[str, Any]]:
    bs = 8
    seq_len = 16
    n_heads = 8
    hidden_size = 512
    head_dim = hidden_size // n_heads
    return [
        {
            "search_fn": _sfdp_pattern_1,
            "replace_fn": _sfdp_replacement_1,
            "dummy_args": [
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, 1, 1, seq_len, device="cuda", dtype=torch.bfloat16),
                0.1234743,  # scaling
                0.85849734,  # dropout
            ],
            "scalar_workaround": {"dropout": 0.85849734, "scaling": 0.1234743},
            "op_ignore_types": {
                torch.ops.aten.to.dtype: (torch.dtype,),
            },
        },
        {
            "search_fn": _sfdp_pattern_2,
            "replace_fn": _sfdp_replacement_2,
            "dummy_args": [
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                0.234743,  # scaling
                0.5849734,  # dropout
            ],
            "scalar_workaround": {"dropout": 0.5849734, "scaling": 0.234743},
            "op_ignore_types": {
                torch.ops.aten.to.dtype: (torch.dtype,),
            },
        },
        {
            "search_fn": _sfdp_pattern_3,
            "replace_fn": _sfdp_replacement_3,
            "dummy_args": [
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, 1, 1, seq_len, device="cuda", dtype=torch.bfloat16),
                0.34743,  # scaling
                0.849734,  # dropout
            ],
            "scalar_workaround": {
                "scaling": 0.34743,
                "dropout": 0.849734,
            },
            "op_ignore_types": {
                torch.ops.aten.to.dtype: (torch.dtype,),
            },
        },
        {
            "search_fn": _sfdp_pattern_4,
            "replace_fn": _sfdp_replacement_4,
            "dummy_args": [
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                0.74321,  # scaling
                0.9734,  # dropout
            ],
            "scalar_workaround": {
                "scaling": 0.74321,
                "dropout": 0.9734,
            },
            "op_ignore_types": {
                torch.ops.aten.to.dtype: (torch.dtype,),
            },
        },
        {
            "search_fn": _sfdp_pattern_5,
            "replace_fn": _sfdp_replacement_5,
            "dummy_args": [
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                0.874321,  # scaling
                0.89734,  # dropout
            ],
            "scalar_workaround": {
                "scaling": 0.874321,
                "dropout": 0.89734,
            },
            "op_ignore_types": {
                torch.ops.aten.to.dtype: (torch.dtype,),
            },
        },
        {
            "search_fn": _sfdp_pattern_6,
            "replace_fn": _sfdp_replacement_6,
            "dummy_args": [
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16),
                torch.randn(bs, 1, 1, seq_len, device="cuda", dtype=torch.bfloat16),
                0.634743,  # scaling
                0.6849734,  # dropout
            ],
            "scalar_workaround": {
                "scaling": 0.634743,
                "dropout": 0.6849734,
            },
            "op_ignore_types": {
                torch.ops.aten.to.dtype: (torch.dtype,),
            },
        },
    ]


def match_eager_attention(gm: GraphModule) -> None:
    """Match and replace the eager attention pattern in fx graphs.

    This is replaced with torch.ops.auto_deploy.torch_attention_sdpa.
    """
    graph = gm.graph
    patterns = ADPatternMatcherPass()

    for pattern_config in _get_sfdp_patterns():
        register_ad_pattern(**pattern_config, patterns=patterns)

    # Apply patterns and clean-up
    num_matches = patterns.apply(graph)
    gm = canonicalize_graph(gm)
    ad_logger.info(f"Found and matched {num_matches} Eager Attention patterns")


def _grouped_attn_pattern(q, k, v, n_rep, attn_mask, dropout_p, scale):
    k = torch.ops.auto_deploy.torch_attention_repeat_kv(k, n_rep)
    v = torch.ops.auto_deploy.torch_attention_repeat_kv(v, n_rep)
    return torch.ops.auto_deploy.torch_attention_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def _grouped_attn_replacement(q, k, v, n_rep, attn_mask, dropout_p, scale):
    return torch.ops.auto_deploy.torch_attention_grouped_sdpa.default(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False, scale=scale
    )


def match_grouped_attention(gm: GraphModule) -> None:
    """Match and replace the grouped attention pattern using pattern matcher util."""
    graph = gm.graph
    patterns = ADPatternMatcherPass()

    # Dummy inputs: meta tensors for FX tracing
    bs, seqlen, n_heads, hidden_size = 8, 16, 8, 512
    head_dim = hidden_size // n_heads

    dummy_args = [
        torch.randn(bs, n_heads, seqlen, head_dim, device="meta", dtype=torch.float16),  # q
        torch.randn(bs, 1, seqlen, head_dim, device="meta", dtype=torch.float16),  # k
        torch.randn(bs, 1, seqlen, head_dim, device="meta", dtype=torch.float16),  # v
        7,  # n_rep
        torch.randn(bs, 1, 1, seqlen, device="meta", dtype=torch.float16),  # attn_mask
        0.12345,  # dropout
        0.56789,  # scale
    ]

    scalar_workaround = {"scale": 0.56789, "dropout_p": 0.12345, "n_rep": 7}

    register_ad_pattern(
        search_fn=_grouped_attn_pattern,
        replace_fn=_grouped_attn_replacement,
        patterns=patterns,
        dummy_args=dummy_args,
        op_ignore_types={},
        scalar_workaround=scalar_workaround,
    )

    num_matches = patterns.apply(graph)
    gm = canonicalize_graph(gm)
    ad_logger.info(f"Found and matched {num_matches} Grouped Attention patterns")


def match_causal_attn_mask(gm: GraphModule) -> None:
    """
    Match attention operations with causal attention masks and optimize them.

    For operations that use explicit causal masks, this replaces:
    - sdpa(q, k, v, causal_mask, dropout_p, False, scale)
    with:
    - sdpa(q, k, v, None, dropout_p, True, scale)

    This optimization enables more efficient implementations on supported backends.
    """
    graph = gm.graph

    # Track replacements to avoid processing nodes multiple times
    num_causal_patterns = 0

    # Iterate through nodes in the graph
    for node in list(graph.nodes):
        # Look for SDPA nodes or grouped SDPA nodes
        if not (
            is_op(node, torch.ops.auto_deploy.torch_attention_sdpa)
            or is_op(node, torch.ops.auto_deploy.torch_attention_grouped_sdpa)
        ):
            continue

        # Get the attention mask argument (4th argument)
        if len(node.args) < 4 or node.args[3] is None:
            continue

        attn_mask = node.args[3]

        # Check if this mask is a causal mask
        if not _is_causal_mask(attn_mask):
            ad_logger.debug(f"Found non-causal attention mask at {node=}!")
            continue

        ad_logger.debug(f"Found causal attention mask at {node}")

        # construct the new args list with args provided to the node and the default values otherwise
        new_args = []
        for idx, arg in enumerate(node.target._schema.arguments):
            # In case arg is provided to the node, use it
            if idx < len(node.args):
                new_args.append(node.args[idx])
            # In case arg is not provided to the node, use the default value
            elif arg.has_default_value:
                new_args.append(arg.default_value)
            else:
                raise ValueError(f"Missing required argument: {arg.name}")

        # Create new arguments with None mask and is_causal=True
        new_args[3] = None  # Set mask to None
        new_args[5] = True  # Set is_causal to True

        # Create new node with updated arguments
        with graph.inserting_before(node):
            new_node = graph.call_function(node.target, args=tuple(new_args), kwargs=node.kwargs)

        # Preserve metadata
        new_node.meta = node.meta.copy()

        # Replace the old node with the new one
        node.replace_all_uses_with(new_node)

        num_causal_patterns += 1

    # Clean up the graph if we made any replacements
    if num_causal_patterns:
        canonicalize_graph(gm)
    ad_logger.info(f"Found {num_causal_patterns} causal mask attention patterns")


def _is_causal_mask(mask_node: Node) -> bool:
    """
    Determine if a node represents a causal attention mask.

    Causal masks typically involve:
    1. Creating a matrix with very negative values (e.g., -inf or close to it)
    2. Using triu with offset 1 to create an upper triangular matrix
    3. Usually involves comparison operations (gt, lt) with position indices

    Returns True if the node appears to be a causal mask pattern.
    """
    # Direct pattern from the test case: masked_fill with triu(ones,1) and -inf
    if is_op(mask_node, torch.ops.aten.masked_fill):
        mask_args = mask_node.args
        if len(mask_args) >= 2:
            _ = mask_args[0]  # zero tensor
            mask_tensor = mask_args[1]
            fill_value = mask_args[2] if len(mask_args) > 2 else mask_node.kwargs.get("value", None)

            # Check if fill value is very negative (e.g., -inf)
            if fill_value is not None and (
                fill_value == float("-inf")
                or (isinstance(fill_value, (int, float)) and fill_value < -1e4)
            ):
                # Try to trace back to find a triu pattern
                if _has_triu_ancestor(mask_tensor, offset=1):
                    return True

    # Pattern from negative_fill test case: masked_fill with ~triu(ones,1) and 0.0
    # The negative_fill pattern has a pre-filled tensor with very negative values
    # and zeros in the lower triangle
    if is_op(mask_node, torch.ops.aten.masked_fill):
        mask_args = mask_node.args
        if len(mask_args) >= 2:
            negative_tensor = mask_args[0]
            mask_tensor = mask_args[1]
            fill_value = mask_args[2] if len(mask_args) > 2 else mask_node.kwargs.get("value", None)

            # Check if fill value is zero and the tensor is pre-filled with negative values
            if fill_value == 0.0 or fill_value == 0:
                # Check for the full tensor with negative values
                if is_op(negative_tensor, torch.ops.aten.full):
                    fill_args = negative_tensor.args
                    if (
                        len(fill_args) > 1
                        and isinstance(fill_args[1], (int, float))
                        and fill_args[1] < -1e4
                    ):
                        # This is likely a negative-filled tensor
                        # Now check if the mask is a bitwise_not of triu
                        if is_op(mask_tensor, torch.ops.aten.bitwise_not):
                            if len(mask_tensor.args) > 0 and _has_triu_ancestor(
                                mask_tensor.args[0], offset=1
                            ):
                                return True

    # Pattern for llama-3.1 style causal mask: slice of expand(unsqueeze(unsqueeze(mul_(triu, gt))))
    if is_op(mask_node, torch.ops.aten.slice):
        # Follow the chain backward to the source of the slice
        if len(mask_node.args) == 0:
            return False
        slice_source = mask_node.args[0]

        # Check for typical expand pattern
        if not (slice_source and is_op(slice_source, torch.ops.aten.expand)):
            return False

        # Continue tracing back through the pattern
        if len(slice_source.args) == 0:
            return False
        expand_source = slice_source.args[0]

        # Check for first unsqueeze operation
        if not (expand_source and is_op(expand_source, torch.ops.aten.unsqueeze)):
            return False

        # Look for the source of first unsqueeze
        if len(expand_source.args) == 0:
            return False
        first_unsqueeze_source = expand_source.args[0]

        # Check for second unsqueeze operation
        if not (first_unsqueeze_source and is_op(first_unsqueeze_source, torch.ops.aten.unsqueeze)):
            return False

        # Look for the source of the second unsqueeze
        if len(first_unsqueeze_source.args) == 0:
            return False
        second_unsqueeze_source = first_unsqueeze_source.args[0]

        # Check for mul_ operation
        if is_op(second_unsqueeze_source, torch.ops.aten.mul_):
            # Check if one of the mul_ arguments is a triu operation
            has_triu = False
            for arg in second_unsqueeze_source.args:
                if is_op(arg, torch.ops.aten.triu):
                    if len(arg.args) > 1 and arg.args[1] == 1:
                        has_triu = True
                        break

            if has_triu:
                # Check if one of the mul_ arguments involves a full tensor with negative values
                for arg in second_unsqueeze_source.args:
                    if is_op(arg, torch.ops.aten.full):
                        if (
                            len(arg.args) > 1
                            and isinstance(arg.args[1], (int, float))
                            and arg.args[1] < -1e4
                        ):
                            return True

            return has_triu

    # Original implementation for backward compatibility
    if is_op(mask_node, torch.ops.aten.slice):
        # Follow the chain backward to the source of the slice
        if len(mask_node.args) == 0:
            return False
        slice_source = mask_node.args[0]

        # Check for typical expand pattern
        if not (slice_source and is_op(slice_source, torch.ops.aten.expand)):
            return False

        # Continue tracing back through the pattern
        if len(slice_source.args) == 0:
            return False
        expand_source = slice_source.args[0]

        # Check for unsqueeze operations
        if not (expand_source and is_op(expand_source, torch.ops.aten.unsqueeze)):
            return False

        # Look for the source of the unsqueeze
        if len(expand_source.args) == 0:
            return False
        unsqueeze_source = expand_source.args[0]

        if not unsqueeze_source:
            return False

        # Check for triu pattern which is common in causal masks
        if is_op(unsqueeze_source, torch.ops.aten.mul_):
            for arg in unsqueeze_source.args:
                if not is_op(arg, torch.ops.aten.triu):
                    continue

                if len(arg.args) <= 1:
                    continue

                triu_offset = arg.args[1]
                # Causal masks typically use triu with offset 1
                if triu_offset == 1:
                    return True

            return False

        # Check if we have a full tensor filled with a very negative number
        if not is_op(unsqueeze_source, torch.ops.aten.full):
            return False

        if len(unsqueeze_source.args) <= 1:
            return False

        fill_value = unsqueeze_source.args[1]
        # Check if the fill value is very negative (likely -inf or close)
        if isinstance(fill_value, float) and fill_value < -1e10:
            return True

    # If we can't definitively identify it as causal, return False
    return False


def _has_triu_ancestor(node: Node, offset: int = 1, depth: int = 0, max_depth: int = 5) -> bool:
    """Helper function to find a triu operation in the ancestry of a node."""
    if depth > max_depth:  # Prevent infinite recursion
        return False

    if is_op(node, torch.ops.aten.triu):
        if len(node.args) > 1 and node.args[1] == offset:
            return True

    # Check if any of the arguments has a triu ancestor
    for arg in node.args:
        if isinstance(arg, Node) and _has_triu_ancestor(arg, offset, depth + 1, max_depth):
            return True

    # Check if any of the kwargs has a triu ancestor
    for value in node.kwargs.values():
        if isinstance(value, Node) and _has_triu_ancestor(value, offset, depth + 1, max_depth):
            return True

    return False


def match_attention_layout(gm: GraphModule, attention_op: Type[AttentionDescriptor]) -> None:
    """
    Match and transform attention operations to match the layout expected by the attention backend.

    If the attention backend expects 'bnsd' layout (batch, num_heads, seq_len, head_dim), which
    is the default for SDPA operations, we don't need to transform anything.

    If the backend expects 'bsnd' layout (batch, seq_len, num_heads, head_dim), we insert
    appropriate transposes before and after SDPA operations and replace them with bsnd_grouped_sdpa.
    """
    # Get attention layout from attention_op
    attention_layout = attention_op.get_attention_layout()

    # List of SDPA operations to look for
    sdpa_ops = {
        torch.ops.auto_deploy.torch_attention_sdpa,
        torch.ops.auto_deploy.torch_attention_grouped_sdpa,
    }

    graph = gm.graph
    num_bsnd_patterns = 0

    # Look for SDPA operations
    for sdpa_node in list(graph.nodes):
        if sdpa_node.op != "call_function" or not is_op(sdpa_node, sdpa_ops):
            continue

        ad_logger.debug(f"Found SDPA node to transform for bsnd layout: {sdpa_node}")

        # Extract q, k, v inputs
        q, k, v = sdpa_node.args[:3]

        # Check if we need to transpose the inputs
        if attention_layout == "bsnd":
            # Add transposes before the node (from bnsd to bsnd)
            with graph.inserting_before(sdpa_node):
                q_updated = graph.call_function(torch.ops.aten.transpose.int, args=(q, 1, 2))
                k_updated = graph.call_function(torch.ops.aten.transpose.int, args=(k, 1, 2))
                v_updated = graph.call_function(torch.ops.aten.transpose.int, args=(v, 1, 2))

            # Preserve fake tensor in meta["val"] for the transposed inputs
            q_updated.meta["val"] = q.meta["val"].transpose(1, 2)
            k_updated.meta["val"] = k.meta["val"].transpose(1, 2)
            v_updated.meta["val"] = v.meta["val"].transpose(1, 2)
        elif attention_layout == "bnsd":
            # we don't need to do anything...
            q_updated = q
            k_updated = k
            v_updated = v
        else:
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        # Create bsnd_grouped_sdpa node with the same args as the original node
        # but using the transposed inputs
        with graph.inserting_before(sdpa_node):
            source_sdpa_node = graph.call_function(
                attention_op.get_source_attention_op(),
                args=(q_updated, k_updated, v_updated) + sdpa_node.args[3:],
                kwargs=sdpa_node.kwargs,
            )

        # Check if need to update the output node to match the layout
        if attention_layout == "bsnd":
            # Add transpose for the output (from bsnd back to bnsd)
            with graph.inserting_after(source_sdpa_node):
                output_updated = graph.call_function(
                    torch.ops.aten.transpose.int, args=(source_sdpa_node, 1, 2)
                )

            # Preserve fake tensor in meta["val"] for the transposed inputs
            source_sdpa_node.meta["val"] = sdpa_node.meta["val"].transpose(1, 2).contiguous()
            output_updated.meta["val"] = source_sdpa_node.meta["val"].transpose(1, 2)
        elif attention_layout == "bnsd":
            output_updated = source_sdpa_node
        else:
            raise ValueError(f"Unsupported attention layout: {attention_layout}")

        # Replace the old node with the transposed output
        sdpa_node.replace_all_uses_with(output_updated)

        num_bsnd_patterns += 1

    # Clean up the graph if we made any replacements
    if num_bsnd_patterns:
        canonicalize_graph(gm)
        ad_logger.debug(f"Transformed graph for bsnd layout: {gm}")

    ad_logger.info(f"Found and matched {num_bsnd_patterns} attention layouts")
