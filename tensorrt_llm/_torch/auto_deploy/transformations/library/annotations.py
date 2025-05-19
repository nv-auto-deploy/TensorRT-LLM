"""Contains transformations that add perf related (nvtx) annotations to the graph."""

import torch
from torch.fx import GraphModule

from ...custom_ops.nvtx import *  # noqa
from ...utils.logger import ad_logger


def add_nvtx_annotations(gm: GraphModule, module_name: str) -> GraphModule:
    """Add NVTX profiling ranges around linear operations in the graph.

    This transformation:
    1. Traverses the graph looking for linear operations
    2. For each linear operation, adds:
       - start_range before the operation
       - end_range after the operation
    3. Uses the node's name as the range identifier

    Args:
        gm: The graph module to transform

    Returns:
        The transformed graph module with NVTX annotations
    """
    ad_logger.info("Adding NVTX annotations around linear operations")
    ad_logger.debug("Before adding NVTX annotations: " + str(gm))

    graph = gm.graph

    ad_logger.info(f"Graph nodes: {graph.nodes}")
    # Insert NVTX markers in every node
    idx = 0
    for node in graph.nodes:
        idx += 1
        if node.meta.get("nn_module_stack"):
            range_name = list(node.meta["nn_module_stack"].keys())[-1]
            ad_logger.info(f"Range name: {range_name}")
            # Add start_range before the node
            with graph.inserting_before(node):
                start_node = graph.call_function(
                    torch.ops.nvtx_ops.start_range, args=(f"{range_name}_{idx}",), kwargs={}
                )
            # Add end_range after the node
            with graph.inserting_after(node):
                end_node = graph.call_function(
                    torch.ops.nvtx_ops.end_range, args=(f"{range_name}_{idx}",), kwargs={}
                )
        else:
            range_name = f"{node}_{idx}"
            if "start_range" in node.name or "end_range" in node.name or "output" in node.name:
                continue
            # Add start_range before the node
            with graph.inserting_before(node):
                start_node = graph.call_function(
                    torch.ops.nvtx_ops.start_range, args=(f"{range_name}_{idx}",), kwargs={}
                )
            # Add end_range after the node
            with graph.inserting_after(node):
                end_node = graph.call_function(
                    torch.ops.nvtx_ops.end_range, args=(f"{range_name}_{idx}",), kwargs={}
                )
            ad_logger.info(f"Node {node} does not have nn_module_stack")
    ad_logger.info(f"Graph nodes after inserting NVTX markers: {graph.nodes}")

    # Preserve metadata
    start_node.meta = start_node.meta.copy()
    end_node.meta = end_node.meta.copy()
    # Clean up the graph
    gm.recompile()
    ad_logger.debug("After adding NVTX annotations: " + str(gm))

    return gm
