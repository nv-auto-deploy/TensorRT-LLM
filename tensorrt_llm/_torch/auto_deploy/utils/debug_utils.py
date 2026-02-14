# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug visualization utilities for torch.fx graph modules.

Two-pass approach:
  1. Render the full graph with ``FxGraphDrawer`` (coloured by node type).
     Graphviz ``dot`` handles the entire layout — all edges point downward.
  2. Parse the resulting SVG, compute per-layer bounding boxes from actual
     node positions, and insert semi-transparent layer rectangles + labels.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple

from torch.fx import GraphModule, Node

from .node_utils import (
    LayerSubgraph,
    is_any_attention_op,
    is_any_delta_op,
    is_any_lin_op,
    is_any_moe_op,
    is_any_ssm_op,
    is_weight_node,
)

# ── Colour palettes ──────────────────────────────────────────────────────────

# (border_color, fill_color) for each node category.
_NODE_COLORS: Dict[str, Tuple[str, str]] = {
    "linear": ("#1565C0", "#BBDEFB"),
    "attention": ("#E65100", "#FFE0B2"),
    "ssm": ("#2E7D32", "#C8E6C9"),
    "moe": ("#6A1B9A", "#E1BEE7"),
    "delta": ("#C62828", "#FFCDD2"),
    "weight": ("#F9A825", "#FFF9C4"),
    "residual": ("#AD1457", "#F8BBD0"),
    "default": (None, None),  # keep FxGraphDrawer default
}

# (border_color, fill_color) for layer cluster rectangles.
_LAYER_COLORS: Dict[str, Tuple[str, str]] = {
    "attention": ("#E65100", "#FFF3E0"),
    "ssm": ("#2E7D32", "#E8F5E9"),
    "mlp": ("#1565C0", "#E3F2FD"),
    "moe": ("#6A1B9A", "#F3E5F5"),
    "mla": ("#00838F", "#E0F7FA"),
    "delta": ("#C62828", "#FFEBEE"),
    "unknown": ("#616161", "#FAFAFA"),
}

_LAYER_PAD = 20  # padding around nodes inside a layer rectangle


# ── Helpers ──────────────────────────────────────────────────────────────────


def _classify_node(node: Node, residuals_set: Set[Node]) -> str:
    """Return the category key for *node*."""
    if is_any_lin_op(node):
        return "linear"
    if is_any_attention_op(node):
        return "attention"
    if is_any_ssm_op(node):
        return "ssm"
    if is_any_moe_op(node):
        return "moe"
    if is_any_delta_op(node):
        return "delta"
    if is_weight_node(node):
        return "weight"
    if node in residuals_set:
        return "residual"
    return "default"


def _extract_bbox(
    g_elem: ET.Element,
    ns: str,
) -> Optional[Tuple[float, float, float, float]]:
    """Extract the bounding box ``(x0, y0, x1, y1)`` from a graphviz SVG node ``<g>``."""
    # Polygon — used for box / record shapes.
    polygon = g_elem.find(f"{{{ns}}}polygon")
    if polygon is not None:
        pts = polygon.get("points", "")
        coords = []
        for p in pts.strip().split():
            xy = p.split(",")
            if len(xy) == 2:
                coords.append((float(xy[0]), float(xy[1])))
        if coords:
            return (
                min(x for x, _ in coords),
                min(y for _, y in coords),
                max(x for x, _ in coords),
                max(y for _, y in coords),
            )

    # Path — used for rounded-box shapes.
    path = g_elem.find(f"{{{ns}}}path")
    if path is not None:
        nums = re.findall(r"[-+]?\d*\.?\d+", path.get("d", ""))
        if len(nums) >= 4:
            pairs = list(zip(nums[0::2], nums[1::2]))
            xs = [float(a) for a, _ in pairs]
            ys = [float(b) for _, b in pairs]
            if xs and ys:
                return (min(xs), min(ys), max(xs), max(ys))

    # Ellipse — used for placeholder / output shapes.
    ellipse = g_elem.find(f"{{{ns}}}ellipse")
    if ellipse is not None:
        cx = float(ellipse.get("cx", "0"))
        cy = float(ellipse.get("cy", "0"))
        rx = float(ellipse.get("rx", "0"))
        ry = float(ellipse.get("ry", "0"))
        return (cx - rx, cy - ry, cx + rx, cy + ry)

    return None


# ── Public API ───────────────────────────────────────────────────────────────


def draw_layered_graph(
    gm: GraphModule,
    layer_subgraphs: List[LayerSubgraph],
    unprocessed_linear_nodes: Set[Node],
    residuals: List[Node],
    filename: str,
) -> None:
    """Draw a layered graph of the ``GraphModule`` and save it as SVG.

    **Step 1** — Render the full graph using :class:`FxGraphDrawer` (pydot /
    graphviz ``dot``).  Before rendering, node fill-colours are overridden
    based on their semantic role (linear → blue, attention → orange, …).

    **Step 2** — Parse the SVG, look up each node's position, compute a
    bounding box per :class:`LayerSubgraph`, and insert coloured rectangles
    (with a label) behind the nodes.

    Args:
        gm: The :class:`GraphModule` to visualise.
        layer_subgraphs: Layer clusters (topologically sorted).
        unprocessed_linear_nodes: Linear nodes not in any layer.
        residuals: Residual-add nodes (rendered in pink).
        filename: Stem — ``<filename>.svg`` is written.
    """
    from torch.fx.passes.graph_drawer import FxGraphDrawer

    residuals_set: Set[Node] = set(residuals) if residuals else set()
    fx_nodes: Dict[str, Node] = {n.name: n for n in gm.graph.nodes}

    # ── Step 1: render with FxGraphDrawer, override colours ──────────────
    drawer = FxGraphDrawer(gm, filename)
    dot_graph = drawer.get_dot_graph()

    for dot_node in dot_graph.get_nodes():
        name = dot_node.get_name().strip('"')
        if name not in fx_nodes:
            continue
        cat = _classify_node(fx_nodes[name], residuals_set)
        border, fill = _NODE_COLORS[cat]
        if fill is not None:
            dot_node.set_fillcolor(fill)
        if border is not None:
            dot_node.set_color(border)
            dot_node.set_penwidth(2)

    svg_bytes: bytes = dot_graph.create_svg()

    # ── Step 2: parse SVG, add layer rectangles ──────────────────────────
    SVG_NS = "http://www.w3.org/2000/svg"
    ET.register_namespace("", SVG_NS)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    root = ET.fromstring(svg_bytes)

    # Locate the main graph <g> (graphviz gives it id="graph0" or similar).
    graph_g: Optional[ET.Element] = None
    for g in root.iter(f"{{{SVG_NS}}}g"):
        cls = g.get("class", "")
        if cls == "graph" or g.get("id", "").startswith("graph"):
            graph_g = g
            break

    if graph_g is None:
        # Cannot locate graph group — just write the raw SVG.
        with open(f"{filename}.svg", "wb") as f:
            f.write(svg_bytes)
        print(f"[draw_layered_graph] Saved {filename}.svg (no layer overlays)")
        return

    # Map node name → bounding box by inspecting child <g> elements.
    node_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    for child in graph_g:
        if child.tag != f"{{{SVG_NS}}}g":
            continue
        title_el = child.find(f"{{{SVG_NS}}}title")
        if title_el is None or title_el.text not in fx_nodes:
            continue
        bbox = _extract_bbox(child, SVG_NS)
        if bbox is not None:
            node_bboxes[title_el.text] = bbox

    # Find the insertion index: right after <title> and background <polygon>
    # so that layer rectangles sit behind every node and edge.
    insert_idx = 0
    for i, child in enumerate(list(graph_g)):
        if child.tag in (f"{{{SVG_NS}}}title", f"{{{SVG_NS}}}polygon"):
            insert_idx = i + 1
        else:
            break

    # Insert layer rectangles + labels (iterate in reverse so indices stay
    # valid when inserting at the same position).
    for i, ls in enumerate(layer_subgraphs):
        all_layer_nodes = list(ls.opening_nodes) + list(ls.subgraph_nodes)
        # remove nodes that do not contain "model_layers" in their name
        all_layer_nodes = [n for n in all_layer_nodes if "model_layers" in n.name]
        if ls.terminating_node is not None:
            all_layer_nodes.append(ls.terminating_node)

        boxes = [node_bboxes[n.name] for n in all_layer_nodes if n.name in node_bboxes]
        # if not boxes:
        #     continue

        x0 = min(b[0] for b in boxes) - _LAYER_PAD
        y0 = min(b[1] for b in boxes) - _LAYER_PAD - 18  # room for label
        x1 = max(b[2] for b in boxes) + _LAYER_PAD
        y1 = max(b[3] for b in boxes) + _LAYER_PAD

        lt = ls.layer_type.value
        border, fill = _LAYER_COLORS.get(lt, ("#616161", "#FAFAFA"))

        rect = ET.Element(
            f"{{{SVG_NS}}}rect",
            {
                "x": f"{x0:.1f}",
                "y": f"{y0:.1f}",
                "width": f"{x1 - x0:.1f}",
                "height": f"{y1 - y0:.1f}",
                "rx": "10",
                "fill": fill,
                "stroke": border,
                "stroke-width": "2",
                "opacity": "0.55",
            },
        )
        graph_g.insert(insert_idx, rect)

        label = ET.Element(
            f"{{{SVG_NS}}}text",
            {
                "x": f"{x0 + 8:.1f}",
                "y": f"{y0 + 14:.1f}",
                "font-family": "Helvetica, Arial, sans-serif",
                "font-size": "12",
                "font-weight": "bold",
                "fill": border,
            },
        )
        label.text = f"Layer: {lt.upper()} {i}"
        graph_g.insert(insert_idx + 1, label)

    # ── Write final SVG ──────────────────────────────────────────────────
    svg_path = f"{filename}.svg"
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(svg_path, encoding="unicode", xml_declaration=True)
    print(f"[draw_layered_graph] Saved {svg_path}")
