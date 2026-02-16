# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug visualization utilities for torch.fx graph modules.

Two-pass approach:
  1. Render the full graph with ``FxGraphDrawer`` (coloured by node type).
     Graphviz ``dot`` handles the entire layout — all edges point downward.
  2. Parse the resulting SVG, compute per-layer bounding boxes from actual
     node positions, and insert semi-transparent layer rectangles + labels.
"""

import inspect
import re
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Set, Tuple

from torch.fx import GraphModule, Node

from .node_utils import (
    LayerSubgraph,
    has_shape,
    is_any_attention_op,
    is_any_delta_op,
    is_any_lin_op,
    is_any_moe_op,
    is_any_ssm_op,
    is_weight_node,
    shape,
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


def _short_target_sig(node: Node) -> str:
    """Build a concise ``name(param, param, ...)`` string for *call_function* nodes.

    For ``OpOverload`` targets the ``_schema`` is parsed to extract parameter
    names and optional/default markers.  For regular callables
    ``inspect.signature`` is used as a fallback.  Non-``call_function`` nodes
    get a plain short target name.
    """
    target = node.target
    if node.op != "call_function":
        if isinstance(target, str):
            return target
        name = getattr(target, "__name__", None)
        return name if name else str(target)

    # Determine the function name (stripped of namespace).
    name = getattr(target, "__name__", None) or str(target)
    for prefix in ("torch.ops.auto_deploy.", "torch.ops.aten.", "torch.ops."):
        if name.startswith(prefix):
            name = name[len(prefix) :]

    # remove ".default" from the name
    name = name.replace(".default", "")

    # Try _schema first (torch OpOverload).
    if hasattr(target, "_schema"):
        schema = str(target._schema)
        # Schema looks like: "ns::name(Type param, Type param=default, ...) -> RetType"
        m = re.search(r"\(([^)]*)\)", schema)
        if m:
            raw_params = m.group(1)
            params: list = []
            for p in raw_params.split(","):
                p = p.strip()
                if not p:
                    continue
                # "Type name" or "Type name=default" or "*, ..."
                parts = p.split()
                if len(parts) >= 2:
                    pname = parts[-1]
                    if "=" in pname:
                        pname = pname.split("=")[0] + "[opt]"
                    params.append(pname)
                elif parts:
                    params.append(parts[0])

            # add "non-node" args
            nonparametric_args = [p for p in node.args if p not in node.all_input_nodes]
            for i, p in enumerate(nonparametric_args):
                if isinstance(p, Iterable):
                    nonparametric_args[i] = [pp if not isinstance(pp, Node) else "p" for pp in p]
            nonparametric_args = [str(p) for p in nonparametric_args]
            return f"{name}({', '.join(params)}, {', '.join(nonparametric_args)})"

    # Fallback: inspect.signature.
    try:
        sig = inspect.signature(target)
        params = []
        for pname, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                params.append(f"{pname}[opt]")
            elif param.annotation is not inspect.Parameter.empty:
                ann = getattr(param.annotation, "__name__", str(param.annotation))
                if "Optional" in ann:
                    params.append(f"{pname}[opt]")
                else:
                    params.append(pname)
            else:
                params.append(pname)
        return f"{name}({', '.join(params)})"
    except (ValueError, TypeError):
        return name


def _node_shape_str(node: Node) -> str:
    """Return a compact shape string like ``[2, 1024, 4096]``."""
    if has_shape(node):
        s = shape(node)
        if s is not None:
            return str(list(s))
    return ""


def _node_dtype_str(node: Node) -> str:
    """Return the dtype string (without ``torch.`` prefix)."""
    val = getattr(node, "meta", {}).get("val", None)
    if val is not None and hasattr(val, "dtype"):
        return str(val.dtype).replace("torch.", "")
    return ""


def _record_escape(text: str) -> str:
    """Escape characters special in graphviz record labels."""
    return (
        text.replace("{", r"\{")
        .replace("}", r"\}")
        .replace("|", r"\|")
        .replace("<", r"\<")
        .replace(">", r"\>")
    )


def _custom_label(node: Node) -> str:
    """Build a graphviz record label showing name, target sig, shape, dtype."""
    display_name = node.name
    if display_name.startswith("model_"):
        display_name = display_name[len("model_") :]
    parts = [f"name={_record_escape(display_name)}"]
    target_sig = _record_escape(_short_target_sig(node))
    if target_sig:
        parts.append(f"target={target_sig}")
    shape_s = _record_escape(_node_shape_str(node))
    if shape_s:
        parts.append(f"shape={shape_s}")
    dtype_s = _record_escape(_node_dtype_str(node))
    if dtype_s:
        parts.append(f"dtype={dtype_s}")
    return "{" + "|".join(parts) + "}"


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
    skip_aux_nodes: bool = True,
) -> None:
    """Draw a layered graph of the ``GraphModule`` and save it as SVG.

    **Step 1** — Render the full graph using :class:`FxGraphDrawer` (pydot /
    graphviz ``dot``).  Before rendering, node fill-colours are overridden
    based on their semantic role (linear → blue, attention → orange, …).
    Auxiliary nodes (e.g., ``sym_size``) are optionally removed for clarity.

    **Step 2** — Parse the SVG, look up each node's position, compute a
    bounding box per :class:`LayerSubgraph`, and insert coloured rectangles
    (with a label) behind the nodes.

    Args:
        gm: The :class:`GraphModule` to visualise.
        layer_subgraphs: Layer clusters (topologically sorted).
        unprocessed_linear_nodes: Linear nodes not in any layer.
        residuals: Residual-add nodes (rendered in pink).
        filename: Stem — ``<filename>.svg`` is written.
        skip_aux_nodes: If ``True``, removes auxiliary nodes (e.g., ``sym_size``)
            from the graph for a cleaner diagram.
    """
    from torch.fx.passes.graph_drawer import FxGraphDrawer

    residuals_set: Set[Node] = set(residuals) if residuals else set()
    fx_nodes: Dict[str, Node] = {n.name: n for n in gm.graph.nodes}

    # ── Step 1: render with FxGraphDrawer, override colours ──────────────
    drawer = FxGraphDrawer(gm, filename)
    dot_graph = drawer.get_dot_graph()

    # Remove auxiliary nodes if requested.
    aux_node_names: Set[str] = set()
    if skip_aux_nodes:
        # Identify nodes to delete
        nodes_to_delete = []
        for dot_node in dot_graph.get_nodes():
            name = dot_node.get_name().strip('"')
            if "sym_size" in name:
                aux_node_names.add(name)
                nodes_to_delete.append(dot_node)

        # Delete edges connected to these nodes (both incoming and outgoing)
        edges_to_delete = []
        for edge in dot_graph.get_edges():
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')
            if src in aux_node_names or dst in aux_node_names:
                edges_to_delete.append(edge)

        for edge in edges_to_delete:
            dot_graph.del_edge(edge)
            dot_graph.del_edge(edge.get_source(), edge.get_destination())

        # Now delete the nodes themselves
        for dot_node in nodes_to_delete:
            dot_graph.del_node(dot_node)

    for dot_node in dot_graph.get_nodes():
        name = dot_node.get_name().strip('"')
        if name not in fx_nodes or name in aux_node_names:
            continue
        fx_node = fx_nodes[name]

        # Override label: name, target(params), shape, dtype.
        dot_node.set_label(_custom_label(fx_node))

        # Override colours by node category.
        cat = _classify_node(fx_node, residuals_set)
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
        # remove nodes that do not contain "_layers" in their name
        all_layer_nodes = [n for n in all_layer_nodes if "_layers" in n.name]
        # remove nodes from rotary embedding cache
        all_layer_nodes = [n for n in all_layer_nodes if "rotary_emb" not in n.name]
        all_layer_nodes = [n for n in all_layer_nodes if "attn_index" not in n.name]

        if ls.terminating_nodes is not None:
            all_layer_nodes.extend(ls.terminating_nodes)

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
