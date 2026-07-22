# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ``fuse_mxfp4_moe`` POST_LOAD_FUSION transform.

Pins the post-quantize / post-load contract of :class:`FuseMXFP4Moe`:

* After running, raw HF MXFP4 buffers on the experts module
  (``gate_up_proj_{blocks,scales,bias}`` / ``down_proj_{blocks,scales,bias}``)
  are deleted and replaced by the six kernel-layout ``*_trtllm`` params
  produced by :func:`prepare_trtllm_gen_moe_mxfp4_weights`.
* The ``trtllm_quant_mxfp4_trtllm_gen_moe_fused`` op's weight/bias arg slots
  are re-pointed at the new prepared get_attr nodes — the op is again
  runnable.
* When ``moe_tp_size > 1``, the prepared fc2 bias is divided by
  ``moe_tp_size`` so that the post-AR sum reproduces the unsharded bias.

Also covers ``quantize_mxfp4_moe`` (backend dispatch, dense-MoE rewrite,
DSV4 checkpoint-layout load hooks) and ``fuse_moe_routing_localization``.
"""

import operator
from types import SimpleNamespace
from typing import Tuple

import pytest
import torch
import torch.nn as nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 (op registration)
import tensorrt_llm._torch.auto_deploy.transform.library.fused_moe_mxfp4 as mxfp4_transform_mod
from tensorrt_llm._torch.auto_deploy._compat import is_sm_100f
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.prepare_trtllm_gen_moe_mxfp4_weights import (
    prepare_trtllm_gen_moe_mxfp4_weights,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    build_deepseek_v4_packed_mxfp4_experts_layout,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig, TransformRegistry
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig

# The transform calls ``prepare_trtllm_gen_moe_mxfp4_weights`` which itself
# invokes ``torch.ops.trtllm.shuffle_matrix`` — registered CUDA-only.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fuse_mxfp4_moe runs prepare_trtllm_gen_moe_mxfp4_weights which is CUDA-only",
)


# Small shapes that still respect the MXFP4 block size (32) and the kernel
# weight alignment (128) so prep runs without padding surprises.
E = 4
H = 128
I = 128  # noqa: E741


def _make_raw_mxfp4_tensors(device: str = "cuda") -> Tuple[torch.Tensor, ...]:
    """Build a deterministic raw-HF-layout MXFP4 expert set on ``device``."""
    g = torch.Generator(device="cpu").manual_seed(0)
    gu_blocks = torch.randint(0, 256, (E, 2 * I, H // 32, 16), dtype=torch.uint8, generator=g).to(
        device
    )
    gu_scales = torch.randint(126, 130, (E, 2 * I, H // 32), dtype=torch.uint8, generator=g).to(
        device
    )
    gu_bias = (torch.randn(E, 2 * I, dtype=torch.bfloat16, generator=g) * 0.01).to(device)
    dn_blocks = torch.randint(0, 256, (E, H, I // 32, 16), dtype=torch.uint8, generator=g).to(
        device
    )
    dn_scales = torch.randint(126, 130, (E, H, I // 32), dtype=torch.uint8, generator=g).to(device)
    dn_bias = (torch.randn(E, H, dtype=torch.bfloat16, generator=g) * 0.01).to(device)
    return gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias


def _build_pre_fuse_gm(raw_tensors: Tuple[torch.Tensor, ...]) -> torch.fx.GraphModule:
    """Build a tiny GM in the exact pre-``FuseMXFP4Moe`` shape ``QuantizeMXFP4MOE`` leaves.

    Shape mirrors ``_apply_trtllm``'s output:
      * root has an ``experts`` submodule with the six raw HF MXFP4 params
        and the three SwiGLU constant params.
      * Graph: ``(hidden, router_w, router_b) -> trtllm_quant_mxfp4_trtllm_gen_moe_fused``
        whose weight/bias args are get_attrs pointing at the raw experts params.
    """
    gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias = raw_tensors

    root = nn.Module()
    root.experts = nn.Module()
    raw_specs = [
        ("gate_up_proj_blocks", gu_blocks),
        ("gate_up_proj_scales", gu_scales),
        ("gate_up_proj_bias", gu_bias),
        ("down_proj_blocks", dn_blocks),
        ("down_proj_scales", dn_scales),
        ("down_proj_bias", dn_bias),
    ]
    for name, t in raw_specs:
        root.experts.register_parameter(name, nn.Parameter(t.clone(), requires_grad=False))

    # SwiGLU constants (gpt-oss defaults). Must live on the experts module
    # because their get_attr nodes are inserted with that path.
    a = torch.full((E,), 1.702, dtype=torch.float32, device=gu_blocks.device)
    b = torch.full((E,), 1.0, dtype=torch.float32, device=gu_blocks.device)
    c = torch.full((E,), 7.0, dtype=torch.float32, device=gu_blocks.device)
    root.experts.register_parameter("swiglu_alpha_trtllm", nn.Parameter(a, requires_grad=False))
    root.experts.register_parameter("swiglu_beta_trtllm", nn.Parameter(b, requires_grad=False))
    root.experts.register_parameter("swiglu_limit_trtllm", nn.Parameter(c, requires_grad=False))

    graph = torch.fx.Graph()
    hidden = graph.placeholder("hidden")
    router_w = graph.placeholder("router_w")
    router_b = graph.placeholder("router_b")

    gu_blocks_n = graph.get_attr("experts.gate_up_proj_blocks")
    dn_blocks_n = graph.get_attr("experts.down_proj_blocks")
    gu_scales_n = graph.get_attr("experts.gate_up_proj_scales")
    dn_scales_n = graph.get_attr("experts.down_proj_scales")
    gu_bias_n = graph.get_attr("experts.gate_up_proj_bias")
    dn_bias_n = graph.get_attr("experts.down_proj_bias")
    sa_n = graph.get_attr("experts.swiglu_alpha_trtllm")
    sb_n = graph.get_attr("experts.swiglu_beta_trtllm")
    sl_n = graph.get_attr("experts.swiglu_limit_trtllm")

    moe = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_mxfp4_trtllm_gen_moe_fused.default,
        args=(
            hidden,
            router_w,
            router_b,
            2,  # top_k
            gu_blocks_n,
            dn_blocks_n,
            gu_scales_n,
            dn_scales_n,
            gu_bias_n,
            dn_bias_n,
            sa_n,
            sb_n,
            sl_n,
            H,  # valid_hidden_size
            I,  # valid_intermediate_size
            "mxfp8",  # act_dtype
            0,  # local_expert_offset
            E,  # num_local_experts
            1,  # routing_method_type = Renormalize
        ),
    )
    graph.output(moe)

    return torch.fx.GraphModule(root, graph)


def _run_fuse(gm: torch.fx.GraphModule, dist_config: DistConfig):
    """Apply just ``FuseMXFP4Moe`` with the given ``dist_config``."""
    shared_config = SharedConfig(
        local_rank=dist_config.rank,
        world_size=dist_config.world_size,
        dist_config=dist_config,
    )
    config_cls = TransformRegistry.get_config_class("fuse_mxfp4_moe")
    config = config_cls(stage="post_load_fusion")
    transform = TransformRegistry.get("fuse_mxfp4_moe")(config)
    return transform._apply(gm, cm=None, factory=None, shared_config=shared_config)


def _moe_node(gm: torch.fx.GraphModule) -> torch.fx.Node:
    target_op = torch.ops.auto_deploy.trtllm_quant_mxfp4_trtllm_gen_moe_fused.default
    nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target_op]
    assert len(nodes) == 1, f"expected exactly one MoE op node, found {len(nodes)}"
    return nodes[0]


# ---------------------------------------------------------------------------
# TP=1 — single-rank: raw → prepared swap, no bias /= moe_tp_size
# ---------------------------------------------------------------------------


def test_fuse_mxfp4_moe_tp1_raw_to_prepared_swap():
    """Single-rank: every raw HF buffer becomes a prepared ``*_trtllm`` buffer; arg slots re-pointed."""
    device = "cuda"
    raw = _make_raw_mxfp4_tensors(device=device)
    gm = _build_pre_fuse_gm(raw)

    dc = DistConfig(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1)
    _, info = _run_fuse(gm, dc)

    # TransformInfo: exactly one MoE node was prepped, not idempotent-skip.
    assert info.skipped is False
    assert info.num_matches == 1

    # Raw HF params are gone.
    raw_names = (
        "gate_up_proj_blocks",
        "gate_up_proj_scales",
        "gate_up_proj_bias",
        "down_proj_blocks",
        "down_proj_scales",
        "down_proj_bias",
    )
    for name in raw_names:
        assert not hasattr(gm.experts, name) or getattr(gm.experts, name, None) is None, (
            f"raw param {name!r} should have been removed"
        )

    # Prepared params are registered (six kinds).
    prepared_names = (
        "fc1_w_trtllm",
        "fc1_w_scale_trtllm",
        "fc1_bias_trtllm",
        "fc2_w_trtllm",
        "fc2_w_scale_trtllm",
        "fc2_bias_trtllm",
    )
    for name in prepared_names:
        assert hasattr(gm.experts, name), f"prepared param {name!r} missing"

    # Op args 4..9 (fc1_w, fc2_w, fc1_s, fc2_s, fc1_b, fc2_b) point at prepared get_attrs.
    n = _moe_node(gm)
    ARG_FC1_W, ARG_FC2_W, ARG_FC1_S, ARG_FC2_S, ARG_FC1_B, ARG_FC2_B = 4, 5, 6, 7, 8, 9
    expected_targets = {
        ARG_FC1_W: "experts.fc1_w_trtllm",
        ARG_FC2_W: "experts.fc2_w_trtllm",
        ARG_FC1_S: "experts.fc1_w_scale_trtllm",
        ARG_FC2_S: "experts.fc2_w_scale_trtllm",
        ARG_FC1_B: "experts.fc1_bias_trtllm",
        ARG_FC2_B: "experts.fc2_bias_trtllm",
    }
    for slot, want in expected_targets.items():
        arg = n.args[slot]
        assert isinstance(arg, torch.fx.Node) and arg.op == "get_attr", (
            f"arg slot {slot} is not a get_attr Node (got {arg!r})"
        )
        assert arg.target == want, f"arg slot {slot} target = {arg.target!r}, want {want!r}"

    # TP=1: fc2_bias matches the raw prep output exactly (no /= moe_tp_size division).
    prep = prepare_trtllm_gen_moe_mxfp4_weights(
        *raw, hidden_size=H, intermediate_size=I, tp_size=1, tp_rank=0
    )
    torch.testing.assert_close(gm.experts.fc2_bias_trtllm.data, prep.fc2_bias_f32, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# TP=2 — fc2 bias must be divided by moe_tp_size so post-AR sum reproduces the unsharded bias
# ---------------------------------------------------------------------------


def test_fuse_mxfp4_moe_tp2_divides_fc2_bias_by_moe_tp_size():
    """``moe_tp_size > 1`` divides only the prepared ``fc2_bias`` by ``moe_tp_size``.

    Other prepared tensors (fc1/fc2 weights, fc1/fc2 scales, fc1 bias) must
    match the TP=1 prep output 1:1 — the transform leaves them alone in the
    scratch path; only ``fc2_bias`` is scaled.
    """
    device = "cuda"
    raw = _make_raw_mxfp4_tensors(device=device)
    gm = _build_pre_fuse_gm(raw)

    moe_tp_size = 2
    dc = DistConfig(
        world_size=2,
        rank=0,
        tp_size=moe_tp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=1,
    )
    _, info = _run_fuse(gm, dc)
    assert info.num_matches == 1

    # Golden: run prep on the SAME raw tensors at tp=1 (the transform path with
    # scratch skips the helper's tp_size > 1 branch and does the division itself).
    prep = prepare_trtllm_gen_moe_mxfp4_weights(
        *raw, hidden_size=H, intermediate_size=I, tp_size=1, tp_rank=0
    )

    # fc2_bias was divided by moe_tp_size; everything else matches 1:1.
    torch.testing.assert_close(
        gm.experts.fc2_bias_trtllm.data, prep.fc2_bias_f32 / moe_tp_size, atol=0, rtol=0
    )
    torch.testing.assert_close(gm.experts.fc1_bias_trtllm.data, prep.fc1_bias_f32, atol=0, rtol=0)
    assert torch.equal(gm.experts.fc1_w_trtllm.data, prep.fc1_weights_mxfp4)
    assert torch.equal(gm.experts.fc2_w_trtllm.data, prep.fc2_weights_mxfp4)
    assert torch.equal(gm.experts.fc1_w_scale_trtllm.data, prep.fc1_weights_scale_ue8m0)
    assert torch.equal(gm.experts.fc2_w_scale_trtllm.data, prep.fc2_weights_scale_ue8m0)


# ---------------------------------------------------------------------------
# Idempotency: re-running on an already-prepped graph is a no-op
# ---------------------------------------------------------------------------


def test_fuse_mxfp4_moe_idempotent_on_already_prepped_graph():
    """Re-running ``FuseMXFP4Moe`` on its own output skips (no double-prep)."""
    device = "cuda"
    raw = _make_raw_mxfp4_tensors(device=device)
    gm = _build_pre_fuse_gm(raw)

    dc = DistConfig(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1)
    _, info1 = _run_fuse(gm, dc)
    assert info1.num_matches == 1

    _, info2 = _run_fuse(gm, dc)
    assert info2.skipped is True, "second run should skip — no raw HF buffers left to prep"
    assert info2.num_matches == 0


# ---------------------------------------------------------------------------
# quantize_mxfp4_moe + fuse_moe_routing_localization
# ---------------------------------------------------------------------------

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm_100f(),
    reason="trtllm-gen MXFP4 runners require CUDA + SM100 (Blackwell)",
)

_FROM_ROUTING = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing.default
_FROM_ROUTING_EP = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing_ep.default
_DSV4_GATE = torch.ops.auto_deploy.deepseek_v4_routing.default
_DSV4_GATE_LOCALIZED = torch.ops.auto_deploy.deepseek_v4_routing_localized.default

_EXPERT_BUFFER_NAMES = (
    "gate_up_blocks",
    "gate_up_bias",
    "gate_up_scales",
    "down_blocks",
    "down_bias",
    "down_scales",
)


class _QuantConfigFactory:
    def __init__(self, qcfg: dict) -> None:
        self._qcfg = qcfg

    def get_quant_config(self) -> dict:
        return self._qcfg


def _apply_transform(name: str, gm, *, factory=None, shared_config=None, **config_kwargs):
    config = TransformRegistry.get_config_class(name)(stage="pattern_matcher", **config_kwargs)
    transform = TransformRegistry.get(name)(config)
    return transform._apply(gm, cm=None, factory=factory, shared_config=shared_config)


def _single_op_node(gm: torch.fx.GraphModule, target) -> torch.fx.Node:
    nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is target]
    assert len(nodes) == 1, f"expected exactly one {target} node, found {len(nodes)}"
    return nodes[0]


def _register_expert_buffers(mod: nn.Module, e: int, h: int, i: int, device="cpu") -> None:
    g = torch.Generator(device="cpu").manual_seed(1)
    specs = {
        "gate_up_blocks": torch.randint(
            0, 256, (e, 2 * i, h // 32, 16), dtype=torch.uint8, generator=g
        ),
        "gate_up_bias": torch.randn(e, 2 * i, generator=g) * 0.05,
        "gate_up_scales": torch.randint(
            124, 131, (e, 2 * i, h // 32), dtype=torch.uint8, generator=g
        ),
        "down_blocks": torch.randint(0, 256, (e, h, i // 32, 16), dtype=torch.uint8, generator=g),
        "down_bias": torch.randn(e, h, generator=g) * 0.05,
        "down_scales": torch.randint(124, 131, (e, h, i // 32), dtype=torch.uint8, generator=g),
    }
    for name, t in specs.items():
        mod.register_buffer(name, t.to(device))


def _from_routing_node(graph, x, sel, rw, layout_args, target=_FROM_ROUTING, **kwargs):
    gub, gu_bias, gus, dnb, dn_bias, dns = layout_args
    return graph.call_function(
        target,
        (x, sel, rw, gub, gu_bias, gus, 1.0, 10.0, dnb, dn_bias, dns),
        {"gate_up_order": "up_gate", "swiglu_mode": "deepseek", **kwargs},
    )


@requires_sm100
def test_fuse_moe_routing_localization_rewrites_dsv4_gate():
    E_LOCAL, E_TOTAL, TOP_K, EXPERT_START, H_DSV4, I_DSV4 = 8, 16, 4, 8, 512, 256

    root = nn.Module()
    root.experts = nn.Module()
    _register_expert_buffers(root.experts, E_LOCAL, H_DSV4, I_DSV4, device="cuda")
    root.register_buffer("gate_bias", torch.randn(E_TOTAL, device="cuda") * 0.1)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    logits = graph.placeholder("logits")
    bias_n = graph.get_attr("gate_bias")
    gate = graph.call_function(_DSV4_GATE, (logits, bias_n, TOP_K, 1.5, True))
    sel = graph.call_function(operator.getitem, (gate, 0))
    rw = graph.call_function(operator.getitem, (gate, 1))
    layout_args = tuple(graph.get_attr(f"experts.{n}") for n in _EXPERT_BUFFER_NAMES)
    moe = _from_routing_node(
        graph, x, sel, rw, layout_args, target=_FROM_ROUTING_EP, expert_start=EXPERT_START
    )
    graph.output(moe)
    gm = torch.fx.GraphModule(root, graph)

    torch.manual_seed(0)
    x_in = torch.randn(8, H_DSV4, dtype=torch.bfloat16, device="cuda") * 0.1
    logits_in = torch.randn(8, E_TOTAL, device="cuda")
    out_ref = gm(x_in, logits_in)

    _, info = _apply_transform("fuse_moe_routing_localization", gm)
    assert info.skipped is False
    assert info.num_matches == 1

    gate_node = _single_op_node(gm, _DSV4_GATE_LOCALIZED)
    assert gate_node.kwargs["expert_start"] == EXPERT_START
    assert gate_node.kwargs["local_experts"] == E_LOCAL
    moe_node = _single_op_node(gm, _FROM_ROUTING_EP)
    assert moe_node.kwargs["routing_localized"] is True

    gm.recompile()
    out_fused = gm(x_in, logits_in)
    assert torch.equal(out_fused, out_ref)


def test_fuse_moe_routing_localization_skips_non_matching_moe_nodes(monkeypatch):
    monkeypatch.setattr(mxfp4_transform_mod, "is_sm_100f", lambda *a, **k: True)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    logits = graph.placeholder("logits")
    bias = graph.placeholder("bias")
    sel_ph = graph.placeholder("sel")
    rw_ph = graph.placeholder("rw")
    blocks = graph.placeholder("blocks")
    zero_blocks = graph.placeholder("zero_blocks")
    zero_blocks.meta["val"] = torch.empty(0, 8, device="meta")
    aux = tuple(graph.placeholder(f"aux{i}") for i in range(5))

    def gate_pair(swap=False):
        g = graph.call_function(_DSV4_GATE, (logits, bias, 4, 1.5, True))
        s = graph.call_function(operator.getitem, (g, 1 if swap else 0))
        w = graph.call_function(operator.getitem, (g, 0 if swap else 1))
        return g, s, w

    def moe(sel, rw, blk, **kwargs):
        return _from_routing_node(graph, x, sel, rw, (blk, *aux), **kwargs)

    _, s1, w1 = gate_pair()
    m1 = moe(s1, w1, blocks, gate_up_order="gate_up")  # non-trtllm gate_up layout
    m2 = moe(sel_ph, rw_ph, blocks)  # routing inputs are not gate getitems
    add = graph.call_function(torch.ops.aten.add.Tensor, (x, x))
    m3 = moe(
        graph.call_function(operator.getitem, (add, 0)),
        graph.call_function(operator.getitem, (add, 1)),
        blocks,
    )  # producer is not a DSV4 gate
    _, s4, w4 = gate_pair(swap=True)
    m4 = moe(s4, w4, blocks)  # getitem indices swapped
    g5, s5, w5 = gate_pair()
    m5 = moe(s5, w5, blocks)  # rw has a second consumer (graph output)
    _, s6, w6 = gate_pair()
    m6 = moe(s6, w6, zero_blocks)  # local expert count resolves to 0
    g7, _, w7 = gate_pair()
    m7 = moe(graph.call_function(torch.ops.aten.clone.default, (g7,)), w7, blocks)  # not a getitem
    m8 = moe(7, rw_ph, blocks)  # selected_experts is not a Node
    _, s9, w9 = gate_pair()
    m9 = moe(s9, w9, blocks)  # blocks placeholder without meta: no local expert count
    _, s10, w10 = gate_pair()
    m10 = moe(s10, w10, 7)  # blocks arg is not a Node
    graph.output((m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, w5))
    gm = torch.fx.GraphModule(nn.Module(), graph)

    _, info = _apply_transform("fuse_moe_routing_localization", gm)
    assert info.skipped is True
    assert info.num_matches == 0
    assert all(n.target is not _DSV4_GATE_LOCALIZED for n in gm.graph.nodes)
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target is _FROM_ROUTING:
            assert "routing_localized" not in n.kwargs


# ---------------------------------------------------------------------------
# quantize_mxfp4_moe: DSV4 checkpoint-layout load hooks
# ---------------------------------------------------------------------------

_LAYOUT_E, _LAYOUT_H, _LAYOUT_I = 2, 32, 32


def _build_layout_hook_gm() -> torch.fx.GraphModule:
    root = nn.Module()
    layer0 = nn.Module()
    layer0.ffn = nn.Module()
    layer0.ffn.experts = nn.Module()
    _register_expert_buffers(layer0.ffn.experts, _LAYOUT_E, _LAYOUT_H, _LAYOUT_I)
    root.layers = nn.ModuleList([layer0])
    root.plain = nn.Module()
    _register_expert_buffers(root.plain, _LAYOUT_E, _LAYOUT_H, _LAYOUT_I)

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    sel = graph.placeholder("sel")
    rw = graph.placeholder("rw")

    def layout_args(prefix):
        return tuple(graph.get_attr(f"{prefix}.{n}") for n in _EXPERT_BUFFER_NAMES)

    m_a = _from_routing_node(graph, x, sel, rw, layout_args("layers.0.ffn.experts"))
    m_dup = _from_routing_node(graph, x, sel, rw, layout_args("layers.0.ffn.experts"))
    m_placeholder_args = _from_routing_node(graph, x, sel, rw, (x, x, x, x, x, x))
    m_no_layer = _from_routing_node(graph, x, sel, rw, layout_args("plain"))
    graph.output((m_a, m_dup, m_placeholder_args, m_no_layer))
    return torch.fx.GraphModule(root, graph)


def _fake_dsv4_checkpoint() -> dict:
    g = torch.Generator(device="cpu").manual_seed(2)
    sd = {}
    for e in range(_LAYOUT_E):
        for proj, rows, cols in (
            ("w1", _LAYOUT_I, _LAYOUT_H),
            ("w3", _LAYOUT_I, _LAYOUT_H),
            ("w2", _LAYOUT_H, _LAYOUT_I),
        ):
            base = f"layers.0.ffn.experts.{e}.{proj}"
            sd[f"{base}.weight"] = torch.randint(
                0, 256, (rows, cols // 2), dtype=torch.uint8, generator=g
            )
            sd[f"{base}.scale"] = torch.randint(
                120, 132, (rows, cols // 32), dtype=torch.uint8, generator=g
            )
    return sd


def test_quantize_mxfp4_moe_registers_dsv4_layout_load_hooks():
    gm = _build_layout_hook_gm()
    layout = build_deepseek_v4_packed_mxfp4_experts_layout()
    factory = _QuantConfigFactory(
        {
            "quant_method": "mxfp4",
            "checkpoint_layout": SimpleNamespace(checkpoint_consumers=(object(), layout)),
        }
    )
    _, info = _apply_transform("quantize_mxfp4_moe", gm, factory=factory)
    # Exactly one hook: dup targets are deduped, placeholder-arg and
    # no-layer-path nodes are rejected.
    assert info.skipped is False
    assert info.num_matches == 1

    sd = _fake_dsv4_checkpoint()
    reference = layout.pack_experts(
        dict(sd),
        layer=0,
        hidden_size=_LAYOUT_H,
        intermediate_size=_LAYOUT_I,
        num_experts=_LAYOUT_E,
    )
    gm.load_state_dict(sd, strict=False)

    experts = gm.get_submodule("layers.0.ffn.experts")
    assert torch.equal(experts.gate_up_blocks, reference.gate_up_blocks)
    assert torch.equal(experts.gate_up_scales, reference.gate_up_scales)
    assert torch.equal(experts.down_blocks, reference.down_blocks)
    assert torch.equal(experts.down_scales, reference.down_scales)

    # Packed runtime keys already present: the hook early-returns untouched.
    sd_packed = {
        f"layers.0.ffn.experts.{name}": getattr(reference, name)
        for name in ("gate_up_blocks", "gate_up_scales", "down_blocks", "down_scales")
    }
    gm.load_state_dict(sd_packed, strict=False)
    assert torch.equal(experts.gate_up_blocks, reference.gate_up_blocks)


def test_quantize_mxfp4_moe_foreign_quant_method_skips():
    gm = _build_layout_hook_gm()
    factory = _QuantConfigFactory(
        {"quant_method": "fp8", "checkpoint_layout": SimpleNamespace(checkpoint_consumers=())}
    )
    _, info = _apply_transform("quantize_mxfp4_moe", gm, factory=factory)
    assert info.skipped is True
    assert info.num_matches == 0


# ---------------------------------------------------------------------------
# quantize_mxfp4_moe: dense-MoE rewrite per backend
# ---------------------------------------------------------------------------


class _DenseMoeModule(nn.Module):
    def __init__(self, num_experts=2, hidden=32, inter=32):
        super().__init__()
        self.router_weight = nn.Parameter(torch.randn(num_experts, hidden))
        self.router_bias = nn.Parameter(torch.randn(num_experts))
        self.gate_up_w = nn.Parameter(torch.randn(num_experts, hidden, 2 * inter))
        self.gate_up_b = nn.Parameter(torch.randn(num_experts, 2 * inter))
        self.down_w = nn.Parameter(torch.randn(num_experts, inter, hidden))
        self.down_b = nn.Parameter(torch.randn(num_experts, hidden))

    def forward(self, hidden_states):
        routing = torch.ops.auto_deploy.torch_moe_router(
            hidden_states, self.router_weight, self.router_bias, 1
        )
        return torch.ops.auto_deploy.torch_moe_dense_mlp(
            hidden_states,
            routing,
            self.gate_up_w,
            self.gate_up_b,
            self.down_w,
            self.down_b,
            1.0,
            10.0,
        )


@pytest.mark.parametrize("backend", ["triton", "trtllm"])
def test_quantize_mxfp4_moe_rewrites_dense_moe(backend, monkeypatch):
    monkeypatch.setattr(mxfp4_transform_mod, "get_sm_version", lambda: 100)
    gm = torch.fx.symbolic_trace(_DenseMoeModule())
    factory = _QuantConfigFactory({"quant_method": "mxfp4"})
    shared = None
    if backend == "trtllm":
        dc = DistConfig(world_size=2, rank=0, tp_size=2, moe_ep_size=2)
        shared = SharedConfig(local_rank=0, world_size=2, dist_config=dc)

    _, info = _apply_transform(
        "quantize_mxfp4_moe", gm, factory=factory, shared_config=shared, backend=backend
    )
    assert info.skipped is False
    assert info.num_matches == 1
    assert "gate_up_w" not in dict(gm.named_parameters())

    if backend == "triton":
        node = _single_op_node(gm, torch.ops.auto_deploy.triton_mxfp4_moe.default)
        assert node.args[3] == 1  # top_k
        blocks = gm.get_parameter("gate_up_proj_blocks")
        assert blocks.shape == (2, 64, 1, 16) and blocks.dtype == torch.uint8
        assert gm.get_parameter("gate_up_proj_scales").shape == (2, 64, 1)
    else:
        node = _single_op_node(
            gm, torch.ops.auto_deploy.trtllm_quant_mxfp4_trtllm_gen_moe_fused.default
        )
        # EP=2: raw HF params registered at the EP-sliced expert count.
        assert gm.get_parameter("gate_up_proj_blocks").shape == (1, 64, 1, 16)
        assert gm.get_parameter("down_proj_blocks").shape == (1, 32, 1, 16)
        assert torch.allclose(gm.get_parameter("swiglu_alpha_trtllm"), torch.full((1,), 1.702))
        ar_targets = {
            torch.ops.auto_deploy.torch_dist_all_reduce.default,
            torch.ops.auto_deploy.trtllm_dist_all_reduce.default,
        }
        ar_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and n.target in ar_targets]
        assert len(ar_nodes) == 1 and ar_nodes[0].args[0] is node
