# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused checks for multi-stream overlap of the DSV4-style MXFP4 MoE pattern.

``multi_stream_moe`` now recognizes ``torch_mxfp4_moe_from_routing`` (and its
``_ep`` variant): the shared-expert MLP moves to the auxiliary CUDA stream while
the routed experts stay on the main stream, rejoining right before the merge
``add`` (the single fused all_reduce stays on the main stream, after the add).

Checked here on a single GPU (torch-reference MXFP4 path, forced via the
gpt-oss swiglu mode so tiny shapes stay off the trtllm-gen runner):
  1. The transform matches the from-routing op and inserts the
     begin/end/wait passthrough triplet exactly once per MoE.
  2. Overlap precondition: the aux-stream region (shared branch) is dispatched
     BEFORE the routed op in graph order. ``begin_aux_stream_passthrough``
     makes the aux stream wait for all main-stream work enqueued so far, so a
     routed-first order would serialize shared after routed (parity, no
     overlap). DSV4 modeling dispatches the shared MLP first for this reason.
  3. The rewrite is bit-exact in eager execution.
  4. The rewrite is bit-exact under CUDA-graph capture + replay (the
     deployment path: monolithic torch-cudagraph decode).
"""

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (register ops)
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import (
    SharedConfig,
    Stages,
    TransformConfig,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_quant import FuseFP8SwigluActQuant
from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
    _execute_shared_expert_in_aux_stream,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import run_shape_prop
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
)

E, TOP_K, H, INTER = 8, 2, 64, 32
BLOCK = 32
PACKED = BLOCK // 2


class _SharedPlusRoutedMoE(nn.Module):
    """Mirror of the DSV4 MoE dataflow: shared MLP dispatched first, then the
    from-routing MXFP4 op, merged with a single add (routed + shared)."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.w1 = nn.Linear(H, INTER, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(H, INTER, bias=False, dtype=torch.bfloat16)
        self.w2 = nn.Linear(INTER, H, bias=False, dtype=torch.bfloat16)
        for lin in (self.w1, self.w3, self.w2):
            with torch.no_grad():
                lin.weight.copy_(
                    torch.randn(lin.weight.shape, generator=g, dtype=torch.float32) * 0.05
                )
        self.register_buffer(
            "gate_up_blocks",
            torch.randint(
                0, 256, (E, 2 * INTER, H // BLOCK, PACKED), dtype=torch.uint8, generator=g
            ),
        )
        self.register_buffer(
            "gate_up_scales",
            torch.randint(110, 130, (E, 2 * INTER, H // BLOCK), dtype=torch.uint8, generator=g),
        )
        self.register_buffer(
            "gate_up_bias",
            torch.randn(E, 2 * INTER, generator=g, dtype=torch.float32) * 0.1,
        )
        self.register_buffer(
            "down_blocks",
            torch.randint(0, 256, (E, H, INTER // BLOCK, PACKED), dtype=torch.uint8, generator=g),
        )
        self.register_buffer(
            "down_scales",
            torch.randint(110, 130, (E, H, INTER // BLOCK), dtype=torch.uint8, generator=g),
        )
        self.register_buffer("down_bias", torch.randn(E, H, generator=g, dtype=torch.float32) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Shared-expert MLP dispatched FIRST (see module docstring / DSV4 modeling).
        shared = self.w2((F.silu(self.w1(x).float()) * self.w3(x).float()).to(x.dtype))
        # gpt-oss swiglu mode keeps tiny shapes on the torch-reference path
        # (the SM100 trtllm-gen fast path only engages for up_gate/deepseek).
        routed = torch.ops.auto_deploy.torch_mxfp4_moe_from_routing(
            x,
            selected_experts,
            routing_weights,
            self.gate_up_blocks,
            self.gate_up_bias,
            self.gate_up_scales,
            1.0,
            7.0,
            self.down_blocks,
            self.down_bias,
            self.down_scales,
            "interleaved",
            "gpt_oss",
            "moe",
        )
        return routed + shared


def _make_inputs(num_tokens: int, seed: int, device: str):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(num_tokens, H, generator=g, dtype=torch.float32) * 0.3).to(
        dtype=torch.bfloat16, device=device
    )
    selected = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(num_tokens)]).to(
        device
    )
    weights = torch.rand(num_tokens, TOP_K, generator=g, dtype=torch.float32).to(device)
    return x, selected, weights


def _export_gm(mod: nn.Module, args) -> torch.fx.GraphModule:
    ep = torch.export.export(mod, args)
    return ep.module()


def _node_index(gm: torch.fx.GraphModule, matcher) -> int:
    for i, n in enumerate(gm.graph.nodes):
        if matcher(n):
            return i
    return -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("num_tokens", [2, 8])  # decode (T*K < E) and dense (T*K >= E) paths
def test_multi_stream_moe_matches_mxfp4_from_routing_and_is_bit_exact(num_tokens):
    device = "cuda"
    cuda_stream_manager.add_device(torch.cuda.current_device())
    mod = _SharedPlusRoutedMoE().to(device)
    args = _make_inputs(num_tokens, seed=123, device=device)

    gm_ref = _export_gm(mod, args)
    ref_out = gm_ref(*args)

    gm = copy.deepcopy(gm_ref)
    gm, num_replaced = _execute_shared_expert_in_aux_stream(
        gm,
        [
            torch.ops.auto_deploy.torch_mxfp4_moe_from_routing,
            torch.ops.auto_deploy.torch_mxfp4_moe_from_routing_ep,
        ],
    )
    assert num_replaced == 1, "transform must match the from-routing MXFP4 MoE exactly once"
    gm.graph.lint()
    gm.recompile()

    idx_begin = _node_index(gm, lambda n: n.target is begin_aux_stream_passthrough)
    idx_end = _node_index(gm, lambda n: n.target is end_aux_stream_passthrough)
    idx_wait = _node_index(gm, lambda n: n.target is wait_aux_stream_passthrough)
    idx_moe = _node_index(
        gm,
        lambda n: n.op == "call_function" and "torch_mxfp4_moe_from_routing" in str(n.target),
    )
    assert idx_begin != -1 and idx_end != -1 and idx_wait != -1, (
        "begin/end/wait passthroughs must all be inserted"
    )
    assert idx_begin < idx_end < idx_moe < idx_wait, (
        "overlap precondition violated: the aux-stream shared region "
        f"(begin={idx_begin}, end={idx_end}) must be dispatched before the routed op "
        f"(moe={idx_moe}) and the main stream must wait for aux only at the merge "
        f"(wait={idx_wait})"
    )

    out = gm(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out), "multi-stream rewrite must be bit-exact (eager)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_multi_stream_moe_mxfp4_bit_exact_under_cuda_graph():
    device = "cuda"
    cuda_stream_manager.add_device(torch.cuda.current_device())
    mod = _SharedPlusRoutedMoE().to(device)
    args = _make_inputs(2, seed=321, device=device)

    gm_ref = _export_gm(mod, args)
    ref_out = gm_ref(*args)

    gm = copy.deepcopy(gm_ref)
    gm, num_replaced = _execute_shared_expert_in_aux_stream(
        gm, [torch.ops.auto_deploy.torch_mxfp4_moe_from_routing]
    )
    assert num_replaced == 1
    gm.recompile()

    # Warm up on a side stream, then capture the whole forward monolithically —
    # the same shape as the torch-cudagraph decode path this transform targets.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            warm = gm(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = gm(*args)

    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(static_out, ref_out), (
        "multi-stream rewrite must be bit-exact under CUDA-graph capture/replay"
    )
    assert torch.equal(warm, ref_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_multi_stream_moe_from_routing_entry_is_scoped_and_checked_in():
    """The checked-in enablement path: ``multi_stream_moe_from_routing`` ships
    enabled in ``default.yaml`` with an op allowlist scoped to the DSV4
    from-routing MXFP4 ops, and the allowlist actually gates the rewrite."""
    import os

    import yaml

    import tensorrt_llm._torch.auto_deploy as ad_pkg
    from tensorrt_llm._torch.auto_deploy.transform.interface import TransformRegistry

    default_yaml = os.path.join(os.path.dirname(ad_pkg.__file__), "config", "default.yaml")
    with open(default_yaml) as f:
        entry = yaml.safe_load(f)["transforms"]["multi_stream_moe_from_routing"]
    assert entry["enabled"] is True
    assert entry["op_allowlist"] == [
        "torch_mxfp4_moe_from_routing",
        "torch_mxfp4_moe_from_routing_ep",
    ]

    device = "cuda"
    cuda_stream_manager.add_device(torch.cuda.current_device())
    mod = _SharedPlusRoutedMoE().to(device)
    args = _make_inputs(2, seed=11, device=device)
    gm_ref = _export_gm(mod, args)
    ref_out = gm_ref(*args)

    transform_cls = TransformRegistry.get("multi_stream_moe_from_routing")

    # The default.yaml allowlist matches the from-routing op exactly once.
    gm = copy.deepcopy(gm_ref)
    t = transform_cls.from_kwargs(stage=entry["stage"], op_allowlist=entry["op_allowlist"])
    gm, info = t._apply(gm, None, None, None)
    assert info.num_matches == 1
    gm.recompile()
    out = gm(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out)

    # An allowlist that excludes the from-routing ops must leave the graph alone.
    gm_other = copy.deepcopy(gm_ref)
    t_other = transform_cls.from_kwargs(stage=entry["stage"], op_allowlist=["trtllm_moe_fused"])
    gm_other, info_other = t_other._apply(gm_other, None, None, None)
    assert info_other.num_matches == 0
    assert not any(n.target is begin_aux_stream_passthrough for n in gm_other.graph.nodes), (
        "op_allowlist must gate the rewrite"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_multi_stream_moe_rewrite_is_idempotent():
    """A second application (e.g. both transform entries enabled with overlapping
    op lists) must be a no-op instead of nesting begin/end stream switches."""
    device = "cuda"
    cuda_stream_manager.add_device(torch.cuda.current_device())
    mod = _SharedPlusRoutedMoE().to(device)
    args = _make_inputs(2, seed=99, device=device)
    gm = _export_gm(mod, args)
    ref_out = gm(*args)

    moe_ops = [torch.ops.auto_deploy.torch_mxfp4_moe_from_routing]
    gm, first = _execute_shared_expert_in_aux_stream(gm, moe_ops)
    assert first == 1
    gm, second = _execute_shared_expert_in_aux_stream(gm, moe_ops)
    assert second == 0, "re-applying the rewrite must skip already-rewritten MoE nodes"
    gm.recompile()

    n_begin = sum(1 for n in gm.graph.nodes if n.target is begin_aux_stream_passthrough)
    n_end = sum(1 for n in gm.graph.nodes if n.target is end_aux_stream_passthrough)
    n_wait = sum(1 for n in gm.graph.nodes if n.target is wait_aux_stream_passthrough)
    assert (n_begin, n_end, n_wait) == (1, 1, 1)

    out = gm(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out)


# ---------------------------------------------------------------------------
# Post-idea_0007 graph shape: block-FP8 shared expert with the fused
# swiglu+act-quant tail feeding the residual-add-prequant down projection
# (residual = routed MoE output).  The overlap must survive this form.
# ---------------------------------------------------------------------------

H_FP8, I_SH = 256, 256  # block-128 FP8 quant needs multiples of 128
QBLK = 128
LIMIT = 7.0


def _fp8_weight(n, k, g):
    w = (torch.randn(n, k, generator=g, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
    s = torch.rand(n // QBLK, k // QBLK, generator=g, dtype=torch.float32) * 0.01 + 0.005
    return w, s


class _MXFP4RoutedBuffers(nn.Module):
    """Routed-expert MXFP4 buffers for hidden size ``H_FP8`` (gpt-oss swiglu mode
    keeps the tiny shapes on the torch-reference path)."""

    def __init__(self, g) -> None:
        super().__init__()
        self.register_buffer(
            "gate_up_blocks",
            torch.randint(
                0, 256, (E, 2 * INTER, H_FP8 // BLOCK, PACKED), dtype=torch.uint8, generator=g
            ),
        )
        self.register_buffer(
            "gate_up_scales",
            torch.randint(110, 130, (E, 2 * INTER, H_FP8 // BLOCK), dtype=torch.uint8, generator=g),
        )
        self.register_buffer(
            "gate_up_bias", torch.randn(E, 2 * INTER, generator=g, dtype=torch.float32) * 0.1
        )
        self.register_buffer(
            "down_blocks",
            torch.randint(
                0, 256, (E, H_FP8, INTER // BLOCK, PACKED), dtype=torch.uint8, generator=g
            ),
        )
        self.register_buffer(
            "down_scales",
            torch.randint(110, 130, (E, H_FP8, INTER // BLOCK), dtype=torch.uint8, generator=g),
        )
        self.register_buffer(
            "down_bias", torch.randn(E, H_FP8, generator=g, dtype=torch.float32) * 0.1
        )

    def routed(self, x, selected_experts, routing_weights):
        return torch.ops.auto_deploy.torch_mxfp4_moe_from_routing(
            x,
            selected_experts,
            routing_weights,
            self.gate_up_blocks,
            self.gate_up_bias,
            self.gate_up_scales,
            1.0,
            LIMIT,
            self.down_blocks,
            self.down_bias,
            self.down_scales,
            "interleaved",
            "gpt_oss",
            "moe",
        )


class _FP8SharedRoutedMoEStrandedTail(_MXFP4RoutedBuffers):
    """Post-idea_0007 node ORDER as ``fuse_fp8_swiglu_act_quant`` used to emit it:
    the shared-expert head (merged gate_up FP8 GEMM + narrows) is dispatched before
    the routed op, but the fused swiglu+act-quant tail and the residual-add-prequant
    down projection sit AFTER the routed op in graph order (the fusion inserted them
    at the old down-projection site, which follows the MoE op once the merge add is
    folded into it as its residual)."""

    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        super().__init__(g)
        w13, s13 = _fp8_weight(2 * I_SH, H_FP8, g)
        w2, s2 = _fp8_weight(H_FP8, I_SH, g)
        self.register_buffer("w13_q", w13)
        self.register_buffer("w13_s", s13)
        self.register_buffer("w2_q", w2)
        self.register_buffer("w2_s", s2)

    def forward(self, x, selected_experts, routing_weights):
        gate_up = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
            x, self.w13_q, None, [], [self.w13_s], [], [], "none", None, 1, "unknown", ""
        )
        gate = torch.narrow(gate_up, -1, 0, I_SH)
        up = torch.narrow(gate_up, -1, I_SH, I_SH)
        routed = self.routed(x, selected_experts, routing_weights)
        # Shared-expert tail stranded after the routed op.
        q, s = torch.ops.auto_deploy.torch_fp8_swiglu_clamp_act_quant(gate, up, LIMIT, QBLK, "")
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add_prequant(
            q, s, self.w2_q, None, [self.w2_s], routed
        )


class _FP8SharedRoutedMoEResidualDown(_MXFP4RoutedBuffers):
    """Post-idea_0008 form (the input ``fuse_fp8_swiglu_act_quant`` consumes): the
    unfused clamped-SwiGLU chain feeding the residual-add down projection whose
    residual is the routed MoE output."""

    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        super().__init__(g)
        w13, s13 = _fp8_weight(2 * I_SH, H_FP8, g)
        w2, s2 = _fp8_weight(H_FP8, I_SH, g)
        self.register_buffer("w13_q", w13)
        self.register_buffer("w13_s", s13)
        self.register_buffer("w2_q", w2)
        self.register_buffer("w2_s", s2)

    def forward(self, x, selected_experts, routing_weights):
        gate_up = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
            x, self.w13_q, None, [], [self.w13_s], [], [], "none", None, 1, "unknown", ""
        )
        gate = torch.narrow(gate_up, -1, 0, I_SH)
        up = torch.narrow(gate_up, -1, I_SH, I_SH)
        routed = self.routed(x, selected_experts, routing_weights)
        gate = torch.clamp(gate, max=LIMIT)
        up = torch.clamp(up, min=-LIMIT, max=LIMIT)
        hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
            hidden,
            self.w2_q,
            None,
            [],
            [self.w2_s],
            [],
            [],
            input_scale_fmt="",
            residual=routed,
        )


def _make_fp8_inputs(num_tokens, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(num_tokens, H_FP8, generator=g, dtype=torch.float32) * 0.3).to(
        dtype=torch.bfloat16, device=device
    )
    selected = torch.stack([torch.randperm(E, generator=g)[:TOP_K] for _ in range(num_tokens)]).to(
        device
    )
    weights = torch.rand(num_tokens, TOP_K, generator=g, dtype=torch.float32).to(device)
    return x, selected, weights


def _fp8_supported():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _overlap_indices(gm):
    idx = {}
    for i, n in enumerate(gm.graph.nodes):
        if n.target is begin_aux_stream_passthrough:
            idx["begin"] = i
        elif n.target is end_aux_stream_passthrough:
            idx["end"] = i
        elif n.target is wait_aux_stream_passthrough:
            idx["wait"] = i
        elif n.op == "call_function" and "torch_mxfp4_moe_from_routing" in str(n.target):
            idx["moe"] = i
    return idx


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_multi_stream_moe_hoists_stranded_fp8_prequant_tail():
    """The rewrite must compact a shared-expert tail stranded after the routed op
    back in front of it: ``begin``/``end`` bracket a positional aux-stream window,
    so a routed op inside the window would serialize both branches on aux."""
    device = "cuda"
    cuda_stream_manager.add_device(torch.cuda.current_device())
    mod = _FP8SharedRoutedMoEStrandedTail().to(device)
    args = _make_fp8_inputs(2, seed=123, device=device)

    gm_ref = _export_gm(mod, args)
    ref_out = gm_ref(*args)
    torch.cuda.synchronize()

    gm = copy.deepcopy(gm_ref)
    gm, num_replaced = _execute_shared_expert_in_aux_stream(
        gm, [torch.ops.auto_deploy.torch_mxfp4_moe_from_routing]
    )
    assert num_replaced == 1
    gm.graph.lint()
    gm.recompile()

    idx = _overlap_indices(gm)
    assert idx["begin"] < idx["end"] < idx["moe"] < idx["wait"], (
        "stranded shared-expert tail must be hoisted ahead of the routed op so the "
        f"aux window excludes it (begin={idx['begin']}, end={idx['end']}, "
        f"moe={idx['moe']}, wait={idx['wait']})"
    )

    out = gm(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out), "hoisting rewrite must be bit-exact (eager)"

    # Deployment path: monolithic CUDA-graph capture + replay.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            gm(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = gm(*args)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, ref_out), (
        "hoisting rewrite must be bit-exact under CUDA-graph capture/replay"
    )


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_swiglu_fusion_keeps_shared_first_dispatch_for_overlap():
    """End-to-end at the real pipeline seam: ``fuse_fp8_swiglu_act_quant`` on the
    residual-add down projection must emit the fused tail ahead of the routed op
    (anchored at its gate/up sources), and the multi-stream rewrite must then
    bracket only the shared branch."""
    device = "cuda"
    cuda_stream_manager.add_device(torch.cuda.current_device())
    mod = _FP8SharedRoutedMoEResidualDown().to(device)
    args = _make_fp8_inputs(2, seed=321, device=device)

    gm = torch_export_to_gm(mod, args=args)
    ref_out = gm(*args)
    ref_out = ref_out[0] if isinstance(ref_out, (tuple, list)) else ref_out
    torch.cuda.synchronize()

    gm, info = FuseFP8SwigluActQuant(TransformConfig(stage=Stages.POST_LOAD_FUSION))._apply(
        gm, None, None, SharedConfig()
    )
    assert info.num_matches == 1
    gm.recompile()
    # The pipeline entry sets ``run_shape_prop: true``: post-transform cleanup
    # repopulates meta on the new nodes, which downstream matchers (the
    # multi-stream merge search) rely on. Mimic it here.
    run_shape_prop(gm, args_static=args)

    # Lever under test: the fused act-quant chain must precede the routed op in
    # graph order (shared-first dispatch), not sit at the down-projection site.
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    act_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and "torch_fp8_swiglu_clamp_act_quant" in str(n.target)
    ]
    moe_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and "torch_mxfp4_moe_from_routing" in str(n.target)
    ]
    assert len(act_nodes) == 1 and len(moe_nodes) == 1
    assert order[act_nodes[0]] < order[moe_nodes[0]], (
        "fuse_fp8_swiglu_act_quant must anchor the fused act-quant chain at its "
        "gate/up sources, ahead of the routed op"
    )

    gm, num_replaced = _execute_shared_expert_in_aux_stream(
        gm, [torch.ops.auto_deploy.torch_mxfp4_moe_from_routing]
    )
    assert num_replaced == 1
    gm.graph.lint()
    gm.recompile()

    idx = _overlap_indices(gm)
    assert idx["begin"] < idx["end"] < idx["moe"] < idx["wait"], (
        f"overlap precondition violated after the fused rewrite (begin={idx['begin']}, "
        f"end={idx['end']}, moe={idx['moe']}, wait={idx['wait']})"
    )

    out = gm(*args)
    out = out[0] if isinstance(out, (tuple, list)) else out
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out), "fusion + multi-stream rewrite must be bit-exact"
