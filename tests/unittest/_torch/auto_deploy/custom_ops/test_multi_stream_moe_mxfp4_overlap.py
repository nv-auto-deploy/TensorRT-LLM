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

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe  # noqa: F401
from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_moe import (
    _execute_shared_expert_in_aux_stream,
)
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
