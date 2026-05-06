"""Tests for multi-stream Q/K/V projection parallelism in GQA attention.

The transform should:
  1. Detect a fork point with three or more linear users (q/k/v).
  2. Identify the heaviest one (Q) and leave it on the main stream.
  3. Move the lighter linears (K, V) onto the auxiliary CUDA stream.
  4. Preserve numerical correctness.
  5. Be compatible with CUDA graph capture & replay.
  6. Skip MHA (equal-sized q/k/v) since GQA-only is the gain case.
"""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_qkv import (
    _execute_qkv_lighter_in_aux_stream,
    _find_qkv_fork_lighter_linears,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import cuda_stream_manager


class MockGQABlock(nn.Module):
    """Simplified GQA-style attention with q/k/v fork + final out_proj."""

    def __init__(self, hidden_dim: int, q_dim: int, kv_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, kv_dim, bias=False)
        # The "attention" — a content-aware combine that defeats the BFS
        # downstream-linear check on q/k/v without inserting another linear.
        self.layernorm = nn.LayerNorm(q_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Pad k/v to q's width and combine without a linear.
        pad_kv = q.shape[-1] - k.shape[-1]
        k = torch.nn.functional.pad(k, (0, pad_kv))
        v = torch.nn.functional.pad(v, (0, pad_kv))
        return self.layernorm(q + k + v)


def _build_gm(model, example_input):
    egm = torch.export.export(model, (example_input,))
    return egm.module()


def test_pattern_matching_single_block():
    """A GQA fork with 3 linear users (q heavier than k/v) yields one match."""
    model = MockGQABlock(128, 256, 64).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_qkv_fork_lighter_linears(gm)
    assert len(pairs) == 1, f"Expected 1 fork-point pair, got {len(pairs)}"
    fork, lighters = pairs[0]
    assert len(lighters) == 2, f"Expected 2 lighter linears (k,v), got {len(lighters)}"


def test_no_match_on_mha_equal_dims():
    """MHA-style (equal q/k/v out features) should not match."""
    model = MockGQABlock(128, 128, 128).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_qkv_fork_lighter_linears(gm)
    assert len(pairs) == 0, f"Expected 0 matches for MHA, got {len(pairs)}"


def test_no_match_on_two_way_fork():
    """A 2-way fork (e.g. MLA) is handled by multi_stream_mla_attn, not here."""

    class TwoWay(nn.Module):
        def __init__(self, dim, q_dim, kv_dim):
            super().__init__()
            self.q = nn.Linear(dim, q_dim, bias=False)
            self.kv = nn.Linear(dim, kv_dim, bias=False)
            self.norm = nn.LayerNorm(q_dim)

        def forward(self, x):
            q = self.q(x)
            kv = self.kv(x)
            kv = torch.nn.functional.pad(kv, (0, q.shape[-1] - kv.shape[-1]))
            return self.norm(q + kv)

    model = TwoWay(128, 256, 64).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_qkv_fork_lighter_linears(gm)
    assert len(pairs) == 0, f"Expected 0 matches for 2-way fork, got {len(pairs)}"


def test_numerical_correctness():
    """After the transform the GraphModule must match the original output."""
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockGQABlock(128, 256, 64).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, 128, device="cuda")
    ref = model(test_x)

    gm, num_replaced = _execute_qkv_lighter_in_aux_stream(gm)
    assert num_replaced == 2, f"Expected 2 replacements (k, v), got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref).abs().max().item()}"
    )


def test_cuda_graph_compatibility():
    """The transformed GraphModule must work under CUDA graph capture and replay."""
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockGQABlock(128, 256, 64).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, 128, device="cuda")
    ref = model(test_x)

    gm, num_replaced = _execute_qkv_lighter_in_aux_stream(gm)
    assert num_replaced == 2

    static_x = torch.randn(4, 128, device="cuda")
    static_y = torch.randn(4, 256, device="cuda")

    for _ in range(3):
        static_y.copy_(gm(static_x))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_y.copy_(gm(static_x))

    static_x.copy_(test_x)
    graph.replay()

    assert torch.allclose(static_y, ref, atol=1e-5), (
        f"CUDA graph output mismatch: max diff = {(static_y - ref).abs().max().item()}"
    )
