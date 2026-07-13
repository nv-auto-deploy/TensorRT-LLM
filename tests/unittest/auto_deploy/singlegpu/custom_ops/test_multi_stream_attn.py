# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for multi-stream Q/KV projection parallelism in MLA attention.

The test builds a minimal mock model that mirrors the MLA fork pattern:
a shared input feeds two parallel linear chains (one heavier "Q-like",
one lighter "KV-like") whose outputs are combined with an add.

The transform should:
  1. Detect the fork point (shared input with 2+ linear users).
  2. Identify the lighter KV-like linear (no downstream linear within
     a few hops) vs. the heavier Q-like chain (has a downstream linear).
  3. Move the KV linear onto the auxiliary CUDA stream.
  4. Preserve numerical correctness.
  5. Be compatible with CUDA graph capture & replay.
"""

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_attn import (
    _execute_kv_proj_in_aux_stream,
    _execute_kv_proj_in_aux_stream_extended,
    _find_kv_proj_linears,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
    begin_aux_stream_passthrough,
    cuda_stream_manager,
    end_aux_stream_passthrough,
    wait_aux_stream_passthrough,
)

# ---------------------------------------------------------------------------
# Helpers -- mock MLA-like module
# ---------------------------------------------------------------------------


class MockMLABlock(nn.Module):
    """Simplified MLA-like attention block with Q and KV projection chains.

    Q chain (heavier):  q_a_proj -> relu (stand-in for rms_norm) -> q_b_proj
    KV chain (lighter):  kv_a_proj
    Merge: add(q_b_proj_output, kv_a_proj_output)

    The layernorm at the output simulates the inter-layer distance in a real
    transformer (output projection, residual add, layernorm) so that the
    next layer's fork point is beyond the BFS max_depth from this layer's
    KV linear.
    """

    def __init__(self, hidden_dim: int, q_inner_dim: int, kv_out_dim: int):
        super().__init__()
        # Q chain: two linears with a non-linearity in between
        self.q_a_proj = nn.Linear(hidden_dim, q_inner_dim, bias=False)
        self.q_b_proj = nn.Linear(q_inner_dim, kv_out_dim, bias=False)
        # KV chain: single linear
        self.kv_a_proj = nn.Linear(hidden_dim, kv_out_dim, bias=False)
        # Inter-layer distance (layernorm + relu simulate residual + norm)
        self.layernorm = nn.LayerNorm(kv_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Q chain: q_a_proj -> relu -> q_b_proj
        q = self.q_a_proj(x)
        q = torch.nn.functional.relu(q)
        q = self.q_b_proj(q)
        # KV chain: kv_a_proj
        kv = self.kv_a_proj(x)
        out = q + kv
        # Inter-layer distance to push next layer's linears beyond BFS depth
        return self.layernorm(torch.nn.functional.relu(out))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_gm(model, example_input):
    """Export *model* to an FX GraphModule."""
    egm = torch.export.export(model, (example_input,))
    return egm.module()


def test_pattern_matching_single_block():
    """The pattern matcher should find exactly one pair for a single MLA block."""
    model = MockMLABlock(128, 64, 128).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 1, f"Expected 1 fork-point pair, got {len(pairs)}"


def test_pattern_matching_multi_block():
    """Multiple layers with sufficient inter-layer distance should all be matched."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    model = (
        nn.Sequential(
            MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim),
            MockMLABlock(kv_out_dim, q_inner_dim, kv_out_dim),
        )
        .eval()
        .to("cuda")
    )
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 2, f"Expected 2 fork-point pairs, got {len(pairs)}"


def test_numerical_correctness():
    """After the transform the GraphModule must produce the same output as the original model."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm)

    assert num_replaced == 1, f"Expected 1 replacement, got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


def test_numerical_correctness_multi_block():
    """Multi-block correctness test."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = (
        nn.Sequential(
            MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim),
            MockMLABlock(kv_out_dim, q_inner_dim, kv_out_dim),
        )
        .eval()
        .to("cuda")
    )
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm)

    assert num_replaced == 2, f"Expected 2 replacements, got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


def test_cuda_graph_compatibility():
    """The transformed GraphModule must work under CUDA graph capture and replay."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm)
    assert num_replaced == 1

    # Allocate static buffers for CUDA graph capture.
    static_x = torch.randn(4, hidden_dim, device="cuda")
    static_output = torch.randn(4, kv_out_dim, device="cuda")

    # Warm up (required before capture).
    for _ in range(3):
        static_output.copy_(gm(static_x))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output.copy_(gm(static_x))

    static_x.copy_(test_x)
    graph.replay()

    assert torch.allclose(static_output, ref_output, atol=1e-5), (
        f"CUDA graph output mismatch: max diff = {(static_output - ref_output).abs().max().item()}"
    )


class MockFusedMLABlock(nn.Module):
    """Mirrors the post-fusion MLA fork with a fused Q/KV GEMM.

    The fused GEMM's output split interposes narrow+contiguous before the norm
    (relu stand-in), so the Q chain's next linear sits at BFS depth 4:
        fused_qkv -> narrow -> contiguous -> relu -> q_b_proj
    The side projection (replicated GEMV) has no downstream linear.
    """

    def __init__(self, hidden_dim: int, q_inner_dim: int, kv_dim: int, side_dim: int):
        super().__init__()
        self.q_inner_dim = q_inner_dim
        self.kv_dim = kv_dim
        self.fused_qkv = nn.Linear(hidden_dim, q_inner_dim + kv_dim, bias=False)
        self.q_b_proj = nn.Linear(q_inner_dim, kv_dim, bias=False)
        self.side_proj = nn.Linear(hidden_dim, side_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fused_qkv(x)
        q = y.narrow(-1, 0, self.q_inner_dim).contiguous()
        q = torch.nn.functional.relu(q)
        q = self.q_b_proj(q)
        kv = y.narrow(-1, self.q_inner_dim, self.kv_dim)
        side = self.side_proj(x)
        return torch.cat([q + kv, side], dim=-1)


def test_fused_split_needs_deeper_classification():
    """Depth-4 classification for fused splits.

    narrow+contiguous push the Q chain's next linear to depth 4: no match at
    the default depth 3, one (fork, side GEMV) pair at depth 4.
    """
    hidden_dim, q_inner_dim, kv_dim, side_dim = 128, 64, 32, 16
    model = MockFusedMLABlock(hidden_dim, q_inner_dim, kv_dim, side_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    assert len(_find_kv_proj_linears(gm)) == 0, "depth 3 should not classify the fused Q chain"

    pairs = _find_kv_proj_linears(gm, max_depth=4)
    assert len(pairs) == 1, f"Expected 1 fork-point pair at depth 4, got {len(pairs)}"
    kv_linear = pairs[0][1]
    kv_out = kv_linear.meta["val"].shape[-1]
    assert kv_out == side_dim, f"Expected side GEMV (dim {side_dim}) as KV, got dim {kv_out}"


def test_numerical_correctness_fused_split():
    """Aux-stream rewrite at depth 4 must preserve numerics for the fused fork."""
    hidden_dim, q_inner_dim, kv_dim, side_dim = 128, 64, 32, 16
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockFusedMLABlock(hidden_dim, q_inner_dim, kv_dim, side_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm, max_depth=4)
    assert num_replaced == 1, f"Expected 1 replacement, got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Extended aux window (multi-node window + late join)
# ---------------------------------------------------------------------------


class MockExtendedSparseBlock(nn.Module):
    """Mirrors the extended pattern-1 shape around a sparse-attention-like join.

    Fork ``x`` feeds a fused Q chain and a side GEMV whose outputs are read
    only through narrows.  The last Q-chain linear forks into a heavy main
    branch and a light kernel side-cone (rope/quant stand-in).  Everything
    meets only at a late ``cat`` (the attention-op stand-in).
    """

    def __init__(
        self,
        hidden: int = 128,
        q_inner: int = 64,
        main_w: int = 256,
        idx_w: int = 32,
        side_w: int = 48,
    ):
        super().__init__()
        self.q_inner = q_inner
        self.main_w = main_w
        self.idx_w = idx_w
        self.side_w = side_w
        self.fused_a = nn.Linear(hidden, q_inner, bias=False)
        self.q_b = nn.Linear(q_inner, main_w + idx_w, bias=False)
        self.side_proj = nn.Linear(hidden, side_w, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Q chain: fused_a -> narrow -> contiguous -> relu (norm stand-in) -> q_b
        q = self.fused_a(x).narrow(-1, 0, self.q_inner).contiguous()
        qb = self.q_b(torch.nn.functional.relu(q))
        # Heavy main branch (rope stand-in) — must stay on main.
        main = torch.nn.functional.relu(qb.narrow(-1, 0, self.main_w).contiguous())
        # Light side cone (rope + quant stand-in) — movable to aux.
        idx = torch.tanh(qb.narrow(-1, self.main_w, self.idx_w).contiguous()) * 2.0
        # Side GEMV split back with narrows (fused replicated projection shape).
        s = self.side_proj(x)
        s1 = s.narrow(-1, 0, self.side_w // 2)
        s2 = s.narrow(-1, self.side_w // 2, self.side_w - self.side_w // 2)
        # Late join: the only consumer of every side output.
        return torch.cat([main, idx, s1, s2], dim=-1)


class MockExtendedTwoLayer(nn.Module):
    """Two extended blocks with inter-layer distance (as in a real stack)."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.block1 = MockExtendedSparseBlock(hidden)
        out_w = self.block1.main_w + self.block1.idx_w + self.block1.side_w
        self.norm1 = nn.LayerNorm(out_w)
        self.o_proj = nn.Linear(out_w, hidden, bias=False)
        self.block2 = MockExtendedSparseBlock(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # relu + layernorm add BFS distance so the side GEMV of block1 does
        # not see o_proj as a downstream linear.
        y = self.o_proj(self.norm1(torch.nn.functional.relu(self.block1(x))))
        return self.block2(y)


def _count_target(gm, target) -> int:
    return sum(1 for n in gm.graph.nodes if n.op == "call_function" and n.target is target)


def test_extended_window_structure():
    """Extended rewrite: two aux windows, wait immediately before the join."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = MockExtendedSparseBlock().eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    gm, num = _execute_kv_proj_in_aux_stream_extended(gm, max_depth=4)
    assert num == 1, f"Expected 1 extended match, got {num}"

    assert _count_target(gm, begin_aux_stream_passthrough) == 2
    assert _count_target(gm, end_aux_stream_passthrough) == 2
    assert _count_target(gm, wait_aux_stream_passthrough) == 1

    gm.graph.lint()

    # wait_aux must sit immediately before the join consumer (the cat).
    (wait_node,) = [n for n in gm.graph.nodes if n.target is wait_aux_stream_passthrough]
    join = wait_node.next
    assert join.target is torch.ops.aten.cat.default, (
        f"wait_aux not immediately before the join: next is {join.target}"
    )
    # The movable side cone (contiguous/tanh/mul) landed inside window 2.
    begins = [n for n in gm.graph.nodes if n.target is begin_aux_stream_passthrough]
    window2 = []
    n = begins[1].next
    while n.target is not end_aux_stream_passthrough:
        window2.append(n)
        n = n.next
    window2_targets = {n.target for n in window2}
    assert torch.ops.aten.tanh.default in window2_targets, f"window2={window2}"


def test_extended_window_numerics():
    """Extended rewrite must preserve numerics."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = MockExtendedSparseBlock().eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, 128, device="cuda")
    ref_output = model(test_x)

    gm, num = _execute_kv_proj_in_aux_stream_extended(gm, max_depth=4)
    assert num == 1

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


def test_extended_window_numerics_two_layer_cuda_graph():
    """Two stacked windows (repeated event reuse) + CUDA graph capture/replay."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = MockExtendedTwoLayer().eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, 128, device="cuda")
    ref_output = model(test_x)

    gm, num = _execute_kv_proj_in_aux_stream_extended(gm, max_depth=4)
    assert num == 2, f"Expected 2 extended matches, got {num}"
    assert _count_target(gm, begin_aux_stream_passthrough) == 4
    assert _count_target(gm, wait_aux_stream_passthrough) == 2
    gm.graph.lint()

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )

    out_w = ref_output.shape[-1]
    static_x = torch.randn(4, 128, device="cuda")
    static_output = torch.randn(4, out_w, device="cuda")
    for _ in range(3):
        static_output.copy_(gm(static_x))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output.copy_(gm(static_x))

    static_x.copy_(test_x)
    graph.replay()

    assert torch.allclose(static_output, ref_output, atol=1e-5), (
        f"CUDA graph output mismatch: max diff = {(static_output - ref_output).abs().max().item()}"
    )


def test_extended_window_fallback_to_single_op():
    """Fallback coverage: no view-split join shape.

    The extended path must fall back to the single-op rewrite and stay
    numerically correct.
    """
    hidden_dim, q_inner_dim, kv_dim, side_dim = 128, 64, 32, 16
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockFusedMLABlock(hidden_dim, q_inner_dim, kv_dim, side_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num = _execute_kv_proj_in_aux_stream_extended(gm, max_depth=4)
    assert num == 1, f"Expected 1 match, got {num}"
    # Fallback path: no begin/end windows, the derived _aux op instead.
    assert _count_target(gm, begin_aux_stream_passthrough) == 0

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


def test_no_match_on_single_linear():
    """A node with only one linear user should not be matched."""

    class SingleLinear(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = SingleLinear(64).eval().to("cuda")
    example_input = torch.randn(4, 64, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 0, f"Expected 0 matches, got {len(pairs)}"


def test_no_match_when_both_have_downstream_linear():
    """When *both* branches have downstream linears the pattern should not match."""

    class BothHeavy(nn.Module):
        def __init__(self, dim, inner):
            super().__init__()
            self.fc_a1 = nn.Linear(dim, inner, bias=False)
            self.fc_a2 = nn.Linear(inner, dim, bias=False)
            self.fc_b1 = nn.Linear(dim, inner, bias=False)
            self.fc_b2 = nn.Linear(inner, dim, bias=False)

        def forward(self, x):
            a = self.fc_a2(torch.relu(self.fc_a1(x)))
            b = self.fc_b2(torch.relu(self.fc_b1(x)))
            return a + b

    model = BothHeavy(64, 32).eval().to("cuda")
    example_input = torch.randn(4, 64, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 0, f"Expected 0 matches, got {len(pairs)}"


def test_decode_selection_aux_config_flips_op_flag():
    """The transform's decode_selection_aux knob sets the op-internal flag.

    The knob performs no graph rewrite: the graph must be structurally
    unchanged apart from the (independent) pattern rewrites, and the
    sparse-attention op module flag must flip exactly when the knob is on.
    """
    import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as dsv4
    from tensorrt_llm._torch.auto_deploy.transform.interface import SharedConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_attn import (
        MultiStreamMLAAttn,
        MultiStreamMLAAttnConfig,
    )

    # Default off in code (yaml opts in).
    assert MultiStreamMLAAttnConfig(stage="compile").decode_selection_aux is False

    shared = SharedConfig()
    try:
        dsv4.set_decode_selection_aux(False)

        model = MockMLABlock(128, 64, 128).eval().to("cuda")
        gm = _build_gm(model, torch.randn(4, 128, device="cuda"))
        transform = MultiStreamMLAAttn.from_kwargs(stage="compile", decode_selection_aux=False)
        transform._apply(gm, None, None, shared)
        assert dsv4._DECODE_SELECTION_AUX is False, "knob off must not flip the flag"

        gm = _build_gm(model, torch.randn(4, 128, device="cuda"))
        transform = MultiStreamMLAAttn.from_kwargs(stage="compile", decode_selection_aux=True)
        transform._apply(gm, None, None, shared)
        assert dsv4._DECODE_SELECTION_AUX is True, "knob on must flip the flag"
    finally:
        dsv4.set_decode_selection_aux(False)
