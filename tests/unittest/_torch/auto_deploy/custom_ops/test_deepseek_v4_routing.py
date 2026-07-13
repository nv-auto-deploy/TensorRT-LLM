# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Parity test for the fused DeepSeek V4 sqrtsoftplus top-k routing custom op."""

import pytest
import torch
import torch.nn.functional as F

# Importing the module registers the auto_deploy::deepseek_v4_routing custom op.
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import deepseek_v4_routing  # noqa: F401


def _reference_routing(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """Byte-faithful copy of DeepseekV4MoEGate.forward (non-hash branch)."""
    scores = F.softplus(router_logits).sqrt()
    selected_experts = (scores + bias).topk(top_k, dim=-1).indices
    routing_weights = scores.gather(1, selected_experts)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * routed_scaling_factor
    return selected_experts, routing_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 2, 5, 17, 128])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("top_k", [6])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.5])
def test_deepseek_v4_routing_parity(
    num_tokens, num_experts, top_k, norm_topk_prob, routed_scaling_factor
):
    torch.manual_seed(num_tokens * 131 + num_experts + int(norm_topk_prob))
    device = "cuda"
    # Spread logits over a wide range to exercise both softplus branches (x>20 linear).
    router_logits = (torch.randn(num_tokens, num_experts, device=device) * 8.0).float()
    bias = (torch.randn(num_experts, device=device) * 0.5).float()

    ref_idx, ref_w = _reference_routing(
        router_logits, bias, top_k, routed_scaling_factor, norm_topk_prob
    )
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, routed_scaling_factor, norm_topk_prob
    )

    assert out_idx.dtype == torch.int64
    assert out_w.dtype == torch.float32
    assert out_idx.shape == (num_tokens, top_k)
    assert out_w.shape == (num_tokens, top_k)

    # Indices: random fp32 logits make ties measure-zero -> exact match expected.
    assert torch.equal(out_idx, ref_idx.to(torch.int64)), (
        f"index mismatch\nref={ref_idx}\nout={out_idx}"
    )
    # Weights: sqrtsoftplus done in fp32 in-register; expect tight numeric parity.
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_softplus_threshold():
    """Large logits (>20) take the softplus linear branch; verify parity there."""
    torch.manual_seed(0)
    device = "cuda"
    num_tokens, num_experts, top_k = 4, 256, 6
    # Push many logits well past the softplus threshold of 20.
    router_logits = (torch.randn(num_tokens, num_experts, device=device) * 5.0 + 18.0).float()
    bias = torch.zeros(num_experts, device=device).float()

    ref_idx, ref_w = _reference_routing(router_logits, bias, top_k, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx.to(torch.int64))
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_exact_tie_smallest_index():
    """Exact value ties: deterministic smallest-index selection (documented contract).

    A rank-6/rank-7 exact tie must resolve to the smaller expert index; a tie
    group wider than the remaining slots is masked as one (the kernel's
    documented all-tied-at-once masking), so only its smallest index appears.
    """
    device = "cuda"
    num_experts, top_k = 256, 6
    bias = torch.zeros(num_experts, device=device).float()

    # Case 1: experts 0..4 strictly descending winners; experts 100 and 200
    # share the exact rank-6 logit (bitwise equal) -> 100 must win the last slot.
    logits = torch.full((1, num_experts), -10.0, device=device)
    logits[0, :5] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0], device=device)
    logits[0, 100] = 4.0
    logits[0, 200] = 4.0
    outs = [
        torch.ops.auto_deploy.deepseek_v4_routing(logits, bias, top_k, 1.5, True) for _ in range(10)
    ]
    for idx, _ in outs[1:]:
        assert torch.equal(idx, outs[0][0]), "tie selection must be deterministic"
    idx = outs[0][0][0].tolist()
    assert idx[:5] == [0, 1, 2, 3, 4]
    assert idx[5] == 100, f"smallest tied index must win the boundary slot, got {idx[5]}"

    # Case 2: 3-way tie (experts 50/150/250) across ranks 5-7 with 2 slots left.
    # The kernel masks the whole tied group after selecting its smallest index
    # (documented v5 semantics), so exactly one group member is selected.
    logits2 = torch.full((1, num_experts), -10.0, device=device)
    logits2[0, :4] = torch.tensor([9.0, 8.0, 7.0, 6.0], device=device)
    for e in (50, 150, 250):
        logits2[0, e] = 4.0
    logits2[0, 10] = 3.0  # next-best distinct value fills the final slot
    idx2 = torch.ops.auto_deploy.deepseek_v4_routing(logits2, bias, top_k, 1.5, True)[0][0].tolist()
    assert idx2[:4] == [0, 1, 2, 3]
    assert idx2[4] == 50, "smallest index of the tied group must be selected"
    assert idx2[5] == 10, "remaining tied members are masked with the group"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_near_tie_ulp_selection():
    """Rank-6/rank-7 logits a few ULPs apart: deterministic, matches reference.

    The strictly-greater score must win regardless of index order (larger
    score deliberately placed at the LARGER index); any kernel-vs-reference
    flip fails loudly here rather than being normalized away.
    """
    device = "cuda"
    num_experts, top_k = 256, 6
    bias = torch.zeros(num_experts, device=device).float()

    for ulps in (4, 8):
        logits = torch.full((1, num_experts), -10.0, device=device)
        logits[0, :5] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0], device=device)
        base = torch.tensor(4.0, device=device)
        stepped = base
        for _ in range(ulps):
            stepped = torch.nextafter(stepped, torch.tensor(float("inf"), device=device))
        logits[0, 100] = base
        logits[0, 200] = stepped  # larger score at the LARGER index
        ref_idx, _ = _reference_routing(logits, bias, top_k, 1.5, True)
        for _ in range(10):
            out_idx, _ = torch.ops.auto_deploy.deepseek_v4_routing(logits, bias, top_k, 1.5, True)
            assert torch.equal(out_idx, ref_idx.to(torch.int64)), (
                f"near-tie ({ulps} ulps) selection flip: ref={ref_idx}, out={out_idx}"
            )
        assert out_idx[0, 5].item() == 200, "strictly-greater score must win the boundary slot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_pdl_bitexact_and_graph():
    """AD_GATE_PDL on/off must be bit-identical (pure load reorder) and the PDL
    launch must survive CUDA-graph capture/replay."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import (
        deepseek_v4_routing as routing_mod,
    )

    torch.manual_seed(3)
    device = "cuda"
    num_tokens, num_experts, top_k = 2, 256, 6
    logits = (torch.randn(num_tokens, num_experts, device=device) * 8.0).float()
    bias = (torch.randn(num_experts, device=device) * 0.5).float()

    old = routing_mod._AD_GATE_PDL
    try:
        routing_mod._AD_GATE_PDL = False
        idx_off, w_off = routing_mod.deepseek_v4_routing_fn(logits, bias, top_k, 1.5, True)
        routing_mod._AD_GATE_PDL = True
        idx_on, w_on = routing_mod.deepseek_v4_routing_fn(logits, bias, top_k, 1.5, True)
        assert torch.equal(idx_off, idx_on)
        assert torch.equal(w_off, w_on)

        # Capture the PDL-launched GEMV+routing chain and replay with new data.
        x = torch.randn(1, 4096, device=device, dtype=torch.float32)
        W = (torch.randn(num_experts, 4096, device=device) * 0.05).float()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for _ in range(3):  # warmup on the side stream
                logits_g = torch.ops.auto_deploy.deepseek_v4_gate_gemv(x, W, None)
                out_g = routing_mod.deepseek_v4_routing_fn(logits_g, bias, top_k, 1.5, True)
            stream.synchronize()
            with torch.cuda.graph(graph):
                logits_g = torch.ops.auto_deploy.deepseek_v4_gate_gemv(x, W, None)
                out_g = routing_mod.deepseek_v4_routing_fn(logits_g, bias, top_k, 1.5, True)
        torch.cuda.current_stream().wait_stream(stream)

        for seed in (11, 12):
            torch.manual_seed(seed)
            x.copy_(torch.randn_like(x))
            graph.replay()
            torch.cuda.synchronize()
            logits_e = torch.ops.auto_deploy.deepseek_v4_gate_gemv(x, W, None)
            idx_e, w_e = routing_mod.deepseek_v4_routing_fn(logits_e, bias, top_k, 1.5, True)
            assert torch.equal(out_g[0], idx_e), "graph replay selection mismatch"
            assert torch.equal(out_g[1], w_e), "graph replay weights mismatch"
    finally:
        routing_mod._AD_GATE_PDL = old


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_gate_gemv_selection_parity_random():
    """Triton gate GEMV vs the per-token cuBLAS M=1 reference over >=10k tokens.

    The GEMV changes fp32 sum order (~1e-6 logits), so this asserts ZERO expert
    selection flips on random tokens — any flip fails loudly (the transform
    installing the op stays default-off precisely because this is not provable).
    """
    torch.manual_seed(0)
    device = "cuda"
    num_tokens, num_experts, hidden, top_k = 10000, 256, 4096, 6
    W = (torch.randn(num_experts, hidden, device=device) * 0.05).float()
    bias = (torch.randn(num_experts, device=device) * 0.5).float()
    X = torch.randn(num_tokens, hidden, device=device, dtype=torch.float32)

    logits_ref = torch.empty(num_tokens, num_experts, device=device)
    logits_tri = torch.empty(num_tokens, num_experts, device=device)
    for t in range(num_tokens):
        x_t = X[t : t + 1]
        logits_ref[t] = F.linear(x_t, W)  # cuBLAS gemv at M=1 (production reference)
        logits_tri[t] = torch.ops.auto_deploy.deepseek_v4_gate_gemv(x_t, W, None)

    torch.testing.assert_close(logits_tri, logits_ref, rtol=1e-5, atol=1e-5)
    idx_ref, w_ref = torch.ops.auto_deploy.deepseek_v4_routing(logits_ref, bias, top_k, 1.5, True)
    idx_tri, w_tri = torch.ops.auto_deploy.deepseek_v4_routing(logits_tri, bias, top_k, 1.5, True)
    flips = (idx_ref != idx_tri).any(dim=-1)
    assert not flips.any(), (
        f"expert selection flipped on {int(flips.sum())}/{num_tokens} tokens; "
        f"first flip token {int(flips.nonzero()[0])}"
    )
    torch.testing.assert_close(w_tri, w_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_gate_gemv_duplicate_row_ties():
    """Duplicate router weight rows: both backends compute bit-equal logits for
    the duplicated rows (deterministic per-row reductions), so the exact tie
    resolves to the smallest index on BOTH paths — no flip possible."""
    torch.manual_seed(1)
    device = "cuda"
    num_experts, hidden, top_k = 256, 4096, 6
    W = (torch.randn(num_experts, hidden, device=device) * 0.05).float()
    # Make three exact duplicates of a strong row at spread-out indices.
    W[97] = W[13]
    W[201] = W[13]
    bias = torch.zeros(num_experts, device=device).float()

    for seed in range(20):
        torch.manual_seed(100 + seed)
        x = torch.randn(1, hidden, device=device, dtype=torch.float32) + 0.2
        logits_ref = F.linear(x, W)
        logits_tri = torch.ops.auto_deploy.deepseek_v4_gate_gemv(x, W, None)
        # Bit-equal logits among duplicated rows within each backend.
        assert logits_ref[0, 13] == logits_ref[0, 97] == logits_ref[0, 201]
        assert logits_tri[0, 13] == logits_tri[0, 97] == logits_tri[0, 201]
        idx_ref, _ = torch.ops.auto_deploy.deepseek_v4_routing(logits_ref, bias, top_k, 1.5, True)
        idx_tri, _ = torch.ops.auto_deploy.deepseek_v4_routing(logits_tri, bias, top_k, 1.5, True)
        assert torch.equal(idx_ref, idx_tri), f"tie flip: ref={idx_ref}, tri={idx_tri}"
        sel = idx_tri[0].tolist()
        # If any duplicate made top-k, it must be the smallest index (13) only.
        assert 97 not in sel and 201 not in sel, f"non-smallest tied index selected: {sel}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_gate_gemv_fallback_paths():
    """Off the single-token fp32 bias-free contract the op must be bit-identical
    to F.linear (it IS F.linear): multi-token, bf16 input, and bias cases."""
    torch.manual_seed(2)
    device = "cuda"
    W = (torch.randn(256, 4096, device=device) * 0.05).float()
    # Multi-token decode / prefill shape.
    x4 = torch.randn(4, 4096, device=device, dtype=torch.float32)
    assert torch.equal(torch.ops.auto_deploy.deepseek_v4_gate_gemv(x4, W, None), F.linear(x4, W))
    # 3-D prefill shape.
    x3d = torch.randn(2, 8, 4096, device=device, dtype=torch.float32)
    assert torch.equal(torch.ops.auto_deploy.deepseek_v4_gate_gemv(x3d, W, None), F.linear(x3d, W))
    # Non-fp32 input falls back.
    xb = torch.randn(1, 4096, device=device, dtype=torch.bfloat16)
    Wb = W.to(torch.bfloat16)
    assert torch.equal(torch.ops.auto_deploy.deepseek_v4_gate_gemv(xb, Wb, None), F.linear(xb, Wb))
    # Bias falls back.
    b = torch.randn(256, device=device, dtype=torch.float32)
    x1 = torch.randn(1, 4096, device=device, dtype=torch.float32)
    assert torch.equal(torch.ops.auto_deploy.deepseek_v4_gate_gemv(x1, W, b), F.linear(x1, W, b))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_fp32_mirror_input_parity():
    """Learned-router fp32-mirror contract for the HC-pre y32 output.

    ``deepseek_v4_hc_pre_mix_combine_partials_y32`` emits ``y32 == y.float()``
    bit-for-bit, so feeding the mirror to the unchanged fp32 router GEMV
    (``torch_linear_simple``) must reproduce the exact logits, selected expert
    ids, and routing weights of the old ``y.to(torch.float32)`` boundary-copy
    path in the same process.
    """
    import triton

    from tensorrt_llm._torch.auto_deploy.custom_ops import hc_composition
    from tensorrt_llm._torch.auto_deploy.custom_ops.linear import linear  # noqa: F401

    torch.manual_seed(17)
    B, S, hc_mult, H = 2, 1, 4, 4096
    num_experts, top_k = 256, 6
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"

    x = torch.randn(B, S, hc_mult * H, device=dev, dtype=torch.bfloat16)
    hc_fn = (torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02).contiguous()
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1
    norm_w = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    n, dim = B * S, hc_mult * H
    chunk, split = hc_composition.hc_partials_layout(dim)
    parts = torch.empty(n, mix_hc + 1, split, device=dev, dtype=torch.float32)
    hc_composition._hc_fn_partials_kernel[(n, split)](
        x.reshape(n, dim).contiguous(),
        hc_fn,
        parts,
        n,
        dim,
        split,
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        num_warps=4,
    )

    y, y32, _, _ = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
        parts, x, hc_fn, hc_scale, hc_base, norm_w, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )

    router_w = (torch.randn(num_experts, H, device=dev, dtype=torch.float32) * 0.05).contiguous()
    bias = (torch.randn(num_experts, device=dev) * 0.5).float()

    # Old gate path: bf16 -> fp32 boundary copy feeding the fp32 GEMV.
    logits_cast = torch.ops.auto_deploy.torch_linear_simple(
        y.reshape(-1, H).to(router_w.dtype), router_w, None
    ).float()
    # New gate path: the kernel-emitted fp32 mirror feeding the same GEMV.
    logits_mirror = torch.ops.auto_deploy.torch_linear_simple(
        y32.reshape(-1, H), router_w, None
    ).float()

    assert torch.equal(y32, y.float())
    assert torch.equal(logits_cast, logits_mirror)

    idx_c, w_c = torch.ops.auto_deploy.deepseek_v4_routing(logits_cast, bias, top_k, 1.5, True)
    idx_m, w_m = torch.ops.auto_deploy.deepseek_v4_routing(logits_mirror, bias, top_k, 1.5, True)
    assert torch.equal(idx_c, idx_m)
    assert torch.equal(w_c, w_m)
