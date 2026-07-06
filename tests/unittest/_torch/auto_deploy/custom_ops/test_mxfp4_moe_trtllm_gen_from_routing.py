# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity checks: trtllm-gen MXFP4 MoE from-routing path (W4A8 mxfp8-act default, W4A16
bf16-act fallback) vs torch reference.

Covers the DSV4-Flash routed MXFP4 MLP on Blackwell (SM100). The trtllm-gen runners
(``mxe4m3_mxe2m1_block_scale_moe_runner`` for ``act_dtype="mxfp8"`` — the default — and
``bf16_mxe2m1_block_scale_moe_runner`` for ``act_dtype="bf16"``) replace the fp32 dequant+bmm
reference for the ``up_gate``/``deepseek`` signature. This validates the integration details the
idea_0008 microbench did NOT exercise: re-interleave of the ``[up|gate]`` split-half layout,
remote-route masking under EP (some routes off-rank, one token fully remote must be exactly
zero), and NON-ZERO expert biases. Output is a reduced-precision kernel vs fp32-reference, so we
assert high cosine similarity (not bit-exactness); the W4A8-vs-W4A16 check isolates the
activation-quantization error with a tighter bound.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy._compat import is_sm_100f
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _make_case(batch: int, dev: str = "cuda"):
    """Scaled-down DSV4-Flash MoE shapes (per-rank, EP) with EP-remote routes."""
    torch.manual_seed(0)
    H = 512
    INTER = 256
    E_LOCAL = 8
    TOP_K = 4
    EXPERT_START = 8  # rank>0: routing spans local [8,16) and remote experts -> exercises masking
    BLK = 32
    GU = 2 * INTER

    gu_blocks = torch.randint(0, 256, (E_LOCAL, GU, H // BLK, 16), dtype=torch.uint8, device=dev)
    gu_scales = torch.randint(124, 131, (E_LOCAL, GU, H // BLK), dtype=torch.uint8, device=dev)
    gu_bias = (torch.randn(E_LOCAL, GU, device=dev) * 0.05).to(torch.float32)
    dn_blocks = torch.randint(0, 256, (E_LOCAL, H, INTER // BLK, 16), dtype=torch.uint8, device=dev)
    dn_scales = torch.randint(124, 131, (E_LOCAL, H, INTER // BLK), dtype=torch.uint8, device=dev)
    dn_bias = (torch.randn(E_LOCAL, H, device=dev) * 0.05).to(torch.float32)

    x = torch.randn(batch, H, dtype=torch.bfloat16, device=dev) * 0.1
    # Global expert ids spanning [0, 4*E_LOCAL): most routes are remote (masked off-rank). Pin the
    # first route per token to a local expert so each token has a nonzero output (a fully-remote
    # token is a correct but degenerate all-zero case that makes cosine ill-defined) — except the
    # last token when batch > 1, which is forced FULLY remote to assert the exact-zero invariant
    # EP partial-sum correctness depends on.
    sel = torch.randint(0, 4 * E_LOCAL, (batch, TOP_K), dtype=torch.int32, device=dev)
    sel[:, 0] = torch.randint(EXPERT_START, EXPERT_START + E_LOCAL, (batch,), device=dev)
    if batch > 1:
        sel[-1] = torch.randint(0, EXPERT_START, (TOP_K,), device=dev)
    rw = torch.softmax(torch.randn(batch, TOP_K, device=dev), dim=-1) * 1.5  # mimic routed_scaling

    ALPHA, LIMIT = 1.0, 10.0  # DSV4: alpha=1.0, swiglu_limit=10.0 (deepseek SwiGLU with clamp)
    weights = (gu_blocks, gu_bias, gu_scales, dn_blocks, dn_bias, dn_scales)
    return x, sel, rw, weights, EXPERT_START, ALPHA, LIMIT, H


def _torch_ref(x, sel, rw, weights, expert_start, alpha, limit):
    """Torch reference (force the SM100 gate off so we hit the fp32 dequant+bmm path)."""
    gu_blocks, gu_bias, gu_scales, dn_blocks, dn_bias, dn_scales = weights
    orig = mxfp4_moe.is_sm_100f
    mxfp4_moe.is_sm_100f = lambda *a, **k: False
    try:
        return mxfp4_moe._run_torch_mxfp4_from_routing_core(
            x,
            sel,
            rw,
            gu_blocks,
            gu_bias,
            gu_scales,
            alpha,
            limit,
            dn_blocks,
            dn_bias,
            dn_scales,
            expert_start,
            "up_gate",
            "deepseek",
        )
    finally:
        mxfp4_moe.is_sm_100f = orig


def _trtllm_gen(x, sel, rw, weights, expert_start, alpha, limit, act_dtype):
    gu_blocks, gu_bias, gu_scales, dn_blocks, dn_bias, dn_scales = weights
    return mxfp4_moe._run_trtllm_gen_mxfp4_from_routing(
        x,
        sel,
        rw,
        gu_blocks,
        gu_bias,
        gu_scales,
        dn_blocks,
        dn_bias,
        dn_scales,
        expert_start,
        alpha,
        limit,
        act_dtype=act_dtype,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not is_sm_100f(), reason="trtllm-gen MXFP4 runners require SM100 (Blackwell)")
@pytest.mark.parametrize("act_dtype", ["mxfp8", "bf16"])
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_trtllm_gen_mxfp4_from_routing_matches_torch_ref(batch, act_dtype):
    x, sel, rw, weights, expert_start, alpha, limit, H = _make_case(batch)
    out_ref = _torch_ref(x, sel, rw, weights, expert_start, alpha, limit)
    out_trt = _trtllm_gen(x, sel, rw, weights, expert_start, alpha, limit, act_dtype)

    assert out_trt.shape == out_ref.shape == (batch, H)
    assert out_trt.dtype == x.dtype
    assert torch.isfinite(out_trt).all()
    # EP masking must survive the runner swap exactly: a fully-remote token contributes a
    # zero partial (any nonzero here would double-count through the EP all_reduce).
    if batch > 1:
        assert (out_ref[-1] == 0).all(), "torch ref: fully-remote token must be exactly zero"
        assert (out_trt[-1] == 0).all(), (
            f"{act_dtype}: fully-remote token must be exactly zero (EP partial-sum invariant)"
        )
    cos = _cos(out_trt, out_ref)
    # Reduced-precision kernel vs fp32 reference on pathological random mxfp4 weights; real
    # weights are tighter. Same bar for both act dtypes: the mxfp8 activation-quant error is
    # small relative to the bf16-vs-fp32 gap at this scale (see the W4A8-vs-W4A16 test below).
    assert cos > 0.85, (
        f"trtllm-gen[{act_dtype}] vs torch-ref cosine too low: {cos:.4f} (batch={batch})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(not is_sm_100f(), reason="trtllm-gen MXFP4 runners require SM100 (Blackwell)")
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_trtllm_gen_mxfp4_from_routing_w4a8_close_to_w4a16(batch):
    """Isolate the activation-quantization delta: W4A8 (mxfp8 act) vs W4A16 (bf16 act) on the
    SAME prepared weights, routing localization, and top-k ordering. The only difference is the
    MXFP8 quantization of the activations, so the bound is much tighter than vs the fp32 ref."""
    x, sel, rw, weights, expert_start, alpha, limit, H = _make_case(batch)
    out_bf16 = _trtllm_gen(x, sel, rw, weights, expert_start, alpha, limit, "bf16")
    out_mxfp8 = _trtllm_gen(x, sel, rw, weights, expert_start, alpha, limit, "mxfp8")

    assert out_mxfp8.shape == out_bf16.shape == (batch, H)
    assert torch.isfinite(out_mxfp8).all()
    cos = _cos(out_mxfp8, out_bf16)
    assert cos > 0.98, f"W4A8 vs W4A16 cosine too low: {cos:.4f} (batch={batch})"


if __name__ == "__main__":
    for b in (1, 2, 8):
        for a in ("mxfp8", "bf16"):
            test_trtllm_gen_mxfp4_from_routing_matches_torch_ref(b, a)
            print(f"batch={b} act={a}: OK")
        test_trtllm_gen_mxfp4_from_routing_w4a8_close_to_w4a16(b)
        print(f"batch={b} w4a8-vs-w4a16: OK")
