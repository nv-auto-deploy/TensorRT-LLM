# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity check: trtllm-gen W4A16 (bf16-act) MXFP4 MoE from-routing path vs torch reference.

Covers the DSV4-Flash routed MXFP4 MLP on Blackwell (SM100). The trtllm-gen runner
(``bf16_mxe2m1_block_scale_moe_runner``) replaces the fp32 dequant+bmm reference for the
``up_gate``/``deepseek`` signature. This validates the integration details the idea_0008
microbench did NOT exercise: re-interleave of the ``[up|gate]`` split-half layout, remote-route
masking under EP (some routes off-rank), and NON-ZERO expert biases. Output is bf16-kernel vs
fp32-reference, so we assert high cosine similarity (not bit-exactness).
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy._compat import is_sm_100f
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import mxfp4_moe


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not is_sm_100f(), reason="trtllm-gen W4A16 MXFP4 runner requires SM100 (Blackwell)"
)
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_trtllm_gen_mxfp4_from_routing_matches_torch_ref(batch):
    torch.manual_seed(0)
    dev = "cuda"
    # Scaled-down DSV4-Flash MoE shapes (per-rank, EP).
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
    # token is a correct but degenerate all-zero case that makes cosine ill-defined).
    sel = torch.randint(0, 4 * E_LOCAL, (batch, TOP_K), dtype=torch.int32, device=dev)
    sel[:, 0] = torch.randint(EXPERT_START, EXPERT_START + E_LOCAL, (batch,), device=dev)
    rw = torch.softmax(torch.randn(batch, TOP_K, device=dev), dim=-1) * 1.5  # mimic routed_scaling

    ALPHA, LIMIT = 1.0, 10.0  # DSV4: alpha=1.0, swiglu_limit=10.0 (deepseek SwiGLU with clamp)

    # torch reference (force the gate off so we hit the fp32 dequant+bmm path)
    orig = mxfp4_moe.is_sm_100f
    mxfp4_moe.is_sm_100f = lambda *a, **k: False
    try:
        out_ref = mxfp4_moe._run_torch_mxfp4_from_routing_core(
            x,
            sel,
            rw,
            gu_blocks,
            gu_bias,
            gu_scales,
            ALPHA,
            LIMIT,
            dn_blocks,
            dn_bias,
            dn_scales,
            EXPERT_START,
            "up_gate",
            "deepseek",
        )
    finally:
        mxfp4_moe.is_sm_100f = orig

    out_trt = mxfp4_moe._run_trtllm_gen_mxfp4_from_routing(
        x,
        sel,
        rw,
        gu_blocks,
        gu_bias,
        gu_scales,
        dn_blocks,
        dn_bias,
        dn_scales,
        EXPERT_START,
        ALPHA,
        LIMIT,
    )

    assert out_trt.shape == out_ref.shape == (batch, H)
    assert out_trt.dtype == x.dtype
    assert torch.isfinite(out_trt).all()
    cos = _cos(out_trt, out_ref)
    # bf16 kernel vs fp32 reference on pathological random mxfp4 weights; real weights are tighter.
    assert cos > 0.85, f"trtllm-gen vs torch-ref cosine too low: {cos:.4f} (batch={batch})"


if __name__ == "__main__":
    for b in (1, 2, 8):
        test_trtllm_gen_mxfp4_from_routing_matches_torch_ref(b)
        print(f"batch={b}: OK")
