# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the fused DeepSeek-V4 HC ``_hc_post`` Triton kernel.

``auto_deploy::deepseek_v4_hc_post`` collapses the ``_hc_post`` residual-stream
composition (two broadcast muls -> M-axis ``sum`` -> ``add`` -> bf16 ``cast``)
into a single launch. This test guards it against a pure-torch reference that
mirrors the original ``_hc_post`` body line-for-line.

The eager reference's ``torch.sum(..., dim=2)`` is an *absolute* axis, so the
math is rank-sensitive: the model always calls ``_hc_post`` with a 3D
``x = [B, S, H]`` (and 4D ``residual = [B, S, hc_mult, H]``), summing over
``comb``'s first ``hc_mult`` axis. The tests therefore exercise exactly that
rank, covering decode (``S == 1``) and prefill (``S > 1``).
"""

import pytest
import torch

# Registers auto_deploy::deepseek_v4_hc_post.
from tensorrt_llm._torch.auto_deploy.custom_ops import deepseek_v4_hc_post  # noqa: F401


def _ref_hc_post(x, residual, post, comb):
    """Exact mirror of DeepseekV4Block._hc_post (modeling_deepseek_v4.py)."""
    y = post.unsqueeze(-1) * x.unsqueeze(-2)
    y = y + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.to(x.dtype)


def _make_inputs(B, S, hc_mult, H, dev="cuda", dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, S, H, device=dev, dtype=dtype)
    residual = torch.randn(B, S, hc_mult, H, device=dev, dtype=dtype)
    post = 2.0 * torch.sigmoid(torch.randn(B, S, hc_mult, device=dev, dtype=torch.float32))
    # comb: doubly-stochastic-ish, like the sinkhorn output (rows ~ sum to 1).
    comb = torch.randn(B, S, hc_mult, hc_mult, device=dev, dtype=torch.float32).softmax(dim=-1)
    return x, residual, post, comb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 1), (2, 1), (1, 1000), (2, 3), (3, 17), (1, 257)])
@pytest.mark.parametrize("hc_mult", [4, 2, 3])
@pytest.mark.parametrize("H", [4096, 2048, 512, 127])
def test_hc_post_matches_reference(BS, hc_mult, H):
    B, S = BS
    x, residual, post, comb = _make_inputs(B, S, hc_mult, H)

    ref = _ref_hc_post(x, residual, post, comb)
    out = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)

    assert out.shape == ref.shape == (B, S, hc_mult, H)
    assert out.dtype == x.dtype
    # bf16 output; only the fp32 reduction association differs from the
    # reference. Allow a couple of bf16 ULP (assert_close bf16 default rtol is
    # 1.6e-2; bump atol slightly to absorb rare LSB flips near rounding bounds).
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_post_expanded_residual_layer0():
    """Layer 0 passes a stride-0 expand as residual; the op must handle it."""
    B, S, hc_mult, H = 2, 5, 4, 4096
    torch.manual_seed(1)
    x = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    # residual = hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1) -> stride 0 on dim 2.
    residual = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    residual = residual.unsqueeze(2).expand(-1, -1, hc_mult, -1)
    assert not residual.is_contiguous()
    post = 2.0 * torch.sigmoid(torch.randn(B, S, hc_mult, device="cuda", dtype=torch.float32))
    comb = torch.randn(B, S, hc_mult, hc_mult, device="cuda", dtype=torch.float32).softmax(dim=-1)

    ref = _ref_hc_post(x, residual, post, comb)
    out = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_post_zero_comb_zero_post_identity():
    """post == 0 and comb == 0 -> output all zeros (sanity on the accumulation)."""
    B, S, hc_mult, H = 2, 3, 4, 512
    torch.manual_seed(2)
    x = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(B, S, hc_mult, H, device="cuda", dtype=torch.bfloat16)
    post = torch.zeros(B, S, hc_mult, device="cuda", dtype=torch.float32)
    comb = torch.zeros(B, S, hc_mult, hc_mult, device="cuda", dtype=torch.float32)
    out = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)
    assert torch.count_nonzero(out) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_post_identity_comb_passthrough():
    """comb == I, post == 0 -> output stream o == residual stream o (per-row identity)."""
    B, S, hc_mult, H = 1, 4, 4, 2048
    torch.manual_seed(3)
    x = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(B, S, hc_mult, H, device="cuda", dtype=torch.bfloat16)
    post = torch.zeros(B, S, hc_mult, device="cuda", dtype=torch.float32)
    eye = torch.eye(hc_mult, device="cuda", dtype=torch.float32)
    comb = eye.expand(B, S, hc_mult, hc_mult).contiguous()
    out = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)
    # y[o] = sum_m I[m,o] * residual[m] = residual[o].
    torch.testing.assert_close(out, residual, rtol=1.6e-2, atol=8e-3)
