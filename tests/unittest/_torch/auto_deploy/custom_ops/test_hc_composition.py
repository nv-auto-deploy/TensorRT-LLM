# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the fused DeepSeek-V4 HC composition Triton kernel.

``auto_deploy::hc_split_sinkhorn`` collapses the HC ``_hc_pre`` chain
(sigmoid + softmax + the ``sinkhorn_iters``-step alternating row/col normalize
loop) into a single Triton kernel. This test guards it against a pure-PyTorch
fp32 reference that mirrors the original ``_hc_split_sinkhorn`` line-for-line, so
the launch collapse cannot silently corrupt the doubly-stochastic ``comb``
matrix or the ``pre`` / ``post`` gates.
"""

import pytest
import torch

# Registers auto_deploy::hc_split_sinkhorn.
from tensorrt_llm._torch.auto_deploy.custom_ops import hc_composition  # noqa: F401


def _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Exact mirror of modeling_deepseek_v4._hc_split_sinkhorn (fp32)."""
    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    post_logits = mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]

    pre = torch.sigmoid(pre_logits) + eps
    post = 2.0 * torch.sigmoid(post_logits)
    comb = comb_logits.view(*comb_logits.shape[:-1], hc_mult, hc_mult)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 17), (1, 257)])
@pytest.mark.parametrize("hc_mult,iters", [(4, 20), (4, 1), (2, 5), (3, 8)])
def test_hc_split_sinkhorn_matches_reference(shape, hc_mult, iters):
    torch.manual_seed(0)
    eps = 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    mixes = torch.randn(*shape, mix_hc, device=dev, dtype=torch.float32)
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1

    pre_r, post_r, comb_r = _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    pre, post, comb = torch.ops.auto_deploy.hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, iters, eps
    )

    assert pre.shape == pre_r.shape and post.shape == post_r.shape and comb.shape == comb_r.shape
    assert pre.dtype == torch.float32 and comb.dtype == torch.float32

    torch.testing.assert_close(pre, pre_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(post, post_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(comb, comb_r, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_split_sinkhorn_doubly_stochastic():
    """After enough sinkhorn iters, comb is ~doubly stochastic (sanity)."""
    torch.manual_seed(1)
    hc_mult, iters, eps = 4, 20, 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(8, mix_hc, device="cuda", dtype=torch.float32)
    hc_scale = torch.ones(3, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(mix_hc, device="cuda", dtype=torch.float32)
    _, _, comb = torch.ops.auto_deploy.hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, iters, eps
    )
    # Columns sum to ~1 after the final col-normalize step.
    col_sums = comb.sum(dim=-2)
    torch.testing.assert_close(col_sums, torch.ones_like(col_sums), rtol=0, atol=1e-3)
