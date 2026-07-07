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

import contextlib

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


@contextlib.contextmanager
def _tf32_disabled():
    """Force true-fp32 torch matmul for the reference computation.

    At the captured decode shape (n == 1) cublas dispatches ``F.linear`` to a
    true-fp32 SIMT gemv; at n >= 2 it may pick a TF32 tensor-core GEMM
    (~1e-3 relative truncation). The fused Triton kernel is uniformly true
    fp32, so the reference must be computed with TF32 off to model the decode
    semantics at every test shape.
    """
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def _ref_hc_pre_mix(x, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps, norm_eps):
    """Exact mirror of the eager modeling _hc_pre front + _hc_split_sinkhorn."""
    flat = x.float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(flat, hc_fn) * rsqrt
    return _ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# (1, 1) / (2, 1) are the captured decode shapes; (4, 16) sits on the fused-path
# boundary (n == 64); all take the two-kernel split-D path.
@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (4, 16)])
@pytest.mark.parametrize("hc_mult,H", [(4, 4096), (4, 512), (2, 384), (3, 100)])
def test_hc_pre_mix_matches_eager_reference(shape, hc_mult, H):
    torch.manual_seed(0)
    eps, norm_eps, iters = 1e-6, 1e-6, 20
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    x = torch.randn(*shape, hc_mult * H, device=dev, dtype=torch.bfloat16)
    hc_fn = (torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02).contiguous()
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1

    with _tf32_disabled():
        pre_r, post_r, comb_r = _ref_hc_pre_mix(
            x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps
        )
    pre, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix(
        x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps
    )

    assert pre.shape == pre_r.shape and post.shape == post_r.shape and comb.shape == comb_r.shape
    assert pre.dtype == torch.float32 and comb.dtype == torch.float32

    # The split-D partial reduction orders the fp32 sums differently from
    # cublas/torch (a ~1 ULP effect on ``mixes``); sigmoid/sinkhorn are
    # contractive, so pre/post stay tight and comb matches the tolerance the
    # sinkhorn kernel itself needs vs eager.
    torch.testing.assert_close(pre, pre_r, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(post, post_r, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(comb, comb_r, rtol=1e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 200), (4, 17)])
def test_hc_pre_mix_large_n_fallback_bit_exact(shape):
    """n > 64 falls back to the eager torch front + production sinkhorn op.

    That path executes the exact same torch kernels (including today's cublas
    TF32 dispatch, if any) followed by ``hc_split_sinkhorn``, so its outputs
    must be bit-identical to composing those pieces by hand.
    """
    torch.manual_seed(1)
    hc_mult, H, iters, eps, norm_eps = 4, 512, 20, 1e-6, 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    x = torch.randn(*shape, hc_mult * H, device=dev, dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1

    n = shape[0] * shape[1]
    flat = x.reshape(n, -1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(flat, hc_fn) * rsqrt
    pre_r, post_r, comb_r = torch.ops.auto_deploy.hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, iters, eps
    )

    pre, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix(
        x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps
    )

    assert torch.equal(pre.reshape_as(pre_r), pre_r)
    assert torch.equal(post.reshape_as(post_r), post_r)
    assert torch.equal(comb.reshape_as(comb_r), comb_r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_pre_mix_deterministic():
    """No atomics: repeated fused-path calls must be bit-identical."""
    torch.manual_seed(3)
    hc_mult, H, iters, eps, norm_eps = 4, 4096, 20, 1e-6, 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(2, 1, hc_mult * H, device="cuda", dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, hc_mult * H, device="cuda", dtype=torch.float32) * 0.02
    hc_scale = torch.ones(3, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(mix_hc, device="cuda", dtype=torch.float32)

    outs = [
        torch.ops.auto_deploy.deepseek_v4_hc_pre_mix(
            x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps
        )
        for _ in range(2)
    ]
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (1, 200)])
def test_hc_pre_chain_end_to_end_vs_current(shape):
    """Full _hc_pre chain (new bf16 front + combine) vs the current HEAD path.

    The current production path materializes ``flat = x.float()`` and feeds the
    eager rsqrt/GEMV/mul front into ``hc_split_sinkhorn`` and the fp32-``flat``
    combine. The new path must reproduce y (bf16) / post / comb within the
    reduction-order tolerance of the fused GEMV.
    """
    from tensorrt_llm._torch.auto_deploy.custom_ops import deepseek_v4_hc_pre_norm  # noqa: F401

    torch.manual_seed(4)
    hc_mult, H, iters, eps, norm_eps = 4, 4096, 20, 1e-6, 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    x = torch.randn(*shape, hc_mult * H, device=dev, dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    # Current HEAD path. TF32 off for the fused-path shapes so the reference
    # GEMV models the true-fp32 SIMT gemv cublas uses at the decode shape
    # (n == 1); the n > 64 fallback keeps today's dispatch bit-exactly.
    n = shape[0] * shape[1]
    flat = x.float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    with _tf32_disabled() if n <= 64 else contextlib.nullcontext():
        mixes = torch.nn.functional.linear(flat, hc_fn) * rsqrt
    pre_r, post_r, comb_r = torch.ops.auto_deploy.hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, iters, eps
    )
    y_r = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre_r, flat, weight, norm_eps, hc_mult, torch.bfloat16
    )

    # New fused path (bf16 end to end).
    pre, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix(
        x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps
    )
    y = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, x, weight, norm_eps, hc_mult, torch.bfloat16
    )

    torch.testing.assert_close(pre, pre_r, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(post, post_r, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(comb, comb_r, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(y, y_r)  # bf16 default tolerances


def _two_op_reference(x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps):
    """The landed two-op HC-pre path the fused op must reproduce bit-for-bit."""
    from tensorrt_llm._torch.auto_deploy.custom_ops import deepseek_v4_hc_pre_norm  # noqa: F401

    pre, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix(
        x, hc_fn, hc_scale, hc_base, hc_mult, iters, eps, norm_eps
    )
    y = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, x, weight, rms_eps, hc_mult, x.dtype
    )
    return y, post, comb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# (1, 1) / (2, 1) are the captured decode shapes; (4, 16) sits on the fused-path
# boundary (n == 64); (1, 200) / (4, 17) take the n > 64 eager fallback.
@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (1, 3), (4, 16), (1, 200), (4, 17)])
@pytest.mark.parametrize("hc_mult,H", [(4, 4096), (4, 512), (2, 384), (3, 100)])
def test_hc_pre_mix_combine_bit_exact_vs_two_op_path(shape, hc_mult, H):
    """Fused composition+combine op vs the landed two-op sequence: torch.equal.

    ``rms_eps`` is deliberately distinct from ``norm_eps`` so an argument swap
    between the mix-statistic epsilon and the RMSNorm epsilon cannot pass.
    """
    torch.manual_seed(0)
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    x = torch.randn(*shape, hc_mult * H, device=dev, dtype=torch.bfloat16)
    hc_fn = (torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02).contiguous()
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    y_r, post_r, comb_r = _two_op_reference(
        x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps
    )
    y, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine(
        x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )

    assert y.shape == y_r.shape and y.dtype == x.dtype
    assert post.shape == post_r.shape and comb.shape == comb_r.shape
    assert post.dtype == torch.float32 and comb.dtype == torch.float32
    assert torch.equal(y, y_r)
    assert torch.equal(post, post_r)
    assert torch.equal(comb, comb_r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_hc_pre_mix_combine_bit_exact_non_bf16(dtype):
    """Non-bf16 residual dtypes route through the same bit-exact kernels."""
    torch.manual_seed(2)
    hc_mult, H, iters, eps, norm_eps, rms_eps = 4, 512, 20, 1e-6, 1e-6, 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    x = torch.randn(2, 1, hc_mult * H, device=dev, dtype=dtype)
    hc_fn = torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    y_r, post_r, comb_r = _two_op_reference(
        x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps
    )
    y, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine(
        x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )
    assert torch.equal(y, y_r)
    assert torch.equal(post, post_r)
    assert torch.equal(comb, comb_r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_pre_mix_combine_deterministic():
    """No atomics: repeated fused-path calls must be bit-identical."""
    torch.manual_seed(3)
    hc_mult, H, iters, eps, norm_eps, rms_eps = 4, 4096, 20, 1e-6, 1e-6, 1e-6
    mix_hc = (2 + hc_mult) * hc_mult
    x = torch.randn(2, 1, hc_mult * H, device="cuda", dtype=torch.bfloat16)
    hc_fn = torch.randn(mix_hc, hc_mult * H, device="cuda", dtype=torch.float32) * 0.02
    hc_scale = torch.ones(3, device="cuda", dtype=torch.float32)
    hc_base = torch.zeros(mix_hc, device="cuda", dtype=torch.float32)
    weight = torch.ones(H, device="cuda", dtype=torch.float32)

    outs = [
        torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine(
            x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
        )
        for _ in range(2)
    ]
    for a, b in zip(outs[0], outs[1]):
        assert torch.equal(a, b)


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
