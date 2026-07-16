# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the fused DeepSeek-V4 HC composition Triton kernel.

``auto_deploy::hc_split_sinkhorn`` collapses the HC ``_hc_pre`` chain
(sigmoid + softmax + the ``sinkhorn_iters``-step alternating row/col normalize
loop) into a single Triton kernel. This test guards it against a pure-PyTorch
fp32 reference that mirrors the original eager modeling helper line-for-line, so
the launch collapse cannot silently corrupt the doubly-stochastic ``comb``
matrix or the ``pre`` / ``post`` gates.
"""

import pytest
import torch
import triton

# Registers auto_deploy::hc_split_sinkhorn.
from tensorrt_llm._torch.auto_deploy.custom_ops import hc_composition  # noqa: F401


def _ref_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
    """Exact mirror of the eager modeling split + sinkhorn chain (fp32)."""
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

    pre_r, post_r, comb_r = _ref_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps)
    pre, post, comb = torch.ops.auto_deploy.hc_split_sinkhorn(
        mixes, hc_scale, hc_base, hc_mult, iters, eps
    )

    assert pre.shape == pre_r.shape and post.shape == post_r.shape and comb.shape == comb_r.shape
    assert pre.dtype == torch.float32 and comb.dtype == torch.float32

    torch.testing.assert_close(pre, pre_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(post, post_r, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(comb, comb_r, rtol=1e-3, atol=1e-5)


def _two_op_reference(x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps):
    """The two-op HC-pre path the fused op must reproduce bit-for-bit.

    A standalone split-D partials launch feeding the partials-consuming op
    (identical kernels on the decode path; the identical eager cublas front on
    the n > 64 prefill fallback, where the partials are ignored).
    """
    parts = _make_partials(x.reshape(-1, x.shape[-1]), hc_fn)
    return torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
        parts, x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )


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


def _make_partials(flat2d: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """The standalone split-D partials launch (what the hc_post seam op emits)."""
    n, dim = flat2d.shape
    mix_hc = fn.shape[0]
    chunk, split = hc_composition.hc_partials_layout(dim)
    partials = torch.empty(n, mix_hc + 1, split, device=flat2d.device, dtype=torch.float32)
    hc_composition._hc_fn_partials_kernel[(n, split)](
        flat2d.contiguous(),
        fn.contiguous().float(),
        partials,
        n,
        dim,
        split,
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        num_warps=4,
    )
    return partials


def _y32_op_inputs(shape, hc_mult, H, dtype=torch.bfloat16, seed=7):
    torch.manual_seed(seed)
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"
    x = torch.randn(*shape, hc_mult * H, device=dev, dtype=dtype)
    hc_fn = (torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02).contiguous()
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0
    parts = _make_partials(x.reshape(-1, hc_mult * H), hc_fn)
    return x, hc_fn, hc_scale, hc_base, weight, parts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# (1, 1) / (2, 1) are the captured decode shapes; (4, 16) sits on the fused-path
# boundary (n == 64); (1, 200) takes the n > 64 eager fallback.
@pytest.mark.parametrize("shape", [(1, 1), (2, 1), (4, 16), (1, 200)])
@pytest.mark.parametrize("hc_mult,H", [(4, 4096), (4, 512)])
def test_hc_pre_mix_combine_partials_y32_bit_exact(shape, hc_mult, H):
    """The y32 op must bit-match the 3-output op AND emit exactly y.float().

    ``y32`` must be the *stored* (out_dtype-rounded) values widened to fp32 —
    the same values the router's ``hidden_states.to(torch.float32)`` boundary
    copy used to produce — never the kernel's pre-rounding fp32 accumulator.
    """
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    x, hc_fn, hc_scale, hc_base, weight, parts = _y32_op_inputs(shape, hc_mult, H)
    args = (parts, x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps)

    y_r, post_r, comb_r = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
        *args, x.dtype
    )
    y, y32, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
        *args, x.dtype
    )

    assert y.shape == y_r.shape and y.dtype == y_r.dtype
    assert torch.equal(y, y_r)
    assert torch.equal(post, post_r)
    assert torch.equal(comb, comb_r)
    assert y32.dtype == torch.float32 and y32.shape == y.shape
    assert torch.equal(y32, y.float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_hc_pre_mix_combine_partials_y32_non_bf16(dtype):
    """Non-bf16 out dtypes: y32 is still an exact, alias-free fp32 widening."""
    hc_mult, H = 4, 512
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    x, hc_fn, hc_scale, hc_base, weight, parts = _y32_op_inputs((2, 1), hc_mult, H, dtype=dtype)
    y, y32, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
        parts, x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )
    y_r, post_r, comb_r = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
        parts, x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )
    assert torch.equal(y, y_r)
    assert torch.equal(post, post_r)
    assert torch.equal(comb, comb_r)
    assert y32.dtype == torch.float32
    assert torch.equal(y32, y.float())
    assert y32.data_ptr() != y.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_pre_mix_combine_partials_y32_cudagraph_fresh_input_replay():
    """Capture the y32 op in a CUDA graph, replay on fresh inputs.

    Outputs must equal a direct call on the same fresh inputs (the deployment mode).
    """
    shape, hc_mult, H = (2, 1), 4, 4096
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    x0, hc_fn, hc_scale, hc_base, weight, parts0 = _y32_op_inputs(shape, hc_mult, H, seed=11)
    flat_s = x0.clone()
    parts_s = parts0.clone()

    def call():
        return torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
            parts_s,
            flat_s,
            hc_fn,
            hc_scale,
            hc_base,
            weight,
            hc_mult,
            iters,
            eps,
            norm_eps,
            rms_eps,
            flat_s.dtype,
        )

    # Warm up (compiles the Triton kernel) on a side stream, then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            call()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_g, y32_g, post_g, comb_g = call()

    # Fresh activations against the SAME weights, written into the captured buffers.
    torch.manual_seed(23)
    x1 = torch.randn_like(x0)
    parts1 = _make_partials(x1.reshape(-1, hc_mult * H), hc_fn)
    flat_s.copy_(x1)
    parts_s.copy_(parts1)
    g.replay()
    torch.cuda.synchronize()

    y_e, y32_e, post_e, comb_e = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
        parts1,
        x1,
        hc_fn,
        hc_scale,
        hc_base,
        weight,
        hc_mult,
        iters,
        eps,
        norm_eps,
        rms_eps,
        x1.dtype,
    )
    assert torch.equal(y_g, y_e)
    assert torch.equal(y32_g, y32_e)
    assert torch.equal(post_g, post_e)
    assert torch.equal(comb_g, comb_e)
    assert torch.equal(y32_g, y_g.float())
