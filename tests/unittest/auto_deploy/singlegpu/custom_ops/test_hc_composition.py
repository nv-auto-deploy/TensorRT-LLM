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

"""Fused DeepSeek-V4 hyper-connection custom ops vs the eager modeling chains."""

import pytest
import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops import deepseek_v4_hyper_connections as hc

# hc_split_sinkhorn ----------------------------------------------------------


def _ref_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps):
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
@pytest.mark.parametrize("shape", [(1, 1), (4, 17)])
@pytest.mark.parametrize("hc_mult,iters", [(4, 20), (4, 1), (3, 8)])
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


# Shared helpers -------------------------------------------------------------


def _make_partials(flat2d: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    # The standalone split-D partials launch (what the hc_post seam op emits).
    n, dim = flat2d.shape
    mix_hc = fn.shape[0]
    chunk, split = hc.hc_partials_layout(dim)
    partials = torch.empty(n, mix_hc + 1, split, device=flat2d.device, dtype=torch.float32)
    hc._hc_fn_partials_kernel[(n, split)](
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


def _assert_ulp_close(actual, ref, rtol, atol, what, max_diff_frac=None):
    # Magnitude at ULP scale; for bf16 outputs also bound the COUNT of flipped elements.
    torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol)
    if max_diff_frac is not None:
        n_diff = (actual != ref).sum().item()
        frac = n_diff / max(actual.numel(), 1)
        assert frac <= max_diff_frac, (
            f"{what}: {n_diff}/{actual.numel()} elements differ ({frac:.2e} > {max_diff_frac:.2e})"
        )


# deepseek_v4_hc_pre_mix_combine[_partials] ----------------------------------


def _two_op_reference(x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps):
    # Standalone partials launch feeding the partials-consuming op.
    parts = _make_partials(x.reshape(-1, x.shape[-1]), hc_fn)
    return torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
        parts, x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )


# (1, 1) = captured decode shape; (4, 16) = fused-path boundary (n == 64);
# (1, 200) = n > 64 eager fallback.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1), (4, 16), (1, 200)])
@pytest.mark.parametrize("hc_mult,H", [(4, 4096), (3, 100)])
def test_hc_pre_mix_combine_bit_exact_vs_two_op_path(shape, hc_mult, H):
    torch.manual_seed(0)
    # rms_eps != norm_eps so an epsilon-argument swap cannot pass.
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
    assert post.dtype == torch.float32 and comb.dtype == torch.float32
    assert torch.equal(y, y_r)
    assert torch.equal(post, post_r)
    assert torch.equal(comb, comb_r)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_hc_pre_mix_combine_bit_exact_non_bf16(dtype):
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


# deepseek_v4_hc_pre_mix_combine_partials_y32 --------------------------------


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
@pytest.mark.parametrize("shape", [(1, 1), (4, 16), (1, 200)])
def test_hc_pre_mix_combine_partials_y32_bit_exact(shape):
    # y32 must be the STORED (out_dtype-rounded) values widened to fp32, never the
    # kernel's pre-rounding fp32 accumulator.
    hc_mult, H = 4, 4096
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
def test_hc_pre_mix_combine_partials_y32_fp32_alias_free():
    # For fp32 x, y.float() could alias y; y32 must still be a fresh exact copy.
    hc_mult, H = 4, 512
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    x, hc_fn, hc_scale, hc_base, weight, parts = _y32_op_inputs(
        (2, 1), hc_mult, H, dtype=torch.float32
    )
    y, y32, post, comb = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
        parts, x, hc_fn, hc_scale, hc_base, weight, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )
    assert y32.dtype == torch.float32
    assert torch.equal(y32, y.float())
    assert y32.data_ptr() != y.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_pre_mix_combine_partials_y32_cudagraph_fresh_input_replay():
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

    # Replay on fresh inputs written into the captured buffers.
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


# deepseek_v4_hc_combine_rmsnorm ---------------------------------------------


def _ref_hc_combine_rmsnorm(pre, flat, weight, eps, hc_mult, out_dtype):
    # Mirror of the modeling _hc_pre tail + DeepseekV4RMSNorm, incl. both bf16 roundings.
    H = weight.shape[0]
    lead = pre.shape[:-1]
    original_shape = (*lead, hc_mult, H)
    y = torch.sum(pre.unsqueeze(-1) * flat.view(original_shape), dim=2)
    y = y.to(out_dtype)
    out = torch.empty_like(y)
    yf = y.to(torch.float32)
    variance = yf.pow(2).mean(-1, keepdim=True)
    yf = yf * torch.rsqrt(variance + eps)
    out.copy_(weight * yf.to(out.dtype))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1), (4, 17), (1, 1000)])
@pytest.mark.parametrize("hc_mult", [4, 3])
@pytest.mark.parametrize("H", [4096, 512, 127])
def test_hc_combine_rmsnorm_matches_reference(shape, hc_mult, H):
    torch.manual_seed(0)
    eps = 1e-6
    dev = "cuda"
    out_dtype = torch.bfloat16
    pre = torch.rand(*shape, hc_mult, device=dev, dtype=torch.float32) + 0.1
    x = torch.randn(*shape, hc_mult, H, device=dev, dtype=out_dtype)
    flat = x.flatten(2).float()
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    ref = _ref_hc_combine_rmsnorm(pre, flat, weight, eps, hc_mult, out_dtype)
    out = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, flat, weight, eps, hc_mult, out_dtype
    )

    assert out.shape == ref.shape
    assert out.dtype == out_dtype
    # A couple of bf16 ULP: reduction order / rsqrt differ from the reference.
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_combine_rmsnorm_bf16_input_bit_exact():
    # bf16 flat must be byte-identical to pre-cast fp32 flat (in-register cast is exact).
    torch.manual_seed(5)
    hc_mult, H, eps = 4, 4096, 1e-6
    dev = "cuda"
    pre = torch.rand(2, 3, hc_mult, device=dev, dtype=torch.float32) + 0.1
    x = torch.randn(2, 3, hc_mult * H, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    out_fp32 = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, x.float(), weight, eps, hc_mult, torch.bfloat16
    )
    out_bf16 = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, x, weight, eps, hc_mult, torch.bfloat16
    )
    assert torch.equal(out_bf16, out_fp32)


# deepseek_v4_hc_post_next_partials + consumers (layer-boundary seam) --------


def _make_seam_inputs(B, S, hc_mult, H, mix_hc, dev="cuda", dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, S, H, device=dev, dtype=dtype)
    residual = torch.randn(B, S, hc_mult, H, device=dev, dtype=dtype)
    post = 2.0 * torch.sigmoid(torch.randn(B, S, hc_mult, device=dev, dtype=torch.float32))
    comb = torch.randn(B, S, hc_mult, hc_mult, device=dev, dtype=torch.float32).softmax(dim=-1)
    next_fn = 0.02 * torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32)
    return x, residual, post, comb, next_fn


def _ref_hc_post(x, residual, post, comb):
    # The standalone _hc_post_compose_kernel launch the seam op subsumes.
    lead = list(x.shape[:-1])
    H = x.shape[-1]
    hc_mult = post.shape[-1]
    n = 1
    for s in lead:
        n *= s
    x_f = x.contiguous().reshape(n, H)
    res_f = residual.contiguous().reshape(n, hc_mult, H)
    post_f = post.contiguous().reshape(n, hc_mult).float()
    comb_f = comb.contiguous().reshape(n, hc_mult * hc_mult).float()
    out = torch.empty((n, hc_mult, H), device=x.device, dtype=x.dtype)
    num_warps, num_stages, block_h_max, o_per_cta = hc._hc_post_launch_config(n, hc_mult)
    block_h = min(block_h_max, triton.next_power_of_2(H))
    grid = (n, triton.cdiv(hc_mult, o_per_cta), triton.cdiv(H, block_h))
    hc._hc_post_compose_kernel[grid](
        x_f,
        res_f,
        post_f,
        comb_f,
        out,
        n,
        H,
        HM=hc_mult,
        BM=triton.next_power_of_2(hc_mult),
        O_PER_CTA=o_per_cta,
        BLOCK_H=block_h,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out.reshape(*lead, hc_mult, H)


# (1, 64) sits on the fused-path boundary; (4, ..., 4) = head-fn seam.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 1), (1, 64)])
@pytest.mark.parametrize("hm_H_mix", [(4, 4096, 24), (3, 512, 15), (4, 4096, 4)])
def test_post_next_partials_ulp_close_decode(BS, hm_H_mix):
    B, S = BS
    hc_mult, H, mix_hc = hm_H_mix
    x, residual, post, comb, next_fn = _make_seam_inputs(B, S, hc_mult, H, mix_hc)

    out_ref = _ref_hc_post(x, residual, post, comb)
    out, parts = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
        x, residual, post, comb, next_fn
    )

    assert out.shape == out_ref.shape and out.dtype == out_ref.dtype
    # ptxas contracts fp32 mul+add chains into FMAs differently per kernel body:
    # ~1-2 fp32 ULP, so rare bf16 LSB flips whose count must stay ~per-mille.
    _assert_ulp_close(out, out_ref, rtol=1.6e-2, atol=1e-5, what="hc_post out", max_diff_frac=2e-3)

    # Seam invariant: emitted partials == standalone partials kernel on the tensor
    # the op actually stored (a bf16 tie-flip legitimately moves that chunk's dot).
    parts_ref = _make_partials(out.reshape(B * S, hc_mult * H), next_fn)
    assert parts.shape == parts_ref.shape
    _assert_ulp_close(parts, parts_ref, rtol=1e-5, atol=1e-5, what="partials")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 1), (1, 17)])
def test_seam_end_to_end_ulp_close_decode(BS):
    # post_next_partials -> pre_mix_combine_partials == hc_post -> pre_mix_combine.
    B, S = BS
    hc_mult, H, mix_hc = 4, 4096, 24
    x, residual, post, comb, next_fn = _make_seam_inputs(B, S, hc_mult, H, mix_hc)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = 0.02 * torch.randn(mix_hc, device="cuda", dtype=torch.float32)
    norm_w = 1.0 + 0.05 * torch.randn(H, device="cuda", dtype=torch.float32)

    out_ref = _ref_hc_post(x, residual, post, comb)
    y_ref, post_ref, comb_ref = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine(
        out_ref.flatten(2),
        next_fn,
        hc_scale,
        hc_base,
        norm_w,
        hc_mult,
        20,
        1e-4,
        1e-6,
        1e-6,
        torch.bfloat16,
    )

    out, parts = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
        x, residual, post, comb, next_fn
    )
    y, post2, comb2 = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
        parts,
        out.flatten(2),
        next_fn,
        hc_scale,
        hc_base,
        norm_w,
        hc_mult,
        20,
        1e-4,
        1e-6,
        1e-6,
        torch.bfloat16,
    )

    _assert_ulp_close(y, y_ref, rtol=1.6e-2, atol=1e-5, what="pre y", max_diff_frac=2e-3)
    _assert_ulp_close(post2, post_ref, rtol=1e-4, atol=1e-6, what="pre post")
    _assert_ulp_close(comb2, comb_ref, rtol=1e-4, atol=1e-6, what="pre comb")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_post_next_partials_prefill_matches_old():
    # Above the decode threshold the op must run the unchanged compose kernel, and the
    # prefill consumer must ignore the unfilled partials entirely.
    B, S = 1, 257
    hc_mult, H, mix_hc = 4, 4096, 24
    x, residual, post, comb, next_fn = _make_seam_inputs(B, S, hc_mult, H, mix_hc)

    out_ref = _ref_hc_post(x, residual, post, comb)
    out, parts = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
        x, residual, post, comb, next_fn
    )
    assert torch.equal(out, out_ref)
    _, split = hc.hc_partials_layout(hc_mult * H)
    assert parts.shape == (B * S, mix_hc + 1, split)

    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = 0.02 * torch.randn(mix_hc, device="cuda", dtype=torch.float32)
    norm_w = 1.0 + 0.05 * torch.randn(H, device="cuda", dtype=torch.float32)
    y_ref, post_ref, comb_ref = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine(
        out_ref.flatten(2),
        next_fn,
        hc_scale,
        hc_base,
        norm_w,
        hc_mult,
        20,
        1e-4,
        1e-6,
        1e-6,
        torch.bfloat16,
    )
    y, post2, comb2 = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
        parts,
        out.flatten(2),
        next_fn,
        hc_scale,
        hc_base,
        norm_w,
        hc_mult,
        20,
        1e-4,
        1e-6,
        1e-6,
        torch.bfloat16,
    )
    assert torch.equal(y, y_ref)
    assert torch.equal(post2, post_ref)
    assert torch.equal(comb2, comb_ref)


# deepseek_v4_hc_head_norm ----------------------------------------------------


def _ref_hc_head_norm(hidden4d, head_fn, head_scale, head_base, norm_w, hc_eps, norm_eps, rms_eps):
    # Exact mirror of the eager _hc_head + self.norm + .float() modeling tail.
    original_dtype = hidden4d.dtype
    original_shape = hidden4d.shape
    flat = hidden4d.flatten(2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = torch.nn.functional.linear(flat, head_fn) * rsqrt
    pre = torch.sigmoid(mixes * head_scale + head_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * flat.view(original_shape), dim=2)
    y = y.to(original_dtype)
    y = torch.ops.auto_deploy.torch_rmsnorm(y, norm_w, rms_eps)
    return y.float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 1), (1, 33)])
@pytest.mark.parametrize("hm_H", [(4, 4096), (4, 96)])
def test_hc_head_norm_decode(BS, hm_H):
    B, S = BS
    hc_mult, H = hm_H
    torch.manual_seed(1)
    hidden = torch.randn(B, S, hc_mult, H, device="cuda", dtype=torch.bfloat16)
    head_fn = 0.02 * torch.randn(hc_mult, hc_mult * H, device="cuda", dtype=torch.float32)
    head_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    head_base = torch.zeros(hc_mult, device="cuda", dtype=torch.float32)
    norm_w = 1.0 + 0.05 * torch.randn(H, device="cuda", dtype=torch.float32)

    ref = _ref_hc_head_norm(hidden, head_fn, head_scale, head_base, norm_w, 1e-4, 1e-6, 1e-6)
    parts = _make_partials(hidden.reshape(B * S, hc_mult * H), head_fn)
    out = torch.ops.auto_deploy.deepseek_v4_hc_head_norm(
        parts, hidden.flatten(2), head_fn, head_scale, head_base, norm_w, 1e-4, 1e-6, 1e-6
    )

    assert out.shape == ref.shape == (B, S, H)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=8e-3)
    # The store must widen exactly to bf16-representable values (absorbed .float()).
    assert torch.equal(out, out.to(torch.bfloat16).float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_head_norm_prefill_bitexact():
    B, S, hc_mult, H = 1, 257, 4, 4096
    torch.manual_seed(2)
    hidden = torch.randn(B, S, hc_mult, H, device="cuda", dtype=torch.bfloat16)
    head_fn = 0.02 * torch.randn(hc_mult, hc_mult * H, device="cuda", dtype=torch.float32)
    head_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    head_base = torch.zeros(hc_mult, device="cuda", dtype=torch.float32)
    norm_w = 1.0 + 0.05 * torch.randn(H, device="cuda", dtype=torch.float32)

    ref = _ref_hc_head_norm(hidden, head_fn, head_scale, head_base, norm_w, 1e-4, 1e-6, 1e-6)
    dummy = torch.empty(B * S, hc_mult + 1, 1, device="cuda", dtype=torch.float32)
    out = torch.ops.auto_deploy.deepseek_v4_hc_head_norm(
        dummy, hidden.flatten(2), head_fn, head_scale, head_base, norm_w, 1e-4, 1e-6, 1e-6
    )
    assert torch.equal(out, ref)
