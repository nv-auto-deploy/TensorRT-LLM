# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the fused DeepSeek-V4 HC layer-boundary seam ops.

``auto_deploy::deepseek_v4_hc_post_next_partials`` fuses the ``_hc_post``
residual composition with the NEXT HC site's ``_hc_fn_partials_kernel``
launch, and ``auto_deploy::deepseek_v4_hc_pre_mix_combine_partials`` /
``auto_deploy::deepseek_v4_hc_head_norm`` consume those partials. The seam op
mirrors the landed two-kernel sequence on the decode path (same math, same
rounding points, one launch fewer); the head op mirrors the
eager ``_hc_head`` + ``torch_rmsnorm`` + ``.float()`` chain with partial-sum
``mixes`` that match cublas to ~1 ULP (same contract as the landed
``deepseek_v4_hc_pre_mix``).

ptxas contracts the fp32 mul+add chains into FMAs differently across kernel
bodies, so the seam op matches the two-kernel sequence to ~1-2 fp32 ULP, not
bit-for-bit (measured 1/16384 output elements and ~45/3200 partials at the
model shape, warp-count invariant). The tests below pin that contract: ULP-
scale tolerances plus a hard bound on the FRACTION of differing elements.
"""

import pytest
import torch
import triton

# Registers all auto_deploy HC ops (hc_composition imports deepseek_v4_hc_pre_norm).
from tensorrt_llm._torch.auto_deploy.custom_ops import hc_composition as hc_comp_mod


def _assert_ulp_close(actual, ref, rtol, atol, what, max_diff_frac=None):
    """Magnitude stays at ULP scale; optionally bound the COUNT of diffs.

    The count bound is meaningful only for bf16 outputs (rounding absorbs
    almost all fp32 association noise, so flips must stay ~per-mille); fp32
    tensors legitimately expose 1-ULP association diffs on many elements.
    """
    torch.testing.assert_close(actual, ref, rtol=rtol, atol=atol)
    if max_diff_frac is not None:
        n_diff = (actual != ref).sum().item()
        frac = n_diff / max(actual.numel(), 1)
        assert frac <= max_diff_frac, (
            f"{what}: {n_diff}/{actual.numel()} elements differ ({frac:.2e} > {max_diff_frac:.2e})"
        )


def _make_inputs(B, S, hc_mult, H, mix_hc, dev="cuda", dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, S, H, device=dev, dtype=dtype)
    residual = torch.randn(B, S, hc_mult, H, device=dev, dtype=dtype)
    post = 2.0 * torch.sigmoid(torch.randn(B, S, hc_mult, device=dev, dtype=torch.float32))
    comb = torch.randn(B, S, hc_mult, hc_mult, device=dev, dtype=torch.float32).softmax(dim=-1)
    next_fn = 0.02 * torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32)
    return x, residual, post, comb, next_fn


def _ref_partials(flat2d: torch.Tensor, fn: torch.Tensor) -> torch.Tensor:
    """Reference: the standalone ``_hc_fn_partials_kernel`` launch the fused op replaces."""
    n, dim = flat2d.shape
    mix_hc = fn.shape[0]
    chunk, split = hc_comp_mod.hc_partials_layout(dim)
    partials = torch.empty(n, mix_hc + 1, split, device=flat2d.device, dtype=torch.float32)
    hc_comp_mod._hc_fn_partials_kernel[(n, split)](
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 1), (2, 1), (4, 3), (1, 64)])
@pytest.mark.parametrize(
    "hm_H_mix",
    [(4, 4096, 24), (4, 96, 24), (3, 512, 15), (4, 4096, 4)],  # (4, ..., 4) = head-fn seam
)
def test_post_next_partials_ulp_close_decode(BS, hm_H_mix):
    B, S = BS
    hc_mult, H, mix_hc = hm_H_mix
    x, residual, post, comb, next_fn = _make_inputs(B, S, hc_mult, H, mix_hc)

    out_ref = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)

    out, parts = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
        x, residual, post, comb, next_fn
    )

    assert out.shape == out_ref.shape and out.dtype == out_ref.dtype
    # A rare fp32 FMA-association tie flips bf16 LSBs (<= ~2 bf16 ULP
    # relative); the count of affected elements must stay ~per-mille.
    _assert_ulp_close(out, out_ref, rtol=1.6e-2, atol=1e-5, what="hc_post out", max_diff_frac=2e-3)

    # THE seam invariant: the emitted partials must equal what the standalone
    # partials kernel computes from the tensor the op actually stored (the
    # downstream composition consumes exactly that tensor). Reference off the
    # fused ``out``, not ``out_ref`` — a bf16 tie-flip in ``out`` legitimately
    # moves the affected chunk's dot. Remaining drift is association-only.
    parts_ref = _ref_partials(out.reshape(B * S, hc_mult * H), next_fn)
    assert parts.shape == parts_ref.shape
    _assert_ulp_close(parts, parts_ref, rtol=1e-5, atol=1e-5, what="partials")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 1), (2, 1), (1, 17)])
def test_seam_end_to_end_ulp_close_decode(BS):
    """post_next_partials -> pre_mix_combine_partials == hc_post -> pre_mix_combine."""
    B, S = BS
    hc_mult, H, mix_hc = 4, 4096, 24
    x, residual, post, comb, next_fn = _make_inputs(B, S, hc_mult, H, mix_hc)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = 0.02 * torch.randn(mix_hc, device="cuda", dtype=torch.float32)
    norm_w = 1.0 + 0.05 * torch.randn(H, device="cuda", dtype=torch.float32)

    out_ref = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)
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

    # The composition kernel is unchanged; only its partials input carries
    # ~ULP drift. The sinkhorn chain keeps the drift at the few-ULP level.
    _assert_ulp_close(y, y_ref, rtol=1.6e-2, atol=1e-5, what="pre y", max_diff_frac=2e-3)
    _assert_ulp_close(post2, post_ref, rtol=1e-4, atol=1e-6, what="pre post")
    _assert_ulp_close(comb2, comb_ref, rtol=1e-4, atol=1e-6, what="pre comb")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("BS", [(1, 257), (2, 128)])
def test_post_next_partials_prefill_matches_old(BS):
    """Above the decode threshold the op must run the unchanged compose kernel."""
    B, S = BS
    hc_mult, H, mix_hc = 4, 4096, 24
    x, residual, post, comb, next_fn = _make_inputs(B, S, hc_mult, H, mix_hc)

    out_ref = torch.ops.auto_deploy.deepseek_v4_hc_post(x, residual, post, comb)
    out, parts = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
        x, residual, post, comb, next_fn
    )
    assert torch.equal(out, out_ref)
    # partials exist (fixed op signature) but are unfilled/unread on this path.
    _, split = hc_comp_mod.hc_partials_layout(hc_mult * H)
    assert parts.shape == (B * S, mix_hc + 1, split)

    # The prefill consumer ignores partials entirely: bit-identical to the
    # partials-free op even when handed the unfilled buffer.
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


def _ref_hc_head_norm(hidden4d, head_fn, head_scale, head_base, norm_w, hc_eps, norm_eps, rms_eps):
    """Exact mirror of the eager _hc_head + self.norm + .float() modeling tail."""
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
@pytest.mark.parametrize("BS", [(1, 1), (2, 1), (1, 33)])
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

    parts = _ref_partials(hidden.reshape(B * S, hc_mult * H), head_fn)
    out = torch.ops.auto_deploy.deepseek_v4_hc_head_norm(
        parts, hidden.flatten(2), head_fn, head_scale, head_base, norm_w, 1e-4, 1e-6, 1e-6
    )

    assert out.shape == ref.shape == (B, S, H)
    assert out.dtype == torch.float32
    # Output values are bf16-rounded then widened; mixes differ from cublas by
    # ~1 fp32 ULP (partial-sum association), so allow 1 bf16 ULP on the result.
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
    assert torch.equal(out, ref), "prefill path must be the identical eager chain"
