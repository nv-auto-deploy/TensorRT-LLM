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

"""Unit tests for ``auto_deploy::deepseek_v4_fused_rope_concat``.

The fused op must reproduce ``cat((nope, _apply_interleaved_rope(pe, cos, sin)))``
to within ~1 ULP. cos/sin are fp32 (the DeepSeek-V4 rope tables are built as fp32),
so the only deviation from the reference's bf16*fp32 promotion is the kernel folding
``even*cos - odd*sin`` into an FMA — i.e. it is at least as accurate as the original.
"""

import pytest
import torch

# Register the custom op (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_rope  # noqa: F401


def _apply_interleaved_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Verbatim reference from modeling_deepseek_v4.py."""
    if inverse:
        sin = -sin
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


def _ref(nope, pe, cos, sin, head_broadcast, inverse=False):
    c = cos.unsqueeze(2) if head_broadcast else cos
    s = sin.unsqueeze(2) if head_broadcast else sin
    pe_rot = _apply_interleaved_rope(pe, c, s, inverse=inverse)
    return torch.cat((nope, pe_rot), dim=-1)


def _fused(nope, pe, cos, sin, inverse=False):
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope, pe, cos, sin, inverse)


def _assert_match(out, ref):
    """The kernel folds ``even*cos - odd*sin`` into an FMA (one rounding) while the
    reference does separate mul/sub (three roundings), so results can differ by up
    to ~1 ULP. Real bugs (wrong index/formula/broadcast) are orders of magnitude
    larger, so a ~1-ULP tolerance still catches them."""
    tol = {torch.float32: 1e-5, torch.float16: 1.5e-3, torch.bfloat16: 1e-2}[out.dtype]
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


DEV = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "B,S,H,Dn,D",
    [
        (1, 1, 8, 512, 64),  # main q/out, decode (per-rank H=8)
        (1, 1, 1, 512, 64),  # main kv, decode (no head dim -> H=1)
        (2, 5, 16, 128, 64),  # indexer-like 4D, prefill-ish
        (1, 1, 64, 128, 64),  # indexer query, single head-group large H
    ],
)
def test_4d_head_broadcast(B, S, H, Dn, D, inverse, dtype):
    torch.manual_seed(0)
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=dtype)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=dtype)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=True, inverse=inverse)
    out = _fused(nope, pe, cos, sin, inverse=inverse)
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "B,L,Dn,D",
    [
        (1, 1, 512, 64),  # main kv decode
        (1, 257, 128, 64),  # compressor over compressed history (large rows)
        (3, 40, 512, 64),  # batched 3D
    ],
)
def test_3d_no_head(B, L, Dn, D, inverse, dtype):
    torch.manual_seed(1)
    nope = torch.randn(B, L, Dn, device=DEV, dtype=dtype)
    pe = torch.randn(B, L, D, device=DEV, dtype=dtype)
    cos = torch.randn(B, L, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, L, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=False, inverse=inverse)
    out = _fused(nope, pe, cos, sin, inverse=inverse)
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_strided_views_from_split():
    """nope/pe as (leading-contiguous) split views of one head — the real call shape.

    Row stride of the slices is the full head_dim, not the slice width; the kernel
    must use stride(-2), not infer it from the slice width.
    """
    torch.manual_seed(2)
    B, S, H, Dn, D = 1, 1, 8, 512, 64
    x = torch.randn(B, S, H, Dn + D, device=DEV, dtype=torch.bfloat16)
    nope, pe = torch.split(x, [Dn, D], dim=-1)
    assert pe.stride(-2) == Dn + D  # confirm strided (not contiguous slice)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=True)
    out = _fused(nope, pe, cos, sin)
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_mismatched_row_strides():
    """KV path: contiguous (quantized) nope + strided pe slice -> different row strides."""
    torch.manual_seed(3)
    B, S, Dn, D = 1, 1, 512, 64
    kv = torch.randn(B, S, Dn + D, device=DEV, dtype=torch.bfloat16)
    _, pe = torch.split(kv, [Dn, D], dim=-1)
    nope = torch.randn(B, S, Dn, device=DEV, dtype=torch.bfloat16)  # fresh contiguous (post-quant)
    assert nope.stride(-2) == Dn and pe.stride(-2) == Dn + D
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=False)
    out = _fused(nope, pe, cos, sin)
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_cos_is_close():
    """When cos/sin are bf16, the kernel's fp32 math is *more* accurate than the
    reference's bf16 math — outputs are close but not bit-exact (justified)."""
    torch.manual_seed(4)
    B, S, H, Dn, D = 1, 1, 8, 512, 64
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.bfloat16)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.bfloat16)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.bfloat16)
    ref = _ref(nope, pe, cos, sin, head_broadcast=True)
    out = _fused(nope, pe, cos, sin)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# Folded per-head weightless RMS norm (rms_eps > 0)                            #
# --------------------------------------------------------------------------- #


def _ref_norm(nope, pe, cos, sin, eps, head_broadcast, inverse=False):
    """Reference: weightless per-head RMS norm over the FULL (nope||pe) head,
    applied BEFORE the split, then the plain rope/concat. Mirrors modeling line
    ``q = q * rsqrt(q.float().square().mean(-1)+eps).to(q.dtype)`` then split +
    ``deepseek_v4_fused_rope_concat``."""
    x = torch.cat((nope, pe), dim=-1)
    x = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
    nope_n, pe_n = torch.split(x, [nope.shape[-1], pe.shape[-1]], dim=-1)
    return _ref(nope_n, pe_n, cos, sin, head_broadcast, inverse=inverse)


def _fused_norm(nope, pe, cos, sin, eps, inverse=False):
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope, pe, cos, sin, inverse, eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "B,S,H,Dn,D",
    [
        (1, 1, 8, 448, 64),  # main q decode, per-rank H=8 (real DSV4-Flash shape)
        (1, 1, 1, 448, 64),  # single head
        (2, 5, 8, 448, 64),  # batched / multi-position
        (1, 1, 8, 120, 8),  # small odd-ish head to stress masked reduction lanes
    ],
)
def test_norm_fold_fp32(B, S, H, Dn, D, inverse):
    """fp32: the rsqrt factor is not bf16-rounded, so the only deviation from the
    split reference is fp32 reduction order (~1e-6). A tight tolerance therefore
    proves the norm math itself — reduction over the full head, /head_dim, factor
    applied to both nope and pe — is exactly right."""
    torch.manual_seed(5)
    eps = 1e-6
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.float32)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.float32)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_norm(nope, pe, cos, sin, eps, head_broadcast=True, inverse=inverse)
    out = _fused_norm(nope, pe, cos, sin, eps, inverse=inverse)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "B,S,H,Dn,D",
    [
        (1, 1, 8, 448, 64),  # main q decode
        (2, 5, 8, 448, 64),  # batched / multi-position
    ],
)
def test_norm_fold_bf16(B, S, H, Dn, D, inverse):
    """bf16: the folded op rounds the rsqrt factor to bf16 (matching `.to(q.dtype)`)
    and materializes each normalized value in bf16 before RoPE, so it is bit-faithful
    to the split reference up to the rope FMA (~1 ULP) and fp32 reduction order."""
    torch.manual_seed(6)
    eps = 1e-6
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.bfloat16)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_norm(nope, pe, cos, sin, eps, head_broadcast=True, inverse=inverse)
    out = _fused_norm(nope, pe, cos, sin, eps, inverse=inverse)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_norm_fold_strided_split_views():
    """The real call passes raw split views of one head (row stride = full head_dim)."""
    torch.manual_seed(7)
    B, S, H, Dn, D = 1, 1, 8, 448, 64
    eps = 1e-6
    x = torch.randn(B, S, H, Dn + D, device=DEV, dtype=torch.bfloat16)
    nope, pe = torch.split(x, [Dn, D], dim=-1)
    assert nope.stride(-2) == Dn + D and pe.stride(-2) == Dn + D
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_norm(nope, pe, cos, sin, eps, head_broadcast=True)
    out = _fused_norm(nope, pe, cos, sin, eps)
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rms_eps_zero_is_plain_rope():
    """rms_eps defaulting to 0 must leave every non-Q call site (kv/indexer/...)
    byte-identical to the plain rope/concat."""
    torch.manual_seed(8)
    B, S, H, Dn, D = 1, 1, 8, 448, 64
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.bfloat16)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    plain = _fused(nope, pe, cos, sin)
    explicit0 = _fused_norm(nope, pe, cos, sin, 0.0)
    assert torch.equal(plain, explicit0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
