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

"""Fused DeepSeek-V4 RoPE ops vs the eager modeling chains they replace."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.rope.deepseek_v4_rope_fusion  # noqa: F401

DEV = "cuda"


def _apply_interleaved_rope(x, cos, sin, inverse=False):
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
    # ~1 ULP: the kernel fuses even*cos - odd*sin into an FMA; the reference doesn't.
    tol = {torch.float32: 1e-5, torch.float16: 1.5e-3, torch.bfloat16: 1e-2}[out.dtype]
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    "B,S,H,Dn,D",
    [
        (1, 1, 8, 512, 64),  # main q/out decode
        (1, 1, 1, 512, 64),  # main kv decode (H=1)
        (2, 5, 16, 128, 64),  # indexer-like 4D prefill
        (1, 1, 64, 128, 64),  # indexer query, large H
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
@pytest.mark.parametrize(
    "B,L,Dn,D",
    [
        (1, 1, 512, 64),
        (1, 257, 128, 64),
        (3, 40, 512, 64),
    ],
)
def test_3d_no_head(B, L, Dn, D):
    torch.manual_seed(1)
    dtype = torch.bfloat16
    nope = torch.randn(B, L, Dn, device=DEV, dtype=dtype)
    pe = torch.randn(B, L, D, device=DEV, dtype=dtype)
    cos = torch.randn(B, L, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, L, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=False)
    out = _fused(nope, pe, cos, sin)
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_strided_views_from_split():
    # Real call shape: nope/pe are split views, so row stride == full head_dim.
    torch.manual_seed(2)
    B, S, H, Dn, D = 1, 1, 8, 512, 64
    x = torch.randn(B, S, H, Dn + D, device=DEV, dtype=torch.bfloat16)
    nope, pe = torch.split(x, [Dn, D], dim=-1)
    assert pe.stride(-2) == Dn + D
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=True)
    out = _fused(nope, pe, cos, sin)
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_mismatched_row_strides():
    # Contiguous (post-quant) nope + strided pe slice -> different row strides.
    torch.manual_seed(3)
    B, S, Dn, D = 1, 1, 512, 64
    kv = torch.randn(B, S, Dn + D, device=DEV, dtype=torch.bfloat16)
    _, pe = torch.split(kv, [Dn, D], dim=-1)
    nope = torch.randn(B, S, Dn, device=DEV, dtype=torch.bfloat16)
    assert nope.stride(-2) == Dn and pe.stride(-2) == Dn + D
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref(nope, pe, cos, sin, head_broadcast=False)
    out = _fused(nope, pe, cos, sin)
    _assert_match(out, ref)


# Folded per-head weightless RMS norm (rms_eps > 0), main-Q path.


def _ref_norm(nope, pe, cos, sin, eps, head_broadcast):
    x = torch.cat((nope, pe), dim=-1)
    x = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
    nope_n, pe_n = torch.split(x, [nope.shape[-1], pe.shape[-1]], dim=-1)
    return _ref(nope_n, pe_n, cos, sin, head_broadcast)


def _fused_norm(nope, pe, cos, sin, eps):
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope, pe, cos, sin, False, eps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "B,S,H,Dn,D",
    [
        (1, 1, 8, 448, 64),  # main q decode (real DSV4-Flash shape)
        (1, 1, 1, 448, 64),
        (2, 5, 8, 448, 64),
        (1, 1, 8, 120, 8),  # masked reduction lanes
    ],
)
def test_norm_fold_fp32(B, S, H, Dn, D):
    torch.manual_seed(5)
    eps = 1e-6
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.float32)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.float32)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_norm(nope, pe, cos, sin, eps, head_broadcast=True)
    out = _fused_norm(nope, pe, cos, sin, eps)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "B,S,H,Dn,D",
    [
        (1, 1, 8, 448, 64),
        (2, 5, 8, 448, 64),
    ],
)
def test_norm_fold_bf16(B, S, H, Dn, D):
    torch.manual_seed(6)
    eps = 1e-6
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.bfloat16)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_norm(nope, pe, cos, sin, eps, head_broadcast=True)
    out = _fused_norm(nope, pe, cos, sin, eps)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    _assert_match(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_norm_fold_strided_split_views():
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
    torch.manual_seed(8)
    B, S, H, Dn, D = 1, 1, 8, 448, 64
    nope = torch.randn(B, S, H, Dn, device=DEV, dtype=torch.bfloat16)
    pe = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    assert torch.equal(_fused(nope, pe, cos, sin), _fused_norm(nope, pe, cos, sin, 0.0))


# auto_deploy::deepseek_v4_kv_norm_rope_concat — fused main-KV front-end.


def _eager_fake_fp8(x, block_size=64):
    # Verbatim eager body of utils.quantization_utils.fake_fp8_act_quant (the chain the
    # decode graph runs; the Triton op's byte-equality is pinned in test_fake_fp8_quant.py).
    dim = x.shape[-1]
    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(-1, dim // block_size, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(1.0e-4) / 448.0)))
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _ref_kv(kv, weight, eps, cos, sin, block_size=64):
    Dn = kv.shape[-1] - cos.shape[-1] * 2
    kv_normed = torch.ops.auto_deploy.torch_rmsnorm(kv, weight, eps)
    nope, pe = torch.split(kv_normed, [Dn, kv.shape[-1] - Dn], dim=-1)
    nope_q = _eager_fake_fp8(nope, block_size)
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope_q, pe, cos, sin, False)


def _fused_kv(kv, weight, eps, cos, sin, block_size=64):
    # The fused op takes the RAW (pre-norm) split views, exactly like modeling.
    Dn = kv.shape[-1] - cos.shape[-1] * 2
    nope, pe = torch.split(kv, [Dn, kv.shape[-1] - Dn], dim=-1)
    return torch.ops.auto_deploy.deepseek_v4_kv_norm_rope_concat(
        nope, pe, weight, cos, sin, eps, block_size
    )


def _mk(B, S, Dn, D, dtype, weight_dtype, seed):
    torch.manual_seed(seed)
    head = Dn + D
    kv = torch.randn(B, S, head, device=DEV, dtype=dtype)
    weight = (0.5 + torch.rand(head, device=DEV, dtype=torch.float32)).to(weight_dtype)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    return kv, weight, cos, sin


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "B,S,Dn,D",
    [
        (1, 1, 448, 64),  # real KV decode shape (7 fp8 blocks of 64)
        (2, 5, 448, 64),
        (1, 1, 128, 64),
    ],
)
def test_kv_fold_fp32(B, S, Dn, D):
    kv, weight, cos, sin = _mk(B, S, Dn, D, torch.float32, torch.float32, seed=1)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1])
def test_kv_fold_bf16_byte_exact_decode(weight_dtype, seed):
    kv, weight, cos, sin = _mk(1, 1, 448, 64, torch.bfloat16, weight_dtype, seed)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(out, ref), (
        f"byte mismatch: {(out != ref).sum().item()} / {ref.numel()} elems differ"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "B,S,Dn,D",
    [
        (2, 5, 448, 64),
        (1, 1, 128, 64),
    ],
)
def test_kv_fold_bf16_faithful(B, S, Dn, D):
    kv, weight, cos, sin = _mk(B, S, Dn, D, torch.bfloat16, torch.float32, seed=7)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_weight_split_mapping():
    # Strictly increasing weight makes nope / pe-even / pe-odd all distinct, so a
    # weight-to-lane mis-mapping breaks the tight tolerance.
    B, S, Dn, D = 1, 1, 448, 64
    torch.manual_seed(11)
    kv = torch.randn(B, S, Dn + D, device=DEV, dtype=torch.float32)
    weight = torch.linspace(0.3, 3.0, Dn + D, device=DEV, dtype=torch.float32)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kv_eps_used():
    B, S, Dn, D = 1, 4, 448, 64
    kv, weight, cos, sin = _mk(B, S, Dn, D, torch.float32, torch.float32, seed=3)
    for eps in (1e-6, 1e-1, 1.0):
        ref = _ref_kv(kv, weight, eps, cos, sin)
        out = _fused_kv(kv, weight, eps, cos, sin)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
