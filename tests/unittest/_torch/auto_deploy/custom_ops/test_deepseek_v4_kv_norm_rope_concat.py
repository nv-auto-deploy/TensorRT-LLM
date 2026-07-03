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

"""Unit tests for ``auto_deploy::deepseek_v4_kv_norm_rope_concat``.

The fused KV front-end op must reproduce, bit-for-bit, the main-KV chain it replaces:

    kv_normed = torch_rmsnorm(kv, weight, eps)               # weighted RMS norm
    nope, pe  = split(kv_normed, [Dn, D])
    nope_q    = fake_fp8_act_quant(nope, block_size=64)       # per-block fake FP8
    out       = deepseek_v4_fused_rope_concat(nope_q, pe, cos, sin)

Because the norm's sum-of-squares uses a different (tree) fp32 reduction order than
torch's ``mean``, the bf16 result is bit-identical only up to the bf16 rounding
absorbing that ~1e-6 fp32 difference. The fp32 test (unrounded factor, no fp8
mantissa rounding) pins the math structure with a tight tolerance; the bf16 tests
assert byte-equality on the real decode shape and ~1-ULP faithfulness on batched
shapes against the exact production chain.
"""

import pytest
import torch

# Register the custom ops (side-effect imports).
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_fake_fp8  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_rope  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fake_fp8_act_quant

DEV = "cuda"


def _ref_kv(kv, weight, eps, cos, sin, block_size=64):
    """Verbatim main-KV reference chain (all production custom ops)."""
    Dn = kv.shape[-1] - cos.shape[-1] * 2
    kv_normed = torch.ops.auto_deploy.torch_rmsnorm(kv, weight, eps)
    nope, pe = torch.split(kv_normed, [Dn, kv.shape[-1] - Dn], dim=-1)
    nope_q = fake_fp8_act_quant(nope, block_size=block_size)
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope_q, pe, cos, sin, False)


def _eager_fake_fp8(x, block_size=64):
    """Verbatim *eager* body of ``utils.quantization_utils.fake_fp8_act_quant`` — the
    ``abs/amax/log2/ceil/pow/clamp/cast/mul`` decomposition that the DSV4 decode graph
    actually runs for the main-KV nope (the fused Triton op only fires for the
    compressor/indexer paths). Kept here so the fold is validated against the exact
    kernels it replaces in production, independent of the Triton-op equivalence."""
    dim = x.shape[-1]
    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(-1, dim // block_size, block_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(1.0e-4) / 448.0)))
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _ref_kv_eager_fp8(kv, weight, eps, cos, sin, block_size=64):
    """Reference chain using the eager fp8 decomposition instead of the Triton op."""
    Dn = kv.shape[-1] - cos.shape[-1] * 2
    kv_normed = torch.ops.auto_deploy.torch_rmsnorm(kv, weight, eps)
    nope, pe = torch.split(kv_normed, [Dn, kv.shape[-1] - Dn], dim=-1)
    nope_q = _eager_fake_fp8(nope, block_size)
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope_q, pe, cos, sin, False)


def _fused_kv(kv, weight, eps, cos, sin, block_size=64):
    """The fused op — fed the RAW (pre-norm) split views, exactly like modeling."""
    Dn = kv.shape[-1] - cos.shape[-1] * 2
    nope, pe = torch.split(kv, [Dn, kv.shape[-1] - Dn], dim=-1)
    return torch.ops.auto_deploy.deepseek_v4_kv_norm_rope_concat(
        nope, pe, weight, cos, sin, eps, block_size
    )


def _mk(B, S, Dn, D, dtype, weight_dtype, seed):
    torch.manual_seed(seed)
    head = Dn + D
    kv = torch.randn(B, S, head, device=DEV, dtype=dtype)
    # non-uniform weight so a wrong nope/pe-even/pe-odd split shows up loudly
    weight = (0.5 + torch.rand(head, device=DEV, dtype=torch.float32)).to(weight_dtype)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    return kv, weight, cos, sin


# --------------------------------------------------------------------------- #
# fp32: unrounded factor, no fp8 mantissa rounding -> only fp32 reduction order #
# deviates from the reference, so a tight tolerance pins the norm/fp8/rope math #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "B,S,Dn,D",
    [
        (1, 1, 448, 64),  # real KV decode shape (head=512, 7 fp8 blocks of 64)
        (2, 5, 448, 64),  # batched / multi-position
        (1, 1, 128, 64),  # smaller head, 2 fp8 blocks
        (1, 3, 256, 64),  # 4 fp8 blocks
    ],
)
def test_kv_fold_fp32(B, S, Dn, D):
    kv, weight, cos, sin = _mk(B, S, Dn, D, torch.float32, torch.float32, seed=1)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# bf16: byte-exact against the production chain on the real decode shape        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kv_fold_bf16_byte_exact_decode(weight_dtype, seed):
    """Production dtype (bf16 acts, fp32 or bf16 weight), real KV decode shape."""
    kv, weight, cos, sin = _mk(1, 1, 448, 64, torch.bfloat16, weight_dtype, seed)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(out, ref), (
        f"byte mismatch: {(out != ref).sum().item()} / {ref.numel()} elems differ"
    )


# --------------------------------------------------------------------------- #
# bf16: faithful (~1 ULP) on batched / multi-position shapes                    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "B,S,Dn,D",
    [
        (2, 5, 448, 64),
        (1, 8, 448, 64),
        (1, 1, 128, 64),
    ],
)
def test_kv_fold_bf16_faithful(B, S, Dn, D):
    kv, weight, cos, sin = _mk(B, S, Dn, D, torch.bfloat16, torch.float32, seed=7)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    # ~1 bf16 ULP; a wrong index/formula/split is orders of magnitude larger.
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


# --------------------------------------------------------------------------- #
# The weight must map to nope[:Dn] and pe even/odd lanes correctly              #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weight_split_mapping():
    """A distinctive per-position weight (so nope vs pe-even vs pe-odd are all
    different) forces any mis-mapping of the weight onto the wrong head lanes to
    produce a large, tolerance-breaking error."""
    B, S, Dn, D = 1, 1, 448, 64
    torch.manual_seed(11)
    kv = torch.randn(B, S, Dn + D, device=DEV, dtype=torch.float32)
    # strictly increasing distinctive weight across the whole head
    weight = torch.linspace(0.3, 3.0, Dn + D, device=DEV, dtype=torch.float32)
    cos = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    sin = torch.randn(B, S, D // 2, device=DEV, dtype=torch.float32)
    ref = _ref_kv(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# eps actually participates (large eps must shift the result the same way)      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_eps_used():
    B, S, Dn, D = 1, 4, 448, 64
    kv, weight, cos, sin = _mk(B, S, Dn, D, torch.float32, torch.float32, seed=3)
    for eps in (1e-6, 1e-1, 1.0):
        ref = _ref_kv(kv, weight, eps, cos, sin)
        out = _fused_kv(kv, weight, eps, cos, sin)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------- #
# bf16: byte-exact against the EAGER fp8 chain that production actually runs     #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kv_fold_bf16_byte_exact_vs_eager_fp8(seed):
    """The decode graph runs the eager fp8 decomposition for the main-KV nope, so
    validate the fold against that exact chain (not only the Triton fp8 op)."""
    kv, weight, cos, sin = _mk(1, 1, 448, 64, torch.bfloat16, torch.float32, seed)
    ref = _ref_kv_eager_fp8(kv, weight, 1e-6, cos, sin)
    out = _fused_kv(kv, weight, 1e-6, cos, sin)
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert torch.equal(out, ref), (
        f"byte mismatch vs eager fp8: {(out != ref).sum().item()} / {ref.numel()} elems differ"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
