# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the fused DeepSeek-V4 HC combine + RMSNorm Triton kernel.

``auto_deploy::deepseek_v4_hc_combine_rmsnorm`` collapses the ``_hc_pre`` tail
(``y = sum_m pre[m] * flat[m*H:]`` -> bf16 cast -> ``attn_norm`` / ``ffn_norm``
``DeepseekV4RMSNorm``) into one launch. This test guards it against a pure-torch
reference that mirrors the original chain line-for-line, *including* the bf16
round-trip ``y`` takes through ``_hc_pre``'s return and ``torch_rmsnorm``'s
``input.to(fp32)``, so the fusion cannot silently change numerics.
"""

import pytest
import torch

# Registers auto_deploy::deepseek_v4_hc_combine_rmsnorm.
from tensorrt_llm._torch.auto_deploy.custom_ops import deepseek_v4_hc_pre_norm  # noqa: F401


def _ref_hc_combine_rmsnorm(pre, flat, weight, eps, hc_mult, out_dtype):
    """Exact mirror of the modeling _hc_pre tail + DeepseekV4RMSNorm (torch_rmsnorm)."""
    H = weight.shape[0]
    lead = pre.shape[:-1]
    original_shape = (*lead, hc_mult, H)
    # weighted-combine over the hc_mult axis (fp32), then the bf16 return cast.
    y = torch.sum(pre.unsqueeze(-1) * flat.view(original_shape), dim=2)
    y = y.to(out_dtype)
    # torch_rmsnorm(y, weight, eps): input.to(fp32) -> normalize -> bf16(weight * bf16(normed))
    out = torch.empty_like(y)
    yf = y.to(torch.float32)
    variance = yf.pow(2).mean(-1, keepdim=True)
    yf = yf * torch.rsqrt(variance + eps)
    out.copy_(weight * yf.to(out.dtype))
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1), (1, 1000), (2, 3), (4, 17), (1, 257)])
@pytest.mark.parametrize("hc_mult", [4, 2, 3])
@pytest.mark.parametrize("H", [4096, 2048, 512, 127])
def test_hc_combine_rmsnorm_matches_reference(shape, hc_mult, H):
    torch.manual_seed(0)
    eps = 1e-6
    dev = "cuda"
    out_dtype = torch.bfloat16
    pre = torch.rand(*shape, hc_mult, device=dev, dtype=torch.float32) + 0.1
    # flat mimics x.flatten(2).float() with bf16-representable values.
    x = torch.randn(*shape, hc_mult, H, device=dev, dtype=out_dtype)
    flat = x.flatten(2).float()
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    ref = _ref_hc_combine_rmsnorm(pre, flat, weight, eps, hc_mult, out_dtype)
    out = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, flat, weight, eps, hc_mult, out_dtype
    )

    assert out.shape == ref.shape
    assert out.dtype == out_dtype
    # bf16 output; only reduction order / rsqrt differ from the reference. Allow a
    # couple of bf16 ULP. assert_close's bf16 default is rtol=1.6e-2, atol=1e-5;
    # bump atol slightly to absorb rare LSB flips near rounding boundaries.
    torch.testing.assert_close(out, ref, rtol=1.6e-2, atol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("shape", [(1, 1), (1, 1000), (2, 3), (4, 17)])
@pytest.mark.parametrize("hc_mult", [4, 3])
@pytest.mark.parametrize("H", [4096, 512, 127])
def test_hc_combine_rmsnorm_bf16_input_bit_exact(shape, hc_mult, H):
    """bf16 ``flat`` must produce byte-identical output to pre-cast fp32 ``flat``.

    The kernel's in-register bf16 -> fp32 conversion is exact, so skipping the
    HBM fp32 materialization cannot change a single bit.
    """
    torch.manual_seed(5)
    eps = 1e-6
    dev = "cuda"
    pre = torch.rand(*shape, hc_mult, device=dev, dtype=torch.float32) + 0.1
    x = torch.randn(*shape, hc_mult * H, device=dev, dtype=torch.bfloat16)
    weight = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    out_fp32 = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, x.float(), weight, eps, hc_mult, torch.bfloat16
    )
    out_bf16 = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, x, weight, eps, hc_mult, torch.bfloat16
    )
    assert torch.equal(out_bf16, out_fp32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_combine_rmsnorm_zero_weight_identity():
    """weight == 0 -> output all zeros (sanity on the weight broadcast)."""
    torch.manual_seed(1)
    hc_mult, H, eps = 4, 512, 1e-6
    pre = torch.rand(8, hc_mult, device="cuda", dtype=torch.float32) + 0.1
    flat = torch.randn(8, hc_mult * H, device="cuda", dtype=torch.float32)
    weight = torch.zeros(H, device="cuda", dtype=torch.float32)
    out = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, flat, weight, eps, hc_mult, torch.bfloat16
    )
    assert torch.count_nonzero(out) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hc_combine_rmsnorm_unit_rms():
    """ones weight, unit-variance constant row -> normalized output ~= the sign pattern."""
    torch.manual_seed(2)
    hc_mult, H, eps = 4, 2048, 1e-6
    # Build flat so the combine yields a known constant-magnitude vector.
    pre = torch.ones(3, hc_mult, device="cuda", dtype=torch.float32)
    base = torch.randn(3, H, device="cuda", dtype=torch.bfloat16).float()
    flat = base.repeat(1, hc_mult) / hc_mult  # sum_m pre*flat = base
    weight = torch.ones(H, device="cuda", dtype=torch.float32)
    out = torch.ops.auto_deploy.deepseek_v4_hc_combine_rmsnorm(
        pre, flat, weight, eps, hc_mult, torch.bfloat16
    )
    # RMS-normalized rows have unit mean-square (within bf16 tolerance).
    ms = out.float().pow(2).mean(-1)
    torch.testing.assert_close(ms, torch.ones_like(ms), rtol=5e-2, atol=5e-2)
