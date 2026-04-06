# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from _torch_test_utils import fp8_compatible

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm as flashinfer_mod
import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.triton_fused_add_rms_norm_quant_fp8 as fused_mod

torch.manual_seed(0)


def _fused_add_rms_norm_quant_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
):
    add_out = x + residual
    add_f32 = add_out.to(torch.float32)
    rrms = torch.rsqrt(add_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    normed = add_f32 * rrms * weight.to(torch.float32)
    normed_out = normed.to(x.dtype)
    out_fp8 = (
        (normed_out.to(torch.float32) / scale.to(torch.float32))
        .clamp(fused_mod._FP8_MIN, fused_mod._FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    return normed_out, out_fp8, add_out


def _baseline_fused_add_quant_linear_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    input_scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
):
    norm_out, add_out = flashinfer_mod.flashinfer_fused_add_rms_norm(
        x.clone(), residual.clone(), norm_weight, eps
    )
    linear_out = torch.ops.auto_deploy.trtllm_quant_fp8_linear(
        norm_out,
        weight_fp8,
        None,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )
    return norm_out, add_out, linear_out


class _KernelStub:
    def __init__(self, launch_fn):
        self._launch_fn = launch_fn

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            return self._launch_fn(grid, *args, **kwargs)

        return _launch


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_triton_fused_add_rms_norm_quant_fp8_matches_reference_decode_batch(seq_len, dtype):
    batch = 1
    hidden_size = 2688
    eps = 1e-5

    x = torch.randn(batch, seq_len, hidden_size, device="cuda", dtype=dtype) * 0.25
    residual = torch.randn_like(x) * 0.25
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype) * 0.25
    input_scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)
    weight_scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)
    linear_weight = torch.randn(hidden_size, hidden_size, device="cuda", dtype=dtype) * 0.05
    weight_fp8 = (linear_weight / weight_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    out_bf16, out_fp8, out_add = fused_mod.fused_add_rms_norm_quant_fp8(
        x, residual, weight, eps, input_scale
    )
    ref_bf16, ref_add, ref_linear = _baseline_fused_add_quant_linear_ref(
        x, residual, weight, eps, input_scale, weight_fp8, weight_scale
    )
    fused_linear = torch.ops.auto_deploy.trtllm_fp8_prequant_linear(
        out_fp8,
        weight_fp8,
        None,
        input_scale=input_scale,
        weight_scale=weight_scale,
        out_dtype=str(dtype).replace("torch.", ""),
    )

    torch.testing.assert_close(out_add, ref_add, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_bf16, ref_bf16, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(fused_linear, ref_linear, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_len", [1, 4])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_triton_fused_add_rms_norm_quant_fp8_matches_static_quantize(seq_len, dtype):
    hidden_size = 2688
    eps = 1e-5

    x = torch.randn(1, seq_len, hidden_size, device="cuda", dtype=dtype) * 0.25
    residual = torch.randn_like(x) * 0.25
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype) * 0.25
    scale = torch.tensor(0.5, device="cuda", dtype=torch.float32)

    out_norm, out_fp8, _ = fused_mod.fused_add_rms_norm_quant_fp8(x, residual, weight, eps, scale)
    ref_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(out_norm, scale)

    torch.testing.assert_close(out_fp8.float(), ref_fp8.float(), rtol=0, atol=0)
