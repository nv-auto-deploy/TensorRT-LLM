# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the block-wise FP8 activation-quant Triton kernel.

``_safe_act_quant`` (and its ``_act_quant_kernel``) is the per-block FP8
activation quantizer that runs upstream of *every* finegrained block-FP8 linear
(MLA / dense projections) on the decode path. This test guards the kernel
against a pure-PyTorch fp32 reference so that occupancy tuning (``num_warps`` /
``num_stages`` on the launch) can be exercised without silently corrupting
results.

The kernel computes, per contiguous ``block_size`` chunk:
    amax  = max(|x|)            (fp32)
    s     = max(amax / 448, 1e-12)            [default]
            exp2(ceil(log2(max(amax,1e-4)/448)))  [ue8m0 / ROUND_SCALE]
    y_fp8 = (x / s) -> float8_e4m3fn          (division done in fp32)
and stores ``s`` in the model dtype (so the returned scale is bf16-rounded).

Because the division is done in fp32 and the fp8 cast is round-to-nearest, the
quantized output is *bit-exact* against the fp32 reference for the default
(non-rounded) scale. The ue8m0 path uses transcendentals (log2/exp2/ceil) whose
last-ULP behavior can differ between Triton and torch, so it is checked on the
dequantized value with an fp8-appropriate bar.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import _safe_act_quant

FP8_MAX = 448.0


def _ref_act_quant(x: torch.Tensor, block_size: int, round_scale: bool):
    """Pure-PyTorch reference mirroring ``_act_quant_kernel`` exactly."""
    orig_shape = x.shape
    K = x.shape[-1]
    nblocks = K // block_size
    xb = x.float().reshape(*x.shape[:-1], nblocks, block_size)
    amax = xb.abs().amax(dim=-1)  # fp32, exact max
    if round_scale:
        amax = amax.clamp(min=1e-4)
        s = amax / FP8_MAX
        s = torch.exp2(torch.ceil(torch.log2(s)))
    else:
        s = amax / FP8_MAX
        s = s.clamp(min=1e-12)
    y = (xb / s.unsqueeze(-1)).reshape(orig_shape).to(torch.float8_e4m3fn)
    # The kernel stores the scale in the input dtype (bf16 here).
    s = s.to(x.dtype)
    return y, s


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9  # Hopper+ for fp8


# Decode (M=1, M=2) is the optimization target; a couple of prefill M values are
# included to guard against a compute-regime correctness regression. K values are
# the DeepSeek-V4-class TP8 per-rank projection K's (all multiples of block=128).
SHAPES_K = [512, 1536, 2048, 7168]
SHAPES_M = [1, 2, 16, 256]


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("round_scale", [False, True])
@pytest.mark.parametrize("M", SHAPES_M)
@pytest.mark.parametrize("K", SHAPES_K)
def test_act_quant_matches_reference(M, K, round_scale):
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    fmt = "ue8m0" if round_scale else ""
    y, s = _safe_act_quant(x, block_size=128, input_scale_fmt=fmt)
    y_ref, s_ref = _ref_act_quant(x, 128, round_scale)

    assert y.dtype == torch.float8_e4m3fn
    assert y.shape == x.shape
    assert s.shape == (M, K // 128)
    assert s.dtype == x.dtype
    assert torch.isfinite(s.float()).all()
    assert torch.isfinite(y.float()).all()

    if not round_scale:
        # fp32 division + RNE cast -> bit-exact quantized output and scale.
        assert torch.equal(y.float(), y_ref.float()), f"M={M} K={K}: y mismatch"
        assert torch.equal(s.float(), s_ref.float()), f"M={M} K={K}: s mismatch"
    else:
        # Transcendental scale path: compare dequantized values.
        deq = y.float() * s.float().repeat_interleave(128, dim=-1)
        deq_ref = y_ref.float() * s_ref.float().repeat_interleave(128, dim=-1)
        denom = x.float().abs().amax().clamp(min=1e-6)
        assert ((deq - deq_ref).abs().amax() / denom).item() < 1e-3, f"M={M} K={K}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("round_scale", [False, True])
def test_act_quant_all_zero_block_is_nan_safe(round_scale):
    """All-zero block must produce 0 (clamped scale), never 0/0 = NaN."""
    x = torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16)
    # Put one non-zero block so the tensor is not uniformly zero.
    x[0, :128] = 0.05
    fmt = "ue8m0" if round_scale else ""
    y, s = _safe_act_quant(x.contiguous(), block_size=128, input_scale_fmt=fmt)
    assert torch.isfinite(y.float()).all(), "NaN/Inf in quantized output"
    assert torch.isfinite(s.float()).all(), "NaN/Inf in scales"
    # The all-zero blocks must dequantize back to exactly zero.
    assert (y.float()[1] == 0).all()
    assert (y.float()[0, 128:] == 0).all()


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_act_quant_multi_token_block_alignment():
    """For M>1 the flat 128-element blocks must align with per-row K-blocks."""
    torch.manual_seed(1)
    M, K = 4, 512
    x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16)).contiguous()
    y, s = _safe_act_quant(x, block_size=128)
    y_ref, s_ref = _ref_act_quant(x, 128, False)
    assert torch.equal(y.float(), y_ref.float())
    assert torch.equal(s.float(), s_ref.float())
