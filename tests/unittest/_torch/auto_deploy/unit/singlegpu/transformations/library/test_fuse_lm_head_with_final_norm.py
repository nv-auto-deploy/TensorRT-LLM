# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerical-correctness test for the fused_rmsnorm_lm_head op.

The op replaces ``flashinfer_rms_norm(x, w_norm, eps) -> aten.linear(_, w_lm, None)``
with a single op that absorbs ``w_norm`` into ``w_lm`` and computes::

    rstd = 1 / sqrt(mean(x ^ 2) + eps)
    out = (x * rstd) @ (w_lm * w_norm[None, :]).T

We verify the fused op matches the unfused reference within bf16 tolerance.
"""

import pytest
import torch

# Force registration of the fused op via the transform module import.
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.transform.library.fuse_lm_head_with_final_norm  # noqa: F401


def _reference_norm_then_linear(
    x: torch.Tensor, w_norm: torch.Tensor, w_lm: torch.Tensor, eps: float
) -> torch.Tensor:
    in_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    normed = (x_fp32 * torch.rsqrt(variance + eps)) * w_norm.to(torch.float32)
    return torch.nn.functional.linear(normed.to(in_dtype), w_lm, None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(1, 1, 2880), (1, 7, 2880), (4, 16, 2880)])
def test_fused_rmsnorm_lm_head_matches_reference(dtype, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = "cuda"
    hidden = shape[-1]
    vocab = 4096  # keep small for speed

    x = torch.randn(*shape, dtype=dtype, device=device) * 0.5
    w_norm = torch.randn(hidden, dtype=dtype, device=device) * 0.1 + 1.0
    w_lm = torch.randn(vocab, hidden, dtype=dtype, device=device) * 0.02
    ones = torch.ones(hidden, dtype=dtype, device=device)
    eps = 1e-5

    # Reference: rmsnorm(x, w_norm) @ w_lm.T
    out_ref = _reference_norm_then_linear(x, w_norm, w_lm, eps)

    # Fused: pre-scale w_lm by w_norm and call the fused op.
    fused_w = (
        (w_lm.to(torch.float32) * w_norm.to(torch.float32).reshape(1, -1)).to(dtype).contiguous()
    )
    out_fused = torch.ops.auto_deploy.fused_rmsnorm_lm_head.default(x, ones, fused_w, eps)

    assert out_fused.shape == out_ref.shape
    assert out_fused.dtype == out_ref.dtype
    # bf16/fp16 tolerance: the only numerical difference is operation order
    # (fp32 mul of two scaled-down tensors vs. one combined scale). Both stay
    # comfortably inside half-precision rounding noise.
    torch.testing.assert_close(out_fused, out_ref, atol=2e-2, rtol=2e-2)


def test_fused_rmsnorm_lm_head_fake_shape_inference():
    """The custom op's fake/meta kernel must produce the right output shape."""
    x = torch.empty((2, 5, 1024), dtype=torch.bfloat16, device="meta")
    ones = torch.empty((1024,), dtype=torch.bfloat16, device="meta")
    w = torch.empty((8192, 1024), dtype=torch.bfloat16, device="meta")
    out = torch.ops.auto_deploy.fused_rmsnorm_lm_head.default(x, ones, w, 1e-5)
    assert tuple(out.shape) == (2, 5, 8192)
    assert out.dtype == torch.bfloat16
