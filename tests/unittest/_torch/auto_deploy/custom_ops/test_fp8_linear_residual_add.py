# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness checks for the fused block-FP8 linear + merge-add epilogue.

``auto_deploy::torch_fake_quant_finegrained_fp8_linear_residual_add`` is emitted by
the ``fuse_fp8_linear_allreduce_add`` transform for the DeepSeek-V4 MoE seam
``all_reduce(routed_moe_out + shared_down_proj_out)``: the standalone elementwise
merge add collapses into the W8A8 matmul epilogue so the projection writes the
collective's input buffer directly.

The claim under test is *exact rounding equivalence*, not approximate accuracy:
the fused epilogue rounds the fp32 accumulator to the output dtype first (the
base kernel's store rounding) and then performs the add in fp32 opmath with one
final rounding (aten's elementwise add semantics). Every case therefore asserts
``torch.equal`` against the unfused two-op reference
``torch_fake_quant_finegrained_fp8_linear(...) + residual`` — including shapes
that dispatch to the split-K decode path (where the op intentionally keeps the
add as a separate elementwise epilogue with identical rounding).
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (  # noqa: F401
    _use_splitk_decode,
)

torch.manual_seed(0)

# (N, K) shapes. The DeepSeek-V4-Flash TP4 shared-expert down projection is
# N=4096, K=512 (full-K kernel path). N=576 exercises the masked-tile store with
# the residual load mask; (1024, 4096) dispatches to the split-K decode path at
# M<=4 and exercises the op's separate-add fallback.
SHAPES = [
    (4096, 512),  # DSV4 shared-expert w2 (the fusion's production shape)
    (2048, 512),
    (576, 7168),  # N not a multiple of 128 -> masked store + masked residual load
    (1024, 4096),  # -> split-K decode dispatch at M<=4
]

BLOCK_SIZE = 128


def _make_inputs(M: int, N: int, K: int, dtype: torch.dtype):
    device = "cuda"
    x = torch.randn(M, K, device=device, dtype=dtype) / K**0.5
    weight = torch.randn(N, K, device=device, dtype=torch.float32) / K**0.5
    weight_fp8 = weight.to(torch.float8_e4m3fn)
    scale_n = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale_k = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    weight_scale_inv = torch.rand(scale_n, scale_k, device=device, dtype=torch.float32) * 0.5 + 0.75
    residual = torch.randn(M, N, device=device, dtype=dtype)
    return x, weight_fp8, weight_scale_inv, residual


@pytest.mark.parametrize("N,K", SHAPES)
@pytest.mark.parametrize("M", [1, 2, 4, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_linear_residual_add_bit_exact(N: int, K: int, M: int, dtype: torch.dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x, weight_fp8, weight_scale_inv, residual = _make_inputs(M, N, K, dtype)

    ref_linear = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
        x, weight_fp8, None, [], [weight_scale_inv], [], []
    )
    ref = ref_linear + residual

    fused = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
        x, weight_fp8, None, [], [weight_scale_inv], [], [], residual=residual
    )

    assert fused.shape == ref.shape and fused.dtype == ref.dtype
    assert torch.equal(fused, ref), (
        f"fused residual-add mismatch (M={M}, N={N}, K={K}, dtype={dtype}): "
        f"max abs diff {(fused.float() - ref.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("order", ["other_first", "linear_first"])
def test_fp8_linear_residual_add_commutes(order: str):
    """The seam add is commutative bit-for-bit; both matched operand orders agree."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    M, N, K = 1, 4096, 512
    x, weight_fp8, weight_scale_inv, residual = _make_inputs(M, N, K, torch.bfloat16)

    ref_linear = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
        x, weight_fp8, None, [], [weight_scale_inv], [], []
    )
    ref = residual + ref_linear if order == "other_first" else ref_linear + residual

    fused = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add(
        x, weight_fp8, None, [], [weight_scale_inv], [], [], residual=residual
    )
    assert torch.equal(fused, ref)


def test_fp8_linear_residual_add_splitk_dispatch_covered():
    """Guard the shape list: (1024, 4096) at M=1 must exercise the split-K fallback."""
    assert _use_splitk_decode(1, 1024, 4096)
    assert not _use_splitk_decode(1, 4096, 512)
