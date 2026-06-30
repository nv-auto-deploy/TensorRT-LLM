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

"""Correctness of the direct grouped block-FP8 W8A8 matmul (idea_0025).

``torch_fake_quant_grouped_finegrained_fp8_linear`` now keeps both operands in FP8 and runs a
direct block-FP8 GEMM (the same ``_w8a8_block_fp8_matmul_triton`` kernel the non-grouped
FineGrained FP8 op uses) instead of dequantizing the operands and running a BF16 grouped BMM.

These tests prove the new path is correct:
  * ``num_groups == 1`` (the DeepSeek-V4 ``wo_a`` per-rank case under TP) dispatches to exactly the
    proven non-grouped FineGrained FP8 op -- bit-for-bit identical on the deterministic full-K
    decode kernel, and equal up to fp32 atomic-reduction reorder (~1 ULP) on the split-K decode
    kernel (K>=4096), which itself is run-to-run non-deterministic by construction.
  * ``num_groups > 1`` matches a per-group launch of the non-grouped op bit-for-bit (full-K).
  * the direct FP8 GEMM agrees to high cosine similarity with the prior dequant->BF16-BMM path,
    so model-level accuracy is preserved (the direct path is in fact more faithful: it avoids the
    BF16 rounding of the dequantized operands).
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _dequant_block_fp8_weight,
)

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)

pytestmark = [
    pytest.mark.skipif(_FP8_DTYPE is None, reason="Requires torch.float8_e4m3fn"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA for Triton kernels"),
]

_GROUPED_OP = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear
_LINEAR_OP = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
_BLOCK = 128


def _make_inputs(num_groups, rank, in_features, batch=2, seed=0):
    """Build a grouped FP8 weight + per-(128,128)-block scale + grouped BF16 activation."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out_rows = num_groups * rank
    assert out_rows % _BLOCK == 0 and in_features % _BLOCK == 0
    w_fp8 = torch.randn(
        out_rows, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
    ).to(_FP8_DTYPE)
    scale_shape = (out_rows // _BLOCK, in_features // _BLOCK)
    scale = torch.rand(scale_shape, generator=gen, device="cuda", dtype=torch.float32) + 0.5
    x = torch.randn(
        batch, num_groups, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    return x, w_fp8, scale


@pytest.mark.parametrize("input_scale_fmt", ["", "ue8m0"])
@pytest.mark.parametrize("with_bias", [False, True])
# K=2048 (<4096) -> deterministic full-K decode kernel, so grouped == non-grouped bit-for-bit.
def test_single_group_matches_nongrouped_fp8_bitwise(input_scale_fmt, with_bias):
    rank, in_features = 256, 2048
    x, w_fp8, scale = _make_inputs(1, rank, in_features, seed=1)
    bias = torch.randn(rank, device="cuda", dtype=torch.bfloat16) if with_bias else None

    out_grouped = _GROUPED_OP(x, w_fp8, bias, [], [scale], [], [], input_scale_fmt=input_scale_fmt)
    # Equivalent flattened non-grouped call ([B, 1, K] -> [B, K]).
    out_linear = _LINEAR_OP(
        x.reshape(x.shape[0], in_features),
        w_fp8,
        bias,
        [],
        [scale],
        [],
        [],
        input_scale_fmt=input_scale_fmt,
    )

    assert out_grouped.shape == (x.shape[0], rank)
    assert out_grouped.dtype == torch.bfloat16
    assert torch.equal(out_grouped, out_linear.reshape_as(out_grouped))


@pytest.mark.parametrize("input_scale_fmt", ["", "ue8m0"])
# rank=1024,in_features=4096 is the DeepSeek-V4 wo_a per-rank shape (K>=4096 -> split-K decode).
# The split-K kernel reduces via fp32 ``tl.atomic_add`` whose order is non-deterministic, so two
# separate launches (grouped vs non-grouped) agree only up to ~1 ULP -- assert closeness, not
# bit-equality. This still proves the grouped path dispatches to the same kernel.
def test_single_group_matches_nongrouped_fp8_splitk_close(input_scale_fmt):
    rank, in_features = 1024, 4096
    x, w_fp8, scale = _make_inputs(1, rank, in_features, seed=1)

    out_grouped = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [], input_scale_fmt=input_scale_fmt)
    out_linear = _LINEAR_OP(
        x.reshape(x.shape[0], in_features),
        w_fp8,
        None,
        [],
        [scale],
        [],
        [],
        input_scale_fmt=input_scale_fmt,
    ).reshape_as(out_grouped)

    assert out_grouped.shape == (x.shape[0], rank)
    torch.testing.assert_close(out_grouped, out_linear, rtol=2e-2, atol=1.0)
    cos = torch.nn.functional.cosine_similarity(
        out_grouped.float().reshape(-1), out_linear.float().reshape(-1), dim=0
    )
    assert cos > 0.9999, f"cosine similarity too low: {cos.item()}"


def test_multi_group_matches_per_group_nongrouped_bitwise():
    num_groups, rank, in_features = 3, 256, 512  # K=512 (<4096) -> deterministic full-K
    x, w_fp8, scale = _make_inputs(num_groups, rank, in_features, seed=2)

    out_grouped = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [])

    wq = w_fp8.view(num_groups, rank, in_features)
    sg = scale.view(num_groups, rank // _BLOCK, in_features // _BLOCK)
    parts = [
        _LINEAR_OP(
            x[:, g, :].contiguous(), wq[g].contiguous(), None, [], [sg[g].contiguous()], [], []
        )
        for g in range(num_groups)
    ]
    ref = torch.stack(parts, dim=1).reshape(x.shape[0], num_groups * rank)

    assert out_grouped.shape == (x.shape[0], num_groups * rank)
    assert torch.equal(out_grouped, ref)


def test_direct_fp8_close_to_prior_dequant_bmm():
    """The direct FP8 GEMM changes the arithmetic vs the prior dequant->BF16-BMM; confirm they
    agree to high cosine similarity so model accuracy is preserved."""
    rank, in_features = 1024, 4096
    x, w_fp8, scale = _make_inputs(1, rank, in_features, seed=3)

    out_fp8 = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [])  # direct FP8 path
    # The prior behavior == feeding a pre-baked BF16 weight (the op's fallback branch).
    out_rows, _ = w_fp8.shape
    block_n = out_rows // scale.shape[0]
    block_k = in_features // scale.shape[1]
    w_bf16 = _dequant_block_fp8_weight(w_fp8, scale, block_n, block_k)
    out_bmm = _GROUPED_OP(x, w_bf16, None, [], [scale], [], [])  # baked BF16 -> dequant BMM

    cos = torch.nn.functional.cosine_similarity(
        out_fp8.float().reshape(-1), out_bmm.float().reshape(-1), dim=0
    )
    assert cos > 0.99, f"cosine similarity too low: {cos.item()}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
