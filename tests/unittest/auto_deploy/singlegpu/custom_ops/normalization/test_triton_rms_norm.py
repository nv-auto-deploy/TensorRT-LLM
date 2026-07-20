# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for the Triton RMSNorm kernel and the DSV4 ``deepseek_v4_q_rmsnorm`` alias."""

import pytest
import torch

# Register the custom ops (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.deepseek_v4_q_rmsnorm  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.triton_rms_norm import rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

Q_LORA = 1024
FUSED_OUT = 1536
EPS = 1e-6


@pytest.mark.parametrize(
    "num_tokens,full_dim,norm_dim",
    [
        (4032, 576, 512),  # DeepSeek-V3-Lite kv_a_layernorm shape
        (128, 256, 128),
        (2, 64, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_exact", [False, True])
def test_rms_norm_matches_torch_rmsnorm(num_tokens, full_dim, norm_dim, dtype, torch_exact):
    """Column-slice input matches the eager reference without a copy.

    The strided fast path must be bitwise-identical to the contiguous run and
    return a contiguous tensor. ``torch_exact`` matches ``torch_rmsnorm`` to
    single-ulp rounding flips at any shape (bit-exactness on the DSV4 envelope
    is pinned by the ``deepseek_v4_q_rmsnorm`` test below); the default recipe
    is a couple of ulps looser.
    """
    torch.manual_seed(0)
    full_tensor = torch.randn(num_tokens, full_dim, dtype=dtype, device="cuda")
    weight = torch.randn(norm_dim, dtype=dtype, device="cuda")

    non_contiguous = full_tensor[:, :norm_dim]
    assert not non_contiguous.is_contiguous()

    out_nc = rms_norm(non_contiguous, weight, EPS, torch_exact=torch_exact)
    out_c = rms_norm(non_contiguous.contiguous(), weight, EPS, torch_exact=torch_exact)
    ref = torch.ops.auto_deploy.torch_rmsnorm(non_contiguous, weight, EPS)

    assert out_nc.shape == (num_tokens, norm_dim)
    assert out_nc.is_contiguous()
    assert torch.equal(out_nc, out_c)
    one_ulp_rtol = 1e-2 if dtype == torch.bfloat16 else 2e-3
    if torch_exact:
        assert torch.allclose(out_nc, ref, rtol=one_ulp_rtol, atol=1e-3)
    else:
        assert torch.allclose(out_nc, ref, rtol=2 * one_ulp_rtol, atol=1e-2)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rows", [1, 37])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_deepseek_v4_q_rmsnorm_matches_bf16_reference(seed, rows, weight_dtype):
    """The alias op keeps the DSV4 contract.

    BF16 in/out, bit-equal to ``torch_rmsnorm`` on the strided narrow Q child of
    the fused projection.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    full = (torch.randn((1, rows, FUSED_OUT), generator=generator, device="cuda") * 8.0).to(
        torch.bfloat16
    )
    q = full.narrow(-1, 0, Q_LORA)
    generator.manual_seed(seed + 100)
    weight = (torch.rand(Q_LORA, generator=generator, device="cuda") * 2 - 0.5).to(weight_dtype)

    ref = torch.ops.auto_deploy.torch_rmsnorm(q, weight, EPS)
    out = torch.ops.auto_deploy.deepseek_v4_q_rmsnorm(q, weight, EPS)

    assert out.dtype == torch.bfloat16
    assert ref.dtype == torch.bfloat16
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "input_dtype,weight_dtype,match",
    [
        (torch.float16, torch.float32, "input must be bfloat16"),
        (torch.float32, torch.float32, "input must be bfloat16"),
        (torch.bfloat16, torch.float16, "weight must be bfloat16 or float32"),
    ],
)
def test_deepseek_v4_q_rmsnorm_rejects_bad_dtypes(input_dtype, weight_dtype, match):
    q = torch.randn((1, 1, Q_LORA), device="cuda", dtype=input_dtype)
    weight = torch.ones(Q_LORA, device="cuda", dtype=weight_dtype)
    with pytest.raises(TypeError, match=match):
        torch.ops.auto_deploy.deepseek_v4_q_rmsnorm(q, weight, EPS)
