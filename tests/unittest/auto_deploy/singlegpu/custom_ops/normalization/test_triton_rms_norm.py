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

"""Tests for the Triton RMSNorm kernel."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm import *  # noqa
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.triton_rms_norm import rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

EPS = 1e-6


@pytest.mark.parametrize(
    "num_tokens,full_dim,norm_dim",
    [
        (4032, 576, 512),  # DeepSeek-V3-Lite kv_a_layernorm shape
        (37, 1536, 1024),  # DeepSeek-V4 Q-LoRA narrow of the fused Q/KV projection
        (128, 256, 128),
        (2, 64, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm_matches_torch_rmsnorm(num_tokens, full_dim, norm_dim, dtype):
    """Column-slice input matches the eager reference without a copy.

    The strided fast path must be bitwise-identical to the contiguous run and
    return a contiguous tensor; both must match ``torch_rmsnorm`` to a couple of
    output-dtype ulps.
    """
    torch.manual_seed(0)
    full_tensor = torch.randn(num_tokens, full_dim, dtype=dtype, device="cuda")
    weight = torch.randn(norm_dim, dtype=dtype, device="cuda")

    non_contiguous = full_tensor[:, :norm_dim]
    assert not non_contiguous.is_contiguous()

    out_nc = rms_norm(non_contiguous, weight, EPS)
    out_c = rms_norm(non_contiguous.contiguous(), weight, EPS)
    ref = torch.ops.auto_deploy.torch_rmsnorm(non_contiguous, weight, EPS)

    assert out_nc.shape == (num_tokens, norm_dim)
    assert out_nc.is_contiguous()
    assert torch.equal(out_nc, out_c)
    rtol = 2e-2 if dtype == torch.bfloat16 else 4e-3
    assert torch.allclose(out_nc, ref, rtol=rtol, atol=1e-2)
