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
    """Column-slice input: copy-free strided path == contiguous run, both match eager."""
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


@pytest.mark.parametrize(
    "make_input",
    [
        # 3D narrow view: leading dims flatten to regular rows -> in-place fast path.
        lambda: torch.randn(2, 5, 96, dtype=torch.float16, device="cuda")[..., :64],
        # 3D middle-dim slice: dim-0 stride mismatch -> contiguous() fallback.
        lambda: torch.randn(2, 5, 64, dtype=torch.float16, device="cuda")[:, :3, :],
        # Transposed: last dim not unit-stride -> contiguous() fallback.
        lambda: torch.randn(64, 8, dtype=torch.float16, device="cuda").t(),
        # Overlapping rows (row stride < feat_size) -> contiguous() fallback.
        lambda: torch.randn(4 * 64, dtype=torch.float16, device="cuda").as_strided(
            (4, 64), (32, 1)
        ),
    ],
    ids=["3d_narrow", "3d_sliced", "transposed", "overlapping"],
)
def test_rms_norm_irregular_and_3d_layouts(make_input):
    """Regular 3D views stay copy-free; irregular layouts fall back to contiguous()."""
    torch.manual_seed(0)
    x = make_input()
    weight = torch.randn(x.shape[-1], dtype=x.dtype, device="cuda")

    out = rms_norm(x, weight, EPS)
    ref = torch.ops.auto_deploy.torch_rmsnorm(x.contiguous(), weight, EPS)

    assert out.shape == x.shape
    assert out.is_contiguous()
    assert torch.allclose(out, ref, rtol=4e-3, atol=1e-2)
