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
"""Unit test for the fused DeepSeek-V4 compressor attention-pool op."""

import pytest
import torch

# Registers auto_deploy::deepseek_v4_compress_pool
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_compressor  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_compressor import _compress_pool_ref


def _ref(kv, gate):
    # Match the exact expression replaced at every call site, including the fp32
    # softmax torch performs internally regardless of input dtype.
    return (kv * gate.softmax(dim=-2)).sum(dim=-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (4, 512),  # rank-2 (dim=0 site, e.g. _compressed_row_from_paged_state)
        (2, 4, 512),  # rank-3 (dim=1 decode new-row site), main head_dim
        (2, 8, 512),  # rank-3 overlap (2*ratio), main head_dim
        (1, 128, 8, 512),  # rank-4 (dim=2 context/full-range), main head_dim
        (2, 130, 8, 128),  # rank-4 overlap, indexer head_dim, odd row count
        (3, 5, 96),  # non-pow2 channel dim
    ],
)
def test_compress_pool_matches_reference(shape, dtype):
    torch.manual_seed(0)
    kv = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn(shape, device="cuda", dtype=dtype)

    out = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    ref = _ref(kv, gate)

    assert out.shape == ref.shape
    assert out.dtype == kv.dtype
    atol, rtol = (2e-4, 2e-4) if dtype == torch.float32 else (8e-3, 8e-3)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compress_pool_validity_masking(dtype):
    """Rows masked to -1e20 (invalid candidates) must get ~zero softmax weight,
    and fully-masked columns must reduce to the (uniform) mean exactly as torch."""
    torch.manual_seed(1)
    B, R, D = 2, 8, 512
    kv = torch.randn(B, R, D, device="cuda", dtype=dtype)
    gate = torch.randn(B, R, D, device="cuda", dtype=dtype)
    # Mask the second half of the ratio axis for row 0; mask ALL of row 1.
    gate[0, R // 2 :, :] = -1.0e20
    gate[1, :, :] = -1.0e20

    out = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    ref = _ref(kv, gate)

    atol, rtol = (2e-4, 2e-4) if dtype == torch.float32 else (8e-3, 8e-3)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
    # fully-masked row -> uniform average over the ratio axis
    assert torch.isfinite(out[1]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compress_pool_ref_helper_consistency():
    """The exported reference helper equals the inline expression."""
    kv = torch.randn(2, 4, 64, device="cuda")
    gate = torch.randn(2, 4, 64, device="cuda")
    torch.testing.assert_close(_compress_pool_ref(kv, gate), _ref(kv, gate))
