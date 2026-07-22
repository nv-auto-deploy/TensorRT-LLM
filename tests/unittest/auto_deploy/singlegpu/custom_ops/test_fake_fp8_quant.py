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

"""Unit tests for ``auto_deploy::fake_fp8_act_quant``.

The fused op must reproduce the eager reference in
utils/quantization_utils.fake_fp8_act_quant byte-for-byte (bf16 in/out, no FP8 hardware).
"""

import pytest
import torch

# Register the custom op (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.fake_fp8_quant  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _fake_fp8_act_quant_ref(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Verbatim eager reference from utils/quantization_utils.fake_fp8_act_quant."""
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x

    def ceil_pow2_scale(amax, max_value, min_value):
        return torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(min_value) / max_value)))

    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(-1, dim // block_size, block_size)
    scale = ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 448.0, 1.0e-4)
    quant = torch.clamp(grouped / scale, -448.0, 448.0).to(dtype).float()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _op(x, block_size=64):
    return torch.ops.auto_deploy.fake_fp8_act_quant(x, block_size)


@pytest.mark.parametrize("shape", [(64,), (2, 7, 192)])
@pytest.mark.parametrize("scale_mag", [1.0, 1e3])
def test_byte_exact_contiguous(shape, scale_mag):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * scale_mag).to(torch.bfloat16)
    ref = _fake_fp8_act_quant_ref(x, 64)
    got = _op(x, 64)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got, ref)


def test_byte_exact_strided_views_and_fallback():
    torch.manual_seed(1)
    full = torch.randn(2, 6, 320, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    # split(dim=-1): the DSV4 call-site layout (strided rows, unit last dim).
    nope, pe = torch.split(full, [192, 128], dim=-1)
    assert not nope.is_contiguous() and nope.stride(-1) == 1
    # Non-row-pitch-indexable layouts must take the `.contiguous()` fallback.
    t2d = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16).t()
    assert t2d.stride(-1) != 1
    t3d = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert t3d.stride(-1) == 1 and t3d.stride(0) != t3d.size(1) * t3d.stride(1)
    for sl in (nope, pe, t2d, t3d):
        assert torch.equal(_op(sl, 64), _fake_fp8_act_quant_ref(sl.contiguous(), 64))


def test_byte_exact_zero_and_extremes():
    blocks = [
        torch.zeros(64),  # amax floor
        torch.full((64,), 1e-6),
        torch.full((64,), 1e4),
        torch.tensor([448.0 * 2.0 ** (k % 9 - 4) for k in range(64)]),  # exact pow2*448 boundary
    ]
    x = torch.stack(blocks).to(torch.bfloat16).cuda()
    assert torch.equal(_op(x, 64), _fake_fp8_act_quant_ref(x, 64))


def test_guard_passthrough_and_empty():
    x = torch.randn(4, 100, device="cuda", dtype=torch.bfloat16)  # 100 % 64 != 0
    assert torch.equal(_op(x, 64), x)
    empty = torch.empty(0, 64, device="cuda", dtype=torch.bfloat16)
    out = _op(empty, 64)
    assert out.shape == empty.shape and out.dtype == empty.dtype
