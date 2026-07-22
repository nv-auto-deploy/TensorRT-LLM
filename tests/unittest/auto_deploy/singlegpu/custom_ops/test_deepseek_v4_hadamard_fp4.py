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

"""Unit tests for ``auto_deploy::deepseek_v4_hadamard_fp4``.

The fused op must reproduce ``fake_fp4_act_quant(hadamard_rotate(x), 32)`` bit-identically
(fake quant, no FP4 hardware).
"""

import pytest
import torch

# Register the custom op (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.deepseek_v4_hadamard_fp4  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# Eager reference (the production helpers were removed from the modeling code).
def _ceil_pow2_scale(amax, max_value, min_value):
    return torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(min_value) / max_value)))


def _fake_fp4_act_quant(x, block_size=32):
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x
    dtype = x.dtype
    x_float = x.float()
    grouped = x_float.reshape(-1, dim // block_size, block_size)
    scale = _ceil_pow2_scale(grouped.abs().amax(dim=-1, keepdim=True), 6.0, 6.0 * 2.0**-126)
    normalized = torch.clamp(grouped / scale, -6.0, 6.0)
    abs_normalized = normalized.abs()
    quant_abs = torch.zeros_like(abs_normalized)
    for thr, val in [
        (0.25, 0.5),
        (0.75, 1.0),
        (1.25, 1.5),
        (1.75, 2.0),
        (2.5, 3.0),
        (3.5, 4.0),
        (5.0, 6.0),
    ]:
        quant_abs = torch.where(abs_normalized > thr, torch.full_like(quant_abs, val), quant_abs)
    quant = quant_abs * normalized.sign()
    return (quant * scale).reshape_as(x_float).to(dtype)


def _hadamard_rotate(x):
    dim = x.shape[-1]
    if dim <= 1:
        return x
    out = x.reshape(-1, dim).float()
    width = 1
    while width < dim:
        out = out.reshape(-1, dim // (2 * width), 2, width)
        left = out[..., 0, :]
        right = out[..., 1, :]
        out = torch.cat((left + right, left - right), dim=-1).flatten(-2)
        width *= 2
    return (out * (dim**-0.5)).reshape_as(x).to(x.dtype)


def _ref(x):
    return _fake_fp4_act_quant(_hadamard_rotate(x), block_size=32)


def _fused(x):
    return torch.ops.auto_deploy.deepseek_v4_hadamard_fp4(x, 32)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(2, 17, 128), (1, 128)])
@pytest.mark.parametrize("scale", [0.02, 40.0])
def test_matches_reference_dim128(dtype, shape, scale):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()
    out = _fused(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(out, _ref(x))


@pytest.mark.parametrize("dim", [32, 256])  # fewest / most butterfly stages
def test_matches_reference_other_pow2_dims(dim):
    torch.manual_seed(1)
    x = (torch.randn((5, dim), device="cuda", dtype=torch.bfloat16) * 1.3).contiguous()
    assert torch.equal(_fused(x), _ref(x))


def test_noncontiguous_leading_dims_write_returned_output():
    torch.manual_seed(4)
    x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert x.stride(-1) == 1 and not x.is_contiguous()
    out = _fused(x)
    assert out.is_contiguous()
    assert torch.equal(out, _ref(x))


# R=1024 is the BLOCK_R=2 threshold (even, no tail); R=2049 masks the trailing row.
@pytest.mark.parametrize("R", [1024, 2049])
def test_blocked_path_matches_reference(R):
    torch.manual_seed(2)
    x = torch.randn((R, 128), device="cuda", dtype=torch.bfloat16).contiguous()
    out = _fused(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(out, _ref(x))


def test_zero_and_empty_input():
    x = torch.zeros((3, 128), device="cuda", dtype=torch.bfloat16)  # log2(0) guard path
    out = _fused(x)
    assert torch.equal(out, _ref(x))
    assert torch.equal(out, torch.zeros_like(out))
    empty = torch.empty((0, 128), device="cuda", dtype=torch.bfloat16)  # R == 0 early return
    out_empty = _fused(empty)
    assert out_empty.shape == empty.shape and out_empty.dtype == empty.dtype
