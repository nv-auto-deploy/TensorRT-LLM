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

The fused op must reproduce ``fake_fp4_act_quant(hadamard_rotate(x), 32)`` exactly.
Both the Walsh-Hadamard butterfly (a tree of single ``add``/``sub`` per output) and
the FP4 ladder use the same fp32 ops as the reference — including the intermediate
``bf16`` round-trip the reference incurs between ``hadamard_rotate``'s ``.to(dtype)``
and ``fake_fp4_act_quant``'s ``.float()`` — so the result is bit-identical.
"""

import pytest
import torch

# Register the custom op (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_hadamard_fp4  # noqa: F401


# --- verbatim reference from utils/quantization_utils.py ---
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 17, 128),  # compressor-rotate-like [B, L, head_dim]
        (2, 1, 8, 128),  # indexer-q decode-like [B, S, H, head_dim]
        (2, 7, 8, 128),  # indexer-q prefill-like
        (1, 128),
        (33, 128),
    ],
)
@pytest.mark.parametrize("scale", [0.02, 0.3, 1.0, 8.0, 40.0])
def test_matches_reference_dim128(dtype, shape, scale):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda", dtype=dtype) * scale).contiguous()
    out = _fused(x)
    ref = _ref(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.equal(out, ref), (
        f"max abs diff {(out.float() - ref.float()).abs().max().item():.3e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dim", [32, 64, 256])
def test_matches_reference_other_pow2_dims(dim):
    torch.manual_seed(1)
    x = (torch.randn((5, dim), device="cuda", dtype=torch.bfloat16) * 1.3).contiguous()
    out = _fused(x)
    ref = _ref(x)
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_input_is_zero():
    # all-zero input exercises the amax==0 / log2(0) guard path of ceil_pow2_scale;
    # the rotation of zeros is zero and the fp4 quant of zeros is zero.
    x = torch.zeros((3, 128), device="cuda", dtype=torch.bfloat16)
    out = _fused(x)
    ref = _ref(x)
    assert torch.equal(out, ref)
    assert torch.equal(out, torch.zeros_like(out))
