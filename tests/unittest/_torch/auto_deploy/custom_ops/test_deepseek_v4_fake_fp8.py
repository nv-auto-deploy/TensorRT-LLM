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

"""Unit tests for ``auto_deploy::deepseek_v4_fake_fp8_act_quant``.

The fused Triton op must reproduce the eager reference in
``utils/quantization_utils.fake_fp8_act_quant`` **byte-for-byte** for bf16 inputs:
all intermediate math is fp32, the two bf16 round-trips use RNE in both, and the
power-of-two scale ``2**ceil(log2(clamp_min(amax, 1e-4)/448))`` lands on the same
integer exponent because the bf16-precision amax stays far from power-of-two
boundaries in log2 space.
"""

import pytest
import torch

# Register the custom op (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_fake_fp8  # noqa: F401


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
    return torch.ops.auto_deploy.deepseek_v4_fake_fp8_act_quant(x, block_size)


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (64,),  # single group, 1-D
        (8, 128),  # 2 groups/row
        (2, 7, 192),  # 3 groups/row, 3-D
        (4, 16, 64),  # decode-ish: many tiny rows
        (1, 1, 128, 64),  # 4-D
        (3, 5, 576),  # 9 groups/row (nope_head_dim-like)
    ],
)
@pytest.mark.parametrize("scale_mag", [1.0, 1e-3, 1e3, 17.0])
def test_byte_exact_contiguous(shape, dtype, scale_mag):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda", dtype=torch.float32) * scale_mag).to(dtype)
    ref = _fake_fp8_act_quant_ref(x, 64)
    got = _op(x, 64)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        f"byte mismatch: {(got != ref).sum().item()} / {ref.numel()} elements differ"
    )


@requires_cuda
def test_byte_exact_split_view():
    """Call sites feed last-dim slices of a contiguous tensor (strided rows)."""
    torch.manual_seed(1)
    full = torch.randn(2, 6, 320, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    # split(dim=-1) -> a non-contiguous view with stride(-2) == 320, stride(-1) == 1
    nope, pe = torch.split(full, [192, 128], dim=-1)
    assert not nope.is_contiguous() and nope.stride(-1) == 1
    for sl in (nope, pe):
        ref = _fake_fp8_act_quant_ref(sl, 64)
        got = _op(sl, 64)
        assert torch.equal(got, ref)


@requires_cuda
def test_byte_exact_zero_and_extremes():
    """Zero blocks (amax floor), tiny, and large blocks all hit the scale edges."""
    blocks = [
        torch.zeros(64),
        torch.full((64,), 1e-6),
        torch.full((64,), 1e4),
        # exact power-of-2 * 448 boundaries (amax/448 == exact 2**k -> log2 hits an int)
        torch.tensor([448.0 * 2.0 ** (k % 9 - 4) for k in range(64)]),
    ]
    x = torch.stack(blocks).to(torch.bfloat16).cuda()
    ref = _fake_fp8_act_quant_ref(x, 64)
    got = _op(x, 64)
    assert torch.equal(got, ref)


@requires_cuda
def test_byte_exact_many_random_seeds():
    """Stress byte-exactness across many random draws (catches rare scale-boundary drift)."""
    mismatches = 0
    for seed in range(50):
        torch.manual_seed(seed)
        x = (torch.randn(13, 256, device="cuda", dtype=torch.float32) * 5.0).to(torch.bfloat16)
        if not torch.equal(_op(x, 64), _fake_fp8_act_quant_ref(x, 64)):
            mismatches += 1
    assert mismatches == 0, f"{mismatches}/50 random draws were not byte-exact"


@requires_cuda
def test_guard_passthrough():
    """dim not divisible by block_size returns input unchanged (mirrors the helper guard)."""
    x = torch.randn(4, 100, device="cuda", dtype=torch.bfloat16)  # 100 % 64 != 0
    got = _op(x, 64)
    assert torch.equal(got, x)
