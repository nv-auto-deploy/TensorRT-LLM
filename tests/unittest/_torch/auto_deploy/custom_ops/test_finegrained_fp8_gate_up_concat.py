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
"""Bit-exactness proof for the gate/up block-FP8 projection concatenation.

The ``fuse_finegrained_fp8_gate_up`` transform replaces two sibling
``torch_fake_quant_finegrained_fp8_linear_prequant`` matmuls (shared-expert w1/w3)
that consume the same (qfp8, qscale) activation with a single matmul over the
concatenated ``[2N, K]`` weight + ``[2N/block_n, K/block_k]`` scale, then slices the
result back into the two ``[..., N]`` tensors. This test verifies the numeric identity
    prequant(x, cat([w1, w3]))[:, :N]  ==  prequant(x, w1)
    prequant(x, cat([w1, w3]))[:, N:]  ==  prequant(x, w3)
directly on the custom op.
"""

import pytest
import torch


def _make_fp8_weight(n, k, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device=device, dtype=torch.bfloat16) * 0.1).to(
        torch.float8_e4m3fn
    )
    # per-block weight scale [N/128, K/128], strictly positive
    ws = (
        torch.rand(n // 128, k // 128, generator=g, device=device, dtype=torch.float32) * 0.05
        + 0.01
    )
    return w, ws


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("M", [8, 16])
def test_gate_up_concat_bit_exact_base_kernel(M):
    """M > 4 uses the deterministic (non split-K) base kernel -> byte-exact merge."""
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (register ops)

    device = "cuda"
    K, N = 4096, 2048  # shared-expert w1/w3 shape (hidden_size x moe_intermediate_size)

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    qfp8, qscale = torch.ops.auto_deploy.torch_fp8_finegrained_act_quant(x, 128, "")

    w1, s1 = _make_fp8_weight(N, K, device, seed=1)
    w3, s3 = _make_fp8_weight(N, K, device, seed=3)

    prequant = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant
    gate = prequant(qfp8, qscale, w1, None, [s1])
    up = prequant(qfp8, qscale, w3, None, [s3])

    cat_w = torch.cat([w1, w3], dim=0)
    cat_s = torch.cat([s1, s3], dim=0)
    merged = prequant(qfp8, qscale, cat_w, None, [cat_s])

    assert merged.shape == (M, 2 * N)
    assert torch.equal(merged[:, :N], gate), "gate half of concatenated matmul is not bit-exact"
    assert torch.equal(merged[:, N:], up), "up half of concatenated matmul is not bit-exact"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gate_up_concat_matches_splitk_decode():
    """M=1, K>=4096 hits the split-K atomic decode path; the concat is algebraically
    identical, differing only by the split-K's own atomic-reduction rounding."""
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    device = "cuda"
    K, N = 4096, 2048

    x = torch.randn(1, K, device=device, dtype=torch.bfloat16)
    qfp8, qscale = torch.ops.auto_deploy.torch_fp8_finegrained_act_quant(x, 128, "")

    w1, s1 = _make_fp8_weight(N, K, device, seed=11)
    w3, s3 = _make_fp8_weight(N, K, device, seed=13)

    prequant = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant
    gate = prequant(qfp8, qscale, w1, None, [s1])
    up = prequant(qfp8, qscale, w3, None, [s3])

    merged = prequant(qfp8, qscale, torch.cat([w1, w3], dim=0), None, [torch.cat([s1, s3], dim=0)])

    # Atomic split-K reduction is non-deterministic in ordering; both paths carry the
    # same rounding characteristics, so a tight relative tolerance is expected.
    torch.testing.assert_close(merged[:, :N], gate, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(merged[:, N:], up, rtol=1e-2, atol=1e-2)
