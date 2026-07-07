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

"""Byte-identity checks for the fused act-quant + block-FP8 matmul decode path (idea_0043).

At decode (M<=4) ``torch_fake_quant_finegrained_fp8_linear`` (and the grouped op's
``num_groups == 1`` branch) fold the standalone ``_act_quant_kernel`` launch into the
W8A8 matmul prologue. The fused kernels must be *byte-identical* to the two-launch
reference (``_safe_act_quant`` + ``_w8a8_block_fp8_matmul_triton``): same fp32
amax/scale math (both ROUND_SCALE branches), same fp32 division by the unrounded
fp32 scale, same RNE fp8 cast, same model-dtype scale rounding, and the same
dot/accumulation tiling.

The split-K consumer (M<=4, K>=4096) reduces via fp32 ``tl.atomic_add`` whose
arrival order is not deterministic run-to-run -- a pre-existing property of the
standalone path. For those shapes the test measures the reference's own run-to-run
stability first and demands bit-equality whenever the reference is bit-stable;
otherwise the fused result must stay inside the same one-ULP envelope. A
``SPLIT_K=1`` launch (single atomic writer per output element -> deterministic) is
additionally checked bit-exactly to pin the fused quant + strided-K-loop dot math.
"""

import pytest
import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _safe_act_quant,
    _use_fused_act_quant_matmul,
    _w8a8_block_fp8_matmul_fused_act_quant,
    _w8a8_block_fp8_matmul_splitk,
    _w8a8_block_fp8_matmul_triton,
)

fp8 = torch.float8_e4m3fn
lin_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
grouped_op = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear


def _fp8_supported():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9  # Hopper+ for fp8


pytestmark = pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")


def _make_weight(N, K, seed=0):
    """Random block-FP8 weight + per-128x128-block fp32 scale (checkpoint layout)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.1).to(fp8)
    ws = (
        torch.rand(triton.cdiv(N, 128), triton.cdiv(K, 128), generator=gen, device="cuda").float()
        * 0.05
        + 0.01
    )
    return w, ws


def _rand_x(M, K, seed, scale=0.1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(M, K, generator=gen, device="cuda", dtype=torch.bfloat16) * scale
    ).contiguous()


def _reference_linear(x, w, ws, bias, fmt):
    """The pre-fusion two-launch op body: standalone quant + unfused matmul."""
    N, K = w.shape
    block_n = triton.cdiv(N, ws.shape[0])
    block_k = triton.cdiv(K, ws.shape[1])
    q, s = _safe_act_quant(x, block_k, fmt)
    out = _w8a8_block_fp8_matmul_triton(q, w, s, ws, [block_n, block_k], output_dtype=x.dtype)
    if bias is not None:
        out = out + bias
    return out.to(x.dtype)


# DeepSeek-V4-Flash TP4 per-rank decode shapes plus off-band tile counts. Full-K
# (K < 4096 at M <= 4) is deterministic -> bit-equality is demanded outright.
FULLK_SHAPES = [
    (1, 4096, 2048),  # wo_b (pinned decode config)
    (1, 16384, 1024),  # fused wq_b + indexer.wq_b (pinned)
    (2, 8192, 1024),  # wq_b band at M=2
    (4, 4096, 512),  # shared w2
    (3, 384, 256),  # small/ragged tile counts
]
# Split-K (M <= 4, K >= 4096) covers every tuned schedule band.
SPLITK_SHAPES = [
    (1, 1536, 4096),  # fused wq_a + wkv (pinned SPLIT_K=24 schedule)
    (1, 1024, 4096),  # shared w1+w3 / grouped wo_a (pinned SPLIT_K=32, nw=2)
    (1, 256, 7168),  # narrow-N deep-split band (SPLIT_K=48)
    (2, 2304, 7168),  # wide-N shallow-split band (SPLIT_K=16)
    (4, 576, 4096),  # mid-N band at the M-gate edge
]


@pytest.mark.parametrize("fmt", ["", "ue8m0"])
@pytest.mark.parametrize("shape", FULLK_SHAPES)
def test_fused_full_k_bit_exact(shape, fmt):
    M, N, K = shape
    x = _rand_x(M, K, seed=M * 7 + K)
    w, ws = _make_weight(N, K, seed=N)
    assert _use_fused_act_quant_matmul(x, N, K, 128)
    out = lin_op(x, w, None, [], [ws], [], [], input_scale_fmt=fmt)
    ref = _reference_linear(x, w, ws, None, fmt)
    assert out.dtype == x.dtype
    assert torch.equal(out, ref), f"{shape} fmt={fmt!r}: fused != standalone reference"


@pytest.mark.parametrize("fmt", ["", "ue8m0"])
@pytest.mark.parametrize("shape", SPLITK_SHAPES)
def test_fused_splitk_matches_reference(shape, fmt):
    M, N, K = shape
    x = _rand_x(M, K, seed=M * 11 + N)
    w, ws = _make_weight(N, K, seed=N + 1)
    assert _use_fused_act_quant_matmul(x, N, K, 128)
    out = lin_op(x, w, None, [], [ws], [], [], input_scale_fmt=fmt)
    ref1 = _reference_linear(x, w, ws, None, fmt)
    ref2 = _reference_linear(x, w, ws, None, fmt)
    if torch.equal(ref1, ref2):
        # Reference is bit-stable at this shape: demand bit-equality of the fusion.
        assert torch.equal(out, ref1), f"{shape} fmt={fmt!r}: fused != standalone reference"
    else:
        # Pre-existing fp32-atomic arrival-order jitter: the fused result must stay
        # inside the reference's own one-ULP run-to-run envelope.
        assert torch.allclose(out.float(), ref1.float(), rtol=2**-7, atol=1e-6)
        mism_out = (out != ref1).float().mean().item()
        mism_ref = (ref2 != ref1).float().mean().item()
        assert mism_out <= max(4.0 * mism_ref, 1e-3), (
            f"{shape} fmt={fmt!r}: fused mismatch rate {mism_out} exceeds the "
            f"reference's own jitter envelope {mism_ref}"
        )


@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_fused_splitk_split1_bit_exact(fmt):
    """SPLIT_K=1 -> one atomic writer per element -> deterministic on both sides;
    pins the fused quant + strided-K-loop dot math bit-exactly."""
    M, N, K = 2, 512, 7168
    x = _rand_x(M, K, seed=3)
    w, ws = _make_weight(N, K, seed=5)
    overrides = dict(SPLIT_K=1, BLOCK_SIZE_N=64, num_warps=4, num_stages=3)
    out = _w8a8_block_fp8_matmul_fused_act_quant(
        x, w, ws, [128, 128], output_dtype=x.dtype, input_scale_fmt=fmt, **overrides
    )
    q, s = _safe_act_quant(x, 128, fmt)
    ref = _w8a8_block_fp8_matmul_splitk(
        q, w, s, ws, 128, 128, x.dtype, M, N, K, BLOCK_SIZE_K=128, **overrides
    )
    assert torch.equal(out, ref), f"fmt={fmt!r}: SPLIT_K=1 fused != unfused"


@pytest.mark.parametrize("fmt", ["", "ue8m0"])
def test_fused_zero_blocks_and_extremes(fmt):
    """All-zero scale groups (clamped-scale path) and large magnitudes stay bit-exact."""
    M, N, K = 2, 384, 512
    x = _rand_x(M, K, seed=9, scale=100.0)
    x[0, :128] = 0.0  # all-zero scale group in row 0
    x[1, :] = 0.0  # fully zero row
    w, ws = _make_weight(N, K, seed=11)
    out = lin_op(x, w, None, [], [ws], [], [], input_scale_fmt=fmt)
    ref = _reference_linear(x, w, ws, None, fmt)
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("fmt", ["", "ue8m0"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_grouped_g1_and_bias(fmt, with_bias):
    """The grouped op's num_groups==1 fused branch (TP wo_a case), K<4096 deterministic."""
    B, rank, K = 2, 256, 2048
    gen = torch.Generator(device="cuda").manual_seed(21)
    x3d = torch.randn(B, 1, K, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.1
    w, ws = _make_weight(rank, K, seed=13)
    bias = (
        torch.randn(rank, generator=gen, device="cuda", dtype=torch.bfloat16) if with_bias else None
    )
    out = grouped_op(x3d, w, bias, [], [ws], [], [], input_scale_fmt=fmt)
    ref = _reference_linear(x3d.reshape(B, K).contiguous(), w, ws, bias, fmt)
    assert out.shape == (B, rank)
    assert torch.equal(out, ref.reshape_as(out))


def test_prefill_path_gate_and_result_unchanged():
    """M>4 must NOT take the fused path and stays bit-identical to the reference."""
    M, N, K = 64, 512, 2048
    x = _rand_x(M, K, seed=17)
    assert not _use_fused_act_quant_matmul(x, N, K, 128)
    w, ws = _make_weight(N, K, seed=19)
    out = lin_op(x, w, None, [], [ws], [], [])
    ref = _reference_linear(x, w, ws, None, "")
    assert torch.equal(out, ref)
