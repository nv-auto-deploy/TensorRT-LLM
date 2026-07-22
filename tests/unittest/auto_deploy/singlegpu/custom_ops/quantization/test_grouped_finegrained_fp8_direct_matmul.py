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

"""Tests for ``auto_deploy::torch_fake_quant_grouped_finegrained_fp8_linear``.

Covers direct block-FP8 dispatch, the dense-dequant fallback, the quant prologue, and
the shared fp32 decode accumulator (incl. CUDA-graph re-zeroing).
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization import torch_quant

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 9),
    reason="Requires CUDA + FP8 (SM89+)",
)

_GROUPED_OP = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear
_LINEAR_OP = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
_BLOCK = 128
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _make_inputs(num_groups, rank, in_features, batch=2, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    out_rows = num_groups * rank
    assert out_rows % _BLOCK == 0 and in_features % _BLOCK == 0
    w_fp8 = torch.randn(
        out_rows, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    scale_shape = (out_rows // _BLOCK, in_features // _BLOCK)
    scale = torch.rand(scale_shape, generator=gen, device="cuda", dtype=torch.float32) + 0.5
    x = torch.randn(
        batch, num_groups, in_features, generator=gen, device="cuda", dtype=torch.bfloat16
    )
    return x, w_fp8, scale


def _dense_dequant_ref(x, weight_fp8, scale, bias=None, input_scale_fmt=""):
    """Pure-torch reference on the flattened checkpoint's global scale grid."""
    num_groups = x.shape[-2]
    rank = weight_fp8.shape[0] // num_groups
    x_blocks = x.contiguous().view(*x.shape[:-1], -1, _BLOCK)
    amax = x_blocks.abs().float().amax(dim=-1)
    if input_scale_fmt.lower() == "ue8m0":
        input_scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp(min=1e-4) / FP8_MAX)))
    else:
        input_scale = torch.clamp(amax / FP8_MAX, min=1e-12)
    qinput = (x_blocks.float() / input_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    input_dequant = (qinput.float() * input_scale.unsqueeze(-1)).view_as(x).to(x.dtype)

    weight_scale = scale.repeat_interleave(_BLOCK, dim=0).repeat_interleave(_BLOCK, dim=1)
    weight_scale = weight_scale[: weight_fp8.shape[0], : weight_fp8.shape[1]]
    weight = (weight_fp8.float() * weight_scale).to(x.dtype).view(num_groups, rank, x.shape[-1])
    out = torch.matmul(input_dequant.unsqueeze(-2), weight.transpose(-1, -2)).squeeze(-2)
    out = out.flatten(-2)
    if bias is not None:
        out = out + bias.reshape(weight_fp8.shape[0]).to(out.dtype)
    return out


# fmt="" quantizes standalone in the op body; "ue8m0" defers into the kernel prologue.
@pytest.mark.parametrize("input_scale_fmt", ["", "ue8m0"])
def test_single_group_matches_nongrouped_fp8_bitwise(input_scale_fmt):
    rank, in_features = 256, 2048  # K < 4096 -> deterministic full-K kernel
    x, w_fp8, scale = _make_inputs(1, rank, in_features, seed=1)

    out_grouped = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [], input_scale_fmt=input_scale_fmt)
    out_linear = _LINEAR_OP(
        x.reshape(x.shape[0], in_features),
        w_fp8,
        None,
        [],
        [scale],
        [],
        [],
        input_scale_fmt=input_scale_fmt,
    )

    assert out_grouped.shape == (x.shape[0], rank)
    assert out_grouped.dtype == torch.bfloat16
    assert torch.equal(out_grouped, out_linear.reshape_as(out_grouped))


def test_single_group_matches_nongrouped_fp8_splitk_close():
    rank, in_features = 1024, 4096  # DSV4 wo_a per-rank shape; K >= 4096 -> split-K
    x, w_fp8, scale = _make_inputs(1, rank, in_features, seed=1)

    out_grouped = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [])
    out_linear = _LINEAR_OP(
        x.reshape(x.shape[0], in_features), w_fp8, None, [], [scale], [], []
    ).reshape_as(out_grouped)

    # Split-K reduces via fp32 atomics: two launches agree only up to ~1 ULP.
    torch.testing.assert_close(out_grouped, out_linear, rtol=2e-2, atol=1.0)
    cos = torch.nn.functional.cosine_similarity(
        out_grouped.float().reshape(-1), out_linear.float().reshape(-1), dim=0
    )
    assert cos > 0.9999, f"cosine similarity too low: {cos.item()}"


def test_multi_group_matches_per_group_nongrouped_bitwise():
    num_groups, rank, in_features = 3, 256, 512  # K < 4096 -> deterministic full-K
    x, w_fp8, scale = _make_inputs(num_groups, rank, in_features, seed=2)

    out_grouped = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [])

    wq = w_fp8.view(num_groups, rank, in_features)
    sg = scale.view(num_groups, rank // _BLOCK, in_features // _BLOCK)
    parts = [
        _LINEAR_OP(
            x[:, g, :].contiguous(), wq[g].contiguous(), None, [], [sg[g].contiguous()], [], []
        )
        for g in range(num_groups)
    ]
    ref = torch.stack(parts, dim=1).reshape(x.shape[0], num_groups * rank)

    assert out_grouped.shape == (x.shape[0], num_groups * rank)
    assert torch.equal(out_grouped, ref)


def test_multi_group_matches_dense_dequant_ue8m0_reference():
    num_groups, rank = 2, 128
    x, w_fp8, scale = _make_inputs(num_groups, rank, 128, batch=6, seed=3)
    x = x.reshape(2, 3, num_groups, 128)  # 4-D lead shape, prefill M -> stacked path
    bias = torch.randn(num_groups * rank, device="cuda", dtype=torch.bfloat16)

    out = _GROUPED_OP(x, w_fp8, bias, [], [scale], [], [], input_scale_fmt="ue8m0")
    ref = _dense_dequant_ref(x, w_fp8, scale, bias=bias, input_scale_fmt="ue8m0")

    assert out.shape == (2, 3, num_groups * rank)
    torch.testing.assert_close(out, ref, rtol=0.02, atol=1.0)


def test_ragged_output_rows_keep_canonical_scale_blocks():
    rank, in_features = 576, 256  # rank % 128 != 0 with the canonical 128-row grid
    x = torch.ones(1, 1, in_features, device="cuda", dtype=torch.bfloat16)
    weight_fp8 = torch.ones(rank, in_features, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    scale = torch.arange(1, 6, device="cuda", dtype=torch.float32).unsqueeze(1).repeat(1, 2)

    output = _GROUPED_OP(x, weight_fp8, None, [], [scale], [], [])
    assert output.shape == (1, rank)
    ref = _dense_dequant_ref(x, weight_fp8, scale)
    torch.testing.assert_close(output, ref, rtol=2e-2, atol=1.0)


def test_global_scale_blocks_may_span_group_boundaries():
    num_groups, rank, in_features = 2, 192, 128  # rank % 128 != 0 -> dense-dequant fallback
    x = torch.ones(1, num_groups, in_features, device="cuda", dtype=torch.bfloat16)
    x[:, 1, :].mul_(2)
    weight_fp8 = torch.ones(num_groups * rank, in_features, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([[1.0], [2.0], [4.0]], device="cuda", dtype=torch.float32)
    bias = torch.arange(num_groups * rank, device="cuda", dtype=torch.bfloat16) / 128

    output = _GROUPED_OP(x, weight_fp8, bias, [], [scale], [], [])
    ref = _dense_dequant_ref(x, weight_fp8, scale, bias=bias)

    assert output.shape == (1, num_groups * rank)
    torch.testing.assert_close(output, ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("num_groups", [1, 2])
def test_grouped_uses_quant_prologue(monkeypatch, num_groups):
    # ue8m0 decode must never call the standalone activation quant.
    rank, K = 1024, 4096
    torch.manual_seed(43)
    vals = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0], device="cuda")
    x = vals[torch.randint(0, 7, (1, 1, num_groups, K), device="cuda")].to(torch.bfloat16)
    w_fp8 = (
        torch.randint(-2, 3, (num_groups * rank, K), device="cuda").float().to(torch.float8_e4m3fn)
    )
    ws_inv = torch.ones(
        num_groups * rank // _BLOCK, K // _BLOCK, device="cuda", dtype=torch.float32
    )

    def _fail(*args, **kwargs):
        raise AssertionError("supported decode path must fuse activation quantization")

    monkeypatch.setattr(torch_quant, "_safe_act_quant", _fail)
    out = _GROUPED_OP(x, w_fp8, None, [], [ws_inv], [], [], input_scale_fmt="ue8m0")
    assert out.shape == (1, 1, num_groups * rank)


# --- shared fp32 decode accumulator (split-K branch writes disjoint column
# --- slices of one pre-zeroed buffer, single finish cast) ---


def _assert_equal_up_to_splitk_atomic_wiggle(out, ref, max_frac=2e-3):
    if torch.equal(out, ref):
        return
    diff = out != ref
    assert diff.float().mean().item() <= max_frac
    assert torch.allclose(out[diff].float(), ref[diff].float(), rtol=1.6e-2, atol=1e-20)


@pytest.mark.parametrize("tokens", [1, 2])
def test_grouped_decode_shared_accumulator_matches_per_group_chain(tokens):
    num_groups, rank, K = 2, 1024, 4096
    x3, w_fp8, scale = _make_inputs(num_groups, rank, K, batch=tokens, seed=5)
    x = x3.unsqueeze(0)  # [1, tokens, G, K]

    out = _GROUPED_OP(x, w_fp8, None, [], [scale], [], [])

    qinput, input_scales = torch_quant._safe_act_quant(x.contiguous(), _BLOCK, "")
    qin = qinput.reshape(tokens, num_groups, K)
    sin = input_scales.reshape(tokens, num_groups, input_scales.shape[-1])
    wq = w_fp8.view(num_groups, rank, K)
    sq = scale.view(num_groups, scale.shape[0] // num_groups, scale.shape[1])
    parts = [
        torch_quant._w8a8_block_fp8_matmul_triton(
            qin[:, g, :].contiguous(),
            wq[g].contiguous(),
            sin[:, g, :].contiguous(),
            sq[g].contiguous(),
            [_BLOCK, _BLOCK],
            output_dtype=x.dtype,
        )
        for g in range(num_groups)
    ]
    ref = torch.stack(parts, dim=1).reshape(1, tokens, num_groups * rank)

    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    _assert_equal_up_to_splitk_atomic_wiggle(out, ref)


def test_grouped_decode_cuda_graph_replays_reset_shared_accumulator():
    num_groups, rank, K = 2, 1024, 4096
    _, w_fp8, scale = _make_inputs(num_groups, rank, K, batch=1, seed=91)

    def _input(seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(1, 1, num_groups, K, generator=gen, device="cuda").to(torch.bfloat16)

    static_input = _input(92)

    def run(inp):
        return _GROUPED_OP(inp, w_fp8, None, [], [scale], [], [])

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            run(static_input)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = run(static_input)

    replay_outputs = []
    for seed in (93, 94):
        fresh_input = _input(seed)
        ref = run(fresh_input)
        static_input.copy_(fresh_input)
        graph.replay()
        torch.cuda.synchronize()
        # Stale (non-re-zeroed) accumulator contents would break this equality.
        _assert_equal_up_to_splitk_atomic_wiggle(static_out, ref)
        replay_outputs.append(static_out.clone())

    assert not torch.equal(replay_outputs[0], replay_outputs[1])
