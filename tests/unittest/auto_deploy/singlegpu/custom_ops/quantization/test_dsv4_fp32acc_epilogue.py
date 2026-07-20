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

"""Correctness tests for the DeepSeek-V4 grouped fine-grained FP8 epilogue fold."""

import pytest
import torch

# Register the custom ops (side-effect import).
import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    torch_fake_quant_grouped_finegrained_fp8_linear,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEVICE = "cuda"
DSV4_FLASH_HIDDEN = 4096


def _rand_acc(shape: tuple[int, ...], seed: int, scale: float = 8.0) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float32) * scale


def _fp8_weight(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 400)
    weight = torch.randn((n, k), generator=generator, device=DEVICE, dtype=torch.float32)
    weight_fp8 = weight.clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(
            ((n + 127) // 128, (k + 127) // 128),
            generator=generator,
            device=DEVICE,
            dtype=torch.float32,
        )
        * 0.02
        + 0.005
    )
    return weight_fp8, scale


def _assert_equal_up_to_splitk_atomic_wiggle(
    out: torch.Tensor, ref: torch.Tensor, max_frac: float = 2e-3
) -> None:
    """Allow only the pre-existing cross-launch split-K atomic-order variation."""
    if torch.equal(out, ref):
        return
    diff = out != ref
    mismatch_fraction = diff.float().mean().item()
    assert mismatch_fraction <= max_frac, (
        f"mismatch fraction {mismatch_fraction} exceeds atomic-wiggle allowance"
    )
    actual = out[diff].float()
    expected = ref[diff].float()
    assert torch.allclose(actual, expected, rtol=1.6e-2, atol=1e-20), (
        "mismatches exceed BF16 rounding-boundary flips: "
        f"{actual[:4].tolist()} vs {expected[:4].tolist()}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("tokens", [1, 2])
def test_grouped_decode_single_accumulator_matches_per_group_chain(seed: int, tokens: int) -> None:
    """The shared accumulator preserves the prior per-group split-K result."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
        _safe_act_quant,
        _w8a8_block_fp8_matmul_triton,
    )

    num_groups = 2
    rank = 1024
    input_tensor = _rand_acc((1, tokens, num_groups, DSV4_FLASH_HIDDEN), seed, scale=1.0).to(
        torch.bfloat16
    )
    weight_fp8, weight_scale = _fp8_weight(num_groups * rank, DSV4_FLASH_HIDDEN, seed)

    out = torch_fake_quant_grouped_finegrained_fp8_linear(
        input_tensor, weight_fp8, None, [], [weight_scale], [], []
    )

    qinput, input_scales = _safe_act_quant(input_tensor.contiguous(), 128, "")
    m_tokens = qinput.numel() // (num_groups * DSV4_FLASH_HIDDEN)
    grouped_input = qinput.reshape(m_tokens, num_groups, DSV4_FLASH_HIDDEN)
    grouped_input_scale = input_scales.reshape(m_tokens, num_groups, input_scales.shape[-1])
    grouped_weight = weight_fp8.view(num_groups, rank, DSV4_FLASH_HIDDEN)
    grouped_weight_scale = weight_scale.view(
        num_groups, weight_scale.shape[0] // num_groups, weight_scale.shape[1]
    )
    parts = [
        _w8a8_block_fp8_matmul_triton(
            grouped_input[:, group, :].contiguous(),
            grouped_weight[group].contiguous(),
            grouped_input_scale[:, group, :].contiguous(),
            grouped_weight_scale[group].contiguous(),
            [128, 128],
            output_dtype=input_tensor.dtype,
        )
        for group in range(num_groups)
    ]
    ref = torch.stack(parts, dim=1).reshape(1, tokens, num_groups * rank)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    _assert_equal_up_to_splitk_atomic_wiggle(out, ref)


@pytest.mark.parametrize("rank", [256, 1024])
def test_grouped_decode_cuda_graph_replays_reset_shared_accumulator(rank: int) -> None:
    """Each replay zeros both legacy and TP4-tuned grouped accumulators.

    ``rank=1024`` is the exact DeepSeek-V4-Flash TP4 per-group shape and engages
    the M=1, K=4096, SPLIT_K=32, BLOCK_SIZE_N=64, two-warp schedule.
    """
    num_groups = 2
    tokens = 1
    weight_fp8, weight_scale = _fp8_weight(num_groups * rank, DSV4_FLASH_HIDDEN, 91)
    static_input = _rand_acc((1, tokens, num_groups, DSV4_FLASH_HIDDEN), 92, scale=1.0).to(
        torch.bfloat16
    )

    def run(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch_fake_quant_grouped_finegrained_fp8_linear(
            input_tensor, weight_fp8, None, [], [weight_scale], [], []
        )

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
        fresh_input = _rand_acc((1, tokens, num_groups, DSV4_FLASH_HIDDEN), seed, scale=1.0).to(
            torch.bfloat16
        )
        ref = run(fresh_input)
        static_input.copy_(fresh_input)
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal_up_to_splitk_atomic_wiggle(static_out, ref)
        replay_outputs.append(static_out.clone())

    assert not torch.equal(replay_outputs[0], replay_outputs[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
