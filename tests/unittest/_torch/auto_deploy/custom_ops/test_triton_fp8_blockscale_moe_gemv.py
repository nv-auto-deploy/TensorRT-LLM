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

"""Unit tests for the M==1 Triton fp8-blockscale MoE GEMV chain.

The chain must reproduce the FineGrainedFP8 gated-MLP MoE semantics
(``out = sum_k rw[k] * W2_k @ (silu(W1_k x) * W3_k x)`` with per-128x128-block
weight dequant, FC1 stacked as ``cat([w3, w1], dim=1)``) for the slots whose
GLOBAL expert id falls in the local EP range, and produce zeros when no slot is
local. It must also stay CUDA-graph capturable with expert selection read
on-device at replay time.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.triton_fp8_blockscale_moe_gemv import (
    can_use_fp8_blockscale_moe_gemv,
    fp8_blockscale_moe_gemv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# (num_local_experts, intermediate, hidden): production is the Step-3.7-Flash
# per-rank EP8 routed expert shape (36, 1280, 4096) with a reduced expert count.
SHAPES = [
    (4, 1280, 4096),
    (3, 256, 512),
]
TOP_K = 8


def _make_moe_weights(num_experts, inter, hidden, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=gen, dtype=torch.float32)

    fc1 = (0.5 * randn(num_experts, 2 * inter, hidden)).to(torch.float8_e4m3fn)
    fc2 = (0.5 * randn(num_experts, hidden, inter)).to(torch.float8_e4m3fn)
    fc1_s = (0.02 * randn(num_experts, 2 * inter // 128, hidden // 128).abs() + 0.005).contiguous()
    fc2_s = (0.02 * randn(num_experts, hidden // 128, inter // 128).abs() + 0.005).contiguous()
    x = (0.1 * randn(1, hidden)).to(torch.bfloat16)
    rw = torch.rand(1, TOP_K, device="cuda", generator=gen, dtype=torch.float32)
    rw = (rw / rw.sum()).contiguous()
    return x, rw, fc1, fc1_s, fc2, fc2_s


def _dequant(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k = w_fp8.shape
    s = scale.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)[:, :k]
    return w_fp8.to(torch.float32) * s


def _ref_moe(x, se, rw, fc1, fc1_s, fc2, fc2_s, local_offset):
    """fp32 reference with the same per-block dequant and bf16 act intermediate."""
    num_local, two_inter, _ = fc1.shape
    inter = two_inter // 2
    xf = x.to(torch.float32)
    out = torch.zeros(1, fc2.shape[1], dtype=torch.float32, device=x.device)
    for k in range(se.shape[-1]):
        e = int(se[0, k]) - local_offset
        if 0 <= e < num_local:
            w1full = _dequant(fc1[e], fc1_s[e])
            up = xf @ w1full[:inter].T
            gate = xf @ w1full[inter:].T
            act = torch.nn.functional.silu(gate) * up
            # the kernel stores the SwiGLU intermediate as bf16
            act = act.to(torch.bfloat16).to(torch.float32)
            out += rw[0, k].item() * (act @ _dequant(fc2[e], fc2_s[e]).T)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("num_local,inter,hidden", SHAPES)
@pytest.mark.parametrize(
    "local_offset,expert_pick",
    [
        (0, "all_local"),  # single-rank layout: every slot local
        (72, "mixed"),  # EP layout: some slots local, some remote
        (72, "none_local"),  # EP layout: no slot local -> zeros
    ],
)
def test_matches_blockscale_reference(num_local, inter, hidden, local_offset, expert_pick):
    x, rw, fc1, fc1_s, fc2, fc2_s = _make_moe_weights(num_local, inter, hidden)
    gen = torch.Generator(device="cuda").manual_seed(1234)
    if expert_pick == "all_local":
        se = torch.randint(
            local_offset, local_offset + num_local, (1, TOP_K), device="cuda", generator=gen
        ).to(torch.int32)
    elif expert_pick == "mixed":
        # half the slots in the local range, half below/above it
        local = torch.randint(
            local_offset, local_offset + num_local, (1, TOP_K // 2), device="cuda", generator=gen
        )
        remote = torch.randint(
            0, local_offset, (1, TOP_K - TOP_K // 2), device="cuda", generator=gen
        )
        se = torch.cat([local, remote], dim=1).to(torch.int32)
    else:
        se = torch.randint(0, local_offset, (1, TOP_K), device="cuda", generator=gen).to(
            torch.int32
        )
    se = se.contiguous()

    assert can_use_fp8_blockscale_moe_gemv(x, se, rw, fc1, fc2, fc1_s, fc2_s)
    out = fp8_blockscale_moe_gemv(x, se, rw, fc1, fc1_s, fc2, fc2_s, local_offset)
    ref = _ref_moe(x, se, rw, fc1, fc1_s, fc2, fc2_s, local_offset)

    if expert_pick == "none_local":
        assert out.abs().max().item() == 0.0
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_cuda_graph_replay_tracks_expert_selection():
    """Replay must honor selected_experts/routing values written after capture."""
    num_local, inter, hidden = SHAPES[0]
    x, rw, fc1, fc1_s, fc2, fc2_s = _make_moe_weights(num_local, inter, hidden, seed=7)
    se = torch.randint(0, num_local, (1, TOP_K), device="cuda").to(torch.int32).contiguous()

    static_out = torch.empty(1, hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        static_out.copy_(fp8_blockscale_moe_gemv(x, se, rw, fc1, fc1_s, fc2, fc2_s, 0))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):  # warmup compiles the kernels outside capture
            run()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for trial_seed in range(3):
        gen = torch.Generator(device="cuda").manual_seed(trial_seed)
        se.copy_(
            torch.randint(0, 2 * num_local, (1, TOP_K), device="cuda", generator=gen).to(
                torch.int32
            )
        )
        rw.copy_(torch.rand(1, TOP_K, device="cuda", generator=gen, dtype=torch.float32))
        graph.replay()
        torch.cuda.synchronize()
        ref = _ref_moe(x, se, rw, fc1, fc1_s, fc2, fc2_s, 0)
        torch.testing.assert_close(static_out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_gate_rejects_multi_token_and_bad_dtypes():
    num_local, inter, hidden = SHAPES[1]
    x, rw, fc1, fc1_s, fc2, fc2_s = _make_moe_weights(num_local, inter, hidden, seed=3)
    se = torch.zeros(1, TOP_K, dtype=torch.int32, device="cuda")

    assert can_use_fp8_blockscale_moe_gemv(x, se, rw, fc1, fc2, fc1_s, fc2_s)
    x2 = torch.cat([x, x], dim=0)
    assert not can_use_fp8_blockscale_moe_gemv(x2, se, rw, fc1, fc2, fc1_s, fc2_s)
    assert not can_use_fp8_blockscale_moe_gemv(x, se, rw, fc1.view(torch.int8), fc2, fc1_s, fc2_s)
    assert not can_use_fp8_blockscale_moe_gemv(x, se.long(), rw, fc1, fc2, fc1_s, fc2_s)
    assert not can_use_fp8_blockscale_moe_gemv(x, se, rw.half(), fc1, fc2, fc1_s, fc2_s)
