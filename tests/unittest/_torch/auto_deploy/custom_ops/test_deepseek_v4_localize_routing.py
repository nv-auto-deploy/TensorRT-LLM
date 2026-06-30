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

"""Bit-exact parity test for the fused DeepSeek V4 EP localize-routing custom op."""

import pytest
import torch

# Importing the module registers the auto_deploy::deepseek_v4_localize_routing custom op.
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import (  # noqa: F401
    deepseek_v4_localize_routing,
)


def _reference_localize(
    selected_experts: torch.Tensor,
    routing_weights_f32: torch.Tensor,
    expert_start: int,
    local_experts: int,
):
    """Byte-faithful copy of the original (model-cast + op-side) localization chain.

    Mirrors modeling line ``routing_weights.to(bf16)`` followed by the unfused chain in
    ``_run_trtllm_gen_mxfp4_from_routing``.
    """
    num_tokens = selected_experts.shape[0]
    # Model-side bf16 cast (now fused away — kept here to prove bit-equality).
    rw_bf16 = routing_weights_f32.to(torch.bfloat16)
    sel = selected_experts.reshape(num_tokens, -1)
    top_k = sel.shape[-1]
    local_idx = sel.to(torch.int64) - int(expert_start)
    valid = (local_idx >= 0) & (local_idx < local_experts)
    local_idx = torch.where(valid, local_idx, torch.full_like(local_idx, local_experts)).to(
        torch.int32
    )
    weights = (rw_bf16.reshape(num_tokens, top_k).to(torch.float32) * valid).to(torch.bfloat16)
    return local_idx, weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 2, 256, 1000])
@pytest.mark.parametrize("top_k", [6, 8])
# (expert_start, local_experts, num_global): non-EP (start=0, full set) and EP shards
# (start>0, a window of experts) so off-rank / out-of-range routes hit the sentinel path.
@pytest.mark.parametrize(
    "expert_start,local_experts,num_global",
    [(0, 256, 256), (32, 32, 256), (224, 32, 256), (96, 32, 256)],
)
def test_localize_routing_bit_exact(num_tokens, top_k, expert_start, local_experts, num_global):
    torch.manual_seed(num_tokens * 977 + top_k * 31 + expert_start + local_experts)
    device = "cuda"
    # Global expert ids spanning the full routing space so a large fraction lands off-rank
    # (sentinel) for the EP shards.
    selected_experts = torch.randint(
        0, num_global, (num_tokens, top_k), device=device, dtype=torch.int64
    )
    routing_weights = torch.randn(num_tokens, top_k, device=device, dtype=torch.float32)

    ref_idx, ref_w = _reference_localize(
        selected_experts, routing_weights, expert_start, local_experts
    )
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_localize_routing(
        selected_experts, routing_weights, expert_start, local_experts
    )

    assert out_idx.dtype == torch.int32
    assert out_w.dtype == torch.bfloat16
    assert out_idx.shape == (num_tokens, top_k)
    assert out_w.shape == (num_tokens, top_k)

    # Deterministic integer + masked-bf16 outputs -> byte-exact.
    assert torch.equal(out_idx, ref_idx), f"local_idx mismatch\nref={ref_idx}\nout={out_idx}"
    assert torch.equal(out_w, ref_w), f"weights mismatch\nref={ref_w}\nout={out_w}"

    # Sanity: every off-rank route carries the invalid sentinel and a zero weight.
    sentinel = out_idx == local_experts
    if sentinel.any():
        assert torch.equal(out_w[sentinel], torch.zeros_like(out_w[sentinel]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_localize_routing_bf16_input_matches_f32_input():
    """Feeding pre-cast bf16 weights (old call site) == feeding fp32 (new call site)."""
    device = "cuda"
    torch.manual_seed(1234)
    selected_experts = torch.randint(0, 256, (2, 8), device=device, dtype=torch.int64)
    routing_weights = torch.randn(2, 8, device=device, dtype=torch.float32)

    out_idx_f32, out_w_f32 = torch.ops.auto_deploy.deepseek_v4_localize_routing(
        selected_experts, routing_weights, 32, 32
    )
    out_idx_bf16, out_w_bf16 = torch.ops.auto_deploy.deepseek_v4_localize_routing(
        selected_experts, routing_weights.to(torch.bfloat16), 32, 32
    )
    assert torch.equal(out_idx_f32, out_idx_bf16)
    assert torch.equal(out_w_f32, out_w_bf16)
