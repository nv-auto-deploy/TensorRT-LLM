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

"""Bit-exact parity tests for the EP-LOCALIZED DeepSeek V4 router gate ops.

``deepseek_v4_routing_localized`` / ``deepseek_v4_hash_routing_localized`` fold the
EP global->local localization (``deepseek_v4_localize_routing``) into the gate
kernel tail. The contract is byte-exact equality with running the two ops back to
back, across decode (fused kernel) and prefill (hash reference-chain branch)
token counts, EP shards including the last-rank remainder, and fully-off-rank
tokens (sentinel + zero weight).
"""

import pytest
import torch

# Importing the modules registers the auto_deploy custom ops.
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import (  # noqa: F401
    deepseek_v4_localize_routing,
    deepseek_v4_routing,
)

# (expert_start, local_experts, num_global): non-EP full set, EP4 shards (incl. first,
# middle, and a last-rank remainder split of 250 -> 62/62/62/64).
_SHARDS = [
    (0, 256, 256),
    (0, 64, 256),
    (128, 64, 256),
    (186, 64, 250),
]


def _compose_reference(sel_global, rw_global, expert_start, local_experts):
    """The unfused chain: global gate outputs -> standalone localize op."""
    return torch.ops.auto_deploy.deepseek_v4_localize_routing(
        sel_global, rw_global, expert_start, local_experts
    )


def _assert_localized_pair_equal(out, ref, local_experts):
    out_idx, out_w = out
    ref_idx, ref_w = ref
    assert out_idx.dtype == torch.int32
    assert out_w.dtype == torch.bfloat16
    assert torch.equal(out_idx, ref_idx), f"local_idx mismatch\nref={ref_idx}\nout={out_idx}"
    assert torch.equal(out_w, ref_w), f"weights mismatch\nref={ref_w}\nout={out_w}"
    sentinel = out_idx == local_experts
    if sentinel.any():
        assert torch.equal(out_w[sentinel], torch.zeros_like(out_w[sentinel]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 2, 64])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
@pytest.mark.parametrize("expert_start,local_experts,num_global", _SHARDS)
def test_routing_localized_bit_exact(
    num_tokens, norm_topk_prob, expert_start, local_experts, num_global
):
    torch.manual_seed(num_tokens * 131 + expert_start * 7 + local_experts)
    device = "cuda"
    top_k = 8
    rsf = 1.5

    router_logits = torch.randn(num_tokens, num_global, device=device, dtype=torch.float32)
    bias = torch.randn(num_global, device=device, dtype=torch.float32) * 0.1

    sel_g, rw_g = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, rsf, norm_topk_prob
    )
    ref = _compose_reference(sel_g, rw_g, expert_start, local_experts)
    out = torch.ops.auto_deploy.deepseek_v4_routing_localized(
        router_logits, bias, top_k, rsf, norm_topk_prob, expert_start, local_experts
    )
    _assert_localized_pair_equal(out, ref, local_experts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# 1/2 exercise the fused decode kernel; 64 exercises the prefill reference-chain
# branch (> _HASH_ROUTING_DECODE_MAX_TOKENS) with the eager localization mirror.
@pytest.mark.parametrize("num_tokens", [1, 2, 64])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
@pytest.mark.parametrize("expert_start,local_experts,num_global", _SHARDS)
def test_hash_routing_localized_bit_exact(
    num_tokens, norm_topk_prob, expert_start, local_experts, num_global
):
    torch.manual_seed(num_tokens * 977 + expert_start * 13 + local_experts)
    device = "cuda"
    top_k = 8
    hidden = 512
    vocab = 1024
    rsf = 2.5

    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(num_global, hidden, device=device, dtype=torch.float32) * 0.05
    tid2eid = torch.randint(0, num_global, (vocab, top_k), device=device, dtype=torch.int64)
    # Force one vocab row fully OFF-rank (when the shard is a strict subset) so a token
    # can hit the all-sentinel/all-zero case.
    if local_experts < num_global:
        off_rank = (expert_start + local_experts) % num_global
        tid2eid[0] = off_rank
    input_ids = torch.randint(0, vocab, (num_tokens,), device=device, dtype=torch.int64)
    input_ids[0] = 0

    sel_g, rw_g = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden_states, weight, tid2eid, input_ids, rsf, norm_topk_prob
    )
    ref = _compose_reference(sel_g, rw_g, expert_start, local_experts)
    out = torch.ops.auto_deploy.deepseek_v4_hash_routing_localized(
        hidden_states, weight, tid2eid, input_ids, rsf, norm_topk_prob, expert_start, local_experts
    )
    _assert_localized_pair_equal(out, ref, local_experts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_routing_localized_rejects_nonpositive_local_experts():
    device = "cuda"
    router_logits = torch.randn(1, 16, device=device, dtype=torch.float32)
    bias = torch.zeros(16, device=device, dtype=torch.float32)
    with pytest.raises(Exception, match="local_experts"):
        torch.ops.auto_deploy.deepseek_v4_routing_localized(router_logits, bias, 4, 1.0, True, 0, 0)


if __name__ == "__main__":
    for t in (1, 2, 64):
        for norm in (True, False):
            for shard in _SHARDS:
                test_routing_localized_bit_exact(t, norm, *shard)
                test_hash_routing_localized_bit_exact(t, norm, *shard)
    test_routing_localized_rejects_nonpositive_local_experts()
    print("all localized-routing parity cases OK")
