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

"""DeepSeek V4 router-gate custom-op tests.

Learned sqrtsoftplus gate, hash gate, and EP-localized variants vs independent torch references.
"""

import pytest
import torch
import torch.nn.functional as F

# Importing the module registers the auto_deploy::deepseek_v4_* routing custom ops.
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import deepseek_v4_routing  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.deepseek_v4_routing import (
    _localize_routing_eager,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# ---------------------------------------------------------------------------
# Learned sqrtsoftplus gate (deepseek_v4_routing)
# ---------------------------------------------------------------------------


def _reference_routing(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    scores = F.softplus(router_logits).sqrt()
    selected_experts = (scores + bias).topk(top_k, dim=-1).indices
    routing_weights = scores.gather(1, selected_experts)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * routed_scaling_factor
    return selected_experts, routing_weights


@requires_cuda
@pytest.mark.parametrize(
    "num_tokens,num_experts,norm_topk_prob",
    [
        (1, 256, True),
        (1, 256, False),
        (128, 256, True),
        (5, 250, True),  # BLOCK_E > num_experts: padding experts must never be selected
    ],
)
def test_deepseek_v4_routing_parity(num_tokens, num_experts, norm_topk_prob):
    torch.manual_seed(num_tokens * 131 + num_experts + int(norm_topk_prob))
    top_k, routed_scaling_factor = 6, 1.5
    device = "cuda"
    # *8 spreads logits over both softplus branches (x>20 linear).
    router_logits = (torch.randn(num_tokens, num_experts, device=device) * 8.0).float()
    bias = (torch.randn(num_experts, device=device) * 0.5).float()

    ref_idx, ref_w = _reference_routing(
        router_logits, bias, top_k, routed_scaling_factor, norm_topk_prob
    )
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, routed_scaling_factor, norm_topk_prob
    )

    assert out_idx.dtype == torch.int64
    assert out_w.dtype == torch.float32
    assert out_idx.shape == (num_tokens, top_k)
    assert out_w.shape == (num_tokens, top_k)
    assert torch.equal(out_idx, ref_idx.to(torch.int64))
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_deepseek_v4_routing_softplus_threshold():
    torch.manual_seed(0)
    device = "cuda"
    num_tokens, num_experts, top_k = 4, 256, 6
    # Push many logits past the softplus linear-branch threshold of 20.
    router_logits = (torch.randn(num_tokens, num_experts, device=device) * 5.0 + 18.0).float()
    bias = torch.zeros(num_experts, device=device).float()

    ref_idx, ref_w = _reference_routing(router_logits, bias, top_k, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx.to(torch.int64))
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_deepseek_v4_routing_preserves_large_negative_softplus(norm_topk_prob):
    device = "cuda"
    num_experts, top_k = 256, 6
    router_logits = torch.stack(
        [
            torch.full((num_experts,), -20.0, device=device),
            torch.full((num_experts,), -30.0, device=device),
        ]
    )
    # Bias forces selection of experts whose softplus weights are tiny (~3e-7) but positive.
    bias = torch.full((num_experts,), -100.0, device=device)
    bias[:top_k] = torch.arange(top_k, 0, -1, device=device)

    ref_idx, ref_w = _reference_routing(router_logits, bias, top_k, 1.5, norm_topk_prob)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, 1.5, norm_topk_prob
    )

    assert torch.equal(out_idx, ref_idx.to(torch.int64))
    assert (out_w > 0).all()
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-9)


@requires_cuda
def test_deepseek_v4_routing_exact_tie_smallest_index():
    device = "cuda"
    num_experts, top_k = 256, 6
    bias = torch.zeros(num_experts, device=device).float()

    # Experts 100/200 share the exact rank-6 logit: the smaller index must win, deterministically.
    logits = torch.full((1, num_experts), -10.0, device=device)
    logits[0, :5] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0], device=device)
    logits[0, 100] = 4.0
    logits[0, 200] = 4.0
    outs = [
        torch.ops.auto_deploy.deepseek_v4_routing(logits, bias, top_k, 1.5, True) for _ in range(10)
    ]
    for idx, _ in outs[1:]:
        assert torch.equal(idx, outs[0][0])
    idx = outs[0][0][0].tolist()
    assert idx[:5] == [0, 1, 2, 3, 4]
    assert idx[5] == 100

    # 3-way tie (50/150/250) with 2 slots left: the kernel masks the whole tied group after
    # selecting its smallest index, so exactly one member is selected and 10 fills the last slot.
    logits2 = torch.full((1, num_experts), -10.0, device=device)
    logits2[0, :4] = torch.tensor([9.0, 8.0, 7.0, 6.0], device=device)
    for e in (50, 150, 250):
        logits2[0, e] = 4.0
    logits2[0, 10] = 3.0
    idx2 = torch.ops.auto_deploy.deepseek_v4_routing(logits2, bias, top_k, 1.5, True)[0][0].tolist()
    assert idx2[:4] == [0, 1, 2, 3]
    assert idx2[4] == 50
    assert idx2[5] == 10


@requires_cuda
def test_deepseek_v4_routing_near_tie_ulp_selection():
    device = "cuda"
    num_experts, top_k = 256, 6
    bias = torch.zeros(num_experts, device=device).float()

    # Rank-6/7 logits a few ULPs apart, larger score at the LARGER index: the strictly
    # greater score must win deterministically and match the reference.
    for ulps in (4, 8):
        logits = torch.full((1, num_experts), -10.0, device=device)
        logits[0, :5] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0], device=device)
        base = torch.tensor(4.0, device=device)
        stepped = base
        for _ in range(ulps):
            stepped = torch.nextafter(stepped, torch.tensor(float("inf"), device=device))
        logits[0, 100] = base
        logits[0, 200] = stepped
        ref_idx, _ = _reference_routing(logits, bias, top_k, 1.5, True)
        for _ in range(10):
            out_idx, _ = torch.ops.auto_deploy.deepseek_v4_routing(logits, bias, top_k, 1.5, True)
            assert torch.equal(out_idx, ref_idx.to(torch.int64))
        assert out_idx[0, 5].item() == 200


# ---------------------------------------------------------------------------
# Hash gate (deepseek_v4_hash_routing)
# ---------------------------------------------------------------------------


def _reference_hash_routing(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    router_logits = F.linear(hidden_states.to(weight.dtype), weight).float()
    scores = F.softplus(router_logits).sqrt()
    selected_experts = tid2eid[input_ids.to(torch.long)].to(torch.int64)
    routing_weights = scores.gather(1, selected_experts)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * routed_scaling_factor
    return selected_experts, routing_weights


def _make_hash_inputs(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    vocab_size: int,
    top_k: int,
    weight_std: float,
    hidden_dtype: torch.dtype,
    ids_dtype: torch.dtype,
    seed: int,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    hidden = (torch.randn(num_tokens, hidden_size, device=device) * 2.0).to(hidden_dtype)
    weight = (torch.randn(num_experts, hidden_size, device=device) * weight_std).float()
    tid2eid = torch.randint(0, num_experts, (vocab_size, top_k), device=device, dtype=torch.int64)
    input_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=ids_dtype)
    return hidden, weight, tid2eid, input_ids


@requires_cuda
@pytest.mark.parametrize(
    "num_tokens,hidden_dtype,ids_dtype,norm_topk_prob",
    [
        (1, torch.bfloat16, torch.int32, True),  # T=1: fp32-gemv cuBLAS mirror (no TF32)
        (2, torch.float32, torch.int64, False),  # T=2: TF32 tensor-core mirror onset
        (8, torch.bfloat16, torch.int64, True),  # decode-boundary token count
    ],
)
def test_hash_routing_decode_parity(num_tokens, hidden_dtype, ids_dtype, norm_topk_prob):
    hidden, weight, tid2eid, input_ids = _make_hash_inputs(
        num_tokens=num_tokens,
        hidden_size=4096,
        num_experts=256,
        vocab_size=512,
        top_k=6,
        weight_std=0.02,
        hidden_dtype=hidden_dtype,
        ids_dtype=ids_dtype,
        seed=num_tokens * 131 + int(norm_topk_prob) + (0 if hidden_dtype is torch.float32 else 7),
    )

    ref_idx, ref_w = _reference_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, norm_topk_prob
    )
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, norm_topk_prob
    )

    assert out_idx.dtype == torch.int64
    assert out_w.dtype == torch.float32
    assert out_idx.shape == (num_tokens, 6)
    assert out_w.shape == (num_tokens, 6)
    assert torch.equal(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-7)


@requires_cuda
def test_hash_routing_softplus_branches():
    # weight_std=0.3 -> logit tails well past +-20, covering both softplus regimes.
    hidden, weight, tid2eid, input_ids = _make_hash_inputs(
        num_tokens=8,
        hidden_size=4096,
        num_experts=256,
        vocab_size=512,
        top_k=6,
        weight_std=0.3,
        hidden_dtype=torch.bfloat16,
        ids_dtype=torch.int32,
        seed=1234,
    )
    logits = F.linear(hidden.float(), weight)
    selected_logits = logits.gather(1, tid2eid[input_ids.to(torch.long)])
    assert (selected_logits > 20.0).any(), "fixture no longer covers softplus linear branch"
    assert (selected_logits < -17.0).any(), "fixture no longer covers log1p u==1 regime"

    ref_idx, ref_w = _reference_hash_routing(hidden, weight, tid2eid, input_ids, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-7)


@requires_cuda
def test_hash_routing_duplicate_experts():
    hidden, weight, _, input_ids = _make_hash_inputs(
        num_tokens=2,
        hidden_size=4096,
        num_experts=256,
        vocab_size=512,
        top_k=6,
        weight_std=0.02,
        hidden_dtype=torch.bfloat16,
        ids_dtype=torch.int32,
        seed=99,
    )
    # Every slot routes to expert 0 (hash-collision degenerate case).
    tid2eid = torch.zeros(512, 6, device="cuda", dtype=torch.int64)

    ref_idx, ref_w = _reference_hash_routing(hidden, weight, tid2eid, input_ids, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-7)


@requires_cuda
def test_hash_routing_prefill_fallback_bit_exact():
    # 16 tokens > _HASH_ROUTING_DECODE_MAX_TOKENS: the op runs the reference chain itself.
    hidden, weight, tid2eid, input_ids = _make_hash_inputs(
        num_tokens=16,
        hidden_size=4096,
        num_experts=256,
        vocab_size=512,
        top_k=6,
        weight_std=0.02,
        hidden_dtype=torch.bfloat16,
        ids_dtype=torch.int32,
        seed=7,
    )
    ref_idx, ref_w = _reference_hash_routing(hidden, weight, tid2eid, input_ids, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx)
    assert torch.equal(out_w, ref_w), "prefill fallback must be bit-identical to the reference"


@requires_cuda
def test_ambient_cublas_dispatch_contract():
    # Pins the cuBLAS behavior the fused kernel's tf32_trunc heuristic mirrors: fp32 gemv at
    # T=1, TF32 with RNE input conversion at T>=2. If this fails after a torch/cuBLAS upgrade,
    # re-derive the heuristic in deepseek_v4_hash_routing_fn.
    if not torch.backends.cuda.matmul.allow_tf32:
        pytest.skip("TF32 disabled in this environment; fp32 path is trivially matched")
    import struct

    # tie = halfway between 1.0 (even in tf32) and 1 + 2^-10.
    tie = struct.unpack(
        "<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", 1.0))[0] | 0x1000)
    )[0]
    W = torch.zeros(256, 4096, device="cuda")
    W[0, 0] = 1.0
    x1 = torch.zeros(1, 4096, device="cuda")
    x1[:, 0] = tie
    assert F.linear(x1, W)[0, 0].item() == tie, "T=1 is no longer a passthrough fp32 gemv"
    x2 = torch.zeros(2, 4096, device="cuda")
    x2[:, 0] = tie
    assert F.linear(x2, W)[0, 0].item() == 1.0, "T>=2 is no longer TF32 with RNE conversion"


# ---------------------------------------------------------------------------
# EP-localized variants: byte-exact vs global op + eager localization
# ---------------------------------------------------------------------------

# (expert_start, local_experts, num_global): full set, first/middle EP4 shards, and a
# last-rank remainder split (250 -> 62/62/62/64).
_SHARDS = [
    (0, 256, 256),
    (0, 64, 256),
    (128, 64, 256),
    (186, 64, 250),
]


def _assert_localized_pair_equal(out, ref, local_experts):
    out_idx, out_w = out
    ref_idx, ref_w = ref
    assert out_idx.dtype == torch.int32
    assert out_w.dtype == torch.bfloat16
    assert torch.equal(out_idx, ref_idx)
    assert torch.equal(out_w, ref_w)
    sentinel = out_idx == local_experts
    if sentinel.any():
        assert torch.equal(out_w[sentinel], torch.zeros_like(out_w[sentinel]))


@requires_cuda
@pytest.mark.parametrize("expert_start,local_experts,num_global", _SHARDS)
def test_routing_localized_bit_exact(expert_start, local_experts, num_global):
    torch.manual_seed(2 * 131 + expert_start * 7 + local_experts)
    device = "cuda"
    num_tokens, top_k, rsf = 2, 8, 1.5

    router_logits = torch.randn(num_tokens, num_global, device=device, dtype=torch.float32)
    bias = torch.randn(num_global, device=device, dtype=torch.float32) * 0.1

    sel_g, rw_g = torch.ops.auto_deploy.deepseek_v4_routing(router_logits, bias, top_k, rsf, True)
    ref = _localize_routing_eager(sel_g, rw_g, expert_start, local_experts)
    out = torch.ops.auto_deploy.deepseek_v4_routing_localized(
        router_logits, bias, top_k, rsf, True, expert_start, local_experts
    )
    _assert_localized_pair_equal(out, ref, local_experts)


@requires_cuda
@pytest.mark.parametrize("num_tokens", [1, 64])  # fused decode kernel / prefill eager mirror
@pytest.mark.parametrize("expert_start,local_experts,num_global", _SHARDS)
def test_hash_routing_localized_bit_exact(num_tokens, expert_start, local_experts, num_global):
    torch.manual_seed(num_tokens * 977 + expert_start * 13 + local_experts)
    device = "cuda"
    top_k, hidden, vocab, rsf = 8, 512, 1024, 2.5

    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    weight = torch.randn(num_global, hidden, device=device, dtype=torch.float32) * 0.05
    tid2eid = torch.randint(0, num_global, (vocab, top_k), device=device, dtype=torch.int64)
    # Force token 0 fully OFF-rank (strict-subset shards) to hit the all-sentinel/zero case.
    if local_experts < num_global:
        off_rank = (expert_start + local_experts) % num_global
        tid2eid[0] = off_rank
    input_ids = torch.randint(0, vocab, (num_tokens,), device=device, dtype=torch.int64)
    input_ids[0] = 0

    sel_g, rw_g = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden_states, weight, tid2eid, input_ids, rsf, True
    )
    ref = _localize_routing_eager(sel_g, rw_g, expert_start, local_experts)
    out = torch.ops.auto_deploy.deepseek_v4_hash_routing_localized(
        hidden_states, weight, tid2eid, input_ids, rsf, True, expert_start, local_experts
    )
    _assert_localized_pair_equal(out, ref, local_experts)


@requires_cuda
def test_localized_ops_reject_nonpositive_local_experts():
    device = "cuda"
    router_logits = torch.randn(1, 16, device=device, dtype=torch.float32)
    bias = torch.zeros(16, device=device, dtype=torch.float32)
    with pytest.raises(Exception, match="local_experts"):
        torch.ops.auto_deploy.deepseek_v4_routing_localized(router_logits, bias, 4, 1.0, True, 0, 0)

    hidden_states = torch.randn(1, 32, device=device, dtype=torch.bfloat16)
    weight = torch.randn(16, 32, device=device, dtype=torch.float32)
    tid2eid = torch.zeros(8, 4, device=device, dtype=torch.int64)
    input_ids = torch.zeros(1, device=device, dtype=torch.int64)
    with pytest.raises(Exception, match="local_experts"):
        torch.ops.auto_deploy.deepseek_v4_hash_routing_localized(
            hidden_states, weight, tid2eid, input_ids, 1.0, True, 0, 0
        )


def test_routing_fake_registrations_match_real_output_metadata():
    # Export relies on the fakes reproducing the real ops' output shapes/dtypes.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        logits = torch.empty(3, 64)
        bias = torch.empty(64)
        idx, w = torch.ops.auto_deploy.deepseek_v4_routing(logits, bias, 4, 1.5, True)
        assert (tuple(idx.shape), idx.dtype) == ((3, 4), torch.int64)
        assert (tuple(w.shape), w.dtype) == ((3, 4), torch.float32)

        idx, w = torch.ops.auto_deploy.deepseek_v4_routing_localized(
            logits, bias, 4, 1.5, True, 0, 16
        )
        assert (tuple(idx.shape), idx.dtype) == ((3, 4), torch.int32)
        assert (tuple(w.shape), w.dtype) == ((3, 4), torch.bfloat16)

        hidden = torch.empty(3, 128)
        weight = torch.empty(64, 128)
        tid2eid = torch.empty(32, 4, dtype=torch.int64)
        input_ids = torch.empty(3, dtype=torch.int64)
        idx, w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
            hidden, weight, tid2eid, input_ids, 1.5, True
        )
        assert (tuple(idx.shape), idx.dtype) == ((3, 4), torch.int64)
        assert (tuple(w.shape), w.dtype) == ((3, 4), torch.float32)

        idx, w = torch.ops.auto_deploy.deepseek_v4_hash_routing_localized(
            hidden, weight, tid2eid, input_ids, 1.5, True, 0, 16
        )
        assert (tuple(idx.shape), idx.dtype) == ((3, 4), torch.int32)
        assert (tuple(w.shape), w.dtype) == ((3, 4), torch.bfloat16)
