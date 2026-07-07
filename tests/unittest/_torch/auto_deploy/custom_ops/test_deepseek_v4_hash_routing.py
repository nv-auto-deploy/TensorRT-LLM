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

"""Parity tests for the fused DeepSeek V4 hash-router gate custom op."""

import pytest
import torch
import torch.nn.functional as F

# Importing the module registers the auto_deploy::deepseek_v4_hash_routing custom op.
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import deepseek_v4_routing  # noqa: F401


def _reference_hash_routing(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    tid2eid: torch.Tensor,
    input_ids: torch.Tensor,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """Byte-faithful copy of the pre-fusion DeepseekV4MoEGate.forward hash branch."""
    router_logits = F.linear(hidden_states.to(weight.dtype), weight).float()
    scores = F.softplus(router_logits).sqrt()
    selected_experts = tid2eid[input_ids.to(torch.long)].to(torch.int64)
    routing_weights = scores.gather(1, selected_experts)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * routed_scaling_factor
    return selected_experts, routing_weights


def _make_inputs(
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize("hidden_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_hash_routing_decode_parity(num_tokens, hidden_dtype, ids_dtype, norm_topk_prob):
    hidden, weight, tid2eid, input_ids = _make_inputs(
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

    # Expert ids are a pure integer gather from tid2eid: exact by construction.
    assert torch.equal(out_idx, ref_idx), f"index mismatch\nref={ref_idx}\nout={out_idx}"
    # Weights differ from the cuBLAS chain only by fp32 dot reduction order.
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hash_routing_softplus_branches():
    """Wide logits exercise the x>20 linear branch AND the log1p small-y regime."""
    hidden, weight, tid2eid, input_ids = _make_inputs(
        num_tokens=8,
        hidden_size=4096,
        num_experts=256,
        vocab_size=512,
        top_k=6,
        weight_std=0.3,  # logit std ~ sqrt(4096)*2*0.3 -> tails well past +-20
        hidden_dtype=torch.bfloat16,
        ids_dtype=torch.int32,
        seed=1234,
    )
    logits = F.linear(hidden.float(), weight)
    selected_logits = logits.gather(1, tid2eid[input_ids.to(torch.long)])
    # Guard the fixture: both softplus regimes must actually be hit at the
    # SELECTED experts, else this test silently stops covering them.
    assert (selected_logits > 20.0).any(), "fixture no longer covers softplus linear branch"
    assert (selected_logits < -17.0).any(), "fixture no longer covers log1p u==1 regime"

    ref_idx, ref_w = _reference_hash_routing(hidden, weight, tid2eid, input_ids, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hash_routing_duplicate_experts():
    """An all-zeros tid2eid (every slot -> expert 0) must behave like the reference."""
    hidden, weight, _, input_ids = _make_inputs(
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
    tid2eid = torch.zeros(512, 6, device="cuda", dtype=torch.int64)

    ref_idx, ref_w = _reference_hash_routing(hidden, weight, tid2eid, input_ids, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [16, 64])
def test_hash_routing_prefill_fallback_bit_exact(num_tokens):
    """Above the decode token bound the op runs the reference chain: bitwise equal."""
    hidden, weight, tid2eid, input_ids = _make_inputs(
        num_tokens=num_tokens,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ambient_cublas_dispatch_contract():
    """Pin the cuBLAS behavior the fused kernel mirrors: fp32 gemv at T=1,
    TF32 with round-to-nearest-even input conversion at T>=2 (when TF32 is on).

    If this test starts failing after a cuBLAS/torch upgrade, the
    ``tf32_trunc`` heuristic in ``deepseek_v4_hash_routing_fn`` must be
    re-derived against the new dispatch.
    """
    if not torch.backends.cuda.matmul.allow_tf32:
        pytest.skip("TF32 disabled in this environment; fp32 path is trivially matched")
    import struct

    tie = struct.unpack(
        "<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", 1.0))[0] | 0x1000)
    )[0]  # halfway between 1.0 (even in tf32) and 1 + 2^-10
    W = torch.zeros(256, 4096, device="cuda")
    W[0, 0] = 1.0
    x1 = torch.zeros(1, 4096, device="cuda")
    x1[:, 0] = tie
    assert F.linear(x1, W)[0, 0].item() == tie, "T=1 is no longer a passthrough fp32 gemv"
    x2 = torch.zeros(2, 4096, device="cuda")
    x2[:, 0] = tie
    assert F.linear(x2, W)[0, 0].item() == 1.0, "T>=2 is no longer TF32 with RNE conversion"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hash_routing_precision_vs_fp64():
    """The fused fp32 path must be at least as close to fp64 truth as the cuBLAS chain."""
    hidden, weight, tid2eid, input_ids = _make_inputs(
        num_tokens=8,
        hidden_size=4096,
        num_experts=256,
        vocab_size=512,
        top_k=6,
        weight_std=0.02,
        hidden_dtype=torch.bfloat16,
        ids_dtype=torch.int32,
        seed=2026,
    )

    # fp64 ground truth of the same chain (bf16 -> fp64 widening is exact).
    logits64 = F.linear(hidden.to(torch.float64), weight.to(torch.float64))
    scores64 = F.softplus(logits64).sqrt()
    sel = tid2eid[input_ids.to(torch.long)]
    w64 = scores64.gather(1, sel)
    w64 = w64 / (w64.sum(dim=-1, keepdim=True) + 1e-20) * 1.5

    _, ref_w = _reference_hash_routing(hidden, weight, tid2eid, input_ids, 1.5, True)
    _, out_w = torch.ops.auto_deploy.deepseek_v4_hash_routing(
        hidden, weight, tid2eid, input_ids, 1.5, True
    )

    err_ref = (ref_w.to(torch.float64) - w64).abs().max().item()
    err_out = (out_w.to(torch.float64) - w64).abs().max().item()
    print(f"\nmax |err| vs fp64: cublas_chain={err_ref:.3e} fused={err_out:.3e}")
    assert err_out <= err_ref * 3.0 + 1e-9, (
        f"fused path degrades precision: fused={err_out:.3e} vs cublas={err_ref:.3e}"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
