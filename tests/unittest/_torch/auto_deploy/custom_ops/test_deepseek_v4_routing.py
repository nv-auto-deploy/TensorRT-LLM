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

"""Parity test for the fused DeepSeek V4 sqrtsoftplus top-k routing custom op."""

import pytest
import torch
import torch.nn.functional as F

# Importing the module registers the auto_deploy::deepseek_v4_routing custom op.
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe import deepseek_v4_routing  # noqa: F401


def _reference_routing(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    norm_topk_prob: bool,
):
    """Byte-faithful copy of DeepseekV4MoEGate.forward (non-hash branch)."""
    scores = F.softplus(router_logits).sqrt()
    selected_experts = (scores + bias).topk(top_k, dim=-1).indices
    routing_weights = scores.gather(1, selected_experts)
    if norm_topk_prob:
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routing_weights = routing_weights * routed_scaling_factor
    return selected_experts, routing_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_tokens", [1, 2, 5, 17, 128])
@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("top_k", [6])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.5])
def test_deepseek_v4_routing_parity(
    num_tokens, num_experts, top_k, norm_topk_prob, routed_scaling_factor
):
    torch.manual_seed(num_tokens * 131 + num_experts + int(norm_topk_prob))
    device = "cuda"
    # Spread logits over a wide range to exercise both softplus branches (x>20 linear).
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

    # Indices: random fp32 logits make ties measure-zero -> exact match expected.
    assert torch.equal(out_idx, ref_idx.to(torch.int64)), (
        f"index mismatch\nref={ref_idx}\nout={out_idx}"
    )
    # Weights: sqrtsoftplus done in fp32 in-register; expect tight numeric parity.
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_softplus_threshold():
    """Large logits (>20) take the softplus linear branch; verify parity there."""
    torch.manual_seed(0)
    device = "cuda"
    num_tokens, num_experts, top_k = 4, 256, 6
    # Push many logits well past the softplus threshold of 20.
    router_logits = (torch.randn(num_tokens, num_experts, device=device) * 5.0 + 18.0).float()
    bias = torch.zeros(num_experts, device=device).float()

    ref_idx, ref_w = _reference_routing(router_logits, bias, top_k, 1.5, True)
    out_idx, out_w = torch.ops.auto_deploy.deepseek_v4_routing(
        router_logits, bias, top_k, 1.5, True
    )
    assert torch.equal(out_idx, ref_idx.to(torch.int64))
    torch.testing.assert_close(out_w, ref_w, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deepseek_v4_routing_fp32_mirror_input_parity():
    """Learned-router fp32-mirror contract for the HC-pre y32 output.

    ``deepseek_v4_hc_pre_mix_combine_partials_y32`` emits ``y32 == y.float()``
    bit-for-bit, so feeding the mirror to the unchanged fp32 router GEMV
    (``torch_linear_simple``) must reproduce the exact logits, selected expert
    ids, and routing weights of the old ``y.to(torch.float32)`` boundary-copy
    path in the same process.
    """
    import triton

    from tensorrt_llm._torch.auto_deploy.custom_ops import hc_composition
    from tensorrt_llm._torch.auto_deploy.custom_ops.linear import linear  # noqa: F401

    torch.manual_seed(17)
    B, S, hc_mult, H = 2, 1, 4, 4096
    num_experts, top_k = 256, 6
    eps, norm_eps, rms_eps, iters = 1e-6, 1e-6, 3e-5, 20
    mix_hc = (2 + hc_mult) * hc_mult
    dev = "cuda"

    x = torch.randn(B, S, hc_mult * H, device=dev, dtype=torch.bfloat16)
    hc_fn = (torch.randn(mix_hc, hc_mult * H, device=dev, dtype=torch.float32) * 0.02).contiguous()
    hc_scale = torch.rand(3, device=dev, dtype=torch.float32) + 0.5
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32) * 0.1
    norm_w = torch.randn(H, device=dev, dtype=torch.float32) * 0.1 + 1.0

    n, dim = B * S, hc_mult * H
    chunk, split = hc_composition.hc_partials_layout(dim)
    parts = torch.empty(n, mix_hc + 1, split, device=dev, dtype=torch.float32)
    hc_composition._hc_fn_partials_kernel[(n, split)](
        x.reshape(n, dim).contiguous(),
        hc_fn,
        parts,
        n,
        dim,
        split,
        MIX_HC=mix_hc,
        KBLOCK=triton.next_power_of_2(mix_hc),
        CHUNK=chunk,
        num_warps=4,
    )

    y, y32, _, _ = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
        parts, x, hc_fn, hc_scale, hc_base, norm_w, hc_mult, iters, eps, norm_eps, rms_eps, x.dtype
    )

    router_w = (torch.randn(num_experts, H, device=dev, dtype=torch.float32) * 0.05).contiguous()
    bias = (torch.randn(num_experts, device=dev) * 0.5).float()

    # Old gate path: bf16 -> fp32 boundary copy feeding the fp32 GEMV.
    logits_cast = torch.ops.auto_deploy.torch_linear_simple(
        y.reshape(-1, H).to(router_w.dtype), router_w, None
    ).float()
    # New gate path: the kernel-emitted fp32 mirror feeding the same GEMV.
    logits_mirror = torch.ops.auto_deploy.torch_linear_simple(
        y32.reshape(-1, H), router_w, None
    ).float()

    assert torch.equal(y32, y.float())
    assert torch.equal(logits_cast, logits_mirror)

    idx_c, w_c = torch.ops.auto_deploy.deepseek_v4_routing(logits_cast, bias, top_k, 1.5, True)
    idx_m, w_m = torch.ops.auto_deploy.deepseek_v4_routing(logits_mirror, bias, top_k, 1.5, True)
    assert torch.equal(idx_c, idx_m)
    assert torch.equal(w_c, w_m)
