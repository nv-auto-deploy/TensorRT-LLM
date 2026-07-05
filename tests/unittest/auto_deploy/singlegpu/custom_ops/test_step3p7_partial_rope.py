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

"""Tests for the fused Step-3.7 partial-rotary RoPE custom op (step3p7_partial_rope)."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_step3p7 import (
    Step3p7RotaryEmbedding,
    _build_step3p7_fused_cos_sin_cache,
    _compute_step3p7_inv_freq,
)

MAX_POS = 4096


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _reference_partial_rope(q, k, inv_freq, position_ids):
    """The pre-fusion reference chain: slice -> rope(rotate_half) -> cat, fp32 cos/sin."""
    rotary_dim = inv_freq.shape[0] * 2
    positions = torch.arange(MAX_POS, dtype=inv_freq.dtype, device=inv_freq.device)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[position_ids].unsqueeze(2)  # [B, S, 1, rotary_dim] fp32
    sin = emb.sin()[position_ids].unsqueeze(2)

    def _one(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_rot = (x_rot.float() * cos + _rotate_half(x_rot.float()) * sin).to(x.dtype)
        return torch.cat([x_rot, x_pass], dim=-1)

    return _one(q), _one(k)


@pytest.mark.parametrize(
    "partial_rotary_factor,base,rope_scaling",
    [
        # full-attention layers: half rotation, llama3 scaling
        (
            0.5,
            5e6,
            {
                "rope_type": "llama3",
                "factor": 2.0,
                "original_max_position_embeddings": 131072,
                "low_freq_factor": 1.0,
                "high_freq_factor": 32.0,
            },
        ),
        # sliding-attention layers: full rotation, no scaling
        (1.0, 1e4, None),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_step3p7_partial_rope_matches_reference(partial_rotary_factor, base, rope_scaling):
    torch.manual_seed(0)
    device = "cuda"
    B, S, NQ, NK, D = 2, 5, 8, 1, 128

    inv_freq = _compute_step3p7_inv_freq(D, partial_rotary_factor, base, rope_scaling).to(device)
    cache = _build_step3p7_fused_cos_sin_cache(inv_freq, MAX_POS)
    rotary_dim = int(D * partial_rotary_factor)
    assert cache.shape == (MAX_POS, rotary_dim)
    assert cache.dtype == torch.float32

    q = torch.randn(B, S, NQ, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, S, NK, D, dtype=torch.bfloat16, device=device)
    position_ids = torch.randint(0, MAX_POS, (B, S), device=device)

    q_out, k_out = torch.ops.auto_deploy.step3p7_partial_rope(q, k, position_ids, cache)
    q_ref, k_ref = _reference_partial_rope(q, k, inv_freq, position_ids)

    assert q_out.shape == q.shape and k_out.shape == k.shape
    assert q_out.dtype == q.dtype and k_out.dtype == k.dtype
    # pass-through half must be bit-exact
    assert torch.equal(q_out[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_out[..., rotary_dim:], k[..., rotary_dim:])
    # rotated half matches the fp32 reference within bf16 rounding
    torch.testing.assert_close(
        q_out[..., :rotary_dim].float(), q_ref[..., :rotary_dim].float(), atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        k_out[..., :rotary_dim].float(), k_ref[..., :rotary_dim].float(), atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_step3p7_rotary_embedding_cache_survives_blanket_dtype_cast():
    """model.to(bf16) (HFQuantConfigReader.post_process_model) must not downcast the cache."""
    emb = Step3p7RotaryEmbedding(
        head_dim=128, partial_rotary_factor=0.5, base=5e6, max_position_embeddings=MAX_POS
    )
    emb = emb.to(torch.bfloat16)
    assert emb.inv_freq.dtype == torch.float32
    assert emb.cos_sin_cache.dtype == torch.float32
    # The rebuild must be driven by the module's own (re-pinned) inv_freq — the same values
    # the pre-fusion optimize_rope cache materialization read at transform time.
    expected = _build_step3p7_fused_cos_sin_cache(emb.inv_freq, MAX_POS)
    torch.testing.assert_close(emb.cos_sin_cache, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_step3p7_partial_rope_fake_registration():
    """Meta/fake propagation (export path) must preserve shapes and dtypes."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(1, 7, 8, 128, dtype=torch.bfloat16, device="cuda")
        k = torch.empty(1, 7, 1, 128, dtype=torch.bfloat16, device="cuda")
        pos = torch.empty(1, 7, dtype=torch.int64, device="cuda")
        cache = torch.empty(MAX_POS, 64, dtype=torch.float32, device="cuda")
        q_out, k_out = torch.ops.auto_deploy.step3p7_partial_rope(q, k, pos, cache)
        assert tuple(q_out.shape) == (1, 7, 8, 128) and q_out.dtype == torch.bfloat16
        assert tuple(k_out.shape) == (1, 7, 1, 128) and k_out.dtype == torch.bfloat16
