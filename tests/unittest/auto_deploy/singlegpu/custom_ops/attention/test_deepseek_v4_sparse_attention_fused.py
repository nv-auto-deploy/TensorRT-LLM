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

"""CUDA correctness tests for the fused DeepSeek V4 sparse-attention attend kernel.

These exercise ``_deepseek_v4_sparse_attention`` on CUDA bf16 tensors (the path the
real model takes), where idea_0001's fused Triton kernel replaces the torch
gather+matmul+softmax+matmul body.  The reference is an independent fp32
implementation of the documented semantics (sink term, negative/out-of-range
masking, duplicate-index independence).
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention import (  # noqa: E402
    deepseek_v4_sparse_attention as dsv4,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel requires CUDA")


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """fp32 reference mirroring the documented sparse-attention semantics."""
    batch_size, seq_len, num_heads, _ = q.shape
    kv_rows = kv.shape[1]
    batch_idx = torch.arange(batch_size, device=q.device).view(batch_size, 1, 1)
    batch_idx = batch_idx.expand(batch_size, seq_len, topk_idxs.shape[-1])

    valid = (topk_idxs >= 0) & (topk_idxs < kv_rows)
    gather_idxs = topk_idxs.to(torch.long).clamp(min=0, max=kv_rows - 1)
    selected_kv = kv[batch_idx, gather_idxs].float()  # [B, S, K, D]
    logits = torch.matmul(q.float(), selected_kv.transpose(-1, -2))  # [B, S, H, K]
    logits = logits * softmax_scale
    logits = logits.masked_fill((~valid).unsqueeze(2), float("-inf"))

    sink = attn_sink.float().view(1, 1, num_heads, 1).expand(batch_size, seq_len, num_heads, 1)
    weights = torch.softmax(torch.cat([logits, sink], dim=-1), dim=-1, dtype=torch.float32)
    out = torch.matmul(weights[..., :-1], selected_kv)
    return out.to(q.dtype)


def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    denom = b.pow(2).mean().sqrt().clamp(min=1e-6)
    return float((a - b).pow(2).mean().sqrt() / denom)


def _check(q, kv, sink, topk, scale, tol=2e-2):
    out = dsv4._deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    ref = _reference(q, kv, sink, topk, scale)
    assert torch.isfinite(out).all(), "output has non-finite values"
    r = _rmse_ratio(out, ref)
    assert r < tol, f"rmse_ratio={r:.4e} exceeds tol={tol:.4e}"
    return out, ref


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_shape_matches_reference(dtype):
    """B=1, S=1, H=64, D=512, L=640 (identity arange selection, the decode path)."""
    torch.manual_seed(0)
    B, S, H, D, L = 1, 1, 64, 512, 640
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    kv = torch.randn(B, L, D, device="cuda", dtype=dtype)
    sink = torch.randn(H, device="cuda", dtype=dtype)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L).expand(B, S, L)
    _check(q, kv, sink, topk, D**-0.5)


def test_prefill_shape_matches_reference():
    """Multi-token prefill with random selection + ~10% masked (-1)."""
    torch.manual_seed(1)
    B, S, H, D, kv_rows, K = 1, 128, 64, 512, 512, 256
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, kv_rows, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.randint(0, kv_rows, (B, S, K), device="cuda", dtype=torch.int64)
    topk = torch.where(torch.rand(B, S, K, device="cuda") < 0.1, torch.full_like(topk, -1), topk)
    _check(q, kv, sink, topk, D**-0.5)


def test_batched_multi_token():
    """B=2, several heads, exercise batch_idxs path."""
    torch.manual_seed(2)
    B, S, H, D, kv_rows, K = 2, 16, 8, 512, 128, 64
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, kv_rows, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.randint(-1, kv_rows, (B, S, K), device="cuda", dtype=torch.int64)
    _check(q, kv, sink, topk, D**-0.5)


def test_all_negative_topk_yields_zero():
    """All-invalid selection -> sink dominates, output is finite zero."""
    torch.manual_seed(3)
    B, S, H, D, L = 1, 1, 64, 512, 640
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((B, L, D), 1000.0, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((B, S, L), -1, device="cuda", dtype=torch.int64)
    out = dsv4._deepseek_v4_sparse_attention(q, kv, sink, topk, D**-0.5)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_duplicate_and_out_of_range_indices():
    """Duplicates get independent mass; out-of-range indices are masked."""
    torch.manual_seed(4)
    B, S, H, D, kv_rows, K = 1, 4, 16, 512, 32, 48
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, kv_rows, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    # mix of duplicates, valid, -1, and out-of-range (>= kv_rows)
    topk = torch.randint(0, kv_rows, (B, S, K), device="cuda", dtype=torch.int64)
    topk[..., 0] = topk[..., 1]  # duplicate
    topk[..., 2] = -1  # masked
    topk[..., 3] = 9999  # out of range -> masked
    _check(q, kv, sink, topk, D**-0.5)


def test_non_power_of_two_head_dim():
    """D not a power of two (576 = 512+64) — kernel must mask the D tail."""
    torch.manual_seed(5)
    B, S, H, D, L = 1, 1, 32, 576, 320
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, L, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L)
    _check(q, kv, sink, topk, D**-0.5)
