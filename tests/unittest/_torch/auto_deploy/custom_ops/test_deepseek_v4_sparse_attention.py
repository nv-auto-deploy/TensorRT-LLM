# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the DeepSeek V4 fused sparse-attention Triton kernels.

``_deepseek_v4_sparse_attention`` dispatches to one of three un-autotuned
``@triton.jit`` kernels in ``deepseek_v4_sparse_attention.py``:

* ``_fused_sparse_attention_splitk_kernel`` + ``_fused_sparse_attention_reduce_kernel``
  -- the **decode** path (few query tokens, key reduction split across CTAs to fill
  idle SMs). This is idea_0020's autotune target: the B=1 TP8 per-rank shape (H=8,
  L=640, D=512).
* ``_fused_sparse_attention_kernel`` -- the **prefill** path (many tokens, the simple
  flash-MQA kernel saturates the GPU with token*head parallelism).

This test guards all three against an fp64 from-scratch ground truth so that
``@triton.autotune`` / launch-config tuning (BLOCK sizes, num_warps, num_stages)
can be exercised without silently corrupting results. Both the split-K decode
path and the simple prefill path are covered, plus the sink-only (all keys
masked) edge case which exercises the NaN-safe online-softmax floor.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _can_use_fused_sparse_attention,
    _deepseek_v4_sparse_attention,
)


def _attn_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    # bf16 tl.dot contraction needs Ampere+ tensor cores.
    return torch.cuda.get_device_capability()[0] >= 8


def _ref_sparse_attention(
    q: torch.Tensor,  # [B, S, H, D]
    kv: torch.Tensor,  # [B, R, D]
    sink: torch.Tensor,  # [H]
    topk: torch.Tensor,  # [B, S, K]
    scale: float,
) -> torch.Tensor:
    """fp64 ground truth: per-token/head softmax over selected rows + raw sink logit.

    Masked slots (idx < 0 or idx >= R) get -inf score -> zero weight. When every
    key of a query is masked the weight collapses onto the (value-less) sink, so
    the output is zero -- matching the kernel's sink fold.
    """
    B, S, H, D = q.shape
    R = kv.shape[1]
    K = topk.shape[-1]
    qf = q.double()
    kvf = kv.double()
    out = torch.zeros(B, S, H, D, dtype=torch.double, device=q.device)
    for b in range(B):
        for s in range(S):
            idx = topk[b, s].long()
            valid = (idx >= 0) & (idx < R)
            idxc = idx.clamp(0, R - 1)
            sel = kvf[b, idxc]  # [K, D]
            scores = (qf[b, s] @ sel.t()) * scale  # [H, K]
            scores = torch.where(valid.view(1, K), scores, torch.full_like(scores, float("-inf")))
            sink_col = sink.double().view(H, 1)  # raw sink logit, NOT scaled
            logits = torch.cat([scores, sink_col], dim=-1)  # [H, K+1]
            w = torch.softmax(logits, dim=-1)
            out[b, s] = w[:, :K] @ sel
    return out


def _check(out: torch.Tensor, ref: torch.Tensor, tag: str, cos_bar=0.999, amax_bar=4e-2):
    out_f = out.double()
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out_f - ref).abs().amax() / scale).item()
    cos = torch.nn.functional.cosine_similarity(out_f.reshape(-1), ref.reshape(-1), dim=0).item()
    assert cos > cos_bar, f"{tag}: cosine={cos:.6f} (<{cos_bar})"
    assert max_rel < amax_bar, f"{tag}: max_abs_err/amax={max_rel:.4e} (>{amax_bar})"


@pytest.mark.skipif(not _attn_supported(), reason="Requires Ampere+ tensor cores")
@pytest.mark.parametrize("num_heads", [8, 16, 64])
def test_decode_splitk(num_heads):
    """Decode (num_tokens=1): the split-K + reduce path. H=8 is the TP8 per-rank
    autotune target; H=16/64 guard the head-grouping heuristic."""
    torch.manual_seed(0)
    D, L = 512, 640
    q = torch.randn(1, 1, num_heads, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, L, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(num_heads, device="cuda", dtype=torch.bfloat16)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L)
    scale = D**-0.5

    assert _can_use_fused_sparse_attention(q.reshape(1, num_heads, D), kv, topk.reshape(1, L))
    out = _deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    ref = _ref_sparse_attention(q, kv, sink, topk, scale)
    _check(out, ref, f"decode-H{num_heads}")


@pytest.mark.skipif(not _attn_supported(), reason="Requires Ampere+ tensor cores")
@pytest.mark.parametrize("D", [64, 128, 512])
def test_decode_head_dim(D):
    """Decode across head_dim (D_BLOCK = next_pow2(D))."""
    torch.manual_seed(1)
    H, L = 8, 384
    q = torch.randn(1, 1, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, L, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L)
    scale = D**-0.5
    out = _deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    ref = _ref_sparse_attention(q, kv, sink, topk, scale)
    _check(out, ref, f"decode-D{D}")


@pytest.mark.skipif(not _attn_supported(), reason="Requires Ampere+ tensor cores")
def test_prefill_simple_kernel():
    """Prefill (num_tokens >> 8): the simple flash-MQA kernel, with ~10% masked
    (-1) topk slots and indices spanning the full kv range."""
    torch.manual_seed(2)
    T, H, D, R, K = 256, 64, 512, 2048, 640
    q = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, R, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.randint(0, R, (1, T, K), device="cuda", dtype=torch.int64)
    mask = torch.rand(1, T, K, device="cuda") < 0.1
    topk = torch.where(mask, torch.full_like(topk, -1), topk)
    scale = D**-0.5
    out = _deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    ref = _ref_sparse_attention(q, kv, sink, topk, scale)
    _check(out, ref, "prefill", amax_bar=6e-2)


@pytest.mark.skipif(not _attn_supported(), reason="Requires Ampere+ tensor cores")
def test_sink_only_all_masked():
    """Every key masked -> all softmax mass on the value-less sink -> zero output.
    Exercises the NaN-safe running-max floor in the online softmax."""
    torch.manual_seed(3)
    H, D, L = 8, 512, 640
    q = torch.randn(1, 1, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, L, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.full((1, 1, L), -1, device="cuda", dtype=torch.int64)
    scale = D**-0.5
    out = _deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    assert torch.isfinite(out).all(), "sink-only output must be finite"
    assert out.abs().amax().item() < 1e-3, "sink-only output must be ~zero"


@pytest.mark.skipif(not _attn_supported(), reason="Requires Ampere+ tensor cores")
def test_partial_mask_consistency():
    """Decode with a mix of valid + masked (-1) slots in the same query."""
    torch.manual_seed(4)
    H, D, L = 8, 512, 512
    q = torch.randn(1, 1, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, L, D, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    topk = torch.arange(L, device="cuda", dtype=torch.int64).view(1, 1, L)
    topk[0, 0, ::3] = -1  # mask every third slot
    scale = D**-0.5
    out = _deepseek_v4_sparse_attention(q, kv, sink, topk, scale)
    ref = _ref_sparse_attention(q, kv, sink, topk, scale)
    _check(out, ref, "partial-mask")
