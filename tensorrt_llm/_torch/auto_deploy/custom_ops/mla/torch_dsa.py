# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Torch reference implementation for DeepSeek Sparse Attention (DSA).

DSA extends MLA with an Indexer submodule that enforces token sparsity.
The Indexer produces top-k token indices whose sparse mask is added to
the MLA attention scores before softmax, zeroing out non-selected positions.

This source op accepts pre-computed Indexer tensors (index_q, index_k,
index_weights), all with RoPE already applied by the caller, keeping the
same design philosophy as torch_mla which accepts pre-rotated q_pe/kpe.

Reference: https://huggingface.co/zai-org/GLM-5 (GlmMoeDsaAttention)
           DeepSeek-V3.2 inference/model.py (MLA with Indexer)
"""

import math
from typing import Optional

import torch


def _compute_dsa_index_mask(
    index_q: torch.Tensor,  # [B, S_q, H, D_idx]
    index_k: torch.Tensor,  # [B, S_k, D_idx]
    index_weights: torch.Tensor,  # [B, S_q, H]
    index_topk: int,
    is_causal: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """Compute the DSA sparse mask from Indexer tensors.

    index_score[b,s,t] = softmax_scale * sum_h(index_weights[b,s,h] * dot(index_q[b,s,h], index_k[b,t]))

    Returns index_mask [B, S_q, S_k] with 0.0 at selected top-k positions and -inf elsewhere.
    """
    bs, s_q, num_idx_heads, _ = index_q.shape
    s_k = index_k.shape[1]

    # scores per head: [B, S_q, H, S_k]
    # compute in float32 for numerical stability
    per_head_scores = torch.einsum(
        "bshd,btd->bsht",
        index_q.float(),
        index_k.float(),
    )  # [B, S_q, H, S_k]

    # weighted sum over heads: [B, S_q, S_k]
    index_score = torch.einsum("bsht,bsh->bst", per_head_scores, index_weights.float())
    index_score = index_score * softmax_scale

    # Apply causal mask so future tokens cannot be selected
    if is_causal and s_q == s_k:
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, device=index_q.device, dtype=torch.bool),
            diagonal=1,
        )
        index_score.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

    # Select top-k valid positions per query token
    effective_topk = min(index_topk, s_k)
    topk_indices = index_score.topk(effective_topk, dim=-1).indices  # [B, S_q, topk]

    # Build mask: -inf everywhere, 0.0 at selected positions
    index_mask = index_score.new_full(index_score.shape, float("-inf"))
    index_mask.scatter_(-1, topk_indices, 0.0)  # [B, S_q, S_k]

    return index_mask


@torch.library.custom_op("auto_deploy::torch_dsa", mutates_args=())
def torch_dsa(
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim] (RoPE applied)
    compressed_kv: torch.Tensor,  # [B, S, kv_lora_rank]
    kpe: torch.Tensor,  # [B, S, 1, qk_rope_head_dim] (RoPE applied)
    kv_b_proj_weight: torch.Tensor,  # [N*(qk_nope_head_dim + v_head_dim), kv_lora_rank]
    index_q: torch.Tensor,  # [B, S, index_n_heads, index_head_dim] (RoPE applied)
    index_k: torch.Tensor,  # [B, S, index_head_dim] (shared across heads, RoPE applied)
    index_weights: torch.Tensor,  # [B, S, index_n_heads] (from weights_proj, pre-scaled)
    index_topk: int = 64,
    is_causal: bool = True,
    scale: Optional[float] = None,
    layout: str = "bsnd",
) -> torch.Tensor:
    """DeepSeek Sparse Attention (DSA) reference implementation.

    Extends MLA with an Indexer that selects top-k KV positions per query token.
    The sparse mask is added to MLA attention scores before softmax.

    Args:
        q_nope: Query non-positional component [B, S, N, qk_nope_head_dim]
        q_pe: Query positional component (RoPE applied) [B, S, N, qk_rope_head_dim]
        compressed_kv: Compressed KV latent [B, S, kv_lora_rank]
        kpe: Key positional encoding (RoPE applied) [B, S, 1, qk_rope_head_dim]
        kv_b_proj_weight: KV expansion weights [N*(qk_nope+v), kv_lora_rank]
        index_q: Indexer query (RoPE applied) [B, S, index_n_heads, index_head_dim]
        index_k: Indexer key (RoPE applied) [B, S, index_head_dim]
        index_weights: Per-head importance weights [B, S, index_n_heads]
        index_topk: Number of KV positions to attend to per query token
        is_causal: Whether to apply causal masking
        scale: Softmax scale (default: 1/sqrt(qk_nope_head_dim + qk_rope_head_dim))
        layout: Input/output layout, "bsnd" or "bnsd"

    Returns:
        Attention output [B, S, N, v_head_dim] (bsnd layout)
    """
    if layout not in ("bnsd", "bsnd"):
        raise ValueError(f"layout must be 'bnsd' or 'bsnd', got {layout!r}")

    # Infer dimensions
    if layout == "bsnd":
        bs, s_q, num_heads, qk_nope_head_dim = q_nope.shape
        qk_rope_head_dim = q_pe.shape[-1]
    else:
        bs, num_heads, s_q, qk_nope_head_dim = q_nope.shape
        qk_rope_head_dim = q_pe.shape[-1]

    s_k = compressed_kv.shape[1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # DSA index softmax scale: 1/sqrt(index_head_dim)
    index_head_dim = index_q.shape[-1]
    index_softmax_scale = 1.0 / math.sqrt(index_head_dim)

    # =========================================================================
    # Indexer: compute sparse mask
    # =========================================================================
    index_mask = _compute_dsa_index_mask(
        index_q, index_k, index_weights, index_topk, is_causal, index_softmax_scale
    )  # [B, S_q, S_k]

    # =========================================================================
    # MLA: expand compressed_kv and compute attention with sparse mask
    # =========================================================================
    # compressed_kv: [B, S, kv_lora_rank] -> [B, S, N, kv_head_dim]
    kv = torch.matmul(compressed_kv, kv_b_proj_weight.t())
    kv = kv.view(bs, s_k, num_heads, kv_head_dim)
    k_nope, value_states = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)

    # Convert to [B, N, S, D] for attention computation
    k_nope = k_nope.transpose(1, 2).contiguous()
    value_states = value_states.transpose(1, 2).contiguous()

    if layout == "bsnd":
        q_nope = q_nope.transpose(1, 2).contiguous()
        q_pe = q_pe.transpose(1, 2).contiguous()
        kpe = kpe.transpose(1, 2).contiguous()

    # kpe: [B, 1, S, rope_head_dim] -> expand to all heads
    kpe_expanded = kpe.expand(bs, num_heads, s_k, qk_rope_head_dim)

    # Full query and key: [B, N, S, qk_head_dim]
    query_states = torch.cat([q_nope, q_pe], dim=-1)
    key_states = torch.cat([k_nope, kpe_expanded], dim=-1)

    # Attention scores: [B, N, S_q, S_k]
    attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

    # Apply causal mask (full upper-triangular mask)
    if is_causal and s_q == s_k:
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, device=q_nope.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Apply DSA sparse mask: [B, S_q, S_k] -> broadcast to [B, N, S_q, S_k]
    attn_scores = attn_scores + index_mask.unsqueeze(1)

    # Softmax + output
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)
    attn_out = torch.matmul(attn_weights, value_states)  # [B, N, S_q, v_head_dim]

    if layout == "bsnd":
        return attn_out.transpose(1, 2).contiguous()  # [B, S, N, v_head_dim]
    else:
        return attn_out.contiguous()  # [B, N, S, v_head_dim]


@torch_dsa.register_fake
def torch_dsa_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    compressed_kv: torch.Tensor,
    kpe: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    index_weights: torch.Tensor,
    index_topk: int = 64,
    is_causal: bool = True,
    scale: Optional[float] = None,
    layout: str = "bsnd",
) -> torch.Tensor:
    """Fake implementation for torch_dsa."""
    qk_nope_head_dim = q_nope.shape[-1]
    num_heads = q_nope.shape[2] if layout == "bsnd" else q_nope.shape[1]
    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim

    if layout == "bsnd":
        return q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
        ).contiguous()
    else:
        return q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
        ).contiguous()
