# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Gemma decode Kernel A custom op.

This op packages the attention half of a Gemma4MoE decoder layer behind a
single custom-op boundary. The initial implementation is correctness-first and
stitches together the existing Triton/FlashInfer building blocks:

- fused QKV projection via ``aten.linear``
- per-head RMSNorm via the Triton RMSNorm helper
- RoPE via ``auto_deploy::flashinfer_rope``
- paged decode attention via ``auto_deploy::triton_paged_mha_with_cache``
- O projection via ``aten.linear``
- post-attention/pre-FFN RMSNorms via the Triton RMSNorm helper

The intent is to stabilize the op contract first so the internals can be
replaced with a true micro-pipelined Triton kernel in follow-up work without
changing callers again.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..normalization.triton_rms_norm import rms_norm


def _reshape_hidden_for_attention(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Normalize hidden-state layout to ``[batch, seq, hidden]``."""
    original_shape = tuple(hidden_states.shape)
    if hidden_states.ndim == 3:
        return hidden_states, original_shape
    if hidden_states.ndim != 2:
        raise ValueError(f"Expected hidden_states to be rank-2 or rank-3, got {hidden_states.ndim}")

    if position_ids.ndim == 1:
        batch_size = hidden_states.shape[0]
        seq_len = 1
    elif position_ids.ndim == 2:
        batch_size, seq_len = position_ids.shape
        if batch_size * seq_len != hidden_states.shape[0]:
            raise ValueError(
                "Flattened hidden_states shape is incompatible with position_ids: "
                f"{tuple(hidden_states.shape)} vs {tuple(position_ids.shape)}"
            )
    else:
        raise ValueError(f"Expected position_ids to be rank-1 or rank-2, got {position_ids.ndim}")

    return hidden_states.reshape(batch_size, seq_len, hidden_states.shape[-1]), original_shape


def _restore_hidden_layout(
    hidden_states: torch.Tensor, original_shape: Tuple[int, ...]
) -> torch.Tensor:
    if len(original_shape) == 2:
        return hidden_states.reshape(original_shape)
    return hidden_states


def _split_qkv(
    qkv: torch.Tensor,
    o_proj_weight: torch.Tensor,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_width = int(o_proj_weight.shape[1])
    total_width = int(qkv.shape[-1])
    kv_width_total = total_width - q_width
    if kv_width_total <= 0 or kv_width_total % 2 != 0:
        raise ValueError(
            "Expected fused QKV width to be q_width + 2 * kv_width, got "
            f"{total_width} with q_width={q_width}"
        )
    kv_width = kv_width_total // 2
    if q_width % head_dim != 0 or kv_width % head_dim != 0:
        raise ValueError(
            "Projected widths must be multiples of head_dim: "
            f"q_width={q_width}, kv_width={kv_width}, head_dim={head_dim}"
        )

    q, k, v = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
    q = q.reshape(*q.shape[:-1], q_width // head_dim, head_dim)
    k = k.reshape(*k.shape[:-1], kv_width // head_dim, head_dim)
    v = v.reshape(*v.shape[:-1], kv_width // head_dim, head_dim)
    return q, k, v


@torch.library.custom_op("auto_deploy::triton_gemma_kernel_a_decode", mutates_args=("kv_cache",))
def triton_gemma_kernel_a_decode(
    residual_in: torch.Tensor,
    attn_normed_in: torch.Tensor,
    qkv_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v_norm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    pre_feedforward_layernorm_weight: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    triton_batch_indices: torch.Tensor,
    triton_positions: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute Gemma's attention half and emit the FFN handoff tensors."""
    residual_bsh, original_shape = _reshape_hidden_for_attention(residual_in, position_ids)
    attn_normed_bsh, _ = _reshape_hidden_for_attention(attn_normed_in, position_ids)

    if residual_bsh.shape != attn_normed_bsh.shape:
        raise ValueError(
            "Expected residual_in and attn_normed_in to have matching shapes, got "
            f"{tuple(residual_bsh.shape)} and {tuple(attn_normed_bsh.shape)}"
        )

    head_dim = int(q_norm_weight.numel())
    qkv = torch.ops.aten.linear(attn_normed_bsh, qkv_weight, None)
    q, k, v = _split_qkv(qkv, o_proj_weight, head_dim)

    q_normed = rms_norm(q, q_norm_weight, eps=eps)
    k_normed = rms_norm(k, k_norm_weight, eps=eps)
    v_normed = rms_norm(v, v_norm_weight, eps=eps)

    q_rope, k_rope = torch.ops.auto_deploy.flashinfer_rope.default(
        q_normed, k_normed, position_ids, cos_sin_cache, True
    )

    attn_out = torch.ops.auto_deploy.triton_paged_mha_with_cache.default(
        q_rope,
        k_rope,
        v_normed,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache,
        scale,
        sliding_window,
    )

    attn_out_flat = attn_out.reshape(*attn_out.shape[:-2], -1)
    o_proj = torch.ops.aten.linear(attn_out_flat, o_proj_weight, None)
    attn_branch = rms_norm(o_proj, post_attention_layernorm_weight, eps=eps)
    post_attn_residual = residual_bsh + attn_branch
    pre_ffn_normed = rms_norm(post_attn_residual, pre_feedforward_layernorm_weight, eps=eps)

    return (
        _restore_hidden_layout(post_attn_residual, original_shape),
        _restore_hidden_layout(pre_ffn_normed, original_shape),
    )


@triton_gemma_kernel_a_decode.register_fake
def triton_gemma_kernel_a_decode_fake(
    residual_in: torch.Tensor,
    attn_normed_in: torch.Tensor,
    qkv_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v_norm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    pre_feedforward_layernorm_weight: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    triton_batch_indices: torch.Tensor,
    triton_positions: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del (
        attn_normed_in,
        qkv_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight,
        position_ids,
        cos_sin_cache,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache,
        scale,
        sliding_window,
        eps,
    )
    return torch.empty_like(residual_in), torch.empty_like(residual_in)
