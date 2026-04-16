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
single custom-op boundary. The hot path is implemented as one Triton kernel
that performs:

- fused Q/K/V projections directly from the normalized hidden state
- per-head RMSNorm and RoPE
- paged decode attention with KV-cache update
- O-proj accumulation
- post-attention residual add and pre-FFN RMSNorm
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..normalization.triton_rms_norm import rms_norm


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


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


@triton.jit
def _load_half_vectors(
    scratch_ptr,
    base_offset,
    global_row_idx,
    row_stride,
    HALF_D: tl.constexpr,
):
    half_offsets = tl.arange(0, HALF_D)
    first = tl.load(
        scratch_ptr + base_offset + global_row_idx * row_stride + half_offsets,
    )
    second = tl.load(
        scratch_ptr + base_offset + global_row_idx * row_stride + HALF_D + half_offsets,
    )
    return first, second


@triton.jit
def _gemma_kernel_a_decode_single_kernel(
    residual_ptr,
    attn_ptr,
    qkv_weight_ptr,
    o_proj_weight_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    v_norm_weight_ptr,
    post_attention_layernorm_weight_ptr,
    pre_feedforward_layernorm_weight_ptr,
    triton_batch_indices_ptr,
    triton_positions_ptr,
    cu_num_pages_ptr,
    cache_loc_ptr,
    last_page_len_ptr,
    cos_sin_cache_ptr,
    kv_cache_ptr,
    qkv_scratch_ptr,
    attn_scratch_ptr,
    o_proj_scratch_ptr,
    post_out_ptr,
    pre_out_ptr,
    residual_stride_t,
    residual_stride_h,
    attn_stride_t,
    attn_stride_h,
    qkv_weight_stride_o,
    qkv_weight_stride_i,
    o_proj_weight_stride_o,
    o_proj_weight_stride_i,
    cache_stride_block,
    cache_stride_kv,
    cache_stride_head,
    cache_stride_token,
    cos_sin_cache_stride_s,
    cos_sin_cache_stride_d,
    qkv_scratch_stride_t,
    qkv_scratch_stride_h,
    attn_scratch_stride_t,
    attn_scratch_stride_h,
    scratch_stride_t,
    scratch_stride_h,
    post_out_stride_t,
    post_out_stride_h,
    pre_out_stride_t,
    pre_out_stride_h,
    scale,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_NUM_PAGES: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    BLOCK_QKV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_TC_QKV: tl.constexpr,
    USE_TC_OPROJ: tl.constexpr,
    USE_DOT_ATTN: tl.constexpr,
    USE_ONE_SHOT_OPROJ: tl.constexpr,
):
    token_id = tl.program_id(0)
    d2: tl.constexpr = D_HEAD // 2
    d2_offsets = tl.arange(0, d2)
    d2_mask = d2_offsets < d2
    h_offsets = tl.arange(0, BLOCK_H)
    q_heads_per_kv: tl.constexpr = N_Q_HEADS // N_KV_HEADS
    q_local_offsets = tl.arange(0, q_heads_per_kv)
    q_base_offset: tl.constexpr = 0
    k_base_offset: tl.constexpr = N_Q_HEADS * D_HEAD
    v_base_offset: tl.constexpr = (N_Q_HEADS + N_KV_HEADS) * D_HEAD

    residual_row_offset = token_id * residual_stride_t
    attn_row_offset = token_id * attn_stride_t
    batch_idx = tl.load(triton_batch_indices_ptr + token_id).to(tl.int32)
    position = tl.load(triton_positions_ptr + token_id).to(tl.int32)
    page_start = tl.load(cu_num_pages_ptr + batch_idx).to(tl.int32)
    page_end = tl.load(cu_num_pages_ptr + batch_idx + 1).to(tl.int32)
    num_pages = page_end - page_start
    last_page_len = tl.load(last_page_len_ptr + batch_idx).to(tl.int32)
    cache_row = position.to(tl.int64) * cos_sin_cache_stride_s
    cos = tl.load(
        cos_sin_cache_ptr + cache_row + d2_offsets * cos_sin_cache_stride_d,
        mask=d2_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_row + (d2 + d2_offsets) * cos_sin_cache_stride_d,
        mask=d2_mask,
        other=0.0,
    ).to(tl.float32)
    q_weight_first = tl.load(q_norm_weight_ptr + d2_offsets, mask=d2_mask, other=0.0).to(tl.float32)
    q_weight_second = tl.load(q_norm_weight_ptr + d2 + d2_offsets, mask=d2_mask, other=0.0).to(
        tl.float32
    )
    k_weight_first = tl.load(k_norm_weight_ptr + d2_offsets, mask=d2_mask, other=0.0).to(tl.float32)
    k_weight_second = tl.load(k_norm_weight_ptr + d2 + d2_offsets, mask=d2_mask, other=0.0).to(
        tl.float32
    )
    v_weight_first = tl.load(v_norm_weight_ptr + d2_offsets, mask=d2_mask, other=0.0).to(tl.float32)
    v_weight_second = tl.load(v_norm_weight_ptr + d2 + d2_offsets, mask=d2_mask, other=0.0).to(
        tl.float32
    )
    qkv_row_offset = token_id * qkv_scratch_stride_t

    qkv_offsets = tl.arange(0, BLOCK_QKV)
    for qkv_start in range(0, QKV_WIDTH, BLOCK_QKV):
        qkv_cols = qkv_start + qkv_offsets
        qkv_mask = qkv_cols < QKV_WIDTH
        if USE_TC_QKV:
            qkv_acc_tc = tl.zeros([1, BLOCK_QKV], dtype=tl.float32)
        else:
            qkv_acc = tl.zeros([BLOCK_QKV], dtype=tl.float32)
        for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < HIDDEN_SIZE
            if USE_TC_QKV:
                x = tl.expand_dims(
                    tl.load(
                        attn_ptr + attn_row_offset + k_offsets * attn_stride_h,
                        mask=k_mask,
                        other=0.0,
                    ),
                    0,
                )
                w = tl.load(
                    qkv_weight_ptr
                    + k_offsets[:, None] * qkv_weight_stride_i
                    + qkv_cols[None, :] * qkv_weight_stride_o,
                    mask=k_mask[:, None] & qkv_mask[None, :],
                    other=0.0,
                )
                qkv_acc_tc = tl.dot(x, w, acc=qkv_acc_tc)
            else:
                x = tl.load(
                    attn_ptr + attn_row_offset + k_offsets * attn_stride_h,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)
                w = tl.load(
                    qkv_weight_ptr
                    + qkv_cols[:, None] * qkv_weight_stride_o
                    + k_offsets[None, :] * qkv_weight_stride_i,
                    mask=qkv_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                qkv_acc += tl.sum(w * x[None, :], axis=1)

        tl.store(
            qkv_scratch_ptr + qkv_row_offset + qkv_cols * qkv_scratch_stride_h,
            tl.sum(qkv_acc_tc, axis=0) if USE_TC_QKV else qkv_acc,
            mask=qkv_mask,
        )

    if not USE_ONE_SHOT_OPROJ:
        for h_start in range(0, HIDDEN_SIZE, BLOCK_H):
            hidden_offsets = h_start + h_offsets
            hidden_mask = hidden_offsets < HIDDEN_SIZE
            tl.store(
                o_proj_scratch_ptr
                + token_id * scratch_stride_t
                + hidden_offsets * scratch_stride_h,
                0.0,
                mask=hidden_mask,
            )

    for kv_head_idx in range(N_KV_HEADS):
        global_k_row = token_id * N_KV_HEADS + kv_head_idx

        k_first, k_second = _load_half_vectors(
            qkv_scratch_ptr,
            k_base_offset,
            global_k_row,
            D_HEAD,
            d2,
        )
        v_first, v_second = _load_half_vectors(
            qkv_scratch_ptr,
            v_base_offset,
            global_k_row,
            D_HEAD,
            d2,
        )
        k_first_f32 = k_first.to(tl.float32)
        k_second_f32 = k_second.to(tl.float32)
        v_first_f32 = v_first.to(tl.float32)
        v_second_f32 = v_second.to(tl.float32)
        k_var = (
            tl.sum(k_first_f32 * k_first_f32, axis=0) + tl.sum(k_second_f32 * k_second_f32, axis=0)
        ) / D_HEAD
        v_var = (
            tl.sum(v_first_f32 * v_first_f32, axis=0) + tl.sum(v_second_f32 * v_second_f32, axis=0)
        ) / D_HEAD
        k_inv_denom = tl.rsqrt(k_var + eps)
        v_inv_denom = tl.rsqrt(v_var + eps)

        k_first = (k_first_f32 * k_inv_denom * k_weight_first).to(k_first.dtype)
        k_second = (k_second_f32 * k_inv_denom * k_weight_second).to(k_second.dtype)
        v_first = (v_first_f32 * v_inv_denom * v_weight_first).to(v_first.dtype)
        v_second = (v_second_f32 * v_inv_denom * v_weight_second).to(v_second.dtype)
        k_rope_first = k_first.to(tl.float32) * cos - k_second.to(tl.float32) * sin
        k_rope_second = k_second.to(tl.float32) * cos + k_first.to(tl.float32) * sin

        page_idx_in_seq = position // PAGE_SIZE
        token_offset_in_page = position % PAGE_SIZE
        physical_page = tl.load(cache_loc_ptr + page_start + page_idx_in_seq).to(tl.int64)
        cache_base = (
            physical_page * cache_stride_block
            + kv_head_idx * cache_stride_head
            + token_offset_in_page.to(tl.int64) * cache_stride_token
        )
        tl.store(
            kv_cache_ptr + cache_base + d2_offsets,
            k_rope_first,
        )
        tl.store(
            kv_cache_ptr + cache_base + d2 + d2_offsets,
            k_rope_second,
        )
        tl.store(
            kv_cache_ptr + cache_base + cache_stride_kv + d2_offsets,
            v_first,
            mask=d2_mask,
        )
        tl.store(
            kv_cache_ptr + cache_base + cache_stride_kv + d2 + d2_offsets,
            v_second,
            mask=d2_mask,
        )

        q_head_indices = kv_head_idx * q_heads_per_kv + q_local_offsets
        global_q_rows = token_id * N_Q_HEADS + q_head_indices
        q_first = tl.load(
            qkv_scratch_ptr + q_base_offset + global_q_rows[:, None] * D_HEAD + d2_offsets[None, :],
            mask=q_local_offsets[:, None] < q_heads_per_kv,
            other=0.0,
        ).to(tl.float32)
        q_second = tl.load(
            qkv_scratch_ptr
            + q_base_offset
            + global_q_rows[:, None] * D_HEAD
            + (d2 + d2_offsets)[None, :],
            mask=q_local_offsets[:, None] < q_heads_per_kv,
            other=0.0,
        ).to(tl.float32)

        q_var = (tl.sum(q_first * q_first, axis=1) + tl.sum(q_second * q_second, axis=1)) / D_HEAD
        q_denom = tl.sqrt(q_var + eps)
        q_first = (q_first / q_denom[:, None]) * q_weight_first[None, :]
        q_second = (q_second / q_denom[:, None]) * q_weight_second[None, :]

        q_rope_first = q_first * cos[None, :] - q_second * sin[None, :]
        q_rope_second = q_second * cos[None, :] + q_first * sin[None, :]
        if USE_DOT_ATTN:
            q_rope_first_mat = q_rope_first.to(k_first.dtype)
            q_rope_second_mat = q_rope_second.to(k_first.dtype)

        m_i = tl.full([q_heads_per_kv], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([q_heads_per_kv], dtype=tl.float32)
        acc_first = tl.zeros([q_heads_per_kv, d2], dtype=tl.float32)
        acc_second = tl.zeros([q_heads_per_kv, d2], dtype=tl.float32)

        for page_number in range(MAX_NUM_PAGES):
            page_valid = page_number < num_pages
            valid_tokens = tl.where(page_number == (num_pages - 1), last_page_len, PAGE_SIZE)
            token_offsets = tl.arange(0, PAGE_SIZE)
            token_mask = page_valid & (token_offsets < valid_tokens)
            page_physical = tl.load(
                cache_loc_ptr + page_start + page_number, mask=page_valid, other=0
            )
            page_base = (
                page_physical.to(tl.int64) * cache_stride_block
                + kv_head_idx * cache_stride_head
                + token_offsets[:, None].to(tl.int64) * cache_stride_token
            )
            k_page_first = tl.load(
                kv_cache_ptr + page_base + d2_offsets[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            k_page_second = tl.load(
                kv_cache_ptr + page_base + d2 + d2_offsets[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            v_page = tl.load(
                kv_cache_ptr + page_base + cache_stride_kv + d2_offsets[None, :],
                mask=token_mask[:, None] & d2_mask[None, :],
                other=0.0,
            )
            v_page_second = tl.load(
                kv_cache_ptr + page_base + cache_stride_kv + d2 + d2_offsets[None, :],
                mask=token_mask[:, None] & d2_mask[None, :],
                other=0.0,
            )
            page_has_tokens = tl.sum(token_mask.to(tl.int32), axis=0) > 0
            if USE_DOT_ATTN:
                scores = (
                    tl.dot(q_rope_first_mat, tl.trans(k_page_first))
                    + tl.dot(q_rope_second_mat, tl.trans(k_page_second))
                ) * scale
                scores = tl.where(token_mask[None, :], scores, float("-inf"))
                page_m = tl.max(scores, axis=1)
                new_m = tl.where(page_has_tokens, tl.maximum(m_i, page_m), m_i)
                alpha = tl.exp(m_i - new_m)
                probs = tl.where(token_mask[None, :], tl.exp(scores - new_m[:, None]), 0.0)
                acc_first = tl.dot(
                    probs.to(v_page.dtype),
                    v_page,
                    acc=acc_first * alpha[:, None],
                )
                acc_second = tl.dot(
                    probs.to(v_page_second.dtype),
                    v_page_second,
                    acc=acc_second * alpha[:, None],
                )
                l_i = l_i * alpha + tl.sum(probs, axis=1)
            else:
                k_page_first_f32 = k_page_first.to(tl.float32)
                k_page_second_f32 = k_page_second.to(tl.float32)
                v_page_f32 = v_page.to(tl.float32)
                v_page_second_f32 = v_page_second.to(tl.float32)
                scores = (
                    tl.sum(k_page_first_f32[:, None, :] * q_rope_first[None, :, :], axis=2)
                    + tl.sum(k_page_second_f32[:, None, :] * q_rope_second[None, :, :], axis=2)
                ) * scale
                scores = tl.where(token_mask[:, None], scores, float("-inf"))
                page_m = tl.max(scores, axis=0)
                new_m = tl.where(page_has_tokens, tl.maximum(m_i, page_m), m_i)
                alpha = tl.exp(m_i - new_m)
                probs = tl.where(token_mask[:, None], tl.exp(scores - new_m[None, :]), 0.0)
                acc_first = acc_first * alpha[:, None] + tl.sum(
                    probs[:, :, None] * v_page_f32[:, None, :], axis=0
                )
                acc_second = acc_second * alpha[:, None] + tl.sum(
                    probs[:, :, None] * v_page_second_f32[:, None, :], axis=0
                )
                l_i = l_i * alpha + tl.sum(probs, axis=0)
            m_i = new_m

        attn_first = acc_first / l_i[:, None]
        attn_second = acc_second / l_i[:, None]

        if USE_ONE_SHOT_OPROJ:
            attn_head_offsets = q_head_indices[:, None] * D_HEAD + d2_offsets[None, :]
            attn_row_offset = token_id * attn_scratch_stride_t
            attn_mask = q_local_offsets[:, None] < q_heads_per_kv
            tl.store(
                attn_scratch_ptr + attn_row_offset + attn_head_offsets * attn_scratch_stride_h,
                attn_first.to(k_first.dtype),
                mask=attn_mask,
            )
            tl.store(
                attn_scratch_ptr
                + attn_row_offset
                + (q_head_indices[:, None] * D_HEAD + d2 + d2_offsets[None, :])
                * attn_scratch_stride_h,
                attn_second.to(k_second.dtype),
                mask=attn_mask,
            )
        else:
            for h_start in range(0, HIDDEN_SIZE, BLOCK_H):
                hidden_offsets = h_start + h_offsets
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                scratch_ptr = (
                    o_proj_scratch_ptr
                    + token_id * scratch_stride_t
                    + hidden_offsets * scratch_stride_h
                )
                current = tl.load(scratch_ptr, mask=hidden_mask, other=0.0)

                for q_local_idx in range(q_heads_per_kv):
                    o_proj_col_start = (kv_head_idx * q_heads_per_kv + q_local_idx) * D_HEAD
                    q_local_mask = q_local_offsets == q_local_idx
                    attn_first_local = tl.sum(
                        tl.where(q_local_mask[:, None], attn_first, 0.0),
                        axis=0,
                    )
                    attn_second_local = tl.sum(
                        tl.where(q_local_mask[:, None], attn_second, 0.0),
                        axis=0,
                    )
                    w_first = tl.load(
                        o_proj_weight_ptr
                        + hidden_offsets[:, None] * o_proj_weight_stride_o
                        + (o_proj_col_start + d2_offsets)[None, :] * o_proj_weight_stride_i,
                        mask=hidden_mask[:, None] & d2_mask[None, :],
                        other=0.0,
                    )
                    w_second = tl.load(
                        o_proj_weight_ptr
                        + hidden_offsets[:, None] * o_proj_weight_stride_o
                        + (o_proj_col_start + d2 + d2_offsets)[None, :] * o_proj_weight_stride_i,
                        mask=hidden_mask[:, None] & d2_mask[None, :],
                        other=0.0,
                    )
                    if USE_TC_OPROJ:
                        partial_first = tl.dot(
                            tl.expand_dims(attn_first_local.to(w_first.dtype), 0),
                            tl.trans(w_first),
                        )
                        partial_second = tl.dot(
                            tl.expand_dims(attn_second_local.to(w_second.dtype), 0),
                            tl.trans(w_second),
                        )
                        partial = tl.sum(partial_first, axis=0) + tl.sum(partial_second, axis=0)
                    else:
                        w_first_f32 = w_first.to(tl.float32)
                        w_second_f32 = w_second.to(tl.float32)
                        partial = tl.sum(w_first_f32 * attn_first_local[None, :], axis=1) + tl.sum(
                            w_second_f32 * attn_second_local[None, :], axis=1
                        )
                    current += partial

                tl.store(scratch_ptr, current, mask=hidden_mask)

    sumsq_attn = 0.0
    if USE_ONE_SHOT_OPROJ:
        o_proj_offsets = tl.arange(0, BLOCK_QKV)
        attn_row_offset = token_id * attn_scratch_stride_t
        q_width: tl.constexpr = N_Q_HEADS * D_HEAD
        for h_start in range(0, HIDDEN_SIZE, BLOCK_H):
            hidden_offsets = h_start + h_offsets
            hidden_mask = hidden_offsets < HIDDEN_SIZE
            if USE_TC_OPROJ:
                o_proj_acc_tc = tl.zeros([1, BLOCK_H], dtype=tl.float32)
            else:
                o_proj_acc = tl.zeros([BLOCK_H], dtype=tl.float32)

            for q_start in range(0, q_width, BLOCK_QKV):
                q_cols = q_start + o_proj_offsets
                q_mask = q_cols < q_width
                if USE_TC_OPROJ:
                    x = tl.expand_dims(
                        tl.load(
                            attn_scratch_ptr + attn_row_offset + q_cols * attn_scratch_stride_h,
                            mask=q_mask,
                            other=0.0,
                        ),
                        0,
                    )
                    w = tl.load(
                        o_proj_weight_ptr
                        + q_cols[:, None] * o_proj_weight_stride_i
                        + hidden_offsets[None, :] * o_proj_weight_stride_o,
                        mask=q_mask[:, None] & hidden_mask[None, :],
                        other=0.0,
                    )
                    o_proj_acc_tc = tl.dot(x, w, acc=o_proj_acc_tc)
                else:
                    x = tl.load(
                        attn_scratch_ptr + attn_row_offset + q_cols * attn_scratch_stride_h,
                        mask=q_mask,
                        other=0.0,
                    ).to(tl.float32)
                    w = tl.load(
                        o_proj_weight_ptr
                        + hidden_offsets[:, None] * o_proj_weight_stride_o
                        + q_cols[None, :] * o_proj_weight_stride_i,
                        mask=hidden_mask[:, None] & q_mask[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    o_proj_acc += tl.sum(w * x[None, :], axis=1)

            tl.store(
                o_proj_scratch_ptr
                + token_id * scratch_stride_t
                + hidden_offsets * scratch_stride_h,
                tl.sum(o_proj_acc_tc, axis=0) if USE_TC_OPROJ else o_proj_acc,
                mask=hidden_mask,
            )
            o_block = tl.sum(o_proj_acc_tc, axis=0) if USE_TC_OPROJ else o_proj_acc
            o_block_f32 = o_block.to(tl.float32)
            sumsq_attn += tl.sum(o_block_f32 * o_block_f32, axis=0)
    else:
        for h_start in range(0, HIDDEN_SIZE, BLOCK_H):
            hidden_offsets = h_start + h_offsets
            hidden_mask = hidden_offsets < HIDDEN_SIZE
            o_block = tl.load(
                o_proj_scratch_ptr
                + token_id * scratch_stride_t
                + hidden_offsets * scratch_stride_h,
                mask=hidden_mask,
                other=0.0,
            )
            o_block_f32 = o_block.to(tl.float32)
            sumsq_attn += tl.sum(o_block_f32 * o_block_f32, axis=0)
    inv_rms_attn = tl.rsqrt(sumsq_attn / HIDDEN_SIZE + eps)

    sumsq_post = 0.0
    for h_start in range(0, HIDDEN_SIZE, BLOCK_H):
        hidden_offsets = h_start + h_offsets
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        o_block = tl.load(
            o_proj_scratch_ptr + token_id * scratch_stride_t + hidden_offsets * scratch_stride_h,
            mask=hidden_mask,
            other=0.0,
        )
        residual_block = tl.load(
            residual_ptr + residual_row_offset + hidden_offsets * residual_stride_h,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        post_weight = tl.load(
            post_attention_layernorm_weight_ptr + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        attn_branch = o_block * inv_rms_attn * post_weight
        post_block = residual_block + attn_branch
        tl.store(
            post_out_ptr + token_id * post_out_stride_t + hidden_offsets * post_out_stride_h,
            post_block,
            mask=hidden_mask,
        )
        sumsq_post += tl.sum(post_block * post_block, axis=0)

    inv_rms_post = tl.rsqrt(sumsq_post / HIDDEN_SIZE + eps)
    for h_start in range(0, HIDDEN_SIZE, BLOCK_H):
        hidden_offsets = h_start + h_offsets
        hidden_mask = hidden_offsets < HIDDEN_SIZE
        post_block = tl.load(
            post_out_ptr + token_id * post_out_stride_t + hidden_offsets * post_out_stride_h,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        pre_weight = tl.load(
            pre_feedforward_layernorm_weight_ptr + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        pre_block = post_block * inv_rms_post * pre_weight
        tl.store(
            pre_out_ptr + token_id * pre_out_stride_t + hidden_offsets * pre_out_stride_h,
            pre_block,
            mask=hidden_mask,
        )


def _split_qkv(
    qkv: torch.Tensor,
    q_width: int,
    kv_width: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_width = int(qkv.shape[-1])
    if q_width <= 0 or kv_width <= 0 or total_width != q_width + 2 * kv_width:
        raise ValueError(
            "Expected fused QKV width to equal q_width + 2 * kv_width, got "
            f"{total_width} with q_width={q_width} and kv_width={kv_width}"
        )
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


def _run_reference_kernel_a(
    residual_bsh: torch.Tensor,
    attn_normed_bsh: torch.Tensor,
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
    scale: Optional[float],
    sliding_window: Optional[int],
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head_dim = int(q_norm_weight.numel())
    q_width = int(o_proj_weight.shape[1])
    kv_width_total = int(qkv_weight.shape[0]) - q_width
    if kv_width_total <= 0 or kv_width_total % 2 != 0:
        raise ValueError(
            "Expected fused QKV weight rows to be q_width + 2 * kv_width, got "
            f"{int(qkv_weight.shape[0])} with q_width={q_width}"
        )
    kv_width = kv_width_total // 2

    qkv = torch.ops.aten.linear(attn_normed_bsh, qkv_weight, None)
    q, k, v = _split_qkv(qkv, q_width, kv_width, head_dim)

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
    return post_attn_residual, pre_ffn_normed


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

    if residual_bsh.shape[1] != 1:
        raise ValueError(
            "triton_gemma_kernel_a_decode currently supports decode-only seq_len=1, got "
            f"{residual_bsh.shape[1]}"
        )
    if sliding_window is not None and sliding_window > 0:
        raise NotImplementedError(
            "triton_gemma_kernel_a_decode single-kernel path does not support sliding_window yet"
        )

    hidden_size = int(residual_bsh.shape[-1])
    head_dim = int(q_norm_weight.numel())
    q_width = int(o_proj_weight.shape[1])
    kv_width_total = int(qkv_weight.shape[0]) - q_width
    if kv_width_total <= 0 or kv_width_total % 2 != 0:
        raise ValueError(
            "Expected fused QKV weight rows to be q_width + 2 * kv_width, got "
            f"{int(qkv_weight.shape[0])} with q_width={q_width}"
        )
    kv_width = kv_width_total // 2
    if q_width % head_dim != 0 or kv_width % head_dim != 0:
        raise ValueError(
            "Projected widths must be multiples of head_dim: "
            f"q_width={q_width}, kv_width={kv_width}, head_dim={head_dim}"
        )
    if head_dim % 2 != 0:
        raise ValueError(f"Single-kernel path requires an even head_dim for RoPE, got {head_dim}")

    num_q_heads = q_width // head_dim
    num_kv_heads = kv_width // head_dim
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "Single-kernel path requires grouped-query attention with q_heads divisible by kv_heads, got "
            f"q_heads={num_q_heads}, kv_heads={num_kv_heads}"
        )
    if qkv_weight.dtype != residual_bsh.dtype or o_proj_weight.dtype != residual_bsh.dtype:
        raise ValueError(
            "Single-kernel path expects linear weights to match the hidden-state dtype"
        )

    residual_flat = residual_bsh.reshape(-1, hidden_size).contiguous()
    attn_normed_flat = attn_normed_bsh.reshape(-1, hidden_size).contiguous()
    qkv_weight = qkv_weight.contiguous()
    o_proj_weight = o_proj_weight.contiguous()
    kv_cache = kv_cache.contiguous()

    num_tokens = residual_flat.shape[0]
    post_attn_flat = torch.empty_like(residual_flat)
    pre_ffn_flat = torch.empty_like(residual_flat)
    qkv_scratch = torch.empty(
        (num_tokens, int(qkv_weight.shape[0])),
        device=residual_flat.device,
        dtype=residual_flat.dtype,
    )
    attn_scratch = torch.empty(
        (num_tokens, q_width),
        device=residual_flat.device,
        dtype=residual_flat.dtype,
    )
    o_proj_scratch = torch.empty(
        (num_tokens, hidden_size), device=residual_flat.device, dtype=torch.float32
    )
    scale = float(scale) if scale is not None else 1.0 / math.sqrt(head_dim)
    page_size = int(kv_cache.shape[3])
    max_seq_with_cache = int(seq_len_with_cache_host.max().item())
    max_num_pages = max(1, (max_seq_with_cache + page_size - 1) // page_size)
    use_tc_qkv = (
        hidden_size >= 1024
        and head_dim >= 128
        and residual_flat.dtype in (torch.float16, torch.bfloat16)
    )
    use_tc_oproj = (
        hidden_size >= 1024
        and head_dim >= 128
        and residual_flat.dtype in (torch.float16, torch.bfloat16)
    )
    use_dot_attn = hidden_size >= 1024 and head_dim >= 128
    use_one_shot_oproj = hidden_size >= 1024 and head_dim >= 128

    block_h = _get_env_int(
        "AD_GEMMA_KERNEL_A_BLOCK_H",
        128 if hidden_size >= 128 else triton.next_power_of_2(hidden_size),
    )
    block_k = _get_env_int(
        "AD_GEMMA_KERNEL_A_BLOCK_K",
        128
        if hidden_size >= 1024
        else 64
        if hidden_size >= 64
        else triton.next_power_of_2(hidden_size),
    )
    block_qkv = _get_env_int("AD_GEMMA_KERNEL_A_BLOCK_QKV", 512 if use_tc_qkv else 128)
    num_warps = _get_env_int("AD_GEMMA_KERNEL_A_NUM_WARPS", 8 if hidden_size >= 1024 else 4)
    num_stages = _get_env_int("AD_GEMMA_KERNEL_A_NUM_STAGES", 2)
    if "AD_GEMMA_KERNEL_A_USE_TC_PATH" in os.environ:
        use_tc_override = bool(int(os.environ["AD_GEMMA_KERNEL_A_USE_TC_PATH"]))
        use_tc_qkv = use_tc_override
        use_tc_oproj = use_tc_override
    if "AD_GEMMA_KERNEL_A_USE_TC_QKV" in os.environ:
        use_tc_qkv = bool(int(os.environ["AD_GEMMA_KERNEL_A_USE_TC_QKV"]))
    if "AD_GEMMA_KERNEL_A_USE_TC_OPROJ" in os.environ:
        use_tc_oproj = bool(int(os.environ["AD_GEMMA_KERNEL_A_USE_TC_OPROJ"]))
    if "AD_GEMMA_KERNEL_A_USE_DOT_ATTN" in os.environ:
        use_dot_attn = bool(int(os.environ["AD_GEMMA_KERNEL_A_USE_DOT_ATTN"]))
    if "AD_GEMMA_KERNEL_A_USE_ONE_SHOT_OPROJ" in os.environ:
        use_one_shot_oproj = bool(int(os.environ["AD_GEMMA_KERNEL_A_USE_ONE_SHOT_OPROJ"]))
    _gemma_kernel_a_decode_single_kernel[(num_tokens,)](
        residual_flat,
        attn_normed_flat,
        qkv_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight,
        triton_batch_indices,
        triton_positions,
        cu_num_pages,
        cache_loc,
        last_page_len,
        cos_sin_cache,
        kv_cache,
        qkv_scratch,
        attn_scratch,
        o_proj_scratch,
        post_attn_flat,
        pre_ffn_flat,
        residual_flat.stride(0),
        residual_flat.stride(1),
        attn_normed_flat.stride(0),
        attn_normed_flat.stride(1),
        qkv_weight.stride(0),
        qkv_weight.stride(1),
        o_proj_weight.stride(0),
        o_proj_weight.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        cos_sin_cache.stride(0),
        cos_sin_cache.stride(1),
        qkv_scratch.stride(0),
        qkv_scratch.stride(1),
        attn_scratch.stride(0),
        attn_scratch.stride(1),
        o_proj_scratch.stride(0),
        o_proj_scratch.stride(1),
        post_attn_flat.stride(0),
        post_attn_flat.stride(1),
        pre_ffn_flat.stride(0),
        pre_ffn_flat.stride(1),
        scale,
        eps,
        HIDDEN_SIZE=hidden_size,
        D_HEAD=head_dim,
        N_Q_HEADS=num_q_heads,
        N_KV_HEADS=num_kv_heads,
        PAGE_SIZE=page_size,
        MAX_NUM_PAGES=max_num_pages,
        QKV_WIDTH=int(qkv_weight.shape[0]),
        BLOCK_QKV=block_qkv,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        USE_TC_QKV=use_tc_qkv,
        USE_TC_OPROJ=use_tc_oproj,
        USE_DOT_ATTN=use_dot_attn,
        USE_ONE_SHOT_OPROJ=use_one_shot_oproj,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        _restore_hidden_layout(post_attn_flat.reshape_as(residual_bsh), original_shape),
        _restore_hidden_layout(pre_ffn_flat.reshape_as(residual_bsh), original_shape),
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
