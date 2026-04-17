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

"""AutoDeploy custom op wrapper for the Gemma4 CUDA megakernel (Kernel A).

This registers ``auto_deploy::megakernel_gemma_kernel_a_decode`` with the
exact same signature as ``auto_deploy::triton_gemma_kernel_a_decode``,
making it a drop-in replacement for the attention sublayer.

The op internally builds the instruction stream, schedules multi-SM attention,
and launches the persistent CUDA megakernel.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

# Constants matching gemma4_config.cuh
_HIDDEN_SIZE = 2816
_HEAD_DIM = 256
_NUM_Q_HEADS = 16
_NUM_KV_HEADS = 8
_GQA_RATIO = _NUM_Q_HEADS // _NUM_KV_HEADS
_Q_WIDTH = _NUM_Q_HEADS * _HEAD_DIM
_KV_WIDTH = _NUM_KV_HEADS * _HEAD_DIM
_QKV_WIDTH = _Q_WIDTH + 2 * _KV_WIDTH
_NUM_SMS = 132
_TOTAL_HEADS = _NUM_Q_HEADS + 2 * _NUM_KV_HEADS


def _dist_rows(total_rows: int, num_sms: int) -> list[tuple[int, int]]:
    rows_per_sm = total_rows // num_sms
    remainder = total_rows % num_sms
    ranges = []
    start = 0
    for i in range(num_sms):
        count = rows_per_sm + (1 if i < remainder else 0)
        ranges.append((start, start + count))
        start += count
    return ranges


def _build_kernel_a_instructions(
    *,
    num_tokens: int,
    total_pages: int,
    page_size: int,
    sliding_window: int,
    shared_kv: bool,
    rope_dim: int,
):
    """Build the instruction stream for one Kernel A decode step."""
    from .launcher import InstructionBuilder

    builder = InstructionBuilder(num_sms=_NUM_SMS)
    all_sms = range(_NUM_SMS)

    for token_id in range(num_tokens):
        # Phase 1a: QKV GEMV
        builder.add_gemv_qkv(all_sms, _dist_rows(_QKV_WIDTH, _NUM_SMS), token_id=token_id)
        builder.add_barrier(all_sms, barrier_id=token_id * 10 + 0)

        # Phase 1b: QKV post (norms + RoPE + cache write)
        builder.add_qkv_post(
            range(_TOTAL_HEADS),
            [(h, h + 1) for h in range(_TOTAL_HEADS)],
            token_id=token_id,
            shared_kv=shared_kv,
            rope_dim=rope_dim,
        )
        builder.add_barrier(all_sms, barrier_id=token_id * 10 + 1)

        # Phase 2: Attention (multi-SM)
        if total_pages > 2:
            num_pp = min((total_pages + 1) // 2, 16)
        else:
            num_pp = 1
        use_multi = num_pp > 1

        if use_multi:
            sm_ids, kv_heads, page_ranges, partial_ids = [], [], [], []
            sm_counter = 0
            pps = (total_pages + num_pp - 1) // num_pp
            for kv_h in range(_NUM_KV_HEADS):
                for pi in range(num_pp):
                    ps_start = pi * pps
                    ps_end = min(ps_start + pps, total_pages)
                    if ps_start >= total_pages:
                        break
                    sm_ids.append(sm_counter)
                    kv_heads.append(kv_h)
                    page_ranges.append((ps_start, ps_end))
                    partial_ids.append(kv_h * num_pp + pi)
                    sm_counter += 1
            builder.add_paged_attn(
                sm_ids,
                kv_heads,
                token_id=token_id,
                page_ranges=page_ranges,
                partial_indices=partial_ids,
                is_single=False,
                sliding_window=sliding_window,
            )
            builder.add_barrier(all_sms, barrier_id=token_id * 10 + 2)
            partial_starts = [kv_h * num_pp for kv_h in range(_NUM_KV_HEADS)]
            builder.add_attn_reduce(
                range(_NUM_KV_HEADS),
                list(range(_NUM_KV_HEADS)),
                num_pp,
                partial_starts=partial_starts,
                token_id=token_id,
            )
            builder.add_barrier(all_sms, barrier_id=token_id * 10 + 3)
            next_bid = token_id * 10 + 4
        else:
            builder.add_paged_attn(
                range(_NUM_KV_HEADS),
                list(range(_NUM_KV_HEADS)),
                token_id=token_id,
                sliding_window=sliding_window,
            )
            builder.add_barrier(all_sms, barrier_id=token_id * 10 + 2)
            next_bid = token_id * 10 + 3

        # Phase 3: O-proj GEMV + OPROJ_POST
        builder.add_gemv_oproj(all_sms, _dist_rows(_HIDDEN_SIZE, _NUM_SMS), token_id=token_id)
        builder.add_barrier(all_sms, barrier_id=next_bid)
        builder.add_oproj_post(0, token_id=token_id)

    builder.add_done(all_sms)

    max_partials = _NUM_KV_HEADS * max(num_pp, 1)
    return builder, use_multi, max_partials


# Lazily initialized launcher (JIT-compiled on first call)
_launcher = None


def _get_launcher():
    global _launcher
    if _launcher is None:
        from .launcher import MegakernelLauncher

        _launcher = MegakernelLauncher(num_sms=_NUM_SMS)
    return _launcher


@torch.library.custom_op(
    "auto_deploy::megakernel_gemma_kernel_a_decode", mutates_args=("kv_cache",)
)
def megakernel_gemma_kernel_a_decode(
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
    """Execute Gemma's attention half via the persistent CUDA megakernel."""
    device = residual_in.device
    hidden_size = residual_in.shape[-1]
    num_tokens = residual_in.reshape(-1, hidden_size).shape[0]
    page_size = kv_cache.shape[3]

    if scale is None:
        scale = 1.0 / math.sqrt(_HEAD_DIM)
    sw = sliding_window if sliding_window is not None and sliding_window > 0 else 0

    # Compute total pages from metadata
    max_seq_with_cache = int(seq_len_with_cache_host.max().item())
    total_pages = max(1, (max_seq_with_cache + page_size - 1) // page_size)

    # Build instruction stream
    builder, use_multi, max_partials = _build_kernel_a_instructions(
        num_tokens=num_tokens,
        total_pages=total_pages,
        page_size=page_size,
        sliding_window=sw,
        shared_kv=False,  # TODO: detect from model config
        rope_dim=0,  # TODO: detect from model config
    )

    # Allocate scratch buffers
    dtype = residual_in.dtype
    flat_residual = residual_in.reshape(num_tokens, hidden_size).contiguous()
    flat_normed = attn_normed_in.reshape(num_tokens, hidden_size).contiguous()
    qkv_scratch = torch.empty(num_tokens, _QKV_WIDTH, device=device, dtype=dtype)
    attn_scratch = torch.empty(num_tokens, _Q_WIDTH, device=device, dtype=dtype)
    o_proj_scratch = torch.empty(num_tokens, hidden_size, device=device, dtype=torch.float32)
    post_attn_out = torch.empty(num_tokens, hidden_size, device=device, dtype=torch.float32)
    pre_ffn_out = torch.empty(num_tokens, hidden_size, device=device, dtype=torch.float32)
    partial_scr = (
        torch.zeros(max_partials, _NUM_Q_HEADS, _HEAD_DIM + 2, device=device, dtype=torch.float32)
        if use_multi
        else None
    )

    # Launch megakernel
    launcher = _get_launcher()
    launcher.launch_gemv_qkv(
        builder,
        flat_normed,
        qkv_weight.contiguous(),
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        cos_sin_cache,
        kv_cache,
        cache_loc.to(dtype=torch.int32, device=device),
        cu_num_pages.to(dtype=torch.int32, device=device),
        triton_positions.to(dtype=torch.int32, device=device),
        triton_batch_indices.to(dtype=torch.int32, device=device),
        last_page_len.to(dtype=torch.int32, device=device),
        qkv_scratch,
        attn_scratch,
        eps=eps,
        page_size=page_size,
        attn_scale=scale,
        o_proj_weight=o_proj_weight.contiguous(),
        residual=flat_residual,
        post_attn_norm_weight=post_attention_layernorm_weight,
        pre_ffn_norm_weight=pre_feedforward_layernorm_weight,
        o_proj_scratch=o_proj_scratch,
        post_attn_out=post_attn_out,
        pre_ffn_out=pre_ffn_out,
        partial_attn_scratch=partial_scr,
    )

    # Convert outputs to bf16 and reshape to match input shape
    original_shape = residual_in.shape
    post_attn = post_attn_out.to(dtype).reshape(original_shape)
    pre_ffn = pre_ffn_out.to(dtype).reshape(original_shape)

    return post_attn, pre_ffn


@megakernel_gemma_kernel_a_decode.register_fake
def megakernel_gemma_kernel_a_decode_fake(
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
    """Fake implementation for symbolic tracing and shape inference."""
    return torch.empty_like(residual_in), torch.empty_like(residual_in)
