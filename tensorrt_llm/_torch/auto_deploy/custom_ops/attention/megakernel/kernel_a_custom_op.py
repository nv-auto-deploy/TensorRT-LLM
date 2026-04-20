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
_OPROJ_POST_SMS = 32


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
    from .launcher import (
        InstructionBuilder,
        choose_attention_num_partials,
        partition_attention_pages,
    )

    builder = InstructionBuilder(num_sms=_NUM_SMS)
    all_sms = range(_NUM_SMS)

    def add_barrier_noops(active_sms, barrier_id: int) -> None:
        active_set = set(active_sms)
        idle_sms = [sm_id for sm_id in all_sms if sm_id not in active_set]
        if idle_sms:
            builder.add_noop(idle_sms, barrier=(_NUM_SMS, barrier_id))

    for token_id in range(num_tokens):
        # Phase 1a: QKV GEMV
        builder.add_gemv_qkv(
            all_sms,
            _dist_rows(_QKV_WIDTH, _NUM_SMS),
            token_id=token_id,
            barrier=(_NUM_SMS, token_id * 10 + 0),
        )

        # Phase 1b: QKV post (norms + RoPE + cache write)
        builder.add_qkv_post(
            range(_TOTAL_HEADS),
            [(h, h + 1) for h in range(_TOTAL_HEADS)],
            token_id=token_id,
            barrier=(_NUM_SMS, token_id * 10 + 1),
            shared_kv=shared_kv,
            rope_dim=rope_dim,
        )
        add_barrier_noops(range(_TOTAL_HEADS), token_id * 10 + 1)

        # Phase 2: Attention (multi-SM)
        num_pp = choose_attention_num_partials(total_pages)
        use_multi = num_pp > 1

        if use_multi:
            sm_ids, kv_heads, page_ranges, partial_ids = [], [], [], []
            sm_counter = 0
            ranges = partition_attention_pages(total_pages, num_pp)
            for kv_h in range(_NUM_KV_HEADS):
                for pi, (ps_start, ps_end) in enumerate(ranges):
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
                barrier=(_NUM_SMS, token_id * 10 + 2),
                sliding_window=sliding_window,
            )
            add_barrier_noops(sm_ids, token_id * 10 + 2)
            partial_starts = [kv_h * num_pp for kv_h in range(_NUM_KV_HEADS)]
            builder.add_attn_reduce(
                range(_NUM_KV_HEADS),
                list(range(_NUM_KV_HEADS)),
                num_pp,
                partial_starts=partial_starts,
                token_id=token_id,
                barrier=(_NUM_SMS, token_id * 10 + 3),
            )
            add_barrier_noops(range(_NUM_KV_HEADS), token_id * 10 + 3)
            next_bid = token_id * 10 + 4
        else:
            builder.add_paged_attn(
                range(_NUM_KV_HEADS),
                list(range(_NUM_KV_HEADS)),
                token_id=token_id,
                barrier=(_NUM_SMS, token_id * 10 + 2),
                sliding_window=sliding_window,
            )
            add_barrier_noops(range(_NUM_KV_HEADS), token_id * 10 + 2)
            next_bid = token_id * 10 + 3

        # Phase 3: O-proj GEMV + distributed OPROJ_POST
        builder.add_gemv_oproj(
            all_sms,
            _dist_rows(_HIDDEN_SIZE, _NUM_SMS),
            token_id=token_id,
            barrier=(_NUM_SMS, next_bid),
        )
        oproj_post_sms = range(_OPROJ_POST_SMS)
        oproj_post_rows = _dist_rows(_HIDDEN_SIZE, _OPROJ_POST_SMS)
        builder.add_oproj_post_stats(
            oproj_post_sms,
            oproj_post_rows,
            token_id=token_id,
            barrier=(_NUM_SMS, next_bid + 1),
        )
        add_barrier_noops(oproj_post_sms, next_bid + 1)
        builder.add_oproj_post_apply(
            oproj_post_sms,
            oproj_post_rows,
            token_id=token_id,
            barrier=(_NUM_SMS, next_bid + 2),
        )
        add_barrier_noops(oproj_post_sms, next_bid + 2)
        builder.add_pre_ffn_post(
            oproj_post_sms,
            oproj_post_rows,
            token_id=token_id,
        )

    builder.add_done(all_sms)

    max_partials = _NUM_KV_HEADS * max(num_pp, 1)
    return builder, use_multi, max_partials


# Lazily initialized launcher and cached resources
_launcher = None
_cached_resources = {}  # keyed by (num_tokens, total_pages, page_size, sliding_window)


def _get_launcher():
    global _launcher
    if _launcher is None:
        from .launcher import MegakernelLauncher

        _launcher = MegakernelLauncher(num_sms=_NUM_SMS)
    return _launcher


def _get_cached_resources(num_tokens, total_pages, page_size, sw, device):
    """Get or create cached instruction builder + scratch buffers.

    All CUDA allocations happen here on first call. Subsequent calls
    reuse the cached tensors (no CUDA alloc in the hot path).
    """
    key = (num_tokens, total_pages, page_size, sw)
    if key not in _cached_resources:
        builder, use_multi, max_partials = _build_kernel_a_instructions(
            num_tokens=num_tokens,
            total_pages=total_pages,
            page_size=page_size,
            sliding_window=sw,
            shared_kv=False,
            rope_dim=0,
        )
        hidden_size = _HIDDEN_SIZE
        dtype = torch.bfloat16
        scratch = {
            "builder": builder,
            "use_multi": use_multi,
            "max_partials": max_partials,
            "barrier_slots": torch.zeros(256, dtype=torch.int32, device=device),
            "debug_output": torch.zeros(_NUM_SMS, dtype=torch.int32, device=device),
            "qkv_scratch": torch.empty(num_tokens, _QKV_WIDTH, device=device, dtype=dtype),
            "attn_scratch": torch.empty(num_tokens, _Q_WIDTH, device=device, dtype=dtype),
            "o_proj_scratch": torch.empty(
                num_tokens, hidden_size, device=device, dtype=torch.float32
            ),
            "post_attn_out": torch.empty(
                num_tokens, hidden_size, device=device, dtype=torch.float32
            ),
            "pre_ffn_out": torch.empty(num_tokens, hidden_size, device=device, dtype=torch.float32),
            "partial_scr": (
                torch.zeros(
                    max_partials, _NUM_Q_HEADS, _HEAD_DIM + 2, device=device, dtype=torch.float32
                )
                if use_multi
                else None
            ),
        }
        _cached_resources[key] = scratch
    return _cached_resources[key]


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
    """Execute Gemma's attention half via the persistent CUDA megakernel.

    For decode (num_tokens=1): uses the persistent CUDA megakernel.
    For prefill (num_tokens>1): falls back to the decomposed reference path.
    """
    device = residual_in.device
    hidden_size = residual_in.shape[-1]
    num_tokens = residual_in.reshape(-1, hidden_size).shape[0]
    page_size = kv_cache.shape[3]

    # Prefill fallback: megakernel is decode-only (batch_size=1, seq_len=1)
    if num_tokens > 1:
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_gemma_kernel_a_decode import (
            _run_reference_kernel_a,
        )

        original_shape = residual_in.shape
        ref_post, ref_pre = _run_reference_kernel_a(
            residual_in.unsqueeze(1) if residual_in.ndim == 2 else residual_in,
            attn_normed_in.unsqueeze(1) if attn_normed_in.ndim == 2 else attn_normed_in,
            qkv_weight,
            o_proj_weight,
            q_norm_weight.float(),
            k_norm_weight.float(),
            v_norm_weight.float(),
            post_attention_layernorm_weight.float(),
            pre_feedforward_layernorm_weight.float(),
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
        # Reshape to match the graph's expected shape (same as residual_in)
        return ref_post.reshape(original_shape), ref_pre.reshape(original_shape)

    if scale is None:
        scale = 1.0 / math.sqrt(_HEAD_DIM)
    sw = sliding_window if sliding_window is not None and sliding_window > 0 else 0

    # Compute total pages from host metadata (no device sync)
    swc = seq_len_with_cache_host
    if swc.is_cuda:
        swc = swc.cpu()
    max_seq_with_cache = int(swc.max())
    total_pages = max(1, (max_seq_with_cache + page_size - 1) // page_size)

    # Get cached instruction builder + scratch buffers (no CUDA alloc in hot path)
    cached = _get_cached_resources(num_tokens, total_pages, page_size, sw, device)
    builder = cached["builder"]

    flat_residual = residual_in.reshape(num_tokens, hidden_size).contiguous()
    flat_normed = attn_normed_in.reshape(num_tokens, hidden_size).contiguous()
    qkv_scratch = cached["qkv_scratch"]
    attn_scratch = cached["attn_scratch"]
    o_proj_scratch = cached["o_proj_scratch"]
    post_attn_out = cached["post_attn_out"]
    pre_ffn_out = cached["pre_ffn_out"]
    partial_scr = cached["partial_scr"]
    barrier_slots = cached["barrier_slots"]
    debug_output = cached["debug_output"]

    # Launch megakernel
    launcher = _get_launcher()
    launcher.launch_gemv_qkv(
        builder,
        flat_normed,
        qkv_weight.contiguous(),
        q_norm_weight.float(),
        k_norm_weight.float(),
        v_norm_weight.float(),
        cos_sin_cache,
        kv_cache,
        cache_loc,
        cu_num_pages,
        triton_positions,
        triton_batch_indices,
        last_page_len,
        qkv_scratch,
        attn_scratch,
        eps=eps,
        page_size=page_size,
        attn_scale=scale,
        o_proj_weight=o_proj_weight.contiguous(),
        residual=flat_residual,
        post_attn_norm_weight=post_attention_layernorm_weight.float(),
        pre_ffn_norm_weight=pre_feedforward_layernorm_weight.float(),
        o_proj_scratch=o_proj_scratch,
        post_attn_out=post_attn_out,
        pre_ffn_out=pre_ffn_out,
        partial_attn_scratch=partial_scr,
        barrier_slots=barrier_slots,
        debug_output=debug_output,
    )

    # TODO: The megakernel causes illegal memory access when called from the
    # FX graph in the executor context. This needs debugging with compute-sanitizer.
    # The graph structure is verified correct (dummy zeros work).
    # Suspected cause: KV cache page_size=32 (executor default) vs page_size=16
    # (megakernel test default), or shared memory config not set in child process.

    # Convert outputs to bf16 and reshape to match input shape
    original_shape = residual_in.shape
    out_dtype = residual_in.dtype
    post_attn = post_attn_out.to(out_dtype).reshape(original_shape).clone()
    pre_ffn = pre_ffn_out.to(out_dtype).reshape(original_shape).clone()

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
