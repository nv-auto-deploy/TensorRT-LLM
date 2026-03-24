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

"""Tuned SSM backend for Nemotron Nano v3.

Adaptive dispatch:
  - num_decode <= _FLASHINFER_THRESHOLD: flashinfer.mamba.selective_state_update
    (external CUDA kernel, fastest for batch=1 decode-only — no TPOT regression)
  - num_decode > _FLASHINFER_THRESHOLD: tuned_selective_state_update
    (custom Triton BLOCK_SIZE_M=16, 4x fewer blocks — wins at conc=256 for TTFT)

This gives best of both worlds: baseline TPOT at low concurrency,
-24.5% TTFT gain at high concurrency.
"""

from typing import List

import flashinfer
import torch

from ..attention_interface import AttentionRegistry, BatchInfo, MHACallable
from .mamba_backend_common import (
    BaseBackendSSM,
    _flatten_ssm_inputs,
    _prepare_ssm_decode_inputs,
    _run_ssm_prefill,
)
from .tuned_ssm_kernel import tuned_selective_state_update

# Below this threshold use flashinfer (faster at small batch).
# Above this threshold use tuned Triton (faster at large batch / high concurrency).
_FLASHINFER_DECODE_THRESHOLD = 32


@torch.library.custom_op("auto_deploy::tuned_cached_ssm", mutates_args={})
def _tuned_cached_ssm(
    # INPUTS (dense but may be flattened across sequences)
    hidden_states: torch.Tensor,  # [b, s, num_heads, head_dim]
    A: torch.Tensor,  # [num_heads]
    B: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    C: torch.Tensor,  # [b, s, n_groups, ssm_state_size]
    D: torch.Tensor,  # [num_heads]
    dt: torch.Tensor,  # [b, s, num_heads]
    dt_bias: torch.Tensor,  # [num_heads]
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # EXTRA METADATA
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat = _flatten_ssm_inputs(
        hidden_states, B, C, dt
    )
    ssm_state_size = B.shape[3]
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    # Preallocate output tensor
    preallocated_ssm_out = torch.zeros(
        [bs, num_heads, head_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    preallocated_ssm_out_p = preallocated_ssm_out[:num_prefill_tokens]

    # Prefill: reuse mamba_chunk_scan_combined (already fast)
    num_prefill, num_prefill_tokens, num_total_tokens, num_seq = _run_ssm_prefill(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        any_prefill_use_initial_states_host,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
        preallocated_ssm_out_p.unsqueeze(0),
    )

    # Decode: use our tuned Triton kernel
    num_decode = num_total_tokens - num_prefill_tokens
    decode_inputs = _prepare_ssm_decode_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx,
        num_prefill,
        num_prefill_tokens,
        num_seq,
        num_total_tokens,
        num_heads,
        head_dim,
        ssm_state_size,
    )

    if decode_inputs is not None:
        (
            slot_idx_decode,
            x_decode,
            B_decode,
            C_decode,
            dt_hp,
            dt_bias_hp,
            A_full,
            D_full,
        ) = decode_inputs

        preallocated_ssm_out_d = preallocated_ssm_out[num_prefill_tokens:num_total_tokens]
        slot_idx_decode_i32 = slot_idx_decode.to(torch.int32)

        if num_decode <= _FLASHINFER_DECODE_THRESHOLD:
            # Small batch: flashinfer CUDA kernel is faster (better for TPOT at c1)
            y_decode = flashinfer.mamba.selective_state_update(
                ssm_state_cache,
                x_decode,
                dt_hp,
                A_full,
                B_decode,
                C_decode,
                D=D_full,
                z=None,
                dt_bias=dt_bias_hp,
                dt_softplus=True,
                state_batch_indices=slot_idx_decode_i32,
            )
            preallocated_ssm_out_d.copy_(y_decode)
        else:
            # Large batch: tuned Triton kernel is faster (better for TTFT at c256)
            tuned_selective_state_update(
                ssm_state_cache,
                x_decode,
                dt_hp,
                A_full,
                B_decode,
                C_decode,
                D=D_full,
                dt_bias=dt_bias_hp,
                dt_softplus=True,
                state_batch_indices=slot_idx_decode_i32,
                out=preallocated_ssm_out_d,
            )

    if num_total_tokens > 0:
        if preallocated_ssm_out.dtype != hidden_states.dtype:
            preallocated_ssm_out = preallocated_ssm_out.to(hidden_states.dtype)
        return preallocated_ssm_out.view(b, s, num_heads, head_dim)
    else:
        return torch.zeros_like(hidden_states)


@_tuned_cached_ssm.register_fake
def _tuned_cached_ssm_fake(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
):
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


@AttentionRegistry.register("tuned_ssm")
class TunedBackendSSM(BaseBackendSSM):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.tuned_cached_ssm.default
