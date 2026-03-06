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

from typing import List, Optional

import torch

from tensorrt_llm._torch.modules.mamba.selective_state_update import selective_state_update

from .. import attention_interface
from ..attention_interface import AttentionRegistry, BatchInfo, MHACallable
from .mamba_backend_common import (
    BaseBackendSSM,
    _flatten_ssm_inputs,
    _prepare_ssm_decode_inputs,
    _prepare_ssm_extend_inputs,
    _run_ssm_prefill,
)


def _triton_cached_ssm_impl(
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
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
    intermediate_ssm_state_cache: Optional[
        torch.Tensor
    ] = None,  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]
) -> torch.Tensor:
    b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat = _flatten_ssm_inputs(
        hidden_states, B, C, dt
    )
    ssm_state_size = B.shape[3]
    # Preallocate output tensor (zeros so padding positions are clean)
    preallocated_ssm_out = torch.zeros(
        [bs, num_heads, head_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_extend, num_decode = batch_info.get_num_sequences()
    num_prefill_tokens, num_extend_tokens, num_decode_tokens = batch_info.get_num_tokens()
    num_total_tokens = num_prefill_tokens + num_extend_tokens + num_decode_tokens

    preallocated_ssm_out_p = preallocated_ssm_out[:num_prefill_tokens]
    preallocated_ssm_out_e = preallocated_ssm_out[
        num_prefill_tokens : num_prefill_tokens + num_extend_tokens
    ]
    preallocated_ssm_out_d = preallocated_ssm_out[
        num_prefill_tokens + num_extend_tokens : num_total_tokens
    ]

    _run_ssm_prefill(
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

    # EXTEND: use verify/decode-style state-update kernel with intermediate-write semantics.
    extend_inputs = _prepare_ssm_extend_inputs(
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
        num_extend,
        num_extend_tokens,
        num_heads,
        head_dim,
        ssm_state_size,
    )
    if extend_inputs is not None:
        if intermediate_ssm_state_cache is None:
            raise RuntimeError(
                "triton_cached_ssm requires intermediate_ssm_state_cache "
                "when extend tokens are present"
            )

        (
            slot_idx_extend,
            x_extend,
            B_extend,
            C_extend,
            dt_extend,
            A_full,
            D_full,
            dt_bias_hp,
        ) = extend_inputs

        intermediate_state_indices = torch.arange(
            num_extend, dtype=torch.int32, device=slot_idx_extend.device
        )
        tokens_per_extend = num_extend_tokens // num_extend
        preallocated_ssm_out_e = preallocated_ssm_out_e.view(
            num_extend, tokens_per_extend, num_heads, head_dim
        )
        selective_state_update(
            ssm_state_cache,
            x_extend,
            dt_extend,
            A_full,
            B_extend,
            C_extend,
            D=D_full,
            z=None,
            dt_bias=dt_bias_hp,
            dt_softplus=True,
            state_batch_indices=slot_idx_extend,
            out=preallocated_ssm_out_e,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_ssm_state_cache,
            cache_steps=tokens_per_extend,
            intermediate_state_indices=intermediate_state_indices,
        )

    # DECODE
    decode_inputs = _prepare_ssm_decode_inputs(
        hs_flat,
        B_flat,
        C_flat,
        dt_flat,
        A,
        D,
        dt_bias,
        slot_idx,
        num_prefill + num_extend,
        num_prefill_tokens + num_extend_tokens,
        num_decode,
        num_decode_tokens,
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
        selective_state_update(
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
            state_batch_indices=slot_idx_decode,
            out=preallocated_ssm_out_d,
        )
    if num_total_tokens > 0:
        # Cast to input dtype if needed (prefill may compute in higher precision)
        if preallocated_ssm_out.dtype != hidden_states.dtype:
            preallocated_ssm_out = preallocated_ssm_out.to(hidden_states.dtype)
        return preallocated_ssm_out.view(b, s, num_heads, head_dim)
    else:
        return torch.zeros_like(hidden_states)


@torch.library.custom_op("auto_deploy::triton_cached_ssm", mutates_args={})
def _triton_cached_ssm(
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
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    return _triton_cached_ssm_impl(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
    )


@_triton_cached_ssm.register_fake
def _triton_cached_ssm_fake(
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
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
):
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


@torch.library.custom_op("auto_deploy::triton_cached_ssm_spec", mutates_args={})
def _triton_cached_ssm_spec(
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
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    intermediate_ssm_state_cache: torch.Tensor,  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
) -> torch.Tensor:
    return _triton_cached_ssm_impl(
        hidden_states,
        A,
        B,
        C,
        D,
        dt,
        dt_bias,
        batch_info_host,
        cu_seqlen,
        slot_idx,
        use_initial_states,
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
        intermediate_ssm_state_cache=intermediate_ssm_state_cache,
    )


@_triton_cached_ssm_spec.register_fake
def _triton_cached_ssm_spec_fake(
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
    # EXTRA METADATA
    chunk_indices: torch.Tensor,  # [num_logical_chunks]
    chunk_offsets: torch.Tensor,  # [num_logical_chunks]
    seq_idx_prefill: torch.Tensor,  # [1, num_prefill_tokens]
    # CACHES
    ssm_state_cache: torch.Tensor,  # [max_batch_size, num_heads, head_dim, ssm_state_size]
    intermediate_ssm_state_cache: torch.Tensor,  # [spec_state_size, max_draft_len+1, num_heads, head_dim, d_state]
    # CONSTANTS
    time_step_limit: List[float],
    chunk_size: int,
):
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


@AttentionRegistry.register("triton_ssm")
class TritonBackendSSM(BaseBackendSSM):
    @classmethod
    def get_cached_attention_op(cls, spec_config=None) -> MHACallable:
        if spec_config is not None:
            return torch.ops.auto_deploy.triton_cached_ssm_spec.default
        return torch.ops.auto_deploy.triton_cached_ssm.default

    @classmethod
    def get_cache_initializers(cls, source_attn_node, cache_config, spec_config=None):
        cache_initializers = super().get_cache_initializers(source_attn_node, cache_config)
        if spec_config is None or spec_config.max_draft_len is None:
            return cache_initializers

        base_handler = cache_initializers["ssm_state_cache"]
        cache_initializers["intermediate_ssm_state_cache"] = (
            attention_interface.SpecSSMResourceHandler(
                num_heads=base_handler.num_heads,
                head_dim=base_handler.head_dim,
                d_state=base_handler.d_state,
                dtype=base_handler.dtype,
                cache_steps=spec_config.max_draft_len + 1,
            )
        )
        return cache_initializers
