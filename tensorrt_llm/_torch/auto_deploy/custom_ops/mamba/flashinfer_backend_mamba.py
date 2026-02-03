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

from typing import List

import torch
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import AttentionRegistry, MHACallable, ResourceHandlerDict
from .mamba_backend_common import (
    BaseBackendSSM,
    _flatten_ssm_inputs,
    _prepare_ssm_decode_inputs,
    _run_ssm_prefill,
)


@torch.library.custom_op("auto_deploy::flashinfer_cached_ssm", mutates_args={})
def _flashinfer_cached_ssm(
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
    b, s, num_heads, head_dim, bs, hs_flat, B_flat, C_flat, dt_flat = _flatten_ssm_inputs(
        hidden_states, B, C, dt
    )
    ssm_state_size = B.shape[3]
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode
    # Preallocate output tensor to avoid memcpy cost for merging prefill
    # and decode outputs
    preallocated_ssm_out = torch.empty(
        [bs, num_heads, head_dim],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    preallocated_ssm_out_p = preallocated_ssm_out[:num_prefill_tokens]

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
        chunk_indices,
        chunk_offsets,
        seq_idx_prefill,
        ssm_state_cache,
        time_step_limit,
        chunk_size,
        preallocated_ssm_out_p.unsqueeze(0),
    )

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

    y_decode = None
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

        import flashinfer

        slot_idx_decode_i32 = slot_idx_decode.to(torch.int32)

        # FlashInfer's selective_state_update has VERY SPECIFIC stride requirements:
        # - x: [batch, nheads, dim] contiguous (stride(1)=dim, stride(2)=1)
        # - dt: [batch, nheads, dim] BROADCASTED (stride(1)=1, stride(2)=0)
        # - dt_bias: [nheads, dim] BROADCASTED (stride(0)=1, stride(1)=0)
        # - A: [nheads, dim, dstate] BROADCASTED (stride(1)=0, stride(2)=0)
        # - D: [nheads, dim] BROADCASTED (stride(0)=1, stride(1)=0)
        # - B, C: [batch, ngroups, dstate] contiguous
        #
        # The _prepare_ssm_decode_inputs creates expanded views with expand(), which have
        # the correct stride=0 pattern. However, we need to cast dtypes, and .to(dtype)
        # on an expanded tensor creates a contiguous copy, breaking the stride pattern.
        #
        # Solution: Extract base tensors, cast to correct dtype, then re-expand.

        # x, B, C are contiguous - just cast to bfloat16
        x_decode_bf16 = x_decode.to(torch.bfloat16)
        B_decode_bf16 = B_decode.to(torch.bfloat16)
        C_decode_bf16 = C_decode.to(torch.bfloat16)

        # dt: extract base [nd, nheads], cast, then re-expand with stride(2)=0
        # dt_hp has shape [nd, num_heads, head_dim] from expand with stride pattern [nheads, 1, 0]
        dt_base = dt_hp[:, :, 0]  # [nd, num_heads] - extract base (all values same along dim 2)
        dt_base_bf16 = dt_base.to(torch.bfloat16)
        dt_bf16 = dt_base_bf16.unsqueeze(-1).expand(
            -1, -1, head_dim
        )  # [nd, nheads, head_dim] stride(2)=0

        # dt_bias: extract base [nheads], cast, then re-expand with stride(1)=0
        dt_bias_base = dt_bias_hp[:, 0]  # [num_heads]
        dt_bias_base_bf16 = dt_bias_base.to(torch.bfloat16)
        dt_bias_bf16 = dt_bias_base_bf16.unsqueeze(-1).expand(
            -1, head_dim
        )  # [nheads, head_dim] stride(1)=0

        # A: extract base [nheads], cast to float32, then re-expand with stride(1)=0, stride(2)=0
        # A_full has shape [nheads, head_dim, ssm_state_size] from expand
        A_base = A_full[:, 0, 0]  # [num_heads]
        A_base_fp32 = A_base.to(torch.float32)
        A_fp32 = A_base_fp32.unsqueeze(-1).unsqueeze(-1).expand(-1, head_dim, ssm_state_size)

        # D: extract base [nheads], cast, then re-expand with stride(1)=0
        D_base = D_full[:, 0]  # [num_heads]
        D_base_bf16 = D_base.to(torch.bfloat16)
        D_bf16 = D_base_bf16.unsqueeze(-1).expand(-1, head_dim)  # [nheads, head_dim] stride(1)=0

        y_decode = flashinfer.mamba.selective_state_update(
            ssm_state_cache,
            x_decode_bf16,
            dt_bf16,
            A_fp32,
            B_decode_bf16,
            C_decode_bf16,
            D=D_bf16,
            z=None,
            dt_bias=dt_bias_bf16,
            dt_softplus=True,
            state_batch_indices=slot_idx_decode_i32,
        )
        preallocated_ssm_out[num_prefill_tokens:num_total_tokens].copy_(y_decode)
    if num_total_tokens > 0:
        return (
            preallocated_ssm_out[:num_total_tokens]
            .view(b, s, num_heads, head_dim)
            .to(hidden_states.dtype)
        )
    else:
        return torch.empty_like(hidden_states)


@_flashinfer_cached_ssm.register_fake
def _flashinfer_cached_ssm_fake(
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
):
    # Return a correctly-shaped tensor for tracing with fake tensors
    return torch.empty_like(
        hidden_states,
        memory_format=torch.contiguous_format,
        dtype=hidden_states.dtype,
    )


# Flashinfer's selective_state_update kernel only supports these head dimensions
FLASHINFER_SUPPORTED_HEAD_DIMS = [64, 128]


@AttentionRegistry.register("flashinfer_ssm")
class FlashinferBackendSSM(BaseBackendSSM):
    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_cached_ssm.default

    # flashinfer's selective_state_update only supports these state dtypes
    FLASHINFER_SUPPORTED_STATE_DTYPES = {torch.bfloat16, torch.float16, torch.float32}

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        ret = super().get_cache_initializers(source_attn_node, cache_config)

        # check head_dim is supported by flashinfer
        if ret["ssm_state_cache"].head_dim not in FLASHINFER_SUPPORTED_HEAD_DIMS:
            raise ValueError(
                f"flashinfer_ssm only supports head_dim in {FLASHINFER_SUPPORTED_HEAD_DIMS}. "
                f"Got head_dim={ret['ssm_state_cache'].head_dim}. "
                "Consider using 'triton_ssm' backend instead."
            )

        return ret
