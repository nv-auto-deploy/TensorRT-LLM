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

"""Fused conv1d + SSM custom op for Mamba decode.

Replaces the separate cuda_cached_causal_conv1d + tuned_cached_ssm chain with
a single Triton kernel for decode tokens, eliminating the intermediate HBM
write/read of the [batch, conv_dim] conv output tensor.

Prefill tokens fall back to the existing CUDA ops (causal_conv1d_fn +
mamba_chunk_scan_combined) which are already optimal for long sequences.
"""

from typing import List, Optional

import torch

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn

from ..attention_interface import BatchInfo
from .fused_mamba_decode import fused_conv_ssm_decode
from .mamba_backend_common import _run_ssm_prefill


@torch.library.custom_op(
    "auto_deploy::fused_cached_conv_ssm",
    mutates_args={"conv_state_cache", "ssm_state_cache"},
)
def _fused_cached_conv_ssm(
    # Conv inputs
    conv_input: torch.Tensor,  # [b, s, conv_dim]
    weight: torch.Tensor,  # [conv_dim, kernel_width]
    bias: Optional[torch.Tensor],  # [conv_dim]
    # SSM weights
    A: torch.Tensor,  # [nheads]
    D: torch.Tensor,  # [nheads]
    dt: torch.Tensor,  # [b, s, nheads]
    dt_bias: torch.Tensor,  # [nheads]
    # Standard metadata
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    any_prefill_use_initial_states_host: torch.Tensor,
    # Extra metadata
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    seq_idx_prefill: torch.Tensor,
    # Caches (mutated in-place)
    conv_state_cache: torch.Tensor,  # [max_batch, conv_dim, kernel_width-1]
    ssm_state_cache: torch.Tensor,  # [max_batch, nheads, dim, dstate]
    # Constants
    time_step_limit: List[float],
    chunk_size: int,
    intermediate_size: int,  # nheads * dim
    ngroups: int,
) -> torch.Tensor:  # [b, s, nheads, dim]
    """Fused conv1d_update + SiLU + SSM_update.

    For decode tokens: single Triton kernel (fused_conv_ssm_decode).
    For prefill tokens: causal_conv1d_fn + mamba_chunk_scan_combined.
    """
    b, s, conv_dim = conv_input.shape
    nheads = ssm_state_cache.shape[1]
    dim = ssm_state_cache.shape[2]
    dstate = ssm_state_cache.shape[3]
    bs = b * s

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    inp_flat = conv_input.reshape(bs, conv_dim)  # [bs, conv_dim]
    dt_flat = dt.reshape(bs, nheads)  # [bs, nheads]

    # Normalize weight to 2D [conv_dim, kernel_width].
    # The weight may arrive as 3D [conv_dim, 1, kernel_width] (depthwise Conv1d layout)
    # from the graph; causal_conv1d_fn and the Triton decode kernel both expect 2D.
    if weight.ndim == 3:
        assert weight.shape[-2] == 1, (
            f"Expected depthwise weight with shape[-2]==1, got {weight.shape}"
        )
        weight = weight.squeeze(-2)  # [conv_dim, kernel_width]

    # Preallocate output
    ssm_out_flat = torch.zeros(bs, nheads, dim, dtype=conv_input.dtype, device=conv_input.device)

    # ----------------------------------------------------------------
    # PREFILL: CUDA conv + mamba_chunk_scan_combined (existing fast path)
    # ----------------------------------------------------------------
    if num_prefill > 0:
        w2d = weight  # [conv_dim, kernel_width]
        x_varlen = inp_flat[:num_prefill_tokens].T.contiguous()  # [conv_dim, prefill_tokens]
        y_varlen = causal_conv1d_fn(
            x_varlen,
            w2d,
            bias,
            query_start_loc=cu_seqlen[: num_prefill + 1],
            cache_indices=slot_idx[:num_prefill].to(torch.int32),
            has_initial_state=use_initial_states[:num_prefill],
            conv_states=conv_state_cache,
            activation=None,  # SiLU applied inline below (avoids large-batch kernel regression)
            pad_slot_id=PAD_SLOT_ID,
        )  # [conv_dim, prefill_tokens]
        inp_flat[:num_prefill_tokens] = y_varlen.T

        # Apply SiLU to ALL channels (x, B, C) to match the original model, which applies
        # self.act (SiLU) to the entire conv output before splitting into x/B/C.
        torch.nn.functional.silu(inp_flat[:num_prefill_tokens], inplace=True)

        # Split conv output for SSM
        x_p = inp_flat[:num_prefill_tokens, :intermediate_size]
        B_p = inp_flat[
            :num_prefill_tokens, intermediate_size : intermediate_size + ngroups * dstate
        ]
        C_p = inp_flat[
            :num_prefill_tokens,
            intermediate_size + ngroups * dstate : intermediate_size + 2 * ngroups * dstate,
        ]

        # Reshape for SSM prefill: [1, prefill_tokens, nheads, dim] etc.
        hs_flat = x_p.view(num_prefill_tokens, nheads, dim)
        B_flat = B_p.view(num_prefill_tokens, ngroups, dstate)
        C_flat = C_p.view(num_prefill_tokens, ngroups, dstate)
        dt_p = dt_flat[:num_prefill_tokens]

        _run_ssm_prefill(
            hs_flat,
            B_flat,
            C_flat,
            dt_p,
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
            out=ssm_out_flat[:num_prefill_tokens].unsqueeze(0),
        )

    # ----------------------------------------------------------------
    # DECODE: fused Triton kernel (single kernel, no intermediate HBM)
    # ----------------------------------------------------------------
    if num_decode > 0:
        conv_in_dec = inp_flat[num_prefill_tokens:num_total_tokens]  # [nd, conv_dim]
        dt_dec = dt_flat[num_prefill_tokens:num_total_tokens]  # [nd, nheads]
        slot_idx_dec = slot_idx[num_prefill:num_seq]
        dt_clamp_min = float(time_step_limit[0]) if time_step_limit else 0.0
        dt_clamp_max = float(time_step_limit[1]) if time_step_limit else float("inf")

        fused_conv_ssm_decode(
            conv_in_dec,
            conv_state_cache,
            weight,
            bias,
            dt_dec,
            dt_bias,
            A,
            D,
            ssm_state_cache,
            slot_idx_dec,
            slot_idx_dec,
            ssm_out_flat[num_prefill_tokens:num_total_tokens],
            dt_clamp_min=dt_clamp_min,
            dt_clamp_max=dt_clamp_max,
        )

    if num_total_tokens > 0:
        return ssm_out_flat[:num_total_tokens].view(b, s, nheads, dim)
    else:
        return torch.zeros(b, s, nheads, dim, dtype=conv_input.dtype, device=conv_input.device)


@_fused_cached_conv_ssm.register_fake
def _fused_cached_conv_ssm_fake(
    conv_input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    A: torch.Tensor,
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
    conv_state_cache: torch.Tensor,
    ssm_state_cache: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
    intermediate_size: int,
    ngroups: int,
):
    b, s, _ = conv_input.shape
    nheads = ssm_state_cache.shape[1]
    dim = ssm_state_cache.shape[2]
    return torch.empty(b, s, nheads, dim, dtype=conv_input.dtype, device=conv_input.device)
