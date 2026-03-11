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

"""Cached Mamba v1 backend for AutoDeploy.

Mamba v1 (as used in Jamba) has a per-element state matrix A: [D_inner, d_state],
unlike Mamba v2/SSD which uses a scalar A per head. This backend provides:

1. A cached custom op that manages SSM state for autoregressive generation.
2. An AttentionDescriptor that integrates with the AD cache insertion framework.

The SSM state has shape [batch, D_inner, d_state] and is stored in the
standard SSMResourceHandler cache as [max_batch_size, D_inner, 1, d_state]
(with num_heads=D_inner, head_dim=1).
"""

from typing import List

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    ResourceHandlerDict,
    SSMResourceHandler,
)
from .torch_mamba_v1 import _torch_mamba_v1_selective_scan  # noqa: F401 — ensures op registered


def _mamba_v1_prefill(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
):
    """Run Mamba v1 sequential scan and return (output, final_ssm_state).

    Args:
        hidden_states: [batch, seq_len, D_inner]
        A: [D_inner, d_state] (already negated)
        B: [batch, seq_len, d_state]
        C: [batch, seq_len, d_state]
        D: [D_inner]
        dt: [batch, seq_len, D_inner] (post-softplus)

    Returns:
        output: [batch, seq_len, D_inner]
        ssm_state: [batch, D_inner, d_state] — final state after processing all tokens
    """
    batch_size, seq_len, d_inner = hidden_states.shape
    dtype = hidden_states.dtype
    d_state = B.shape[-1]

    hidden_states_f = hidden_states.float()
    B_f = B.float()
    C_f = C.float()
    dt_f = dt.float()
    A_f = A.float()

    # Discretize
    discrete_A = torch.exp(A_f.unsqueeze(0).unsqueeze(0) * dt_f.unsqueeze(-1))
    discrete_B = dt_f.unsqueeze(-1) * B_f.unsqueeze(2)
    deltaB_u = discrete_B * hidden_states_f.unsqueeze(-1)

    # Sequential scan
    ssm_state = torch.zeros(
        batch_size, d_inner, d_state, device=hidden_states.device, dtype=torch.float32
    )
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, i, :, :] * ssm_state + deltaB_u[:, i, :, :]
        scan_output_i = torch.sum(ssm_state * C_f[:, i, :].unsqueeze(1), dim=-1)
        scan_outputs.append(scan_output_i)
    scan_output = torch.stack(scan_outputs, dim=1)

    # Skip connection
    scan_output = scan_output + hidden_states_f * D.float().unsqueeze(0).unsqueeze(0)

    return scan_output.to(dtype), ssm_state


def _mamba_v1_decode(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    ssm_state: torch.Tensor,
):
    """Single-step Mamba v1 decode from cached state.

    Args:
        hidden_states: [batch, D_inner] — single token
        A: [D_inner, d_state]
        B: [batch, d_state]
        C: [batch, d_state]
        D: [D_inner]
        dt: [batch, D_inner] — single timestep (post-softplus)
        ssm_state: [batch, D_inner, d_state] — cached SSM state

    Returns:
        output: [batch, D_inner]
        updated_ssm_state: [batch, D_inner, d_state]
    """
    A_f = A.float()
    dt_f = dt.float()
    x_f = hidden_states.float()
    B_f = B.float()
    C_f = C.float()

    # Discretize: dA = exp(A * dt), where A: [D, N], dt: [B, D]
    dA = torch.exp(A_f.unsqueeze(0) * dt_f.unsqueeze(-1))  # [B, D, N]
    dB = dt_f.unsqueeze(-1) * B_f.unsqueeze(1)  # [B, D, N]

    # Update state
    updated_state = dA * ssm_state.float() + dB * x_f.unsqueeze(-1)  # [B, D, N]

    # Output: y = sum(state * C, dim=-1) + D * x
    y = torch.sum(updated_state * C_f.unsqueeze(1), dim=-1)  # [B, D]
    y = y + x_f * D.float().unsqueeze(0)

    return y.to(hidden_states.dtype), updated_state


# ---------------------------------------------------------------
# Cached op integrating with the AD attention interface
# ---------------------------------------------------------------


@torch.library.custom_op("auto_deploy::torch_cached_mamba_v1", mutates_args={"ssm_state_cache"})
def _torch_cached_mamba_v1(
    # INPUTS
    hidden_states: torch.Tensor,  # [b, s, D_inner]
    A: torch.Tensor,  # [D_inner, d_state]
    B: torch.Tensor,  # [b, s, d_state]
    C: torch.Tensor,  # [b, s, d_state]
    D: torch.Tensor,  # [D_inner]
    dt: torch.Tensor,  # [b, s, D_inner]
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # CACHES — shape [max_batch_size, D_inner, 1, d_state] from SSMResourceHandler
    ssm_state_cache: torch.Tensor,
) -> torch.Tensor:
    """Slot-indexed cached Mamba v1 selective scan.

    Handles both prefill (sequential scan saving final state) and
    decode (single-step update from cached state).
    """
    b, s = hidden_states.shape[:2]
    d_inner = hidden_states.shape[2]

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    seq_len_t = seq_len[:num_seq]
    slot_idx_t = slot_idx[:num_seq].to(torch.long)

    # Squeeze head_dim=1 from cache: [max_bs, D_inner, 1, d_state] -> [max_bs, D_inner, d_state]
    cache_3d = ssm_state_cache.squeeze(2)

    if s == 1:
        # === Decode-only: single token per sequence ===
        ssm_batch = cache_3d.index_select(0, slot_idx_t)  # [num_seq, D, N]
        x = hidden_states.squeeze(1)  # [b, D]
        dt_step = dt.squeeze(1)  # [b, D]
        B_step = B.squeeze(1)  # [b, d_state]
        C_step = C.squeeze(1)  # [b, d_state]

        y, updated_state = _mamba_v1_decode(x, A, B_step, C_step, D, dt_step, ssm_batch)

        # Write back to cache (unsqueeze head_dim=1 for 4D cache)
        ssm_state_cache.index_copy_(
            0, slot_idx_t, updated_state.unsqueeze(2).to(ssm_state_cache.dtype)
        )
        return y.unsqueeze(1).to(hidden_states.dtype)  # [b, 1, D]

    # === Prefill (flattened sequences) ===
    bs = b * s
    hs_flat = hidden_states.reshape(bs, d_inner)
    B_flat = B.reshape(bs, B.shape[-1])
    C_flat = C.reshape(bs, C.shape[-1])
    dt_flat = dt.reshape(bs, d_inner)

    y = torch.empty(bs, d_inner, device=hidden_states.device, dtype=hidden_states.dtype)

    seq_start = cu_seqlen[:num_seq]

    for i in range(num_seq):
        length_i = seq_len_t[i].item()
        if length_i == 0:
            continue

        start_i = seq_start[i].item()
        end_i = start_i + length_i

        hs_seq = hs_flat[start_i:end_i].unsqueeze(0)  # [1, L, D]
        B_seq = B_flat[start_i:end_i].unsqueeze(0)  # [1, L, N]
        C_seq = C_flat[start_i:end_i].unsqueeze(0)  # [1, L, N]
        dt_seq = dt_flat[start_i:end_i].unsqueeze(0)  # [1, L, D]

        out_seq, final_state = _mamba_v1_prefill(hs_seq, A, B_seq, C_seq, D, dt_seq)

        y[start_i:end_i] = out_seq[0].to(y.dtype)

        # Store final state to cache (unsqueeze head_dim=1 for 4D)
        slot_i = slot_idx_t[i].unsqueeze(0)
        ssm_state_cache.index_copy_(0, slot_i, final_state.unsqueeze(2).to(ssm_state_cache.dtype))

    return y.reshape(b, s, d_inner)


@_torch_cached_mamba_v1.register_fake
def _torch_cached_mamba_v1_fake(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    ssm_state_cache: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


# ---------------------------------------------------------------
# AttentionDescriptor registration
# ---------------------------------------------------------------


@AttentionRegistry.register("torch_mamba_v1")
class TorchBackendMambaV1(AttentionDescriptor):
    """Attention descriptor for Mamba v1 selective scan (Jamba-style)."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        # Mamba v1 uses [b, s, D_inner] — no head structure
        # But the interface requires a layout; we use "bsnd" since that's
        # what the cache insertion code expects (it just checks the string).
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # torch_mamba_v1_selective_scan(hidden_states, A, B, C, D, dt) = 6 args
        return 6

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_mamba_v1_selective_scan

    @classmethod
    def get_cached_attention_op(cls):
        return torch.ops.auto_deploy.torch_cached_mamba_v1.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        # A: [D_inner, d_state] — args[1] is the A parameter, always has meta["val"]
        A_fake: torch.Tensor = source_attn_node.args[1].meta["val"]
        d_inner, d_state = A_fake.shape

        # Infer dtype from hidden_states if available, else from A
        hs_node = source_attn_node.args[0]
        hs_dtype = hs_node.meta["val"].dtype if "val" in hs_node.meta else A_fake.dtype

        # Map to SSMResourceHandler: num_heads=D_inner, head_dim=1, d_state=d_state
        ssm_state_dtype = cls.resolve_cache_dtype(cache_config.mamba_ssm_cache_dtype, hs_dtype)

        return {
            "ssm_state_cache": SSMResourceHandler(
                num_heads=d_inner,
                head_dim=1,
                d_state=d_state,
                dtype=ssm_state_dtype,
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        # No constants for Mamba v1 (unlike v2 which has time_step_limit, chunk_size)
        return []
