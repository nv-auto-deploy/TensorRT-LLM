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

"""Cached attention op for the gated delta rule using FlashInfer CuTe DSL decode kernel.

Uses the CuTe DSL GDN decode kernel (gdn_decode.py) for decode and the FLA Triton
chunked kernel for prefill. Requires SM90+ (Hopper) GPU architecture.

Gated Delta Rule reference: https://arxiv.org/abs/2412.06464
CuTe DSL kernel from FlashInfer PR #2370.
"""

from typing import List

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ....modules.fla.chunk import chunk_gated_delta_rule
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    StateResourceHandler,
)
from .torch_gated_delta_rule import compute_g_beta


@torch.library.custom_op(
    "auto_deploy::flashinfer_cached_gated_delta_rule", mutates_args=("delta_cache",)
)
def flashinfer_cached_gated_delta_rule(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    b_size, s, num_heads, _ = q.shape

    # Derive g and beta from raw parameters
    g, beta = compute_g_beta(a, b, A_log, dt_bias)

    # flatten batch and sequence dims
    q_flat = q.view(b_size * s, num_heads, -1)
    k_flat = k.view(b_size * s, num_heads, -1)
    v_flat = v.view(b_size * s, num_heads, -1)
    g_flat = g.view(b_size * s, num_heads)
    beta_flat = beta.view(b_size * s, num_heads)

    # pre-allocate output
    y = torch.empty_like(v, memory_format=torch.contiguous_format)
    y_flat = y.view(b_size * s, num_heads, -1)

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # clean up metadata
    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    if num_prefill > 0:
        initial_states = None
        if torch.any(use_initial_states[:num_prefill]):
            initial_states = torch.where(
                use_initial_states[:num_prefill, None, None, None],
                delta_cache[slot_idx[:num_prefill]],
                0,
            )

        y_prefill, final_state = chunk_gated_delta_rule(
            q=q_flat[None, :num_prefill_tokens],
            k=k_flat[None, :num_prefill_tokens],
            v=v_flat[None, :num_prefill_tokens],
            g=g_flat[None, :num_prefill_tokens],
            beta=beta_flat[None, :num_prefill_tokens],
            scale=scale,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlen_prefill,
        )

        y_flat[None, :num_prefill_tokens] = y_prefill.to(y_flat.dtype)
        delta_cache.index_copy_(0, slot_idx[:num_prefill], final_state.to(delta_cache.dtype))

        del y_prefill, initial_states, final_state

    if num_decode > 0:
        from .gdn_decode import gated_delta_rule_decode_pooled

        decode_start = num_prefill_tokens
        decode_end = num_prefill_tokens + num_decode
        decode_slots = slot_idx[num_prefill:num_seq]

        q_decode = q_flat[decode_start:decode_end].unsqueeze(1)  # [D, 1, H, K]
        k_decode = k_flat[decode_start:decode_end].unsqueeze(1)  # [D, 1, H, K]
        v_decode = v_flat[decode_start:decode_end].unsqueeze(1)  # [D, 1, H, V]
        a_decode = a.view(b_size * s, num_heads)[decode_start:decode_end].unsqueeze(1)
        b_decode = b.view(b_size * s, num_heads)[decode_start:decode_end].unsqueeze(1)

        # Pool-indexed decode: pass delta_cache + slot indices directly.
        # The kernel reads/writes state in-place at indexed positions.
        # No gather, no scatter, no .contiguous(), no state.copy_().
        y_decode = gated_delta_rule_decode_pooled(
            q=q_decode,
            k=k_decode,
            v=v_decode,
            state_pool=delta_cache,
            state_indices=decode_slots.to(torch.int32),
            A_log=A_log.detach().float(),
            a=a_decode,
            dt_bias=dt_bias.detach(),
            b=b_decode,
            scale=scale,
            use_qk_l2norm=False,
        )

        y_flat[decode_start:decode_end] = y_decode.squeeze(1).to(y_flat.dtype)

        del y_decode

    return y


@flashinfer_cached_gated_delta_rule.register_fake
def flashinfer_cached_gated_delta_rule_fake(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    delta_cache: torch.Tensor,  # [max_batch_size, H, K, V]
    # CONSTANTS
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(v)


@AttentionRegistry.register("flashinfer_gated_delta")
class FlashinferGatedDeltaBackend(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q, k, v, a, b, A_log, dt_bias
        return 7

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_gated_delta_rule

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.flashinfer_cached_gated_delta_rule.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        key_node = source_attn_node.args[1]
        value_node = source_attn_node.args[2]
        num_heads = key_node.meta["val"].shape[-2]
        key_dim = key_node.meta["val"].shape[-1]
        value_dim = value_node.meta["val"].shape[-1]
        key_dtype = key_node.meta["val"].dtype

        return {
            "delta_cache": StateResourceHandler(
                num_heads,
                key_dim,
                value_dim,
                # NOTE: not configurable at the moment, using auto to match the key dtype
                dtype=cls.resolve_cache_dtype("auto", key_dtype),
            )
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        scale = extract_op_args(source_attn_node, "scale")[0]
        if scale is None:
            key_node = source_attn_node.args[1]
            key_dim = key_node.meta["val"].shape[-1]
            scale = key_dim**-0.5
        return [scale]
