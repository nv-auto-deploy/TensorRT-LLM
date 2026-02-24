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

"""Cached attention op for the gated delta rule using fla kernels.

Gated Delta Rule is based on this paper: https://arxiv.org/abs/2412.06464

Kernels are based on this repo: https://github.com/fla-org/flash-linear-attention
"""

import os
from typing import List

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ....modules.fla.chunk import chunk_gated_delta_rule
from ....modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update_fwd
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

# ---------------------------------------------------------------------------
# Debug guard: set TRTLLM_FLA_DEBUG=1 to enable cache-isolation checks.
# Set TRTLLM_FLA_DUMP_DIR=<path> to also dump cache snapshots to disk.
# ---------------------------------------------------------------------------
_FLA_DEBUG = os.environ.get("TRTLLM_FLA_DEBUG", "0") == "1"
_FLA_DUMP_DIR = os.environ.get("TRTLLM_FLA_DUMP_DIR", "")
_fla_debug_step_counter = [0]  # mutable container for step tracking


def _fla_debug_pre_check(delta_cache, slot_idx, num_seq, batch_info_host, q):
    """Pre-op debug checks: metadata sanity + cache snapshot for post-check."""
    if torch.cuda.is_current_stream_capturing():
        return None
    max_slots = delta_cache.shape[0]
    active_slots = slot_idx[:num_seq].long()

    # Slot index bounds
    assert active_slots.numel() == 0 or active_slots.max().item() < max_slots, (
        f"[FLA DEBUG] slot_idx out of bounds: max={active_slots.max().item()}, "
        f"cache_slots={max_slots}"
    )
    assert active_slots.numel() == 0 or active_slots.min().item() >= 0, (
        f"[FLA DEBUG] negative slot_idx: {active_slots.min().item()}"
    )

    # batch_info_host consistency
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    total_tokens = q.shape[0] * q.shape[1]
    assert num_prefill_tokens + num_decode == total_tokens, (
        f"[FLA DEBUG] batch_info_host inconsistent: "
        f"{num_prefill_tokens} prefill_tokens + {num_decode} decode != {total_tokens} total. "
        f"batch_info_host={batch_info_host.tolist()}"
    )
    assert num_prefill + num_decode == num_seq, (
        f"[FLA DEBUG] num_prefill({num_prefill}) + num_decode({num_decode}) != num_seq({num_seq})"
    )

    # Duplicate slot indices (two sequences sharing one cache slot)
    if active_slots.numel() > 1:
        unique_slots = active_slots.unique()
        assert unique_slots.numel() == active_slots.numel(), (
            f"[FLA DEBUG] DUPLICATE slot_idx detected! "
            f"slots={active_slots.tolist()}, unique={unique_slots.tolist()}"
        )

    return delta_cache.clone()


def _fla_debug_post_check(delta_cache, cache_before, slot_idx, num_seq):
    """Post-op debug checks: slot isolation + NaN/Inf detection."""
    if cache_before is None:
        return
    step = _fla_debug_step_counter[0]
    _fla_debug_step_counter[0] += 1

    active_slots = slot_idx[:num_seq].long()
    max_slots = delta_cache.shape[0]

    # Build mask of slots that SHOULD have changed
    expected_modified = torch.zeros(max_slots, dtype=torch.bool, device=delta_cache.device)
    if active_slots.numel() > 0:
        expected_modified[active_slots] = True

    # Check which slots actually changed
    diff = (
        (delta_cache.float() - cache_before.float())
        .abs()
        .amax(dim=tuple(range(1, delta_cache.ndim)))
    )
    actually_modified = diff > 0

    # Slots that changed but shouldn't have
    unexpected = actually_modified & ~expected_modified
    if unexpected.any():
        bad_slots = unexpected.nonzero(as_tuple=True)[0].tolist()
        bad_diffs = diff[unexpected].tolist()
        raise RuntimeError(
            f"[FLA DEBUG step={step}] CACHE CONTAMINATION: "
            f"slots {bad_slots} were modified but are NOT in active "
            f"slot_idx={active_slots.tolist()}! Max diffs: {bad_diffs}"
        )

    # NaN/Inf in active slots
    if active_slots.numel() > 0:
        active_cache = delta_cache[active_slots]
        if torch.isnan(active_cache).any():
            nan_slots = [
                active_slots[i].item()
                for i in range(active_slots.numel())
                if torch.isnan(delta_cache[active_slots[i]]).any()
            ]
            raise RuntimeError(f"[FLA DEBUG step={step}] NaN in cache slots {nan_slots}")
        if torch.isinf(active_cache).any():
            inf_slots = [
                active_slots[i].item()
                for i in range(active_slots.numel())
                if torch.isinf(delta_cache[active_slots[i]]).any()
            ]
            raise RuntimeError(f"[FLA DEBUG step={step}] Inf in cache slots {inf_slots}")

    # Optional dump
    if _FLA_DUMP_DIR:
        os.makedirs(_FLA_DUMP_DIR, exist_ok=True)
        dump_path = os.path.join(_FLA_DUMP_DIR, f"fla_step_{step:06d}.safetensors")
        try:
            from safetensors import torch as safetensors_torch

            safetensors_torch.save_file(
                {
                    "cache_before": cache_before.cpu(),
                    "cache_after": delta_cache.cpu(),
                    "slot_idx": slot_idx[:num_seq].cpu().int(),
                },
                dump_path,
            )
        except ImportError:
            torch.save(
                {
                    "cache_before": cache_before.cpu(),
                    "cache_after": delta_cache.cpu(),
                    "slot_idx": slot_idx[:num_seq].cpu(),
                },
                dump_path.replace(".safetensors", ".pt"),
            )


@torch.library.custom_op("auto_deploy::fla_cached_gated_delta_rule", mutates_args=("delta_cache",))
def fla_cached_gated_delta_rule(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
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
    b, s, num_heads, _ = q.shape

    # flatten batch and sequence dims
    q_flat = q.view(b * s, num_heads, -1)
    k_flat = k.view(b * s, num_heads, -1)
    v_flat = v.view(b * s, num_heads, -1)
    g_flat = g.view(b * s, num_heads)
    beta_flat = beta.view(b * s, num_heads)

    # pre-allocate output
    y = torch.zeros_like(v, memory_format=torch.contiguous_format)
    y_flat = y.view(b * s, num_heads, -1)

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode

    # clean up metadata
    cu_seqlen_prefill = cu_seqlen[: num_prefill + 1]
    slot_idx = slot_idx[:num_seq].to(torch.long)
    use_initial_states = use_initial_states[:num_seq]

    # Debug guard: pre-check
    _cache_snapshot = None
    if _FLA_DEBUG:
        _cache_snapshot = _fla_debug_pre_check(delta_cache, slot_idx, num_seq, batch_info_host, q)

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
        cu_seqlen_decode = torch.arange(0, num_decode + 1, device=q.device, dtype=torch.long)
        y_decode = fused_recurrent_gated_delta_rule_update_fwd(
            q=q_flat[None, num_prefill_tokens:].contiguous(),
            k=k_flat[None, num_prefill_tokens:].contiguous(),
            v=v_flat[None, num_prefill_tokens:].contiguous(),
            g=g_flat[None, num_prefill_tokens:].contiguous(),
            beta=beta_flat[None, num_prefill_tokens:].contiguous(),
            scale=scale,
            initial_state_source=delta_cache,
            initial_state_indices=slot_idx[num_prefill:].contiguous(),
            cu_seqlens=cu_seqlen_decode,
        )

        y_flat[None, num_prefill_tokens:] = y_decode.to(y_flat.dtype)

        del y_decode

    # Debug guard: post-check
    if _FLA_DEBUG and _cache_snapshot is not None:
        _fla_debug_post_check(delta_cache, _cache_snapshot, slot_idx, num_seq)

    return y


@fla_cached_gated_delta_rule.register_fake
def fla_cached_gated_delta_rule_fake(
    # INPUTS (dense but may be flattened across sequences)
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
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


@AttentionRegistry.register("fla_gated_delta")
class FlaGatedDeltaBackend(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # q, k, v, g, beta
        return 5

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_gated_delta_rule

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.fla_cached_gated_delta_rule.default

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

        return {
            "delta_cache": StateResourceHandler(
                num_heads,
                key_dim,
                value_dim,
                # NOTE: float32 cache to avoid bfloat16 quantization errors
                # that accumulate across autoregressive decode steps.
                dtype=torch.float32,
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
