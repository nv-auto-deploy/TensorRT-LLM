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

"""TRT-LLM attention backend for Auto-Deploy.

This module wraps TRT-LLM's optimized ``thop.attention`` kernel for use in Auto-Deploy,
following the same design pattern as the FlashInfer backend:

- Minimal module-level state (``_TrtllmPlanner``, analogous to ``_FlashInferPlanner``)
- SequenceInfo fields used directly as thop.attention metadata
- Pool pointers derived lazily from ``kv_cache.data_ptr()``
- Workspace managed as module-level state (not a ResourceHandler / graph input)
- All possible "constants" inferred from tensor shapes at runtime
"""

from typing import List, Optional

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from tensorrt_llm._utils import get_sm_version, prefer_pinned
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.quantization import QuantMode

from .....llmapi.llm_args import KvCacheConfig
from ...utils.cuda_graph import cuda_graph_state
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    KVPagedResourceHandler,
    MHACallable,
    PrepareMetadataHostCallable,
    ResourceHandlerDict,
)

# =============================================================================
# Module-level planner (analogous to _GlobalFlashInferPlanner)
# =============================================================================


class _TrtllmPlanner:
    """Minimal planner for TRT-LLM attention backend.

    Analogous to ``_FlashInferPlanner`` in the FlashInfer backend. Only stores
    data that cannot be derived from SequenceInfo or tensor shapes.

    Two main entry points:
    - ``reset()``: one-time allocation of ALL persistent buffers.
    - ``plan()``: per-forward host metadata (host_request_types, block_offsets, host_total_kv_lens).

    Pool pointer management:
    Each attention layer needs its own ``host_pool_pointers`` tensor so that CUDA graph
    replay reads the correct (layer-specific) pool base from a stable tensor address.
    Tensors are created lazily on first access per layer and never re-allocated.
    """

    def __init__(self):
        self.workspace: Optional[torch.Tensor] = None
        # Per-layer pool pointer tensors, keyed by kv_cache.data_ptr().
        # Each is [1, 2] int64 pinned, created lazily on first access per layer.
        # This ensures each layer's attention kernel in a CUDA graph is captured
        # with its own stable tensor address, avoiding the issue where a shared
        # tensor would hold only the last-set layer's pointer during graph replay.
        self._per_layer_pool_ptrs: dict = {}
        # pool_mapping: fixed [1, 2] all zeros since we always pass layer_idx=0
        # and pool_pointers already encodes the layer offset via kv_cache.data_ptr()
        self.host_pool_mapping: Optional[torch.Tensor] = None  # [1, 2] int32 pinned
        # thop-specific host metadata NOT available from SequenceInfo
        self.host_request_types: Optional[torch.Tensor] = None  # [max_batch] int32 pinned
        self.host_total_kv_lens: Optional[torch.Tensor] = None  # [2] int64 pinned
        self.host_total_kv_lens_ctx: Optional[torch.Tensor] = None  # [2] int64 pinned
        self.host_total_kv_lens_gen: Optional[torch.Tensor] = None  # [2] int64 pinned
        # thop variant of input_pos_host and seq_len_host
        # keeping a separate copy here since we sometimes have to overwrite the original values
        self.host_past_kv_lengths: Optional[torch.Tensor] = None  # [max_batch] int32 pinned
        self.host_context_lengths: Optional[torch.Tensor] = None  # [max_batch] int32 pinned
        # Persistent block_offsets buffer for CUDA graph compatibility.
        # Pre-allocated to max size so the tensor address is stable across replays.
        self.block_offsets: Optional[torch.Tensor] = None
        # GPU context_lengths for thop: context_len for prefill, 0 for decode
        self.context_lengths_cuda: Optional[torch.Tensor] = None
        # FP8 scale tensors (lazily initialized from constants on first FP8 use)
        self.kv_scale_orig_quant: Optional[torch.Tensor] = None
        self.kv_scale_quant_orig: Optional[torch.Tensor] = None

    def reset(self, device: torch.device, max_batch: int, max_blocks_per_seq: int) -> None:
        """One-time allocation of ALL persistent buffers.

        Guards against double-init. Called lazily from ``prepare_trtllm_metadata_host``
        on the first forward pass after cache initialization.
        """
        if self.workspace is not None:
            return  # already initialized

        # Workspace: pre-allocate a modest initial buffer (like flashinfer's 320MB).
        # thop.attention auto-resizes via resize_() if more space is needed during warm-up.
        self.workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.host_pool_mapping = torch.zeros(
            1, 2, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_total_kv_lens = torch.zeros(
            2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_total_kv_lens_ctx = torch.zeros(
            2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_total_kv_lens_gen = torch.zeros(
            2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_request_types = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.block_offsets = torch.zeros(
            1, max_batch, 2, max_blocks_per_seq, dtype=torch.int32, device=device
        )
        self.host_past_kv_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.host_context_lengths = torch.zeros(
            max_batch, dtype=torch.int32, device="cpu", pin_memory=prefer_pinned()
        )
        self.context_lengths_cuda = torch.zeros(max_batch, dtype=torch.int32, device=device)

    def plan(
        self,
        num_prefill: int,
        num_decode: int,
        max_context_length: int,
        block_offset_multiplier: int,
        seq_len_with_cache_host: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        page_seq_indices: torch.Tensor,
        page_in_seq: torch.Tensor,
        input_pos_host: torch.Tensor,
        seq_len_host: torch.Tensor,
    ) -> None:
        """Per-forward host metadata: fills host_request_types, block_offsets, host_total_kv_lens.

        Called from ``prepare_trtllm_metadata_host`` before every forward (including replays).
        """
        num_seq = num_prefill + num_decode

        # host_request_types: 0 = prefill (context), 1 = decode (generation)
        self.host_request_types[:num_prefill].fill_(0)
        self.host_request_types[num_prefill:num_seq].fill_(1)

        # Compute block_offsets for thop.attention using pre-computed page indices.
        block_offsets = self.block_offsets
        total_pages = int(cu_num_pages_host[num_seq])

        # Zero block_offsets for current batch sequences before filling.
        # This prevents stale entries from previous batches (which may have had
        # more pages per sequence) from causing incorrect attention reads.
        block_offsets[:, :num_seq, :, :].zero_()

        base_offsets = cache_loc[:total_pages] * block_offset_multiplier
        seq_idx = page_seq_indices[:total_pages]
        pg_idx = page_in_seq[:total_pages]
        block_offsets[0, seq_idx, 0, pg_idx] = base_offsets  # K
        block_offsets[0, seq_idx, 1, pg_idx] = base_offsets + 1  # V

        # host_total_kv_lens: [context_total_kv, gen_total_kv]
        is_capturing = torch.cuda.is_current_stream_capturing() or cuda_graph_state.in_warm_up()
        if is_capturing:
            # CUDA graph capture: set host tensors to MAX values so the kernel captures
            # the worst-case execution pattern.
            self.host_total_kv_lens[0] = max_context_length * num_prefill
            self.host_total_kv_lens[1] = max_context_length * num_decode
            self.host_past_kv_lengths[:num_seq].fill_(max_context_length)
            self.host_context_lengths[:num_seq].fill_(max_context_length)
        else:
            ctx_total = int(seq_len_with_cache_host[:num_prefill].sum())
            gen_total = int(seq_len_with_cache_host[num_prefill:num_seq].sum())
            self.host_total_kv_lens[0] = ctx_total
            self.host_total_kv_lens[1] = gen_total
            # Per-sub-batch tensors for mixed batch splitting: each sub-batch
            # should only report its own KV total, with the other component zeroed.
            self.host_total_kv_lens_ctx[0] = ctx_total
            self.host_total_kv_lens_ctx[1] = 0
            self.host_total_kv_lens_gen[0] = 0
            self.host_total_kv_lens_gen[1] = gen_total
            # host_past_kv_lengths must equal sequence_length (total KV length
            # including past + current tokens), matching the standard backend's
            # kv_lens_runtime = cached_token_lens + seq_lens_kv.
            self.host_past_kv_lengths[:num_seq] = seq_len_with_cache_host[:num_seq]
            # host_context_lengths: context length for prefill seqs, 0 for decode seqs.
            # The standard backend uses prompt_lens_cpu which is 0 for decode.
            self.host_context_lengths[:num_prefill] = seq_len_host[:num_prefill]
            self.host_context_lengths[num_prefill:num_seq].zero_()

    def get_pool_pointers_for_layer(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """Return a per-layer ``host_pool_pointers`` tensor for this kv_cache view.

        Each attention layer receives a different ``kv_cache`` tensor (a strided view
        into the pool).  We create one pinned [1, 2] int64 tensor per unique
        ``data_ptr`` and cache it forever.  This guarantees that each layer's
        ``thop.attention`` call in a CUDA graph is captured with a *stable, distinct*
        tensor address, so graph replay reads the correct pool base for every layer.
        """
        ptr = kv_cache.data_ptr()
        t = self._per_layer_pool_ptrs.get(ptr)
        if t is not None:
            return t

        t = torch.zeros(1, 2, dtype=torch.int64, device="cpu", pin_memory=prefer_pinned())
        t[0, 0] = ptr
        self._per_layer_pool_ptrs[ptr] = t
        return t


_GlobalTrtllmPlanner = _TrtllmPlanner()

# Debug: per-forward layer counter (reset in prepare_trtllm_metadata_host, incremented in mha_with_cache)
_debug_layer_counter = 0
_debug_forward_counter = 0

_SDPA_OVERRIDE = False
_SDPA_OVERRIDE_DECODE_ONLY = False

# =============================================================================
# Host-side prepare function (analogous to prepare_flashinfer_metadata_host)
# =============================================================================


def prepare_trtllm_metadata_host(
    batch_info_host: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    page_seq_indices: torch.Tensor,
    page_in_seq: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_host: torch.Tensor,
) -> None:
    """Fill thop-specific host metadata and compute block_offsets.

    This runs OUTSIDE the CUDA graph before every forward (including replays).
    Block offsets MUST be computed here (not in the device-side prepare_metadata op)
    because they are batch-dependent and need to be updated before each replay.

    All max-size constants are read from ``max_seq_info_host`` which is set once via
    ``SequenceInfo.update_cache_information()`` after cache initialization:
    ``[max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size]``

    ``page_seq_indices`` and ``page_in_seq`` are pre-computed in SequenceInfo from
    ``pages_per_seq`` and avoid the expensive GPU searchsorted that was previously needed.
    """
    global _debug_layer_counter, _debug_forward_counter
    num_prefill, _, num_decode = batch_info_host.tolist()

    # Read all max-size constants from max_seq_info_host (set at cache init time)
    max_context_length, max_blocks_per_seq, block_offset_multiplier, max_batch_size = (
        max_seq_info_host.tolist()
    )

    # One-time allocation of all persistent buffers (lazy, guards against double-init)
    _GlobalTrtllmPlanner.reset(cache_loc.device, max_batch_size, max_blocks_per_seq)

    # Debug: reset layer counter at start of forward, increment forward counter
    _debug_layer_counter = 0
    _debug_forward_counter += 1
    num_seq = num_prefill + num_decode
    ad_logger.info(
        f"[plan] fwd={_debug_forward_counter} prefill={num_prefill} decode={num_decode} "
        f"input_pos={input_pos_host[:num_seq].tolist()} seq_len={seq_len_host[:num_seq].tolist()} "
        f"seq_len_with_cache={seq_len_with_cache_host[:num_seq].tolist()} "
        f"cache_loc[:8]={cache_loc[: min(8, len(cache_loc))].tolist()} "
        f"block_offset_mult={block_offset_multiplier}"
    )

    # Per-forward: fill host_request_types, block_offsets, host_total_kv_lens
    _GlobalTrtllmPlanner.plan(
        num_prefill=num_prefill,
        num_decode=num_decode,
        max_context_length=max_context_length,
        block_offset_multiplier=block_offset_multiplier,
        seq_len_with_cache_host=seq_len_with_cache_host,
        cu_num_pages_host=cu_num_pages_host,
        cache_loc=cache_loc,
        page_seq_indices=page_seq_indices,
        page_in_seq=page_in_seq,
        input_pos_host=input_pos_host,
        seq_len_host=seq_len_host,
    )


# =============================================================================
# Cached attention op (analogous to flashinfer_mha_with_cache)
# =============================================================================


@torch.library.custom_op("auto_deploy::trtllm_attention_mha_with_cache", mutates_args=("kv_cache",))
def trtllm_mha_with_cache(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (SequenceInfo fields used directly by thop)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # CACHE
    kv_cache: torch.Tensor,
    # CONSTANTS (only truly un-inferable values)
    scale: Optional[float],
    sliding_window: Optional[int] = None,
    kv_scale_orig_quant: float = 1.0,
    kv_scale_quant_orig: float = 1.0,
) -> torch.Tensor:
    """TRT-LLM attention with paged KV cache for Auto-Deploy.

    Infers num_heads, num_kv_heads, head_dim, and tokens_per_block from tensor shapes.
    All max-size constants (max_num_requests, max_context_length) are read from
    ``max_seq_info_host`` which is set once via ``SequenceInfo.update_cache_information()``.

    Note: ``prepare_trtllm_metadata_host`` is guaranteed to be called before this op,
    so all persistent planner buffers are already initialized.

    Note: layer_idx is always passed as 0 to thop.attention because
    the kv_cache tensor is already a strided view for the correct layer,
    pool_pointers encodes kv_cache.data_ptr() (layer-specific), and
    pool_mapping is all zeros. See module docstring for details.
    """
    global _debug_layer_counter
    _debug_layer_counter += 1
    _dbg_layer = _debug_layer_counter

    # Infer dimensions from tensor shapes (bsnd layout)
    num_heads = q.shape[2]
    num_kv_heads = k.shape[2]
    head_dim = q.shape[3]
    tokens_per_block = kv_cache.shape[3]  # HND: [blocks, 2, heads, tpb, head_dim]

    # Get batch dimensions and model-level constants from host tensors (no device sync)
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_tokens = num_prefill_tokens + num_decode
    max_context_length = int(max_seq_info_host[0])
    max_num_requests = int(max_seq_info_host[3])
    # Use sliding_window for attention_window_size if provided, else full context length
    attention_window_size = (
        sliding_window
        if isinstance(sliding_window, int) and sliding_window > 0
        else max_context_length
    )

    # Get per-layer pool pointer tensor (stable address for CUDA graph replay)
    host_kv_cache_pool_pointers = _GlobalTrtllmPlanner.get_pool_pointers_for_layer(kv_cache)

    # FP8 KV cache: lazily create scale tensors from float constants on first use
    if kv_cache.dtype == torch.float8_e4m3fn:
        if _GlobalTrtllmPlanner.kv_scale_orig_quant is None:
            _GlobalTrtllmPlanner.kv_scale_orig_quant = torch.tensor(
                [kv_scale_orig_quant], dtype=torch.float32, device=q.device
            )
            _GlobalTrtllmPlanner.kv_scale_quant_orig = torch.tensor(
                [kv_scale_quant_orig], dtype=torch.float32, device=q.device
            )
        quant_mode = int(QuantMode.FP8_KV_CACHE)
    else:
        quant_mode = 0

    # Reshape Q, K, V to [num_tokens, num_heads * head_dim] and fuse
    # Input is always [bs, 1] (generate-only) or [1, total_seq_len] (prefill/mixed),
    # so b * s == num_tokens always holds.
    q_shape_og = q.shape

    # Debug: compare context attention against PyTorch SDPA at ALL token positions.
    # Run at ALL layers for fwd=4 (mixed batch) to find the first divergent layer.
    # For other fwd passes, only run at layer 1.
    _ref_sdpa_out = None
    _do_ctx_cmp = (
        num_prefill > 0
        and num_prefill_tokens > 1
        and (_dbg_layer == 1 or _debug_forward_counter == 4)
    )
    if _do_ctx_cmp:
        with torch.no_grad():
            q_ctx = q[:, :num_prefill_tokens, :, :]
            k_ctx = k[:, :num_prefill_tokens, :, :]
            v_ctx = v[:, :num_prefill_tokens, :, :]
            if num_kv_heads != num_heads:
                rep = num_heads // num_kv_heads
                k_ctx = k_ctx.repeat_interleave(rep, dim=2)
                v_ctx = v_ctx.repeat_interleave(rep, dim=2)
            _ref_sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                q_ctx.transpose(1, 2),
                k_ctx.transpose(1, 2),
                v_ctx.transpose(1, 2),
                is_causal=True,
                scale=scale,
            ).transpose(1, 2)  # [B, S, H, D]
            ref_last = _ref_sdpa_out[0, num_prefill_tokens - 1]
            ad_logger.info(
                f"[ref_sdpa] fwd={_debug_forward_counter} prefill_tokens={num_prefill_tokens} "
                f"last_tok_abs_mean={ref_last.abs().mean().item():.6f} "
                f"last_tok[:4]={ref_last.reshape(-1)[:4].tolist()}"
            )

    q_flat = q.reshape(num_tokens, num_heads * head_dim)
    k_flat = k.reshape(num_tokens, num_kv_heads * head_dim)
    v_flat = v.reshape(num_tokens, num_kv_heads * head_dim)
    qkv_fused = torch.cat([q_flat, k_flat, v_flat], dim=-1).contiguous()

    # Prepare output
    output = torch.empty(num_tokens, num_heads * head_dim, dtype=q.dtype, device=q.device)

    # Common metadata
    host_total_kv_lens = _GlobalTrtllmPlanner.host_total_kv_lens
    host_kv_cache_pool_mapping = _GlobalTrtllmPlanner.host_pool_mapping

    # Pack parameters for thop.attention
    rotary_embedding_scales = [1.0, 1.0, 1.0]
    rotary_embedding_max_position_info = [max_context_length, max_context_length]
    spec_decoding_bool_params = [False, False, False]
    spec_decoding_tensor_params = [None, None, None]

    sm_version = get_sm_version()
    if sm_version >= 89:  # Ada/Hopper
        spec_decoding_tensor_params.extend([None, None, None])

    mla_tensor_params = [None, None]

    # AttentionInputType enum: 0=Mixed, 1=ContextOnly, 2=GenerationOnly
    # The C++ run() method does not offset host tensor pointers
    # (host_past_key_value_lengths, host_context_lengths) by seq_offset when
    # splitting a Mixed batch into context and generation sub-batches. The XQA
    # kernel then reads wrong per-sequence metadata for the generation requests.
    # The standard PyTorch backend avoids this by always calling thop.attention
    # with ContextOnly or GenerationOnly (never Mixed). We follow the same
    # pattern: for mixed batches, make two separate calls with correctly sliced
    # per-sequence tensors.
    _CONTEXT_ONLY = 1
    _GENERATION_ONLY = 2

    def _invoke_thop(
        qkv_slice,
        out_slice,
        seq_len_slice,
        ctx_len_slice,
        host_past_kv_slice,
        host_ctx_slice,
        host_req_slice,
        block_offsets_view,
        attn_input_type,
        total_kv_lens,
    ):
        thop.attention(
            qkv_slice,  # q (actually fused QKV)
            None,  # k (None when using fused QKV)
            None,  # v (None when using fused QKV)
            out_slice,  # output
            None,  # output_sf (NVFP4)
            _GlobalTrtllmPlanner.workspace,
            seq_len_slice,  # sequence_length
            host_past_kv_slice,  # host_past_key_value_lengths
            total_kv_lens,  # host_total_kv_lens
            ctx_len_slice,  # context_lengths
            host_ctx_slice,  # host_context_lengths
            host_req_slice,  # host_request_types
            block_offsets_view,  # kv_cache_block_offsets
            host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping,
            None,  # cache_indirection (beam search)
            _GlobalTrtllmPlanner.kv_scale_orig_quant,
            _GlobalTrtllmPlanner.kv_scale_quant_orig,
            None,  # out_scale
            None,  # rotary_inv_freq
            None,  # rotary_cos_sin
            None,  # latent_cache (MLA)
            None,  # q_pe (MLA)
            None,  # block_ids_per_seq
            None,  # attention_sinks
            True,  # is_fused_qkv
            True,  # update_kv_cache
            1,  # predicted_tokens_per_seq
            0,  # layer_idx (always 0; pool_pointers encodes the layer offset)
            num_heads,
            num_kv_heads,
            head_dim,
            tokens_per_block,
            max_num_requests,
            max_context_length,
            attention_window_size,
            0,  # sink_token_length
            1,  # beam_width
            int(AttentionMaskType.causal),
            quant_mode,
            1.0,  # q_scaling
            0,  # position_embedding_type
            0,  # rotary_embedding_dim
            10000.0,  # rotary_embedding_base
            0,  # rotary_embedding_scale_type
            rotary_embedding_scales,
            rotary_embedding_max_position_info,
            True,  # use_paged_context_fmha
            attn_input_type,
            False,  # is_mla_enable
            max_num_requests,  # chunked_prefill_buffer_batch_size
            None,  # q_lora_rank (MLA)
            None,  # kv_lora_rank (MLA)
            None,  # qk_nope_head_dim (MLA)
            None,  # qk_rope_head_dim (MLA)
            None,  # v_head_dim (MLA)
            None,  # mrope_rotary_cos_sin
            None,  # mrope_position_deltas
            mla_tensor_params,
            None,  # attention_chunk_size
            None,  # softmax_stats_tensor
            spec_decoding_bool_params,
            spec_decoding_tensor_params,
            None,  # sparse_kv_indices
            None,  # sparse_kv_offsets
            None,  # sparse_attn_indices
            None,  # sparse_attn_offsets
            1,  # sparse_attn_indices_block_size
            0,  # sparse_mla_topk
            None,  # skip_softmax_threshold_scale_factor_prefill
            None,  # skip_softmax_threshold_scale_factor_decode
            None,  # skip_softmax_stat
            None,  # cu_q_seqlens
            None,  # cu_kv_seqlens
            None,  # fmha_scheduler_counter
            None,  # mla_bmm1_scale
            None,  # mla_bmm2_scale
            None,  # quant_q_buffer
        )

    planner = _GlobalTrtllmPlanner
    block_offsets = planner.block_offsets

    # GPU context_lengths: context length for prefill, 0 for decode.
    # Matches the standard backend's prompt_lens_cuda (0 for generation requests).
    ctx_lens_gpu = planner.context_lengths_cuda
    ctx_lens_gpu[:num_prefill].copy_(seq_len[:num_prefill])
    ctx_lens_gpu[num_prefill:num_seq].zero_()

    is_mixed = num_prefill > 0 and num_decode > 0
    if is_mixed:
        ad_logger.info(
            f"[trtllm_attn] MIXED batch split: {num_prefill} ctx ({num_prefill_tokens} tok) "
            f"+ {num_decode} gen, host_past_kv={planner.host_past_kv_lengths[:num_seq].tolist()}, "
            f"host_ctx_len={planner.host_context_lengths[:num_seq].tolist()}"
        )
        # Context (prefill) sub-batch — only context KV total, zero generation
        _invoke_thop(
            qkv_fused[:num_prefill_tokens],
            output[:num_prefill_tokens],
            seq_len_with_cache[:num_prefill],
            ctx_lens_gpu[:num_prefill],
            planner.host_past_kv_lengths[:num_prefill],
            planner.host_context_lengths[:num_prefill],
            planner.host_request_types[:num_prefill],
            block_offsets,
            _CONTEXT_ONLY,
            planner.host_total_kv_lens_ctx,
        )
        # Generation (decode) sub-batch — only generation KV total, zero context
        _invoke_thop(
            qkv_fused[num_prefill_tokens:],
            output[num_prefill_tokens:],
            seq_len_with_cache[num_prefill:num_seq],
            ctx_lens_gpu[num_prefill:num_seq],
            planner.host_past_kv_lengths[num_prefill:num_seq],
            planner.host_context_lengths[num_prefill:num_seq],
            planner.host_request_types[num_prefill:num_seq],
            block_offsets[:, num_prefill:, :, :],
            _GENERATION_ONLY,
            planner.host_total_kv_lens_gen,
        )
    else:
        attn_type = _CONTEXT_ONLY if num_prefill > 0 else _GENERATION_ONLY
        _invoke_thop(
            qkv_fused,
            output,
            seq_len_with_cache[:num_seq],
            ctx_lens_gpu[:num_seq],
            planner.host_past_kv_lengths[:num_seq],
            planner.host_context_lengths[:num_seq],
            planner.host_request_types[:num_seq],
            block_offsets,
            attn_type,
            host_total_kv_lens,
        )

    # Debug: compare thop context output vs SDPA reference at ALL token positions
    if _ref_sdpa_out is not None:
        with torch.no_grad():
            thop_ctx_out = output[:num_prefill_tokens].reshape(
                1, num_prefill_tokens, num_heads, head_dim
            )
            diff = (_ref_sdpa_out - thop_ctx_out).abs()
            per_tok_max = diff.reshape(num_prefill_tokens, -1).max(dim=1).values
            worst_tok = per_tok_max.argmax().item()
            ad_logger.info(
                f"[ctx_full_cmp] fwd={_debug_forward_counter} layer={_dbg_layer} ntok={num_prefill_tokens} "
                f"all_max_diff={per_tok_max.max().item():.8f} "
                f"all_mean_diff={diff.mean().item():.8f} "
                f"worst_tok={worst_tok} worst_tok_max={per_tok_max[worst_tok].item():.8f} "
                f"tok0_max={per_tok_max[0].item():.8f} "
                f"last_tok_max={per_tok_max[-1].item():.8f}"
            )
            if per_tok_max.max().item() > 0.01:
                bad_toks = (per_tok_max > 0.01).nonzero(as_tuple=True)[0]
                ad_logger.warning(
                    f"[ctx_full_cmp] MISMATCH at {len(bad_toks)} tokens! "
                    f"bad_tok_indices={bad_toks[:10].tolist()} "
                    f"max_diffs={per_tok_max[bad_toks[:10]].tolist()}"
                )

    # Debug: verify KV cache write correctness after context call in mixed batch
    # Check at layers 1, 16, and 32 (first, middle, last of a 32-layer model)
    # Skip for FP8 KV cache since dtype promotion is not supported
    if (
        _dbg_layer in (1, 16, 32)
        and is_mixed
        and num_prefill > 0
        and kv_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        with torch.no_grad():
            torch.cuda.synchronize()
            k_input = k[:, :num_prefill_tokens, :, :].reshape(
                num_prefill_tokens, num_kv_heads, head_dim
            )
            v_input = v[:, :num_prefill_tokens, :, :].reshape(
                num_prefill_tokens, num_kv_heads, head_dim
            )
            k_block_off = block_offsets[0, 0, 0, 0].item()
            v_block_off = block_offsets[0, 0, 1, 0].item()
            bom = int(max_seq_info_host[2])
            page_id = k_block_off // bom if bom > 0 else 0
            ctx_len = int(seq_len_host[0])
            k_cached = kv_cache[page_id, 0, :, :ctx_len, :]
            v_cached = kv_cache[page_id, 1, :, :ctx_len, :]
            k_input_t0 = k_input[0]
            k_cached_t0 = k_cached[:, 0, :]
            k_diff = (k_input_t0 - k_cached_t0).abs().max().item()
            k_input_last = k_input[ctx_len - 1]
            k_cached_last = k_cached[:, ctx_len - 1, :]
            k_diff_last = (k_input_last - k_cached_last).abs().max().item()
            v_input_t0 = v_input[0]
            v_cached_t0 = v_cached[:, 0, :]
            v_diff = (v_input_t0 - v_cached_t0).abs().max().item()
            v_input_last = v_input[ctx_len - 1]
            v_cached_last = v_cached[:, ctx_len - 1, :]
            v_diff_last = (v_input_last - v_cached_last).abs().max().item()
            ad_logger.info(
                f"[kv_verify] fwd={_debug_forward_counter} page_id={page_id} ctx_len={ctx_len} "
                f"k_block_off={k_block_off} v_block_off={v_block_off} bom={bom} "
                f"K_diff_t0={k_diff:.8f} K_diff_last={k_diff_last:.8f} "
                f"V_diff_t0={v_diff:.8f} V_diff_last={v_diff_last:.8f}"
            )

    # Debug: log output statistics at key layers to detect corruption
    if _dbg_layer in (1, 2, 16, 32):
        with torch.no_grad():
            for seq_i in range(num_seq):
                if seq_i < num_prefill:
                    tok_start = sum(seq_len_host[:seq_i].tolist()) if seq_i > 0 else 0
                    tok_end = tok_start + int(seq_len_host[seq_i])
                    last_tok = output[tok_end - 1]
                else:
                    idx = num_prefill_tokens + (seq_i - num_prefill)
                    last_tok = output[idx]
                ad_logger.info(
                    f"[attn_out] fwd={_debug_forward_counter} layer={_dbg_layer} seq={seq_i} "
                    f"last_tok_abs_mean={last_tok.abs().mean().item():.6f} "
                    f"last_tok[:4]={last_tok[:4].tolist()}"
                )

    # Debug: decode-time SDPA comparison — run at ALL layers for fwd=5 to find worst layer
    # Skip for FP8 KV cache since dtype promotion is not supported
    if (
        num_prefill == 0
        and num_decode >= 2
        and _debug_forward_counter == 5
        and kv_cache.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        with torch.no_grad():
            torch.cuda.synchronize()
            bom = int(max_seq_info_host[2])
            # seq 1 = prompt 1 in decode-only batches
            seq_idx_to_check = 1
            past_len = int(input_pos_host[seq_idx_to_check])
            total_len = past_len + 1  # past + new decode token

            k_off = block_offsets[0, seq_idx_to_check, 0, 0].item()
            page_id = k_off // bom if bom > 0 else 0

            # After thop.attention, cache has all tokens including the new one
            k_all = kv_cache[page_id, 0, :, :total_len, :]  # [kv_heads, total_len, hd]
            v_all = kv_cache[page_id, 1, :, :total_len, :]

            # GQA expand
            if num_kv_heads != num_heads:
                rep = num_heads // num_kv_heads
                k_all = k_all.repeat_interleave(rep, dim=0)
                v_all = v_all.repeat_interleave(rep, dim=0)

            # Query for this sequence
            q_tok = q.reshape(num_tokens, num_heads, head_dim)[seq_idx_to_check]  # [nh, hd]
            q_sdpa = q_tok.unsqueeze(0).unsqueeze(2)  # [1, nh, 1, hd]
            k_sdpa = k_all.unsqueeze(0)  # [1, nh, total_len, hd]
            v_sdpa = v_all.unsqueeze(0)

            ref = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=False,
                scale=scale,
            )  # [1, nh, 1, hd]
            ref_flat = ref.squeeze(0).squeeze(1)  # [nh, hd]

            thop_out = output[num_prefill_tokens + seq_idx_to_check].reshape(num_heads, head_dim)
            diff = (ref_flat - thop_out).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            ad_logger.info(
                f"[decode_sdpa] fwd={_debug_forward_counter} layer={_dbg_layer} seq={seq_idx_to_check} "
                f"past_len={past_len} page_id={page_id} "
                f"max_diff={max_diff:.8f} mean_diff={mean_diff:.8f} "
                f"ref[:4]={ref_flat.reshape(-1)[:4].tolist()} "
                f"thop[:4]={thop_out.reshape(-1)[:4].tolist()}"
            )
            if max_diff > 0.01:
                # Verify the new decode token was written correctly
                k_new_cached = kv_cache[page_id, 0, :, past_len, :]  # [kv_heads, hd]
                k_new_input = k.reshape(num_tokens, num_kv_heads, head_dim)[seq_idx_to_check]
                k_new_diff = (k_new_cached - k_new_input).abs().max().item()
                # Verify some past tokens
                k_past_0 = kv_cache[page_id, 0, :, 0, :]
                ad_logger.warning(
                    f"[decode_sdpa] MISMATCH! k_new_write_diff={k_new_diff:.8f} "
                    f"k_cached_t0[:4]={k_past_0.reshape(-1)[:4].tolist()} "
                    f"k_new_cached[:4]={k_new_cached.reshape(-1)[:4].tolist()}"
                )

    # Debug: override thop output with pure SDPA output (thop still ran for KV cache update)
    _do_override = (_SDPA_OVERRIDE and _debug_forward_counter >= 3) or (
        _SDPA_OVERRIDE_DECODE_ONLY and _debug_forward_counter >= 3 and num_decode > 0
    )
    if _do_override:
        with torch.no_grad():
            bom = int(max_seq_info_host[2])
            # Context tokens: compute SDPA from input Q/K/V (skip if decode-only override)
            if num_prefill_tokens > 0 and not _SDPA_OVERRIDE_DECODE_ONLY:
                q_ctx = q[:, :num_prefill_tokens, :, :]
                k_ctx = k[:, :num_prefill_tokens, :, :]
                v_ctx = v[:, :num_prefill_tokens, :, :]
                if num_kv_heads != num_heads:
                    rep = num_heads // num_kv_heads
                    k_ctx = k_ctx.repeat_interleave(rep, dim=2)
                    v_ctx = v_ctx.repeat_interleave(rep, dim=2)
                sdpa_ctx = torch.nn.functional.scaled_dot_product_attention(
                    q_ctx.transpose(1, 2),
                    k_ctx.transpose(1, 2),
                    v_ctx.transpose(1, 2),
                    is_causal=True,
                    scale=scale,
                ).transpose(1, 2)
                output[:num_prefill_tokens] = sdpa_ctx.reshape(
                    num_prefill_tokens, num_heads * head_dim
                )

            # Decode tokens: compute SDPA from cached K/V + new token
            for dec_i in range(num_decode):
                seq_i = num_prefill + dec_i
                tok_i = num_prefill_tokens + dec_i
                past_len = int(input_pos_host[seq_i])
                total_len = past_len + 1

                # Find page(s) for this sequence
                k_off = block_offsets[0, seq_i, 0, 0].item()
                page_id = k_off // bom if bom > 0 else 0

                # Read full K/V from cache (including new token just written by thop)
                pages_needed = (total_len + tokens_per_block - 1) // tokens_per_block
                if pages_needed == 1:
                    k_all = kv_cache[page_id, 0, :, :total_len, :]
                    v_all = kv_cache[page_id, 1, :, :total_len, :]
                else:
                    k_pages = []
                    v_pages = []
                    remaining = total_len
                    for pg in range(pages_needed):
                        pg_off = block_offsets[0, seq_i, 0, pg].item()
                        pg_id = pg_off // bom if bom > 0 else 0
                        n = min(tokens_per_block, remaining)
                        k_pages.append(kv_cache[pg_id, 0, :, :n, :])
                        v_pages.append(kv_cache[pg_id, 1, :, :n, :])
                        remaining -= n
                    k_all = torch.cat(k_pages, dim=1)
                    v_all = torch.cat(v_pages, dim=1)

                if num_kv_heads != num_heads:
                    rep = num_heads // num_kv_heads
                    k_all = k_all.repeat_interleave(rep, dim=0)
                    v_all = v_all.repeat_interleave(rep, dim=0)

                q_tok = q.reshape(num_tokens, num_heads, head_dim)[tok_i]
                q_sdpa = q_tok.unsqueeze(0).unsqueeze(2)
                k_sdpa = k_all.unsqueeze(0)
                v_sdpa = v_all.unsqueeze(0)
                ref = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa,
                    k_sdpa,
                    v_sdpa,
                    is_causal=False,
                    scale=scale,
                )
                output[tok_i] = ref.squeeze(0).squeeze(1).reshape(num_heads * head_dim)

    return output.view(*q_shape_og)


@trtllm_mha_with_cache.register_fake
def trtllm_mha_with_cache_fake(
    # Q, K, V inputs
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # STANDARD METADATA (SequenceInfo fields used directly by thop)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    seq_len_host: torch.Tensor,
    input_pos_host: torch.Tensor,
    seq_len_with_cache: torch.Tensor,
    max_seq_info_host: torch.Tensor,
    # CACHE
    kv_cache: torch.Tensor,
    # CONSTANTS (only truly un-inferable values)
    scale: Optional[float],
    sliding_window: Optional[int] = None,
    kv_scale_orig_quant: float = 1.0,
    kv_scale_quant_orig: float = 1.0,
) -> torch.Tensor:
    """Fake implementation for torch.compile tracing."""
    return torch.empty_like(q.contiguous())


# =============================================================================
# AttentionDescriptor (analogous to FlashInferAttention)
# =============================================================================


@AttentionRegistry.register("trtllm")
class TrtllmAttention(AttentionDescriptor):
    """TRT-LLM attention backend for Auto-Deploy.

    Follows the same stateless descriptor pattern as ``FlashInferAttention``.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op."""
        return torch.ops.auto_deploy.trtllm_attention_mha_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments from SequenceInfo."""
        return [
            "batch_info_host",
            "seq_len",
            "seq_len_host",
            "input_pos_host",
            "seq_len_with_cache",
            "max_seq_info_host",
        ]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Return only KV cache handler (no workspace handler, managed like flashinfer)."""
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]

        return {
            "kv_cache": KVPagedResourceHandler(
                num_kv_heads,
                head_dim,
                dtype=cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype),
                kv_factor=2,
                kv_layout="HND",
            )
        }

    @classmethod
    def get_host_prepare_metadata_function(cls) -> Optional[PrepareMetadataHostCallable]:
        """Return host-side prepare function for thop-specific metadata."""
        return prepare_trtllm_metadata_host

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Extract constants from the source attention node.

        Returns scale, sliding_window, kv_scale_orig_quant, and kv_scale_quant_orig.
        Everything else (num_heads, head_dim, max_context_length, etc.) is inferred
        from tensor shapes or SequenceInfo metadata at runtime.
        """
        # Sanity check: layout == "bsnd"
        layout = source_attn_node.kwargs.get("layout", None)
        if (
            layout is None
            and len(source_attn_node.args) > 0
            and isinstance(source_attn_node.args[-1], str)
        ):
            layout = source_attn_node.args[-1]
        if layout != "bsnd":
            raise RuntimeError(
                f"Expected torch_attention layout='bsnd' but got {layout!r} "
                f"for node: {source_attn_node.format_node()}"
            )

        # Check other arguments
        _attn_mask, _dropout_p, _is_causal = extract_op_args(
            source_attn_node, "attn_mask", "dropout_p", "is_causal"
        )

        # Get scale
        if len(source_attn_node.args) > 6:
            scale = source_attn_node.args[6]
        else:
            scale = source_attn_node.kwargs.get("scale", None)

        if not (isinstance(scale, float) or scale is None):
            ad_logger.warning(f"Provided {scale=}, is not a float. Using default scale instead.")
            scale = None

        # Get sliding_window from source attention node
        sliding_window = extract_op_args(source_attn_node, "sliding_window")[0]

        return [
            scale,
            sliding_window,
            1.0,  # kv_scale_orig_quant (hard-coded, same as FlashInfer)
            1.0,  # kv_scale_quant_orig (hard-coded, same as FlashInfer)
        ]
