# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash speculative-decoding draft attention ops.

DFlash drafts a whole block ``[last_accepted, MASK, ..., MASK]`` in one **non-causal** masked pass
over a persistent drafter-side context K/V cache (derived from accepted target hidden states) plus
the current query block. Two ops:

- ``auto_deploy::dflash_attention`` -- the **source** op the DFlash draft layer emits at export. Its
  eager body delegates to the well-tested ``auto_deploy::torch_attention(..., is_causal=False)``, so
  source/export-mode math is canonical non-causal SDPA. It is a *distinct* op (not ``torch_attention``)
  so routing is by op-type (a dedicated ``insert_cached_dflash_attention`` transform), and it carries
  ``ctx_len`` (ignored by the SDPA math) so that graph input survives export and flows to the cached op.

- ``auto_deploy::dflash_attention_with_kvcache`` -- the **cached** op, a thin wrapper over
  ``flash_attn.flash_attn_with_kvcache(..., causal=False)`` (the same kernel the PyTorch DFlash branch
  uses). It reads the persistent per-slot context K/V (``cache_batch_idx=slot_idx``,
  ``cache_seqlens=ctx_len``) and **appends** the current query-block K/V in place into the cache slack
  at ``ctx_len`` (hence the ctx caches are declared mutated). The DFlash wrapper writes only accepted
  target-derived K/V rows before this op runs; the query-block append is transient scratch.

Layout is ``bsnd`` throughout: ``q`` is ``[B, block, n_heads, head_dim]``; ``k``/``v`` (current
query-block K/V to append) are ``[B, block, n_kv, head_dim]``; the ctx caches are dense per-slot
``[max_slots, max_ctx + block, n_kv, head_dim]``.
"""

from typing import List, Optional

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

from ..._compat import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandler,
    ResourceHandlerDict,
    SequenceInfo,
)

__all__ = [
    "dflash_attention",
    "dflash_attention_with_kvcache",
    "DFlashCtxKVResourceHandler",
    "DFlashAttention",
]


# ============================================================================ #
#  Source op (export-time): non-causal SDPA, delegates to torch_attention      #
# ============================================================================ #
@torch.library.custom_op("auto_deploy::dflash_attention", mutates_args=())
def dflash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx_len: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Non-causal masked-block attention (source/export representation).

    Delegates the math to ``auto_deploy::torch_attention(is_causal=False)`` over the provided
    ``q``/``k``/``v`` (bsnd). ``ctx_len`` is carried for the cached-op lowering contract and is
    intentionally unused here -- in source/export mode there is no persistent ctx cache; the honest
    representation is bidirectional attention over the K/V tensors the export graph constructs.
    """
    return torch.ops.auto_deploy.torch_attention(
        q, k, v, is_causal=False, scale=scale, layout="bsnd"
    )


@dflash_attention.register_fake
def dflash_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ctx_len: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    # bsnd: output keeps q's [B, block, n_heads] and adopts v's head_dim.
    return q.new_empty(*q.shape[:-1], v.shape[-1]).contiguous()


# ============================================================================ #
#  Cached op (runtime): wraps flash_attn_with_kvcache(causal=False)            #
# ============================================================================ #
@torch.library.custom_op(
    "auto_deploy::dflash_attention_with_kvcache",
    mutates_args=("ctx_k_cache", "ctx_v_cache"),
)
def dflash_attention_with_kvcache(
    # Q + current query-block K/V (to append), bsnd
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    # METADATA
    slot_idx: torch.Tensor,  # [B] int32 -> cache_batch_idx (per-request ctx slot)
    ctx_len: torch.Tensor,  # [B] int32 -> cache_seqlens (persistent context length)
    # CACHES (mutated: flash appends the query block into the slack at ctx_len)
    ctx_k_cache: torch.Tensor,  # [max_slots, max_ctx + block, n_kv, head_dim]
    ctx_v_cache: torch.Tensor,
    # CONSTANTS (scale is positional-required, matching trtllm/flashinfer cached ops: the first
    # CONSTANT is always supplied positionally by the descriptor's get_constants).
    scale: Optional[float],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Read persistent ctx K/V + append current query-block K/V, non-causal block attention.

    Thin wrapper over ``flash_attn_with_kvcache``: ``q`` attends (non-causally) over
    ``[ctx_k_cache[slot, :ctx_len] || query-block]``; the query-block K/V are appended in place at
    ``ctx_len`` for each request (``cache_batch_idx=slot_idx``). Returns ``[B, block, n_heads, head_dim]``.
    """
    # Lazy import so AutoDeploy import does not hard-depend on flash_attn in envs without it.
    from flash_attn import flash_attn_with_kvcache

    result = flash_attn_with_kvcache(
        q,
        ctx_k_cache,
        ctx_v_cache,
        k=k,
        v=v,
        cache_seqlens=ctx_len.to(torch.int32),
        cache_batch_idx=slot_idx.to(torch.int32),
        softmax_scale=scale,
        causal=False,
    )
    if out is not None:
        # AD CUDA-graph convention (mirrors trtllm/flashinfer cached ops): when the runtime injects a
        # pre-allocated, fixed-address output buffer, write the result into it and return an *empty*
        # fresh tensor. We must NOT return ``out`` itself -- a returned tensor may not alias an input
        # (``torch._library`` aliasing check). ``flash_attn_with_kvcache`` has no ``out=`` param, so a
        # copy into the stable buffer is unavoidable; the copy keeps the output at a graph-stable addr.
        out.copy_(result)
        return out.new_empty(0)
    return result


@dflash_attention_with_kvcache.register_fake
def dflash_attention_with_kvcache_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_idx: torch.Tensor,
    ctx_len: torch.Tensor,
    ctx_k_cache: torch.Tensor,
    ctx_v_cache: torch.Tensor,
    scale: Optional[float],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out.new_empty(0)
    return q.new_empty(*q.shape[:-1], ctx_v_cache.shape[-1]).contiguous()


# ============================================================================ #
#  Resource handler: dense slack-sized per-slot context K/V (bypasses KVCache) #
# ============================================================================ #
class DFlashCtxKVResourceHandler(ResourceHandler):
    """Unpaged dense per-slot context K/V for DFlash draft attention.

    Shape ``[max_num_state_slots, max_seq_len + block_size, n_kv, head_dim]`` -- one such resource
    per draft attention node (a ``ctx_k_cache`` and a ``ctx_v_cache``). The persistent context lives
    in ``[:ctx_len]`` (written by the wrapper from accepted target-derived K/V); the ``+block_size``
    slack is where ``flash_attn_with_kvcache`` transiently appends the current query block at row
    ``ctx_len`` (see ``dflash_attention_with_kvcache``). Unpaged + slot-indexed (``slot_idx`` ==
    ``cache_batch_idx``), so it bypasses the paged ``KVCacheManager`` -- the Eagle ``hidden_states``
    precedent. ``max_ctx = max_seq_len`` so ``ctx_len`` can never exceed the buffer (no clamp).
    """

    def __init__(
        self, num_kv_heads: int, head_dim: int, block_size: int, dtype: torch.dtype
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype

    def allocate(self, sequence_info: SequenceInfo) -> torch.Tensor:
        return torch.empty(
            sequence_info.max_num_state_slots,
            sequence_info.max_seq_len
            + self.block_size,  # +slack for the transient query-block append
            self.num_kv_heads,
            self.head_dim,
            device=sequence_info.device,
            dtype=self.dtype,
        )


# ============================================================================ #
#  Attention descriptor: routes dflash_attention -> dflash_attention_with_kvcache
# ============================================================================ #
@AttentionRegistry.register("dflash")
class DFlashAttention(AttentionDescriptor):
    """Descriptor for the DFlash draft attention path.

    Matches the **distinct** source op ``auto_deploy::dflash_attention`` (NOT ``torch_attention``), so
    the dedicated ``insert_cached_dflash_attention`` transform and the default ``insert_cached_attention``
    (which matches ``torch_attention``) never collide -- routing is purely by op-type. Lowers to the
    cached ``auto_deploy::dflash_attention_with_kvcache`` over per-slot dense ctx K/V resources.

    Metadata: ``slot_idx`` is *added* from ``SequenceInfo`` (no placeholder in the draft graph ->
    activate_arg), ``ctx_len`` is *retrieved* from the existing draft-graph placeholder (carried by the
    source op). The cached op's positional arg order
    ``(q, k, v, slot_idx, ctx_len, ctx_k_cache, ctx_v_cache, scale)`` is exactly
    ``(*qkv, *standard_metadata, *caches, *constants)`` produced by the generic insertion transform.
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.dflash_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.dflash_attention_with_kvcache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        # Order MUST match the cached op signature: slot_idx then ctx_len.
        # slot_idx -> added from SequenceInfo; ctx_len -> retrieved from the draft-graph placeholder.
        return ["slot_idx", "ctx_len"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        # Source op is dflash_attention(q, k, v, ctx_len, scale), bsnd layout.
        q_fake: FakeTensor = source_attn_node.args[0].meta["val"]
        k_fake: FakeTensor = source_attn_node.args[1].meta["val"]
        block_size = q_fake.shape[1]  # query-block width (drafter block_size)
        num_kv_heads = k_fake.shape[2]
        head_dim = k_fake.shape[3]
        dtype = cls.resolve_cache_dtype(cache_config.dtype, k_fake.dtype)
        # One resource per draft attention node; two per node (k and v).
        return {
            "ctx_k_cache": DFlashCtxKVResourceHandler(num_kv_heads, head_dim, block_size, dtype),
            "ctx_v_cache": DFlashCtxKVResourceHandler(num_kv_heads, head_dim, block_size, dtype),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        (scale,) = extract_op_args(source_attn_node, "scale")
        if not (isinstance(scale, float) or scale is None):
            scale = None
        return [scale]
