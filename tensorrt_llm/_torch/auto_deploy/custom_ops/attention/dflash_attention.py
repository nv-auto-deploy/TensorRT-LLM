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

from typing import Optional

import torch

__all__ = ["dflash_attention", "dflash_attention_with_kvcache"]


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
    # CONSTANTS
    scale: Optional[float] = None,
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
        out.copy_(result)
        return out
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
    scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out
    return q.new_empty(*q.shape[:-1], ctx_v_cache.shape[-1]).contiguous()
