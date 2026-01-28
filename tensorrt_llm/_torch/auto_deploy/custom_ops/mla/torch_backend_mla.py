"""Custom ops for MultiHead Latent Attention (MLA).

This module provides the attention descriptor for MLA, which uses:
- torch_mla: source op (without fused RoPE)
- torch_cached_mla_with_cache: cached backend op with unified FlashInfer cache layout

FlashInfer MLA Cache Layout:
    mla_cache: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
    - No num_heads dimension (MLA-specific optimization)
    - ckv_cached = mla_cache[:, :, :head_dim_ckv]  (zero-copy slice)
    - kpe_cached = mla_cache[:, :, head_dim_ckv:]  (zero-copy slice)

Reference: https://docs.flashinfer.ai/tutorials/kv_layout.html#mla-page-layout
"""

import math
from typing import List, Optional

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    Constant,
    MHACallable,
    ResourceHandlerDict,
    UnpagedResourceHandler,
)


def _update_mla_cache(
    ckv: torch.Tensor,  # [total_tokens, head_dim_ckv]
    kpe: torch.Tensor,  # [total_tokens, head_dim_kpe]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_start: torch.Tensor,
    head_dim_ckv: int,
) -> None:
    """Update unified MLA cache with ckv and kpe values.

    FlashInfer MLA cache layout: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
    - First head_dim_ckv dims: compressed key/value
    - Last head_dim_kpe dims: key positional encoding
    """
    for idx in range(seq_len.shape[0]):
        start = seq_start[idx].item()
        length = seq_len[idx].item()
        cache_idx = cache_loc[idx].item()
        pos = input_pos[idx].item()

        # Update ckv portion
        mla_cache[cache_idx, pos : pos + length, :head_dim_ckv] = ckv[start : start + length]
        # Update kpe portion
        mla_cache[cache_idx, pos : pos + length, head_dim_ckv:] = kpe[start : start + length]


def _torch_mla_generate(
    q_nope: torch.Tensor,  # [B, 1, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [B, 1, N, qk_rope_head_dim]
    ckv: torch.Tensor,  # [B, 1, 1, head_dim_ckv]
    kpe: torch.Tensor,  # [B, 1, 1, head_dim_kpe]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
    cache_loc: torch.Tensor,
    input_pos: torch.Tensor,
    scale: float,
    head_dim_ckv: int,
    out: torch.Tensor,
) -> None:
    """Generate-only MLA attention (single token per sequence)."""
    b = q_nope.shape[0]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    # qk_rope_head_dim = q_pe.shape[3]
    # head_dim_kpe = kpe.shape[3]

    # Flatten ckv and kpe for cache update: [B, 1, 1, D] -> [B, D]
    ckv_flat = ckv.squeeze(1).squeeze(1)  # [B, head_dim_ckv]
    kpe_flat = kpe.squeeze(1).squeeze(1)  # [B, head_dim_kpe]

    # Update cache for each sequence
    for i in range(b):
        cache_idx = cache_loc[i].item()
        pos = input_pos[i].item()
        mla_cache[cache_idx, pos, :head_dim_ckv] = ckv_flat[i]
        mla_cache[cache_idx, pos, head_dim_ckv:] = kpe_flat[i]

    # Compute attention for each sequence
    for i in range(b):
        cache_idx = cache_loc[i].item()
        pos = input_pos[i].item()

        # Get query for this sequence: [N, qk_nope_head_dim], [N, qk_rope_head_dim]
        q_nope_i = q_nope[i, 0]  # [N, qk_nope_head_dim]
        q_pe_i = q_pe[i, 0]  # [N, qk_rope_head_dim]

        # Retrieve cached ckv and kpe up to current position
        cached_data = mla_cache[cache_idx, : pos + 1]  # [seq_len, head_dim_ckv + head_dim_kpe]
        ckv_cached = cached_data[:, :head_dim_ckv]  # [seq_len, head_dim_ckv]
        kpe_cached = cached_data[:, head_dim_ckv:]  # [seq_len, head_dim_kpe]

        # Split ckv into k_nope and value
        # v_head_dim = head_dim_ckv - qk_nope_head_dim
        k_nope_cached = ckv_cached[:, :qk_nope_head_dim]  # [seq_len, qk_nope_head_dim]
        v_cached = ckv_cached[:, qk_nope_head_dim:]  # [seq_len, v_head_dim]

        # Construct full query: [N, qk_head_dim]
        query_full = torch.cat(
            [q_nope_i, q_pe_i], dim=-1
        )  # [N, qk_nope_head_dim + qk_rope_head_dim]

        # Construct full key: expand to num_heads, [seq_len, N, qk_head_dim]
        # k_nope and kpe are shared across heads, expand them
        k_nope_expanded = k_nope_cached.unsqueeze(1).expand(
            -1, num_heads, -1
        )  # [seq_len, N, qk_nope_head_dim]
        kpe_expanded = kpe_cached.unsqueeze(1).expand(
            -1, num_heads, -1
        )  # [seq_len, N, qk_rope_head_dim]
        key_full = torch.cat([k_nope_expanded, kpe_expanded], dim=-1)  # [seq_len, N, qk_head_dim]

        # Transpose for attention computation
        # query: [N, 1, qk_head_dim], key: [N, seq_len, qk_head_dim]
        query_t = query_full.unsqueeze(1)  # [N, 1, qk_head_dim]
        key_t = key_full.transpose(0, 1)  # [N, seq_len, qk_head_dim]

        # Compute attention scores: [N, 1, seq_len]
        attn_scores = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)

        # Value: expand to num_heads [seq_len, N, v_head_dim] -> [N, seq_len, v_head_dim]
        v_expanded = v_cached.unsqueeze(1).expand(-1, num_heads, -1)  # [seq_len, N, v_head_dim]
        v_t = v_expanded.transpose(0, 1)  # [N, seq_len, v_head_dim]

        # Compute output: [N, 1, v_head_dim] -> [N, v_head_dim]
        attn_out = torch.matmul(attn_weights, v_t).squeeze(1)  # [N, v_head_dim]

        out[i] = attn_out


def _torch_mla_context(
    q_nope: torch.Tensor,  # [total_tokens, N, qk_nope_head_dim]
    q_pe: torch.Tensor,  # [total_tokens, N, qk_rope_head_dim]
    ckv: torch.Tensor,  # [total_tokens, 1, head_dim_ckv]
    kpe: torch.Tensor,  # [total_tokens, 1, head_dim_kpe]
    mla_cache: torch.Tensor,  # [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    seq_len: torch.Tensor,
    seq_start: torch.Tensor,
    scale: float,
    head_dim_ckv: int,
    out: torch.Tensor,
) -> None:
    """Context MLA attention (multiple tokens, potentially multiple sequences)."""
    num_heads = q_nope.shape[1]
    qk_nope_head_dim = q_nope.shape[2]
    # qk_rope_head_dim = q_pe.shape[2]

    # Flatten ckv and kpe: [total_tokens, 1, D] -> [total_tokens, D]
    ckv_flat = ckv.squeeze(1)  # [total_tokens, head_dim_ckv]
    kpe_flat = kpe.squeeze(1)  # [total_tokens, head_dim_kpe]

    # Update cache first
    _update_mla_cache(
        ckv_flat, kpe_flat, mla_cache, seq_len, input_pos, cache_loc, seq_start, head_dim_ckv
    )

    # Compute attention for each sequence
    attn_outputs = []
    for idx in range(seq_len.shape[0]):
        seq_len_i = seq_len[idx].item()
        input_pos_i = input_pos[idx].item()
        cache_loc_i = cache_loc[idx].item()
        seq_start_i = seq_start[idx].item()

        if seq_len_i == 0:
            continue

        # Get query for this sequence
        q_nope_seq = q_nope[
            seq_start_i : seq_start_i + seq_len_i
        ]  # [seq_len_i, N, qk_nope_head_dim]
        q_pe_seq = q_pe[seq_start_i : seq_start_i + seq_len_i]  # [seq_len_i, N, qk_rope_head_dim]

        # Construct full query
        query_full = torch.cat([q_nope_seq, q_pe_seq], dim=-1)  # [seq_len_i, N, qk_head_dim]

        # Get cached ckv and kpe
        kv_seq_len = input_pos_i + seq_len_i
        cached_data = mla_cache[
            cache_loc_i, :kv_seq_len
        ]  # [kv_seq_len, head_dim_ckv + head_dim_kpe]
        ckv_cached = cached_data[:, :head_dim_ckv]  # [kv_seq_len, head_dim_ckv]
        kpe_cached = cached_data[:, head_dim_ckv:]  # [kv_seq_len, head_dim_kpe]

        # Split ckv into k_nope and value
        # v_head_dim = head_dim_ckv - qk_nope_head_dim
        k_nope_cached = ckv_cached[:, :qk_nope_head_dim]  # [kv_seq_len, qk_nope_head_dim]
        v_cached = ckv_cached[:, qk_nope_head_dim:]  # [kv_seq_len, v_head_dim]

        # Construct full key: expand to num_heads
        k_nope_expanded = k_nope_cached.unsqueeze(1).expand(
            -1, num_heads, -1
        )  # [kv_seq_len, N, qk_nope_head_dim]
        kpe_expanded = kpe_cached.unsqueeze(1).expand(
            -1, num_heads, -1
        )  # [kv_seq_len, N, qk_rope_head_dim]
        key_full = torch.cat(
            [k_nope_expanded, kpe_expanded], dim=-1
        )  # [kv_seq_len, N, qk_head_dim]

        # Transpose for attention: [1, N, seq_len, head_dim]
        query_t = query_full.transpose(0, 1).unsqueeze(0)  # [1, N, seq_len_i, qk_head_dim]
        key_t = key_full.transpose(0, 1).unsqueeze(0)  # [1, N, kv_seq_len, qk_head_dim]

        # Compute attention scores
        attn_scores = (
            torch.matmul(query_t, key_t.transpose(-2, -1)) * scale
        )  # [1, N, seq_len_i, kv_seq_len]

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len_i, kv_seq_len, device=q_nope.device, dtype=torch.bool),
            diagonal=kv_seq_len - seq_len_i + 1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)

        # Value: expand to num_heads
        v_expanded = v_cached.unsqueeze(1).expand(-1, num_heads, -1)  # [kv_seq_len, N, v_head_dim]
        v_t = v_expanded.transpose(0, 1).unsqueeze(0)  # [1, N, kv_seq_len, v_head_dim]

        # Compute output
        attn_out = torch.matmul(attn_weights, v_t)  # [1, N, seq_len_i, v_head_dim]
        attn_out = attn_out[0].transpose(0, 1)  # [seq_len_i, N, v_head_dim]

        attn_outputs.append(attn_out)

    # Concatenate all outputs
    if len(attn_outputs) == 0:
        out.zero_()
    elif len(attn_outputs) == 1:
        out.copy_(attn_outputs[0])
    else:
        out.copy_(torch.cat(attn_outputs, dim=0))


@torch.library.custom_op("auto_deploy::torch_cached_mla_with_cache", mutates_args=())
def torch_backend_mla_with_cache(
    # Q components (2 args)
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim] - query non-positional
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim] - query positional (RoPE applied)
    # KV components (2 args)
    ckv: torch.Tensor,  # [B, S, 1, head_dim_ckv] - compressed key/value
    kpe: torch.Tensor,  # [B, S, 1, head_dim_kpe] - key positional (RoPE applied)
    # STANDARD METADATA (same as torch_backend_mha_with_cache)
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # UNIFIED CACHE (FlashInfer layout)
    mla_cache: torch.Tensor,  # [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
    # CONSTANTS
    scale: Optional[float] = None,
    head_dim_ckv: int = 512,  # dimension split point for ckv vs kpe
) -> torch.Tensor:
    """Torch backend MLA with unified FlashInfer cache layout.

    FlashInfer MLA Cache Layout:
        mla_cache: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
        - ckv_cached = mla_cache[:, :, :head_dim_ckv]  (zero-copy slice)
        - kpe_cached = mla_cache[:, :, head_dim_ckv:]  (zero-copy slice)

    Reference: https://docs.flashinfer.ai/tutorials/kv_layout.html#mla-page-layout
    """
    # Get dimensions
    b, s = q_nope.shape[:2]
    num_heads = q_nope.shape[2]
    qk_nope_head_dim = q_nope.shape[3]
    qk_rope_head_dim = q_pe.shape[3]
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Compute v_head_dim from ckv
    v_head_dim = head_dim_ckv - qk_nope_head_dim

    # Get cleaned up metadata
    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    seq_len = seq_len[:num_seq]
    input_pos = input_pos[:num_seq]
    cache_loc = cache_loc[:num_seq]
    seq_start = cu_seqlen[:num_seq]

    # Set scale
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Define output shape: [B, S, N, v_head_dim]
    output_shape = (b, s, num_heads, v_head_dim)

    # Reshape inputs based on phase
    if s == 1:
        # Generate phase: keep [B, 1, ...] shape
        bs_view = (b, s)

        # Create output tensor
        y = q_nope.new_empty(b, num_heads, v_head_dim).contiguous()

        # Compute MLA attention
        _torch_mla_generate(
            q_nope, q_pe, ckv, kpe, mla_cache, cache_loc, input_pos, scale, head_dim_ckv, y
        )

        return y.unsqueeze(1)  # [B, 1, N, v_head_dim]
    else:
        # Context phase: flatten to [total_tokens, ...]
        bs_view = (b * s,)

        q_nope_flat = q_nope.contiguous().view(*bs_view, num_heads, qk_nope_head_dim)
        q_pe_flat = q_pe.contiguous().view(*bs_view, num_heads, qk_rope_head_dim)
        ckv_flat = ckv.contiguous().view(*bs_view, 1, head_dim_ckv)
        kpe_flat = kpe.contiguous().view(*bs_view, 1, kpe.shape[-1])

        # Create output tensor
        y = q_nope.new_empty(*bs_view, num_heads, v_head_dim).contiguous()

        # Compute MLA attention
        _torch_mla_context(
            q_nope_flat,
            q_pe_flat,
            ckv_flat,
            kpe_flat,
            mla_cache,
            input_pos,
            cache_loc,
            seq_len,
            seq_start,
            scale,
            head_dim_ckv,
            y,
        )

        return y.view(*output_shape)


@torch_backend_mla_with_cache.register_fake
def torch_backend_mla_with_cache_fake(
    # Q components
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    # KV components
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_seqlen: torch.Tensor,
    # CACHE
    mla_cache: torch.Tensor,
    # CONSTANTS
    scale: Optional[float] = None,
    head_dim_ckv: int = 512,
) -> torch.Tensor:
    """Fake implementation for torch_backend_mla_with_cache."""
    qk_nope_head_dim = q_nope.shape[-1]
    v_head_dim = head_dim_ckv - qk_nope_head_dim
    # Output: [B, S, N, v_head_dim]
    return q_nope.new_empty(
        q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
    ).contiguous()


@AttentionRegistry.register("MultiHeadLatentAttention")
class MultiHeadLatentAttention(AttentionDescriptor):
    """Attention descriptor for Multi-head Latent Attention (MLA).

    This descriptor uses the new torch_mla source op (without fused RoPE) and
    torch_cached_mla_with_cache backend op with unified FlashInfer cache layout.

    FlashInfer MLA Cache Layout:
        mla_cache: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
        - No num_heads dimension (MLA-specific optimization)
        - ckv_cached = mla_cache[:, :, :head_dim_ckv]  (zero-copy slice)
        - kpe_cached = mla_cache[:, :, head_dim_ckv:]  (zero-copy slice)

    Reference: https://docs.flashinfer.ai/tutorials/kv_layout.html#mla-page-layout
    """

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        """Get the attention layout expected by the backend."""
        return "bsnd"  # Align with TorchBackendAttention

    @classmethod
    def get_num_qkv_args(cls) -> int:
        """Get the number of qkv arguments expected by the source op."""
        return 4  # q_nope, q_pe, ckv, kpe

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        """Get the source attention op that we target for replacement."""
        return torch.ops.auto_deploy.torch_mla

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Get the cached attention op."""
        return torch.ops.auto_deploy.torch_cached_mla_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        """Get the list of standard metadata arguments."""
        # Same as TorchBackendAttention
        return ["batch_info_host", "seq_len", "input_pos", "cache_loc", "cu_seqlen"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        """Get cache initializers using unified FlashInfer MLA cache layout."""
        # Extract dimensions from source node args
        # torch_mla signature: q_nope, q_pe, ckv, kpe, ...
        q_pe_fake = source_attn_node.args[1].meta["val"]
        ckv_fake = source_attn_node.args[2].meta["val"]

        # Get dimensions
        # q_pe: [B, S, N, qk_rope_head_dim]
        # ckv: [B, S, 1, head_dim_ckv]
        head_dim_kpe = q_pe_fake.shape[-1]  # qk_rope_head_dim
        head_dim_ckv = ckv_fake.shape[-1]  # kv_lora_rank

        # Unified FlashInfer MLA cache: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
        # No num_heads dimension - MLA-specific optimization
        return {
            "mla_cache": UnpagedResourceHandler(
                head_dim_ckv + head_dim_kpe,  # unified dimension, no num_heads
                dtype=cls.resolve_cache_dtype(cache_config.dtype, ckv_fake.dtype),
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        """Get constants to pass to the cached attention op."""
        # Extract head_dim_ckv for cache slicing in the backend
        ckv_fake = source_attn_node.args[2].meta["val"]
        head_dim_ckv = ckv_fake.shape[-1]

        # Get scale from kwargs or use default
        scale = source_attn_node.kwargs.get("scale", None)

        return [scale, head_dim_ckv]
