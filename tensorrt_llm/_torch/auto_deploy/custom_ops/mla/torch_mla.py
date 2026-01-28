"""Torch reference implementations for attention."""

import math
from typing import Optional

import torch


@torch.library.custom_op("auto_deploy::torch_mla", mutates_args=())
def torch_mla(
    q_nope: torch.Tensor,  # [B, S, N, qk_nope_head_dim] or [B, N, S, ...] - query non-positional
    q_pe: torch.Tensor,  # [B, S, N, qk_rope_head_dim] or [B, N, S, ...] - query positional (RoPE applied)
    ckv: torch.Tensor,  # [B, S, 1, head_dim_ckv] or [B, 1, S, ...] - compressed key/value
    kpe: torch.Tensor,  # [B, S, 1, head_dim_kpe] or [B, 1, S, ...] - key positional (RoPE applied)
    is_causal: bool = True,
    scale: Optional[float] = None,
    layout: str = "bsnd",  # "bsnd" or "bnsd"
) -> torch.Tensor:
    """Multi-head Latent Attention (MLA) without fused RoPE.

    This op follows the torch_attention signature style and uses FlashInfer naming:
    - ckv: compressed key/value (what DeepSeek calls kv after kv_b_proj)
    - kpe: key positional encoding (k_pe after RoPE is applied)

    The attention computation follows the MLA paper:
    - Query = concat(q_nope, q_pe) expanded to all heads
    - Key = concat(k_nope, kpe) where k_nope is derived from ckv
    - Value is derived from ckv

    Args:
        q_nope: Query non-positional component [B, S, N, qk_nope_head_dim] (bsnd)
        q_pe: Query positional component with RoPE already applied [B, S, N, qk_rope_head_dim] (bsnd)
        ckv: Compressed key/value from kv_b_proj [B, S, 1, head_dim_ckv] (bsnd)
        kpe: Key positional encoding with RoPE already applied [B, S, 1, head_dim_kpe] (bsnd)
        is_causal: Whether to apply causal masking (default: True)
        scale: Softmax scale factor (default: 1/sqrt(qk_head_dim))
        layout: Input/output layout, either "bsnd" or "bnsd" (default: "bsnd")

    Returns:
        Attention output with shape [B, S, N, v_head_dim] (bsnd) or [B, N, S, v_head_dim] (bnsd)
    """
    if layout not in ("bnsd", "bsnd"):
        raise ValueError(f"layout must be 'bnsd' or 'bsnd', got {layout!r}")

    # Convert to bnsd format for computation
    if layout == "bsnd":
        q_nope = q_nope.transpose(1, 2).contiguous()  # [B, N, S, qk_nope_head_dim]
        q_pe = q_pe.transpose(1, 2).contiguous()  # [B, N, S, qk_rope_head_dim]
        ckv = ckv.transpose(1, 2).contiguous()  # [B, 1, S, head_dim_ckv]
        kpe = kpe.transpose(1, 2).contiguous()  # [B, 1, S, head_dim_kpe]

    # Get dimensions
    bs, num_heads, s_q, qk_nope_head_dim = q_nope.shape
    qk_rope_head_dim = q_pe.shape[-1]
    head_dim_ckv = ckv.shape[-1]
    head_dim_kpe = kpe.shape[-1]
    s_k = ckv.shape[2]

    # MLA uses compressed KV: ckv contains both k_nope and value_states
    # Typically: head_dim_ckv = qk_nope_head_dim + v_head_dim
    # For DeepSeek: ckv is [B, 1, S, kv_lora_rank] where kv_lora_rank includes both
    v_head_dim = head_dim_ckv - qk_nope_head_dim
    if v_head_dim <= 0:
        # If ckv doesn't contain separate k_nope and v, treat entire ckv as the compressed representation
        # This handles the absorb case where the latent representation is used directly
        v_head_dim = head_dim_ckv

    # Set scale
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Split ckv into k_nope and value_states
    if head_dim_ckv > qk_nope_head_dim:
        k_nope, value_states = torch.split(
            ckv, [qk_nope_head_dim, head_dim_ckv - qk_nope_head_dim], dim=-1
        )
    else:
        # Absorb case: use ckv directly
        k_nope = ckv
        value_states = ckv

    # Expand kpe to match num_heads (kpe is shared across heads)
    # kpe shape: [B, 1, S, head_dim_kpe] -> [B, N, S, head_dim_kpe]
    kpe_expanded = kpe.expand(bs, num_heads, s_k, head_dim_kpe)

    # Expand k_nope to match num_heads (k_nope is shared across heads)
    # k_nope shape: [B, 1, S, qk_nope_head_dim] -> [B, N, S, qk_nope_head_dim]
    k_nope_expanded = k_nope.expand(bs, num_heads, s_k, qk_nope_head_dim)

    # Expand value_states to match num_heads
    value_expanded = value_states.expand(bs, num_heads, s_k, -1)

    # Construct full query and key states
    # query_states: [B, N, S, qk_head_dim]
    query_states = torch.cat([q_nope, q_pe], dim=-1)
    # key_states: [B, N, S, qk_head_dim]
    key_states = torch.cat([k_nope_expanded, kpe_expanded], dim=-1)

    # Compute attention scores: Q @ K^T
    attn_scores = (
        torch.matmul(query_states, key_states.transpose(-2, -1)) * scale
    )  # [B, N, s_q, s_k]

    # Apply causal mask if specified
    if is_causal and s_q == s_k:
        causal_mask = torch.triu(
            torch.ones(s_q, s_k, device=q_nope.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Compute attention weights and output
    attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_nope.dtype)
    attn_out = torch.matmul(attn_weights, value_expanded)  # [B, N, s_q, v_head_dim]

    # Convert back to requested layout
    if layout == "bsnd":
        return attn_out.transpose(1, 2).contiguous()  # [B, S, N, v_head_dim]
    else:
        return attn_out.contiguous()  # [B, N, S, v_head_dim]


@torch_mla.register_fake
def torch_mla_fake(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    is_causal: bool = True,
    scale: Optional[float] = None,
    layout: str = "bsnd",
) -> torch.Tensor:
    """Fake implementation for torch_mla."""
    # Compute v_head_dim from ckv
    qk_nope_head_dim = q_nope.shape[-1]
    head_dim_ckv = ckv.shape[-1]
    v_head_dim = (
        head_dim_ckv - qk_nope_head_dim if head_dim_ckv > qk_nope_head_dim else head_dim_ckv
    )

    # Output shape depends on layout
    if layout == "bsnd":
        # Input: [B, S, N, D], Output: [B, S, N, v_head_dim]
        return q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
        ).contiguous()
    else:
        # Input: [B, N, S, D], Output: [B, N, S, v_head_dim]
        return q_nope.new_empty(
            q_nope.shape[0], q_nope.shape[1], q_nope.shape[2], v_head_dim
        ).contiguous()
