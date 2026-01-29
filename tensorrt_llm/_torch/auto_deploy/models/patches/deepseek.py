import types
import warnings
from typing import Dict, Optional

import torch
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache


def deepseek_v3_attention(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.IntTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    """DeepSeekV3Attention forward function rewritten to wrap MLA as a custom op.

    This version uses FlashInfer-compatible compressed KV cache:
    - compressed_kv: latent before kv_b_proj [B, S, kv_lora_rank]
    - kpe: key positional encoding (after RoPE) [B, S, 1, qk_rope_head_dim]
    - kv_b_proj_weight: projection weights passed to attention op

    The torch_mla op handles kv_b_proj expansion internally:
    - Prefill: expand compressed_kv -> full K, V for attention
    - Generate: use weight absorption for efficiency
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    bsz, q_len, _ = hidden_states.size()

    # =========================================================================
    # Query projection
    # =========================================================================
    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    # Shape: [B, S, N, q_head_dim] (BSND layout)
    q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    # =========================================================================
    # KV projection - keep compressed form for FlashInfer cache
    # =========================================================================
    kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
    compressed_kv, k_pe = torch.split(
        kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )

    # Apply layernorm to compressed_kv (this is what we store in cache)
    # Shape: [B, S, kv_lora_rank]
    compressed_kv = self.kv_a_layernorm(compressed_kv)

    # k_pe: [B, S, 1, qk_rope_head_dim] (BSND layout, shared across heads)
    k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

    kv_seq_len = q_len
    if past_key_value is not None:
        raise ValueError("past_key_value is not supported")

    # =========================================================================
    # Apply RoPE BEFORE the MLA call (separation of concerns)
    # =========================================================================
    cos, sin = self.rotary_emb(hidden_states, seq_len=kv_seq_len)
    cos = cos[position_ids]  # [B, S, head_dim]
    sin = sin[position_ids]  # [B, S, head_dim]

    # Apply RoPE to q_pe and k_pe using existing custom op
    # Input layout for torch_rope_with_qk_interleaving: [B, S, N, D] with unsqueeze_dim=2
    q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q_pe,
        k_pe,
        cos,
        sin,
        2,  # unsqueeze_dim=2 for BSND layout
    )

    # =========================================================================
    # Call MLA with compressed KV (FlashInfer-compatible)
    # =========================================================================
    # torch_mla signature (5 tensor args):
    # - q_nope: [B, S, N, qk_nope_head_dim]
    # - q_pe: [B, S, N, qk_rope_head_dim] (RoPE already applied)
    # - compressed_kv: [B, S, kv_lora_rank] (BEFORE kv_b_proj)
    # - kpe: [B, S, 1, qk_rope_head_dim] (RoPE already applied)
    # - kv_b_proj_weight: [num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    attn_output = torch.ops.auto_deploy.torch_mla(
        q_nope,  # [B, S, N, qk_nope_head_dim]
        q_pe_rotated,  # [B, S, N, qk_rope_head_dim]
        compressed_kv,  # [B, S, kv_lora_rank] - compressed latent
        kpe,  # [B, S, 1, qk_rope_head_dim]
        self.kv_b_proj.weight,  # [out_features, in_features] = [N*(qk_nope+v), kv_lora_rank]
        True,  # is_causal
        self.softmax_scale,
        "bsnd",  # layout
    )

    # Output shape: [B, S, N, v_head_dim] -> [B, S, N * v_head_dim]
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


@torch.inference_mode()
def deepseek_v3_moe(self, hidden_states):
    """DeepSeekV3MoE forward function rewritten in Mixtral style to enable torch export."""

    selected_experts, routing_weights, *_ = self.gate(hidden_states)
    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[expert.gate_proj.weight for expert in self.experts],
        w2_weight=[expert.down_proj.weight for expert in self.experts],
        w3_weight=[expert.up_proj.weight for expert in self.experts],
    )

    if self.config.n_shared_experts is not None:
        final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)

    return final_hidden_states.to(hidden_states.dtype)


def deepseek_v3_rope(self, x, seq_len=None):
    """DeepSeekV3 Rotary Embedding forward function rewritten to enable torch export.
    We return the full cached cos and sin values, instead of slicing them based on seq_len as this
    would cause an issue during the generate phase (when seq_len=1 from input_ids). We also move the cos
    sin buffers to appropriate device to enable export.
    """

    return (
        self.cos_cached.to(dtype=x.dtype).to(device=x.device),
        self.sin_cached.to(dtype=x.dtype).to(device=x.device),
    )


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "DeepseekV3MoE": deepseek_v3_moe,
    "DeepseekV2MoE": deepseek_v3_moe,
    "DeepseekV3RotaryEmbedding": deepseek_v3_rope,
    "DeepseekV3YarnRotaryEmbedding": deepseek_v3_rope,
    "DeepseekV2RotaryEmbedding": deepseek_v3_rope,
    "DeepseekV2YarnRotaryEmbedding": deepseek_v3_rope,
    "DeepseekV3Attention": deepseek_v3_attention,
}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES.keys():
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
