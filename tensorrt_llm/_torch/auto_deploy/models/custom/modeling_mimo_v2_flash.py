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

"""Slimmed down PyTorch MiMo-V2-Flash model implementation for auto_deploy export.

Source:
https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

This implementation differs from the original HuggingFace version in the following ways:
* Bundled config class (model requires trust_remote_code in transformers)
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)

Architecture highlights:
* Hybrid attention: alternating full causal and sliding window attention layers
* Different Q/K head_dim (192) vs V head_dim (128) with partial rotary (64 dims)
* Two RoPE embeddings: full attention (theta=5M) and SWA (theta=10K)
* Attention sink bias on SWA layers
* MoE with 256 experts, top-8 noaux_tc routing (layer 0 is dense)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType


class MiMoV2FlashConfig(PretrainedConfig):
    """Configuration class for MiMo-V2-Flash model.

    Bundled with the custom model implementation since the model uses
    trust_remote_code and is not natively registered in transformers.
    """

    model_type = "mimo_v2_flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 152576,
        hidden_size: int = 4096,
        intermediate_size: int = 16384,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 4,
        head_dim: int = 192,
        v_head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        layernorm_epsilon: float = 1e-5,
        rope_theta: float = 5000000.0,
        partial_rotary_factor: float = 0.334,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        # Hybrid attention
        hybrid_layer_pattern: Optional[List[int]] = None,
        # SWA parameters
        swa_rope_theta: float = 10000.0,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 8,
        swa_head_dim: int = 192,
        swa_v_head_dim: int = 128,
        sliding_window: int = 128,
        sliding_window_size: int = 128,
        # Sink bias
        add_swa_attention_sink_bias: bool = True,
        add_full_attention_sink_bias: bool = False,
        # MoE parameters
        moe_layer_freq: Optional[List[int]] = None,
        moe_intermediate_size: int = 2048,
        n_routed_experts: int = 256,
        n_shared_experts: Optional[int] = None,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        scoring_func: str = "sigmoid",
        n_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        routed_scaling_factor: Optional[float] = None,
        # Other
        initializer_range: float = 0.02,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.layernorm_epsilon = layernorm_epsilon
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Hybrid
        self.hybrid_layer_pattern = hybrid_layer_pattern
        # SWA
        self.swa_rope_theta = swa_rope_theta
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window_size
        # Sink
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        # MoE
        self.moe_layer_freq = moe_layer_freq
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.n_group = n_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.routed_scaling_factor = routed_scaling_factor
        self.initializer_range = initializer_range

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Register config with AutoConfig
try:
    AutoConfig.register("mimo_v2_flash", MiMoV2FlashConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("mimo_v2_flash", MiMoV2FlashConfig)
    except ValueError:
        pass


class MiMoV2FlashRMSNorm(nn.Module):
    """RMS Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class MiMoV2FlashRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for partial rotary dims.

    Precomputes cos/sin for rope_dim and caches with _ad_ prefix.
    The forward method slices by position_ids once so layers don't repeat the work.
    """

    def __init__(
        self,
        rope_dim: int,
        max_position_embeddings: int = 262144,
        base: float = 10000.0,
    ):
        super().__init__()
        self.rope_dim = rope_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return full cached cos/sin (not sliced) for export compatibility
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class MiMoV2FlashMLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(self, config: MiMoV2FlashConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiMoV2FlashMoEGate(nn.Module):
    """noaux_tc MoE gating using fused TRT-LLM ops.

    Uses trtllm.noaux_tc_op for fused sigmoid + bias + group top-k + normalize + scale.
    """

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = (
            config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0
        )
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, h = hidden_states.shape
        hidden_flat = hidden_states.view(-1, h)

        # Router GEMM
        if self.weight.dtype == torch.float32:
            router_logits = F.linear(hidden_flat.float(), self.weight)
        else:
            router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_flat, self.weight.t(), bias=None, out_dtype=torch.float32
            )

        # Fused routing: sigmoid + bias + group top-k + normalize + scale
        topk_weights, topk_indices = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            self.e_score_correction_bias,
            self.n_group,
            self.topk_group,
            self.top_k,
            self.routed_scaling_factor,
        )

        return topk_indices, topk_weights


class MiMoV2FlashMoE(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                MiMoV2FlashMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MiMoV2FlashMoEGate(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)

        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states.view(-1, hidden_states.shape[-1]),
            topk_indices,
            topk_weights,
            w1_weight=[expert.gate_proj.weight for expert in self.experts],
            w2_weight=[expert.down_proj.weight for expert in self.experts],
            w3_weight=[expert.up_proj.weight for expert in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )

        return final_hidden_states.view(*orig_shape)


class MiMoV2FlashAttention(nn.Module):
    """Hybrid attention supporting both full causal and sliding window modes.

    Handles:
    * Different Q/K head_dim vs V head_dim
    * Partial rotary: only rope_dim of head_dim gets RoPE
    * GQA with different head counts for full vs SWA layers
    * Optional attention sink bias
    """

    def __init__(self, config: MiMoV2FlashConfig, is_swa: bool, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_swa = is_swa

        if is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.num_heads = config.swa_num_attention_heads
            self.num_kv_heads = config.swa_num_key_value_heads
            self.sliding_window = config.sliding_window
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads
            self.sliding_window = None

        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.scaling = self.head_dim ** (-0.5)

        # Q/K/V/O projections with different dims for value
        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.v_head_dim
        o_size = self.num_heads * self.v_head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, k_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, v_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(o_size, config.hidden_size, bias=False)

        # Attention sink bias (optional, per-head learned parameter)
        has_sink = (config.add_full_attention_sink_bias and not is_swa) or (
            config.add_swa_attention_sink_bias and is_swa
        )
        if has_sink:
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.num_heads), requires_grad=False
            )
        else:
            self.attention_sink_bias = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V to BSND layout
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Partial rotary: split into rope and nope parts
        q_rope, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_rope, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)

        # Slice cos/sin from full cache by position_ids
        cos, sin = position_embeddings  # Full table: [max_seq_len, rope_dim]
        cos = cos[position_ids]  # [B, S, rope_dim]
        sin = sin[position_ids]  # [B, S, rope_dim]

        # Apply RoPE to rope parts only (BSND layout, unsqueeze_dim=2)
        q_rope, k_rope = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_rope, k_rope, cos, sin, 2
        )

        # Reassemble Q/K
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)

        # V: project and pad to head_dim so Q/K/V have uniform last dim in BSND.
        # AD's cached attention transform expects uniform head_dim across Q/K/V.
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.v_head_dim)
        v = F.pad(v, (0, self.head_dim - self.v_head_dim))  # [B, S, N_kv, head_dim]

        # Attention with GQA support, optional sinks and sliding window (BSND layout)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,
            self.attention_sink_bias,
            self.sliding_window,
            None,  # logit_cap
            "bsnd",
        )

        # Slice back to v_head_dim and reshape
        attn_output = attn_output[..., : self.v_head_dim]  # [B, S, N, v_head_dim]
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


class MiMoV2FlashDecoderLayer(nn.Module):
    """Decoder layer with hybrid attention and dense/MoE FFN."""

    def __init__(self, config: MiMoV2FlashConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention type based on hybrid pattern
        is_swa = (
            config.hybrid_layer_pattern is not None and config.hybrid_layer_pattern[layer_idx] == 1
        )
        self.is_swa = is_swa
        self.self_attn = MiMoV2FlashAttention(config, is_swa=is_swa, layer_idx=layer_idx)

        # MLP or MoE based on moe_layer_freq
        use_moe = (
            config.n_routed_experts is not None
            and config.moe_layer_freq is not None
            and config.moe_layer_freq[layer_idx] == 1
        )
        if use_moe:
            self.mlp = MiMoV2FlashMoE(config)
        else:
            self.mlp = MiMoV2FlashMLP(config)

        self.input_layernorm = MiMoV2FlashRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = MiMoV2FlashRMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class MiMoV2FlashOutput(ModelOutput):
    """Output for MiMoV2FlashModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class MiMoV2FlashCausalLMOutput(ModelOutput):
    """Output for MiMoV2FlashForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class MiMoV2FlashPreTrainedModel(PreTrainedModel):
    """Base class for MiMo-V2-Flash models."""

    config_class = MiMoV2FlashConfig
    base_model_prefix = "model"
    _no_split_modules = ["MiMoV2FlashDecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MiMoV2FlashModel(MiMoV2FlashPreTrainedModel):
    """MiMo-V2-Flash transformer decoder model."""

    def __init__(self, config: MiMoV2FlashConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                MiMoV2FlashDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = MiMoV2FlashRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

        # Compute rope_dim from head_dim (same for full and SWA since head_dim matches)
        rope_dim = int(config.head_dim * config.partial_rotary_factor)

        # Two RoPE embeddings: full attention and SWA use different theta
        self.rotary_emb = MiMoV2FlashRotaryEmbedding(
            rope_dim=rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.swa_rotary_emb = MiMoV2FlashRotaryEmbedding(
            rope_dim=rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.swa_rope_theta,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> MiMoV2FlashOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cast to compute dtype for FP8 models
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # Compute position embeddings once (full cache tables, not sliced)
        full_position_embeddings = self.rotary_emb(inputs_embeds)
        swa_position_embeddings = self.swa_rotary_emb(inputs_embeds)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            pos_emb = swa_position_embeddings if decoder_layer.is_swa else full_position_embeddings
            hidden_states = decoder_layer(hidden_states, position_ids, pos_emb)

        hidden_states = self.norm(hidden_states)

        return MiMoV2FlashOutput(last_hidden_state=hidden_states)


class MiMoV2FlashForCausalLM(MiMoV2FlashPreTrainedModel, GenerationMixin):
    """MiMo-V2-Flash model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = MiMoV2FlashModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> MiMoV2FlashCausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return MiMoV2FlashCausalLMOutput(logits=logits)


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("MiMoV2FlashConfig", MiMoV2FlashForCausalLM)
