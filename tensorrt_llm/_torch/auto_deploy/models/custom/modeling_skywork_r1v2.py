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

"""Slimmed down PyTorch Skywork-R1V2 model implementation for auto_deploy export.

Source:
https://huggingface.co/Skywork/Skywork-R1V2-38B

Skywork-R1V2-38B is a multimodal VLM (InternVL-based) with a Qwen2 LLM backbone.
For AutoDeploy, only the LLM (text) path is exported; the vision tower stays in eager.

This implementation differs from the original HuggingFace version in the following ways:
* Only the LLM backbone is instantiated (vision model is not needed for AD export)
* Simplified for prefill-only inference (no KV caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants (uses torch_attention custom op)
* Removed gradient checkpointing and training code paths
* Removed attention dropout (inference only)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import Qwen2Config
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# ---------------------------------------------------------------------------
# Load SkyworkChatConfig from the HF checkpoint via trust_remote_code.
#
# The config class is defined in configuration_skywork_chat.py inside the HF
# repo (model_type = "skywork_chat").  It is NOT in standard transformers, so
# we load it from the local HF cache.  tie_word_embeddings is already False in
# the checkpoint's config.json at both the top level and inside llm_config, so
# no override is needed here.
# ---------------------------------------------------------------------------


def _load_skywork_chat_config_cls():
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            "Skywork/Skywork-R1V2-38B", trust_remote_code=True, local_files_only=True
        )
        return type(cfg)
    except Exception:
        return None


SkyworkChatConfig = _load_skywork_chat_config_cls()


# ---------------------------------------------------------------------------
# Model components (Qwen2-based LLM backbone)
# ---------------------------------------------------------------------------


class SkyworkR1V2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class SkyworkR1V2RotaryEmbedding(nn.Module):
    """Standard RoPE with precomputed cos/sin cache.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
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
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        # Return full cached tables; slicing by position_ids happens downstream in attention.
        return cos, sin


class SkyworkR1V2MLP(nn.Module):
    """SwiGLU MLP (identical to Qwen2)."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SkyworkR1V2Attention(nn.Module):
    """Grouped Query Attention for Qwen2 backbone.

    Qwen2 uses bias on Q/K/V projections but no bias on O projection.
    No per-head Q/K normalization (unlike Qwen3).
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        # Slice full RoPE tables by position_ids here (D3 convention).
        cos, sin = position_embeddings
        cos = cos[position_ids]  # [B, S, head_dim]
        sin = sin[position_ids]  # [B, S, head_dim]

        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q,
            k,
            cos,
            sin,
            2,
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,
            None,  # sinks
            None,  # sliding_window
            None,  # logit_cap
            "bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class SkyworkR1V2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = SkyworkR1V2Attention(config, layer_idx=layer_idx)
        self.mlp = SkyworkR1V2MLP(config)
        self.input_layernorm = SkyworkR1V2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SkyworkR1V2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Dataclass outputs
# ---------------------------------------------------------------------------


@dataclass
class SkyworkR1V2Output(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class SkyworkR1V2CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Transformer model (language_model.model.*)
# ---------------------------------------------------------------------------


class SkyworkR1V2PreTrainedModel(PreTrainedModel):
    config_class = SkyworkChatConfig
    base_model_prefix = "language_model"
    _no_split_modules = ["SkyworkR1V2DecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SkyworkR1V2TransformerModel(nn.Module):
    """Qwen2 transformer body (maps to language_model.model.*)."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                SkyworkR1V2DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = SkyworkR1V2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = SkyworkR1V2RotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SkyworkR1V2Output:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        assert position_ids is not None, "position_ids must be provided for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings, position_ids)

        hidden_states = self.norm(hidden_states)
        return SkyworkR1V2Output(last_hidden_state=hidden_states)


class SkyworkR1V2LanguageModel(nn.Module):
    """Wraps transformer + lm_head (maps to language_model.*)."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.model = SkyworkR1V2TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SkyworkR1V2CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return SkyworkR1V2CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Top-level model (registered with AD)
# ---------------------------------------------------------------------------


class SkyworkR1V2ForCausalLM(SkyworkR1V2PreTrainedModel, GenerationMixin):
    """Skywork-R1V2 model for AutoDeploy (text-only, no vision tower).

    Weight hierarchy matches the HF checkpoint:
      language_model.model.embed_tokens.weight
      language_model.model.layers.X.self_attn.{q,k,v}_proj.{weight,bias}
      language_model.model.layers.X.self_attn.o_proj.weight
      language_model.model.layers.X.mlp.{gate_proj,up_proj,down_proj}.weight
      language_model.model.layers.X.{input_layernorm,post_attention_layernorm}.weight
      language_model.model.norm.weight
      language_model.lm_head.weight

    Vision weights (vision_model.*, mlp1.*) are silently skipped during loading.
    """

    def __init__(self, config: SkyworkChatConfig, **kwargs):
        super().__init__(config)
        llm_config = config.llm_config
        self.language_model = SkyworkR1V2LanguageModel(llm_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def get_decoder(self):
        return self.language_model.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> SkyworkR1V2CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


# Register with AutoModelForCausalLMFactory
AutoModelForCausalLMFactory.register_custom_model_cls("SkyworkChatConfig", SkyworkR1V2ForCausalLM)
