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

"""Prefill-only OLMo-2 model for AutoDeploy.

Source: https://huggingface.co/allenai/OLMo-2-0325-32B-DPO

Key differences from HF modeling_olmo2.py:
- KV cache, training paths, dropout, and flash attention variants removed.
- Uses AD custom ops (torch_rmsnorm, torch_rope_with_explicit_cos_sin, torch_attention).
- RoPE precomputed with _ad_ prefix buffers for lift_to_meta compatibility.

Architecture notes (differs from Llama):
- Post-norm: RMSNorm applied after sublayer output, before residual addition.
- Q/K normalization: RMSNorm on Q and K projections within attention.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class Olmo2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class Olmo2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=500000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute full cos/sin table
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return full table (not sliced); indexing by position_ids happens downstream.
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class Olmo2MLP(nn.Module):
    def __init__(self, config: Olmo2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Olmo2Attention(nn.Module):
    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # Q/K normalization (unique to OLMo-2): applied to flat projections before reshape
        self.q_norm = Olmo2RMSNorm(self.num_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo2RMSNorm(self.num_kv_heads * self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Apply Q/K normalization on flat projections (before reshape)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape to [B, S, N, D] (BSND layout)
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_kv_heads, self.head_dim)
        v = v.view(B, S, self.num_kv_heads, self.head_dim)

        # Index cos/sin by position_ids: [max_seq_len, head_dim] -> [B, S, head_dim]
        cos, sin = position_embeddings
        cos = cos[position_ids]
        sin = sin[position_ids]

        # Apply RoPE (unsqueeze_dim=2 for BSND layout)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q, k, cos, sin, unsqueeze_dim=2
        )

        # Attention (handles GQA internally)
        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            layout="bsnd",
        )

        # Reshape and project output
        attn_output = attn_output.view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Olmo2DecoderLayer(nn.Module):
    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Olmo2MLP(config)
        # Post-norm: applied after sublayer output, before residual addition
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # Attention with post-norm
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with post-norm
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class Olmo2Output(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Olmo2CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class Olmo2PreTrainedModel(PreTrainedModel):
    config_class = Olmo2Config
    base_model_prefix = "model"
    _no_split_modules = ["Olmo2DecoderLayer"]
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


class Olmo2Model(Olmo2PreTrainedModel):
    def __init__(self, config: Olmo2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Olmo2RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Olmo2Output:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if position_ids is None:
            position_ids = (
                torch.arange(hidden_states.shape[1], device=hidden_states.device)
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1)
            )

        # Get full RoPE table
        position_embeddings = self.rotary_emb(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return Olmo2Output(last_hidden_state=hidden_states)


class Olmo2ForCausalLM(Olmo2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Olmo2Config):
        super().__init__(config)
        self.model = Olmo2Model(config)
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Olmo2CausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return Olmo2CausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("Olmo2Config", Olmo2ForCausalLM)
