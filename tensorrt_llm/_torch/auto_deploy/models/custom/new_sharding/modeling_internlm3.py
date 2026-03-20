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

"""InternLM3 model with explicit sharding hint ops.

This is a port of modeling_internlm3.py where shardable operations use
AutoDeploy custom ops with sharding hint kwargs. The FX graph is a complete
specification of how this model should be sharded; ``apply_sharding_hints``
applies deterministic, node-local sharding from those hints.

Shardable custom ops used:
  - torch.ops.auto_deploy.torch_linear_simple  (tp_mode, tp_min_local_shape, layer_type)
  - torch.ops.auto_deploy.view                 (tp_scaled_dim, layer_type)
  - torch.ops.auto_deploy.all_reduce           (identity / dist.all_reduce, layer_type)

Config is loaded from the HF checkpoint via trust_remote_code=True (not bundled here).

The InternLM3 model uses GQA with SwiGLU MLP, RMSNorm, and dynamic NTK-scaled RoPE.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class InternLM3RMSNorm(nn.Module):
    """RMS Normalization using AutoDeploy torch_rmsnorm reference op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class InternLM3RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for InternLM3.

    Supports all rope types (default, dynamic, linear, etc.) via
    transformers ROPE_INIT_FUNCTIONS. Precomputes and caches cos/sin values.
    Slices by position_ids once and returns pre-sliced cos/sin to all layers.

    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(self, config):
        super().__init__()
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
        else:
            rope_type = "default"

        inv_freq, self.attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, device=None)

        max_pos = config.max_position_embeddings
        t = torch.arange(max_pos, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos() * self.attention_scaling, persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin() * self.attention_scaling, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._ad_cos_cached.to(dtype=x.dtype, device=x.device)
        sin = self._ad_sin_cached.to(dtype=x.dtype, device=x.device)
        return cos[position_ids], sin[position_ids]


class InternLM3MLP(nn.Module):
    """MLP layer for InternLM3 (SwiGLU) with sharding hints.

    Sharding strategy:
      gate_proj -> colwise
      up_proj   -> colwise
      down_proj -> rowwise + all_reduce
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias,
            tp_mode="colwise",
            layer_type="mlp",
        )
        up = torch.ops.auto_deploy.torch_linear_simple(
            x,
            self.up_proj.weight,
            self.up_proj.bias,
            tp_mode="colwise",
            layer_type="mlp",
        )
        down = torch.ops.auto_deploy.torch_linear_simple(
            self.act_fn(gate) * up,
            self.down_proj.weight,
            self.down_proj.bias,
            tp_mode="rowwise",
            layer_type="mlp",
        )
        down = torch.ops.auto_deploy.all_reduce(down, layer_type="mlp")
        return down


class InternLM3Attention(nn.Module):
    """Grouped Query Attention for InternLM3 with sharding hints.

    Uses AD canonical ops for attention and RoPE. GQA is handled natively
    by torch_attention — no repeat_kv needed.

    Sharding strategy:
      q_proj -> colwise (+ tp_min_local_shape for GQA)
      k_proj -> colwise (+ tp_min_local_shape for GQA)
      v_proj -> colwise (+ tp_min_local_shape for GQA)
      view   -> tp_scaled_dim=2 (head count dimension)
      o_proj -> rowwise + all_reduce
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.q_proj.weight,
            self.q_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.k_proj.weight,
            self.k_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.v_proj.weight,
            self.v_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )

        q = torch.ops.auto_deploy.view(
            q,
            [bsz, q_len, self.num_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.view(
            k,
            [bsz, q_len, self.num_kv_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.view(
            v,
            [bsz, q_len, self.num_kv_heads, self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )

        cos, sin = position_embeddings

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
            None,
            0.0,
            True,
            self.scaling,
            None,
            None,
            None,
            "bsnd",
        )

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )

        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output,
            self.o_proj.weight,
            self.o_proj.bias,
            tp_mode="rowwise",
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")

        return attn_output


class InternLM3DecoderLayer(nn.Module):
    """Transformer decoder layer for InternLM3."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = InternLM3Attention(config, layer_idx=layer_idx)
        self.mlp = InternLM3MLP(config)
        self.input_layernorm = InternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = InternLM3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@dataclass
class InternLM3Output(ModelOutput):
    """Output for InternLM3Model."""

    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class InternLM3CausalLMOutput(ModelOutput):
    """Output for InternLM3ForCausalLM."""

    logits: Optional[torch.FloatTensor] = None


class InternLM3PreTrainedModel(PreTrainedModel):
    """Base class for InternLM3 models."""

    base_model_prefix = "model"
    _no_split_modules = ["InternLM3DecoderLayer"]
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


class InternLM3Model(InternLM3PreTrainedModel):
    """InternLM3 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                InternLM3DecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = InternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = InternLM3RotaryEmbedding(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        **kwargs,
    ) -> InternLM3Output:
        assert position_ids is not None, "position_ids must be provided for AD export"

        inputs_embeds = self.embed_tokens(input_ids)

        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return InternLM3Output(last_hidden_state=hidden_states)


class InternLM3ForCausalLM(InternLM3PreTrainedModel, GenerationMixin):
    """InternLM3 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.model = InternLM3Model(config)
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
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        **kwargs,
    ) -> InternLM3CausalLMOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        return InternLM3CausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("InternLM3Config", InternLM3ForCausalLM)
