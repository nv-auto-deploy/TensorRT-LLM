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

"""Prefill-only PyTorch Jamba model for auto_deploy export.

Source: https://huggingface.co/ai21labs/AI21-Jamba-Reasoning-3B

This implementation differs from the original HuggingFace version:
* Simplified for prefill-only inference (no KV caching, no Mamba state caching)
* Uses auto_deploy custom ops for export compatibility
* Removed flash attention variants, training paths, dropout
* Removed MoE load-balancing loss

Supports the full Jamba architecture including:
* Hybrid Mamba v1 / Attention layers (pattern determined by config)
* Optional Sparse MoE (when num_experts > 1 per layer)
* Mamba v1 selective scan with layernorm on dt/B/C projections
* GQA / MQA attention (no rotary embeddings)
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

import tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_causal_conv  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_mamba_v1  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

try:
    from transformers.models.jamba.configuration_jamba import JambaConfig
except ImportError:
    from transformers import AutoConfig, PretrainedConfig

    class JambaConfig(PretrainedConfig):
        """Fallback config for Jamba when not in installed transformers."""

        model_type = "jamba"

        def __init__(
            self,
            vocab_size=65536,
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_act="silu",
            max_position_embeddings=262144,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            num_experts=1,
            num_experts_per_tok=2,
            expert_layer_offset=1,
            expert_layer_period=2,
            attn_layer_offset=4,
            attn_layer_period=8,
            attention_dropout=0.0,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_dt_rank=256,
            mamba_expand=2,
            mamba_proj_bias=False,
            mamba_conv_bias=True,
            use_mamba_kernels=True,
            pad_token_id=0,
            tie_word_embeddings=False,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = num_key_value_heads
            self.hidden_act = hidden_act
            self.max_position_embeddings = max_position_embeddings
            self.initializer_range = initializer_range
            self.rms_norm_eps = rms_norm_eps
            self.use_cache = use_cache
            self.num_experts = num_experts
            self.num_experts_per_tok = num_experts_per_tok
            self.expert_layer_offset = expert_layer_offset
            self.expert_layer_period = expert_layer_period
            self.attn_layer_offset = attn_layer_offset
            self.attn_layer_period = attn_layer_period
            self.attention_dropout = attention_dropout
            self.mamba_d_state = mamba_d_state
            self.mamba_d_conv = mamba_d_conv
            self.mamba_dt_rank = mamba_dt_rank
            self.mamba_expand = mamba_expand
            self.mamba_proj_bias = mamba_proj_bias
            self.mamba_conv_bias = mamba_conv_bias
            self.use_mamba_kernels = use_mamba_kernels
            super().__init__(
                pad_token_id=pad_token_id,
                tie_word_embeddings=tie_word_embeddings,
                **kwargs,
            )

        @property
        def layers_block_type(self):
            return [
                "attention"
                if (i - self.attn_layer_offset) % self.attn_layer_period == 0
                else "mamba"
                for i in range(self.num_hidden_layers)
            ]

        @property
        def layers_num_experts(self):
            return [
                self.num_experts
                if (i - self.expert_layer_offset) % self.expert_layer_period == 0
                else 1
                for i in range(self.num_hidden_layers)
            ]

    AutoConfig.register("jamba", JambaConfig, exist_ok=True)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class JambaModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class JambaCausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class JambaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


# ---------------------------------------------------------------------------
# Attention (no positional encoding — Jamba attention layers don't use RoPE)
# ---------------------------------------------------------------------------


class JambaAttention(nn.Module):
    """GQA/MQA attention without positional encoding."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [B, S, N, D] for bsnd layout
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # No RoPE applied — Jamba attention layers use no positional encoding

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim),
            layout="bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ---------------------------------------------------------------------------
# Mamba v1 Mixer
# ---------------------------------------------------------------------------


class JambaMambaMixer(nn.Module):
    """Mamba v1 selective SSM mixer using AD custom ops.

    Architecture:
    1. in_proj: [hidden_size] -> [2 * intermediate_size] (gated)
    2. conv1d on first half (causal, depthwise)
    3. SiLU activation
    4. x_proj: [intermediate_size] -> [dt_rank + 2*d_state]
    5. LayerNorms on dt, B, C
    6. dt_proj: [dt_rank] -> [intermediate_size], then softplus
    7. Mamba v1 selective scan + D skip connection
    8. Gate: multiply by silu(gate)
    9. out_proj: [intermediate_size] -> [hidden_size]
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.act = ACT2FN[config.hidden_act]

        # Input projection: hidden_size -> 2 * intermediate_size (for gating)
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)

        # Causal 1D convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        # Selective projection: intermediate_size -> dt_rank + 2*d_state
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False,
        )

        # Time step projection: dt_rank -> intermediate_size
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # SSM parameters
        A = torch.arange(1, self.ssm_state_size + 1)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))

        # Output projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        # Jamba-specific: layernorms on dt, B, C projections
        self.dt_layernorm = JambaRMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = JambaRMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated projection
        projected_states = self.in_proj(hidden_states)  # [B, L, 2*D_inner]
        hidden_states_inner, gate = projected_states.chunk(2, dim=-1)  # each [B, L, D_inner]

        # 2. Causal 1D convolution using AD custom op
        hidden_states_inner = torch.ops.auto_deploy.torch_causal_conv1d(
            hidden_states_inner,
            self.conv1d.weight,
            self.conv1d.bias,
            self.conv1d.stride[0],
            self.conv1d.padding[0],
            self.conv1d.dilation[0],
            self.conv1d.groups,
            self.conv1d.padding_mode,
        )
        hidden_states_inner = self.act(hidden_states_inner)  # [B, L, D_inner]

        # 3. Selective projection -> dt, B, C
        ssm_parameters = self.x_proj(hidden_states_inner)  # [B, L, dt_rank + 2*d_state]
        time_step, B, C = torch.split(
            ssm_parameters,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1,
        )

        # 4. LayerNorms on dt, B, C (Jamba-specific)
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        # 5. Time step projection + softplus
        discrete_time_step = self.dt_proj(time_step)  # [B, L, D_inner]
        discrete_time_step = F.softplus(discrete_time_step)

        # 6. Selective scan using Mamba v1 custom op
        A = -torch.exp(self.A_log.float())  # [D_inner, d_state]
        scan_output = torch.ops.auto_deploy.torch_mamba_v1_selective_scan(
            hidden_states_inner,
            A,
            B,
            C,
            self.D,
            discrete_time_step,
        )  # [B, L, D_inner]

        # 7. Gating: multiply by activated gate
        scan_output = scan_output * self.act(gate)

        # 8. Output projection
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states


# ---------------------------------------------------------------------------
# MLP and optional MoE
# ---------------------------------------------------------------------------


class JambaMLP(nn.Module):
    """Gated MLP with SiLU activation."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class JambaSparseMoeBlock(nn.Module):
    """Sparse MoE block with top-k routing (used when num_experts > 1)."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([JambaMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router: softmax then top-k
        router_logits = self.router(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        # Dispatch to experts
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states_flat[top_x]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        return final_hidden_states.view(batch_size, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Decoder layers
# ---------------------------------------------------------------------------


class JambaAttentionDecoderLayer(nn.Module):
    """Decoder layer with attention + MLP/MoE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        self.self_attn = JambaAttention(config, layer_idx)

        if num_experts > 1:
            self.feed_forward = JambaSparseMoeBlock(config)
        else:
            self.feed_forward = JambaMLP(config)

        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)

        return hidden_states


class JambaMambaDecoderLayer(nn.Module):
    """Decoder layer with Mamba v1 mixer + MLP/MoE."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        num_experts = config.layers_num_experts[layer_idx]
        self.mamba = JambaMambaMixer(config, layer_idx)

        if num_experts > 1:
            self.feed_forward = JambaSparseMoeBlock(config)
        else:
            self.feed_forward = JambaMLP(config)

        self.input_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_ff_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Mamba
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states = residual + self.feed_forward(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

LAYER_TYPE_MAP = {
    "attention": JambaAttentionDecoderLayer,
    "mamba": JambaMambaDecoderLayer,
}


class JambaPreTrainedModel(PreTrainedModel):
    config_class = JambaConfig
    base_model_prefix = "model"
    _no_split_modules = ["JambaAttentionDecoderLayer", "JambaMambaDecoderLayer"]
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, JambaRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, JambaMambaMixer):
            A = torch.arange(1, module.ssm_state_size + 1)[None, :]
            A = A.expand(module.intermediate_size, -1).contiguous()
            module.A_log.data.copy_(torch.log(A))
            module.D.data.fill_(1.0)


class JambaModel(JambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        layers = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layers_block_type[i]
            layer_cls = LAYER_TYPE_MAP[layer_type]
            layers.append(layer_cls(config, layer_idx=i))
        self.layers = nn.ModuleList(layers)

        self.final_layernorm = JambaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JambaModelOutput:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.final_layernorm(hidden_states)

        return JambaModelOutput(last_hidden_state=hidden_states)


class JambaForCausalLM(JambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = JambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> JambaCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return JambaCausalLMOutput(logits=logits)


AutoModelForCausalLMFactory.register_custom_model_cls("JambaConfig", JambaForCausalLM)
