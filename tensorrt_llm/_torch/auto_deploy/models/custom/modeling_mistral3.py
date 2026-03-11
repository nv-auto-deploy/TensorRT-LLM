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

"""Mistral3 (Mistral-Small-3.1) model for auto_deploy (text + vision).

Reference HF modeling file:
  transformers/models/mistral3/modeling_mistral3.py

This implementation differs from the HuggingFace original in the following ways:
  * The text backbone uses AD canonical ops (torch_rmsnorm, torch_attention,
    torch_rope_with_explicit_cos_sin) for export compatibility.
  * Cache-related code paths have been removed (prefill-only).
  * Training-related code paths have been removed.
  * Unnecessary output fields have been removed.
  * Vision tower (Pixtral) is imported from transformers and kept in eager mode.

The Mistral3 text model is a standard Mistral architecture with:
  * Grouped Query Attention (GQA)
  * SwiGLU MLP activation
  * RMSNorm normalization
  * Standard Rotary Position Embeddings (RoPE)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

# =============================================================================
# Text Model Components (exported via torch.export with AD canonical ops)
# =============================================================================


class Mistral3TextRMSNorm(nn.Module):
    """RMS Normalization using AD torch_rmsnorm canonical op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class Mistral3TextRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for the Mistral text backbone.

    Precomputes and caches cos/sin values. Returns full table (not sliced);
    slicing by position_ids happens downstream.
    Uses _ad_ prefix for buffer names to work with AutoDeploy's lift_to_meta.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 1000000000.0,
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
        return cos[position_ids], sin[position_ids]


class Mistral3TextMLP(nn.Module):
    """SwiGLU MLP for the Mistral text backbone."""

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Mistral3TextAttention(nn.Module):
    """Grouped Query Attention for the Mistral text backbone.

    Uses AD canonical ops for RoPE and attention. GQA is handled natively
    by torch_attention — no repeat_interleave/repeat_kv needed.
    """

    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # Project Q/K/V and reshape to BSND layout
        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        cos, sin = position_embeddings

        # Apply RoPE using AD canonical op (BSND layout, unsqueeze_dim=2)
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        # Attention using AD canonical op with native GQA support (BSND layout)
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


class Mistral3TextDecoderLayer(nn.Module):
    """Transformer decoder layer for the Mistral text backbone."""

    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Mistral3TextAttention(config, layer_idx=layer_idx)
        self.mlp = Mistral3TextMLP(config)
        self.input_layernorm = Mistral3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Mistral3TextRMSNorm(
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
class Mistral3TextOutput(ModelOutput):
    """Output for Mistral3TextModel."""

    last_hidden_state: Optional[torch.FloatTensor] = None


class Mistral3TextPreTrainedModel(PreTrainedModel):
    """Base class for the Mistral3 text model."""

    config_class = MistralConfig
    base_model_prefix = "model"
    _no_split_modules = ["Mistral3TextDecoderLayer"]
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


class Mistral3TextModel(Mistral3TextPreTrainedModel):
    """Mistral text transformer decoder model (exported by AD).

    This module's nn.Module hierarchy matches HF MistralModel for checkpoint
    compatibility: embed_tokens, layers[i].self_attn.{q,k,v,o}_proj,
    layers[i].mlp.{gate,up,down}_proj, layers[i].{input,post_attention}_layernorm, norm.
    """

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Mistral3TextDecoderLayer(config, layer_idx=idx)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Mistral3TextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.rotary_emb = Mistral3TextRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
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
    ) -> Mistral3TextOutput:
        assert position_ids is not None, "position_ids must be provided for AD export"

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Cast to model compute dtype
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)

        hidden_states = self.norm(hidden_states)

        return Mistral3TextOutput(last_hidden_state=hidden_states)


# =============================================================================
# Multimodal Components (eager, NOT exported)
# =============================================================================


class Mistral3EagerRMSNorm(nn.Module):
    """Plain RMSNorm for multimodal components (not exported, no canonical ops needed)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Mistral3PatchMerger(nn.Module):
    """Learned merging of spatial_merge_size**2 patches."""

    def __init__(self, config: Mistral3Config):
        super().__init__()
        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]
        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = F.unfold(
                image_grid, kernel_size=self.spatial_merge_size, stride=self.spatial_merge_size
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        return self.merging_layer(image_features)


class Mistral3MultiModalProjector(nn.Module):
    """Multimodal projector: norm + patch_merger + MLP.

    Weight layout matches HF checkpoint:
      multi_modal_projector.norm.weight
      multi_modal_projector.patch_merger.merging_layer.weight
      multi_modal_projector.linear_1.weight
      multi_modal_projector.linear_2.weight
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.norm = Mistral3EagerRMSNorm(
            config.vision_config.hidden_size, eps=config.text_config.rms_norm_eps
        )
        self.patch_merger = Mistral3PatchMerger(config)
        num_feature_layers = (
            1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        )
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, image_sizes)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# =============================================================================
# Top-Level Model
# =============================================================================


@dataclass
class Mistral3ConditionalOutput(ModelOutput):
    """Output for Mistral3ForConditionalGeneration."""

    logits: Optional[torch.FloatTensor] = None


class Mistral3PreTrainedModel(PreTrainedModel):
    """Base class for the full Mistral3 multimodal model."""

    config_class = Mistral3Config
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class Mistral3Model(nn.Module):
    """Multimodal wrapper: vision tower + projector + language model.

    This module is NOT exported. It orchestrates the vision pipeline in
    eager PyTorch and calls the (potentially exported) language model.
    """

    def __init__(self, config: Mistral3Config):
        super().__init__()
        self.config = config

        # Vision tower: Pixtral, stays in eager mode
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = Mistral3MultiModalProjector(config)

        # Text model: uses AD canonical ops, will be exported
        self.language_model = Mistral3TextModel(config.text_config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
    ) -> List[torch.Tensor]:
        vision_feature_layer = (
            vision_feature_layer
            if vision_feature_layer is not None
            else self.config.vision_feature_layer
        )

        image_outputs = self.vision_tower(
            pixel_values, image_sizes=image_sizes, output_hidden_states=True
        )

        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature.squeeze(0), image_sizes)

        downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
        split_sizes = [
            (height // downsample_ratio) * (width // downsample_ratio)
            for height, width in image_sizes
        ]
        image_features = torch.split(image_features.squeeze(0), split_sizes)
        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Mistral3TextOutput:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                inputs_embeds.device
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs,
        )


class Mistral3ForConditionalGeneration(Mistral3PreTrainedModel, GenerationMixin):
    """Top-level multimodal model: vision + language model + lm_head.

    Wraps Mistral3Model (vision tower + projector + text model) and adds
    an lm_head at the top level — matching the HF checkpoint weight layout.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Mistral3Config, **kwargs):
        super().__init__(config)
        self.model = Mistral3Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Expose submodules for backward compatibility
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Mistral3ConditionalOutput:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        return Mistral3ConditionalOutput(logits=logits)


# =============================================================================
# Registration
# =============================================================================

AutoModelForCausalLMFactory.register_custom_model_cls(
    "Mistral3Config", Mistral3ForConditionalGeneration
)
