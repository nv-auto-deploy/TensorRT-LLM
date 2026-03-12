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

"""Pixtral-12B text-only wrapper for auto_deploy.

Reference HF model: mistral-community/pixtral-12b

The mistral-community/pixtral-12b checkpoint uses LlavaConfig (model_type="llava"),
which AutoModelForCausalLM cannot load. This wrapper registers for LlavaConfig and
loads only the text backbone (standard MistralForCausalLM), discarding the vision
tower and multimodal projector weights.

Checkpoint weight layout:
  language_model.model.{embed_tokens,layers.*,norm}.*   ← text backbone (loaded)
  language_model.lm_head.weight                         ← LM head (loaded)
  multi_modal_projector.*                               ← discarded
  vision_tower.*                                        ← discarded

The text backbone is standard Mistral (GQA, SwiGLU, RMSNorm, RoPE) — AD transforms
handle it natively without custom canonical ops.
"""

from typing import Optional

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llava.configuration_llava import LlavaConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory


class PixtralPreTrainedModel(PreTrainedModel):
    """Base class for the Pixtral text-only wrapper."""

    config_class = LlavaConfig
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


class PixtralForCausalLM(PixtralPreTrainedModel, GenerationMixin):
    """Text-only wrapper that loads the Mistral text backbone from a Llava checkpoint.

    Module hierarchy matches the checkpoint layout:
      language_model.model.*     → MistralModel (text backbone)
      language_model.lm_head.*   → LM head

    Vision tower and projector weights are silently ignored during loading.
    """

    # Ignore vision tower and projector weights from checkpoint
    _keys_to_ignore_on_load_unexpected = [
        r"vision_tower\..*",
        r"multi_modal_projector\..*",
    ]

    def __init__(self, config: LlavaConfig, **kwargs):
        super().__init__(config)
        self.language_model = MistralForCausalLM(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        assert position_ids is not None, "position_ids must be provided for AD export"
        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
        )


# Registration — visible to both AutoModelForCausalLM and
# AutoModelForImageTextToText factories (shared _custom_model_mapping).
AutoModelForCausalLMFactory.register_custom_model_cls("LlavaConfig", PixtralForCausalLM)
