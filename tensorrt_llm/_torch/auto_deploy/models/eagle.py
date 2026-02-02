# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Factory definitions for building models related to Eagle in AutoDeploy.

This module provides EagleDrafterFactory, a specialized factory for building
Eagle speculative decoding draft models. It extends AutoModelForCausalLMFactory
to handle Eagle-specific configuration.

The EagleDrafterForCausalLM class builds its own model-specific layers internally
based on model_type, so the factory just needs to create the drafter with the config.
"""

from contextlib import nullcontext

import torch.nn as nn
from accelerate import init_empty_weights
from torch._prims_common import DeviceLikeType

from ..utils.logger import ad_logger
from .custom.modeling_eagle import EagleConfig, EagleDrafterForCausalLM
from .factory import ModelFactoryRegistry
from .hf import AutoModelForCausalLMFactory


@ModelFactoryRegistry.register("EagleDrafter")
class EagleDrafterFactory(AutoModelForCausalLMFactory):
    """Factory for building Eagle drafter models.

    The drafter builds its own model-specific layers internally based on
    config.model_type, allowing it to work with different base models
    (Llama, NemotronH, etc.) without the factory needing to know the details.

    The checkpoint config is expected to have the base model's model_type
    (e.g., "llama") along with Eagle-specific fields like draft_vocab_size.
    """

    def _build_model(self, device: DeviceLikeType) -> nn.Module:
        model_config, unused_kwargs = self._get_model_config()

        # Get model type for config
        model_type = model_config.model_type
        ad_logger.info(f"EagleDrafterFactory: building drafter for model_type='{model_type}'")

        # Convert base config to EagleConfig, preserving existing values
        # and applying model-specific defaults based on model_type
        model_config = EagleConfig(model_config, model_type)

        with (init_empty_weights if device == "meta" else nullcontext)():
            model = EagleDrafterForCausalLM(model_config)

        if device == "meta":
            # post-init must be called explicitly for HF models with init_empty_weights
            if hasattr(model, "post_init"):
                model.post_init()
        else:
            model.to(device)

        # Store checkpoint conversion mapping if present
        self._checkpoint_conversion_mapping = getattr(model, "_checkpoint_conversion_mapping", None)

        model.eval()

        return model

    def build_and_load_model(self, _device: DeviceLikeType) -> nn.Module:
        raise NotImplementedError(
            "EagleDrafterFactory does not support build_and_load_model(). "
            "Use build_model() + load_or_random_init() instead."
        )
