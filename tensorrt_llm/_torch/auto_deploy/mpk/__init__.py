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

"""AutoDeploy -> MPK translation utilities."""

from .buffer_planner import GemmaBufferPlanner
from .gemma_analyzer import GemmaGraphAnalyzer
from .gemma_layer_lowering import GemmaLayerLoweringPlanner
from .mirage_bridge import (
    MirageBindingResult,
    create_test_persistent_kernel,
    execute_layer_plan_reference,
    exercise_layer_plan_against_mirage,
    exercise_mirage_task_registration,
    resolve_layer_plan_against_mirage,
)
from .runtime_wrapper import GemmaMpkRuntimeWrapper
from .translator import GemmaMpkTranslator
from .types import (
    SUPPORTED_SOURCE_OP_FAMILIES,
    GemmaBufferPlan,
    GemmaBufferSpec,
    GemmaCanonicalOp,
    GemmaGraphInfo,
    GemmaLayerInfo,
    GemmaLayerLoweringPlan,
    GemmaLayerSchema,
    GemmaLoweringStatus,
    GemmaMpkStep,
    GemmaMpkTranslationPlan,
    GemmaNodeRef,
)

__all__ = [
    "GemmaBufferPlan",
    "GemmaBufferPlanner",
    "GemmaBufferSpec",
    "GemmaCanonicalOp",
    "GemmaGraphAnalyzer",
    "GemmaGraphInfo",
    "GemmaLayerInfo",
    "GemmaLayerLoweringPlan",
    "GemmaLayerLoweringPlanner",
    "GemmaLayerSchema",
    "GemmaLoweringStatus",
    "GemmaMpkStep",
    "GemmaMpkTranslationPlan",
    "GemmaMpkTranslator",
    "GemmaMpkRuntimeWrapper",
    "GemmaNodeRef",
    "MirageBindingResult",
    "SUPPORTED_SOURCE_OP_FAMILIES",
    "create_test_persistent_kernel",
    "execute_layer_plan_reference",
    "exercise_layer_plan_against_mirage",
    "exercise_mirage_task_registration",
    "resolve_layer_plan_against_mirage",
]
