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

"""Initial Gemma4MoE MPK translator.

The current implementation is deliberately analysis-first.  It validates the
supported no-MLIR-fusion graph surface and emits a structured translation plan.
Subsequent phases can extend this object with concrete buffer planning and MPK
builder emission without changing the transform entrypoint.
"""

from __future__ import annotations

from torch.fx import GraphModule

from .buffer_planner import GemmaBufferPlanner
from .gemma_analyzer import GemmaGraphAnalyzer
from .gemma_layer_lowering import GemmaLayerLoweringPlanner
from .types import GemmaMpkTranslationPlan


class GemmaMpkTranslator:
    """Build an initial MPK translation plan from a Gemma4MoE FX graph."""

    def __init__(self, *, require_no_mlir_fusion: bool = True):
        self.analyzer = GemmaGraphAnalyzer(require_no_mlir_fusion=require_no_mlir_fusion)
        self.buffer_planner = GemmaBufferPlanner()
        self.layer_lowering_planner = GemmaLayerLoweringPlanner()

    def build_plan(self, gm: GraphModule) -> GemmaMpkTranslationPlan:
        graph_info = self.analyzer.analyze(gm)
        buffer_plan = self.buffer_planner.build(graph_info)
        layer_lowerings = [
            self.layer_lowering_planner.build(layer_info) for layer_info in graph_info.layer_infos
        ]
        notes = [
            "Current implementation emits a dry-run MPK translation skeleton from the analyzed FX graph.",
            "The plan includes explicit graph/layer buffers plus per-layer canonical and MPK step sequences.",
            "Steps marked as gaps identify backend work still needed before executable MPK emission.",
        ]
        return GemmaMpkTranslationPlan(
            graph_info=graph_info,
            buffer_plan=buffer_plan,
            layer_lowerings=layer_lowerings,
            notes=notes,
        )
