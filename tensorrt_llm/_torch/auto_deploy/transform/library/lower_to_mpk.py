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

"""Compile-stage MPK lowering hook for Gemma4MoE.

The initial implementation intentionally focuses on:
- config gating
- strict source-contract validation
- Gemma graph analysis
- emission of a structured translation plan into module metadata

This lets us validate the translator contract before replacing graph regions or
compiling MPK artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple, Type

import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...mpk.translator import GemmaMpkTranslator
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class LowerToMpkConfig(TransformConfig):
    """Configuration for the initial Gemma4MoE -> MPK lowering hook."""

    run_per_gm: bool = Field(
        default=False,
        description="The initial Gemma MPK translator analyzes the full model graph.",
    )

    model_family: str = Field(
        default="gemma4moe",
        description="Model family currently supported by the MPK translator.",
    )
    require_no_mlir_fusion: bool = Field(
        default=True,
        description="Require the no-MLIR-fusion graph surface for the Gemma MPK analyzer.",
    )
    dry_run_only: bool = Field(
        default=True,
        description=(
            "Analyze and emit a translation plan without replacing the graph or compiling MPK."
        ),
    )
    debug_dump_plan_path: Optional[str] = Field(
        default=None,
        description="Optional path to dump the JSON translation plan for debugging.",
    )


@TransformRegistry.register("lower_to_mpk")
class LowerToMpk(BaseTransform):
    """Compile-stage hook that prepares a Gemma4MoE graph for future MPK lowering."""

    config: LowerToMpkConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return LowerToMpkConfig

    def _apply_to_full_model(
        self,
        model: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        if self.config.model_family.lower() != "gemma4moe":
            raise ValueError(f"Unsupported MPK model_family: {self.config.model_family}")

        if not isinstance(model, GraphModule):
            raise ValueError(
                "lower_to_mpk currently expects the full model to be a GraphModule at compile stage."
            )

        translator = GemmaMpkTranslator(require_no_mlir_fusion=self.config.require_no_mlir_fusion)
        plan = translator.build_plan(model)

        autodeploy_meta = self._get_autodeploy_meta(model)
        autodeploy_meta["mpk_translation_plan"] = plan.to_dict()
        self._set_autodeploy_meta(model, autodeploy_meta)

        if self.config.debug_dump_plan_path is not None:
            dump_path = Path(self.config.debug_dump_plan_path)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(json.dumps(plan.to_dict(), indent=2))
            ad_logger.info(f"Dumped Gemma MPK translation plan to {dump_path}")

        if not self.config.dry_run_only:
            raise NotImplementedError(
                "lower_to_mpk runtime/module replacement is not implemented yet. "
                "Set dry_run_only=true to run analyzer-backed planning."
            )

        summary = plan.summary()
        graph_summary = summary["graph"]
        ad_logger.info(
            "Gemma MPK analyzer recovered "
            f"{graph_summary['num_layers']} layers with schemas {graph_summary['schemas']}"
        )
        ad_logger.info(
            "Gemma MPK dry-run plan contains "
            f"{summary['num_layer_lowerings']} layer lowerings, "
            f"{summary['num_gap_steps']} gap steps, and "
            f"{summary['num_partial_steps']} partial steps"
        )

        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)
        return model, info
