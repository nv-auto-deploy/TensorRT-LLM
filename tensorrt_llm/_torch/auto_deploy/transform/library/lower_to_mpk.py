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
import os
from pathlib import Path
from typing import Optional, Tuple, Type

import torch.nn as nn
from pydantic import Field
from torch.fx import Graph, GraphModule

from ...models.factory import ModelFactory
from ...mpk.mirage_bridge import build_gemma_mirage_runtime_callable
from ...mpk.runtime_wrapper import GemmaMpkRuntimeWrapper
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
    mirage_cache_dir: Optional[str] = Field(
        default=None,
        description=(
            "Optional cache directory to force MIRAGE_MPK_CACHE_DIR when the live "
            "Mirage runtime is installed."
        ),
    )
    force_enable_mirage_cache: bool = Field(
        default=False,
        description=(
            "If true, clear MIRAGE_MPK_DISABLE_CACHE and ensure a stable cache root "
            "is configured for the live Mirage runtime."
        ),
    )
    prewarm_decode_executors: bool = Field(
        default=False,
        description=(
            "If true, prewarm the Gemma decode executor set during lower_to_mpk "
            "installation so the first request does not discover/compile new MPK "
            "blocks on the hot path."
        ),
    )
    max_batch_size: Optional[int] = Field(
        default=None,
        description=(
            "Maximum batch size (num_tokens) for the Mirage MPK runtime. "
            "Executors compile once at this envelope and use "
            "pk.update_dynamic_dims() for smaller actual batches at runtime. "
            "If None, falls back to per-shape compilation (legacy behavior)."
        ),
    )
    max_seq_length: Optional[int] = Field(
        default=None,
        description=(
            "Maximum sequence length for the single-PK layer executor. "
            "The executor compiles once at this envelope so growing sequences "
            "do not trigger recompilation. If None, the executor grows "
            "incrementally as sequences get longer (legacy behavior)."
        ),
    )


@TransformRegistry.register("lower_to_mpk")
class LowerToMpk(BaseTransform):
    """Compile-stage hook that prepares a Gemma4MoE graph for future MPK lowering."""

    config: LowerToMpkConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return LowerToMpkConfig

    def _configure_mirage_cache_env(self) -> Optional[str]:
        cache_dir = self.config.mirage_cache_dir
        if cache_dir is None and not self.config.force_enable_mirage_cache:
            return None

        resolved_cache_dir: Optional[Path] = None
        if cache_dir is not None:
            resolved_cache_dir = Path(cache_dir).expanduser().resolve()
        else:
            env_cache_dir = os.environ.get("MIRAGE_MPK_CACHE_DIR")
            if env_cache_dir:
                resolved_cache_dir = Path(env_cache_dir).expanduser().resolve()
            else:
                resolved_cache_dir = Path(".tmp/mirage_mpk_cache").resolve()

        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MIRAGE_MPK_CACHE_DIR"] = str(resolved_cache_dir)
        if self.config.force_enable_mirage_cache:
            os.environ["MIRAGE_MPK_DISABLE_CACHE"] = "0"
        return str(resolved_cache_dir)

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

        summary = plan.summary()
        graph_summary = summary["graph"]
        ad_logger.info(
            "Gemma MPK analyzer recovered "
            f"{graph_summary['num_layers']} layers with schemas {graph_summary['schemas']}"
        )
        ad_logger.info(
            "Gemma MPK translation plan contains "
            f"{summary['num_layer_lowerings']} layer lowerings, "
            f"{summary['num_gap_steps']} gap steps, and "
            f"{summary['num_partial_steps']} partial steps"
        )

        if not self.config.dry_run_only:
            cache_dir = self._configure_mirage_cache_env()
            runtime_callable = build_gemma_mirage_runtime_callable(
                plan.to_dict(),
                source_model=model,
                max_batch_size=self.config.max_batch_size,
                max_seq_length=self.config.max_seq_length,
            )
            if self.config.prewarm_decode_executors:
                prewarm = getattr(runtime_callable, "prewarm_decode_executors", None)
                if callable(prewarm):
                    ad_logger.info("Prewarming Gemma MPK decode executor set...")
                    prewarm()
                    ad_logger.info("Gemma MPK decode executor prewarm finished.")
            model = self._wrap_graphmodule_with_runtime_wrapper(
                model,
                plan.to_dict(),
                runtime_callable=runtime_callable,
            )
            wrapped_meta = self._get_autodeploy_meta(model)
            wrapped_meta["mpk_runtime_mode"] = "mirage_runtime"
            self._set_autodeploy_meta(model, wrapped_meta)
            cache_message = (
                f", Mirage cache enabled at {cache_dir}" if cache_dir is not None else ""
            )
            ad_logger.info(
                "Gemma MPK blockwise decode runtime installed with generate-only Mirage routing, "
                "original-graph execution for non-generate-only batches, and single-PK layers "
                f"kept behind explicit opt-in{cache_message}"
            )

        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)
        return model, info

    def _wrap_graphmodule_with_runtime_wrapper(
        self,
        model: GraphModule,
        translation_plan: dict,
        runtime_callable: Optional[nn.Module] = None,
    ) -> GraphModule:
        """Wrap the full graph in a runtime wrapper while preserving GraphModule shape.

        The wrapper is the first strict runtime integration point for MPK.
        Generate-only batches are routed to the Mirage-backed decode runtime,
        while prefill and mixed batches continue through the original graph.
        """

        wrapper = GemmaMpkRuntimeWrapper(
            mpk_callable=runtime_callable
            if runtime_callable is not None
            else build_gemma_mirage_runtime_callable(
                translation_plan,
                source_model=model,
            ),
            original_model=model,
            translation_plan=translation_plan,
            input_names=[node.name for node in model.graph.nodes if node.op == "placeholder"],
        )

        outer_root = nn.Module()
        outer_root.add_module("gemma_mpk_runtime", wrapper)

        outer_graph = Graph()
        placeholder_nodes = []
        for node in model.graph.nodes:
            if node.op != "placeholder":
                continue
            placeholder = outer_graph.placeholder(node.name)
            placeholder.meta = dict(getattr(node, "meta", {}))
            placeholder_nodes.append(placeholder)

        wrapper_out = outer_graph.call_module("gemma_mpk_runtime", args=tuple(placeholder_nodes))
        output_node = outer_graph.output(wrapper_out)
        output_node.meta = {}

        wrapped_model = GraphModule(outer_root, outer_graph)
        wrapped_model.meta = dict(getattr(model, "meta", {}))
        return wrapped_model
