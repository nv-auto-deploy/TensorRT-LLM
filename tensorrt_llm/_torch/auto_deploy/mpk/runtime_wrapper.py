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

"""Runtime wrapper scaffold for future AutoDeploy -> MPK integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import torch.nn as nn

from ..custom_ops.attention_interface import BatchInfo


class GemmaMpkRuntimeWrapper(nn.Module):
    """Wrapper that owns the Gemma MPK runtime boundary.

    The wrapper is intentionally policy-aware:
    - generate-only batches use the Mirage-backed runtime
    - non-generate-only batches use the original AutoDeploy GraphModule

    This keeps Mirage focused on decode-only acceleration while preserving the
    existing AutoDeploy path for prefill and mixed steps.
    """

    def __init__(
        self,
        *,
        mpk_callable: Optional[Callable[..., Any]] = None,
        original_model: Optional[nn.Module] = None,
        translation_plan: Optional[Dict[str, Any]] = None,
        input_names: Optional[Sequence[str]] = None,
        batch_info_input_name: str = "batch_info_host",
    ) -> None:
        super().__init__()
        self.mpk_callable = mpk_callable
        self.original_model = original_model
        self.translation_plan = translation_plan or {}
        self.input_names = tuple(input_names or ())
        self.batch_info_input_name = batch_info_input_name

    def _bind_inputs(self, args, kwargs) -> Dict[str, Any]:
        if len(args) > len(self.input_names):
            raise RuntimeError(
                "GemmaMpkRuntimeWrapper received more positional inputs than placeholder names."
            )

        bound_inputs = dict(kwargs)
        for name, value in zip(self.input_names, args):
            if name in bound_inputs:
                raise RuntimeError(f"GemmaMpkRuntimeWrapper received duplicate input for '{name}'.")
            bound_inputs[name] = value
        return bound_inputs

    def _is_generate_only(self, bound_inputs: Dict[str, Any]) -> bool:
        batch_info_host = bound_inputs.get(self.batch_info_input_name)
        if batch_info_host is None:
            raise RuntimeError(
                "GemmaMpkRuntimeWrapper requires 'batch_info_host' to decide whether "
                "to route to Mirage decode or the original AutoDeploy graph."
            )
        return BatchInfo(batch_info_host).is_generate_only()

    def forward(self, *args, **kwargs):
        bound_inputs = self._bind_inputs(args, kwargs)
        if self._is_generate_only(bound_inputs):
            if self.mpk_callable is not None:
                return self.mpk_callable(*args, **kwargs)
            raise RuntimeError(
                "GemmaMpkRuntimeWrapper was invoked on a generate-only batch without a live MPK "
                "callable. The Gemma MPK path no longer supports eager fallback for decode."
            )

        if self.original_model is not None:
            return self.original_model(**bound_inputs)

        raise RuntimeError(
            "GemmaMpkRuntimeWrapper received a non-generate-only batch without an original "
            "AutoDeploy model to execute."
        )

    def extra_repr(self) -> str:
        mode = "mpk" if self.mpk_callable is not None else "missing_mpk_callable"
        has_plan = bool(self.translation_plan)
        original = self.original_model is not None
        return (
            f"mode={mode}, has_plan={has_plan}, has_original_model={original}, "
            f"num_inputs={len(self.input_names)}"
        )
