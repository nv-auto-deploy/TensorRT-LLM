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


class GemmaMpkRuntimeWrapper(nn.Module):
    """Wrapper that owns the Gemma MPK runtime boundary.

    Once the MPK path is selected we intentionally do not retain an eager
    fallback. This avoids silently executing the original graph when the
    Mirage-backed runtime path is expected.
    """

    def __init__(
        self,
        *,
        mpk_callable: Optional[Callable[..., Any]] = None,
        translation_plan: Optional[Dict[str, Any]] = None,
        input_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.mpk_callable = mpk_callable
        self.translation_plan = translation_plan or {}
        self.input_names = tuple(input_names or ())

    def forward(self, *args, **kwargs):
        if self.mpk_callable is not None:
            return self.mpk_callable(*args, **kwargs)
        raise RuntimeError(
            "GemmaMpkRuntimeWrapper was invoked without a live MPK callable. "
            "The Gemma MPK path no longer supports eager fallback."
        )

    def extra_repr(self) -> str:
        mode = "mpk" if self.mpk_callable is not None else "missing_mpk_callable"
        has_plan = bool(self.translation_plan)
        return f"mode={mode}, has_plan={has_plan}, num_inputs={len(self.input_names)}"
