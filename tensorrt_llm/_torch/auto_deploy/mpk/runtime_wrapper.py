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
    """Wrapper that will eventually own a compiled MPK artifact.

    The current implementation is intentionally conservative:
    - if a compiled MPK callable is provided, it is invoked directly
    - otherwise, an eager fallback module is used

    This gives the MPK lowering path a concrete runtime integration target
    without changing the current execution behavior.
    """

    def __init__(
        self,
        *,
        eager_fallback: Optional[nn.Module] = None,
        mpk_callable: Optional[Callable[..., Any]] = None,
        translation_plan: Optional[Dict[str, Any]] = None,
        input_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.eager_fallback = eager_fallback
        self.mpk_callable = mpk_callable
        self.translation_plan = translation_plan or {}
        self.input_names = tuple(input_names or ())

    def forward(self, *args, **kwargs):
        if self.mpk_callable is not None:
            return self.mpk_callable(*args, **kwargs)
        if self.eager_fallback is not None:
            if not kwargs and self.input_names and len(args) == len(self.input_names):
                kwargs = dict(zip(self.input_names, args))
                args = ()
            return self.eager_fallback(*args, **kwargs)
        raise RuntimeError(
            "GemmaMpkRuntimeWrapper has neither an MPK callable nor an eager fallback."
        )

    def extra_repr(self) -> str:
        mode = "mpk" if self.mpk_callable is not None else "fallback"
        has_plan = bool(self.translation_plan)
        return f"mode={mode}, has_plan={has_plan}, num_inputs={len(self.input_names)}"
