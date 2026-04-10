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

from types import SimpleNamespace

import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.mpk import (
    GemmaMpkRuntimeWrapper,
    build_gemma_mirage_runtime_callable,
)
from tensorrt_llm._torch.auto_deploy.transform.library.compile_model import CompileModel


class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta = {}


class _InfoShouldNotReset:
    def reset(self):
        raise AssertionError("compile_model should not reset sequence info in Mirage runtime mode")


def test_gemma_mpk_runtime_wrapper_requires_live_callable():
    wrapper = GemmaMpkRuntimeWrapper(
        mpk_callable=None,
        translation_plan={"layer_lowerings": []},
        input_names=("input_ids",),
    )

    try:
        wrapper("fake_input")
    except RuntimeError as exc:
        assert "no longer supports eager fallback" in str(exc)
    else:
        raise AssertionError("Expected strict MPK wrapper to reject missing runtime callable.")


def test_gemma_mirage_runtime_callable_reports_remaining_lowering_gaps():
    runtime_callable = build_gemma_mirage_runtime_callable(
        {
            "layer_lowerings": [
                {
                    "mpk_steps": [
                        {"status": "supported"},
                        {"status": "gap"},
                        {"status": "partial"},
                    ]
                }
            ]
        }
    )

    try:
        runtime_callable()
    except NotImplementedError as exc:
        message = str(exc)
        assert "1 gap steps" in message
        assert "1 partial steps" in message
    else:
        raise AssertionError("Expected strict Mirage runtime callable to report remaining gaps.")


def test_compile_model_skips_when_mirage_runtime_mode_is_active():
    transform = CompileModel.from_kwargs(stage="compile", backend="torch-cudagraph")
    mod = _DummyModule()
    mod.meta["_autodeploy"] = {"mpk_runtime_mode": "mirage_runtime"}

    cm = SimpleNamespace(info=_InfoShouldNotReset())
    compiled_mod, info = transform._apply_to_full_model(  # noqa: SLF001
        mod=mod,
        cm=cm,
        factory=SimpleNamespace(),
        shared_config=SimpleNamespace(),
    )

    assert compiled_mod is mod
    assert info.skipped is True
    assert info.is_clean is True
