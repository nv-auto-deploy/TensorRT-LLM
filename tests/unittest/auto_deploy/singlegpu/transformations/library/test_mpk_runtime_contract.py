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

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.mpk import (
    GemmaMpkRuntimeWrapper,
    build_gemma_mirage_runtime_callable,
)
from tensorrt_llm._torch.auto_deploy.mpk.mirage_bridge import _should_use_single_pk_layer
from tensorrt_llm._torch.auto_deploy.transform.library.compile_model import CompileModel


class _DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.meta = {}

    def forward(self, **kwargs):
        return {"logits": kwargs["input_ids"] + 1}


class _InfoShouldNotReset:
    def reset(self):
        raise AssertionError("compile_model should not reset sequence info in Mirage runtime mode")


def _make_batch_info_host(*, num_prefill: int = 0, num_extend: int = 0, num_decode: int = 0):
    batch_info_host = torch.zeros(12, dtype=torch.int32)
    batch_info_host[0] = num_prefill
    batch_info_host[2] = num_extend
    batch_info_host[4] = num_decode
    return batch_info_host


def test_gemma_mpk_runtime_wrapper_requires_live_callable():
    wrapper = GemmaMpkRuntimeWrapper(
        mpk_callable=None,
        translation_plan={"layer_lowerings": []},
        input_names=("input_ids", "batch_info_host"),
    )

    try:
        wrapper(torch.tensor([1]), _make_batch_info_host(num_decode=1))
    except RuntimeError as exc:
        assert "no longer supports eager fallback for decode" in str(exc)
    else:
        raise AssertionError("Expected strict MPK wrapper to reject missing runtime callable.")


def test_gemma_mpk_runtime_wrapper_routes_generate_only_to_mirage():
    calls = []

    def _mpk_callable(*args, **kwargs):
        calls.append(("mpk", len(args), sorted(kwargs)))
        return {"logits": torch.tensor([7.0])}

    wrapper = GemmaMpkRuntimeWrapper(
        mpk_callable=_mpk_callable,
        original_model=_DummyModule(),
        translation_plan={"layer_lowerings": []},
        input_names=("input_ids", "batch_info_host"),
    )

    output = wrapper(torch.tensor([1.0]), _make_batch_info_host(num_decode=1))

    assert calls == [("mpk", 2, [])]
    assert torch.equal(output["logits"], torch.tensor([7.0]))


def test_gemma_mpk_runtime_wrapper_routes_prefill_to_original_model():
    wrapper = GemmaMpkRuntimeWrapper(
        mpk_callable=lambda *args, **kwargs: {"logits": torch.tensor([-1.0])},
        original_model=_DummyModule(),
        translation_plan={"layer_lowerings": []},
        input_names=("input_ids", "batch_info_host"),
    )

    output = wrapper(torch.tensor([3.0]), _make_batch_info_host(num_prefill=1))

    assert torch.equal(output["logits"], torch.tensor([4.0]))


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


def test_single_pk_layer_path_is_opt_in(monkeypatch):
    monkeypatch.delenv("AD_MPK_ENABLE_SINGLE_PK_LAYER", raising=False)
    assert _should_use_single_pk_layer(batch_size=1, seq_len=1, num_prefill=0) is False

    monkeypatch.setenv("AD_MPK_ENABLE_SINGLE_PK_LAYER", "1")
    assert _should_use_single_pk_layer(batch_size=1, seq_len=1, num_prefill=0) is True
    assert _should_use_single_pk_layer(batch_size=1, seq_len=2, num_prefill=0) is False
    assert _should_use_single_pk_layer(batch_size=2, seq_len=1, num_prefill=0) is False
    assert _should_use_single_pk_layer(batch_size=1, seq_len=1, num_prefill=1) is False
