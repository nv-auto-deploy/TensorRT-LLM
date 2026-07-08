# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the pure scoring/plumbing in accuracy_core (no TRT-LLM/datasets)."""

import math
import os

import pytest
from accuracy_core import (
    AccuracyTask,
    EvalResult,
    acceptance_rate_from_stats,
    accuracy_percent,
    check_acceptance_rate,
    compute_threshold,
    extract_gsm8k_answer,
    extract_mmlu_answer,
    norm_ppf,
    registry_yaml_extra,
    select_reference_entry,
)


def _spec_stat(num_draft, num_accepted):
    return {"specDecodingStats": {"numDraftTokens": num_draft, "numAcceptedTokens": num_accepted}}


class TestAcceptanceRate:
    def test_aggregates_over_drafting_iters(self):
        stats = [_spec_stat(4, 3), _spec_stat(4, 2), _spec_stat(0, 0), {"otherStats": {}}]
        rate, accepted, drafted, iters = acceptance_rate_from_stats(stats)
        assert (accepted, drafted, iters) == (5, 8, 2)  # zero/absent-draft iters ignored
        assert rate == pytest.approx(5 / 8)

    def test_empty_or_no_spec(self):
        assert acceptance_rate_from_stats([]) == (0.0, 0, 0, 0)
        assert acceptance_rate_from_stats([{"otherStats": {}}]) == (0.0, 0, 0, 0)

    def test_check_raises_below_floor(self):
        class _LLM:
            @staticmethod
            def get_stats():
                return [_spec_stat(10, 2)]  # 20%

        assert check_acceptance_rate(_LLM(), 0.10) == pytest.approx(0.2)
        with pytest.raises(AssertionError):
            check_acceptance_rate(_LLM(), 0.50)


class TestExtraction:
    def test_mmlu_shapes(self):
        assert extract_mmlu_answer("B") == "B"
        assert extract_mmlu_answer("Answer: C") == "C"
        assert extract_mmlu_answer("The answer is D.") == "D"
        assert extract_mmlu_answer("(A)") == "A"
        assert extract_mmlu_answer("nonsense") is None

    def test_gsm8k_shapes(self):
        assert extract_gsm8k_answer("... so the total is #### 42") == "42"
        assert extract_gsm8k_answer("The result is 3,000 dollars") == "3000"
        assert extract_gsm8k_answer("answer: 7.0") == "7"
        assert extract_gsm8k_answer("no digits here") is None

    def test_accuracy_percent(self):
        assert accuracy_percent(["A", "B", None], ["A", "C", "D"]) == 100 / 3
        assert accuracy_percent([], []) == 0.0


class TestStatistics:
    def test_norm_ppf_matches_known_quantiles(self):
        assert math.isclose(norm_ppf(0.05), -1.6448536269514722, abs_tol=1e-6)
        assert math.isclose(norm_ppf(0.5), 0.0, abs_tol=1e-6)
        assert math.isclose(norm_ppf(0.975), 1.959963984540054, abs_tol=1e-6)

    def test_threshold_is_below_reference_and_widens_with_fewer_samples(self):
        ref, sigma = 88.0, 50.0
        t_full = compute_threshold(1319, ref, sigma)
        t_few = compute_threshold(64, ref, sigma)
        assert t_full < ref  # acceptance margin below the reference
        assert t_few < t_full  # fewer samples -> wider (lower) threshold

    def test_select_reference_entry(self):
        entries = [
            {"accuracy": 77.8},
            {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8", "accuracy": 73.9},
        ]
        assert select_reference_entry(entries, None, None)["accuracy"] == 77.8
        assert select_reference_entry(entries, "FP8", "FP8")["accuracy"] == 73.9
        assert select_reference_entry(entries, "NVFP4", None) is None


class TestRegistry:
    def test_resolves_active_model_config(self):
        # An active registry model resolves to its config yamls (gpt-oss @
        # world_size_2). Skips where the mirrored registry isn't present (e.g. the
        # TRT-LLM source tree; it exists in the generated standalone package).
        paths = registry_yaml_extra("openai/gpt-oss-120b", "gpt_oss")
        if not paths:
            pytest.skip("model registry not present in this checkout")
        names = [os.path.basename(p) for p in paths]
        assert any(n.startswith("world_size_") for n in names)
        assert all(p.endswith(".yaml") for p in paths)

    def test_unknown_model_returns_empty(self):
        # A model absent from (or commented-out/disabled in) the registry resolves
        # to []; the accuracy suite waives such models.
        assert registry_yaml_extra("does/not-exist") == []


class _FakeCompletion:
    def __init__(self, text):
        self.text = text


class _FakeOutput:
    def __init__(self, text):
        self.outputs = [_FakeCompletion(text)]


class _FakeLLM:
    """Minimal stand-in; no quant_config -> reference lookup uses the bare entry."""

    def __init__(self, answers):
        self._answers = answers

    def generate(self, prompts, sampling_params):
        return [_FakeOutput(a) for a in self._answers]


class _CannedMMLU(AccuracyTask):
    name = "mmlu"
    SIGMA = 50.0
    ALPHA = 0.05

    def build_prompts(self):
        return ["q1", "q2", "q3", "q4"], ["A", "B", "C", "D"]

    def score(self, generated, references):
        return accuracy_percent([extract_mmlu_answer(t) for t in generated], references)

    def sampling_params(self):  # avoid importing tensorrt_llm in the unit test
        return None


def test_evaluate_uses_statistical_threshold_from_reference():
    # references/mmlu.yaml seeds this model's bare MMLU accuracy from upstream
    # TensorRT-LLM (nvidia/Nemotron-3-Nano = 73.85).
    task = _CannedMMLU("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    result = task.evaluate(_FakeLLM(["A", "B", "C", "D"]))  # 100% on the canned set
    assert isinstance(result, EvalResult)
    assert result.num_samples == 4
    assert result.score == 100.0
    assert result.ref_accuracy == 73.85
    # threshold = 73.85 + norm_ppf(0.05)*sqrt(2*50^2/4); loose on 4 samples, 100 passes.
    assert result.threshold == compute_threshold(4, 73.85, 50.0, 0.05)
    assert result.passed


def test_evaluate_unknown_model_passes_trivially():
    task = _CannedMMLU("some/unregistered-model")
    result = task.evaluate(_FakeLLM(["A", "X", "X", "X"]))  # 25%
    assert result.ref_accuracy is None
    assert result.threshold == 0.0
    assert result.passed  # no reference -> trivial pass until seeded


def test_explicit_threshold_can_fail():
    task = _CannedMMLU("some/unregistered-model")
    result = task.evaluate(_FakeLLM(["A", "X", "X", "X"]), threshold=90.0)
    assert result.score == 25.0
    assert not result.passed
