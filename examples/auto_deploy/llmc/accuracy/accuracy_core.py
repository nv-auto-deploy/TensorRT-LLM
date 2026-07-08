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

"""Native task-accuracy harness for llm-compiler (MMLU + GSM8K).

Runs real models end-to-end through ``llmc.llm.LLM`` (TensorRT-LLM provides the
runtime/kernels only) and scores task accuracy against per-model reference
thresholds. This is a native reimplementation of TensorRT-LLM's
``tests/integration/defs/accuracy`` harness so llm-compiler owns its own
accuracy tooling; it mirrors upstream conventions that matter for comparability:

  * scores are on a **0-100** scale;
  * the pass/fail threshold is a single-tail statistical bound
    ``ref_accuracy + z_alpha * sqrt(2 * sigma^2 / n)`` (``z_alpha = Phi^-1(alpha)``,
    negative for ``alpha < 0.5``), so the margin widens automatically when fewer
    samples are run;
  * references are keyed by model id with a list of quant variants (bare entry =
    unquantized/bf16), matched on ``quant_algo`` / ``kv_cache_quant_algo``.

Model weights resolve to a local dir under ``LLM_MODELS_ROOT`` (cluster-provided;
see ``resolve_model_path``), and MMLU/GSM8K datasets under
``LLM_MODELS_ROOT/datasets`` (see ``resolve_dataset_path``); both fall back to the
HuggingFace hub. ``datasets`` and ``tensorrt_llm.SamplingParams`` are imported
lazily so the pure scoring logic is importable/unit-testable without TRT-LLM or
the datasets lib.

``LLMC_ACCURACY_MAX_SAMPLES`` caps the sample count per task for cheap iteration.
"""

from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import yaml

REFERENCES_DIR = Path(__file__).parent / "references"
_ENV_MAX_SAMPLES = os.environ.get("LLMC_ACCURACY_MAX_SAMPLES")

# llm-compiler's model registry (per-model config yamls: world size,
# attn backend, quant, model-specific tweaks like nano_v3.yaml / gpt_oss.yaml).
# NOTE: in the standalone package this file lives at
# ``<repo>/tests/accuracy/accuracy_core.py``, so the repo root is two parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY_YAML = _REPO_ROOT / "runners/trtllm/model_registry/models.yaml"
_REGISTRY_CONFIGS_DIR = _REPO_ROOT / "runners/trtllm/model_registry/configs"

def _load_registry_entries(path: Path) -> List[dict]:
    """Return the ``models:`` list from a registry yaml, or [] if absent/empty."""
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text()) or {}
    entries = data.get("models", data) if isinstance(data, dict) else data
    return list(entries) if entries else []


def registry_yaml_extra(model_name: str, config_id: Optional[str] = None) -> List[str]:
    """Resolve a registry model's ``yaml_extra`` to absolute config paths.

    Mirrors ``build_and_run_llmc_trtllm.get_registry_yaml_extra`` but is dependency
    -free (plain YAML, no llmc/trtllm import) so this module stays importable for
    unit tests. Returns [] if the model entry is absent -- e.g. models that are
    commented-out (disabled) in the registry, such as nano_v3 / super_v3, whose
    accuracy tests are waived accordingly.
    """
    entries = _load_registry_entries(_REGISTRY_YAML)
    if not entries:
        return []
    matches = [e for e in entries if e.get("name") == model_name]
    if config_id is not None:
        matches = [e for e in matches if e.get("config_id", "default") == config_id]
    if not matches:
        return []
    if len(matches) > 1:
        defaults = [e for e in matches if e.get("config_id", "default") == "default"]
        selected = defaults[0] if defaults else matches[0]
    else:
        selected = matches[0]
    return [str(_REGISTRY_CONFIGS_DIR / cfg) for cfg in selected.get("yaml_extra", [])]


def registry_config_paths(*names: str) -> List[str]:
    """Absolute paths to named config files in the model-registry ``configs/`` dir.

    For tests that pass a specific config directly (not via a registry model
    entry), e.g. the MTP spec-dec test loading ``super_v3_mtp.yaml``.
    """
    return [str(_REGISTRY_CONFIGS_DIR / n) for n in names]


def resolve_model_path(hf_id: str) -> str:
    """Resolve a HF model id to a local dir under ``LLM_MODELS_ROOT``.

    Mirrors TensorRT-LLM's ``hf_id_to_local_model_dir``: returns
    ``<LLM_MODELS_ROOT>/<repo-name>`` (the id's last path segment) if it exists,
    then ``<LLM_MODELS_ROOT>/<full id>``, otherwise falls back to the HF id (HF
    hub download). ``LLM_MODELS_ROOT`` is provided by the cluster.
    """
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        return hf_id
    root = Path(root)
    for candidate in (root / hf_id.split("/")[-1], root / hf_id):
        if candidate.exists():
            return str(candidate)
    return hf_id


def resolve_dataset_path(dataset_id: str) -> str:
    """Resolve a HF dataset id to a local dir under ``LLM_MODELS_ROOT/datasets``.

    TRT-LLM stages datasets at ``<LLM_MODELS_ROOT>/datasets/<...>``. Returns
    ``<root>/datasets/<full id>`` if present, then ``<root>/datasets/<name>``
    (last segment), otherwise the HF id (hub download).
    """
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        return dataset_id
    datasets_root = Path(root) / "datasets"
    for candidate in (datasets_root / dataset_id, datasets_root / dataset_id.split("/")[-1]):
        if candidate.exists():
            return str(candidate)
    return dataset_id


# ---------------------------------------------------------------------------
# Statistics: inverse normal CDF + hypothesis-test threshold (no scipy dep)
# ---------------------------------------------------------------------------
def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF via the Acklam rational approximation.

    Accurate to ~1e-9 over (0, 1); avoids a scipy dependency in the harness.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def compute_threshold(
    num_samples: int,
    ref_accuracy: float,
    sigma: float,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> float:
    """Single-tail acceptance threshold around ``ref_accuracy`` (upstream formula)."""
    if num_samples <= 0:
        return ref_accuracy
    scale = (2 * sigma**2 / num_samples) ** 0.5
    z_alpha = norm_ppf(alpha)  # negative for alpha < 0.5
    return ref_accuracy + z_alpha * scale if higher_is_better else ref_accuracy - z_alpha * scale


# ---------------------------------------------------------------------------
# Pure scoring helpers (unit-tested; no LLM / dataset dependency)
# ---------------------------------------------------------------------------
_MMLU_CHOICES = ("A", "B", "C", "D")


def extract_mmlu_answer(text: str) -> Optional[str]:
    """First A-D choice letter in a generated answer, else None.

    Upstream scores ``output.strip().startswith(gold)``; we additionally tolerate
    "Answer: B" / "(C)" shapes that appear once few-shot formatting drifts.
    """
    if not text:
        return None
    stripped = text.strip()
    # A standalone leading choice letter ("B", "C.", "A)") -- but not a word like "Answer".
    if stripped[:1].upper() in _MMLU_CHOICES and (len(stripped) == 1 or not stripped[1].isalpha()):
        return stripped[0].upper()
    m = re.search(r"answer\b[^A-Za-z]*([A-D])\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else None


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _normalize_number(token: str) -> Optional[str]:
    token = token.replace(",", "").rstrip(".")
    if not token:
        return None
    try:
        val = float(token)
    except ValueError:
        return None
    return str(int(val)) if val.is_integer() else repr(val)


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Final numeric answer: prefer the FIRST ``#### N`` marker, else last number.

    Uses the first ``####`` (not the last): without a stop sequence a model can
    continue past its answer and hallucinate further ``Question:/Answer: #### N``
    turns, so the last marker belongs to fabricated text. We also cut anything
    after a new ``Question:`` for the same reason.
    """
    if not text:
        return None
    text = re.split(r"\n\s*Question:", text, maxsplit=1)[0]
    marker = text.split("####", 1)
    if len(marker) == 2:
        nums = _NUMBER_RE.findall(marker[1])
        if nums:
            return _normalize_number(nums[0])
    nums = _NUMBER_RE.findall(text)
    return _normalize_number(nums[-1]) if nums else None


def accuracy_percent(predictions: Sequence[Optional[str]], references: Sequence[Any]) -> float:
    """Percent (0-100) of predictions exactly matching their reference."""
    if not references:
        return 0.0
    correct = sum(
        1
        for pred, ref in zip(predictions, references)
        if pred is not None and str(pred).strip().upper() == str(ref).strip().upper()
    )
    return 100.0 * correct / len(references)


# ---------------------------------------------------------------------------
# References + result
# ---------------------------------------------------------------------------
def select_reference_entry(
    entries: List[dict], quant_algo: Optional[str], kv_cache_quant_algo: Optional[str]
) -> Optional[dict]:
    """First entry whose quant match-keys equal the run's (upstream semantics)."""
    for entry in entries or []:
        if (
            entry.get("quant_algo") == quant_algo
            and entry.get("kv_cache_quant_algo") == kv_cache_quant_algo
        ):
            return entry
    return None


# ---------------------------------------------------------------------------
# Speculative decoding (MTP / Eagle3) acceptance rate
# ---------------------------------------------------------------------------
def acceptance_rate_from_stats(stats) -> Tuple[float, int, int, int]:
    """Aggregate the spec-dec acceptance rate from ``llm.get_stats()`` output.

    Mirrors TensorRT-LLM's ``_check_acceptance_rate_stats``: sum accepted /
    drafted tokens over every iteration that actually drafted. Each ``stat`` is a
    dict carrying a ``specDecodingStats`` sub-dict (populated only when the LLM is
    built with ``enable_iter_perf_stats=True``). Returns
    ``(rate, accepted, drafted, num_spec_iters)``.
    """
    total_drafted = 0
    total_accepted = 0
    num_spec_iterations = 0
    for stat in stats or []:
        spec_stats = (stat.get("specDecodingStats") if isinstance(stat, dict) else None) or {}
        num_draft = spec_stats.get("numDraftTokens", 0)
        if num_draft <= 0:
            continue
        num_spec_iterations += 1
        total_drafted += num_draft
        total_accepted += spec_stats.get("numAcceptedTokens", 0)
    rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    return rate, total_accepted, total_drafted, num_spec_iterations


def check_acceptance_rate(llm, min_acceptance_rate: float) -> float:
    """Assert the spec-dec acceptance rate for the current run meets a floor.

    Reads ``llm.get_stats()`` (requires ``enable_iter_perf_stats=True`` on the
    LLM). Returns the measured rate; raises AssertionError if below the floor.
    """
    rate, accepted, drafted, num_iters = acceptance_rate_from_stats(llm.get_stats())
    print(
        f"\n[MTP] acceptance rate {rate:.2%} ({accepted}/{drafted} tokens across "
        f"{num_iters} speculative iterations)"
    )
    assert rate >= min_acceptance_rate, (
        f"Acceptance rate {rate:.2%} below threshold {min_acceptance_rate:.0%}"
    )
    return rate


@dataclass
class EvalResult:
    task: str
    model: str
    score: float  # 0-100
    threshold: float  # 0-100
    ref_accuracy: Optional[float]
    num_samples: int
    higher_is_better: bool = True

    @property
    def passed(self) -> bool:
        return (
            self.score >= self.threshold if self.higher_is_better else self.score <= self.threshold
        )

    def assert_passed(self) -> "EvalResult":
        assert self.passed, (
            f"{self.task} accuracy {self.score:.2f} for {self.model} is below "
            f"threshold {self.threshold:.2f} (ref={self.ref_accuracy}, n={self.num_samples})"
        )
        return self


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
class AccuracyTask:
    """Base task: build prompts, generate, score (0-100), compare to a stat threshold."""

    name: str = "task"
    NUM_SAMPLES: int = 512
    NUM_FEWSHOT: int = 0
    MAX_OUTPUT_LEN: int = 256
    SIGMA: float = 50.0
    ALPHA: float = 0.05
    HIGHER_IS_BETTER: bool = True
    # Used only when the model has no reference entry (keeps first runs green).
    DEFAULT_REF_ACCURACY: float = 0.0
    seed: int = 0

    def __init__(self, model_name: str):
        self.model_name = model_name
        path = REFERENCES_DIR / f"{self.name}.yaml"
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        self.reference: List[dict] = (data or {}).get(model_name, [])

    def _effective_num_samples(self) -> int:
        n = self.NUM_SAMPLES
        if _ENV_MAX_SAMPLES:
            try:
                n = min(n, int(_ENV_MAX_SAMPLES))
            except ValueError:
                pass
        return n

    def build_prompts(self) -> Tuple[List[str], List[Any]]:
        raise NotImplementedError

    def score(self, generated: List[str], references: List[Any]) -> float:
        raise NotImplementedError

    # Stop strings to end generation (per task). Critical for few-shot tasks
    # like GSM8K: without it the model runs to max_tokens and hallucinates
    # follow-on "Question:/Answer:" turns, wrecking answer extraction.
    STOP: Optional[List[str]] = None

    def sampling_params(self):
        from tensorrt_llm import SamplingParams

        return SamplingParams(
            max_tokens=self.MAX_OUTPUT_LEN, temperature=0.0, top_p=1.0, stop=self.STOP
        )

    @staticmethod
    def _quant_of(llm) -> Tuple[Optional[str], Optional[str]]:
        # AutoDeploy/llmc detects quantization on the model factory (parsed from
        # the checkpoint's hf_quant_config), NOT on llm.args.quant_config, which
        # stays a default-empty QuantConfig. Read the factory first so quantized
        # runs match their FP8/NVFP4 reference entry; fall back to args (empty on
        # this path, populated on a plain TRT-LLM path).
        try:
            factory = getattr(llm, "factory", None)
            qcfg = factory.get_quant_config() if factory is not None else None
        except Exception:
            qcfg = None
        if qcfg:
            qa = qcfg.get("quant_algo")
            kv = qcfg.get("kv_cache_quant_algo")
            if qa or kv:
                return (str(qa) if qa else None), (str(kv) if kv else None)
        qc = getattr(getattr(llm, "args", None), "quant_config", None)
        if qc is None:
            return None, None
        qa = getattr(qc, "quant_algo", None)
        kv = getattr(qc, "kv_cache_quant_algo", None)
        return (str(qa) if qa else None), (str(kv) if kv else None)

    def _reference_accuracy(self, llm) -> Optional[float]:
        quant_algo, kv = self._quant_of(llm)
        entry = select_reference_entry(self.reference, quant_algo, kv)
        return float(entry["accuracy"]) if entry and "accuracy" in entry else None

    def evaluate(self, llm, threshold: Optional[float] = None) -> EvalResult:
        prompts, references = self.build_prompts()
        outputs = llm.generate(prompts, self.sampling_params())
        generated = [self._extract_text(o) for o in outputs]
        score = self.score(generated, references)
        n = len(references)

        ref_accuracy = self._reference_accuracy(llm)
        if threshold is not None:
            thr = threshold
        elif ref_accuracy is not None:
            thr = compute_threshold(n, ref_accuracy, self.SIGMA, self.ALPHA, self.HIGHER_IS_BETTER)
        else:
            thr = self.DEFAULT_REF_ACCURACY
        return EvalResult(
            self.name, self.model_name, score, thr, ref_accuracy, n, self.HIGHER_IS_BETTER
        )

    @staticmethod
    def _extract_text(output: Any) -> str:
        try:
            return output.outputs[0].text
        except (AttributeError, IndexError):
            return str(output)


# The 57 MMLU subject configs. cais/mmlu's aggregate "all" config is frequently
# absent from a local HF cache (only the per-subject configs get pulled), so we
# load and concatenate the subjects -- equivalent to "all" and offline-friendly.
_MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


class MMLU(AccuracyTask):
    name = "mmlu"
    NUM_SAMPLES = 4096
    NUM_FEWSHOT = 5
    MAX_OUTPUT_LEN = 4  # a couple of tokens is enough for the answer letter
    SIGMA = 50.0
    ALPHA = 0.05
    DATASET = "cais/mmlu"
    SUBJECTS = _MMLU_SUBJECTS

    def _format_example(self, row: dict, include_answer: bool) -> str:
        text = row["question"].strip()
        for letter, choice in zip(_MMLU_CHOICES, row["choices"]):
            text += f"\n{letter}. {choice}"
        text += "\nAnswer:"
        if include_answer:
            text += f" {_MMLU_CHOICES[int(row['answer'])]}\n\n"
        return text

    def build_prompts(self) -> Tuple[List[str], List[Any]]:
        from datasets import load_dataset

        # (subject, test_row) across all subjects, plus a per-subject few-shot
        # pool from each subject's dev split. Subject comes from the config name,
        # so we don't rely on a "subject" column being present in the rows.
        dataset = resolve_dataset_path(self.DATASET)
        test_examples: List[Tuple[str, dict]] = []
        dev_by_subject: dict = {}
        for subject in self.SUBJECTS:
            try:
                test_s = load_dataset(dataset, subject, split="test")
                dev_s = load_dataset(dataset, subject, split="dev")
            except Exception:  # subject missing from the cache -- skip it
                continue
            dev_by_subject[subject] = list(dev_s)
            test_examples.extend((subject, row) for row in test_s)

        rng = random.Random(self.seed)
        n = min(self._effective_num_samples(), len(test_examples))
        indices = rng.sample(range(len(test_examples)), n)

        prompts, references = [], []
        for i in indices:
            subject, row = test_examples[i]
            shots = dev_by_subject.get(subject, [])[: self.NUM_FEWSHOT]
            preamble = (
                "The following are multiple choice questions (with answers) about "
                f"{subject.replace('_', ' ')}.\n\n"
            )
            fewshot = "".join(self._format_example(s, include_answer=True) for s in shots)
            prompts.append(preamble + fewshot + self._format_example(row, include_answer=False))
            references.append(_MMLU_CHOICES[int(row["answer"])])
        return prompts, references

    def score(self, generated: List[str], references: List[Any]) -> float:
        return accuracy_percent([extract_mmlu_answer(t) for t in generated], references)


class GSM8K(AccuracyTask):
    name = "gsm8k"
    NUM_SAMPLES = 1319  # full GSM8K test split
    NUM_FEWSHOT = 5
    MAX_OUTPUT_LEN = 256
    SIGMA = 50.0
    ALPHA = 0.05
    DATASET = "openai/gsm8k"
    CONFIG = "main"
    # End generation once the model starts a fresh few-shot turn.
    STOP = ["\nQuestion:", "\n\nQuestion:"]

    def build_prompts(self) -> Tuple[List[str], List[Any]]:
        from datasets import load_dataset

        dataset = resolve_dataset_path(self.DATASET)
        test = load_dataset(dataset, self.CONFIG, split="test")
        # Only the first NUM_FEWSHOT train rows are used as shots -- slice the
        # split so we don't decode the full ~7.5k-row train set to keep 5.
        train = load_dataset(dataset, self.CONFIG, split=f"train[:{self.NUM_FEWSHOT}]")

        fewshot = "".join(
            f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n"
            for i in range(len(train))
        )

        rng = random.Random(self.seed)
        n = min(self._effective_num_samples(), len(test))
        indices = rng.sample(range(len(test)), n)

        prompts, references = [], []
        for i in indices:
            row = test[i]
            prompts.append(f"{fewshot}Question: {row['question']}\nAnswer:")
            references.append(extract_gsm8k_answer(row["answer"]))
        return prompts, references

    def score(self, generated: List[str], references: List[Any]) -> float:
        return accuracy_percent([extract_gsm8k_answer(t) for t in generated], references)


TASKS = {MMLU.name: MMLU, GSM8K.name: GSM8K}
