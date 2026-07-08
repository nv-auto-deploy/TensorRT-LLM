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

"""Native task-accuracy tests for the llm-compiler (llmc) path.

Each model is run end-to-end through ``llmc.llm.LLM`` (TensorRT-LLM provides the
runtime/kernels only) and scored on MMLU + GSM8K against per-model reference
thresholds (see ``references/*.yaml``). Model ids and world sizes track
``runners/trtllm/model_registry/models.yaml``; weights resolve to a local dir
under ``LLM_MODELS_ROOT`` (falling back to the HF id).

TRT-LLM is required at runtime; the whole module is skipped where it is absent
so ``--collect-only`` still works in a standalone checkout.
"""

from typing import TYPE_CHECKING

import pytest
from accuracy_core import (
    GSM8K,
    MMLU,
    check_acceptance_rate,
    registry_config_paths,
    registry_yaml_extra,
    resolve_model_path,
)

if TYPE_CHECKING:  # import only for type hints; keeps the module importable
    from tensorrt_llm._torch.auto_deploy.llm import (
        LLM,  # (and thus collectable) without tensorrt_llm/GPU
    )


class LlmapiAccuracyTestHarness:
    """Base harness: constructs an llmc LLM for the subclass's model.

    The model's per-model config from the llmc model registry
    (runners/trtllm/model_registry/models.yaml -> configs/*.yaml, e.g. nano_v3.yaml,
    gpt_oss.yaml) is loaded via ``yaml_extra`` so accuracy runs match how the model
    is actually built. Those configs also set world size (world_size_N.yaml);
    ``WORLD_SIZE`` mirrors the registry and is passed explicitly for clarity.
    Weights resolve to a local dir under LLM_MODELS_ROOT (else the HF id).
    """

    MODEL_NAME: str = ""
    WORLD_SIZE: int = 1
    # Registry config_id selector (only needed if a model has multiple entries).
    CONFIG_ID: str = None
    # Extra kwargs forwarded to llmc.llm.LLM, merged after the registry yaml_extra.
    EXTRA_LLM_ARGS: dict = {}

    # Enough context for the eval prompts: 5-shot GSM8K runs ~600-1500 tokens +
    # 256 generated, MMLU similar. Several registry configs only inherit
    # dashboard_default.yaml's max_seq_len=512 (a perf-smoke default), which
    # truncates these prompts; an explicit kwarg overrides the yaml value.
    MAX_SEQ_LEN: int = 4096

    def get_llm(self) -> "LLM":
        # tensorrt_llm (and llmc.llm, which imports it) are required only to run;
        # import lazily so the module still collects on a CPU-only runner (e.g.
        # the CI job that discovers tests to build the accuracy workload matrix).
        pytest.importorskip("tensorrt_llm")
        from tensorrt_llm._torch.auto_deploy.llm import LLM

        yaml_extra = registry_yaml_extra(self.MODEL_NAME, self.CONFIG_ID)
        kwargs = dict(self.EXTRA_LLM_ARGS)
        # Registry configs first so explicit EXTRA_LLM_ARGS.yaml_extra can override.
        kwargs["yaml_extra"] = [*yaml_extra, *kwargs.get("yaml_extra", [])]
        kwargs.setdefault("max_seq_len", self.MAX_SEQ_LEN)
        # Resolve weights to a local dir under LLM_MODELS_ROOT (cluster-provided);
        # falls back to the HF id if unset/absent. References still key by HF id.
        return LLM(
            model=resolve_model_path(self.MODEL_NAME), world_size=self.WORLD_SIZE, **kwargs
        )

    @pytest.fixture(scope="class")
    def llm(self):
        # Class-scoped: build the (multi-GPU) engine + load weights ONCE per model
        # and share it across test_mmlu/test_gsm8k, instead of paying the full
        # build twice per class. Closed after the last test in the class.
        with self.get_llm() as _llm:
            yield _llm

    @staticmethod
    def _report(result) -> None:
        # Always surface the measured score/threshold (assert_passed is silent on
        # success), so accuracy runs are visible in the captured (-s) output.
        print(
            f"\n[ACCURACY] {result.task} {result.model}: score={result.score:.2f} "
            f"threshold={result.threshold:.2f} ref={result.ref_accuracy} "
            f"n={result.num_samples}"
        )

    def test_mmlu(self, llm):
        result = MMLU(self.MODEL_NAME).evaluate(llm)
        self._report(result)
        result.assert_passed()

    def test_gsm8k(self, llm):
        result = GSM8K(self.MODEL_NAME).evaluate(llm)
        self._report(result)
        result.assert_passed()


# Models that need more than a single 8x80GB node -- either a bigger-memory GPU
# (B200/GB200) or a multi-node allocation. The @large marker routes them to the
# big/Blackwell cluster (see scripts/generate_accuracy_products.py). They all run
# for now (no opt-in gate).
_LARGE = pytest.mark.large


# --- Models that run end-to-end on a single 8x80GB node -----------------------
class TestLlama31_8B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    WORLD_SIZE = 1


@pytest.mark.xfail(
    reason="llmc accuracy below TRT-LLM ref: MMLU 63.99 vs 71.30, GSM8K 36.39 vs 90.83",
    strict=False,
)
class TestGemma4MoE(LlmapiAccuracyTestHarness):
    MODEL_NAME = "google/gemma-4-26B-A4B-it"
    WORLD_SIZE = 1


@pytest.mark.blackwell  # NVFP4 requires a Blackwell GPU (routed to the B200 cluster)
@pytest.mark.xfail(
    reason="llmc gemma-4-31B load/export bug (NVFP4 input_scale / bf16 get_per_layer_inputs)",
    strict=False,
)
class TestGemma4Dense31B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/Gemma-4-31B-IT-NVFP4"
    WORLD_SIZE = 2


# nano_v3 is commented out (disabled) in the shared registry, so load its config
# directly -- the config files live in configs/ regardless of the registry entry,
# so the test does not depend on models.yaml (nor on re-enabling it there).
class TestNemotronNanoV3(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    WORLD_SIZE = 4
    EXTRA_LLM_ARGS = {
        "yaml_extra": registry_config_paths(
            "dashboard_default.yaml", "world_size_4.yaml", "nano_v3.yaml"
        )
    }


@_LARGE
@pytest.mark.xfail(
    reason="gpt-oss-120b MXFP4 experts unsharded (~66GB/rank); needs a big-memory GPU (B200/GB200) -- more nodes don't shard the experts",
    strict=False,
)
class TestGPTOSS(LlmapiAccuracyTestHarness):
    MODEL_NAME = "openai/gpt-oss-120b"
    WORLD_SIZE = 2  # matches the shared registry (gpt_oss -> world_size_2)
    EXTRA_LLM_ARGS = {"kv_cache_config": {"free_gpu_memory_fraction": 0.2}}


# super_v3 is commented out (disabled) in the shared registry; load its config
# directly (see TestNemotronNanoV3).
class TestNemotronSuper120B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-BF16-BF16KV-010726"
    WORLD_SIZE = 8
    EXTRA_LLM_ARGS = {
        "yaml_extra": registry_config_paths(
            "dashboard_default.yaml", "world_size_8.yaml", "super_v3.yaml"
        )
    }

    @pytest.mark.xfail(
        reason="GSM8K 89.23 vs TRT-LLM ref 92.70 (thr 89.50); MMLU passes", strict=False
    )
    def test_gsm8k(self, llm):
        super().test_gsm8k(llm)


# --- Models too large for a single 8x80GB node (need a suitable platform) -----
# Tagged @_LARGE -> routed to the big/Blackwell cluster. WORLD_SIZE=8 fits a
# big-memory single node (e.g. 8xB200 = 1440GB). For an 80GB-per-GPU platform,
# run multi-node: the generator already emits nodes = ceil(world_size /
# LLMC_ACCURACY_GPUS_PER_NODE), so bump WORLD_SIZE to 16/32 -- but first add the
# matching world_size_16/32.yaml (only world_size_{1,2,4,8}.yaml ship upstream;
# a new one under the managed configs/ dir must be added upstream to survive sync).
@_LARGE
class TestDeepSeekR1(LlmapiAccuracyTestHarness):
    # Full-weight config (deepseek-r1.yaml, no layer/expert reduction); references
    # exist -> a real accuracy run once it fits the target platform.
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
    WORLD_SIZE = 8


@_LARGE
@pytest.mark.xfail(
    reason="registry config qwen3.5_moe_400b.yaml exports only 2 of the MoE experts (truncated smoke) and no full-model reference exists yet; needs a full-weight config + reference on the target platform",
    strict=False,
)
class TestQwen35_397B(LlmapiAccuracyTestHarness):
    MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
    WORLD_SIZE = 8


@_LARGE
@pytest.mark.xfail(
    reason="registry config truncates GLM-4.7 to 5 layers (num_hidden_layers_5.yaml) + 2 experts (glm4_moe.yaml); needs a full-weight config + reference on the target platform",
    strict=False,
)
class TestGLM47(LlmapiAccuracyTestHarness):
    MODEL_NAME = "zai-org/GLM-4.7"
    WORLD_SIZE = 8


# --- MTP (multi-token-prediction speculative decoding) ------------------------
# Ports TRT-LLM's TestNemotronSuperV3::test_mtp[fp8_ws4_80gb]: Nemotron-Super-V3
# A12B FP8, world_size 4 (fits one 8x80GB node). MTP is turned on purely by loading
# the super_v3_mtp.yaml config directly (speculative_config: {decoding_type: MTP,
# num_nextn_predict_layers: 6, mtp_eagle_one_model: true}) -- this is a spec-dec
# test config, not a registry model variant, so it does not go through the registry
# overlay (matching upstream test_mtp, which also passes super_v3_mtp.yaml directly
# with world_size=4). We score GSM8K AND assert a spec-dec acceptance-rate floor
# (the real MTP signal); MMLU is not part of the upstream MTP test.
# enable_iter_perf_stats is required for get_stats() to carry specDecodingStats.
class TestNemotronSuperV3MTP(LlmapiAccuracyTestHarness):
    MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
    WORLD_SIZE = 4
    EXTRA_LLM_ARGS = {
        "enable_iter_perf_stats": True,
        "yaml_extra": registry_config_paths("super_v3_mtp.yaml"),
    }
    MIN_ACCEPTANCE_RATE = 0.50

    @pytest.mark.skip(
        reason="MTP accuracy is validated via GSM8K + acceptance rate, not MMLU (matches TRT-LLM test_mtp)"
    )
    def test_mmlu(self, llm):
        pass

    def test_gsm8k(self, llm):
        result = GSM8K(self.MODEL_NAME).evaluate(llm)
        self._report(result)
        result.assert_passed()
        check_acceptance_rate(llm, self.MIN_ACCEPTANCE_RATE)
