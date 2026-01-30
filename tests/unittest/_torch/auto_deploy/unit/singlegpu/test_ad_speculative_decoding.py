# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import with_mocked_hf_download

from tensorrt_llm._torch.auto_deploy.llm import DemoLLM
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, Eagle3DecodingConfig, KvCacheConfig


@pytest.mark.parametrize("use_hf_speculative_model", [False])
@with_mocked_hf_download
def test_ad_speculative_decoding_smoke(use_hf_speculative_model: bool):
    """Test speculative decoding with AutoDeploy using the build_and_run_ad main()."""

    # Use a simple test prompt
    test_prompt = "What is the capital of France?"

    # Get base model config
    experiment_config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")
    speculative_model_hf_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if use_hf_speculative_model:
        # NOTE: this will still mock out the actual HuggingFace download
        speculative_model = speculative_model_hf_id
    else:
        speculative_model = get_small_model_config(speculative_model_hf_id)["args"]["model"]

    # Configure speculative decoding with a draft model
    spec_config = DraftTargetDecodingConfig(max_draft_len=3, speculative_model=speculative_model)

    # Configure KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.01,
    )

    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["speculative_config"] = spec_config
    experiment_config["args"]["kv_cache_config"] = kv_cache_config
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["max_num_tokens"] = 64

    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    print(f"Experiment config: {experiment_config}")

    cfg = ExperimentConfig(**experiment_config)

    # Add sampling parameters (deterministic with temperature=0.0)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 50,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    print(f"Experiment config: {experiment_config}")
    print("Generating outputs with speculative decoding...")
    results = main(cfg)

    # Validate that we got output
    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1, "Should have exactly one prompt/output pair"

    prompt, generated_text = prompts_and_outputs[0]
    assert prompt == test_prompt, f"Prompt mismatch: expected '{test_prompt}', got '{prompt}'"
    assert len(generated_text) > 0, "Generated text should not be empty"

    print("Speculative decoding smoke test passed!")


# Maybe this test would be better checking a variety of settings of spec config and overlap scheduler
# and being a test for the KV cache manager creation.
def test_kv_cache_manager_spec_dec():
    """Tests that KV cache manager is created correctly with spec decoding related parameters."""
    print("\n" + "=" * 80)
    print("Testing Au")
    print("=" * 80)

    base_model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    eagle_model_id = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"

    base_model_config = get_small_model_config(base_model_id)
    eagle_model_config = get_small_model_config(eagle_model_id)

    print(f"\nBase Model Config: {base_model_config}")
    print(f"Eagle Model Config: {eagle_model_config}")

    max_draft_len = 3
    use_one_model_spec_dec = True

    speculative_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model_config["args"]["model"],
        eagle3_one_model=use_one_model_spec_dec,
        eagle3_layers_to_capture={1, 15, 28},
    )

    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.001,
    )

    llm = DemoLLM(
        model=base_model_config["args"]["model"],
        skip_loading_weights=True,
        world_size=0,
        kv_cache_config=kv_cache_config,
        speculative_config=speculative_config,
        disable_overlap_scheduler=True,
        max_num_tokens=128,
    )

    engine = llm._executor.engine_executor
    cache_interface = engine.cache_seq_interface
    kv_cache_manager = cache_interface._kv_cache_manager

    kv_num_extra_kv_tokens = getattr(kv_cache_manager, "num_extra_kv_tokens", None)
    kv_max_draft_len = getattr(kv_cache_manager, "max_draft_len", None)
    kv_max_total_draft_tokens = getattr(kv_cache_manager, "max_total_draft_tokens", None)

    csi_extra_seq_len_for_kv_cache = getattr(cache_interface, "_extra_seq_len_for_kv_cache", None)
    csi_spec_config = getattr(cache_interface, "_spec_config", None)

    print("\n" + "=" * 60)
    print("KVCacheManager Parameters:")
    print("=" * 60)
    print(f"  kv_num_extra_kv_tokens:     {kv_num_extra_kv_tokens}")
    print(f"  kv_max_draft_len:           {kv_max_draft_len}")
    print(f"  kv_max_total_draft_tokens:  {kv_max_total_draft_tokens}")
    print("=" * 60)
    print("\nCachedSequenceInterface Parameters:")
    print("=" * 60)
    print(f"  csi_extra_seq_len_for_kv_cache: {csi_extra_seq_len_for_kv_cache}")
    print(f"  csi_spec_config:                {csi_spec_config}")
    print("=" * 60)

    assert kv_max_draft_len == max_draft_len, (
        f"Expected kv_max_draft_len={max_draft_len}, got {kv_max_draft_len}"
    )
    assert kv_max_total_draft_tokens == max_draft_len, (
        f"Expected kv_max_total_draft_tokens={max_draft_len}, got {kv_max_total_draft_tokens}"
    )

    expected_num_extra_kv_tokens = max_draft_len - 1 if use_one_model_spec_dec else 0
    assert kv_num_extra_kv_tokens == expected_num_extra_kv_tokens, (
        f"Expected kv_num_extra_kv_tokens={expected_num_extra_kv_tokens}, "
        f"got {kv_num_extra_kv_tokens}"
    )

    # csi_extra_seq_len_for_kv_cache = max_total_draft_tokens + num_extra_kv_tokens
    # (no overlap scheduler contribution since disable_overlap_scheduler=True)
    expected_extra_seq_len = max_draft_len + expected_num_extra_kv_tokens  # 3 + 2 = 5
    assert csi_extra_seq_len_for_kv_cache == expected_extra_seq_len, (
        f"Expected csi_extra_seq_len_for_kv_cache={expected_extra_seq_len}, "
        f"got {csi_extra_seq_len_for_kv_cache}"
    )

    print("\n" + "=" * 80)
    print("SUCCESS! All KV cache spec-related assertions passed!")
    print("=" * 80)

    # Shutdown the LLM (no generation was performed)
    llm.shutdown()
