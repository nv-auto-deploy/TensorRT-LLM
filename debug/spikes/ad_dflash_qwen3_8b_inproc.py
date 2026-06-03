# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash E2E IN-PROCESS (world_size=0, DemoLLM) via build_and_run_ad.main.

Decisive test: plain Qwen3-8B is coherent in-process (world_size=0) but crashes/garbages at
world_size=1 (LLM-API worker). This runs DFlash through the SAME in-process path to isolate whether
the E2E garbage is a world_size=1 worker artifact or a real DFlash bug.

Run: CUDA_VISIBLE_DEVICES=<free> TRITON_CACHE_DIR=... LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     python debug/spikes/ad_dflash_qwen3_8b_inproc.py
"""

import sys

sys.path.append("tests/unittest/utils")
sys.path.append("examples/auto_deploy")
from build_and_run_ad import ExperimentConfig, main  # noqa: E402
from llm_data import llm_models_root  # noqa: E402

from tensorrt_llm.llmapi import DFlashDecodingConfig  # noqa: E402


def run():
    root = llm_models_root()
    target = f"{root}/Qwen3/Qwen3-8B"
    draft = f"{root}/Qwen3-8B-DFlash-b16"
    spec = DFlashDecodingConfig(
        max_draft_len=4,
        speculative_model=draft,
        target_layer_ids=[1, 9, 17, 25, 33],
    )
    experiment_config = {
        "args": {
            "model": target,
            "world_size": 0,  # in-process (DemoLLM) -- the verified-good plain-target path
            "runtime": "demollm",
            "attn_backend": "trtllm",
            "compile_backend": "torch-simple",
            "max_seq_len": 2048,
            "max_batch_size": 4,
            "speculative_config": spec,
            "cuda_graph_config": {"batch_sizes": [1, 2, 4], "enable_padding": True},
            "kv_cache_config": {"enable_block_reuse": False, "max_tokens": 2048},
        },
        "prompt": {
            "batch_size": 1,
            "queries": ["The capital of France is"],
            "sp_kwargs": {"max_tokens": 64, "temperature": 0.0, "top_k": None},
        },
    }
    cfg = ExperimentConfig(**experiment_config)
    print(">>> running DFlash in-process (world_size=0) ...")
    main(cfg)


if __name__ == "__main__":
    run()
