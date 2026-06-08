# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash / plain Qwen3-8B E2E via build_and_run_ad.main, world_size=1 + TRTLLM runtime.

The canonical deployment path (the trusted example harness). Toggle DFlash with SPEC=1 (default) or
plain target with SPEC=0. world_size=0/DemoLLM is intentionally NOT used (DemoLLM doesn't support
spec-dec multi-token output bookkeeping).

Run: CUDA_VISIBLE_DEVICES=<free> SPEC=0|1 TRITON_CACHE_DIR=... \
     LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     python debug/spikes/ad_dflash_qwen3_8b_bnr.py
"""

import os
import sys

sys.path.append("tests/unittest/utils")
sys.path.append("examples/auto_deploy")
from build_and_run_ad import ExperimentConfig, main  # noqa: E402
from llm_data import llm_models_root  # noqa: E402

from tensorrt_llm.llmapi import DFlashDecodingConfig  # noqa: E402


def run():
    root = llm_models_root()
    # Env-parameterized target/draft so we can isolate Qwen3-target vs Llama-target (same Qwen3-arch
    # DFlash draft architecture). TGT_LAYER_IDS must match the draft's dflash_config.target_layer_ids.
    target = os.environ.get("TARGET_MODEL", f"{root}/Qwen3/Qwen3-8B")
    draft = os.environ.get("DRAFT_MODEL", f"{root}/Qwen3-8B-DFlash-b16")
    layer_ids = [int(x) for x in os.environ.get("TGT_LAYER_IDS", "1,9,17,25,33").split(",")]
    print(f">>> target={target}\n>>> draft={draft}\n>>> target_layer_ids={layer_ids}")
    use_spec = os.environ.get("SPEC", "1") == "1"

    regstyle = os.environ.get("REGSTYLE", "0") == "1"
    attn = os.environ.get("ATTN", "trtllm")  # "flashinfer" -> plain prefill (no spec-dec mask path)
    if regstyle:
        # Mirror the registry dashboard_default.yaml (the canonical supported config) at ws1.
        args = {
            "model": target,
            "world_size": 1,
            "runtime": "trtllm",
            "attn_backend": attn,
            "compile_backend": "torch-cudagraph",
            "max_seq_len": 512,
            "max_batch_size": 128,
        }
    else:
        args = {
            "model": target,
            "world_size": 1,  # canonical worker-process deployment
            "runtime": "trtllm",  # the real runtime (NOT demollm)
            "attn_backend": attn,
            "compile_backend": "torch-simple",
            "max_seq_len": 2048,
            "max_batch_size": 4,
            "cuda_graph_config": {"batch_sizes": [1, 2, 4], "enable_padding": True},
            "kv_cache_config": {"enable_block_reuse": False, "max_tokens": 2048},
        }
    if os.environ.get("NORESIZE") == "1":
        # free_gpu_memory_fraction=0.0 -> needs_resize()==False -> the resize_kv_cache pass (whose
        # large sample forward corrupts the target lm_head) is skipped entirely; the estimation-mode
        # pool is kept. TEMPORARY: trades away KV capacity tuning. See debug/dflash_worklog_accuracy.md.
        args.setdefault("kv_cache_config", {})["free_gpu_memory_fraction"] = 0.0
    if use_spec:
        args["speculative_config"] = DFlashDecodingConfig(
            max_draft_len=4, speculative_model=draft, target_layer_ids=layer_ids
        )
        # Spec decoding bring-up requires the overlap scheduler OFF (the AD spec-dec smoke does this
        # for every spec case). Overlap pipelines next-step scheduling against in-flight tokens, which
        # is incompatible with the one-model verify wrapper -> garbage if left on.
        args["disable_overlap_scheduler"] = True
        # DEBUG bisection: NOCAP=1 disables the hidden-state capture transform. Only valid with
        # DFLASH_SKIP_CTX=1 (the wrapper then never reads the capture buffers). Isolates whether the
        # capture graph-rewrite corrupts the target verify path.
        if os.environ.get("NOCAP") == "1":
            args["transforms"] = {"detect_hidden_states_for_capture": {"enabled": False}}
    experiment_config = {
        "args": args,
        "prompt": {
            "batch_size": 1,
            "queries": ["The capital of France is"],
            "sp_kwargs": {"max_tokens": 64, "temperature": 0.0, "top_k": None},
        },
    }
    cfg = ExperimentConfig(**experiment_config)
    print(f">>> build_and_run_ad world_size=1 runtime=trtllm SPEC={use_spec}")
    main(cfg)


if __name__ == "__main__":
    run()
