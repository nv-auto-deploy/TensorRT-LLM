# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AutoDeploy-backend DFlash E2E smoke on Qwen3-8B (internal checkpoints).

First end-to-end bring-up of the AD DFlash path (factory -> export -> cache insertion ->
DFlashWrapper._forward_with_kv_cache). torch-simple compile backend for bring-up. Prints the
acceptance rate (avg_decoded_tokens_per_iter - 1) to compare against the PyTorch oracle (~1.325).

Run: CUDA_VISIBLE_DEVICES=<free> TRITON_CACHE_DIR=... LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     python debug/spikes/ad_dflash_qwen3_8b_smoke.py
"""

import sys

sys.path.append("tests/unittest/utils")
from llm_data import llm_models_root  # noqa: E402

from tensorrt_llm._torch.auto_deploy import LLM  # noqa: E402
from tensorrt_llm.llmapi import DFlashDecodingConfig, KvCacheConfig, SamplingParams  # noqa: E402

PROMPTS = ["The capital of France is"]


def main():
    root = llm_models_root()
    target = f"{root}/Qwen3/Qwen3-8B"
    draft = f"{root}/Qwen3-8B-DFlash-b16"
    spec = DFlashDecodingConfig(
        max_draft_len=4,
        speculative_model=draft,
        target_layer_ids=[1, 9, 17, 25, 33],
    )
    llm = LLM(
        model=target,
        speculative_config=spec,
        attn_backend="trtllm",
        max_seq_len=2048,
        max_batch_size=4,
        world_size=1,
        compile_backend="torch-simple",  # bring-up phase (no cudagraph yet)
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=2048),
    )
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=64, temperature=0))
    print("\n================ AD DFlash Qwen3-8B smoke ================")
    for i, o in enumerate(outputs):
        apt = o.avg_decoded_tokens_per_iter
        print(f"[{i}] avg_decoded_tokens_per_iter={apt:.3f}  accepted/iter={apt - 1.0:.3f}")
        print(f"     gen: {o.outputs[0].text[:140]!r}")
    print("(PyTorch oracle ~1.325 accepted/iter)")
    print("==========================================================\n")
    llm.shutdown()


if __name__ == "__main__":
    main()
