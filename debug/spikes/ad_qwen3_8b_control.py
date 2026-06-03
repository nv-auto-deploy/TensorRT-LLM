# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control: plain AD Qwen3-8B (NO DFlash spec) to check the target forward in isolation.

If this is coherent, the DFlash E2E garbage is a DFlashWrapper bug; if garbage, it's a target setup
issue. Same target + settings as ad_dflash_qwen3_8b_smoke.py.

Run: CUDA_VISIBLE_DEVICES=<free> LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     python debug/spikes/ad_qwen3_8b_control.py
"""

import sys

sys.path.append("tests/unittest/utils")
from llm_data import llm_models_root  # noqa: E402

from tensorrt_llm._torch.auto_deploy import LLM  # noqa: E402
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, SamplingParams  # noqa: E402

PROMPTS = ["The capital of France is"]


def main():
    root = llm_models_root()
    target = f"{root}/Qwen3/Qwen3-8B"
    import os

    variant = os.environ.get("CTRL_VARIANT", "vanilla")
    kwargs = dict(
        model=target,
        attn_backend="trtllm",
        max_seq_len=2048,
        world_size=int(os.environ.get("CTRL_WORLD_SIZE", "1")),
        compile_backend="torch-simple",
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=2048),
    )
    if variant == "vanilla":
        # No cuda_graph_config (default max_batch_size=128 -> match it), most vanilla path.
        kwargs["max_batch_size"] = 128
    else:  # "dflash_like" -- exactly the DFlash smoke's config minus spec
        kwargs["max_batch_size"] = 4
        kwargs["cuda_graph_config"] = CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True)
    print(f">>> CONTROL variant={variant} kwargs_keys={sorted(kwargs)}")
    llm = LLM(**kwargs)
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=64, temperature=0))
    print("\n================ AD Qwen3-8B CONTROL (no DFlash) ================")
    for i, o in enumerate(outputs):
        print(f"[{i}] gen: {o.outputs[0].text[:160]!r}")
    print("=================================================================\n")
    llm.shutdown()


if __name__ == "__main__":
    main()
