# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 0 reference (CI cluster): PyTorch DFlash on Qwen3-8B using the INTERNAL checkpoints.

Uses paths under llm_models_root() -- exactly what tests/.../test_dflash.py::test_dflash_qwen3_8b
uses. Prints the real acceptance rate (avg_decoded_tokens_per_iter - 1) per request + mean, which the
unit test asserts >= 1.0 but does not print. Establishes the oracle number the AutoDeploy port matches.
"""

import sys

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig

sys.path.append("tests/unittest/utils")
from llm_data import llm_models_root  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "The future of AI is",
]


def main(disable_overlap_scheduler: bool):
    root = llm_models_root()
    target = f"{root}/Qwen3/Qwen3-8B"
    draft = f"{root}/Qwen3-8B-DFlash-b16"
    print(
        f"target={target}\ndraft ={draft}\noverlap_scheduler_disabled={disable_overlap_scheduler}"
    )

    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=2048)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True)
    spec_config = DFlashDecodingConfig(max_draft_len=4, speculative_model=draft)
    llm = LLM(
        model=target,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=4,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        enable_chunked_prefill=False,
        speculative_config=spec_config,
    )
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=256, temperature=0))

    print("\n================ DFlash Qwen3-8B PyTorch reference (INTERNAL ckpt) ================")
    avg_accepted = []
    for i, o in enumerate(outputs):
        apt = o.avg_decoded_tokens_per_iter
        acc = apt - 1.0
        avg_accepted.append(acc)
        print(f"[{i}] avg_decoded_tokens_per_iter={apt:.3f}  accepted/iter={acc:.3f}")
        print(f"     gen: {o.outputs[0].text[:140]!r}")
    mean_accepted = sum(avg_accepted) / len(avg_accepted)
    print(
        f"\nMEAN ACCEPTED (draft tokens/iter) = {mean_accepted:.3f}  (unit-test threshold >= 1.0)"
    )
    print("max_draft_len=4 => theoretical max accepted/iter = 4.0")
    print("==================================================================================\n")
    llm.shutdown()


if __name__ == "__main__":
    overlap_off = "--overlap-on" not in sys.argv
    main(disable_overlap_scheduler=overlap_off)
