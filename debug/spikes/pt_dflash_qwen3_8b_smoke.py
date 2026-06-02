# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 0 reference: PyTorch-backend DFlash on Qwen3-8B — acceptance-rate baseline.

Mirrors tests/unittest/_torch/speculative/hw_agnostic/test_dflash.py::test_dflash_qwen3_8b but uses
HF ids (Qwen/Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16, both cached). Establishes the unwaived oracle's
acceptance rate that the AutoDeploy port must match. Acceptance = avg_decoded_tokens_per_iter - 1.
"""

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig

PROMPTS = [
    "The capital of France is",
    "The president of the United States is",
    "The future of AI is",
]


def main():
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=2048)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True)
    spec_config = DFlashDecodingConfig(
        max_draft_len=4,
        speculative_model="z-lab/Qwen3-8B-DFlash-b16",
    )
    llm = LLM(
        model="Qwen/Qwen3-8B",
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,  # phase-1 bring-up parity
        cuda_graph_config=cuda_graph_config,
        max_batch_size=4,
        kv_cache_config=kv_cache_config,
        max_seq_len=2048,
        enable_chunked_prefill=False,
        speculative_config=spec_config,
    )
    outputs = llm.generate(PROMPTS, SamplingParams(max_tokens=256, temperature=0))

    print("\n================ DFlash Qwen3-8B PyTorch reference ================")
    avg_accepted = []
    for i, o in enumerate(outputs):
        apt = o.avg_decoded_tokens_per_iter
        acc = apt - 1.0
        avg_accepted.append(acc)
        print(f"[{i}] avg_decoded_tokens_per_iter={apt:.3f}  accepted/iter={acc:.3f}")
        print(f"     prompt: {PROMPTS[i]!r}")
        print(f"     gen:    {o.outputs[0].text[:140]!r}")
    mean_accepted = sum(avg_accepted) / len(avg_accepted)
    print(
        f"\nMEAN ACCEPTED (draft tokens/iter) = {mean_accepted:.3f}  "
        f"(unit-test threshold >= 1.0; gsm8k ref acc ~ 87.11)"
    )
    print("==================================================================\n")
    llm.shutdown()


if __name__ == "__main__":
    main()
