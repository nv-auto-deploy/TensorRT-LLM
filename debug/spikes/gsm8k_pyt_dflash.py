# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSM8K accuracy + acceptance for the PyTorch backend running DFlash + cudagraph.

The native-PyTorch DFlash counterpart to debug/spikes/gsm8k_ad_compare.py, so AD-vs-PyT can be
compared apples-to-apples: SAME GSM8K protocol (5-shot, greedy, num_samples, dataset) and SAME
acceptance metric (numAcceptedTokens/numDraftTokens from specDecodingStats). Config mirrors
tests/integration/defs/accuracy/test_llm_api_pytorch.py::test_dflash (max_batch_size=8, overlap ON,
CudaGraphConfig(max_batch_size=8), kv free_gpu_memory_fraction=0.6 -- PyT has no resize-corruption
bug, so no free_gpu_memory_fraction=0.0 workaround is needed here).

Env: MODEL=llama|qwen3 (default llama), NUM_SAMPLES=<int> (default 1319 = full GSM8K).
Run: CUDA_VISIBLE_DEVICES=0 MODEL=llama TRITON_CACHE_DIR=... LLM_MODELS_ROOT=... \
       python -u debug/spikes/gsm8k_pyt_dflash.py
"""

import os

from tensorrt_llm import LLM
from tensorrt_llm.evaluate import GSM8K
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

_MODELS = {
    "llama": ("llama-3.1-model/Llama-3.1-8B-Instruct", "LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
    "qwen3": ("Qwen3/Qwen3-8B", "Qwen3-8B-DFlash-b16"),
}


def _acceptance_from_stats(stats):
    """Identical to debug/spikes/gsm8k_ad_compare.py so the rate is directly comparable."""
    total_drafted = total_accepted = num_iters = 0
    for stat in stats:
        spec = stat.get("specDecodingStats", {}) or {}
        nd = spec.get("numDraftTokens", 0)
        na = spec.get("numAcceptedTokens", 0)
        if nd <= 0:
            continue
        num_iters += 1
        total_drafted += nd
        total_accepted += na
    rate = total_accepted / total_drafted if total_drafted else 0.0
    return rate, total_accepted, total_drafted, num_iters


def main():
    root = os.environ["LLM_MODELS_ROOT"]
    key = os.environ.get("MODEL", "llama")
    target_rel, draft_rel = _MODELS[key]
    target = f"{root}/{target_rel}"
    draft = f"{root}/{draft_rel}"
    num_samples = int(os.environ.get("NUM_SAMPLES", "1319"))

    tag = f"PYT MODEL={key} DFLASH+cudagraph N={num_samples}"
    print(f">>> {tag}\n>>> target={target}\n>>> draft={draft}", flush=True)

    llm = LLM(
        model=target,
        max_batch_size=8,
        disable_overlap_scheduler=False,  # overlap ON (matches test_llm_api_pytorch::test_dflash)
        cuda_graph_config=CudaGraphConfig(max_batch_size=8, enable_padding=True),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.6),
        speculative_config=DFlashDecodingConfig(max_draft_len=4, speculative_model=draft),
        enable_iter_perf_stats=True,
    )
    with llm:
        task = GSM8K(
            dataset_path=f"{root}/datasets/openai/gsm8k",
            num_samples=num_samples,
            random_seed=0,
        )
        sp = SamplingParams(
            max_tokens=256, truncate_prompt_tokens=4096, temperature=0.0, top_k=None
        )
        score = task.evaluate(llm, sampling_params=sp)
        print(f">>> RESULT {tag} GSM8K_accuracy={score:.3f}", flush=True)
        try:
            rate, na, nd, niter = _acceptance_from_stats(llm.get_stats(timeout=10))
            k = 4  # max_draft_len
            print(
                f">>> RESULT {tag} accept_rate={rate:.2%} "
                f"accepted_draft_per_step≈{rate * k:.3f} tokens_per_step≈{rate * k + 1:.3f} "
                f"(accepted_draft={na}/{nd} over {niter} batch-stat entries; oracle≈1.325)",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - acceptance is diagnostic; don't fail the run
            print(f">>> WARN could not compute acceptance from stats: {e!r}", flush=True)


if __name__ == "__main__":
    main()
