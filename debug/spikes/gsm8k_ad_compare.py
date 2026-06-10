# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSM8K accuracy + DFlash acceptance comparison on the AutoDeploy backend.

Goal (per user): AD DFlash-ON GSM8K accuracy must MATCH AD DFlash-OFF (spec-dec is lossless under
greedy decoding), and the acceptance rate should be reasonable (compare vs the ~1.325 PyT oracle).
This validates the DFlash AD port end-to-end on a real benchmark before we turn on overlap
scheduling + monolithic cudagraph capture.

Env knobs:
  MODEL=llama|qwen3        target model (default llama)
  DFLASH=0|1               spec-dec off/on (default 1)
  NUM_SAMPLES=<int>        GSM8K samples (default 200; 1319 = full)
  ATTN=trtllm|flashinfer   attention backend (default trtllm)
  CUDA_VISIBLE_DEVICES=<n>

Run:
  CUDA_VISIBLE_DEVICES=0 MODEL=llama DFLASH=0 NUM_SAMPLES=200 \
    TRITON_CACHE_DIR=... LLM_MODELS_ROOT=... python -u debug/spikes/gsm8k_ad_compare.py
"""

import os

from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM
from tensorrt_llm.evaluate import GSM8K
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.sampling_params import SamplingParams

# (target, draft, target_layer_ids, hf_model_name) per model key. Paths are relative to
# LLM_MODELS_ROOT; layer_ids must match the draft's dflash_config.target_layer_ids.
_MODELS = {
    "llama": (
        "llama-3.1-model/Llama-3.1-8B-Instruct",
        "LLaMA3.1-8B-Instruct-DFlash-UltraChat",
        [1, 8, 15, 22, 29],
        "meta-llama/Llama-3.1-8B-Instruct",
    ),
    "qwen3": (
        "Qwen3/Qwen3-8B",
        "Qwen3-8B-DFlash-b16",
        [1, 9, 17, 25, 33],
        "Qwen/Qwen3-8B",
    ),
}


def _acceptance_from_stats(stats):
    """Mirror tests/.../test_llm_api_autodeploy.py::_check_acceptance_rate_stats."""
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
    # accept_rate = fraction of drafted tokens accepted (a ratio -> batch-aggregation safe).
    rate = total_accepted / total_drafted if total_drafted else 0.0
    return rate, total_accepted, total_drafted, num_iters


def main():
    root = os.environ["LLM_MODELS_ROOT"]
    key = os.environ.get("MODEL", "llama")
    target_rel, draft_rel, layer_ids, model_name = _MODELS[key]
    target = f"{root}/{target_rel}"
    draft = f"{root}/{draft_rel}"
    use_dflash = os.environ.get("DFLASH", "1") == "1"
    num_samples = int(os.environ.get("NUM_SAMPLES", "200"))
    attn = os.environ.get("ATTN", "trtllm")
    # Roadmap step #12: CUDAGRAPH=1 -> monolithic cudagraph capture (torch-cudagraph) instead of the
    # no-op torch-simple. Confirm accuracy+acceptance unchanged vs torch-simple.
    compile_backend = (
        "torch-cudagraph" if os.environ.get("CUDAGRAPH", "0") == "1" else "torch-simple"
    )

    kwargs = dict(
        tokenizer=target,
        runtime="trtllm",
        world_size=1,
        attn_backend=attn,
        compile_backend=compile_backend,
        skip_tokenizer_init=False,
        trust_remote_code=True,
        max_seq_len=8192,
        max_num_tokens=8192,
        enable_iter_perf_stats=True,
        kv_cache_config={"enable_block_reuse": False, "free_gpu_memory_fraction": 0.7},
    )
    if use_dflash:
        kwargs["speculative_config"] = DFlashDecodingConfig(
            max_draft_len=4, speculative_model=draft, target_layer_ids=layer_ids
        )
        # Spec-dec bring-up ran with the overlap scheduler OFF (one-model verify wrapper). Roadmap
        # step #11: re-enable overlap (OVERLAP=1) and confirm accuracy+acceptance are unchanged.
        # Eagle3 one-model AD already runs with overlap ON, so the infra supports it.
        if os.environ.get("OVERLAP", "0") != "1":
            kwargs["disable_overlap_scheduler"] = True
        # Keep the eval batch bounded for developer runs, but leave resize_kv_cache enabled through
        # the base free_gpu_memory_fraction setting so this harness exercises the production path.
        kwargs["max_batch_size"] = 16
        kwargs["cuda_graph_config"] = {"batch_sizes": [1, 2, 4, 8, 16], "enable_padding": True}

    tag = f"MODEL={key} DFLASH={int(use_dflash)} N={num_samples} ATTN={attn}"
    print(
        f">>> {tag}\n>>> target={target}\n>>> draft={draft if use_dflash else '(none)'}", flush=True
    )

    with AutoDeployLLM(model=target, **kwargs) as llm:
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

        if use_dflash:
            try:
                rate, na, nd, niter = _acceptance_from_stats(llm.get_stats(timeout=10))
                # Per-sequence accepted DRAFT tokens/step ≈ rate*K (numDraftTokens are batch-summed, so
                # dividing by niter would mix in batch size). +1 bonus token => tokens/step vs oracle.
                k = 4  # max_draft_len
                acc_draft_per_step = rate * k
                print(
                    f">>> RESULT {tag} accept_rate={rate:.2%} "
                    f"accepted_draft_per_step≈{acc_draft_per_step:.3f} "
                    f"tokens_per_step≈{acc_draft_per_step + 1:.3f} "
                    f"(accepted_draft={na}/{nd} over {niter} batch-stat entries; oracle≈1.325)",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001 - acceptance is diagnostic; don't fail the run
                print(f">>> WARN could not compute acceptance from stats: {e!r}", flush=True)


if __name__ == "__main__":
    main()
