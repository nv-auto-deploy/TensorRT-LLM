# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
from pathlib import Path

from examples.auto_deploy.build_and_run_ad import ExperimentConfig, build_llm_from_config
from tensorrt_llm.evaluate import GSM8K
from tensorrt_llm.sampling_params import SamplingParams


def _summarize_acceptance(stats):
    total_drafted = 0
    total_accepted = 0
    spec_iters = 0

    for stat in stats:
        spec = stat.get("specDecodingStats") or {}
        drafted = int(spec.get("numDraftTokens", 0) or 0)
        accepted = int(spec.get("numAcceptedTokens", 0) or 0)
        if drafted <= 0:
            continue
        spec_iters += 1
        total_drafted += drafted
        total_accepted += accepted

    rate = total_accepted / total_drafted if total_drafted else None
    return {
        "spec_iters": spec_iters,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_rate": rate,
    }


def main():
    model = os.environ.get("QWEN35_MTP_MODEL", "Qwen/Qwen3.5-397B-A17B-FP8")
    registry_config_id = os.environ.get("QWEN35_MTP_REGISTRY_CONFIG_ID", "qwen3_5_moe_400b_fp8_mtp")
    compile_backend = os.environ.get("QWEN35_MTP_COMPILE_BACKEND", "torch-simple")
    attn_backend = os.environ.get("QWEN35_MTP_ATTN_BACKEND")
    max_batch_size = int(os.environ.get("QWEN35_MTP_MAX_BATCH_SIZE", "4"))
    num_samples = int(os.environ.get("QWEN35_GSM8K_NUM_SAMPLES", "128"))
    dataset_path = os.environ.get(
        "QWEN35_GSM8K_DATASET_PATH",
        "/home/gramnarayan/dev/model-symlinks/datasets/openai/gsm8k",
    )
    output_dir = os.environ.get("QWEN35_GSM8K_OUTPUT_DIR")
    out_path = Path(os.environ.get("QWEN35_GSM8K_OUT", "/tmp/qwen35_gsm8k_probe.json"))

    llm_args = {
        "compile_backend": compile_backend,
        "disable_overlap_scheduler": False,
        "enable_chunked_prefill": False,
        "max_num_tokens": 8192,
        "max_batch_size": max_batch_size,
        "cuda_graph_config": {
            "max_batch_size": max_batch_size,
            "batch_sizes": [1, max_batch_size],
        },
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": 0.7,
            "tokens_per_block": 32,
        },
        "transforms": {
            "insert_cached_causal_conv": {
                "backend": "triton_causal_conv",
            },
            "multi_stream_gemm": {
                "enabled": False,
            },
            "multi_stream_moe": {
                "enabled": False,
            },
        },
        "enable_iter_perf_stats": True,
        "enable_iter_req_stats": True,
        "iter_stats_max_iterations": 100000,
        "print_iter_log": False,
    }
    if attn_backend is not None:
        llm_args["attn_backend"] = attn_backend

    config = ExperimentConfig(
        extra_cli_args=[],
        model=model,
        use_registry=True,
        registry_config_id=registry_config_id,
        args=llm_args,
    )

    llm = build_llm_from_config(config)
    try:
        evaluator = GSM8K(
            dataset_path=dataset_path,
            num_samples=num_samples,
            random_seed=0,
            output_path=output_dir,
        )
        sampling_params = SamplingParams(
            max_tokens=256,
            truncate_prompt_tokens=4096,
            temperature=0,
        )
        accuracy = evaluator.evaluate(
            llm,
            sampling_params,
            streaming=False,
            scores_filter=None,
        )
        stats = llm.get_stats()
        summary = _summarize_acceptance(stats)
        result = {
            "config": {
                "model": model,
                "registry_config_id": registry_config_id,
                "compile_backend": compile_backend,
                "attn_backend": attn_backend,
                "max_batch_size": max_batch_size,
                "num_samples": num_samples,
                "dataset_path": dataset_path,
                "llm_args": llm_args,
            },
            "accuracy": accuracy,
            "summary": summary,
            "num_stats": len(stats),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
