# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary Qwen3.5 MTP GSM8K validation matrix for AutoDeploy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "integration"))

from defs.accuracy.accuracy_core import GSM8K  # noqa: E402
from test_common.llm_data import hf_id_to_local_model_dir  # noqa: E402

from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM  # noqa: E402
from tensorrt_llm.sampling_params import SamplingParams  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-397B-A17B"
MODEL_NAME_FP8 = "Qwen/Qwen3.5-397B-A17B-FP8"

REGISTRY_DIR = REPO_ROOT / "examples" / "auto_deploy" / "model_registry"
CONFIG_DIR = REGISTRY_DIR / "configs"


def _registry_yaml_extra(config_id: str) -> list[str]:
    with open(REGISTRY_DIR / "models.yaml") as f:
        registry = yaml.safe_load(f)
    for entry in registry["models"]:
        if entry.get("config_id") == config_id:
            return [str(CONFIG_DIR / cfg) for cfg in entry["yaml_extra"]]
    raise ValueError(f"Unknown AutoDeploy registry config_id: {config_id}")


def _write_overlay(args: argparse.Namespace, variant_name: str, compile_backend: str) -> str:
    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes = [size for size in batch_sizes if size <= args.max_batch_size]
    overlay: dict[str, Any] = {
        "compile_backend": compile_backend,
        "attn_backend": "trtllm",
        "runtime": "trtllm",
        "max_seq_len": args.max_seq_len,
        "max_num_tokens": args.max_num_tokens,
        "max_batch_size": args.max_batch_size,
        "enable_iter_perf_stats": args.enable_iter_stats,
        "enable_iter_req_stats": args.enable_iter_stats,
        "print_iter_log": args.print_iter_log,
        "max_stats_len": args.max_stats_len,
        "kv_cache_config": {
            "enable_block_reuse": False,
            "free_gpu_memory_fraction": args.free_gpu_memory_fraction,
            "tokens_per_block": 32,
        },
        "transforms": {
            "multi_stream_gemm": {
                "enabled": False,
            },
            "multi_stream_moe": {
                "enabled": False,
            },
        },
    }
    if compile_backend == "torch-cudagraph":
        overlay["cuda_graph_config"] = {
            "max_batch_size": args.max_batch_size,
            "batch_sizes": batch_sizes,
        }

    overlay_path = Path(args.output_dir) / f"{variant_name}_overlay.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overlay_path, "w") as f:
        yaml.safe_dump(overlay, f, sort_keys=False)
    return str(overlay_path)


def _sum_spec_stats(obj: Any) -> tuple[int, int]:
    if isinstance(obj, dict):
        drafted = 0
        accepted = 0
        if "numDraftTokens" in obj:
            drafted += int(obj.get("numDraftTokens") or 0)
            accepted += int(obj.get("numAcceptedTokens") or 0)
        for value in obj.values():
            child_drafted, child_accepted = _sum_spec_stats(value)
            drafted += child_drafted
            accepted += child_accepted
        return drafted, accepted
    if isinstance(obj, list):
        drafted = 0
        accepted = 0
        for value in obj:
            child_drafted, child_accepted = _sum_spec_stats(value)
            drafted += child_drafted
            accepted += child_accepted
        return drafted, accepted
    return 0, 0


def _run_variant(args: argparse.Namespace, *, compile_backend: str, mtp: bool) -> dict[str, Any]:
    variant_name = f"{compile_backend}_{'mtp' if mtp else 'no_mtp'}".replace("-", "_")
    config_id = "qwen3_5_moe_400b_fp8_mtp" if mtp else "qwen3_5_moe_400b_fp8"
    yaml_extra = _registry_yaml_extra(config_id)
    yaml_extra.append(_write_overlay(args, variant_name, compile_backend))

    model_path = hf_id_to_local_model_dir(MODEL_NAME_FP8)
    sampling_params = SamplingParams(
        end_id=-1,
        pad_id=-1,
        n=1,
        use_beam_search=False,
        max_tokens=args.max_output_len,
        truncate_prompt_tokens=GSM8K.MAX_INPUT_LEN,
    )

    evaluator = GSM8K.EVALUATOR_CLS(
        dataset_path=GSM8K.DATASET_DIR,
        num_samples=args.num_samples,
        random_seed=0,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
        chat_template_kwargs={"enable_thinking": False},
        log_samples=args.log_samples,
        output_path=str(Path(args.output_dir) / variant_name),
    )

    with AutoDeployLLM(
        model=model_path,
        tokenizer=model_path,
        yaml_extra=yaml_extra,
        trust_remote_code=True,
        skip_tokenizer_init=False,
    ) as llm:
        score = evaluator.evaluate(llm, sampling_params=sampling_params, scores_filter=None)
        stats = llm.get_stats(timeout=args.stats_timeout)

    drafted, accepted = _sum_spec_stats(stats)
    acceptance = accepted / drafted if drafted else None
    result = {
        "variant": variant_name,
        "model": MODEL_NAME_FP8,
        "reference_model": MODEL_NAME,
        "compile_backend": compile_backend,
        "mtp": mtp,
        "num_samples": args.num_samples,
        "accuracy": score,
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": acceptance,
        "yaml_extra": yaml_extra,
    }
    result_path = Path(args.output_dir) / f"{variant_name}_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _selected_variants(args: argparse.Namespace) -> list[tuple[str, bool]]:
    if args.variant != "matrix":
        backend, mtp_text = args.variant.rsplit("_", 1)
        return [(backend, mtp_text == "mtp")]
    return [
        ("torch-simple", False),
        ("torch-simple", True),
        ("torch-cudagraph", False),
        ("torch-cudagraph", True),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=[
            "matrix",
            "torch-simple_no_mtp",
            "torch-simple_mtp",
            "torch-cudagraph_no_mtp",
            "torch-cudagraph_mtp",
        ],
        default="matrix",
    )
    parser.add_argument("--num-samples", type=int, default=1319)
    parser.add_argument("--max-output-len", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--free-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--max-stats-len", type=int, default=100000)
    parser.add_argument("--stats-timeout", type=float, default=5)
    parser.add_argument("--output-dir", default="/tmp/qwen35_gsm8k_matrix")
    parser.add_argument("--enable-iter-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-iter-log", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-samples", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = [
        _run_variant(args, compile_backend=backend, mtp=mtp)
        for backend, mtp in _selected_variants(args)
    ]

    if args.variant == "matrix":
        summary_path = Path(args.output_dir) / "matrix_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
