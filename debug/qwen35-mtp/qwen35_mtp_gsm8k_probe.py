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
import sys
from pathlib import Path

from examples.auto_deploy.build_and_run_ad import ExperimentConfig, build_llm_from_config
from tensorrt_llm.sampling_params import SamplingParams

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(_REPO_ROOT / "tests" / "integration"))

from defs.accuracy.accuracy_core import GSM8K as GSM8KTask  # noqa: E402


def _summarize_acceptance(stats):
    total_drafted = 0
    total_accepted = 0
    spec_iters = 0
    rows = []

    for stat in stats:
        spec = stat.get("specDecodingStats", {})
        drafted = int(spec.get("numDraftTokens", 0) or 0)
        accepted = int(spec.get("numAcceptedTokens", 0) or 0)
        if drafted <= 0:
            continue
        spec_iters += 1
        total_drafted += drafted
        total_accepted += accepted
        rows.append(
            {
                "iter": stat.get("iter"),
                "drafted": drafted,
                "accepted": accepted,
                "acceptanceLength": spec.get("acceptanceLength"),
                "numRequestsWithDraftTokens": spec.get("numRequestsWithDraftTokens"),
            }
        )

    rate = total_accepted / total_drafted if total_drafted else 0.0
    return {
        "spec_iters": spec_iters,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_rate": rate,
        "rows": rows,
    }


def main():
    config = ExperimentConfig(
        extra_cli_args=[],
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        use_registry=True,
        registry_config_id="qwen3_5_moe_400b_fp8_mtp",
        args={
            "compile_backend": "torch-simple",
            "max_batch_size": 1,
            "cuda_graph_config": {
                "max_batch_size": 1,
                "batch_sizes": [1],
            },
            "enable_iter_perf_stats": True,
            "enable_iter_req_stats": True,
            "iter_stats_max_iterations": 1000,
            "print_iter_log": False,
        },
    )

    output_path = Path("/tmp/qwen35_mtp_gsm8k_lm_eval")
    llm = build_llm_from_config(config)
    try:
        evaluator = GSM8KTask.EVALUATOR_CLS(
            dataset_path=GSM8KTask.DATASET_DIR,
            random_seed=0,
            num_samples=1,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            chat_template_kwargs={"enable_thinking": False},
            log_samples=True,
            output_path=str(output_path),
        )
        score = evaluator.evaluate(
            llm,
            sampling_params=SamplingParams(
                max_tokens=512,
                truncate_prompt_tokens=GSM8KTask.MAX_INPUT_LEN,
            ),
            **GSM8KTask.EVALUATE_KWARGS,
        )
        stats = llm.get_stats()
        summary = _summarize_acceptance(stats)
        result = {
            "score": score,
            "summary": summary,
            "num_stats": len(stats),
            "output_path": str(output_path),
            "stats": stats,
        }
        result_path = Path("/tmp/qwen35_mtp_gsm8k_probe.json")
        result_path.write_text(json.dumps(result, indent=2))
        print(json.dumps({"score": score, "summary": summary, "num_stats": len(stats)}, indent=2))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
