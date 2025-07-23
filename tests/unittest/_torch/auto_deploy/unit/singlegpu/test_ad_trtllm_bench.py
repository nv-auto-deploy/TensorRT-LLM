import json
import subprocess
import tempfile
from pathlib import Path

import yaml
from _model_test_utils import _hf_model_dir_or_hub_id
from click.testing import CliRunner
from utils.cpp_paths import llm_root  # noqa: F401
from utils.llm_data import llm_models_root

from tensorrt_llm.commands.bench import main


def prepare_dataset(root_dir: str, temp_dir: str, model_name: str):
    _DATASET_NAME = "synthetic_128_128.txt"
    dataset_path = Path(temp_dir, _DATASET_NAME)
    dataset_tool = Path(root_dir, "benchmarks", "cpp", "prepare_dataset.py")
    script_dir = Path(root_dir, "benchmarks", "cpp")

    # Generate a small dataset to run a test.
    command = [
        "python3",
        f"{dataset_tool}",
        "--stdout",
        "--tokenizer",
        model_name,
        "token-norm-dist",
        "--input-mean",
        "128",
        "--output-mean",
        "128",
        "--input-stdev",
        "0",
        "--output-stdev",
        "0",
        "--num-requests",
        "10",
    ]
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, cwd=str(script_dir), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to prepare dataset: {result.stderr}")
    # Grab the stdout and write it to a dataset file for passing to suite.
    with open(dataset_path, "w") as dataset:
        dataset.write(result.stdout)
    return dataset_path


def run_benchmark(model_name: str, dataset_path: str, temp_dir: str, report_json_path: str = None):
    """Run benchmark and optionally capture JSON performance report."""
    runner = CliRunner()

    args = [
        "--model",
        model_name,
        "throughput",
        "--backend",
        "_autodeploy",
        "--dataset",
        dataset_path,
        "--extra_llm_api_options",
        f"{temp_dir}/model_kwargs.yaml",
    ]

    # Add report_json argument if path is provided
    if report_json_path:
        args.extend(["--report_json", report_json_path])

    result = runner.invoke(main, args, catch_exceptions=False)
    assert result.exit_code == 0

    # Return parsed JSON report if requested
    if report_json_path and Path(report_json_path).exists():
        with open(report_json_path, "r") as f:
            return json.load(f)
    return None


def assert_performance_within_tolerance(
    actual_tokens_per_sec: float,
    golden_tokens_per_sec: float,
    relative_tolerance: float = 0.15,
    absolute_tolerance: float = 10.0,
):
    """
    Assert that actual performance is within tolerance of golden result.

    Args:
        actual_tokens_per_sec: Measured performance metric
        golden_tokens_per_sec: Expected performance metric
        relative_tolerance: Relative tolerance (15% by default)
        absolute_tolerance: Absolute tolerance (10 tokens/sec by default)
    """
    absolute_diff = abs(actual_tokens_per_sec - golden_tokens_per_sec)
    relative_diff = (
        absolute_diff / golden_tokens_per_sec if golden_tokens_per_sec > 0 else float("inf")
    )

    # Performance should be within relative tolerance OR absolute tolerance
    within_relative_tolerance = relative_diff <= relative_tolerance
    within_absolute_tolerance = absolute_diff <= absolute_tolerance

    assert within_relative_tolerance or within_absolute_tolerance, (
        f"Performance regression detected! "
        f"Actual: {actual_tokens_per_sec:.2f} tokens/sec/user, "
        f"Golden: {golden_tokens_per_sec:.2f} tokens/sec/user, "
        f"Relative diff: {relative_diff:.2%}, "
        f"Absolute diff: {absolute_diff:.2f} tokens/sec, "
        f"Tolerance: {relative_tolerance:.2%} relative OR {absolute_tolerance:.2f} tokens/sec absolute"
    )


def test_trtllm_bench(llm_root):  # noqa: F811
    model_name = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/model_kwargs.yaml", "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": 2},
                    "cuda_graph_batch_sizes": [1, 2],
                    "max_batch_size": 128,
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_name)
        run_benchmark(model_name, dataset_path, temp_dir)


def test_trtllm_bench_perf_golden_comparison(llm_root):  # noqa: F811
    """Test that runs autodeploy backend and compares per-user tokens per second to golden result."""
    model_name = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    # Golden performance value for TinyLlama-1.1B-Chat-v1.0 with autodeploy backend
    # This should be updated based on expected performance on the test hardware
    # For now, using a conservative value that should be achievable
    GOLDEN_TOKENS_PER_SEC_PER_USER = 40  # tokens/sec/user

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/model_kwargs.yaml", "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": 10},
                    "cuda_graph_batch_sizes": [1, 2],
                    "max_batch_size": 128,
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_name)
        report_json_path = f"{temp_dir}/benchmark_report.json"

        # Run benchmark and capture performance results
        report_data = run_benchmark(model_name, dataset_path, temp_dir, report_json_path)

        # Extract per-user tokens per second from the performance results
        assert report_data is not None, "Failed to capture benchmark report"
        assert "performance" in report_data, "Performance metrics not found in benchmark report"

        actual_tokens_per_sec = report_data["performance"].get("output_throughput_per_user_tok_s")
        assert actual_tokens_per_sec is not None, (
            "output_throughput_per_user_tok_s not found in performance metrics"
        )

        print(f"Measured performance: {actual_tokens_per_sec:.2f} tokens/sec/user")
        print(f"Golden performance: {GOLDEN_TOKENS_PER_SEC_PER_USER:.2f} tokens/sec/user")

        # Compare against golden result with tolerance
        assert_performance_within_tolerance(
            actual_tokens_per_sec,
            GOLDEN_TOKENS_PER_SEC_PER_USER,
            relative_tolerance=0.1,  # 10% relative tolerance
            absolute_tolerance=5.0,  # 5 tokens/sec absolute tolerance
        )
