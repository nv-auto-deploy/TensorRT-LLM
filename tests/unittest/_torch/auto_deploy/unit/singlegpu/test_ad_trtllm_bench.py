import json
import re
import subprocess
import tempfile
from pathlib import Path

import yaml
from _model_test_utils import _hf_model_dir_or_hub_id
from utils.cpp_paths import llm_root  # noqa: F401
from utils.llm_data import llm_models_root


def parse_kv_cache_metrics(log_output: str, free_mem_ratio: float = 0.8):
    """Parse KV cache metrics from the benchmark log output."""
    metrics = {}

    # Simple patterns based on actual log format
    patterns = {
        "current_cache_size": r"Current cache size:\s*(\d+)",
        "free_mem_pre_mb": r"Free memory before forward pass \(MB\):\s*(\d+)",
        "free_mem_post_mb": r"Free memory after forward pass \(MB\):\s*(\d+)",
    }

    # Extract metrics using simple regex patterns
    for metric_name, pattern in patterns.items():
        match = re.search(pattern, log_output, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            metrics[metric_name] = value
            print(f"  ✅ Found {metric_name}: {value}")
        else:
            print(f"  ❌ Could not find {metric_name}")

    # Calculate new_cache_size using the same formula as in resize_kv_cache
    # new_cache_size = free_mem_post * 1024 * 1024 * free_mem_ratio + current_cache_size
    if "free_mem_post_mb" in metrics and "current_cache_size" in metrics:
        metrics["new_cache_size"] = int(
            metrics["free_mem_post_mb"] * 1024 * 1024 * free_mem_ratio
            + metrics["current_cache_size"]
        )
        print(
            f"  ✅ Calculated new_cache_size: {metrics['new_cache_size']} (using free_mem_ratio={free_mem_ratio})"
        )
    else:
        print("  ❌ Cannot calculate new_cache_size - missing required metrics")

    return metrics


def run_benchmark(model_name: str, dataset_path: str, temp_dir: str, report_json_path: str = None):
    """Run benchmark and capture KV cache metrics from log output."""

    # Read the test config to get free_mem_ratio
    config_path = f"{temp_dir}/model_kwargs.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    free_mem_ratio = config.get("free_mem_ratio", 0.8)  # Default to 0.8 if not specified

    # Build the command to run the benchmark
    cmd = [
        "python",
        "-m",
        "tensorrt_llm.commands.bench",
        "--model",
        model_name,
        "throughput",
        "--backend",
        "_autodeploy",
        "--dataset",
        str(dataset_path),
        "--extra_llm_api_options",
        config_path,
    ]

    # Add report_json argument if path is provided
    if report_json_path:
        cmd.extend(["--report_json", report_json_path])

    print(f"🚀 Running benchmark command: {' '.join(cmd)}")
    print(f"📋 Using free_mem_ratio from config: {free_mem_ratio}")

    # Run benchmark as subprocess to capture ALL output
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check if the command succeeded
    assert result.returncode == 0, (
        f"Benchmark failed with return code {result.returncode}:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    # Combine stdout and stderr for parsing
    full_log_output = f"{result.stdout}\n{result.stderr}"

    # Parse KV cache metrics from the combined log output
    kv_cache_metrics = parse_kv_cache_metrics(full_log_output, free_mem_ratio)
    print("📊 KV Cache Metrics parsed from logs:")
    if kv_cache_metrics:
        for key, value in kv_cache_metrics.items():
            if "mb" in key.lower():
                print(f"  {key}: {value}MB")
            else:
                print(f"  {key}: {value} bytes")
    else:
        print("  ⚠️ No KV cache metrics were parsed successfully")

    # Return parsed JSON report with KV cache metrics if requested
    if report_json_path and Path(report_json_path).exists():
        with open(report_json_path, "r") as f:
            report_data = json.load(f)

        # Add KV cache metrics to the report
        report_data["kv_cache_metrics"] = kv_cache_metrics
        return report_data
    return None


def assert_kv_cache_metrics_within_bounds(
    actual_metrics: dict,
    golden_metrics: dict,
    tolerance_percentage: float = 10.0,
):
    """
    Assert that KV cache metrics are within tolerance of golden values.

    Args:
        actual_metrics: Measured KV cache metrics
        golden_metrics: Expected KV cache metrics
        tolerance_percentage: Relative tolerance percentage (10% by default)
    """
    for metric_name, golden_value in golden_metrics.items():
        if metric_name not in actual_metrics:
            raise AssertionError(f"KV cache metric '{metric_name}' not found in actual metrics")

        actual_value = actual_metrics[metric_name]
        relative_diff = (
            abs(actual_value - golden_value) / golden_value if golden_value > 0 else float("inf")
        )

        assert relative_diff <= tolerance_percentage / 100.0, (
            f"KV cache metric '{metric_name}' outside expected bounds! "
            f"Actual: {actual_value}, "
            f"Golden: {golden_value}, "
            f"Relative diff: {relative_diff:.2%}, "
            f"Tolerance: {tolerance_percentage}%"
        )


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


def assert_performance_within_tolerance(
    actual_tokens_per_sec: float,
    golden_tokens_per_sec: float,
    relative_tolerance: float = 0.15,
    absolute_tolerance: float = 10.0,
):
    """
    Assert that actual performance is within tolerance of golden result.
    Only fails if performance is WORSE than golden - improvements always pass.

    Args:
        actual_tokens_per_sec: Measured performance metric
        golden_tokens_per_sec: Expected performance metric
        relative_tolerance: Relative tolerance (15% by default)
        absolute_tolerance: Absolute tolerance (10 tokens/sec by default)
    """
    # If actual performance is better than or equal to golden, always pass
    if actual_tokens_per_sec >= golden_tokens_per_sec:
        print(
            f"✅ Performance improvement detected:"
            f" {actual_tokens_per_sec:.2f} >= {golden_tokens_per_sec:.2f} tokens/sec/user"
        )
        return

    # Performance is worse than golden - check if it's within acceptable tolerance
    performance_drop = golden_tokens_per_sec - actual_tokens_per_sec
    relative_drop = (
        performance_drop / golden_tokens_per_sec if golden_tokens_per_sec > 0 else float("inf")
    )

    # Performance should be within relative tolerance OR absolute tolerance
    within_relative_tolerance = relative_drop <= relative_tolerance
    within_absolute_tolerance = performance_drop <= absolute_tolerance

    assert within_relative_tolerance or within_absolute_tolerance, (
        f"Performance regression detected! "
        f"Actual: {actual_tokens_per_sec:.2f} tokens/sec/user, "
        f"Golden: {golden_tokens_per_sec:.2f} tokens/sec/user, "
        f"Performance drop: {performance_drop:.2f} tokens/sec ({relative_drop:.2%}), "
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
    """Test that runs autodeploy backend and compares performance and KV cache metrics to golden results."""
    model_name = _hf_model_dir_or_hub_id(
        f"{llm_models_root()}/TinyLlama-1.1B-Chat-v1.0", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    # Golden performance value for TinyLlama-1.1B-Chat-v1.0 with autodeploy backend
    GOLDEN_TOKENS_PER_SEC_PER_USER = 30.41  # tokens/sec/user (updated from actual test result)

    # Golden KV cache metrics (updated from actual test results)
    GOLDEN_KV_CACHE_METRICS = {
        "current_cache_size": 83886080,  # bytes - from logs
        "free_mem_pre_mb": 76151,  # MB - from logs
        "free_mem_post_mb": 75407,  # MB - from logs
        "new_cache_size": 71258625997,  # bytes - calculated: int(75407 * 1024 * 1024 * 0.9) + 83886080
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(f"{temp_dir}/model_kwargs.yaml", "w") as f:
            yaml.dump(
                {
                    "model_kwargs": {"num_hidden_layers": 10},
                    "cuda_graph_batch_sizes": [1, 2],
                    "max_batch_size": 128,
                    "free_mem_ratio": 0.9,
                },
                f,
            )

        dataset_path = prepare_dataset(llm_root, temp_dir, model_name)
        report_json_path = f"{temp_dir}/benchmark_report.json"

        # Run benchmark and capture performance + KV cache results
        report_data = run_benchmark(model_name, dataset_path, temp_dir, report_json_path)

        # Extract performance metrics
        assert report_data is not None, "Failed to capture benchmark report"
        assert "performance" in report_data, "Performance metrics not found in benchmark report"

        actual_tokens_per_sec = report_data["performance"].get("output_throughput_per_user_tok_s")
        assert actual_tokens_per_sec is not None, (
            "output_throughput_per_user_tok_s not found in performance metrics"
        )

        # Extract KV cache metrics (REQUIRED - test fails if not found)
        kv_cache_metrics = report_data.get("kv_cache_metrics", {})

        # Fail if no KV cache metrics were captured
        assert kv_cache_metrics, (
            "REQUIRED KV cache metrics not found! "
            "The autodeploy backend must log memory statistics for this test to pass. "
            "Expected metrics: current_cache_size, free_mem_pre_mb, free_mem_post_mb, new_cache_size"
        )

        # Ensure we have the minimum required metrics
        required_metrics = [
            "current_cache_size",
            "free_mem_pre_mb",
            "free_mem_post_mb",
            "new_cache_size",
        ]
        missing_metrics = [metric for metric in required_metrics if metric not in kv_cache_metrics]
        assert not missing_metrics, (
            f"Missing required KV cache metrics: {missing_metrics}. "
            f"Found metrics: {list(kv_cache_metrics.keys())}. "
            f"All of {required_metrics} are required for the test to pass."
        )

        # Print all metrics for visibility
        print("=== PERFORMANCE METRICS ===")
        print(f"Measured performance: {actual_tokens_per_sec:.2f} tokens/sec/user")
        print(f"Golden performance: {GOLDEN_TOKENS_PER_SEC_PER_USER:.2f} tokens/sec/user")

        print("=== KV CACHE METRICS (REQUIRED) ===")
        for metric_name, actual_value in kv_cache_metrics.items():
            golden_value = GOLDEN_KV_CACHE_METRICS.get(metric_name, "N/A")
            if "mb" in metric_name.lower():
                print(f"{metric_name}: {actual_value}MB (golden: {golden_value})")
            else:
                print(f"{metric_name}: {actual_value} bytes (golden: {golden_value})")

        # Performance validation (always required)
        assert_performance_within_tolerance(
            actual_tokens_per_sec,
            GOLDEN_TOKENS_PER_SEC_PER_USER,
            relative_tolerance=0.1,  # 10% relative tolerance
            absolute_tolerance=5.0,  # 5 tokens/sec absolute tolerance
        )

        # KV cache metrics validation
        print(f"Validating {len(kv_cache_metrics)} KV cache metrics against golden values...")
        tolerance_percentage = 1.0  # % tolerance for KV cache metrics
        assert_kv_cache_metrics_within_bounds(
            kv_cache_metrics, GOLDEN_KV_CACHE_METRICS, tolerance_percentage=tolerance_percentage
        )

        print("=== ALL TESTS PASSED ===")
        print(f"Performance: ✅ {actual_tokens_per_sec:.2f} tokens/sec/user within bounds")
        print(
            f"KV Cache Metrics: ✅ All {len(kv_cache_metrics)} metrics within {tolerance_percentage}% of golden values"
        )
