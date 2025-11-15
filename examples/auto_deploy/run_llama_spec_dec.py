"""Test models with speculative decoding using TRT-LLM or AutoDeploy backend.

This script tests models using either the TRT-LLM engine directly or the AutoDeploy backend,
with optional speculative decoding.

Usage:
    # TRT-LLM backend (default) with downloaded speculative model adapter
    python run_llama_spec_dec.py --model meta-llama/Llama-3.1-8B-Instruct \
        --speculative-model-dir /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy\
            /autodeploy_data/hf_home\
            /hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6

    # TRT-LLM backend with auto-download speculative model adapter (only for Llama-3.1-8B-Instruct)
    python run_llama_spec_dec.py --model meta-llama/Llama-3.1-8B-Instruct --auto-download-draft-model

    # AutoDeploy backend with speculative model adapter
    python run_llama_spec_dec.py --backend autodeploy --model meta-llama/Llama-3.1-8B-Instruct \
        --auto-download-draft-model

    # Disable speculative decoding (baseline mode) with either backend
    python run_llama_spec_dec.py --backend trtllm --model meta-llama/Llama-3.1-8B-Instruct --no-spec-dec
"""

import argparse
import os
import shlex
import sys

from build_and_run_ad import ExperimentConfig
from build_and_run_ad import main as ad_main

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import DraftTargetDecodingConfig, KvCacheConfig

NUM_DRAFT_TOKENS = 3

# Test prompts
prompts = [
    "What is the capital of France?",
    "Please explain the concept of gravity in simple words and a single sentence.",
    "What is the capital of Norway?",
    "What is the highest mountain in the world?",
]


def download_model(model_id: str, cache_dir: str = None):
    """Download a model from Hugging Face Hub.

    Args:
        model_id: The model ID on Hugging Face Hub
        cache_dir: Optional cache directory

    Returns:
        Path to the downloaded model
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    print(f"\n{'=' * 80}")
    print(f"[INFO] Downloading model: {model_id}")
    print("=" * 80)

    try:
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
        )

        print("[SUCCESS] Model downloaded successfully!")
        print(f"[INFO] Model path: {model_path}")
        return model_path

    except Exception as e:
        print(f"[ERROR] Failed to download {model_id}: {e}")

        # Check if it's a gated model
        if "401" in str(e) or "gated" in str(e).lower() or "access" in str(e).lower():
            print("\n[HINT] This might be a gated model. You need to:")
            print("1. Accept the model's license on Hugging Face Hub")
            print(f"   Visit: https://huggingface.co/{model_id}")
            print("2. Login with your HF token:")
            print("   huggingface-cli login")

        raise


def ensure_speculative_model(
    speculative_model_dir: str = None,
    auto_download_draft_model: bool = False,
    model_name: str = None,
):
    """Ensure speculative model is available, downloading if necessary.

    Args:
        speculative_model_dir: Path to local speculative model directory
        auto_download_draft_model: If True, auto-download default speculative model (Llama-3.1-8B only)
        model_name: Base model name to validate speculative model compatibility

    Returns:
        Path to speculative model, or None if disabled

    Raises:
        SystemExit: If both flags are specified or if any operation fails
    """
    # Default speculative model for auto-download (only for Llama-3.1-8B-Instruct)
    DEFAULT_SPEC_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    # Check for conflicting flags
    if auto_download_draft_model and speculative_model_dir:
        print("[ERROR] Cannot specify both --auto-download-draft-model and --speculative-model-dir")
        print("[ERROR] Please use only one of these options")
        sys.exit(1)

    # Case 1: Auto-download speculative model
    if auto_download_draft_model:
        # Validate that we're using the right base model
        if model_name and DEFAULT_BASE_MODEL not in model_name:
            print(
                f"[WARNING] Auto-download speculative model is only supported for {DEFAULT_BASE_MODEL}"
            )
            print(f"[WARNING] You specified: {model_name}")
            print("[WARNING] The speculative model adapter may not be compatible with your model")

        print(f"[INFO] Auto-downloading speculative model adapter: {DEFAULT_SPEC_MODEL}")
        print(f"[INFO] This adapter is designed for: {DEFAULT_BASE_MODEL}")

        # Download the model (this will raise an exception if it fails)
        spec_model_path = download_model(DEFAULT_SPEC_MODEL)
        return spec_model_path

    # Case 2: Use specified speculative model directory
    if speculative_model_dir:
        print(f"[INFO] Using specified speculative model directory: {speculative_model_dir}")

        # Check if the directory exists
        if not os.path.exists(speculative_model_dir):
            print(f"[ERROR] Speculative model directory does not exist: {speculative_model_dir}")
            print("[ERROR] Please check the path and try again")
            sys.exit(1)

        return speculative_model_dir

    # Case 3: Neither flag specified - baseline mode
    print("[INFO] No speculative model specified. Running in baseline mode.")
    return None


def test_llama_spec_dec_with_trtllm(
    model: str, speculative_model_dir: str = None, enable_spec_dec: bool = True
):
    """Test model with TRT-LLM engine directly.

    Args:
        model: Model name or HuggingFace model ID
        speculative_model_dir: Path to speculative model (optional)
        enable_spec_dec: Whether to enable speculative decoding
    """
    print(f"\n{'=' * 80}")
    print("TRT-LLM Test Configuration")
    print("=" * 80)
    print(f"Base Model: {model}")
    speculative_info = speculative_model_dir if speculative_model_dir else "None (baseline mode)"
    print(f"Speculative Model: {speculative_info}")
    print("=" * 80 + "\n")

    # Configure speculative decoding (using DraftTargetDecodingConfig)
    spec_config = None
    if enable_spec_dec and speculative_model_dir:
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=NUM_DRAFT_TOKENS, speculative_model_dir=speculative_model_dir
        )

        print(f"[TRACE] Created DraftTargetDecodingConfig: {spec_config}")
        print(f"[TRACE] Speculative model dir: {speculative_model_dir}")
    else:
        print("[TRACE] Running without speculative decoding (baseline mode)")

    # Configure KV cache
    # Allocate 80% of free GPU memory for KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.3,
    )
    print("[TRACE] Created KvCacheConfig with free_gpu_memory_fraction=0.8")

    # Create TRT-LLM instance
    spec_mode = "speculative" if (enable_spec_dec and speculative_model_dir) else "baseline"
    print(f"\n[TRACE] Creating LLM instance ({spec_mode} mode)...")
    print(f"[TRACE] Model: {model}")

    try:
        # Try to use flashinfer attention backend to match AutoDeploy's attn
        llm = LLM(
            model=model,
            speculative_config=spec_config,
            kv_cache_config=kv_cache_config,
            attn_backend="flashinfer",
            tensor_parallel_size=2,
        )
        print("[TRACE] LLM instance created successfully!")

    except Exception as e:
        print("\n[ERROR] ===== FAILED! =====")
        print(f"[ERROR] Failed to create LLM instance: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        print("\n[DEBUG HINTS]:")
        print("1. Check if the model name is correct and accessible")
        print("2. Verify the speculative model directory is correct")
        print("3. Ensure sufficient GPU memory is available")
        print("4. Try running with --no-spec-dec to test baseline first")

        raise

    # Configure sampling parameters
    sampling_params = SamplingParams(max_tokens=100)
    print("[TRACE] Sampling parameters: max_tokens=100")

    # Run generation
    print(f"\n[TRACE] Starting generation with {len(prompts)} prompts in batch...")

    try:
        # Generate responses for all prompts in batch
        print(f"\n[TRACE] Processing {len(prompts)} prompts in batch...")
        responses = llm.generate(prompts, sampling_params)

        # Process results
        results = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            output_text = response.outputs[0].text
            results.append((prompt, output_text))
            print(f"[TRACE] Generated {len(output_text)} characters for prompt {i + 1}")

        print("\n[TRACE] ===== SUCCESS! =====")
        print(f"[TRACE] Generation completed with {model} using TRT-LLM!")

        # Print all results
        print(f"\n[TRACE] Generated {len(results)} outputs:")
        for i, (prompt, output) in enumerate(results):
            print(f"\n[TRACE] Output {i}:")
            print(f"  Prompt: {prompt}")
            print(f"  Response: {output}")

        return {
            "prompts_and_outputs": results,
            "mode": spec_mode,
            "model": model,
        }

    except Exception as e:
        print("\n[ERROR] ===== FAILED! =====")
        print(f"[ERROR] Generation failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        print("\n[DEBUG HINTS]:")
        print("1. Check if the generation parameters are valid")
        print("2. Verify sufficient GPU memory for generation")
        print("3. Check if the model was loaded correctly")

        raise


def test_llama_spec_dec_with_autodeploy(
    model: str, speculative_model_dir: str = None, enable_spec_dec: bool = True
):
    """Test model with AutoDeploy backend.

    Args:
        model: Model name or HuggingFace model ID or local path
        speculative_model_dir: Path to speculative model (optional)
        enable_spec_dec: Whether to enable speculative decoding
    """
    print(f"\n{'=' * 80}")
    print("AutoDeploy Test Configuration")
    print("=" * 80)
    print(f"Base Model: {model}")
    speculative_info = speculative_model_dir if speculative_model_dir else "None (baseline mode)"
    print(f"Speculative Model: {speculative_info}")
    print("=" * 80 + "\n")

    # Configure speculative decoding (using DraftTargetDecodingConfig)
    spec_config = None
    if enable_spec_dec and speculative_model_dir:
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=NUM_DRAFT_TOKENS, speculative_model_dir=speculative_model_dir
        )
        print(f"[TRACE] Created DraftTargetDecodingConfig: {spec_config}")
        print(f"[TRACE] Speculative model dir: {speculative_model_dir}")
    else:
        print("[TRACE] Running without speculative decoding (baseline mode)")

    # Configure AutoDeploy LLM arguments
    llm_args = {
        "model": model,
        "skip_loading_weights": False,  # We want to load weights
        "speculative_config": spec_config,
        "runtime": "trtllm",  # AutoDeploy runtime
        "world_size": 1,  # 2 GPUs
    }

    # Configure experiment with prompts
    experiment_config = {
        "args": llm_args,
        "benchmark": {"enabled": False},  # Disable benchmarking
        "prompt": {
            "batch_size": 4,
            "queries": prompts,
        },
    }

    # Create ExperimentConfig
    spec_mode = "speculative" if (enable_spec_dec and speculative_model_dir) else "baseline"
    print(f"\n[TRACE] Creating ExperimentConfig ({spec_mode} mode)...")
    print(f"[TRACE] Model: {model}")

    try:
        cfg = ExperimentConfig(**experiment_config)
        print("[TRACE] ExperimentConfig created successfully!")
    except Exception as e:
        print("\n[ERROR] ===== FAILED! =====")
        print(f"[ERROR] Failed to create ExperimentConfig: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        print("\n[DEBUG HINTS]:")
        print("1. Check if all llm_args fields are valid for AutoDeploy's LlmArgs")
        print("2. Verify the model path is correct")
        print("3. Check if speculative_config is supported by AutoDeploy")

        raise

    # Add sampling parameters
    cfg.prompt.sp_kwargs = {
        "max_tokens": 50,
        "top_k": None,
        "temperature": 0.0,
    }

    # Run the experiment
    print(f"\n[TRACE] Starting AutoDeploy generation ({spec_mode} mode)...")

    try:
        result = ad_main(cfg)

        print("\n[TRACE] ===== SUCCESS! =====")
        print(f"[TRACE] Generation completed with {model} using AutoDeploy!")

        # Check if we got valid outputs
        if "prompts_and_outputs" in result:
            print(f"\n[TRACE] Generated {len(result['prompts_and_outputs'])} outputs:")
            for i, (prompt, output) in enumerate(result["prompts_and_outputs"]):
                print(f"\n[TRACE] Output {i}:")
                print(f"  Prompt: {prompt}")
                print(f"  Response: {output}")

        return result

    except Exception as e:
        print("\n[ERROR] ===== FAILED! =====")
        print(f"[ERROR] Generation failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        print("\n[DEBUG HINTS]:")
        print("1. Check if 'speculative_config' is supported in AutoDeploy LlmArgs")
        print("2. Verify sufficient GPU memory for generation")
        print("3. Check if AutoDeploy backend supports speculative decoding")
        print("4. Try running with --no-spec-dec to test baseline first")

        raise


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test models with speculative decoding using TRT-LLM or AutoDeploy backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use TRT-LLM backend (default) with downloaded speculative model adapter
  %(prog)s --model meta-llama/Llama-3.1-8B-Instruct \\
      --speculative-model-dir /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/\\
autodeploy_data/hf_home/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6

  # Use TRT-LLM backend with auto-download speculative model adapter (simplest option for Llama-3.1-8B)
  %(prog)s --model meta-llama/Llama-3.1-8B-Instruct --auto-download-draft-model

  # Use AutoDeploy backend with auto-download speculative model adapter
  %(prog)s --backend autodeploy --model meta-llama/Llama-3.1-8B-Instruct --auto-download-draft-model

  # Run in baseline mode without speculative decoding (TRT-LLM)
  %(prog)s --model meta-llama/Llama-3.1-8B-Instruct --no-spec-dec

  # Run in baseline mode without speculative decoding (AutoDeploy)
  %(prog)s --backend autodeploy --model meta-llama/Llama-3.1-8B-Instruct --no-spec-dec

Notes:
  - --backend: Choose 'trtllm' (default) or 'autodeploy'
  - Use exactly ONE of: --auto-download-draft-model, --speculative-model-dir, or --no-spec-dec
  - --auto-download-draft-model: Downloads speculative model adapter for Llama-3.1-8B-Instruct
  - --speculative-model-dir: Uses an already-downloaded local directory path
  - --no-spec-dec: Runs in baseline mode without speculative decoding
  - If none specified, defaults to baseline mode
        """,
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["trtllm", "autodeploy"],
        default="trtllm",
        help="Backend to use: 'trtllm' for TRT-LLM engine directly, "
        "'autodeploy' for AutoDeploy backend (default: trtllm)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or HuggingFace model ID (default: meta-llama/Llama-3.1-8B-Instruct)",
    )

    parser.add_argument(
        "--speculative-model-dir",
        type=str,
        default=None,
        help="Path to local speculative model directory. "
        "Cannot be used with --auto-download-draft-model",
    )

    parser.add_argument(
        "--auto-download-draft-model",
        action="store_true",
        help="Auto-download speculative model adapter for Llama-3.1-8B-Instruct (TinyLlama/TinyLlama-1.1B-Chat-v1.0). "
        "Cannot be used with --speculative-model-dir",
    )

    parser.add_argument(
        "--no-spec-dec",
        action="store_true",
        help="Disable speculative decoding (baseline mode)",
    )

    args = parser.parse_args()

    # Ensure speculative model is available
    speculative_model_path = None
    if not args.no_spec_dec:
        speculative_model_path = ensure_speculative_model(
            speculative_model_dir=args.speculative_model_dir,
            auto_download_draft_model=args.auto_download_draft_model,
            model_name=args.model,
        )

    # Run the test with the appropriate backend
    if args.backend == "trtllm":
        result = test_llama_spec_dec_with_trtllm(
            model=args.model,
            speculative_model_dir=speculative_model_path,
            enable_spec_dec=not args.no_spec_dec,
        )
    elif args.backend == "autodeploy":
        result = test_llama_spec_dec_with_autodeploy(
            model=args.model,
            speculative_model_dir=speculative_model_path,
            enable_spec_dec=not args.no_spec_dec,
        )
    else:
        print(f"[ERROR] Unknown backend: {args.backend}")
        sys.exit(1)

    return result


if __name__ == "__main__":
    cmd = " ".join(shlex.quote(arg) for arg in sys.argv)
    print(f"Run command: {cmd}")
    main()
