"""Test models with Eagle3 speculative decoding using TRT-LLM engine directly.

This script tests models using the TRT-LLM engine with optional Eagle3 speculative decoding.

Usage:
    # Use specific model with downloaded Eagle3 adapter (for Llama-3.1-8B)
    python run_trtllm_llama_eagle.py --model meta-llama/Llama-3.1-8B-Instruct \
        --speculative-model-dir /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy\
            /autodeploy_data/hf_home\
            /hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots/ada412b672e293d682423de84a095447bf38a637

    # Auto-download Eagle3 adapter (only supported for Llama-3.1-8B-Instruct)
    python run_trtllm_llama_eagle.py --model meta-llama/Llama-3.1-8B-Instruct --auto-download-eagle

    # Disable speculative decoding (baseline mode)
    python run_trtllm_llama_eagle.py --model meta-llama/Llama-3.1-8B-Instruct --no-eagle
"""

import argparse
import os
import sys

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import EagleDecodingConfig, KvCacheConfig


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
    speculative_model_dir: str = None, auto_download_eagle: bool = False, model_name: str = None
):
    """Ensure speculative model is available, downloading if necessary.

    Args:
        speculative_model_dir: Path to local Eagle3 speculative model directory
        auto_download_eagle: If True, auto-download default Eagle3 model (Llama-3.1-8B only)
        model_name: Base model name to validate Eagle compatibility

    Returns:
        Path to speculative model, or None if disabled

    Raises:
        SystemExit: If both flags are specified or if any operation fails
    """
    # Default Eagle3 model for auto-download (only for Llama-3.1-8B-Instruct)
    DEFAULT_EAGLE_MODEL = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
    DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    # Check for conflicting flags
    if auto_download_eagle and speculative_model_dir:
        print("[ERROR] Cannot specify both --auto-download-eagle and --speculative-model-dir")
        print("[ERROR] Please use only one of these options")
        sys.exit(1)

    # Case 1: Auto-download Eagle3 model
    if auto_download_eagle:
        # Validate that we're using the right base model
        if model_name and DEFAULT_BASE_MODEL not in model_name:
            print(f"[WARNING] Auto-download Eagle is only supported for {DEFAULT_BASE_MODEL}")
            print(f"[WARNING] You specified: {model_name}")
            print("[WARNING] The Eagle adapter may not be compatible with your model")

        print(f"[INFO] Auto-downloading Eagle3 adapter: {DEFAULT_EAGLE_MODEL}")
        print(f"[INFO] This adapter is designed for: {DEFAULT_BASE_MODEL}")

        # Download the model (this will raise an exception if it fails)
        eagle_path = download_model(DEFAULT_EAGLE_MODEL)
        return eagle_path

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


def test_llama_eagle_with_trtllm(
    model: str, speculative_model_dir: str = None, enable_eagle: bool = True
):
    """Test model with TRT-LLM engine directly.

    Args:
        model: Model name or HuggingFace model ID
        speculative_model_dir: Path to Eagle3 speculative model (optional)
        enable_eagle: Whether to enable Eagle3 speculative decoding
    """
    print(f"\n{'=' * 80}")
    print("TRT-LLM Test Configuration")
    print("=" * 80)
    print(f"Base Model: {model}")
    speculative_info = speculative_model_dir if speculative_model_dir else "None (baseline mode)"
    print(f"Speculative Model: {speculative_info}")
    print("Eagle3 One Model: False")
    print("=" * 80 + "\n")

    # Configure Eagle3 speculative decoding
    eagle3_config = None
    if enable_eagle and speculative_model_dir:
        eagle3_config = EagleDecodingConfig(
            max_draft_len=3, speculative_model_dir=speculative_model_dir, eagle3_one_model=False
        )
        print(f"[TRACE] Created EagleDecodingConfig: {eagle3_config}")
        print(f"[TRACE] Speculative model dir: {speculative_model_dir}")
    else:
        print("[TRACE] Running without speculative decoding (baseline mode)")

    # Configure KV cache
    # Allocate 80% of free GPU memory for KV cache
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.8,
    )
    print("[TRACE] Created KvCacheConfig with free_gpu_memory_fraction=0.8")

    # Test prompts
    prompts = [
        "What is the capital of France?",
        "Explain what machine learning is in one sentence.",
    ]

    # Create TRT-LLM instance
    spec_mode = "Eagle3" if (enable_eagle and speculative_model_dir) else "baseline"
    print(f"\n[TRACE] Creating LLM instance ({spec_mode} mode)...")
    print(f"[TRACE] Model: {model}")

    try:
        llm = LLM(model=model, speculative_config=eagle3_config, kv_cache_config=kv_cache_config)
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
        print("4. Try running with --no-eagle to test baseline first")

        raise

    # Configure sampling parameters
    sampling_params = SamplingParams(max_tokens=100)
    print("[TRACE] Sampling parameters: max_tokens=100")

    # Run generation
    print(f"\n[TRACE] Starting generation with {len(prompts)} prompts...")

    try:
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n[TRACE] Processing prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

            # Generate response
            response = llm.generate(prompt, sampling_params)

            # Extract output text
            output_text = response.outputs[0].text
            results.append((prompt, output_text))

            print(f"[TRACE] Generated {len(output_text)} characters")

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


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test models with Eagle3 speculative decoding using TRT-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Llama-3.1-8B with downloaded Eagle3 adapter (recommended if already downloaded)
  %(prog)s --model meta-llama/Llama-3.1-8B-Instruct \\
      --speculative-model-dir /lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/\\
autodeploy_data/hf_home/hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots/ada412b672e293d682423de84a095447bf38a637

  # Auto-download Eagle3 adapter for Llama-3.1-8B (simplest option for Llama-3.1-8B)
  %(prog)s --model meta-llama/Llama-3.1-8B-Instruct --auto-download-eagle

  # Run in baseline mode without speculative decoding
  %(prog)s --model meta-llama/Llama-3.1-8B-Instruct --no-eagle

Notes:
  - Use exactly ONE of: --auto-download-eagle, --speculative-model-dir, or --no-eagle
  - --auto-download-eagle: Downloads Eagle3 adapter for Llama-3.1-8B-Instruct
  - --speculative-model-dir: Uses an already-downloaded local directory path
  - --no-eagle: Runs in baseline mode without speculative decoding
  - If none specified, defaults to baseline mode
        """,
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
        help="Path to local Eagle3 speculative model directory. "
        "Cannot be used with --auto-download-eagle",
    )

    parser.add_argument(
        "--auto-download-eagle",
        action="store_true",
        help="Auto-download Eagle3 adapter for Llama-3.1-8B-Instruct (yuhuili/EAGLE3-LLaMA3.1-Instruct-8B). "
        "Cannot be used with --speculative-model-dir",
    )

    parser.add_argument(
        "--no-eagle",
        action="store_true",
        help="Disable Eagle3 speculative decoding (baseline mode)",
    )

    args = parser.parse_args()

    # Ensure speculative model is available
    speculative_model_path = None
    if not args.no_eagle:
        speculative_model_path = ensure_speculative_model(
            speculative_model_dir=args.speculative_model_dir,
            auto_download_eagle=args.auto_download_eagle,
            model_name=args.model,
        )

    # Run the test
    result = test_llama_eagle_with_trtllm(
        model=args.model,
        speculative_model_dir=speculative_model_path,
        enable_eagle=not args.no_eagle,
    )

    return result


if __name__ == "__main__":
    main()
