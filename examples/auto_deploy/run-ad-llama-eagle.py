"""Test GPT-OSS-120B with Eagle3 speculative decoding using AutoDeploy flow.

This script tests the same model as run-gptoss-no-ad.py but uses the AutoDeploy backend
with the ExperimentConfig pattern.
"""


# Add parent directory to path to import build_and_run_ad

from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm.llmapi import EagleDecodingConfig

# Model configuration
# MODEL_PATH = "openai/gpt-oss-120b"
MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home\
    /hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Eagle3 draft model path
# EAGLE3_MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home\
# /hub/models--nvidia--gpt-oss-120b-Eagle3/snapshots/6511991ada827d7cdda41fc2ff1e211e7e833605"
# EAGLE3_MODEL_NAME = "nvidia/gpt-oss-120b-Eagle3"

EAGLE3_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
EAGLE3_MODEL_PATH = "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data\
    /hf_home/hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots/ada412b672e293d682423de84a095447bf38a637"

# Enable/disable speculative decoding for testing

# TODO: When speculative decoding is enabled, we should test that it is at least doing something
# by checking that the outputs differ (and still make sense).
ENABLE_EAGLE3 = True


def test_ad_llama_eagle():
    """Test Llama-3.1-8B-Instruct with AutoDeploy backend."""
    print(
        "[TRACE] Starting AutoDeploy test for Llama-3.1-8B-Instruct with Eagle3 speculative decoding..."
    )

    # ⚠️ POTENTIAL ERROR #1: AutoDeploy LlmArgs might not support speculative_config
    # The AutoDeploy LlmArgs inherits from BaseLlmArgs which should have speculative_config,
    # but it's unclear if AutoDeploy backend actually supports it.
    eagle3_config = None
    if ENABLE_EAGLE3:
        eagle3_config = EagleDecodingConfig(
            max_draft_len=3,
            speculative_model_dir=EAGLE3_MODEL_PATH,
            eagle3_one_model=False,  # Testing two-model implementation
        )
        print(f"[TRACE] Created EagleDecodingConfig: {eagle3_config}")
        print(f"[TRACE] Eagle3 draft model: {EAGLE3_MODEL_PATH}")
    else:
        print("[TRACE] Running WITHOUT speculative decoding (baseline mode)")

    # Configure AutoDeploy LLM arguments
    llm_args = {
        "model": MODEL_PATH,
        "skip_loading_weights": False,  # We want to load weights
        "speculative_config": eagle3_config,
    }

    # Configure experiment with prompts
    experiment_config = {
        "args": llm_args,
        "benchmark": {
            "enabled": False  # Disable benchmarking for now
        },
        "prompt": {
            "batch_size": 1,
            "queries": [
                "What is the capital of France?",
            ],
        },
    }

    # DemoLLM runtime does not support guided decoding. Need to set runtime to trtllm.
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1

    # Create ExperimentConfig
    # ⚠️ POTENTIAL ERROR #10: ExperimentConfig validation might fail if any of the
    # llm_args fields are invalid or not recognized by AutoDeploy's LlmArgs
    try:
        cfg = ExperimentConfig(**experiment_config)
        print("[TRACE] ExperimentConfig created successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to create ExperimentConfig: {e}")
        import traceback

        traceback.print_exc()
        print("\n[HINT] Check if all llm_args fields are valid for AutoDeploy's LlmArgs")
        raise

    # Add sampling parameters
    cfg.prompt.sp_kwargs = {
        "max_tokens": 100,
        "top_k": None,
        "temperature": 1.0,
    }

    # Run the experiment
    spec_mode = "Eagle3" if ENABLE_EAGLE3 else "baseline"
    print(f"\n[TRACE] Starting AutoDeploy test ({spec_mode} mode)...")
    print(f"[TRACE] Model: {MODEL_PATH}")
    print(f"[TRACE] World size: {llm_args['world_size']}")

    try:
        # ⚠️ POTENTIAL ERROR #11: The main() function might fail during LLM creation
        # if speculative_config is not supported by AutoDeploy backend
        result = main(cfg)

        print("\n[TRACE] ===== SUCCESS! =====")
        print(f"[TRACE] Generation completed with {MODEL_PATH} using AutoDeploy!")
        print(f"[TRACE] Results: {result}")

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
        print(f"[ERROR] Test failed: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()

        print("\n[DEBUG HINTS]:")
        print("1. Check if 'speculative_config' is supported in AutoDeploy LlmArgs")
        print("2. Verify 'max_num_tokens' vs 'max_seq_len' field name")
        print("3. Check if AutoDeploy backend supports Eagle3 at all")
        print("4. Look at AutoDeploy's LlmArgs class for valid field names")
        print("5. Try running with ENABLE_EAGLE3 = False to test baseline first")

        raise


if __name__ == "__main__":
    test_ad_llama_eagle()
