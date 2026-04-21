"""Testing module patches that enable export of MiniMax-M2 model.

This test verifies that the patched MiniMaxM2SparseMoeBlock forward function
produces identical outputs to the original HuggingFace implementation.
"""

import types

import torch
from transformers import MiniMaxM2Config
from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2SparseMoeBlock

# Import custom_ops to register torch.ops.auto_deploy.torch_moe
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.models.patches.minimax_m2 import minimax_m2_moe


def _build_minimax_m2_moe_layer():
    """Construct a small MiniMaxM2SparseMoeBlock directly.

    Building the standalone block (rather than loading a full model from a
    local HF cache) keeps the test hermetic and avoids pitfalls with
    cached remote-code configs on CI nodes.
    """
    config = MiniMaxM2Config()
    config.num_hidden_layers = 1
    config.use_cache = False
    config.hidden_size = 16  # Small hidden size
    config.intermediate_size = 8  # For MLP within experts
    config.num_local_experts = 4  # Small number of experts
    config.num_experts_per_tok = 2  # Top-k experts
    config.router_jitter_noise = 0.0  # Disable jitter for deterministic tests

    module = MiniMaxM2SparseMoeBlock(config)
    module.eval()
    # Initialize weights to something non-default so routing exercises the experts.
    for p in module.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)
    # `e_score_correction_bias` is a buffer that is zero-initialized; that is fine.
    return module


def test_minimax_m2_moe_patch():
    """Test that the patched MiniMaxM2SparseMoeBlock forward matches HF implementation.

    The patch rewrites the forward to use torch.ops.auto_deploy.torch_moe
    for torch.export compatibility while maintaining numerical equivalence.
    """
    # Set seed for reproducibility
    torch.manual_seed(42)

    module = _build_minimax_m2_moe_layer()

    # Convert module to bfloat16 to match input dtype
    module = module.to(torch.bfloat16)

    # hidden_size=16 matches the config in _build_minimax_m2_moe_layer
    hidden_size = 16
    inputs = torch.randn(2, 6, hidden_size, dtype=torch.bfloat16)

    # Reference: original (unpatched) class method. In transformers 5.x this
    # returns just `hidden_states` (no router_logits).
    original_class_forward = type(module).forward
    with torch.no_grad():
        ref_output = original_class_forward(module, inputs)

    # Apply the export-friendly patch and run the same input through it.
    module.forward = types.MethodType(minimax_m2_moe, module)
    with torch.no_grad():
        test_output = module(inputs)

    # Final hidden states should be very close
    # (small tolerance for different computation order in torch_moe)
    torch.testing.assert_close(
        ref_output,
        test_output,
        atol=1e-3,
        rtol=1e-3,
        msg="Output mismatch between original and patched MoE",
    )
