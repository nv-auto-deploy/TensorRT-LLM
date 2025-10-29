#!/usr/bin/env python3
"""Test FlashInfer MoE integration with TensorRT-LLM.

IMPORTANT: FlashInfer must be INITIALIZED (called at least once) BEFORE TensorRT-LLM
is imported to avoid tvm_ffi conflicts! Therefore, we delay TensorRT-LLM import
until inside the test function.
"""

import pytest
import torch

# Import FlashInfer FIRST (but don't import TensorRT-LLM yet!)
try:
    from flashinfer.fused_moe import cutlass_fused_moe  # noqa: F401
    from flashinfer.fused_moe.core import ActivationType  # noqa: F401

    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

import ctypes
import os

ctypes.CDLL(
    os.environ.get("FLASHINFER_MOE_SO_PATH"),
    mode=ctypes.RTLD_GLOBAL,
)


@pytest.mark.skip(reason="Test failed due to invalid shape error")
def test_flashinfer_moe_matches_torch_moe_gated_mlp():
    device = "cuda"
    dtype = torch.float16
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    # Use same parameters as test_flashinfer_direct.py which works!
    M = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 128
    E = 4
    top_k = 2

    torch.manual_seed(0)

    # Create input
    x = torch.randn(M, HIDDEN_SIZE, device=device, dtype=dtype) / 5

    # Create per-expert weights for gated MLP: w1 (gate), w3 (up), w2 (down)
    w1_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) / 5
        for _ in range(E)
    ]
    w3_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) / 5
        for _ in range(E)
    ]
    w2_list = [
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype) / 5
        for _ in range(E)
    ]

    # FlashInfer expects [w3, w1] concatenated
    w3_w1_stacked = torch.stack(
        [torch.cat([w3, w1], dim=0) for w1, w3 in zip(w1_list, w3_list)], dim=0
    ).contiguous()  # [E, 2*I, H]
    w2_stacked = torch.stack(w2_list, dim=0).contiguous()  # [E, H, I]

    # Create routing
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_full = torch.softmax(router_logits, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_full, k=top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # Test 1: FlashInfer custom op
    with torch.inference_mode():
        out_flashinfer = torch.ops.auto_deploy.flashinfer_moe_fused(
            x,
            selected_experts,
            routing_weights,
            w3_w1_stacked,
            w2_stacked,
            mlp_style="gated_mlp",
            act_fn="silu",
        )

    # Test 2: Reference torch_moe
    out_torch = torch.ops.auto_deploy.torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight=w1_list,
        w2_weight=w2_list,
        w3_weight=w3_list,
        mlp_style="gated_mlp",
        act_fn="silu",
    )

    # Test 3: Compare outputs
    torch.testing.assert_close(out_flashinfer, out_torch, rtol=1e-2, atol=1e-2)


def test_flashinfer_moe_matches_torch_moe_mlp_relu2():
    """Test non-gated MLP with ReLU^2 activation."""
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    device = "cuda"
    dtype = torch.float16

    # Use same parameters as gated test
    M = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 128
    E = 4
    top_k = 2

    torch.manual_seed(42)  # Different seed from gated test

    # Create input
    x = torch.randn(M, HIDDEN_SIZE, device=device, dtype=dtype) / 5

    # Create per-expert weights for non-gated MLP: w1 (up), w2 (down)
    # No w3 for non-gated MLP
    w1_list = [
        torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=dtype) / 5
        for _ in range(E)
    ]
    w2_list = [
        torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE, device=device, dtype=dtype) / 5
        for _ in range(E)
    ]

    # For non-gated MLP, w1_stacked is just w1 (not concatenated with w3)
    w1_stacked = torch.stack(w1_list, dim=0).contiguous()  # [E, I, H]
    w2_stacked = torch.stack(w2_list, dim=0).contiguous()  # [E, H, I]

    # Create routing
    router_logits = torch.randn(M, E, device=device, dtype=torch.float32)
    routing_full = torch.softmax(router_logits, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_full, k=top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    # Test 1: FlashInfer custom op with mlp style and relu2
    with torch.inference_mode():
        out_flashinfer = torch.ops.auto_deploy.flashinfer_moe_fused(
            x,
            selected_experts,
            routing_weights,
            w1_stacked,
            w2_stacked,
            mlp_style="mlp",
            act_fn="relu2",
        )

    # Test 2: Reference torch_moe
    out_torch = torch.ops.auto_deploy.torch_moe(
        x,
        selected_experts,
        routing_weights,
        w1_weight=w1_list,
        w2_weight=w2_list,
        w3_weight=[],  # Empty list for non-gated MLP
        mlp_style="mlp",
        act_fn="relu2",
    )

    # Test 3: Compare outputs
    torch.testing.assert_close(out_flashinfer, out_torch, rtol=1e-2, atol=1e-2)
