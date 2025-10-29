#!/usr/bin/env python3
"""Test FlashInfer MoE integration with TensorRT-LLM.

IMPORTANT: FlashInfer must be INITIALIZED (called at least once) BEFORE TensorRT-LLM
is imported to avoid tvm_ffi conflicts! Therefore, we delay TensorRT-LLM import
until inside the test function.
"""

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


def test_flashinfer_moe_matches_torch_moe_gated_mlp():
    device = "cuda"
    dtype = torch.float16

    # Use same parameters as test_flashinfer_direct.py which works!
    M = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 128
    E = 4
    top_k = 2

    torch.manual_seed(0)

    print("\nTest Configuration:")
    print(f"  Tokens (M): {M}")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Intermediate Size: {INTERMEDIATE_SIZE}")
    print(f"  Experts (E): {E}")
    print(f"  Top-K: {top_k}")
    print(f"  Dtype: {dtype}")

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

    print("\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  w3_w1_stacked: {w3_w1_stacked.shape}")
    print(f"  w2_stacked: {w2_stacked.shape}")
    print(f"  selected_experts: {selected_experts.shape}")
    print(f"  routing_weights: {routing_weights.shape}")

    # NOW safe to import TensorRT-LLM (AFTER FlashInfer is initialized)
    print("\nImporting TensorRT-LLM custom ops...")
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    # Test 1: FlashInfer custom op
    print("\n[1/3] Testing FlashInfer custom op...")
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
    print(f"      ✓ FlashInfer output shape: {out_flashinfer.shape}")

    # Test 2: Reference torch_moe
    print("\n[2/3] Testing torch_moe reference...")
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
    print(f"      ✓ Torch reference output shape: {out_torch.shape}")

    # Test 3: Compare outputs
    print("\n[3/3] Comparing FlashInfer vs Torch Reference...")
    print("-" * 70)
    diff = (out_flashinfer - out_torch).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  Max absolute difference:  {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    # Assert close with relaxed tolerance (FlashInfer uses different precision)
    try:
        torch.testing.assert_close(out_flashinfer, out_torch, rtol=1e-2, atol=1e-2)
        print("\n" + "=" * 70)
        print("✅ TEST PASSED! FlashInfer matches torch_moe reference.")
        print("=" * 70 + "\n")
    except AssertionError as err:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {err}\n")
        raise


if __name__ == "__main__":
    # Run directly without pytest
    print("\nRunning test directly (not through pytest)...")
    try:
        test_flashinfer_moe_matches_torch_moe_gated_mlp()
    except Exception:
        print("\n❌ Test failed with error:")
        import traceback

        traceback.print_exc()
        exit(1)
