"""
FlashInfer CUTLASS implementation of Fused MoE operations.

This module provides a wrapper around FlashInfer's cutlass_fused_moe kernel
to integrate with TensorRT-LLM's auto_deploy system.
"""

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType


@torch.library.custom_op("auto_deploy::flashinfer_moe_fused", mutates_args=())
def flashinfer_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    """
    FlashInfer CUTLASS-based fused MoE for BF16/FP16 inputs.

    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the indices
            of the selected experts for each token.
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w1_stacked_weight (torch.Tensor):
            - For mlp_style=="gated_mlp": Shape (NUM_EXPERTS, 2*INTERMEDIATE_SIZE, HIDDEN_SIZE)
              containing stacked [w3, w1] weights (up and gate projections).
            - For mlp_style=="mlp": Shape (NUM_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE)
              containing w_up weights.
        w2_stacked_weight (torch.Tensor): A tensor of shape (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            containing the w2 (down projection) weights for each expert.
        mlp_style (str): Type of MLP - "gated_mlp" (default) or "mlp".
        act_fn (str): Activation function - "silu" (default) or "relu" for relu^2.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.

    Note:
        This op requires FlashInfer to be installed. For gated_mlp, w1_stacked_weight should
        have w3 and w1 concatenated along the intermediate dimension as [w3, w1].
    """

    # Store original shape and flatten to 2D
    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])

    # Validate dtypes
    if x2d.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"Unsupported input dtype: {x2d.dtype}. Expected float16, bfloat16, or float32."
        )

    # Determine activation type
    mlp_style = mlp_style.lower()
    act_fn = act_fn.lower()

    if mlp_style == "gated_mlp":
        # Gated MLP uses Silu: silu(x @ w1.T) * (x @ w3.T)
        if act_fn == "silu":
            activation_type = ActivationType.Silu
        else:
            raise ValueError(f"Unsupported activation '{act_fn}' for gated_mlp. Use 'silu'.")
    elif mlp_style == "mlp":
        # For non-gated MLP with ReLU^2
        if act_fn == "relu2":
            activation_type = ActivationType.Relu2
        else:
            raise ValueError(f"Unsupported activation '{act_fn}' for mlp. Use 'relu2'.")
    else:
        raise ValueError(f"Unknown mlp_style '{mlp_style}'. Use 'gated_mlp' or 'mlp'.")

    # Ensure contiguous tensors and correct dtypes
    x2d = x2d.contiguous()
    # FlashInfer requires int32 for selected_experts
    selected_experts = selected_experts.to(torch.int32).contiguous()
    routing_weights = routing_weights.contiguous()
    w1_stacked = w1_stacked_weight.contiguous()
    w2_stacked = w2_stacked_weight.contiguous()

    # Call FlashInfer's cutlass_fused_moe (BF16/FP16 path)
    # Following FlashInfer's test pattern (lines 263-275)
    output = torch.empty_like(x2d)
    result = cutlass_fused_moe(
        input=x2d,
        token_selected_experts=selected_experts,
        token_final_scales=routing_weights,
        fc1_expert_weights=w1_stacked,
        fc2_expert_weights=w2_stacked,
        output_dtype=x2d.dtype,
        quant_scales=None,  # None for BF16/FP16
        output=output,  # Pre-allocate output buffer
        activation_type=activation_type,
    )

    # cutlass_fused_moe returns a list [output, ...], extract the output tensor
    if isinstance(result, (list, tuple)):
        output = result[0]
    else:
        output = result

    return output.view(x_shape)


@flashinfer_fused_moe.register_fake
def flashinfer_fused_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::flashinfer_quant_fp8_moe", mutates_args=())
def flashinfer_quant_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,  # [E, I, H] stacked FP8 weights
    w2_weight: torch.Tensor,  # [E, H, I] stacked FP8 weights
    w3_weight: torch.Tensor,  # [E, I, H] for gated_mlp, unused for mlp
    w1_input_scale: torch.Tensor,  # [E] stacked input scales
    w2_input_scale: torch.Tensor,  # [E] stacked input scales
    w3_input_scale: torch.Tensor,  # [E] or unused
    w1_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w2_weight_scale: torch.Tensor,  # [E] stacked weight scales
    w3_weight_scale: torch.Tensor,  # [E] or unused
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    """
    FlashInfer CUTLASS-based FP8 W8A8 MoE for gated MLP.

    Parameters:
        x: BF16/FP16 input tensor of shape (B, H) or (B, S, H)
        selected_experts: Expert indices (B*S, TOP_K)
        routing_weights: Routing weights (B*S, TOP_K)
        w1_weight: FP8 w1 weights [E, I, H]
        w2_weight: FP8 w2 weights [E, H, I]
        w3_weight: FP8 w3 weights [E, I, H] (for gated_mlp)
        w1_input_scale: Input scales for w1 [E]
        w2_input_scale: Input scales for w2 [E]
        w3_input_scale: Input scales for w3 [E]
        w1_weight_scale: Weight scales for w1 [E]
        w2_weight_scale: Weight scales for w2 [E]
        w3_weight_scale: Weight scales for w3 [E]
        mlp_style: "gated_mlp" or "mlp"
        act_fn: "silu" for gated_mlp
    """

    if mlp_style != "gated_mlp":
        raise NotImplementedError("FlashInfer FP8 MoE currently only supports gated_mlp")

    # Store original shape and flatten to 2D
    x_shape = x.shape
    x2d = x.view(-1, x_shape[-1])

    # For gated MLP, concatenate w1 and w3
    # FlashInfer expects [w1, w3] concatenated
    w1_w3_stacked = torch.cat([w1_weight, w3_weight], dim=1).contiguous()  # [E, 2*I, H]

    # Prepare quant_scales for FlashInfer FP8 format:
    # [gemm1_dequant_scale, gemm2_act_quant_scale, gemm2_dequant_scale, gemm1_input_dequant_scale]
    # For gated MLP:
    # - gemm1_dequant_scale: w1_weight_scale * w1_input_scale (combined for w1 and w3)
    # - gemm2_act_quant_scale: 1 / w2_input_scale
    # - gemm2_dequant_scale: w2_weight_scale * w2_input_scale
    # - gemm1_input_dequant_scale: w1_input_scale

    # Compute combined scales
    gemm1_dequant = (w1_weight_scale * w1_input_scale).contiguous()  # [E]
    gemm2_act_quant = (1.0 / w2_input_scale).contiguous()  # [E]
    gemm2_dequant = (w2_weight_scale * w2_input_scale).contiguous()  # [E]
    gemm1_input_dequant = w1_input_scale.contiguous()  # [E]

    quant_scales = [gemm1_dequant, gemm2_act_quant, gemm2_dequant, gemm1_input_dequant]

    # Ensure contiguous tensors
    selected_experts = selected_experts.contiguous()
    routing_weights = routing_weights.contiguous()
    w2_stacked = w2_weight.contiguous()

    # Determine activation type
    if act_fn.lower() == "silu":
        activation_type = ActivationType.Silu
    else:
        raise ValueError(f"Unsupported activation '{act_fn}' for FP8 gated_mlp. Use 'silu'.")

    # Call FlashInfer's cutlass_fused_moe with FP8 support
    output = cutlass_fused_moe(
        input=x2d,
        token_selected_experts=selected_experts,
        token_final_scales=routing_weights,
        fc1_expert_weights=w1_w3_stacked,
        fc2_expert_weights=w2_stacked,
        output_dtype=x2d.dtype,
        quant_scales=quant_scales,
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        input_sf=None,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        output=None,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        tune_max_num_tokens=8192,
        enable_pdl=None,
        activation_type=activation_type,
    )

    # Restore original shape
    return output.view(x_shape)


@flashinfer_quant_fp8_moe.register_fake
def flashinfer_quant_fp8_moe_fake(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    w1_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
    w3_input_scale: torch.Tensor,
    w1_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w3_weight_scale: torch.Tensor,
    mlp_style: str = "gated_mlp",
    act_fn: str = "silu",
) -> torch.Tensor:
    return torch.empty_like(x)
