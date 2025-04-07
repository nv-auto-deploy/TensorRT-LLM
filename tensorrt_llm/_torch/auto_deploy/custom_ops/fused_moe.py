from typing import List

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.fused_moe import FusedMoE  # noqa: F401


@torch.library.custom_op("moe::torch_moe", mutates_args=())
def torch_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
) -> torch.Tensor:
    """
    A reference implementation of a Mixture-of-Experts (MoE) layer computation in the Mixtral style,
    compatible with DeepSeek by using index_add_ for in-place updates.
    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the indices
            of the selected experts for each token. Only experts within range [0,num_experts) is processed
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w1_weight (List[torch.Tensor]): A list of expert weight tensors for w1, each of shape
            (I, H), where I is the intermediate size.
        w2_weight (List[torch.Tensor]): A list of expert weight tensors for w2, each of shape
            (H, I).
        w3_weight (List[torch.Tensor]): A list of expert weight tensors for w3, each of shape
            (I, H).
    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.
    """

    hidden_dim = x.shape[-1]
    num_experts = len(w1_weight)

    final_hidden_states = torch.zeros_like(x)
    valid_mask = (selected_experts >= 0) & (selected_experts < num_experts)
    # For out-of-range indices, set them to num_experts
    selected_experts_fixed = torch.where(
        valid_mask, selected_experts, torch.full_like(selected_experts, num_experts)
    )
    # Create one-hot encoding with an extra class.
    one_hot = F.one_hot(selected_experts_fixed, num_classes=num_experts + 1)
    expert_mask = one_hot[..., :num_experts].permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        tokens_for_this_expert = x[None, top_x].reshape(-1, hidden_dim)

        gate_out = F.linear(tokens_for_this_expert, w1_weight[expert_idx])
        up_out = F.linear(tokens_for_this_expert, w3_weight[expert_idx])
        activated = F.silu(gate_out)
        prod = activated * up_out
        expert_out = F.linear(prod, w2_weight[expert_idx])

        current_hidden_states = expert_out * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(
            0, top_x, current_hidden_states.to(final_hidden_states.dtype)
        )

    return final_hidden_states.view_as(x)


@torch_moe.register_fake
def torch_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("moe::torch_fused_moe", mutates_args=())
def torch_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    """
    A reference implementation of a fused MoE layer computation.

    Parameters:
        x (torch.Tensor): Input tensor of shape (B, H) or (B, S, H), where B is the batch size,
            S is the sequence length, and H is the hidden size.
        selected_experts (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the
            indices of the selected experts for each token.
        routing_weights (torch.Tensor): A tensor of shape (B, TOP_K) or (B*S, TOP_K) containing the normalized
            routing weights for the selected experts.
        w3_w1_stacked_weight (torch.Tensor): A tensor of shape (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            containing the fused weights for w3 and w1 for each expert.
        w2_stacked_weight (torch.Tensor): A tensor of shape (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            containing the weights for w2 for each expert.

    Returns:
        torch.Tensor: Output tensor with the same shape as the input x.
    """
    num_experts = w2_stacked_weight.shape[0]
    intermediate_size = w3_w1_stacked_weight.shape[1] // 2
    results = torch.zeros_like(x)

    for expert_id in range(num_experts):
        batch_idx, nth_expert = torch.where(selected_experts == expert_id)
        if batch_idx.numel() == 0:
            continue

        expert_inputs = x[batch_idx]

        stacked = w3_w1_stacked_weight[expert_id]
        w3 = stacked[:intermediate_size, :]
        w1 = stacked[intermediate_size:, :]
        w2 = w2_stacked_weight[expert_id]

        # Compute expert output:
        #   expert_out = (F.silu(x @ w1.t()) * (x @ w3.t())) @ w2.t()
        out_w1 = expert_inputs @ w1.t()
        out_w3 = expert_inputs @ w3.t()
        expert_out = (F.silu(out_w1) * out_w3) @ w2.t()

        scaling = routing_weights[batch_idx, nth_expert].unsqueeze(-1)
        results[batch_idx] += scaling * expert_out

    return results.view_as(x)


@torch_fused_moe.register_fake
def torch_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("moe::trtllm_fused_moe", mutates_args=())
def trtllm_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    routing_weights = routing_weights.to(torch.float32)
    selected_experts = selected_experts.to(torch.int32)
    quant_scales = []

    return torch.ops.trtllm.fused_moe(
        x,
        selected_experts,
        routing_weights,
        w3_w1_stacked_weight,
        w2_stacked_weight,
        x.dtype,
        quant_scales,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
    )[0]


@trtllm_fused_moe.register_fake
def trtllm_fused_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w3_w1_stacked_weight: torch.Tensor,
    w2_stacked_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("moe::torch_fp8_moe", mutates_args=())
def torch_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
) -> torch.Tensor:
    """
    FP8 MoE op using quantized linear operations.

    Computes a Mixture-of-Experts layer similar to the reference moe::torch_moe op, but uses the
    quantized FP8 linear op for expert computations.

    Args:
        x: Input tensor of shape (B, H) or (B, S, H).
        selected_experts: Tensor (B, TOP_K) or (B*S, TOP_K) containing expert indices.
        routing_weights: Tensor of normalized routing weights.
        w1_weight, w2_weight, w3_weight: Lists of pre-quantized weight tensors for the three linear ops.
        w1_input_scale, w2_input_scale, w3_input_scale: Lists of input scale tensors for the corresponding ops.
        w1_weight_scale, w2_weight_scale, w3_weight_scale: Lists of weight scale tensors for the corresponding ops.

    """
    hidden_dim = x.shape[-1]
    num_experts = len(w1_weight)

    final_hidden_states = torch.zeros_like(x)
    valid_mask = (selected_experts >= 0) & (selected_experts < num_experts)
    selected_experts_fixed = torch.where(valid_mask, selected_experts, torch.full_like(selected_experts, num_experts))
    one_hot = F.one_hot(selected_experts_fixed, num_classes=num_experts + 1)
    expert_mask = one_hot[..., :num_experts].permute(2, 1, 0)

    for expert_idx in range(num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        tokens_for_expert = x[None, top_x].reshape(-1, hidden_dim)

        gate_out = torch.ops.quant.fp8_linear(
            tokens_for_expert,
            w1_weight[expert_idx],
            bias=None,
            input_scale=w1_input_scale[expert_idx],
            weight_scale=w1_weight_scale[expert_idx],
        )
        up_out = torch.ops.quant.fp8_linear(
            tokens_for_expert,
            w3_weight[expert_idx],
            bias=None,
            input_scale=w3_input_scale[expert_idx],
            weight_scale=w3_weight_scale[expert_idx],
        )
        activated = F.silu(gate_out)
        prod = activated * up_out
        expert_out = torch.ops.quant.fp8_linear(
            prod,
            w2_weight[expert_idx],
            bias=None,
            input_scale=w2_input_scale[expert_idx],
            weight_scale=w2_weight_scale[expert_idx],
        )

        current_hidden_states = expert_out * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states)

    return final_hidden_states.view_as(x)


@torch_fp8_moe.register_fake
def torch_fp8_moe(
    x: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    w1_weight: List[torch.Tensor],
    w2_weight: List[torch.Tensor],
    w3_weight: List[torch.Tensor],
    w1_input_scale: List[torch.Tensor],
    w2_input_scale: List[torch.Tensor],
    w3_input_scale: List[torch.Tensor],
    w1_weight_scale: List[torch.Tensor],
    w2_weight_scale: List[torch.Tensor],
    w3_weight_scale: List[torch.Tensor],
) -> torch.Tensor:
    return torch.empty_like(x)
