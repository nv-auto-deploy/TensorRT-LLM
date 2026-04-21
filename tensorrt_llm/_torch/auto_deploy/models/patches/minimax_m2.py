"""A patch for MiniMax-M2 MoE to make it compatible with torch.export.

MiniMax-M2 is loaded from HuggingFace hub (trust_remote_code), so we cannot
import MiniMaxM2SparseMoeBlock directly. Instead, we use the same pattern as
DeepSeek: patching AutoModelForCausalLM.from_config to iterate over modules
and patch by class name.
"""

import types
from typing import Dict

import torch
from transformers import AutoModelForCausalLM


def minimax_m2_moe(self, hidden_states: torch.Tensor):
    """MiniMaxM2SparseMoeBlock forward function rewritten to enable torch.export.

    Targets the transformers 5.x layout:
      - self.gate: MiniMaxM2TopKRouter with `weight` of shape (E, H)
      - self.experts: MiniMaxM2Experts holding packed 3D weights
          * gate_up_proj: (E, 2*I, H) -- gate (w1) and up (w3) stacked along dim 1
          * down_proj:    (E, H, I)   -- down (w2)
      - self.e_score_correction_bias: buffer of shape (E,)
      - forward returns only `hidden_states` (no router_logits), matching the
        native signature consumed by MiniMaxM2DecoderLayer.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        )
    hidden_states = hidden_states.view(-1, hidden_dim)

    # Router logits from the gate's raw weight (matches MiniMaxM2TopKRouter.forward).
    router_logits = torch.nn.functional.linear(
        hidden_states.to(self.gate.weight.dtype), self.gate.weight
    )

    # MiniMax-M2 routing: sigmoid + bias for selection, original sigmoid for combine.
    routing_weights = torch.sigmoid(router_logits.float())
    scores_for_choice = routing_weights + self.e_score_correction_bias
    _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
    top_k_weights = routing_weights.gather(1, selected_experts)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
    top_k_weights = top_k_weights.to(hidden_states.dtype)

    # Split packed gate_up_proj into per-expert gate (w1) and up (w3) weights.
    # F.linear(x, W) does x @ W.T; the new experts' forward chunks the output
    # along the last dim, so rows [:I] of gate_up_proj are w1, rows [I:] are w3.
    gate_up = self.experts.gate_up_proj  # (E, 2*I, H)
    down = self.experts.down_proj  # (E, H, I)
    num_experts = gate_up.shape[0]
    intermediate_dim = gate_up.shape[1] // 2
    w1_weight = [gate_up[e, :intermediate_dim, :] for e in range(num_experts)]
    w3_weight = [gate_up[e, intermediate_dim:, :] for e in range(num_experts)]
    w2_weight = [down[e] for e in range(num_experts)]

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        top_k_weights,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states


_from_config_previous = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, callable] = {"MiniMaxM2SparseMoeBlock": minimax_m2_moe}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_previous(config, **kwargs)
    # Patch modules by class name
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES:
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# Patch AutoModelForCausalLM.from_config
AutoModelForCausalLM.from_config = get_model_from_config_patched
