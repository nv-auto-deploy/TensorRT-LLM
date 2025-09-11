import types
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple


@can_return_tuple
def forward_no_mask(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = False
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # if self.gradient_checkpointing and self.training and use_cache:
    #     logger.warning_once(
    #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
    #     )
    #     use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    hidden_states = inputs_embeds

    # It may already have been prepared by e.g. `generate`
    # if not isinstance(causal_mask_mapping := attention_mask, dict):
    #     # Prepare mask arguments
    #     mask_kwargs = {
    #         "config": self.config,
    #         "input_embeds": inputs_embeds,
    #         "attention_mask": attention_mask,
    #         "cache_position": cache_position,
    #         "past_key_values": past_key_values,
    #         "position_ids": position_ids,
    #     }
    #     # Create the masks
    #     causal_mask_mapping = {
    #         "full_attention": create_causal_mask(**mask_kwargs),
    #     }

    # create position embeddings to be shared across the decoder layers
    freq_cis = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    # i = 0
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=None,  # causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=freq_cis,
            **kwargs,
        )

        hidden_states = layer_outputs

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def _forward_moe(self, hidden_states: torch.Tensor):
    # # check if we can apply the patch
    # use_original_forward = False
    # if not all(isinstance(expert.act_fn, nn.SiLU) for expert in self.experts):
    #     use_original_forward = True

    # if any(getattr(mod, "bias", None) is not None for mod in self.experts.modules()):
    #     use_original_forward = True

    # # rely on original forward instead
    # if use_original_forward:
    #     return self._original_forward(hidden_states)

    batch_size, sequence_length, hidden_dim = hidden_states.shape
    if self.training and self.jitter_noise > 0:
        hidden_states *= torch.empty_like(hidden_states).uniform_(
            1.0 - self.jitter_noise, 1.0 + self.jitter_noise
        )
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    # NOTE: no routing weight normalization unlike Mixtral!
    # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.ops.auto_deploy.torch_moe(
        hidden_states,
        selected_experts,
        routing_weights,
        w1_weight=[self.gate_proj.weight[i] for i in range(self.num_experts)],  # gate projection
        w2_weight=[self.down_proj.weight[i] for i in range(self.num_experts)],  # down projection
        w3_weight=[self.up_proj.weight[i] for i in range(self.num_experts)],  # up projection
    )
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states  # NOTE: only return hidden states for step 3


CUSTOM_MODULE_PATCHES: Dict[str, callable] = {
    "Step3Model": forward_no_mask,
    "Step3vMoEMLP": _forward_moe,
}
_from_config_original = AutoModelForCausalLM.from_config


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if type(module).__name__ in CUSTOM_MODULE_PATCHES.keys():
            print(f"Patching {type(module).__name__}")
            module.forward = types.MethodType(CUSTOM_MODULE_PATCHES[type(module).__name__], module)

    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
