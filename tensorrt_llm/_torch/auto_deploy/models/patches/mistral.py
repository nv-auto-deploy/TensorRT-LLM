"""A patch for the Mistral3Model to make it compatible with torch.export."""

from typing import Optional, Union

import torch
from transformers.models.mistral3.modeling_mistral3 import (
    Mistral3Model,
    Mistral3ModelOutputWithPast,
)

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _mistral_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, list[int]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    image_sizes: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[tuple, Mistral3ModelOutputWithPast]:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer
        if vision_feature_layer is not None
        else self.config.vision_feature_layer
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if pixel_values is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
        )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    def _no_vision_branch(
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor],
    ):
        return inputs_embeds

    def _vision_branch(
        # ! The type annotations in the original transformers code are all wrong.
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        pixel_values: torch.Tensor,
        image_sizes: Optional[torch.Tensor],
    ):
        pixel_values = pixel_values.to(torch.bfloat16)
        image_features = self.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            image_sizes=image_sizes,
        )
        image_features = torch.cat(image_features, dim=0)

        special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        # if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
        #     n_image_tokens = (input_ids == self.config.image_token_id).sum()
        #     n_image_features = image_features.shape[0] * image_features.shape[1]
        #     raise ValueError(
        #         f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
        #     )
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        return inputs_embeds

    # Decide by whether there is any non-zero pixel_values.
    has_image: torch.Tensor = torch.any(pixel_values != 0)

    inputs_embeds = torch.cond(
        has_image,
        _vision_branch,
        _no_vision_branch,
        (input_ids, inputs_embeds, pixel_values, image_sizes),  # , vision_feature_layer),
    )

    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return Mistral3ModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        # image_hidden_states=image_features if pixel_values is not None else None,
        image_hidden_states=None,
    )


@ExportPatchRegistry.register("hf_mistral3")
class Mistral3ModelPatch(BaseExportPatch):
    """Patch for `Mistral3Model`."""

    def _apply_patch(self):
        """Apply the Mistral3Model patch."""
        # Store original forward method
        self.original_values["Mistral3Model.forward"] = Mistral3Model.forward

        # Apply patch by replacing the forward method
        Mistral3Model._original_forward = Mistral3Model.forward  # type: ignore
        Mistral3Model.forward = _mistral_forward  # type: ignore

    def _revert_patch(self):
        """Revert the Mistral3Model patch."""
        # Restore original forward method.
        Mistral3Model.forward = self.original_values["Mistral3Model.forward"]  # type: ignore

        # Clean up the temporary attribute.
        if hasattr(Mistral3Model, "_original_forward"):
            delattr(Mistral3Model, "_original_forward")
