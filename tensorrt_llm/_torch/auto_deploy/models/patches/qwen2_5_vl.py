"""Patches for Qwen2.5-VL model to make it compatible with torch.export."""

from typing import Optional

import torch

from ...export.interface import BaseExportPatch, ExportPatchRegistry


def _patched_vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs):
    """
    Patched forward method for Qwen2.5-VL vision transformer.

    This patch moves ALL data-dependent operations into custom ops to make
    the method fully compatible with torch.export's symbolic tracing.
    """
    # Original patch_embed processing (no data dependencies)
    hidden_states = self.patch_embed(hidden_states)

    # Use custom op to handle ALL data-dependent operations including advanced indexing
    hidden_states, pos_emb_cos, pos_emb_sin, cu_window_seqlens, cu_seqlens, reverse_indices = (
        torch.ops.auto_deploy.qwen_vision_data_dependent_ops(
            grid_thw,
            hidden_states,
            self.spatial_merge_size,
            self.window_size,
            self.patch_size,
            self.spatial_merge_unit,
        )
    )

    # Create position_embeddings tuple from separate tensors
    position_embeddings = (pos_emb_cos, pos_emb_sin)

    # Process through attention blocks (using custom op for attention mask)
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        # Use custom op for _prepare_attention_mask (handles data-dependent operations)
        attention_mask_tensor = torch.ops.auto_deploy.qwen_prepare_attention_mask(
            hidden_states, cu_seqlens_now, self.config._attn_implementation
        )

        # Convert empty tensor marker back to None
        attention_mask = None if attention_mask_tensor.numel() == 0 else attention_mask_tensor
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )
    # Final merger (no data dependencies)
    hidden_states = self.merger(hidden_states)

    # Use custom op for reverse indexing (handles data-dependent operations)
    hidden_states = torch.ops.auto_deploy.qwen_reverse_indexing(hidden_states, reverse_indices)

    return hidden_states


def _patched_rope_forward(self, x, position_ids):
    """
    Patched forward method for Qwen2_5_VLRotaryEmbedding to handle 'meta' device during torch.export.

    This patch fixes the device_type issue when using torch.export where device becomes 'meta'.
    """
    # In contrast to other models, Qwen2_5_VL has different position ids for the grids
    # So we expand the inv_freq to shape (3, ...)
    inv_freq_expanded = (
        self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    )
    position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

    # Fix device type handling for torch.export (where device can be 'meta')
    device_type = (
        x.device.type
        if isinstance(x.device.type, str) and x.device.type not in ["mps", "meta"]
        else "cpu"
    )

    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _patched_create_causal_mask(**kwargs):
    """
    Patched create_causal_mask function that returns None to avoid issues during model export/execution.

    This is a temporary workaround for compatibility issues with create_causal_mask during
    TensorRT-LLM model execution.
    """
    return None


def _patched_get_image_features_flat(
    self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None
):
    """
    WAR: Return flat image features directly; avoid Python list/split that specializes num_images.
    """
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
    return image_embeds


def _patched_model_forward_export_war(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[list[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs,
):
    """
    WAR forward: (a) do not call get_rope_index; synthesize tensor position_ids if None,
    (b) consume flat image features directly (no split/cat).
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModelOutputWithPast
    from transformers.utils import is_torchdynamo_compiling

    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum()
            n_image_features = image_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match tokens:{n_image_tokens}, features:{n_image_features}"
                )
            mask = input_ids == self.config.image_token_id
            image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            # Keep videos flat as well
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            if not is_torchdynamo_compiling() and n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match tokens:{n_video_tokens}, features:{n_video_features}"
                )
            mask = input_ids == self.config.video_token_id
            video_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    # Bypass get_rope_index: synthesize simple tensor position_ids if None
    if position_ids is None:
        bsz, seqlen = inputs_embeds.shape[0], inputs_embeds.shape[1]
        base = torch.arange(seqlen, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        position_ids = base.view(1, 1, -1).expand(3, bsz, -1)
        # keep rope_deltas as zeros per batch
        self.rope_deltas = torch.zeros(
            (bsz, 1), device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()


@ExportPatchRegistry.register("qwen2_5_vl_vision")
class Qwen2_5_VLVisionPatch(BaseExportPatch):
    """
    Patch for Qwen2.5-VL model to make it compatible with torch.export.

    This patch applies fixes for:
    1. Vision transformer forward method (using custom ops for data-dependent operations)
    2. Rotary embedding forward method (handling 'meta' device during export)
    3. create_causal_mask function (returns None to avoid execution issues)
    4. WAR for multimodal export: flat image features and synthetic tensor position_ids
    """

    def _apply_patch(self):
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_modeling
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2_5_VLModel,
            Qwen2_5_VLRotaryEmbedding,
        )

        # Store original methods
        self.original_values["vision_forward"] = Qwen2_5_VisionTransformerPretrainedModel.forward
        self.original_values["rope_forward"] = Qwen2_5_VLRotaryEmbedding.forward
        self.original_values["create_causal_mask"] = create_causal_mask
        self.original_values["qwen_create_causal_mask"] = qwen_modeling.create_causal_mask
        # Store originals for model-level methods
        self.original_values["model_get_image_features"] = Qwen2_5_VLModel.get_image_features
        self.original_values["model_forward"] = Qwen2_5_VLModel.forward

        # Apply patches
        Qwen2_5_VisionTransformerPretrainedModel.forward = _patched_vision_forward
        Qwen2_5_VLRotaryEmbedding.forward = _patched_rope_forward
        Qwen2_5_VLModel.get_image_features = _patched_get_image_features_flat
        Qwen2_5_VLModel.forward = _patched_model_forward_export_war

        # Patch the create_causal_mask function in both the masking_utils module
        # and the locally imported reference in the qwen2_5_vl modeling module
        import transformers.masking_utils

        transformers.masking_utils.create_causal_mask = _patched_create_causal_mask
        qwen_modeling.create_causal_mask = _patched_create_causal_mask

    def _revert_patch(self):
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_modeling
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2_5_VLModel,
            Qwen2_5_VLRotaryEmbedding,
        )

        # Restore original methods
        Qwen2_5_VisionTransformerPretrainedModel.forward = self.original_values["vision_forward"]
        Qwen2_5_VLRotaryEmbedding.forward = self.original_values["rope_forward"]
        Qwen2_5_VLModel.get_image_features = self.original_values["model_get_image_features"]
        Qwen2_5_VLModel.forward = self.original_values["model_forward"]

        # Restore original create_causal_mask function in both locations
        import transformers.masking_utils

        transformers.masking_utils.create_causal_mask = self.original_values["create_causal_mask"]
        qwen_modeling.create_causal_mask = self.original_values["qwen_create_causal_mask"]
