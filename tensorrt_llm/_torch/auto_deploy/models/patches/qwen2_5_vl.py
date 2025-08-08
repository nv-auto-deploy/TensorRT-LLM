"""Patches for Qwen2.5-VL model to make it compatible with torch.export."""

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


@ExportPatchRegistry.register("qwen2_5_vl_vision")
class Qwen2_5_VLVisionPatch(BaseExportPatch):
    """
    Patch for Qwen2.5-VL model to make it compatible with torch.export.

    This patch applies fixes for:
    1. Vision transformer forward method (using custom ops for data-dependent operations)
    2. Rotary embedding forward method (handling 'meta' device during export)
    3. create_causal_mask function (returns None to avoid execution issues)
    """

    def _apply_patch(self):
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_modeling
        from transformers.masking_utils import create_causal_mask
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2_5_VLRotaryEmbedding,
        )

        # Store original methods
        self.original_values["vision_forward"] = Qwen2_5_VisionTransformerPretrainedModel.forward
        self.original_values["rope_forward"] = Qwen2_5_VLRotaryEmbedding.forward
        self.original_values["create_causal_mask"] = create_causal_mask
        self.original_values["qwen_create_causal_mask"] = qwen_modeling.create_causal_mask

        # Apply patches
        Qwen2_5_VisionTransformerPretrainedModel.forward = _patched_vision_forward
        Qwen2_5_VLRotaryEmbedding.forward = _patched_rope_forward

        # Patch the create_causal_mask function in both the masking_utils module
        # and the locally imported reference in the qwen2_5_vl modeling module
        import transformers.masking_utils

        transformers.masking_utils.create_causal_mask = _patched_create_causal_mask
        qwen_modeling.create_causal_mask = _patched_create_causal_mask

    def _revert_patch(self):
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_modeling
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VisionTransformerPretrainedModel,
            Qwen2_5_VLRotaryEmbedding,
        )

        # Restore original methods
        Qwen2_5_VisionTransformerPretrainedModel.forward = self.original_values["vision_forward"]
        Qwen2_5_VLRotaryEmbedding.forward = self.original_values["rope_forward"]

        # Restore original create_causal_mask function in both locations
        import transformers.masking_utils

        transformers.masking_utils.create_causal_mask = self.original_values["create_causal_mask"]
        qwen_modeling.create_causal_mask = self.original_values["qwen_create_causal_mask"]
