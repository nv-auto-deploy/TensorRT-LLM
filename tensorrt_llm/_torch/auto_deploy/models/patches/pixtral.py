"""A patch for the PixtralVisionModel to make it compatible with torch.export."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mistral3.modeling_mistral3 import Mistral3PatchMerger
from transformers.models.pixtral.modeling_pixtral import (
    generate_block_attention_mask,
    PixtralVisionModel,
    position_ids_in_meshgrid,
)

from ...export.interface import BaseExportPatch, ExportPatchRegistry

# @williamz notes:
# 1. everything decorated by a `custom_op` must be type annotated.
#    a. It must be one of the internally supported param types. As such, `self: PixtralVisionModel`
#       is a no-go.
#    As such, pretty much only free-standing functions with tensor inputs are supported - instance
#    methods cannot be decorated.

@torch.library.custom_op("auto_deploy::process_pixtral_patch_embeds", mutates_args={})
def _process_patch_embeds(
    patch_embeds: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    hidden_size: int,
    max_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_embeds_list = [
        embed[..., : (size[0] // patch_size), : (size[1] // patch_size)]
        for embed, size in zip(patch_embeds, image_sizes)
    ]

    # flatten to a single sequence
    patch_embeds = torch.cat([p.flatten(1).T for p in patch_embeds_list], dim=0).unsqueeze(0)

    position_ids = position_ids_in_meshgrid(patch_embeds_list, max_width=max_width)

    return patch_embeds, position_ids


@_process_patch_embeds.register_fake
def _process_patch_embeds_meta(
    patch_embeds: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    hidden_size: int,
    max_widht: int,
):
    B = (image_sizes // patch_size).prod(dim=1).sum()
    device = patch_embeds.device
    return (
        # Leading 1 = `unsqueeze(0)`.
        # The symbolic tracing will actually not complain if `1` is missing - I guess because
        # the number of elements in the underlying tensor is the same?
        torch.empty(1, B, hidden_size, device=device),
        torch.empty(hidden_size, device=device, dtype=torch.int64),
    )


def _pixtral_forward(
    self: PixtralVisionModel,
    pixel_values: torch.Tensor,
    image_sizes: torch.Tensor | None,
    output_hidden_states: bool | None = None,
    output_attentions: bool | None = None,
    return_dict: bool | None = None,
    *args,
    **kwargs,
):
    if image_sizes is None:
        batch_size, _, height, width = pixel_values.shape
        image_sizes = torch.tensor([(height, width)] * batch_size, device=pixel_values.device)

    # pass images through initial convolution independently
    patch_embeds = self.patch_conv(pixel_values)
    patch_embeds, position_ids = torch.ops.auto_deploy.process_pixtral_patch_embeds(
        patch_embeds=patch_embeds,
        image_sizes=image_sizes,
        patch_size=self.patch_size,
        hidden_size=self.config.hidden_size,
        max_width=self.config.image_size // self.config.patch_size,
    )

    patch_embeds = self.ln_pre(patch_embeds)

    kwargs["position_ids"] = position_ids

    position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

    if self.config._attn_implementation == "flash_attention_2":
        # We only rely on position_ids when using flash_attention_2
        attention_mask = None
    else:
        attention_mask = generate_block_attention_mask(
            (image_sizes // self.config.patch_size).prod(dim=1),
            patch_embeds,
        )

    return self.transformer(
        patch_embeds,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_dict=True,
        **kwargs,
    )


def generate_block_attention_mask(num_ids_per_image, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min

    idx = torch.arange(seq_len, device=device)
    block_end_idx = num_ids_per_image.cumsum(-1)
    block_start_idx = torch.cat(
        [
            num_ids_per_image.new_zeros((1,)),
            num_ids_per_image[:-1],
        ]
    ).cumsum(-1)

    # Build a mask where positions outside each [start, end) block are 1, inside are 0.
    mask = torch.ones((seq_len, seq_len), device=device, dtype=dtype)
    for start, end in zip(block_start_idx, block_end_idx):
        block = (idx >= start) & (idx < end)
        mask[block.unsqueeze(0) & block.unsqueeze(1)] = 0

    return mask


@torch.library.custom_op("auto_deploy::unfold_to_2d_grid", mutates_args={})
def _unfold_to_2d_grid(
    image_features: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    image_sizes = [
        (image_size[0] // patch_size, image_size[1] // patch_size) for image_size in image_sizes
    ]

    tokens_per_image = [h * w for h, w in image_sizes]
    d = image_features.shape[-1]

    permuted_tensor = []
    for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
        # Reshape image_tokens into a 2D grid
        h, w = image_sizes[image_index]
        image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
        grid = torch.nn.functional.unfold(
            image_grid, kernel_size=spatial_merge_size, stride=spatial_merge_size
        )
        grid = grid.view(d * spatial_merge_size**2, -1).t()
        permuted_tensor.append(grid)

    image_features = torch.cat(permuted_tensor, dim=0)


@_unfold_to_2d_grid.register_fake
def _unfold_to_2d_grid_meta(
    image_features: torch.Tensor,
    image_sizes: torch.Tensor,
    patch_size: int,
    spatial_merge_size: int,
):
    embedding_sizes = (image_sizes // patch_size).prod(dim=1)
    spatial_factor = spatial_merge_size * spatial_merge_size
    grid_sizes = embedding_sizes // spatial_factor
    total_size = grid_sizes.sum()

    return image_features.new_empty(total_size, image_features.shape[-1] * spatial_factor)


def _patch_merger_forward(
    self, image_features: torch.Tensor, image_sizes: torch.Tensor
) -> torch.Tensor:
    unfolded_features = torch.ops.auto_deploy.unfold_to_2d_grid(
        image_features=image_features,
        image_sizes=image_sizes,
        patch_size=self.patch_size,
        spatial_merge_size=self.spatial_merge_size,
    )
    image_features = self.merging_layer(unfolded_features)
    return image_features


@ExportPatchRegistry.register("hf_pixtral_vit")
class PixtralVisionModelPatch(BaseExportPatch):
    """Patch for `PixtralVisionModel`."""

    def _apply_patch(self):
        """Apply the PixtralVisionModel patch."""
        # Store original forward method
        self.original_values["PixtralVisionModel.forward"] = PixtralVisionModel.forward
        self.original_values["Mistral3PatchMerger.forward"] = Mistral3PatchMerger.forward

        # Apply patch by replacing the forward method
        PixtralVisionModel._original_forward = PixtralVisionModel.forward  # type: ignore
        PixtralVisionModel.forward = _pixtral_forward  # type: ignore

        Mistral3PatchMerger._original_forward = Mistral3PatchMerger.forward
        Mistral3PatchMerger.forward = _patch_merger_forward

    def _revert_patch(self):
        """Revert the PixtralVisionModel patch."""
        # Restore original forward method.
        PixtralVisionModel.forward = self.original_values["PixtralVisionModel.forward"]  # type: ignore
        Mistral3PatchMerger.forward = self.original_values["Mistral3PatchMerger.forward"]

        # Clean up the temporary attribute.
        if hasattr(PixtralVisionModel, "_original_forward"):
            delattr(PixtralVisionModel, "_original_forward")

        if hasattr(Mistral3PatchMerger, "_original_forward"):
            delattr(Mistral3PatchMerger, "_original_forward")
