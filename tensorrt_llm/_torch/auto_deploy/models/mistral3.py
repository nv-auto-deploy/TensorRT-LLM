"""Auto-deploy model factory for Mistral3 models."""

import types
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.custom_ops import attention_interface
from tensorrt_llm._torch.auto_deploy.models import factory, hf

from ..custom_ops.attention_interface import Dim


@factory.ModelFactoryRegistry.register("Mistral3VLM")
class Mistral3VLM(hf.AutoModelForImageTextToTextFactory):
    def get_submodule_export_config(self) -> Dict[str, Any]:
        def _post_process(sub_mod: nn.Module, key: str, sub_gm: GraphModule):
            sub_gm.embed_tokens = sub_mod.get_input_embeddings()
            sub_gm.get_input_embeddings = types.MethodType(
                sub_mod.get_input_embeddings.__func__, sub_gm
            )

            # add a dummy node to the graph for making the embedding module "sticky/impure"
            # TODO (lucaslie): is there a better way to make it "sticky"?
            n_embed_tokens = sub_gm.graph.get_attr("embed_tokens.weight")
            sub_gm.graph.call_function(
                torch._assert, args=(n_embed_tokens, "Avoid embedding getting deleted from graph.")
            )

        def _get_dynamic_shapes():
            seq_batch_size = Dim.DYNAMIC
            seq_len = Dim.DYNAMIC
            img_batch_size = Dim.DYNAMIC
            img_height = Dim.DYNAMIC
            img_width = Dim.DYNAMIC
            # seq_batch_size = Dim("batch_size", min=1)
            # seq_len = Dim("seq_len", min=1)
            # img_batch_size = Dim("img_batch_size")
            # img_height = Dim("img_height", min=32)
            # img_width = Dim("img_width", min=32)

            return {
                "input_ids": {0: seq_batch_size, 1: seq_len},
                "inputs_embeds": {0: seq_batch_size, 1: seq_len},
                "position_ids": {0: seq_batch_size, 1: seq_len},
                "pixel_values": {0: img_batch_size, 2: img_height, 3: img_width},
                "image_sizes": {0: img_batch_size},
            }

        return {"model.language_model": (_get_dynamic_shapes, _post_process)}

    def get_extra_inputs(
        self,
    ) -> Dict[str, Tuple[torch.Tensor, attention_interface.DynamicShapeCallback]]:
        """Return a dictionary of extra inputs for the model.

        Returns:
            A dictionary of extra inputs for the model where the key corresponds to the argument
            name and the value corresponds to a tuple of (example_input, dynamic_shape_callback).
            The dynamic shape callback is a function that returns the dynamic shape of the extra
            input.
        """
        return {}
        extra_inputs = super().get_extra_inputs()
        # Reuse the same dynamic batch dimension for `image_sizes`.
        batch_dim = extra_inputs["pixel_values"][1]()[0]
        extra_inputs["image_sizes"] = (torch.zeros(0, 2, dtype=torch.long), lambda: {0: batch_dim})

        return extra_inputs

    def get_example_inputs(self) -> Dict[str, torch.Tensor]:
        """Return a dictionary of example inputs for the model."""
        return {}
        # return super().get_example_inputs()

    @property
    def _example_image_dims(self) -> Tuple[int, int]:
        # The pixtral processor requires a minimum image size, which is larger than the default (16, 16)
        # in the parent class.
        # TODO: figure this out on the model config somehow (patch size value, etc.).
        return (64, 64)
