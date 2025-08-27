"""Auto-deploy model factory for Mistral3 models."""

from typing import Dict, Tuple

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops import attention_interface
from tensorrt_llm._torch.auto_deploy.models import factory, hf


@factory.ModelFactoryRegistry.register("Mistral3VLM")
class Mistral3VLM(hf.AutoModelForImageTextToTextFactory):
    def __init__(self, *args, **kwargs):
        if kwargs.get("example_input_names"):
            raise ValueError(f"`example_input_names` cannot be specified for {type(self)}.")

        example_image_dims = kwargs.pop("example_image_dims", (64, 64))
        for i, dim in enumerate(example_image_dims):
            if dim < 32:
                raise ValueError(
                    f"The {i}-th example image dimension {dim} needs to be larger than 32."
                )

        super().__init__(
            *args,
            example_image_dims=example_image_dims,
            example_input_names=["input_ids", "pixel_values", "image_sizes"],
            **kwargs,
        )

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
        extra_inputs = super().get_extra_inputs()
        # Reuse the same dynamic batch dimension for `image_sizes`.
        batch_dim = extra_inputs["pixel_values"][1]()[0]
        extra_inputs["image_sizes"] = (torch.zeros(0, 2), lambda: {0: batch_dim})

        return extra_inputs

    # ? Is this necessary if mistral3's forward is patched in e.g. `models/patches/foo.py`?
    @staticmethod
    def _simple_forward(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
    ):
        """A simple forward pass for the model to functionalize the args.

        This follows the standard function signature as expected by factory.py.
        """
        return type(model).forward(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        )
