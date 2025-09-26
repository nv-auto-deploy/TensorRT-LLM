"""A simple wrapper transform to export a model to a graph module."""

import types
from typing import List, Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule

from ...export import torch_export_to_gm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class ExportToGMConfig(TransformConfig):
    """Configuration for the export to graph module transform."""

    strict: bool = Field(
        description="Whether to export in strict mode. NOTE: we generally export in non-strict mode"
        "for now as it relaxes some assumptions around tracing. Strict mode uses torchdynamo"
        "(symbolic bytecode analysis), which can be brittle since it relies on the exact bytecode"
        "representation of the model see here as well: https://pytorch.org/docs/stable/export.html#non-strict-export",
        default=False,
    )
    clone_state_dict: bool = Field(
        description="Whether to clone the state_dict of the model. This is useful to avoid"
        "modifying the original state_dict of the model.",
        default=False,
    )
    patch_list: Optional[List[str]] = Field(
        description="List of patch names to apply with export. "
        "Default is to apply all registered patches.",
        default=None,
    )


@TransformRegistry.register("export_to_gm")
class ExportToGM(BaseTransform):
    """A simple wrapper transform to export a model to a graph module."""

    config: ExportToGMConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ExportToGMConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # at this point we assume the gm is just a dummy graph module
        assert len(gm.graph.nodes) == 0, "Expected empty graph module."

        # retrieve the actual model from the dummy graph module
        model = gm.get_submodule("factory_model")

        # set the example sequence
        cm.info.set_example_sequence(**factory.get_example_inputs())

        # update example sequence
        # kwargs = cm.named_args
        # kwargs["inputs_embeds"] = model.get_input_embeddings()(kwargs.pop("input_ids"))

        # do some more kwargs
        kwargs = {}
        kwargs["attention_mask"] = None
        kwargs["position_ids"] = cm.named_args["position_ids"]
        kwargs["past_key_values"] = None
        kwargs["inputs_embeds"] = model.get_input_embeddings()(cm.named_args["input_ids"])
        kwargs["use_cache"] = None
        kwargs["cache_position"] = None

        # do some dummy dynamic shapes
        d_shapes = {}
        d_shapes["attention_mask"] = None
        d_shapes["position_ids"] = cm.named_dynamic_shapes["position_ids"]
        d_shapes["past_key_values"] = None
        d_shapes["inputs_embeds"] = cm.named_dynamic_shapes["input_ids"]
        d_shapes["use_cache"] = None
        d_shapes["cache_position"] = None

        # export the model to a graph module
        gm = torch_export_to_gm(
            model,
            args=(),
            kwargs=kwargs,
            dynamic_shapes=d_shapes,
            clone=self.config.clone_state_dict,
            strict=self.config.strict,
            patch_list=self.config.patch_list,
        )

        # some more stuff
        gm.embed_tokens = model.get_input_embeddings()
        gm.get_input_embeddings = types.MethodType(model.get_input_embeddings.__func__, gm)

        # add a dummy node to the graph for making the embedding module "sticky/impure"
        # TODO (lucaslie): is there a better way to make it "sticky"?
        n_embed_tokens = gm.graph.get_attr("embed_tokens.weight")
        gm.graph.call_function(
            torch._assert, args=(n_embed_tokens, "Avoid embedding getting deleted from graph.")
        )

        # this is a clean graph by definition since it was just exported
        info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

        return gm, info
