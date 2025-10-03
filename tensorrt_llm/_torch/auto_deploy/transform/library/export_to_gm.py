"""A simple wrapper transform to export a model to a graph module."""

import types
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule

from ...export import run_forward_for_capture, torch_export_to_gm
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

    # def _apply_to_full_model(
    #     self,
    #     mod: nn.Module,
    #     cm: CachedSequenceInterface,
    #     factory: ModelFactory,
    #     shared_config: SharedConfig,
    # ) -> Tuple[nn.Module, TransformInfo]:
    #     # set the example sequence
    #     cm.info.set_example_sequence(**factory.get_example_inputs())

    #     # export the model to a graph module
    #     gm = torch_export_to_gm(
    #         mod,
    #         args=(),
    #         kwargs=cm.named_args,
    #         dynamic_shapes=cm.named_dynamic_shapes,
    #         clone=self.config.clone_state_dict,
    #         strict=self.config.strict,
    #         patch_list=self.config.patch_list,
    #     )

    #     # this is a clean graph by definition since it was just exported
    #     info = TransformInfo(skipped=False, num_matches=1, is_clean=True, has_valid_shapes=True)

    #     return gm, info

    def _apply_to_full_model(
        self,
        mod: nn.Module,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[nn.Module, TransformInfo]:
        # set the example sequence
        cm.info.set_example_sequence(**factory.get_example_inputs())

        captured_kwargs = {}

        def _forward_for_capture(mod: nn.Module, **kwargs):
            captured_kwargs.clear()
            captured_kwargs.update(kwargs)
            return mod._original_forward(**kwargs)

        def _capture_fn(mod: nn.Module, args, kwargs):
            mod(*args, **kwargs)
            return mod

        num_matches = 0

        for key, (_get_dyn_shape, _post_process) in factory.get_submodule_export_config().items():
            sub_mod = mod.get_submodule(key)
            sub_mod._original_forward = sub_mod.forward
            sub_mod.forward = types.MethodType(_forward_for_capture, sub_mod)
            run_forward_for_capture(
                mod,
                _capture_fn,
                args=(),
                kwargs=cm.named_args,
                clone=self.config.clone_state_dict,
                patch_list=self.config.patch_list,
            )
            sub_mod.forward = sub_mod._original_forward
            del sub_mod._original_forward

            # construct dynamic shapes
            dyn_shape_lookup = _get_dyn_shape()
            dynamic_shapes = {
                k: dyn_shape_lookup.get(k, {} if isinstance(v, torch.Tensor) else None)
                for k, v in captured_kwargs.items()
            }

            # export the model to a graph module
            if True:
                sub_gm = torch_export_to_gm(
                    sub_mod,
                    args=(),
                    kwargs=captured_kwargs,
                    dynamic_shapes=dynamic_shapes,
                    clone=self.config.clone_state_dict,
                    strict=self.config.strict,
                    patch_list=self.config.patch_list,
                )

            # post process the sub graph module
            _post_process(sub_mod, key, sub_gm)

            # set the sub graph module
            mod.set_submodule(key, sub_gm)

            # increment the number of matches
            num_matches += 1

        # this is a clean graph by definition since it was just exported
        info = TransformInfo(
            skipped=False, num_matches=num_matches, is_clean=True, has_valid_shapes=True
        )

        return mod, info
