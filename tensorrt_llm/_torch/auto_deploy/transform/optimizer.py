"""High-level entrypoint to transform a model into an efficient inference model."""

import gc
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.fx import GraphModule

from ..distributed import common as dist_ad
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from ..utils.node_utils import draw_graph
from .interface import (
    InferenceOptimizerConfig,
    SharedConfig,
    Stages,
    StrictInferenceOptimizerConfig,
    TransformConfig,
    TransformRegistry,
)


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, config: InferenceOptimizerConfig):
        self.factory = factory
        self.config = self._clean_config(config)
        if not dist.is_initialized():
            local_rank, world_size = 0, 1
        else:
            local_rank, world_size = dist_ad.get_rank_world_size()
        self.shared_config = SharedConfig(local_rank=local_rank, world_size=world_size)

    def _clean_config(self, config: InferenceOptimizerConfig) -> StrictInferenceOptimizerConfig:
        """Get a typed checked ("strict") config with sorted keys according to stages."""
        # convert to nested kwargs, no TransformConfig objects allowed
        nested_kwargs = {
            k: v.model_dump() if isinstance(v, TransformConfig) else v for k, v in config.items()
        }
        # sort by stage
        keys_sorted = sorted(nested_kwargs.keys(), key=lambda k: Stages(nested_kwargs[k]["stage"]))
        # create strict config with correct config classes and correct order
        strict_config: StrictInferenceOptimizerConfig = {
            k: TransformRegistry.get_config_class(k)(**nested_kwargs[k]) for k in keys_sorted
        }
        # return strict config
        return strict_config

    def __call__(self, cm: CachedSequenceInterface, mod: Optional[nn.Module] = None) -> nn.Module:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.
            mod: The model to transform.

        Returns:
            A nn.Module representing the optimized inference model.
        """
        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty model if not provided
        if mod is None:
            mod = nn.Module()

        # iterate over all transforms sorted by stage in the config
        for t_name, t_config in self.config.items():
            # instantiate transform
            transform = TransformRegistry.get(t_name)(t_config)
            # run transform
            mod = transform(mod, cm, self.factory, self.shared_config)

            # draw graph after each pass if the module is a GraphModule
#            if isinstance(mod, GraphModule):
#                try:
#                    filename = f"graph_after_{t_name}"
#                    # Try to use FXGraphDrawer (requires graphviz)
#                    draw_graph(mod, filename)
#                    print(f"Graph saved to {filename}.svg after transform: {t_name}")
#                except Exception:
#                    # Fallback: save graph as text representation
#                    try:
#                        filename_txt = f"{filename}.txt"
#                        with open(filename_txt, "w") as f:
#                            f.write(f"Graph after {t_name}\n")
#                            f.write("=" * 80 + "\n\n")
#                            f.write(str(mod.graph))
#                        print(f"Graph saved to {filename_txt} after transform: {t_name} (text format)")
#                    except Exception as e:
#                        print(f"Warning: Failed to save graph after {t_name}: {e}")

        ############################################################################################
        # RETURN OPTIMIZED MODEL
        ############################################################################################
        torch.cuda.empty_cache()
        gc.collect()
        return mod
