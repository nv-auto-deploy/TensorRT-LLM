"""High-level entrypoint to transform a model into an efficient inference model."""

import torch.nn as nn
from torch.fx import Graph, GraphModule

from ..llm_args import AutoDeployConfig
from ..models.factory import ModelFactory
from ..shim.interface import CachedSequenceInterface
from .interface import TransformRegistry


class InferenceOptimizer:
    def __init__(self, factory: ModelFactory, ad_config: AutoDeployConfig):
        self.factory = factory
        self.ad_config = ad_config

    def __call__(self, cm: CachedSequenceInterface) -> GraphModule:
        """Transform a model into an optimized inference model.

        Args:
            cm: The cached sequence interface defining the sequence interface.

        Returns:
            A GraphModule representing the optimized inference model.
        """
        ############################################################################################
        # RUN THROUGH CONFIGURED TRANSFORMATIONS
        ############################################################################################

        # start with an empty fake graph module
        gm = GraphModule(nn.Module(), Graph())

        # retrieve transforms config from ad_config
        t_configs = self.ad_config.transforms

        # sort transforms by stage
        keys_transforms = sorted(t_configs.keys(), key=lambda k: t_configs[k].stage)

        # iterate over all transforms now sorted by stage
        for t_key in keys_transforms:
            # instantiate transform
            transform = TransformRegistry.get(t_key)(**t_configs[t_key].model_dump())
            # run transform
            gm = transform(gm, cm, self.factory)

        ############################################################################################
        # RETURN OPTIMIZED GRAPH
        ############################################################################################
        return gm
