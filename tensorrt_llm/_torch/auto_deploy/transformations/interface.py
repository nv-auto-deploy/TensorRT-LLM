"""The interface for all transforms.

This module defines the base classes and interfaces for all transforms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Type, final

from pydantic import BaseModel, Field
from torch.fx import GraphModule


class Stages(Enum):
    """Enumerated (ordered!) stages of the transformation pipeline.

    This is used to classify and pre-order transforms.
    """

    POST_EXPORT = auto()  # low-level cleanups of the exported graph
    PATTERN_MATCHER = auto()  # high-level pattern matching to standardize graph representation
    SHARDING = auto()  # auto-sharding of the graph
    WEIGHT_LOAD = auto()  # loading of the model weights
    POST_LOAD_FUSION = auto()  # post-loading fusion and perf optimizations of the graph
    CACHE_INIT = auto()  # initialization of cached attention + cache initialization (e.g. KV cache)
    COMPILE = auto()  # graph compilation stage using low-level compilers like torch.compile


class TransformConfig(BaseModel):
    """A simple configuration class that can be extended by a transform for configurability."""

    model_config = {
        "extra": "allow",  # to provide an easy way to do config validation without
    }

    enabled: bool = Field(
        default=True,
        description="Whether to enable this transform.",
    )
    run_graph_cleanup: bool = Field(
        default=True,
        description="Whether to run graph cleanup/canonicalization after this transform.",
    )
    requires_shape_prop: bool = Field(
        default=False,
        description="Whether this transform requires shape propagation before.",
    )


@dataclass
class TransformInfo:
    """Information about the result of a transform."""

    num_matches: int


class BaseTransform(ABC):
    """A base class for all transforms."""

    config: TransformConfig  # overwrite type hint if other config cls is used in subclass!

    @classmethod
    @abstractmethod
    def get_transform_key(cls) -> str:
        """Get the short name of the transform.

        This is used to identify the transform in the transformation pipeline.
        """

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        """Get the configuration class for the transform.

        This is used to validate the configuration of the transform.
        """
        return TransformConfig

    @final
    def __init__(self, config: TransformConfig):
        """Initialize the transform.

        Args:
            config: The configuration for the transform.

        To customize the initialization, override the `_post_init` method.
        """
        self.config = self.get_config_class()(**config.model_dump())
        self._post_init()

    def _post_init(self):
        """Post-initialization hook that can be overridden by subclasses."""
        pass

    @final
    def __call__(self, gm: GraphModule) -> TransformInfo:
        """Apply the transform to the graph.

        Args:
            gm: The graph module to apply the transform to.

        Returns:
            TransformInfo: Information about the result of the transform.

        This method is the main entry point for any transforms and is called by the
        InferenceOptimizer pipeline.
        """
        return self._apply(gm)

    @abstractmethod
    def _apply(self, gm: GraphModule) -> TransformInfo:
        """Apply the transform to the graph.

        This is the core method that should be implemented by subclasses.
        """
