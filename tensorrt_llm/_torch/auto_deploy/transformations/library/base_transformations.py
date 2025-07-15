"""Base classes for transformations."""

from abc import ABC, abstractmethod
from typing import List

import torch
from pydantic import BaseModel, ConfigDict, Field

from ...utils.logger import ad_logger
from .._graph import canonicalize_graph


class TransformationInfo(BaseModel, ABC):
    """Abstract base class for transformation configurations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    anchor_gm: torch.fx.GraphModule
    anchor_node: torch.fx.Node

    def __eq__(self, other):
        """Custom equality comparison ignoring GraphModule and Node objects."""
        if not isinstance(other, self.__class__):
            return False

        # Compare all fields except anchor_gm and anchor_node
        self_dict = self.model_dump(exclude={"anchor_gm", "anchor_node"})
        other_dict = other.model_dump(exclude={"anchor_gm", "anchor_node"})

        # Compare node names instead of objects
        self_dict["anchor_node_name"] = getattr(self.anchor_node, "name", str(self.anchor_node))
        other_dict["anchor_node_name"] = getattr(other.anchor_node, "name", str(other.anchor_node))

        return self_dict == other_dict

    def __hash__(self):
        """Custom hash function for set operations."""
        # Create a hashable representation excluding anchor_gm and anchor_node
        hashable_dict = self.model_dump(exclude={"anchor_gm", "anchor_node"})
        hashable_dict["anchor_node_name"] = getattr(self.anchor_node, "name", str(self.anchor_node))

        # Convert dict to tuple of sorted items for hashing
        return hash(tuple(sorted(hashable_dict.items())))

    @abstractmethod
    def apply(self) -> None:
        """Apply the transformation to the graph module.

        This method must be implemented by each transformation class.
        """
        pass


class TransformationConfig(BaseModel):
    """Configuration for transformations."""

    transformation_list: List[TransformationInfo] = Field(default_factory=list)


def transformation_executor(config: TransformationConfig) -> None:
    """Apply transformations to the graph module.

    Args:
        config: Transformation configuration containing list of transformations to apply
    """
    gms = set()
    for transformation in config.transformation_list:
        if transformation.anchor_gm is None or transformation.anchor_node is None:
            continue
        gms.add(transformation.anchor_gm)
        transformation.apply()

    # canonicalize and return
    for gm in gms:
        gm = canonicalize_graph(gm)
        ad_logger.debug("After applying graph transformations: " + str(gm))
