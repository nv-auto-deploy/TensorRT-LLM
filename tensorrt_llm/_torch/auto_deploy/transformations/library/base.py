"""Base classes for transformations."""

from abc import ABC

import torch
from pydantic import BaseModel, ConfigDict


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
