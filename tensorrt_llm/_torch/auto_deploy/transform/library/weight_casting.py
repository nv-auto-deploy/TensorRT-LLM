from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("weight_casting")
class WeightCasting(BaseTransform):
    """Any casts of weights are expressed as load hooks.
    This is a post-load transform to convert the weights to the correct dtype.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0

        with open("graph-before-weight-casting.txt", "w") as f:
            f.write(str(gm.graph))
        for node in gm.graph.nodes:
            # looking for to nodes
            if not is_op(node, torch.ops.aten.to):
                continue

            # weight node is the first argument
            target_dtype = node.args[1]
            if node.args[0].op == "get_attr":
                casted_weight_tensor = gm.get_parameter(node.args[0].target).to(target_dtype)
                # Sanitize parameter name: PyTorch does not allow '.' in parameter names.
                param_base = str(node.args[0].target).replace(".", "_")
                dtype_suffix = str(target_dtype)
                if dtype_suffix.startswith("torch."):
                    dtype_suffix = dtype_suffix.split(".", 1)[1]
                dtype_suffix = dtype_suffix.replace(".", "_")
                new_key = f"{param_base}_as_{dtype_suffix}"
                gm.register_parameter(
                    new_key, torch.nn.Parameter(casted_weight_tensor, requires_grad=False)
                )

                with gm.graph.inserting_before(node):
                    new_weight_node = gm.graph.create_node("get_attr", new_key)
                    node.replace_all_uses_with(new_weight_node)
                gm.graph.erase_node(node)
                num_matches += 1

        gm.graph.eliminate_dead_code()
        gm.delete_all_unused_submodules()

        # store info object about the transform
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        with open("graph-after-weight-casting.txt", "w") as f:
            f.write(str(gm.graph))
        return gm, info
