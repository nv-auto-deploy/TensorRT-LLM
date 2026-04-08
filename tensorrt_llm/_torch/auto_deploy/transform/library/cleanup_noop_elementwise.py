import operator
from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# Map target -> identity value that makes the op a no-op.
# For add/sub the identity is 0 (x + 0 = x, x - 0 = x).
# For mul the identity is 1 (x * 1 = x).
_BUILTIN_TARGETS = {operator.sub: 0, operator.mul: 1, operator.add: 0}
_ATEN_TARGETS = {torch.ops.aten.sub.Tensor: 0, torch.ops.aten.mul.Tensor: 1}


@TransformRegistry.register("cleanup_noop_elementwise")
class CleanupNoopElementwise(BaseTransform):
    """Eliminate elementwise nodes from the graph that are no-ops.

    This covers patterns where an elementwise operation has no effect on the input tensor:
      - operator.sub(x, 0) / aten.sub.Tensor(x, 0): subtracting zero
      - operator.mul(x, 1) / aten.mul.Tensor(x, 1): multiplying by one
      - operator.add(x, 0): adding zero via the Python built-in operator

    These no-ops each launch a CUDA kernel for nothing. Removing them is safe as long as the
    second argument is a scalar constant equal to the identity value for that operation.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue

            # Determine the identity value for this target, if applicable
            identity_value = None

            # Check built-in operator targets
            if node.target in _BUILTIN_TARGETS:
                identity_value = _BUILTIN_TARGETS[node.target]
            # Check aten targets
            elif node.target in _ATEN_TARGETS:
                identity_value = _ATEN_TARGETS[node.target]
            else:
                continue

            # We expect at least 2 positional args: (input_tensor, scalar)
            if len(node.args) < 2:
                continue

            # Check if the second argument is a scalar constant equal to the identity value
            second_arg = node.args[1]
            if not isinstance(second_arg, (int, float)):
                continue
            if second_arg != identity_value:
                continue

            # Do the replacement and clean-up
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
            num_matches += 1

        # Store info object about the transform
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )

        return gm, info
