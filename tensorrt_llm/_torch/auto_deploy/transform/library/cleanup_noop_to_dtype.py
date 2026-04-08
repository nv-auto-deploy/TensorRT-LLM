from typing import Tuple

import torch
from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


@TransformRegistry.register("cleanup_noop_to_dtype")
class CleanupNoopToDtype(BaseTransform):
    """Eliminate aten.to.dtype nodes from the graph that are no-ops.

    When the input tensor's dtype already matches the target dtype, the to.dtype call is a no-op
    that still launches a CUDA kernel. This transform removes those redundant nodes.

    For example, in the DeepSeek-R1 compiled graph, triton_rms_norm outputs BF16 tensors that are
    immediately followed by aten.to.dtype(..., torch.bfloat16), which is a no-op.
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
            # looking for aten.to.dtype nodes
            if not is_op(node, torch.ops.aten.to.dtype):
                continue

            # the pattern is aten.to.dtype(input_tensor, target_dtype)
            input_node = node.args[0]
            target_dtype = node.args[1]

            # determine the input tensor's dtype from metadata
            input_dtype = None
            val = input_node.meta.get("val")
            if val is not None and hasattr(val, "dtype"):
                input_dtype = val.dtype
            else:
                tensor_meta = input_node.meta.get("tensor_meta")
                if tensor_meta is not None and hasattr(tensor_meta, "dtype"):
                    input_dtype = tensor_meta.dtype

            if input_dtype is None:
                continue

            # if dtypes match, this to.dtype is a no-op
            if input_dtype != target_dtype:
                continue

            # do the replacement and clean-up
            node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
            num_matches += 1

        # store info object about the transform
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )

        return gm, info
