"""A set of transforms to handle SSM cache transforms."""

from typing import List

from torch.fx import GraphModule, Node

from ...custom_ops.attention_interface import Constant
from ...utils.node_utils import extract_op_args, is_op
from ..interface import TransformRegistry
from .kvcache import InsertCachedAttention


# TODO: think about separating valid attention backends per transform better in the future
@TransformRegistry.register("insert_cached_ssm_attention")
class SSMCacheTransform(InsertCachedAttention):
    """A transform to handle SSM cache operations."""

    def _process_get_metadata(
        self, gm: GraphModule, m_args: List[str], const_args: List[Constant]
    ) -> List[Node]:
        mamba_nodes = [
            n for n in gm.graph.nodes if is_op(n, self.attn_descriptor.get_source_attention_op())
        ]

        chunk_size = extract_op_args(mamba_nodes[0], "chunk_size")
        const_args.append(chunk_size)
        return super()._process_get_metadata(gm, m_args, const_args)


@TransformRegistry.register("insert_cached_causal_conv")
class InitializeCausalConvCache(InsertCachedAttention):
    """A transform to handle causal conv cache operations."""
