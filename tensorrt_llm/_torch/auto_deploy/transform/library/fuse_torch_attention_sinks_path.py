# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-cast attention sinks to float32 for the trtllm cached attention path.

The thop.attention C++ kernel (used by the trtllm AutoDeploy attention backend)
expects ``attention_sinks`` in ``float32``.  GPT-OSS-style models hold ``sinks``
as a per-head learnable scalar in the model dtype (typically ``bfloat16``), so
``trtllm_attention_mha_with_cache`` casts via ``attention_sinks.to(float32)``
on every forward.  Under CUDA-graph capture the cast becomes a per-layer
in-graph kernel.

This post_load_fusion transform inspects each ``torch_attention`` call whose
``sinks=`` kwarg references a non-fp32 tensor parameter, registers a sibling
``float32`` buffer on the parent module that holds the pre-cast values, and
rewires the ``sinks`` kwarg to read from that buffer.  The runtime dtype check
in ``trtllm_attention_mha_with_cache`` then short-circuits and the per-layer
cast disappears from the captured graph, while the original parameter remains
in place for any non-trtllm consumer that needs the model-dtype copy.

Disabled by default; enable in model configs that route ``torch_attention``
calls to ``attn_backend: trtllm`` and carry sinks (GPT-OSS family).
"""

from typing import Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args, is_op, set_op_args
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _get_nested_attr(module: torch.nn.Module, target: str) -> torch.Tensor:
    """Resolve a dotted attribute path (e.g. ``model.layers.0.self_attn.sinks``)."""
    obj = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


@TransformRegistry.register("fuse_torch_attention_sinks_path")
class FuseTorchAttentionSinksPath(BaseTransform):
    """Pre-cast sinks tensors to float32 for trtllm cached attention.

    For each ``torch_attention`` call with a non-fp32 ``sinks`` get_attr Node,
    register a sibling ``<name>_ad_fp32`` non-persistent buffer on the parent
    module and rewire the ``sinks`` kwarg to it.  Skips calls whose sinks is
    already fp32, missing, on the meta device, or not a get_attr Node — those
    are not safe to rewrite without changing semantics.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        num_fused = 0
        # Cache the new fp32 buffer Node per (sinks_target, parent) pair so multiple
        # attention calls that share a sinks parameter all rewire to the same buffer.
        rewired: dict[str, Node] = {}

        for attn_node in list(graph.nodes):
            if not is_op(attn_node, torch.ops.auto_deploy.torch_attention):
                continue
            if self._try_precast_one(gm, attn_node, rewired):
                num_fused += 1

        info = TransformInfo(
            skipped=num_fused == 0,
            num_matches=num_fused,
            is_clean=num_fused == 0,
            has_valid_shapes=num_fused == 0,
        )
        return gm, info

    @staticmethod
    def _try_precast_one(gm: GraphModule, attn_node: Node, rewired: dict[str, Node]) -> bool:
        sinks_arg = extract_op_args(attn_node, "sinks")[0]
        if not isinstance(sinks_arg, Node) or sinks_arg.op != "get_attr":
            return False

        target: str = sinks_arg.target  # type: ignore[assignment]

        if target in rewired:
            set_op_args(attn_node, sinks=rewired[target])
            return True

        try:
            sinks_tensor = _get_nested_attr(gm, target)
        except AttributeError:
            return False
        if not isinstance(sinks_tensor, torch.Tensor) or sinks_tensor.is_meta:
            return False
        if sinks_tensor.dtype == torch.float32:
            return False

        if "." in target:
            parent_path, leaf = target.rsplit(".", 1)
            parent = _get_nested_attr(gm, parent_path)
            new_target = f"{parent_path}.{leaf}_ad_fp32"
            new_leaf = f"{leaf}_ad_fp32"
        else:
            parent = gm
            parent_path = ""
            leaf = target
            new_target = f"{leaf}_ad_fp32"
            new_leaf = new_target

        sinks_fp32 = sinks_tensor.detach().to(torch.float32).contiguous()

        # Idempotency: if a prior run registered the buffer, reuse it.
        existing = getattr(parent, new_leaf, None)
        if (
            isinstance(existing, torch.Tensor)
            and existing.dtype == torch.float32
            and existing.shape == sinks_fp32.shape
        ):
            sinks_fp32 = existing
        else:
            parent.register_buffer(new_leaf, sinks_fp32, persistent=False)

        graph = attn_node.graph
        with graph.inserting_before(attn_node):
            new_node = graph.create_node("get_attr", new_target)
        new_node.meta["val"] = sinks_fp32

        try:
            set_op_args(attn_node, sinks=new_node)
        except RuntimeError as exc:
            ad_logger.debug(
                f"fuse_torch_attention_sinks_path: failed to rewire sinks for {attn_node}: {exc}"
            )
            graph.erase_node(new_node)
            return False

        rewired[target] = new_node
        return True
