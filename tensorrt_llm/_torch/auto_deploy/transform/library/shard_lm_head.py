# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph transform: shard LM head weight along the vocab dimension for tensor parallelism.

At TP degree > 1, after the backbone AllReduce all TP ranks hold identical hidden
states and compute the full vocab projection redundantly.  This transform slices
lm_head.weight to the local rank's vocab shard and inserts an AllGather along
dim=-1 after the partial logits, turning one fat GEMV into tp_size smaller ones.

Works even when lm_head.weight is tied to embed_tokens.weight: it uses
get_lm_head_node() (position-based) to locate the LM-head linear regardless of
the weight tensor's attribute name in the exported graph.

Run in the post_load_fusion stage (after weights are loaded and fuse_fp8_linear
has run, so the LM head remains a standard aten.linear).
"""

from typing import Tuple, Type

from torch.fx import GraphModule

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import get_lm_head_node
from ...utils.node_utils import is_linear_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from .sharding import _get_dist_ops


@TransformRegistry.register("shard_lm_head_vocab_parallel")
class ShardLmHeadVocabParallel(BaseTransform):
    """Shard the LM head weight along vocab dim and AllGather partial logits.

    Turns the redundant full-vocab projection on each TP rank into a
    local-shard projection + AllGather.  Expected benefit: ~tp_size×
    reduction in LM-head GEMV kernel time.

    Run after ``fuse_fp8_linear`` (post_load_fusion stage).
    """

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return TransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Determine TP rank / size
        mapping = shared_config.mapping
        if mapping is not None:
            tp_rank = mapping.tp_rank
            tp_size = mapping.tp_size
        else:
            tp_rank = shared_config.local_rank
            tp_size = shared_config.world_size

        if tp_size <= 1:
            self._log_info("tp_size=1, skipping vocab-parallel LM head sharding")
            return gm, TransformInfo(skipped=True, num_matches=0)

        # Find the LM head linear node (position-based, robust to weight tying)
        lm_head_node = get_lm_head_node(gm)
        if not is_linear_op(lm_head_node):
            self._log_info(f"LM head node {lm_head_node} is not a standard linear op, skipping")
            return gm, TransformInfo(skipped=True, num_matches=0)

        # aten.linear / torch_linear_simple: args = (input, weight, bias?)
        if len(lm_head_node.all_input_nodes) < 2:
            self._log_info("LM head linear has fewer than 2 inputs, skipping")
            return gm, TransformInfo(skipped=True, num_matches=0)

        weight_node = lm_head_node.all_input_nodes[1]
        if weight_node.op != "get_attr":
            self._log_info(f"LM head weight input is not get_attr (op={weight_node.op}), skipping")
            return gm, TransformInfo(skipped=True, num_matches=0)

        # Retrieve the weight tensor (may be a Parameter or buffer in gm)
        weight_key = weight_node.target
        weight_tensor = None
        try:
            weight_tensor = gm.get_parameter(weight_key)
        except AttributeError:
            pass
        if weight_tensor is None:
            try:
                weight_tensor = gm.get_buffer(weight_key)
            except AttributeError:
                pass
        if weight_tensor is None:
            self._log_info(f"Cannot retrieve weight for attr={weight_key!r}, skipping")
            return gm, TransformInfo(skipped=True, num_matches=0)

        vocab_size = weight_tensor.shape[0]
        if vocab_size % tp_size != 0:
            self._log_info(f"vocab_size={vocab_size} not divisible by tp_size={tp_size}, skipping")
            return gm, TransformInfo(skipped=True, num_matches=0)

        chunk = vocab_size // tp_size
        start = tp_rank * chunk
        end = start + chunk

        # Slice to local shard — detach so it's independent of the tied embed_tokens weight
        local_weight = weight_tensor[start:end, :].contiguous().detach()
        shard_attr = "lm_head_weight_vocab_shard"
        gm.register_buffer(shard_attr, local_weight)

        # Insert a new get_attr node for the shard just before the lm_head linear
        with gm.graph.inserting_before(lm_head_node):
            shard_node = gm.graph.get_attr(shard_attr)

        # Rewire: replace the old full-vocab weight with the local shard
        lm_head_node.replace_input_with(weight_node, shard_node)

        # Insert AllGather along dim=-1 (vocab) after the partial logits
        all_gather_op, _ = _get_dist_ops("auto")
        with gm.graph.inserting_after(lm_head_node):
            gather_node = gm.graph.call_function(all_gather_op, args=(lm_head_node, -1))
            lm_head_node.replace_all_uses_with(gather_node)
            # Fix the self-edge introduced by replace_all_uses_with
            gather_node.replace_input_with(gather_node, lm_head_node)

        gm.recompile()

        self._log_info(
            f"shard_lm_head_vocab_parallel: rank={tp_rank}/{tp_size}, "
            f"vocab={vocab_size} → shard=[{start}:{end}] (chunk={chunk}), "
            f"AllGather along dim=-1"
        )

        return gm, TransformInfo(skipped=False, num_matches=1)
