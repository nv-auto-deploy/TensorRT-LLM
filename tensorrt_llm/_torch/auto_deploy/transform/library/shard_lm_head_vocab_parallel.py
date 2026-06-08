# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Vocab-parallel (column) sharding for an otherwise-replicated ``lm_head``.

Hint-driven sharding (``apply_sharding_hints``) only shards linear nodes that
carry an explicit ``torch_linear_simple`` sharding hint. A plain ``nn.Linear``
``lm_head`` carries no hint, so it is left fully REPLICATED: every TP rank holds
and reads the entire ``[vocab_size, hidden_size]`` weight on every decode step.
For a large vocabulary at high TP this replicated GEMV weight read is a
non-negligible slice of the per-step HBM traffic.

This transform finds the ``lm_head`` linear in the exported graph and applies a
vocab-parallel **column** shard (split the output / vocab dimension across the
TP group) followed by an **all_gather** over the vocab dimension to reconstruct
the full ``[..., vocab_size]`` logits. Each rank then reads only
``vocab_size / world_size`` rows of the weight, cutting the lm_head weight read
~``world_size``x. The result is numerically bit-exact: column sharding +
all_gather is a pure concatenation (no cross-rank reduction).

It reuses the exact machinery the heuristic simple-shard path uses
(``WeightShardingInfo`` with ``split_dim=COLUMN`` + ``dist_op="all_gather"`` ->
``_shard_parameter_node``), so weight tensors, load hooks, the backend-aware
all_gather op, and shape propagation are handled identically. Running this as a
graph transform (after ``torch.export``) — rather than emitting an all_gather in
the model's ``forward`` — is what makes it correct: the weight is sharded first,
so the inserted all_gather's ``register_fake`` re-propagates the gathered shape
back to ``[..., vocab_size]`` instead of baking in a ``vocab_size * world_size``
constant at export time.
"""

from typing import Tuple, Type

from torch.fx import GraphModule

from ...utils._graph import get_lm_head_node
from ...utils.node_utils import LayerType, is_linear_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)
from .sharding import ShardingTransformConfig, SplitDimension, WeightShardingInfo


@TransformRegistry.register("shard_lm_head_vocab_parallel")
class ShardLmHeadVocabParallel(BaseTransform):
    """Column-shard the replicated ``lm_head`` weight across the TP group + all_gather logits."""

    config: ShardingTransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return ShardingTransformConfig

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        local_rank, world_size = shared_config.local_rank, shared_config.world_size

        # No-op for single-device setups (nothing to shard / gather).
        if world_size < 2:
            self._log_info("world_size < 2; lm_head vocab-parallel sharding is a no-op")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Populate the sharding config with the runtime distributed context, mirroring
        # the `detect_sharding` transform's setup so `_shard_parameter_node` sees a fully
        # initialized config (rank / world_size / dist_config / strategies).
        config = self.config
        config.rank = local_rank
        config.world_size = world_size
        if shared_config.dist_config is not None:
            config.dist_config = shared_config.dist_config
        else:
            config._init_mapping()

        # The lm_head is the (unwrapped) linear feeding the graph output.
        lm_head_node = get_lm_head_node(gm)
        if not is_linear_op(lm_head_node):
            self._log_info(
                f"lm_head node '{lm_head_node}' is not a linear op (lm_head likely applied "
                f"outside this graph); skipping"
            )
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Idempotency guard: `_shard_parameter_node` also checks this, but bail early so the
        # match count stays honest if this transform is ever run twice (e.g. per-gm).
        if lm_head_node.meta.get("sharded", False):
            self._log_info(f"lm_head node '{lm_head_node.name}' is already sharded; skipping")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Column split (dim 0 of the weight = vocab dim) + all_gather over the output vocab dim.
        # `from_node` resolves the correct (FP8/FP4/base) WeightShardingInfo subclass; the Step
        # lm_head is bf16 so this is the base class with a no-op quantization callback.
        info = WeightShardingInfo.from_node(
            lm_head_node,
            split_dim=SplitDimension.COLUMN,
            config=config,
            dist_op="all_gather",
            min_local_shape=1,
            layer_type=LayerType.UNKNOWN,
        )
        applied = info.check_and_apply(gm, lm_head_node)
        if not applied:
            self._log_info(
                f"WeightShardingInfo declined to shard lm_head node '{lm_head_node.name}'"
            )
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        self._log_info(
            f"Vocab-parallel column-sharded lm_head node '{lm_head_node.name}' "
            f"across {world_size} TP ranks (+ all_gather over vocab dim)"
        )
        return gm, TransformInfo(
            skipped=False, num_matches=1, is_clean=False, has_valid_shapes=False
        )
