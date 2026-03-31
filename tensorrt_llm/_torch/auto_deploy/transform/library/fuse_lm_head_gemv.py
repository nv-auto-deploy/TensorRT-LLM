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

"""Graph transform: replace BF16 LM head linear with a custom GEMV.

At decode time the LM head input is a single token ([1, hidden_size]) thanks
to gather_logits_before_lm_head.  A pre-transposed weight stored as
[hidden_size, vocab_size] row-major allows fully-coalesced HBM reads and
improves bandwidth utilization from ~40% to ~70-80% of H100 peak.

This transform:
  1. Finds the LM head linear via get_lm_head_node.
  2. Checks the weight is BF16 (not FP8 — FP8 LM head is unsupported here).
  3. Pre-transposes the weight to [K, N] = [hidden, vocab] contiguous.
  4. Registers the transposed weight as a graph buffer.
  5. Replaces the linear op with auto_deploy::bf16_lm_head_gemv.

run_in_stage: post_load_fusion  (after gather_logits_before_lm_head)
"""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule

from ...custom_ops.linear.bf16_gemv import _bf16_lm_head_gemv  # noqa: F401 (registers op)
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


@TransformRegistry.register("fuse_lm_head_gemv")
class FuseLmHeadGemv(BaseTransform):
    """Replace BF16 LM head linear with a coalesced-access custom GEMV.

    Skipped if the LM head weight is not BF16 (e.g. already FP8).
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
        lm_head_node = get_lm_head_node(gm)

        if not is_linear_op(lm_head_node):
            self._log_info("LM head node is not a linear op; skipping fuse_lm_head_gemv.")
            return gm, TransformInfo(skipped=True, num_matches=0)

        # args: (input, weight, bias?)
        input_nodes = lm_head_node.all_input_nodes
        if len(input_nodes) < 2:
            self._log_info("LM head linear has unexpected arg count; skipping.")
            return gm, TransformInfo(skipped=True, num_matches=0)

        weight_node = input_nodes[1]
        if weight_node.op != "get_attr":
            self._log_info("LM head weight is not a get_attr node; skipping.")
            return gm, TransformInfo(skipped=True, num_matches=0)

        weight_key = weight_node.target
        try:
            weight_tensor = gm.get_parameter(weight_key)
        except AttributeError:
            try:
                weight_tensor = getattr(gm, weight_key)
            except AttributeError:
                self._log_info(f"Cannot find weight {weight_key}; skipping.")
                return gm, TransformInfo(skipped=True, num_matches=0)

        if weight_tensor.dtype != torch.bfloat16:
            self._log_info(
                f"LM head weight dtype is {weight_tensor.dtype}, not bfloat16; skipping."
            )
            return gm, TransformInfo(skipped=True, num_matches=0)

        # weight: [N, K] = [vocab, hidden].  Transpose to [K, N] contiguous.
        weight_T = weight_tensor.T.contiguous().detach()  # [K, N]
        gemv_weight_key = "lm_head_gemv_weight_T"
        gm.register_buffer(gemv_weight_key, weight_T)

        graph = gm.graph
        with graph.inserting_before(lm_head_node):
            weight_T_node = graph.get_attr(gemv_weight_key)
            gemv_node = graph.call_function(
                torch.ops.auto_deploy.bf16_lm_head_gemv.default,
                args=(input_nodes[0], weight_T_node),
            )

        lm_head_node.replace_all_uses_with(gemv_node)
        graph.erase_node(lm_head_node)
        graph.eliminate_dead_code()
        gm.recompile()

        self._log_info(
            f"Replaced LM head linear with bf16_lm_head_gemv "
            f"(weight_T shape: {list(weight_T.shape)})."
        )
        return gm, TransformInfo(skipped=False, num_matches=1)
