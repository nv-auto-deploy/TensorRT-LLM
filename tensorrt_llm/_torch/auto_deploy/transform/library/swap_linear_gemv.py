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

"""Swap decode bf16 projection GEMMs to a runtime-M==1-dispatched Triton GEMV.

At batch=1 decode every bf16 ``torch_linear_simple`` projection is a
``[1, K] x [N, K]^T`` GEMV that cuBLAS serves via a split-K kernel plus a
``splitKreduce`` launch. ``auto_deploy::triton_bf16_gemv_linear`` replaces the
pair with one single-pass Triton GEMV (fp32 accumulation) when the flattened
token count is 1 and falls back to ``aten.linear`` otherwise, so prefill and
multi-token decode batches stay bit-identical to the pre-swap graph.

This config-gated transform retargets eligible ``torch_linear_simple`` nodes to
the custom op. It must run after weight sharding and after
``fuse_gemms_mixed_children`` (both consume ``torch_linear_simple`` hints /
nodes): scheduling it at ``post_load_fusion`` from the model config places it
behind the default-config fusion passes. Eligibility is shape-gated via config
so that only weights where the Triton GEMV measurably beats cuBLAS are swapped
(per-rank fused qkvg / o_proj / dense-MLP projections), leaving tiny
projections and the lm_head on cuBLAS.
"""

from typing import Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


class SwapLinearGemvConfig(TransformConfig):
    """Configuration for the linear -> Triton GEMV swap."""

    min_out_features: int = Field(
        default=1024,
        description="Only swap linears with weight dim-0 (out_features) at least this large; "
        "smaller GEMVs cannot fill the SMs single-pass.",
    )
    min_in_features: int = Field(
        default=512,
        description="Only swap linears with weight dim-1 (in_features) at least this large.",
    )
    max_out_features: int = Field(
        default=8192,
        description="Only swap linears with out_features at most this large; very tall weights "
        "(e.g. the vocab-sharded lm_head) see no benefit over cuBLAS split-K.",
    )


@TransformRegistry.register("swap_linear_gemv")
class SwapLinearGemv(BaseTransform):
    """Retarget eligible bf16 ``torch_linear_simple`` nodes to the Triton M==1 GEMV op."""

    config: SwapLinearGemvConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return SwapLinearGemvConfig

    def _eligible_weight(self, gm: GraphModule, weight_node: Node) -> Optional[torch.Tensor]:
        if not isinstance(weight_node, Node) or weight_node.op != "get_attr":
            return None
        try:
            weight = gm.get_parameter(weight_node.target)
        except AttributeError:
            return None
        if (
            weight.dtype == torch.bfloat16
            and weight.ndim == 2
            and weight.is_contiguous()
            and self.config.min_out_features <= weight.shape[0] <= self.config.max_out_features
            and weight.shape[1] >= self.config.min_in_features
            and weight.shape[1] % 128 == 0
        ):
            return weight
        return None

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_linear_simple):
                continue
            if len(node.args) < 2:
                continue
            input_node, weight_node = node.args[0], node.args[1]
            bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
            if bias is not None:
                continue
            weight = self._eligible_weight(gm, weight_node)
            if weight is None:
                continue
            with gm.graph.inserting_before(node):
                new_node = gm.graph.call_function(
                    torch.ops.auto_deploy.triton_bf16_gemv_linear.default,
                    args=(input_node, weight_node),
                )
            new_node.meta.update(node.meta)
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)
            self._log_info(
                f"Swapped linear '{new_node.name}' (weight {tuple(weight.shape)}) to Triton GEMV"
            )
            num_matches += 1

        info = TransformInfo(
            skipped=num_matches == 0,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=True,
        )
        return gm, info
