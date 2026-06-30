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

"""Hoist the static FP8->BF16 weight dequant of grouped FineGrained FP8 linears to load time."""

from typing import Optional, Tuple, Type

import torch
from torch import nn
from torch.fx import GraphModule, Node

from ...custom_ops.quantization.torch_quant import _dequant_block_fp8_weight
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


def _get_attr_tensor(gm: GraphModule, target: str) -> Optional[torch.Tensor]:
    """Fetch a parameter/buffer tensor from ``gm`` by dotted attribute path."""
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, torch.Tensor) else None


@TransformRegistry.register("bake_grouped_finegrained_fp8_weight")
class BakeGroupedFineGrainedFP8Weight(BaseTransform):
    """Pre-materialize the exact BF16 runtime value of each grouped FineGrained FP8 weight.

    ``torch_fake_quant_grouped_finegrained_fp8_linear`` dequantizes its static FP8 weight
    (FP8->BF16 cast + per-block scale expansion + multiply) on *every* call. Those stages
    depend only on the checkpoint weight and per-block scale, never on the activation, so
    this transform runs them once at ``post_load_fusion`` (after sharding + weight load, when
    weights are real and already sliced per rank) and replaces the FP8 weight parameter with
    its dequantized BF16 value. The op detects the non-FP8 weight and skips the per-call
    dequant, leaving the dynamic input quantize-dequantize and grouped BMM bit-for-bit
    identical (verified by the custom-op unit test).

    On DeepSeek-V4-Flash only MLA ``wo_a`` (the output-projection-A grouped linear) emits the
    grouped FineGrained FP8 op, so this matches one weight per decoder layer.
    """

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
        target = torch.ops.auto_deploy.torch_fake_quant_grouped_finegrained_fp8_linear
        cnt = 0
        for node in list(gm.graph.nodes):
            if not is_op(node, target):
                continue

            # args = (input, weight, bias, input_scale[], weight_scale[], input_zp[], weight_zp[])
            weight_node = node.args[1]
            scale_list = node.args[4]
            if not isinstance(weight_node, Node) or weight_node.op != "get_attr":
                continue
            if not isinstance(scale_list, (list, tuple)) or len(scale_list) == 0:
                continue
            scale_node = scale_list[0]
            if not isinstance(scale_node, Node) or scale_node.op != "get_attr":
                continue

            w = _get_attr_tensor(gm, weight_node.target)
            scale = _get_attr_tensor(gm, scale_node.target)
            if w is None or scale is None:
                continue
            if w.dtype != torch.float8_e4m3fn:
                # Already baked (idempotent re-run) or not an FP8 checkpoint weight.
                continue
            if w.dim() != 2 or scale.dim() != 2:
                continue

            out_rows, in_features = w.shape
            scale_n, scale_k = scale.shape
            if scale_n == 0 or scale_k == 0:
                continue
            # Match the per-call block inference in the op exactly (ceil-div).
            block_n = -(-out_rows // scale_n)
            block_k = -(-in_features // scale_k)

            # Identical computation to the op's per-call weight dequant -> bit-for-bit equal.
            w_bf16 = _dequant_block_fp8_weight(w, scale, block_n, block_k, dtype=torch.bfloat16)

            modname, _, attrname = weight_node.target.rpartition(".")
            submod = gm.get_submodule(modname) if modname else gm
            setattr(submod, attrname, nn.Parameter(w_bf16, requires_grad=False))
            weight_node.meta["val"] = w_bf16.detach()
            cnt += 1

        info = TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info
