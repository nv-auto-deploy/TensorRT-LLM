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

"""Graph transform: fuse gated-RMSNorm + FP8 quantization for Mamba out_proj.

In Nemotron-H Mamba2 layers, the decode path produces:

    y_norm = triton_rmsnorm_gated(y, norm_weight, gate, eps, group_size, norm_before_gate)
    [optional view/cast ops]
    out = trtllm_quant_fp8_linear(y_norm, w_fp8, bias, in_scale, w_scale)

where trtllm_quant_fp8_linear internally runs scaleMatrixPerTensorVec (FP8 quant) + GEMM.

This transform replaces the chain with:

    y_fp8 = gated_rms_norm_quant_fp8(y, gate, norm_weight, in_scale, eps, group_size, nbg)
    [re-apply view ops if any]
    out = trtllm_fp8_prequant_linear(y_fp8, w_fp8, bias, in_scale, w_scale)

Eliminates one kernel per Mamba layer (the scaleMatrixPerTensorVec inside trtllm_quant_fp8_linear),
and avoids writing/reading the intermediate BF16 tensor between norm and quantization.

Run after fuse_fp8_linear (post_load_fusion stage).
"""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.node_utils import extract_op_args
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

# FP8 linear ops that may precede gated_rmsnorm in the Mamba path.
_FP8_LINEAR_OPS = frozenset(
    [
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        torch.ops.auto_deploy.torch_quant_fp8_linear.default,
    ]
)

# View/reshape/cast ops that may appear between gated_rmsnorm and the linear.
_VIEW_OPS = frozenset(
    [
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
        # .to(dtype) when dtype == tensor.dtype may appear as a no-op cast
        torch.ops.aten.to.dtype,
        # torch.export normalizes dtype casts to prims.convert_element_type or aten._to_copy
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
    ]
)

# Dtype-changing cast ops that must NOT be re-applied after fusion.
# In the original graph these are BF16→BF16 no-ops; after fusion the fused
# op outputs FP8, so re-applying them would cast FP8→BF16 and break the
# dtype contract of trtllm_fp8_prequant_linear.
_DTYPE_CAST_OPS = frozenset(
    [
        torch.ops.aten.to.dtype,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten._to_copy.default,
    ]
)


def _trace_through_views(node: Node, max_depth: int = 8) -> Node:
    """Trace backward through view/reshape/cast ops."""
    cur = node
    for _ in range(max_depth):
        if not isinstance(cur, Node) or cur.op != "call_function":
            break
        if cur.target in _VIEW_OPS:
            cur = cur.args[0]
        else:
            break
    return cur


def _is_gated_rmsnorm(node: Node) -> bool:
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is torch.ops.auto_deploy.triton_rmsnorm_gated.default
    )


def _only_used_by(node: Node, consumer: Node) -> bool:
    """True if node (and all its view-chain users) are only consumed by consumer."""
    for u in node.users.keys():
        if u is consumer:
            continue
        if isinstance(u, Node) and u.op == "call_function" and u.target in _VIEW_OPS:
            if not _only_used_by(u, consumer):
                return False
        else:
            return False
    return True


@TransformRegistry.register("fuse_gated_rmsnorm_quant_fp8")
class FuseGatedRMSNormQuantFP8(BaseTransform):
    """Fuse gated-RMSNorm + FP8 quantization into a single Triton kernel.

    Matches ``trtllm_quant_fp8_linear(triton_rmsnorm_gated(y, w, gate, ...), ...)``
    and replaces with
    ``trtllm_fp8_prequant_linear(gated_rms_norm_quant_fp8(y, gate, w, scale, ...), ...)``.

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
        graph = gm.graph
        cnt = 0

        for node in list(graph.nodes):
            if node.target not in _FP8_LINEAR_OPS:
                continue

            # Extract args: (input, weight_fp8, bias, input_scale, weight_scale)
            args = extract_op_args(
                node, "input", "weight_fp8", "bias", "input_scale", "weight_scale"
            )
            inp, w_fp8, bias, in_scale, w_scale = args
            if not isinstance(inp, Node) or not isinstance(in_scale, Node):
                continue

            # Trace through views to find the activation source
            act_node = _trace_through_views(inp)

            # Must be triton_rmsnorm_gated
            if not _is_gated_rmsnorm(act_node):
                continue

            # Verify exclusivity: gated_rmsnorm result only consumed by this linear
            if act_node is not inp:
                if not _only_used_by(act_node, inp if act_node is not inp else node):
                    continue
            if not _only_used_by(inp, node):
                continue

            # Extract gated_rmsnorm args:
            #   triton_rmsnorm_gated(x, weight, gate, eps, group_size, norm_before_gate=False)
            rm_args = act_node.args
            if len(rm_args) < 5:
                continue
            rm_x = rm_args[0]  # input (SSM output)
            rm_w = rm_args[1]  # norm weight
            rm_gate = rm_args[2]  # gate (may be None node)
            rm_eps = rm_args[3]  # eps (float constant)
            rm_gs = rm_args[4]  # group_size (int constant)
            rm_nbg = rm_args[5] if len(rm_args) > 5 else False  # norm_before_gate

            if not isinstance(rm_x, Node):
                continue

            # Insert fused op before the fp8_linear
            with graph.inserting_before(node):
                # gated_rms_norm_quant_fp8(x, gate, weight, in_scale, eps, group_size, nbg)
                fp8_act = graph.call_function(
                    torch.ops.auto_deploy.gated_rms_norm_quant_fp8.default,
                    args=(rm_x, rm_gate, rm_w, in_scale, rm_eps, rm_gs, rm_nbg),
                )

                # Re-apply shape-only view ops between act_node and inp.
                # Skip dtype cast ops: in the original graph they were BF16→BF16
                # no-ops; after fusion the fused kernel outputs FP8, so
                # re-applying them would cast FP8→BF16 and crash
                # trtllm_fp8_prequant_linear.
                fp8_input = fp8_act
                if act_node is not inp:
                    chain = []
                    cur = inp
                    while cur is not act_node:
                        chain.append(cur)
                        cur = cur.args[0]
                    for view_node in reversed(chain):
                        if view_node.target in _DTYPE_CAST_OPS:
                            continue
                        fp8_input = graph.call_function(
                            view_node.target,
                            args=(fp8_input, *view_node.args[1:]),
                            kwargs=view_node.kwargs,
                        )

                new_gemm = graph.call_function(
                    torch.ops.auto_deploy.trtllm_fp8_prequant_linear.default,
                    args=(fp8_input, w_fp8, bias),
                    kwargs={
                        "input_scale": in_scale,
                        "weight_scale": w_scale,
                        "out_dtype": "bfloat16",
                    },
                )

            node.replace_all_uses_with(new_gemm)
            graph.erase_node(node)
            cnt += 1

        if cnt > 0:
            eliminate_dead_code(gm)
            gm.recompile()

        return gm, TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
