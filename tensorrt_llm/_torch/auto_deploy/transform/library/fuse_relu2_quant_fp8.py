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

"""Graph transform: fuse ReLU² activation + FP8 quantization.

After fuse_fp8_linear runs, the shared expert MLP in Nemotron-style models
has the pattern:

    relu_out = aten.relu(up_proj_out)
    sq_out   = aten.pow(relu_out, 2)           # or aten.square / aten.mul
    [reshape/view ops ...]
    out = trtllm_quant_fp8_linear(sq_out, weight, bias, in_scale, w_scale)

This transform replaces the chain with:

    fp8_act = auto_deploy::relu2_quant_fp8(up_proj_out, in_scale)
    [reshape/view ops ...]
    out = trtllm_fp8_prequant_linear(fp8_act, weight, bias,
                                     input_scale=in_scale, weight_scale=w_scale,
                                     out_dtype="bfloat16")

Eliminates three separate GPU kernels (relu, pow, scaleMatrixPerTensorVec)
and replaces them with one Triton kernel.
"""

from typing import Optional, Tuple, Type

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

# Ops that fuse_fp8_linear may produce; we handle both backends.
_FP8_LINEAR_OPS = frozenset(
    [
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        torch.ops.auto_deploy.torch_quant_fp8_linear.default,
    ]
)

# relu2 square patterns (all three are equivalent after torch.export)
_RELU2_SQUARE_OPS = frozenset(
    [
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.square.default,
    ]
)

# Relu ops (plain and in-place variants)
_RELU_OPS = frozenset(
    [
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.clamp_min.default,
    ]
)

# Shape-only passthrough ops we can trace through
_VIEW_OPS = frozenset(
    [
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.clone.default,
    ]
)


def _trace_through_views(node: Node, max_depth: int = 8) -> Node:
    """Trace backwards through view/reshape ops."""
    cur = node
    for _ in range(max_depth):
        if not isinstance(cur, Node) or cur.op != "call_function":
            break
        if cur.target in _VIEW_OPS:
            cur = cur.args[0]
        else:
            break
    return cur


def _is_fp8_linear(node: Node) -> bool:
    return isinstance(node, Node) and node.target in _FP8_LINEAR_OPS


def _get_out_dtype_str(fp8_linear_node: Node) -> str:
    """Extract output dtype string from the fp8_linear node's fake tensor."""
    val = fp8_linear_node.meta.get("val")
    if hasattr(val, "dtype") and val.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return str(val.dtype).replace("torch.", "")
    return "bfloat16"


def _match_relu2_pattern(
    sq_node: Node,
) -> Optional[Node]:
    """Return the relu input x if node matches relu2(x), else None.

    Handles:
      - pow(relu(x), 2)  / pow(relu(x), 2.0)
      - square(relu(x))
      - mul(relu(x), relu(x))  [same relu node used twice]
    """
    if not isinstance(sq_node, Node) or sq_node.op != "call_function":
        return None

    target = sq_node.target

    if target in _RELU2_SQUARE_OPS:
        # pow(relu_out, 2) or square(relu_out)
        relu_node = sq_node.args[0]
        if target is torch.ops.aten.pow.Tensor_Scalar:
            exp = sq_node.args[1]
            if exp not in (2, 2.0):
                return None
        if not isinstance(relu_node, Node) or relu_node.target not in _RELU_OPS:
            return None
        return relu_node.args[0]  # x

    if target is torch.ops.aten.mul.Tensor:
        # mul(relu(x), relu(x)) — both operands must be the same relu node
        lhs, rhs = sq_node.args[0], sq_node.args[1]
        if lhs is not rhs:
            return None
        relu_node = lhs
        if not isinstance(relu_node, Node) or relu_node.target not in _RELU_OPS:
            return None
        return relu_node.args[0]

    return None


def _only_used_by(node: Node, consumer: Node) -> bool:
    """True if node (and all its view-users) are only consumed by consumer."""
    users = list(node.users.keys())
    # Allow intermediate view nodes if they also end up at consumer
    for u in users:
        if u is consumer:
            continue
        # Allow passthrough (view/reshape) if their only terminal is consumer
        if isinstance(u, Node) and u.op == "call_function" and u.target in _VIEW_OPS:
            if not _only_used_by(u, consumer):
                return False
        else:
            return False
    return True


@TransformRegistry.register("fuse_relu2_quant_fp8")
class FuseRelu2QuantFP8(BaseTransform):
    """Fuse relu2 + FP8 quantization into a single Triton kernel.

    Matches ``trtllm_quant_fp8_linear(pow(relu(x), 2), ...)`` and replaces
    with ``trtllm_fp8_prequant_linear(relu2_quant_fp8(x, scale), ...)``.

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

        # Snapshot nodes to avoid mutation-during-iteration issues
        for node in list(graph.nodes):
            if not _is_fp8_linear(node):
                continue

            # Extract args: (input, weight_fp8, bias, input_scale, weight_scale)
            args = extract_op_args(
                node, "input", "weight_fp8", "bias", "input_scale", "weight_scale"
            )
            inp, weight, bias, in_scale, w_scale = args
            if not isinstance(inp, Node) or not isinstance(in_scale, Node):
                continue

            # Trace through views to find the activation source
            act_node = _trace_through_views(inp)

            # Match relu2 pattern
            x_node = _match_relu2_pattern(act_node)
            if x_node is None:
                continue

            # Verify relu2 result is only consumed by this fp8_linear
            # (possibly via view/reshape intermediates that we just traced)
            if act_node is not inp:
                # there are view nodes between act_node and node; check act_node exclusivity
                if not _only_used_by(act_node, inp if act_node is not inp else node):
                    continue
            if not _only_used_by(inp, node):
                continue

            # Get output dtype string
            out_dtype_str = _get_out_dtype_str(node)

            # Insert fused op before the fp8_linear
            with graph.inserting_before(node):
                fp8_act = graph.call_function(
                    torch.ops.auto_deploy.relu2_quant_fp8.default,
                    args=(x_node, in_scale),
                )
                # Re-apply any view/reshape ops that were between act_node and inp
                fp8_input = fp8_act
                if act_node is not inp:
                    # Re-trace forward: collect view chain from act_node → inp
                    chain = []
                    cur = inp
                    while cur is not act_node:
                        chain.append(cur)
                        cur = cur.args[0]
                    for view_node in reversed(chain):
                        fp8_input = graph.call_function(
                            view_node.target,
                            args=(fp8_input, *view_node.args[1:]),
                            kwargs=view_node.kwargs,
                        )

                new_gemm = graph.call_function(
                    torch.ops.auto_deploy.trtllm_fp8_prequant_linear.default,
                    args=(fp8_input, weight, bias),
                    kwargs={
                        "input_scale": in_scale,
                        "weight_scale": w_scale,
                        "out_dtype": out_dtype_str,
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
