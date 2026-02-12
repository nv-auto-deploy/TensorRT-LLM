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

"""Transformation for fusing RMSNorm + FP8 quantization.

Matches patterns where flashinfer_rms_norm feeds into trtllm_quant_fp8_linear,
and replaces them with a fused Triton kernel (RMSNorm + FP8 quant) followed by
a GEMM-only op that takes pre-quantized FP8 input.

This eliminates the DRAM round-trip between the normalization and quantization
steps by producing both BF16 (for residual / other consumers) and FP8 (for GEMM)
outputs in a single pass.
"""

import operator
from typing import List, Tuple, Type

import torch
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


def _get_fp8_linear_args(node: Node):
    """Extract arguments from a trtllm_quant_fp8_linear node.

    The op can be called with positional or keyword args:
        op(input, weight_fp8, bias, input_scale=..., weight_scale=...)
    or:
        op(input, weight_fp8, None, input_scale=..., weight_scale=...)

    Returns:
        Tuple of (input, weight_fp8, bias, input_scale, weight_scale) nodes.
    """
    input_arg = node.args[0]
    weight_arg = node.args[1]
    bias_arg = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
    in_scale = node.args[3] if len(node.args) > 3 else node.kwargs.get("input_scale")
    w_scale = node.args[4] if len(node.args) > 4 else node.kwargs.get("weight_scale")
    return input_arg, weight_arg, bias_arg, in_scale, w_scale


def _same_scale_source(a: Node, b: Node) -> bool:
    """Check if two FX nodes reference the same underlying scale value.

    FX tracing may create separate get_attr nodes for the same parameter
    (e.g., self.in_scale accessed 3 times yields in_scale, in_scale_1,
    in_scale_2). These are different Node objects but point to the same
    parameter via their `target` attribute.
    """
    if a is b:
        return True
    # Both are get_attr referencing the same module attribute
    if a.op == "get_attr" and b.op == "get_attr" and a.target == b.target:
        return True
    return False


def _find_fp8_linear_consumers(
    norm_node: Node,
) -> Tuple[List[Node], "Node | None"]:
    """Find trtllm_quant_fp8_linear consumers of a norm node.

    Returns:
        Tuple of (list of fp8_linear consumer nodes, shared input_scale node).
        If consumers have different input_scales, returns ([], None).
    """
    fp8_linear_users = []
    shared_scale = None

    for user in norm_node.users:
        if is_op(user, torch.ops.auto_deploy.trtllm_quant_fp8_linear):
            _, _, _, in_scale, _ = _get_fp8_linear_args(user)
            if shared_scale is None:
                shared_scale = in_scale
            elif not _same_scale_source(in_scale, shared_scale):
                # Different input_scales -- cannot share a single FP8 output
                return [], None
            fp8_linear_users.append(user)

    return fp8_linear_users, shared_scale


def _get_out_dtype_str(norm_node: Node) -> str:
    """Determine the output dtype string from a norm node's metadata."""
    if "val" in norm_node.meta:
        val = norm_node.meta["val"]
        if hasattr(val, "dtype"):
            return str(val.dtype).replace("torch.", "")
    return "bfloat16"


@TransformRegistry.register("fuse_rmsnorm_quant_fp8")
class FuseRMSNormQuantFP8(BaseTransform):
    """Fuse RMSNorm + FP8 quantization into a single Triton kernel.

    Matches:
        norm_out = flashinfer_rms_norm(x, weight, eps)
        linear_out = trtllm_quant_fp8_linear(norm_out, w_fp8, bias, in_scale, w_scale)

    Replaces with:
        bf16_out, fp8_out = triton_rms_norm_quant_fp8(x, weight, eps, in_scale)
        linear_out = trtllm_fp8_gemm(fp8_out, w_fp8, bias, in_scale, w_scale, dtype)
        (other consumers of norm_out use bf16_out)

    Handles multi-consumer case: when norm_out feeds multiple fp8_linear ops
    (e.g., Q, K, V projections), all share the fused FP8 output.
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
            if not is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
                continue

            print(f"GAGAM fuse_rmsnorm_quant_fp8: found flashinfer_rms_norm node: {node.name}")
            for user in node.users:
                print(f"GAGAM   direct user: op={user.op}, name={user.name}, target={user.target}")

            fp8_linear_users, shared_scale = _find_fp8_linear_consumers(node)
            if not fp8_linear_users:
                print(f"GAGAM   -> no fp8_linear consumers found (count={len(fp8_linear_users)})")
                continue

            # Determine output dtype from the norm node metadata
            out_dtype_str = _get_out_dtype_str(node)

            # --- Insert fused RMSNorm + FP8 quant ---
            # triton_rms_norm_quant_fp8(input, weight, eps, scale)
            #   -> (bf16_out, fp8_out)
            #
            # Insert right before the first fp8_linear consumer to ensure
            # all argument nodes (including shared_scale, which may be a
            # get_attr defined after the norm node) are available.
            norm_args = node.args  # (input, weight, eps)
            first_fp8_user = fp8_linear_users[0]
            with graph.inserting_before(first_fp8_user):
                fused_node = graph.call_function(
                    torch.ops.auto_deploy.triton_rms_norm_quant_fp8.default,
                    args=(*norm_args, shared_scale),
                )
                bf16_node = graph.call_function(operator.getitem, args=(fused_node, 0))
                fp8_node = graph.call_function(operator.getitem, args=(fused_node, 1))

            # --- Replace each fp8_linear consumer with fp8_gemm ---
            for fp8_user in fp8_linear_users:
                _, weight_arg, bias_arg, in_scale, w_scale = _get_fp8_linear_args(fp8_user)

                with graph.inserting_after(fp8_user):
                    gemm_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_fp8_gemm.default,
                        args=(fp8_node, weight_arg, bias_arg),
                        kwargs={
                            "input_scale": in_scale,
                            "weight_scale": w_scale,
                            "out_dtype": out_dtype_str,
                        },
                    )
                    fp8_user.replace_all_uses_with(gemm_node)
                graph.erase_node(fp8_user)
                cnt += 1

            # --- Rewire remaining consumers of the original norm node ---
            # At this point, only non-fp8-linear consumers remain
            node.replace_all_uses_with(bf16_node)
            graph.erase_node(node)

        gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
