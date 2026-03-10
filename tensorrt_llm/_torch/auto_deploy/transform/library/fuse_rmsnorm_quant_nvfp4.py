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

"""Direct FX fusion for RMSNorm (+ optional Add) + NVFP4 linear."""

import operator
from typing import Optional, Tuple, Type

import torch
from pydantic import Field
from torch.fx import GraphModule, Node

from ...custom_ops.normalization.flashinfer_fused_add_rms_norm import flashinfer_fused_add_rms_norm
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import eliminate_dead_code
from ...utils.node_utils import extract_op_args, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_SUPPORTED_OUT_DTYPES = {torch.float16, torch.bfloat16}


class MatchRMSNormQuantNVFP4PatternConfig(TransformConfig):
    """Compatibility config for the no-op match stage."""

    requires_shape_prop: bool = Field(
        default=True,
        description="Kept for compatibility; direct FX fusion uses shape metadata when available.",
    )


@torch.library.custom_op("auto_deploy::torch_fused_add_rmsnorm_quant_nvfp4_linear", mutates_args=())
def torch_fused_add_rmsnorm_quant_nvfp4_linear(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compatibility canonical op for add-aware RMSNorm + NVFP4 linear."""
    del input_scale, weight_scale, alpha
    add_out = torch.ops.aten.add.Tensor(input, residual)
    norm_out = torch.ops.auto_deploy.torch_rmsnorm(add_out, norm_weight, eps)
    return torch.ops.aten.linear(norm_out, weight_fp4.repeat(1, 2).to(norm_out.dtype), bias)


@torch_fused_add_rmsnorm_quant_nvfp4_linear.register_fake
def _torch_fused_add_rmsnorm_quant_nvfp4_linear_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del residual, norm_weight, eps, input_scale, weight_scale, alpha
    out_shape = (*input.shape[:-1], weight_fp4.shape[0])
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


@torch.library.custom_op("auto_deploy::torch_rmsnorm_quant_nvfp4_linear", mutates_args=())
def torch_rmsnorm_quant_nvfp4_linear(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compatibility canonical op for direct RMSNorm + NVFP4 linear."""
    del input_scale, weight_scale, alpha
    norm_out = torch.ops.auto_deploy.torch_rmsnorm(input, norm_weight, eps)
    return torch.ops.aten.linear(norm_out, weight_fp4.repeat(1, 2).to(norm_out.dtype), bias)


@torch_rmsnorm_quant_nvfp4_linear.register_fake
def _torch_rmsnorm_quant_nvfp4_linear_fake(
    input: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    del norm_weight, eps, input_scale, weight_scale, alpha
    out_shape = (*input.shape[:-1], weight_fp4.shape[0])
    return torch.empty(out_shape, dtype=input.dtype, device=input.device)


def _get_meta_dtype(node: Optional[Node]) -> Optional[torch.dtype]:
    if node is None or "val" not in node.meta:
        return None

    val = node.meta["val"]
    if hasattr(val, "dtype"):
        return val.dtype

    return None


def _get_expected_out_dtype(*nodes: Optional[Node]) -> torch.dtype:
    for node in nodes:
        dtype = _get_meta_dtype(node)
        if dtype in _SUPPORTED_OUT_DTYPES:
            return dtype

    return torch.bfloat16


def _is_view_like(node: Node) -> bool:
    return is_op(node, torch.ops.aten.view.default) or is_op(node, torch.ops.aten.reshape.default)


def _unwrap_post_norm_nodes(node):
    current = node
    post_nodes: list[Node] = []
    while isinstance(current, Node) and _is_view_like(current):
        post_nodes.append(current)
        current = current.args[0]
    return current, post_nodes


def _packed_shape(shape_arg):
    if not isinstance(shape_arg, (list, tuple)):
        return shape_arg

    packed = list(shape_arg)
    if packed and isinstance(packed[-1], int) and packed[-1] > 0:
        packed[-1] = packed[-1] // 2

    return tuple(packed) if isinstance(shape_arg, tuple) else packed


def _reapply_post_norm_nodes(graph, fp4_node: Node, post_nodes: list[Node]) -> Node:
    current = fp4_node
    for post_node in reversed(post_nodes):
        current = graph.call_function(
            post_node.target,
            args=(current, _packed_shape(post_node.args[1])),
            kwargs=post_node.kwargs,
        )
    return current


def _extract_nvfp4_linear_args(node: Node):
    if is_op(node, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear):
        (
            input_arg,
            weight_arg,
            bias_arg,
            input_scale_args,
            weight_scale_args,
            _input_zp,
            _weight_zp,
        ) = extract_op_args(
            node,
            "input",
            "weight_quantized",
            "bias",
            "input_scale",
            "weight_scale",
            "input_zp",
            "weight_zp",
        )

        input_scale_arg = input_scale_args[0] if input_scale_args else None
        weight_scale_arg = weight_scale_args[0] if weight_scale_args else None
        alpha_arg = weight_scale_args[1] if len(weight_scale_args) > 1 else None
        return input_arg, weight_arg, bias_arg, input_scale_arg, weight_scale_arg, alpha_arg

    if is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear):
        (
            input_arg,
            weight_arg,
            bias_arg,
            input_scale_arg,
            weight_scale_arg,
            alpha_arg,
        ) = extract_op_args(
            node,
            "input",
            "weight_fp4",
            "bias",
            "input_scale",
            "weight_scale",
            "alpha",
        )
        return input_arg, weight_arg, bias_arg, input_scale_arg, weight_scale_arg, alpha_arg

    raise ValueError(f"Unsupported NVFP4 linear node target: {node.target}")


def _match_add_aware_source(node: Node):
    if is_op(node, operator.getitem):
        source = node.args[0]
        if (
            isinstance(source, Node)
            and node.args[1] == 0
            and source.target is flashinfer_fused_add_rms_norm
        ):
            return {
                "kind": "fused_add",
                "x": source.args[0],
                "residual": source.args[1],
                "weight": source.args[2],
                "eps": source.args[3],
            }

    if is_op(node, torch.ops.auto_deploy.torch_rmsnorm):
        norm_input = node.args[0]
        if isinstance(norm_input, Node) and is_op(norm_input, torch.ops.aten.to.dtype):
            norm_input = norm_input.args[0]

        if isinstance(norm_input, Node) and is_op(norm_input, torch.ops.aten.add.Tensor):
            return {
                "kind": "raw_add",
                "x": norm_input.args[0],
                "residual": norm_input.args[1],
                "weight": node.args[1],
                "eps": node.args[2],
            }

    return None


def _match_direct_rmsnorm_source(node: Node):
    if not is_op(node, torch.ops.auto_deploy.torch_rmsnorm):
        return None

    norm_input = node.args[0]
    if isinstance(norm_input, Node) and is_op(norm_input, torch.ops.aten.to.dtype):
        norm_input = norm_input.args[0]

    if isinstance(norm_input, Node) and is_op(norm_input, torch.ops.aten.add.Tensor):
        return None

    return {
        "input": norm_input,
        "weight": node.args[1],
        "eps": node.args[2],
    }


@TransformRegistry.register("match_rmsnorm_quant_nvfp4_pattern")
class MatchRMSNormQuantNVFP4Pattern(BaseTransform):
    """Compatibility no-op; direct FX fusion happens later."""

    config: TransformConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return MatchRMSNormQuantNVFP4PatternConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        info = TransformInfo(
            skipped=True,
            num_matches=0,
            is_clean=True,
            has_valid_shapes=True,
        )
        return gm, info


@TransformRegistry.register("fuse_rmsnorm_quant_nvfp4")
class FuseRMSNormQuantNVFP4(BaseTransform):
    """Fuse add+rms+quant first; otherwise fall back to rms+quant."""

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
            if not (
                is_op(node, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear)
                or is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_linear)
            ):
                continue

            (
                fake_quant_input,
                weight_arg,
                bias_arg,
                input_scale_arg,
                weight_scale_arg,
                alpha_arg,
            ) = _extract_nvfp4_linear_args(node)
            if input_scale_arg is None or weight_scale_arg is None or alpha_arg is None:
                continue

            source_node, post_nodes = _unwrap_post_norm_nodes(fake_quant_input)
            add_match = _match_add_aware_source(source_node)
            direct_match = (
                None if add_match is not None else _match_direct_rmsnorm_source(source_node)
            )
            if add_match is None and direct_match is None:
                continue

            out_dtype = _get_expected_out_dtype(node, source_node)

            with graph.inserting_before(node):
                if add_match is not None:
                    fused_quant_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_fused_add_rms_norm_quant_nvfp4.default,
                        args=(
                            add_match["x"],
                            add_match["residual"],
                            add_match["weight"],
                            add_match["eps"],
                            input_scale_arg,
                        ),
                    )
                    fp4_node = graph.call_function(operator.getitem, args=(fused_quant_node, 1))
                    sf_node = graph.call_function(operator.getitem, args=(fused_quant_node, 2))
                else:
                    fused_quant_node = graph.call_function(
                        torch.ops.auto_deploy.trtllm_rms_norm_quant_nvfp4.default,
                        args=(
                            direct_match["input"],
                            direct_match["weight"],
                            direct_match["eps"],
                            input_scale_arg,
                        ),
                    )
                    fp4_node = graph.call_function(operator.getitem, args=(fused_quant_node, 1))
                    sf_node = graph.call_function(operator.getitem, args=(fused_quant_node, 2))

                fp4_input = _reapply_post_norm_nodes(graph, fp4_node, post_nodes)
                gemm_node = graph.call_function(
                    torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default,
                    args=(fp4_input, weight_arg, sf_node, weight_scale_arg, alpha_arg),
                    kwargs={
                        "bias": bias_arg,
                        "out_dtype": out_dtype,
                    },
                )

            node.replace_all_uses_with(gemm_node)
            graph.erase_node(node)
            cnt += 1

        if cnt > 0:
            eliminate_dead_code(gm)
        gm.recompile()

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
