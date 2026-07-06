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

A second pass fuses the head-wise attention gate into the swapped o_proj GEMV:
``triton_bf16_gemv_linear(reshape(step3p7_head_gate(attn, gate)), w)`` becomes
``triton_bf16_gemv_head_gate_linear(attn, gate, w)``, applying the sigmoid gate
as an in-kernel prologue while the GEMV streams its activation input. This
drops one launch-bound elementwise kernel and one ``[1, K]`` bf16 round trip
per attention layer at batch=1 decode, bit-identical to the unfused chain.
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

    def _fuse_head_gate_prologue(self, gm: GraphModule) -> int:
        """Fuse ``step3p7_head_gate`` into the swapped GEMV as an in-kernel prologue.

        Matches ``triton_bf16_gemv_linear(reshape(step3p7_head_gate(attn, gate)), w)``
        where the reshape is a flatten of the trailing ``[H, D]`` head dims (the
        sharding-lowered ``auto_deploy.view`` between the gate and o_proj) and both
        intermediates are single-use, then retargets the GEMV to
        ``triton_bf16_gemv_head_gate_linear(attn, gate, w)``. The fused op takes the
        pre-gate attention output directly, so the head_gate and reshape nodes die.
        """
        num_fused = 0
        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.triton_bf16_gemv_linear):
                continue
            reshape_node, weight_node = node.args[0], node.args[1]
            if not is_op(reshape_node, torch.ops.aten.reshape) or len(reshape_node.users) != 1:
                continue
            hg_node = reshape_node.args[0]
            if not is_op(hg_node, torch.ops.auto_deploy.step3p7_head_gate):
                continue
            if len(hg_node.users) != 1 or len(hg_node.args) < 2:
                continue
            attn_node, gate_node = hg_node.args[0], hg_node.args[1]
            weight = self._eligible_weight(gm, weight_node)
            if weight is None:
                continue
            hg_val = hg_node.meta.get("val")
            rs_val = reshape_node.meta.get("val")
            if hg_val is None or rs_val is None or hg_val.dim() != rs_val.dim() + 1:
                continue
            num_heads, head_dim = int(hg_val.shape[-2]), int(hg_val.shape[-1])
            # The fused kernel indexes gates as k // head_dim; require pow2 head_dim and
            # an exact flatten of the head dims into the GEMV reduction dim (leading
            # dims unchanged, compared symbolically so no shape guards are introduced).
            if head_dim & (head_dim - 1) != 0 or num_heads * head_dim != weight.shape[1]:
                continue
            if [str(s) for s in hg_val.shape[:-2]] != [str(s) for s in rs_val.shape[:-1]]:
                continue
            with gm.graph.inserting_before(node):
                fused_node = gm.graph.call_function(
                    torch.ops.auto_deploy.triton_bf16_gemv_head_gate_linear.default,
                    args=(attn_node, gate_node, weight_node),
                )
            fused_node.meta.update(node.meta)
            node.replace_all_uses_with(fused_node)
            gm.graph.erase_node(node)
            gm.graph.erase_node(reshape_node)
            gm.graph.erase_node(hg_node)
            self._log_info(
                f"Fused head-gate prologue into GEMV '{fused_node.name}' "
                f"(weight {tuple(weight.shape)}, {num_heads} heads x {head_dim})"
            )
            num_fused += 1
        return num_fused

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

        num_matches += self._fuse_head_gate_prologue(gm)

        info = TransformInfo(
            skipped=num_matches == 0,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=True,
        )
        return gm, info
