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

"""Graph transform swapping the DSV4 router-gate GEMV producer for the Triton op.

Matches the learned-router gate head emitted by the DSV4 modeling code::

    logits = torch_linear_simple(x_fp32, W_fp32(E, H), None)   # cuBLAS gemvx @ M=1
    [logits = aten.to.dtype(logits, fp32)]                     # identity cast (optional)
    sel, w = deepseek_v4_routing[_localized](logits, bias, ...)

and retargets the producer node IN PLACE to
``auto_deploy::deepseek_v4_gate_gemv`` (same signature — args/kwargs, node
order, and every other node are untouched, so downstream stream/PDL transforms
see an identical graph shape). The op runs a one-CTA-per-expert-row Triton
GEMV at single-token decode and falls back to the identical cuBLAS reference
for every other shape (prefill, multi-token decode graphs).

DEFAULT OFF: the Triton GEMV changes the fp32 summation order of the logits
(~1e-6 relative); expert selection can only flip on a rank-k/rank-k+1 near-tie
inside that band (0/10k random tokens in unit tests), but enabling requires a
selection-parity check per deployment. The transform runs at post_load_fusion,
AFTER the pipeline-cache point (stage ``sharding``), so — as with
``fuse_hc_seam_pair`` — the gate needs no factory pipeline-cache identifier.
"""

from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.fused_moe import (
    deepseek_v4_routing as _routing_ops,  # noqa: F401 (registers ops)
)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_CAST_OPS = (
    torch.ops.aten.to,
    torch.ops.aten._to_copy,
)

_ROUTING_OPS = (
    torch.ops.auto_deploy.deepseek_v4_routing,
    torch.ops.auto_deploy.deepseek_v4_routing_localized,
)


def _routing_consumer(node: Node) -> bool:
    """True iff ``node``'s logits feed exactly one DSV4 routing op as ``router_logits``.

    Tolerates one interposed identity fp32 dtype cast (the modeling code's
    ``.float()`` on the already-fp32 GEMV output), which may or may not survive
    export/cleanup.
    """
    users = list(node.users)
    if len(users) != 1:
        return False
    consumer = users[0]
    if is_op(consumer, _CAST_OPS):
        val = consumer.meta.get("val")
        if val is None or val.dtype != torch.float32:
            return False
        cast_users = list(consumer.users)
        if len(cast_users) != 1:
            return False
        consumer = cast_users[0]
    if not is_op(consumer, _ROUTING_OPS):
        return False
    return bool(consumer.args) and consumer.args[0] in (node, users[0])


def _get_attr_tensor(gm: GraphModule, node: Node):
    if not isinstance(node, Node) or node.op != "get_attr":
        return None
    mod_name, _, attr_name = str(node.target).rpartition(".")
    submod = gm.get_submodule(mod_name) if mod_name else gm
    return getattr(submod, attr_name, None)


@TransformRegistry.register("fuse_deepseek_v4_gate_gemv")
class FuseDeepseekV4GateGemv(BaseTransform):
    """Retarget the DSV4 router-gate ``torch_linear_simple`` to the Triton GEMV op."""

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
        num_matches = 0
        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_linear_simple):
                continue
            inp, weight_node, bias = extract_op_args(node, "input", "weight", "bias")
            if bias is not None:
                continue
            weight = _get_attr_tensor(gm, weight_node)
            if weight is None or weight.ndim != 2 or weight.dtype != torch.float32:
                continue
            inp_val = inp.meta.get("val") if isinstance(inp, Node) else None
            if inp_val is None or inp_val.dtype != torch.float32:
                continue
            if not _routing_consumer(node):
                continue

            # Pure in-place target swap: identical signature, args/kwargs kept.
            node.target = torch.ops.auto_deploy.deepseek_v4_gate_gemv.default
            num_matches += 1

        if num_matches:
            ad_logger.info(
                f"fuse_deepseek_v4_gate_gemv: swapped {num_matches} DSV4 router-gate GEMV(s) "
                f"to auto_deploy.deepseek_v4_gate_gemv"
            )

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=(num_matches == 0),
            has_valid_shapes=(num_matches == 0),
        )
        return gm, info
