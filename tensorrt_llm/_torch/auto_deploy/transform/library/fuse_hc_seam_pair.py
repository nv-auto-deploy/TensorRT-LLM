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

"""Graph transform fusing the DeepSeek-V4 HC seam producer/consumer op pair.

Replaces each ``deepseek_v4_hc_post_next_partials`` (A) whose partials output
feeds exactly one ``deepseek_v4_hc_pre_mix_combine_partials[_y32]`` (B) — with
``B.flat`` a view of ``A.out`` and ``B.hc_fn is A.next_hc_fn`` — by the single
barrier-kernel op ``deepseek_v4_hc_post_pre_combine[_y32]``::

    A = hc_post_next_partials(x, res, post, comb, fn)     # -> (out, partials)
    B = hc_pre_mix_combine_partials(partials, out.flatten(2), fn, ...)
                                                          # -> (y, post', comb')
      ==>
    M = hc_post_pre_combine(x, res, post, comb, fn, ...)  # -> (out, y, post', comb')

The head pair (A -> deepseek_v4_hc_head_norm) is intentionally NOT matched.
DEFAULT OFF: the merged kernel measured slower than the pair at decode n=1
(see deepseek_v4_hc_post.py); kept as a gated experiment. Launch-only PDL
gating (AD_HC_PDL) is shared with the pair, so toggling this transform never
changes PDL semantics; it runs post weight-load, after the pipeline-cache
point, so the gate has no cache-identifier impact.
"""

import operator
from typing import Tuple, Type

import torch
from torch.fx import GraphModule, Node

from ...custom_ops import deepseek_v4_hc_post as _hc_post_ops  # noqa: F401 (registers ops)
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import extract_op_args, is_op
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)

_VIEW_OPS = (
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.contiguous.default,
)


def _resolves_to(node: Node, producer: Node, max_hops: int = 4) -> bool:
    """True if ``node`` is ``producer`` or a short view-only chain over it."""
    cur = node
    for _ in range(max_hops + 1):
        if cur is producer:
            return True
        if not isinstance(cur, Node) or cur.op != "call_function" or cur.target not in _VIEW_OPS:
            return False
        cur = cur.args[0]
    return False


@TransformRegistry.register("fuse_hc_seam_pair")
class FuseHcSeamPair(BaseTransform):
    """Fuse adjacent HC seam producer/consumer custom ops into the pair op."""

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
            if not is_op(node, torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials):
                continue

            out_gi = parts_gi = None
            for u in node.users:
                if u.op == "call_function" and u.target == operator.getitem:
                    if u.args[1] == 0:
                        out_gi = u
                    elif u.args[1] == 1:
                        parts_gi = u
            if out_gi is None or parts_gi is None:
                continue

            # The partials must feed exactly one consumer: the combine op.
            parts_users = list(parts_gi.users)
            if len(parts_users) != 1:
                continue
            combine = parts_users[0]
            if is_op(combine, torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials):
                merged_op = torch.ops.auto_deploy.deepseek_v4_hc_post_pre_combine.default
            elif is_op(combine, torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32):
                merged_op = torch.ops.auto_deploy.deepseek_v4_hc_post_pre_combine_y32.default
            else:
                continue

            a_x, a_res, a_post, a_comb, a_fn = extract_op_args(
                node, "x", "residual", "post", "comb", "next_hc_fn"
            )
            (
                b_partials,
                b_flat,
                b_fn,
                b_scale,
                b_base,
                b_weight,
                b_hc_mult,
                b_sinkhorn,
                b_eps,
                b_norm_eps,
                b_rms_eps,
                b_out_dtype,
            ) = extract_op_args(
                combine,
                "partials",
                "flat",
                "hc_fn",
                "hc_scale",
                "hc_base",
                "norm_weight",
                "hc_mult",
                "sinkhorn_iters",
                "eps",
                "norm_eps",
                "rms_eps",
                "out_dtype",
            )
            # The consumer must read the producer's partials against the SAME
            # hc_fn table and combine the producer's own residual stream.
            if b_partials is not parts_gi or b_fn is not a_fn:
                continue
            if not _resolves_to(b_flat, out_gi):
                continue

            # Collect the consumer's getitems (bail on any direct tuple use).
            gi_map = {}
            direct_use = False
            for u in list(combine.users):
                if u.op == "call_function" and u.target == operator.getitem:
                    gi_map[u.args[1]] = u
                else:
                    direct_use = True
            if direct_use:
                continue

            with graph.inserting_before(combine):
                merged = graph.call_function(
                    merged_op,
                    args=(
                        a_x,
                        a_res,
                        a_post,
                        a_comb,
                        a_fn,
                        b_scale,
                        b_base,
                        b_weight,
                        b_hc_mult,
                        b_sinkhorn,
                        b_eps,
                        b_norm_eps,
                        b_rms_eps,
                        b_out_dtype,
                    ),
                )
                merged.meta = combine.meta.copy()
                out_val = out_gi.meta.get("val")
                b_val = combine.meta.get("val")
                if out_val is not None and b_val is not None:
                    merged.meta["val"] = (out_val, *tuple(b_val))
                m_out = graph.call_function(operator.getitem, (merged, 0))
                m_out.meta = out_gi.meta.copy()
                repl = []
                for idx, gi in gi_map.items():
                    m_gi = graph.call_function(operator.getitem, (merged, idx + 1))
                    m_gi.meta = gi.meta.copy()
                    repl.append((gi, m_gi))

            for gi, m_gi in repl:
                gi.replace_all_uses_with(m_gi)
                graph.erase_node(gi)
            graph.erase_node(combine)
            # out keeps its non-consumer users (the next seam's residual input
            # and the — now dead — view chain that fed the consumer).
            out_gi.replace_all_uses_with(m_out)
            graph.erase_node(parts_gi)
            graph.erase_node(out_gi)
            graph.erase_node(node)
            cnt += 1

        info = TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info
