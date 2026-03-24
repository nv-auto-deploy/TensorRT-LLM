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

"""Graph transform to fuse cuda_cached_causal_conv1d + tuned_cached_ssm.

After insert_cached_causal_conv and insert_cached_ssm_attention run, the graph has:

    conv_out = cuda_cached_causal_conv1d_wrapper(conv_input, weight, bias,
                   batch_info_host, cu_seqlen, slot_idx, use_initial_states,
                   conv_state_cache, stride, padding, dilation, groups,
                   padding_mode, "silu")   # conv_out IS conv_input (in-place)
    ...reshape/split/narrow ops...
    x  = conv_out[..., :intermediate_size].view(b, s, nheads, dim)
    B  = conv_out[..., intermediate_size:intermediate_size+ngroups*dstate].view(b,s,ngroups,dstate)
    C  = conv_out[..., ...].view(b, s, ngroups, dstate)
    ssm_out = tuned_cached_ssm(x, A, B, C, D, dt, dt_bias,
                   batch_info_host, cu_seqlen, slot_idx, use_initial_states,
                   any_prefill_use_initial_states_host,
                   chunk_indices, chunk_offsets, seq_idx_prefill,
                   ssm_state_cache, time_step_limit, chunk_size)

This transform replaces the entire chain with:

    ssm_out = fused_cached_conv_ssm(conv_input, weight, bias,
                   A, D, dt, dt_bias,
                   batch_info_host, cu_seqlen, slot_idx, use_initial_states,
                   any_prefill_use_initial_states_host,
                   chunk_indices, chunk_offsets, seq_idx_prefill,
                   conv_state_cache, ssm_state_cache,
                   time_step_limit, chunk_size,
                   intermediate_size, ngroups)
"""

import operator
from typing import List, Tuple

import torch
from torch.fx import GraphModule, Node

from ...custom_ops.mamba.cuda_backend_causal_conv import cuda_cached_causal_conv1d_wrapper
from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

_PASSTHROUGH_OPS = frozenset(
    [
        torch.ops.aten.view.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
    ]
)


def _trace_to_source(node: Node, max_depth: int = 12) -> Node:
    """Trace through shape/copy ops and split getitems to find the originating node."""
    current = node
    for _ in range(max_depth):
        if not isinstance(current, Node):
            break
        if current.op != "call_function":
            break
        target = current.target
        if target in _PASSTHROUGH_OPS:
            current = current.args[0]
        elif target is operator.getitem:
            # getitem(split_result, idx) — trace into the split node
            current = current.args[0]
        elif target in (
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes.default,
            torch.ops.aten.unbind.int,
            torch.ops.aten.narrow.default,
            torch.ops.aten.slice.Tensor,
            torch.ops.aten.select.int,
        ):
            current = current.args[0]
        else:
            break
    return current


def _find_fuse_pairs(
    gm: GraphModule,
) -> List[Tuple[Node, Node, int, int]]:
    """Return list of (conv_node, ssm_node, intermediate_size, ngroups) to fuse."""
    ssm_op = torch.ops.auto_deploy.tuned_cached_ssm.default

    pairs = []
    for node in gm.graph.nodes:
        if not is_op(node, ssm_op):
            continue

        # hidden_states = node.args[0], B = node.args[2], C = node.args[3]
        hidden_states_node = node.args[0]
        B_node = node.args[2]
        C_node = node.args[3]

        hs_source = _trace_to_source(hidden_states_node)
        B_source = _trace_to_source(B_node)
        C_source = _trace_to_source(C_node)

        # All three should originate from the same conv node
        if hs_source is not B_source or hs_source is not C_source:
            continue
        if not is_op(hs_source, cuda_cached_causal_conv1d_wrapper):
            continue

        conv_node = hs_source

        # Extract intermediate_size and ngroups from fake tensor shapes
        try:
            hs_fake = hidden_states_node.meta["val"]
            B_fake = B_node.meta["val"]
        except KeyError:
            continue

        nheads = hs_fake.shape[-2]
        dim_h = hs_fake.shape[-1]
        intermediate_size = nheads * dim_h
        ngroups = B_fake.shape[-2]

        pairs.append((conv_node, node, intermediate_size, ngroups))

    return pairs


@TransformRegistry.register("fuse_conv_ssm")
class FuseConvSSM(BaseTransform):
    """Fuses cuda_cached_causal_conv1d + tuned_cached_ssm into fused_cached_conv_ssm.

    Run this AFTER insert_cached_causal_conv and insert_cached_ssm_attention.
    For decode tokens the fused Triton kernel eliminates the intermediate
    [batch, conv_dim] HBM round-trip.  Prefill falls back to the CUDA path.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        pairs = _find_fuse_pairs(gm)

        fused_op = torch.ops.auto_deploy.fused_cached_conv_ssm.default

        for conv_node, ssm_node, intermediate_size, ngroups in pairs:
            # --- Extract conv node args ---
            # cuda_cached_causal_conv1d_wrapper(input, weight, bias,
            #     batch_info_host, cu_seqlen, slot_idx, use_initial_states,
            #     conv_state_cache, stride, padding, dilation, groups,
            #     padding_mode, activation)
            conv_input = conv_node.args[0]
            conv_weight = conv_node.args[1]
            conv_bias = conv_node.args[2]
            conv_state_cache = conv_node.args[7]

            # --- Extract SSM node args ---
            # tuned_cached_ssm(hidden_states, A, B, C, D, dt, dt_bias,
            #     batch_info_host, cu_seqlen, slot_idx, use_initial_states,
            #     any_prefill_use_initial_states_host,
            #     chunk_indices, chunk_offsets, seq_idx_prefill,
            #     ssm_state_cache, time_step_limit, chunk_size)
            A = ssm_node.args[1]
            D = ssm_node.args[4]
            dt = ssm_node.args[5]
            dt_bias = ssm_node.args[6]
            batch_info_host = ssm_node.args[7]
            cu_seqlen = ssm_node.args[8]
            slot_idx = ssm_node.args[9]
            use_initial_states = ssm_node.args[10]
            any_prefill_use_initial_states_host = ssm_node.args[11]
            chunk_indices = ssm_node.args[12]
            chunk_offsets = ssm_node.args[13]
            seq_idx_prefill = ssm_node.args[14]
            ssm_state_cache = ssm_node.args[15]
            time_step_limit = ssm_node.args[16]
            chunk_size = ssm_node.args[17]

            # --- Insert fused node before ssm_node ---
            with graph.inserting_before(ssm_node):
                fused_node = graph.call_function(
                    fused_op,
                    args=(
                        conv_input,
                        conv_weight,
                        conv_bias,
                        A,
                        D,
                        dt,
                        dt_bias,
                        batch_info_host,
                        cu_seqlen,
                        slot_idx,
                        use_initial_states,
                        any_prefill_use_initial_states_host,
                        chunk_indices,
                        chunk_offsets,
                        seq_idx_prefill,
                        conv_state_cache,
                        ssm_state_cache,
                        time_step_limit,
                        chunk_size,
                        intermediate_size,
                        ngroups,
                    ),
                )

            # Replace SSM node output with fused node output.
            # After this, the reshape/split/conv chain feeding into ssm_node becomes
            # dead code and will be cleaned up by eliminate_dead_code() below.
            ssm_node.replace_all_uses_with(fused_node)
            graph.erase_node(ssm_node)

        graph.eliminate_dead_code()

        return gm, TransformInfo(
            skipped=len(pairs) == 0,
            num_matches=len(pairs),
            is_clean=len(pairs) == 0,
            has_valid_shapes=len(pairs) == 0,
        )
