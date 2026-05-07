# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fuse Q/K/V (and similar) linear projections that share one activation.

For models where ``q_proj``/``k_proj``/``v_proj`` are three separate
``nn.Linear`` layers driven by the same hidden state, this transform
concatenates their weights (and biases, when present) along the output
dim into a single GEMM and narrows the result back into the original
per-projection slices. This collapses three kernel launches into one
per attention layer.

Distinct from :class:`FuseGemms` (which only handles bias-less linear
ops) — this transform supports the bias case that GPT-OSS-style
attention uses (``attention_bias=True``).

This module also hosts :class:`FuseAttnOutputViewChain`, which mirrors
the QKV-side fusion on the attention OUTPUT side: it folds the
``cached_attn[B,S,N,D] → reshape([B,S,-1]) → o_proj`` chain into
``cached_attn → view([B,S,-1]) → o_proj`` so the metadata op is a strict
view (no contiguity-check fallback path).
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import delete_all_unused_submodules, eliminate_dead_code
from ...utils.cuda_mem_tracker import cuda_memory_tracker
from ...utils.logger import ad_logger
from ...utils.node_utils import is_linear_op, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry


def _insert_fused_qkv_gemm(
    gm: GraphModule,
    idx: int,
    parent_node: Node,
    linear_nodes: List[Node],
) -> bool:
    """Fuse linear ops sharing one activation, with concatenated bias.

    Mirrors :func:`fusion._insert_fused_gemm` but threads a fused bias
    parameter into the linear call (``args[2]``). All input linears must
    have a non-None bias.
    """
    weight_keys: List[str] = []
    bias_keys: List[str] = []
    for n in linear_nodes:
        w = n.args[1]
        b = n.args[2]
        if w.op != "get_attr" or b is None or b.op != "get_attr":
            ad_logger.debug("Skipping QKV fusion: linear has non-attr weight/bias args.")
            return False
        weight_keys.append(w.target)
        bias_keys.append(b.target)

    weight_params = [gm.get_parameter(k) for k in weight_keys]
    bias_params = [gm.get_parameter(k) for k in bias_keys]
    sizes = [p.size(0) for p in weight_params]

    weight_dtypes = {p.dtype for p in weight_params}
    bias_dtypes = {p.dtype for p in bias_params}
    if len(weight_dtypes) != 1 or len(bias_dtypes) != 1:
        ad_logger.warning(
            f"Skipping QKV fusion for {weight_keys}: mixed dtypes "
            f"weights={weight_dtypes} biases={bias_dtypes}"
        )
        return False
    weight_dtype = weight_dtypes.pop()
    bias_dtype = bias_dtypes.pop()

    # Sanity: each bias length must match its weight's out features.
    for w, b in zip(weight_params, bias_params):
        if b.dim() != 1 or b.size(0) != w.size(0):
            ad_logger.warning(
                f"Skipping QKV fusion: bias shape {tuple(b.shape)} does not match "
                f"weight shape {tuple(w.shape)}"
            )
            return False

    key_w = f"fused_qkv_weight_{idx}"
    key_b = f"fused_qkv_bias_{idx}"
    fused_w = torch.cat(weight_params, dim=0).to(weight_dtype)
    fused_b = torch.cat(bias_params, dim=0).to(bias_dtype)
    setattr(gm, key_w, nn.Parameter(fused_w, requires_grad=False))
    setattr(gm, key_b, nn.Parameter(fused_b, requires_grad=False))

    ad_logger.warning(
        f"Fusing {len(linear_nodes)} QKV GEMMs (weights={weight_keys}, "
        f"biases={bias_keys}) into {key_w}/{key_b} (dtype={weight_dtype})"
    )

    fused_kwargs = dict(linear_nodes[0].kwargs)
    ref_val = linear_nodes[0].meta.get("val")

    with gm.graph.inserting_before(linear_nodes[0]):
        get_w_node = gm.graph.get_attr(key_w, torch.Tensor)
        get_b_node = gm.graph.get_attr(key_b, torch.Tensor)
        fused_linear_node = gm.graph.call_function(
            linear_nodes[0].target,
            args=(parent_node, get_w_node, get_b_node),
            kwargs=fused_kwargs,
        )
        if ref_val is not None:
            fused_out_shape = (*ref_val.shape[:-1], sum(sizes))
            fused_linear_node.meta["val"] = torch.empty(
                fused_out_shape, dtype=ref_val.dtype, device="meta"
            )

    # Use narrow (zero-copy view) to split — downstream view([B, S, N, D])
    # only requires the inner dim to be contiguous, which narrow preserves.
    offset = 0
    for i, n in enumerate(linear_nodes):
        size = sizes[i]
        with gm.graph.inserting_before(n):
            narrow_node = gm.graph.call_function(
                torch.narrow, args=(fused_linear_node, -1, offset, size)
            )
            n_ref = n.meta.get("val")
            if n_ref is not None:
                narrow_node.meta["val"] = torch.empty(n_ref.shape, dtype=n_ref.dtype, device="meta")
        n.replace_all_uses_with(narrow_node)
        offset += size

    eliminate_dead_code(gm)
    delete_all_unused_submodules(gm)
    return True


@TransformRegistry.register("fuse_qkv_proj")
class FuseQKVProj(BaseTransform):
    """Fuse Q/K/V (or any pair+ of) bf16 linear projections sharing one input.

    Operates on bias-carrying ``aten.linear`` / ``torch_linear_simple`` nodes.
    The bias-less case is already handled by :class:`FuseGemms` (which
    requires ``args[2] is None``); this transform covers the GPT-OSS-style
    case where ``attention_bias=True``.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Group bias-carrying linear ops by shared parent activation.
        groups: Dict[Node, List[Node]] = defaultdict(list)
        for node in gm.graph.nodes:
            if not is_linear_op(node):
                continue
            if len(node.args) < 3 or node.args[2] is None:
                continue
            groups[node.args[0]].append(node)

        idx = -1
        num_matches = 0
        with cuda_memory_tracker():
            for parent_node, linear_nodes in groups.items():
                if len(linear_nodes) < 2:
                    continue
                # Sub-group by op target so only same-target bias linears are
                # fused together. Non-linear users of the parent (e.g. sym_size
                # for shape access) are left alone — fusing only removes the
                # original linear nodes; the parent activation still feeds any
                # remaining users.
                by_target: Dict[object, List[Node]] = defaultdict(list)
                for n in linear_nodes:
                    by_target[n.target].append(n)
                for same_target_group in by_target.values():
                    if len(same_target_group) < 2:
                        continue
                    if _insert_fused_qkv_gemm(gm, idx := idx + 1, parent_node, same_target_group):
                        num_matches += 1

        torch.cuda.empty_cache()

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )


# Cached-attention ops whose output flows directly into o_proj via a
# tail reshape. The cached attention kernel returns a contiguous tensor
# (``output.view(*q_shape_og)`` over a freshly-allocated ``torch.zeros``),
# so the reshape is provably a view and can be expressed as ``aten.view``.
_CACHED_ATTN_OUTPUT_OPS = (torch.ops.auto_deploy.trtllm_attention_mha_with_cache,)


@TransformRegistry.register("fuse_attn_output_view_chain")
class FuseAttnOutputViewChain(BaseTransform):
    """Rewrite the tail reshape after cached attention into a strict view.

    The GPT-OSS attention forward emits, per layer:

        attn_out = trtllm_attention_mha_with_cache(q, k, v, ...)  # [B, S, N, D]
        attn_out = aten.reshape(attn_out, [B, S, -1])             # [B, S, N*D]
        out      = o_proj(attn_out)

    The cached attention op returns a contiguous tensor (its underlying
    buffer is a freshly-allocated ``torch.zeros(B*S, N*D, ...)`` viewed
    back to ``q``'s shape). The tail ``aten.reshape`` therefore always
    resolves to a view, never a copy. Replacing it with
    ``aten.view.default`` removes the contiguity-check fallback path
    from the dispatch and makes the no-copy property explicit in the
    graph (analogous to how ``fuse_qkv_proj`` collapses the input-side
    chain into a single fused linear + ``narrow`` view per slice).
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        num_matches = 0
        for attn_node in list(gm.graph.nodes):
            if not is_op(attn_node, _CACHED_ATTN_OUTPUT_OPS):
                continue
            for user in list(attn_node.users):
                if not is_op(user, torch.ops.aten.reshape):
                    continue
                # Skip if reshape input isn't the attention output (defensive).
                if user.args[0] is not attn_node:
                    continue
                with gm.graph.inserting_before(user):
                    view_node = gm.graph.call_function(
                        torch.ops.aten.view.default,
                        args=user.args,
                        kwargs=user.kwargs,
                    )
                ref_val = user.meta.get("val")
                if ref_val is not None:
                    view_node.meta["val"] = torch.empty(
                        ref_val.shape, dtype=ref_val.dtype, device="meta"
                    )
                user.replace_all_uses_with(view_node)
                gm.graph.erase_node(user)
                num_matches += 1

        if num_matches:
            ad_logger.warning(
                f"FuseAttnOutputViewChain: rewrote {num_matches} reshape→view "
                f"after cached attention output."
            )
            eliminate_dead_code(gm)

        return gm, TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
