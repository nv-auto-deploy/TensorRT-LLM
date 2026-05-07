# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fuse the model's final RMSNorm with the lm_head GEMM.

Models typically end with::

    hidden = rmsnorm(hidden, norm_weight, eps)  # last per-token normalization
    logits = linear(hidden, lm_head_weight)  # tied or separate lm_head

The two ops are run back-to-back by a single token-output path, so the
``rmsnorm`` write and the ``lm_head`` read both touch the same hidden-state
buffer.  Folding the per-channel ``norm_weight`` into ``lm_head_weight`` at
load time lets us drop the explicit ``mul`` step inside RMSNorm: only the
``x * rstd`` scaling needs to remain at runtime, and that result is consumed
directly by the lm_head GEMM.

Algebraically::

    rmsnorm(x) @ W_lm.T
        = (x * rstd(x) * w_norm) @ W_lm.T
        = (x * rstd(x))         @ (W_lm * w_norm[None, :]).T

So we can pre-scale ``W_lm`` once by ``w_norm`` and replace the runtime
RMSNorm with a "weight-less" RMSNorm that only computes ``x * rstd``.  The
fused custom op below packages this as one node so the ``flashinfer_rms_norm``
+ ``aten.linear`` pair collapses into a single graph node.

The fused op is implemented by reusing the optimized ``flashinfer_rms_norm``
kernel with a unit weight buffer — the per-channel multiply is folded into
``W_lm`` once at transform time, so the runtime kernel just performs the
rstd compute and a unit-weight scale.  The unit weight is baked as a
GraphModule buffer so cudagraph capture sees a stable static tensor and
the fused op signature does not depend on a runtime dictionary cache.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils._graph import get_lm_head_node
from ...utils.logger import ad_logger
from ...utils.node_utils import is_linear_op, is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# RMSNorm op variants this transform recognises as the "final norm" feeding lm_head.
# We accept any of the three because the fuse_rmsnorm transform may have replaced
# the original ``torch_rmsnorm`` with a backend-specific variant by the time we run.
_RMSNORM_OPS = (
    torch.ops.auto_deploy.flashinfer_rms_norm,
    torch.ops.auto_deploy.triton_rms_norm,
    torch.ops.auto_deploy.torch_rmsnorm,
)


# ---------------------------------------------------------------------------
# Fused custom op: rmsnorm(x, ones) @ W.T   with norm_weight already folded
# into W. The op only computes ``x * rstd(x)`` and feeds that into the GEMM,
# matching the algebra above.
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::fused_rmsnorm_lm_head", mutates_args=())
def fused_rmsnorm_lm_head(
    input: torch.Tensor,
    ones: torch.Tensor,
    lm_head_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused (RMSNorm without per-channel weight) + linear.

    ``lm_head_weight`` is expected to have already absorbed the rmsnorm's
    ``weight`` parameter (i.e. ``lm_head_weight = original_lm_head * norm_weight[None, :]``).
    The runtime cost is therefore ``rmsnorm(x, ones)`` + ``_ @ W.T`` which is
    one elementwise pass shorter than the unfused ``rmsnorm + linear`` pair.

    ``ones`` is supplied by the transform as a baked-in graph buffer of shape
    ``(H,)`` and the same dtype/device as ``input``.  Reusing the optimized
    flashinfer kernel with a unit weight handles the rstd compute and per-row
    scale in one launch; the per-channel multiply by ``norm_weight`` has been
    folded into ``lm_head_weight`` so the unit weight scale is mathematically
    a no-op the kernel still has to execute (so this saves the *load* of
    ``norm_weight`` and the *write* of the scaled tensor only — graph-level
    savings come from removing the explicit rmsnorm node from cudagraph).

    Args:
        input: Activations of shape ``(..., H)``.
        ones: Baked unit-weight buffer of shape ``(H,)``, dtype/device matching ``input``.
        lm_head_weight: Pre-scaled lm_head weight of shape ``(V, H)``.
        eps: RMSNorm epsilon.

    Returns:
        Logits of shape ``(..., V)``.
    """
    normed = torch.ops.auto_deploy.flashinfer_rms_norm(input, ones, eps)
    return torch.nn.functional.linear(normed, lm_head_weight, None)


@fused_rmsnorm_lm_head.register_fake
def _fused_rmsnorm_lm_head_fake(
    input: torch.Tensor,
    ones: torch.Tensor,
    lm_head_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    out_shape = (*input.shape[:-1], lm_head_weight.shape[0])
    return input.new_empty(out_shape, dtype=input.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_rms_norm_input(node: Node) -> Optional[Node]:
    """If *node* is one of the RMSNorm variants, return that node; else None.

    Handles a possible ``aten.to.dtype`` cast inserted between the rms_norm
    and the lm_head linear.
    """
    if isinstance(node, Node) and is_op(node, torch.ops.aten.to):
        node = node.all_input_nodes[0]
    if isinstance(node, Node) and is_op(node, _RMSNORM_OPS):
        return node
    return None


def _materialise_attr(gm: GraphModule, key: str) -> torch.Tensor:
    """Resolve a get_attr target to its underlying parameter or buffer tensor."""
    if hasattr(gm, key):
        return getattr(gm, key)
    return gm.get_parameter(key)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


@TransformRegistry.register("fuse_lm_head_with_final_norm")
class FuseLmHeadWithFinalNorm(BaseTransform):
    """Fold the trailing ``rmsnorm`` weight into ``lm_head`` and replace the pair
    with a single ``fused_rmsnorm_lm_head`` op.

    This collapses the model-level
    ``flashinfer_rms_norm(x, w_norm, eps) -> aten.linear(_, w_lm, None)``
    pair into one graph node.  The ``w_norm`` parameter is multiplied into
    ``w_lm`` once at transform time; runtime cost is then:

        fused_rmsnorm_lm_head(x, ones_buf, w_lm * w_norm[None, :], eps)

    Only the head's no-bias linear is targeted (the lm_head). Layers earlier
    in the model are untouched because their rmsnorm output is consumed by
    multiple users (residuals, MoE routing, …) which would still need the
    unscaled normalised tensor.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph

        # Locate the lm_head linear via the standard helper. This walks past
        # an optional ``aten.to`` cast and any sharded all_gather; if the
        # backbone graph does not contain an lm_head linear (e.g. for VLM
        # text-only export) the helper returns the output node itself, which
        # we filter out below.
        try:
            lm_head_node = get_lm_head_node(gm)
        except Exception as exc:  # pragma: no cover - defensive
            ad_logger.debug(f"fuse_lm_head_with_final_norm: lm_head lookup failed: {exc}")
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        if not is_linear_op(lm_head_node):
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # lm_head must be a no-bias linear (lm_head typically has bias=False).
        if len(lm_head_node.args) >= 3 and lm_head_node.args[2] is not None:
            ad_logger.debug("fuse_lm_head_with_final_norm: lm_head has a bias; skipping.")
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # The input to the lm_head must be one of the RMSNorm variants
        # (optionally through an aten.to cast).
        pre_lm_input = lm_head_node.all_input_nodes[0]
        rmsnorm_node = _find_rms_norm_input(pre_lm_input)
        if rmsnorm_node is None:
            ad_logger.debug(
                "fuse_lm_head_with_final_norm: lm_head input is not an RMSNorm; skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Conservative match constraints — the rmsnorm output must be consumed
        # only by this lm_head (and possibly an aten.to cast). If other users
        # exist we cannot safely drop the per-channel weight multiply.
        intermediate = pre_lm_input  # may equal rmsnorm_node, or be the aten.to cast above it
        rmsnorm_users = list(rmsnorm_node.users)
        intermediate_users = list(intermediate.users)
        if len(rmsnorm_users) != 1 or len(intermediate_users) != 1:
            ad_logger.debug(
                "fuse_lm_head_with_final_norm: rmsnorm/cast has additional users; skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Extract the rmsnorm parameters: (input, weight, eps)
        if len(rmsnorm_node.args) < 3:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        norm_input_node = rmsnorm_node.args[0]
        norm_weight_node = rmsnorm_node.args[1]
        eps = rmsnorm_node.args[2]

        if not isinstance(norm_weight_node, Node) or norm_weight_node.op != "get_attr":
            ad_logger.debug(
                "fuse_lm_head_with_final_norm: rmsnorm weight is not a parameter; skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        lm_weight_node = lm_head_node.args[1]
        if not isinstance(lm_weight_node, Node) or lm_weight_node.op != "get_attr":
            ad_logger.debug(
                "fuse_lm_head_with_final_norm: lm_head weight is not a parameter; skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # eps must be a literal float we can carry into the custom op.
        if not isinstance(eps, (int, float)):
            ad_logger.debug("fuse_lm_head_with_final_norm: eps is not a constant; skipping.")
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Materialise the underlying tensors.
        try:
            norm_weight = _materialise_attr(gm, norm_weight_node.target)
            lm_weight = _materialise_attr(gm, lm_weight_node.target)
        except (AttributeError, KeyError) as exc:
            ad_logger.debug(f"fuse_lm_head_with_final_norm: could not resolve weights: {exc}")
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        if norm_weight.numel() != lm_weight.shape[1]:
            ad_logger.debug("fuse_lm_head_with_final_norm: norm/lm shape mismatch; skipping.")
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # If the lm_head weight is also referenced by another node (e.g. a
        # tied input embedding), do NOT mutate it in place — the embedding
        # would silently get the scaled weights too. In that case skip.
        param_users = [
            n
            for n in graph.nodes
            if n is not lm_weight_node and n.op == "get_attr" and n.target == lm_weight_node.target
        ]
        if param_users:
            ad_logger.debug(
                "fuse_lm_head_with_final_norm: lm_head weight has additional get_attr users (tied?); skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Build the fused (pre-scaled) lm_head weight: lm_w * w_norm[None, :].
        # Cast through the lm_head weight dtype so we don't change storage type.
        fused_lm_weight = (
            (lm_weight.to(torch.float32) * norm_weight.to(torch.float32).reshape(1, -1))
            .to(lm_weight.dtype)
            .contiguous()
        )

        fused_key = "fused_lm_head_with_norm_weight"
        # Avoid colliding with prior runs (per-gm). Pick a unique name if needed.
        suffix = 0
        while hasattr(gm, fused_key):
            suffix += 1
            fused_key = f"fused_lm_head_with_norm_weight_{suffix}"
        setattr(gm, fused_key, nn.Parameter(fused_lm_weight, requires_grad=False))

        # Bake a unit-weight buffer into the GraphModule so the fused op
        # signature carries a stable get_attr-fed tensor (no runtime dict
        # lookup, no allocation on first call). Match the lm_head input
        # dtype/device so flashinfer_rms_norm runs without dtype shuffling.
        ones_dtype = norm_weight.dtype
        ones_device = norm_weight.device
        ones_buf = torch.ones(norm_weight.numel(), dtype=ones_dtype, device=ones_device)
        ones_key = "fused_lm_head_norm_unit_weight"
        ones_suffix = 0
        while hasattr(gm, ones_key):
            ones_suffix += 1
            ones_key = f"fused_lm_head_norm_unit_weight_{ones_suffix}"
        # Register as a buffer (not a parameter) since it has no learnable role.
        gm.register_buffer(ones_key, ones_buf, persistent=False)

        ad_logger.info(
            f"Fusing rmsnorm(weight={norm_weight_node.target}) into lm_head "
            f"(weight={lm_weight_node.target}) -> {fused_key} (ones={ones_key}); "
            f"replacing {rmsnorm_node.name} + {lm_head_node.name} with fused op."
        )

        # Insert the fused op right before the lm_head linear, then redirect users.
        with graph.inserting_before(lm_head_node):
            new_w_node = graph.get_attr(fused_key, torch.Tensor)
            ones_node = graph.get_attr(ones_key, torch.Tensor)
            fused_node = graph.call_function(
                torch.ops.auto_deploy.fused_rmsnorm_lm_head.default,
                args=(norm_input_node, ones_node, new_w_node, float(eps)),
            )
            # Preserve metadata so downstream transforms keep shape info.
            ref_val = lm_head_node.meta.get("val")
            if ref_val is not None:
                fused_node.meta["val"] = torch.empty(
                    ref_val.shape, dtype=ref_val.dtype, device="meta"
                )

        lm_head_node.replace_all_uses_with(fused_node)
        graph.erase_node(lm_head_node)
        # If the (optional) cast and rmsnorm now have no users, drop them too.
        if intermediate is not rmsnorm_node and len(intermediate.users) == 0:
            graph.erase_node(intermediate)
        if len(rmsnorm_node.users) == 0:
            graph.erase_node(rmsnorm_node)

        return gm, TransformInfo(skipped=False, num_matches=1)
