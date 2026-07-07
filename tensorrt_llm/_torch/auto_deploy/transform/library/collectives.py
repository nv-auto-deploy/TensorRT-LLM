# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Transformations for fusing collective operations.

This module registers TRT-LLM backend patterns only. Fusion is only applied
when TRT-LLM is available (MPI mode) since it provides optimized fused kernels.
The torch backend (demollm mode) does not benefit from fusion.
"""

from functools import partial
from typing import Optional, Tuple

import torch
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# TODO: This is an overly simplified model that works well for vanilla Llama models.
# However, we eventually want to consider more sophisticated patterns such as
# * all_reduce(lin1(x) + lin2(x))
# * version above with fused GEMMs (i.e. with a split node)
# * all_reduce(pointwise_op(linear(x)))
# * ...


# ============================================================================
# Pattern Template Factory Functions
# ============================================================================


_RMSNORM_OPS = {
    "torch_rmsnorm": torch.ops.auto_deploy.torch_rmsnorm,
    "triton_rms_norm": torch.ops.auto_deploy.triton_rms_norm,
}


def _make_allreduce_residual_rmsnorm_pattern(
    add_order: str = "residual_first",
    strategy: str = "AUTO",
    rmsnorm_op_name: str = "torch_rmsnorm",
):
    """Factory function to create pattern functions for allreduce+residual+rmsnorm fusion.

    Args:
        add_order: Either "residual_first" (residual + x) or "x_first" (x + residual)
        strategy: AllReduce strategy to use in the pattern
        rmsnorm_op_name: Which rmsnorm op to match ("torch_rmsnorm" or "triton_rms_norm")

    Returns:
        A pattern function that can be used with register_ad_pattern
    """
    rmsnorm_op = _RMSNORM_OPS[rmsnorm_op_name]

    def pattern_fn(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 0.1253
    ):
        """Pattern: trtllm_dist_all_reduce(x) -> add residual -> rmsnorm

        Reference PyTorch composition:
            y = trtllm_dist_all_reduce(x)
            z = residual + y  (or y + residual)
            normed = rmsnorm_op(z, weight, eps)
        Returns (normed, z)
        """
        hidden_states = torch.ops.auto_deploy.trtllm_dist_all_reduce(x, strategy)

        # Handle addition order
        if add_order == "residual_first":
            add = residual + hidden_states
        else:  # x_first
            add = hidden_states + residual

        normed = rmsnorm_op(add, weight, eps)

        return normed, add

    return pattern_fn


def _allreduce_residual_rmsnorm_replacement(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, strategy: str
):
    """Replacement using TRT-LLM fused kernel."""
    return torch.ops.dist.trtllm_fused_allreduce_residual_rmsnorm(
        x, residual, weight, eps, strategy
    )


# ============================================================================
# Transform Implementation
# ============================================================================


@TransformRegistry.register("fuse_allreduce_residual_rmsnorm")
class FuseAllreduceResidualRMSNorm(BaseTransform):
    """Fuse (allreduce + residual add + RMSNorm) into one fused op with tuple output.

    This transform only applies when TRT-LLM ops are used (MPI mode), as it provides
    optimized fused kernels. The torch backend (demollm mode) does not benefit from
    this fusion and uses unfused operations.

    Note: This transform expects torch_rmsnorm ops in the graph, which are created
    by the match_rmsnorm_pattern transform that runs earlier in the pipeline.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Collectives fusion depends on sharding (reads _sharding_transform_container).
        # Draft models are not sharded, so skip them.
        if getattr(gm, "is_draft", False):
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        if shared_config.dist_config is not None:
            # Primary production path: DistConfig built by LlmArgs.init_dist_config
            # with allreduce_strategy populated from YAML.
            strategy = shared_config.dist_config.allreduce_strategy
        elif hasattr(gm, "_sharding_transform_container"):
            # Legacy fallback: entered only by external invocations that construct
            # InferenceOptimizer without a dist_config kwarg (e.g.
            # tests/unittest/auto_deploy/multigpu/transformations/library/
            # test_allreduce_residual_rmsnorm_fusion.py). Will be removed together
            # with the legacy sharding pipeline (sharding.py).
            strategy = gm._sharding_transform_container.config.allreduce_strategy.name
        else:
            ad_logger.warning("No dist config found, skipping allreduce-residual-rmsnorm fusion")
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        ad_logger.info(f"allreduce strategy selected = {strategy!r}")

        # ============================================================================
        # Instantiate Pattern Functions
        # ============================================================================

        patterns = ADPatternMatcherPass()

        # Dummy shapes for tracing
        bsz, hidden = 8, 512
        dummy_args = [
            torch.randn(bsz, hidden, device="meta", dtype=torch.bfloat16),  # x
            torch.randn(bsz, hidden, device="meta", dtype=torch.bfloat16),  # residual
            torch.randn(hidden, device="meta", dtype=torch.bfloat16),  # weight
            0.1253,  # eps
        ]
        scalar_workaround = {"eps": 0.1253}

        for rmsnorm_op_name in _RMSNORM_OPS:
            for add_order in ("residual_first", "x_first"):
                pattern = _make_allreduce_residual_rmsnorm_pattern(
                    add_order=add_order, strategy=strategy, rmsnorm_op_name=rmsnorm_op_name
                )
                register_ad_pattern(
                    search_fn=pattern,
                    replace_fn=partial(_allreduce_residual_rmsnorm_replacement, strategy=strategy),
                    patterns=patterns,
                    dummy_args=dummy_args,
                    scalar_workaround=scalar_workaround,
                )

        num_matches = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


# ============================================================================
# Collinear all-reduce fusion: all_reduce(a) + all_reduce(b) -> all_reduce(a + b)
# ============================================================================

# Real (post-sharding) distributed all-reduce ops. Both reduce over the *world*
# process group -- ``trtllm_allreduce`` builds its Mapping with
# ``tp_size == world_size`` (custom_ops/distributed/trtllm_dist.py) and
# ``torch_dist_all_reduce`` uses the default group -- so any two of them can be
# folded: AR(a) + AR(b) == AR(a + b).
_DIST_ALLREDUCE_OPS = (
    torch.ops.auto_deploy.trtllm_dist_all_reduce,
    torch.ops.auto_deploy.torch_dist_all_reduce,
)

# No-op (shape- and dtype-preserving) wrappers we may walk past between an
# all_reduce and the consuming add. View/reshape are intentionally excluded:
# peeling a shape-changing op would leave the two all_reduce inputs with
# mismatched shapes, so ``in_a + in_b`` would not reproduce the original add.
_NOOP_WRAPPER_OPS = (
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
)

# Cast ops we may walk past *only* when they are no-ops (input dtype == output
# dtype). The row-parallel MLP's trailing ``.to(x.dtype)`` lowers to
# ``aten.to.dtype`` (and ``aten._to_copy`` in other paths); when the activation is
# already in the target dtype this is a pure identity, so folding the add in front
# of the all_reduce is value-preserving.
_CAST_OPS = (
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
)


def _peel_to_allreduce(node: Node) -> Optional[Node]:
    """Walk a single-user chain of no-op wrappers back to a distributed all_reduce.

    Returns the all_reduce ``Node`` if ``node`` is -- after skipping no-op
    clone/contiguous/dtype-preserving cast wrappers -- produced by one of the
    distributed all-reduce ops, *and* every link on the way (including the
    all_reduce itself) has exactly one user. The single-user requirement makes the
    whole chain dead once the add is rewritten, so the fold is a net -1 collective
    rather than +1. Returns ``None`` if any condition fails.
    """
    current = node
    while isinstance(current, Node):
        if is_op(current, _DIST_ALLREDUCE_OPS):
            return current if len(current.users) == 1 else None
        if len(current.users) != 1:
            return None
        if is_op(current, _NOOP_WRAPPER_OPS):
            current = current.args[0]
            continue
        if is_op(current, _CAST_OPS):
            src = current.args[0]
            if not isinstance(src, Node):
                return None
            src_val = src.meta.get("val")
            cur_val = current.meta.get("val")
            # Only skip a genuine no-op cast (same dtype). A real dtype change
            # before vs. after the reduction would alter accumulation precision.
            if src_val is not None and cur_val is not None and src_val.dtype == cur_val.dtype:
                current = src
                continue
            return None
        return None
    return None


@TransformRegistry.register("fuse_collinear_allreduce")
class FuseCollinearAllreduce(BaseTransform):
    """Fold two same-group all-reduces feeding one add into a single all-reduce.

    all_reduce is linear across ranks, so ``AR(a) + AR(b) == AR(a + b)``. In
    AutoDeploy every ``*_dist_all_reduce`` reduces over the *world* process group,
    so the EP all_reduce inserted by ``apply_sharding_hints`` for a routed MoE op
    and the TP all_reduce of a row-parallel linear are both world reductions and
    can be folded when they meet at an add.

    Canonical case: DeepSeek-V4's MoE forward returns
    ``routed_experts (EP all_reduce) + shared_experts (TP all_reduce)`` -- two
    collectives per MoE layer where one suffices.

    Fires only when both all_reduces (a) use the same op + strategy, (b) feed the
    add as their sole consumer (so the fold is a net -1 collective, not +1), and
    (c) carry matching input shapes (so ``a + b`` reproduces the original add).
    This runs in post_load_fusion, after ``apply_sharding_hints`` has materialized
    the real collectives.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        graph = gm.graph
        cnt = 0
        for node in list(graph.nodes):
            if not is_op(node, torch.ops.aten.add.Tensor) or len(node.args) < 2:
                continue
            lhs, rhs = node.args[0], node.args[1]
            if not (isinstance(lhs, Node) and isinstance(rhs, Node)):
                continue
            ar_a = _peel_to_allreduce(lhs)
            ar_b = _peel_to_allreduce(rhs)
            if ar_a is None or ar_b is None or ar_a is ar_b:
                continue
            # Same collective op (=> same world group) and same strategy.
            if ar_a.target is not ar_b.target:
                continue
            strat_a = ar_a.args[1] if len(ar_a.args) > 1 else ar_a.kwargs.get("strategy")
            strat_b = ar_b.args[1] if len(ar_b.args) > 1 else ar_b.kwargs.get("strategy")
            if strat_a != strat_b:
                continue
            in_a, in_b = ar_a.args[0], ar_b.args[0]
            if not (isinstance(in_a, Node) and isinstance(in_b, Node)):
                continue
            val_a = in_a.meta.get("val")
            val_b = in_b.meta.get("val")
            # Matching shapes => (in_a + in_b) reproduces the original add exactly.
            if val_a is None or val_b is None or tuple(val_a.shape) != tuple(val_b.shape):
                continue

            with graph.inserting_before(node):
                summed = graph.call_function(torch.ops.aten.add.Tensor, args=(in_a, in_b))
                summed.meta["val"] = torch.empty(val_a.shape, dtype=val_a.dtype, device="meta")
                fused_ar = graph.call_function(ar_a.target, args=(summed, strat_a))
                ref_val = node.meta.get("val")
                if ref_val is not None:
                    fused_ar.meta["val"] = torch.empty(
                        ref_val.shape, dtype=ref_val.dtype, device="meta"
                    )
            node.replace_all_uses_with(fused_ar)
            cnt += 1

        if cnt > 0:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(f"fused {cnt} collinear all_reduce pair(s) -> single all_reduce")

        info = TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info


# ============================================================================
# Merge-add fusion: all_reduce(other + fp8_linear(x)) folds the add into the
# linear's epilogue so the projection writes the collective's input directly.
# ============================================================================

# Positional argument order + keyword defaults of
# ``auto_deploy::torch_fake_quant_finegrained_fp8_linear`` (used to normalize a
# matched graph node's args/kwargs into the full positional tuple for the fused
# residual-add variant, whose signature is the same list + trailing ``residual``).
_FINEGRAINED_FP8_LINEAR_ARG_NAMES = (
    "input",
    "weight_quantized",
    "bias",
    "input_scale",
    "weight_scale",
    "input_zp",
    "weight_zp",
    "tp_mode",
    "output_sizes",
    "tp_min_local_shape",
    "layer_type",
    "input_scale_fmt",
)
_FINEGRAINED_FP8_LINEAR_ARG_DEFAULTS = {
    "tp_mode": "none",
    "output_sizes": None,
    "tp_min_local_shape": 1,
    "layer_type": "unknown",
    "input_scale_fmt": "",
}


@TransformRegistry.register("fuse_fp8_linear_allreduce_add")
class FuseFp8LinearAllreduceAdd(BaseTransform):
    """Fold an all_reduce input's merge add into the producing block-FP8 linear.

    Matches ``all_reduce(add(other, linear(x)))`` (either operand order) where the
    linear is a bias-free ``torch_fake_quant_finegrained_fp8_linear`` whose sole
    consumer is the add, and the add's sole consumer is a distributed all_reduce.
    The add is replaced by ``torch_fake_quant_finegrained_fp8_linear_residual_add``,
    which folds the elementwise add into the W8A8 matmul epilogue so the projection
    writes the summed tensor -- the collective's input buffer -- directly (bit-exact:
    the accumulator is rounded to the output dtype before the fp32-opmath add).

    Canonical case: DeepSeek-V4's MoE seam after ``fuse_collinear_allreduce``,
    ``AR(routed_moe_out + shared_down_proj_out)`` -- one standalone bf16 add per MoE
    layer collapses into the shared-expert down projection. The rewrite keeps the
    merge-node data dependencies intact (the routed output becomes the fused node's
    ``residual`` arg, passed positionally), so the multi-stream MoE transform still
    classifies the fused node as the shared/routed merge point and inserts its
    aux-stream sync around it.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if not self.config.enabled:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        linear_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
        fused_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_residual_add

        graph = gm.graph
        cnt = 0
        for node in list(graph.nodes):
            if not is_op(node, torch.ops.aten.add.Tensor) or len(node.args) < 2:
                continue
            # The add must feed exactly one distributed all_reduce, so the fused
            # projection output is precisely the collective's input buffer.
            if len(node.users) != 1:
                continue
            (consumer,) = node.users
            if not is_op(consumer, _DIST_ALLREDUCE_OPS):
                continue
            lhs, rhs = node.args[0], node.args[1]
            if not (isinstance(lhs, Node) and isinstance(rhs, Node)):
                continue
            for linear, other in ((lhs, rhs), (rhs, lhs)):
                if not is_op(linear, linear_op):
                    continue
                # Sole consumer => the original linear node is dead after the
                # rewrite (net -1 kernel, not a duplicated matmul).
                if len(linear.users) != 1:
                    continue
                vals = dict(zip(_FINEGRAINED_FP8_LINEAR_ARG_NAMES, linear.args))
                vals.update(linear.kwargs)
                for name, default in _FINEGRAINED_FP8_LINEAR_ARG_DEFAULTS.items():
                    vals.setdefault(name, default)
                # The fused epilogue reproduces add(matmul_out, residual); a bias
                # would introduce a second, differently-ordered rounding point.
                if vals.get("bias") is not None:
                    continue
                lin_val = linear.meta.get("val")
                other_val = other.meta.get("val")
                if lin_val is None or other_val is None:
                    continue
                # Matching shape + dtype => the epilogue add is elementwise with no
                # broadcast and reproduces the original add exactly.
                if (
                    tuple(lin_val.shape) != tuple(other_val.shape)
                    or lin_val.dtype != other_val.dtype
                ):
                    continue

                # Insert at the add's position: both the linear's inputs and
                # ``other`` are already defined there regardless of which branch
                # was emitted first. ``residual`` is passed positionally so
                # downstream arg-rewriting passes (e.g. the multi-stream wait
                # insertion) see it in ``node.args``.
                with graph.inserting_before(node):
                    fused = graph.call_function(
                        fused_op,
                        args=tuple(vals[n] for n in _FINEGRAINED_FP8_LINEAR_ARG_NAMES) + (other,),
                    )
                ref_val = node.meta.get("val")
                if ref_val is not None:
                    fused.meta["val"] = torch.empty(
                        ref_val.shape, dtype=ref_val.dtype, device="meta"
                    )
                node.replace_all_uses_with(fused)
                cnt += 1
                break

        if cnt > 0:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(f"fused {cnt} allreduce-input merge add(s) into block-FP8 linear(s)")

        info = TransformInfo(
            skipped=False,
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info
