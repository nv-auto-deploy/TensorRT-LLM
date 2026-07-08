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
import operator
from collections import defaultdict
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import triton
from pydantic import Field
from torch.fx import GraphModule, Node

from tensorrt_llm.quantization.utils.fp8_utils import (
    resmooth_to_fp8_e8m0,
    transform_sf_into_required_layout,
)

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.node_utils import extract_op_args, is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import (
    BaseTransform,
    SharedConfig,
    TransformConfig,
    TransformInfo,
    TransformRegistry,
)


# with bias=None
def _fp8_ref_pattern_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
        x,
        w_fp8,
        None,
        input_scale=[input_scale],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


# with bias!=None
def _fp8_ref_pattern_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
        x,
        w_fp8,
        bias,
        input_scale=[input_scale],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


# NVFP4: with bias=None
def _fp4_ref_pattern_1(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        w_fp4,
        None,
        input_scale=[input_scale],
        weight_scale=[weight_scale, alpha],
        input_zp=[],
        weight_zp=[],
    )


def _fp4_ref_repl_1(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input=x,
        weight_fp4=w_fp4,
        bias=None,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=alpha,
    )


# with bias!=None
def _fp4_ref_pattern_2(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    bias: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        w_fp4,
        bias,
        input_scale=[input_scale],
        weight_scale=[weight_scale, alpha],
        input_zp=[],
        weight_zp=[],
    )


def _fp4_ref_repl_2(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    bias: torch.Tensor | None,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input=x,
        weight_fp4=w_fp4,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
        alpha=alpha,
    )


def _register_quant_fp8_linear_patterns(patterns: ADPatternMatcherPass, op) -> None:
    """
    Register FP8 linear patterns with robust dummy args and minimal ignores.
    """

    # Define replacement functions that use the provided op.
    # Use keyword-only binding for input/weight/bias so the call stays robust
    # against any FX-state perturbation that affects positional arg layout
    # (e.g., sharding placeholder insertion in this PR).
    def _fp8_ref_repl_1(
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        return op(
            input=x,
            weight_fp8=w_fp8,
            bias=None,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )

    def _fp8_ref_repl_2(
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        bias: torch.Tensor,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ):
        return op(
            input=x,
            weight_fp8=w_fp8,
            bias=bias,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )

    # FP8 dummy tensors
    x_fp8 = torch.randn(3, 16, device="meta", dtype=torch.float16)
    w_fp8 = torch.randn(32, 16, device="meta", dtype=torch.float16)
    bias32 = torch.randn(32, device="meta", dtype=torch.float32)
    one = torch.tensor(1.0, device="meta", dtype=torch.float32)

    # no-bias variant
    dummy_args_fp8 = [
        x_fp8,
        w_fp8,
        one,
        torch.tensor(0.5, device="meta", dtype=torch.float32),
    ]
    register_ad_pattern(
        search_fn=_fp8_ref_pattern_1,
        replace_fn=_fp8_ref_repl_1,
        patterns=patterns,
        dummy_args=dummy_args_fp8,
    )

    # bias variant
    dummy_args_fp8_2 = [
        x_fp8,
        w_fp8,
        bias32,
        one,
        torch.tensor(0.5, device="meta", dtype=torch.float32),
    ]
    register_ad_pattern(
        search_fn=_fp8_ref_pattern_2,
        replace_fn=_fp8_ref_repl_2,
        patterns=patterns,
        dummy_args=dummy_args_fp8_2,
    )


def _register_quant_fp4_linear_patterns(patterns: ADPatternMatcherPass) -> None:
    """
    Register FP4 linear patterns with robust dummy args and minimal ignores.
    """
    # FP4 shape params
    N = 32
    K_packed = 32  # weight is packed by 2 FP4 per byte
    K_eff = 2 * K_packed

    # FP4 dummy tensors
    x_fp4 = torch.randn(3, K_eff, device="meta", dtype=torch.float16)
    w_fp4 = torch.randint(0, 255, (N, K_packed), device="meta", dtype=torch.uint8)

    s_in2 = torch.tensor(0.01, device="meta", dtype=torch.float32)
    alpha = torch.tensor(1.2345, device="meta", dtype=torch.float32)

    cutlass_len = N * (K_eff // 16)  # 32 * (64/16) = 128
    cutlass_vec = torch.randint(0, 255, (cutlass_len,), device="meta", dtype=torch.uint8)

    # no-bias variant
    dummy_args_fp4_1 = [
        x_fp4,
        w_fp4,
        s_in2,
        cutlass_vec,
        alpha,
    ]
    register_ad_pattern(
        search_fn=_fp4_ref_pattern_1,
        replace_fn=_fp4_ref_repl_1,
        patterns=patterns,
        dummy_args=dummy_args_fp4_1,
    )

    # bias variant
    dummy_args_fp4_2 = [
        x_fp4,
        w_fp4,
        torch.randn(N, device="meta", dtype=torch.float16),  # bias
        s_in2,
        cutlass_vec,
        alpha,
    ]
    register_ad_pattern(
        search_fn=_fp4_ref_pattern_2,
        replace_fn=_fp4_ref_repl_2,
        patterns=patterns,
        dummy_args=dummy_args_fp4_2,
    )


class FuseFP8LinearConfig(TransformConfig):
    """Configuration for FP8 linear fusion transform."""

    backend: str = Field(
        default="torch",
        description="Backend to use for FP8 linear computation (default: 'torch').",
    )


@TransformRegistry.register("fuse_fp8_linear")
class FuseFP8Linear(BaseTransform):
    """Matches and replaces FP8 fake quantized linear ops with fused torch backend ops."""

    config: FuseFP8LinearConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseFP8LinearConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend.lower() not in ["torch", "trtllm"]:
            raise ValueError(f"Unsupported FP8 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        op = (
            torch.ops.auto_deploy.trtllm_quant_fp8_linear
            if self.config.backend.lower() == "trtllm"
            else torch.ops.auto_deploy.torch_quant_fp8_linear
        )

        _register_quant_fp8_linear_patterns(patterns, op)
        cnt = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=cnt == 0,
            has_valid_shapes=cnt == 0,
        )
        return gm, info


class FuseNVFP4LinearConfig(TransformConfig):
    """Configuration for NVFP4 linear fusion transform."""

    backend: str = Field(
        default="trtllm",
        description="Backend to use for NVFP4 linear computation (default: 'trtllm').",
    )


@TransformRegistry.register("fuse_nvfp4_linear")
class FuseNVFP4Linear(BaseTransform):
    """Matches and replaces NVFP4 fake quantized linear ops with fused TensorRT-LLM ops."""

    config: FuseNVFP4LinearConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseNVFP4LinearConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend.lower() != "trtllm":
            raise ValueError(f"Unsupported NVFP4 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        _register_quant_fp4_linear_patterns(patterns)
        cnt = patterns.apply(gm.graph)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info


# ============================================================================
# FineGrained FP8 Linear Patterns (for MiniMax M2, DeepSeek, etc.)
# ============================================================================


# FineGrained FP8: with bias=None
def _finegrained_fp8_pattern_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
        x,
        w_fp8,
        None,
        input_scale=[],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


def _finegrained_fp8_repl_1(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.trtllm_finegrained_fp8_linear(
        x,
        w_fp8,
        None,
        weight_scale,
    )


# FineGrained FP8: with bias!=None
def _finegrained_fp8_pattern_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
        x,
        w_fp8,
        bias,
        input_scale=[],
        weight_scale=[weight_scale],
        input_zp=[],
        weight_zp=[],
    )


def _finegrained_fp8_repl_2(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    bias: torch.Tensor,
    weight_scale: torch.Tensor,
):
    return torch.ops.auto_deploy.trtllm_finegrained_fp8_linear(
        x,
        w_fp8,
        bias,
        weight_scale,
    )


def _register_finegrained_fp8_linear_patterns(patterns: ADPatternMatcherPass) -> None:
    """
    Register FineGrained FP8 linear patterns.

    FineGrained FP8 uses block-wise weight quantization with per-block scales.
    The replacement uses TRT-LLM's optimized fp8_block_scaling_gemm kernel.
    """
    # FineGrained FP8 dummy tensors
    # weight shape: [N, K], weight_scale shape: [N/128, K/128]
    N, K = 256, 256  # Must be multiples of 128 for block quantization
    x_fg_fp8 = torch.randn(3, K, device="meta", dtype=torch.bfloat16)
    w_fg_fp8 = torch.randn(N, K, device="meta", dtype=torch.float8_e4m3fn)
    bias_fg = torch.randn(N, device="meta", dtype=torch.bfloat16)
    # Per-block weight scale: [N/128, K/128]
    weight_scale_fg = torch.randn(N // 128, K // 128, device="meta", dtype=torch.float32)

    # no-bias variant
    dummy_args_fg_fp8_1 = [
        x_fg_fp8,
        w_fg_fp8,
        weight_scale_fg,
    ]
    register_ad_pattern(
        search_fn=_finegrained_fp8_pattern_1,
        replace_fn=_finegrained_fp8_repl_1,
        patterns=patterns,
        dummy_args=dummy_args_fg_fp8_1,
    )

    # bias variant
    dummy_args_fg_fp8_2 = [
        x_fg_fp8,
        w_fg_fp8,
        bias_fg,
        weight_scale_fg,
    ]
    register_ad_pattern(
        search_fn=_finegrained_fp8_pattern_2,
        replace_fn=_finegrained_fp8_repl_2,
        patterns=patterns,
        dummy_args=dummy_args_fg_fp8_2,
    )


class FuseFineGrainedFP8LinearConfig(TransformConfig):
    """Configuration for FineGrained FP8 linear fusion transform."""

    backend: str = Field(
        default="trtllm",
        description="Backend to use for FineGrained FP8 linear computation (default: 'trtllm').",
    )


def _resolve_attr_tensor(gm: GraphModule, attr_node: Node) -> Optional[torch.Tensor]:
    """Resolve a get_attr node's target to the live tensor on `gm`, or None.

    The `weight_scale` arg may be either a registered buffer (common) or a
    parameter. We fall back to a plain getattr walk to remain tolerant of both.
    """
    if not isinstance(attr_node, Node) or attr_node.op != "get_attr":
        return None
    target = attr_node.target
    if not isinstance(target, str):
        return None
    try:
        return gm.get_buffer(target)
    except AttributeError:
        pass
    try:
        return gm.get_parameter(target)
    except AttributeError:
        pass
    obj = gm
    for name in target.split("."):
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj if isinstance(obj, torch.Tensor) else None


def _replace_attr_tensor(gm: GraphModule, attr_node: Node, new_tensor: torch.Tensor) -> bool:
    """Replace the live tensor backing a get_attr node on `gm`, preserving its
    original storage class (parameter vs buffer).

    Walks the dotted target to the parent module, finds whether the attr was
    registered as a parameter or a buffer, then re-registers `new_tensor` under
    the same class. Keeping the storage class stable is important because
    downstream code (e.g., parameter counting in unit-test helpers) treats
    the two differently.
    """
    if not isinstance(attr_node, Node) or attr_node.op != "get_attr":
        return False
    target = attr_node.target
    if not isinstance(target, str):
        return False

    *path, attr_name = target.split(".")
    obj = gm
    for p in path:
        obj = getattr(obj, p, None)
        if obj is None:
            return False

    was_parameter = hasattr(obj, "_parameters") and attr_name in obj._parameters
    was_buffer = hasattr(obj, "_buffers") and attr_name in obj._buffers

    # Drop any existing registration so we can re-register cleanly.
    if was_parameter:
        del obj._parameters[attr_name]
    if was_buffer:
        del obj._buffers[attr_name]
    if attr_name in obj.__dict__:
        del obj.__dict__[attr_name]

    if was_parameter and not was_buffer:
        # Preserve parameter storage. Note: fp8 dtypes are non-differentiable;
        # the original FineGrainedFP8 model registers fp8 weights as buffers,
        # so this branch typically won't fire for them — included for safety.
        setattr(obj, attr_name, nn.Parameter(new_tensor.detach(), requires_grad=False))
    else:
        # Default to buffer (matches FineGrainedFP8 model's storage class for
        # both weight_fp8 and weight_scale_inv).
        obj.register_buffer(attr_name, new_tensor.detach())
    return True


def _dispatch_trtllm_finegrained_fp8_to_deepgemm(gm: GraphModule) -> int:
    """Compile-time dispatch: rewrite to DeepGEMM and convert scales atomically.

    For each `trtllm_finegrained_fp8_linear` node we choose to swap to
    `trtllm_fp8_deepgemm`, we *also* convert that node's weight + weight_scale
    in place to UE8M0 packed int + TMA col-major layout, in a single pass.

    Doing the scale conversion here (instead of in a separate post_load_hook)
    guarantees the graph never holds a UE8M0 scale paired with a raw-FP32-scale
    op (`trtllm_finegrained_fp8_linear` or `torch_fake_quant_finegrained_fp8_linear`),
    which would otherwise produce NaN. Nodes that fail any precondition
    (op not present, weight not 128-aligned, fp8_utils missing) keep raw FP32
    scales and stay on `trtllm_finegrained_fp8_linear` (cuBLAS / fp8_block_scaling
    fallback).

    Returns the number of rewritten nodes.
    """
    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        return 0
    # Positional index of weight_scale in trtllm_finegrained_fp8_linear signature:
    #   (input, weight, bias, weight_scale, tp_mode=..., ...)
    weight_scale_arg = 3

    src_op = torch.ops.auto_deploy.trtllm_finegrained_fp8_linear
    dst_op = getattr(torch.ops.auto_deploy, "trtllm_fp8_deepgemm", None)
    if dst_op is None:
        return 0

    num_rewrites = 0
    for node in gm.graph.nodes:
        if not is_op(node, src_op):
            continue
        if len(node.args) <= weight_scale_arg:
            continue

        weight_arg = node.args[1]
        scale_arg = node.args[weight_scale_arg]

        weight_tensor = _resolve_attr_tensor(gm, weight_arg)
        scale_tensor = _resolve_attr_tensor(gm, scale_arg)
        if weight_tensor is None or scale_tensor is None:
            continue
        if weight_tensor.dtype != torch.float8_e4m3fn:
            continue

        # If a previous run already converted this scale (e.g., re-applying the
        # transform), just ensure the op target points at deepgemm.
        if scale_tensor.dtype == torch.int:
            node.target = dst_op.default
            num_rewrites += 1
            continue

        N, K = weight_tensor.shape[-2], weight_tensor.shape[-1]
        if N % 128 != 0 or K % 128 != 0:
            # TP-misaligned projections fall back to cuBLAS with raw FP32 scale.
            continue

        try:
            with torch.no_grad():
                weight_new, scale_new = resmooth_to_fp8_e8m0(weight_tensor, scale_tensor.float())
                N_new, K_new = weight_new.shape[-2], weight_new.shape[-1]
                transformed_scale = transform_sf_into_required_layout(
                    scale_new,
                    mn=N_new,
                    k=K_new,
                    recipe=(1, 128, 128),
                    is_sfa=False,
                )
        except Exception as exc:  # pragma: no cover - defensive: keep raw path on error
            ad_logger.warning(
                f"DeepGEMM scale conversion failed for {scale_arg.target}: {exc}; "
                f"keeping trtllm_finegrained_fp8_linear (raw FP32 scale) for this node."
            )
            continue

        if not _replace_attr_tensor(gm, weight_arg, weight_new):
            continue
        if not _replace_attr_tensor(gm, scale_arg, transformed_scale):
            continue

        # Signatures match positionally; safe to swap target now that buffers
        # have been converted in lock-step.
        node.target = dst_op.default
        num_rewrites += 1

    return num_rewrites


@TransformRegistry.register("fuse_finegrained_fp8_linear")
class FuseFineGrainedFP8Linear(BaseTransform):
    """Matches and replaces FineGrained FP8 fake quantized linear ops with TRT-LLM ops.

    Two-stage pipeline:
      1. Pattern matcher rewrites ``torch_fake_quant_finegrained_fp8_linear``
         (HuggingFace triton kernel) to ``trtllm_finegrained_fp8_linear``
         (TRT-LLM ``fp8_block_scaling_gemm`` with FP32 per-block scales).
      2. A compile-time dispatch pass further rewrites any nodes whose
         ``weight_scale`` buffer is UE8M0 packed int (produced by
         ``FineGrainedFP8LinearQuantization.post_load_hook`` on SM100f) to
         the dedicated ``trtllm_fp8_deepgemm`` op. Keeping the SM100f/UE8M0
         path in a separate op avoids per-call hardware / dtype branching
         inside the runtime op.

    Used for models like MiniMax M2 and DeepSeek that use HuggingFace's FineGrained FP8
    quantization format with 128x128 block sizes.
    """

    config: FuseFineGrainedFP8LinearConfig

    @classmethod
    def get_config_class(cls) -> Type[TransformConfig]:
        return FuseFineGrainedFP8LinearConfig

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        if self.config.backend.lower() != "trtllm":
            raise ValueError(f"Unsupported FineGrained FP8 backend: {self.config.backend}")

        patterns = ADPatternMatcherPass()
        _register_finegrained_fp8_linear_patterns(patterns)
        cnt = patterns.apply(gm.graph)

        # Compile-time dispatch to the UE8M0 fast-path op. Counts toward
        # num_matches so downstream graph invariants get re-checked.
        cnt += _dispatch_trtllm_finegrained_fp8_to_deepgemm(gm)

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info


# ============================================================================
# FineGrained FP8 activation-quant common-subexpression elimination
# ============================================================================


def _finegrained_fp8_block_k(
    gm: GraphModule, weight_node: object, weight_scale_arg: object
) -> Optional[int]:
    """Derive the activation-quant block size (block_k) of a FineGrained FP8 linear.

    ``torch_fake_quant_finegrained_fp8_linear`` infers ``block_k = cdiv(K, scale_k)``
    from the weight ``[N, K]`` and per-block weight scale ``[N/block_n, K/block_k]``
    shapes and quantizes the activation in groups of that size. Two linears can share
    one activation quant only when this block size matches, so the transform groups on
    it. Returns ``None`` when the shapes cannot be resolved (the node is skipped).
    """
    if not isinstance(weight_node, Node):
        return None
    if not isinstance(weight_scale_arg, (list, tuple)) or len(weight_scale_arg) == 0:
        return None
    scale_node = weight_scale_arg[0]
    weight = _resolve_attr_tensor(gm, weight_node)
    scale = _resolve_attr_tensor(gm, scale_node)
    if weight is None or scale is None or weight.dim() != 2 or scale.dim() != 2:
        return None
    K = weight.shape[1]
    scale_k = scale.shape[1]
    if scale_k == 0:
        return None
    return int(triton.cdiv(K, scale_k))


@TransformRegistry.register("fuse_fp8_act_quant_cse")
class FuseFP8ActQuantCSE(BaseTransform):
    """Share one block-FP8 activation quant across sibling FineGrained FP8 linears.

    Models with MLA + MoE (DeepSeek-V4, etc.) feed the *same* activation tensor into
    several FineGrained FP8 linears, e.g.:
      * ``attn.wq_a`` + ``attn.wkv``               (both on the attention-norm output)
      * ``attn.wq_b`` + ``attn.indexer.wq_b``      (both on the q-lora)
      * shared-expert ``w1`` + ``w3``              (both on the MoE/MLP input)

    Each ``torch_fake_quant_finegrained_fp8_linear`` re-runs ``_safe_act_quant`` (the
    ``_act_quant_kernel`` Triton launch) on that identical tensor with the identical
    block size, producing a byte-identical ``(fp8, scale)`` pair every time -- pure
    redundant work (this kernel is the largest single hit-count Triton launch in the
    DeepSeek-V4 decode window).

    This transform hoists one ``torch_fp8_finegrained_act_quant`` per
    ``(input, block_k, input_scale_fmt)`` group with >= 2 members and rewrites every
    member to the matmul-only ``torch_fake_quant_finegrained_fp8_linear_prequant``,
    deleting the redundant quant launches. Singleton groups are left untouched
    (splitting them would be neutral). Reference-exact: the quant kernel is a
    deterministic pure function, so one shared launch equals each per-linear recompute
    bit for bit.

    Runs in post_load_fusion AFTER ``fuse_finegrained_fp8_linear``. When the TRT-LLM
    finegrained fuse is enabled the linears are already routed to deepgemm / cuBLAS
    (which quantize internally), so this transform matches nothing -- it only fires on
    the torch reference path (fuse disabled), which is what DeepSeek-V4 runs.
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
        lin_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear
        act_op = torch.ops.auto_deploy.torch_fp8_finegrained_act_quant.default
        prequant_op = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant.default

        # Group linear nodes by (shared input node, block_k, input_scale_fmt). Nodes are
        # visited in topological order, so each group's members[] is already ordered and
        # members[0] is the earliest -- a safe insertion point for the shared quant.
        groups: "dict[Tuple[Node, int, str], list[Node]]" = defaultdict(list)
        for node in gm.graph.nodes:
            if not is_op(node, lin_op):
                continue
            inp, weight, weight_scale, fmt = extract_op_args(
                node, "input", "weight_quantized", "weight_scale", "input_scale_fmt"
            )
            if not isinstance(inp, Node):
                continue
            block_k = _finegrained_fp8_block_k(gm, weight, weight_scale)
            if block_k is None:
                continue
            groups[(inp, block_k, fmt or "")].append(node)

        num_groups = 0
        num_linears = 0
        num_quant_removed = 0
        for (inp, block_k, fmt), members in groups.items():
            if len(members) < 2:
                continue
            num_groups += 1
            num_quant_removed += len(members) - 1  # N quant launches collapse to 1

            # One shared activation quant, inserted right before the earliest member
            # (which already dominates -- it consumes `inp`, so it follows `inp` and all
            # graph placeholders). The getitems unpack the (qfp8, scale) tuple.
            first = members[0]
            with gm.graph.inserting_before(first):
                act_node = gm.graph.call_function(act_op, args=(inp, int(block_k), fmt))
                qfp8 = gm.graph.call_function(operator.getitem, args=(act_node, 0))
                qscale = gm.graph.call_function(operator.getitem, args=(act_node, 1))

            for member in members:
                _, weight, bias, weight_scale = extract_op_args(
                    member, "input", "weight_quantized", "bias", "weight_scale"
                )
                with gm.graph.inserting_before(member):
                    new_node = gm.graph.call_function(
                        prequant_op,
                        args=(qfp8, qscale, weight, bias, weight_scale),
                    )
                member.replace_all_uses_with(new_node)
                gm.graph.erase_node(member)
                num_linears += 1

        if num_groups:
            ad_logger.info(
                f"fuse_fp8_act_quant_cse: shared {num_groups} activation-quant group(s) "
                f"across {num_linears} FineGrained FP8 linears, removed "
                f"{num_quant_removed} redundant act-quant launch(es) per forward"
            )

        info = TransformInfo(
            skipped=(num_groups == 0),
            num_matches=num_groups,
            is_clean=(num_groups == 0),
            has_valid_shapes=(num_groups == 0),
        )
        return gm, info


# ============================================================================
# DeepSeek-V4 Q-LoRA RMS-norm fusion
# ============================================================================


_DSV4_HIDDEN_WIDTH = 4096
_DSV4_Q_LORA_WIDTH = 1024
_DSV4_KV_NOPE_WIDTH = 448
_DSV4_KV_ROPE_WIDTH = 64
_DSV4_KV_WIDTH = _DSV4_KV_NOPE_WIDTH + _DSV4_KV_ROPE_WIDTH
_DSV4_QKV_WIDTH = _DSV4_Q_LORA_WIDTH + _DSV4_KV_WIDTH


def _positional_or_keyword(node: Node, position: int, name: str, default=None):
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > position:
        return node.args[position]
    return default


def _is_last_dim(dim, source: Node) -> bool:
    if dim == -1:
        return True
    val = source.meta.get("val")
    return isinstance(dim, int) and isinstance(val, torch.Tensor) and dim == val.dim() - 1


def _last_dim_interval(node: Node, producer: Node) -> Optional[Tuple[int, int]]:
    """Return the half-open producer interval represented by a view node."""
    if node is producer:
        return (0, _DSV4_QKV_WIDTH)
    if node.op == "call_method" and node.target == "contiguous":
        return _last_dim_interval(node.args[0], producer)
    if is_op(node, torch.ops.aten.contiguous):
        return _last_dim_interval(node.args[0], producer)
    if is_op(node, torch.narrow) or is_op(node, torch.ops.aten.narrow):
        source = node.args[0]
        parent = _last_dim_interval(source, producer) if isinstance(source, Node) else None
        dim = _positional_or_keyword(node, 1, "dim")
        start = _positional_or_keyword(node, 2, "start")
        length = _positional_or_keyword(node, 3, "length")
        if (
            parent is None
            or not _is_last_dim(dim, source)
            or not isinstance(start, int)
            or not isinstance(length, int)
            or start < 0
            or length < 0
            or start + length > parent[1] - parent[0]
        ):
            return None
        return (parent[0] + start, parent[0] + start + length)
    if is_op(node, operator.getitem):
        split_node = node.args[0]
        index = node.args[1]
        if not isinstance(split_node, Node) or not isinstance(index, int):
            return None
        if not is_op(split_node, torch.ops.aten.split_with_sizes):
            return None
        source = split_node.args[0]
        parent = _last_dim_interval(source, producer) if isinstance(source, Node) else None
        sizes = _positional_or_keyword(split_node, 1, "split_sizes")
        dim = _positional_or_keyword(split_node, 2, "dim", 0)
        if (
            parent is None
            or not _is_last_dim(dim, source)
            or not isinstance(sizes, (list, tuple))
            or not all(isinstance(size, int) for size in sizes)
            or any(size < 0 for size in sizes)
            or sum(sizes) != parent[1] - parent[0]
            or index < 0
            or index >= len(sizes)
        ):
            return None
        start = parent[0] + sum(sizes[:index])
        return (start, start + sizes[index])
    return None


def _is_supported_dsv4_view(node: Node) -> bool:
    return (
        (node.op == "call_method" and node.target == "contiguous")
        or is_op(node, torch.narrow)
        or is_op(node, torch.ops.aten.narrow)
        or is_op(node, torch.ops.aten.contiguous)
        or is_op(node, torch.ops.aten.split_with_sizes)
        or is_op(node, operator.getitem)
    )


def _bf16_meta_with_width(node: Node, width: int) -> bool:
    val = node.meta.get("val")
    return (
        isinstance(val, torch.Tensor)
        and val.dtype == torch.bfloat16
        and val.dim() > 0
        and val.shape[-1] == width
    )


def _match_dsv4_qkv_consumers(gm: GraphModule, producer: Node) -> Optional[Tuple[Node, Node]]:
    """Match the exact BF16 Q1024 + KV(448,64) DeepSeek-V4 projection fanout."""
    rms_op = torch.ops.auto_deploy.torch_rmsnorm
    kv_op = torch.ops.auto_deploy.deepseek_v4_kv_norm_rope_concat
    visited = {producer}
    terminals = set()
    stack = [producer]
    while stack:
        current = stack.pop()
        for user in current.users:
            if _is_supported_dsv4_view(user):
                if user.args[0] is not current:
                    return None
                if user not in visited:
                    visited.add(user)
                    stack.append(user)
            elif is_op(user, rms_op) or is_op(user, kv_op):
                terminals.add(user)
            else:
                return None

    rms_nodes = [node for node in terminals if is_op(node, rms_op)]
    kv_nodes = [node for node in terminals if is_op(node, kv_op)]
    if len(rms_nodes) != 1 or len(kv_nodes) != 1 or len(terminals) != 2:
        return None
    rms_node = rms_nodes[0]
    kv_node = kv_nodes[0]
    q_input, q_weight = extract_op_args(rms_node, "input", "weight")
    kv_nope, kv_pe, kv_weight = extract_op_args(kv_node, "nope", "pe", "weight")
    if not all(isinstance(node, Node) for node in (q_input, q_weight, kv_nope, kv_pe, kv_weight)):
        return None
    if _last_dim_interval(q_input, producer) != (0, _DSV4_Q_LORA_WIDTH):
        return None
    if _last_dim_interval(kv_nope, producer) != (
        _DSV4_Q_LORA_WIDTH,
        _DSV4_Q_LORA_WIDTH + _DSV4_KV_NOPE_WIDTH,
    ):
        return None
    if _last_dim_interval(kv_pe, producer) != (
        _DSV4_Q_LORA_WIDTH + _DSV4_KV_NOPE_WIDTH,
        _DSV4_QKV_WIDTH,
    ):
        return None
    if not (
        _bf16_meta_with_width(producer, _DSV4_QKV_WIDTH)
        and _bf16_meta_with_width(q_input, _DSV4_Q_LORA_WIDTH)
        and _bf16_meta_with_width(rms_node, _DSV4_Q_LORA_WIDTH)
        and _bf16_meta_with_width(kv_nope, _DSV4_KV_NOPE_WIDTH)
        and _bf16_meta_with_width(kv_pe, _DSV4_KV_ROPE_WIDTH)
        and _bf16_meta_with_width(kv_node, _DSV4_KV_WIDTH)
    ):
        return None
    q_weight_tensor = _resolve_attr_tensor(gm, q_weight)
    kv_weight_tensor = _resolve_attr_tensor(gm, kv_weight)
    if q_weight_tensor is None or tuple(q_weight_tensor.shape) != (_DSV4_Q_LORA_WIDTH,):
        return None
    if kv_weight_tensor is None or tuple(kv_weight_tensor.shape) != (_DSV4_KV_WIDTH,):
        return None
    return rms_node, kv_node


@TransformRegistry.register("fuse_deepseek_v4_q_rmsnorm")
class FuseDeepSeekV4QRMSNorm(BaseTransform):
    """Replace only the exact DeepSeek-V4 Q-LoRA BF16 RMS-norm decomposition.

    The transform recognizes the fused FineGrained-FP8 ``wq_a+wkv`` projection
    with weight shape ``[1536, 4096]`` and proves its complete fanout is Q1024 plus
    KV512 split as 448 no-PE and 64 RoPE dimensions. It replaces only the Q child's
    ``torch_rmsnorm`` with a fixed-BF16 one-kernel implementation. Projection
    outputs, the KV path, and the full-K prefill path remain unchanged.

    The exact DeepSeek-V4 op provenance and shapes make the transform a no-op for
    every other model, so it needs no model- or campaign-specific configuration.
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
        linear_ops = (
            torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear,
            torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant,
        )
        q_rmsnorm_op = torch.ops.auto_deploy.deepseek_v4_q_rmsnorm.default
        num_matches = 0
        for producer in list(gm.graph.nodes):
            if not any(is_op(producer, linear_op) for linear_op in linear_ops):
                continue
            weight, bias = extract_op_args(producer, "weight_quantized", "bias")
            weight_tensor = _resolve_attr_tensor(gm, weight)
            if bias is not None or weight_tensor is None:
                continue
            if tuple(weight_tensor.shape) != (_DSV4_QKV_WIDTH, _DSV4_HIDDEN_WIDTH):
                continue
            match = _match_dsv4_qkv_consumers(gm, producer)
            if match is None:
                continue
            rms_node, _ = match
            q_input, q_weight, eps = extract_op_args(rms_node, "input", "weight", "eps")
            with gm.graph.inserting_before(rms_node):
                replacement = gm.graph.call_function(
                    q_rmsnorm_op,
                    args=(q_input, q_weight, eps),
                )
            replacement.meta["val"] = rms_node.meta["val"]
            rms_node.replace_all_uses_with(replacement)
            gm.graph.erase_node(rms_node)
            num_matches += 1

        if num_matches:
            gm.graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(
                "fuse_deepseek_v4_q_rmsnorm: replaced "
                f"{num_matches} exact Q-LoRA RMS-norm decomposition(s)"
            )

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=(num_matches == 0),
            has_valid_shapes=(num_matches == 0),
        )
        return gm, info


# ============================================================================
# FineGrained FP8 gate/up projection concatenation
# ============================================================================


@TransformRegistry.register("fuse_finegrained_fp8_gate_up")
class FuseFineGrainedFP8GateUp(BaseTransform):
    """Merge sibling gate/up block-FP8 matmuls into one concatenated projection.

    Runs in post_load_fusion AFTER ``fuse_fp8_act_quant_cse``. That transform hoists a
    shared activation quant, leaving the SwiGLU gate (``w1``) and up (``w3``) projections
    as two ``torch_fake_quant_finegrained_fp8_linear_prequant`` nodes that consume the
    *same* ``(qfp8, qscale)`` pair and have identical ``[N, K]`` weight shapes (both are
    ``moe_intermediate_size x hidden_size``). The default SwiGLU matcher
    (``match_finegrained_fp8_swiglu_pattern``) cannot span the ``clamp`` + FP32-cast
    nodes DeepSeek-V4 inserts between these linears and the ``silu * mul``
    (``swiglu_limit=10``), so the pair is never fused and launches two separate,
    CTA-starved block-FP8 GEMVs on every layer at batch=1.

    This transform concatenates each such sibling pair's weight (and per-block weight
    scale) along dim 0 into one ``[2N, K]`` projection, runs a single prequant matmul,
    and slices the ``[..., 2N]`` result back into the two original ``[..., N]`` views --
    leaving the clamp / SiLU / mul / down chain that consumes gate and up byte-for-byte
    unchanged (only the two matmuls collapse to one). The block-FP8 matmul computes each
    output row independently and ``N`` is required to be a multiple of the weight scale's
    ``block_n``, so the seam lands on a block boundary and the concatenated result equals
    the two separate results element-for-element (reference-exact on the deterministic
    base kernel; the split-K decode path differs only by its own atomic-reduction
    rounding, which is orthogonal to the N-concatenation).

    Scope: only bias-free sibling groups with >= 2 members of identical weight+scale
    shape are merged. In DeepSeek-V4 this uniquely selects shared-expert ``w1``+``w3``
    (the MLA siblings ``wq_a``+``wkv`` and ``wq_b``+``indexer.wq_b`` differ in ``N``).
    Concatenation is applied to the already-sharded local weights, so it is per-rank
    correct regardless of TP layout.
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
        prequant_packet = torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant
        prequant_op = prequant_packet.default
        slice_op = torch.ops.aten.slice.Tensor

        # Group matmul-only prequant nodes by their shared (qfp8, qscale) activation
        # (args[0], args[1]). Sibling linears on one input share these two nodes after
        # fuse_fp8_act_quant_cse. Visit in topological order so members[0] is the
        # earliest node -- a valid insertion point that dominates every use.
        groups: "dict[Tuple[Node, Node], list[Node]]" = defaultdict(list)
        for node in gm.graph.nodes:
            if not is_op(node, prequant_packet):
                continue
            qinput, input_scale, bias = extract_op_args(node, "qinput", "input_scale", "bias")
            if not isinstance(qinput, Node) or not isinstance(input_scale, Node):
                continue
            if bias is not None:
                # Concatenating biased projections is out of scope; gate/up are bias-free.
                continue
            groups[(qinput, input_scale)].append(node)

        num_groups = 0
        num_matmuls_removed = 0
        fused_idx = 0
        for (qinput, input_scale), members in groups.items():
            if len(members) < 2:
                continue

            # Bucket shared-activation siblings by identical (weight_shape, scale_shape).
            # gate/up share moe_intermediate_size x hidden_size; MLA siblings differ.
            by_shape: "dict[Tuple[tuple, tuple], list[Tuple[Node, object, Node]]]" = defaultdict(
                list
            )
            for m in members:
                weight, weight_scale = extract_op_args(m, "weight_quantized", "weight_scale")
                s_node = (
                    weight_scale[0]
                    if isinstance(weight_scale, (list, tuple)) and weight_scale
                    else None
                )
                w = _resolve_attr_tensor(gm, weight)
                s = _resolve_attr_tensor(gm, s_node) if isinstance(s_node, Node) else None
                if w is None or s is None or w.dim() != 2 or s.dim() != 2:
                    continue
                N = w.shape[0]
                block_n = triton.cdiv(N, s.shape[0])
                # The seam between concatenated weights must land on a block-scale
                # boundary, otherwise the merged scale would be misapplied at the join.
                if block_n == 0 or N % block_n != 0:
                    continue
                by_shape[(tuple(w.shape), tuple(s.shape))].append((m, weight, s_node))

            for shape_members in by_shape.values():
                if len(shape_members) < 2:
                    continue

                nodes = [sm[0] for sm in shape_members]
                weight_tensors = [_resolve_attr_tensor(gm, sm[1]) for sm in shape_members]
                scale_tensors = [_resolve_attr_tensor(gm, sm[2]) for sm in shape_members]
                sizes = [w.shape[0] for w in weight_tensors]

                cat_w = torch.cat(weight_tensors, dim=0)  # [sum_N, K] float8_e4m3fn
                cat_s = torch.cat(scale_tensors, dim=0)  # [sum_N/block_n, K/block_k] float32
                w_name = f"fused_fp8_gate_up_weight_{fused_idx}"
                s_name = f"fused_fp8_gate_up_weight_scale_{fused_idx}"
                fused_idx += 1
                gm.register_buffer(w_name, cat_w.detach())
                gm.register_buffer(s_name, cat_s.detach())

                first = nodes[0]
                with gm.graph.inserting_before(first):
                    w_attr = gm.graph.get_attr(w_name)
                    s_attr = gm.graph.get_attr(s_name)
                    merged = gm.graph.call_function(
                        prequant_op, args=(qinput, input_scale, w_attr, None, [s_attr])
                    )
                    offset = 0
                    slices = []
                    for size in sizes:
                        slices.append(
                            gm.graph.call_function(
                                slice_op, args=(merged, -1, offset, offset + size)
                            )
                        )
                        offset += size

                for node, sl in zip(nodes, slices):
                    node.replace_all_uses_with(sl)
                    gm.graph.erase_node(node)

                num_groups += 1
                num_matmuls_removed += len(nodes) - 1

        if num_groups:
            ad_logger.info(
                f"fuse_finegrained_fp8_gate_up: merged {num_groups} gate/up sibling group(s), "
                f"removed {num_matmuls_removed} block-FP8 matmul launch(es) per forward"
            )

        info = TransformInfo(
            skipped=(num_groups == 0),
            num_matches=num_groups,
            is_clean=(num_groups == 0),
            has_valid_shapes=(num_groups == 0),
        )
        return gm, info


# ============================================================================
# DeepSeek-V4 shared-expert clamped-SwiGLU + down-input act-quant fusion
# ============================================================================


_SWIGLU_CAST_OPS = (
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
)
_SWIGLU_CLAMP_OPS = (
    torch.ops.aten.clamp.default,
    torch.ops.aten.clamp_max.default,
    torch.ops.aten.clamp_min.default,
)


def _sole_user(node: object) -> bool:
    return isinstance(node, Node) and len(node.users) == 1


def _cast_target_dtype(node: Node) -> Optional[torch.dtype]:
    """Target dtype of an ``aten._to_copy`` / ``aten.to.dtype`` node."""
    if "dtype" in node.kwargs:
        dtype = node.kwargs["dtype"]
        return dtype if isinstance(dtype, torch.dtype) else None
    if len(node.args) >= 2 and isinstance(node.args[1], torch.dtype):
        return node.args[1]
    return None


def _clamp_scalar_bounds(node: Node) -> Tuple[Optional[float], Optional[float]]:
    """(min, max) python-scalar bounds of a clamp/clamp_min/clamp_max node."""

    def _as_scalar(v: object) -> Optional[float]:
        return float(v) if isinstance(v, (int, float)) else None

    if is_op(node, torch.ops.aten.clamp_max):
        mx = node.kwargs.get("max", node.args[1] if len(node.args) > 1 else None)
        return None, _as_scalar(mx)
    if is_op(node, torch.ops.aten.clamp_min):
        mn = node.kwargs.get("min", node.args[1] if len(node.args) > 1 else None)
        return _as_scalar(mn), None
    args = list(node.args) + [None, None]
    mn = node.kwargs.get("min", args[1])
    mx = node.kwargs.get("max", args[2])
    return _as_scalar(mn), _as_scalar(mx)


def _match_clamped_swiglu_chain(h: object) -> Optional[dict]:
    """Match the DeepSeek-V4 shared-expert activation chain feeding a down projection.

    Expected producer chain of ``h`` (every intermediate single-user, so the rewrite
    strands it for dead-code elimination)::

        gate_src -> clamp(max=L)  -> to(f32) -> silu \
                                                      mul -> to(model_dtype) == h
        up_src   -> clamp(-L, L)  -> to(f32) ---------/

    ``gate_src`` / ``up_src`` are typically the two ``torch.narrow`` views of one
    merged gate_up projection (``fuse_gemms_mixed_children``), optionally wrapped in
    ``.contiguous()`` -- the wrapper is bypassed since the fused kernel reads strided
    views directly. Returns the fused-op arguments or None if the chain differs.
    """
    if not _sole_user(h) or not is_op(h, _SWIGLU_CAST_OPS):
        return None
    model_dtype = _cast_target_dtype(h)
    if model_dtype is None:
        return None
    mul = h.args[0]
    if not _sole_user(mul) or not is_op(mul, torch.ops.aten.mul.Tensor) or len(mul.args) < 2:
        return None
    lhs, rhs = mul.args[0], mul.args[1]
    if isinstance(lhs, Node) and is_op(lhs, torch.ops.aten.silu):
        silu, up_f32 = lhs, rhs
    elif isinstance(rhs, Node) and is_op(rhs, torch.ops.aten.silu):
        silu, up_f32 = rhs, lhs
    else:
        return None
    if not _sole_user(silu) or not _sole_user(up_f32) or not is_op(up_f32, _SWIGLU_CAST_OPS):
        return None
    gate_f32 = silu.args[0]
    if not _sole_user(gate_f32) or not is_op(gate_f32, _SWIGLU_CAST_OPS):
        return None
    if _cast_target_dtype(gate_f32) != torch.float32 or _cast_target_dtype(up_f32) != torch.float32:
        return None

    clamp_g, clamp_u = gate_f32.args[0], up_f32.args[0]
    if not _sole_user(clamp_g) or not is_op(clamp_g, _SWIGLU_CLAMP_OPS):
        return None
    if not _sole_user(clamp_u) or not is_op(clamp_u, _SWIGLU_CLAMP_OPS):
        return None
    gate_min, gate_max = _clamp_scalar_bounds(clamp_g)
    up_min, up_max = _clamp_scalar_bounds(clamp_u)
    # Gate is clamped from above only; up symmetrically -- both at the same +limit
    # (the modeling emits torch.clamp(gate, max=L) / torch.clamp(up, -L, L)).
    if gate_min is not None or gate_max is None or gate_max <= 0:
        return None
    limit = gate_max
    if up_min != -limit or up_max != limit:
        return None

    def _bypass_contiguous(src: object) -> object:
        if isinstance(src, Node) and src.op == "call_method" and src.target == "contiguous":
            return src.args[0]
        return src

    gate_src = _bypass_contiguous(clamp_g.args[0])
    up_src = _bypass_contiguous(clamp_u.args[0])
    if not isinstance(gate_src, Node) or not isinstance(up_src, Node):
        return None
    gate_val = gate_src.meta.get("val")
    up_val = up_src.meta.get("val")
    if gate_val is None or up_val is None:
        return None
    if gate_val.dim() != 2 or tuple(gate_val.shape) != tuple(up_val.shape):
        return None
    if gate_val.dtype != model_dtype or up_val.dtype != model_dtype:
        return None
    if gate_val.stride(-1) != 1 or up_val.stride(-1) != 1:
        return None
    return {"gate": gate_src, "up": up_src, "limit": limit, "width": int(gate_val.shape[-1])}


@TransformRegistry.register("fuse_fp8_swiglu_act_quant")
class FuseFP8SwigluActQuant(BaseTransform):
    """Fuse the clamped-SwiGLU + down-input act-quant chain into one kernel.

    DeepSeek-V4's shared-expert epilogue between the (merged) gate/up projection and
    the down projection runs seven tiny elementwise launches per MoE layer --
    ``clamp(gate)``, ``clamp(up)``, two FP32 casts, ``silu``, ``mul``, a cast back to
    the model dtype -- plus the down linear's internal ``_act_quant_kernel`` launch.
    All of them stream the same [tokens, moe_intermediate/tp] activation through HBM
    again and again. The default SwiGLU matcher cannot span the clamp/FP32-cast nodes
    (``swiglu_limit=10``), so this chain survives every generic fusion pass.

    This transform rewrites, per matched down projection::

        lin[_residual_add](to(silu(to_f32(clamp(g))) * to_f32(clamp(u)), dt), w2, ...)
      ->
        q, s = torch_fp8_swiglu_clamp_act_quant(g, u, limit, block_k, fmt)
        lin[_residual_add]_prequant(q, s, w2, ...)

    consuming the gate/up ``torch.narrow`` views of the merged gate_up GEMM in place
    (any ``.contiguous()`` wrappers are bypassed; the kernel reads strided views).
    The fused kernel reproduces the aten chain bit for bit -- fp32 opmath, aten's
    silu formula, NaN-propagating clamps, a model-dtype round at the reference's
    store point, and ``_act_quant_kernel``'s exact scale math -- so the down matmul
    consumes byte-identical ``(qfp8, scale)`` inputs.

    Runs in post_load_fusion AFTER ``fuse_fp8_linear_allreduce_add`` so it sees the
    down projection in its final form (plain or with the folded merge add) and keeps
    that fusion's collective-input epilogue: the residual stays a positional data
    dependency on the rewritten node, preserving the shared/routed merge point that
    stream-overlap transforms classify. A no-op when linears route to
    trtllm/deepgemm (those quantize internally).
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
        ad_ops = torch.ops.auto_deploy
        plain_lin = ad_ops.torch_fake_quant_finegrained_fp8_linear
        residual_lin = ad_ops.torch_fake_quant_finegrained_fp8_linear_residual_add
        quant_op = ad_ops.torch_fp8_swiglu_clamp_act_quant.default
        prequant_op = ad_ops.torch_fake_quant_finegrained_fp8_linear_prequant.default
        residual_prequant_op = (
            ad_ops.torch_fake_quant_finegrained_fp8_linear_residual_add_prequant.default
        )

        graph = gm.graph
        cnt = 0
        node_order = {n: i for i, n in enumerate(graph.nodes)}
        for node in list(graph.nodes):
            if is_op(node, residual_lin):
                is_residual = True
            elif is_op(node, plain_lin):
                is_residual = False
            else:
                continue
            inp, weight, bias, weight_scale, fmt = extract_op_args(
                node, "input", "weight_quantized", "bias", "weight_scale", "input_scale_fmt"
            )
            if bias is not None:
                continue
            matched = _match_clamped_swiglu_chain(inp)
            if matched is None:
                continue
            block_k = _finegrained_fp8_block_k(gm, weight, weight_scale)
            if block_k is None or matched["width"] % block_k != 0:
                continue
            residual = None
            if is_residual:
                (residual,) = extract_op_args(node, "residual")
                if not isinstance(residual, Node):
                    continue

            # Insert the fused act-quant chain right after its gate/up sources, NOT
            # at the down-projection site: with the residual-add form the down node
            # sits AFTER the routed MoE op (its residual input), and parking the
            # shared-expert tail there would pull it inside the aux-stream window
            # that multi_stream_moe brackets around the shared branch, serializing
            # shared and routed on one stream (see multi_stream_moe.py).
            anchor = (
                matched["up"]
                if node_order.get(matched["up"], 0) >= node_order.get(matched["gate"], 0)
                else matched["gate"]
            )
            with graph.inserting_after(anchor):
                act = graph.call_function(
                    quant_op,
                    args=(
                        matched["gate"],
                        matched["up"],
                        matched["limit"],
                        int(block_k),
                        fmt or "",
                    ),
                )
            with graph.inserting_after(act):
                qfp8 = graph.call_function(operator.getitem, args=(act, 0))
            with graph.inserting_after(qfp8):
                qscale = graph.call_function(operator.getitem, args=(act, 1))
            with graph.inserting_before(node):
                if is_residual:
                    new_node = graph.call_function(
                        residual_prequant_op,
                        args=(qfp8, qscale, weight, None, weight_scale, residual),
                    )
                else:
                    new_node = graph.call_function(
                        prequant_op, args=(qfp8, qscale, weight, None, weight_scale)
                    )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            cnt += 1

        if cnt:
            graph.eliminate_dead_code()
            gm.recompile()
            ad_logger.info(
                f"fuse_fp8_swiglu_act_quant: fused {cnt} clamped-SwiGLU + act-quant chain(s) "
                "into single-kernel prequant down projections"
            )

        info = TransformInfo(
            skipped=(cnt == 0),
            num_matches=cnt,
            is_clean=(cnt == 0),
            has_valid_shapes=(cnt == 0),
        )
        return gm, info
