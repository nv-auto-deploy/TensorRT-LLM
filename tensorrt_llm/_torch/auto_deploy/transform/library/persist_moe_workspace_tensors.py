# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-allocate persistent workspace tensors for MXFP4 MoE ops.

The fused ``triton_mxfp4_moe`` op (and its multi-stream ``_aux`` derivative)
internally calls ``matmul_ogs`` twice per layer — once for the gate_up
(SwiGLU-fused) GEMM and once for the down GEMM.  ``matmul_ogs`` allocates
two intermediate tensors per call when its ``y`` argument is None:

    - ``inter`` :  shape (1, num_tokens * top_k, intermediate_size)  bf16
    - ``y``     :  shape (1, num_tokens, hidden_size)                bf16

When the model runs under the ``torch-cudagraph`` compile backend the
allocations are made via the CUDA-graph private memory pool at *capture*
time, so replay does not pay an allocator cost — but the captured pool
still grows with the number of distinct allocation sites it sees across
all 24 MoE layers and across the 8 captured decode batch sizes
(``cuda_graph_config.batch_sizes = [1,2,4,8,16,32,64,128]``).  By
threading two graph-baked workspace buffers through the MoE call we
collapse the per-layer × per-batch-size allocations down to a single
shared address per role: fewer distinct addresses for the captured
graphs to track and a tighter L2 working set across layers.

The transform locates every ``triton_mxfp4_moe`` (or ``_aux`` variant)
node in the FX graph, registers two ``GraphModule`` buffers sized for the
largest captured decode batch, and replaces each call with a Python
function (``mxfp4_moe_persistent_main`` or ``..._aux``) that narrows the
workspaces to the current batch's M and threads them through ``matmul_ogs``
via its ``y=`` parameter.  The Python-function form is intentional:
``matmul_ogs`` returns a view of ``y`` when supplied, so a plain function
side-steps ``torch.library``'s no-alias rule for custom-op outputs while
still being captured cleanly by AD's monolithic CUDA graph.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.logger import ad_logger
from ...utils.multi_stream_utils import cuda_stream_manager
from ...utils.node_utils import is_op
from ..interface import BaseTransform, SharedConfig, TransformInfo, TransformRegistry

# ---------------------------------------------------------------------------
# Persistent-workspace MoE callables
# ---------------------------------------------------------------------------
#
# These are plain ``@torch._dynamo.disable`` Python functions (not custom
# ops).  matmul_ogs writes directly into the supplied ``y=`` buffer and
# returns a view of it; ``torch.library``'s no-alias contract forbids that
# for custom ops, but plain FX call_function targets are unconstrained.


@torch._dynamo.disable
def mxfp4_moe_persistent_main(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    inter_workspace: torch.Tensor,
    y_workspace: torch.Tensor,
    layer_type: str = "moe",
) -> torch.Tensor:
    return _run_persistent_core(
        hidden_states,
        router_weight,
        router_bias,
        top_k,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        inter_workspace,
        y_workspace,
    )


@torch._dynamo.disable
def mxfp4_moe_persistent_aux(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    inter_workspace: torch.Tensor,
    y_workspace: torch.Tensor,
    layer_type: str = "moe",
) -> torch.Tensor:
    """Aux-stream variant.  Mirrors ``_make_aux_stream_impl`` from
    ``multi_stream_utils`` but inlined so we don't have to derive a new
    custom op for the aux variant — these are plain FX call_function
    targets.
    """
    device = torch.cuda.current_device()
    aux_stream = cuda_stream_manager.get_stream(device, cuda_stream_manager.AUX_STREAM_NAME)
    main_event = cuda_stream_manager.get_event(device, cuda_stream_manager.MAIN_STREAM_NAME)
    aux_event = cuda_stream_manager.get_event(device, cuda_stream_manager.AUX_STREAM_NAME)

    # Record current-stream progress so aux can wait on it.  current_stream
    # rather than default_stream because we may be inside cuda graph capture.
    main_event.record(torch.cuda.current_stream(device))
    with torch.cuda.stream(aux_stream):
        aux_stream.wait_event(main_event)
        out = _run_persistent_core(
            hidden_states,
            router_weight,
            router_bias,
            top_k,
            gate_up_blocks,
            gate_up_bias,
            gate_up_scales,
            alpha,
            limit,
            down_blocks,
            down_bias,
            down_scales,
            inter_workspace,
            y_workspace,
        )
        aux_event.record(aux_stream)
    torch.cuda.current_stream(device).wait_event(aux_event)
    return out


def _run_persistent_core(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    inter_workspace: torch.Tensor,
    y_workspace: torch.Tensor,
) -> torch.Tensor:
    """MXFP4 MoE forward backed by graph-baked inter/y workspace tensors.

    Mirrors ``_run_mxfp4_mlp_core`` from ``custom_ops/fused_moe/mxfp4_moe.py``
    but threads ``y=`` through both ``matmul_ogs`` calls so the kernel writes
    into the persistent workspace instead of allocating fresh.
    """
    # Lazy imports to keep this module's import graph light.
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul_ogs,
    )
    from triton_kernels.numerics import InFlexData
    from triton_kernels.swiglu import swiglu_fn

    from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonEPRouter

    from ...custom_ops.fused_moe.mxfp4_moe import (
        _MAX_MXFP4_IMPRECISE_ACC,
        _bias_as_fp32,
        _prepare_weights_scales_cached,
    )

    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)
    M_in = x.shape[0]
    M_inter = M_in * top_k

    router_logits = F.linear(x, router_weight, router_bias)
    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = TritonEPRouter()(router_logits, top_k)

    (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    ) = _prepare_weights_scales_cached(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )

    gate_pc = PrecisionConfig(
        weight_scale=gate_up_w_scale_raw,
        flex_ctx=FlexCtx(rhs_data=InFlexData()),
        max_num_imprecise_acc=_MAX_MXFP4_IMPRECISE_ACC,
    )
    down_pc = PrecisionConfig(
        weight_scale=down_w_scale_raw,
        flex_ctx=FlexCtx(rhs_data=InFlexData()),
        max_num_imprecise_acc=_MAX_MXFP4_IMPRECISE_ACC,
    )
    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (float(alpha), float(limit)),
    )

    # Workspaces are 2D (max_M, N); we narrow to the current row count.
    # Narrowing on the leading dim of a contiguous 2D tensor yields a view
    # that is itself contiguous (strides = (N, 1)), which matmul_ogs accepts
    # via the ``output.ndim == 2`` branch in apply_allocation.
    #
    # Workspaces are sized for max captured decode batch (cuda_graph_config.
    # batch_sizes max).  Prefill batches (eager path, piecewise_enabled=false)
    # may have many more tokens than the largest captured decode bucket, so
    # fall back to None (matmul_ogs allocates fresh) when the workspace is
    # too small.  This keeps prefill correct without paying for a workspace
    # that would have to grow to max_num_tokens × top_k.
    use_workspace = inter_workspace.shape[0] >= M_inter and y_workspace.shape[0] >= M_in
    if use_workspace:
        inter_buf = inter_workspace.narrow(0, 0, M_inter)
        y_buf = y_workspace.narrow(0, 0, M_in)
    else:
        inter_buf = None
        y_buf = None

    # gate_up (with SWiGLU fused).  The SwiGLU reduction halves the output's
    # last dim, so inter has shape (M_inter, intermediate_size).
    inter = matmul_ogs(
        x,
        triton_gate_up_w,
        _bias_as_fp32(gate_up_bias),
        routing_data,
        gather_indx=gather_idx,
        precision_config=gate_pc,
        gammas=None,
        fused_activation=act,
        y=inter_buf,
    )

    # down -> y has shape (M_in, hidden_size).
    y = matmul_ogs(
        inter,
        triton_down_w,
        _bias_as_fp32(down_bias),
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=down_pc,
        gammas=routing_data.gate_scal,
        y=y_buf,
    )

    return y.reshape(*leading_shape, hidden_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BASE_OP = torch.ops.auto_deploy.triton_mxfp4_moe


def _maybe_aux_op() -> Optional[object]:
    """Return the dynamically derived ``triton_mxfp4_moe_aux`` op or None."""
    return getattr(torch.ops.auto_deploy, "triton_mxfp4_moe_aux", None)


def _is_target_moe(node: Node) -> Tuple[bool, bool]:
    """Return ``(is_match, is_aux)`` for a node that may be a MoE call.

    Accepts the bare ``triton_mxfp4_moe`` op (multi_stream_router disabled)
    or its ``_aux`` derivative (multi_stream_router enabled).  EP variants
    are intentionally NOT matched — they have a different signature.
    """
    if not is_op(node, _BASE_OP):
        aux_op = _maybe_aux_op()
        if aux_op is not None and is_op(node, aux_op):
            return True, True
        return False, False
    return True, False


def _resolve_param(gm: GraphModule, node: Node) -> Optional[torch.Tensor]:
    """Resolve a ``get_attr`` node to its underlying tensor on *gm*."""
    if not isinstance(node, Node) or node.op != "get_attr":
        return None
    target = node.target
    try:
        return gm.get_parameter(target)
    except (AttributeError, KeyError):
        pass
    return getattr(gm, target, None)


def _gate_up_intermediate_size(gm: GraphModule, gate_up_blocks_node: Node) -> Optional[int]:
    """Read intermediate_size from ``gate_up_blocks`` shape ``[E, 2I, H//32, 16]``."""
    val = gate_up_blocks_node.meta.get("val")
    if val is not None and hasattr(val, "shape") and len(val.shape) >= 2:
        twoI = int(val.shape[1])
        if twoI % 2 == 0:
            return twoI // 2
    tensor = _resolve_param(gm, gate_up_blocks_node)
    if tensor is not None and tensor.dim() >= 2 and int(tensor.shape[1]) % 2 == 0:
        return int(tensor.shape[1]) // 2
    return None


def _down_hidden_size(gm: GraphModule, down_blocks_node: Node) -> Optional[int]:
    """Read hidden_size from ``down_blocks`` shape ``[E, H, I//32, 16]``."""
    val = down_blocks_node.meta.get("val")
    if val is not None and hasattr(val, "shape") and len(val.shape) >= 2:
        return int(val.shape[1])
    tensor = _resolve_param(gm, down_blocks_node)
    if tensor is not None and tensor.dim() >= 2:
        return int(tensor.shape[1])
    return None


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


@TransformRegistry.register("persist_moe_workspace_tensors")
class PersistMoeWorkspaceTensors(BaseTransform):
    """Bake per-model MoE workspace buffers into the GraphModule.

    Replaces every ``triton_mxfp4_moe`` (and ``_aux``) call with a
    workspace-threaded equivalent so ``matmul_ogs`` writes into stable,
    graph-baked tensors instead of allocating fresh per-layer/per-batch.
    """

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph

        moe_nodes = []
        for node in graph.nodes:
            ok, is_aux = _is_target_moe(node)
            if ok:
                moe_nodes.append((node, is_aux))

        if not moe_nodes:
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Pull dimensions from the first matched op.  Op signature (per
        # mxfp4_moe.py): hidden_states, router_weight, router_bias, top_k,
        # gate_up_blocks, gate_up_bias, gate_up_scales, alpha, limit,
        # down_blocks, down_bias, down_scales, layer_type
        first_node = moe_nodes[0][0]
        if len(first_node.args) < 12:
            ad_logger.warning(
                "persist_moe_workspace_tensors: unexpected op arity "
                f"({len(first_node.args)} args); skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        gate_up_blocks_node = first_node.args[4]
        down_blocks_node = first_node.args[9]

        intermediate_size = _gate_up_intermediate_size(gm, gate_up_blocks_node)
        hidden_size = _down_hidden_size(gm, down_blocks_node)

        if intermediate_size is None or hidden_size is None:
            ad_logger.warning(
                "persist_moe_workspace_tensors: could not resolve "
                f"intermediate_size={intermediate_size} or hidden_size={hidden_size}; skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # top_k from the third positional arg.
        top_k = first_node.args[3]
        if not isinstance(top_k, int):
            ad_logger.warning(
                f"persist_moe_workspace_tensors: top_k is not an int (got {type(top_k)}); skipping."
            )
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        # Size workspaces for the largest captured decode batch (M_in_max).
        # For prefill chunks larger than max_batch_size the function will
        # narrow against a workspace that is too small; we guard that path
        # below by NOT replacing the call — instead we leave the original op
        # in place for safety.  In practice this transform runs at compile
        # stage on a graph whose live cuda-graph captures only see
        # cuda_graph_config.batch_sizes (decode), so the bound is sufficient.
        max_batch_size = int(getattr(cm.info, "max_batch_size", 0) or 0)
        if max_batch_size <= 0:
            ad_logger.warning("persist_moe_workspace_tensors: max_batch_size <= 0; skipping.")
            return gm, TransformInfo(
                skipped=False, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        max_m_inter = max_batch_size * top_k
        max_m_in = max_batch_size

        # Resolve dtype/device from the FX meta of the hidden_states input.
        first_hidden = first_node.args[0]
        meta_val = first_hidden.meta.get("val") if isinstance(first_hidden, Node) else None
        if meta_val is None or not hasattr(meta_val, "dtype") or not hasattr(meta_val, "device"):
            # Fall back to bf16 + cuda — gpt-oss inference path.
            buf_dtype = torch.bfloat16
            buf_device = torch.device("cuda", torch.cuda.current_device())
        else:
            buf_dtype = meta_val.dtype
            buf_device = meta_val.device
            if buf_device.type == "meta":
                buf_device = torch.device("cuda", torch.cuda.current_device())

        # Bake two graph-level buffers (shared across all matched layers).
        # Sharing is safe: each layer's MoE result is consumed by its
        # immediately-following residual add before the next layer's MoE
        # writes begin, so the workspaces' lifetimes are non-overlapping
        # along the captured kernel-launch sequence.
        inter_buf = torch.empty(
            (max_m_inter, intermediate_size),
            dtype=buf_dtype,
            device=buf_device,
        )
        y_buf = torch.empty(
            (max_m_in, hidden_size),
            dtype=buf_dtype,
            device=buf_device,
        )

        # Avoid name collisions with any existing buffers.
        inter_key = "persist_moe_inter_workspace"
        suffix = 0
        while hasattr(gm, inter_key):
            suffix += 1
            inter_key = f"persist_moe_inter_workspace_{suffix}"
        y_key = "persist_moe_y_workspace"
        suffix = 0
        while hasattr(gm, y_key):
            suffix += 1
            y_key = f"persist_moe_y_workspace_{suffix}"

        gm.register_buffer(inter_key, inter_buf, persistent=False)
        gm.register_buffer(y_key, y_buf, persistent=False)

        ad_logger.info(
            "persist_moe_workspace_tensors: registering buffers "
            f"{inter_key}{tuple(inter_buf.shape)} and {y_key}{tuple(y_buf.shape)}; "
            f"replacing {len(moe_nodes)} MoE node(s) "
            f"(intermediate_size={intermediate_size}, hidden_size={hidden_size}, top_k={top_k})."
        )

        # Make sure the multi_stream stream manager has registered a stream
        # for the current device — the aux variant references the singleton
        # at runtime.
        cuda_stream_manager.add_device(torch.cuda.current_device())

        num_replaced = 0

        for node, is_aux in moe_nodes:
            with graph.inserting_before(node):
                inter_attr = graph.get_attr(inter_key, torch.Tensor)
                y_attr = graph.get_attr(y_key, torch.Tensor)

            target_fn = mxfp4_moe_persistent_aux if is_aux else mxfp4_moe_persistent_main

            # New args = original args (sans trailing layer_type kwarg) +
            # workspace tensors.  ``triton_mxfp4_moe`` has 13 positional
            # args (12 mandatory + layer_type=str default); we splice
            # workspaces in BEFORE layer_type so the python function's
            # signature matches.
            orig_args = list(node.args)
            if len(orig_args) >= 13:
                layer_type_arg = orig_args[12]
                core_args = orig_args[:12]
            else:
                layer_type_arg = "moe"
                core_args = orig_args[:12]

            new_args = tuple(core_args) + (inter_attr, y_attr, layer_type_arg)

            with graph.inserting_after(node):
                new_node = graph.call_function(target_fn, args=new_args, kwargs={})

            # Carry over the FX shape meta so downstream transforms keep
            # tensor info.
            ref_val = node.meta.get("val")
            if ref_val is not None:
                new_node.meta["val"] = ref_val

            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            num_replaced += 1

        return gm, TransformInfo(
            skipped=False, num_matches=num_replaced, is_clean=False, has_valid_shapes=True
        )
