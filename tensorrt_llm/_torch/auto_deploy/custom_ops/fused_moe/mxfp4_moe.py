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

# MXFP4 MoE ops with packed expert decode and SwiGLU activation.

import weakref
from collections import OrderedDict
from collections.abc import Callable

import torch
import torch.nn.functional as F

from ..._compat import is_sm_100f

# NOTE: ``triton_kernels`` is an optional dependency. The torch-reference MXFP4 ops
# (``torch_mxfp4_moe`` etc.) must remain importable and usable without it, so all
# ``triton_kernels`` symbols are imported lazily inside the functions that need them.

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_E2M1_PACKED_VALUES = tuple(
    (_E2M1_VALUES[byte & 0x0F], _E2M1_VALUES[(byte >> 4) & 0x0F]) for byte in range(256)
)
_E8M0_EXPONENT_BIAS = 127
_MXFP4_BLOCK_SIZE = 32
_TORCH_MXFP4_ROUTED_MOE_TOKEN_CHUNK = 16

# Prepared (swizzled) triton_kernels tensors; typed as ``object`` so the module
# imports without ``triton_kernels``.
PreparedWeights = tuple[object, object, object, object]
TensorCacheKey = tuple[
    str,
    str,
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    int | None,
]
WeightCacheKey = tuple[object, ...]

# ``convert_layout`` swizzles the packed expert weights. Cache the result by
# underlying storage/view metadata so decode steps do not repeat that work.
_MXFP4_WEIGHT_CACHE: OrderedDict[WeightCacheKey, tuple[PreparedWeights, list[weakref.finalize]]] = (
    OrderedDict()
)
_E2M1_PACKED_TABLE_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


# copied from transformers.integrations.mxfp4::swizzle_mxfp4 with minor modification
def _mxfp4_value_layout(mx_axis: int):
    # Blackwell's default value layout is only supported by the persistent TMA
    # kernel. GPT-OSS MoE can select the non-persistent kernel for small shapes,
    # where unswizzled values use the native MXFP4 dot_scaled path.
    from triton_kernels.target_info import cuda_capability_geq
    from triton_kernels.tensor_details import layout
    from triton_kernels.tensor_details.layout import StridedLayout

    if cuda_capability_geq(10):
        return StridedLayout, {}
    return layout.make_default_matmul_mxfp4_w_layout(mx_axis=mx_axis)


def _swizzle_mxfp4(w, w_scale):
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details.layout import StridedLayout

    value_layout, value_layout_opts = _mxfp4_value_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


# route_fn produces ``(routing_data, gather_idx, scatter_idx)`` from triton_kernels;
# typed as ``object`` so the module imports without ``triton_kernels``.
RouteFn = Callable[[torch.Tensor], tuple[object, object, object]]


def _mxfp4_layout_cache_key() -> tuple[object, ...]:
    value_layout, value_layout_opts = _mxfp4_value_layout(mx_axis=1)
    return (
        value_layout.__module__,
        value_layout.__qualname__,
        tuple(sorted(value_layout_opts.items())),
    )


def _source_tensor(tensor: torch.Tensor) -> torch.Tensor:
    base = getattr(tensor, "_base", None)
    return base if isinstance(base, torch.Tensor) else tensor


def _tensor_cache_key(tensor: torch.Tensor) -> TensorCacheKey:
    source = _source_tensor(tensor)
    try:
        version = source._version
    except RuntimeError:
        version = None
    return (
        str(tensor.device),
        str(tensor.dtype),
        source.untyped_storage().data_ptr(),
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        version,
    )


def _detach_finalizers(finalizers: list[weakref.finalize]) -> None:
    for finalizer in finalizers:
        finalizer.detach()


def _evict_mxfp4_weight_cache_entry(key: WeightCacheKey) -> None:
    entry = _MXFP4_WEIGHT_CACHE.pop(key, None)
    if entry is None:
        return
    _, finalizers = entry
    _detach_finalizers(finalizers)


def _trim_mxfp4_weight_cache() -> None:
    max_entries = 256
    while len(_MXFP4_WEIGHT_CACHE) > max_entries:
        _, (_, finalizers) = _MXFP4_WEIGHT_CACHE.popitem(last=False)
        _detach_finalizers(finalizers)


def _clear_mxfp4_weight_cache() -> None:
    while _MXFP4_WEIGHT_CACHE:
        _, (_, finalizers) = _MXFP4_WEIGHT_CACHE.popitem()
        _detach_finalizers(finalizers)


def _register_cache_finalizers(
    key: WeightCacheKey, tensors: tuple[torch.Tensor, ...]
) -> list[weakref.finalize]:
    finalizers: list[weakref.finalize] = []
    seen_sources: set[int] = set()
    for tensor in tensors:
        source = _source_tensor(tensor)
        source_id = id(source)
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        finalizers.append(weakref.finalize(source, _evict_mxfp4_weight_cache_entry, key))
    return finalizers


def _as_uint8(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype == torch.uint8:
        return tensor
    if tensor.dtype == torch.int8:
        return tensor.view(torch.uint8)
    raise TypeError(f"{name} should contain packed MXFP4 bytes with dtype uint8 or int8.")


def _decode_e8m0_scales(scales: torch.Tensor) -> torch.Tensor:
    scales_u8 = _as_uint8(scales, "scales")
    exponents = scales_u8.to(torch.int32) - _E8M0_EXPONENT_BIAS
    return torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)


def _get_e2m1_packed_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    table = _E2M1_PACKED_TABLE_CACHE.get(key)
    if table is not None:
        return table
    if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "MXFP4 E2M1 decode table should be initialized before CUDA graph capture."
        )
    table = torch.tensor(_E2M1_PACKED_VALUES, device=device, dtype=dtype)
    _E2M1_PACKED_TABLE_CACHE[key] = table
    return table


def _decode_mxfp4_blocks(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    blocks_u8 = _as_uint8(blocks, "blocks")
    if blocks_u8.dim() < 2:
        raise ValueError("MXFP4 blocks should have at least block and packed-byte dimensions.")
    if blocks_u8.shape[-1] * 2 != _MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 blocks should pack {_MXFP4_BLOCK_SIZE} values per scale block, "
            f"got last dimension {blocks_u8.shape[-1]}."
        )
    if scales.shape != blocks_u8.shape[:-1]:
        raise ValueError(
            f"MXFP4 scales shape {tuple(scales.shape)} should match blocks shape "
            f"{tuple(blocks_u8.shape[:-1])} without the packed-byte dimension."
        )

    table = _get_e2m1_packed_table(blocks.device, torch.float32)
    values = table[blocks_u8.to(torch.int64)].reshape(*blocks_u8.shape[:-1], _MXFP4_BLOCK_SIZE)
    decoded = values * _decode_e8m0_scales(scales).unsqueeze(-1)
    return decoded.reshape(*blocks_u8.shape[:-2], blocks_u8.shape[-2] * _MXFP4_BLOCK_SIZE)


def _split_gate_up(
    gate_up: torch.Tensor,
    gate_up_order: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate_up_order in ("interleaved", "gpt_oss_interleaved"):
        return gate_up[..., ::2], gate_up[..., 1::2]

    gate_up_size = gate_up.shape[-1]
    if gate_up_size % 2 != 0:
        raise ValueError(f"gate_up output size should be even, got {gate_up_size}.")
    midpoint = gate_up_size // 2
    first, second = gate_up[..., :midpoint], gate_up[..., midpoint:]
    if gate_up_order in ("up_gate", "w3_w1"):
        return second, first
    if gate_up_order in ("gate_up", "w1_w3"):
        return first, second
    raise ValueError(f"Unsupported gate_up_order: {gate_up_order}.")


def _apply_swiglu(
    gate_up: torch.Tensor,
    alpha: float,
    limit: float,
    gate_up_order: str,
    swiglu_mode: str,
) -> torch.Tensor:
    gate, up = _split_gate_up(gate_up, gate_up_order)
    if swiglu_mode == "gpt_oss" or limit > 0.0:
        gate = gate.clamp(max=float(limit))
        up = up.clamp(min=-float(limit), max=float(limit))
    gate = gate * torch.sigmoid(gate * float(alpha))
    if swiglu_mode == "deepseek":
        return gate * up
    if swiglu_mode == "gpt_oss":
        return gate * (up + 1.0)
    raise ValueError(f"Unsupported swiglu_mode: {swiglu_mode}.")


def _router_topk(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k <= 0:
        raise ValueError(f"top_k should be positive, got {top_k}.")

    bias = router_bias.to(torch.float32) if router_bias.numel() > 0 else None
    router_logits = F.linear(hidden_states.to(torch.float32), router_weight.to(torch.float32), bias)
    if top_k > router_logits.shape[-1]:
        raise ValueError(f"top_k={top_k} should be <= number of experts {router_logits.shape[-1]}.")
    router_top_value, router_indices = torch.topk(router_logits, top_k, dim=-1)
    routing_weights = torch.softmax(router_top_value, dim=-1, dtype=torch.float32)
    return router_indices, routing_weights


def _split_range_last_remainder(num_experts: int, world_size: int, rank: int) -> tuple[int, int]:
    if world_size <= 0:
        raise ValueError(f"ep_size should be positive, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"ep_rank should be in [0, {world_size}), got {rank}.")
    base = num_experts // world_size
    lo = base * rank
    hi = num_experts if rank == world_size - 1 else base * (rank + 1)
    return lo, hi


def _check_decoded_mxfp4_shapes(
    gate_up_weight: torch.Tensor, down_weight: torch.Tensor, hidden_size: int
) -> None:
    if gate_up_weight.shape[-1] != hidden_size:
        raise ValueError(
            f"Decoded gate/up MXFP4 weight hidden dimension {gate_up_weight.shape[-1]} "
            f"does not match hidden states dimension {hidden_size}."
        )
    if down_weight.shape[-2] != hidden_size:
        raise ValueError(
            f"Decoded down MXFP4 weight output dimension {down_weight.shape[-2]} "
            f"does not match hidden states dimension {hidden_size}."
        )


def _run_torch_mxfp4_dense_slots(
    x: torch.Tensor,
    leading_shape: torch.Size,
    local_expert_idx: torch.Tensor,
    valid_route: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    alpha: float,
    limit: float,
    gate_up_order: str,
    swiglu_mode: str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """All local experts already dequantized; index_select the routed expert per slot.

    Used for large token counts (prefill / warmup), where dequantizing the full
    expert set once and reusing it across tokens is cheaper than gathering and
    dequantizing ``num_tokens * top_k`` slots. ``local_expert_idx`` is the clamped
    [num_tokens, top_k] local index; ``valid_route`` masks out non-local routes.
    """
    hidden_size = x.shape[-1]
    output = torch.zeros((x.shape[0], hidden_size), device=x.device, dtype=torch.float32)

    x_for_bmm = x.unsqueeze(-1)
    for route_idx in range(local_expert_idx.shape[1]):
        for start in range(0, x.shape[0], _TORCH_MXFP4_ROUTED_MOE_TOKEN_CHUNK):
            end = min(start + _TORCH_MXFP4_ROUTED_MOE_TOKEN_CHUNK, x.shape[0])
            token_slice = slice(start, end)
            expert_idx = local_expert_idx[token_slice, route_idx]
            gate_up = torch.bmm(
                gate_up_weight.index_select(0, expert_idx), x_for_bmm[token_slice]
            ).squeeze(-1)
            gate_up = gate_up + gate_up_bias.index_select(0, expert_idx).to(torch.float32)
            inter = _apply_swiglu(gate_up, alpha, limit, gate_up_order, swiglu_mode)
            expert_output = torch.bmm(
                down_weight.index_select(0, expert_idx), inter.unsqueeze(-1)
            ).squeeze(-1)
            expert_output = expert_output + down_bias.index_select(0, expert_idx).to(torch.float32)
            route_scale = routing_weights[token_slice, route_idx, None] * valid_route[
                token_slice, route_idx, None
            ].to(torch.float32)
            output[token_slice] = output[token_slice] + expert_output * route_scale

    return output.reshape(*leading_shape, hidden_size).to(output_dtype)


def _run_torch_mxfp4_selected_slots(
    x: torch.Tensor,
    leading_shape: torch.Size,
    top_k: int,
    valid_route: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    alpha: float,
    limit: float,
    gate_up_order: str,
    swiglu_mode: str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Per-slot dequantized weights/biases laid out [num_tokens * top_k, ...].

    One row per (token, route) slot. The slot's expert weight is selected by a plain
    slice of the [num_tokens, top_k, ...] view instead of an index_select over the
    full expert set.
    """
    hidden_size = x.shape[-1]
    num_tokens = x.shape[0]
    output = torch.zeros((num_tokens, hidden_size), device=x.device, dtype=torch.float32)

    gate_up_weight = gate_up_weight.reshape(num_tokens, top_k, *gate_up_weight.shape[1:])
    down_weight = down_weight.reshape(num_tokens, top_k, *down_weight.shape[1:])
    gate_up_bias = gate_up_bias.reshape(num_tokens, top_k, -1)
    down_bias = down_bias.reshape(num_tokens, top_k, -1)

    x_for_bmm = x.unsqueeze(-1)
    for route_idx in range(top_k):
        for start in range(0, num_tokens, _TORCH_MXFP4_ROUTED_MOE_TOKEN_CHUNK):
            end = min(start + _TORCH_MXFP4_ROUTED_MOE_TOKEN_CHUNK, num_tokens)
            token_slice = slice(start, end)
            gate_up = torch.bmm(
                gate_up_weight[token_slice, route_idx], x_for_bmm[token_slice]
            ).squeeze(-1)
            gate_up = gate_up + gate_up_bias[token_slice, route_idx]
            inter = _apply_swiglu(gate_up, alpha, limit, gate_up_order, swiglu_mode)
            expert_output = torch.bmm(
                down_weight[token_slice, route_idx], inter.unsqueeze(-1)
            ).squeeze(-1)
            expert_output = expert_output + down_bias[token_slice, route_idx]
            route_scale = routing_weights[token_slice, route_idx, None] * valid_route[
                token_slice, route_idx, None
            ].to(torch.float32)
            output[token_slice] = output[token_slice] + expert_output * route_scale

    return output.reshape(*leading_shape, hidden_size).to(output_dtype)


def _run_torch_mxfp4_mlp_core(
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
    expert_start: int = 0,
) -> torch.Tensor:
    selected_experts, routing_weights = _router_topk(
        hidden_states.reshape(-1, hidden_states.shape[-1]),
        router_weight,
        router_bias,
        top_k,
    )
    return _run_torch_mxfp4_from_routing_core(
        hidden_states,
        selected_experts,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        expert_start=expert_start,
        gate_up_order="interleaved",
        swiglu_mode="gpt_oss",
    )


# ---------------------------------------------------------------------------
# TRT-LLM-Gen W4A16 (bf16-act) MXFP4 MoE decode path (Blackwell / SM100)
# ---------------------------------------------------------------------------
# DSV4-Flash feeds *precomputed* routing (selected_experts / routing_weights, its
# custom sqrtsoftplus+hash routing stays upstream) to ``torch_mxfp4_moe_from_routing(_ep)``.
# The torch reference then dequantizes the routed expert slots and runs the SwiGLU MLP in
# fp32 bmm — the #1 decode-time op (idea_0002 routed it off the full-expert dequant; the
# remaining per-slot dequant+bmm is the dominant residual). The trtllm-gen W4A16 runner
# ``bf16_mxe2m1_block_scale_moe_runner`` does the same math in one hand-tuned cubin and is
# 21-28x faster at DSV4 EP=8 decode shapes (idea_0008 microbench), 17x faster than matmul_ogs
# (which regressed +34%, idea_0005). The kernel consumes mxfp4 weights + bf16 activations
# directly; precomputed routing is forwarded as topk_ids/topk_weights (used verbatim — PT folds
# routed_scaling_factor into token_final_scales the same way, routing.py:380/419).

# Kernel-ready (pad/shard/shuffle) weights + per-expert SwiGLU tensors, keyed by raw-weight
# identity. The prep is expensive (per-expert TMA-layout shuffle) and must run once before
# CUDA-graph capture; captured decode steps reuse the cache, mirroring
# ``_prepare_weights_scales_cached`` for the matmul_ogs path.
_TRTLLM_GEN_MXFP4_CACHE: OrderedDict[WeightCacheKey, tuple[object, list[weakref.finalize]]] = (
    OrderedDict()
)


def _reinterleave_up_gate(t: torch.Tensor, intermediate_size: int) -> torch.Tensor:
    """Convert DSV4 ``up_gate`` split-half (``[up(I) | gate(I)]`` on the 2I axis) into the
    HF-interleaved layout (gate at even rows, up at odd rows) that
    :func:`prepare_trtllm_gen_moe_mxfp4_weights` (gpt-oss layout) expects.

    The torch reference (``gate_up_order='up_gate'``) reads ``up = t[:, :I]`` /
    ``gate = t[:, I:]``; prepare's ``_deinterleave_gate_up`` reads ``gate = t[:, 0::2]`` /
    ``up = t[:, 1::2]``. Re-interleaving so even=gate, odd=up makes prepare see the same
    gate/up halves the reference does — THE fix that lifts cos 0.012 -> 0.977 (idea_0008).
    Handles weights ``[E, 2I, H/32, 16]``, scales ``[E, 2I, H/32]`` and bias ``[E, 2I]``.
    """
    up = t[:, :intermediate_size]
    gate = t[:, intermediate_size:]
    interleaved = torch.stack([gate, up], dim=2)  # [E, I, 2, ...] -> rows g0,u0,g1,u1,...
    return interleaved.reshape(t.shape[0], 2 * intermediate_size, *t.shape[2:]).contiguous()


def _prepare_trtllm_gen_mxfp4_cached(
    hidden_size: int,
    intermediate_size: int,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    alpha: float,
    limit: float,
):
    """Re-interleave DSV4 gate/up + run :func:`prepare_trtllm_gen_moe_mxfp4_weights`, cached.

    Returns ``(prepared_weights, swiglu_alpha, swiglu_beta, swiglu_limit)``. The SwiGLU params
    (alpha from the op, beta=0, limit from the op) make the kernel's gated activation
    ``clamp(gate, max=limit) * sigmoid(alpha * clamp(gate, max=limit)) * clamp(up, ±limit)``
    bit-equivalent to the torch reference's deepseek SwiGLU (``swiglu_mode='deepseek'`` with the
    same clamp), so values stay faithful even when activations exceed ``limit``.
    """
    raw_tensors = (
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias,
        down_blocks,
        down_scales,
        down_bias,
    )
    key: WeightCacheKey = (
        "trtllm_gen_w4a16",
        hidden_size,
        intermediate_size,
        float(alpha),
        float(limit),
        *(_tensor_cache_key(tensor) for tensor in raw_tensors),
    )
    entry = _TRTLLM_GEN_MXFP4_CACHE.get(key)
    if entry is not None:
        _TRTLLM_GEN_MXFP4_CACHE.move_to_end(key)
        return entry[0]

    from .prepare_trtllm_gen_moe_mxfp4_weights import prepare_trtllm_gen_moe_mxfp4_weights

    gu_blocks_i = _reinterleave_up_gate(gate_up_blocks, intermediate_size)
    gu_scales_i = _reinterleave_up_gate(gate_up_scales, intermediate_size)
    gu_bias_i = _reinterleave_up_gate(gate_up_bias, intermediate_size)
    prepared = prepare_trtllm_gen_moe_mxfp4_weights(
        gu_blocks_i,
        gu_scales_i,
        gu_bias_i,
        down_blocks,
        down_scales,
        down_bias,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tp_size=1,
        tp_rank=0,
    )
    e_local = gate_up_blocks.shape[0]
    dev = gate_up_blocks.device
    # deepseek SwiGLU: silu-style gate (alpha, no +1 term -> beta=0) with the op's clamp limit.
    # limit<=0 in the reference means "no clamp"; the kernel uses +inf for that.
    kernel_limit = float(limit) if limit > 0.0 else float("inf")
    swiglu_alpha = torch.full((e_local,), float(alpha), dtype=torch.float32, device=dev)
    swiglu_beta = torch.zeros(e_local, dtype=torch.float32, device=dev)
    swiglu_limit = torch.full((e_local,), kernel_limit, dtype=torch.float32, device=dev)
    bundle = (prepared, swiglu_alpha, swiglu_beta, swiglu_limit)

    _TRTLLM_GEN_MXFP4_CACHE[key] = (bundle, _register_cache_finalizers(key, raw_tensors))
    while len(_TRTLLM_GEN_MXFP4_CACHE) > 256:
        _, (_, finalizers) = _TRTLLM_GEN_MXFP4_CACHE.popitem(last=False)
        _detach_finalizers(finalizers)
    return bundle


def _run_trtllm_gen_mxfp4_from_routing(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    expert_start: int,
    alpha: float = 1.0,
    limit: float = float("inf"),
) -> torch.Tensor:
    """Run the routed MXFP4 MLP via the trtllm-gen W4A16 bf16-act runner.

    Produces this rank's *local* partial output (tokens routed to off-rank experts contribute
    zero, exactly like the torch reference's ``valid_route`` masking); the sharding transform's
    all_reduce sums local partials across EP ranks. Mirrors the trtllm-gen EP-allreduce pattern
    used by the NVFP4/FP8 paths (local expert coords, ``local_expert_offset=0``).
    """
    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x2d = hidden_states.reshape(-1, hidden_size).to(torch.bfloat16)
    num_tokens = x2d.shape[0]

    local_experts = gate_up_blocks.shape[0]
    intermediate_size = gate_up_blocks.shape[1] // 2  # gate_up "2I" rows / 2

    prepared, sg_alpha, sg_beta, sg_limit = _prepare_trtllm_gen_mxfp4_cached(
        hidden_size,
        intermediate_size,
        gate_up_blocks,
        gate_up_scales,
        gate_up_bias,
        down_blocks,
        down_bias,
        down_scales,
        alpha,
        limit,
    )

    # Global -> local expert coords + remote-route masking. Off-rank routes are tagged with the
    # invalid sentinel ``expert_id == local_experts`` (== num_experts here) so the kernel SKIPS
    # them — matching the trtllm-gen EP convention (``invalid_expert_id = num_experts`` in
    # ``_run_moe_with_alltoall``). Clamping them onto a valid expert instead floods that expert's
    # routing histogram and drops real routes (capacity overflow), under-computing the output.
    sel = selected_experts.reshape(num_tokens, -1)
    top_k = sel.shape[-1]
    # Fuse the global->local subtraction, range mask, invalid-sentinel select, int32 cast,
    # and routing-weight mask/bf16-cast (plus the upstream model-side bf16 cast) into one
    # Triton program. Off-rank / out-of-range routes get the sentinel ``local_experts`` so
    # the runner SKIPS them; valid weights are bit-identical bf16, invalid ones are bf16(0).
    local_idx, weights = torch.ops.auto_deploy.deepseek_v4_localize_routing(
        sel,
        routing_weights.reshape(num_tokens, top_k),
        int(expert_start),
        int(local_experts),
    )

    # Pad activations to the kernel's expected (padded) hidden dim (multiple of 512).
    expected_hidden = int(prepared.fc1_weights_mxfp4.shape[-1] * 2)
    if expected_hidden > hidden_size:
        x2d = torch.nn.functional.pad(x2d, (0, expected_hidden - hidden_size))

    intermediate_size_padded = int(prepared.fc1_weights_mxfp4.shape[1] // 2)
    result = torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner(
        None,  # routing_logits (precomputed routing -> None)
        None,  # routing_bias
        x2d,
        prepared.fc1_weights_mxfp4,
        prepared.fc1_weights_scale_ue8m0,
        prepared.fc1_bias_f32,
        sg_alpha,
        sg_beta,
        sg_limit,
        prepared.fc2_weights_mxfp4,
        prepared.fc2_weights_scale_ue8m0,
        prepared.fc2_bias_f32,
        local_experts,  # num_experts (local routing space; topk_ids in [0, local_experts))
        int(top_k),
        None,  # n_group
        None,  # topk_group
        intermediate_size_padded,
        prepared.valid_hidden_size,
        prepared.valid_intermediate_size,
        0,  # local_expert_offset (local expert coords)
        local_experts,  # local_num_experts
        None,  # routed_scaling_factor (already folded into routing_weights upstream)
        1,  # routing_method_type = Renormalize (no-op: topk precomputed)
        0,  # act_type = SwiGlu
        topk_weights=weights,
        topk_ids=local_idx,
    )

    valid_h = prepared.valid_hidden_size
    if result.shape[-1] > valid_h:
        result = result[..., :valid_h].contiguous()
    return result.reshape(*leading_shape, valid_h).to(hidden_states.dtype)


def _run_torch_mxfp4_from_routing_core(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    expert_start: int = 0,
    gate_up_order: str = "up_gate",
    swiglu_mode: str = "deepseek",
) -> torch.Tensor:
    # Blackwell fast path: the DSV4 (up_gate / deepseek-SwiGLU) routed MXFP4 MLP runs on the
    # trtllm-gen W4A16 bf16-act runner instead of the fp32 dequant+bmm reference. gpt-oss
    # (interleaved / gpt_oss-SwiGLU) and non-SM100 keep the torch reference below.
    if is_sm_100f() and gate_up_order == "up_gate" and swiglu_mode == "deepseek":
        return _run_trtllm_gen_mxfp4_from_routing(
            hidden_states,
            selected_experts,
            routing_weights,
            gate_up_blocks,
            gate_up_bias,
            gate_up_scales,
            down_blocks,
            down_bias,
            down_scales,
            expert_start,
            alpha,
            limit,
        )

    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size).to(torch.float32)
    num_tokens = x.shape[0]

    selected_experts = selected_experts.reshape(num_tokens, -1).to(torch.int64)
    routing_weights = routing_weights.reshape_as(selected_experts).to(torch.float32)
    top_k = selected_experts.shape[-1]
    local_experts = gate_up_blocks.shape[0]
    if local_experts <= 0:
        raise ValueError("MXFP4 MoE requires at least one local expert.")

    local_expert_idx = selected_experts - int(expert_start)
    valid_route = (local_expert_idx >= 0) & (local_expert_idx < local_experts)
    local_expert_idx = local_expert_idx.clamp(0, local_experts - 1)

    # Decode regime: when the routing references fewer expert slots than there are
    # local experts (num_tokens * top_k < local_experts), gather each slot's *packed*
    # (uint8) blocks and dequantize only those slots instead of materializing the full
    # [local_experts, ...] fp32 weight every step. The (data-volume-bound)
    # `table[blocks]` dequant then runs over num_tokens * top_k slots — the dominant
    # per-step decode cost. `_decode_mxfp4_blocks` is elementwise over the leading
    # expert dim, so decode(blocks[idx]) == decode(blocks)[idx]: bit-identical to
    # dequantizing the full expert set and index_select-ing afterwards. For large
    # token counts (prefill / warmup), dequantizing all experts once and reusing them
    # across tokens is cheaper, so fall back to the dense path there.
    if num_tokens * top_k < local_experts:
        slot_idx = local_expert_idx.reshape(-1)
        gate_up_weight = _decode_mxfp4_blocks(
            gate_up_blocks.index_select(0, slot_idx),
            gate_up_scales.index_select(0, slot_idx),
        )
        down_weight = _decode_mxfp4_blocks(
            down_blocks.index_select(0, slot_idx),
            down_scales.index_select(0, slot_idx),
        )
        _check_decoded_mxfp4_shapes(gate_up_weight, down_weight, hidden_size)
        return _run_torch_mxfp4_selected_slots(
            x,
            leading_shape,
            top_k,
            valid_route,
            routing_weights,
            gate_up_weight,
            gate_up_bias.index_select(0, slot_idx).to(torch.float32),
            down_weight,
            down_bias.index_select(0, slot_idx).to(torch.float32),
            alpha,
            limit,
            gate_up_order,
            swiglu_mode,
            hidden_states.dtype,
        )

    gate_up_weight = _decode_mxfp4_blocks(gate_up_blocks, gate_up_scales)
    down_weight = _decode_mxfp4_blocks(down_blocks, down_scales)
    _check_decoded_mxfp4_shapes(gate_up_weight, down_weight, hidden_size)
    return _run_torch_mxfp4_dense_slots(
        x,
        leading_shape,
        local_expert_idx,
        valid_route,
        routing_weights,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        alpha,
        limit,
        gate_up_order,
        swiglu_mode,
        hidden_states.dtype,
    )


def _prepare_weights_scales(
    hidden_size: int,
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in unit8
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in unit8
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
):
    local_experts = gate_up_blocks.size(0)
    intermediate_size = gate_up_blocks.shape[1] // 2

    # canon shapes for swizzling (use last two dims as [K, N] style)
    gate_up_blocks = gate_up_blocks.view(local_experts, intermediate_size * 2, -1)
    triton_gate_up_w, gate_up_w_scale_raw = _swizzle_mxfp4(
        gate_up_blocks.transpose(-2, -1), gate_up_scales.transpose(-2, -1)
    )
    triton_gate_up_w.shape = torch.Size([local_experts, hidden_size, intermediate_size * 2])

    down_blocks = down_blocks.view(local_experts, -1, intermediate_size // 2)
    triton_down_w, down_w_scale_raw = _swizzle_mxfp4(
        down_blocks.transpose(-2, -1), down_scales.transpose(-2, -1)
    )
    triton_down_w.shape = torch.Size([local_experts, intermediate_size, hidden_size])

    return (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    )


def _prepare_weights_scales_cached(
    hidden_size: int,
    gate_up_blocks: torch.Tensor,
    gate_up_scales: torch.Tensor,
    down_blocks: torch.Tensor,
    down_scales: torch.Tensor,
) -> PreparedWeights:
    raw_tensors = (gate_up_blocks, gate_up_scales, down_blocks, down_scales)
    key: WeightCacheKey = (
        hidden_size,
        _mxfp4_layout_cache_key(),
        *(_tensor_cache_key(tensor) for tensor in raw_tensors),
    )

    entry = _MXFP4_WEIGHT_CACHE.get(key)
    if entry is not None:
        _MXFP4_WEIGHT_CACHE.move_to_end(key)
        prepared_weights, _ = entry
        return prepared_weights

    prepared_weights = _prepare_weights_scales(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )
    _MXFP4_WEIGHT_CACHE[key] = (
        prepared_weights,
        _register_cache_finalizers(key, raw_tensors),
    )
    _trim_mxfp4_weight_cache()
    return prepared_weights


def _run_mxfp4_mlp_core(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    route_fn: RouteFn,  # injects routing variant
) -> torch.Tensor:
    """
    Shared core for both triton_mxfp4_moe and triton_mxfp4_moe_ep.
    - route_fn encapsulates the only difference: how we produce (routing_data, gather_idx, scatter_idx).
    """
    from triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
        matmul_ogs,
    )
    from triton_kernels.numerics import InFlexData
    from triton_kernels.swiglu import swiglu_fn

    leading_shape = hidden_states.shape[:-1]
    hidden_size = hidden_states.shape[-1]
    x = hidden_states.reshape(-1, hidden_size)

    router_logits = F.linear(x, router_weight, router_bias)
    # route (global vs EP-aware)
    with torch.cuda.device(router_logits.device):
        routing_data, gather_idx, scatter_idx = route_fn(router_logits)

    (
        triton_gate_up_w,
        gate_up_w_scale_raw,
        triton_down_w,
        down_w_scale_raw,
    ) = _prepare_weights_scales_cached(
        hidden_size, gate_up_blocks, gate_up_scales, down_blocks, down_scales
    )

    gate_pc = PrecisionConfig(
        weight_scale=gate_up_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )
    down_pc = PrecisionConfig(
        weight_scale=down_w_scale_raw, flex_ctx=FlexCtx(rhs_data=InFlexData())
    )

    act = FusedActivation(
        FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
        (float(alpha), float(limit)),
    )

    # gate_up (with SWiGLU fused)
    inter = matmul_ogs(
        x,
        triton_gate_up_w,
        gate_up_bias.to(torch.float32),
        routing_data,
        gather_indx=gather_idx,
        precision_config=gate_pc,
        gammas=None,
        fused_activation=act,
    )

    # down
    y = matmul_ogs(
        inter,
        triton_down_w,
        down_bias.to(torch.float32),
        routing_data,
        scatter_indx=scatter_idx,
        precision_config=down_pc,
        gammas=routing_data.gate_scal,
    )

    y = y.reshape(*leading_shape, hidden_size)
    return y


@torch.library.custom_op("auto_deploy::triton_mxfp4_moe", mutates_args=())
def triton_mxfp4_moe(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router
    router_weight: torch.Tensor,  # [E, H]
    router_bias: torch.Tensor,  # [E]
    top_k: int,
    # gate_up path
    gate_up_blocks: torch.Tensor,  # [E, 2I, H//32, 16] in unit8
    gate_up_bias: torch.Tensor,  # [E, 2I]
    gate_up_scales: torch.Tensor,  # [E, 2I, H//32] in unit8
    alpha: float,
    limit: float,
    # down path
    down_blocks: torch.Tensor,  # [E, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E, H]
    down_scales: torch.Tensor,  # [E, H, I//32] in uint8
    layer_type: str = "moe",
) -> torch.Tensor:
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonEPRouter

    def _global_route_fn(logits: torch.Tensor):
        # routing() removed in triton_kernels 3.6.0
        # TritonEPRouter(ep=1) is equivalent
        return TritonEPRouter()(logits, top_k)

    return _run_mxfp4_mlp_core(
        hidden_states,
        router_weight,
        router_bias,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        route_fn=_global_route_fn,
    )


@triton_mxfp4_moe.register_fake
def _mxfp4_mlp_fake(
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
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::torch_mxfp4_moe", mutates_args=())
def torch_mxfp4_moe(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router
    router_weight: torch.Tensor,  # [E, H]
    router_bias: torch.Tensor,  # [E]
    top_k: int,
    # gate_up path
    gate_up_blocks: torch.Tensor,  # [E, 2I, H//32, 16] in uint8
    gate_up_bias: torch.Tensor,  # [E, 2I]
    gate_up_scales: torch.Tensor,  # [E, 2I, H//32] in uint8
    alpha: float,
    limit: float,
    # down path
    down_blocks: torch.Tensor,  # [E, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E, H]
    down_scales: torch.Tensor,  # [E, H, I//32] in uint8
    layer_type: str = "moe",
) -> torch.Tensor:
    return _run_torch_mxfp4_mlp_core(
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
    )


@torch_mxfp4_moe.register_fake
def _torch_mxfp4_mlp_fake(
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
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::torch_mxfp4_moe_from_routing", mutates_args=())
def torch_mxfp4_moe_from_routing(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    selected_experts: torch.Tensor,  # [B*S, top_k]
    routing_weights: torch.Tensor,  # [B*S, top_k]
    # gate_up path
    gate_up_blocks: torch.Tensor,  # [E, 2I, H//32, 16] in uint8
    gate_up_bias: torch.Tensor,  # [E, 2I]
    gate_up_scales: torch.Tensor,  # [E, 2I, H//32] in uint8
    alpha: float,
    limit: float,
    # down path
    down_blocks: torch.Tensor,  # [E, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E, H]
    down_scales: torch.Tensor,  # [E, H, I//32] in uint8
    gate_up_order: str = "up_gate",
    swiglu_mode: str = "deepseek",
    layer_type: str = "moe",
) -> torch.Tensor:
    return _run_torch_mxfp4_from_routing_core(
        hidden_states,
        selected_experts,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        gate_up_order=gate_up_order,
        swiglu_mode=swiglu_mode,
    )


@torch_mxfp4_moe_from_routing.register_fake
def _torch_mxfp4_mlp_from_routing_fake(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    gate_up_order: str = "up_gate",
    swiglu_mode: str = "deepseek",
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::triton_mxfp4_moe_ep", mutates_args=())
def triton_mxfp4_moe_ep(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router (replicated across EP)
    router_weight: torch.Tensor,  # [E_total, H]
    router_bias: torch.Tensor,  # [E_total]
    top_k: int,
    # expert params (already sharded along dim 0)
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in unit8
    gate_up_bias: torch.Tensor,  # [E_local, 2I]
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in unit8
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E_local, H]
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
    # EP topology
    ep_size: int,
    ep_rank: int,
    layer_type: str = "moe",
) -> torch.Tensor:
    from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import TritonEPRouter

    triton_ep_router = TritonEPRouter()

    def _ep_route_fn(logits: torch.Tensor):
        return triton_ep_router(logits, top_k, ep=ep_size, node_idx=ep_rank)

    return _run_mxfp4_mlp_core(
        hidden_states,
        router_weight,
        router_bias,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        route_fn=_ep_route_fn,
    )


@triton_mxfp4_moe_ep.register_fake
def _mxfp4_mlp_ep_fake(
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
    ep_size: int,
    ep_rank: int,
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::torch_mxfp4_moe_from_routing_ep", mutates_args=())
def torch_mxfp4_moe_from_routing_ep(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    selected_experts: torch.Tensor,  # [B*S, top_k]
    routing_weights: torch.Tensor,  # [B*S, top_k]
    # expert params (already sharded along dim 0)
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in uint8
    gate_up_bias: torch.Tensor,  # [E_local, 2I]
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in uint8
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E_local, H]
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
    gate_up_order: str = "up_gate",
    swiglu_mode: str = "deepseek",
    layer_type: str = "moe",
    expert_start: int = 0,
) -> torch.Tensor:
    return _run_torch_mxfp4_from_routing_core(
        hidden_states,
        selected_experts,
        routing_weights,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        expert_start=expert_start,
        gate_up_order=gate_up_order,
        swiglu_mode=swiglu_mode,
    )


@torch_mxfp4_moe_from_routing_ep.register_fake
def _torch_mxfp4_mlp_from_routing_ep_fake(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_blocks: torch.Tensor,
    gate_up_bias: torch.Tensor,
    gate_up_scales: torch.Tensor,
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,
    down_bias: torch.Tensor,
    down_scales: torch.Tensor,
    gate_up_order: str = "up_gate",
    swiglu_mode: str = "deepseek",
    layer_type: str = "moe",
    expert_start: int = 0,
):
    return torch.empty_like(hidden_states)


@torch.library.custom_op("auto_deploy::torch_mxfp4_moe_ep", mutates_args=())
def torch_mxfp4_moe_ep(
    hidden_states: torch.Tensor,  # [B, S, H] or [B*S, H]
    # router (replicated across EP)
    router_weight: torch.Tensor,  # [E_total, H]
    router_bias: torch.Tensor,  # [E_total]
    top_k: int,
    # expert params (already sharded along dim 0)
    gate_up_blocks: torch.Tensor,  # [E_local, 2I, H//32, 16] in uint8
    gate_up_bias: torch.Tensor,  # [E_local, 2I]
    gate_up_scales: torch.Tensor,  # [E_local, 2I, H//32] in uint8
    alpha: float,
    limit: float,
    down_blocks: torch.Tensor,  # [E_local, H, I//32, 16] in uint8
    down_bias: torch.Tensor,  # [E_local, H]
    down_scales: torch.Tensor,  # [E_local, H, I//32] in uint8
    # EP topology
    ep_size: int,
    ep_rank: int,
    layer_type: str = "moe",
) -> torch.Tensor:
    expert_start, _ = _split_range_last_remainder(router_weight.shape[0], ep_size, ep_rank)
    return _run_torch_mxfp4_mlp_core(
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
        expert_start=expert_start,
    )


@torch_mxfp4_moe_ep.register_fake
def _torch_mxfp4_mlp_ep_fake(
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
    ep_size: int,
    ep_rank: int,
    layer_type: str = "moe",
):
    return torch.empty_like(hidden_states)
