# Copyright contributors to the SGLang project
# Licensed under the Apache License, Version 2.0.
# Original source: https://github.com/sgl-project/sglang
#
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

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from ...utils.fp8_dequant import dequant_fp8_weight_two_dim_block_grid
from ...utils.quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    unpack_uint8_to_int4_weight_2d,
)

# FP4 tables (E2M1)
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


# ===== Helpers =====
def _expect_single_scale(scales: List[Optional[torch.Tensor]], name: str) -> torch.Tensor:
    if len(scales) == 0 or scales[0] is None:
        raise ValueError(f"{name} must provide at least one scale tensor (scales[0]).")
    return scales[0]


def _to_fp8_fake(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).to(torch.float8_e4m3fn)


def _from_fp8(x_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x_fp8.to(dtype) * scale


def _dequant_weight_fp8(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return weight_fp8.to(dtype) * weight_scale


def _dequant_block_fp8_weight(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    block_n: int,
    block_k: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    return dequant_fp8_weight_two_dim_block_grid(
        weight_fp8, weight_scale, block_n, block_k, dtype=dtype
    )


# The NVFP4 helpers below are adapted from modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor
def _nvfp4_get_weights_scaling_factor(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
):
    """Returns quantized per block weight scaling factor."""
    if weights_scaling_factor_2 is None:
        # per-tensor scale-2 = amax / (6 * 448)
        weights_scaling_factor_2 = input.abs().amax().float() / (6.0 * 448.0)

    # Get per_block amax
    [n, k] = input.shape[-2:]
    assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

    assert k % block_size == 0, (
        "Weight shape is not divisible for block size for block quantization."
    )

    input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
    # Get per block amax
    per_block_amax = input.abs().amax(dim=-1).float()
    # Get per-block-scale
    per_block_scale = per_block_amax / 6.0
    # Quantize per_block_scale to FP8
    q_per_block_scale = per_block_scale / weights_scaling_factor_2
    # Set all zero values in scale to 1.0
    q_per_block_scale[per_block_scale == 0] = 1.0
    # Convert to torch.float8_e4m3fn
    if not keep_high_precision:
        q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
    return q_per_block_scale, weights_scaling_factor_2


def _cast_fp4(weight: torch.Tensor):
    """Converts tensor to uint4."""
    # Get device
    device = weight.device

    # Define mask to perform rounding
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
    mask_shape = list(weight.shape)
    mask = mask.expand([*mask_shape, 7])

    sign_bit = (weight < 0).to(torch.uint8)

    weight_abs = weight.abs()  # avoid in-place modification to input
    # Calculate the ordinal value based on the bounds
    ord = torch.searchsorted(e2m1_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
    # All values equal to e2m1_bounds at odd indices are rounded up and even indices are rounded down
    round = torch.any((weight_abs.unsqueeze(-1) == e2m1_bounds.to(device)) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
    return fp4_val


def _quantize_nvfp4(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
):
    """Converting a tensor to a quantized format based on NVFP4 quantization.

    Args:
        input (torch.Tensor): The input tensor to be quantized.
        block_size (int): The size of each block for quantization.
        weights_scaling_factor_2 (torch.Tensor): The per-tensor scaling factor for the weights.
    Returns:
    tuple: Contains quantized data and quantized per block scaling factor
    """

    weights_scaling_factor, weights_scaling_factor_2 = _nvfp4_get_weights_scaling_factor(
        input, block_size, weights_scaling_factor_2
    )

    # Reshape the weight and scale factors
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))

    # Scale weights
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )

    # Reshape weights to original
    scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))

    # Cast weights to fp4
    q_weight = _cast_fp4(scaled_weight)
    # Pack weights
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    return packed_weight, weights_scaling_factor


def _dequantize_nvfp4(
    quantized_t: torch.Tensor,  # [N, K/2] uint8
    scale_1: torch.Tensor,  # q_per_block_scale (FP8/FP32), flat or shaped
    scale_2: torch.Tensor,  # per-tensor scale-2 (FP32 scalar)
    orig_shape: tuple,  # (N, K)
    orig_dtype: torch.dtype,
) -> torch.Tensor:
    device = quantized_t.device
    N, K = orig_shape
    # slice/pad handling for the scale vector: take exactly N*K/16 entries
    num_blocks = N * (K // 16)
    s1 = scale_1.reshape(-1)[:num_blocks]

    high = (quantized_t >> 4) & 0x0F
    low = quantized_t & 0x0F
    idx = torch.empty(N, (K // 2) * 2, dtype=torch.long, device=device)
    idx[..., 0::2] = low.long()
    idx[..., 1::2] = high.long()

    vals = e2m1_values.to(device)[idx]  # [N, K], float32

    scale_real = (s1.to(torch.float32) * scale_2.to(torch.float32)).view(N, K // 16, 1)
    vals = vals.view(N, K // 16, 16) * scale_real
    return vals.view(N, K).to(orig_dtype)


@torch.library.custom_op("auto_deploy::torch_fake_quant_fp8_linear", mutates_args=())
def torch_fake_quant_fp8_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """
    Reference (eager) implementation for multiple quant formats via `format_type`.
    For FP8:
      - input_scale[0] and weight_scale[0] are required (amax/448 style)
      - input_zp / weight_zp ignored
    """
    if weight_quantized.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 path requires weight_quantized.dtype == torch.float8_e4m3fn")
    s_in = _expect_single_scale(input_scale, "input_scale")
    s_w = _expect_single_scale(weight_scale, "weight_scale")

    in_dtype = input.dtype
    out_features, in_features = weight_quantized.shape

    input_fp8 = _to_fp8_fake(input, s_in)
    input_deq = _from_fp8(input_fp8, s_in, in_dtype)

    weight_deq = _dequant_weight_fp8(weight_quantized, s_w, out_features, in_dtype)

    out = torch.matmul(input_deq.reshape(-1, in_features), weight_deq.t())
    if bias is not None:
        out = out + bias
    return out.reshape(*input.shape[:-1], out_features)


@torch_fake_quant_fp8_linear.register_fake
def torch_fake_quant_fp8_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    w = weight_quantized.to(input.dtype)
    return torch.ops.aten.linear(input, w, bias)


@torch.library.custom_op("auto_deploy::torch_fake_quant_nvfp4_linear", mutates_args=())
def torch_fake_quant_nvfp4_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """
    Reference (eager) implementation for multiple quant formats via `format_type`.
    For FP4:
      - input_scale[0]  = s_in2   (scalar, amax/(448*6))
      - weight_scale[0] = q_per_block_scale_w  (len >= N*K/16; may be padded)
      - weight_scale[1] = alpha = s_in2 * s_w2 (combined per-tensor scales)
    """
    if weight_quantized.dtype != torch.uint8:
        raise TypeError("NVFP4 path requires packed uint8 weights (2x FP4 per byte).")

    inv_x = _expect_single_scale(input_scale, "input_scale")
    if len(weight_scale) < 2 or weight_scale[0] is None or weight_scale[1] is None:
        raise ValueError(
            "NVFP4 needs weight_scale[0] (per-block vector) and weight_scale[1] (alpha)."
        )
    cutlass_qscale = weight_scale[0]
    alpha = weight_scale[1]

    if cutlass_qscale.dtype != torch.uint8:
        raise TypeError("NVFP4 expects CUTLASS per-block scale vector in uint8 (same as fused op).")

    inv_w = 1 / (inv_x * alpha)
    s2_x = 1.0 / inv_x
    s2_w = 1.0 / inv_w

    # Shapes
    in_dtype = input.dtype
    input_shape = input.shape
    N, K_packed = weight_quantized.shape[-2], weight_quantized.shape[-1]
    K = K_packed * 2
    assert K % 16 == 0, "NVFP4 requires K to be a multiple of 16"
    num_blocks_w = N * (K // 16)

    q_scale_w_slice = cutlass_fp4_scale_to_modelopt_fp4_scale(cutlass_qscale, (N, K))
    # (1) Dequantize weights with scale_1 = q_scale_w (sliced), scale_2 = s_w2
    q_scale_w_slice = q_scale_w_slice.reshape(-1)[:num_blocks_w]
    W_deq = _dequantize_nvfp4(weight_quantized, q_scale_w_slice, s2_w, (N, K), in_dtype)  # [N, K]

    # (2) Quantize+dequantize inputs with _quantize_nvfp4/_dequantize_nvfp4
    # Flatten batch for NVFP4 block processing
    X_2d = input.reshape(-1, K)

    X_packed, X_q_scale = _quantize_nvfp4(X_2d, block_size=16, weights_scaling_factor_2=s2_x)
    X_deq = _dequantize_nvfp4(X_packed, X_q_scale, s2_x, (X_2d.shape[0], K), in_dtype)  # [B, K]

    # (3) GEMM + bias (float GEMM with codec error baked in)
    out_2d = torch.matmul(X_deq, W_deq.t())  # [B, N]
    if bias is not None:
        out_2d = out_2d + bias
    return out_2d.reshape(*input_shape[:-1], N)


@torch_fake_quant_nvfp4_linear.register_fake
def torch_fake_quant_nvfp4_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_quantized.repeat(1, 2).to(input.dtype), bias)


@torch.library.custom_op("auto_deploy::torch_fake_quant_int4_linear", mutates_args=())
def torch_fake_quant_int4_linear(
    input: torch.Tensor,  # [..., K]
    weight_quantized: torch.Tensor,  # [N//2, K] unit8 (packed)
    bias: Optional[torch.Tensor],  # [N] or None
    input_scale: List[torch.Tensor],  # [ pre_quant_scale ]
    weight_scale: List[torch.Tensor],  # [ weight_scale ]
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    BLOCK_SIZE = 128
    # activation pre-scale
    pre_quant_scale = input_scale[0].to(dtype=input.dtype)
    x_scaled = torch.mul(input, pre_quant_scale)

    q_int4 = unpack_uint8_to_int4_weight_2d(weight_quantized, weight_scale[0])  # (N,K), int8
    amax_2d = (weight_scale[0] * 7.0).to(input.dtype)  # (N, K//128)

    scale_blocks = (7.0 / amax_2d).to(torch.float32)  # (N, K//128)
    scale_full = scale_blocks.repeat_interleave(BLOCK_SIZE, dim=1)  # (N,K)

    # Dequantize
    w_deq = (q_int4.to(torch.float32) / scale_full).to(input.dtype)

    return torch.ops.auto_deploy.torch_linear_simple.default(
        x_scaled,
        w_deq,
        bias,
        tp_mode=tp_mode,
        output_sizes=output_sizes,
        tp_min_local_shape=tp_min_local_shape,
        layer_type=layer_type,
    )


@torch_fake_quant_int4_linear.register_fake
def _fake(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    N_half = weight_quantized.shape[-2]
    N = N_half * 2
    return torch.empty((*input.shape[:-1], N), dtype=input.dtype, device=input.device)


@torch.library.custom_op("auto_deploy::torch_fake_quant_int4_gptq_linear", mutates_args=())
def torch_fake_quant_int4_gptq_linear(
    input: torch.Tensor,  # [..., K]
    weight_quantized: torch.Tensor,  # qweight [K/8, N] int32 (packed)
    bias: Optional[torch.Tensor],  # [N] or None
    input_scale: List[torch.Tensor],  # unused for GPTQ
    weight_scale: List[torch.Tensor],  # GPTQ scales [G, N]
    input_zp: List[torch.Tensor],  # unused for GPTQ
    weight_zp: List[torch.Tensor],  # GPTQ qzeros [G, N/8] int32
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    """
    GPTQ INT4 linear with compatible signature to other quant ops.
    - weight_quantized: qweight [K/8, N] packed int32
    - weight_scale[0]: scales [G, N]
    - weight_zp[0]: qzeros [G, N/8] packed int32
    """
    PACK_FACTOR = 8
    MAXQ = 15
    dequant_dtype = torch.int8

    qweight = weight_quantized
    scales = _expect_single_scale(weight_scale, "weight_scale")
    qzeros = _expect_single_scale(weight_zp, "weight_zp")

    dev = qweight.device
    input_shape = input.shape
    in_features = input_shape[-1]

    if qweight.dim() != 2:
        raise RuntimeError("qweight must be 2D [K/8, N]")
    K = qweight.size(0) * PACK_FACTOR
    N = qweight.size(1)

    if scales.dim() != 2 or scales.size(1) != N:
        raise RuntimeError(f"scales must be [G, N={N}]")
    G = scales.size(0)

    if K % G != 0:
        raise RuntimeError(f"K ({K}) must be divisible by G ({G})")
    block_size = K // G

    if qzeros.dim() != 2 or qzeros.size(0) != G or qzeros.size(1) * PACK_FACTOR != N:
        raise RuntimeError(f"qzeros must be [G={G}, N/8={N // 8}]")

    # Reshape input to 2D if needed
    x_2d = input.reshape(-1, in_features)

    # Build g_idx and shift tables
    g_idx = torch.arange(K, device=dev, dtype=torch.int32) // block_size  # [K]
    wf = torch.arange(PACK_FACTOR, device=dev, dtype=torch.int32) * 4  # [8]
    wf_unsqueeze_zero = wf.view(1, 1, PACK_FACTOR)  # [1,1,8]
    wf_unsqueeze_neg_one = wf.view(1, PACK_FACTOR, 1)  # [1,8,1]

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, PACK_FACTOR),
        wf_unsqueeze_zero,
    ).to(dequant_dtype)
    zeros = torch.bitwise_and(zeros, MAXQ).reshape(scales.shape)

    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            torch.unsqueeze(qweight, 1).expand(-1, PACK_FACTOR, -1),
            wf_unsqueeze_neg_one,
        ).to(dequant_dtype),
        MAXQ,
    )
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    weights = (scales[g_idx.long()] * (weight - zeros[g_idx.long()])).to(input.dtype)

    out = torch.matmul(x_2d, weights)

    if bias is not None:
        out = out + bias

    # Reshape output back to match input batch dimensions
    out = out.reshape(*input_shape[:-1], N)

    return out


@torch_fake_quant_int4_gptq_linear.register_fake
def torch_fake_quant_int4_gptq_linear_fake(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
) -> torch.Tensor:
    N = weight_quantized.size(1)
    return torch.empty((*input.shape[:-1], N), dtype=input.dtype, device=input.device)


@triton.jit
def _act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, ROUND_SCALE: tl.constexpr):
    """Block-wise FP8 activation quantization, safe for all-zero blocks.

    Identical to HuggingFace's act_quant_kernel except that the per-block scale
    is clamped to a minimum of 1e-12 before dividing.  This avoids 0/0 = NaN
    when every element in a block is zero.
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    amax = tl.max(tl.abs(x))
    if ROUND_SCALE:
        amax = tl.maximum(amax, 1e-4)
        s = amax / 448.0
        s = tl.exp2(tl.ceil(tl.log2(s)))
    else:
        s = amax / 448.0
        # Clamp scale so that all-zero blocks produce 0/eps = 0 instead of 0/0 = NaN.
        s = tl.maximum(s, 1e-12)
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)


def _safe_act_quant(x: torch.Tensor, block_size: int = 128, input_scale_fmt: str = "") -> tuple:
    """Block-wise FP8 activation quantization (CUDA-graph safe).

    Drop-in replacement for ``transformers.integrations.finegrained_fp8.act_quant``
    that fixes the NaN-on-zero-block bug by clamping the per-block scale inside
    the Triton kernel itself.  No post-hoc fixup tensors are created, so the
    op is fully compatible with CUDA graphs.
    """
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # Keep scale metadata in the model dtype to avoid FP32->BF16 cast kernels
    # when the tensor is consumed by downstream MoE/quantized paths.
    s = x.new_empty(*x.shape[:-1], x.shape[-1] // block_size, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)  # noqa: E731
    round_scale = input_scale_fmt.lower() == "ue8m0"
    # Each program reduces exactly one `block_size` (=128) chunk to a single scale
    # -- a tiny single-reduction workload. One warp (32 lanes, ~4 elems/lane,
    # intra-warp shuffle, no shared-mem barrier) beats the Triton default
    # num_warps=4 (128 lanes + a cross-warp reduction) by ~5% on the DeepSeek-V4
    # decode mean and ~1-2% at prefill, with zero numeric change (drift-controlled
    # round-robin CUDA-graph microbench on B200/sm100). num_stages is left at the
    # default -- this kernel is loop-free, so software pipelining is inert.
    _act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, ROUND_SCALE=round_scale, num_warps=1)
    return y, s


# Adapted from sgl-project/sglang fp8 block matmul kernel, vendored here to
# decouple from transformers.integrations.finegrained_fp8 (which removed
# w8a8_block_fp8_matmul_triton in transformers 5.5.x).
#
# Autotune over (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, num_warps, num_stages).
# BLOCK_SIZE_K is intentionally NOT autotuned -- it is passed at the call site and
# pinned to the quantization group_k so the in-loop scale index
# ``offs_ks = (k * BLOCK_SIZE_K) // group_k`` never straddles a scale block
# (a K-tile that does not divide group_k would load one scale for several groups
# and corrupt the result). BLOCK_SIZE_N / BLOCK_SIZE_M fetch per-element scales so
# they are free to tune.
#
# Tuning was driven by a CUDA-graph-amortized microbench on B200 (sm100) over the
# DeepSeek-V4 MLA/dense per-rank projection shapes:
#   * M=1/2 decode is a K-loop-latency-bound GEMV (K=7168 hits an ~18us floor
#     independent of N). BLOCK_SIZE_N=64 + tuned warps/stages beats the old fixed
#     BLOCK_SIZE_N=128 by ~28% on the decode mean.
#   * For M>=64 large-K (>=2048) prefill the stock kernel is run-to-run
#     NON-deterministic on sm100 with num_warps=4 (a sparse fp8-MMA pipelining
#     glitch; global RMSE stays ~1e-3 but a few near-cancellation outputs flip).
#     Every BLOCK_SIZE_M>=64 config below therefore uses num_warps=8, which was
#     verified deterministic at the worst shape AND ~2.8x faster than the old
#     BLOCK_SIZE_M=128/num_warps=4 launch. The autotuner selects on latency only,
#     so the racy num_warps=4 large-tile configs are deliberately excluded.
_W8A8_BLOCK_FP8_MATMUL_CONFIGS = [
    # Decode / small-M (BLOCK_SIZE_M=16), num_warps=4. The M=1 GEMV is dominated by
    # the K=7168 projections; the right BLOCK_SIZE_N depends on N:
    #   * small N (<=~4k): BLOCK_SIZE_N=32 spreads the work over more CTAs and is
    #     ~20% faster than 64 (e.g. N=256/K=7168: 14.3 vs 18.1 us).
    #   * large N (e.g. 7168): BLOCK_SIZE_N=64 wins -- 32 would launch >2x the SM
    #     count and run a second wave (N=7168/K=2048: 8.3 vs 11.9 us).
    # The autotuner (keyed on N) resolves the 20-30% gap reliably. num_warps is
    # pinned to 4 (vs 8) because the K=7168 GEMV is ~15% faster at 4 and the 4-vs-8
    # gap is too small for do_bench to resolve, which would re-introduce run-to-run
    # selection flicker. num_stages 3/4 is a harmless near-tie.
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=4
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1}, num_warps=4, num_stages=3
    ),
    # Prefill / large-M (BLOCK_SIZE_M>=64): num_warps=8, BLOCK_SIZE_N=128 only.
    # IMPORTANT: the stock kernel is run-to-run NON-deterministic on sm100 at large
    # M (>=256) with large K (>=2048) for several (BLOCK_SIZE_*, num_warps) combos
    # -- a sparse fp8-MMA pipelining glitch (global RMSE stays ~1e-3 but a few
    # near-cancellation outputs flip). The old BLOCK_SIZE_M=128/num_warps=4 launch
    # was itself racy. BLOCK_SIZE_N=256 is faster but RACY at some shapes (e.g.
    # N=7168,K=2304); BLOCK_SIZE_N=128 with num_warps=8 was verified deterministic
    # at every measured shape AND still ~2x faster than the old launch, so prefill is
    # both faster and deterministic. (Prefill does not affect tpot; correctness wins.)
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4
    ),
]


@triton.autotune(configs=_W8A8_BLOCK_FP8_MATMUL_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _w8a8_block_fp8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert As.numel() != 1, "per-tensor scales unsupported in vendored path"
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    assert B.ndim == 2 and B.is_contiguous()
    assert Bs.ndim == 2
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)

    # Split-K decode path: at small M (decode GEMV) with a long K reduction the
    # default one-CTA-per-(M,N)-tile kernel walks the entire K dimension in a single
    # serial loop and launches only ``cdiv(N, BLOCK_N)`` CTAs (e.g. M=1,N=256,K=7168
    # -> 4-8 CTAs over 56 K-blocks), leaving the GPU almost idle and bottlenecked on
    # K-loop latency. ``_w8a8_block_fp8_matmul_splitk`` partitions the K reduction
    # across ``SPLIT_K`` CTAs (grid axis 1) and reduces the fp32 partials, raising
    # occupancy ~SPLIT_K x. Restricted to small M + large K where the base grid is
    # CTA-starved; everything else keeps the autotuned full-K kernel.
    if _use_splitk_decode(M, N, K):
        return _w8a8_block_fp8_matmul_splitk(A, B, As, Bs, block_n, block_k, output_dtype, M, N, K)

    C = A.new_empty(C_shape, dtype=output_dtype)

    # BLOCK_SIZE_M / BLOCK_SIZE_N / GROUP_SIZE_M / num_warps / num_stages come from
    # the @triton.autotune config (keyed on M, N, K). BLOCK_SIZE_K is pinned to the
    # quantization block size for scale-indexing correctness (see kernel comment).
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    _w8a8_block_fp8_matmul_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C.stride(-2),
        C.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_K=block_k,
    )
    return C


# Split-K decode GEMV reduction layout (kernel_layout axis, idea_0025).
#
# The full-K kernel above assigns one CTA per (M,N) output tile and walks the whole
# K dimension serially. At decode M (1/2) with K=7168 (56 K-blocks of 128) the base
# grid is only cdiv(N, BLOCK_N) CTAs (4-36 for the DeepSeek-V4 MLA/dense projection
# Ns), so the kernel is reduction-/latency-bound and the GPU is almost idle.
#
# This kernel splits the K reduction across ``SPLIT_K`` CTAs (grid axis 1). Each CTA
# strides through ``k = pid_sk, pid_sk+SPLIT_K, ...`` K-blocks (balanced + handles a
# K-block count not divisible by SPLIT_K via the existing K-mask), accumulates its
# partial in fp32, and ``tl.atomic_add``s into a pre-zeroed fp32 accumulator. The
# split is along the contraction only, so the result equals the serial sum up to
# fp32 add-ordering (~1e-7 rel, far under the fp8 quant error); decode is bit-stable
# enough for the strict M=1/2 accuracy bar. BLOCK_SIZE_K stays pinned to the
# quantization group_k so the per-block scale index never straddles a scale block.
@triton.jit
def _w8a8_block_fp8_matmul_splitk_kernel(
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    group_n,
    group_k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # Start each split at its own K-block; stride by SPLIT_K K-blocks per loop step.
    a_ptrs = A + (
        offs_am[:, None] * stride_am + (pid_sk * BLOCK_SIZE_K + offs_k)[None, :] * stride_ak
    )
    b_ptrs = B + (
        (pid_sk * BLOCK_SIZE_K + offs_k)[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    for k in range(pid_sk, num_k, SPLIT_K):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        offs_ks = (k * BLOCK_SIZE_K) // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_ak
        b_ptrs += SPLIT_K * BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


# Split-K launch config for the decode GEMV. Tuned (kernel_layout axis) on B200 over
# the DeepSeek-V4 MLA/dense K=7168 decode projection shapes (N in {256,576,1536,2304}).
_SPLITK_BLOCK_SIZE_M = 16
# Mid-N default / fallback SPLIT_K (see ``_splitk_split_k`` for the per-N schedule).
_SPLITK_SPLIT_K = 24
_SPLITK_NUM_WARPS = 4
_SPLITK_NUM_STAGES = 3
# Gate: small M (decode) + long K reduction, where the base grid is CTA-starved.
_SPLITK_MAX_M = 4
_SPLITK_MIN_K = 4096


def _use_splitk_decode(M: int, N: int, K: int) -> bool:
    return M <= _SPLITK_MAX_M and K >= _SPLITK_MIN_K


def _splitk_block_n(N: int) -> int:
    """BLOCK_SIZE_N for the split-K decode GEMV, scaled with N.

    Small N is CTA-starved so a narrow N-tile spreads work over more CTAs; large N
    has enough output tiles that a wide N-tile (better MMA / fewer atomic stores)
    wins. Measured B200 optima: N=256->32, N=576->64, N>=1024 (1536/2304)->128.
    """
    if N <= 512:
        return 32
    if N >= 1024:
        return 128
    return 64


def _splitk_split_k(N: int) -> int:
    """SPLIT_K (K-reduction CTA fan-out) for the split-K decode GEMV, scaled with N.

    The K=7168 reduction is partitioned across ``SPLIT_K`` CTAs per output tile, so
    the launch grid is ``cdiv(N, BLOCK_SIZE_N) * SPLIT_K`` and the atomic-reduction
    count scales with ``SPLIT_K``. The best grid is ~2-3 waves on the B200 SM array:
    narrow-N tiles (``N < 1024`` -> BLOCK_SIZE_N 32/64) yield few n-tiles and are
    CTA-starved, so they want a *deeper* K-split (more CTAs); wide 128-N tiles
    already have enough n-tiles, so a *shallower* split cuts atomic-reduction
    over-subscription. Measured B200 optima (BLOCK_SIZE_K=128, K=7168, M=1):
    N=256->48, N=576->48, N=1536->24, N=2304->16 (each ~-3.5..-4.0% vs the old
    fixed SPLIT_K=24; idea_0063, kernel_tile). idea_0025's fixed 24 was tuned over
    SPLIT_K<=32 and missed the deeper split the narrow-N shapes want.
    """
    if N < 1024:
        return 48
    if N <= 1792:
        return _SPLITK_SPLIT_K  # 24
    return 16


def _splitk_block_k(N: int, block_k: int) -> int:
    """MMA contraction-tile depth (``BLOCK_SIZE_K``) for the split-K decode GEMV.

    Decoupled from the quantization scale group ``block_k`` (idea_0063): it must
    divide ``block_k`` so an MMA tile stays inside one scale block, but a *smaller*
    tile keeps the atomic-reduction count fixed at ``SPLIT_K`` while raising the
    K-loop trip count, which deepens the software pipeline (more in-flight B-tile
    loads to hide HBM latency on this memory-bound GEMV). Pinned to ``block_k`` here;
    the kernel_tile sweep tunes per-N.
    """
    return block_k


def _validate_splitk_c_out(
    C_out: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
) -> None:
    """Validate a caller-owned split-K accumulator without reading device data."""
    expected_shape = (M, N)
    if A.dim() != 2 or tuple(A.shape) != (M, K):
        raise ValueError(f"A must have shape {(M, K)} when C_out is used, got {tuple(A.shape)}")
    if C_out.dtype != torch.float32:
        raise TypeError(f"C_out must have dtype float32, got {C_out.dtype}")
    if output_dtype != torch.float32:
        raise ValueError("C_out requires output_dtype=torch.float32")
    if C_out.layout != torch.strided:
        raise ValueError(f"C_out must use strided layout, got {C_out.layout}")
    for name, tensor in (("A", A), ("B", B), ("As", As), ("Bs", Bs)):
        if C_out.device != tensor.device:
            raise ValueError(
                f"C_out and {name} must be on the same device, "
                f"got {C_out.device} and {tensor.device}"
            )
    if C_out.dim() != 2 or tuple(C_out.shape) != expected_shape:
        raise ValueError(f"C_out must have shape {expected_shape}, got {tuple(C_out.shape)}")
    if C_out.stride(1) != 1 or C_out.stride(0) < N:
        raise ValueError(
            "C_out must have unit column stride and row stride >= N; "
            f"got stride {tuple(C_out.stride())} for N={N}"
        )
    for name, tensor in (("A", A), ("B", B), ("As", As), ("Bs", Bs)):
        if torch._C._overlaps(C_out, tensor):
            raise ValueError(f"C_out must not overlap {name}")


def _w8a8_block_fp8_matmul_splitk(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_n: int,
    block_k: int,
    output_dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
    *,
    SPLIT_K: Optional[int] = None,
    BLOCK_SIZE_N: Optional[int] = None,
    BLOCK_SIZE_K: Optional[int] = None,
    num_warps: int = _SPLITK_NUM_WARPS,
    num_stages: int = _SPLITK_NUM_STAGES,
    C_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Split-K block-FP8 GEMM for the decode GEMV (see kernel docstring above).

    Accumulates fp32 partials from ``SPLIT_K`` contraction-slices via atomics into a
    pre-zeroed fp32 buffer, then casts to ``output_dtype``. ``SPLIT_K`` /
    ``BLOCK_SIZE_N`` / ``BLOCK_SIZE_K`` default (``None``) to the tuned heuristic
    (``_splitk_split_k`` / ``_splitk_block_n`` / ``_splitk_block_k``); the microbench
    passes them explicitly to sweep the config.

    ``BLOCK_SIZE_K`` is the MMA contraction-tile depth and is *decoupled* from the
    quantization ``block_k`` (the scale group). It must divide ``block_k`` so a tile
    never straddles a scale-block boundary (the kernel loads one scale per tile,
    indexed ``(k * BLOCK_SIZE_K) // group_k``); a smaller tile keeps the same atomic
    count as ``SPLIT_K`` but raises the K-loop iteration count (deeper pipelining).

    ``C_out``, when given, is a caller-provided, pre-zeroed FP32 accumulator. It must
    be a non-overlapping ``[M, N]`` view with unit column stride, must not alias any
    input, and is returned directly (``output_dtype`` must be FP32). This lets
    grouped GEMVs use disjoint column slices of one allocation and one later finish
    cast. Split-K uses FP32 atomics, so launch-to-launch accumulation order is not
    deterministic; a later BF16 cast normally absorbs the variation but values on a
    BF16 rounding boundary can differ by one ULP.
    """
    if SPLIT_K is None:
        SPLIT_K = _splitk_split_k(N)
    if BLOCK_SIZE_N is None:
        BLOCK_SIZE_N = _splitk_block_n(N)
    if BLOCK_SIZE_K is None:
        BLOCK_SIZE_K = _splitk_block_k(N, block_k)
    # The MMA tile must fit inside a single scale block (one scale loaded per tile).
    assert block_k % BLOCK_SIZE_K == 0, (
        f"BLOCK_SIZE_K={BLOCK_SIZE_K} must divide quant block_k={block_k}"
    )
    if C_out is not None:
        _validate_splitk_c_out(C_out, A, B, As, Bs, output_dtype, M, N, K)
        C_acc = C_out
    else:
        C_shape = A.shape[:-1] + (N,)
        # fp32 accumulator, pre-zeroed for the atomic reduction across SPLIT_K CTAs.
        C_acc = A.new_zeros(C_shape, dtype=torch.float32)

    grid = (
        triton.cdiv(M, _SPLITK_BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        SPLIT_K,
    )
    _w8a8_block_fp8_matmul_splitk_kernel[grid](
        A,
        B,
        C_acc,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        C_acc.stride(-2),
        C_acc.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=_SPLITK_BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SPLIT_K=SPLIT_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if output_dtype == torch.float32:
        return C_acc
    return C_acc.to(output_dtype)


@torch.library.custom_op("auto_deploy::torch_fake_quant_finegrained_fp8_linear", mutates_args=())
def torch_fake_quant_finegrained_fp8_linear(
    input: torch.Tensor,  # [..., K]
    weight_quantized: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # [N] or None
    input_scale: List[torch.Tensor],  # unused for FineGrained FP8 (input quantized on the fly)
    weight_scale: List[torch.Tensor],  # [weight_scale_inv]
    input_zp: List[torch.Tensor],  # unused
    weight_zp: List[torch.Tensor],  # unused
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
    input_scale_fmt: str = "",
) -> torch.Tensor:
    """FineGrainedFP8 linear operation.
    - weight_scale[0] = weight_scale_inv (per-block weight scale)
    - input_scale, input_zp, weight_zp are unused
    - block_size is inferred from weight and weight_scale_inv shapes
    """
    weight_scale_inv = weight_scale[0]

    # Infer block_size from weight and weight_scale_inv shapes
    # weight shape: [N, K], weight_scale_inv shape: [ceil(N/block_n), ceil(K/block_k)]
    N, K = weight_quantized.shape
    scale_n, scale_k = weight_scale_inv.shape
    block_n = triton.cdiv(N, scale_n)
    block_k = triton.cdiv(K, scale_k)
    block_size = [block_n, block_k]

    qinput, scale = _safe_act_quant(input, block_size[1], input_scale_fmt)
    output = _w8a8_block_fp8_matmul_triton(
        qinput,
        weight_quantized,
        scale,
        weight_scale_inv,
        block_size,
        output_dtype=input.dtype,
    )

    if bias is not None:
        output = output + bias

    return output.to(dtype=input.dtype)


@torch_fake_quant_finegrained_fp8_linear.register_fake
def _torch_fake_quant_finegrained_fp8_linear_fake(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
    input_scale_fmt: str = "",
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    out_features = weight_quantized.shape[0]
    return torch.empty((*input.shape[:-1], out_features), dtype=input.dtype, device=input.device)


@torch.library.custom_op("auto_deploy::torch_fp8_finegrained_act_quant", mutates_args=())
def torch_fp8_finegrained_act_quant(
    input: torch.Tensor,  # [..., K]
    block_size: int,  # block_k (the activation quant group, == matmul block_k)
    input_scale_fmt: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stand-alone block-wise FP8 activation quantization.

    This is exactly the activation-quant half of
    ``torch_fake_quant_finegrained_fp8_linear`` (the ``_safe_act_quant`` /
    ``_act_quant_kernel`` Triton launch), split out as its own op so the
    ``fuse_fp8_act_quant_cse`` graph transform can hoist it out of the linear and
    share one ``(qfp8, scale)`` pair across sibling linears that consume the same
    activation tensor with the same block size. Because ``_safe_act_quant`` is a
    deterministic pure function of ``(input, block_size, input_scale_fmt)``, the
    shared result is byte-identical to each per-linear recompute.
    """
    return _safe_act_quant(input, block_size, input_scale_fmt)


@torch_fp8_finegrained_act_quant.register_fake
def _torch_fp8_finegrained_act_quant_fake(
    input: torch.Tensor,
    block_size: int,
    input_scale_fmt: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    qinput = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scale = input.new_empty(*input.shape[:-1], input.shape[-1] // block_size, dtype=input.dtype)
    return qinput, scale


@torch.library.custom_op(
    "auto_deploy::torch_fake_quant_finegrained_fp8_linear_prequant", mutates_args=()
)
def torch_fake_quant_finegrained_fp8_linear_prequant(
    qinput: torch.Tensor,  # [..., K] float8_e4m3fn (pre-quantized activation)
    input_scale: torch.Tensor,  # [..., K//block_k] per-block act scale (model dtype)
    weight_quantized: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # [N] or None
    weight_scale: List[torch.Tensor],  # [weight_scale_inv]
) -> torch.Tensor:
    """Matmul half of ``torch_fake_quant_finegrained_fp8_linear``.

    Consumes a pre-quantized activation + per-block scale (produced once by
    ``torch_fp8_finegrained_act_quant`` and shared across sibling linears) and runs
    the same block-FP8 W8A8 matmul + bias the original op runs after its in-line
    ``_safe_act_quant``. The output dtype is recovered from ``input_scale.dtype``,
    which ``_safe_act_quant`` allocates in the original activation dtype, so the
    result is bit-for-bit identical to the fused op.
    """
    weight_scale_inv = weight_scale[0]
    out_dtype = input_scale.dtype
    N, K = weight_quantized.shape
    scale_n, scale_k = weight_scale_inv.shape
    block_n = triton.cdiv(N, scale_n)
    block_k = triton.cdiv(K, scale_k)

    output = _w8a8_block_fp8_matmul_triton(
        qinput,
        weight_quantized,
        input_scale,
        weight_scale_inv,
        [block_n, block_k],
        output_dtype=out_dtype,
    )

    if bias is not None:
        output = output + bias

    return output.to(dtype=out_dtype)


@torch_fake_quant_finegrained_fp8_linear_prequant.register_fake
def _torch_fake_quant_finegrained_fp8_linear_prequant_fake(
    qinput: torch.Tensor,
    input_scale: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: List[torch.Tensor],
) -> torch.Tensor:
    out_features = weight_quantized.shape[0]
    return torch.empty(
        (*qinput.shape[:-1], out_features), dtype=input_scale.dtype, device=qinput.device
    )


@torch.library.custom_op(
    "auto_deploy::torch_fake_quant_grouped_finegrained_fp8_linear",
    mutates_args=(),
)
def torch_fake_quant_grouped_finegrained_fp8_linear(
    input: torch.Tensor,  # [..., G, K]
    weight_quantized: torch.Tensor,  # [G * R, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # [G * R], [G, R], or None
    input_scale: List[torch.Tensor],  # unused for FineGrained FP8 (input quantized on the fly)
    weight_scale: List[torch.Tensor],  # [weight_scale_inv]
    input_zp: List[torch.Tensor],  # unused
    weight_zp: List[torch.Tensor],  # unused
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
    input_scale_fmt: str = "",
) -> torch.Tensor:
    """Grouped FineGrainedFP8 projection for flattened checkpoint weights.

    Consumes a flattened checkpoint weight layout ``[G * R, K]`` and matching
    per-block ``weight_scale_inv`` buffers.
    """
    del input_scale, input_zp, weight_zp, tp_mode, output_sizes, tp_min_local_shape, layer_type
    if input.dim() < 2:
        raise ValueError(f"input must have at least grouped and K dimensions, got {input.shape}")
    if weight_quantized.dim() != 2:
        raise ValueError(f"weight must have shape [G * R, K], got {weight_quantized.shape}")
    # The weight is either the raw ``float8_e4m3fn`` checkpoint tensor (consumed directly by the
    # block-FP8 matmul below) or a tensor whose exact BF16 runtime value was pre-materialized at
    # load time by the ``bake_grouped_finegrained_fp8_weight`` post_load_fusion transform. When
    # the weight is FP8 we keep both operands in FP8 and run a direct grouped block-FP8 W8A8
    # matmul (idea_0025); when it has already been baked to a floating-point dtype we fall back to
    # the dynamic input quantize-dequantize + grouped BMM, which stays bit-for-bit identical to
    # the prior behavior on that branch.
    weight_is_fp8 = weight_quantized.dtype == torch.float8_e4m3fn
    if not weight_is_fp8 and not weight_quantized.is_floating_point():
        raise TypeError(
            "Grouped FineGrained FP8 path requires a float8_e4m3fn weight or a pre-dequantized "
            f"floating-point weight, got {weight_quantized.dtype}"
        )

    weight_scale_inv = _expect_single_scale(weight_scale, "weight_scale")
    num_groups = input.shape[-2]
    in_features = input.shape[-1]
    out_rows, weight_in_features = weight_quantized.shape
    if weight_in_features != in_features:
        raise ValueError(f"weight K ({weight_in_features}) must match input K ({in_features})")
    if out_rows % num_groups != 0:
        raise ValueError(f"weight rows ({out_rows}) must be divisible by groups ({num_groups})")

    scale_n, scale_k = weight_scale_inv.shape
    if scale_n == 0 or scale_k == 0:
        raise ValueError(f"weight_scale has zero dimension {tuple(weight_scale_inv.shape)}")
    block_n = triton.cdiv(out_rows, scale_n)
    block_k = triton.cdiv(in_features, scale_k)

    input_contiguous = input.contiguous()
    rank = out_rows // num_groups
    lead_shape = input.shape[:-2]

    if weight_is_fp8:
        # Direct grouped block-FP8 W8A8 matmul: keep both operands in FP8 and let the proven
        # block-FP8 kernel apply the per-block input/weight scales inside its FP32 accumulator.
        # This removes the per-call input quantize->dequantize round-trip and the FP8->BF16 weight
        # dequant entirely, and reads the FP8 weight (1 byte/elem) instead of a dequantized BF16
        # weight (2 byte/elem) -- the dominant HBM cost of this memory-bound decode GEMV. It is the
        # same arithmetic the non-grouped FineGrained FP8 path already uses, and is numerically
        # *more* faithful than the prior dequant->BF16-BMM (no BF16 rounding of the dequantized
        # operands; the per-block scale is constant within a block so factoring it out of the
        # contraction is exact). Under tensor parallelism the DeepSeek-V4 MLA ``wo_a`` per-rank
        # group count is 1, so this is a single 2D block-FP8 GEMM (K=4096 -> the split-K decode
        # path); ``num_groups > 1`` falls back to a per-group launch of the same proven kernel.
        qinput, input_scales = _safe_act_quant(input_contiguous, block_k, input_scale_fmt)
        m_tokens = qinput.numel() // (num_groups * in_features)
        qin = qinput.reshape(m_tokens, num_groups, in_features)
        sin = input_scales.reshape(m_tokens, num_groups, input_scales.shape[-1])
        if num_groups == 1:
            out2d = _w8a8_block_fp8_matmul_triton(
                qin[:, 0, :],
                weight_quantized,
                sin[:, 0, :],
                weight_scale_inv,
                [block_n, block_k],
                output_dtype=input.dtype,
            )
            output = out2d.reshape(*lead_shape, out_rows)
        elif _use_splitk_decode(m_tokens, rank, in_features):
            # Decode GEMV epilogue collapse (idea_0003): every per-rank group takes the
            # split-K path here, and the old per-group dispatch paid a (zero-fill +
            # fp32->bf16 finish cast) pair per group plus a ``torch.stack`` copy to
            # re-concatenate the group outputs. Instead, atomically accumulate all
            # groups into ONE pre-zeroed fp32 buffer laid out exactly like the stacked
            # result ([M, G*rank], group-major columns) — the split-K kernel writes
            # each group's disjoint column slice through explicit strides — then run
            # ONE finish cast over the whole buffer. This preserves the kernel, launch
            # configuration, and mathematical FP32 reduction for each group while
            # removing redundant fills/casts and the stack copy. Split-K atomic arrival
            # order remains nondeterministic; values exactly on a BF16 rounding boundary
            # can vary by one ULP just as they can on the original per-group path. The
            # strided ``qin`` / ``sin`` slices need no ``.contiguous()`` because the
            # kernel consumes explicit row strides.
            weight_grouped = weight_quantized.view(num_groups, rank, in_features)
            scale_rows = scale_n // num_groups
            scale_grouped = weight_scale_inv.view(num_groups, scale_rows, scale_k)
            acc = qinput.new_zeros((m_tokens, out_rows), dtype=torch.float32)
            for g in range(num_groups):
                _w8a8_block_fp8_matmul_splitk(
                    qin[:, g, :],
                    weight_grouped[g],
                    sin[:, g, :],
                    scale_grouped[g],
                    block_n,
                    block_k,
                    torch.float32,
                    m_tokens,
                    rank,
                    in_features,
                    C_out=acc[:, g * rank : (g + 1) * rank],
                )
            output = acc.to(input.dtype).reshape(*lead_shape, out_rows)
        else:
            weight_grouped = weight_quantized.view(num_groups, rank, in_features)
            scale_rows = scale_n // num_groups
            scale_grouped = weight_scale_inv.view(num_groups, scale_rows, scale_k)
            parts = [
                _w8a8_block_fp8_matmul_triton(
                    qin[:, g, :].contiguous(),
                    weight_grouped[g].contiguous(),
                    sin[:, g, :].contiguous(),
                    scale_grouped[g].contiguous(),
                    [block_n, block_k],
                    output_dtype=input.dtype,
                )
                for g in range(num_groups)
            ]
            output = torch.stack(parts, dim=1).reshape(*lead_shape, out_rows)
    else:
        # Weight already holds its dequantized floating-point runtime value (baked at load time by
        # the ``bake_grouped_finegrained_fp8_weight`` transform). Fall back to the dynamic input
        # quantize-dequantize + grouped BMM; bit-for-bit identical to the prior behavior here.
        qinput, input_scales = _safe_act_quant(input_contiguous, block_k, input_scale_fmt)
        qinput_blocks = qinput.reshape(*input_contiguous.shape[:-1], -1, block_k)
        input_dequant = (qinput_blocks.to(input.dtype) * input_scales.unsqueeze(-1)).reshape_as(
            input_contiguous
        )
        weight_grouped = weight_quantized.to(input.dtype).view(num_groups, rank, in_features)
        output = torch.matmul(
            input_dequant.unsqueeze(-2),
            weight_grouped.transpose(-1, -2),
        ).squeeze(-2)
        output = output.flatten(-2)

    if bias is not None:
        output = output + bias.reshape(out_rows).to(output.dtype)
    return output.to(dtype=input.dtype)


@torch_fake_quant_grouped_finegrained_fp8_linear.register_fake
def _torch_fake_quant_grouped_finegrained_fp8_linear_fake(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
    input_scale_fmt: str = "",
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    del bias, input_scale, weight_scale, input_zp, weight_zp
    del tp_mode, output_sizes, tp_min_local_shape, layer_type, input_scale_fmt
    out_features = weight_quantized.shape[0]
    return torch.empty((*input.shape[:-2], out_features), dtype=input.dtype, device=input.device)
