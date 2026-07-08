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

# DeepSeek-V4-Flash TP4 per-rank decode additions (idea_0040). Each exact M=1
# full-K key is pinned to its measured winner so independent ranks cannot choose
# different near-tie configs during Triton autotuning.
_W8A8_TP4_QIDX_CONFIG = triton.Config(
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 1},
    num_warps=8,
    num_stages=4,
)
_W8A8_TP4_Q_CONFIG = triton.Config(
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 1},
    num_warps=8,
    num_stages=4,
)
_W8A8_TP4_WO_B_CONFIG = triton.Config(
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "GROUP_SIZE_M": 1},
    num_warps=8,
    num_stages=5,
)
_W8A8_TP4_DECODE_CONFIG_BY_KEY = {
    (1, 16384, 1024): _W8A8_TP4_QIDX_CONFIG,  # fused wq_b + indexer.wq_b
    (1, 8192, 1024): _W8A8_TP4_Q_CONFIG,  # wq_b
    (1, 4096, 2048): _W8A8_TP4_WO_B_CONFIG,  # wo_b
}
_W8A8_TP4_DECODE_CONFIGS = tuple(_W8A8_TP4_DECODE_CONFIG_BY_KEY.values())
_W8A8_TP4_DECODE_KEYS = frozenset(_W8A8_TP4_DECODE_CONFIG_BY_KEY)

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
    # The exact TP4 full-K GEMVs are much wider / shorter-K than the K=7168 shapes
    # the configs above were tuned on. With 8-16 K-blocks, round-robin CUDA-graph
    # microbenchmarks (drift controlled, L2-cold and L2-hot weight regimes) show:
    #   * N=16384 K=1024: BLOCK_SIZE_N=128/num_warps=8/num_stages=4 (~128 CTAs
    #     ~= 1 wave) -21% cold / -30% hot vs the BLOCK_SIZE_N=64/num_warps=4 pick.
    #   * N=8192 K=1024: BLOCK_SIZE_N=64/num_warps=8/num_stages=4, ~-10%.
    #   * N=4096 K=2048: BLOCK_SIZE_N=32/num_warps=8/num_stages=5, ~4% in the
    #     cold-weight measurements.
    # Each key is pinned to the matching winner above. Every other key, including
    # M=2, prefill, and N=4096/K=512 shared w2, sees the pre-idea config set.
    *_W8A8_TP4_DECODE_CONFIGS,
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


def _w8a8_prune_tp4_decode_configs(configs, nargs, **kwargs):
    """Pin exact measured TP4 keys and preserve the old list everywhere else.

    Pinning prevents per-rank autotuner noise from selecting a slower near-tie.
    Other models, shared w2, chunked prefill, and larger-batch selection behavior
    remain identical to the pre-idea config set.
    """
    key = (nargs["M"], nargs["N"], nargs["K"])
    pinned_config = _W8A8_TP4_DECODE_CONFIG_BY_KEY.get(key)
    if pinned_config is not None:
        return [pinned_config]
    return [config for config in configs if config not in _W8A8_TP4_DECODE_CONFIGS]


@triton.autotune(
    configs=_W8A8_BLOCK_FP8_MATMUL_CONFIGS,
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _w8a8_prune_tp4_decode_configs},
)
@triton.jit
def _w8a8_block_fp8_matmul_kernel(
    A,
    B,
    C,
    As,
    Bs,
    R,
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
    stride_rm,
    stride_rn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
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
    if HAS_RESIDUAL:
        # Fused merge add (e.g. routed-MoE output + this shared-expert down
        # projection feeding one all_reduce). ``c`` was already rounded to the
        # output dtype above, so widening both addends to fp32 and rounding once
        # reproduces the eager two-kernel sequence ``add(matmul_out, residual)``
        # bit-for-bit (aten elementwise add computes in fp32 opmath).
        r_ptrs = R + stride_rm * offs_cm[:, None] + stride_rn * offs_cn[None, :]
        r = tl.load(r_ptrs, mask=c_mask, other=0.0)
        c = (c.to(tl.float32) + r.to(tl.float32)).to(C.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def _w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float32,
    residual: Optional[torch.Tensor] = None,
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
    if residual is not None:
        assert residual.shape == C_shape, "residual must match the matmul output shape"
        assert residual.dtype == output_dtype, "residual must match the matmul output dtype"
        assert residual.dim() >= 2 and residual.is_contiguous()

    # Rowwise direct-store decode GEMV (idea_0009): the exact measured M=1
    # DeepSeek-V4-Flash TP4 per-rank shapes bypass both incumbent paths -- one
    # flat rowwise kernel replaces full-K MMA launches and the split-K
    # (zero-fill + atomic matmul + finish-cast) triple, with the residual merge
    # add fused in the same epilogue as the full-K kernel. See the kernel
    # docstring/comment block for the measured schedule and numerics.
    if _use_rowwise_gemv(M, N, K, block_n, block_k, A, B, As, Bs):
        return _w8a8_gemv_rowwise(
            A, B, As, Bs, block_n, block_k, output_dtype, N, K, residual=residual
        )

    if _use_splitk_decode(M, N, K):
        C = _w8a8_block_fp8_matmul_splitk(A, B, As, Bs, block_n, block_k, output_dtype, M, N, K)
        if residual is not None:
            # The split-K path materializes its output via an fp32 atomic
            # accumulator + cast; keep the merge add as a separate elementwise op
            # there (identical rounding to the unfused sequence).
            C = C + residual
        return C

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
        C if residual is None else residual,
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
        0 if residual is None else residual.stride(-2),
        0 if residual is None else residual.stride(-1),
        BLOCK_SIZE_K=block_k,
        HAS_RESIDUAL=residual is not None,
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


# Split-K launch config for the decode GEMV. Two tuned bands (see the per-knob
# heuristics below): the legacy schedule tuned on B200 over K=7168 decode
# projection shapes (N in {256,576,1536,2304}; kernel_layout axis), and exact M=1,
# K=4096 DeepSeek-V4-Flash TP4 per-rank shapes (N in {1024,1536}; idea_0040,
# kernel_autotune axis).
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


def _use_dsv4_tp4_m1_splitk_schedule(M: Optional[int], N: int, K: Optional[int]) -> bool:
    """Return whether the exact measured DeepSeek-V4-Flash TP4 schedule applies."""
    return M == 1 and K == 4096 and N in (1024, 1536)


def _splitk_block_n(N: int, K: Optional[int] = None, M: Optional[int] = None) -> int:
    """BLOCK_SIZE_N for the split-K decode GEMV, scaled with N and measured shape.

    Small N is CTA-starved so a narrow N-tile spreads work over more CTAs; large N
    has enough output tiles that a wide N-tile (better MMA / fewer atomic stores)
    wins. Measured B200 optima at K=7168: N=256->32, N=576->64, N>=1024
    (1536/2304)->128.

    Exact M=1, K=4096 shapes (idea_0040, DeepSeek-V4-Flash TP4 per-rank decode: fused
    wq_a+wkv N=1536, shared w1+w3 / grouped wo_a N=1024, all K=4096): with only 32
    K-blocks the wide 128-N tile leaves too few CTAs in flight; BLOCK_SIZE_N=64
    wins in the measured L2-cold and L2-hot regimes. All other shapes preserve
    the legacy K=7168-tuned schedule.
    """
    if _use_dsv4_tp4_m1_splitk_schedule(M, N, K):
        return 64
    if N <= 512:
        return 32
    if N >= 1024:
        return 128
    return 64


def _splitk_split_k(N: int, K: Optional[int] = None, M: Optional[int] = None) -> int:
    """SPLIT_K (K-reduction CTA fan-out) for the split-K decode GEMV.

    The K reduction is partitioned across ``SPLIT_K`` CTAs per output tile, so
    the launch grid is ``cdiv(N, BLOCK_SIZE_N) * SPLIT_K`` and the atomic-reduction
    count scales with ``SPLIT_K``. The best grid is ~2-3 waves on the B200 SM array:
    narrow-N tiles (``N < 1024`` -> BLOCK_SIZE_N 32/64) yield few n-tiles and are
    CTA-starved, so they want a *deeper* K-split (more CTAs); wide 128-N tiles
    already have enough n-tiles, so a *shallower* split cuts atomic-reduction
    over-subscription. Measured B200 optima (BLOCK_SIZE_K=128, K=7168, M=1):
    N=256->48, N=576->48, N=1536->24, N=2304->16 (each ~-3.5..-4.0% vs the old
    fixed SPLIT_K=24; idea_0063, kernel_tile). idea_0025's fixed 24 was tuned over
    SPLIT_K<=32 and missed the deeper split the narrow-N shapes want.

    Exact M=1, K=4096 shapes (idea_0040): K=4096 has exactly 32 K-blocks, so the
    SPLIT_K=24 is ragged -- 8 CTAs per tile do 2 K-blocks while 16 do 1, and the
    2-block stragglers set the kernel tail. SPLIT_K=32 (1 K-block per CTA,
    balanced) + BLOCK_SIZE_N=64 + num_warps=2 wins the measured N=1024 single and
    grouped sites. N=1536 instead keeps SPLIT_K=24, whose drift-controlled cold
    result was slightly faster than 32. All other shapes preserve the legacy
    schedule.
    """
    if _use_dsv4_tp4_m1_splitk_schedule(M, N, K):
        return 32 if N == 1024 else _SPLITK_SPLIT_K
    if N < 1024:
        return 48
    if N <= 1792:
        return _SPLITK_SPLIT_K  # 24
    return 16


def _splitk_num_warps(N: int, K: Optional[int] = None, M: Optional[int] = None) -> int:
    """num_warps for the split-K decode GEMV.

    The exact idea_0040 M=1/K=4096/N={1024,1536} shapes use 2 warps; their short
    per-CTA K reduction does not benefit from the legacy 4-warps schedule. Every
    other shape keeps the tuned 4-warps default.
    """
    if _use_dsv4_tp4_m1_splitk_schedule(M, N, K):
        return 2
    return _SPLITK_NUM_WARPS


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
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    C_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Split-K block-FP8 GEMM for the decode GEMV (see kernel docstring above).

    Accumulates fp32 partials from ``SPLIT_K`` contraction-slices via atomics into a
    pre-zeroed fp32 buffer, then casts to ``output_dtype``. ``SPLIT_K`` /
    ``BLOCK_SIZE_N`` / ``BLOCK_SIZE_K`` / ``num_warps`` / ``num_stages`` default
    (``None``) to the tuned heuristics (``_splitk_split_k`` / ``_splitk_block_n`` /
    ``_splitk_block_k`` / ``_splitk_num_warps`` / ``_SPLITK_NUM_STAGES``), which
    select per (N, K) band; the microbench passes them explicitly to sweep the
    config.

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
        SPLIT_K = _splitk_split_k(N, K, M)
    if BLOCK_SIZE_N is None:
        BLOCK_SIZE_N = _splitk_block_n(N, K, M)
    if BLOCK_SIZE_K is None:
        BLOCK_SIZE_K = _splitk_block_k(N, block_k)
    if num_warps is None:
        num_warps = _splitk_num_warps(N, K, M)
    if num_stages is None:
        num_stages = _SPLITK_NUM_STAGES
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


# Rowwise direct-store decode GEMV backend (idea_0009, kernel_backend axis).
#
# At M=1 both incumbent paths carry structural overhead this memory-bound GEMV
# does not need:
#   * the full-K MMA kernel loads B as (BLOCK_K, BLOCK_N) tiles -- 128-byte
#     segments per output column -- and pads the M=1 row to a 16-row MMA tile;
#   * the split-K path adds a zero-fill kernel before and an fp32->bf16 finish
#     cast kernel after every call (three CUDA-graph nodes per GEMV) plus an
#     fp32 atomic reduction.
# This kernel instead assigns each CTA a small band of B *rows* and streams them
# along K in (BLOCK_N, GROUPS, group_k) tiles: per row the K segment read per
# load is ``GROUPS * group_k`` bytes (up to a whole 4 KiB row) instead of 128,
# there are no atomics, no MMA row padding, and the bf16 result (plus the
# optional fused residual add) is stored directly -- one kernel, no fill/cast.
#
# Numerics: for each row the per-scale-group partial dot products are formed in
# fp32 and accumulated in a deterministic order (sequential over GROUPS-sized
# chunks, tree-reduced within a chunk), with the same ``(partial * a_s) * b_s``
# scale association as the incumbent kernels. It therefore matches the full-K
# kernel up to the intra-group summation-tree order (measured 0-1 / 16384
# bf16-boundary flips vs an fp64 dequant ground truth, the same count the
# incumbent MMA kernel shows) and *removes* the split-K path's atomic-order
# run-to-run nondeterminism on the shapes it covers.
#
# The launch schedule was selected by a drift-controlled round-robin CUDA-graph
# microbench on B200 (L2-cold weight rotation) over the exact DeepSeek-V4-Flash
# TP4 per-rank M=1 decode shapes; each entry below beat the incumbent dispatch
# (including the split-K fill+cast overhead) in 3/3 alternating repeats.
# Persistent / work-queue variants of this kernel (grid-strided task loop,
# one-wave grid) were also swept and LOST to the flat one-task-per-CTA launch at
# every shape, so the backend intentionally launches a plain flat grid.
# Non-listed shapes, M>=2, and non-128 quant blocks keep the incumbent paths.

# (N, K) -> (BLOCK_N rows/CTA, GROUPS k-groups/iter, num_warps, num_stages)
_W8A8_TP4_M1_ROWWISE_CFG = {
    (1536, 4096): (1, 32, 2, 3),  # fused wq_a + wkv          (was split-K 24x24)
    (1024, 4096): (1, 32, 2, 5),  # shared w1+w3 / wo_a sites (was split-K 16x32)
    (16384, 1024): (32, 8, 4, 3),  # fused wq_b + indexer.wq_b (was full-K BN128)
    (8192, 1024): (8, 8, 4, 3),  # wq_b                      (was full-K BN64)
    (4096, 512): (4, 4, 2, 5),  # shared w2 (+residual)     (was full-K BN32)
    (4096, 2048): (4, 8, 2, 5),  # wo_b                      (was full-K BN32)
}


def _use_rowwise_gemv(
    M: int,
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
) -> bool:
    """Gate the rowwise decode GEMV to the exact measured M=1 TP4 shapes."""
    if M != 1 or block_n != 128 or block_k != 128:
        return False
    cfg = _W8A8_TP4_M1_ROWWISE_CFG.get((N, K))
    if cfg is None:
        return False
    block_rows, groups = cfg[0], cfg[1]
    # Defensive: every table shape satisfies these exactly-divisible tilings.
    if N % block_rows or K % (groups * block_k):
        return False
    # The kernel reads flat contiguous rows (A, As) and row-major B / Bs.
    return A.stride(-1) == 1 and B.stride(1) == 1 and As.stride(-1) == 1 and Bs.stride(1) == 1


@triton.jit
def _w8a8_gemv_rowwise_kernel(
    A,
    B,
    C,
    As,
    Bs,
    R,
    K,
    stride_bn,
    stride_bsn,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_g = tl.arange(0, GROUPS)
    offs_k = tl.arange(0, GROUP_K)
    num_k_groups = K // GROUP_K

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    bs_row = offs_n // GROUP_N
    for g0 in range(0, num_k_groups, GROUPS):
        kk = (g0 + offs_g)[:, None] * GROUP_K + offs_k[None, :]
        a = tl.load(A + kk).to(tl.float32)  # (GROUPS, GROUP_K)
        b = tl.load(B + offs_n[:, None, None] * stride_bn + kk[None, :, :]).to(tl.float32)
        part = tl.sum(b * a[None, :, :], axis=2)  # (BLOCK_N, GROUPS)
        a_s = tl.load(As + g0 + offs_g).to(tl.float32)
        b_s = tl.load(Bs + bs_row[:, None] * stride_bsn + (g0 + offs_g)[None, :]).to(tl.float32)
        # Same scale association as the incumbent kernels: (partial * a_s) * b_s.
        acc += tl.sum(part * a_s[None, :] * b_s, axis=1)

    c = acc.to(C.dtype.element_ty)
    if HAS_RESIDUAL:
        # Mirror the full-K epilogue bit-for-bit: round the accumulator to the
        # output dtype first, then add the residual in fp32 and round once.
        r = tl.load(R + offs_n).to(tl.float32)
        c = (c.to(tl.float32) + r).to(C.dtype.element_ty)
    tl.store(C + offs_n, c)


def _w8a8_gemv_rowwise(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_n: int,
    block_k: int,
    output_dtype: torch.dtype,
    N: int,
    K: int,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch the rowwise direct-store decode GEMV (gate via _use_rowwise_gemv)."""
    block_rows, groups, num_warps, num_stages = _W8A8_TP4_M1_ROWWISE_CFG[(N, K)]
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    grid = (N // block_rows,)
    _w8a8_gemv_rowwise_kernel[grid](
        A,
        B,
        C,
        As,
        Bs,
        C if residual is None else residual,
        K,
        B.stride(0),
        Bs.stride(0),
        GROUP_N=block_n,
        GROUP_K=block_k,
        BLOCK_N=block_rows,
        GROUPS=groups,
        HAS_RESIDUAL=residual is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


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
    "auto_deploy::torch_fake_quant_finegrained_fp8_linear_residual_add", mutates_args=()
)
def torch_fake_quant_finegrained_fp8_linear_residual_add(
    input: torch.Tensor,  # [..., K]
    weight_quantized: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # must be None (the fusion only matches bias-free linears)
    input_scale: List[torch.Tensor],  # unused for FineGrained FP8 (input quantized on the fly)
    weight_scale: List[torch.Tensor],  # [weight_scale_inv]
    input_zp: List[torch.Tensor],  # unused
    weight_zp: List[torch.Tensor],  # unused
    tp_mode: str = "none",
    output_sizes: Optional[List[int]] = None,
    tp_min_local_shape: int = 1,
    layer_type: str = "unknown",
    input_scale_fmt: str = "",
    residual: Optional[torch.Tensor] = None,  # [..., N] added to the matmul output
) -> torch.Tensor:
    """``torch_fake_quant_finegrained_fp8_linear`` with a fused trailing merge add.

    Computes ``torch_fake_quant_finegrained_fp8_linear(input, ...) + residual`` with
    the add folded into the W8A8 block-FP8 matmul epilogue (one kernel instead of
    matmul + standalone elementwise add). The matmul accumulator is rounded to the
    output dtype *before* the add, so the result is bit-for-bit identical to the
    unfused sequence. Emitted by the ``fuse_fp8_linear_allreduce_add`` transform for
    the MoE routed+shared merge add feeding a distributed all_reduce (the summed
    tensor is the collective's input buffer).
    """
    assert bias is None, "fused residual-add linear only supports bias-free linears"
    assert residual is not None, "residual tensor is required"
    weight_scale_inv = weight_scale[0]

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
        residual=residual.contiguous(),
    )

    return output.to(dtype=input.dtype)


@torch_fake_quant_finegrained_fp8_linear_residual_add.register_fake
def _torch_fake_quant_finegrained_fp8_linear_residual_add_fake(
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
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    out_features = weight_quantized.shape[0]
    return torch.empty((*input.shape[:-1], out_features), dtype=input.dtype, device=input.device)


@triton.jit
def _swiglu_clamp_act_quant_kernel(
    g_ptr,
    u_ptr,
    y_ptr,
    s_ptr,
    limit,
    g_row_stride,
    u_row_stride,
    GROUPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
):
    """Fused (clamped) SwiGLU + block-wise FP8 activation quantization.

    One program handles one ``BLOCK_SIZE`` (=quant group) chunk of one row. It
    reproduces the eager chain
    ``clamp(gate, max=L); clamp(up, -L, L); silu(gate.float()) * up.float();
    .to(model_dtype); _act_quant_kernel`` bit for bit:

    * clamps use ``tl.where`` so NaN inputs propagate exactly like ``aten.clamp``
      (a plain ``tl.minimum`` would replace NaN with the bound);
    * comparing/selecting in fp32 after the exact bf16->fp32 widening selects the
      same value ``aten.clamp`` picks in bf16 (the bound is bf16-representable);
    * silu uses aten's fp32 formula ``x / (1 + expf(-x))``;
    * the product is rounded to the model dtype in-register at the same point the
      reference chain stores it, then re-widened, so the quantization sees the
      identical values ``_act_quant_kernel`` loads from memory;
    * the scale math (both fmt branches) matches ``_act_quant_kernel`` line for
      line, including storing the fp32-divided payload with a bf16-rounded scale.
    """
    pid = tl.program_id(axis=0)
    row = pid // GROUPS
    grp = pid % GROUPS
    cols = grp * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gate = tl.load(g_ptr + row * g_row_stride + cols).to(tl.float32)
    up = tl.load(u_ptr + row * u_row_stride + cols).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.where(gate > limit, limit, gate)
        up = tl.where(up < -limit, -limit, up)
        up = tl.where(up > limit, limit, up)
    hidden = (gate / (1.0 + tl.exp(-gate))) * up
    hidden = hidden.to(g_ptr.dtype.element_ty).to(tl.float32)
    amax = tl.max(tl.abs(hidden))
    if ROUND_SCALE:
        amax = tl.maximum(amax, 1e-4)
        s = amax / 448.0
        s = tl.exp2(tl.ceil(tl.log2(s)))
    else:
        s = amax / 448.0
        # Clamp scale so that all-zero blocks produce 0/eps = 0 instead of 0/0 = NaN.
        s = tl.maximum(s, 1e-12)
    y = hidden / s
    y = y.to(y_ptr.dtype.element_ty)
    # Outputs are contiguous [M, GROUPS*BLOCK_SIZE] / [M, GROUPS], so the row-major
    # program id addresses both directly.
    tl.store(y_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), y)
    tl.store(s_ptr + pid, s)


@torch.library.custom_op("auto_deploy::torch_fp8_swiglu_clamp_act_quant", mutates_args=())
def torch_fp8_swiglu_clamp_act_quant(
    gate: torch.Tensor,  # [M, I], strided views allowed (stride(-1)==1)
    up: torch.Tensor,  # [M, I], same shape/dtype as gate
    limit: Optional[float],  # swiglu clamp bound; None/<=0 disables the clamps
    block_size: int,  # activation quant group (== matmul block_k)
    input_scale_fmt: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused clamped-SwiGLU + block-FP8 activation quant feeding a *_prequant linear.

    Computes, in one kernel launch, the DeepSeek-V4 shared-expert epilogue between
    the merged gate/up projection and the down projection::

        gate = clamp(gate, max=limit)
        up = clamp(up, min=-limit, max=limit)
        hidden = (silu(gate.float()) * up.float()).to(gate.dtype)
        return _safe_act_quant(hidden, block_size, input_scale_fmt)

    ``gate``/``up`` may be non-contiguous last-dim-unit-stride views (e.g. the two
    ``torch.narrow`` halves of one fused gate_up GEMM output), so the sliced halves
    are consumed in place without materializing them. Bit-for-bit identical to the
    unfused aten chain + ``_act_quant_kernel`` (see the kernel docstring).
    """
    assert gate.dim() == 2 and up.dim() == 2, "expected flattened [tokens, features] inputs"
    assert gate.shape == up.shape and gate.dtype == up.dtype
    assert gate.stride(-1) == 1 and up.stride(-1) == 1
    num_tokens, width = gate.shape
    assert width % block_size == 0
    groups = width // block_size
    y = torch.empty((num_tokens, width), dtype=torch.float8_e4m3fn, device=gate.device)
    # Keep scale metadata in the model dtype (matches _safe_act_quant).
    s = torch.empty((num_tokens, groups), dtype=gate.dtype, device=gate.device)
    has_limit = limit is not None and limit > 0
    round_scale = input_scale_fmt.lower() == "ue8m0"
    grid = (num_tokens * groups,)
    # One quant group per program: a tiny single-reduction workload, one warp
    # (same rationale as _act_quant_kernel's num_warps=1).
    _swiglu_clamp_act_quant_kernel[grid](
        gate,
        up,
        y,
        s,
        float(limit) if has_limit else 0.0,
        gate.stride(0),
        up.stride(0),
        GROUPS=groups,
        BLOCK_SIZE=block_size,
        HAS_LIMIT=has_limit,
        ROUND_SCALE=round_scale,
        num_warps=1,
    )
    return y, s


@torch_fp8_swiglu_clamp_act_quant.register_fake
def _torch_fp8_swiglu_clamp_act_quant_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: Optional[float],
    block_size: int,
    input_scale_fmt: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    qhidden = torch.empty(gate.shape, dtype=torch.float8_e4m3fn, device=gate.device)
    scale = torch.empty(
        (*gate.shape[:-1], gate.shape[-1] // block_size), dtype=gate.dtype, device=gate.device
    )
    return qhidden, scale


@torch.library.custom_op(
    "auto_deploy::torch_fake_quant_finegrained_fp8_linear_residual_add_prequant",
    mutates_args=(),
)
def torch_fake_quant_finegrained_fp8_linear_residual_add_prequant(
    qinput: torch.Tensor,  # [..., K] float8_e4m3fn (pre-quantized activation)
    input_scale: torch.Tensor,  # [..., K//block_k] per-block act scale (model dtype)
    weight_quantized: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # must be None (the fusion only matches bias-free linears)
    weight_scale: List[torch.Tensor],  # [weight_scale_inv]
    residual: torch.Tensor,  # [..., N] added to the matmul output
) -> torch.Tensor:
    """Matmul half of ``torch_fake_quant_finegrained_fp8_linear_residual_add``.

    Consumes a pre-quantized activation + per-block scale (e.g. produced by
    ``torch_fp8_swiglu_clamp_act_quant``) and runs the same block-FP8 W8A8 matmul
    with the merge add folded into the epilogue. Output dtype is recovered from
    ``input_scale.dtype`` exactly like the prequant linear, so the result is
    bit-for-bit identical to the internally-quantizing residual-add op.
    """
    assert bias is None, "fused residual-add linear only supports bias-free linears"
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
        residual=residual.contiguous(),
    )

    return output.to(dtype=out_dtype)


@torch_fake_quant_finegrained_fp8_linear_residual_add_prequant.register_fake
def _torch_fake_quant_finegrained_fp8_linear_residual_add_prequant_fake(
    qinput: torch.Tensor,
    input_scale: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: List[torch.Tensor],
    residual: torch.Tensor,
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
