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


def _use_quant_prologue(input_scale_fmt: str) -> bool:
    """Use the measured in-kernel quant path for power-of-two activation scales."""
    return input_scale_fmt.lower() == "ue8m0"


@triton.jit
def _act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr, ROUND_SCALE: tl.constexpr):
    """Block-wise FP8 activation quantization (HF's act_quant_kernel plus a
    >=1e-12 scale clamp so all-zero blocks produce 0 instead of 0/0 = NaN)."""
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


@triton.jit
def _ue8m0_quant_rows(x):
    """In-register ue8m0 block-FP8 quant of one whole quant group per row of ``x``
    (identical scale math and rounding to ``_act_quant_kernel``'s ROUND_SCALE path)."""
    amax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
    s = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    return (x / s[:, None]).to(tl.float8e4nv), s


def _safe_act_quant(x: torch.Tensor, block_size: int = 128, input_scale_fmt: str = "") -> tuple:
    """Block-wise FP8 activation quantization (drop-in for HF's ``act_quant``);
    the NaN fix lives inside the kernel, so no fixup tensors and CUDA-graph safe."""
    assert x.is_contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # Keep scale metadata in the model dtype to avoid FP32->BF16 cast kernels
    # when the tensor is consumed by downstream MoE/quantized paths.
    s = x.new_empty(*x.shape[:-1], x.shape[-1] // block_size, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)  # noqa: E731
    round_scale = input_scale_fmt.lower() == "ue8m0"
    # num_warps=1: each program is a single 128-elem reduction, so one warp keeps it
    # intra-warp (no smem barrier); measured faster than the 4-warp default.
    _act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size, ROUND_SCALE=round_scale, num_warps=1)
    return y, s


# Adapted from sgl-project/sglang, vendored because transformers 5.5.x removed
# w8a8_block_fp8_matmul_triton. BLOCK_SIZE_K is NOT autotuned: it must stay pinned
# to the quantization group_k or the in-loop scale index straddles a scale block
# and corrupts the result. Configs measured on B200 over the DSV4 per-rank shapes.

_W8A8_BLOCK_FP8_MATMUL_CONFIGS = [
    # Decode / small-M: BLOCK_SIZE_N 32 vs 64 selected by the autotuner; num_warps
    # pinned to 4 (the 4-vs-8 gap is below do_bench resolution -> selection flicker).
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
    # Prefill / large-M: ONLY BLOCK_SIZE_N=128 + num_warps=8. Other large-tile
    # configs are run-to-run NON-deterministic on sm100 at M>=256/K>=2048 (fp8-MMA
    # pipelining glitch); do not add faster-but-racy configs — the autotuner
    # selects on latency only.
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4
    ),
]


@triton.autotune(
    configs=_W8A8_BLOCK_FP8_MATMUL_CONFIGS,
    key=["M", "N", "K"],
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
        # Fused merge add: ``c`` is already rounded to the output dtype, so fp32
        # widen + add + one rounding reproduces the eager add(matmul_out, residual).
        r_ptrs = R + stride_rm * offs_cm[:, None] + stride_rn * offs_cn[None, :]
        r = tl.load(r_ptrs, mask=c_mask, other=0.0)
        c = (c.to(tl.float32) + r.to(tl.float32)).to(C.dtype.element_ty)
    tl.store(c_ptrs, c, mask=c_mask)


def _w8a8_block_fp8_matmul_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    As: Optional[torch.Tensor],
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float32,
    residual: Optional[torch.Tensor] = None,
    input_scale_fmt: str = "",
) -> torch.Tensor:
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    if As is None:
        # Deferred activation quant: the decode GEMV/split-K kernels fuse the ue8m0
        # quant into their prologue; other shapes quantize standalone below.
        assert A.is_contiguous() and A.shape[-1] % block_k == 0
    else:
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

    if residual is not None:
        assert residual.shape == C_shape, "residual must match the matmul output shape"
        assert residual.dtype == output_dtype, "residual must match the matmul output dtype"
        assert residual.dim() >= 2 and residual.is_contiguous()

    # Dispatch: exact measured M=1 TP4 shapes -> rowwise direct-store GEMV;
    # small-M/long-K (CTA-starved base grid) -> split-K; else autotuned full-K.
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

    if As is None:
        # No fused-prologue decode path for this shape (e.g. prefill).
        A, As = _safe_act_quant(A, block_k, input_scale_fmt)

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


# Split-K decode GEMV: at decode M with long K the full-K kernel launches only
# cdiv(N, BLOCK_N) CTAs and is K-loop-latency-bound. This kernel fans the K
# reduction out across SPLIT_K CTAs (grid axis 1, strided K-blocks) and
# tl.atomic_add's fp32 partials into a pre-zeroed accumulator — equal to the
# serial sum up to fp32 add ordering (~1e-7 rel, far under fp8 quant error).
# BLOCK_SIZE_K stays pinned to group_k (scale-index correctness).
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
    QUANT_PROLOGUE: tl.constexpr,
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
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        offs_ks = (k * BLOCK_SIZE_K) // group_k
        if QUANT_PROLOGUE:
            # A is the raw activation; BLOCK_SIZE_K == group_k makes each tile row
            # one quant group, so quantize in-register and feed fp8 to tl.dot.
            x = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0).to(tl.float32)
            a, a_s = _ue8m0_quant_rows(x)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
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


# Split-K launch config for the decode GEMV, tuned on B200 over the K=7168 decode
# projection shapes (N in {256,576,1536,2304}) plus the exact M=1/K=4096
# DeepSeek-V4-Flash TP4 per-rank shapes (N in {1024,1536}).
_SPLITK_BLOCK_SIZE_M = 16
_SPLITK_NUM_STAGES = 3
# Gate: small M (decode) + long K reduction, where the base grid is CTA-starved.
_SPLITK_MAX_M = 4
_SPLITK_MIN_K = 4096


def _use_splitk_decode(M: int, N: int, K: int) -> bool:
    return M <= _SPLITK_MAX_M and K >= _SPLITK_MIN_K


def _splitk_schedule(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """(BLOCK_SIZE_N, SPLIT_K, num_warps) for the split-K decode GEMV.

    Small N is CTA-starved: narrow N-tiles plus a deeper K-split spread work over
    more CTAs; wide-N shapes prefer fat tiles and a shallower split (fewer
    atomics). The exact M=1/K=4096 TP4 shapes were re-measured separately: with
    only 32 K-blocks a 128-wide tile is too coarse (BLOCK_N=64 wins), N=1024
    wants the balanced SPLIT_K=32 (24 leaves 2-K-block stragglers), and the
    short per-CTA reduction only needs 2 warps.
    """
    if M == 1 and K == 4096 and N in (1024, 1536):
        return 64, (32 if N == 1024 else 24), 2
    block_n = 32 if N <= 512 else (64 if N < 1024 else 128)
    split_k = 48 if N < 1024 else (24 if N <= 1792 else 16)
    return block_n, split_k, 4


def _w8a8_block_fp8_matmul_splitk(
    A: torch.Tensor,
    B: torch.Tensor,
    As: Optional[torch.Tensor],
    Bs: torch.Tensor,
    block_n: int,
    block_k: int,
    output_dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
    *,
    C_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Split-K block-FP8 GEMM for the decode GEMV (see kernel comment above).

    ``C_out`` is a pre-zeroed fp32 column slice from the grouped-GEMV path (all
    groups share one allocation and one finish cast). Atomic accumulation order is
    nondeterministic: values on a BF16 rounding boundary can differ by one ULP.
    """
    block_size_n, split_k, num_warps = _splitk_schedule(M, N, K)
    if As is None:
        # Deferred-quant path: the per-row group amax is reduced over one K tile,
        # so the tile must cover exactly one whole quant group.
        assert K % block_k == 0, f"quant prologue requires K % block_k == 0 (K={K})"
    if C_out is not None:
        C_acc = C_out
    else:
        C_shape = A.shape[:-1] + (N,)
        # fp32 accumulator, pre-zeroed for the atomic reduction across SPLIT_K CTAs.
        C_acc = A.new_zeros(C_shape, dtype=torch.float32)

    grid = (
        triton.cdiv(M, _SPLITK_BLOCK_SIZE_M) * triton.cdiv(N, block_size_n),
        split_k,
    )
    _w8a8_block_fp8_matmul_splitk_kernel[grid](
        A,
        B,
        C_acc,
        A if As is None else As,
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
        0 if As is None else As.stride(-2),
        0 if As is None else As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        BLOCK_SIZE_M=_SPLITK_BLOCK_SIZE_M,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_k,
        SPLIT_K=split_k,
        QUANT_PROLOGUE=As is None,
        num_warps=num_warps,
        num_stages=_SPLITK_NUM_STAGES,
    )
    if output_dtype == torch.float32:
        return C_acc
    return C_acc.to(output_dtype)


# Rowwise direct-store decode GEMV: gives each CTA a band of B rows and streams
# whole K segments per row — one kernel (no zero-fill/cast pair), no atomics,
# deterministic accumulation. Schedules below were measured per exact DSV4-Flash
# TP4 M=1 shape on B200; non-listed shapes / M>=2 keep the incumbent paths.

# (N, K) -> (BLOCK_N rows/CTA, GROUPS k-groups/iter, num_warps, num_stages)
_W8A8_TP4_M1_ROWWISE_CFG = {
    (1536, 4096): (1, 32, 2, 3),  # fused wq_a + wkv          (was split-K 24x24)
    (1024, 4096): (1, 32, 2, 5),  # shared w1+w3 / wo_a sites (was split-K 16x32)
    (16384, 1024): (32, 8, 4, 3),  # fused wq_b + indexer.wq_b (was full-K BN128)
    (8192, 1024): (8, 8, 4, 3),  # wq_b                      (was full-K BN64)
    (4096, 512): (4, 4, 2, 5),  # shared w2 (+residual)     (was full-K BN32)
    (4096, 2048): (4, 8, 2, 5),  # wo_b                      (was full-K BN32)
}

# Fused-quant-prologue variant: the in-kernel quant shifts the register balance,
# so each shape was re-swept. Key set must match _W8A8_TP4_M1_ROWWISE_CFG
# (shared _use_rowwise_gemv gate).
_W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG = {
    (1536, 4096): (4, 32, 4, 3),  # 4.01 vs 4.08 us pair (BLOCK_N=1 regressed)
    (1024, 4096): (2, 32, 4, 5),  # 3.53 vs 4.20 us pair
    (16384, 1024): (32, 8, 4, 3),  # 4.88 vs 5.58 us pair (standalone cfg kept)
    (8192, 1024): (32, 8, 4, 2),  # 3.51 vs 4.92 us pair
    (4096, 512): (8, 4, 2, 5),  # 2.34 vs 3.56 us pair
    (4096, 2048): (4, 16, 4, 2),  # 4.11 vs 5.42 us pair
}


def _rowwise_gemv_cfg(N: int, K: int, prequantized: bool) -> Optional[Tuple[int, int, int, int]]:
    """(BLOCK_N, GROUPS, num_warps, num_stages) for the rowwise GEMV, or None.

    ``prequantized`` (``As is not None``) selects the standalone-quant table;
    the deferred-quant prologue variant uses its own re-swept table.
    """
    table = _W8A8_TP4_M1_ROWWISE_CFG if prequantized else _W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG
    return table.get((N, K))


def _use_rowwise_gemv(
    M: int,
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    A: torch.Tensor,
    B: torch.Tensor,
    As: Optional[torch.Tensor],
    Bs: torch.Tensor,
) -> bool:
    """Gate the rowwise decode GEMV to the exact measured M=1 TP4 shapes."""
    if M != 1 or block_n != 128 or block_k != 128:
        return False
    cfg = _rowwise_gemv_cfg(N, K, As is not None)
    if cfg is None:
        return False
    block_rows, groups = cfg[0], cfg[1]
    # Defensive: every table shape satisfies these exactly-divisible tilings.
    if N % block_rows or K % (groups * block_k):
        return False
    # The kernel reads flat contiguous rows (A, As) and row-major B / Bs.
    # As is None on the deferred-quant path (in-kernel activation-quant prologue).
    if As is not None and As.stride(-1) != 1:
        return False
    return A.stride(-1) == 1 and B.stride(1) == 1 and Bs.stride(1) == 1


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
    QUANT_PROLOGUE: tl.constexpr,
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
        if QUANT_PROLOGUE:
            # A is the raw model-dtype activation: ue8m0 fp8 round-trip in-register.
            x = tl.load(A + kk).to(tl.float32)  # (GROUPS, GROUP_K)
            a, a_s = _ue8m0_quant_rows(x)
            a = a.to(tl.float32)
        else:
            a = tl.load(A + kk).to(tl.float32)  # (GROUPS, GROUP_K)
            a_s = tl.load(As + g0 + offs_g).to(tl.float32)
        b = tl.load(B + offs_n[:, None, None] * stride_bn + kk[None, :, :]).to(tl.float32)
        part = tl.sum(b * a[None, :, :], axis=2)  # (BLOCK_N, GROUPS)
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
    As: Optional[torch.Tensor],
    Bs: torch.Tensor,
    block_n: int,
    block_k: int,
    output_dtype: torch.dtype,
    N: int,
    K: int,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch the rowwise direct-store decode GEMV (gate via _use_rowwise_gemv).

    ``As is None`` selects the deferred-quant path: ``A`` is the raw model-dtype
    activation and the kernel quantizes each 128-group in its prologue.
    """
    block_rows, groups, num_warps, num_stages = _rowwise_gemv_cfg(N, K, As is not None)
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)
    grid = (N // block_rows,)
    _w8a8_gemv_rowwise_kernel[grid](
        A,
        B,
        C,
        A if As is None else As,
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
        QUANT_PROLOGUE=As is None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


_FINEGRAINED_FP8_CANONICAL_BLOCK_N = 128


def _finegrained_fp8_block_sizes(
    weight_quantized: torch.Tensor, weight_scale_inv: torch.Tensor
) -> Tuple[int, int]:
    """Infer (block_n, block_k) from the weight and per-block scale shapes.

    Recognize the standard 128-row checkpoint grid before falling back to shape inference,
    since a partial final N block makes the block size ambiguous from shape alone.
    """
    N, K = weight_quantized.shape
    scale_n, scale_k = weight_scale_inv.shape
    block_n = (
        _FINEGRAINED_FP8_CANONICAL_BLOCK_N
        if scale_n == triton.cdiv(N, _FINEGRAINED_FP8_CANONICAL_BLOCK_N)
        else triton.cdiv(N, scale_n)
    )
    block_k = triton.cdiv(K, scale_k)
    return block_n, block_k


def _finegrained_fp8_matmul(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    out_dtype: torch.dtype,
    *,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    input_scale_fmt: str = "",
    allow_quant_prologue: bool = False,
) -> torch.Tensor:
    """Shared body of the FineGrained FP8 linear ops: (quantize +) block-FP8 W8A8
    matmul (+ bias / fused residual add). ``input_scale is None`` means a raw
    activation — quantized here, or in the decode kernels' prologue when
    ``allow_quant_prologue``. Block sizes are inferred from the scale shapes."""
    block_n, block_k = _finegrained_fp8_block_sizes(weight_quantized, weight_scale_inv)
    if input_scale is not None:
        qinput, scale = input, input_scale
    elif allow_quant_prologue and _use_quant_prologue(input_scale_fmt):
        # Defer the activation quant into the decode kernels' prologue (or the
        # standalone fallback inside the matmul dispatch for non-decode shapes).
        qinput, scale = input, None
    else:
        qinput, scale = _safe_act_quant(input, block_k, input_scale_fmt)
    output = _w8a8_block_fp8_matmul_triton(
        qinput,
        weight_quantized,
        scale,
        weight_scale_inv,
        [block_n, block_k],
        output_dtype=out_dtype,
        residual=None if residual is None else residual.contiguous(),
        input_scale_fmt=input_scale_fmt,
    )
    if bias is not None:
        output = output + bias
    return output.to(dtype=out_dtype)


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
    return _finegrained_fp8_matmul(
        input,
        weight_quantized,
        weight_scale[0],
        input.dtype,
        bias=bias,
        input_scale_fmt=input_scale_fmt,
        allow_quant_prologue=True,
    )


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


@torch.library.custom_op(
    "auto_deploy::torch_fake_quant_finegrained_fp8_linear_residual_add", mutates_args=()
)
def torch_fake_quant_finegrained_fp8_linear_residual_add(
    input: torch.Tensor,  # [..., K]
    weight_quantized: torch.Tensor,  # [N, K] float8_e4m3fn
    bias: Optional[torch.Tensor],  # must be None (the fusion only matches bias-free linears)
    input_scale: List[torch.Tensor],  # [] for raw input, [scale] for pre-quantized input
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
    """``torch_fake_quant_finegrained_fp8_linear`` with the merge add fused into the
    matmul epilogue (one kernel instead of matmul + elementwise add); same rounding
    as the unfused sequence. ``input`` is raw (``input_scale=[]``) or pre-quantized
    FP8 (``input_scale=[scale]``). Emitted by ``fuse_fp8_linear_allreduce_add`` for
    the MoE routed+shared merge add feeding an all_reduce.
    """
    assert bias is None, "fused residual-add linear only supports bias-free linears"
    assert residual is not None, "residual tensor is required"
    activation_scale = input_scale[0] if input_scale else None
    out_dtype = activation_scale.dtype if activation_scale is not None else input.dtype
    return _finegrained_fp8_matmul(
        input,
        weight_quantized,
        weight_scale[0],
        out_dtype,
        input_scale=activation_scale,
        residual=residual,
        input_scale_fmt=input_scale_fmt,
    )


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
    out_dtype = input_scale[0].dtype if input_scale else input.dtype
    return torch.empty((*input.shape[:-1], out_features), dtype=out_dtype, device=input.device)


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

    One program per quant-group chunk of one row, reproducing the eager chain
    ``clamp(gate/up) -> silu(gate.f32) * up.f32 -> .to(model_dtype) ->
    _act_quant_kernel`` with the same rounding points. Clamps use ``tl.where`` so
    NaN propagates like ``aten.clamp`` (``tl.minimum`` would swallow it), and the
    product is rounded to the model dtype in-register at the same point the
    reference stores it.
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
    """Fused clamped-SwiGLU + block-FP8 activation quant feeding a down projection.

    Computes, in one kernel launch, the DeepSeek-V4 shared-expert epilogue between
    the merged gate/up projection and the down projection::

        gate = clamp(gate, max=limit)
        up = clamp(up, min=-limit, max=limit)
        hidden = (silu(gate.float()) * up.float()).to(gate.dtype)
        return _safe_act_quant(hidden, block_size, input_scale_fmt)

    ``gate``/``up`` may be strided views (the two narrow halves of one fused
    gate_up GEMM output), consumed in place without materializing them.
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
    if weight_quantized.dtype != torch.float8_e4m3fn:
        raise TypeError("Grouped FineGrained FP8 path requires float8_e4m3fn weight")

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
    block_n, block_k = _finegrained_fp8_block_sizes(weight_quantized, weight_scale_inv)

    input_contiguous = input.contiguous()
    rank = out_rows // num_groups
    lead_shape = input.shape[:-2]

    if num_groups > 1 and (rank % block_n != 0 or scale_n % num_groups != 0):
        # The checkpoint scale grid covers the flattened weight. If a scale block crosses a
        # logical group boundary, it cannot be reshaped into independent per-group grids.
        qinput, input_scales = _safe_act_quant(input_contiguous, block_k, input_scale_fmt)
        qinput_blocks = qinput.reshape(*input_contiguous.shape[:-1], -1, block_k)
        input_dequant = (qinput_blocks.to(input.dtype) * input_scales.unsqueeze(-1)).reshape_as(
            input_contiguous
        )
        weight_dequant = dequant_fp8_weight_two_dim_block_grid(
            weight_quantized,
            weight_scale_inv,
            block_n,
            block_k,
            dtype=input.dtype,
        )
        weight_grouped = weight_dequant.view(num_groups, rank, in_features)
        output = torch.matmul(
            input_dequant.unsqueeze(-2),
            weight_grouped.transpose(-1, -2),
        ).squeeze(-2)
        output = output.flatten(-2)
        if bias is not None:
            output = output + bias.reshape(out_rows).to(output.dtype)
        return output.to(dtype=input.dtype)

    # Direct grouped block-FP8 W8A8 matmul: keep both operands in FP8 and let the
    # block-FP8 kernel apply the per-block scales in its fp32 accumulator — no
    # quant round-trip, no bf16 weight dequant (halves the HBM read of this
    # memory-bound GEMV). Per-rank DSV4 wo_a has num_groups == 1 (a single 2D GEMM);
    # block-aligned num_groups > 1 launches the same kernel per group.
    m_tokens = input_contiguous.numel() // (num_groups * in_features)
    # Deferred activation quant: the decode split-K/GEMV kernels fuse the
    # ue8m0 quant into their prologue; other branches quantize standalone.
    defer_quant = _use_quant_prologue(input_scale_fmt) and (
        num_groups == 1 or _use_splitk_decode(m_tokens, rank, in_features)
    )
    if defer_quant:
        qin = input_contiguous.reshape(m_tokens, num_groups, in_features)
        sin = None
    else:
        qinput, input_scales = _safe_act_quant(input_contiguous, block_k, input_scale_fmt)
        qin = qinput.reshape(m_tokens, num_groups, in_features)
        sin = input_scales.reshape(m_tokens, num_groups, input_scales.shape[-1])
    if num_groups == 1:
        out2d = _w8a8_block_fp8_matmul_triton(
            qin[:, 0, :],
            weight_quantized,
            None if defer_quant else sin[:, 0, :],
            weight_scale_inv,
            [block_n, block_k],
            output_dtype=input.dtype,
            input_scale_fmt=input_scale_fmt,
        )
        output = out2d.reshape(*lead_shape, out_rows)
    elif _use_splitk_decode(m_tokens, rank, in_features):
        # Accumulate every group into ONE pre-zeroed fp32 buffer laid out like the
        # stacked result (each group writes its disjoint column slice via strides),
        # then one finish cast — instead of a per-group zero-fill/cast pair plus a
        # torch.stack copy. Per-group math and launch config are unchanged.
        weight_grouped = weight_quantized.view(num_groups, rank, in_features)
        scale_rows = scale_n // num_groups
        scale_grouped = weight_scale_inv.view(num_groups, scale_rows, scale_k)
        acc = qin.new_zeros((m_tokens, out_rows), dtype=torch.float32)
        for g in range(num_groups):
            _w8a8_block_fp8_matmul_splitk(
                qin[:, g, :],
                weight_grouped[g],
                None if defer_quant else sin[:, g, :],
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
