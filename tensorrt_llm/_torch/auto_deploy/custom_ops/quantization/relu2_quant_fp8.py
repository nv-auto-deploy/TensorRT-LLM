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

"""Fused ReLU² + FP8 quantization Triton kernel.

Replaces two PyTorch elementwise kernels (relu, pow) plus the
scaleMatrixPerTensorVec FP8-quantize step inside trtllm_quant_fp8_linear
with a single Triton kernel:

    out_fp8 = clamp(max(x, 0)^2 / scale, FP8_MIN, FP8_MAX).to(float8_e4m3fn)

Scale convention matches torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor:
    scale = amax(x) / 448.0  →  x_fp8 = x / scale (then clamp+cast).
"""

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)  # -448.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


@triton.jit
def _relu2_quant_fp8_kernel(
    x_ptr,
    out_fp8_ptr,
    scale_ptr,
    n_elements,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Elementwise relu2 + FP8 quantize: out = clamp(relu(x)^2 / scale).

    Structural choices (iter 4-8 sweep):
    - relu in bf16 space (tl.maximum on input dtype) before fp32 upcast:
      bf16 SIMD is wider than fp32, saves the early upcast for negative elements.
    - tl.clamp instead of manual max(min()): compiles to a single PTX instruction.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)

    # relu in bf16 space (wider SIMD, avoids premature fp32 upcast)
    r_bf16 = tl.maximum(x, 0.0)
    # upcast to fp32 for squaring (bf16 max^2 ≈ 4e9 would overflow)
    r = r_bf16.to(tl.float32)
    relu2 = r * r

    # FP8 per-tensor quantize: divide by scale then clamp to FP8 range
    scale = tl.load(scale_ptr)
    out_scaled = relu2 / scale
    out_fp8 = tl.clamp(out_scaled, FP8_MIN, FP8_MAX).to(tl.float8e4nv)

    tl.store(out_fp8_ptr + offs, out_fp8, mask=mask)


def _run_relu2_quant_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    orig_shape = x.shape
    x_flat = x.reshape(-1)
    n = x_flat.numel()
    out_fp8 = torch.empty(orig_shape, dtype=torch.float8_e4m3fn, device=x.device)
    # BLOCK=1024, W=4: best balanced config across all shapes.
    # Re-sweep (iter 51) confirmed BLOCK=4096 regressed D4 (5.57 vs 5.31µs baseline)
    # while BLOCK=1024 beats baseline on all three decode targets (D1/D4/D16)
    # AND on P1K prefill (-5.9% vs original).
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _relu2_quant_fp8_kernel[grid](
        x_flat,
        out_fp8.reshape(-1),
        scale,
        n_elements=n,
        FP8_MIN=_FP8_MIN,
        FP8_MAX=_FP8_MAX,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out_fp8


@torch.library.custom_op("auto_deploy::relu2_quant_fp8", mutates_args=())
def relu2_quant_fp8(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused relu2 + FP8 per-tensor quantization.

    Computes ``clamp(relu(x)^2 / scale, FP8_MIN, FP8_MAX).to(float8_e4m3fn)``.

    Args:
        x: BF16/FP16 input tensor (any shape).
        scale: Scalar FP32 quantization scale (amax / 448.0 convention).

    Returns:
        FP8 (float8_e4m3fn) tensor with the same shape as ``x``.
    """
    return _run_relu2_quant_fp8(x, scale)


@relu2_quant_fp8.register_fake
def _relu2_quant_fp8_fake(
    x: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
