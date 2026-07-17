# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Fused RMSNorm for the DeepSeek-V4 Q-LoRA projection.

Before: ``torch_rmsnorm(Q-LoRA)`` decomposes into separate reduction,
normalization, weight multiplication, and cast operations.
After: ``_deepseek_v4_q_rmsnorm_kernel`` performs the complete RMSNorm in one
Triton kernel while preserving the FP32 reduction and BF16 rounding points.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _deepseek_v4_q_rmsnorm_kernel(
    x_ptr,  # [R, N] bf16 input (strided rows)
    w_ptr,  # [N] RMS-norm weight (native dtype -> fp32)
    out_ptr,  # [R, N] contiguous, model dtype
    N,
    x_row_stride,
    out_row_stride,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    out_ty = out_ptr.dtype.element_ty
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # torch_rmsnorm reference math: fp32 mean(x^2), unrounded fp32 rsqrt factor,
    # then round(x * factor) -> * fp32 weight -> round — the exact two rounding
    # points of ``out.copy_(weight * (input * rsqrt(var + eps)).to(out.dtype))``.
    var = tl.sum(x * x) / N
    factor = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (w * (x * factor).to(out_ty).to(tl.float32)).to(out_ty)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


@torch.library.custom_op("auto_deploy::deepseek_v4_q_rmsnorm", mutates_args=())
def deepseek_v4_q_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """One-kernel BF16 RMS norm for the DeepSeek-V4 Q-LoRA projection.

    This op keeps the projection-to-consumer contract BF16 and reproduces
    ``torch_rmsnorm``'s fp32 reduction and BF16 rounding points. It is selected only
    for the 1024-wide Q child of the DeepSeek-V4 fused Q/KV projection.

    Args:
        input: ``[..., 1024]`` BF16 Q-LoRA projection output. A
            last-dimension-contiguous narrow view is supported.
        weight: ``[1024]`` BF16 or FP32 RMS-norm weight, applied in FP32.
        eps: RMS-norm epsilon.

    Returns:
        A contiguous BF16 tensor with the same shape as ``input``.
    """
    q_lora_width = 1024
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be bfloat16, got {input.dtype}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"weight must be bfloat16 or float32, got {weight.dtype}")
    if input.device != weight.device:
        raise ValueError(
            f"input and weight must be on the same device, got {input.device} and {weight.device}"
        )
    if input.dim() == 0 or input.shape[-1] != q_lora_width:
        raise ValueError(f"input must have shape [..., {q_lora_width}], got {tuple(input.shape)}")
    if input.stride(-1) != 1:
        raise ValueError("input last dimension must be contiguous")
    if input.dim() >= 2 and input.shape[-2] > 1 and input.stride(-2) < q_lora_width:
        raise ValueError("input rows must not overlap")
    if input.dim() > 2:
        expected_stride = input.shape[-2] * input.stride(-2)
        for dim in range(input.dim() - 3, -1, -1):
            if input.shape[dim] > 1 and input.stride(dim) != expected_stride:
                raise ValueError("input leading dimensions must flatten to regularly strided rows")
            expected_stride *= input.shape[dim]
    if weight.dim() != 1 or weight.numel() != q_lora_width or weight.stride(0) != 1:
        raise ValueError(f"weight must be contiguous with shape [{q_lora_width}]")
    N = q_lora_width

    out = torch.empty(input.shape, device=input.device, dtype=torch.bfloat16)
    R = input.numel() // N
    if R == 0:
        return out
    x_row_stride = input.stride(-2) if input.dim() >= 2 else N

    _deepseek_v4_q_rmsnorm_kernel[(R,)](
        input,
        weight,
        out,
        N,
        x_row_stride,
        N,
        eps,
        BLOCK_N=triton.next_power_of_2(N),
        num_warps=4,
    )
    return out


@deepseek_v4_q_rmsnorm.register_fake
def _deepseek_v4_q_rmsnorm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return input.new_empty(input.shape, dtype=torch.bfloat16)
