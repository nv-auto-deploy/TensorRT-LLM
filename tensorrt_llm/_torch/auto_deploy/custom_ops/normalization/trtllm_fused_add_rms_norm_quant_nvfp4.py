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

"""RMSNorm + NVFP4 quantization wrappers.

Wraps the C++ kernel ``trtllm.fused_add_rms_norm_quant`` (NVFP4 mode) with
a thin custom op that:

* Always produce a high-precision (BF16/FP16) normed output so that
  non-quantised consumers can be rewired without an extra norm.
* Convert the int32-packed FP4 output to uint8 (``view(torch.uint8)``).
* Flatten arbitrary leading dims to 2-D before calling the C++ kernel
  (which requires ``[M, N]``) and reshape outputs back.

The provided variants are:

``trtllm_rms_norm_quant_nvfp4``
    Standalone RMSNorm + quant for graphs without a residual add.

``trtllm_fused_add_rms_norm_quant_nvfp4``
    Fused add + norm + quant — the common case after ``fuse_add_rms_norm``
    has already run.
"""

from typing import Tuple

import torch

from tensorrt_llm.quantization.utils import fp4_utils

TRTLLM_NVFP4_SCALING_VECTOR_SIZE = 16


@torch.library.custom_op("auto_deploy::trtllm_rms_norm_quant_nvfp4", mutates_args=())
def trtllm_rms_norm_quant_nvfp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standalone RMSNorm + NVFP4 quantization."""
    assert weight.numel() % 2 == 0, "NVFP4 packing requires even hidden size"

    orig_shape = x.shape
    n = weight.shape[0]
    x_2d = x.reshape(-1, n)
    weight_cast = weight.to(dtype=x.dtype)

    hp_normed = torch.ops.auto_deploy.triton_rms_norm(x_2d, weight_cast, eps)
    fp4_u8, sf_out = torch.ops.trtllm.fp4_quantize(
        hp_normed, sf_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE
    )
    fp4_u8 = fp4_u8.reshape(*orig_shape[:-1], -1)
    return hp_normed.reshape(orig_shape), fp4_u8, sf_out


@trtllm_rms_norm_quant_nvfp4.register_fake
def _trtllm_rms_norm_quant_nvfp4_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del eps, sf_scale
    assert weight.numel() % 2 == 0, "NVFP4 packing requires even hidden size"

    n = weight.shape[0]
    m = x.numel() // n
    hp_normed = torch.empty_like(x)
    _, scale_shape = fp4_utils.get_fp4_shape((m, n), sf_vec_size=16, is_swizzled_layout=True)
    fp4_u8 = x.new_empty((*x.shape[:-1], n // 2), dtype=torch.uint8)
    sf_out = x.new_empty((scale_shape,), dtype=torch.uint8)
    return hp_normed, fp4_u8, sf_out


# ---------------------------------------------------------------------------
# Fused add + norm + quant
# ---------------------------------------------------------------------------
@torch.library.custom_op("auto_deploy::trtllm_fused_add_rms_norm_quant_nvfp4", mutates_args=())
def trtllm_fused_add_rms_norm_quant_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Add + RMSNorm + NVFP4 quantization.

    Computes ``add_out = x + residual``, ``norm_out = rms_norm(add_out)``,
    then NVFP4 block-quantises ``norm_out``.

    Args:
        x: [..., N] first input (bf16/fp16).
        residual: [..., N] second input (residual).
        weight: [N] RMSNorm gamma.
        eps: Layernorm epsilon.
        sf_scale: Global FP4 scale (scalar, float32).

    Returns:
        (bf16_normed, fp4_u8, sf_out, add_out)
    """
    assert weight.numel() % 2 == 0, "NVFP4 packing requires even hidden size"

    orig_shape = x.shape
    n = weight.shape[0]
    x_2d = x.reshape(-1, n)
    residual_2d = residual.reshape(-1, n)

    weight_cast = weight.to(dtype=x.dtype)

    normed_i32, add_out_2d, sf_out, hp_normed = torch.ops.trtllm.fused_add_rms_norm_quant(
        x_2d, residual_2d, weight_cast, sf_scale, True, eps=eps, output_hp_norm=True
    )
    fp4_u8 = normed_i32.view(torch.uint8)
    fp4_u8 = fp4_u8.reshape(*orig_shape[:-1], -1)
    return hp_normed.reshape(orig_shape), fp4_u8, sf_out, add_out_2d.reshape(orig_shape)


@trtllm_fused_add_rms_norm_quant_nvfp4.register_fake
def _fused_add_rms_norm_quant_nvfp4_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    sf_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert weight.numel() % 2 == 0, "NVFP4 packing requires even hidden size"

    n = weight.shape[0]
    m = x.numel() // n
    bf16_out = torch.empty_like(x)
    _, scale_shape = fp4_utils.get_fp4_shape((m, n), sf_vec_size=16, is_swizzled_layout=True)
    fp4_u8 = x.new_empty((*x.shape[:-1], n // 2), dtype=torch.uint8)
    sf_out = x.new_empty((scale_shape,), dtype=torch.uint8)
    add_out = torch.empty_like(x)
    return bf16_out, fp4_u8, sf_out, add_out
