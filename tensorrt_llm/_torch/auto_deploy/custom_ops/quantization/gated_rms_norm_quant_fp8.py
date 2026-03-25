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

"""Fused gated-RMSNorm + FP8 per-tensor quantization Triton kernel.

For Mamba2 decode path, this fuses:
    y_norm = triton_rmsnorm_gated(x, weight, gate, eps, group_size, norm_before_gate)
    # (then scaleMatrixPerTensorVec inside trtllm_quant_fp8_linear)

into a single kernel:
    y_fp8 = gated_rms_norm_quant_fp8(x, gate, weight, in_scale, eps, group_size, norm_before_gate)

Eliminates one kernel launch per Mamba layer (the scaleMatrixPerTensorVec call), plus
avoids writing/reading the intermediate BF16 tensor between normalization and quantization.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_FP8_MIN = float(torch.finfo(torch.float8_e4m3fn).min)  # -448.0
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)  # 448.0


@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
@triton.jit
def _gated_rms_norm_quant_fp8_kernel(
    X,  # pointer to input [M, ngroups*N]
    Y_fp8,  # pointer to FP8 output [M, ngroups*N]
    W,  # pointer to norm weight [ngroups*N]
    Z,  # pointer to gate [M, ngroups*N]  (None → HAS_Z=False)
    Scale,  # pointer to FP8 per-tensor scale (scalar float32)
    Rstd,  # pointer to rstd scratch [ngroups * M]
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N,  # group_size
    eps,
    BLOCK_N: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One CTA processes one (row, group) pair.

    Grid: (M, ngroups)  where ngroups = total_hidden // group_size
    """
    row = tl.program_id(0)
    group = tl.program_id(1)

    X += row * stride_x_row + group * N
    Y_fp8 += row * stride_y_row + group * N
    if HAS_Z:
        # Cast to int64 to avoid overflow (same caution as original layernorm_gated.py)
        Z += tl.cast(row, tl.int64) * stride_z_row + group * N
    Rstd += group * M
    W += group * N

    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)

    # SiLU gating BEFORE normalisation  (norm_before_gate=False, Nemotron default)
    if HAS_Z and not NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        x = x * z * tl.sigmoid(z)

    # RMS normalisation (no mean subtraction = RMSNorm)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)

    w = tl.load(W + cols, mask=cols < N).to(tl.float32)
    y = x * rstd * w

    # SiLU gating AFTER normalisation  (norm_before_gate=True)
    if HAS_Z and NORM_BEFORE_GATE:
        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
        y = y * z * tl.sigmoid(z)

    # FP8 per-tensor quantize: y_fp8 = clamp(y / scale, FP8_MIN, FP8_MAX)
    scale = tl.load(Scale)
    y_scaled = y / scale
    y_clamped = tl.minimum(tl.maximum(y_scaled, FP8_MIN), FP8_MAX)
    y_fp8 = y_clamped.to(tl.float8e4nv)

    tl.store(Y_fp8 + cols, y_fp8, mask=cols < N)


def _run_gated_rms_norm_quant_fp8(
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: torch.Tensor,
    in_scale: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    """Launch the fused gated-RMSNorm + FP8-quantize kernel.

    Args:
        x: Input BF16/FP16 tensor [M, H] (flattened to 2-D inside this function).
        gate: Gate tensor same shape as x, or None.
        weight: RMSNorm weight [H].
        in_scale: Per-tensor FP8 quantization scale (scalar float32 tensor).
        eps: RMSNorm epsilon.
        group_size: Group size for grouped RMSNorm (H % group_size == 0).
        norm_before_gate: If True, apply gate after norm; if False, apply gate before norm.

    Returns:
        FP8 (float8_e4m3fn) tensor of shape [M, H].
    """
    orig_shape = x.shape
    H = weight.numel()
    assert orig_shape[-1] == H
    assert H % group_size == 0
    ngroups = H // group_size

    x2 = x.reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]

    z2 = None
    if gate is not None:
        z2 = gate.reshape(-1, H)
        if z2.stride(-1) != 1:
            z2 = z2.contiguous()

    assert weight.is_contiguous()
    assert in_scale.is_contiguous()

    out_fp8 = torch.empty(orig_shape, dtype=torch.float8_e4m3fn, device=x.device)
    out2 = out_fp8.reshape(-1, H)

    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
    if group_size > BLOCK_N:
        raise RuntimeError("gated_rms_norm_quant_fp8: group_size exceeds 64KB limit.")
    num_warps = min(max(BLOCK_N // 256, 1), 8)

    grid = (M, ngroups)
    with torch.cuda.device(x.device.index):
        _gated_rms_norm_quant_fp8_kernel[grid](
            x2,
            out2,
            weight,
            z2,
            in_scale,
            rstd,
            x2.stride(0),
            out2.stride(0),
            z2.stride(0) if z2 is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            NORM_BEFORE_GATE=norm_before_gate,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            num_warps=num_warps,
        )
    return out_fp8


@torch.library.custom_op("auto_deploy::gated_rms_norm_quant_fp8", mutates_args=())
def gated_rms_norm_quant_fp8(
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: torch.Tensor,
    in_scale: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    """Fused gated-RMSNorm + FP8 per-tensor quantization.

    Equivalent to:
        y = triton_rmsnorm_gated(x, weight, gate, eps, group_size, norm_before_gate)
        fp8_y = clamp(y / in_scale, FP8_MIN, FP8_MAX).to(float8_e4m3fn)

    Args:
        x: Input tensor [*, H] in BF16/FP16.
        gate: Gate tensor same shape as x (or None for ungated).
        weight: RMSNorm scale weights [H].
        in_scale: Per-tensor FP8 quantization scale (scalar float32).
        eps: RMSNorm epsilon.
        group_size: Normalization group size; H must be divisible by group_size.
        norm_before_gate: If True, gate is applied after norm; if False, before.

    Returns:
        FP8 (float8_e4m3fn) tensor of the same shape as x.
    """
    return _run_gated_rms_norm_quant_fp8(
        x, gate, weight, in_scale, eps, group_size, norm_before_gate
    )


@gated_rms_norm_quant_fp8.register_fake
def _gated_rms_norm_quant_fp8_fake(
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    weight: torch.Tensor,
    in_scale: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
