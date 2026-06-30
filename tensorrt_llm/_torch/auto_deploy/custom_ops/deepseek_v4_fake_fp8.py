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

"""Fused fake-FP8 block activation-quant custom op for DeepSeek-V4.

``fake_fp8_act_quant`` in ``utils/quantization_utils.py`` simulates block-wise FP8
activation quantization on a bf16 tensor: per group of ``block_size`` elements along
the last dim it derives a power-of-two scale from the block amax, divides, clamps to
the e4m3 ``[-448, 448]`` range, round-trips through the input dtype (bf16) to emulate
the FP8 mantissa, then multiplies the scale back. In eager / exported form this
decomposes into a long chain of tiny kernels per call -- ``abs``, an ``amax``
reduction, ``clamp_min``, ``div``, ``log2``, ``ceil``, ``pow``, ``div``, ``clamp``,
two bf16 round-trip casts and a ``mul`` -- and the helper is called up to 3x per layer
per step (main q-compressor nope, main kv nope, and the sparse-indexer compressed
nope). The trace shows ~180 such chains; the ``max``/``pow``/``log2``/``ceil`` kernels
alone exceed 1.2 ms before the clamp/cast/div/mul work.

This op collapses the whole chain into a *single* Triton kernel: one program per
``block_size``-element group loads the block, reduces the amax, derives the power-of-
two scale, and writes the dequantized result. All intermediate math stays in fp32 and
the two bf16 round-trips are reproduced bit-for-bit, so the output is byte-identical to
the reference (see ``test_deepseek_v4_fake_fp8.py``). The full-tensor fp32 ``.float()``
materialization the reference performs is avoided entirely (bf16 in, bf16 out).
"""

import torch
import triton
import triton.language as tl

__all__ = ["deepseek_v4_fake_fp8_act_quant"]


@triton.jit
def _fake_fp8_act_quant_kernel(
    x_ptr,  # input, last-dim contiguous, leading dims densely nested with row_stride
    out_ptr,  # output, contiguous (packed)
    n_groups_per_row,  # dim // block_size
    row_stride,  # element stride between consecutive rows of the input
    BLOCK_SIZE: tl.constexpr,  # block_size (real group width)
    BLOCK_POW2: tl.constexpr,  # next_power_of_2(block_size) -- arange width
    MAX_VAL: tl.constexpr,  # 448.0 (e4m3 absmax)
    MIN_VAL: tl.constexpr,  # 1e-4 (amax floor)
):
    gid = tl.program_id(0)
    row = gid // n_groups_per_row
    grp = gid % n_groups_per_row
    in_base = row * row_stride + grp * BLOCK_SIZE
    out_base = gid * BLOCK_SIZE  # output is packed: group gid lives at gid * block_size

    offs = tl.arange(0, BLOCK_POW2)
    mask = offs < BLOCK_SIZE
    # Padding lanes load 0.0; abs(0)=0 never wins the amax, and the masked store drops
    # them -- so masking is a no-op for the pow2 block_size=64 case used in practice.
    x = tl.load(x_ptr + in_base + offs, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x), axis=0)
    # scale = 2 ** ceil(log2(clamp_min(amax, MIN_VAL) / MAX_VAL)). bf16-precision amax
    # keeps the log2 argument far from power-of-two boundaries, so ceil() lands on the
    # same integer as torch and exp2(int) == torch.pow(2.0, int) exactly.
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, MIN_VAL) / MAX_VAL)))

    out_ty = out_ptr.dtype.element_ty
    q = x / scale
    q = tl.minimum(tl.maximum(q, -MAX_VAL), MAX_VAL)
    # Round-trip through the storage dtype then back to fp32 -- mirrors `.to(dtype).float()`.
    q = q.to(out_ty).to(tl.float32)
    out = q * scale
    tl.store(out_ptr + out_base + offs, out.to(out_ty), mask=mask)


def _is_row_pitch_indexable(x: torch.Tensor) -> bool:
    """True if every ``block_size`` group can be addressed via a single row pitch.

    The kernel flattens all leading dims and walks rows with pitch ``stride(-2)``. That
    is valid iff the last dim is unit-stride and the leading dims are densely nested with
    that pitch -- which holds for contiguous tensors and for last-dim slices of
    contiguous tensors (every DeepSeek-V4 call site). Anything else falls back to a
    packed copy in the wrapper.
    """
    if x.stride(-1) != 1:
        return False
    for j in range(x.dim() - 2):
        if x.stride(j) != x.size(j + 1) * x.stride(j + 1):
            return False
    return True


@torch.library.custom_op("auto_deploy::deepseek_v4_fake_fp8_act_quant", mutates_args=())
def deepseek_v4_fake_fp8_act_quant(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Fused, byte-exact replacement for ``utils.quantization_utils.fake_fp8_act_quant``.

    Per ``block_size``-element group along the last dim: dequantized fake-FP8 round trip
    ``(clamp(x/scale, -448, 448) -> bf16 -> fp32) * scale -> bf16`` with
    ``scale = 2**ceil(log2(clamp_min(amax, 1e-4) / 448))``. Returns a contiguous tensor of
    the same shape/dtype as ``x``.
    """
    dim = x.shape[-1]
    if dim == 0 or dim % block_size != 0:
        return x.clone()

    if not _is_row_pitch_indexable(x):
        x = x.contiguous()

    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    num_groups = x.numel() // block_size
    if num_groups == 0:
        return out

    n_groups_per_row = dim // block_size
    row_stride = x.stride(-2) if x.dim() >= 2 else dim
    _fake_fp8_act_quant_kernel[(num_groups,)](
        x,
        out,
        n_groups_per_row,
        row_stride,
        BLOCK_SIZE=block_size,
        BLOCK_POW2=triton.next_power_of_2(block_size),
        MAX_VAL=448.0,
        MIN_VAL=1.0e-4,
        num_warps=1,
    )
    return out


@deepseek_v4_fake_fp8_act_quant.register_fake
def _deepseek_v4_fake_fp8_act_quant_fake(x: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    return torch.empty_like(x, memory_format=torch.contiguous_format)
