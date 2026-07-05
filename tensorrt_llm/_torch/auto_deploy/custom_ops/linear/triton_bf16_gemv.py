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

"""Triton bf16 GEMV custom op for M==1 decode-time projection GEMMs.

At batch=1 decode every bf16 projection is a ``[1, K] x [N, K]^T`` GEMV. cuBLAS
serves these through split-K kernels plus a separate ``splitKreduce`` launch that
round-trips a partials workspace through memory. A single-pass Triton GEMV with a
small ``BLOCK_N`` (many CTAs, fills the SMs) reads each weight row exactly once,
accumulates in fp32 (same accumulation precision as cuBLAS bf16 GEMM), and writes
the bf16 output directly — one kernel instead of two and no workspace traffic.

The custom op dispatches at runtime on the flattened token count: ``M == 1`` (the
cudagraph decode hot path; M is frozen per captured graph at capture time) runs the
Triton GEMV, everything else (prefill, multi-token decode batches) falls back to
``aten.linear`` — bit-identical to the pre-swap graph. Numerics on the M==1 path
match cuBLAS up to fp32 summation order.

The kernel configuration is resolved per weight shape from a measured table for
the shapes this op is deployed on (Step-3.7-Flash per-rank TP8 projections), with
a generic occupancy-driven heuristic as fallback. No runtime autotuning: config
resolution is pure Python on shapes, so it is deterministic and safe under CUDA
graph capture.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Minimum granularity of the K-loop; the production contract requires K % 128 == 0
# so a power-of-two BLOCK_K always exists (tl.arange requires powers of two).
_MIN_BLOCK_K = 128


@triton.jit
def _bf16_gemv_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``y[n] = dot(x, w[n, :])``: bf16 loads, fp32 accumulate, bf16 store.

    One program per ``BLOCK_N`` output rows; rows are masked so N need not divide
    by BLOCK_N. K must divide by BLOCK_K (mask-free K loop).
    """
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_n[:, None], other=0.0
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16), mask=mask_n)


# Best measured (BLOCK_N, BLOCK_K, num_warps, num_stages) per weight shape (N, K),
# from a CUDA-graph replay sweep on H100 over the Step-3.7-Flash per-rank TP8
# decode shapes with one distinct weight per call (L2-busted, HBM-streamed like
# the real 45-layer decode step). Measured speedup vs cuBLAS in parentheses.
_KERNEL_CONFIG_TABLE = {
    (1288, 4096): (1, 4096, 8, 3),  # fused qkvg, full attn (1.50x)
    (1804, 4096): (1, 4096, 8, 3),  # fused qkvg, sliding attn (1.39x)
    (4096, 1024): (2, 1024, 4, 3),  # o_proj, full attn (1.21x)
    (4096, 1536): (2, 512, 4, 3),  # o_proj, sliding attn (1.14x)
    (2816, 4096): (1, 4096, 8, 3),  # dense MLP fused gate_up (1.35x)
    (4096, 1408): (8, 128, 8, 3),  # dense MLP down (1.14x)
}


def _largest_pow2_divisor_block_k(k: int) -> int:
    block_k = _MIN_BLOCK_K
    while k % (block_k * 2) == 0 and block_k * 2 <= 4096:
        block_k *= 2
    return block_k


def _pick_config(n: int, k: int) -> Tuple[int, int, int, int]:
    """Resolve (BLOCK_N, BLOCK_K, num_warps, num_stages) for a weight shape."""
    cfg = _KERNEL_CONFIG_TABLE.get((n, k))
    if cfg is not None:
        return cfg
    # Fallback heuristic following the measured trend: (near-)one row per CTA
    # maximizes CTA count for latency hiding, one K-slab per loop iteration up
    # to 4096 keeps loads wide; more warps help the wide slabs.
    block_k = _largest_pow2_divisor_block_k(k)
    return 1, block_k, 8 if block_k >= 2048 else 4, 3


def _production_contract(x2d: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the M==1 decode Triton GEMV can handle this call."""
    return (
        x2d.is_cuda
        and weight.is_cuda
        and x2d.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and x2d.shape[0] == 1
        and x2d.shape[1] == weight.shape[1]
        and weight.shape[1] % _MIN_BLOCK_K == 0
        and weight.is_contiguous()
    )


def _run_bf16_gemv(x2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, k = weight.shape
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x2d.device)
    block_n, block_k, num_warps, num_stages = _pick_config(n, k)
    grid = (triton.cdiv(n, block_n),)
    _bf16_gemv_kernel[grid](
        x2d,
        weight,
        out,
        n,
        K=k,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


@torch.library.custom_op("auto_deploy::triton_bf16_gemv_linear", mutates_args=())
def triton_bf16_gemv_linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``input @ weight^T`` with a single-pass Triton GEMV on the M==1 decode path.

    Args:
        input: ``(..., in_features)`` activations (bf16 on the production path).
        weight: ``(out_features, in_features)`` bf16 weight; bias is not supported
            (the swap transform only rewrites bias-free linears).

    Any call outside the production contract (M > 1 prefill/multi-token batches,
    non-bf16 dtypes, CPU tensors) falls back to ``aten.linear``, bit-identical to
    the pre-swap graph.
    """
    x2d = input.reshape(-1, input.shape[-1])
    if _production_contract(x2d, weight):
        out = _run_bf16_gemv(x2d.contiguous(), weight)
        return out.reshape(*input.shape[:-1], weight.shape[0])
    return torch.ops.aten.linear.default(input, weight, None)


@triton_bf16_gemv_linear.register_fake
def _triton_bf16_gemv_linear_fake(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return input.new_empty((*input.shape[:-1], weight.shape[0]))
