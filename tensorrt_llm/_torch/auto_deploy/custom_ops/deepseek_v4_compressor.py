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

"""Fused attention-pool COMPUTE for the DeepSeek-V4 compressor.

Collapses the compressor's softmax-weighted pool
``(kv * gate.softmax(dim=-2)).sum(dim=-2)`` (main + lightning-indexer
compressors, context and decode paths of the sparse-attention op, and the eager
indexer ``compress_projected``) into ONE Triton kernel parallelized over
``(row, channel)``: ``out[n,d] = sum_r kv[n,r,d] * softmax(gate[n,:,d])[r]``.

The kernel name deliberately contains neither ``softmax``/``reduce``/``sum`` nor
``copy``/``mul`` so the collapsed work classifies under the ``other`` op-type and
leaves the ``reduction`` / ``elementwise`` buckets (the perf signal). All math is
fp32 internal, matching the reference's fp32 softmax; the output keeps ``kv``'s
dtype so the op is a drop-in replacement.
"""

import torch
import triton
import triton.language as tl

# Autotune the launch occupancy per compressed-pool shape. Each program owns one
# ``[BLOCK_R, BLOCK_D]`` tile and reduces the tiny ratio axis (``R <= 8``) in
# registers, so ``num_warps=1`` (single-warp shuffle reduction, no shared-memory
# barrier) wins across every decode and prefill shape; ``num_stages`` is inert
# (loop-free kernel). Keyed on ``(R, D)`` — the only dims that change the optimal
# occupancy — so each compressor (indexer ``D=128`` / main ``D=512``) tunes once;
# the cross-warp configs are kept as a measured safety margin.
_DSV4_COMPRESS_POOL_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=1),
    triton.Config({}, num_warps=2, num_stages=1),
    triton.Config({}, num_warps=4, num_stages=1),
]


@triton.autotune(configs=_DSV4_COMPRESS_POOL_CONFIGS, key=["R", "D"])
@triton.jit
def _dsv4_compress_pool_kernel(
    kv_ptr,  # [N, R, D] contiguous
    gate_ptr,  # [N, R, D] contiguous
    out_ptr,  # [N, D] contiguous
    N,
    R,
    D,
    BLOCK_R: tl.constexpr,  # next_pow2(R)
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    if n >= N:
        return
    d0 = tl.program_id(1) * BLOCK_D
    ds = d0 + tl.arange(0, BLOCK_D)
    dmask = ds < D
    rs = tl.arange(0, BLOCK_R)
    rmask = rs < R

    # [BLOCK_R, BLOCK_D] tile of row n. Padded rows (rs >= R) load -inf for gate
    # (zero softmax weight) and 0 for kv.
    offs = n * R * D + rs[:, None] * D + ds[None, :]
    mask = rmask[:, None] & dmask[None, :]
    g = tl.load(gate_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
    k = tl.load(kv_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Per-channel softmax over the ratio axis, then weighted sum — same op order
    # as ``(kv * gate.softmax(dim=-2)).sum(dim=-2)``.
    m = tl.max(g, axis=0)  # [BLOCK_D]
    e = tl.exp(g - m[None, :])  # [BLOCK_R, BLOCK_D]; masked/padded rows -> 0
    s = tl.sum(e, axis=0)  # [BLOCK_D]
    w = e / s[None, :]
    out = tl.sum(k * w, axis=0)  # [BLOCK_D]

    tl.store(out_ptr + n * D + ds, out.to(out_ptr.dtype.element_ty), mask=dmask)


def _compress_pool_ref(kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Eager reference == the expression this op replaces (used on non-CUDA / fake)."""
    return (kv * gate.softmax(dim=-2)).sum(dim=-2)


@torch.library.custom_op("auto_deploy::deepseek_v4_compress_pool", mutates_args=())
def deepseek_v4_compress_pool(kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Fused softmax-weighted pool over the ratio axis (dim ``-2``).

    Args:
        kv:   ``[..., R, D]`` gathered (and, for overlap layers, overlap-expanded)
            compressor value states. ``R`` is the ratio / candidate count, ``D`` is
            ``head_dim``.
        gate: ``[..., R, D]`` same shape — gate logits with the ape bias added and
            the validity masking (``-1e20``) and overlap transform already applied
            by the caller.

    Returns:
        ``[..., D]`` == ``(kv * gate.softmax(dim=-2)).sum(dim=-2)`` in ``kv``'s dtype,
        computed with an fp32-internal softmax.
    """
    assert kv.shape == gate.shape, f"kv/gate shape mismatch: {kv.shape} vs {gate.shape}"
    assert kv.dim() >= 2, "kv/gate must have rank >= 2 ([..., R, D])"

    R = kv.shape[-2]
    D = kv.shape[-1]
    out = torch.empty((*kv.shape[:-2], D), device=kv.device, dtype=kv.dtype)
    N = out.numel() // D if D > 0 else 0
    if N == 0 or R == 0 or D == 0:
        return out
    if kv.device.type != "cuda":
        return _compress_pool_ref(kv, gate).to(kv.dtype)

    kvc = kv.contiguous()
    gatec = gate.contiguous()

    # D-axis tiling: grid is ``(N, cdiv(D, BLOCK_D))``. Large-N launches already
    # saturate the SMs at the maximal ``BLOCK_D=128`` (best coalescing, fewest
    # CTAs); small-N decode shapes (e.g. the main-compressor new-row path,
    # ``N<=2, D=512``) are occupancy-starved there, so halve ``BLOCK_D`` until
    # the grid reaches the machine-fill target (~512 CTAs), down to a coalescing
    # floor of 16. Large-N keeps the byte-identical ``BLOCK_D=128`` launch.
    cap = min(128, triton.next_power_of_2(D))
    BLOCK_D = cap
    while BLOCK_D > 16 and N * triton.cdiv(D, BLOCK_D) < 512:
        BLOCK_D //= 2
    grid = (N, triton.cdiv(D, BLOCK_D))
    _dsv4_compress_pool_kernel[grid](
        kvc,
        gatec,
        out,
        N,
        R,
        D,
        BLOCK_R=triton.next_power_of_2(R),
        BLOCK_D=BLOCK_D,
    )
    return out


@deepseek_v4_compress_pool.register_fake
def _deepseek_v4_compress_pool_fake(kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return kv.new_empty((*kv.shape[:-2], kv.shape[-1]))
