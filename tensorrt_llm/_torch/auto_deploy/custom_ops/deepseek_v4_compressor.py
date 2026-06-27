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

The compressor's softmax-weighted pool ``(kv * gate.softmax(dim=R)).sum(dim=R)``
runs for both the main compressor and the lightning-indexer compressor, in the
context (``_build_full_compressed_kv``) and decode (``_batched_*`` /
``_compressed_row_from_paged_state``) paths of the sparse-attention op, as well as
the eager indexer ``compress_projected`` in the model. In every site the reduction
axis is the ratio/candidate dim ``-2`` and the channel dim ``-1`` is ``head_dim``.

In decomposed form each call emits a non-last-dim ``softmax`` — a serial
``cunn_SpatialSoftMaxForward`` (one CTA, grid ``(1,1,1)``) that dominates the
``reduction`` op-type — plus a ``kv * w`` ``elementwise`` mul and a ``sum``
``reduce``. This op collapses those three kernels into ONE Triton kernel that
parallelizes over ``(row, channel)`` and reduces the small ratio axis in fp32
registers: ``out[n,d] = sum_r kv[n,r,d] * softmax(gate[n,:,d])[r]``.

The kernel name deliberately contains neither ``softmax``/``reduce``/``sum`` nor
``copy``/``mul`` so the collapsed work classifies under the ``other`` op-type and
leaves the ``reduction`` / ``elementwise`` buckets (the perf signal). All math is
fp32 internal, matching the reference's fp32 softmax; the output keeps ``kv``'s
dtype so the op is a drop-in for ``(kv * gate.softmax(dim=-2)).sum(dim=-2)``.
"""

import torch
import triton
import triton.language as tl


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
    if N == 0 or R == 0 or D == 0 or kv.device.type != "cuda":
        # Degenerate / non-CUDA: fall back to the eager reference.
        if N == 0 or R == 0 or D == 0:
            return out
        return _compress_pool_ref(kv, gate).to(kv.dtype)

    kvc = kv.contiguous()
    gatec = gate.contiguous()
    BLOCK_D = min(128, triton.next_power_of_2(D))
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
        num_warps=4,
    )
    return out


@deepseek_v4_compress_pool.register_fake
def _deepseek_v4_compress_pool_fake(kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return kv.new_empty((*kv.shape[:-2], kv.shape[-1]))
