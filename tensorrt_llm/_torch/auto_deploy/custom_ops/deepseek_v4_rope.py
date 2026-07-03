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

"""Fused interleaved-RoPE + concat custom op for DeepSeek-V4.

``_apply_interleaved_rope`` in ``modeling_deepseek_v4.py`` does an even/odd split,
4 muls + 1 add + 1 sub, a ``stack`` and a ``flatten`` (the ``stack`` itself emits a
``CatArrayBatchedCopy``), then a ``.to(dtype)`` cast — and every call site is
immediately followed by ``torch.cat((nope, pe))`` to repack the head. In eager /
decomposed form each call therefore emits ~6 tiny ``elementwise`` kernels plus
``stack`` / ``cat`` ``copy_cast`` kernels, run up to 5x per layer per step (main
q/kv/out, the compressor, and the indexer query).

This op collapses the rotation **and** the nope/pe concat into a *single* Triton
kernel: one program per ``(position, head)`` row copies the ``nope`` slice through
and writes the interleaved-rotated ``pe`` slice contiguously next to it — no
intermediate ``stack``, no ``cat``. All rotation math stays in fp32 (matching the
reference's bf16*fp32 type promotion).

The kernel name contains ``rope`` on purpose so the collapsed work classifies
under the ``rope`` op-type and leaves the ``elementwise`` / ``copy_cast`` buckets.

The main-Q path additionally normalizes each head with a weightless RMS norm
(``q *= rsqrt(mean(q^2) + eps).to(q.dtype)``) immediately before the split/rope.
Passing ``rms_eps > 0`` folds that reduction/elementwise chain into this same
kernel: the sum-of-squares is gathered over the full ``nope || pe`` head, the
rsqrt factor is rounded to the output dtype (matching the reference
``.to(q.dtype)``), and each normalized value is materialized in the output dtype
before RoPE — bit-faithful to running the norm as a separate pre-split step.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_interleaved_rope_concat_kernel(
    nope_ptr,  # [R, Dn] (strided) — copied through
    pe_ptr,  # [R, D]  (strided) — interleaved-rotated
    cos_ptr,  # [N_pos, Dh]
    sin_ptr,  # [N_pos, Dh]
    out_ptr,  # [R, Dn + D] contiguous
    R,  # total rows = N_pos * H
    H,  # heads per position (cos/sin broadcast over heads)
    Dn,  # nope width
    D,  # rope width (even)
    Dh,  # D // 2
    nope_row_stride,
    pe_row_stride,
    cossin_row_stride,
    out_row_stride,
    rms_eps,  # per-head weightless RMS-norm epsilon (only read when HAS_NORM)
    INVERSE: tl.constexpr,
    HAS_NORM: tl.constexpr,  # fold q * rsqrt(mean(q^2)+eps) over the full head first
    BLOCK_DN: tl.constexpr,  # next_pow2(Dn)
    BLOCK_DH: tl.constexpr,  # next_pow2(Dh)
):
    row = tl.program_id(0)
    if row >= R:
        return
    pos = row // H
    out_base = out_ptr + row * out_row_stride

    # --- load the nope slice and the pe even/odd lanes ---
    dn = tl.arange(0, BLOCK_DN)
    mn = dn < Dn
    nope = tl.load(nope_ptr + row * nope_row_stride + dn, mask=mn, other=0.0)

    k = tl.arange(0, BLOCK_DH)
    mh = k < Dh
    pe_base = pe_ptr + row * pe_row_stride
    even = tl.load(pe_base + 2 * k, mask=mh, other=0.0).to(tl.float32)
    odd = tl.load(pe_base + 2 * k + 1, mask=mh, other=0.0).to(tl.float32)

    # --- optional per-head weightless RMS norm over the full (nope || pe) head ---
    # Reproduces ``q * rsqrt(mean(q^2) + eps).to(q.dtype)`` applied to the *full*
    # head BEFORE the nope/pe split: sum-of-squares is order-independent so it is
    # gathered from the nope block and the pe even/odd lanes. The rsqrt factor is
    # rounded to the output dtype (matching the reference ``.to(q.dtype)`` on the
    # rsqrt) and every normalized value is materialized in the output dtype before
    # RoPE reads it back, so the fused result is bit-faithful to the split reference.
    if HAS_NORM:
        out_ty0 = out_ptr.dtype.element_ty
        nope_f = nope.to(tl.float32)
        ss = (
            tl.sum(nope_f * nope_f, axis=0)
            + tl.sum(even * even, axis=0)
            + tl.sum(odd * odd, axis=0)
        )
        factor = tl.rsqrt(ss / (Dn + D) + rms_eps).to(out_ty0).to(tl.float32)
        nope = (nope_f * factor).to(out_ty0)
        even = (even * factor).to(out_ty0).to(tl.float32)
        odd = (odd * factor).to(out_ty0).to(tl.float32)

    # --- copy the (optionally normalized) nope slice through ---
    tl.store(out_base + dn, nope.to(out_ptr.dtype.element_ty), mask=mn)

    # --- interleaved RoPE on the pe slice (fp32 math) ---
    cos = tl.load(cos_ptr + pos * cossin_row_stride + k, mask=mh, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + pos * cossin_row_stride + k, mask=mh, other=0.0).to(tl.float32)
    if INVERSE:
        sin = -sin
    out_even = even * cos - odd * sin
    out_odd = even * sin + odd * cos

    # interleaved write: out[Dn + 2k] = out_even, out[Dn + 2k + 1] = out_odd
    pe_out_base = out_base + Dn
    out_ty = out_ptr.dtype.element_ty
    tl.store(pe_out_base + 2 * k, out_even.to(out_ty), mask=mh)
    tl.store(pe_out_base + 2 * k + 1, out_odd.to(out_ty), mask=mh)


@torch.library.custom_op("auto_deploy::deepseek_v4_fused_rope_concat", mutates_args=())
def deepseek_v4_fused_rope_concat(
    nope: torch.Tensor,
    pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
    rms_eps: float = 0.0,
) -> torch.Tensor:
    """Fused replacement for ``cat((nope, _apply_interleaved_rope(pe, cos, sin)))``.

    Args:
        nope: ``[..., Dn]`` slice that is concatenated **before** the rotated pe.
            For the main KV path this is the already-fp8-quantized nope; for every
            other site it is the raw nope slice.
        pe:   ``[..., D]`` slice to interleaved-rotate (``D`` even). Shares ``nope``'s
            leading dims. May be a (leading-contiguous) view of a larger tensor.
        cos:  ``[..M.., Dh]`` (``Dh == D // 2``). Its leading dims are ``pe``'s
            leading dims with the head dim collapsed — i.e. cos/sin broadcast over
            heads exactly like ``cos.unsqueeze(head_dim)`` did in the reference.
        sin:  same shape as ``cos``.
        inverse: if True, negate ``sin`` (the attention-output inverse rotation).
        rms_eps: if ``> 0``, additionally fold a *weightless* per-head RMS
            normalization over the full ``[..., Dn + D]`` head — i.e.
            ``q *= rsqrt(mean(q^2, dim=-1) + rms_eps).to(q.dtype)`` — that would
            otherwise run as a separate reduction/elementwise chain immediately
            before this op (the main-Q path). ``nope``/``pe`` must be the *raw*
            (un-normalized) split views. Defaults to ``0.0`` (no norm) so every
            other call site is unchanged.

    Returns:
        ``[..., Dn + D]`` contiguous tensor == ``cat((nope, rope(pe)), dim=-1)``,
        optionally with the per-head RMS norm applied first.
    """
    assert pe.shape[-1] % 2 == 0, "rope dim must be even"
    assert pe.stride(-1) == 1 and nope.stride(-1) == 1 and cos.stride(-1) == 1, (
        "last dim of nope/pe/cos must be contiguous"
    )
    D = pe.shape[-1]
    Dn = nope.shape[-1]
    Dh = D // 2

    out = torch.empty((*pe.shape[:-1], Dn + D), device=pe.device, dtype=pe.dtype)
    R = pe.numel() // D
    if R == 0:
        return out
    n_pos = cos.numel() // Dh
    # cos/sin broadcast over the head dim; rows of a position are consecutive heads.
    H = R // n_pos

    nope_row_stride = nope.stride(-2) if nope.dim() >= 2 else Dn
    pe_row_stride = pe.stride(-2) if pe.dim() >= 2 else D
    cossin_row_stride = cos.stride(-2) if cos.dim() >= 2 else Dh

    grid = (R,)
    _fused_interleaved_rope_concat_kernel[grid](
        nope,
        pe,
        cos,
        sin,
        out,
        R,
        H,
        Dn,
        D,
        Dh,
        nope_row_stride,
        pe_row_stride,
        cossin_row_stride,
        Dn + D,
        rms_eps,
        INVERSE=inverse,
        HAS_NORM=rms_eps > 0.0,
        BLOCK_DN=triton.next_power_of_2(Dn),
        BLOCK_DH=triton.next_power_of_2(Dh),
        num_warps=4,
    )
    return out


@deepseek_v4_fused_rope_concat.register_fake
def _deepseek_v4_fused_rope_concat_fake(
    nope: torch.Tensor,
    pe: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
    rms_eps: float = 0.0,
) -> torch.Tensor:
    return pe.new_empty((*pe.shape[:-1], nope.shape[-1] + pe.shape[-1]))
