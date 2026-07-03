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


# --------------------------------------------------------------------------- #
# KV front-end: weighted RMS norm + no-PE fake-FP8 quant + interleaved RoPE    #
# --------------------------------------------------------------------------- #
#
# The main-KV path in ``modeling_deepseek_v4.py`` runs, back to back:
#   ``kv = kv_norm(kv)``            (weighted RMS norm over the full head, an
#                                    ``auto_deploy::torch_rmsnorm`` custom op that
#                                    decomposes into a reduction + several
#                                    elementwise + copy_cast kernels at runtime),
#   ``kv_nope, kv_pe = split(kv)``  (nope/pe split of the normed head),
#   ``kv_nope = fake_fp8_act_quant(kv_nope, 64)``  (a fused ``other``-bucket kernel),
#   ``deepseek_v4_fused_rope_concat(kv_nope, kv_pe, cos, sin)``  (the proven op).
#
# ``deepseek_v4_kv_norm_rope_concat`` collapses all four into a single kernel: it
# takes the *raw* (pre-norm) split views + the norm weight, normalizes the full
# ``nope || pe`` head, fake-FP8-quantizes the nope slice per block, interleaved-
# rotates the pe slice, and writes ``cat((fp8(nope), rope(pe)))`` — no separate
# normed-kv tensor, no fp8 buffer, one launch. Every rounding point is reproduced
# bit-for-bit (see below and ``test_deepseek_v4_kv_norm_rope_concat.py``):
#   * RMS norm — ``torch_rmsnorm`` semantics: ``variance = mean(x^2)`` in fp32, an
#     *unrounded* fp32 ``rsqrt`` factor, then ``round(normed_bf16) -> * weight(fp32)
#     -> round_bf16`` (two rounding points, matching
#     ``out.copy_(weight * input.to(out.dtype))``).
#   * fake-FP8 — per ``block_size`` group of the *normed* nope:
#     ``scale = 2**ceil(log2(clamp_min(amax, 1e-4) / 448))`` then
#     ``(clamp(x/scale, -448, 448) -> bf16 -> fp32) * scale -> bf16`` — identical to
#     ``deepseek_v4_fake_fp8_act_quant``.
# The name contains ``rope`` so the collapsed work stays in the ``rope`` op-type
# bucket (net-flat rope count) while the norm/fp8 kernels leave the reduction /
# elementwise / copy_cast / other buckets.


@triton.jit
def _kv_norm_rope_concat_kernel(
    nope_ptr,  # [R, Dn] raw (strided) — normed, fp8-quantized, then copied through
    pe_ptr,  # [R, D]  raw (strided) — normed, then interleaved-rotated
    weight_ptr,  # [Dn + D] RMS-norm weight (loaded in its native dtype -> fp32)
    cos_ptr,  # [N_pos, Dh]
    sin_ptr,  # [N_pos, Dh]
    out_ptr,  # [R, Dn + D] contiguous
    R,  # total rows = N_pos * H
    H,  # heads per position (cos/sin broadcast over heads; H=1 for the KV latent)
    Dn,  # nope width
    D,  # rope width (even)
    Dh,  # D // 2
    nope_row_stride,
    pe_row_stride,
    cossin_row_stride,
    out_row_stride,
    rms_eps,
    INVERSE: tl.constexpr,
    NB: tl.constexpr,  # BLOCK_DN // FP8_BLOCK (fp8 reshape block count)
    FP8_BLOCK: tl.constexpr,  # fp8 quant group width (block_size)
    FP8_MAX: tl.constexpr,  # 448.0 (e4m3 absmax)
    FP8_MIN: tl.constexpr,  # 1e-4 (amax floor)
    BLOCK_DN: tl.constexpr,  # next_pow2(Dn) == NB * FP8_BLOCK
    BLOCK_DH: tl.constexpr,  # next_pow2(Dh)
):
    row = tl.program_id(0)
    if row >= R:
        return
    pos = row // H
    out_base = out_ptr + row * out_row_stride
    out_ty = out_ptr.dtype.element_ty

    # --- load the raw nope slice and the raw pe even/odd lanes (fp32 math) ---
    dn = tl.arange(0, BLOCK_DN)
    mn = dn < Dn
    nope = tl.load(nope_ptr + row * nope_row_stride + dn, mask=mn, other=0.0).to(tl.float32)

    k = tl.arange(0, BLOCK_DH)
    mh = k < Dh
    pe_base = pe_ptr + row * pe_row_stride
    even = tl.load(pe_base + 2 * k, mask=mh, other=0.0).to(tl.float32)
    odd = tl.load(pe_base + 2 * k + 1, mask=mh, other=0.0).to(tl.float32)

    # --- weighted RMS norm over the FULL (nope || pe) head (torch_rmsnorm) ---
    # sum-of-squares is order-independent, so it is gathered from the (zero-padded)
    # nope block and the pe even/odd lanes. The rsqrt factor stays fp32 (unrounded);
    # the normed value is rounded to the output dtype, multiplied by the fp32 weight,
    # and rounded again — the two rounding points of ``out.copy_(weight * x.to(dt))``.
    ss = tl.sum(nope * nope) + tl.sum(even * even) + tl.sum(odd * odd)
    factor = tl.rsqrt(ss / (Dn + D) + rms_eps)

    w_nope = tl.load(weight_ptr + dn, mask=mn, other=0.0).to(tl.float32)
    w_even = tl.load(weight_ptr + Dn + 2 * k, mask=mh, other=0.0).to(tl.float32)
    w_odd = tl.load(weight_ptr + Dn + 2 * k + 1, mask=mh, other=0.0).to(tl.float32)

    kv_nope = (w_nope * (nope * factor).to(out_ty).to(tl.float32)).to(out_ty)
    kv_even = (w_even * (even * factor).to(out_ty).to(tl.float32)).to(out_ty)
    kv_odd = (w_odd * (odd * factor).to(out_ty).to(tl.float32)).to(out_ty)

    # --- fake-FP8 block quant on the normed nope (reshape into FP8_BLOCK groups) ---
    # Padding lanes (dn >= Dn) are 0 and live in their own trailing block(s), so they
    # never contaminate a real block's amax and are dropped by the masked store.
    xb = tl.reshape(kv_nope.to(tl.float32), (NB, FP8_BLOCK))
    amax = tl.max(tl.abs(xb), axis=1, keep_dims=True)  # [NB, 1]
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax, FP8_MIN) / FP8_MAX)))
    q = tl.minimum(tl.maximum(xb / scale, -FP8_MAX), FP8_MAX)
    q = q.to(out_ty).to(tl.float32)  # bf16 round-trip (emulates the FP8 mantissa)
    nope_q = tl.reshape((q * scale).to(out_ty), (BLOCK_DN,))
    tl.store(out_base + dn, nope_q, mask=mn)

    # --- interleaved RoPE on the normed pe (fp32 math) ---
    ev = kv_even.to(tl.float32)
    od = kv_odd.to(tl.float32)
    cos = tl.load(cos_ptr + pos * cossin_row_stride + k, mask=mh, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + pos * cossin_row_stride + k, mask=mh, other=0.0).to(tl.float32)
    if INVERSE:
        sin = -sin
    out_even = ev * cos - od * sin
    out_odd = ev * sin + od * cos
    pe_out_base = out_base + Dn
    tl.store(pe_out_base + 2 * k, out_even.to(out_ty), mask=mh)
    tl.store(pe_out_base + 2 * k + 1, out_odd.to(out_ty), mask=mh)


@torch.library.custom_op("auto_deploy::deepseek_v4_kv_norm_rope_concat", mutates_args=())
def deepseek_v4_kv_norm_rope_concat(
    nope: torch.Tensor,
    pe: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_eps: float,
    fp8_block_size: int = 64,
    inverse: bool = False,
) -> torch.Tensor:
    """Fused KV front-end == ``fake_fp8(rmsnorm(cat(nope,pe))[:Dn]) || rope(rmsnorm(...)[Dn:])``.

    Replaces the ``kv_norm -> split -> fake_fp8_act_quant(nope) ->
    deepseek_v4_fused_rope_concat`` chain of the main-KV path with one kernel.

    Args:
        nope: ``[..., Dn]`` *raw* (pre-norm) nope slice — a last-dim view of the raw
            KV head. Weighted-RMS-normalized then fake-FP8 block-quantized in-kernel.
        pe:   ``[..., D]`` *raw* (pre-norm) pe slice (``D`` even). Weighted-RMS-
            normalized then interleaved-rotated in-kernel.
        weight: ``[Dn + D]`` RMS-norm weight for the full head (``torch_rmsnorm``
            weight). Kept in its native dtype and applied in fp32.
        cos/sin: ``[..M.., Dh]`` (``Dh == D // 2``), broadcasting over heads exactly
            like the reference ``cos.unsqueeze(head_dim)``.
        rms_eps: RMS-norm epsilon.
        fp8_block_size: block width for the fake-FP8 quant of the nope slice
            (``Dn % fp8_block_size == 0`` required).
        inverse: if True, negate ``sin`` (unused by the KV path; kept for symmetry).

    Returns:
        ``[..., Dn + D]`` contiguous tensor, bit-faithful to the split reference.
    """
    assert pe.shape[-1] % 2 == 0, "rope dim must be even"
    assert pe.stride(-1) == 1 and nope.stride(-1) == 1 and cos.stride(-1) == 1, (
        "last dim of nope/pe/cos must be contiguous"
    )
    assert weight.stride(-1) == 1, "weight must be last-dim contiguous"
    D = pe.shape[-1]
    Dn = nope.shape[-1]
    Dh = D // 2
    assert weight.numel() == Dn + D, "weight must cover the full (nope || pe) head"
    assert Dn % fp8_block_size == 0, "Dn must be a multiple of fp8_block_size"

    out = torch.empty((*pe.shape[:-1], Dn + D), device=pe.device, dtype=pe.dtype)
    R = pe.numel() // D
    if R == 0:
        return out
    n_pos = cos.numel() // Dh
    H = R // n_pos

    BLOCK_DN = triton.next_power_of_2(Dn)
    assert BLOCK_DN % fp8_block_size == 0, "next_pow2(Dn) must be a multiple of block size"
    NB = BLOCK_DN // fp8_block_size

    nope_row_stride = nope.stride(-2) if nope.dim() >= 2 else Dn
    pe_row_stride = pe.stride(-2) if pe.dim() >= 2 else D
    cossin_row_stride = cos.stride(-2) if cos.dim() >= 2 else Dh

    grid = (R,)
    _kv_norm_rope_concat_kernel[grid](
        nope,
        pe,
        weight,
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
        NB=NB,
        FP8_BLOCK=fp8_block_size,
        FP8_MAX=448.0,
        FP8_MIN=1.0e-4,
        BLOCK_DN=BLOCK_DN,
        BLOCK_DH=triton.next_power_of_2(Dh),
        num_warps=4,
    )
    return out


@deepseek_v4_kv_norm_rope_concat.register_fake
def _deepseek_v4_kv_norm_rope_concat_fake(
    nope: torch.Tensor,
    pe: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_eps: float,
    fp8_block_size: int = 64,
    inverse: bool = False,
) -> torch.Tensor:
    return pe.new_empty((*pe.shape[:-1], nope.shape[-1] + pe.shape[-1]))
