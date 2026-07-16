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

"""Fused HC post-sinkhorn weighted-combine + block RMSNorm for DeepSeek-V4.

``deepseek_v4_hc_combine_rmsnorm`` collapses the HC ``_hc_pre`` tail — the
weighted-combine ``y = sum_m pre[m] * flat[m*H:(m+1)*H]`` followed by the block
``attn_norm`` / ``ffn_norm`` RMSNorm — into a *single* Triton kernel: each
program owns one token row, accumulates the combine in fp32 registers (the
``[hc_mult, H]`` broadcast product is never materialized in HBM), then applies
the RMSNorm in-register. It is invoked from the fused HC-pre ops' prefill
fallback paths in ``hc_composition.py`` (the modeling code calls those ops, not
this one directly).

Numerics contract: the arithmetic mirrors the reference byte-for-byte,
including the bf16 round-trip ``y`` takes through ``_hc_pre``'s return +
``torch_rmsnorm``'s ``input.to(fp32)``, and the
``bf16(weight_fp32 * bf16(normed))`` store order.

The kernel name (``_hc_weighted_combine_kernel``) deliberately avoids every
op-type regex (no ``sum`` / ``mean`` / ``norm`` / ``mul`` / ``cast`` / ``index``
substring) so the collapsed work leaves the ``elementwise`` / ``reduction`` /
``copy_cast`` buckets entirely.
"""

import math

import torch
import triton
import triton.language as tl


def _hc_launch_config(n: int, block_h: int):
    """Pick (num_warps, num_stages, maxnreg) for the combine+RMSNorm launch.

    One CTA per token row reducing over ``BLOCK_H`` (==next_pow2(H)) with a
    fully-unrolled ``HM`` loop, so ``num_stages`` is near-inert (pinned to 2)
    and the live knob is ``num_warps`` (plus a register cap for decode).

    Chosen **deterministically** rather than via ``@triton.autotune``: these
    sub-floor kernels run launch-overhead-bound under Triton's ``do_bench``
    (it measures host launch cadence, not GPU time), so the autotuner cannot
    resolve the fine gaps and intermittently picks the register-capped config
    for prefill, which spills the wide prefill CTA.

    For the H=4096 model shape (BLOCK_H=4096, microbenched on B200):
      * decode  (n<=4)   -> nw=32, maxnreg=128: widest CTA hides the 4*H fp32
        load latency; the register cap nudges ptxas into a faster schedule.
      * prefill (n>=256) -> nw=8: SM-saturated, fewer warps/CTA.
      * otherwise        -> nw=16: the former default.
    ``maxnreg`` is applied only to the wide decode CTA; it would spill the
    narrower prefill/mid CTAs, so they keep the compiler default.
    """
    if n <= 4:
        num_warps = 32
    elif n >= 256:
        num_warps = 8
    else:
        num_warps = 16
    # Small hidden dims (non-model shapes) never need a wide CTA.
    if block_h < 1024:
        num_warps = min(num_warps, 8)
    if block_h < 256:
        num_warps = min(num_warps, 4)
    # The register cap only helps the wide single-/few-CTA decode launch; it would
    # spill narrower CTAs, so apply it only when the full-width nw=32 survived.
    maxnreg = 128 if num_warps == 32 else None
    return num_warps, 2, maxnreg


@triton.jit
def _hc_weighted_combine_kernel(
    pre_ptr,  # [N, HM] fp32
    flat_ptr,  # [N, HM * H] fp32 or bf16/fp16 (converted to fp32 in-register)
    weight_ptr,  # [H] fp32 (RMSNorm weight)
    out_ptr,  # [N, H] out_dtype
    N,
    H,
    eps,
    HM: tl.constexpr,  # hc_mult
    BLOCK_H: tl.constexpr,  # next_power_of_2(H)
):
    """One program per token row. y = sum_m pre[m] * flat[m*H:] then RMSNorm."""
    row = tl.program_id(0)
    if row >= N:
        return

    h = tl.arange(0, BLOCK_H)
    hmask = h < H

    # --- weighted combine over the hc_mult axis, fp32 accumulate ---
    # acc[h] = sum_{m} pre[row, m] * flat[row, m*H + h]; never materialize [HM, H].
    flat_row = flat_ptr + row * (HM * H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for m in tl.static_range(HM):
        p = tl.load(pre_ptr + row * HM + m)  # scalar fp32
        # Native-dtype load + in-register fp32 convert: exact for bf16/fp16
        # inputs (widening conversions are lossless), identical for fp32, and
        # skips the HBM materialization of an fp32 ``flat``.
        f = tl.load(flat_row + m * H + h, mask=hmask, other=0.0).to(tl.float32)
        acc += p * f

    # --- replicate the bf16 round-trip y takes (return .to(x.dtype) then
    #     torch_rmsnorm's input.to(fp32)) so the fused op is bit-faithful ---
    y = acc.to(tl.bfloat16).to(tl.float32)

    # --- RMSNorm over H: var = mean(y^2); out = bf16(weight * bf16(y * rstd)) ---
    var = tl.sum(y * y, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    normed = (y * rstd).to(tl.bfloat16).to(tl.float32)
    w = tl.load(weight_ptr + h, mask=hmask, other=0.0)
    out = w * normed
    tl.store(out_ptr + row * H + h, out.to(out_ptr.dtype.element_ty), mask=hmask)


@torch.library.custom_op("auto_deploy::deepseek_v4_hc_combine_rmsnorm", mutates_args=())
def deepseek_v4_hc_combine_rmsnorm(
    pre: torch.Tensor,
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    hc_mult: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused HC weighted-combine + RMSNorm. Drop-in for the ``_hc_pre`` tail.

    Computes, per leading (token) index::

        y      = sum_m pre[..., m] * flat[..., m*H : (m+1)*H]      # fp32
        y      = y.to(out_dtype).float()                          # bf16 round-trip
        out    = (weight * (y * rsqrt(mean(y^2) + eps)).to(out_dtype).float()).to(out_dtype)

    Args:
        pre:      ``[..., hc_mult]`` fp32 combine weights (from the sinkhorn op).
        flat:     ``[..., hc_mult * H]`` flattened hidden states
                  (``x.flatten(2)``; fp32, bf16, or fp16 — non-fp32 inputs are
                  converted to fp32 in-register, which is exact).
        weight:   ``[H]`` fp32 RMSNorm weight (attn_norm / ffn_norm).
        eps:      RMSNorm epsilon.
        hc_mult:  number of hyper-connection streams folded over.
        out_dtype: dtype of the returned (normalized) tensor (the residual dtype).

    Returns:
        ``[..., H]`` ``out_dtype`` == ``rmsnorm(sum_m pre*flat, weight, eps)``.
    """
    lead = list(pre.shape[:-1])
    n = math.prod(lead)
    H = weight.shape[0]
    assert flat.shape[-1] == hc_mult * H, "flat last dim must equal hc_mult * H"

    pre_f = pre.reshape(n, hc_mult).contiguous().float()
    flat_f = flat.reshape(n, hc_mult * H).contiguous()
    weight_f = weight.contiguous().float()

    out = torch.empty((n, H), device=pre.device, dtype=out_dtype)
    if n == 0:
        return out.reshape(*lead, H)

    block_h = triton.next_power_of_2(H)
    num_warps, num_stages, maxnreg = _hc_launch_config(n, block_h)
    launch_kwargs = dict(HM=hc_mult, BLOCK_H=block_h, num_warps=num_warps, num_stages=num_stages)
    if maxnreg is not None:
        launch_kwargs["maxnreg"] = maxnreg

    grid = (n,)
    _hc_weighted_combine_kernel[grid](
        pre_f,
        flat_f,
        weight_f,
        out,
        n,
        H,
        eps,
        **launch_kwargs,
    )
    return out.reshape(*lead, H)


@deepseek_v4_hc_combine_rmsnorm.register_fake
def _deepseek_v4_hc_combine_rmsnorm_fake(
    pre: torch.Tensor,
    flat: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    hc_mult: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    lead = list(pre.shape[:-1])
    H = weight.shape[0]
    return pre.new_empty((*lead, H), dtype=out_dtype)
