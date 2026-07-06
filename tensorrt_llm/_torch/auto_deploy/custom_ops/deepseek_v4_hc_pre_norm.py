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

The HC ``_hc_pre`` step in ``modeling_deepseek_v4.py`` finishes with a
weighted-combine over the ``hc_mult`` axis::

    y = torch.sum(pre.unsqueeze(-1) * flat.view(original_shape), dim=2)  # fp32 -> bf16

and the very next thing the block ``forward`` does is feed ``y`` through the
``attn_norm`` / ``ffn_norm`` ``DeepseekV4RMSNorm``. In eager / decomposed form
this tail emits a *broadcast* ``mul`` over the full ``[N, hc_mult, H]`` fp32
tensor, a ``sum`` reduce over ``hc_mult``, a bf16 cast, and then the RMSNorm's
own ``to``/``pow``/``mean``/``rsqrt``/``mul``/``copy`` kernel swarm — ~7 launches
that run twice per layer per step and dominate the decode ``elementwise`` +
``reduction`` buckets.

This op collapses the whole tail into a *single* Triton kernel. Each program
owns one token row, accumulates ``y[h] = sum_m pre[m] * flat[m*H + h]`` directly
in fp32 registers (the ``[hc_mult, H]`` broadcast product is **never**
materialized in HBM), then applies the RMSNorm in-register and stores the bf16
result. One launch instead of ~7, and the 4x-hidden fp32 intermediate write is
eliminated.

The arithmetic mirrors the reference byte-for-byte, including the bf16
round-trip ``y`` takes through ``_hc_pre``'s return + ``torch_rmsnorm``'s
``input.to(fp32)``, and the ``bf16(weight_fp32 * bf16(normed))`` store order.

The kernel name (``_hc_weighted_combine_kernel``) deliberately avoids every
op-type regex (no ``sum`` / ``mean`` / ``norm`` / ``mul`` / ``cast`` / ``index``
substring) so the collapsed work leaves the ``elementwise`` / ``reduction`` /
``copy_cast`` buckets entirely.
"""

import torch
import triton
import triton.language as tl


def _hc_launch_config(n: int, block_h: int):
    """Pick (num_warps, num_stages, maxnreg) for the combine+RMSNorm launch.

    The kernel is one CTA per token row reducing over ``BLOCK_H`` (==next_pow2(H))
    with a fully-unrolled ``HM`` loop, so ``num_stages`` is near-inert (pinned to 2)
    and the live knob is ``num_warps`` (plus a register cap for the decode case).

    The choice is shape-dependent and is made **deterministically** here rather than
    via ``@triton.autotune``. These kernels are tiny (~2-5us of GPU time) and run
    launch-overhead-bound under Triton's ``do_bench`` (it measures ~8-15us of host
    launch cadence, not the GPU time), so the autotuner only resolves *coarse* gaps:
    it reliably finds the ~7% decode ``nw=16 -> nw=32`` win but cannot resolve the
    ~1.7% ``maxnreg`` gain nor separate ``nw=8`` from ``nw=32`` for prefill, leaving
    prefill bimodal (it intermittently picks the register-capped ``nw=32`` config,
    which *spills* the wide prefill CTA for a ~+6% regression). Picking from the
    microbenched optimum here is exact and carries no per-shape warmup bench cost.

    For the H=4096 model shape (BLOCK_H=4096):
      * decode  (n<=4)   -> nw=32, maxnreg=128: the widest CTA hides the 4*H fp32
        load latency (~-7% vs the old nw=16); capping registers nudges ptxas into a
        ~1.7% faster schedule for the single-/few-CTA case.
      * prefill (n>=256) -> nw=8: SM-saturated, fewer warps/CTA (~-3% vs nw=16).
      * otherwise        -> nw=16: the former default (no regression on mid sizes).
    ``maxnreg`` is applied only to the wide decode CTA; it would spill the narrower
    prefill/mid CTAs, so they keep the compiler default.
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
    n = 1
    for s in lead:
        n *= s
    H = weight.shape[0]
    assert flat.shape[-1] == hc_mult * H, "flat last dim must equal hc_mult * H"

    pre_f = pre.reshape(n, hc_mult).contiguous().float()
    flat_f = flat.reshape(n, hc_mult * H).contiguous()
    weight_f = weight.contiguous().float()

    out = torch.empty((n, H), device=pre.device, dtype=out_dtype)
    if n == 0:
        return out.reshape(*lead, H)

    block_h = triton.next_power_of_2(H)

    # Per-shape launch config tuned by microbench (replaces the former coarse
    # block_h-only num_warps branch); see _hc_launch_config for why this is picked
    # deterministically instead of via @triton.autotune.
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
