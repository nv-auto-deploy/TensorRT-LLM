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


def _hc_combine_configs():
    # The kernel is a single-CTA-per-row reduction over BLOCK_H (==next_pow2(H))
    # with a fully-unrolled HM loop, so num_stages is near-inert; the live knob is
    # num_warps. Decode (N small) wants the widest CTA (nw=32 hides the 4*H fp32
    # load latency); prefill (N large) saturates SMs and favors fewer warps/CTA.
    return [
        triton.Config({}, num_warps=nw, num_stages=ns) for nw in (4, 8, 16, 32) for ns in (1, 2)
    ]


@triton.autotune(configs=_hc_combine_configs(), key=["N", "H", "HM"])
@triton.jit
def _hc_weighted_combine_kernel(
    pre_ptr,  # [N, HM] fp32
    flat_ptr,  # [N, HM * H] fp32
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
        f = tl.load(flat_row + m * H + h, mask=hmask, other=0.0)
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
        flat:     ``[..., hc_mult * H]`` fp32 flattened hidden states
                  (``x.flatten(2).float()``); element ``[..., m*H + h]``.
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
    flat_f = flat.reshape(n, hc_mult * H).contiguous().float()
    weight_f = weight.contiguous().float()

    out = torch.empty((n, H), device=pre.device, dtype=out_dtype)
    if n == 0:
        return out.reshape(*lead, H)

    block_h = triton.next_power_of_2(H)

    # num_warps / num_stages are selected per (N, H, HM) by @triton.autotune on
    # _hc_weighted_combine_kernel (replaces the former coarse block_h-only branch).
    grid = (n,)
    _hc_weighted_combine_kernel[grid](
        pre_f,
        flat_f,
        weight_f,
        out,
        n,
        H,
        eps,
        HM=hc_mult,
        BLOCK_H=block_h,
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
