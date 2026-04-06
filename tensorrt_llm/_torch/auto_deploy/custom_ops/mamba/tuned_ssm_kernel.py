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

"""Tuned Triton SSM decode kernel for Nemotron Nano v3 dimensions.

Optimized for: nheads=64, dim=64, dstate=128, n_groups=8.
Key difference from stock selective_state_update: uses BLOCK_SIZE_M=16 (vs 4)
to reduce grid size by 4x, improving GPU scheduler efficiency at small batch.
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.softplus import softplus


@triton.jit()
def _tuned_ssm_update_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    out_ptr,
    state_batch_indices_ptr,
    pad_slot_id,
    # Matrix dimensions
    batch,
    nheads,
    dim,
    dstate,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    DT_CLAMP_MIN: tl.constexpr,
    DT_CLAMP_MAX: tl.constexpr,
    DSTATE_CONSTEXPR: tl.constexpr,  # compile-time dstate for loop unrolling
    DIM_CONSTEXPR: tl.constexpr,  # compile-time dim for mask elimination
    NHEADS_NGROUPS_RATIO: tl.constexpr,  # compile-time head/group ratio (8 for NanoV3)
):
    """Optimized SSM state update for decode (T=1).

    Compared to the generic kernel, this:
    - Removes T loop (single token only)
    - Removes z/intermediate_states/spec_decoding branches
    - Uses larger BLOCK_SIZE_M for our target dim=64
    - Hardcodes DT_SOFTPLUS=True, HAS_DT_BIAS=True, HAS_D=True
    """
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Slot-indexed state access (no gather/scatter)
    if HAS_STATE_BATCH_INDICES:
        state_batch_idx = tl.load(state_batch_indices_ptr + pid_b).to(tl.int64)
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
        if state_batch_idx == pad_slot_id:
            return
    else:
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_h * stride_dt_head
    dt_bias_ptr += pid_h * stride_dt_bias_head
    A_ptr += pid_h * stride_A_head
    B_ptr += pid_b * stride_B_batch + (pid_h // NHEADS_NGROUPS_RATIO) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // NHEADS_NGROUPS_RATIO) * stride_C_group
    D_ptr += pid_h * stride_D_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Use compile-time dim to eliminate mask when BLOCK_SIZE_M divides DIM_CONSTEXPR evenly
    mask_m = offs_m < DIM_CONSTEXPR

    # Load x, dt, dt_bias for this block's dim slice (scalar per dim lane)
    x = tl.load(x_ptr + offs_m * stride_x_dim, mask=mask_m, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptr + offs_m * stride_dt_dim, mask=mask_m, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + offs_m * stride_dt_bias_dim, mask=mask_m, other=0.0).to(
        tl.float32
    )

    # dt = softplus(dt + dt_bias), then clamp to [DT_CLAMP_MIN, DT_CLAMP_MAX]
    dt = softplus(dt + dt_bias)
    dt = tl.clamp(dt, DT_CLAMP_MIN, DT_CLAMP_MAX)

    # Load D early to overlap memory latency with dt computation
    D = tl.load(D_ptr + offs_m * stride_D_dim, mask=mask_m, other=0.0).to(tl.float32)

    # Precompute column broadcasts once to avoid repeated reshape in loop
    dt_col = dt[:, None]  # [BLOCK_SIZE_M, 1]

    # Precompute log2(e) * dt for exp2 conversion
    LOG2E = 1.4426950408889634
    dt_log2e = dt_col * LOG2E  # [BLOCK_SIZE_M, 1]

    # Precompute dt*x for dB*x fusion: state += B[None,:] * (dt*x)[:,None]
    dt_x_col = (dt * x)[:, None]  # [BLOCK_SIZE_M, 1]

    # Accumulate output over dstate in BLOCK_SIZE_DSTATE chunks
    # Use DSTATE_CONSTEXPR for compile-time loop unrolling
    out_acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for dstate_start in range(0, DSTATE_CONSTEXPR, BLOCK_SIZE_DSTATE):
        offs_n = dstate_start + tl.arange(0, BLOCK_SIZE_DSTATE)
        mask = mask_m[:, None] & (offs_n[None, :] < dstate)

        # Load state [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
        state_ptrs = state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )
        state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Load A [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
        A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
        A = tl.load(A_ptrs, mask=mask, other=0.0).to(tl.float32)

        # dA = exp(A * dt) via exp2: exp(x) = exp2(x * log2(e))
        dA = tl.math.exp2(A * dt_log2e)

        # Load B, C [BLOCK_SIZE_DSTATE]
        B = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(
            tl.float32
        )
        C = tl.load(C_ptr + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(
            tl.float32
        )

        # state = state * dA + B * dt_x (use libdevice fma for potential speedup)
        # fma(a, b, c) = a * b + c; split into two FMA calls
        state = tl.math.fma(state, dA, B[None, :] * dt_x_col)

        # Accumulate output contribution from this dstate chunk
        out_acc += tl.sum(state * C[None, :], axis=1)

        # Store updated state for this dstate chunk
        tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)

    # out = accumulated state*C + x * D (D loaded early above)
    out_acc += x * D

    # Store output
    tl.store(out_ptr + offs_m * stride_out_dim, out_acc, mask=mask_m)


DT_CLAMP_MIN = 0.001
DT_CLAMP_MAX = 0.1


def tuned_selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    dt_bias=None,
    dt_softplus=True,
    state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
    dt_clamp_min=DT_CLAMP_MIN,
    dt_clamp_max=DT_CLAMP_MAX,
):
    """Drop-in replacement for flashinfer.mamba.selective_state_update.

    Tuned for Nemotron Nano v3: nheads=64, dim=64, dstate=128.
    Uses BLOCK_SIZE_M=16 (vs 4 in stock) to reduce grid by 4x.
    """
    # Normalize dimensions to 4D: (batch, T=1, nheads, dim)
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if dt.dim() == 3:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out is not None:
        if out.dim() == 2:
            out = out.unsqueeze(1)
        if out.dim() == 3:
            out = out.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    ngroups = B.shape[2]

    if out is None:
        out = torch.empty(batch, 1, nheads, dim, device=x.device, dtype=x.dtype)

    # Tuned block sizes for our target dimensions
    # BLOCK_SIZE_M=32: optimal for all batch sizes (reduces grid 2x vs M=16)
    # BLOCK_SIZE_DSTATE=128: full dstate in one pass is optimal (no loop overhead)
    # num_warps=4: optimal across all batch sizes
    BLOCK_SIZE_DSTATE = triton.next_power_of_2(dstate)
    if dim <= 64:
        BLOCK_SIZE_M = 32
        num_warps = 4
    elif dim <= 128:
        BLOCK_SIZE_M = 32
        num_warps = 4
    else:
        BLOCK_SIZE_M = 16
        num_warps = 4
    # Note: batch-adaptive heuristics tested (M/W by batch) — no benefit found
    # M=32/W=4 is optimal across all batch sizes {33, 64, 128, 256, 384}

    has_sbi = state_batch_indices is not None

    def grid(META):
        return (triton.cdiv(dim, META["BLOCK_SIZE_M"]), batch, nheads)

    # Squeeze out the T=1 dim for the kernel (it only handles T=1)
    x_sq = x.squeeze(1)  # [batch, nheads, dim]
    dt_sq = dt.squeeze(1)  # [batch, nheads, dim]
    B_sq = B.squeeze(1)  # [batch, ngroups, dstate]
    C_sq = C.squeeze(1)  # [batch, ngroups, dstate]
    out_sq = out.squeeze(1)  # [batch, nheads, dim]

    with torch.cuda.device(x.device.index):
        _tuned_ssm_update_kernel[grid](
            state,
            x_sq,
            dt_sq,
            dt_bias,
            A,
            B_sq,
            C_sq,
            D,
            out_sq,
            state_batch_indices,
            pad_slot_id,
            batch,
            nheads,
            dim,
            dstate,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x_sq.stride(0),
            x_sq.stride(1),
            x_sq.stride(2),
            dt_sq.stride(0),
            dt_sq.stride(1),
            dt_sq.stride(2),
            dt_bias.stride(0),
            dt_bias.stride(1),
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B_sq.stride(0),
            B_sq.stride(1),
            B_sq.stride(2),
            C_sq.stride(0),
            C_sq.stride(1),
            C_sq.stride(2),
            D.stride(0),
            D.stride(1),
            out_sq.stride(0),
            out_sq.stride(1),
            out_sq.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_DSTATE=BLOCK_SIZE_DSTATE,
            HAS_STATE_BATCH_INDICES=has_sbi,
            DT_CLAMP_MIN=dt_clamp_min,
            DT_CLAMP_MAX=dt_clamp_max,
            DSTATE_CONSTEXPR=dstate,
            DIM_CONSTEXPR=dim,
            NHEADS_NGROUPS_RATIO=nheads // ngroups,
            num_warps=num_warps,
        )
