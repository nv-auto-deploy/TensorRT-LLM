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
    nheads_ngroups_ratio,
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
    B_ptr += pid_b * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    C_ptr += pid_b * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    D_ptr += pid_h * stride_D_head
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)

    # Load state [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load x, dt, dt_bias for this block's dim slice
    x = tl.load(x_ptr + offs_m * stride_x_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    dt = tl.load(dt_ptr + offs_m * stride_dt_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    dt_bias = tl.load(dt_bias_ptr + offs_m * stride_dt_bias_dim, mask=offs_m < dim, other=0.0).to(
        tl.float32
    )

    # dt = softplus(dt + dt_bias)
    dt = softplus(dt + dt_bias)

    # Load A [BLOCK_SIZE_M, BLOCK_SIZE_DSTATE]
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate
    A = tl.load(A_ptrs, mask=mask, other=0.0).to(tl.float32)

    # dA = exp(A * dt)
    dA = tl.exp(A * dt[:, None])

    # Load B, C [BLOCK_SIZE_DSTATE]
    B = tl.load(B_ptr + offs_n * stride_B_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)
    C = tl.load(C_ptr + offs_n * stride_C_dstate, mask=offs_n < dstate, other=0.0).to(tl.float32)

    # dB = B * dt
    dB = B[None, :] * dt[:, None]

    # state = state * dA + dB * x
    state = state * dA + dB * x[:, None]

    # out = sum(state * C, axis=1) + x * D
    out = tl.sum(state * C[None, :], axis=1)
    D = tl.load(D_ptr + offs_m * stride_D_dim, mask=offs_m < dim, other=0.0).to(tl.float32)
    out += x * D

    # Store output and updated state
    tl.store(out_ptr + offs_m * stride_out_dim, out, mask=offs_m < dim)
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask)


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
    BLOCK_SIZE_DSTATE = triton.next_power_of_2(dstate)
    if dim <= 64:
        BLOCK_SIZE_M = 16
        num_warps = 4
    elif dim <= 128:
        BLOCK_SIZE_M = 16
        num_warps = 4
    else:
        BLOCK_SIZE_M = 8
        num_warps = 4

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
            nheads // ngroups,
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
            num_warps=num_warps,
        )
