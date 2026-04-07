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

"""Fused Mamba decode kernel: conv1d_update + SiLU + SSM_update in one Triton kernel.

Eliminates intermediate tensor writes between conv1d and SSM by computing
both operations in a single kernel launch. The conv1d output feeds directly
into the SSM state update without going through HBM.

Target: Nemotron Nano v3 (nheads=64, dim=64, dstate=128, ngroups=8, conv_dim=6144)
"""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.mamba.softplus import softplus


@triton.jit()
def _bc_conv_compute_kernel(
    conv_input_ptr,  # [batch, conv_dim]
    conv_state_ptr,  # [max_batch, conv_dim, kernel_width-1]
    conv_weight_ptr,  # [conv_dim, kernel_width]
    conv_bias_ptr,  # [conv_dim]
    conv_slot_idx_ptr,  # [batch]
    bc_buf_ptr,  # [batch, 2, ngroups, dstate] — output (B=[:, 0, :, :], C=[:, 1, :, :])
    batch,
    ngroups,
    dstate,
    intermediate_size,
    conv_dim,
    kernel_width: tl.constexpr,
    stride_ci_b,
    stride_ci_d,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    stride_cw_d,
    stride_cw_w,
    BLOCK_DSTATE: tl.constexpr,
):
    """Compute B/C conv values, update B/C state, and store results to bc_buf.

    Grid: (batch, ngroups) — each CTA handles one (batch, group) pair.
    Eliminates 7/8 redundant B/C computation across heads in the same group.
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    conv_slot = tl.load(conv_slot_idx_ptr + pid_b).to(tl.int64)

    offs_n = tl.arange(0, BLOCK_DSTATE)
    mask_n = offs_n < dstate

    B_offset = intermediate_size + pid_g * dstate
    C_offset = intermediate_size + ngroups * dstate + pid_g * dstate

    B_channels = B_offset + offs_n
    C_channels = C_offset + offs_n
    mask_bc = (B_channels < conv_dim) & mask_n

    B_vals = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)
    C_vals = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    for k in range(kernel_width - 1):
        b_state = tl.load(
            conv_state_ptr + conv_slot * stride_cs_b + B_channels * stride_cs_d + k * stride_cs_w,
            mask=mask_bc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        b_w = tl.load(
            conv_weight_ptr + B_channels * stride_cw_d + k * stride_cw_w, mask=mask_bc, other=0.0
        ).to(tl.float32)
        B_vals += b_state * b_w

        c_state = tl.load(
            conv_state_ptr + conv_slot * stride_cs_b + C_channels * stride_cs_d + k * stride_cs_w,
            mask=mask_bc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        c_w = tl.load(
            conv_weight_ptr + C_channels * stride_cw_d + k * stride_cw_w, mask=mask_bc, other=0.0
        ).to(tl.float32)
        C_vals += c_state * c_w

    # Current input for B, C
    b_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + B_channels * stride_ci_d, mask=mask_bc, other=0.0
    ).to(tl.float32)
    b_w_last = tl.load(
        conv_weight_ptr + B_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    b_bias = tl.load(conv_bias_ptr + B_channels, mask=mask_bc, other=0.0).to(tl.float32)
    B_vals += b_in * b_w_last + b_bias

    c_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + C_channels * stride_ci_d, mask=mask_bc, other=0.0
    ).to(tl.float32)
    c_w_last = tl.load(
        conv_weight_ptr + C_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    c_bias = tl.load(conv_bias_ptr + C_channels, mask=mask_bc, other=0.0).to(tl.float32)
    C_vals += c_in * c_w_last + c_bias

    # Apply SiLU to B and C
    B_vals = B_vals * (1.0 / (1.0 + tl.exp(-B_vals)))
    C_vals = C_vals * (1.0 / (1.0 + tl.exp(-C_vals)))

    # Store to bc_buf: [batch, 2, ngroups, dstate]
    # B at [pid_b, 0, pid_g, offs_n], C at [pid_b, 1, pid_g, offs_n]
    b_buf_offs = pid_b * (2 * ngroups * dstate) + 0 * (ngroups * dstate) + pid_g * dstate + offs_n
    c_buf_offs = pid_b * (2 * ngroups * dstate) + 1 * (ngroups * dstate) + pid_g * dstate + offs_n
    tl.store(bc_buf_ptr + b_buf_offs, B_vals.to(tl.bfloat16), mask=mask_n)
    tl.store(bc_buf_ptr + c_buf_offs, C_vals.to(tl.bfloat16), mask=mask_n)

    # Update B/C conv state (shift + append new input) — race-free since one CTA per group
    for k in range(kernel_width - 2):
        old_b = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + B_channels * stride_cs_d
            + (k + 1) * stride_cs_w,
            mask=mask_bc,
            other=0.0,
        )
        tl.store(
            conv_state_ptr + conv_slot * stride_cs_b + B_channels * stride_cs_d + k * stride_cs_w,
            old_b,
            mask=mask_bc,
        )
    tl.store(
        conv_state_ptr
        + conv_slot * stride_cs_b
        + B_channels * stride_cs_d
        + (kernel_width - 2) * stride_cs_w,
        b_in.to(conv_state_ptr.dtype.element_ty),
        mask=mask_bc,
    )

    for k in range(kernel_width - 2):
        old_c = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + C_channels * stride_cs_d
            + (k + 1) * stride_cs_w,
            mask=mask_bc,
            other=0.0,
        )
        tl.store(
            conv_state_ptr + conv_slot * stride_cs_b + C_channels * stride_cs_d + k * stride_cs_w,
            old_c,
            mask=mask_bc,
        )
    tl.store(
        conv_state_ptr
        + conv_slot * stride_cs_b
        + C_channels * stride_cs_d
        + (kernel_width - 2) * stride_cs_w,
        c_in.to(conv_state_ptr.dtype.element_ty),
        mask=mask_bc,
    )


@triton.jit()
def _bc_conv_state_update_kernel(
    conv_input_ptr,  # [batch, conv_dim]
    conv_state_ptr,  # [max_batch, conv_dim, kernel_width-1]
    conv_slot_idx_ptr,  # [batch]
    batch,
    ngroups,
    dstate,
    intermediate_size,
    conv_dim,
    kernel_width: tl.constexpr,
    stride_ci_b,
    stride_ci_d,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    BLOCK_DSTATE: tl.constexpr,
):
    """Update B/C conv state for all groups. Grid: (batch, ngroups).

    This kernel runs AFTER _fused_conv_ssm_kernel so the B/C state shift
    is serialized (no race condition). The main kernel reads the PRE-shift
    state, then this kernel performs the shift.
    """
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    conv_slot = tl.load(conv_slot_idx_ptr + pid_b).to(tl.int64)

    offs_n = tl.arange(0, BLOCK_DSTATE)
    mask_n = offs_n < dstate

    B_offset = intermediate_size + pid_g * dstate
    C_offset = intermediate_size + ngroups * dstate + pid_g * dstate

    B_channels = B_offset + offs_n
    C_channels = C_offset + offs_n
    mask_bc = (B_channels < conv_dim) & mask_n

    # Load the new inputs for B/C
    b_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + B_channels * stride_ci_d,
        mask=mask_bc,
        other=0.0,
    )
    c_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + C_channels * stride_ci_d,
        mask=mask_bc,
        other=0.0,
    )

    # Shift B state: [k] <- [k+1], then store new input at end
    for k in range(kernel_width - 2):
        old_b = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + B_channels * stride_cs_d
            + (k + 1) * stride_cs_w,
            mask=mask_bc,
            other=0.0,
        )
        tl.store(
            conv_state_ptr + conv_slot * stride_cs_b + B_channels * stride_cs_d + k * stride_cs_w,
            old_b,
            mask=mask_bc,
        )
    tl.store(
        conv_state_ptr
        + conv_slot * stride_cs_b
        + B_channels * stride_cs_d
        + (kernel_width - 2) * stride_cs_w,
        b_in,
        mask=mask_bc,
    )

    # Shift C state
    for k in range(kernel_width - 2):
        old_c = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + C_channels * stride_cs_d
            + (k + 1) * stride_cs_w,
            mask=mask_bc,
            other=0.0,
        )
        tl.store(
            conv_state_ptr + conv_slot * stride_cs_b + C_channels * stride_cs_d + k * stride_cs_w,
            old_c,
            mask=mask_bc,
        )
    tl.store(
        conv_state_ptr
        + conv_slot * stride_cs_b
        + C_channels * stride_cs_d
        + (kernel_width - 2) * stride_cs_w,
        c_in,
        mask=mask_bc,
    )


@triton.jit()
def _fused_conv_ssm_kernel(
    # Conv inputs
    conv_input_ptr,  # [batch, conv_dim] — raw input to conv (before split)
    conv_state_ptr,  # [batch, conv_dim, kernel_width-1]
    conv_weight_ptr,  # [conv_dim, kernel_width]
    conv_bias_ptr,  # [conv_dim]
    # SSM inputs
    dt_ptr,  # [batch, nheads] (broadcast to [batch, nheads, dim])
    dt_bias_ptr,  # [nheads] (broadcast to [nheads, dim])
    A_ptr,  # [nheads] (broadcast)
    D_ptr,  # [nheads] (broadcast)
    ssm_state_ptr,  # [max_batch, nheads, dim, dstate] (slot-indexed)
    # Slot indices for cache access
    conv_slot_idx_ptr,  # [batch] int32 — conv state cache slot per sequence
    ssm_slot_idx_ptr,  # [batch] int32 — SSM state cache slot per sequence
    # Output
    ssm_out_ptr,  # [batch, nheads, dim]
    # Dimensions
    batch,
    conv_dim,  # intermediate_size + 2*ngroups*dstate
    intermediate_size,  # nheads * dim
    nheads,
    dim,
    dstate,
    ngroups,
    nheads_per_group,
    kernel_width: tl.constexpr,
    # Strides
    stride_ci_b,
    stride_ci_d,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    stride_cw_d,
    stride_cw_w,
    stride_ss_b,
    stride_ss_h,
    stride_ss_d,
    stride_ss_n,
    stride_so_b,
    stride_so_h,
    stride_so_d,
    # Meta
    BLOCK_DIM: tl.constexpr,  # tiles over head_dim for SSM
    BLOCK_DSTATE: tl.constexpr,
):
    """Fused conv1d_update + SiLU + SSM for one (batch, head) tile.

    Grid: (cdiv(dim, BLOCK_DIM), batch, nheads)

    This kernel:
    1. For each dim element in the tile, computes the conv1d update for the
       corresponding hidden channel (offset = head_idx * dim + dim_offset)
    2. Applies SiLU activation
    3. Also computes conv1d for B and C channels (shared per group)
    4. Runs SSM state update with the conv outputs
    """
    pid_d = tl.program_id(0)  # dim tile
    pid_b = tl.program_id(1)  # batch
    pid_h = tl.program_id(2)  # head

    # Load slot indices for cache access
    conv_slot = tl.load(conv_slot_idx_ptr + pid_b).to(tl.int64)
    ssm_slot = tl.load(ssm_slot_idx_ptr + pid_b).to(tl.int64)

    group_idx = pid_h // nheads_per_group

    offs_d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    offs_n = tl.arange(0, BLOCK_DSTATE)
    mask_d = offs_d < dim
    mask_n = offs_n < dstate
    mask_dn = mask_d[:, None] & mask_n[None, :]

    # ---------------------------------------------------------------
    # Load SSM scalars early (dt, A, D are tiny — a few bytes each).
    # Computing softplus and dA before the state prefetch means the
    # scalar ALU work completes before we issue the large 16KB state
    # load, giving the memory controller maximum time to fetch it.
    # ---------------------------------------------------------------
    dt_val = tl.load(dt_ptr + pid_b * nheads + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    dt_val = softplus(dt_val + dt_bias_val)

    A_val = tl.load(A_ptr + pid_h).to(tl.float32)
    dA = tl.exp(A_val * dt_val)
    D_val = tl.load(D_ptr + pid_h).to(tl.float32)

    # ---------------------------------------------------------------
    # Prefetch SSM state after scalar computation.
    # The load is issued early so the GPU can fetch the 16KB state
    # tile while the warp executes the conv1d sections below.
    # This overlaps memory latency with compute.
    # ---------------------------------------------------------------
    state_ptrs = (
        ssm_state_ptr
        + ssm_slot * stride_ss_b
        + pid_h * stride_ss_h
        + offs_d[:, None] * stride_ss_d
        + offs_n[None, :] * stride_ss_n
    )
    state_prefetch = tl.load(state_ptrs, mask=mask_dn, other=0.0, eviction_policy="evict_last")

    # ---------------------------------------------------------------
    # Step 1: Conv1d update for hidden channels (slot-indexed)
    # Channel index in conv_input: hidden_channel = head_idx * dim + dim_offset
    # ---------------------------------------------------------------
    hidden_channels = pid_h * dim + offs_d  # [BLOCK_DIM]
    mask_hc = hidden_channels < intermediate_size

    conv_out_hidden = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    # Simple depthwise conv1d for kernel_width=4
    # Use conv_slot instead of pid_b for cache access
    # evict_last on conv state: large tensor, evict after read to preserve L2 for weights
    for k in range(kernel_width - 1):  # k=0,1,2
        state_val = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + hidden_channels * stride_cs_d
            + k * stride_cs_w,
            mask=mask_hc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        w_val = tl.load(
            conv_weight_ptr + hidden_channels * stride_cw_d + k * stride_cw_w,
            mask=mask_hc,
            other=0.0,
        ).to(tl.float32)
        conv_out_hidden += state_val * w_val

    # Current input contribution
    x_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + hidden_channels * stride_ci_d,
        mask=mask_hc,
        other=0.0,
    ).to(tl.float32)
    w_last = tl.load(
        conv_weight_ptr + hidden_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_hc,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(conv_bias_ptr + hidden_channels, mask=mask_hc, other=0.0).to(tl.float32)
    conv_out_hidden += x_in * w_last + bias

    # Update conv state: shift left, append new input
    for k in range(kernel_width - 2):
        old_val = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + hidden_channels * stride_cs_d
            + (k + 1) * stride_cs_w,
            mask=mask_hc,
            other=0.0,
        )
        tl.store(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + hidden_channels * stride_cs_d
            + k * stride_cs_w,
            old_val,
            mask=mask_hc,
        )
    tl.store(
        conv_state_ptr
        + conv_slot * stride_cs_b
        + hidden_channels * stride_cs_d
        + (kernel_width - 2) * stride_cs_w,
        x_in.to(conv_state_ptr.dtype.element_ty),
        mask=mask_hc,
    )

    # Apply SiLU: x * sigmoid(x)
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_out_hidden))
    x_hidden = conv_out_hidden * sigmoid_val  # [BLOCK_DIM], this is the SSM input

    # ---------------------------------------------------------------
    # Step 2: Conv1d for B and C channels (shared per group)
    # B channels: offset = intermediate_size + group_idx * dstate
    # C channels: offset = intermediate_size + ngroups * dstate + group_idx * dstate
    # ---------------------------------------------------------------
    B_offset = intermediate_size + group_idx * dstate
    C_offset = intermediate_size + ngroups * dstate + group_idx * dstate

    B_vals = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)
    C_vals = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

    B_channels = B_offset + offs_n
    C_channels = C_offset + offs_n
    mask_bc = (B_channels < conv_dim) & mask_n

    for k in range(kernel_width - 1):
        b_state = tl.load(
            conv_state_ptr + conv_slot * stride_cs_b + B_channels * stride_cs_d + k * stride_cs_w,
            mask=mask_bc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        b_w = tl.load(
            conv_weight_ptr + B_channels * stride_cw_d + k * stride_cw_w, mask=mask_bc, other=0.0
        ).to(tl.float32)
        B_vals += b_state * b_w

        c_state = tl.load(
            conv_state_ptr + conv_slot * stride_cs_b + C_channels * stride_cs_d + k * stride_cs_w,
            mask=mask_bc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        c_w = tl.load(
            conv_weight_ptr + C_channels * stride_cw_d + k * stride_cw_w, mask=mask_bc, other=0.0
        ).to(tl.float32)
        C_vals += c_state * c_w

    # Current input for B, C
    b_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + B_channels * stride_ci_d, mask=mask_bc, other=0.0
    ).to(tl.float32)
    b_w_last = tl.load(
        conv_weight_ptr + B_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    b_bias = tl.load(conv_bias_ptr + B_channels, mask=mask_bc, other=0.0).to(tl.float32)
    B_vals += b_in * b_w_last + b_bias

    c_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + C_channels * stride_ci_d, mask=mask_bc, other=0.0
    ).to(tl.float32)
    c_w_last = tl.load(
        conv_weight_ptr + C_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    c_bias = tl.load(conv_bias_ptr + C_channels, mask=mask_bc, other=0.0).to(tl.float32)
    C_vals += c_in * c_w_last + c_bias

    # Apply SiLU to B and C
    B_vals = B_vals * (1.0 / (1.0 + tl.exp(-B_vals)))
    C_vals = C_vals * (1.0 / (1.0 + tl.exp(-C_vals)))

    # ---------------------------------------------------------------
    # Step 3: SSM state update
    # state = state * exp(A * dt) + x * dt * B
    # out = sum(state * C) + x * D
    # dt, A, D, dA were pre-computed at the top of the kernel.
    # ---------------------------------------------------------------
    # Use the prefetched SSM state (issued before conv computation)
    state = state_prefetch.to(tl.float32)

    # State update
    dB = B_vals[None, :] * dt_val  # [1, BLOCK_DSTATE]
    state = state * dA + dB * x_hidden[:, None]

    # Output: sum(state * C) + x * D
    out = tl.sum(state * C_vals[None, :], axis=1)
    out += x_hidden * D_val

    # Store output and SSM state
    tl.store(
        ssm_out_ptr + pid_b * stride_so_b + pid_h * stride_so_h + offs_d * stride_so_d,
        out.to(tl.bfloat16),
        mask=mask_d,
    )
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask_dn)

    # Update B/C conv state AFTER SSM output stores.
    # Moving the B/C state shift here (instead of before SiLU) allows the SSM
    # computation and output store to execute while the B/C writes are pending in
    # the store buffer, overlapping the B/C write latency with SSM compute.
    # Only head-0 per group writes (race-free guard).
    if pid_h % nheads_per_group == 0:
        for k in range(kernel_width - 2):
            old_b = tl.load(
                conv_state_ptr
                + conv_slot * stride_cs_b
                + B_channels * stride_cs_d
                + (k + 1) * stride_cs_w,
                mask=mask_bc,
                other=0.0,
            )
            tl.store(
                conv_state_ptr
                + conv_slot * stride_cs_b
                + B_channels * stride_cs_d
                + k * stride_cs_w,
                old_b,
                mask=mask_bc,
            )
        tl.store(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + B_channels * stride_cs_d
            + (kernel_width - 2) * stride_cs_w,
            b_in.to(conv_state_ptr.dtype.element_ty),
            mask=mask_bc,
        )

        for k in range(kernel_width - 2):
            old_c = tl.load(
                conv_state_ptr
                + conv_slot * stride_cs_b
                + C_channels * stride_cs_d
                + (k + 1) * stride_cs_w,
                mask=mask_bc,
                other=0.0,
            )
            tl.store(
                conv_state_ptr
                + conv_slot * stride_cs_b
                + C_channels * stride_cs_d
                + k * stride_cs_w,
                old_c,
                mask=mask_bc,
            )
        tl.store(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + C_channels * stride_cs_d
            + (kernel_width - 2) * stride_cs_w,
            c_in.to(conv_state_ptr.dtype.element_ty),
            mask=mask_bc,
        )


@triton.jit()
def _fused_conv_ssm_kernel_bc_buf(
    # Conv inputs (hidden channels only)
    conv_input_ptr,  # [batch, conv_dim]
    conv_state_ptr,  # [max_batch, conv_dim, kernel_width-1]
    conv_weight_ptr,  # [conv_dim, kernel_width]
    conv_bias_ptr,  # [conv_dim]
    # Pre-computed B/C values (from _bc_conv_compute_kernel)
    bc_buf_ptr,  # [batch, 2, ngroups, dstate] — bf16
    # SSM inputs
    dt_ptr,  # [batch, nheads]
    dt_bias_ptr,  # [nheads]
    A_ptr,  # [nheads]
    D_ptr,  # [nheads]
    ssm_state_ptr,  # [max_batch, nheads, dim, dstate]
    # Slot indices
    conv_slot_idx_ptr,  # [batch]
    ssm_slot_idx_ptr,  # [batch]
    # Output
    ssm_out_ptr,  # [batch, nheads, dim]
    # Dimensions
    batch,
    conv_dim,
    intermediate_size,
    nheads,
    dim,
    dstate,
    ngroups,
    nheads_per_group,
    kernel_width: tl.constexpr,
    # Strides
    stride_ci_b,
    stride_ci_d,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    stride_cw_d,
    stride_cw_w,
    stride_ss_b,
    stride_ss_h,
    stride_ss_d,
    stride_ss_n,
    stride_so_b,
    stride_so_h,
    stride_so_d,
    # Meta
    BLOCK_DIM: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,
):
    """Fused conv1d (hidden only) + SiLU + SSM, reading B/C from bc_buf.

    Grid: (cdiv(dim, BLOCK_DIM), batch, nheads)

    This is the second kernel of the two-kernel approach. B/C conv values are
    pre-computed in _bc_conv_compute_kernel (one CTA per group, race-free) and
    stored in bc_buf. This kernel handles hidden channel conv + SSM update only,
    eliminating the redundant 7/8 B/C computation across heads in a group.
    """
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    conv_slot = tl.load(conv_slot_idx_ptr + pid_b).to(tl.int64)
    ssm_slot = tl.load(ssm_slot_idx_ptr + pid_b).to(tl.int64)

    group_idx = pid_h // nheads_per_group

    offs_d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    offs_n = tl.arange(0, BLOCK_DSTATE)
    mask_d = offs_d < dim
    mask_n = offs_n < dstate
    mask_dn = mask_d[:, None] & mask_n[None, :]

    # ---------------------------------------------------------------
    # Step 1: Conv1d update for hidden channels only
    # ---------------------------------------------------------------
    hidden_channels = pid_h * dim + offs_d
    mask_hc = hidden_channels < intermediate_size

    conv_out_hidden = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    for k in range(kernel_width - 1):
        state_val = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + hidden_channels * stride_cs_d
            + k * stride_cs_w,
            mask=mask_hc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        w_val = tl.load(
            conv_weight_ptr + hidden_channels * stride_cw_d + k * stride_cw_w,
            mask=mask_hc,
            other=0.0,
        ).to(tl.float32)
        conv_out_hidden += state_val * w_val

    x_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + hidden_channels * stride_ci_d,
        mask=mask_hc,
        other=0.0,
    ).to(tl.float32)
    w_last = tl.load(
        conv_weight_ptr + hidden_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_hc,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(conv_bias_ptr + hidden_channels, mask=mask_hc, other=0.0).to(tl.float32)
    conv_out_hidden += x_in * w_last + bias

    # Update hidden conv state
    for k in range(kernel_width - 2):
        old_val = tl.load(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + hidden_channels * stride_cs_d
            + (k + 1) * stride_cs_w,
            mask=mask_hc,
            other=0.0,
        )
        tl.store(
            conv_state_ptr
            + conv_slot * stride_cs_b
            + hidden_channels * stride_cs_d
            + k * stride_cs_w,
            old_val,
            mask=mask_hc,
        )
    tl.store(
        conv_state_ptr
        + conv_slot * stride_cs_b
        + hidden_channels * stride_cs_d
        + (kernel_width - 2) * stride_cs_w,
        x_in.to(conv_state_ptr.dtype.element_ty),
        mask=mask_hc,
    )

    # Apply SiLU
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_out_hidden))
    x_hidden = conv_out_hidden * sigmoid_val

    # ---------------------------------------------------------------
    # Step 2: Load pre-computed B/C from bc_buf [batch, 2, ngroups, dstate]
    # ---------------------------------------------------------------
    b_buf_offs = (
        pid_b * (2 * ngroups * dstate) + 0 * (ngroups * dstate) + group_idx * dstate + offs_n
    )
    c_buf_offs = (
        pid_b * (2 * ngroups * dstate) + 1 * (ngroups * dstate) + group_idx * dstate + offs_n
    )
    B_vals = tl.load(bc_buf_ptr + b_buf_offs, mask=mask_n, other=0.0).to(tl.float32)
    C_vals = tl.load(bc_buf_ptr + c_buf_offs, mask=mask_n, other=0.0).to(tl.float32)

    # ---------------------------------------------------------------
    # Step 3: SSM state update
    # ---------------------------------------------------------------
    dt_val = tl.load(dt_ptr + pid_b * nheads + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    dt_val = softplus(dt_val + dt_bias_val)

    A_val = tl.load(A_ptr + pid_h).to(tl.float32)
    dA = tl.exp(A_val * dt_val)

    state_ptrs = (
        ssm_state_ptr
        + ssm_slot * stride_ss_b
        + pid_h * stride_ss_h
        + offs_d[:, None] * stride_ss_d
        + offs_n[None, :] * stride_ss_n
    )
    state = tl.load(state_ptrs, mask=mask_dn, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )

    dB = B_vals[None, :] * dt_val
    state = state * dA + dB * x_hidden[:, None]

    out = tl.sum(state * C_vals[None, :], axis=1)
    D_val = tl.load(D_ptr + pid_h).to(tl.float32)
    out += x_hidden * D_val

    tl.store(
        ssm_out_ptr + pid_b * stride_so_b + pid_h * stride_so_h + offs_d * stride_so_d,
        out.to(tl.bfloat16),
        mask=mask_d,
    )
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask_dn)


def fused_conv_ssm_decode_two_kernel(
    conv_input: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_slot_idx: torch.Tensor,
    ssm_slot_idx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Two-kernel launch: B/C precompute + SSM-only main kernel.

    Kernel 1: _bc_conv_compute_kernel [batch, ngroups] — race-free B/C conv + state update
    Kernel 2: _fused_conv_ssm_kernel_bc_buf [1, batch, nheads] — hidden conv + SSM from bc_buf

    Args: same as fused_conv_ssm_decode
    """
    batch, conv_dim = conv_input.shape
    _, nheads, dim, dstate = ssm_state.shape
    kernel_width = conv_state.shape[-1] + 1
    intermediate_size = nheads * dim
    ngroups = (conv_dim - intermediate_size) // (2 * dstate)
    nheads_per_group = nheads // ngroups

    BLOCK_DIM = max(triton.next_power_of_2(dim), 16)
    BLOCK_DSTATE = triton.next_power_of_2(dstate)

    conv_input = conv_input.contiguous()
    conv_state = conv_state.contiguous()
    conv_weight = conv_weight.contiguous()

    # Allocate B/C buffer: [batch, 2, ngroups, dstate] in bf16
    bc_buf = torch.empty(batch, 2, ngroups, dstate, dtype=torch.bfloat16, device=conv_input.device)

    # Kernel 1: compute B/C values + update B/C conv state (race-free)
    grid_bc = (batch, ngroups)
    _bc_conv_compute_kernel[grid_bc](
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        conv_slot_idx,
        bc_buf,
        batch,
        ngroups,
        dstate,
        intermediate_size,
        conv_dim,
        kernel_width,
        conv_input.stride(0),
        conv_input.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=4,
    )

    # Kernel 2: hidden conv + SSM using pre-computed B/C
    def grid_main(META):
        return (triton.cdiv(dim, META["BLOCK_DIM"]), batch, nheads)

    _fused_conv_ssm_kernel_bc_buf[grid_main](
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        bc_buf,
        dt,
        dt_bias,
        A,
        D,
        ssm_state,
        conv_slot_idx,
        ssm_slot_idx,
        out,
        batch,
        conv_dim,
        intermediate_size,
        nheads,
        dim,
        dstate,
        ngroups,
        nheads_per_group,
        kernel_width,
        conv_input.stride(0),
        conv_input.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=8,
    )


def fused_conv_ssm_decode(
    conv_input: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
    ssm_state: torch.Tensor,
    conv_slot_idx: torch.Tensor,
    ssm_slot_idx: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Launch _fused_conv_ssm_kernel for decode (T=1) tokens.

    Args:
        conv_input:   [batch, conv_dim]  raw conv input (before split), bf16
        conv_state:   [max_batch, conv_dim, kernel_width-1]  slot-indexed
        conv_weight:  [conv_dim, kernel_width]
        conv_bias:    [conv_dim]
        dt:           [batch, nheads]
        dt_bias:      [nheads]
        A:            [nheads]
        D:            [nheads]
        ssm_state:    [max_batch, nheads, dim, dstate]  slot-indexed
        conv_slot_idx:[batch] int32
        ssm_slot_idx: [batch] int32
        out:          [batch, nheads, dim] preallocated output (written in bf16)
    """
    batch, conv_dim = conv_input.shape
    _, nheads, dim, dstate = ssm_state.shape
    kernel_width = conv_state.shape[-1] + 1
    intermediate_size = nheads * dim
    ngroups = (conv_dim - intermediate_size) // (2 * dstate)
    nheads_per_group = nheads // ngroups

    BLOCK_DIM = max(triton.next_power_of_2(dim), 16)
    BLOCK_DSTATE = triton.next_power_of_2(dstate)
    # With bf16 SSM state (production default), num_warps=4 is optimal:
    # halved register pressure allows better occupancy vs num_warps=8.
    num_warps = 4

    def grid(META):
        return (triton.cdiv(dim, META["BLOCK_DIM"]), batch, nheads)

    conv_input = conv_input.contiguous()
    conv_state = conv_state.contiguous()
    conv_weight = conv_weight.contiguous()

    _fused_conv_ssm_kernel[grid](
        conv_input,
        conv_state,
        conv_weight,
        conv_bias,
        dt,
        dt_bias,
        A,
        D,
        ssm_state,
        conv_slot_idx,
        ssm_slot_idx,
        out,
        batch,
        conv_dim,
        intermediate_size,
        nheads,
        dim,
        dstate,
        ngroups,
        nheads_per_group,
        kernel_width,
        conv_input.stride(0),
        conv_input.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        conv_weight.stride(0),
        conv_weight.stride(1),
        ssm_state.stride(0),
        ssm_state.stride(1),
        ssm_state.stride(2),
        ssm_state.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=num_warps,
    )
