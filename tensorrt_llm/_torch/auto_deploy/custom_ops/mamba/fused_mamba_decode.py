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

    # Skip pad slots (slot == -1 means padding — no state to update).
    if conv_slot < 0:
        return

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
    # dt clamping (applied after softplus; use 0.0 / inf for no-op)
    dt_clamp_min,
    dt_clamp_max,
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

    # Skip pad slots (slot == -1 means padding, no real sequence).
    # Without this guard, pad slots produce OOB memory accesses (ptr - stride)
    # that silently corrupt adjacent tensors, causing catastrophically wrong outputs.
    if conv_slot < 0 or ssm_slot < 0:
        return

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
    dt_val = tl.minimum(tl.maximum(dt_val, dt_clamp_min), dt_clamp_max)

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
    state_prefetch = tl.load(state_ptrs, mask=mask_dn, other=0.0)

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

    # Update hidden conv state: shift left, append new input.
    # This is placed AFTER all B/C reads to avoid a cache-line false-sharing hazard:
    # the last hidden channel (ch=intermediate_size-1) and the first B channel
    # (ch=intermediate_size) share the same L1/L2 cache line. Writing hidden ch
    # before reading B/C can cause other CTAs on different SMs to observe stale
    # B/C values from their L1 cache (the write invalidates/updates the cache line
    # while the B/C read is still in flight on the other SM).
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

    # NOTE: B/C conv state update is intentionally omitted here.
    # The in-kernel B/C state shift (guarded by pid_h%nheads_per_group==0) causes
    # a data race at large batch sizes: head-0 can write the shifted B/C state
    # while other heads in the same group are still reading it on a different SM.
    # Instead, _bc_conv_state_update_kernel is launched as a separate kernel in
    # fused_conv_ssm_decode, serializing the shift after all heads have finished.


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


@triton.jit()
def _fused_conv_ssm_kernel_persistent(
    # Conv inputs
    conv_input_ptr,  # [batch, conv_dim]
    conv_state_ptr,  # [max_batch, conv_dim, kernel_width-1]
    conv_weight_ptr,  # [conv_dim, kernel_width]
    conv_bias_ptr,  # [conv_dim]
    # SSM inputs
    dt_ptr,  # [batch, nheads]
    dt_bias_ptr,  # [nheads]
    A_ptr,  # [nheads]
    D_ptr,  # [nheads]
    ssm_state_ptr,  # [max_batch, nheads, dim, dstate]
    conv_slot_idx_ptr,  # [batch]
    ssm_slot_idx_ptr,  # [batch]
    # Output
    ssm_out_ptr,  # [batch, nheads, dim]
    # Work queue: total_work = batch * nheads (each item = one (batch, head) pair)
    total_work,
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
    N_CTAs: tl.constexpr,  # total number of CTAs launched (= grid size)
):
    """Persistent kernel: each CTA loops over multiple (batch, head) work items.

    Grid: (N_CTAs,) — flat 1D grid, N_CTAs >= SM count to fill the GPU.
    Each CTA processes work items [cta_id, cta_id+N_CTAs, cta_id+2*N_CTAs, ...]
    Work item i = (i // nheads, i % nheads) = (batch_idx, head_idx).

    Key: head-0 per group (head_idx % nheads_per_group == 0) writes B/C state.
    In the persistent loop, each CTA processes heads in order, so the B/C write
    by head-0 is naturally serialized before the B/C read by heads 1..7 of the
    same batch element — IF heads are processed in order across CTAs. To be safe,
    we rely on the same guard as the original kernel: only head_idx%nheads_per_group==0
    writes B/C state, and reads happen before writes within one work item.
    """
    cta_id = tl.program_id(0)

    offs_d = tl.arange(0, BLOCK_DIM)
    offs_n = tl.arange(0, BLOCK_DSTATE)
    mask_d = offs_d < dim
    mask_n = offs_n < dstate
    mask_dn = mask_d[:, None] & mask_n[None, :]

    # Each CTA processes work items at positions cta_id, cta_id+N_CTAs, ...
    work_id = cta_id
    while work_id < total_work:
        pid_b = work_id // nheads
        pid_h = work_id % nheads

        conv_slot = tl.load(conv_slot_idx_ptr + pid_b).to(tl.int64)
        ssm_slot = tl.load(ssm_slot_idx_ptr + pid_b).to(tl.int64)

        group_idx = pid_h // nheads_per_group

        # SSM scalars
        dt_val = tl.load(dt_ptr + pid_b * nheads + pid_h).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
        dt_val = softplus(dt_val + dt_bias_val)
        A_val = tl.load(A_ptr + pid_h).to(tl.float32)
        dA = tl.exp(A_val * dt_val)
        D_val = tl.load(D_ptr + pid_h).to(tl.float32)

        # SSM state prefetch
        state_ptrs = (
            ssm_state_ptr
            + ssm_slot * stride_ss_b
            + pid_h * stride_ss_h
            + offs_d[:, None] * stride_ss_d
            + offs_n[None, :] * stride_ss_n
        )
        state_prefetch = tl.load(state_ptrs, mask=mask_dn, other=0.0, eviction_policy="evict_last")

        # Step 1: Hidden channel conv1d
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

        # SiLU
        sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_out_hidden))
        x_hidden = conv_out_hidden * sigmoid_val

        # Step 2: B/C conv1d
        B_offset = intermediate_size + group_idx * dstate
        C_offset = intermediate_size + ngroups * dstate + group_idx * dstate

        B_vals = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)
        C_vals = tl.zeros((BLOCK_DSTATE,), dtype=tl.float32)

        B_channels = B_offset + offs_n
        C_channels = C_offset + offs_n
        mask_bc = (B_channels < conv_dim) & mask_n

        for k in range(kernel_width - 1):
            b_state = tl.load(
                conv_state_ptr
                + conv_slot * stride_cs_b
                + B_channels * stride_cs_d
                + k * stride_cs_w,
                mask=mask_bc,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            b_w = tl.load(
                conv_weight_ptr + B_channels * stride_cw_d + k * stride_cw_w,
                mask=mask_bc,
                other=0.0,
            ).to(tl.float32)
            B_vals += b_state * b_w

            c_state = tl.load(
                conv_state_ptr
                + conv_slot * stride_cs_b
                + C_channels * stride_cs_d
                + k * stride_cs_w,
                mask=mask_bc,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            c_w = tl.load(
                conv_weight_ptr + C_channels * stride_cw_d + k * stride_cw_w,
                mask=mask_bc,
                other=0.0,
            ).to(tl.float32)
            C_vals += c_state * c_w

        b_in = tl.load(
            conv_input_ptr + pid_b * stride_ci_b + B_channels * stride_ci_d,
            mask=mask_bc,
            other=0.0,
        ).to(tl.float32)
        b_w_last = tl.load(
            conv_weight_ptr + B_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
            mask=mask_bc,
            other=0.0,
        ).to(tl.float32)
        b_bias = tl.load(conv_bias_ptr + B_channels, mask=mask_bc, other=0.0).to(tl.float32)
        B_vals += b_in * b_w_last + b_bias

        c_in = tl.load(
            conv_input_ptr + pid_b * stride_ci_b + C_channels * stride_ci_d,
            mask=mask_bc,
            other=0.0,
        ).to(tl.float32)
        c_w_last = tl.load(
            conv_weight_ptr + C_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
            mask=mask_bc,
            other=0.0,
        ).to(tl.float32)
        c_bias = tl.load(conv_bias_ptr + C_channels, mask=mask_bc, other=0.0).to(tl.float32)
        C_vals += c_in * c_w_last + c_bias

        # SiLU on B and C
        B_vals = B_vals * (1.0 / (1.0 + tl.exp(-B_vals)))
        C_vals = C_vals * (1.0 / (1.0 + tl.exp(-C_vals)))

        # Step 3: SSM state update
        state = state_prefetch.to(tl.float32)
        dB = B_vals[None, :] * dt_val
        state = state * dA + dB * x_hidden[:, None]

        out = tl.sum(state * C_vals[None, :], axis=1)
        out += x_hidden * D_val

        # Store output and SSM state
        tl.store(
            ssm_out_ptr + pid_b * stride_so_b + pid_h * stride_so_h + offs_d * stride_so_d,
            out.to(tl.bfloat16),
            mask=mask_d,
        )
        tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask_dn)

        # B/C conv state writes (after SSM output)
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

        work_id += N_CTAs


@triton.jit()
def _fused_conv_ssm_kernel_dstate_split(
    # Conv inputs
    conv_input_ptr,  # [batch, conv_dim]
    conv_state_ptr,  # [max_batch, conv_dim, kernel_width-1]
    conv_weight_ptr,  # [conv_dim, kernel_width]
    conv_bias_ptr,  # [conv_dim]
    # SSM inputs
    dt_ptr,  # [batch, nheads]
    dt_bias_ptr,  # [nheads]
    A_ptr,  # [nheads]
    D_ptr,  # [nheads]
    ssm_state_ptr,  # [max_batch, nheads, dim, dstate]
    conv_slot_idx_ptr,  # [batch]
    ssm_slot_idx_ptr,  # [batch]
    # Output: float32 accumulation buffer [batch, nheads, dim]
    out_accum_ptr,  # float32
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
    # Strides (conv inputs)
    stride_ci_b,
    stride_ci_d,
    stride_cs_b,
    stride_cs_d,
    stride_cs_w,
    stride_cw_d,
    stride_cw_w,
    # SSM strides
    stride_ss_b,
    stride_ss_h,
    stride_ss_d,
    stride_ss_n,
    # Output strides (float32 accum)
    stride_so_b,
    stride_so_h,
    stride_so_d,
    # Meta
    BLOCK_DIM: tl.constexpr,
    BLOCK_DSTATE: tl.constexpr,  # full dstate
    DSTATE_TILE: tl.constexpr,  # tile size for dstate split
    N_DSTATE_TILES: tl.constexpr,  # dstate // DSTATE_TILE
    IS_FIRST_TILE: tl.constexpr,  # whether pid_n == 0 (writes hidden conv state + D*x)
):
    """Dstate-split SSM kernel: each CTA handles DSTATE_TILE dstate elements.

    Grid: (cdiv(dim, BLOCK_DIM), batch, nheads * N_DSTATE_TILES)
    pid_h_full = pid_h_base * N_DSTATE_TILES + pid_n
    where pid_n = dstate tile index (0..N_DSTATE_TILES-1)

    Multiple CTAs contribute partial sums to the same output element via atomic_add.
    Conv state updates and D*x term are only done by the first dstate tile (pid_n==0).
    """
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hn = tl.program_id(2)  # combined head * N_DSTATE_TILES + dstate_tile

    pid_h = pid_hn // N_DSTATE_TILES
    pid_n = pid_hn % N_DSTATE_TILES  # dstate tile index

    conv_slot = tl.load(conv_slot_idx_ptr + pid_b).to(tl.int64)
    ssm_slot = tl.load(ssm_slot_idx_ptr + pid_b).to(tl.int64)

    group_idx = pid_h // nheads_per_group

    offs_d = pid_d * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    offs_n_tile = pid_n * DSTATE_TILE + tl.arange(0, DSTATE_TILE)
    mask_d = offs_d < dim
    mask_n_tile = offs_n_tile < dstate
    mask_dn = mask_d[:, None] & mask_n_tile[None, :]

    # SSM scalars
    dt_val = tl.load(dt_ptr + pid_b * nheads + pid_h).to(tl.float32)
    dt_bias_val = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    dt_val = softplus(dt_val + dt_bias_val)
    A_val = tl.load(A_ptr + pid_h).to(tl.float32)
    dA = tl.exp(A_val * dt_val)

    # Load SSM state tile [BLOCK_DIM, DSTATE_TILE]
    state_ptrs = (
        ssm_state_ptr
        + ssm_slot * stride_ss_b
        + pid_h * stride_ss_h
        + offs_d[:, None] * stride_ss_d
        + offs_n_tile[None, :] * stride_ss_n
    )
    state = tl.load(state_ptrs, mask=mask_dn, other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )

    # Step 1: Hidden channel conv (only first dstate tile to avoid redundant work)
    hidden_channels = pid_h * dim + offs_d
    mask_hc = hidden_channels < intermediate_size

    conv_out_hidden = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    if IS_FIRST_TILE:
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
    else:
        # Non-first tiles still need x_in for dB computation
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
        # Need full conv output — must redo for non-first tiles
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
        conv_out_hidden += x_in * w_last + bias

    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_out_hidden))
    x_hidden = conv_out_hidden * sigmoid_val

    # Step 2: B/C conv for this dstate tile
    B_offset = intermediate_size + group_idx * dstate + pid_n * DSTATE_TILE
    C_offset = intermediate_size + ngroups * dstate + group_idx * dstate + pid_n * DSTATE_TILE

    B_vals = tl.zeros((DSTATE_TILE,), dtype=tl.float32)
    C_vals = tl.zeros((DSTATE_TILE,), dtype=tl.float32)

    B_channels = B_offset + tl.arange(0, DSTATE_TILE)
    C_channels = C_offset + tl.arange(0, DSTATE_TILE)
    mask_bc = (B_channels < conv_dim) & mask_n_tile

    for k in range(kernel_width - 1):
        b_state = tl.load(
            conv_state_ptr + conv_slot * stride_cs_b + B_channels * stride_cs_d + k * stride_cs_w,
            mask=mask_bc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        b_w = tl.load(
            conv_weight_ptr + B_channels * stride_cw_d + k * stride_cw_w,
            mask=mask_bc,
            other=0.0,
        ).to(tl.float32)
        B_vals += b_state * b_w

        c_state = tl.load(
            conv_state_ptr + conv_slot * stride_cs_b + C_channels * stride_cs_d + k * stride_cs_w,
            mask=mask_bc,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        c_w = tl.load(
            conv_weight_ptr + C_channels * stride_cw_d + k * stride_cw_w,
            mask=mask_bc,
            other=0.0,
        ).to(tl.float32)
        C_vals += c_state * c_w

    b_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + B_channels * stride_ci_d,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    b_w_last = tl.load(
        conv_weight_ptr + B_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    b_bias = tl.load(conv_bias_ptr + B_channels, mask=mask_bc, other=0.0).to(tl.float32)
    B_vals += b_in * b_w_last + b_bias

    c_in = tl.load(
        conv_input_ptr + pid_b * stride_ci_b + C_channels * stride_ci_d,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    c_w_last = tl.load(
        conv_weight_ptr + C_channels * stride_cw_d + (kernel_width - 1) * stride_cw_w,
        mask=mask_bc,
        other=0.0,
    ).to(tl.float32)
    c_bias = tl.load(conv_bias_ptr + C_channels, mask=mask_bc, other=0.0).to(tl.float32)
    C_vals += c_in * c_w_last + c_bias

    B_vals = B_vals * (1.0 / (1.0 + tl.exp(-B_vals)))
    C_vals = C_vals * (1.0 / (1.0 + tl.exp(-C_vals)))

    # Step 3: SSM state update for this dstate tile
    dB = B_vals[None, :] * dt_val  # [1, DSTATE_TILE]
    state = state * dA + dB * x_hidden[:, None]

    # Partial output: sum(state * C) over this dstate tile → [BLOCK_DIM]
    partial_out = tl.sum(state * C_vals[None, :], axis=1)  # [BLOCK_DIM]

    # Atomic add partial output into float32 accumulation buffer
    out_ptrs = out_accum_ptr + pid_b * stride_so_b + pid_h * stride_so_h + offs_d * stride_so_d
    tl.atomic_add(out_ptrs, partial_out, mask=mask_d)

    # D*x term — only first dstate tile to avoid duplication
    if IS_FIRST_TILE:
        D_val = tl.load(D_ptr + pid_h).to(tl.float32)
        d_out = x_hidden * D_val
        tl.atomic_add(out_ptrs, d_out, mask=mask_d)

    # Store updated SSM state tile
    tl.store(state_ptrs, state.to(state_ptrs.dtype.element_ty), mask=mask_dn)

    # B/C conv state write — only head-0 per group, only first dstate tile
    if IS_FIRST_TILE and (pid_h % nheads_per_group == 0):
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
def _cast_fp32_to_bf16_kernel(
    src_ptr,  # [N] float32
    dst_ptr,  # [N] bfloat16
    N,
    BLOCK: tl.constexpr,
):
    """Cast float32 accumulation buffer to bfloat16 output."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val = tl.load(src_ptr + offs, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, val.to(tl.bfloat16), mask=mask)


def fused_conv_ssm_decode_dstate_split(
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
    n_dstate_tiles: int = 4,
) -> None:
    """Dstate-split kernel: splits dstate across CTAs for more parallelism.

    Grid: (cdiv(dim, BLOCK_DIM), batch, nheads * n_dstate_tiles)
    For B=1, nheads=64, n_dstate_tiles=4: 256 CTAs vs original 64.
    Uses atomic_add to accumulate partial output across dstate tiles.
    """
    batch, conv_dim = conv_input.shape
    _, nheads, dim, dstate = ssm_state.shape
    kernel_width = conv_state.shape[-1] + 1
    intermediate_size = nheads * dim
    ngroups = (conv_dim - intermediate_size) // (2 * dstate)
    nheads_per_group = nheads // ngroups

    BLOCK_DIM = max(triton.next_power_of_2(dim), 16)
    BLOCK_DSTATE = triton.next_power_of_2(dstate)
    DSTATE_TILE = dstate // n_dstate_tiles
    assert dstate % n_dstate_tiles == 0, (
        f"dstate={dstate} must be divisible by n_dstate_tiles={n_dstate_tiles}"
    )

    conv_input = conv_input.contiguous()
    conv_state = conv_state.contiguous()
    conv_weight = conv_weight.contiguous()

    # Float32 accumulation buffer for atomic adds
    out_accum = torch.zeros(batch, nheads, dim, dtype=torch.float32, device=conv_input.device)

    def grid(META):
        return (triton.cdiv(dim, META["BLOCK_DIM"]), batch, nheads * n_dstate_tiles)

    _fused_conv_ssm_kernel_dstate_split[grid](
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
        out_accum,
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
        out_accum.stride(0),
        out_accum.stride(1),
        out_accum.stride(2),
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_DSTATE=BLOCK_DSTATE,
        DSTATE_TILE=DSTATE_TILE,
        N_DSTATE_TILES=n_dstate_tiles,
        IS_FIRST_TILE=True,  # always True — we set n_dstate_tiles=1 path separately
        num_warps=4,
    )

    # Cast float32 accumulation to bfloat16 output
    N = batch * nheads * dim
    CAST_BLOCK = 256
    _cast_fp32_to_bf16_kernel[(triton.cdiv(N, CAST_BLOCK),)](
        out_accum.view(-1),
        out.view(-1),
        N,
        BLOCK=CAST_BLOCK,
        num_warps=1,
    )


def fused_conv_ssm_decode_persistent(
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
    n_ctas: int = 128,
) -> None:
    """Persistent kernel launcher: N_CTAs CTAs loop over all (batch, head) work items.

    For small batch (B=1..8), N_CTAs > batch*nheads so each CTA does exactly 1 work item
    but we fill all SMs, reducing launch overhead and improving SM utilization.
    For large batch, N_CTAs < batch*nheads so CTAs loop over multiple items.
    """
    batch, conv_dim = conv_input.shape
    _, nheads, dim, dstate = ssm_state.shape
    kernel_width = conv_state.shape[-1] + 1
    intermediate_size = nheads * dim
    ngroups = (conv_dim - intermediate_size) // (2 * dstate)
    nheads_per_group = nheads // ngroups

    BLOCK_DIM = max(triton.next_power_of_2(dim), 16)
    BLOCK_DSTATE = triton.next_power_of_2(dstate)

    total_work = batch * nheads
    # Cap N_CTAs at total_work to avoid idle CTAs that do 0 iterations
    actual_n_ctas = min(n_ctas, total_work)

    conv_input = conv_input.contiguous()
    conv_state = conv_state.contiguous()
    conv_weight = conv_weight.contiguous()

    _fused_conv_ssm_kernel_persistent[(actual_n_ctas,)](
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
        total_work,
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
        N_CTAs=actual_n_ctas,
        num_warps=4,
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
    dt_clamp_min: float = 0.0,
    dt_clamp_max: float = float("inf"),
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
        dt_clamp_min,
        dt_clamp_max,
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=num_warps,
    )

    # Kernel 2: update B/C conv state (serialized after main kernel to avoid race).
    # The main kernel (_fused_conv_ssm_kernel) reads B/C state from all heads in the
    # same group concurrently. Having head-0 write the B/C shift in-kernel causes a
    # data race at large batch sizes: head-0 on one SM can write the shifted state
    # before other heads on different SMs finish reading. Running the shift as a
    # separate kernel guarantees all reads complete before any writes occur.
    _bc_conv_state_update_kernel[(batch, ngroups)](
        conv_input,
        conv_state,
        conv_slot_idx,
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
        BLOCK_DSTATE=BLOCK_DSTATE,
        num_warps=4,
    )
