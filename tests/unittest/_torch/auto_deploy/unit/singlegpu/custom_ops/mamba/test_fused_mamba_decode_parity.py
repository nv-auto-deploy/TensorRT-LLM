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

"""Parity tests for the fused Mamba decode launcher.

These tests guard the `fuse_conv_ssm` path used by Nemotron Nano v3. The fused
decode launcher must match an unfused reference (depthwise conv + tuned SSM)
for multi-sequence decode, which is the regime that exposed the GSM8K accuracy
regression.
"""

import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.mamba.fused_mamba_decode import (
    fused_conv_ssm_decode,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.mamba.tuned_ssm_kernel import (
    tuned_selective_state_update,
)
from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_update

# Nemotron Nano v3 decode dimensions
NHEADS = 64
DIM = 64
DSTATE = 128
NGROUPS = 8
KERNEL_WIDTH = 4
INTERMEDIATE_SIZE = NHEADS * DIM
CONV_DIM = INTERMEDIATE_SIZE + 2 * NGROUPS * DSTATE


def _make_inputs(batch, steps, seed=123, device="cuda"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    conv_inputs = [
        torch.randn(batch, CONV_DIM, dtype=torch.bfloat16, device=device, generator=g)
        for _ in range(steps)
    ]
    dt_steps = [
        torch.randn(batch, NHEADS, dtype=torch.bfloat16, device=device, generator=g) * 0.1
        for _ in range(steps)
    ]

    conv_state = torch.randn(
        batch,
        CONV_DIM,
        KERNEL_WIDTH - 1,
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    )
    conv_weight = torch.randn(
        CONV_DIM, KERNEL_WIDTH, dtype=torch.bfloat16, device=device, generator=g
    )
    conv_bias = torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device, generator=g)
    dt_bias = torch.randn(NHEADS, dtype=torch.bfloat16, device=device, generator=g) * 0.1
    A = -torch.rand(NHEADS, dtype=torch.float32, device=device, generator=g)
    D = torch.randn(NHEADS, dtype=torch.bfloat16, device=device, generator=g)
    ssm_state = torch.randn(
        batch,
        NHEADS,
        DIM,
        DSTATE,
        dtype=torch.bfloat16,
        device=device,
        generator=g,
    )
    slot_idx = torch.arange(batch, dtype=torch.int32, device=device)

    return (
        conv_inputs,
        dt_steps,
        conv_state,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        D,
        ssm_state,
        slot_idx,
    )


def _reference_decode_step(
    conv_input,
    dt,
    conv_state,
    conv_weight,
    conv_bias,
    dt_bias,
    A,
    D,
    ssm_state,
    slot_idx,
):
    conv_out = causal_conv1d_update(
        conv_input.clone(),
        conv_state,
        conv_weight,
        conv_bias,
        activation="silu",
        cache_seqlens=None,
        conv_state_indices=slot_idx,
        pad_slot_id=PAD_SLOT_ID,
    )

    x = conv_out[:, :INTERMEDIATE_SIZE].view(conv_input.shape[0], NHEADS, DIM)
    B = conv_out[:, INTERMEDIATE_SIZE : INTERMEDIATE_SIZE + NGROUPS * DSTATE].view(
        conv_input.shape[0], NGROUPS, DSTATE
    )
    C = conv_out[:, INTERMEDIATE_SIZE + NGROUPS * DSTATE :].view(
        conv_input.shape[0], NGROUPS, DSTATE
    )

    dt_hp = dt.unsqueeze(-1).expand(-1, -1, DIM)
    dt_bias_hp = dt_bias.unsqueeze(-1).expand(-1, DIM)
    A_full = A[:, None, None].expand(-1, DIM, DSTATE)
    D_full = D[:, None].expand(-1, DIM)

    out = torch.empty(
        conv_input.shape[0], NHEADS, DIM, dtype=conv_input.dtype, device=conv_input.device
    )
    tuned_selective_state_update(
        ssm_state,
        x,
        dt_hp,
        A_full,
        B,
        C,
        D=D_full,
        dt_bias=dt_bias_hp,
        dt_softplus=True,
        state_batch_indices=slot_idx,
        out=out,
        dt_clamp_min=None,
        dt_clamp_max=None,
    )

    return out


def test_fused_conv_ssm_decode_matches_unfused_reference_multi_sequence():
    batch = 64
    steps = 4

    (
        conv_inputs,
        dt_steps,
        conv_state,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        D,
        ssm_state,
        slot_idx,
    ) = _make_inputs(batch=batch, steps=steps)

    fused_conv_state = conv_state.clone()
    ref_conv_state = conv_state.clone()
    fused_ssm_state = ssm_state.clone()
    ref_ssm_state = ssm_state.clone()

    for step_idx, (conv_input, dt) in enumerate(zip(conv_inputs, dt_steps), start=1):
        fused_out = torch.empty(
            batch, NHEADS, DIM, dtype=conv_input.dtype, device=conv_input.device
        )
        fused_conv_ssm_decode(
            conv_input,
            fused_conv_state,
            conv_weight,
            conv_bias,
            dt,
            dt_bias,
            A,
            D,
            fused_ssm_state,
            slot_idx,
            slot_idx,
            fused_out,
        )

        ref_out = _reference_decode_step(
            conv_input,
            dt,
            ref_conv_state,
            conv_weight,
            conv_bias,
            dt_bias,
            A,
            D,
            ref_ssm_state,
            slot_idx,
        )

        out_max_diff = (fused_out.float() - ref_out.float()).abs().max().item()
        ssm_max_diff = (fused_ssm_state.float() - ref_ssm_state.float()).abs().max().item()

        assert torch.equal(fused_conv_state, ref_conv_state), (
            f"step={step_idx}: fused conv state diverged from the unfused reference"
        )
        assert out_max_diff < 0.5, (
            f"step={step_idx}: fused decode output diverged from the unfused reference "
            f"(max_diff={out_max_diff:.4f})"
        )
        assert ssm_max_diff < 0.5, (
            f"step={step_idx}: fused SSM state diverged from the unfused reference "
            f"(max_diff={ssm_max_diff:.4f})"
        )


def test_fused_conv_ssm_decode_handles_padded_rows():
    batch = 64
    steps = 3

    (
        conv_inputs,
        dt_steps,
        conv_state,
        conv_weight,
        conv_bias,
        dt_bias,
        A,
        D,
        ssm_state,
        _slot_idx,
    ) = _make_inputs(batch=batch, steps=steps)

    slot_idx = torch.arange(batch, dtype=torch.int32, device=conv_state.device)
    slot_idx[-8:] = PAD_SLOT_ID

    fused_conv_state = conv_state.clone()
    ref_conv_state = conv_state.clone()
    fused_ssm_state = ssm_state.clone()
    ref_ssm_state = ssm_state.clone()

    for step_idx, (conv_input, dt) in enumerate(zip(conv_inputs, dt_steps), start=1):
        fused_out = torch.zeros(
            batch, NHEADS, DIM, dtype=conv_input.dtype, device=conv_input.device
        )
        fused_conv_ssm_decode(
            conv_input,
            fused_conv_state,
            conv_weight,
            conv_bias,
            dt,
            dt_bias,
            A,
            D,
            fused_ssm_state,
            slot_idx,
            slot_idx,
            fused_out,
        )

        ref_out = _reference_decode_step(
            conv_input,
            dt,
            ref_conv_state,
            conv_weight,
            conv_bias,
            dt_bias,
            A,
            D,
            ref_ssm_state,
            slot_idx,
        )

        active = slot_idx != PAD_SLOT_ID
        out_max_diff = (fused_out[active].float() - ref_out[active].float()).abs().max().item()
        ssm_max_diff = (
            (fused_ssm_state[active].float() - ref_ssm_state[active].float()).abs().max().item()
        )

        assert torch.equal(fused_conv_state[active], ref_conv_state[active]), (
            f"step={step_idx}: fused conv state diverged on active rows with padded replay"
        )
        assert out_max_diff < 0.5, (
            f"step={step_idx}: fused decode output diverged on active rows with padded replay "
            f"(max_diff={out_max_diff:.4f})"
        )
        assert ssm_max_diff < 0.5, (
            f"step={step_idx}: fused SSM state diverged on active rows with padded replay "
            f"(max_diff={ssm_max_diff:.4f})"
        )
        assert torch.count_nonzero(fused_out[~active]).item() == 0, (
            f"step={step_idx}: padded rows should stay zeroed in fused decode output"
        )
