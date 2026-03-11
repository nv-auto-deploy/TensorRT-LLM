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

"""Custom op for Mamba v1 selective scan (as used in Jamba and similar models).

Unlike Mamba2/SSD (torch_ssm), Mamba v1 uses per-element state matrices with shape
[intermediate_size, d_state] and a sequential scan over the sequence dimension.
"""

import torch


@torch.library.custom_op("auto_deploy::torch_mamba_v1_selective_scan", mutates_args={})
def _torch_mamba_v1_selective_scan(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Mamba v1 selective scan with skip connection.

    Args:
        hidden_states: [batch, seq_len, intermediate_size] - input after conv1d + activation
        A: [intermediate_size, d_state] - state matrix (already negated: -exp(A_log))
        B: [batch, seq_len, d_state] - input-dependent B
        C: [batch, seq_len, d_state] - input-dependent C
        D: [intermediate_size] - skip connection coefficient
        dt: [batch, seq_len, intermediate_size] - time step (after softplus)

    Returns:
        scan_output: [batch, seq_len, intermediate_size]
    """
    batch_size, seq_len, d_inner = hidden_states.shape
    dtype = hidden_states.dtype

    # Work in float32 for numerical stability
    hidden_states_f = hidden_states.float()
    B_f = B.float()
    C_f = C.float()
    dt_f = dt.float()
    A_f = A.float()

    # Discretize: A_discrete = exp(A * dt), where A is [D, N] and dt is [B, L, D]
    # discrete_A: [B, L, D, N]
    discrete_A = torch.exp(A_f.unsqueeze(0).unsqueeze(0) * dt_f.unsqueeze(-1))

    # discrete_B = dt * B: [B, L, D, N]
    discrete_B = dt_f.unsqueeze(-1) * B_f.unsqueeze(2)

    # deltaB_u = discrete_B * x: [B, L, D, N]
    deltaB_u = discrete_B * hidden_states_f.unsqueeze(-1)

    # Sequential scan
    ssm_state = torch.zeros(
        batch_size, d_inner, B.shape[-1], device=hidden_states.device, dtype=torch.float32
    )
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, i, :, :] * ssm_state + deltaB_u[:, i, :, :]  # [B, D, N]
        # output_i = sum(ssm_state * C[:, i, :], dim=-1)  ->  [B, D]
        scan_output_i = torch.sum(ssm_state * C_f[:, i, :].unsqueeze(1), dim=-1)
        scan_outputs.append(scan_output_i)
    scan_output = torch.stack(scan_outputs, dim=1)  # [B, L, D]

    # Skip connection: y = y + D * x
    scan_output = scan_output + hidden_states_f * D.float().unsqueeze(0).unsqueeze(0)

    return scan_output.to(dtype)


@_torch_mamba_v1_selective_scan.register_fake
def _torch_mamba_v1_selective_scan_meta(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states, dtype=hidden_states.dtype)
