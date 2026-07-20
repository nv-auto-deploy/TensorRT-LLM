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

"""One-kernel RMSNorm for the DeepSeek-V4 Q-LoRA projection.

A stable op name over ``rms_norm(torch_exact=True)``: bit-identical to
``torch_rmsnorm`` (fp32 reduction, same two bf16 rounding points), consuming the
strided narrow Q child of the fused Q/KV projection copy-free.
"""

import torch

from .triton_rms_norm import rms_norm


@torch.library.custom_op("auto_deploy::deepseek_v4_q_rmsnorm", mutates_args=())
def deepseek_v4_q_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Bit-exact ``torch_rmsnorm`` in one Triton launch; returns a contiguous tensor."""
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be bfloat16, got {input.dtype}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"weight must be bfloat16 or float32, got {weight.dtype}")
    return rms_norm(input, weight, eps, torch_exact=True)


@deepseek_v4_q_rmsnorm.register_fake
def _deepseek_v4_q_rmsnorm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return input.new_empty(input.shape)
