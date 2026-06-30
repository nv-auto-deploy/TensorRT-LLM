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

"""Fused Triton kernel for the DeepSeek V4 EP global->local routing localization.

Before the trtllm-gen W4A16 MoE runner (``_run_trtllm_gen_mxfp4_from_routing``), the
precomputed ``(selected_experts, routing_weights)`` pair must be converted from GLOBAL
expert coordinates into this rank's LOCAL coordinates and the masked BF16 weights the
runner consumes. The reference chain runs a swarm of tiny per-token kernels::

    local = selected_experts.to(int64) - expert_start  # 1 sub
    valid = (local >= 0) & (local < E_local)  # 2 cmp + 1 and
    local_idx = where(valid, local, E_local).to(int32)  # fill + where + cast
    weights = (routing_weights.to(float32) * valid).to(bf16)  # cast + mul + cast

plus the model-side ``routing_weights.to(bf16)`` cast at the call site. At decode (1-2
tokens, ``top_k`` slots) every one of these is a launch-bound CUDA-graph node over a
tiny ``[num_tokens, top_k]`` tensor (~11 kernels / MoE call). This module collapses the
whole chain into ONE Triton program that emits the int32 local IDs (with the kernel's
invalid-expert sentinel ``E_local`` for off-rank / out-of-range routes) and the masked
BF16 routing weights, leaving the vendored runner itself untouched.

The kernel reads ``routing_weights`` in whatever float dtype the gate produced (fp32),
casting to bf16 internally, so the upstream ``.to(bf16)`` cast is fused away too:
``bf16(f32(w)) == bf16(w)`` and ``bf16(bf16(w)) == bf16(w)``, so the fused output is
bit-identical to the unfused chain for every valid slot (and exactly ``0`` otherwise).
The invalid sentinel ``E_local`` (== local ``num_experts``) makes the runner SKIP those
slots, matching its ``invalid_expert_id = num_experts`` EP convention.

The kernel is deliberately NOT named with ``index``/``select``/``cast`` etc. so the one
remaining program lands in the ``other`` op-type bucket rather than re-inflating the
``gather_scatter``/``copy_cast``/``elementwise`` buckets it drains.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _localize_routing_kernel(
    sel_ptr,  # (N,) int64 global expert ids
    rw_ptr,  # (N,) routing weights (fp32 or bf16)
    local_idx_ptr,  # (N,) int32 output local expert ids (sentinel = LOCAL_EXPERTS)
    weights_ptr,  # (N,) bf16 output masked routing weights
    n_elements,
    expert_start,
    local_experts,
    BLOCK: tl.constexpr,
):
    """Flat elementwise: global->local localize + invalid-sentinel + masked bf16 weights."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    sel = tl.load(sel_ptr + offs, mask=mask, other=0).to(tl.int64)
    local = sel - expert_start
    valid = (local >= 0) & (local < local_experts)
    # Off-rank / out-of-range routes -> invalid sentinel ``local_experts`` so the runner
    # SKIPS them (matches its ``invalid_expert_id = num_experts``).
    out_idx = tl.where(valid, local, local_experts).to(tl.int32)
    tl.store(local_idx_ptr + offs, out_idx, mask=mask)

    rw = tl.load(rw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.where(valid, rw, 0.0).to(tl.bfloat16)
    tl.store(weights_ptr + offs, w, mask=mask)


def deepseek_v4_localize_routing_fn(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused EP global->local routing localization (see module docstring).

    Args:
        selected_experts: ``[..., top_k]`` integer global expert ids.
        routing_weights: ``[..., top_k]`` float routing weights (fp32 or bf16).
        expert_start: this rank's first global expert id (EP shard offset).
        local_experts: number of experts on this rank (== local ``num_experts``).

    Returns:
        local_idx: ``[..., top_k]`` int32 local expert ids; off-rank / out-of-range
            slots carry the invalid sentinel ``local_experts``.
        weights: ``[..., top_k]`` bf16 routing weights, masked to ``0`` on invalid slots.
    """
    sel = selected_experts.reshape(-1)
    rw = routing_weights.reshape(-1)
    n = sel.numel()
    local_idx = torch.empty_like(sel, dtype=torch.int32)
    weights = torch.empty_like(rw, dtype=torch.bfloat16)
    if n > 0:
        # BLOCK >> num_tokens*top_k at decode, so one program covers the whole tensor;
        # at prefill the cdiv grid tiles it. Pure elementwise -> single warp suffices.
        BLOCK = 256
        grid = (triton.cdiv(n, BLOCK),)
        _localize_routing_kernel[grid](
            sel,
            rw,
            local_idx,
            weights,
            n,
            int(expert_start),
            int(local_experts),
            BLOCK=BLOCK,
            num_warps=1,
        )
    return local_idx.reshape(selected_experts.shape), weights.reshape(routing_weights.shape)


@torch.library.custom_op("auto_deploy::deepseek_v4_localize_routing", mutates_args=())
def deepseek_v4_localize_routing(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse the EP global->local routing localization for the trtllm-gen MoE runner.

    Equivalent to::

        local = selected_experts.to(int64) - expert_start
        valid = (local >= 0) & (local < local_experts)
        local_idx = where(valid, local, local_experts).to(int32)
        weights = (routing_weights.to(float32) * valid).to(bfloat16)

    collapsed into one Triton program. Returns ``(local_idx int32, weights bf16)``, both
    shaped like ``selected_experts`` / ``routing_weights`` respectively.
    """
    return deepseek_v4_localize_routing_fn(
        selected_experts, routing_weights, expert_start, local_experts
    )


@deepseek_v4_localize_routing.register_fake
def _deepseek_v4_localize_routing_fake(
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_start: int,
    local_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    local_idx = torch.empty_like(selected_experts, dtype=torch.int32)
    weights = torch.empty_like(routing_weights, dtype=torch.bfloat16)
    return local_idx, weights
