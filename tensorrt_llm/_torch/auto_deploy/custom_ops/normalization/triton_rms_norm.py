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

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride,
    output_row_stride,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """RMSNorm with an fp32 reduction and fp32 weight multiply."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    w = tl.load(weight + offsets, mask=mask)
    x = tl.load(input + prog_id * input_row_stride + offsets, mask=mask)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = (w.to(tl.float32) * (xf / tl.sqrt(var + eps))).to(x.dtype)

    tl.store(output + prog_id * output_row_stride + offsets, out, mask=mask)


def _flattens_to_regular_rows(t: Tensor, feat_size: int) -> bool:
    """True if ``t`` is a last-dim-contiguous view whose leading dims flatten to
    uniformly strided, non-overlapping ``feat_size``-wide rows (e.g. a narrow view
    of a wider fused projection)."""
    if t.dim() < 2 or t.shape[-1] != feat_size or t.stride(-1) != 1:
        return False
    if t.shape[-2] > 1 and t.stride(-2) < feat_size:
        return False
    expected = t.shape[-2] * t.stride(-2)
    for dim in range(t.dim() - 3, -1, -1):
        if t.shape[dim] > 1 and t.stride(dim) != expected:
            return False
        expected *= t.shape[dim]
    return True


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5):
    """RMSNorm.

    Regularly strided rows (e.g. narrow views) are consumed in place; irregular
    layouts fall back to one ``contiguous()`` copy. The output is always contiguous.
    """
    feat_size = weight.shape[0]
    if not _flattens_to_regular_rows(hidden_states, feat_size):
        hidden_states = hidden_states.contiguous()
    seq_len = hidden_states.numel() // hidden_states.size(-1)

    out = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)

    BLOCK_N = triton.next_power_of_2(feat_size)
    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_row_stride=hidden_states.stride(-2),
        output_row_stride=out.stride(-2),
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return out
