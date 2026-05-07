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
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    """Rms norm kernel.

    Iterates over BLOCK_N in chunks of BLOCK_TILE. With BLOCK_TILE == BLOCK_N
    (default) this is a single-pass kernel; smaller BLOCK_TILE switches to a
    2-pass chunked layout that reduces register pressure per stage.
    """
    prog_id = tl.program_id(0)
    x_base = input + prog_id * input_row_stride
    out_base = output + prog_id * input_row_stride
    inv_n = float(1.0 / N_COLS)

    if BLOCK_TILE >= BLOCK_N:
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N_COLS
        w = tl.load(weight + offsets, mask=mask).to(tl.float32)
        x = tl.load(x_base + offsets, mask=mask)
        xf = x.to(tl.float32)
        var = tl.sum(xf * xf, 0) * inv_n
        rrms = tl.rsqrt(var + eps)
        out = (xf * rrms * w).to(x.dtype)
        tl.store(out_base + offsets, out, mask=mask)
    else:
        sum_sq = 0.0
        for k in tl.static_range(0, BLOCK_N, BLOCK_TILE):
            offs = k + tl.arange(0, BLOCK_TILE)
            mask = offs < N_COLS
            x_chunk = tl.load(x_base + offs, mask=mask).to(tl.float32)
            sum_sq += tl.sum(x_chunk * x_chunk, 0)
        var = sum_sq * inv_n
        rrms = 1.0 / tl.sqrt(var + eps)
        for k in tl.static_range(0, BLOCK_N, BLOCK_TILE):
            offs = k + tl.arange(0, BLOCK_TILE)
            mask = offs < N_COLS
            w_chunk = tl.load(weight + offs, mask=mask).to(tl.float32)
            x_chunk = tl.load(x_base + offs, mask=mask)
            xf = x_chunk.to(tl.float32)
            out = (xf * rrms * w_chunk).to(x_chunk.dtype)
            tl.store(out_base + offs, out, mask=mask)


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5):
    """Rms norm."""
    # Ensure contiguous: the Triton kernel uses the same stride for both input
    # and output pointers, but torch.empty_like always produces a contiguous
    # output. If hidden_states is non-contiguous (e.g. a split_with_sizes view),
    # input_stride != output_stride → out-of-bounds writes → cudaErrorIllegalAddress.
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    BLOCK_TILE = BLOCK_N  # single-pass fast path; smaller value enables chunked path
    out = torch.empty_like(hidden_states)

    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        BLOCK_TILE=BLOCK_TILE,
        num_warps=4,
        num_stages=3,
    )

    return out
