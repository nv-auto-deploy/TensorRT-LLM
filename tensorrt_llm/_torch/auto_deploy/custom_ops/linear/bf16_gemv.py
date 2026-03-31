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

"""Custom BF16 GEMV for LM head decode performance.

At decode time the LM head is a GEMV: [1, K] × [K, N] → [1, N].
The standard cuBLAS/NVJET path accesses weight as [N, K] row-major, where
iterating over different output elements (n) in a warp requires stride-K
jumps through memory — 5376 bytes per step for K=2688 — resulting in
~40% HBM bandwidth utilization.

This kernel pre-transposes the weight to [K, N] row-major, so that a warp
handling 32 consecutive n values reads weight_T[k, n:n+32] — contiguous
in memory — yielding coalesced 64-byte transactions and much better
bandwidth utilization.

Kernel design:
  - weight_T: [K, N] contiguous (pre-transposed once at model load time)
  - Grid: (ceil(N / BLOCK_N),)
  - Each block: accumulates BLOCK_N outputs over K in BLOCK_K-sized tiles
  - tl.dot([1, BLOCK_K] × [BLOCK_K, BLOCK_N]) drives the inner loop
  - For M > 1 (prefill): fall back to torch.mm
"""

import torch
import triton
import triton.language as tl

# Tunable constants for [1, 2688] × [2688, 64000] BF16 on H100
_BLOCK_N = 256  # output elements per SM tile
_BLOCK_K = 32  # K elements per tl.dot inner iteration
_NUM_WARPS = 8  # 8 warps × 32 threads = 256 threads per block
_NUM_STAGES = 3  # software pipeline depth


@triton.jit
def _bf16_gemv_transposed_kernel(
    x_ptr,  # [K] — single-token hidden state (after gather)
    w_ptr,  # [K, N] — weight pre-transposed to [hidden, vocab]
    y_ptr,  # [N] — output logits
    K,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """BF16 GEMV: y = x @ w_T  where w_T is stored as [K, N] row-major.

    For fixed k, w_T[k, n:n+BLOCK_N] is contiguous → coalesced warp loads.
    """
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    # Accumulator [1, BLOCK_N] in fp32 — 2-D so tl.dot can be used
    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for k_base in tl.range(0, K, BLOCK_K):
        k_offs = k_base + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # x[k_base : k_base+BLOCK_K] — contiguous load, reshape to [1, BLOCK_K]
        x_chunk = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0)  # [BLOCK_K]
        x_mat = tl.reshape(x_chunk, (1, BLOCK_K)).to(tl.float32)  # [1, BLOCK_K]

        # w_T[k_base:k_base+BLOCK_K, n_start:n_start+BLOCK_N]
        # Address: w_ptr + k * N + n  ← coalesced for varying n, fixed k
        w_ptrs = w_ptr + k_offs[:, None] * N + n_offs[None, :]  # [BLOCK_K, BLOCK_N]
        w_chunk = tl.load(
            w_ptrs,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # acc += x_mat @ w_chunk  ([1, BLOCK_K] × [BLOCK_K, BLOCK_N] → [1, BLOCK_N])
        acc = tl.dot(x_mat, w_chunk, acc, out_dtype=tl.float32)

    # Flatten [1, BLOCK_N] → [BLOCK_N] and write outputs
    acc_1d = tl.reshape(acc, (BLOCK_N,))
    tl.store(y_ptr + n_offs, acc_1d.to(tl.bfloat16), mask=n_mask)


def _gemv_decode(x_flat: torch.Tensor, weight_T: torch.Tensor) -> torch.Tensor:
    """Run the Triton GEMV for exactly one token.

    Args:
        x_flat:   [K] BF16 tensor (single token hidden state).
        weight_T: [K, N] BF16 tensor (pre-transposed weight).

    Returns:
        y: [N] BF16 logits.
    """
    K, N = weight_T.shape
    assert x_flat.shape == (K,), f"Expected x_flat shape ({K},), got {x_flat.shape}"
    y = torch.empty(N, dtype=torch.bfloat16, device=x_flat.device)
    grid = (triton.cdiv(N, _BLOCK_N),)
    _bf16_gemv_transposed_kernel[grid](
        x_flat,
        weight_T,
        y,
        K,
        N,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return y


@torch.library.custom_op("auto_deploy::bf16_lm_head_gemv", mutates_args=())
def _bf16_lm_head_gemv(
    x: torch.Tensor,
    weight_T: torch.Tensor,
) -> torch.Tensor:
    """BF16 LM head GEMV: y = x @ weight_T.

    Handles x of any shape (..., K): flattens leading dims to M_total,
    dispatches to Triton GEMV for M_total==1 (decode) or torch.mm otherwise.

    Args:
        x:        [..., K] BF16 input (any number of leading dims).
        weight_T: [K, N] BF16 weight (pre-transposed from [N, K]).

    Returns:
        y: [..., N] BF16 output with same leading dims as x.
    """
    K, N = weight_T.shape
    leading = x.shape[:-1]
    x_2d = x.reshape(-1, K)  # [M_total, K]
    M = x_2d.shape[0]
    if M == 1:
        out = _gemv_decode(x_2d[0], weight_T).unsqueeze(0)  # [1, N]
    else:
        out = torch.mm(x_2d, weight_T)  # [M_total, N]
    return out.reshape(*leading, N)


@_bf16_lm_head_gemv.register_fake
def _bf16_lm_head_gemv_fake(
    x: torch.Tensor,
    weight_T: torch.Tensor,
) -> torch.Tensor:
    N = weight_T.shape[1]
    return x.new_empty(*x.shape[:-1], N)


def benchmark_lm_head_gemv(K: int = 2688, N: int = 64000, warmup: int = 25, rep: int = 100):
    """Microbenchmark: compare torch.mm vs custom GEMV for M=1 BF16 GEMV."""
    device = torch.device("cuda")
    x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
    weight = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    weight_T = weight.T.contiguous()  # [K, N]

    def ref():
        return torch.mm(x, weight.T)

    def custom():
        return _bf16_lm_head_gemv(x, weight_T)

    t_ref = triton.testing.do_bench(ref, warmup=warmup, rep=rep)
    t_custom = triton.testing.do_bench(custom, warmup=warmup, rep=rep)
    bw_ref = (N * K * 2) / t_ref / 1e9  # TB/s
    bw_custom = (N * K * 2) / t_custom / 1e9
    print(
        f"[K={K}, N={N}] torch.mm: {t_ref * 1000:.1f}µs ({bw_ref:.2f}TB/s)  "
        f"custom_gemv: {t_custom * 1000:.1f}µs ({bw_custom:.2f}TB/s)  "
        f"speedup: {t_ref / t_custom:.2f}x"
    )
    return t_ref, t_custom
