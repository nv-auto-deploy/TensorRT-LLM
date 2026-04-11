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

"""Fused Gemma4 MoE Router kernel: RMSNorm + scale + proj + softmax + topk + normalize.

Fuses 9 separate PyTorch kernel launches into a single Triton kernel per token.
Key design decisions:
- W is transposed to W_T[H, E] so inner-H loop accesses W_T[h, 0:E] contiguously (coalesced).
- Single pass over H: compute RMSNorm variance AND (x*scale*W) simultaneously.
  This works because score[e] = norm_factor * sum_h(x[h]*scale[h]*W[e,h]), so we
  accumulate raw (un-normalized) scores and multiply by norm_factor at the end.
- Softmax and topk done in registers after the H loop.
- TopK masking: use tl.max + sentinel pattern (from triton_routing.py) to correctly
  handle ties and 0-d index tensors across all T values.
- Multi-token programs: N_TOKENS tokens per program reuse the W_T tile across tokens,
  reducing W_T bandwidth by N_TOKENS for large-T prefill shapes.
- Fast softmax: use exp2(x * log2e) instead of exp(x) for ~10% faster exp.
"""

import math as _math
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_and_store(
    probs,  # [E] softmax probabilities
    e_offs,  # [E] expert offsets
    weights_ptr,
    indices_ptr,
    out_offset,  # base offset for this token's outputs
    E: tl.constexpr,
    K: tl.constexpr,
):
    """Iterative top-K selection with sentinel pattern; stores K weights and indices."""
    k_offs = tl.arange(0, K)
    top_vals = tl.zeros([K], dtype=tl.float32)
    top_idxs = tl.zeros([K], dtype=tl.int32)
    top_sum = tl.zeros([1], dtype=tl.float32)

    for i in tl.static_range(K):
        max_val = tl.max(probs, 0)
        is_max = probs == max_val
        candidate = tl.where(is_max, e_offs, E)
        max_idx = tl.min(candidate, 0)

        ki_mask = k_offs == i
        top_vals = tl.where(ki_mask, max_val, top_vals)
        top_idxs = tl.where(ki_mask, max_idx.to(tl.int32), top_idxs)
        top_sum += max_val
        probs = tl.where(e_offs == max_idx, float("-inf"), probs)

    tl.store(weights_ptr + out_offset + k_offs, top_vals / top_sum)
    tl.store(indices_ptr + out_offset + k_offs, top_idxs)


@triton.jit
def _gemma4_router_fwd(
    hidden_ptr,  # [T, H] input hidden states
    scale_ptr,  # [H]    per-dim router scale
    proj_T_ptr,  # [H, E] transposed proj weight (W.T of nn.Linear weight [E, H])
    weights_ptr,  # [T, K] output top-k weights (normalized)
    indices_ptr,  # [T, K] output top-k expert indices (int32)
    stride_th,  # stride of hidden along token dim (= H)
    stride_hE,  # stride of proj_T along H dim (= E)
    T,  # total number of tokens (runtime, for boundary check)
    H: tl.constexpr,
    E: tl.constexpr,  # must be power of 2 (128)
    K: tl.constexpr,  # must be power of 2 (8)
    root_size,
    eps,
    BLOCK_H: tl.constexpr,  # tile size along H
    N_TOKENS: tl.constexpr,  # tokens per program (1 for decode, 2-4 for prefill)
    LOG2E: tl.constexpr,  # tl.math.log2e() as constexpr for fast exp2
):
    prog_id = tl.program_id(0)
    t_base = prog_id * N_TOKENS

    h_offs = tl.arange(0, BLOCK_H)
    e_offs = tl.arange(0, E)

    # For multi-token programs: loop over N_TOKENS tokens per program.
    # When T is not a multiple of N_TOKENS, the last program may have fewer tokens.
    # We mask out-of-range tokens by conditioning on t < T.
    for n in tl.static_range(N_TOKENS):
        t = t_base + n

        # Accumulators (scalar for var, vector [E] for scores)
        var = tl.zeros([1], dtype=tl.float32)
        scores = tl.zeros([E], dtype=tl.float32)

        # Single pass over H: accumulate variance and (x*scale) @ W simultaneously
        for h_base in tl.range(0, H, BLOCK_H):
            h_idx = h_base + h_offs
            mask = h_idx < H

            # Load hidden [BLOCK_H] and scale [BLOCK_H]
            # Mask out-of-range tokens with other=0.0 (var and scores stay 0)
            token_mask = (t < T) & mask
            x = tl.load(hidden_ptr + t * stride_th + h_idx, mask=token_mask, other=0.0).to(
                tl.float32
            )
            s = tl.load(scale_ptr + h_idx, mask=mask, other=0.0).to(tl.float32)

            # Accumulate variance (without applying norm yet)
            var += tl.sum(x * x, 0)

            # Load W_T[h_base:h_base+BLOCK_H, 0:E] as [BLOCK_H, E] tile
            w_T = tl.load(
                proj_T_ptr + h_idx[:, None] * stride_hE + e_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)  # [BLOCK_H, E]

            # Accumulate scores: scores += (x * s) @ w_T  -> [E]
            xs = (x * s)[:, None]  # [BLOCK_H, 1]
            scores += tl.sum(xs * w_T, 0)  # [E]

        # Only write outputs for valid tokens
        if t < T:
            # Apply RMSNorm scaling
            norm_factor = tl.rsqrt(var / H + eps) * root_size
            scores = scores * norm_factor  # [E]

            # Softmax over E experts using exp2 for speed
            max_s = tl.max(scores, 0)
            exp_s = tl.exp2((scores - max_s) * LOG2E)
            sum_exp = tl.sum(exp_s, 0)
            probs = exp_s / sum_exp  # [E]

            # TopK and store
            _topk_and_store(probs, e_offs, weights_ptr, indices_ptr, t * K, E, K)


def gemma4_router_triton(
    hidden: torch.Tensor,  # [T, H]
    scale: torch.Tensor,  # [H]
    proj_T: torch.Tensor,  # [H, E] transposed proj weight
    root_size: float,
    eps: float,
    top_k: int,
    num_warps: int = 8,
    num_stages: int = 2,
    block_h: int = 512,
    n_tokens: int = 1,
) -> tuple:
    """Fused Gemma4 router: returns (top_k_weights [T,K], top_k_indices [T,K]).

    Default params nw=8, ns=2, bh=512, n_tokens=1 are tuned for H100 decode.
    For large prefill (T>=256), use n_tokens=2-4 to amortize W_T bandwidth.
    """
    T, H = hidden.shape
    E = proj_T.shape[1]
    K = top_k

    out_weights = torch.empty((T, K), dtype=torch.float32, device=hidden.device)
    out_indices = torch.empty((T, K), dtype=torch.int32, device=hidden.device)

    log2e = _math.log2(_math.e)

    grid = (triton.cdiv(T, n_tokens),)
    _gemma4_router_fwd[grid](
        hidden,
        scale,
        proj_T,
        out_weights,
        out_indices,
        hidden.stride(0),  # stride_th = H
        proj_T.stride(0),  # stride_hE = E
        T=T,
        H=H,
        E=E,
        K=K,
        root_size=root_size,
        eps=eps,
        BLOCK_H=block_h,
        N_TOKENS=n_tokens,
        LOG2E=log2e,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_weights, out_indices


# ---------------------------------------------------------------------------
# Custom op registration — needed for FakeTensor / meta tensor tracing in AD
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::gemma4_router", mutates_args=(), device_types="cuda")
def gemma4_router(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    proj_weight: torch.Tensor,
    root_size: float,
    eps: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Gemma4 router custom op: RMSNorm + proj + softmax + topk.

    Accepts ``proj_weight`` in its natural ``[E, H]`` (nn.Linear) layout and
    transposes it to ``[H, E]`` internally so callers can pass ``self.proj.weight``
    directly without maintaining a separate transposed buffer.

    Registered as a torch custom op so that FakeTensor / meta-tensor tracing
    during torch.export sees only shape inference (via ``register_fake``) rather
    than the actual Triton kernel, which cannot execute on meta tensors.

    Kernel parameters are selected based on T for optimal throughput:
    - T < 256: bh=512, nw=8 (best for decode batch sizes 1–128)
    - T >= 256: bh=256, nw=4 (29% faster for large-batch decode / prefill)
    """
    proj_T = proj_weight.t().contiguous()
    T = hidden.shape[0]
    num_warps = 4 if T >= 256 else 8
    block_h = 256 if T >= 256 else 512
    return gemma4_router_triton(
        hidden, scale, proj_T, root_size, eps, top_k, num_warps=num_warps, block_h=block_h
    )


@gemma4_router.register_fake
def _gemma4_router_fake(
    hidden: torch.Tensor,
    scale: torch.Tensor,
    proj_weight: torch.Tensor,
    root_size: float,
    eps: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = hidden.shape[0]
    return (
        torch.empty((T, top_k), dtype=torch.float32, device=hidden.device),
        torch.empty((T, top_k), dtype=torch.int32, device=hidden.device),
    )
