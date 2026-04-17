// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Opcode: OP_GEMV_OPROJ — O-projection + post-attention norms + residual.
//
// Fuses:
//   1. O-proj GEMV: attn_scratch[Q_WIDTH] × o_proj_weight[HIDDEN, Q_WIDTH]^T → [HIDDEN]
//   2. RMSNorm on O-proj output (post_attention_layernorm)
//   3. Residual add: post_attn = residual + normed_oproj
//   4. RMSNorm on post_attn (pre_feedforward_layernorm)
//   5. Output: post_attn_out + pre_ffn_out
//
// The norms + residual are fused into the GEMV epilogue:
// each SM that computes output row h also applies the norms/residual for that row.
// This requires a two-pass approach:
//   Pass 1: compute O-proj GEMV → o_proj_scratch (fp32)
//   Barrier (for RMSNorm variance reduction across SMs)
//   Pass 2: each SM applies norms + residual for its rows
//
// But since the norms are over the full HIDDEN_SIZE, a single SM can't compute the
// variance without reading all rows. So we use a simpler approach:
//   - All SMs cooperate on GEMV, writing to o_proj_scratch
//   - After barrier, ONE SM (or all SMs cooperatively) computes norms + residual
//
// For simplicity and correctness, we write raw O-proj to o_proj_scratch (fp32),
// then a post-processing step applies norms.
//
// Instruction encoding:
//   word[0] = OP_GEMV_OPROJ (5)
//   word[1] = row_start (inclusive)
//   word[2] = row_end (exclusive)
//   word[3] = token_id

#pragma once

#include "../gemma4_config.cuh"
#include "../gemma4_globals.cuh"
#include "../megakernel_framework.cuh"

namespace gemma4_megakernel
{

__device__ void handle_gemv_oproj(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const row_start = instr.field(1);
    int const row_end = instr.field(2);
    int const token_id = instr.field(3);
    int const num_rows = row_end - row_start;

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    // ALL 20 warps participate in GEMV
    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DATA_OFFSET);

    {
        __nv_bfloat16 const* attn_row = gg.attn_scratch + (int64_t) token_id * Q_WIDTH;
        int const tid = threadIdx.x;
        for (int i = tid; i < Q_WIDTH; i += THREADS_PER_BLOCK)
        {
            s_input[i] = attn_row[i];
        }
    }
    __syncthreads();

    int const rows_per_warp = (num_rows + NUM_WARPS - 1) / NUM_WARPS;
    int const warp_row_start = row_start + warp_id * rows_per_warp;
    int const warp_row_end = min(warp_row_start + rows_per_warp, row_end);

    // 128-bit vectorized dot product for Q_WIDTH=4096
    // 4096 / (32 lanes × 8 elems) = 16 iterations
    for (int row = warp_row_start; row < warp_row_end; row++)
    {
        __nv_bfloat16 const* w = gg.o_proj_weight + (int64_t) row * Q_WIDTH;
        float acc = 0.0f;

        constexpr int EPL = 8;              // elements per lane
        constexpr int VS = WARP_SIZE * EPL; // 256
        constexpr int NI = Q_WIDTH / VS;    // 16

#pragma unroll
        for (int vi = 0; vi < NI; vi++)
        {
            int k = vi * VS + lane_id * EPL;
            uint4 w_vec = *reinterpret_cast<uint4 const*>(&w[k]);
            uint4 x_vec = *reinterpret_cast<uint4 const*>(&s_input[k]);
            __nv_bfloat162 const* w2 = reinterpret_cast<__nv_bfloat162 const*>(&w_vec);
            __nv_bfloat162 const* x2 = reinterpret_cast<__nv_bfloat162 const*>(&x_vec);
#pragma unroll
            for (int p = 0; p < 4; p++)
            {
                acc += __low2float(w2[p]) * __low2float(x2[p]) + __high2float(w2[p]) * __high2float(x2[p]);
            }
        }

#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
        }

        if (lane_id == 0)
        {
            gg.o_proj_scratch[(int64_t) token_id * HIDDEN_SIZE + row] = acc;
        }
    }

    // Flush writes for cross-SM visibility after barrier
    __threadfence();
}

// ──────────────────────────────────────────────────────────────
// OP_OPROJ_POST (opcode 7) — apply post-attention norms + residual
//
// Reads: o_proj_scratch (fp32), residual (bf16), norm weights
// Writes: post_attn_out (fp32), pre_ffn_out (fp32)
//
// This runs on a single SM (or few SMs) because RMSNorm needs
// the full HIDDEN_SIZE vector for variance computation.
//
// Instruction:
//   word[0] = 7
//   word[1] = token_id
// ──────────────────────────────────────────────────────────────
__device__ void handle_oproj_post(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const token_id = instr.field(1);
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;
    int const tid = threadIdx.x;

    // ALL 20 warps cooperate (was 1 warp → 20x parallelism for memory loads)
    extern __shared__ char smem_raw[];
    float* s_reduce = reinterpret_cast<float*>(smem_raw + SMEM_DATA_OFFSET);

    float const* oproj = gg.o_proj_scratch + (int64_t) token_id * HIDDEN_SIZE;
    __nv_bfloat16 const* residual = gg.residual + (int64_t) token_id * HIDDEN_SIZE;

    // Step 1: Compute sumsq for RMSNorm of O-proj output (all threads participate)
    float sumsq_attn = 0.0f;
    for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
    {
        float v = oproj[h];
        sumsq_attn += v * v;
    }
// Warp reduction
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sumsq_attn += __shfl_down_sync(0xFFFFFFFF, sumsq_attn, offset);
    }
    if (lane_id == 0)
        s_reduce[warp_id] = sumsq_attn;
    __syncthreads();
    // Final reduction across warps (warp 0)
    if (warp_id == 0)
    {
        float val = (lane_id < NUM_WARPS) ? s_reduce[lane_id] : 0.0f;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0)
            s_reduce[0] = val;
    }
    __syncthreads();
    float inv_rms_attn = rsqrtf(s_reduce[0] / HIDDEN_SIZE + gg.eps);

    // Step 2: post_attn = residual + RMSNorm(o_proj) * post_weight + compute sumsq_post
    float sumsq_post = 0.0f;
    for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
    {
        float o = oproj[h];
        float r = __bfloat162float(residual[h]);
        float pw = gg.post_attn_norm_weight[h];
        float post_val = r + o * inv_rms_attn * pw;
        gg.post_attn_out[(int64_t) token_id * HIDDEN_SIZE + h] = post_val;
        sumsq_post += post_val * post_val;
    }
// Warp + cross-warp reduction for sumsq_post
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sumsq_post += __shfl_down_sync(0xFFFFFFFF, sumsq_post, offset);
    }
    if (lane_id == 0)
        s_reduce[warp_id] = sumsq_post;
    __syncthreads();
    if (warp_id == 0)
    {
        float val = (lane_id < NUM_WARPS) ? s_reduce[lane_id] : 0.0f;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0)
            s_reduce[0] = val;
    }
    __syncthreads();
    float inv_rms_post = rsqrtf(s_reduce[0] / HIDDEN_SIZE + gg.eps);

    // Step 3: pre_ffn = RMSNorm(post_attn) * pre_ffn_weight
    for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
    {
        float post_val = gg.post_attn_out[(int64_t) token_id * HIDDEN_SIZE + h];
        float fw = gg.pre_ffn_norm_weight[h];
        gg.pre_ffn_out[(int64_t) token_id * HIDDEN_SIZE + h] = post_val * inv_rms_post * fw;
    }
}

} // namespace gemma4_megakernel
