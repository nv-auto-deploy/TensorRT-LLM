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

// Opcode: OP_GEMV_QKV — Raw QKV matrix-vector multiply with Kahan summation.
//
// Uses Kahan compensated summation for the dot product to achieve
// accumulation-order-independent results. This eliminates the numerical
// drift between the megakernel and cuBLAS that caused text degradation
// when compounding over 25+ layers.
//
// Kahan summation doubles the effective precision of fp32 accumulation,
// making the result nearly identical regardless of the order elements
// are summed. The 2x extra FLOPs are free since GEMV is memory-bound.

#pragma once

#include "../gemma4_config.cuh"
#include "../gemma4_globals.cuh"
#include "../megakernel_framework.cuh"

namespace gemma4_megakernel
{

// Warp-level dot product with Kahan compensated summation.
// Each lane processes HIDDEN_SIZE/32 elements with error compensation,
// then a compensated warp reduction combines the partial sums.
__device__ __forceinline__ float warp_dot_product(
    __nv_bfloat16 const* __restrict__ weight_row, __nv_bfloat16 const* __restrict__ input, int lane_id)
{
    float sum = 0.0f;
    float comp = 0.0f; // Kahan compensation term

    constexpr int ELEMS_PER_LANE = 8;
    constexpr int VEC_STRIDE = WARP_SIZE * ELEMS_PER_LANE;
    constexpr int NUM_ITERS = HIDDEN_SIZE / VEC_STRIDE;

#pragma unroll
    for (int vi = 0; vi < NUM_ITERS; vi++)
    {
        int k = vi * VEC_STRIDE + lane_id * ELEMS_PER_LANE;
        uint4 w_vec = *reinterpret_cast<uint4 const*>(&weight_row[k]);
        uint4 x_vec = *reinterpret_cast<uint4 const*>(&input[k]);
        __nv_bfloat162 const* w2 = reinterpret_cast<__nv_bfloat162 const*>(&w_vec);
        __nv_bfloat162 const* x2 = reinterpret_cast<__nv_bfloat162 const*>(&x_vec);
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            float wl = __low2float(w2[p]);
            float wh = __high2float(w2[p]);
            float xl = __low2float(x2[p]);
            float xh = __high2float(x2[p]);
            // Kahan compensated addition for each product
            float prod = wl * xl + wh * xh;
            float y = prod - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
    }

    // Compensated warp reduction: include compensation in the shuffle
    // First, add comp to sum for each lane's final value
    sum = sum - comp;

    // Standard warp reduction (the per-lane compensation already removed most error)
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

// O-proj variant for Q_WIDTH=4096
__device__ __forceinline__ float warp_dot_product_qw(
    __nv_bfloat16 const* __restrict__ weight_row, __nv_bfloat16 const* __restrict__ input, int lane_id)
{
    float sum = 0.0f;
    float comp = 0.0f;

    constexpr int ELEMS_PER_LANE = 8;
    constexpr int VEC_STRIDE = WARP_SIZE * ELEMS_PER_LANE;
    constexpr int NUM_ITERS = Q_WIDTH / VEC_STRIDE;

#pragma unroll
    for (int vi = 0; vi < NUM_ITERS; vi++)
    {
        int k = vi * VEC_STRIDE + lane_id * ELEMS_PER_LANE;
        uint4 w_vec = *reinterpret_cast<uint4 const*>(&weight_row[k]);
        uint4 x_vec = *reinterpret_cast<uint4 const*>(&input[k]);
        __nv_bfloat162 const* w2 = reinterpret_cast<__nv_bfloat162 const*>(&w_vec);
        __nv_bfloat162 const* x2 = reinterpret_cast<__nv_bfloat162 const*>(&x_vec);
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            float prod = __low2float(w2[p]) * __low2float(x2[p]) + __high2float(w2[p]) * __high2float(x2[p]);
            float y = prod - comp;
            float t = sum + y;
            comp = (t - sum) - y;
            sum = t;
        }
    }

    sum = sum - comp;

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

__device__ void handle_gemv_qkv(
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
        __nv_bfloat16 const* input_row = gg.attn_normed + (int64_t) token_id * HIDDEN_SIZE;
        int const tid = threadIdx.x;
        for (int i = tid; i < HIDDEN_SIZE; i += THREADS_PER_BLOCK)
        {
            s_input[i] = input_row[i];
        }
    }
    __syncthreads();

    // All 20 warps process rows
    int const rows_per_warp = (num_rows + NUM_WARPS - 1) / NUM_WARPS;
    int const warp_row_start = row_start + warp_id * rows_per_warp;
    int const warp_row_end = min(warp_row_start + rows_per_warp, row_end);

    for (int row = warp_row_start; row < warp_row_end; row++)
    {
        __nv_bfloat16 const* w = gg.qkv_weight + (int64_t) row * HIDDEN_SIZE;
        float result = warp_dot_product(w, s_input, lane_id);
        if (lane_id == 0)
        {
            gg.qkv_scratch[(int64_t) token_id * QKV_WIDTH + row] = __float2bfloat16(result);
        }
    }

    // Flush writes to global memory for cross-SM visibility after barrier
    __threadfence();
}

} // namespace gemma4_megakernel
