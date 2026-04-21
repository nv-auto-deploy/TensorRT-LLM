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

#pragma once

#include "../gemma4_config.cuh"
#include "../gemma4_globals.cuh"
#include "../megakernel_framework.cuh"

namespace gemma4_megakernel
{

static constexpr int FFN_MATH_WARPS = 16;
static constexpr int MOE_FC1_MATH_WARPS = NUM_WARPS;
static constexpr int MOE_FC2_MATH_WARPS = 16;

__device__ __forceinline__ float gelu_tanh_approx(float x)
{
    constexpr float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    float const x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

__device__ __forceinline__ float block_sum(float sum, float* s_reduce, int warp_id, int lane_id)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane_id == 0)
    {
        s_reduce[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        float val = lane_id < NUM_WARPS ? s_reduce[lane_id] : 0.0f;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (lane_id == 0)
        {
            s_reduce[0] = val;
        }
    }
    __syncthreads();
    return s_reduce[0];
}

__device__ __forceinline__ float warp_dot_product_bf16(
    __nv_bfloat16 const* weight_row, __nv_bfloat16 const* input, int dim, int lane_id)
{
    constexpr int kElemsPerLane = 8;
    constexpr int kVecStride = WARP_SIZE * kElemsPerLane;
    float sum = 0.0f;
    int const full_dim = dim - (dim % kVecStride);

#pragma unroll
    for (int k = lane_id * kElemsPerLane; k < full_dim; k += kVecStride)
    {
        uint4 const w_vec = *reinterpret_cast<uint4 const*>(&weight_row[k]);
        uint4 const x_vec = *reinterpret_cast<uint4 const*>(&input[k]);
        __nv_bfloat162 const* w2 = reinterpret_cast<__nv_bfloat162 const*>(&w_vec);
        __nv_bfloat162 const* x2 = reinterpret_cast<__nv_bfloat162 const*>(&x_vec);
        __nv_bfloat162 chunk_acc = __float2bfloat162_rn(0.0f);
#pragma unroll
        for (int p = 0; p < 4; p++)
        {
            chunk_acc = __hfma2(w2[p], x2[p], chunk_acc);
        }
        float2 const chunk = __bfloat1622float2(chunk_acc);
        sum += chunk.x + chunk.y;
    }

    for (int k = full_dim + lane_id * 2; k + 1 < dim; k += WARP_SIZE * 2)
    {
        __nv_bfloat162 const w2 = *reinterpret_cast<__nv_bfloat162 const*>(&weight_row[k]);
        __nv_bfloat162 const x2 = *reinterpret_cast<__nv_bfloat162 const*>(&input[k]);
        float2 const prod = __bfloat1622float2(__hmul2(w2, x2));
        sum += prod.x + prod.y;
    }

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

__device__ __forceinline__ float sum_moe_slots(float const* moeSlotBase)
{
    float4 const a = reinterpret_cast<float4 const*>(moeSlotBase)[0];
    float4 const b = reinterpret_cast<float4 const*>(moeSlotBase)[1];
    return (a.x + a.y + a.z + a.w) + (b.x + b.y + b.z + b.w);
}

__device__ __forceinline__ float load_moe_merged(Gemma4Globals const& gg, int token_id, int h)
{
    if (gg.moe_merged_scratch)
    {
        return gg.moe_merged_scratch[(int64_t) token_id * HIDDEN_SIZE + h];
    }
    float const* moeSlotBase = gg.moe_scratch + ((int64_t) token_id * HIDDEN_SIZE + h) * MOE_TOPK;
    return sum_moe_slots(moeSlotBase);
}

__device__ __forceinline__ void topk_insert(float value, int index, float* topValues, int* topIndices)
{
    if (value <= topValues[MOE_TOPK - 1])
    {
        return;
    }

    int pos = MOE_TOPK - 1;
    while (pos > 0 && value > topValues[pos - 1])
    {
        topValues[pos] = topValues[pos - 1];
        topIndices[pos] = topIndices[pos - 1];
        --pos;
    }
    topValues[pos] = value;
    topIndices[pos] = index;
}

__device__ __forceinline__ float block_max(float value, float* s_reduce, int warp_id, int lane_id)
{
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
    }
    if (lane_id == 0)
    {
        s_reduce[warp_id] = value;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        float val = lane_id < NUM_WARPS ? s_reduce[lane_id] : -3.402823e38F;
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (lane_id == 0)
        {
            s_reduce[0] = val;
        }
    }
    __syncthreads();
    return s_reduce[0];
}

__device__ void handle_ffn_gateup(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const row_start = instr.field(2);
    int const row_end = instr.field(3);
    int const token_id = instr.field(4);
    int const num_rows = row_end - row_start;
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DATA_OFFSET);
    float const* pre_ffn = gg.b_pre_ffn_in + (int64_t) token_id * HIDDEN_SIZE;
    __nv_bfloat162* s_input2 = reinterpret_cast<__nv_bfloat162*>(s_input);

    for (int i = threadIdx.x; i < HIDDEN_SIZE / 2; i += THREADS_PER_BLOCK)
    {
        int const k = 2 * i;
        s_input2[i] = __floats2bfloat162_rn(pre_ffn[k], pre_ffn[k + 1]);
    }
    __syncthreads();

    if (warp_id >= FFN_MATH_WARPS)
    {
        return;
    }

    int const rows_per_warp = (num_rows + FFN_MATH_WARPS - 1) / FFN_MATH_WARPS;
    int const warp_row_start = row_start + warp_id * rows_per_warp;
    int const warp_row_end = min(warp_row_start + rows_per_warp, row_end);

    for (int row = warp_row_start; row < warp_row_end; row++)
    {
        __nv_bfloat16 const* w = gg.ffn_gate_up_weight + (int64_t) row * HIDDEN_SIZE;
        float const sum = warp_dot_product_bf16(w, s_input, HIDDEN_SIZE, lane_id);
        if (lane_id == 0)
        {
            int const out_idx = row < FFN_INTERMEDIATE ? row : row - FFN_INTERMEDIATE;
            __nv_bfloat16* out = row < FFN_INTERMEDIATE ? gg.ffn_gate_scratch : gg.ffn_up_scratch;
            out[(int64_t) token_id * FFN_INTERMEDIATE + out_idx] = __float2bfloat16(sum);
        }
    }
}

__device__ void handle_ffn_down(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const row_start = instr.field(2);
    int const row_end = instr.field(3);
    int const token_id = instr.field(4);
    int const num_rows = row_end - row_start;
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_act = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DATA_OFFSET);
    __nv_bfloat16 const* gate = gg.ffn_gate_scratch + (int64_t) token_id * FFN_INTERMEDIATE;
    __nv_bfloat16 const* up = gg.ffn_up_scratch + (int64_t) token_id * FFN_INTERMEDIATE;
    __nv_bfloat162* s_act2 = reinterpret_cast<__nv_bfloat162*>(s_act);

    for (int i = threadIdx.x; i < FFN_INTERMEDIATE / 2; i += THREADS_PER_BLOCK)
    {
        int const k = 2 * i;
        float const act0 = gelu_tanh_approx(__bfloat162float(gate[k])) * __bfloat162float(up[k]);
        float const act1 = gelu_tanh_approx(__bfloat162float(gate[k + 1])) * __bfloat162float(up[k + 1]);
        s_act2[i] = __floats2bfloat162_rn(act0, act1);
    }
    __syncthreads();

    if (warp_id >= FFN_MATH_WARPS)
    {
        return;
    }

    int const rows_per_warp = (num_rows + FFN_MATH_WARPS - 1) / FFN_MATH_WARPS;
    int const warp_row_start = row_start + warp_id * rows_per_warp;
    int const warp_row_end = min(warp_row_start + rows_per_warp, row_end);

    for (int row = warp_row_start; row < warp_row_end; row++)
    {
        __nv_bfloat16 const* w = gg.ffn_down_weight + (int64_t) row * FFN_INTERMEDIATE;
        float const sum = warp_dot_product_bf16(w, s_act, FFN_INTERMEDIATE, lane_id);
        if (lane_id == 0)
        {
            gg.ffn_down_scratch[(int64_t) token_id * HIDDEN_SIZE + row] = sum;
        }
    }
}

__device__ void handle_b_post(MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const token_id = instr.field(1);
    int const mode = instr.field(2);
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;
    int const tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float* s_reduce = reinterpret_cast<float*>(smem_raw + SMEM_DATA_OFFSET);
    float* s_dense_norm = s_reduce + NUM_WARPS;
    float* s_moe_norm = s_dense_norm + HIDDEN_SIZE;
    float* s_hidden = s_moe_norm + HIDDEN_SIZE;

    float const* dense_down = gg.ffn_down_scratch + (int64_t) token_id * HIDDEN_SIZE;
    float const* post_attn = gg.b_post_attn_in + (int64_t) token_id * HIDDEN_SIZE;
    bool const hasMoe = (gg.moe_scratch || gg.moe_merged_scratch) && gg.post_ffn2_norm_weight;

    if (mode == 0)
    {
        float sumsq_dense = 0.0f;
        for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
        {
            float const v = dense_down[h];
            sumsq_dense += v * v;
        }
        float const inv_rms_dense = rsqrtf(block_sum(sumsq_dense, s_reduce, warp_id, lane_id) / HIDDEN_SIZE + gg.eps);

        float sumsq_ffn = 0.0f;
        for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
        {
            float const ffn_val = dense_down[h] * inv_rms_dense * gg.post_ffn1_norm_weight[h];
            s_dense_norm[h] = ffn_val;
        }

        if (hasMoe)
        {
            float sumsq_moe = 0.0f;
            for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
            {
                float const moe = load_moe_merged(gg, token_id, h);
                sumsq_moe += moe * moe;
            }
            float const inv_rms_moe = rsqrtf(block_sum(sumsq_moe, s_reduce, warp_id, lane_id) / HIDDEN_SIZE + gg.eps);
            for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
            {
                float const moe = load_moe_merged(gg, token_id, h);
                s_moe_norm[h] = moe * inv_rms_moe * gg.post_ffn2_norm_weight[h];
            }
        }
        __syncthreads();

        for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
        {
            float const ffn_val = hasMoe ? (s_dense_norm[h] + s_moe_norm[h]) : s_dense_norm[h];
            sumsq_ffn += ffn_val * ffn_val;
        }
        float const inv_rms_ffn = rsqrtf(block_sum(sumsq_ffn, s_reduce, warp_id, lane_id) / HIDDEN_SIZE + gg.eps);

        float sumsq_hidden = 0.0f;
        for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
        {
            float const merged_ffn = hasMoe ? (s_dense_norm[h] + s_moe_norm[h]) : s_dense_norm[h];
            float const merged = merged_ffn * inv_rms_ffn * gg.post_ffn_norm_weight[h];
            float const layer_scale = gg.layer_scalar[gg.layer_scalar_size == 1 ? 0 : h];
            float const hidden = (post_attn[h] + merged) * layer_scale;
            s_hidden[h] = hidden;
            sumsq_hidden += hidden * hidden;
        }
        float const inv_rms_hidden = rsqrtf(block_sum(sumsq_hidden, s_reduce, warp_id, lane_id) / HIDDEN_SIZE + gg.eps);

        for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
        {
            float const hidden = s_hidden[h];
            gg.hidden_out[(int64_t) token_id * HIDDEN_SIZE + h] = hidden;
            gg.next_attn_normed_out[(int64_t) token_id * HIDDEN_SIZE + h]
                = hidden * inv_rms_hidden * gg.next_input_norm_weight[h];
        }
        return;
    }

    int const row_start = instr.field(3);
    int const row_end = instr.field(4);
    int const num_partials = instr.field(5);
    float* scratch = reinterpret_cast<float*>(mg.debug_output + mg.num_sms);
    float* dense_partials = scratch;
    float* moe_partials = scratch + num_partials;
    float* ffn_partials = scratch + 2 * num_partials;
    float* hidden_partials = scratch + 3 * num_partials;
    float* moe_merged_tmp
        = (gg.moe_merged_scratch ? gg.moe_merged_scratch : gg.next_attn_normed_out) + (int64_t) token_id * HIDDEN_SIZE;

    if (mode == 1)
    {
        float dense_partial = 0.0f;
        float moe_partial = 0.0f;
        for (int h = row_start + tid; h < row_end; h += THREADS_PER_BLOCK)
        {
            float const v = dense_down[h];
            dense_partial += v * v;
            if (hasMoe)
            {
                float const moe = load_moe_merged(gg, token_id, h);
                moe_partial += moe * moe;
                moe_merged_tmp[h] = moe;
            }
        }
        float const dense_total = block_sum(dense_partial, s_reduce, warp_id, lane_id);
        float const moe_total = block_sum(moe_partial, s_reduce, warp_id, lane_id);
        if (tid == 0)
        {
            dense_partials[sm_id] = dense_total;
            moe_partials[sm_id] = moe_total;
        }
        return;
    }

    if (mode == 2)
    {
        float dense_partial = 0.0f;
        float moe_partial = 0.0f;
        for (int i = tid; i < num_partials; i += THREADS_PER_BLOCK)
        {
            dense_partial += dense_partials[i];
            moe_partial += moe_partials[i];
        }
        float const total_dense = block_sum(dense_partial, s_reduce, warp_id, lane_id);
        float const inv_rms_dense = rsqrtf(total_dense / HIDDEN_SIZE + gg.eps);
        float inv_rms_moe = 0.0f;
        if (hasMoe)
        {
            float const total_moe = block_sum(moe_partial, s_reduce, warp_id, lane_id);
            inv_rms_moe = rsqrtf(total_moe / HIDDEN_SIZE + gg.eps);
        }

        float sumsq_ffn = 0.0f;
        for (int h = row_start + tid; h < row_end; h += THREADS_PER_BLOCK)
        {
            float const dense_norm = dense_down[h] * inv_rms_dense * gg.post_ffn1_norm_weight[h];
            float moe_norm = 0.0f;
            if (hasMoe)
            {
                moe_norm = moe_merged_tmp[h] * inv_rms_moe * gg.post_ffn2_norm_weight[h];
            }
            float const merged_ffn = dense_norm + moe_norm;
            gg.hidden_out[(int64_t) token_id * HIDDEN_SIZE + h] = merged_ffn;
            sumsq_ffn += merged_ffn * merged_ffn;
        }
        float const ffn_total = block_sum(sumsq_ffn, s_reduce, warp_id, lane_id);
        if (tid == 0)
        {
            ffn_partials[sm_id] = ffn_total;
        }
        return;
    }

    if (mode == 3)
    {
        float partial = 0.0f;
        for (int i = tid; i < num_partials; i += THREADS_PER_BLOCK)
        {
            partial += ffn_partials[i];
        }
        float const total_ffn = block_sum(partial, s_reduce, warp_id, lane_id);
        float const inv_rms_ffn = rsqrtf(total_ffn / HIDDEN_SIZE + gg.eps);

        float sumsq_hidden = 0.0f;
        for (int h = row_start + tid; h < row_end; h += THREADS_PER_BLOCK)
        {
            float const merged
                = gg.hidden_out[(int64_t) token_id * HIDDEN_SIZE + h] * inv_rms_ffn * gg.post_ffn_norm_weight[h];
            float const layer_scale = gg.layer_scalar[gg.layer_scalar_size == 1 ? 0 : h];
            float const hidden = (post_attn[h] + merged) * layer_scale;
            gg.hidden_out[(int64_t) token_id * HIDDEN_SIZE + h] = hidden;
            sumsq_hidden += hidden * hidden;
        }
        float const hidden_total = block_sum(sumsq_hidden, s_reduce, warp_id, lane_id);
        if (tid == 0)
        {
            hidden_partials[sm_id] = hidden_total;
        }
        return;
    }

    float partial = 0.0f;
    for (int i = tid; i < num_partials; i += THREADS_PER_BLOCK)
    {
        partial += hidden_partials[i];
    }
    float const total_hidden = block_sum(partial, s_reduce, warp_id, lane_id);
    float const inv_rms_hidden = rsqrtf(total_hidden / HIDDEN_SIZE + gg.eps);
    for (int h = row_start + tid; h < row_end; h += THREADS_PER_BLOCK)
    {
        float const hidden = gg.hidden_out[(int64_t) token_id * HIDDEN_SIZE + h];
        gg.hidden_out[(int64_t) token_id * HIDDEN_SIZE + h] = hidden;
        gg.next_attn_normed_out[(int64_t) token_id * HIDDEN_SIZE + h]
            = hidden * inv_rms_hidden * gg.next_input_norm_weight[h];
    }
}

__device__ void handle_router_topk(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    (void) mg;
    (void) sm_id;

    int const token_id = instr.field(2);
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;
    int const tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float* s_reduce = reinterpret_cast<float*>(smem_raw + SMEM_DATA_OFFSET);
    __nv_bfloat16* s_router_input = reinterpret_cast<__nv_bfloat16*>(s_reduce + NUM_WARPS);
    float* s_logits = reinterpret_cast<float*>(s_router_input + HIDDEN_SIZE);
    float const* post_attn = gg.b_post_attn_in + (int64_t) token_id * HIDDEN_SIZE;

    float sumsq_hidden = 0.0f;
    for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
    {
        float const v = post_attn[h];
        sumsq_hidden += v * v;
    }
    float const inv_rms_hidden = rsqrtf(block_sum(sumsq_hidden, s_reduce, warp_id, lane_id) / HIDDEN_SIZE + gg.eps);
    float const router_root_size = gg.router_root_size[0];

    for (int h = tid; h < HIDDEN_SIZE; h += THREADS_PER_BLOCK)
    {
        float const base = post_attn[h] * inv_rms_hidden;
        s_router_input[h] = __float2bfloat16(base * router_root_size * gg.router_scale[h]);
        if (gg.moe_input_scratch && gg.pre_ffn2_norm_weight)
        {
            gg.moe_input_scratch[(int64_t) token_id * HIDDEN_SIZE + h]
                = __float2bfloat16(base * gg.pre_ffn2_norm_weight[h]);
        }
    }
    __syncthreads();

    if (warp_id < NUM_CONSUMERS)
    {
        constexpr int kExpertsPerWarp = (NUM_EXPERTS + NUM_CONSUMERS - 1) / NUM_CONSUMERS;
        int const expert_start = warp_id * kExpertsPerWarp;
        int const expert_end = min(expert_start + kExpertsPerWarp, NUM_EXPERTS);
        for (int expert = expert_start; expert < expert_end; ++expert)
        {
            __nv_bfloat16 const* w = gg.router_proj_weight + (int64_t) expert * HIDDEN_SIZE;
            float const logit = warp_dot_product_bf16(w, s_router_input, HIDDEN_SIZE, lane_id);
            if (lane_id == 0)
            {
                s_logits[expert] = logit;
            }
        }
    }
    __syncthreads();

    if (tid == 0)
    {
        float topValues[MOE_TOPK];
        int topIndices[MOE_TOPK];
#pragma unroll
        for (int i = 0; i < MOE_TOPK; ++i)
        {
            topValues[i] = -3.402823e38F;
            topIndices[i] = -1;
        }

        for (int expert = 0; expert < NUM_EXPERTS; ++expert)
        {
            topk_insert(s_logits[expert], expert, topValues, topIndices);
        }

        float const maxLogit = topValues[0];
        float weightSum = 0.0f;
#pragma unroll
        for (int i = 0; i < MOE_TOPK; ++i)
        {
            topValues[i] = expf(topValues[i] - maxLogit);
            weightSum += topValues[i];
        }
        float const invWeightSum = 1.0f / weightSum;
        for (int i = 0; i < MOE_TOPK; ++i)
        {
            gg.router_topk_weights[(int64_t) token_id * MOE_TOPK + i] = topValues[i] * invWeightSum;
            gg.router_topk_indices[(int64_t) token_id * MOE_TOPK + i] = topIndices[i];
        }
    }
}

__device__ void handle_moe(MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    (void) mg;
    (void) sm_id;

    int const mode = instr.field(1);
    int const slot = instr.field(2);
    int const row_start = instr.field(3);
    int const row_end = instr.field(4);
    int const token_id = instr.field(5);
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ char smem_raw[];
    float* s_reduce = reinterpret_cast<float*>(smem_raw + SMEM_DATA_OFFSET);
    __nv_bfloat16* s_input = reinterpret_cast<__nv_bfloat16*>(s_reduce + NUM_WARPS);
    __nv_bfloat16* s_gate = s_input + HIDDEN_SIZE;
    __nv_bfloat16* s_up = s_gate + EXPERT_INTERMEDIATE;
    __nv_bfloat16* s_act = s_up + EXPERT_INTERMEDIATE;
    __nv_bfloat162* s_input2 = reinterpret_cast<__nv_bfloat162*>(s_input);
    __nv_bfloat162* s_gate2 = reinterpret_cast<__nv_bfloat162*>(s_gate);
    __nv_bfloat162* s_up2 = reinterpret_cast<__nv_bfloat162*>(s_up);
    __nv_bfloat162* s_act2 = reinterpret_cast<__nv_bfloat162*>(s_act);

    __nv_bfloat16 const* moe_input = gg.moe_input_scratch + (int64_t) token_id * HIDDEN_SIZE;
    __nv_bfloat162 const* moe_input2 = reinterpret_cast<__nv_bfloat162 const*>(moe_input);
    for (int i = threadIdx.x; i < HIDDEN_SIZE / 2; i += THREADS_PER_BLOCK)
    {
        s_input2[i] = moe_input2[i];
    }
    __syncthreads();

    int const expert = gg.router_topk_indices[(int64_t) token_id * MOE_TOPK + slot];
    float const routeWeight = gg.router_topk_weights[(int64_t) token_id * MOE_TOPK + slot];
    if (expert < 0)
    {
        return;
    }

    if (mode == 0)
    {
        if (warp_id >= MOE_FC1_MATH_WARPS)
        {
            return;
        }
        for (int row = row_start + warp_id; row < row_end; row += MOE_FC1_MATH_WARPS)
        {
            __nv_bfloat16 const* w13
                = gg.moe_w13_stacked_weight + ((int64_t) expert * (2 * EXPERT_INTERMEDIATE) + row) * HIDDEN_SIZE;
            float const sum = warp_dot_product_bf16(w13, s_input, HIDDEN_SIZE, lane_id);
            if (lane_id == 0)
            {
                int const outIdx = row < EXPERT_INTERMEDIATE ? row : row - EXPERT_INTERMEDIATE;
                __nv_bfloat16* out = row < EXPERT_INTERMEDIATE
                    ? gg.moe_gate_scratch + ((int64_t) token_id * MOE_TOPK + slot) * EXPERT_INTERMEDIATE
                    : gg.moe_up_scratch + ((int64_t) token_id * MOE_TOPK + slot) * EXPERT_INTERMEDIATE;
                out[outIdx] = __float2bfloat16(sum);
            }
        }
        return;
    }

    if (mode == 2)
    {
        float* s_moe_acc = reinterpret_cast<float*>(s_act + EXPERT_INTERMEDIATE);
        int const row_count = row_end - row_start;
        for (int i = threadIdx.x; i < row_count; i += THREADS_PER_BLOCK)
        {
            s_moe_acc[i] = 0.0f;
        }
        __syncthreads();

        for (int slot_idx = 0; slot_idx < MOE_TOPK; ++slot_idx)
        {
            int const slot_expert = gg.router_topk_indices[(int64_t) token_id * MOE_TOPK + slot_idx];
            if (slot_expert < 0)
            {
                continue;
            }
            float const slot_route_weight = gg.router_topk_weights[(int64_t) token_id * MOE_TOPK + slot_idx];
            __nv_bfloat16 const* slot_gate
                = gg.moe_gate_scratch + ((int64_t) token_id * MOE_TOPK + slot_idx) * EXPERT_INTERMEDIATE;
            __nv_bfloat16 const* slot_up
                = gg.moe_up_scratch + ((int64_t) token_id * MOE_TOPK + slot_idx) * EXPERT_INTERMEDIATE;
            __nv_bfloat162 const* slot_gate2 = reinterpret_cast<__nv_bfloat162 const*>(slot_gate);
            __nv_bfloat162 const* slot_up2 = reinterpret_cast<__nv_bfloat162 const*>(slot_up);

            for (int i = threadIdx.x; i < EXPERT_INTERMEDIATE / 2; i += THREADS_PER_BLOCK)
            {
                s_gate2[i] = slot_gate2[i];
                s_up2[i] = slot_up2[i];
            }
            __syncthreads();

            for (int i = threadIdx.x; i < EXPERT_INTERMEDIATE / 2; i += THREADS_PER_BLOCK)
            {
                int const k = 2 * i;
                float const act0 = gelu_tanh_approx(__bfloat162float(s_gate[k])) * __bfloat162float(s_up[k]);
                float const act1 = gelu_tanh_approx(__bfloat162float(s_gate[k + 1])) * __bfloat162float(s_up[k + 1]);
                s_act2[i] = __floats2bfloat162_rn(act0, act1);
            }
            __syncthreads();

            if (warp_id < MOE_FC2_MATH_WARPS)
            {
                for (int row = row_start + warp_id; row < row_end; row += MOE_FC2_MATH_WARPS)
                {
                    __nv_bfloat16 const* w2
                        = gg.moe_w2_weight + ((int64_t) slot_expert * HIDDEN_SIZE + row) * EXPERT_INTERMEDIATE;
                    float const sum = warp_dot_product_bf16(w2, s_act, EXPERT_INTERMEDIATE, lane_id);
                    if (lane_id == 0)
                    {
                        s_moe_acc[row - row_start] += sum * slot_route_weight;
                    }
                }
            }
            __syncthreads();
        }

        for (int i = threadIdx.x; i < row_count; i += THREADS_PER_BLOCK)
        {
            gg.moe_merged_scratch[(int64_t) token_id * HIDDEN_SIZE + row_start + i] = s_moe_acc[i];
        }
        return;
    }

    __nv_bfloat16 const* gate = gg.moe_gate_scratch + ((int64_t) token_id * MOE_TOPK + slot) * EXPERT_INTERMEDIATE;
    __nv_bfloat16 const* up = gg.moe_up_scratch + ((int64_t) token_id * MOE_TOPK + slot) * EXPERT_INTERMEDIATE;
    __nv_bfloat162 const* gate2 = reinterpret_cast<__nv_bfloat162 const*>(gate);
    __nv_bfloat162 const* up2 = reinterpret_cast<__nv_bfloat162 const*>(up);
    for (int i = threadIdx.x; i < EXPERT_INTERMEDIATE / 2; i += THREADS_PER_BLOCK)
    {
        s_gate2[i] = gate2[i];
        s_up2[i] = up2[i];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < EXPERT_INTERMEDIATE / 2; i += THREADS_PER_BLOCK)
    {
        int const k = 2 * i;
        float const act0 = gelu_tanh_approx(__bfloat162float(s_gate[k])) * __bfloat162float(s_up[k]);
        float const act1 = gelu_tanh_approx(__bfloat162float(s_gate[k + 1])) * __bfloat162float(s_up[k + 1]);
        s_act2[i] = __floats2bfloat162_rn(act0, act1);
    }
    __syncthreads();

    if (warp_id >= MOE_FC2_MATH_WARPS)
    {
        return;
    }
    for (int row = row_start + warp_id; row < row_end; row += MOE_FC2_MATH_WARPS)
    {
        __nv_bfloat16 const* w2 = gg.moe_w2_weight + ((int64_t) expert * HIDDEN_SIZE + row) * EXPERT_INTERMEDIATE;
        float const sum = warp_dot_product_bf16(w2, s_act, EXPERT_INTERMEDIATE, lane_id);
        if (lane_id == 0)
        {
            gg.moe_scratch[((int64_t) token_id * HIDDEN_SIZE + row) * MOE_TOPK + slot] = sum * routeWeight;
        }
    }
}

} // namespace gemma4_megakernel
