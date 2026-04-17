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

// Opcode: OP_PAGED_ATTN — Multi-SM paged decode attention with partial outputs.
//
// Each SM handles one KV head over a RANGE of pages (not all pages).
// Multiple SMs can work on the same KV head's different page ranges.
// When num_partials > 1 per head, a reduction step (OP_ATTN_REDUCE) combines them.
//
// Instruction encoding:
//   word[0] = OP_PAGED_ATTN (3)
//   word[1] = kv_head_idx
//   word[2] = token_id
//   word[3] = page_range_start (first page index to process, relative to sequence start)
//   word[4] = page_range_end   (exclusive)
//   word[5] = partial_idx      (which partial slot this SM writes to)
//   word[6] = is_single_partial (1 if only one partial → write directly to attn_scratch)
//
// If is_single_partial: writes final output to attn_scratch (no reduction needed)
// Otherwise: writes partial O + LSE to partial_attn_scratch for later reduction

#pragma once

#include "../gemma4_config.cuh"
#include "../gemma4_globals.cuh"
#include "../megakernel_framework.cuh"

namespace gemma4_megakernel
{

static constexpr int ELEMS_PER_LANE = HEAD_DIM / WARP_SIZE; // 8

__device__ void handle_paged_attn(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const kv_head_idx = instr.field(1);
    int const token_id = instr.field(2);
    int const page_range_start = instr.field(3);
    int const page_range_end = instr.field(4);
    int const partial_idx = instr.field(5);
    int const is_single = instr.field(6);

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id < FIRST_CONSUMER)
        return;
    int const consumer_id = warp_id - FIRST_CONSUMER;

    // Each consumer warp handles one Q head in the GQA group
    if (consumer_id >= GQA_RATIO)
        return;
    int const q_head_idx = kv_head_idx * GQA_RATIO + consumer_id;

    // Load Q vector into registers
    float q_regs[ELEMS_PER_LANE];
    {
        __nv_bfloat16 const* q_src = gg.qkv_scratch + (int64_t) token_id * QKV_WIDTH + (int64_t) q_head_idx * HEAD_DIM;
#pragma unroll
        for (int di = 0; di < ELEMS_PER_LANE; di++)
        {
            q_regs[di] = __bfloat162float(q_src[lane_id + di * WARP_SIZE]);
        }
    }

    // Page table metadata
    int batch_idx = gg.triton_batch_indices[token_id];
    int page_start = gg.cu_num_pages[batch_idx];
    int total_pages = gg.cu_num_pages[batch_idx + 1] - page_start;
    int last_pl = gg.last_page_len[batch_idx];

    // Online softmax state
    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc[ELEMS_PER_LANE];
#pragma unroll
    for (int di = 0; di < ELEMS_PER_LANE; di++)
        acc[di] = 0.0f;

    // Iterate over assigned page range
    for (int page_num = page_range_start; page_num < page_range_end; page_num++)
    {
        if (page_num >= total_pages)
            break;
        int phys_page = gg.cache_loc[page_start + page_num];
        int valid_tokens = (page_num == total_pages - 1) ? last_pl : gg.page_size;

        int64_t k_base = (int64_t) phys_page * gg.cache_stride_block + 0 * gg.cache_stride_kv
            + (int64_t) kv_head_idx * gg.cache_stride_head;
        int64_t v_base = (int64_t) phys_page * gg.cache_stride_block + 1 * gg.cache_stride_kv
            + (int64_t) kv_head_idx * gg.cache_stride_head;

        for (int t = 0; t < valid_tokens; t++)
        {
            int64_t k_off = k_base + (int64_t) t * gg.cache_stride_token;
            int64_t v_off = v_base + (int64_t) t * gg.cache_stride_token;

            // Q · K^T
            float dot = 0.0f;
#pragma unroll
            for (int di = 0; di < ELEMS_PER_LANE; di++)
            {
                dot += q_regs[di] * __bfloat162float(gg.kv_cache[k_off + lane_id + di * WARP_SIZE]);
            }
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }
            float score = __shfl_sync(0xFFFFFFFF, dot, 0) * gg.attn_scale;

            // Online softmax
            float new_m = fmaxf(m_i, score);
            float alpha = expf(m_i - new_m);
            float p = expf(score - new_m);

#pragma unroll
            for (int di = 0; di < ELEMS_PER_LANE; di++)
            {
                acc[di] = acc[di] * alpha + p * __bfloat162float(gg.kv_cache[v_off + lane_id + di * WARP_SIZE]);
            }
            l_i = l_i * alpha + p;
            m_i = new_m;
        }
    }

    if (is_single)
    {
        // Single partial → write final normalized output directly
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        __nv_bfloat16* out = gg.attn_scratch + (int64_t) token_id * Q_WIDTH + (int64_t) q_head_idx * HEAD_DIM;
#pragma unroll
        for (int di = 0; di < ELEMS_PER_LANE; di++)
        {
            out[lane_id + di * WARP_SIZE] = __float2bfloat16(acc[di] * inv_l);
        }
    }
    else
    {
        // Multiple partials → write unnormalized O, m, and l for reduction
        // Layout: partial_attn_scratch[partial_idx, q_head, HEAD_DIM + 2]
        // HEAD_DIM+0 = m_i (running max), HEAD_DIM+1 = l_i (running sum)
        float* partial_out = gg.partial_attn_scratch + (int64_t) partial_idx * NUM_Q_HEADS * (HEAD_DIM + 2)
            + (int64_t) q_head_idx * (HEAD_DIM + 2);
#pragma unroll
        for (int di = 0; di < ELEMS_PER_LANE; di++)
        {
            partial_out[lane_id + di * WARP_SIZE] = acc[di];
        }
        if (lane_id == 0)
        {
            partial_out[HEAD_DIM] = m_i;
            partial_out[HEAD_DIM + 1] = l_i;
        }
    }

    // Flush writes for cross-SM visibility after barrier
    __threadfence();
}

// ──────────────────────────────────────────────────────────────
// OP_ATTN_REDUCE — Combine partial attention outputs via LSE correction
//
// Instruction encoding:
//   word[0] = OP_ATTN_REDUCE (4)
//   word[1] = kv_head_idx
//   word[2] = token_id
//   word[3] = num_partials
// ──────────────────────────────────────────────────────────────
__device__ void handle_attn_reduce(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const kv_head_idx = instr.field(1);
    int const token_id = instr.field(2);
    int const num_partials = instr.field(3);

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id < FIRST_CONSUMER)
        return;
    int const consumer_id = warp_id - FIRST_CONSUMER;
    if (consumer_id >= GQA_RATIO)
        return;

    int const q_head_idx = kv_head_idx * GQA_RATIO + consumer_id;

    // Accumulate partials with LSE correction
    float m_acc = -1e30f;
    float l_acc = 0.0f;
    float o_acc[ELEMS_PER_LANE];
#pragma unroll
    for (int di = 0; di < ELEMS_PER_LANE; di++)
        o_acc[di] = 0.0f;

    // word[4] = partial_start (first partial index for this KV head)
    int const partial_start = instr.field(4);

    for (int pi = 0; pi < num_partials; pi++)
    {
        float const* partial = gg.partial_attn_scratch + (int64_t) (partial_start + pi) * NUM_Q_HEADS * (HEAD_DIM + 2)
            + (int64_t) q_head_idx * (HEAD_DIM + 2);

        // Load m_i and l_i from the partial
        float m_i = 0.0f, l_i = 0.0f;
        if (lane_id == 0)
        {
            m_i = partial[HEAD_DIM];
            l_i = partial[HEAD_DIM + 1];
        }
        m_i = __shfl_sync(0xFFFFFFFF, m_i, 0);
        l_i = __shfl_sync(0xFFFFFFFF, l_i, 0);

        // Skip empty partials
        if (l_i <= 0.0f)
            continue;

        float new_m = fmaxf(m_acc, m_i);
        float alpha = expf(m_acc - new_m); // rescale existing accumulator
        float beta = expf(m_i - new_m);    // rescale this partial

// O_partial is unnormalized: contains Σ exp(score - m_i) * V
// Rescale by exp(m_i - new_m) to bring to common base
#pragma unroll
        for (int di = 0; di < ELEMS_PER_LANE; di++)
        {
            float o_i = partial[lane_id + di * WARP_SIZE];
            o_acc[di] = o_acc[di] * alpha + o_i * beta;
        }
        l_acc = l_acc * alpha + l_i * beta;
        m_acc = new_m;
    }

    // Normalize and write final output
    float inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
    __nv_bfloat16* out = gg.attn_scratch + (int64_t) token_id * Q_WIDTH + (int64_t) q_head_idx * HEAD_DIM;
#pragma unroll
    for (int di = 0; di < ELEMS_PER_LANE; di++)
    {
        out[lane_id + di * WARP_SIZE] = __float2bfloat16(o_acc[di] * inv_l);
    }
}

} // namespace gemma4_megakernel
