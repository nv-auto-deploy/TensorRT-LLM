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
static constexpr int NUM_LOADER_THREADS = FIRST_CONSUMER * WARP_SIZE;
static constexpr int CP_ASYNC_BYTES = 16;
static constexpr int ELEMS_PER_CP_ASYNC = CP_ASYNC_BYTES / sizeof(__nv_bfloat16);
static constexpr int ATTN_WARPS_PER_Q_HEAD = NUM_CONSUMERS / GQA_RATIO;         // 8
static constexpr int REDUCE_WARPS_PER_Q_HEAD = NUM_CONSUMERS / GQA_RATIO;       // 8
static constexpr int REDUCE_DIMS_PER_WARP = HEAD_DIM / REDUCE_WARPS_PER_Q_HEAD; // 32

struct PageStage
{
    bool valid;
    int validTokens;
};

__device__ __forceinline__ void cp_async_16(void* smem_ptr, void const* gmem_ptr)
{
    unsigned const smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all()
{
    asm volatile("cp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ PageStage stage_kv_page_async(Gemma4Globals const& gg, int kv_head_idx, int page_start,
    int total_pages, int last_pl, int page_num, int earliest_visible_page, bool use_sliding_window, int loader_tid,
    bool is_loader, __nv_bfloat16* s_k, __nv_bfloat16* s_v, int page_elems)
{
    PageStage stage{false, 0};
    bool const page_in_range = page_num < total_pages && !(use_sliding_window && page_num < earliest_visible_page);
    if (!page_in_range)
    {
        return stage;
    }

    stage.valid = true;
    stage.validTokens = (page_num == total_pages - 1) ? last_pl : gg.page_size;

    if (!is_loader)
    {
        return stage;
    }

    int const phys_page = gg.cache_loc[page_start + page_num];
    int64_t const k_base = (int64_t) phys_page * gg.cache_stride_block + 0 * gg.cache_stride_kv
        + (int64_t) kv_head_idx * gg.cache_stride_head;
    int64_t const v_base = (int64_t) phys_page * gg.cache_stride_block + 1 * gg.cache_stride_kv
        + (int64_t) kv_head_idx * gg.cache_stride_head;
    int const num_vecs = page_elems / ELEMS_PER_CP_ASYNC;

    for (int vec = loader_tid; vec < num_vecs; vec += NUM_LOADER_THREADS)
    {
        int const elem = vec * ELEMS_PER_CP_ASYNC;
        int const token_in_page = elem / HEAD_DIM;
        int const dim = elem - token_in_page * HEAD_DIM;
        int64_t const cache_off = (int64_t) token_in_page * gg.cache_stride_token + dim;
        cp_async_16(s_k + elem, gg.kv_cache + k_base + cache_off);
        cp_async_16(s_v + elem, gg.kv_cache + v_base + cache_off);
    }
    cp_async_commit();
    return stage;
}

__device__ void handle_paged_attn(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const kv_head_idx = instr.field(1);
    int const token_id = instr.field(2);
    int const page_range_start = instr.field(3);
    int const page_range_end = instr.field(4);
    int const partial_idx = instr.field(5);
    int const is_single = instr.field(6);
    int const sliding_window = instr.field(7); // 0 = full attention, >0 = window size

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    bool const is_loader = warp_id < FIRST_CONSUMER;
    int const loader_tid = threadIdx.x;
    int const consumer_id = warp_id - FIRST_CONSUMER;
    bool const is_consumer = warp_id >= FIRST_CONSUMER && consumer_id < NUM_CONSUMERS;
    int const q_head_group = is_consumer ? (consumer_id / ATTN_WARPS_PER_Q_HEAD) : 0;
    int const consumer_group_lane = is_consumer ? (consumer_id % ATTN_WARPS_PER_Q_HEAD) : 0;
    int const q_head_idx = kv_head_idx * GQA_RATIO + q_head_group;

    // Load Q vector into registers
    float q_regs[ELEMS_PER_LANE];
    if (is_consumer)
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

    // Sliding window: compute the earliest visible token position.
    // Tokens before earliest_visible are masked out (score = -inf → skipped).
    int position = gg.triton_positions[token_id];
    int earliest_visible = 0;
    int earliest_visible_page = 0;
    if (sliding_window > 0)
    {
        earliest_visible = max(0, position - sliding_window + 1);
        earliest_visible_page = earliest_visible / gg.page_size;
    }

    // Online softmax state
    float m_i = -1e30f;
    float l_i = 0.0f;
    float acc[ELEMS_PER_LANE];
#pragma unroll
    for (int di = 0; di < ELEMS_PER_LANE; di++)
        acc[di] = 0.0f;

    // Stage one K/V page in shared memory so both Q-head consumer warps
    // reuse the same loads instead of fetching K/V independently.
    extern __shared__ char smem_raw[];
    __nv_bfloat16* s_page = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DATA_OFFSET);
    int const page_elems = gg.page_size * HEAD_DIM;
    __nv_bfloat16* s_k[2] = {s_page, s_page + 2 * page_elems};
    __nv_bfloat16* s_v[2] = {s_page + page_elems, s_page + 3 * page_elems};
    float* s_consumer_partials = reinterpret_cast<float*>(s_page + 4 * page_elems);
    bool const use_sliding_window = sliding_window > 0;

    PageStage current = stage_kv_page_async(gg, kv_head_idx, page_start, total_pages, last_pl, page_range_start,
        earliest_visible_page, use_sliding_window, loader_tid, is_loader, s_k[0], s_v[0], page_elems);
    if (is_loader && current.valid)
    {
        cp_async_wait_all();
    }
    __syncthreads();

    int current_buffer = 0;
    int next_buffer = 1;
    for (int page_num = page_range_start; page_num < page_range_end; page_num++)
    {
        PageStage next{false, 0};
        if (page_num + 1 < page_range_end)
        {
            next = stage_kv_page_async(gg, kv_head_idx, page_start, total_pages, last_pl, page_num + 1,
                earliest_visible_page, use_sliding_window, loader_tid, is_loader, s_k[next_buffer], s_v[next_buffer],
                page_elems);
        }

        if (current.valid && is_consumer)
        {
            for (int t = consumer_group_lane; t < current.validTokens; t += ATTN_WARPS_PER_Q_HEAD)
            {
                // Skip tokens before the sliding window boundary
                int global_token_pos = page_num * gg.page_size + t;
                if (use_sliding_window && global_token_pos < earliest_visible)
                    continue;

                int const page_row = t * HEAD_DIM;

                // Q · K^T
                float dot = 0.0f;
#pragma unroll
                for (int di = 0; di < ELEMS_PER_LANE; di++)
                {
                    dot += q_regs[di] * __bfloat162float(s_k[current_buffer][page_row + lane_id + di * WARP_SIZE]);
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
                    acc[di] = acc[di] * alpha
                        + p * __bfloat162float(s_v[current_buffer][page_row + lane_id + di * WARP_SIZE]);
                }
                l_i = l_i * alpha + p;
                m_i = new_m;
            }
        }

        if (is_loader && next.valid)
        {
            cp_async_wait_all();
        }
        __syncthreads();
        current = next;
        int const tmp = current_buffer;
        current_buffer = next_buffer;
        next_buffer = tmp;
    }

    if (is_consumer)
    {
        float* partial = s_consumer_partials + consumer_id * (HEAD_DIM + 2);
#pragma unroll
        for (int di = 0; di < ELEMS_PER_LANE; di++)
        {
            partial[lane_id + di * WARP_SIZE] = acc[di];
        }
        if (lane_id == 0)
        {
            partial[HEAD_DIM] = m_i;
            partial[HEAD_DIM + 1] = l_i;
        }
    }
    __syncthreads();

    if (is_consumer && consumer_group_lane == 0)
    {
        float m_acc = -1e30f;
        float l_acc = 0.0f;
        float o_acc[ELEMS_PER_LANE];
#pragma unroll
        for (int di = 0; di < ELEMS_PER_LANE; di++)
        {
            o_acc[di] = 0.0f;
        }

        int const group_start = q_head_group * ATTN_WARPS_PER_Q_HEAD;
        for (int wi = 0; wi < ATTN_WARPS_PER_Q_HEAD; wi++)
        {
            float const* partial = s_consumer_partials + (group_start + wi) * (HEAD_DIM + 2);
            float const m_i_local = partial[HEAD_DIM];
            float const l_i_local = partial[HEAD_DIM + 1];
            if (l_i_local <= 0.0f)
            {
                continue;
            }

            float const new_m = fmaxf(m_acc, m_i_local);
            float const alpha = expf(m_acc - new_m);
            float const beta = expf(m_i_local - new_m);
#pragma unroll
            for (int di = 0; di < ELEMS_PER_LANE; di++)
            {
                float const o_i = partial[lane_id + di * WARP_SIZE];
                o_acc[di] = o_acc[di] * alpha + o_i * beta;
            }
            l_acc = l_acc * alpha + l_i_local * beta;
            m_acc = new_m;
        }

        if (is_single)
        {
            float const inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
            __nv_bfloat16* out = gg.attn_scratch + (int64_t) token_id * Q_WIDTH + (int64_t) q_head_idx * HEAD_DIM;
#pragma unroll
            for (int di = 0; di < ELEMS_PER_LANE; di++)
            {
                out[lane_id + di * WARP_SIZE] = __float2bfloat16(o_acc[di] * inv_l);
            }
        }
        else
        {
            float* partial_out = gg.partial_attn_scratch + (int64_t) partial_idx * NUM_Q_HEADS * (HEAD_DIM + 2)
                + (int64_t) q_head_idx * (HEAD_DIM + 2);
#pragma unroll
            for (int di = 0; di < ELEMS_PER_LANE; di++)
            {
                partial_out[lane_id + di * WARP_SIZE] = o_acc[di];
            }
            if (lane_id == 0)
            {
                partial_out[HEAD_DIM] = m_acc;
                partial_out[HEAD_DIM + 1] = l_acc;
            }
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
    if (consumer_id >= NUM_CONSUMERS)
        return;

    // word[4] = partial_start (first partial index for this KV head)
    int const partial_start = instr.field(4);

    int const q_head_group = consumer_id / REDUCE_WARPS_PER_Q_HEAD;
    int const q_head_idx = kv_head_idx * GQA_RATIO + q_head_group;
    int const q_head_chunk = consumer_id % REDUCE_WARPS_PER_Q_HEAD;
    int const dim = q_head_chunk * REDUCE_DIMS_PER_WARP + lane_id;
    float m_acc = -1e30f;
    float l_acc = 0.0f;
    float o_acc = 0.0f;

    for (int pi = 0; pi < num_partials; pi++)
    {
        float const* partial = gg.partial_attn_scratch + (int64_t) (partial_start + pi) * NUM_Q_HEADS * (HEAD_DIM + 2)
            + (int64_t) q_head_idx * (HEAD_DIM + 2);

        float m_i = 0.0f, l_i = 0.0f;
        if (lane_id == 0)
        {
            m_i = partial[HEAD_DIM];
            l_i = partial[HEAD_DIM + 1];
        }
        m_i = __shfl_sync(0xFFFFFFFF, m_i, 0);
        l_i = __shfl_sync(0xFFFFFFFF, l_i, 0);

        if (l_i <= 0.0f)
        {
            continue;
        }

        float const new_m = fmaxf(m_acc, m_i);
        float const alpha = expf(m_acc - new_m);
        float const beta = expf(m_i - new_m);
        float const o_i = partial[dim];
        o_acc = o_acc * alpha + o_i * beta;
        l_acc = l_acc * alpha + l_i * beta;
        m_acc = new_m;
    }

    float inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
    __nv_bfloat16* out = gg.attn_scratch + (int64_t) token_id * Q_WIDTH + (int64_t) q_head_idx * HEAD_DIM;
    out[dim] = __float2bfloat16(o_acc * inv_l);
}

} // namespace gemma4_megakernel
