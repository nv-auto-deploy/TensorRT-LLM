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

// Opcode: OP_QKV_POST — Per-head RMSNorm + RoPE + paged KV cache write.
//
// Runs AFTER OP_GEMV_QKV + barrier. Reads raw QKV from qkv_scratch,
// applies per-head processing, writes Q back to scratch (for attention)
// and K/V to the paged KV cache.
//
// Instruction encoding:
//   word[0] = OP_QKV_POST (6)
//   word[1] = head_start (first head index for this SM, inclusive)
//   word[2] = head_end   (exclusive)
//   word[3] = token_id
//
// Head layout in QKV: [Q0..Q15 | K0..K7 | V0..V7] = 32 heads total.
// Heads 0-15 = Q, 16-23 = K, 24-31 = V.

#pragma once

#include "../gemma4_config.cuh"
#include "../gemma4_globals.cuh"
#include "../megakernel_framework.cuh"

namespace gemma4_megakernel
{

static constexpr int TOTAL_HEADS = NUM_Q_HEADS + 2 * NUM_KV_HEADS; // 32

__device__ void handle_qkv_post(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id)
{
    int const head_start = instr.field(1);
    int const head_end = instr.field(2);
    int const token_id = instr.field(3);

    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id < FIRST_CONSUMER)
        return;
    int const consumer_id = warp_id - FIRST_CONSUMER;

    int const half_d = HEAD_DIM / 2; // 128

    // Load per-token metadata
    int position, batch_idx, page_start;
    if (lane_id == 0)
    {
        position = gg.triton_positions[token_id];
        batch_idx = gg.triton_batch_indices[token_id];
        page_start = gg.cu_num_pages[batch_idx];
    }
    position = __shfl_sync(0xFFFFFFFF, position, 0);
    batch_idx = __shfl_sync(0xFFFFFFFF, batch_idx, 0);
    page_start = __shfl_sync(0xFFFFFFFF, page_start, 0);

    // RoPE cos/sin for this position
    float const* cos_base = gg.cos_sin_cache + position * gg.cos_sin_stride_s;

    // Cache write metadata
    int page_idx = position / gg.page_size;
    int tok_in_page = position % gg.page_size;
    int phys_page = gg.cache_loc[page_start + page_idx];

    // Distribute heads across consumer warps
    int const num_heads = head_end - head_start;
    for (int hi = consumer_id; hi < num_heads; hi += NUM_CONSUMERS)
    {
        int const head = head_start + hi;
        int const head_row_start = head * HEAD_DIM;

        // Read raw QKV values for this head
        __nv_bfloat16 const* src = gg.qkv_scratch + (int64_t) token_id * QKV_WIDTH + head_row_start;

        // Each lane handles HEAD_DIM/WARP_SIZE = 8 elements
        float local_first[4], local_second[4]; // first-half and second-half
        float sumsq = 0.0f;

#pragma unroll
        for (int di = 0; di < 4; di++)
        {
            int d_first = lane_id + di * WARP_SIZE; // 0..127
            int d_second = d_first + half_d;        // 128..255
            local_first[di] = __bfloat162float(src[d_first]);
            local_second[di] = __bfloat162float(src[d_second]);
            sumsq += local_first[di] * local_first[di] + local_second[di] * local_second[di];
        }

// Warp reduce sumsq
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        {
            sumsq += __shfl_down_sync(0xFFFFFFFF, sumsq, offset);
        }
        float inv_rms = rsqrtf(__shfl_sync(0xFFFFFFFF, sumsq, 0) / HEAD_DIM + gg.eps);

        if (head < NUM_Q_HEADS)
        {
// ── Q head: RMSNorm + RoPE → Q scratch ──
#pragma unroll
            for (int di = 0; di < 4; di++)
            {
                int d_first = lane_id + di * WARP_SIZE;
                int d_second = d_first + half_d;
                float nf = local_first[di] * inv_rms * gg.q_norm_weight[d_first];
                float ns = local_second[di] * inv_rms * gg.q_norm_weight[d_second];
                float cos_d = cos_base[d_first * gg.cos_sin_stride_d];
                float sin_d = cos_base[(d_first + half_d) * gg.cos_sin_stride_d];
                float rf = nf * cos_d - ns * sin_d;
                float rs = ns * cos_d + nf * sin_d;
                int64_t base = (int64_t) token_id * QKV_WIDTH + head_row_start;
                gg.qkv_scratch[base + d_first] = __float2bfloat16(rf);
                gg.qkv_scratch[base + d_second] = __float2bfloat16(rs);
            }
        }
        else if (head < NUM_Q_HEADS + NUM_KV_HEADS)
        {
            // ── K head: RMSNorm + RoPE → paged KV cache (K slot) ──
            int kv_head_idx = head - NUM_Q_HEADS;
            int64_t cache_base = (int64_t) phys_page * gg.cache_stride_block + 0 * gg.cache_stride_kv
                + (int64_t) kv_head_idx * gg.cache_stride_head + (int64_t) tok_in_page * gg.cache_stride_token;

#pragma unroll
            for (int di = 0; di < 4; di++)
            {
                int d_first = lane_id + di * WARP_SIZE;
                int d_second = d_first + half_d;
                float nf = local_first[di] * inv_rms * gg.k_norm_weight[d_first];
                float ns = local_second[di] * inv_rms * gg.k_norm_weight[d_second];
                float cos_d = cos_base[d_first * gg.cos_sin_stride_d];
                float sin_d = cos_base[(d_first + half_d) * gg.cos_sin_stride_d];
                float rf = nf * cos_d - ns * sin_d;
                float rs = ns * cos_d + nf * sin_d;
                gg.kv_cache[cache_base + d_first] = __float2bfloat16(rf);
                gg.kv_cache[cache_base + d_second] = __float2bfloat16(rs);
            }
        }
        else
        {
            // ── V head: RMSNorm only → paged KV cache (V slot) ──
            int kv_head_idx = head - NUM_Q_HEADS - NUM_KV_HEADS;
            int64_t cache_base = (int64_t) phys_page * gg.cache_stride_block + 1 * gg.cache_stride_kv
                + (int64_t) kv_head_idx * gg.cache_stride_head + (int64_t) tok_in_page * gg.cache_stride_token;

#pragma unroll
            for (int di = 0; di < 4; di++)
            {
                int d_first = lane_id + di * WARP_SIZE;
                int d_second = d_first + half_d;
                float nf = local_first[di] * inv_rms * gg.v_norm_weight[d_first];
                float ns = local_second[di] * inv_rms * gg.v_norm_weight[d_second];
                gg.kv_cache[cache_base + d_first] = __float2bfloat16(nf);
                gg.kv_cache[cache_base + d_second] = __float2bfloat16(ns);
            }
        }
    }

    // Flush writes (Q scratch + KV cache) for cross-SM visibility
    __threadfence();
}

} // namespace gemma4_megakernel
