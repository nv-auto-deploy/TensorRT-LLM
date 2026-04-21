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

// Runtime globals for the Gemma4 megakernel.
// All model weights, KV cache, and scratch buffers are referenced here.

#pragma once

#include "gemma4_config.cuh"
#include <cstdint>
#include <cuda_bf16.h>

namespace gemma4_megakernel
{

struct Gemma4Globals
{
    // ── Model weights (bf16) ──
    __nv_bfloat16 const* qkv_weight;    // [QKV_WIDTH, HIDDEN_SIZE]
    __nv_bfloat16 const* o_proj_weight; // [HIDDEN_SIZE, Q_WIDTH]

    // ── Per-head norm weights (fp32) ──
    float const* q_norm_weight; // [HEAD_DIM]
    float const* k_norm_weight; // [HEAD_DIM]
    float const* v_norm_weight; // [HEAD_DIM]

    // ── Layer norm weights (fp32) ──
    float const* post_attn_norm_weight;  // [HIDDEN_SIZE]
    float const* pre_ffn_norm_weight;    // [HIDDEN_SIZE]
    float const* post_ffn1_norm_weight;  // [HIDDEN_SIZE]
    float const* post_ffn_norm_weight;   // [HIDDEN_SIZE]
    float const* next_input_norm_weight; // [HIDDEN_SIZE]
    float const* layer_scalar;           // [1] or [HIDDEN_SIZE]
    int32_t layer_scalar_size;

    // ── Inputs (bf16, per-token) ──
    __nv_bfloat16 const* attn_normed; // [num_tokens, HIDDEN_SIZE]
    __nv_bfloat16 const* residual;    // [num_tokens, HIDDEN_SIZE]
    float const* b_post_attn_in;      // [num_tokens, HIDDEN_SIZE]
    float const* b_pre_ffn_in;        // [num_tokens, HIDDEN_SIZE]

    // ── RoPE cache (fp32) ──
    float const* cos_sin_cache; // [max_positions, HEAD_DIM]
    int32_t cos_sin_stride_s;   // stride along position dim
    int32_t cos_sin_stride_d;   // stride along head_dim dim

    // ── KV cache (bf16, paged) ──
    __nv_bfloat16* kv_cache; // [num_blocks, 2, NUM_KV_HEADS, page_size, HEAD_DIM]
    int64_t cache_stride_block;
    int64_t cache_stride_kv;
    int64_t cache_stride_head;
    int64_t cache_stride_token;

    // ── Page table ──
    int32_t const* cache_loc;    // [total_pages]
    int32_t const* cu_num_pages; // [batch_size + 1]

                                 // ── Per-token metadata ──
    int32_t const* triton_positions;     // [num_tokens]
    int32_t const* triton_batch_indices; // [num_tokens]
    int32_t const* last_page_len;        // [batch_size]

    // ── Scratch buffers ──
    __nv_bfloat16* qkv_scratch;      // [num_tokens, QKV_WIDTH]
    __nv_bfloat16* attn_scratch;     // [num_tokens, Q_WIDTH]
    float* o_proj_scratch;           // [num_tokens, HIDDEN_SIZE]
    __nv_bfloat16* ffn_gate_scratch; // [num_tokens, FFN_INTERMEDIATE]
    __nv_bfloat16* ffn_up_scratch;   // [num_tokens, FFN_INTERMEDIATE]
    float* ffn_down_scratch;         // [num_tokens, HIDDEN_SIZE]
    float* partial_attn_scratch;     // [max_partials, NUM_Q_HEADS, HEAD_DIM+1]

    // ── Outputs (bf16/fp32) ──
    float* post_attn_out;        // [num_tokens, HIDDEN_SIZE]
    float* pre_ffn_out;          // [num_tokens, HIDDEN_SIZE]
    float* hidden_out;           // [num_tokens, HIDDEN_SIZE]
    float* next_attn_normed_out; // [num_tokens, HIDDEN_SIZE]

    // ── Kernel B weights (bf16) ──
    __nv_bfloat16 const* ffn_gate_up_weight; // [2*FFN_INTERMEDIATE, HIDDEN_SIZE]
    __nv_bfloat16 const* ffn_down_weight;    // [HIDDEN_SIZE, FFN_INTERMEDIATE]
    __nv_bfloat16 const* router_proj_weight; // [NUM_EXPERTS, HIDDEN_SIZE]

    // ── Kernel B router/norm weights ──
    float const* router_scale;                   // [HIDDEN_SIZE]
    float const* router_root_size;               // [1]
    float const* pre_ffn2_norm_weight;           // [HIDDEN_SIZE]
    float const* post_ffn2_norm_weight;          // [HIDDEN_SIZE]
    __nv_bfloat16 const* moe_w13_stacked_weight; // [NUM_EXPERTS, 2*EXPERT_INTERMEDIATE, HIDDEN_SIZE]
    __nv_bfloat16 const* moe_w2_weight;          // [NUM_EXPERTS, HIDDEN_SIZE, EXPERT_INTERMEDIATE]

    // ── Kernel B router/MoE scratch ──
    float* router_topk_weights;       // [num_tokens, MOE_TOPK]
    int32_t* router_topk_indices;     // [num_tokens, MOE_TOPK]
    __nv_bfloat16* moe_input_scratch; // [num_tokens, HIDDEN_SIZE]
    __nv_bfloat16* moe_gate_scratch;  // [num_tokens, MOE_TOPK, EXPERT_INTERMEDIATE]
    __nv_bfloat16* moe_up_scratch;    // [num_tokens, MOE_TOPK, EXPERT_INTERMEDIATE]
    float* moe_scratch;               // [num_tokens, HIDDEN_SIZE, MOE_TOPK]
    float* moe_merged_scratch;        // [num_tokens, HIDDEN_SIZE]

    // ── Attention parameters ──
    float attn_scale;
    float eps;
    int32_t page_size;
    int32_t num_tokens;
};

} // namespace gemma4_megakernel
