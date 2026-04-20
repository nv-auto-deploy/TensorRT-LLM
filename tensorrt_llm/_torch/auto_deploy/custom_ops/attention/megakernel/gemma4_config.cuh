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

#include <cstdint>

namespace gemma4_megakernel
{

// ──────────────────────────────────────────────────────────────
// Gemma4 model constants
// ──────────────────────────────────────────────────────────────
static constexpr int HIDDEN_SIZE = 2816;
static constexpr int HEAD_DIM = 256;
static constexpr int NUM_Q_HEADS = 16;
static constexpr int NUM_KV_HEADS = 8;
static constexpr int FFN_INTERMEDIATE = 2112;
static constexpr int NUM_EXPERTS = 128;
static constexpr int MOE_TOPK = 8;
static constexpr int EXPERT_INTERMEDIATE = 704;
static constexpr int GQA_RATIO = NUM_Q_HEADS / NUM_KV_HEADS; // 2
static constexpr int Q_WIDTH = NUM_Q_HEADS * HEAD_DIM;       // 4096
static constexpr int KV_WIDTH = NUM_KV_HEADS * HEAD_DIM;     // 2048
static constexpr int QKV_WIDTH = Q_WIDTH + 2 * KV_WIDTH;     // 8192

// ──────────────────────────────────────────────────────────────
// Megakernel launch configuration
// ──────────────────────────────────────────────────────────────
static constexpr int NUM_SMS = 132;                             // H100
static constexpr int NUM_WARPS = 20;                            // per SM
static constexpr int WARP_SIZE = 32;
static constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE; // 640

// Warp role assignments
static constexpr int CONTROLLER_WARP = 0;
static constexpr int LOADER_WARP = 1;
static constexpr int LAUNCHER_WARP = 2;
static constexpr int STORER_WARP = 3;
static constexpr int FIRST_CONSUMER = 4;
static constexpr int NUM_CONSUMERS = NUM_WARPS - FIRST_CONSUMER; // 16

// Shared memory
static constexpr int SMEM_PAGE_SIZE = 16384;                       // 16 KB per page
static constexpr int NUM_SMEM_PAGES = 12;
static constexpr int TOTAL_SMEM = SMEM_PAGE_SIZE * NUM_SMEM_PAGES; // 192 KB

// ──────────────────────────────────────────────────────────────
// Instruction encoding
// ──────────────────────────────────────────────────────────────
static constexpr int INSTRUCTION_WORDS = 32; // 128 bytes per instruction
static constexpr int MAX_INSTRUCTIONS = 512; // per SM per launch, enough for 30-layer loops

// Instruction preload area at the start of shared memory
static constexpr int MAX_PRELOAD_INSTRUCTIONS = 24;
static constexpr int SMEM_INSTR_BYTES = MAX_PRELOAD_INSTRUCTIONS * INSTRUCTION_WORDS * 4; // 2048 bytes
static constexpr int SMEM_DATA_OFFSET = SMEM_INSTR_BYTES; // Opcode data starts after instructions

// Opcodes
enum Opcode : int32_t
{
    OP_NOOP = 0,
    OP_BARRIER = 1,      // Global SM barrier
    OP_GEMV_QKV = 2,     // Phase 1: Raw QKV GEMV → scratch
    OP_QKV_POST = 6,     // Phase 1: Per-head norms + RoPE + cache write (after GEMV barrier)
    OP_PAGED_ATTN = 3,   // Phase 2: Paged partial attention
    OP_ATTN_REDUCE = 4,  // Phase 2: Attention reduction
    OP_GEMV_OPROJ = 5,   // Phase 3: Raw O-proj GEMV → scratch
    OP_OPROJ_POST = 7,   // Phase 3: Norms + residual + pre-FFN norm
    OP_FFN_GATEUP = 8,   // Phase 4 / Kernel B: Dense FFN gate+up projection
    OP_FFN_DOWN = 9,     // Phase 4 / Kernel B: Dense FFN down projection
    OP_ROUTER_TOPK = 10, // Phase 4 / Kernel B: Router projection + top-k selection
    OP_MOE = 11,         // Phase 4 / Kernel B: MoE expert execution + merge
    OP_B_POST = 12,      // Phase 4 / Kernel B: Post-FFN combine + next-layer RMSNorm
    OP_DONE = 255,
};

} // namespace gemma4_megakernel
