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

// Megakernel framework: persistent kernel dispatch loop with global barriers.
//
// Architecture (adapted from HazyResearch/Megakernels):
//   - Single persistent kernel launched across all SMs (132 on H100)
//   - Each SM processes a per-SM instruction queue sequentially
//   - Inter-SM coordination via global memory barriers (atomicAdd + spin-wait)
//   - Within each SM: 20 warps with specialized roles

#pragma once

#include "gemma4_config.cuh"
#include "gemma4_globals.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace gemma4_megakernel
{

// ──────────────────────────────────────────────────────────────
// Instruction: fixed-size 128-byte packet read from global memory
// ──────────────────────────────────────────────────────────────
struct Instruction
{
    int32_t words[INSTRUCTION_WORDS];

    __device__ __forceinline__ Opcode opcode() const
    {
        return static_cast<Opcode>(words[0]);
    }

    __device__ __forceinline__ int32_t field(int idx) const
    {
        return words[idx];
    }
};

// ──────────────────────────────────────────────────────────────
// Global barrier: atomicAdd + volatile spin-wait
// ──────────────────────────────────────────────────────────────
static constexpr int BARRIER_SLOT_STRIDE = 1; // Legacy: 1 int32 per barrier slot

struct GlobalBarrier
{
    int32_t* slots;

    __device__ __forceinline__ void arrive_and_wait(int /*sm_id*/, int barrier_id, int expected_count) const
    {
        atomicAdd(&slots[barrier_id], 1);
        __threadfence();
        int32_t volatile* v = reinterpret_cast<int32_t volatile*>(&slots[barrier_id]);
        while (*v < expected_count)
        {
            __nanosleep(20);
        }
    }
};

// ──────────────────────────────────────────────────────────────
// Runtime globals passed to the kernel
// ──────────────────────────────────────────────────────────────
struct MegakernelGlobals
{
    // Instruction queues: [num_sms * max_instructions * INSTRUCTION_WORDS]
    int32_t const* instructions;

    // Number of instructions per SM: [num_sms]
    int32_t const* num_instructions_per_sm;

    // Global barrier
    GlobalBarrier barrier;

    // Debug/validation output: [num_sms]
    int32_t* debug_output;

    // Number of SMs in this launch
    int32_t num_sms;

    // Gemma4 model globals (weights, caches, scratch buffers)
    Gemma4Globals gemma;
};

// ──────────────────────────────────────────────────────────────
// Instruction fetch: load instruction for current SM
// ──────────────────────────────────────────────────────────────
__device__ __forceinline__ Instruction fetch_instruction(MegakernelGlobals const& g, int sm_id, int instr_idx)
{
    Instruction instr;
    int32_t const* base = g.instructions + (int64_t) sm_id * MAX_INSTRUCTIONS * INSTRUCTION_WORDS
        + (int64_t) instr_idx * INSTRUCTION_WORDS;

    // All threads in warp 0 cooperatively load 32 words
    int lane = threadIdx.x % WARP_SIZE;
    if (lane < INSTRUCTION_WORDS)
    {
        instr.words[lane] = base[lane];
    }
    // Broadcast to all lanes via shared memory or shuffle
    // For simplicity, use __shfl_sync to broadcast from lane < 32
    for (int w = 0; w < INSTRUCTION_WORDS; w++)
    {
        instr.words[w] = __shfl_sync(0xFFFFFFFF, instr.words[w], w);
    }
    return instr;
}

// ──────────────────────────────────────────────────────────────
// Opcode handlers (forward declarations — implemented per phase)
// ──────────────────────────────────────────────────────────────

// Phase 0: no-op (does nothing, for scaffold testing)
__device__ void handle_noop(MegakernelGlobals const& g, Instruction const& instr, int sm_id);

// Phase 0: global barrier
//   field(1) = barrier_id
//   field(2) = expected arrival count
__device__ void handle_barrier(MegakernelGlobals const& g, Instruction const& instr, int sm_id);

// Phase 1: Raw QKV GEMV (implemented in opcodes/gemv_qkv.cuh)
__device__ void handle_gemv_qkv(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id);

// Phase 1: QKV post-processing — norms + RoPE + cache write
__device__ void handle_qkv_post(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id);

// Phase 2: Paged decode attention (multi-SM per KV head)
__device__ void handle_paged_attn(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id);

// Phase 2: Attention reduction (LSE-corrected combine of partials)
__device__ void handle_attn_reduce(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id);

// Phase 3: O-proj GEMV (raw, distributed)
__device__ void handle_gemv_oproj(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id);

// Phase 3: O-proj post-processing (norms + residual + pre-FFN norm)
__device__ void handle_oproj_post(
    MegakernelGlobals const& mg, Gemma4Globals const& gg, Instruction const& instr, int sm_id);

// ──────────────────────────────────────────────────────────────
// Main dispatch loop: the persistent kernel entry point
// ──────────────────────────────────────────────────────────────
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1) megakernel_dispatch(MegakernelGlobals g)
{
    int const sm_id = blockIdx.x;
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    if (sm_id >= g.num_sms)
        return;

    // ── Preload ALL instructions into shared memory at kernel start ──
    // This eliminates per-instruction global memory fetches.
    // Cost: num_instructions × 128 bytes of smem (typically < 2 KB).
    extern __shared__ char smem_raw[];
    // Reserve first part of smem for instruction preload
    static constexpr int MAX_PRELOAD = 16; // Up to 16 instructions preloaded
    Instruction* s_instrs = reinterpret_cast<Instruction*>(smem_raw);

    __shared__ int32_t s_num_instructions;
    if (threadIdx.x == 0)
    {
        s_num_instructions = g.num_instructions_per_sm[sm_id];
    }
    __syncthreads();
    int const num_instructions = min((int) s_num_instructions, MAX_PRELOAD);

    // All threads cooperate to load instructions (128 bytes each = 32 int32 words)
    {
        int const total_words = num_instructions * INSTRUCTION_WORDS;
        int32_t const* src = g.instructions + (int64_t) sm_id * MAX_INSTRUCTIONS * INSTRUCTION_WORDS;
        int32_t* dst = reinterpret_cast<int32_t*>(s_instrs);
        for (int i = threadIdx.x; i < total_words; i += THREADS_PER_BLOCK)
        {
            dst[i] = src[i];
        }
    }
    __syncthreads();

    for (int i = 0; i < num_instructions; i++)
    {
        // Read instruction directly from preloaded shared memory.
        // All threads read the same words (smem broadcast, no bank conflicts).
        // No shuffle needed — eliminates 32 __shfl_sync per instruction.
        Instruction const& sinstr = s_instrs[i];
        Instruction instr;
#pragma unroll
        for (int w = 0; w < 8; w++)
        { // Only read first 8 words (opcodes use at most 7)
            instr.words[w] = sinstr.words[w];
        }

        // Dispatch based on opcode
        switch (instr.opcode())
        {
        case OP_NOOP: break;
        case OP_BARRIER: handle_barrier(g, instr, sm_id); break;
        case OP_GEMV_QKV: handle_gemv_qkv(g, g.gemma, instr, sm_id); break;
        case OP_QKV_POST: handle_qkv_post(g, g.gemma, instr, sm_id); break;
        case OP_PAGED_ATTN: handle_paged_attn(g, g.gemma, instr, sm_id); break;
        case OP_ATTN_REDUCE: handle_attn_reduce(g, g.gemma, instr, sm_id); break;
        case OP_GEMV_OPROJ: handle_gemv_oproj(g, g.gemma, instr, sm_id); break;
        case OP_OPROJ_POST: handle_oproj_post(g, g.gemma, instr, sm_id); break;
        case OP_DONE:
            if (warp_id == 0 && lane_id == 0 && g.debug_output)
            {
                g.debug_output[sm_id] = sm_id + 1;
            }
            return;
        default: break;
        }
        __syncthreads();
    }

    // Mark SM as completed
    if (warp_id == 0 && lane_id == 0 && g.debug_output)
    {
        g.debug_output[sm_id] = sm_id + 1;
    }
}

// ──────────────────────────────────────────────────────────────
// Opcode implementations
// ──────────────────────────────────────────────────────────────

__device__ void handle_noop(MegakernelGlobals const& g, Instruction const& instr, int sm_id)
{
    // No operation — used for scaffold testing and padding
    (void) g;
    (void) instr;
    (void) sm_id;
}

__device__ void handle_barrier(MegakernelGlobals const& g, Instruction const& instr, int sm_id)
{
    int const barrier_id = instr.field(1);
    int const expected_count = instr.field(2);
    int const warp_id = threadIdx.x / WARP_SIZE;
    int const lane_id = threadIdx.x % WARP_SIZE;

    // __threadfence() is called by each opcode that writes global memory
    // (GEMV, QKV_POST, PAGED_ATTN, GEMV_OPROJ) in their epilogues.
    // No need to repeat it here — the dispatch loop's __syncthreads() ensures
    // all threads within the block have completed their threadfence before
    // we reach this barrier instruction.

    // One thread per SM arrives and waits
    if (warp_id == 0 && lane_id == 0)
    {
        g.barrier.arrive_and_wait(sm_id, barrier_id, expected_count);
    }
    __syncthreads();
}

} // namespace gemma4_megakernel
