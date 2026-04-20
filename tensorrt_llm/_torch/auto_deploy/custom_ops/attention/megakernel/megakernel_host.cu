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

// Host-side launcher for the Gemma4 megakernel.
// Compiled as a torch C++ extension via torch.utils.cpp_extension.load.

#include "megakernel_framework.cuh"
#include "opcodes/gemv_oproj.cuh"
#include "opcodes/gemv_qkv.cuh"
#include "opcodes/kernel_b_dense.cuh"
#include "opcodes/paged_attention.cuh"
#include "opcodes/qkv_post.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace gmk = gemma4_megakernel;

static void configure_smem_once()
{
    static bool done = false;
    if (!done)
    {
        cudaFuncSetAttribute(gmk::megakernel_dispatch, cudaFuncAttributeMaxDynamicSharedMemorySize, gmk::TOTAL_SMEM);
        done = true;
    }
}

// ──────────────────────────────────────────────────────────────
// launch_megakernel: Launch with scaffold globals only (Phase 0)
// ──────────────────────────────────────────────────────────────
torch::Tensor launch_megakernel(torch::Tensor instructions, torch::Tensor num_instr_per_sm, torch::Tensor barrier_slots,
    torch::Tensor debug_output, int64_t num_sms)
{
    TORCH_CHECK(instructions.is_cuda(), "instructions must be on CUDA");
    TORCH_CHECK(num_instr_per_sm.is_cuda(), "num_instr_per_sm must be on CUDA");
    TORCH_CHECK(barrier_slots.is_cuda(), "barrier_slots must be on CUDA");
    TORCH_CHECK(debug_output.is_cuda(), "debug_output must be on CUDA");

    TORCH_CHECK(instructions.dtype() == torch::kInt32, "instructions must be int32");
    TORCH_CHECK(num_instr_per_sm.dtype() == torch::kInt32, "num_instr_per_sm must be int32");
    TORCH_CHECK(barrier_slots.dtype() == torch::kInt32, "barrier_slots must be int32");
    TORCH_CHECK(debug_output.dtype() == torch::kInt32, "debug_output must be int32");

    TORCH_CHECK(num_sms > 0 && num_sms <= gmk::NUM_SMS, "num_sms must be in [1, ", gmk::NUM_SMS, "], got ", num_sms);

    gmk::MegakernelGlobals g{};
    g.instructions = instructions.data_ptr<int32_t>();
    g.num_instructions_per_sm = num_instr_per_sm.data_ptr<int32_t>();
    g.barrier.slots = barrier_slots.data_ptr<int32_t>();
    g.debug_output = debug_output.data_ptr<int32_t>();
    g.num_sms = static_cast<int32_t>(num_sms);

    configure_smem_once();
    auto stream = at::cuda::getCurrentCUDAStream();
    gmk::megakernel_dispatch<<<num_sms, gmk::THREADS_PER_BLOCK, gmk::TOTAL_SMEM, stream>>>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return debug_output;
}

// ──────────────────────────────────────────────────────────────
// launch_gemv_qkv: Launch QKV GEMV opcode (Phase 1)
//
// Runs the full QKV projection + per-head norms + RoPE + cache write
// as a megakernel across all SMs.
// ──────────────────────────────────────────────────────────────
torch::Tensor launch_gemv_qkv(torch::Tensor instructions, torch::Tensor num_instr_per_sm, torch::Tensor barrier_slots,
    torch::Tensor debug_output, int64_t num_sms,
    // Gemma4 model inputs
    torch::Tensor attn_normed,          // [num_tokens, HIDDEN_SIZE] bf16
    torch::Tensor qkv_weight,           // [QKV_WIDTH, HIDDEN_SIZE] bf16
    torch::Tensor q_norm_weight,        // [HEAD_DIM] fp32
    torch::Tensor k_norm_weight,        // [HEAD_DIM] fp32
    torch::Tensor v_norm_weight,        // [HEAD_DIM] fp32
    torch::Tensor cos_sin_cache,        // [max_pos, HEAD_DIM] fp32
    torch::Tensor kv_cache,             // [blocks, 2, kv_heads, page_size, HEAD_DIM] bf16
    torch::Tensor cache_loc,            // [total_pages] int32
    torch::Tensor cu_num_pages,         // [batch+1] int32
    torch::Tensor triton_positions,     // [num_tokens] int32
    torch::Tensor triton_batch_indices, // [num_tokens] int32
    torch::Tensor last_page_len,        // [batch] int32
    torch::Tensor qkv_scratch,          // [num_tokens, QKV_WIDTH] bf16
    torch::Tensor attn_scratch,         // [num_tokens, Q_WIDTH] bf16
    double eps, int64_t page_size, double attn_scale,
    // Phase 3: O-proj + norms + residual (optional — pass empty tensors to skip)
    torch::Tensor o_proj_weight,         // [HIDDEN_SIZE, Q_WIDTH] bf16
    torch::Tensor residual,              // [num_tokens, HIDDEN_SIZE] bf16
    torch::Tensor post_attn_norm_weight, // [HIDDEN_SIZE] fp32
    torch::Tensor pre_ffn_norm_weight,   // [HIDDEN_SIZE] fp32
    torch::Tensor o_proj_scratch,        // [num_tokens, HIDDEN_SIZE] fp32
    torch::Tensor post_attn_out,         // [num_tokens, HIDDEN_SIZE] fp32
    torch::Tensor pre_ffn_out,           // [num_tokens, HIDDEN_SIZE] fp32
    torch::Tensor partial_attn_scratch)  // [max_partials, NUM_Q_HEADS, HEAD_DIM+1] fp32
{
    gmk::MegakernelGlobals g{};
    g.instructions = instructions.data_ptr<int32_t>();
    g.num_instructions_per_sm = num_instr_per_sm.data_ptr<int32_t>();
    g.barrier.slots = barrier_slots.data_ptr<int32_t>();
    g.debug_output = debug_output.data_ptr<int32_t>();
    g.num_sms = static_cast<int32_t>(num_sms);

    // Fill Gemma4 globals
    auto& gg = g.gemma;
    gg.attn_normed = reinterpret_cast<__nv_bfloat16 const*>(attn_normed.data_ptr());
    gg.qkv_weight = reinterpret_cast<__nv_bfloat16 const*>(qkv_weight.data_ptr());
    gg.q_norm_weight = q_norm_weight.data_ptr<float>();
    gg.k_norm_weight = k_norm_weight.data_ptr<float>();
    gg.v_norm_weight = v_norm_weight.data_ptr<float>();
    gg.cos_sin_cache = cos_sin_cache.data_ptr<float>();
    gg.cos_sin_stride_s = static_cast<int32_t>(cos_sin_cache.stride(0));
    gg.cos_sin_stride_d = static_cast<int32_t>(cos_sin_cache.stride(1));
    gg.kv_cache = reinterpret_cast<__nv_bfloat16*>(kv_cache.data_ptr());
    gg.cache_stride_block = kv_cache.stride(0);
    gg.cache_stride_kv = kv_cache.stride(1);
    gg.cache_stride_head = kv_cache.stride(2);
    gg.cache_stride_token = kv_cache.stride(3);
    gg.cache_loc = cache_loc.data_ptr<int32_t>();
    gg.cu_num_pages = cu_num_pages.data_ptr<int32_t>();
    gg.triton_positions = triton_positions.data_ptr<int32_t>();
    gg.triton_batch_indices = triton_batch_indices.data_ptr<int32_t>();
    gg.last_page_len = last_page_len.data_ptr<int32_t>();
    gg.qkv_scratch = reinterpret_cast<__nv_bfloat16*>(qkv_scratch.data_ptr());
    gg.attn_scratch = reinterpret_cast<__nv_bfloat16*>(attn_scratch.data_ptr());
    gg.eps = static_cast<float>(eps);
    gg.attn_scale = static_cast<float>(attn_scale);
    gg.page_size = static_cast<int32_t>(page_size);
    gg.num_tokens = static_cast<int32_t>(attn_normed.size(0));

    // Phase 3 globals (o_proj + norms + residual)
    if (o_proj_weight.numel() > 0)
    {
        gg.o_proj_weight = reinterpret_cast<__nv_bfloat16 const*>(o_proj_weight.data_ptr());
        gg.residual = reinterpret_cast<__nv_bfloat16 const*>(residual.data_ptr());
        gg.post_attn_norm_weight = post_attn_norm_weight.data_ptr<float>();
        gg.pre_ffn_norm_weight = pre_ffn_norm_weight.data_ptr<float>();
        gg.o_proj_scratch = o_proj_scratch.data_ptr<float>();
        gg.post_attn_out = post_attn_out.data_ptr<float>();
        gg.pre_ffn_out = pre_ffn_out.data_ptr<float>();
    }
    if (partial_attn_scratch.numel() > 0)
    {
        gg.partial_attn_scratch = partial_attn_scratch.data_ptr<float>();
    }

    configure_smem_once();
    auto stream = at::cuda::getCurrentCUDAStream();
    gmk::megakernel_dispatch<<<num_sms, gmk::THREADS_PER_BLOCK, gmk::TOTAL_SMEM, stream>>>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return debug_output;
}

torch::Tensor launch_dense_b(torch::Tensor instructions, torch::Tensor num_instr_per_sm, torch::Tensor barrier_slots,
    torch::Tensor debug_output, int64_t num_sms, torch::Tensor post_attn_in, torch::Tensor pre_ffn_in,
    torch::Tensor ffn_gate_up_weight, torch::Tensor ffn_down_weight, torch::Tensor post_ffn1_norm_weight,
    torch::Tensor post_ffn_norm_weight, torch::Tensor layer_scalar, torch::Tensor next_input_norm_weight,
    torch::Tensor ffn_gate_scratch, torch::Tensor ffn_up_scratch, torch::Tensor ffn_down_scratch,
    torch::Tensor hidden_out, torch::Tensor next_attn_normed_out, torch::Tensor router_proj_weight,
    torch::Tensor router_scale, torch::Tensor router_root_size, torch::Tensor pre_ffn2_norm_weight,
    torch::Tensor router_topk_weights, torch::Tensor router_topk_indices, torch::Tensor moe_input_scratch,
    torch::Tensor post_ffn2_norm_weight, torch::Tensor moe_w13_stacked_weight, torch::Tensor moe_w2_weight,
    torch::Tensor moe_gate_scratch, torch::Tensor moe_up_scratch, torch::Tensor moe_scratch, double eps)
{
    gmk::MegakernelGlobals g{};
    g.instructions = instructions.data_ptr<int32_t>();
    g.num_instructions_per_sm = num_instr_per_sm.data_ptr<int32_t>();
    g.barrier.slots = barrier_slots.data_ptr<int32_t>();
    g.debug_output = debug_output.data_ptr<int32_t>();
    g.num_sms = static_cast<int32_t>(num_sms);

    auto& gg = g.gemma;
    gg.b_post_attn_in = post_attn_in.data_ptr<float>();
    gg.b_pre_ffn_in = pre_ffn_in.data_ptr<float>();
    gg.ffn_gate_up_weight = reinterpret_cast<__nv_bfloat16 const*>(ffn_gate_up_weight.data_ptr());
    gg.ffn_down_weight = reinterpret_cast<__nv_bfloat16 const*>(ffn_down_weight.data_ptr());
    gg.post_ffn1_norm_weight = post_ffn1_norm_weight.data_ptr<float>();
    gg.post_ffn_norm_weight = post_ffn_norm_weight.data_ptr<float>();
    gg.layer_scalar = layer_scalar.data_ptr<float>();
    gg.layer_scalar_size = static_cast<int32_t>(layer_scalar.numel());
    gg.next_input_norm_weight = next_input_norm_weight.data_ptr<float>();
    gg.ffn_gate_scratch = reinterpret_cast<__nv_bfloat16*>(ffn_gate_scratch.data_ptr());
    gg.ffn_up_scratch = reinterpret_cast<__nv_bfloat16*>(ffn_up_scratch.data_ptr());
    gg.ffn_down_scratch = ffn_down_scratch.data_ptr<float>();
    gg.hidden_out = hidden_out.data_ptr<float>();
    gg.next_attn_normed_out = next_attn_normed_out.data_ptr<float>();
    if (router_proj_weight.numel() > 0)
    {
        gg.router_proj_weight = reinterpret_cast<__nv_bfloat16 const*>(router_proj_weight.data_ptr());
        gg.router_scale = router_scale.data_ptr<float>();
        gg.router_root_size = router_root_size.data_ptr<float>();
        gg.pre_ffn2_norm_weight = pre_ffn2_norm_weight.data_ptr<float>();
        gg.router_topk_weights = router_topk_weights.data_ptr<float>();
        gg.router_topk_indices = router_topk_indices.data_ptr<int32_t>();
        gg.moe_input_scratch = reinterpret_cast<__nv_bfloat16*>(moe_input_scratch.data_ptr());
    }
    if (moe_w13_stacked_weight.numel() > 0)
    {
        gg.post_ffn2_norm_weight = post_ffn2_norm_weight.data_ptr<float>();
        gg.moe_w13_stacked_weight = reinterpret_cast<__nv_bfloat16 const*>(moe_w13_stacked_weight.data_ptr());
        gg.moe_w2_weight = reinterpret_cast<__nv_bfloat16 const*>(moe_w2_weight.data_ptr());
        gg.moe_gate_scratch = reinterpret_cast<__nv_bfloat16*>(moe_gate_scratch.data_ptr());
        gg.moe_up_scratch = reinterpret_cast<__nv_bfloat16*>(moe_up_scratch.data_ptr());
        gg.moe_scratch = moe_scratch.data_ptr<float>();
    }
    gg.eps = static_cast<float>(eps);
    gg.num_tokens = static_cast<int32_t>(post_attn_in.size(0));

    configure_smem_once();
    auto stream = at::cuda::getCurrentCUDAStream();
    gmk::megakernel_dispatch<<<num_sms, gmk::THREADS_PER_BLOCK, gmk::TOTAL_SMEM, stream>>>(g);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return debug_output;
}

// ──────────────────────────────────────────────────────────────
// get_config: Return megakernel configuration as a dict
// ──────────────────────────────────────────────────────────────
std::unordered_map<std::string, int64_t> get_config()
{
    return {
        {"num_sms", gmk::NUM_SMS},
        {"num_warps", gmk::NUM_WARPS},
        {"threads_per_block", gmk::THREADS_PER_BLOCK},
        {"total_smem", gmk::TOTAL_SMEM},
        {"instruction_words", gmk::INSTRUCTION_WORDS},
        {"max_instructions", gmk::MAX_INSTRUCTIONS},
        {"hidden_size", gmk::HIDDEN_SIZE},
        {"ffn_intermediate", gmk::FFN_INTERMEDIATE},
        {"head_dim", gmk::HEAD_DIM},
        {"num_q_heads", gmk::NUM_Q_HEADS},
        {"num_kv_heads", gmk::NUM_KV_HEADS},
        {"gqa_ratio", gmk::GQA_RATIO},
        {"qkv_width", gmk::QKV_WIDTH},
        {"barrier_slot_stride", gmk::BARRIER_SLOT_STRIDE},
        {"op_noop", gmk::OP_NOOP},
        {"op_barrier", gmk::OP_BARRIER},
        {"op_gemv_qkv", gmk::OP_GEMV_QKV},
        {"op_qkv_post", gmk::OP_QKV_POST},
        {"op_ffn_gateup", gmk::OP_FFN_GATEUP},
        {"op_ffn_down", gmk::OP_FFN_DOWN},
        {"op_router_topk", gmk::OP_ROUTER_TOPK},
        {"op_moe", gmk::OP_MOE},
        {"op_b_post", gmk::OP_B_POST},
        {"op_done", gmk::OP_DONE},
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch_megakernel", &launch_megakernel, "Launch the Gemma4 persistent megakernel (scaffold)",
        py::arg("instructions"), py::arg("num_instr_per_sm"), py::arg("barrier_slots"), py::arg("debug_output"),
        py::arg("num_sms"));
    m.def("launch_gemv_qkv", &launch_gemv_qkv, "Launch QKV GEMV megakernel", py::arg("instructions"),
        py::arg("num_instr_per_sm"), py::arg("barrier_slots"), py::arg("debug_output"), py::arg("num_sms"),
        py::arg("attn_normed"), py::arg("qkv_weight"), py::arg("q_norm_weight"), py::arg("k_norm_weight"),
        py::arg("v_norm_weight"), py::arg("cos_sin_cache"), py::arg("kv_cache"), py::arg("cache_loc"),
        py::arg("cu_num_pages"), py::arg("triton_positions"), py::arg("triton_batch_indices"), py::arg("last_page_len"),
        py::arg("qkv_scratch"), py::arg("attn_scratch"), py::arg("eps"), py::arg("page_size"), py::arg("attn_scale"),
        py::arg("o_proj_weight"), py::arg("residual"), py::arg("post_attn_norm_weight"), py::arg("pre_ffn_norm_weight"),
        py::arg("o_proj_scratch"), py::arg("post_attn_out"), py::arg("pre_ffn_out"), py::arg("partial_attn_scratch"));
    m.def("launch_dense_b", &launch_dense_b, "Launch dense Kernel B megakernel slice", py::arg("instructions"),
        py::arg("num_instr_per_sm"), py::arg("barrier_slots"), py::arg("debug_output"), py::arg("num_sms"),
        py::arg("post_attn_in"), py::arg("pre_ffn_in"), py::arg("ffn_gate_up_weight"), py::arg("ffn_down_weight"),
        py::arg("post_ffn1_norm_weight"), py::arg("post_ffn_norm_weight"), py::arg("layer_scalar"),
        py::arg("next_input_norm_weight"), py::arg("ffn_gate_scratch"), py::arg("ffn_up_scratch"),
        py::arg("ffn_down_scratch"), py::arg("hidden_out"), py::arg("next_attn_normed_out"),
        py::arg("router_proj_weight"), py::arg("router_scale"), py::arg("router_root_size"),
        py::arg("pre_ffn2_norm_weight"), py::arg("router_topk_weights"), py::arg("router_topk_indices"),
        py::arg("moe_input_scratch"), py::arg("post_ffn2_norm_weight"), py::arg("moe_w13_stacked_weight"),
        py::arg("moe_w2_weight"), py::arg("moe_gate_scratch"), py::arg("moe_up_scratch"), py::arg("moe_scratch"),
        py::arg("eps"));
    m.def("get_config", &get_config, "Get megakernel configuration");
}
