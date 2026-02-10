# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kernel micro-benchmarks for AutoDeploy custom ops.

Available benchmarks
--------------------
- bench_rms_norm          : FlashInfer vs Triton vs Torch RMSNorm
- bench_fused_add_rms_norm: FlashInfer fused add+RMSNorm vs Torch baseline
- bench_l2_norm           : Torch vs FLA L2Norm
- bench_gated_rms_norm    : Torch vs Triton gated RMSNorm
- bench_rope              : FlashInfer vs Triton vs Torch RoPE
- bench_fp8_linear        : TRT-LLM vs Torch FP8 linear
- bench_moe               : Torch vs TRT-LLM fused MoE (BF16 gated SiLU)

TODO: The following ops are not yet benchmarked:
- Attention (torch_attention, torch_attention_sdpa, triton decode attention)
    Requires KV cache setup, slot indices, and context/decode phase orchestration.
- MLA (torch_mla)
    Requires compressed_kv, kv_b_proj_weight, q_nope/q_pe decomposition.
- Mamba / SSM (torch_mamba, torch_causal_conv1d)
    Stateful ops requiring conv state and SSM state management.
- Linear (torch_linear_simple)
    Thin wrapper over aten.linear; benchmarking adds no insight.
- Sharded RMSNorm (sharded_rmsnorm)
    Requires multi-GPU dist.all_reduce.
- NVFP4 Linear (torch_quant_nvfp4_linear)
    Requires complex weight packing/scaling setup via trtllm.fp4_quantize.
- FP8 BMM (torch_quant_fp8_bmm)
    Niche batched-matmul op, less priority.
- Fake-quant ops (torch_fake_quant_*)
    Reference-only dequant-requant paths, not deployed kernels.
"""
