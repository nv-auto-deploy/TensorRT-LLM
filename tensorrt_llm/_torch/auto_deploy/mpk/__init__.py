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

"""AutoDeploy -> MPK translation utilities."""

from .buffer_planner import GemmaBufferPlanner
from .gemma_analyzer import GemmaGraphAnalyzer
from .gemma_layer_lowering import GemmaLayerLoweringPlanner
from .mirage_bridge import (
    MirageBindingResult,
    build_gemma_mirage_runtime_callable,
    compile_supported_rmsnorm_linear_smoke,
    create_test_persistent_kernel,
    execute_layer_plan_reference,
    exercise_layer_plan_against_mirage,
    exercise_mirage_task_registration,
    resolve_layer_plan_against_mirage,
    run_mirage_attention_block_pk_forward_correctness,
    run_mirage_attention_sublayer_forward_correctness,
    run_mirage_attention_sublayer_pk_forward_correctness,
    run_mirage_ffn_down_projection_forward_correctness,
    run_mirage_ffn_down_via_moe_w2_forward_correctness,
    run_mirage_gemma_decode_ffn_down_direct_matmul_forward_correctness,
    run_mirage_gemma_decode_ffn_down_via_moe_w2_forward_correctness,
    run_mirage_gemma_full_layer_split_dense_forward_correctness,
    run_mirage_hybrid_attention_sublayer_forward_correctness,
    run_mirage_linear_with_residual_forward_correctness,
    run_mirage_linear_with_residual_pk_forward_correctness,
    run_mirage_moe_gelu_split_block_forward_correctness,
    run_mirage_moe_gelu_split_dense_block_forward_correctness,
    run_mirage_moe_gelu_split_dense_projection_forward_correctness,
    run_mirage_moe_silu_block_forward_correctness,
    run_mirage_moe_split_dense_w2_reduce_forward_correctness,
    run_mirage_norm_linear_forward_correctness,
    run_mirage_paged_attention_forward_correctness,
    run_mirage_rmsnorm_linear_pk_forward_correctness,
)
from .runtime_wrapper import GemmaMpkRuntimeWrapper
from .translator import GemmaMpkTranslator
from .types import (
    SUPPORTED_SOURCE_OP_FAMILIES,
    GemmaBufferPlan,
    GemmaBufferSpec,
    GemmaCanonicalOp,
    GemmaGraphInfo,
    GemmaLayerInfo,
    GemmaLayerLoweringPlan,
    GemmaLayerSchema,
    GemmaLoweringStatus,
    GemmaMpkStep,
    GemmaMpkTranslationPlan,
    GemmaNodeRef,
)

__all__ = [
    "GemmaBufferPlan",
    "GemmaBufferPlanner",
    "GemmaBufferSpec",
    "GemmaCanonicalOp",
    "GemmaGraphAnalyzer",
    "GemmaGraphInfo",
    "GemmaLayerInfo",
    "GemmaLayerLoweringPlan",
    "GemmaLayerLoweringPlanner",
    "GemmaLayerSchema",
    "GemmaLoweringStatus",
    "GemmaMpkStep",
    "GemmaMpkTranslationPlan",
    "GemmaMpkTranslator",
    "GemmaMpkRuntimeWrapper",
    "GemmaNodeRef",
    "MirageBindingResult",
    "SUPPORTED_SOURCE_OP_FAMILIES",
    "build_gemma_mirage_runtime_callable",
    "compile_supported_rmsnorm_linear_smoke",
    "create_test_persistent_kernel",
    "execute_layer_plan_reference",
    "exercise_layer_plan_against_mirage",
    "exercise_mirage_task_registration",
    "resolve_layer_plan_against_mirage",
    "run_mirage_gemma_full_layer_split_dense_forward_correctness",
    "run_mirage_ffn_down_projection_forward_correctness",
    "run_mirage_ffn_down_via_moe_w2_forward_correctness",
    "run_mirage_gemma_decode_ffn_down_direct_matmul_forward_correctness",
    "run_mirage_gemma_decode_ffn_down_via_moe_w2_forward_correctness",
    "run_mirage_moe_gelu_split_block_forward_correctness",
    "run_mirage_moe_gelu_split_dense_block_forward_correctness",
    "run_mirage_moe_gelu_split_dense_projection_forward_correctness",
    "run_mirage_moe_split_dense_w2_reduce_forward_correctness",
    "run_mirage_attention_sublayer_forward_correctness",
    "run_mirage_attention_block_pk_forward_correctness",
    "run_mirage_moe_silu_block_forward_correctness",
    "run_mirage_hybrid_attention_sublayer_forward_correctness",
    "run_mirage_attention_sublayer_pk_forward_correctness",
    "run_mirage_linear_with_residual_forward_correctness",
    "run_mirage_linear_with_residual_pk_forward_correctness",
    "run_mirage_paged_attention_forward_correctness",
    "run_mirage_norm_linear_forward_correctness",
    "run_mirage_rmsnorm_linear_pk_forward_correctness",
]
