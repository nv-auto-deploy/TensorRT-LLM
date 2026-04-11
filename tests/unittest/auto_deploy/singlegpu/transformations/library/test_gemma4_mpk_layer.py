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

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.auto_deploy.mpk import (
    GemmaLayerInfo,
    GemmaLayerLoweringPlanner,
    GemmaLayerSchema,
    GemmaNodeRef,
    execute_layer_plan_reference,
    exercise_layer_plan_against_mirage,
    resolve_layer_plan_against_mirage,
)
from tensorrt_llm._torch.auto_deploy.mpk.mirage_bridge import _require_mirage


def _make_gemma4_layer_info() -> GemmaLayerInfo:
    return GemmaLayerInfo(
        layer_index=0,
        schema=GemmaLayerSchema.REGULAR,
        anchors={
            "qkv_linear": GemmaNodeRef(
                "qkv", "call_function", "auto_deploy::torch_linear_simple", 0
            ),
            "q_norm": GemmaNodeRef("qnorm", "call_function", "auto_deploy::flashinfer_rms_norm", 0),
            "k_norm": GemmaNodeRef("knorm", "call_function", "auto_deploy::flashinfer_rms_norm", 0),
            "v_norm": GemmaNodeRef("vnorm", "call_function", "auto_deploy::flashinfer_rms_norm", 0),
            "rope": GemmaNodeRef("rope", "call_function", "auto_deploy::flashinfer_rope", 0),
            "cached_attention": GemmaNodeRef(
                "attn", "call_function", "auto_deploy::triton_paged_mha_with_cache", 0
            ),
            "o_proj": GemmaNodeRef("oproj", "call_function", "auto_deploy::torch_linear_simple", 0),
            "ffn_gate_up": GemmaNodeRef(
                "ffnup", "call_function", "auto_deploy::torch_linear_simple", 0
            ),
            "ffn_down": GemmaNodeRef(
                "ffndown", "call_function", "auto_deploy::torch_linear_simple", 0
            ),
            "router_proj": GemmaNodeRef(
                "router", "call_function", "auto_deploy::torch_linear_simple", 0
            ),
            "topk": GemmaNodeRef(
                "topk", "call_function", "auto_deploy::triton_fused_topk_softmax", 0
            ),
            "moe_fused": GemmaNodeRef("moe", "call_function", "auto_deploy::trtllm_moe_fused", 0),
            "kv_cache": GemmaNodeRef("r0_kv_cache", "placeholder", "r0_kv_cache", 0),
        },
        hidden_size=8,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        router_top_k=2,
        moe_is_gated_mlp=True,
        moe_act_fn=1,
    )


def _make_reference_inputs():
    hidden_in = torch.tensor(
        [
            [0.5, -1.0, 0.25, 1.5],
            [-0.75, 0.5, 1.0, -0.25],
        ],
        dtype=torch.float32,
    )
    weights = {
        "attn_norm_weight": torch.tensor([1.0, 0.5, 1.5, 0.75], dtype=torch.float32),
        "qkv_weight": torch.arange(48, dtype=torch.float32).reshape(12, 4) / 32.0,
        "q_norm_weight": torch.tensor([1.0, 1.1, 0.9, 1.2], dtype=torch.float32),
        "k_norm_weight": torch.tensor([0.8, 1.0, 1.2, 0.7], dtype=torch.float32),
        "v_norm_weight": torch.tensor([1.2, 0.95, 0.85, 1.05], dtype=torch.float32),
        "cos": torch.tensor([[1.0, 0.5, 1.0, 0.5], [0.75, 0.25, 0.75, 0.25]], dtype=torch.float32),
        "sin": torch.tensor([[0.0, 0.25, 0.0, 0.25], [0.1, 0.2, 0.1, 0.2]], dtype=torch.float32),
        "k_cache": torch.tensor(
            [[0.1, -0.2, 0.05, 0.3], [-0.1, 0.15, -0.05, 0.2]], dtype=torch.float32
        ),
        "v_cache": torch.tensor(
            [[0.2, 0.0, -0.1, 0.05], [0.05, -0.05, 0.15, -0.1]], dtype=torch.float32
        ),
        "o_proj_weight": torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16.0,
        "ffn_norm_weight": torch.tensor([0.9, 1.1, 1.0, 0.8], dtype=torch.float32),
        "ffn_gate_up_weight": torch.arange(48, dtype=torch.float32).reshape(12, 4) / 40.0,
        "ffn_down_weight": torch.arange(24, dtype=torch.float32).reshape(4, 6) / 24.0,
        "router_weight": torch.tensor(
            [
                [0.25, -0.5, 0.75, 0.1],
                [0.5, 0.2, -0.25, 0.3],
                [-0.1, 0.4, 0.15, 0.6],
            ],
            dtype=torch.float32,
        ),
        "moe_w13_weight": torch.arange(72, dtype=torch.float32).reshape(3, 6, 4) / 50.0,
        "moe_w2_weight": torch.arange(36, dtype=torch.float32).reshape(3, 4, 3) / 60.0,
    }
    return hidden_in, weights


def _rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = input_tensor.square().mean(dim=-1, keepdim=True)
    return input_tensor * torch.rsqrt(variance + eps) * weight


def _apply_rope(input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first_half, second_half = torch.chunk(input_tensor, 2, dim=-1)
    cos_half = cos[..., : first_half.shape[-1]]
    sin_half = sin[..., : first_half.shape[-1]]
    return torch.cat(
        [
            first_half * cos_half - second_half * sin_half,
            first_half * sin_half + second_half * cos_half,
        ],
        dim=-1,
    )


def _manual_reference_outputs(hidden_in: torch.Tensor, weights: dict[str, torch.Tensor]):
    qkv_normed = _rms_norm(hidden_in, weights["attn_norm_weight"])
    qkv_packed = qkv_normed @ weights["qkv_weight"].transpose(0, 1)

    q, k, v = torch.chunk(qkv_packed, 3, dim=-1)
    q_ready = _apply_rope(_rms_norm(q, weights["q_norm_weight"]), weights["cos"], weights["sin"])
    k_ready = _apply_rope(_rms_norm(k, weights["k_norm_weight"]), weights["cos"], weights["sin"])
    v_ready = _rms_norm(v, weights["v_norm_weight"])
    attn_out = (
        q_ready
        + k_ready
        + v_ready
        + weights["k_cache"].mean(dim=0)
        + weights["v_cache"].mean(dim=0)
    )

    post_attn_residual = attn_out @ weights["o_proj_weight"].transpose(0, 1) + hidden_in

    ffn_gate_up = _rms_norm(post_attn_residual, weights["ffn_norm_weight"])
    ffn_gate_up = ffn_gate_up @ weights["ffn_gate_up_weight"].transpose(0, 1)
    ffn_gate, ffn_up = torch.chunk(ffn_gate_up, 2, dim=-1)
    ffn_down = (F.gelu(ffn_gate) * ffn_up) @ weights["ffn_down_weight"].transpose(0, 1)

    router_logits = post_attn_residual @ weights["router_weight"].transpose(0, 1)
    router_values, router_indices = torch.topk(router_logits, k=2, dim=-1)
    router_weights = torch.softmax(router_values, dim=-1)
    router_mask = torch.zeros_like(router_logits)
    router_mask.scatter_(1, router_indices, 1.0)

    moe_w13_out = torch.empty(hidden_in.shape[0], 2, 6, dtype=torch.float32)
    moe_w2_out = torch.empty(hidden_in.shape[0], 2, 4, dtype=torch.float32)
    for token_idx in range(hidden_in.shape[0]):
        for route_idx in range(2):
            expert_idx = int(router_indices[token_idx, route_idx].item())
            moe_w13_out[token_idx, route_idx] = post_attn_residual[token_idx] @ weights[
                "moe_w13_weight"
            ][expert_idx].transpose(0, 1)
            moe_gate, moe_up = torch.chunk(moe_w13_out[token_idx, route_idx], 2, dim=-1)
            moe_act = F.gelu(moe_gate) * moe_up
            moe_w2_out[token_idx, route_idx] = moe_act @ weights["moe_w2_weight"][
                expert_idx
            ].transpose(0, 1)

    hidden_out = (moe_w2_out * router_weights.unsqueeze(-1)).sum(dim=1) + ffn_down
    return {
        "qkv_packed": qkv_packed,
        "attn_out": attn_out,
        "post_attn_residual": post_attn_residual,
        "ffn_down": ffn_down,
        "router_logits": router_logits,
        "router_weights": router_weights,
        "router_indices": router_indices,
        "router_mask": router_mask,
        "moe_w13_out": moe_w13_out,
        "moe_w2_out": moe_w2_out,
        "hidden_out": hidden_out,
    }


def test_gemma4_layer_lowering_resolves_and_runs_on_mirage():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    layer_info = _make_gemma4_layer_info()
    layer_plan = GemmaLayerLoweringPlanner().build(layer_info)

    bindings = resolve_layer_plan_against_mirage(layer_plan)
    assert len(bindings) == len(layer_plan.mpk_steps)
    assert sum(1 for item in bindings if item.resolved) >= 6

    result = exercise_layer_plan_against_mirage(layer_plan)

    assert "attn_rmsnorm_linear" in result["executed_steps"]
    assert "paged_attention" in result["executed_steps"]
    assert "router_topk_softmax" in result["executed_steps"]
    assert result["generated_json_len"] > 0
    assert result["generated_cuda_len"] > 0


def test_supported_mirage_smoke_compiles_and_launches():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "compile_supported_rmsnorm_linear_smoke; "
                "print(compile_supported_rmsnorm_linear_smoke("
                "output_dir='./mirage_compile_supported_smoke_test_subproc'))"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "'compiled': True" in proc.stdout
    assert "'launched': True" in proc.stdout


def test_mirage_norm_linear_matches_reference_numerically():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_norm_linear_forward_correctness; "
                "result = run_mirage_norm_linear_forward_correctness(); "
                "print(result); "
                "assert result['max_abs'] < 1.5; "
                "assert result['mean_abs'] < 0.3"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "max_abs" in proc.stdout


def test_mirage_paged_attention_matches_reference_numerically():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_paged_attention_forward_correctness; "
                "result = run_mirage_paged_attention_forward_correctness(); "
                "print(result); "
                "assert result['attn_max_abs'] < 0.1; "
                "assert result['attn_mean_abs'] < 0.01"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "attn_max_abs" in proc.stdout


def test_mirage_linear_with_residual_matches_reference_numerically():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_linear_with_residual_forward_correctness; "
                "result = run_mirage_linear_with_residual_forward_correctness(); "
                "print(result); "
                "assert result['linear_max_abs'] < 1.0; "
                "assert result['linear_mean_abs'] < 0.2"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "linear_max_abs" in proc.stdout


def test_mirage_linear_with_residual_pk_matches_reference_across_repeats():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_linear_with_residual_pk_forward_correctness; "
                "result = run_mirage_linear_with_residual_pk_forward_correctness(); "
                "print(result); "
                "assert result['repeat_0_max_abs'] < 0.01; "
                "assert result['repeat_0_mean_abs'] < 0.001; "
                "assert result['repeat_1_max_abs'] < 0.01; "
                "assert result['repeat_1_mean_abs'] < 0.001"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "repeat_1_max_abs" in proc.stdout


def test_mirage_hybrid_attention_sublayer_single_token_matches_reference_across_repeats():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_hybrid_attention_sublayer_forward_correctness; "
                "result = run_mirage_hybrid_attention_sublayer_forward_correctness("
                "num_tokens=1, repeats=2); "
                "print(result); "
                "assert result['repeat_0_qkv_max_abs'] < 0.03; "
                "assert result['repeat_0_qkv_mean_abs'] < 0.005; "
                "assert result['repeat_0_attn_max_abs'] < 0.005; "
                "assert result['repeat_0_attn_mean_abs'] < 0.001; "
                "assert result['repeat_0_block_max_abs'] < 0.01; "
                "assert result['repeat_0_block_mean_abs'] < 0.002; "
                "assert result['repeat_1_qkv_max_abs'] < 0.03; "
                "assert result['repeat_1_qkv_mean_abs'] < 0.005; "
                "assert result['repeat_1_attn_max_abs'] < 0.005; "
                "assert result['repeat_1_attn_mean_abs'] < 0.001; "
                "assert result['repeat_1_block_max_abs'] < 0.01; "
                "assert result['repeat_1_block_mean_abs'] < 0.002"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "repeat_1_block_max_abs" in proc.stdout


def test_mirage_attention_block_pk_matches_reference_across_repeats():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_attention_block_pk_forward_correctness; "
                "result = run_mirage_attention_block_pk_forward_correctness(); "
                "print(result); "
                "assert result['repeat_0_attn_max_abs'] < 0.005; "
                "assert result['repeat_0_attn_mean_abs'] < 0.001; "
                "assert result['repeat_0_block_max_abs'] < 0.005; "
                "assert result['repeat_0_block_mean_abs'] < 0.001; "
                "assert result['repeat_1_attn_max_abs'] < 0.005; "
                "assert result['repeat_1_attn_mean_abs'] < 0.001; "
                "assert result['repeat_1_block_max_abs'] < 0.005; "
                "assert result['repeat_1_block_mean_abs'] < 0.001"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "repeat_1_block_max_abs" in proc.stdout


def test_mirage_attention_sublayer_pk_matches_reference_across_repeats():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_attention_sublayer_pk_forward_correctness; "
                "result = run_mirage_attention_sublayer_pk_forward_correctness(); "
                "print(result); "
                "assert result['repeat_0_qkv_max_abs'] < 0.02; "
                "assert result['repeat_0_qkv_mean_abs'] < 0.005; "
                "assert result['repeat_0_attn_max_abs'] < 0.05; "
                "assert result['repeat_0_attn_mean_abs'] < 0.01; "
                "assert result['repeat_0_block_max_abs'] < 0.06; "
                "assert result['repeat_0_block_mean_abs'] < 0.01; "
                "assert result['repeat_1_qkv_max_abs'] < 0.02; "
                "assert result['repeat_1_qkv_mean_abs'] < 0.005; "
                "assert result['repeat_1_attn_max_abs'] < 0.05; "
                "assert result['repeat_1_attn_mean_abs'] < 0.01; "
                "assert result['repeat_1_block_max_abs'] < 0.06; "
                "assert result['repeat_1_block_mean_abs'] < 0.01"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "repeat_1_block_max_abs" in proc.stdout


def test_mirage_moe_silu_block_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_moe_silu_block_forward_correctness; "
                "result = run_mirage_moe_silu_block_forward_correctness(); "
                "print(result); "
                "assert result['topk_weight_max_abs'] < 0.005; "
                "assert result['topk_weight_mean_abs'] < 0.0002; "
                "assert result['routing_overlap_count'] >= 7.0; "
                "assert result['w2_max_abs'] < 0.03; "
                "assert result['w2_mean_abs'] < 0.005; "
                "assert result['out_max_abs'] < 0.01; "
                "assert result['out_mean_abs'] < 0.002"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "routing_overlap_count" in proc.stdout


def test_mirage_moe_gelu_split_block_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_moe_gelu_split_block_forward_correctness; "
                "result = run_mirage_moe_gelu_split_block_forward_correctness(); "
                "print(result); "
                "assert result['topk_weight_max_abs'] < 0.005; "
                "assert result['topk_weight_mean_abs'] < 0.0002; "
                "assert result['routing_overlap_count'] >= 7.0; "
                "assert result['act_max_abs'] < 0.06; "
                "assert result['act_mean_abs'] < 0.005; "
                "assert result['w2_max_abs'] < 0.03; "
                "assert result['w2_mean_abs'] < 0.005; "
                "assert result['out_max_abs'] < 0.01; "
                "assert result['out_mean_abs'] < 0.002"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "routing_overlap_count" in proc.stdout


def test_mirage_moe_gelu_split_dense_projection_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_moe_gelu_split_dense_projection_forward_correctness; "
                "result = run_mirage_moe_gelu_split_dense_projection_forward_correctness(); "
                "print(result); "
                "assert result['post_attn_max_abs'] > 5.0; "
                "assert result['post_attn_mean_abs'] > 1.0; "
                "assert result['gate_max_abs'] < 0.02; "
                "assert result['gate_mean_abs'] < 0.004; "
                "assert result['up_max_abs'] < 0.02; "
                "assert result['up_mean_abs'] < 0.004"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "post_attn_max_abs" in proc.stdout


def test_mirage_moe_gelu_split_dense_block_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_moe_gelu_split_dense_block_forward_correctness; "
                "result = run_mirage_moe_gelu_split_dense_block_forward_correctness(); "
                "print(result); "
                "assert result['topk_weight_max_abs'] < 0.005; "
                "assert result['topk_weight_mean_abs'] < 0.002; "
                "assert result['routing_overlap_count'] >= 7.0; "
                "assert result['gate_max_abs'] < 0.02; "
                "assert result['gate_mean_abs'] < 0.004; "
                "assert result['up_max_abs'] < 0.02; "
                "assert result['up_mean_abs'] < 0.004; "
                "assert result['act_live_inputs_max_abs'] < 0.07; "
                "assert result['act_live_inputs_mean_abs'] < 0.004; "
                "assert result['w2_max_abs'] < 0.07; "
                "assert result['w2_mean_abs'] < 0.006; "
                "assert result['out_max_abs'] < 0.03; "
                "assert result['out_mean_abs'] < 0.005"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "routing_overlap_count" in proc.stdout


def test_mirage_ffn_down_via_moe_w2_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_ffn_down_via_moe_w2_forward_correctness; "
                "result = run_mirage_ffn_down_via_moe_w2_forward_correctness("
                "use_generic_activation_input=True, repack_after_activation=True); "
                "print(result); "
                "assert result['w2_max_abs'] < 0.001; "
                "assert result['w2_mean_abs'] < 0.0002; "
                "assert result['out_max_abs'] < 0.001; "
                "assert result['out_mean_abs'] < 0.0002"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "w2_max_abs" in proc.stdout


def test_mirage_gemma_decode_ffn_down_via_moe_w2_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_gemma_decode_ffn_down_via_moe_w2_forward_correctness; "
                "result = run_mirage_gemma_decode_ffn_down_via_moe_w2_forward_correctness(); "
                "print(result); "
                "assert result['out_max_abs'] < 0.01; "
                "assert result['out_mean_abs'] < 0.001"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    assert "out_max_abs" in proc.stdout


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Direct decode FFN-down PK linear specialization remains unstable for the "
        "single-token Gemma decode shape; keep the moe_w2 workaround until this "
        "path becomes reliable."
    ),
)
def test_mirage_direct_ffn_down_decode_path_regression():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_ffn_down_projection_forward_correctness; "
                "result = run_mirage_ffn_down_projection_forward_correctness("
                "use_generic_activation_input=True, repack_after_activation=True); "
                "print(result); "
                "assert result['out_max_abs'] < 0.01; "
                "assert result['out_mean_abs'] < 0.003"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )

    assert proc.returncode == 0, proc.stderr
    assert "out_max_abs" in proc.stdout


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Exact Gemma decode direct matmul path remains unstable for the real "
        "single-token FFN-down shape (1 x 2112 -> 2816); keep the moe_w2 "
        "workaround until this path is proven reliable."
    ),
)
def test_mirage_gemma_decode_direct_ffn_down_matmul_regression():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_gemma_decode_ffn_down_direct_matmul_forward_correctness; "
                "result = run_mirage_gemma_decode_ffn_down_direct_matmul_forward_correctness(); "
                "print(result); "
                "assert result['out_max_abs'] < 0.01; "
                "assert result['out_mean_abs'] < 0.001"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )

    assert proc.returncode == 0, proc.stderr
    assert "out_max_abs" in proc.stdout


def test_mirage_gemma_full_live_layer_matches_reference():
    try:
        _require_mirage()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_pythonpath = os.getcwd()
    mirage_pythonpath = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python"
    )
    env["PYTHONPATH"] = ":".join(
        [item for item in [repo_pythonpath, mirage_pythonpath, existing_pythonpath] if item]
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tensorrt_llm._torch.auto_deploy.mpk import "
                "run_mirage_gemma_full_layer_split_dense_forward_correctness; "
                "result = run_mirage_gemma_full_layer_split_dense_forward_correctness(); "
                "print(result); "
                "assert result['post_attn_max_abs'] < 0.06; "
                "assert result['post_attn_mean_abs'] < 0.01; "
                "assert result['ffn_down_max_abs'] < 0.01; "
                "assert result['ffn_down_mean_abs'] < 0.003; "
                "assert result['topk_weight_max_abs'] < 0.005; "
                "assert result['topk_weight_mean_abs'] < 0.002; "
                "assert result['routing_overlap_count'] >= 8.0; "
                "assert result['moe_act_max_abs'] < 0.2; "
                "assert result['moe_act_mean_abs'] < 0.02; "
                "assert result['w2_max_abs'] < 0.07; "
                "assert result['w2_mean_abs'] < 0.01; "
                "assert result['hidden_out_max_abs'] < 0.03; "
                "assert result['hidden_out_mean_abs'] < 0.01"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "hidden_out_max_abs" in proc.stdout


def test_gemma4_translated_attention_block_matches_reference():
    layer_plan = GemmaLayerLoweringPlanner().build(_make_gemma4_layer_info())
    hidden_in, weights = _make_reference_inputs()
    outputs = execute_layer_plan_reference(layer_plan, hidden_in=hidden_in, weights=weights)
    expected = _manual_reference_outputs(hidden_in, weights)

    assert torch.allclose(
        outputs["layer_0_qkv_packed"], expected["qkv_packed"], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(outputs["layer_0_attn_out"], expected["attn_out"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        outputs["layer_0_post_attn_residual"], expected["post_attn_residual"], atol=1e-5, rtol=1e-5
    )


def test_gemma4_translated_moe_block_matches_reference():
    layer_plan = GemmaLayerLoweringPlanner().build(_make_gemma4_layer_info())
    hidden_in, weights = _make_reference_inputs()
    outputs = execute_layer_plan_reference(layer_plan, hidden_in=hidden_in, weights=weights)
    expected = _manual_reference_outputs(hidden_in, weights)

    assert torch.allclose(
        outputs["layer_0_router_logits"], expected["router_logits"], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        outputs["layer_0_router_weights"], expected["router_weights"], atol=1e-5, rtol=1e-5
    )
    assert torch.equal(outputs["layer_0_router_indices"], expected["router_indices"])
    assert torch.allclose(
        outputs["layer_0_moe_w13_out"], expected["moe_w13_out"], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        outputs["layer_0_moe_w2_out"], expected["moe_w2_out"], atol=1e-5, rtol=1e-5
    )


def test_gemma4_full_layer_numerical_correctness():
    layer_plan = GemmaLayerLoweringPlanner().build(_make_gemma4_layer_info())
    hidden_in, weights = _make_reference_inputs()
    outputs = execute_layer_plan_reference(layer_plan, hidden_in=hidden_in, weights=weights)
    expected = _manual_reference_outputs(hidden_in, weights)

    assert torch.allclose(
        outputs[layer_plan.output_buffer], expected["hidden_out"], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(outputs["layer_0_ffn_down"], expected["ffn_down"], atol=1e-5, rtol=1e-5)
