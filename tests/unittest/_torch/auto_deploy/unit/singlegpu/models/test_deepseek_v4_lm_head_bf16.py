# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""AD_BF16_LM_HEAD gate: run the LM-head logits GEMM in bf16.

Gate off (default) keeps the fp32 head (weight widened once at load, fp32
GEMM). Gate on keeps the head weight in bf16 and rounds the fp32 hidden to
bf16 at the head boundary, so the vocab GEMM streams half the bytes. The
bf16-path logits differ from the fp32 path only by bf16 rounding of the
activation and of the GEMM output (both paths see the same bf16-valued
weight); these tests pin that bound, the bf16 path's definitional
bit-exactness, and gate-off leaving construction dtype + head graph unchanged.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (registers custom ops)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom import modeling_deepseek_v4 as dsv4
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    _linear,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import extract_op_args, is_op

# bf16 keeps 8 significand bits, so round-to-nearest satisfies
# |round_bf16(x) - x| <= 2**-8 * |x|.
_BF16_REL = 2.0**-8

# NOTE: on sm>=100 ``torch_linear_simple`` routes bf16 x bf16 to
# ``trtllm::cublas_mm`` (CUDA-only), so every bf16-path test below runs on GPU.
_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _small_config(**overrides) -> DeepseekV4Config:
    values = {
        "vocab_size": 32,
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 8,
        "qk_rope_head_dim": 2,
        "o_groups": 2,
        "o_lora_rank": 8,
        "sliding_window": 3,
        "compress_ratios": (0, 4, 128),
        "compress_rope_theta": 16000.0,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "moe_intermediate_size": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 1,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.25,
        "norm_topk_prob": True,
        "swiglu_limit": 0.5,
        "hidden_act": "silu",
        "max_position_embeddings": 256,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "rope_type": "yarn",
            "factor": 1.0,
            "original_max_position_embeddings": 256,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        },
        "rms_norm_eps": 1e-6,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "ad_rope_cache_len": 256,
        "ad_compress_max_seq_len": 256,
        "attention_bias": False,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return DeepseekV4Config(**values)


def _fp32_head_logits(hidden_fp32: torch.Tensor, w_fp32: torch.Tensor) -> torch.Tensor:
    """Gate-off model tail: fp32 hidden x fp32 (bf16-valued) weight."""
    return _linear(hidden_fp32, w_fp32, None, layer_type="lm_head").float()


def _bf16_head_logits(hidden_fp32: torch.Tensor, w_bf16: torch.Tensor) -> torch.Tensor:
    """Gate-on model tail: bf16 boundary cast, bf16 weight, fp32 widen after."""
    return _linear(hidden_fp32.to(torch.bfloat16), w_bf16, None, layer_type="lm_head").float()


@_requires_cuda
def test_bf16_head_bitexact_vs_definition() -> None:
    """Gate-on chain must equal F.linear(h.bfloat16(), w_bf16).float() bit-exactly.

    Pinned at the production decode shape (per-rank vocab shard x hidden, one
    token): there both the routed ``trtllm::cublas_mm`` and aten's cuBLAS pick
    fp32-accumulating kernels that agree bitwise. (Tiny shapes can pick
    different split-K kernels whose accumulation order differs by ~1 ulp, so
    the definitional identity is asserted where the model actually runs.)
    """
    torch.manual_seed(0)
    vocab, hidden = 129280 // 4, 4096  # per-rank TP4 vocab shard of the real head
    w_bf16 = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    h = torch.randn(1, 1, hidden, dtype=torch.float32, device="cuda")

    logits = _bf16_head_logits(h, w_bf16)
    reference = F.linear(h.bfloat16(), w_bf16).float()
    assert logits.dtype == torch.float32
    assert torch.equal(logits, reference)


@_requires_cuda
def test_bf16_head_close_to_fp32_head() -> None:
    """bf16-path logits vs fp32-path logits under a derived rounding bound.

    Both paths use the same bf16-valued weight, so the difference is bounded by
    (a) activation rounding, |h16 - h| <= 2**-8 |h|, which perturbs each dot
    product by at most 2**-8 * sum_k |h_k * w_k|, plus (b) bf16 rounding of the
    GEMM output, <= 2**-8 |y|. Accumulation is fp32 in both paths, so
    order-dependent error is negligible; 1.05x covers its interaction with (b).
    """
    torch.manual_seed(1)
    vocab, hidden = 513, 256
    w_bf16 = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    w_fp32 = w_bf16.float()
    h = torch.randn(4, hidden, dtype=torch.float32, device="cuda")

    logits_fp32 = _fp32_head_logits(h, w_fp32)
    logits_bf16 = _bf16_head_logits(h, w_bf16)

    bound = _BF16_REL * (h.abs() @ w_fp32.abs().T + logits_fp32.abs()) * 1.05 + 1e-5
    diff = (logits_fp32 - logits_bf16).abs()
    assert torch.all(diff <= bound), f"max excess {(diff - bound).max().item()}"


@_requires_cuda
def test_bf16_head_on_bf16_representable_hidden() -> None:
    """In-model case: the fused HC head op emits bf16-rounded values widened to
    fp32, so the gate's boundary cast is lossless and only the GEMM-output
    rounding remains (2 ulp: 1 for the bf16 store, 1 for an accumulation-order
    rounding flip between the two cuBLAS kernels)."""
    torch.manual_seed(2)
    vocab, hidden = 129, 64
    w_bf16 = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    # bf16-representable fp32 hidden, as produced by the fused HC head op.
    h = torch.randn(3, hidden, dtype=torch.bfloat16, device="cuda").float()

    logits_fp32 = _fp32_head_logits(h, w_bf16.float())
    logits_bf16 = _bf16_head_logits(h, w_bf16)

    bound = 2 * _BF16_REL * logits_fp32.abs() + 1e-5
    assert torch.all((logits_fp32 - logits_bf16).abs() <= bound)


def test_gate_construction_dtype(monkeypatch) -> None:
    """Gate off -> fp32 head param (unchanged default); gate on -> bf16."""
    monkeypatch.setattr(dsv4, "_AD_BF16_LM_HEAD", False)
    model = DeepseekV4ForCausalLM(_small_config()).eval()
    assert model.head.weight.dtype == torch.float32

    monkeypatch.setattr(dsv4, "_AD_BF16_LM_HEAD", True)
    model = DeepseekV4ForCausalLM(_small_config()).eval()
    assert model.head.weight.dtype == torch.bfloat16


def test_gate_in_pipeline_cache_identifier(monkeypatch, tmp_path) -> None:
    """The gate changes the traced graph, so it must split the AutoDeploy
    pipeline-cache key: gated/ungated runs must not restore each other's
    pre-weight graph snapshot."""
    factory = dsv4.DeepseekV4AutoModelForCausalLMFactory(model=str(tmp_path))

    monkeypatch.setattr(dsv4, "_AD_BF16_LM_HEAD", False)
    ident_off = factory.get_pipeline_cache_model_identifier()
    monkeypatch.setattr(dsv4, "_AD_BF16_LM_HEAD", True)
    ident_on = factory.get_pipeline_cache_model_identifier()

    assert ident_off["ad_bf16_lm_head"] is False
    assert ident_on["ad_bf16_lm_head"] is True
    assert ident_off != ident_on


def _export_head_linear(monkeypatch, gate: bool):
    """Export the tiny model with the gate patched; return (gm, lm-head linear node)."""
    monkeypatch.setattr(dsv4, "_AD_BF16_LM_HEAD", gate)
    torch.manual_seed(3)
    # Real-model-shaped head dims: the fused rope/fp8 CUDA kernels require the
    # nope slice (head_dim - qk_rope_head_dim) to be a multiple of 64.
    config = _small_config(head_dim=128, qk_rope_head_dim=64, index_head_dim=128)
    model = DeepseekV4ForCausalLM(config).eval().to("cuda")
    input_ids = torch.randint(0, config.vocab_size, (2, 6), device="cuda")
    position_ids = torch.arange(6, device="cuda").unsqueeze(0).expand(2, -1)
    gm = torch_export_to_gm(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        dynamic_shapes={
            "input_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
            "position_ids": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        },
        num_moe_experts_for_export=2,
    )
    head_linears = [
        node
        for node in gm.graph.nodes
        if is_op(node, torch.ops.auto_deploy.torch_linear_simple)
        and extract_op_args(node, "layer_type")[0] == "lm_head"
    ]
    assert len(head_linears) == 1
    return gm, head_linears[0]


@_requires_cuda
@pytest.mark.parametrize("gate", [False, True])
def test_gate_export_graph_head_region(monkeypatch, gate: bool) -> None:
    """Gate off: fp32 weight, linear fed directly by the fused HC head op (no
    cast node — same head region as before the gate existed). Gate on: bf16
    weight, a bf16 boundary cast between the HC head op and the linear, and the
    logits still widen to fp32."""
    gm, lin = _export_head_linear(monkeypatch, gate)

    expected = torch.bfloat16 if gate else torch.float32
    act, weight = lin.args[0], lin.args[1]
    assert weight.meta["val"].dtype == expected
    assert act.meta["val"].dtype == expected
    assert lin.meta["val"].dtype == expected

    hc_head_op = torch.ops.auto_deploy.deepseek_v4_hc_head_norm
    if gate:
        # Boundary cast feeding the linear, sourced from the fused HC head op.
        assert not is_op(act, hc_head_op)
        (cast_src,) = [a for a in act.all_input_nodes]
        assert is_op(cast_src, hc_head_op)
    else:
        assert is_op(act, hc_head_op)

    # The graph output logits are fp32 in both paths.
    output_node = next(iter(reversed(gm.graph.nodes)))
    assert output_node.op == "output"
    for out in output_node.all_input_nodes:
        assert out.meta["val"].dtype == torch.float32
