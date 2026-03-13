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

"""Tests for MiMo-V2-Flash custom model implementation.

Architecture: Hybrid attention (full causal + SWA), partial rotary, different
Q/K vs V head dims, MoE with noaux_tc routing, attention sink bias.

Since MiMo-V2-Flash is NOT in the installed transformers, HF reference classes
are defined inline below (minimal faithful copies from the HF modeling code).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.activations import ACT2FN

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mimo_v2_flash import (
    MiMoV2FlashAttention,
    MiMoV2FlashConfig,
    MiMoV2FlashDecoderLayer,
    MiMoV2FlashForCausalLM,
    MiMoV2FlashMLP,
    MiMoV2FlashMoE,
    MiMoV2FlashRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =========================================================================
# Small test config
# =========================================================================


def _create_small_config() -> MiMoV2FlashConfig:
    """Create a small MiMo-V2-Flash config for testing.

    Uses 3 layers: layer 0 = full attention + dense MLP,
    layers 1-2 = SWA + MoE. This covers both layer types.
    """
    return MiMoV2FlashConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        v_head_dim=8,
        hidden_act="silu",
        max_position_embeddings=512,
        layernorm_epsilon=1e-5,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,  # rope_dim = 8
        attention_bias=False,
        # Hybrid: 0=full, 1=SWA
        hybrid_layer_pattern=[0, 1, 1],
        # SWA params
        swa_rope_theta=5000.0,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        swa_v_head_dim=8,
        sliding_window=4,
        sliding_window_size=4,
        # Sink
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        # MoE
        moe_layer_freq=[0, 1, 1],
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=None,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        scoring_func="sigmoid",
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        routed_scaling_factor=1.0,
        # Other
        tie_word_embeddings=False,
        initializer_range=0.02,
    )


# =========================================================================
# HF reference classes (minimal faithful copies for equivalence testing)
# The MiMo-V2-Flash model is not in the installed transformers library,
# so we define standalone HF reference classes here.
# These use per-expert nn.ModuleList format matching the checkpoint.
# =========================================================================


def _hf_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_hf_rotate_half(q) * sin)
    k_embed = (k * cos) + (_hf_rotate_half(k) * sin)
    return q_embed, k_embed


def _hf_repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class HFMiMoV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HFMiMoV2MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class HFMiMoV2MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = (
            config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0
        )
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.e_score_correction_bias = nn.Parameter(torch.empty(self.n_routed_experts))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()
        n = bsz * seq_len
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(n, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(n, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(n, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class HFMiMoV2MoE(nn.Module):
    """HF-style MoE with per-token expert loop (reference implementation)."""

    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList(
            [
                HFMiMoV2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = HFMiMoV2MoEGate(config)

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        final = torch.zeros_like(hidden_flat, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts)).permute(2, 0, 1)
        for expert_idx in range(len(self.experts)):
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_flat[token_indices]
                expert_output = self.experts[expert_idx](expert_input)
                final.index_add_(0, token_indices, expert_output * expert_weights.unsqueeze(-1))
        return final.to(hidden_flat.dtype).view(*orig_shape)


class HFMiMoV2FlashRotaryEmbedding(nn.Module):
    """Minimal HF-style rotary embedding for testing."""

    def __init__(self, rope_dim, max_position_embeddings=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class HFMiMoV2Attention(nn.Module):
    """HF-style attention with sink support for reference testing."""

    def __init__(self, config, is_swa, layer_idx):
        super().__init__()
        self.is_swa = is_swa
        if is_swa:
            self.head_dim = config.swa_head_dim
            self.v_head_dim = config.swa_v_head_dim
            self.num_heads = config.swa_num_attention_heads
            self.num_kv_heads = config.swa_num_key_value_heads
        else:
            self.head_dim = config.head_dim
            self.v_head_dim = config.v_head_dim
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads

        self.rope_dim = int(self.head_dim * config.partial_rotary_factor)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** (-0.5)

        q_size = self.num_heads * self.head_dim
        k_size = self.num_kv_heads * self.head_dim
        v_size = self.num_kv_heads * self.v_head_dim
        o_size = self.num_heads * self.v_head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, k_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, v_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(o_size, config.hidden_size, bias=False)

        has_sink = (config.add_full_attention_sink_bias and not is_swa) or (
            config.add_swa_attention_sink_bias and is_swa
        )
        if has_sink:
            self.attention_sink_bias = nn.Parameter(
                torch.empty(self.num_heads), requires_grad=False
            )
        else:
            self.attention_sink_bias = None

    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.size()
        qk_shape = (bsz, q_len, -1, self.head_dim)
        v_shape = (bsz, q_len, -1, self.v_head_dim)

        q = self.q_proj(hidden_states).view(qk_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(qk_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(v_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q_rope, q_nope = q.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        k_rope, k_nope = k.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        q_rope, k_rope = _hf_apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q = torch.cat([q_rope, q_nope], dim=-1)
        k = torch.cat([k_rope, k_nope], dim=-1)

        # GQA: repeat KV
        k = _hf_repeat_kv(k, self.num_kv_groups)
        v = _hf_repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(q_len, q_len, device=q.device, dtype=torch.bool), diagonal=1
        )
        attn_weights.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Sinks
        if self.attention_sink_bias is not None:
            sinks = self.attention_sink_bias.reshape(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
            logits_max = torch.max(attn_weights, dim=-1, keepdim=True).values
            sink_exp = torch.exp(sinks - logits_max)
            attn_exp = torch.exp(attn_weights - logits_max)
            normalizer = attn_exp.sum(dim=-1, keepdim=True) + sink_exp
            attn_weights = attn_exp / normalizer
        else:
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class HFMiMoV2DecoderLayer(nn.Module):
    """HF-style decoder layer for reference testing."""

    def __init__(self, config, layer_idx):
        super().__init__()
        is_swa = (
            config.hybrid_layer_pattern is not None and config.hybrid_layer_pattern[layer_idx] == 1
        )
        self.is_swa = is_swa
        self.self_attn = HFMiMoV2Attention(config, is_swa=is_swa, layer_idx=layer_idx)
        use_moe = (
            config.n_routed_experts is not None
            and config.moe_layer_freq is not None
            and config.moe_layer_freq[layer_idx] == 1
        )
        self.mlp = HFMiMoV2MoE(config) if use_moe else HFMiMoV2MLP(config)
        self.input_layernorm = HFMiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = HFMiMoV2RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )

    def forward(self, hidden_states, position_embeddings):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HFMiMoV2FlashModel(nn.Module):
    """HF-style full model for reference testing."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [HFMiMoV2DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = HFMiMoV2RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        rope_dim = int(config.head_dim * config.partial_rotary_factor)
        self.rotary_emb = HFMiMoV2FlashRotaryEmbedding(
            rope_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )
        self.swa_rotary_emb = HFMiMoV2FlashRotaryEmbedding(
            rope_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.swa_rope_theta,
        )

    def forward(self, input_ids, position_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        full_pos = self.rotary_emb(inputs_embeds, position_ids)
        swa_pos = self.swa_rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            pos_emb = swa_pos if layer.is_swa else full_pos
            hidden_states = layer(hidden_states, pos_emb)
        return self.norm(hidden_states)


class HFMiMoV2FlashForCausalLM(nn.Module):
    """HF-style CausalLM for reference testing."""

    def __init__(self, config):
        super().__init__()
        self.model = HFMiMoV2FlashModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids):
        hidden_states = self.model(input_ids, position_ids)
        return self.lm_head(hidden_states).float()


# =========================================================================
# Helper to get position embeddings for HF ref and custom model
# =========================================================================


def _get_hf_pos_emb(config, x, position_ids, is_swa=False):
    """Compute position embeddings in HF style (already sliced by position_ids)."""
    rope_dim = int(config.head_dim * config.partial_rotary_factor)
    base = config.swa_rope_theta if is_swa else config.rope_theta
    rotary = HFMiMoV2FlashRotaryEmbedding(rope_dim, base=base).to(device=x.device)
    return rotary(x, position_ids)


def _get_custom_pos_emb(config, x, is_swa=False):
    """Compute position embeddings in AD style (full cache, not sliced)."""
    rope_dim = int(config.head_dim * config.partial_rotary_factor)
    base = config.swa_rope_theta if is_swa else config.rope_theta
    rotary = MiMoV2FlashRotaryEmbedding(rope_dim, base=base).to(device=x.device, dtype=x.dtype)
    return rotary(x)


# =========================================================================
# Block-level tests (CPU)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_mlp_equivalence(B, S, dtype):
    """Test MLP block produces equivalent output to HF reference."""
    config = _create_small_config()

    hf_mlp = HFMiMoV2MLP(config)
    hf_mlp.to(dtype=dtype)
    hf_mlp.eval()

    custom_mlp = MiMoV2FlashMLP(config)
    custom_mlp.to(dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, dtype=dtype)
    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_attention_full_equivalence(B, S, dtype):
    """Test full attention (non-SWA) block produces equivalent output."""
    device = "cuda"
    config = _create_small_config()

    hf_attn = HFMiMoV2Attention(config, is_swa=False, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = MiMoV2FlashAttention(config, is_swa=False, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_pos_emb = _get_hf_pos_emb(config, x, position_ids, is_swa=False)
    hf_out = hf_attn(x, hf_pos_emb)

    custom_pos_emb = _get_custom_pos_emb(config, x, is_swa=False)
    custom_out = custom_attn(x, position_ids, custom_pos_emb)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Full attention")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_attention_swa_with_sinks_equivalence(B, S, dtype):
    """Test SWA attention with sink bias produces equivalent output."""
    device = "cuda"
    config = _create_small_config()

    hf_attn = HFMiMoV2Attention(config, is_swa=True, layer_idx=1)
    hf_attn.to(device=device, dtype=dtype)
    if hf_attn.attention_sink_bias is not None:
        hf_attn.attention_sink_bias.data = torch.randn(
            config.swa_num_attention_heads, device=device
        )
    hf_attn.eval()

    custom_attn = MiMoV2FlashAttention(config, is_swa=True, layer_idx=1)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_pos_emb = _get_hf_pos_emb(config, x, position_ids, is_swa=True)
    hf_out = hf_attn(x, hf_pos_emb)

    custom_pos_emb = _get_custom_pos_emb(config, x, is_swa=True)
    custom_out = custom_attn(x, position_ids, custom_pos_emb)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="SWA attention with sinks")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_moe_equivalence(B, S, dtype):
    """Test MoE block produces equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_moe = HFMiMoV2MoE(config)
    hf_moe.to(device=device, dtype=dtype)
    hf_moe.gate.weight = nn.Parameter(torch.randn_like(hf_moe.gate.weight))
    hf_moe.gate.e_score_correction_bias = nn.Parameter(
        torch.randn_like(hf_moe.gate.e_score_correction_bias)
    )
    hf_moe.eval()

    custom_moe = MiMoV2FlashMoE(config)
    custom_moe.to(device=device, dtype=dtype)
    custom_moe.load_state_dict(hf_moe.state_dict())
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    hf_out = hf_moe(x)
    custom_out = custom_moe(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE block")


# =========================================================================
# Layer-level tests
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_decoder_layer_full_equivalence(B, S, dtype):
    """Test full-attention decoder layer (layer 0: full attn + dense MLP)."""
    device = "cuda"
    config = _create_small_config()
    layer_idx = 0

    hf_layer = HFMiMoV2DecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = MiMoV2FlashDecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_pos_emb = _get_hf_pos_emb(config, x, position_ids, is_swa=False)
    hf_out = hf_layer(x, hf_pos_emb)

    custom_pos_emb = _get_custom_pos_emb(config, x, is_swa=False)
    custom_out = custom_layer(x, position_ids, custom_pos_emb)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Full-attn decoder layer")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_decoder_layer_swa_moe_equivalence(B, S, dtype):
    """Test SWA + MoE decoder layer (layers 1-2)."""
    device = "cuda"
    config = _create_small_config()
    layer_idx = 1

    hf_layer = HFMiMoV2DecoderLayer(config, layer_idx=layer_idx)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.mlp.gate.weight = nn.Parameter(torch.randn_like(hf_layer.mlp.gate.weight))
    hf_layer.mlp.gate.e_score_correction_bias = nn.Parameter(
        torch.randn_like(hf_layer.mlp.gate.e_score_correction_bias)
    )
    if hf_layer.self_attn.attention_sink_bias is not None:
        hf_layer.self_attn.attention_sink_bias.data = torch.randn(
            config.swa_num_attention_heads, device=device
        )
    hf_layer.eval()

    custom_layer = MiMoV2FlashDecoderLayer(config, layer_idx=layer_idx)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_pos_emb = _get_hf_pos_emb(config, x, position_ids, is_swa=True)
    hf_out = hf_layer(x, hf_pos_emb)

    custom_pos_emb = _get_custom_pos_emb(config, x, is_swa=True)
    custom_out = custom_layer(x, position_ids, custom_pos_emb)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="SWA+MoE decoder layer")


# =========================================================================
# Full model equivalence test
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mimo_v2_flash_full_model_numerical_equivalence(B, S, dtype):
    """Test full model logits match HF reference implementation."""
    device = "cuda"
    config = _create_small_config()

    hf_model = HFMiMoV2FlashForCausalLM(config)
    hf_model.to(device=device, dtype=dtype)
    for module in hf_model.modules():
        if isinstance(module, HFMiMoV2MoEGate):
            module.weight = nn.Parameter(torch.randn_like(module.weight))
            module.e_score_correction_bias = nn.Parameter(
                torch.randn_like(module.e_score_correction_bias)
            )
        if isinstance(module, HFMiMoV2Attention) and module.attention_sink_bias is not None:
            module.attention_sink_bias.data = torch.randn_like(module.attention_sink_bias)
    hf_model.eval()

    custom_model = MiMoV2FlashForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids, position_ids)
    custom_logits = custom_model(input_ids=input_ids, position_ids=position_ids).logits

    assert_rmse_close(custom_logits, hf_logits, rmse_ratio_tol=0.05, msg="Full model logits")


# =========================================================================
# Export test
# =========================================================================


def test_mimo_v2_flash_model_can_be_exported():
    """Test that the model exports with torch_export_to_gm and dynamic shapes.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces outputs with correct shape and finite values
    3. Dynamic shapes work with different batch/sequence sizes
    4. Exported model produces numerically equivalent output to the original
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = MiMoV2FlashForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Get reference output before export
    with torch.inference_mode():
        ref_logits = model(input_ids=input_ids, position_ids=position_ids).logits

    batch_dim = Dim.DYNAMIC
    seq_dim = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim, 1: seq_dim},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out
    logits = out["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all()

    # Verify numerical equivalence between original and exported model
    assert_rmse_close(logits, ref_logits, rmse_ratio_tol=0.05, msg="Exported model logits")

    # Test with different shape for dynamic shapes
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    assert out2["logits"].shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(out2["logits"]).all()


# =========================================================================
# Structural tests (CPU, no GPU needed)
# =========================================================================


def test_mimo_v2_flash_layer_types():
    """Test that layer types match hybrid_layer_pattern and moe_layer_freq."""
    config = _create_small_config()
    model = MiMoV2FlashForCausalLM(config)

    layer0 = model.model.layers[0]
    assert not layer0.is_swa
    assert type(layer0.mlp).__name__ == "MiMoV2FlashMLP"
    assert layer0.self_attn.attention_sink_bias is None

    for i in [1, 2]:
        layer = model.model.layers[i]
        assert layer.is_swa
        assert type(layer.mlp).__name__ == "MiMoV2FlashMoE"
        assert layer.self_attn.attention_sink_bias is not None


def test_mimo_v2_flash_expert_structure():
    """Test that MoE experts have correct structure for checkpoint loading."""
    config = _create_small_config()
    model = MiMoV2FlashForCausalLM(config)
    moe = model.model.layers[1].mlp

    assert isinstance(moe.experts, nn.ModuleList)
    assert len(moe.experts) == config.n_routed_experts

    state_dict = moe.state_dict()
    expected_keys = [
        "experts.0.gate_proj.weight",
        "experts.0.up_proj.weight",
        "experts.0.down_proj.weight",
        "gate.weight",
        "gate.e_score_correction_bias",
    ]
    for key in expected_keys:
        assert key in state_dict, f"Missing key '{key}' in MoE state_dict"


def test_mimo_v2_flash_config_registration():
    """Test that config model_type is correct."""
    config = _create_small_config()
    assert config.model_type == "mimo_v2_flash"
    assert hasattr(config, "hybrid_layer_pattern")
    assert hasattr(config, "moe_layer_freq")
    assert hasattr(config, "swa_rope_theta")
