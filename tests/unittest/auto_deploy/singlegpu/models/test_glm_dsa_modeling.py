# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Hierarchical equivalence tests for the GLM MoE DSA (GLM-5) custom model.

Since glm_moe_dsa is not in transformers 4.57, this file contains inline
HF reference classes copied from:
  https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa

Test structure (bottom-up):
  1. Block equivalence   — RMSNorm, MLP, MoE gate routing, MoE layer
  2. Layer equivalence   — full decoder layer (dense and MoE variants)
  3. Full model          — end-to-end logits comparison
  4. Export              — torch_export_to_gm, dynamic shapes, finite output
"""

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy  # noqa: F401 — registers custom ops
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm_dsa import (
    GlmDSADecoderLayer,
    GlmDSAForCausalLM,
    GlmDSAIndexer,
    GlmDSAMLP,
    GlmDSAMoE,
    GlmDSARMSNorm,
    GlmMoeDsaConfig,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

# ---------------------------------------------------------------------------
# Small test config
# ---------------------------------------------------------------------------

_BATCH_AND_SEQ = ((1, 6), (2, 4))


def _small_config(num_hidden_layers: int = 3, first_k_dense_replace: int = 1) -> GlmMoeDsaConfig:
    """Return a tiny GLM-5-like config suitable for CPU tests."""
    return GlmMoeDsaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        # MLA
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        # MoE
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=first_k_dense_replace,
        moe_layer_freq=1,
        # Indexer
        index_topk=4,
        index_head_dim=16,
        index_n_heads=2,
        indexer_rope_interleave=False,  # skip de-interleave for random-weight tests
        # RoPE
        rope_theta=10000.0,
        rope_scaling=None,
        rope_interleave=False,
        # Other
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


# ===========================================================================
# Inline HF reference classes (copied from transformers glm_moe_dsa)
# These are used as numerical ground truth since the model type is not in
# transformers 4.57.  Keep them minimal and identical to the upstream source.
# ===========================================================================


class _HFRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class _HFMoeDsaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn="silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _HFNaiveMoE(nn.Module):
    """Stacked expert storage (HF checkpoint format)."""

    def __init__(self, n_routed_experts, hidden_size, moe_intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(n_routed_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(n_routed_experts, hidden_size, moe_intermediate_size)
        )

    def forward(self, hidden_states, topk_indices, topk_weights):
        """Reference MoE dispatch using scatter-reduce."""
        T, top_k = topk_indices.shape
        output = torch.zeros_like(hidden_states)

        for t in range(T):
            for k in range(top_k):
                expert_idx = topk_indices[t, k].item()
                w = topk_weights[t, k].item()
                gate_up = self.gate_up_proj[expert_idx]  # [2*mid, H]
                down = self.down_proj[expert_idx]  # [H, mid]
                mid = gate_up.shape[0] // 2
                x = hidden_states[t : t + 1]  # [1, H]
                gate_out = F.silu(x @ gate_up[:mid].t()) * (x @ gate_up[mid:].t())
                output[t] += w * (gate_out @ down.t()).squeeze(0)

        return output


def _hf_route_tokens(
    gate_weight,
    e_score_correction_bias,
    hidden_flat,
    n_group,
    topk_group,
    top_k,
    norm_topk_prob,
    routed_scaling_factor,
):
    """Vanilla PyTorch noaux_tc routing — mirrors HF route_tokens_to_experts."""
    router_logits = F.linear(hidden_flat.float(), gate_weight.float())
    scores = router_logits.sigmoid()
    scores_for_choice = scores + e_score_correction_bias

    group_scores = (
        scores_for_choice.view(-1, n_group, scores_for_choice.shape[-1] // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = group_scores.topk(topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, scores_for_choice.shape[-1] // n_group)
        .reshape(-1, scores_for_choice.shape[-1])
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = scores_for_choice.topk(top_k, dim=-1, sorted=False).indices
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


# ---------------------------------------------------------------------------
# Weight conversion helpers (HF stacked → per-expert for loading into custom)
# ---------------------------------------------------------------------------


def _stacked_to_per_expert_state_dict(full_sd: dict, config) -> dict:
    """Convert HF-style stacked expert weights to our per-expert ModuleList format."""
    out = {}
    n = config.n_routed_experts
    mid = config.moe_intermediate_size

    for k, v in full_sd.items():
        if ".mlp.experts.gate_up_proj" in k:
            prefix = k[: k.index(".experts.gate_up_proj") + len(".experts.")]
            for i in range(n):
                out[f"{prefix}{i}.gate_proj.weight"] = v[i, :mid]
                out[f"{prefix}{i}.up_proj.weight"] = v[i, mid:]
        elif ".mlp.experts.down_proj" in k:
            prefix = k[: k.index(".experts.down_proj") + len(".experts.")]
            for i in range(n):
                out[f"{prefix}{i}.down_proj.weight"] = v[i]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Inline reference attention, decoder layer, and full model classes.
# These replicate HF GlmMoeDsaAttention math WITHOUT using torch_dsa so
# they serve as independent ground-truth for equivalence tests.
# ---------------------------------------------------------------------------


class _HFGlmDsaAttention(nn.Module):
    """Reference MLA+DSA attention without torch_dsa custom op.

    Implements the full DSA computation using vanilla PyTorch ops
    (matching the math in torch_dsa.py) and is used as ground truth
    for test_attention_block_equivalence.
    """

    def __init__(self, config):
        super().__init__()
        H = config.hidden_size
        N = config.num_attention_heads
        nope = config.qk_nope_head_dim
        rope_dim = config.qk_rope_head_dim
        v = config.v_head_dim
        kv_lora = config.kv_lora_rank
        q_lora = config.q_lora_rank
        idx_heads = config.index_n_heads
        idx_dim = config.index_head_dim

        self.config = config

        self.q_a_proj = nn.Linear(H, q_lora, bias=False)
        self.q_a_layernorm = _HFRMSNorm(q_lora, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora, N * (nope + rope_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(H, kv_lora + rope_dim, bias=False)
        self.kv_a_layernorm = _HFRMSNorm(kv_lora, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(kv_lora, N * (nope + v), bias=False)
        self.o_proj = nn.Linear(N * v, H, bias=False)

        # Indexer (flat, not nested under .indexer)
        self.wq_b = nn.Linear(q_lora, idx_heads * idx_dim, bias=False)
        self.wk = nn.Linear(H, idx_dim, bias=False)
        self.k_norm = nn.LayerNorm(idx_dim, eps=1e-6)
        self.weights_proj = nn.Linear(H, idx_heads, bias=False)

        self.softmax_scale = (nope + rope_dim) ** (-0.5)

    @staticmethod
    def _rope_bsnd(x, cos_bsd, sin_bsd):
        """NeoX RoPE on [B, S, N, D]; cos/sin are [B, S, D] (full rope_dim).

        Uses the rotate-half formula: out = x * cos + rotate_half(x) * sin
        where rotate_half(x) = cat(-x[..., D//2:], x[..., :D//2]).
        """
        cos = cos_bsd.unsqueeze(2)  # [B, S, 1, D]
        sin = sin_bsd.unsqueeze(2)
        half = x.shape[-1] // 2
        x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos + x_rot * sin

    @staticmethod
    def _rope_bsd(x, cos_bsd, sin_bsd):
        """NeoX RoPE on [B, S, D]; cos/sin are [B, S, D] (full rope_dim)."""
        half = x.shape[-1] // 2
        x_rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
        return x * cos_bsd + x_rot * sin_bsd

    def _cos_sin(self, seq_len, device, dtype):
        """Vanilla RoPE table (no YaRN)."""
        rope_dim = self.config.qk_rope_head_dim
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
        )
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)  # [S, rope_dim]

    def forward(self, hidden_states, position_ids):
        bsz, q_len = hidden_states.shape[:2]
        device = hidden_states.device
        dtype = hidden_states.dtype
        N = self.config.num_attention_heads
        nope = self.config.qk_nope_head_dim
        rope_dim = self.config.qk_rope_head_dim
        v = self.config.v_head_dim
        kv_lora = self.config.kv_lora_rank
        idx_heads = self.config.index_n_heads
        idx_dim = self.config.index_head_dim

        # Q path
        qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(qr).view(bsz, q_len, N, nope + rope_dim)
        q_nope = q[..., :nope]
        q_pe = q[..., nope:]

        # KV path
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = self.kv_a_layernorm(kv_a[..., :kv_lora])
        k_pe_raw = kv_a[..., kv_lora:]  # [B, S, rope_dim]

        # RoPE
        cos_full, sin_full = self._cos_sin(q_len, device, dtype)
        cos = cos_full[position_ids]  # [B, S, rope_dim]
        sin = sin_full[position_ids]
        q_pe_rot = self._rope_bsnd(q_pe, cos, sin)
        k_pe_rot = self._rope_bsd(k_pe_raw, cos, sin)  # [B, S, rope_dim]

        # Indexer Q — layout: [rope | nope] within each head
        idx_q = self.wq_b(qr).view(bsz, q_len, idx_heads, idx_dim)
        idx_q_rope, idx_q_nope = idx_q[..., :rope_dim], idx_q[..., rope_dim:]
        idx_q = torch.cat([self._rope_bsnd(idx_q_rope, cos, sin), idx_q_nope], dim=-1)

        # Indexer K
        idx_k_raw = self.k_norm(self.wk(hidden_states))  # [B, S, idx_dim]
        idx_k_rope, idx_k_nope = idx_k_raw[..., :rope_dim], idx_k_raw[..., rope_dim:]
        idx_k = torch.cat([self._rope_bsd(idx_k_rope, cos, sin), idx_k_nope], dim=-1)

        # Indexer weights (pre-scaled)
        idx_weights = self.weights_proj(hidden_states) * (idx_heads**-0.5)  # [B, S, idx_heads]

        # Expand KV via kv_b_proj
        kv = torch.matmul(compressed_kv, self.kv_b_proj.weight.t())  # [B, S, N*(nope+v)]
        kv = kv.view(bsz, q_len, N, nope + v)
        k_nope_t = kv[..., :nope].transpose(1, 2)  # [B, N, S, nope]
        value_states = kv[..., nope:].transpose(1, 2)  # [B, N, S, v]

        # Full Q, K [B, N, S, qk_head_dim]
        q_full = torch.cat([q_nope.transpose(1, 2), q_pe_rot.transpose(1, 2)], dim=-1)
        k_pe_expand = k_pe_rot.unsqueeze(1).expand(bsz, N, q_len, rope_dim)
        k_full = torch.cat([k_nope_t, k_pe_expand], dim=-1)

        # Attention scores [B, N, S_q, S_k]
        attn = torch.matmul(q_full, k_full.transpose(-2, -1)) * self.softmax_scale

        # Causal mask
        causal = torch.triu(torch.ones(q_len, q_len, device=device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(causal[None, None], float("-inf"))

        # DSA index mask (same math as _compute_dsa_index_mask in torch_dsa.py)
        per_head = torch.einsum("bshd,btd->bsht", idx_q.float(), idx_k.float())
        idx_score = torch.einsum("bsht,bsh->bst", per_head, idx_weights.float())
        idx_score = idx_score * (idx_dim**-0.5)
        idx_score.masked_fill_(causal[None], float("-inf"))
        eff_topk = min(self.config.index_topk, q_len)
        topk_idx = idx_score.topk(eff_topk, dim=-1).indices
        idx_mask = idx_score.new_full(idx_score.shape, float("-inf"))
        idx_mask.scatter_(-1, topk_idx, 0.0)

        attn = attn + idx_mask[:, None, :, :]
        w = torch.softmax(attn, dim=-1, dtype=torch.float32).to(dtype)
        out = torch.matmul(w, value_states)  # [B, N, S, v]
        out = out.transpose(1, 2).reshape(bsz, q_len, N * v)
        return self.o_proj(out)


def _copy_attn_weights(custom_attn, ref_attn):
    """Copy weights from GlmDSAAttention → _HFGlmDsaAttention."""
    for name in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
        getattr(ref_attn, name).weight = getattr(custom_attn, name).weight
    ref_attn.q_a_layernorm.weight = custom_attn.q_a_layernorm.weight
    ref_attn.kv_a_layernorm.weight = custom_attn.kv_a_layernorm.weight
    ref_attn.wq_b.weight = custom_attn.indexer.wq_b.weight
    ref_attn.wk.weight = custom_attn.indexer.wk.weight
    ref_attn.k_norm.weight = custom_attn.indexer.k_norm.weight
    ref_attn.k_norm.bias = custom_attn.indexer.k_norm.bias
    ref_attn.weights_proj.weight = custom_attn.indexer.weights_proj.weight


class _HFDecoderLayer(nn.Module):
    """Reference decoder layer using _HFGlmDsaAttention + HF MLP references."""

    def __init__(self, config, is_moe=False):
        super().__init__()
        H = config.hidden_size
        self.is_moe = is_moe
        self.config = config

        self.input_layernorm = _HFRMSNorm(H, eps=config.rms_norm_eps)
        self.self_attn = _HFGlmDsaAttention(config)
        self.post_attention_layernorm = _HFRMSNorm(H, eps=config.rms_norm_eps)

        if is_moe:
            self.gate_weight = nn.Parameter(torch.empty(config.n_routed_experts, H))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
            self.routed_experts = _HFNaiveMoE(
                config.n_routed_experts, H, config.moe_intermediate_size
            )
            self.shared_expert = _HFMoeDsaMLP(
                H, config.moe_intermediate_size * config.n_shared_experts
            )
        else:
            self.mlp = _HFMoeDsaMLP(H, config.intermediate_size)

    def forward(self, hidden_states, position_ids):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)

        if self.is_moe:
            cfg = self.config
            T = normed.shape[0] * normed.shape[1]
            hidden_flat = normed.view(T, -1)
            topk_indices, topk_weights = _hf_route_tokens(
                self.gate_weight,
                self.e_score_correction_bias,
                hidden_flat,
                cfg.n_group,
                cfg.topk_group,
                cfg.num_experts_per_tok,
                cfg.norm_topk_prob,
                cfg.routed_scaling_factor,
            )
            routed = self.routed_experts(hidden_flat.float(), topk_indices, topk_weights.float())
            routed = routed.view(*normed.shape)
            shared = self.shared_expert(normed.float()).to(hidden_states.dtype)
            mlp_out = (routed + shared).to(hidden_states.dtype)
        else:
            mlp_out = self.mlp(normed)

        return residual + mlp_out


def _build_hf_decoder_layer(custom_layer, config, is_moe):
    """Build _HFDecoderLayer with weights copied from GlmDSADecoderLayer."""
    ref = _HFDecoderLayer(config, is_moe=is_moe)
    ref.input_layernorm.weight = custom_layer.input_layernorm.weight
    ref.post_attention_layernorm.weight = custom_layer.post_attention_layernorm.weight
    _copy_attn_weights(custom_layer.self_attn, ref.self_attn)

    if is_moe:
        cm = custom_layer.mlp
        ref.gate_weight = nn.Parameter(cm.gate.weight.data.clone())
        ref.e_score_correction_bias = nn.Parameter(cm.gate.e_score_correction_bias.data.clone())
        n = config.n_routed_experts
        mid = config.moe_intermediate_size
        for j in range(n):
            ref.routed_experts.gate_up_proj.data[j, :mid] = cm.experts[j].gate_proj.weight.data
            ref.routed_experts.gate_up_proj.data[j, mid:] = cm.experts[j].up_proj.weight.data
            ref.routed_experts.down_proj.data[j] = cm.experts[j].down_proj.weight.data
        ref.shared_expert.gate_proj.weight = cm.shared_experts.gate_proj.weight
        ref.shared_expert.up_proj.weight = cm.shared_experts.up_proj.weight
        ref.shared_expert.down_proj.weight = cm.shared_experts.down_proj.weight
    else:
        cm = custom_layer.mlp
        ref.mlp.gate_proj.weight = cm.gate_proj.weight
        ref.mlp.up_proj.weight = cm.up_proj.weight
        ref.mlp.down_proj.weight = cm.down_proj.weight

    return ref


class _HFGlmDsaForCausalLM(nn.Module):
    """Reference full model using all HF reference components."""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                _HFDecoderLayer(
                    config,
                    is_moe=(
                        config.n_routed_experts is not None
                        and i >= config.first_k_dense_replace
                        and i % config.moe_layer_freq == 0
                    ),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)
        return self.lm_head(self.norm(hidden_states))


def _build_hf_model(custom_model, config):
    """Build _HFGlmDsaForCausalLM with weights copied from GlmDSAForCausalLM."""
    ref = _HFGlmDsaForCausalLM(config)
    ref.embed_tokens.weight = custom_model.model.embed_tokens.weight
    ref.norm.weight = custom_model.model.norm.weight
    ref.lm_head.weight = custom_model.lm_head.weight

    for i, (cust_layer, ref_layer) in enumerate(zip(custom_model.model.layers, ref.layers)):
        ref_layer.input_layernorm.weight = cust_layer.input_layernorm.weight
        ref_layer.post_attention_layernorm.weight = cust_layer.post_attention_layernorm.weight
        _copy_attn_weights(cust_layer.self_attn, ref_layer.self_attn)

        if ref_layer.is_moe:
            cm = cust_layer.mlp
            ref_layer.gate_weight = nn.Parameter(cm.gate.weight.data.clone())
            ref_layer.e_score_correction_bias = nn.Parameter(
                cm.gate.e_score_correction_bias.data.clone()
            )
            n = config.n_routed_experts
            mid = config.moe_intermediate_size
            for j in range(n):
                ref_layer.routed_experts.gate_up_proj.data[j, :mid] = cm.experts[
                    j
                ].gate_proj.weight.data
                ref_layer.routed_experts.gate_up_proj.data[j, mid:] = cm.experts[
                    j
                ].up_proj.weight.data
                ref_layer.routed_experts.down_proj.data[j] = cm.experts[j].down_proj.weight.data
            ref_layer.shared_expert.gate_proj.weight = cm.shared_experts.gate_proj.weight
            ref_layer.shared_expert.up_proj.weight = cm.shared_experts.up_proj.weight
            ref_layer.shared_expert.down_proj.weight = cm.shared_experts.down_proj.weight
        else:
            cm = cust_layer.mlp
            ref_layer.mlp.gate_proj.weight = cm.gate_proj.weight
            ref_layer.mlp.up_proj.weight = cm.up_proj.weight
            ref_layer.mlp.down_proj.weight = cm.down_proj.weight

    return ref


# ===========================================================================
# 1. Block-level equivalence tests
# ===========================================================================


@torch.no_grad()
def test_rmsnorm_equivalence():
    """Custom RMSNorm matches HF reference (identical math)."""
    config = _small_config()
    custom = GlmDSARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    ref = _HFRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    ref.weight = custom.weight  # share weights

    x = torch.randn(2, 6, config.hidden_size)
    torch.testing.assert_close(custom(x), ref(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_mlp_equivalence(B, S):
    """Custom MLP matches HF MLP (identical math)."""
    config = _small_config()
    custom = GlmDSAMLP(config)
    ref = _HFMoeDsaMLP(config.hidden_size, config.intermediate_size)
    ref.gate_proj.weight = custom.gate_proj.weight
    ref.up_proj.weight = custom.up_proj.weight
    ref.down_proj.weight = custom.down_proj.weight

    x = torch.randn(B, S, config.hidden_size)
    torch.testing.assert_close(custom(x), ref(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_moe_gate_routing_equivalence(B, S):
    """Custom MoE gate routing matches HF reference routing (noaux_tc)."""
    config = _small_config()
    custom_moe = GlmDSAMoE(config)
    custom_moe.gate.weight = nn.Parameter(torch.randn_like(custom_moe.gate.weight))

    x = torch.randn(B, S, config.hidden_size)
    T = B * S
    hidden_flat = x.view(T, -1)

    # Custom model routing
    custom_indices, custom_weights = custom_moe.gate(x)

    # HF reference routing
    ref_indices, ref_weights = _hf_route_tokens(
        custom_moe.gate.weight,
        custom_moe.gate.e_score_correction_bias,
        hidden_flat,
        config.n_group,
        config.topk_group,
        config.num_experts_per_tok,
        config.norm_topk_prob,
        config.routed_scaling_factor,
    )

    # Indices might differ in order (topk sorted=False), but selected expert sets should match per token
    custom_sorted = custom_indices.sort(dim=-1).values
    ref_sorted = ref_indices.sort(dim=-1).values
    torch.testing.assert_close(custom_sorted, ref_sorted)
    # Weights (after sorting by index) should match
    custom_w_sorted = custom_weights.gather(1, custom_indices.argsort(dim=-1))
    ref_w_sorted = ref_weights.gather(1, ref_indices.argsort(dim=-1))
    torch.testing.assert_close(custom_w_sorted.float(), ref_w_sorted.float(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_moe_layer_equivalence(B, S):
    """Custom MoE output matches reference MoE using same weights and routing."""
    config = _small_config()
    custom_moe = GlmDSAMoE(config)
    custom_moe.gate.weight = nn.Parameter(torch.randn_like(custom_moe.gate.weight))

    # Build HF-style reference: use same weights, route identically, compute via loop
    hf_naive = _HFNaiveMoE(
        config.n_routed_experts, config.hidden_size, config.moe_intermediate_size
    )
    for i in range(config.n_routed_experts):
        hf_naive.gate_up_proj.data[i, : config.moe_intermediate_size] = custom_moe.experts[
            i
        ].gate_proj.weight.data
        hf_naive.gate_up_proj.data[i, config.moe_intermediate_size :] = custom_moe.experts[
            i
        ].up_proj.weight.data
        hf_naive.down_proj.data[i] = custom_moe.experts[i].down_proj.weight.data

    hf_shared_gate = custom_moe.shared_experts.gate_proj.weight.data.clone()
    hf_shared_up = custom_moe.shared_experts.up_proj.weight.data.clone()
    hf_shared_down = custom_moe.shared_experts.down_proj.weight.data.clone()

    x = torch.randn(B, S, config.hidden_size)
    T = B * S

    # Custom forward
    custom_out = custom_moe(x)

    # Reference forward
    topk_indices, topk_weights = _hf_route_tokens(
        custom_moe.gate.weight,
        custom_moe.gate.e_score_correction_bias,
        x.view(T, -1),
        config.n_group,
        config.topk_group,
        config.num_experts_per_tok,
        config.norm_topk_prob,
        config.routed_scaling_factor,
    )
    ref_routed = hf_naive(x.view(T, -1).float(), topk_indices, topk_weights.float())
    ref_routed = ref_routed.view(B, S, config.hidden_size)

    # Shared expert
    x_shared = x.float()
    shared_out = F.silu(x_shared @ hf_shared_gate.float().t()) * (
        x_shared @ hf_shared_up.float().t()
    )
    shared_out = shared_out @ hf_shared_down.float().t()

    ref_out = (ref_routed + shared_out).to(x.dtype)

    assert_rmse_close(custom_out.float(), ref_out.float(), rmse_ratio_tol=0.02, msg="MoE layer: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_attention_block_equivalence(B, S):
    """GlmDSAAttention (uses torch_dsa) matches _HFGlmDsaAttention (vanilla torch).

    This test verifies that the projection/RoPE/indexer wiring in GlmDSAAttention
    correctly feeds into the torch_dsa custom op by comparing against an independent
    reference that implements DSA using raw PyTorch ops.
    """
    config = _small_config()
    custom_attn = GlmDSADecoderLayer(config, layer_idx=0).self_attn
    custom_attn.eval()

    ref_attn = _HFGlmDsaAttention(config)
    _copy_attn_weights(custom_attn, ref_attn)
    ref_attn.eval()

    x = torch.randn(B, S, config.hidden_size)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    custom_out = custom_attn(x, pos)
    ref_out = ref_attn(x, pos)

    assert custom_out.shape == ref_out.shape
    assert_rmse_close(
        custom_out.float(), ref_out.float(), rmse_ratio_tol=0.10, msg="Attention block: "
    )


# ===========================================================================
# 2. Layer-level equivalence tests
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_dense_decoder_layer_equivalence(B, S):
    """Dense decoder layer matches HF reference decoder layer with same weights (CPU)."""
    config = _small_config(num_hidden_layers=3, first_k_dense_replace=1)

    layer = GlmDSADecoderLayer(config, layer_idx=0)  # dense layer
    assert isinstance(layer.mlp, GlmDSAMLP), "Layer 0 should be dense"
    layer.eval()

    ref = _build_hf_decoder_layer(layer, config, is_moe=False)
    ref.eval()

    x = torch.randn(B, S, config.hidden_size)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    custom_out = layer(x, pos)
    ref_out = ref(x, pos)

    assert custom_out.shape == x.shape
    assert torch.isfinite(custom_out).all()
    assert_rmse_close(
        custom_out.float(), ref_out.float(), rmse_ratio_tol=0.05, msg="Dense decoder layer: "
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_moe_decoder_layer_equivalence(B, S):
    """MoE decoder layer matches HF reference decoder layer with same weights (CPU)."""
    config = _small_config(num_hidden_layers=3, first_k_dense_replace=1)

    layer = GlmDSADecoderLayer(config, layer_idx=1)  # MoE layer
    assert isinstance(layer.mlp, GlmDSAMoE), "Layer 1 should be MoE"
    layer.eval()

    ref = _build_hf_decoder_layer(layer, config, is_moe=True)
    ref.eval()

    x = torch.randn(B, S, config.hidden_size)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    custom_out = layer(x, pos)
    ref_out = ref(x, pos)

    assert custom_out.shape == x.shape
    assert torch.isfinite(custom_out).all()
    assert_rmse_close(
        custom_out.float(), ref_out.float(), rmse_ratio_tol=0.05, msg="MoE decoder layer: "
    )


# ===========================================================================
# 3. Full-model equivalence test
# ===========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQ)
@torch.no_grad()
def test_full_model_equivalence(B, S):
    """Full model logits match HF reference model with same weights (CPU)."""
    config = _small_config(num_hidden_layers=3, first_k_dense_replace=1)

    model = GlmDSAForCausalLM(config)
    model.eval()

    ref = _build_hf_model(model, config)
    ref.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S))
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    custom_logits = model(input_ids=input_ids, position_ids=pos).logits
    ref_logits = ref(input_ids, pos)

    assert custom_logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(custom_logits).all(), "Logits contain NaN or Inf"
    assert_rmse_close(
        custom_logits.float(), ref_logits.float(), rmse_ratio_tol=0.05, msg="Full model: "
    )


@torch.no_grad()
def test_full_model_self_consistency():
    """Two forward passes with the same input produce identical outputs (CPU)."""
    config = _small_config(num_hidden_layers=3, first_k_dense_replace=1)

    model = GlmDSAForCausalLM(config)
    model.eval()

    B, S = 2, 6
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)

    out1 = model(input_ids=input_ids, position_ids=pos)
    out2 = model(input_ids=input_ids, position_ids=pos)
    torch.testing.assert_close(out1.logits, out2.logits)


# ===========================================================================
# 4. Structural / config tests
# ===========================================================================


def test_config_registration():
    """Config model_type is correct and has expected attributes."""
    config = _small_config()
    assert config.model_type == "glm_moe_dsa"
    assert hasattr(config, "index_topk")
    assert hasattr(config, "index_n_heads")
    assert hasattr(config, "index_head_dim")
    assert hasattr(config, "kv_lora_rank")
    assert hasattr(config, "qk_rope_head_dim")


def test_config_rope_theta_from_rope_parameters():
    """rope_theta is correctly extracted from rope_parameters dict."""
    config = GlmMoeDsaConfig(rope_parameters={"rope_theta": 500000.0, "rope_type": "default"})
    assert config.rope_theta == 500000.0


def test_layer_types():
    """Layer 0..first_k_dense_replace-1 are dense, rest are MoE."""
    config = _small_config(num_hidden_layers=4, first_k_dense_replace=2)
    model = GlmDSAForCausalLM(config)

    for i in range(2):
        assert isinstance(model.model.layers[i].mlp, GlmDSAMLP), f"Layer {i} should be dense"
    for i in range(2, 4):
        assert isinstance(model.model.layers[i].mlp, GlmDSAMoE), f"Layer {i} should be MoE"


def test_expert_structure():
    """MoE expert list has correct structure for checkpoint loading."""
    config = _small_config()
    moe = GlmDSAMoE(config)

    assert isinstance(moe.experts, nn.ModuleList)
    assert len(moe.experts) == config.n_routed_experts

    sd = moe.state_dict()
    for i in range(config.n_routed_experts):
        assert f"experts.{i}.gate_proj.weight" in sd
        assert f"experts.{i}.up_proj.weight" in sd
        assert f"experts.{i}.down_proj.weight" in sd


def test_indexer_structure():
    """GlmDSAIndexer has correct submodule names for checkpoint key matching."""
    config = _small_config()
    indexer = GlmDSAIndexer(config)

    assert hasattr(indexer, "wq_b")
    assert hasattr(indexer, "wk")
    assert hasattr(indexer, "k_norm")
    assert hasattr(indexer, "weights_proj")
    assert isinstance(indexer.k_norm, nn.LayerNorm)


def test_attention_indexer_submodule():
    """GlmDSAAttention exposes indexer as a submodule (for checkpoint key: self_attn.indexer.*)."""
    config = _small_config()
    model = GlmDSAForCausalLM(config)
    attn = model.model.layers[0].self_attn

    assert hasattr(attn, "indexer"), "indexer must be a submodule of self_attn"
    assert isinstance(attn.indexer, GlmDSAIndexer)


def test_moe_weight_expand_hook():
    """_moe_expert_expand_hook correctly expands stacked expert weights at load time."""
    config = _small_config(num_hidden_layers=2, first_k_dense_replace=1)
    model = GlmDSAForCausalLM(config)

    n = config.n_routed_experts
    mid = config.moe_intermediate_size
    H = config.hidden_size

    # Build a fake stacked-format state_dict (layer 1 is MoE)
    original_sd = model.state_dict()
    stacked_sd = {}
    for k, v in original_sd.items():
        if ".mlp.experts." in k and "gate_proj" in k:
            # We'll replace per-expert keys with stacked ones
            pass
        else:
            stacked_sd[k] = v

    # Stack the expert weights back (simulate HF checkpoint format)
    gate_up = torch.zeros(n, 2 * mid, H)
    down = torch.zeros(n, H, mid)
    for i in range(n):
        gate_up[i, :mid] = original_sd[f"model.layers.1.mlp.experts.{i}.gate_proj.weight"]
        gate_up[i, mid:] = original_sd[f"model.layers.1.mlp.experts.{i}.up_proj.weight"]
        down[i] = original_sd[f"model.layers.1.mlp.experts.{i}.down_proj.weight"]

    stacked_sd["model.layers.1.mlp.experts.gate_up_proj"] = gate_up
    stacked_sd["model.layers.1.mlp.experts.down_proj"] = down

    # Load — the hook should expand stacked → per-expert
    model2 = GlmDSAForCausalLM(config)
    model2.load_state_dict(stacked_sd)

    # Verify weights match
    for i in range(n):
        torch.testing.assert_close(
            model2.model.layers[1].mlp.experts[i].gate_proj.weight,
            original_sd[f"model.layers.1.mlp.experts.{i}.gate_proj.weight"],
        )


# ===========================================================================
# 5. Export test
# ===========================================================================


def test_model_export():
    """Model exports with torch_export_to_gm, produces finite output on two shapes."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _small_config(num_hidden_layers=2, first_k_dense_replace=1)

    model = GlmDSAForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )
    gm = torch_export_to_gm(
        model,
        args=(),
        kwargs={"input_ids": input_ids, "position_ids": pos},
        dynamic_shapes=dynamic_shapes,
    )
    move_to_device(gm, device)

    with torch.inference_mode():
        out = gm(input_ids=input_ids, position_ids=pos)

    assert "logits" in out
    assert out["logits"].shape == (B, S, config.vocab_size)
    assert torch.isfinite(out["logits"]).all()

    # Numerical equivalence between eager and exported graph
    with torch.no_grad():
        eager_out = model(input_ids=input_ids, position_ids=pos)
    assert_rmse_close(
        out["logits"].float(),
        eager_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (shape 1): ",
    )

    # Test second shape
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    pos2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out2 = gm(input_ids=input_ids2, position_ids=pos2)

    assert out2["logits"].shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(out2["logits"]).all()

    with torch.no_grad():
        eager_out2 = model(input_ids=input_ids2, position_ids=pos2)
    assert_rmse_close(
        out2["logits"].float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (shape 2): ",
    )
