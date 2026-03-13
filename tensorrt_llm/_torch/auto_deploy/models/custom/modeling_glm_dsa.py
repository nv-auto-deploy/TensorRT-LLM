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

"""AutoDeploy export-ready implementation of GLM MoE DSA (GLM-5).

Source: https://huggingface.co/zai-org/GLM-5 (model_type = "glm_moe_dsa")
        https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa

Differences from the HuggingFace reference:
- Prefill-only (no KV caching — caching is handled by AutoDeploy graph transforms)
- RMSNorm uses torch.ops.auto_deploy.torch_rmsnorm canonical op
- Attention uses torch_dsa custom op in BSND layout
- MoE gate uses vanilla PyTorch noaux_tc routing (AD transforms can replace with trtllm kernels)
- Indexer is a separate GlmDSAIndexer submodule matching checkpoint key names
- FP8 quantization and Hadamard rotation are omitted (orthogonal to correctness)
- RoPE weights are de-interleaved at load time via load hooks
- MoE expert weights are expanded from stacked checkpoint format to per-expert ModuleList at load time
- Bundled GlmMoeDsaConfig class (not yet in transformers 4.57)
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.utils import ActivationType

# =============================================================================
# Bundled Config
# =============================================================================


class GlmMoeDsaConfig(PretrainedConfig):
    """Configuration class for GLM MoE DSA (GLM-5).

    Bundled here because this model requires transformers 5.0+, but we run on 4.57.
    Field names match the checkpoint config.json exactly.
    """

    model_type = "glm_moe_dsa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 6144,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        # MLA parameters
        q_lora_rank: int = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        # MoE parameters
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 2.5,
        norm_topk_prob: bool = True,
        # Layer type control (checkpoint format uses first_k_dense_replace + moe_layer_freq)
        first_k_dense_replace: int = 3,
        moe_layer_freq: int = 1,
        # RoPE — accept both checkpoint format (rope_parameters dict) and direct rope_theta
        rope_theta: float = 1000000.0,
        rope_scaling: Optional[dict] = None,
        rope_parameters: Optional[dict] = None,  # checkpoint format
        rope_interleave: bool = True,
        # Indexer (DSA) parameters
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        indexer_rope_interleave: bool = True,
        # Other
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 154820,
        # Extra checkpoint fields that we ignore
        ep_size: int = 1,
        head_dim: int = 64,
        qk_head_dim: int = 256,
        num_nextn_predict_layers: int = 0,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        dtype: str = "bfloat16",
        pretraining_tp: int = 1,
        **kwargs,
    ):
        # Model dimensions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        # MLA
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # MoE
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq

        # RoPE — extract rope_theta from rope_parameters dict if provided
        if rope_parameters is not None and isinstance(rope_parameters, dict):
            self.rope_theta = rope_parameters.get("rope_theta", rope_theta)
        else:
            self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.rope_interleave = rope_interleave

        # Indexer (DSA)
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.indexer_rope_interleave = indexer_rope_interleave

        # Other
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            **kwargs,
        )


AutoConfig.register("glm_moe_dsa", GlmMoeDsaConfig, exist_ok=True)


# =============================================================================
# Building blocks
# =============================================================================


class GlmDSARMSNorm(nn.Module):
    """RMS Normalization using the canonical AutoDeploy torch_rmsnorm op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class GlmDSARotaryEmbedding(nn.Module):
    """Rotary Position Embedding (non-interleaved, NeoX style after de-interleave at load time)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("_ad_inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=self._ad_inv_freq.dtype)
        freqs = torch.outer(t, self._ad_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_ad_sin_cached", emb.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._ad_cos_cached.to(dtype=x.dtype, device=x.device),
            self._ad_sin_cached.to(dtype=x.dtype, device=x.device),
        )


class GlmDSAYarnRotaryEmbedding(GlmDSARotaryEmbedding):
    """YaRN-extended rotary embedding."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        low, high = self._yarn_find_correction_range(
            self.beta_fast, self.beta_slow, dim, self.base, self.original_max_position_embeddings
        )
        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("_ad_inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        _mscale = float(
            self._yarn_get_mscale(self.scaling_factor, self.mscale)
            / self._yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_ad_cos_cached", (emb.cos() * _mscale), persistent=False)
        self.register_buffer("_ad_sin_cached", (emb.sin() * _mscale), persistent=False)

    @staticmethod
    def _yarn_find_correction_dim(
        num_rotations: float, dim: int, base: float = 10000, max_position_embeddings: int = 2048
    ) -> float:
        return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_find_correction_range(
        self, low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
    ) -> Tuple[int, int]:
        low = math.floor(
            self._yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
        )
        return max(low, 0), min(high, dim - 1)

    @staticmethod
    def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    @staticmethod
    def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)


class GlmDSAMLP(nn.Module):
    """MLP with SwiGLU activation."""

    def __init__(
        self, config, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GlmDSAMoEGate(nn.Module):
    """MoE gate with noaux_tc top-k routing — vanilla PyTorch implementation.

    Matches HF GlmMoeDsaTopkRouter + GlmMoeDsaMoE.route_tokens_to_experts logic.
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # Router logits and sigmoid scoring (always float32)
        router_logits = F.linear(hidden_flat.float(), self.weight.float())  # [T, n_experts]
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias  # [T, n_experts]

        # Group-level top-k selection
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [T, n_group]
        group_idx = group_scores.topk(
            self.topk_group, dim=-1, sorted=False
        ).indices  # [T, topk_group]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )  # [T, n_experts]

        # Mask out non-selected groups, then take per-token top-k experts
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = scores_for_choice.topk(
            self.top_k, dim=-1, sorted=False
        ).indices  # [T, top_k]

        # Gather weights from original scores (without correction bias)
        topk_weights = scores.gather(1, topk_indices)  # [T, top_k]
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights


class GlmDSAMoE(nn.Module):
    """Mixture of Experts layer."""

    def __init__(self, config):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                GlmDSAMLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = GlmDSAMoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = GlmDSAMLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)

        final_hidden_states = torch.ops.auto_deploy.torch_moe(
            hidden_states.view(-1, hidden_states.shape[-1]),
            topk_indices,
            topk_weights,
            w1_weight=[e.gate_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[e.up_proj.weight for e in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
        )

        final_hidden_states = final_hidden_states.view(*orig_shape)
        if self.shared_experts is not None:
            final_hidden_states = final_hidden_states + self.shared_experts(identity)
        return final_hidden_states.to(hidden_states.dtype)


class GlmDSAIndexer(nn.Module):
    """DSA Indexer — computes per-token importance scores and returns top-k index keys.

    Submodule names match the HF GlmMoeDsaIndexer checkpoint keys:
      self_attn.indexer.wq_b, .wk, .k_norm, .weights_proj
    """

    def __init__(self, config):
        super().__init__()
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank

        # wq_b: q_lora_rank → index_n_heads * index_head_dim
        self.wq_b = nn.Linear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False
        )
        # wk: hidden_size → index_head_dim
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        # k_norm: LayerNorm on indexer key (eps=1e-6 matches HF)
        self.k_norm = nn.LayerNorm(config.index_head_dim, eps=1e-6)
        # weights_proj: hidden_size → index_n_heads (per-head importance scalars)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden_size]
        qr: torch.Tensor,  # [B, S, q_lora_rank] (shared from MLA Q path)
        cos: torch.Tensor,  # [B, S, qk_rope_head_dim]
        sin: torch.Tensor,  # [B, S, qk_rope_head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (index_q, index_k, index_weights) with RoPE applied."""
        bsz, q_len, _ = hidden_states.shape

        # ---- Indexer Q ----
        # wq_b projects qr → [B, S, index_n_heads, index_head_dim]
        # Layout: [rope_dim | nope_dim] within each head (matches HF checkpoint)
        index_q_raw = self.wq_b(qr).view(bsz, q_len, self.index_n_heads, self.index_head_dim)
        index_q_pe_raw = index_q_raw[:, :, :, : self.qk_rope_head_dim]
        index_q_nope = index_q_raw[:, :, :, self.qk_rope_head_dim :]

        # Apply RoPE to indexer Q pe part (use dummy k to satisfy op signature)
        index_k_dummy = index_q_raw.new_zeros(bsz, q_len, 1, self.qk_rope_head_dim)
        index_q_pe_rotated, _ = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            index_q_pe_raw, index_k_dummy, cos, sin, 2
        )
        # Recombine: [pe_rotated | nope]
        index_q = torch.cat([index_q_pe_rotated, index_q_nope], dim=-1)

        # ---- Indexer K ----
        # wk projects hidden_states → [B, S, index_head_dim]
        # Layout: [rope_dim | nope_dim]
        index_k_raw = self.k_norm(self.wk(hidden_states))
        index_k_pe_raw = index_k_raw[:, :, : self.qk_rope_head_dim]
        index_k_nope = index_k_raw[:, :, self.qk_rope_head_dim :]

        # Apply RoPE to indexer K pe part
        index_k_pe_4d = index_k_pe_raw.view(bsz, q_len, 1, self.qk_rope_head_dim)
        index_q_dummy = index_k_pe_4d.new_zeros(bsz, q_len, 1, self.qk_rope_head_dim)
        _, index_k_pe_rotated = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            index_q_dummy, index_k_pe_4d, cos, sin, 2
        )
        index_k_pe_rotated = index_k_pe_rotated.view(bsz, q_len, self.qk_rope_head_dim)
        # Recombine: [pe_rotated | nope]
        index_k = torch.cat([index_k_pe_rotated, index_k_nope], dim=-1)

        # ---- Indexer importance weights ----
        # weights_proj scaled by n_heads^(-0.5), matching HF
        index_weights = self.weights_proj(hidden_states) * (self.index_n_heads**-0.5)

        return index_q, index_k, index_weights


class GlmDSAAttention(nn.Module):
    """MLA + DSA (DeepSeek Sparse Attention) for GLM-5.

    The Indexer computes per-token importance scores; top-k positions are kept and
    the rest are masked to -inf before softmax.  RoPE is de-interleaved at load time.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        # Softmax scale (with optional YaRN mscale correction)
        self.softmax_scale = self.q_head_dim ** (-0.5)
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = GlmDSAYarnRotaryEmbedding._yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        # MLA projections
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = GlmDSARMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = GlmDSARMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        # Indexer submodule — names match HF checkpoint: self_attn.indexer.*
        self.indexer = GlmDSAIndexer(config)

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = GlmDSARotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "yarn":
                kwargs = {
                    k: self.config.rope_scaling[k]
                    for k in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if k in self.config.rope_scaling
                }
                self.rotary_emb = GlmDSAYarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                self.rotary_emb = GlmDSARotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        # ---- MLA Q path — qr is shared with the Indexer ----
        qr = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
        q = self.q_b_proj(qr).view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # ---- MLA KV path ----
        kv_a_output = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            kv_a_output, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        compressed_kv = self.kv_a_layernorm(compressed_kv)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)

        # ---- RoPE (weights de-interleaved at load time → NeoX-style) ----
        cos, sin = self.rotary_emb(hidden_states, seq_len=q_len)
        cos = cos[position_ids]  # [B, S, rope_head_dim]
        sin = sin[position_ids]

        q_pe_rotated, kpe = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
            q_pe,
            k_pe,
            cos,
            sin,
            2,  # unsqueeze_dim=2 for BSND layout
        )

        # ---- Indexer ----
        index_q, index_k, index_weights = self.indexer(hidden_states, qr, cos, sin)

        # ---- DSA attention ----
        attn_output = torch.ops.auto_deploy.torch_dsa(
            q_nope,  # [B, S, N, qk_nope_head_dim]
            q_pe_rotated,  # [B, S, N, qk_rope_head_dim]
            compressed_kv,  # [B, S, kv_lora_rank]
            kpe,  # [B, S, 1, qk_rope_head_dim]
            self.kv_b_proj.weight,  # [N*(qk_nope+v), kv_lora_rank]
            index_q,  # [B, S, index_n_heads, index_head_dim]
            index_k,  # [B, S, index_head_dim]
            index_weights,  # [B, S, index_n_heads]
            self.config.index_topk,
            True,  # is_causal
            self.softmax_scale,
            "bsnd",
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class GlmDSADecoderLayer(nn.Module):
    """Transformer decoder layer for GLM-5."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = GlmDSAAttention(config, layer_idx=layer_idx)

        use_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = GlmDSAMoE(config) if use_moe else GlmDSAMLP(config)

        self.input_layernorm = GlmDSARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GlmDSARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Top-level model classes
# =============================================================================


@dataclass
class GlmDSAOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class GlmDSACausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


class GlmDSAPreTrainedModel(PreTrainedModel):
    config_class = GlmMoeDsaConfig
    base_model_prefix = "model"
    _no_split_modules = ["GlmDSADecoderLayer"]
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GlmDSAModel(GlmDSAPreTrainedModel):
    """GLM-5 transformer decoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GlmDSADecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = GlmDSARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GlmDSAOutput:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert position_ids is not None, "position_ids must be provided for AD export"
        batch_size, seq_length = inputs_embeds.shape[:2]

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, position_ids)

        hidden_states = self.norm(hidden_states)
        return GlmDSAOutput(last_hidden_state=hidden_states)


class GlmDSAForCausalLM(GlmDSAPreTrainedModel, GenerationMixin):
    """GLM-5 model with language modeling head."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = GlmDSAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Load hook 1: De-interleave MLA RoPE weights (q_b_proj, kv_a_proj_with_mqa)
        self._register_load_state_dict_pre_hook(
            partial(
                _mla_rope_deinterleave_hook,
                qk_rope_head_dim=config.qk_rope_head_dim,
                qk_nope_head_dim=config.qk_nope_head_dim,
                num_heads=config.num_attention_heads,
                kv_lora_rank=config.kv_lora_rank,
                num_layers=config.num_hidden_layers,
            )
        )

        # Load hook 2: De-interleave indexer RoPE weights (indexer.wq_b, indexer.wk)
        if config.indexer_rope_interleave:
            self._register_load_state_dict_pre_hook(
                partial(
                    _indexer_rope_deinterleave_hook,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    index_n_heads=config.index_n_heads,
                    index_head_dim=config.index_head_dim,
                    num_layers=config.num_hidden_layers,
                )
            )

        # Load hook 3: Expand stacked MoE expert weights → per-expert ModuleList
        self._register_load_state_dict_pre_hook(
            partial(
                _moe_expert_expand_hook,
                n_routed_experts=config.n_routed_experts,
                moe_intermediate_size=config.moe_intermediate_size,
                first_k_dense_replace=config.first_k_dense_replace,
                moe_layer_freq=config.moe_layer_freq,
                num_layers=config.num_hidden_layers,
            )
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> GlmDSACausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return GlmDSACausalLMOutput(logits=logits)


# =============================================================================
# Load-time weight transformation hooks
# =============================================================================


def _mla_rope_deinterleave_hook(
    state_dict,
    prefix,
    *args,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    num_heads: int,
    kv_lora_rank: int,
    num_layers: int,
):
    """De-interleave MLA RoPE weights from GLM-5 interleaved format to NeoX format.

    For q_b_proj: output shape [num_heads * qk_head_dim, q_lora_rank].
      Each head has [nope_dim | rope_dim]; rope_dim is interleaved and needs reordering.
    For kv_a_proj_with_mqa: output shape [kv_lora_rank + qk_rope_head_dim, hidden_size].
      The last qk_rope_head_dim rows are the rope part and need reordering.
    """
    d = qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])
    qk_head_dim = qk_nope_head_dim + d

    for layer_idx in range(num_layers):
        layer_prefix = f"{prefix}model.layers.{layer_idx}.self_attn."

        q_key = layer_prefix + "q_b_proj.weight"
        if q_key in state_dict:
            w = state_dict[q_key]
            w = w.view(num_heads, qk_head_dim, -1)
            w_nope = w[:, :qk_nope_head_dim, :]
            w_rope = w[:, qk_nope_head_dim:, :]
            w_rope = w_rope[:, perm, :]
            state_dict[q_key] = torch.cat([w_nope, w_rope], dim=1).view(-1, w.shape[-1])

        kv_key = layer_prefix + "kv_a_proj_with_mqa.weight"
        if kv_key in state_dict:
            w = state_dict[kv_key]
            w_kv = w[:kv_lora_rank, :]
            w_pe = w[kv_lora_rank:, :]
            state_dict[kv_key] = torch.cat([w_kv, w_pe[perm, :]], dim=0)

        kv_bias_key = layer_prefix + "kv_a_proj_with_mqa.bias"
        if kv_bias_key in state_dict:
            b = state_dict[kv_bias_key]
            b_kv = b[:kv_lora_rank]
            b_pe = b[kv_lora_rank:]
            state_dict[kv_bias_key] = torch.cat([b_kv, b_pe[perm]])


def _indexer_rope_deinterleave_hook(
    state_dict,
    prefix,
    *args,
    qk_rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    num_layers: int,
):
    """De-interleave indexer RoPE weights from interleaved to non-interleaved format.

    Indexer layout within each head: [rope_dim | nope_dim] (rope is FIRST).

    For indexer.wq_b: shape [index_n_heads * index_head_dim, q_lora_rank].
      Each head has [rope_dim | nope_dim]; rope_dim needs reordering.
    For indexer.wk: shape [index_head_dim, hidden_size].
      First rope_dim rows are the rope part and need reordering.
    """
    d = qk_rope_head_dim
    perm = torch.cat([torch.arange(0, d, 2), torch.arange(1, d, 2)])

    for layer_idx in range(num_layers):
        idx_prefix = f"{prefix}model.layers.{layer_idx}.self_attn.indexer."

        wq_key = idx_prefix + "wq_b.weight"
        if wq_key in state_dict:
            w = state_dict[wq_key]
            # [index_n_heads * index_head_dim, q_lora_rank] → [n_heads, head_dim, q_lora_rank]
            w = w.view(index_n_heads, index_head_dim, -1)
            w_rope = w[:, :d, :]
            w_nope = w[:, d:, :]
            w_rope = w_rope[:, perm, :]
            state_dict[wq_key] = torch.cat([w_rope, w_nope], dim=1).view(-1, w.shape[-1])

        wk_key = idx_prefix + "wk.weight"
        if wk_key in state_dict:
            w = state_dict[wk_key]
            # [index_head_dim, hidden_size]
            w_rope = w[:d, :]
            w_nope = w[d:, :]
            state_dict[wk_key] = torch.cat([w_rope[perm, :], w_nope], dim=0)


def _moe_expert_expand_hook(
    state_dict,
    prefix,
    *args,
    n_routed_experts: int,
    moe_intermediate_size: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
    num_layers: int,
):
    """Expand stacked HF expert weights into per-expert format expected by GlmDSAMoE.

    HF checkpoint stores experts as:
      mlp.experts.gate_up_proj: [n_experts, 2 * moe_intermediate_size, hidden_size]
      mlp.experts.down_proj:    [n_experts, hidden_size, moe_intermediate_size]

    Our model expects:
      mlp.experts.{i}.gate_proj.weight: [moe_intermediate_size, hidden_size]
      mlp.experts.{i}.up_proj.weight:   [moe_intermediate_size, hidden_size]
      mlp.experts.{i}.down_proj.weight: [hidden_size, moe_intermediate_size]
    """
    for layer_idx in range(num_layers):
        is_moe = layer_idx >= first_k_dense_replace and layer_idx % moe_layer_freq == 0
        if not is_moe:
            continue

        mlp_prefix = f"{prefix}model.layers.{layer_idx}.mlp."
        gate_up_key = mlp_prefix + "experts.gate_up_proj"
        down_key = mlp_prefix + "experts.down_proj"

        if gate_up_key not in state_dict:
            continue

        gate_up = state_dict.pop(gate_up_key)  # [n_experts, 2*intermediate, hidden]
        down = state_dict.pop(down_key)  # [n_experts, hidden, intermediate]

        for i in range(n_routed_experts):
            gate_up_i = gate_up[i]  # [2*intermediate, hidden]
            state_dict[mlp_prefix + f"experts.{i}.gate_proj.weight"] = gate_up_i[
                :moe_intermediate_size
            ]
            state_dict[mlp_prefix + f"experts.{i}.up_proj.weight"] = gate_up_i[
                moe_intermediate_size:
            ]
            state_dict[mlp_prefix + f"experts.{i}.down_proj.weight"] = down[i]


# =============================================================================
# Registration
# =============================================================================

AutoModelForCausalLMFactory.register_custom_model_cls("GlmMoeDsaConfig", GlmDSAForCausalLM)
