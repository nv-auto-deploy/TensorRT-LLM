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

"""Tests for Skywork-R1V2 custom model implementation.

Skywork-R1V2-38B is an InternVL-based VLM with a Qwen2 LLM backbone.
The AD custom model only exports the LLM text path.  The LLM is standard
Qwen2 (GQA with bias on Q/K/V, SwiGLU MLP, RMSNorm, RoPE).

Since the HF config/model classes for this model are NOT in standard
transformers (they use trust_remote_code), we include standalone HF
reference implementations directly in this test file.
"""

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim
from transformers import Qwen2Config
from transformers.activations import ACT2FN

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_skywork_r1v2 import (
    SkyworkChatConfig,
    SkyworkR1V2Attention,
    SkyworkR1V2DecoderLayer,
    SkyworkR1V2ForCausalLM,
    SkyworkR1V2MLP,
    SkyworkR1V2RMSNorm,
    SkyworkR1V2RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

# SkyworkChatConfig is loaded from the HF cache (trust_remote_code=True).  If the
# model has not been downloaded, it will be None and all tests are skipped.
if SkyworkChatConfig is None:
    pytest.skip(
        "Skywork/Skywork-R1V2-38B not found in local HF cache; skipping tests.",
        allow_module_level=True,
    )

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Small test config
# ---------------------------------------------------------------------------


def _create_small_llm_config() -> Qwen2Config:
    """Create a small Qwen2 config for the LLM backbone.

    architectures is set explicitly so that to_dict() carries it through to
    SkyworkChatConfig, which uses it to select the backbone class.
    """
    return Qwen2Config(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        tie_word_embeddings=False,
    )


def _create_small_chat_config() -> SkyworkChatConfig:
    """Create a small SkyworkChatConfig wrapping the Qwen2 LLM config.

    tie_word_embeddings is read from the llm_config and forwarded to the outer
    config so that post_init() does not incorrectly tie lm_head to embed_tokens
    (PretrainedConfig defaults tie_word_embeddings=True if not passed explicitly).
    """
    llm_config = _create_small_llm_config()
    llm_dict = llm_config.to_dict()
    return SkyworkChatConfig(
        llm_config=llm_dict,
        tie_word_embeddings=llm_dict.get("tie_word_embeddings", False),
    )


# ---------------------------------------------------------------------------
# Standalone HF Qwen2 reference implementations (for equivalence tests)
#
# Why not import transformers.models.qwen2.modeling_qwen2 directly?
#
# 1. Attention SDPA behavior: HF Qwen2Attention uses F.scaled_dot_product_attention
#    with an explicit attention mask and may dispatch to flash/efficient kernels
#    depending on transformers version and installed backends.  Our inline attention
#    uses the same F.scaled_dot_product_attention call but with is_causal=True and
#    no external mask, matching what our AD custom op produces.  This makes the
#    reference deterministic and independent of transformers installation details.
#
# 2. The RMSNorm and MLP blocks are functionally identical to the HF versions, but
#    keeping all reference implementations inline avoids importing from transformers
#    private module paths (transformers.models.qwen2.modeling_qwen2) which could
#    change across versions.
# ---------------------------------------------------------------------------


class _HFQwen2RMSNorm(nn.Module):
    """Reference RMSNorm from HF Qwen2."""

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


class _HFQwen2RotaryEmbedding(nn.Module):
    """Reference RoPE from HF Qwen2."""

    def __init__(self, dim, max_position_embeddings=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)
        return cos, sin


class _HFQwen2MLP(nn.Module):
    """Reference MLP from HF Qwen2."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _hf_repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class _HFQwen2Attention(nn.Module):
    """Reference Attention from HF Qwen2 (eager mode)."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings):
        bsz, q_len, _ = hidden_states.size()
        q = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        q, k = _hf_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        k = _hf_repeat_kv(k, self.num_kv_groups)
        v = _hf_repeat_kv(v, self.num_kv_groups)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scaling)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class _HFQwen2DecoderLayer(nn.Module):
    """Reference decoder layer from HF Qwen2."""

    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.self_attn = _HFQwen2Attention(config, layer_idx=layer_idx)
        self.mlp = _HFQwen2MLP(config)
        self.input_layernorm = _HFQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _HFQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class _HFQwen2ForCausalLM(nn.Module):
    """Reference full model from HF Qwen2 (for equivalence tests)."""

    def __init__(self, config):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_HFQwen2DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = _HFQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rotary_emb = _HFQwen2RotaryEmbedding(
            head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
        )

    def forward(self, input_ids, position_ids):
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        return logits


def _convert_hf_to_custom_state_dict(hf_state_dict):
    """Convert HF reference state dict keys to match our custom model hierarchy.

    HF reference: embed_tokens.weight, layers.0.*, norm.weight, lm_head.weight
    Custom model: language_model.model.embed_tokens.weight, language_model.model.layers.0.*, ...
    """
    converted = {}
    for key, value in hf_state_dict.items():
        if key.startswith("lm_head."):
            converted[f"language_model.{key}"] = value
        else:
            converted[f"language_model.model.{key}"] = value
    return converted


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm produces numerically equivalent output to HF Qwen2 RMSNorm."""
    device = "cpu"
    config = _create_small_llm_config()

    hf_norm = (
        _HFQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        .to(device=device, dtype=dtype)
        .eval()
    )
    custom_norm = SkyworkR1V2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
        device=device, dtype=dtype
    )
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_norm(x), hf_norm(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF Qwen2 MLP."""
    device = "cpu"
    config = _create_small_llm_config()

    hf_mlp = _HFQwen2MLP(config).to(device=device, dtype=dtype).eval()
    custom_mlp = SkyworkR1V2MLP(config).to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF Qwen2 Attention."""
    device = "cpu"
    config = _create_small_llm_config()

    hf_attn = _HFQwen2Attention(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_attn = SkyworkR1V2Attention(config, layer_idx=0).to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF reference position embeddings
    head_dim = config.hidden_size // config.num_attention_heads
    hf_rotary = _HFQwen2RotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    ).to(device=device, dtype=dtype)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings: rotary returns full tables; attention slices by position_ids.
    custom_rotary = SkyworkR1V2RotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    ).to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)  # full [max_seq_len, head_dim] tables

    hf_out = hf_attn(x, (hf_cos, hf_sin))
    custom_out = custom_attn(x, (custom_cos, custom_sin), position_ids)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_skywork_r1v2_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF Qwen2."""
    device = "cpu"
    config = _create_small_llm_config()

    hf_layer = _HFQwen2DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype).eval()
    custom_layer = SkyworkR1V2DecoderLayer(config, layer_idx=0).to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    head_dim = config.hidden_size // config.num_attention_heads
    hf_rotary = _HFQwen2RotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    ).to(device=device, dtype=dtype)
    hf_pos_emb = hf_rotary(x, position_ids)

    custom_rotary = SkyworkR1V2RotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    ).to(device=device, dtype=dtype)
    custom_pos_emb = custom_rotary(x, position_ids)  # full [max_seq_len, head_dim] tables

    hf_out = hf_layer(x, hf_pos_emb)
    custom_out = custom_layer(x, custom_pos_emb, position_ids)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu"])
@torch.no_grad()
def test_skywork_r1v2_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF Qwen2."""
    config = _create_small_llm_config()
    chat_config = _create_small_chat_config()

    hf_model = _HFQwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()

    custom_model = SkyworkR1V2ForCausalLM(chat_config).to(device=device, dtype=dtype)
    custom_model.load_state_dict(
        _convert_hf_to_custom_state_dict(hf_model.state_dict()), strict=False
    )
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_logits = hf_model(input_ids, position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_skywork_r1v2_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    chat_config = _create_small_chat_config()

    model = SkyworkR1V2ForCausalLM(chat_config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, 1000, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    dynamic_shapes = (
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
        {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, 1000)
    assert_rmse_close(
        logits.float(), eager_out.logits.float(), rmse_ratio_tol=0.05, msg="Export vs eager: "
    )

    # Test dynamic shapes with different input
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, 1000, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)
    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    assert out_gm2["logits"].shape == (B2, S2, 1000)
    assert_rmse_close(
        out_gm2["logits"].float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_skywork_r1v2_config_parsing():
    """Test that SkyworkChatConfig correctly parses nested llm_config."""
    config = _create_small_chat_config()
    assert config.model_type == "skywork_chat"
    assert isinstance(config.llm_config, Qwen2Config)
    assert config.llm_config.hidden_size == 64
    assert config.llm_config.num_attention_heads == 4
    assert config.llm_config.num_key_value_heads == 2


def test_skywork_r1v2_gqa_structure():
    """Test that attention uses GQA with bias on QKV."""
    chat_config = _create_small_chat_config()
    model = SkyworkR1V2ForCausalLM(chat_config)

    attn = model.language_model.model.layers[0].self_attn
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 2
    assert attn.q_proj.bias is not None
    assert attn.k_proj.bias is not None
    assert attn.v_proj.bias is not None
    assert attn.o_proj.bias is None


def test_skywork_r1v2_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    chat_config = _create_small_chat_config()
    model = SkyworkR1V2ForCausalLM(chat_config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.bias",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.bias",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.bias",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )

    # Ensure no unexpected keys outside language_model.* (e.g. no vision_model.*, mlp1.*)
    for key in state_dict:
        assert key.startswith("language_model."), f"Unexpected key outside language_model: '{key}'"
