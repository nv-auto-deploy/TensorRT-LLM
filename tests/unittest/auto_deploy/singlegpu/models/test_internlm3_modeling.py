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

"""Tests for InternLM3 custom model implementation.

This module tests the custom InternLM3 model implementation which uses
auto_deploy custom ops (torch_attention, torch_rope_with_explicit_cos_sin)
for export compatibility. InternLM3 is a Llama-style model with GQA,
SiLU MLP, RMSNorm, and dynamic NTK-aware RoPE.

HF reference classes are defined inline (copied strictly from the InternLM3
HF source) because InternLM3Config/modeling is not part of standard transformers.
"""

import pytest
import torch
import torch.nn.functional as F
from _model_test_utils import assert_rmse_close
from torch import nn
from torch.export import Dim
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_internlm3 import (
    InternLM3Attention,
    InternLM3Config,
    InternLM3DecoderLayer,
    InternLM3ForCausalLM,
    InternLM3MLP,
    InternLM3RMSNorm,
    InternLM3RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> InternLM3Config:
    """Create a small InternLM3 config for testing."""
    return InternLM3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        qkv_bias=False,
        bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


# =========================================================================
# HF Reference implementations (copied strictly from HF InternLM3 source)
# These are used as ground truth for equivalence tests.
# =========================================================================


class _RefInternLM3RMSNorm(nn.Module):
    """Reference InternLM3 RMSNorm from HF source."""

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


class _RefInternLM3RotaryEmbedding(nn.Module):
    """Reference InternLM3 RotaryEmbedding from HF source."""

    def __init__(self, config):
        super().__init__()
        if config.rope_scaling is not None:
            rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )
        else:
            rope_type = "default"
        self.rope_type = rope_type
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device=None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _ref_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_ref_rotate_half(q) * sin)
    k_embed = (k * cos) + (_ref_rotate_half(k) * sin)
    return q_embed, k_embed


def _ref_repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class _RefInternLM3MLP(nn.Module):
    """Reference InternLM3 MLP from HF source."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class _RefInternLM3Attention(nn.Module):
    """Reference InternLM3 eager attention from HF source."""

    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _ref_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = _ref_repeat_kv(key_states, self.num_key_value_groups)
        value_states = _ref_repeat_kv(value_states, self.num_key_value_groups)

        is_causal = attention_mask is None
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None if is_causal else attention_mask,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), None


class _RefInternLM3DecoderLayer(nn.Module):
    """Reference InternLM3 decoder layer from HF source."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = _RefInternLM3Attention(config, layer_idx=layer_idx)
        self.mlp = _RefInternLM3MLP(config)
        self.input_layernorm = _RefInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RefInternLM3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,)


class _RefInternLM3ForCausalLM(PreTrainedModel):
    """Reference InternLM3ForCausalLM (eager attention, no KV cache)."""

    config_class = InternLM3Config

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [
                _RefInternLM3DecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RefInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = _RefInternLM3RotaryEmbedding(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(self, input_ids, position_ids=None):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            outputs = layer(hidden_states, position_embeddings=position_embeddings)
            hidden_states = outputs[0]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        from dataclasses import dataclass

        @dataclass
        class _Out:
            logits: torch.Tensor

        return _Out(logits=logits)


def _build_ref_state_dict_for_custom(ref_model: _RefInternLM3ForCausalLM) -> dict:
    """Map reference model state_dict keys to custom model keys.

    The reference model has a flat layout (embed_tokens, layers, norm, lm_head)
    while the custom model wraps these under 'model.' prefix.
    """
    ref_sd = ref_model.state_dict()
    custom_sd = {}
    for k, v in ref_sd.items():
        if k in ("lm_head.weight",):
            custom_sd[k] = v
        else:
            custom_sd[f"model.{k}"] = v
    return custom_sd


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_rmsnorm_equivalence(B, S, dtype):
    """Test RMSNorm produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    ref_norm = _RefInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    ref_norm.to(device=device, dtype=dtype).eval()

    custom_norm = InternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    custom_norm.to(device=device, dtype=dtype)
    custom_norm.load_state_dict(ref_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_norm(x), ref_norm(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    ref_mlp = _RefInternLM3MLP(config)
    ref_mlp.to(device=device, dtype=dtype).eval()

    custom_mlp = InternLM3MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(ref_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), ref_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    ref_attn = _RefInternLM3Attention(config, layer_idx=0)
    ref_attn.to(device=device, dtype=dtype).eval()

    custom_attn = InternLM3Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(ref_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    ref_rotary = _RefInternLM3RotaryEmbedding(config)
    ref_rotary.to(device=device, dtype=dtype)
    ref_cos, ref_sin = ref_rotary(x, position_ids)

    custom_rotary = InternLM3RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    # custom_rotary returns full table [max_pos, head_dim]; attention slices by position_ids
    custom_full_cos, custom_full_sin = custom_rotary(x)

    ref_out, _ = ref_attn(
        hidden_states=x, position_embeddings=(ref_cos, ref_sin), attention_mask=None
    )
    custom_out = custom_attn(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_full_cos, custom_full_sin),
    )

    assert_rmse_close(custom_out, ref_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    ref_layer = _RefInternLM3DecoderLayer(config, layer_idx=0)
    ref_layer.to(device=device, dtype=dtype).eval()

    custom_layer = InternLM3DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(ref_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    ref_rotary = _RefInternLM3RotaryEmbedding(config)
    ref_rotary.to(device=device, dtype=dtype)
    ref_cos, ref_sin = ref_rotary(x, position_ids)

    custom_rotary = InternLM3RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    # custom_rotary returns full table [max_pos, head_dim]; decoder layer passes position_ids down
    custom_full_cos, custom_full_sin = custom_rotary(x)

    ref_out = ref_layer(hidden_states=x, position_embeddings=(ref_cos, ref_sin))
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]

    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_full_cos, custom_full_sin),
    )

    assert_rmse_close(custom_out, ref_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_internlm3_full_model_equivalence(B, S, dtype, device):
    """Test full model produces numerically equivalent output to HF reference."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_config()

    ref_model = _RefInternLM3ForCausalLM(config)
    ref_model.to(device=device, dtype=dtype).eval()

    custom_model = InternLM3ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(_build_ref_state_dict_for_custom(ref_model))
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    ref_out = ref_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        ref_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_internlm3_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = InternLM3ForCausalLM(config)
    model.to(device=device, dtype=dtype).eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
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
    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected shape {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    assert_rmse_close(
        logits.float(),
        eager_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager: ",
    )

    # Test dynamic shapes with different input shape
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size), (
        f"Dynamic shape test failed: expected {(B2, S2, config.vocab_size)}, got {logits2.shape}"
    )
    assert_rmse_close(
        logits2.float(),
        eager_out2.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_internlm3_config_registration():
    """Test that InternLM3Config is properly configured."""
    config = _create_small_config()
    assert config.model_type == "internlm3"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")
    assert hasattr(config, "head_dim")
    assert hasattr(config, "qkv_bias")
    assert hasattr(config, "bias")


def test_internlm3_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_config()
    model = InternLM3ForCausalLM(config)
    attn = model.model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"


def test_internlm3_state_dict_keys():
    """Test that state_dict keys match expected checkpoint format."""
    config = _create_small_config()
    model = InternLM3ForCausalLM(config)
    state_dict = model.state_dict()

    expected_keys = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    for key in expected_keys:
        assert key in state_dict, (
            f"Expected key '{key}' not in state_dict. First 10 keys: {list(state_dict.keys())[:10]}"
        )
