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

"""Tests for OLMo-2 custom model implementation.

Hierarchical test levels:
1. Block equivalence — MLP, Attention individually
2. Layer equivalence — Full decoder layer (post-norm)
3. Full model equivalence — End-to-end logits comparison
4. Export test — torch_export_to_gm with dynamic shapes
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_olmo2 import (
    Olmo2Attention,
    Olmo2DecoderLayer,
    Olmo2ForCausalLM,
    Olmo2MLP,
    Olmo2RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Helpers
# =============================================================================


def _create_small_config():
    """Create a small OLMo-2 config for testing."""
    from transformers.models.olmo2.configuration_olmo2 import Olmo2Config

    return Olmo2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
    )


# =============================================================================
# HF Reference Import Helpers
# =============================================================================


def _get_hf_model_class():
    try:
        from transformers.models.olmo2.modeling_olmo2 import Olmo2ForCausalLM

        return Olmo2ForCausalLM
    except ImportError:
        return None


def _get_hf_mlp_class():
    try:
        from transformers.models.olmo2.modeling_olmo2 import Olmo2MLP

        return Olmo2MLP
    except ImportError:
        return None


def _get_hf_attention_class():
    try:
        from transformers.models.olmo2.modeling_olmo2 import Olmo2Attention

        return Olmo2Attention
    except ImportError:
        return None


def _get_hf_decoder_layer_class():
    try:
        from transformers.models.olmo2.modeling_olmo2 import Olmo2DecoderLayer

        return Olmo2DecoderLayer
    except ImportError:
        return None


def _create_hf_config():
    """Create HF Olmo2Config matching our small test config."""
    from transformers.models.olmo2.configuration_olmo2 import Olmo2Config

    config = Olmo2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=500000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config


def _create_causal_mask(B, S, device, dtype):
    """Create a 4D causal attention mask for HF eager attention."""
    mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)


# =============================================================================
# Structural Tests
# =============================================================================


def test_olmo2_post_norm_structure():
    """Verify decoder layer has post-norm (not pre-norm) structure."""
    config = _create_small_config()
    layer = Olmo2DecoderLayer(config, layer_idx=0)

    assert hasattr(layer, "post_attention_layernorm"), "Missing post_attention_layernorm"
    assert hasattr(layer, "post_feedforward_layernorm"), "Missing post_feedforward_layernorm"
    assert not hasattr(layer, "input_layernorm"), "Should NOT have input_layernorm (post-norm arch)"


def test_olmo2_qk_norm_structure():
    """Verify attention has Q/K normalization with correct sizes."""
    config = _create_small_config()
    attn = Olmo2Attention(config, layer_idx=0)

    head_dim = config.hidden_size // config.num_attention_heads
    assert hasattr(attn, "q_norm"), "Missing q_norm"
    assert hasattr(attn, "k_norm"), "Missing k_norm"
    assert attn.q_norm.weight.shape[0] == config.num_attention_heads * head_dim
    assert attn.k_norm.weight.shape[0] == config.num_key_value_heads * head_dim


def test_olmo2_no_tied_embeddings():
    """Verify lm_head.weight is separate from embed_tokens.weight."""
    config = _create_small_config()
    model = Olmo2ForCausalLM(config)

    # With tie_word_embeddings=False, these should be different tensors
    assert not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight), (
        "Embeddings should not be tied"
    )


# =============================================================================
# Numerical Equivalence Tests
# =============================================================================

# --- Level 1: Block Equivalence ---


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_olmo2_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF Olmo2MLP."""
    HFMLP = _get_hf_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have Olmo2MLP")

    device = "cpu"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_mlp = HFMLP(hf_config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Olmo2MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_olmo2_attention_numerical_equivalence(B, S, dtype):
    """Test attention produces numerically equivalent output to HF Olmo2Attention."""
    HFAttn = _get_hf_attention_class()
    if HFAttn is None:
        pytest.skip("transformers doesn't have Olmo2Attention")

    device = "cpu"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_attn = HFAttn(hf_config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = Olmo2Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Custom model: full RoPE table + position_ids
    head_dim = config.hidden_size // config.num_attention_heads
    rotary_emb = Olmo2RotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    full_cos, full_sin = rotary_emb(x)

    # HF expects position-indexed cos/sin: [B, S, head_dim]
    hf_cos = full_cos[position_ids]
    hf_sin = full_sin[position_ids]

    causal_mask = _create_causal_mask(B, S, device, dtype)

    # Run HF attention
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
        position_ids=position_ids,
    )

    # Run custom attention
    custom_out = custom_attn(x, position_ids, (full_cos, full_sin))

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# --- Level 2: Layer Equivalence ---


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_olmo2_decoder_layer_numerical_equivalence(B, S, dtype):
    """Test decoder layer matches HF Olmo2DecoderLayer."""
    HFLayer = _get_hf_decoder_layer_class()
    if HFLayer is None:
        pytest.skip("transformers doesn't have Olmo2DecoderLayer")

    device = "cpu"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_layer = HFLayer(hf_config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = Olmo2DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    head_dim = config.hidden_size // config.num_attention_heads
    rotary_emb = Olmo2RotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    )
    full_cos, full_sin = rotary_emb(x)
    hf_cos = full_cos[position_ids]
    hf_sin = full_sin[position_ids]

    causal_mask = _create_causal_mask(B, S, device, dtype)

    # HF layer
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_embeddings=(hf_cos, hf_sin),
    )

    # Custom layer
    custom_out = custom_layer(x, position_ids, (full_cos, full_sin))

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# --- Level 3: Full Model Equivalence ---


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_olmo2_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent logits to HF Olmo2ForCausalLM."""
    HFModel = _get_hf_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have Olmo2ForCausalLM")

    device = "cpu"
    config = _create_small_config()
    hf_config = _create_hf_config()

    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = Olmo2ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(custom_out.logits, hf_out.logits, rmse_ratio_tol=0.05, msg="Full model: ")


# =============================================================================
# Export Test
# =============================================================================


@torch.no_grad()
def test_olmo2_model_can_be_exported():
    """Test that Olmo2ForCausalLM can be exported with torch_export_to_gm."""
    device = "cpu"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = Olmo2ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    # Eager reference output
    eager_out = model(input_ids=input_ids, position_ids=position_ids)

    # Exported graph output
    out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits' key"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected shape {(B, S, config.vocab_size)}, got {logits.shape}"
    )
    torch.testing.assert_close(logits.float(), eager_out.logits.float(), rtol=1e-3, atol=1e-3)

    # Test with different shape to verify dynamic shapes
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)
    out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    expected_shape = (B2, S2, config.vocab_size)
    assert logits2.shape == expected_shape, (
        f"Dynamic shape test failed: expected {expected_shape}, got {logits2.shape}"
    )
    torch.testing.assert_close(logits2.float(), eager_out2.logits.float(), rtol=1e-3, atol=1e-3)
