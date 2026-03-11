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

"""Tests for Mistral3 custom model implementation.

This module tests the custom Mistral3 model implementation which uses
auto_deploy canonical ops (torch_attention, torch_rope_with_explicit_cos_sin,
torch_rmsnorm) for the text backbone. The text model is a standard Mistral
architecture with GQA (no per-head Q/K norms, unlike Qwen3).

Hierarchical test levels:
  1. Block equivalence (MLP, Attention)
  2. Layer equivalence (DecoderLayer)
  3. Full text model equivalence (Mistral3TextModel vs HF MistralModel)
  4. Export test (torch_export_to_gm with dynamic shapes)
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim
from transformers.models.mistral.configuration_mistral import MistralConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
    Mistral3TextAttention,
    Mistral3TextDecoderLayer,
    Mistral3TextMLP,
    Mistral3TextModel,
    Mistral3TextRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_text_config() -> MistralConfig:
    """Create a small Mistral text config for testing."""
    return MistralConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA: 4 Q heads, 2 KV heads
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )


# =========================================================================
# HF reference class helpers
# =========================================================================


def _get_hf_mistral_model_class():
    """Get the HF MistralForCausalLM class."""
    try:
        from transformers.models.mistral.modeling_mistral import (
            MistralForCausalLM as HFMistralForCausalLM,
        )

        return HFMistralForCausalLM
    except ImportError:
        return None


def _get_hf_mistral_attention_class():
    """Get the HF MistralAttention class."""
    try:
        from transformers.models.mistral.modeling_mistral import (
            MistralAttention as HFMistralAttention,
        )

        return HFMistralAttention
    except ImportError:
        return None


def _get_hf_mistral_mlp_class():
    """Get the HF MistralMLP class."""
    try:
        from transformers.models.mistral.modeling_mistral import MistralMLP as HFMistralMLP

        return HFMistralMLP
    except ImportError:
        return None


def _get_hf_mistral_decoder_layer_class():
    """Get the HF MistralDecoderLayer class."""
    try:
        from transformers.models.mistral.modeling_mistral import (
            MistralDecoderLayer as HFMistralDecoderLayer,
        )

        return HFMistralDecoderLayer
    except ImportError:
        return None


def _get_hf_mistral_rotary_class():
    """Get the HF MistralRotaryEmbedding class."""
    try:
        from transformers.models.mistral.modeling_mistral import (
            MistralRotaryEmbedding as HFMistralRotaryEmbedding,
        )

        return HFMistralRotaryEmbedding
    except ImportError:
        return None


# =========================================================================
# Block equivalence tests (Level 1)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mistral3_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF MistralMLP."""
    HFMLP = _get_hf_mistral_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have MistralMLP")

    device = "cuda"
    config = _create_small_text_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = Mistral3TextMLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    # MLP uses identical math — tight tolerance
    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


def _make_causal_mask(S: int, device, dtype) -> torch.Tensor:
    """Create a causal attention mask compatible with HF's eager attention.

    Returns shape [1, 1, S, S] with 0 for allowed positions and -inf for masked.
    """
    mask = torch.full((S, S), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mistral3_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF MistralAttention.

    Note: Our custom attention uses is_causal=True inside torch_attention, so we must
    pass a matching causal mask to HF's eager attention for a fair comparison. Without
    the mask, HF's eager attention does bidirectional attention.
    """
    HFAttention = _get_hf_mistral_attention_class()
    HFRotary = _get_hf_mistral_rotary_class()
    if HFAttention is None or HFRotary is None:
        pytest.skip("transformers doesn't have MistralAttention or MistralRotaryEmbedding")

    device = "cuda"
    config = _create_small_text_config()
    config._attn_implementation = "eager"

    hf_attn = HFAttention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    custom_attn = Mistral3TextAttention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = Mistral3TextRotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Create causal mask for HF attention (custom uses is_causal=True internally)
    causal_mask = _make_causal_mask(S, device, dtype)

    # HF attention with causal mask
    hf_out, _ = hf_attn(
        hidden_states=x,
        position_embeddings=(hf_cos, hf_sin),
        attention_mask=causal_mask,
    )

    # Custom attention (is_causal=True applied inside torch_attention)
    custom_out = custom_attn(
        hidden_states=x,
        position_embeddings=(custom_cos, custom_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


# =========================================================================
# Layer equivalence tests (Level 2)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_mistral3_decoder_layer_equivalence(B, S, dtype):
    """Test decoder layer produces numerically equivalent output to HF MistralDecoderLayer."""
    HFDecoderLayer = _get_hf_mistral_decoder_layer_class()
    HFRotary = _get_hf_mistral_rotary_class()
    if HFDecoderLayer is None or HFRotary is None:
        pytest.skip("transformers doesn't have MistralDecoderLayer or MistralRotaryEmbedding")

    device = "cuda"
    config = _create_small_text_config()
    config._attn_implementation = "eager"

    hf_layer = HFDecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = Mistral3TextDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF position embeddings
    hf_rotary = HFRotary(config=config, device=device)
    hf_cos, hf_sin = hf_rotary(x, position_ids)

    # Custom position embeddings
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    custom_rotary = Mistral3TextRotaryEmbedding(
        head_dim, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta
    )
    custom_rotary.to(device=device, dtype=dtype)
    custom_cos, custom_sin = custom_rotary(x, position_ids)

    # Create causal mask for HF layer (custom uses is_causal=True inside torch_attention)
    causal_mask = _make_causal_mask(S, device, dtype)

    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(hf_cos, hf_sin),
    )
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]

    custom_out = custom_layer(
        hidden_states=x,
        position_embeddings=(custom_cos, custom_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


# =========================================================================
# Full text model equivalence tests (Level 3)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_mistral3_text_model_equivalence(B, S, dtype, device):
    """Test full text model produces numerically equivalent logits to HF MistralModel.

    We compare Mistral3TextModel (our custom) against MistralModel (HF), both using
    the same weights and inputs. The lm_head is tested as part of this comparison
    by wrapping Mistral3TextModel with a linear head manually.
    """
    HFModel = _get_hf_mistral_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have MistralForCausalLM")

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = _create_small_text_config()
    config._attn_implementation = "eager"

    # Create HF model (MistralForCausalLM = MistralModel + lm_head)
    hf_model = HFModel(config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Create our custom text model + lm_head
    custom_text_model = Mistral3TextModel(config)
    custom_text_model.to(device=device, dtype=dtype)
    # Load only the text model weights (HF stores them under "model.")
    custom_text_model.load_state_dict(hf_model.model.state_dict())
    custom_text_model.eval()

    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    lm_head.to(device=device, dtype=dtype)
    lm_head.load_state_dict({"weight": hf_model.lm_head.weight.data})
    lm_head.eval()

    # Create inputs
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # HF forward
    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)

    # Custom forward
    custom_out = custom_text_model(input_ids=input_ids, position_ids=position_ids)
    custom_logits = lm_head(custom_out.last_hidden_state).float()

    assert_rmse_close(
        custom_logits,
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full text model logits: ",
    )


# =========================================================================
# Export test (Level 4)
# =========================================================================


@torch.no_grad()
def test_mistral3_text_model_can_be_exported():
    """Test that the custom text model can be exported with torch_export_to_gm.

    Verifies:
    1. The model exports successfully without graph breaks
    2. The exported graph module produces numerically equivalent output
    3. Dynamic shapes work correctly with a second input shape
    """
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_text_config()

    model = Mistral3TextModel(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = model(input_ids=input_ids, position_ids=position_ids)

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

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "last_hidden_state" in out_gm, "Output should contain 'last_hidden_state' key"
    hidden = out_gm["last_hidden_state"]
    assert hidden.shape == (B, S, config.hidden_size), (
        f"Expected shape {(B, S, config.hidden_size)}, got {hidden.shape}"
    )
    assert_rmse_close(
        hidden.float(),
        eager_out.last_hidden_state.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager: ",
    )

    # Test dynamic shapes with a different input
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = model(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    hidden2 = out_gm2["last_hidden_state"]
    assert hidden2.shape == (B2, S2, config.hidden_size), (
        f"Dynamic shape test failed: expected {(B2, S2, config.hidden_size)}, got {hidden2.shape}"
    )
    assert_rmse_close(
        hidden2.float(),
        eager_out2.last_hidden_state.float(),
        rmse_ratio_tol=0.05,
        msg="Export vs eager (dynamic shape): ",
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_mistral3_text_config_used():
    """Test that the text model uses MistralConfig correctly."""
    config = _create_small_text_config()
    assert config.model_type == "mistral"
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_attention_heads")
    assert hasattr(config, "num_key_value_heads")


def test_mistral3_gqa_structure():
    """Test that attention uses GQA (fewer KV heads than Q heads)."""
    config = _create_small_text_config()
    model = Mistral3TextModel(config)

    attn = model.layers[0].self_attn
    assert attn.num_heads == 4, f"Expected 4 Q heads, got {attn.num_heads}"
    assert attn.num_kv_heads == 2, f"Expected 2 KV heads, got {attn.num_kv_heads}"

    # Mistral does NOT have per-head Q/K norms (unlike Qwen3)
    assert not hasattr(attn, "q_norm"), "Mistral attention should NOT have q_norm"
    assert not hasattr(attn, "k_norm"), "Mistral attention should NOT have k_norm"


def test_mistral3_text_state_dict_keys():
    """Test that state_dict keys match expected HF MistralModel checkpoint format."""
    config = _create_small_text_config()
    model = Mistral3TextModel(config)
    state_dict = model.state_dict()

    expected_key_patterns = [
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "norm.weight",
    ]

    for key in expected_key_patterns:
        assert key in state_dict, (
            f"Expected key '{key}' in state_dict, got keys: {list(state_dict.keys())[:10]}..."
        )


def test_mistral3_multimodal_structure():
    """Test that the full Mistral3ForConditionalGeneration has correct module hierarchy."""
    from transformers.models.mistral3.configuration_mistral3 import (
        Mistral3Config as HFMistral3Config,
    )

    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
        Mistral3ForConditionalGeneration,
    )

    # Create a small multimodal config
    text_config = _create_small_text_config()
    config = HFMistral3Config(text_config=text_config.to_dict())

    model = Mistral3ForConditionalGeneration(config)

    # Check module hierarchy matches expected checkpoint layout
    assert hasattr(model, "model"), "Should have 'model' attribute"
    assert hasattr(model.model, "vision_tower"), "Should have 'model.vision_tower'"
    assert hasattr(model.model, "multi_modal_projector"), (
        "Should have 'model.multi_modal_projector'"
    )
    assert hasattr(model.model, "language_model"), "Should have 'model.language_model'"
    assert hasattr(model, "lm_head"), "Should have 'lm_head'"

    # The language_model should have MistralConfig (needed for TextModelExportInfo)
    assert isinstance(model.model.language_model.config, MistralConfig), (
        "language_model.config should be MistralConfig for TextModelExportInfo auto-discovery"
    )
