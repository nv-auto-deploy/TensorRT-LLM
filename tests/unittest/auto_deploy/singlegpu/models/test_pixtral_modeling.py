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

"""Tests for Pixtral-12B text-only wrapper (PixtralForCausalLM).

This module tests the text-only wrapper that loads a Mistral text backbone from
a Llava-architecture checkpoint (mistral-community/pixtral-12b). The wrapper:
  * Registers for LlavaConfig
  * Wraps standard MistralForCausalLM under self.language_model
  * Discards vision_tower and multi_modal_projector weights
  * Delegates all forward logic to MistralForCausalLM

Since no custom AD canonical ops are used (the wrapper delegates to standard HF
MistralForCausalLM, which AD transforms handle natively), hierarchical block/layer
tests are not applicable — the wrapper has no custom attention, MLP, or norm blocks.

Test levels:
  1. Structural tests (checkpoint layout, registration, delegation)
  2. Full-model equivalence (wrapper vs independent MistralForCausalLM)
  3. Export test (torch_export_to_gm with dynamic shapes)
"""

import pytest
import torch
from torch.export import Dim
from transformers.models.llava.configuration_llava import LlavaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM as HFMistralForCausalLM
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_pixtral import PixtralForCausalLM
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
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        rope_theta=1000000000.0,
        tie_word_embeddings=False,
    )


def _create_small_llava_config() -> LlavaConfig:
    """Create a small LlavaConfig for testing (mirrors pixtral-12b structure)."""
    text_config = _create_small_text_config()
    vision_config = PixtralVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=16,
        image_size=64,
    )
    return LlavaConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
    )


# =========================================================================
# Structural tests
# =========================================================================


def test_pixtral_state_dict_matches_checkpoint_layout():
    """Test that state_dict keys use the language_model.* prefix matching the checkpoint."""
    config = _create_small_llava_config()
    model = PixtralForCausalLM(config)
    state_dict = model.state_dict()

    for key in state_dict.keys():
        assert key.startswith("language_model."), (
            f"Key '{key}' does not start with 'language_model.'"
        )

    expected_keys = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.model.layers.0.input_layernorm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
    ]
    for key in expected_keys:
        assert key in state_dict, f"Missing expected key '{key}'"


def test_pixtral_no_vision_or_projector_keys():
    """Test that the model has NO vision_tower or multi_modal_projector keys."""
    config = _create_small_llava_config()
    model = PixtralForCausalLM(config)

    for key in model.state_dict().keys():
        assert "vision_tower" not in key, f"Unexpected vision key: {key}"
        assert "multi_modal_projector" not in key, f"Unexpected projector key: {key}"


def test_pixtral_ignores_unexpected_checkpoint_keys():
    """Test that vision/projector keys from a full checkpoint are silently ignored."""
    config = _create_small_llava_config()
    model = PixtralForCausalLM(config)

    fake_state_dict = dict(model.state_dict())
    fake_state_dict["vision_tower.patch_conv.weight"] = torch.randn(32, 3, 16, 16)
    fake_state_dict["multi_modal_projector.linear_1.weight"] = torch.randn(64, 32)

    missing, unexpected = model.load_state_dict(fake_state_dict, strict=False)
    assert any("vision_tower" in k for k in unexpected)
    assert any("multi_modal_projector" in k for k in unexpected)
    assert len(missing) == 0


def test_pixtral_registered_for_llava_config():
    """Test that LlavaConfig maps to PixtralForCausalLM in the custom model registry."""
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    cls = AutoModelForCausalLMFactory._custom_model_mapping.get("LlavaConfig")
    assert cls is PixtralForCausalLM


def test_pixtral_position_ids_required():
    """Test that forward raises when position_ids is None."""
    config = _create_small_llava_config()
    model = PixtralForCausalLM(config)
    model.eval()

    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 4))
    with pytest.raises(AssertionError, match="position_ids must be provided"):
        model(input_ids=input_ids, position_ids=None)


def test_pixtral_embedding_delegation():
    """Test that get/set input/output embeddings delegate to inner model."""
    config = _create_small_llava_config()
    model = PixtralForCausalLM(config)

    assert model.get_input_embeddings() is model.language_model.get_input_embeddings()
    assert model.get_output_embeddings() is model.language_model.lm_head


# =========================================================================
# Full-model equivalence test
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_pixtral_logits_match_standalone_mistral(B, S, dtype, device):
    """Test that PixtralForCausalLM produces identical logits to a standalone MistralForCausalLM.

    Instantiates both models independently, copies weights from the wrapper's inner
    language_model into a standalone MistralForCausalLM, and compares logits.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    text_config = _create_small_text_config()
    text_config._attn_implementation = "eager"
    llava_config = _create_small_llava_config()
    llava_config.text_config._attn_implementation = "eager"

    # Create wrapper
    wrapper = PixtralForCausalLM(llava_config)
    wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    # Create independent standalone MistralForCausalLM and load same weights
    standalone = HFMistralForCausalLM(text_config)
    standalone.to(device=device, dtype=dtype)
    standalone.load_state_dict(wrapper.language_model.state_dict())
    standalone.eval()

    input_ids = torch.randint(0, text_config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    wrapper_out = wrapper(input_ids=input_ids, position_ids=position_ids)
    standalone_out = standalone(input_ids=input_ids, position_ids=position_ids, use_cache=False)

    torch.testing.assert_close(wrapper_out.logits, standalone_out.logits, rtol=1e-5, atol=1e-5)


# =========================================================================
# Export test
# =========================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.no_grad()
def test_pixtral_can_be_exported(device):
    """Test that the wrapper can be exported with torch_export_to_gm.

    Exports the full PixtralForCausalLM (which delegates to MistralForCausalLM)
    and verifies end-to-end logits match the eager forward pass.

    Verifies:
    1. The model exports successfully
    2. The exported graph module produces equivalent logits
    3. Dynamic shapes work with a second input shape
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dtype = torch.bfloat16
    text_config = _create_small_text_config()
    llava_config = _create_small_llava_config()

    wrapper = PixtralForCausalLM(llava_config)
    wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, text_config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    eager_out = wrapper(input_ids=input_ids, position_ids=position_ids)

    batch_dim = Dim.DYNAMIC
    seq_dim = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim, 1: seq_dim},
    )

    gm = torch_export_to_gm(
        wrapper,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm, "Output should contain 'logits'"
    logits = out_gm["logits"]
    assert logits.shape == (B, S, text_config.vocab_size)
    torch.testing.assert_close(logits.float(), eager_out.logits.float(), rtol=1e-3, atol=1e-3)

    # Test dynamic shapes with different input
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, text_config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    eager_out2 = wrapper(input_ids=input_ids2, position_ids=position_ids2)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, text_config.vocab_size)
    torch.testing.assert_close(logits2.float(), eager_out2.logits.float(), rtol=1e-3, atol=1e-3)
