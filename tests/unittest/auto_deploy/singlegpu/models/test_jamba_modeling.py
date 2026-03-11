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

"""Tests for Jamba custom model implementation (hybrid Mamba v1 + Attention).

This module tests the custom Jamba model implementation which uses
auto_deploy custom ops (torch_mamba_v1_selective_scan, torch_attention,
torch_causal_conv1d) for export compatibility.

Hierarchical test levels:
1. Block equivalence — MLP, Attention, Mamba mixer individually
2. Layer equivalence — Full decoder layers (attention and mamba types)
3. Full model equivalence — End-to-end logits comparison
4. Export test — torch.export with dynamic shapes
"""

import pytest
import torch
from _model_test_utils import assert_rmse_close
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_jamba import (
    JambaAttention,
    JambaAttentionDecoderLayer,
    JambaForCausalLM,
    JambaMambaDecoderLayer,
    JambaMambaMixer,
    JambaMLP,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# HF reference helpers
# ---------------------------------------------------------------------------


def _get_jamba_config():
    """Get JambaConfig from transformers."""
    try:
        from transformers.models.jamba.configuration_jamba import JambaConfig

        return JambaConfig
    except ImportError:
        return None


def _get_hf_jamba_model():
    """Get HF JambaForCausalLM class."""
    try:
        from transformers.models.jamba.modeling_jamba import JambaForCausalLM as HFJambaForCausalLM

        return HFJambaForCausalLM
    except ImportError:
        return None


def _get_hf_jamba_attention():
    """Get HF JambaAttention class."""
    try:
        from transformers.models.jamba.modeling_jamba import JambaAttention as HFJambaAttention

        return HFJambaAttention
    except ImportError:
        return None


def _get_hf_jamba_mamba_mixer():
    """Get HF JambaMambaMixer class."""
    try:
        from transformers.models.jamba.modeling_jamba import JambaMambaMixer as HFJambaMambaMixer

        return HFJambaMambaMixer
    except ImportError:
        return None


def _get_hf_jamba_mlp():
    """Get HF JambaMLP class."""
    try:
        from transformers.models.jamba.modeling_jamba import JambaMLP as HFJambaMLP

        return HFJambaMLP
    except ImportError:
        return None


def _get_hf_attn_decoder_layer():
    """Get HF JambaAttentionDecoderLayer class."""
    try:
        from transformers.models.jamba.modeling_jamba import (
            JambaAttentionDecoderLayer as HFJambaAttentionDecoderLayer,
        )

        return HFJambaAttentionDecoderLayer
    except ImportError:
        return None


def _get_hf_mamba_decoder_layer():
    """Get HF JambaMambaDecoderLayer class."""
    try:
        from transformers.models.jamba.modeling_jamba import (
            JambaMambaDecoderLayer as HFJambaMambaDecoderLayer,
        )

        return HFJambaMambaDecoderLayer
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Small config for testing
# ---------------------------------------------------------------------------


def _create_small_config():
    """Create a small Jamba config for testing.

    3 layers: [mamba, attention, mamba]
    (attn_layer_offset=1, attn_layer_period=2 -> layer 1 is attention)
    No MoE (num_experts=1).
    """
    try:
        from transformers.models.jamba.configuration_jamba import JambaConfig
    except ImportError:
        from tensorrt_llm._torch.auto_deploy.models.custom.modeling_jamba import JambaConfig

    return JambaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        num_experts=1,
        num_experts_per_tok=2,
        expert_layer_offset=1,
        expert_layer_period=2,
        attn_layer_offset=1,
        attn_layer_period=2,
        attention_dropout=0.0,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_dt_rank=16,
        mamba_expand=2,
        mamba_proj_bias=False,
        mamba_conv_bias=True,
        use_mamba_kernels=False,
        pad_token_id=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )


def _create_hf_config():
    """Create HF-compatible config for equivalence tests."""
    HFConfig = _get_jamba_config()
    if HFConfig is None:
        return None

    config = HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=1,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        num_experts=1,
        num_experts_per_tok=2,
        expert_layer_offset=1,
        expert_layer_period=2,
        attn_layer_offset=1,
        attn_layer_period=2,
        attention_dropout=0.0,
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_dt_rank=16,
        mamba_expand=2,
        mamba_proj_bias=False,
        mamba_conv_bias=True,
        use_mamba_kernels=False,
        pad_token_id=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
    )
    config._attn_implementation = "eager"
    return config


# =========================================================================
# Level 1: Block Equivalence Tests
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_mlp_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF."""
    HFMLP = _get_hf_jamba_mlp()
    if HFMLP is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFMLP(config)
    hf_mlp.to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = JambaMLP(config)
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
def test_jamba_attention_equivalence(B, S, dtype):
    """Test attention produces numerically close output to HF SDPA."""
    HFAttention = _get_hf_jamba_attention()
    if HFAttention is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # HF attention
    hf_attn = HFAttention(hf_config, layer_idx=1)
    hf_attn.to(device=device, dtype=dtype)
    hf_attn.eval()

    # Custom attention
    custom_attn = JambaAttention(config, layer_idx=1)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Build causal mask in HF format: [B, 1, S, S] with 0 for attend, -inf for mask
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((S, S), min_dtype, device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)

    # HF attention returns (attn_output, attn_weights, past_key_values)
    hf_out = hf_attn(hidden_states=x, attention_mask=causal_mask)[0]
    custom_out = custom_attn(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_mamba_mixer_equivalence(B, S, dtype):
    """Test Mamba v1 mixer against HF slow_forward."""
    HFMambaMixer = _get_hf_jamba_mamba_mixer()
    if HFMambaMixer is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()
    hf_config.use_mamba_kernels = False  # Force slow path

    # HF Mamba mixer
    hf_mamba = HFMambaMixer(hf_config, layer_idx=0)
    hf_mamba.to(device=device, dtype=dtype)
    hf_mamba.eval()

    # Custom Mamba mixer
    custom_mamba = JambaMambaMixer(config, layer_idx=0)
    custom_mamba.to(device=device, dtype=dtype)
    custom_mamba.load_state_dict(hf_mamba.state_dict())
    custom_mamba.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # HF slow_forward
    hf_out = hf_mamba.slow_forward(x)
    custom_out = custom_mamba(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Mamba mixer")


# =========================================================================
# Level 2: Layer Equivalence Tests
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_attention_decoder_layer_equivalence(B, S, dtype):
    """Test attention decoder layer against HF."""
    HFLayer = _get_hf_attn_decoder_layer()
    if HFLayer is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()

    # Layer 1 is attention in our config
    hf_layer = HFLayer(hf_config, layer_idx=1)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = JambaAttentionDecoderLayer(config, layer_idx=1)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    # Build causal mask in HF format: [B, 1, S, S] with 0 for attend, -inf for mask
    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full((S, S), min_dtype, device=device, dtype=dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask[None, None, :, :].expand(B, 1, -1, -1)

    # HF returns tuple (hidden_states, ...)
    hf_out = hf_layer(x, attention_mask=causal_mask)[0]
    custom_out = custom_layer(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Attention decoder layer")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_mamba_decoder_layer_equivalence(B, S, dtype):
    """Test mamba decoder layer against HF."""
    HFLayer = _get_hf_mamba_decoder_layer()
    if HFLayer is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()
    hf_config.use_mamba_kernels = False

    # Layer 0 is mamba in our config
    hf_layer = HFLayer(hf_config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype)
    hf_layer.eval()

    custom_layer = JambaMambaDecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_layer(x)[0]
    custom_out = custom_layer(x)

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Mamba decoder layer")


# =========================================================================
# Level 3: Full Model Equivalence
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_full_model_equivalence(B, S, dtype):
    """Test full model logits against HF (slow_forward path) on CUDA."""
    HFModel = _get_hf_jamba_model()
    if HFModel is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cuda"
    config = _create_small_config()
    hf_config = _create_hf_config()
    hf_config.use_mamba_kernels = False
    hf_config._attn_implementation = "eager"

    # HF model
    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    # Custom model
    custom_model = JambaForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model logits",
    )


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@torch.no_grad()
def test_jamba_full_model_equivalence_cpu(B, S):
    """Test full model logits against HF on CPU with float32."""
    HFModel = _get_hf_jamba_model()
    if HFModel is None:
        pytest.skip("transformers doesn't have jamba model")

    device = "cpu"
    dtype = torch.float32
    config = _create_small_config()
    hf_config = _create_hf_config()
    hf_config.use_mamba_kernels = False
    hf_config._attn_implementation = "eager"

    hf_model = HFModel(hf_config)
    hf_model.to(device=device, dtype=dtype)
    hf_model.eval()

    custom_model = JambaForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    custom_model.load_state_dict(hf_model.state_dict())
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits,
        hf_out.logits,
        rmse_ratio_tol=0.05,
        msg="Full model logits (CPU)",
    )


# =========================================================================
# Level 4: Export Test
# =========================================================================


def test_jamba_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_small_config()

    model = JambaForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    # Dynamic shapes for batch and sequence
    batch_dim = Dim.DYNAMIC
    seq_dim = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_dim, 1: seq_dim},
        {0: batch_dim, 1: seq_dim},
    )

    # Get reference output before export
    with torch.inference_mode():
        ref_out = model(input_ids=input_ids, position_ids=position_ids)
    ref_logits = ref_out.logits

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
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all(), "Logits should be finite"

    # Verify exported model matches original model
    assert_rmse_close(
        logits.float(), ref_logits.float(), rmse_ratio_tol=0.05, msg="Export equivalence"
    )

    # Verify dynamic shapes with different input
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all(), "Logits should be finite"


# =========================================================================
# Sanity tests (no HF dependency)
# =========================================================================


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_full_model_forward(B, S, dtype):
    """Test that the full model produces valid output."""
    device = "cuda"
    config = _create_small_config()

    model = JambaForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    output = model(input_ids=input_ids, position_ids=position_ids)

    assert output.logits.shape == (B, S, config.vocab_size)
    assert not torch.isnan(output.logits).any(), "Logits contain NaN"
    assert not torch.isinf(output.logits).any(), "Logits contain Inf"
    assert not torch.allclose(output.logits, torch.zeros_like(output.logits)), "Logits all zeros"


def test_jamba_layer_types():
    """Test that layer types match config (mamba, attention, mamba)."""
    config = _create_small_config()
    model = JambaForCausalLM(config)

    assert isinstance(model.model.layers[0], JambaMambaDecoderLayer)
    assert isinstance(model.model.layers[1], JambaAttentionDecoderLayer)
    assert isinstance(model.model.layers[2], JambaMambaDecoderLayer)


def test_jamba_config_registration():
    """Test config type and attributes."""
    config = _create_small_config()
    assert config.model_type == "jamba"
    assert hasattr(config, "mamba_d_state")
    assert hasattr(config, "mamba_dt_rank")
    assert hasattr(config, "attn_layer_offset")
    assert config.layers_block_type == ["mamba", "attention", "mamba"]


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_jamba_mamba_mixer_forward(B, S, dtype):
    """Test that Mamba mixer produces valid output."""
    device = "cuda"
    config = _create_small_config()

    mixer = JambaMambaMixer(config, layer_idx=0)
    mixer.to(device=device, dtype=dtype)
    mixer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    output = mixer(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert not torch.allclose(output, torch.zeros_like(output)), "Output all zeros"
    assert not torch.allclose(output, x), "Output unchanged from input"
