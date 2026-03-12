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

HF reference classes are loaded directly from the HF snapshot because
InternLM3 is not part of standard transformers (uses auto_map). The HF
modeling file requires a LossKwargs stub for the installed transformers version.
"""

import importlib.util
import os
import sys
import types

import pytest
import torch
import transformers.utils as _transformers_utils
from _model_test_utils import assert_rmse_close
from torch.export import Dim

# ---------------------------------------------------------------------------
# Load HF InternLM3 classes from the local snapshot.
#
# The HF files use relative imports (from .configuration_internlm3 import ...)
# so they must be loaded as a synthetic package via importlib.util.
# The HF modeling file also imports LossKwargs which is absent in the
# installed transformers version — stub it before loading.
# ---------------------------------------------------------------------------
_HF_SNAPSHOT = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/"
    "autodeploy_data/hf_home/hub/models--internlm--internlm3-8b-instruct/"
    "snapshots/28c99415adaf61767bd1c619f4f99f308fdfd223"
)

if not hasattr(_transformers_utils, "LossKwargs"):
    from typing import TypedDict

    _transformers_utils.LossKwargs = TypedDict("LossKwargs", {}, total=False)


def _load_hf_snapshot_module(name: str, filename: str):
    """Load a module from the HF snapshot as part of a synthetic package."""
    _PKG = "_hf_internlm3_snapshot"
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [_HF_SNAPSHOT]
        pkg.__package__ = _PKG
        sys.modules[_PKG] = pkg
    full_name = f"{_PKG}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = importlib.util.spec_from_file_location(full_name, os.path.join(_HF_SNAPSHOT, filename))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _PKG
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod


_hf_cfg_mod = _load_hf_snapshot_module("configuration_internlm3", "configuration_internlm3.py")
_hf_mod = _load_hf_snapshot_module("modeling_internlm3", "modeling_internlm3.py")

InternLM3Config = _hf_cfg_mod.InternLM3Config
HFInternLM3Attention = _hf_mod.InternLM3Attention
HFInternLM3DecoderLayer = _hf_mod.InternLM3DecoderLayer
HFInternLM3ForCausalLM = _hf_mod.InternLM3ForCausalLM
HFInternLM3MLP = _hf_mod.InternLM3MLP
HFInternLM3RMSNorm = _hf_mod.InternLM3RMSNorm

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm  # noqa: E402
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_internlm3 import (  # noqa: E402
    InternLM3Attention,
    InternLM3DecoderLayer,
    InternLM3ForCausalLM,
    InternLM3MLP,
    InternLM3RMSNorm,
    InternLM3RotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device  # noqa: E402

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def _create_small_config() -> InternLM3Config:
    """Create a small InternLM3 config for testing."""
    cfg = InternLM3Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
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
    cfg._attn_implementation = "eager"
    return cfg


def _make_4d_causal_mask(B: int, S: int, device, dtype) -> torch.Tensor:
    """Build additive 4-D causal mask [B, 1, S, S] for HF eager attention."""
    mask = torch.zeros(B, 1, S, S, device=device, dtype=dtype)
    upper = torch.ones(S, S, device=device, dtype=torch.bool).triu(diagonal=1)
    mask.masked_fill_(upper.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask


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

    hf_norm = HFInternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    hf_norm.to(device=device, dtype=dtype).eval()

    custom_norm = InternLM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    custom_norm.to(device=device, dtype=dtype)
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_norm(x), hf_norm(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_mlp_equivalence(B, S, dtype):
    """Test MLP layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_mlp = HFInternLM3MLP(config)
    hf_mlp.to(device=device, dtype=dtype).eval()

    custom_mlp = InternLM3MLP(config)
    custom_mlp.to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    torch.testing.assert_close(custom_mlp(x), hf_mlp(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_internlm3_attention_equivalence(B, S, dtype):
    """Test Attention layer produces numerically equivalent output to HF reference."""
    device = "cuda"
    config = _create_small_config()

    hf_attn = HFInternLM3Attention(config, layer_idx=0)
    hf_attn.to(device=device, dtype=dtype).eval()

    custom_attn = InternLM3Attention(config, layer_idx=0)
    custom_attn.to(device=device, dtype=dtype)
    custom_attn.load_state_dict(hf_attn.state_dict())
    custom_attn.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    causal_mask = _make_4d_causal_mask(B, S, device, dtype)

    # HF: let the attention module compute position_embeddings via its own rotary_emb
    hf_out = hf_attn(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
    )[0]

    # Custom: rotary returns full table [max_pos, head_dim]; attention slices
    custom_rotary = InternLM3RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_full_cos, custom_full_sin = custom_rotary(x)
    custom_out = custom_attn(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_full_cos, custom_full_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


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

    hf_layer = HFInternLM3DecoderLayer(config, layer_idx=0)
    hf_layer.to(device=device, dtype=dtype).eval()

    custom_layer = InternLM3DecoderLayer(config, layer_idx=0)
    custom_layer.to(device=device, dtype=dtype)
    custom_layer.load_state_dict(hf_layer.state_dict())
    custom_layer.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    causal_mask = _make_4d_causal_mask(B, S, device, dtype)

    # HF: let the layer compute position_embeddings via its own rotary_emb
    hf_out = hf_layer(
        hidden_states=x,
        attention_mask=causal_mask,
        position_ids=position_ids,
    )[0]

    custom_rotary = InternLM3RotaryEmbedding(config)
    custom_rotary.to(device=device, dtype=dtype)
    custom_full_cos, custom_full_sin = custom_rotary(x)
    custom_out = custom_layer(
        hidden_states=x,
        position_ids=position_ids,
        position_embeddings=(custom_full_cos, custom_full_sin),
    )

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer: ")


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

    hf_model = HFInternLM3ForCausalLM(config)
    hf_model.to(device=device, dtype=dtype).eval()

    custom_model = InternLM3ForCausalLM(config)
    custom_model.to(device=device, dtype=dtype)
    # State dict keys match directly: both have model.embed_tokens, model.layers.*, lm_head
    custom_model.load_state_dict(hf_model.state_dict(), strict=False)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
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
