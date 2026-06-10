# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Factory-wiring tests for DFlashOneModelFactory (mirrors the Eagle factory tests).

Validates that the combined target+draft model is created through the factory and that the factory's
properties are correct (wrapper structure, export infos, resolved wrapper config, block_size guard).
Builds on the ``meta`` device (no weight allocation) and skips if the checkpoints are absent.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig
from utils.llm_data import llm_models_root

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers factories + ops)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_dflash import (
    DFlashDrafterForCausalLM,
    DFlashWrapper,
)
from tensorrt_llm._torch.auto_deploy.models.dflash import (
    DFlashDraftModelExportInfo,
    DFlashOneModelFactory,
)
from tensorrt_llm._torch.auto_deploy.models.eagle import TargetModelExportInfo
from tensorrt_llm._torch.auto_deploy.models.factory import ModelFactoryRegistry
from tensorrt_llm.llmapi import DFlashDecodingConfig

BLOCK_SIZE_B16 = 16
MASK_TOKEN_ID_B16 = 151669


def _paths():
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM_MODELS_ROOT not set")
    target = Path(root) / "Qwen3" / "Qwen3-8B"
    draft = Path(root) / "Qwen3-8B-DFlash-b16"
    if not target.is_dir() or not draft.is_dir():
        pytest.skip("Qwen3-8B / Qwen3-8B-DFlash-b16 checkpoints not found")
    return str(target), str(draft)


def _make_factory(max_draft_len: int = 4) -> DFlashOneModelFactory:
    target, draft = _paths()
    spec = DFlashDecodingConfig(max_draft_len=max_draft_len, speculative_model=draft)
    return DFlashOneModelFactory(
        model=target, speculative_config=spec, skip_loading_weights=True, max_seq_len=64
    )


def test_factory_registered():
    assert ModelFactoryRegistry.has("dflash_one_model")


def test_factory_builds_wrapper():
    """Factory creates a DFlashWrapper (target + draft) with wrapper config from the draft checkpoint."""
    model = _make_factory(max_draft_len=4).build_model("meta")
    assert isinstance(model, DFlashWrapper)
    assert isinstance(model.draft_model, DFlashDrafterForCausalLM)
    assert model.target_model is not None
    assert model.max_draft_len == 4
    assert model.block_size == BLOCK_SIZE_B16
    assert model.mask_token_id == MASK_TOKEN_ID_B16
    # DFlash shares embed + lm_head from the target.
    assert model.load_embedding_from_target and model.load_lm_head_from_target


def test_factory_export_infos():
    """get_export_infos returns [TargetModelExportInfo, DFlashDraftModelExportInfo] (ctx_len declared)."""
    factory = _make_factory()
    model = factory.build_model("meta")
    infos = factory.get_export_infos(model)
    assert len(infos) == 2
    assert isinstance(infos[0], TargetModelExportInfo)
    assert isinstance(infos[1], DFlashDraftModelExportInfo)
    assert infos[1].submodule_name == "draft_model"
    # ctx_len must be a declared draft graph input (so cached-attn insertion retrieves it).
    assert "ctx_len" in infos[1].dynamic_shape_lookup
    assert {"inputs_embeds", "position_ids"}.issubset(infos[1].dynamic_shape_lookup)


def test_factory_validates_block_size():
    """max_draft_len + 1 must be <= block_size (16); the factory raises otherwise (we don't clamp)."""
    factory = _make_factory(max_draft_len=20)  # 21 > 16
    with pytest.raises(ValueError, match="block_size"):
        factory.build_model("meta")


def test_factory_preserves_zero_mask_token_id(monkeypatch):
    """An explicit mask_token_id=0 must not fall through to the draft config fallback."""
    spec = DFlashDecodingConfig(
        max_draft_len=1,
        speculative_model="test-dflash-draft",
        mask_token_id=0,
        target_layer_ids=[0],
    )
    factory = DFlashOneModelFactory(
        model="test-target",
        speculative_config=spec,
        skip_loading_weights=True,
        max_seq_len=64,
    )
    draft_config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        vocab_size=32,
    )
    draft_config.block_size = 4
    draft_config.dflash_config = {"target_layer_ids": [0], "mask_token_id": 7}
    monkeypatch.setattr(factory, "prefetch_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(factory, "_build_draft_config", lambda: draft_config)
    monkeypatch.setattr(factory.target_factory, "build_model", lambda _device: nn.Module())

    model = factory.build_model("meta")

    assert model.mask_token_id == 0


def test_prefetch_checkpoint_fetches_dflash_draft(monkeypatch):
    spec = DFlashDecodingConfig(
        max_draft_len=1,
        speculative_model="test-dflash-draft",
        target_layer_ids=[0],
    )
    factory = DFlashOneModelFactory(
        model="test-target",
        speculative_config=spec,
        skip_loading_weights=True,
        max_seq_len=64,
    )
    calls = []

    def record_target_prefetch(force=False, skip_loading_weights=None):
        calls.append(("target", force, skip_loading_weights))

    def record_draft_prefetch(force=False, skip_loading_weights=None):
        calls.append(("draft", force, skip_loading_weights))

    monkeypatch.setattr(factory.target_factory, "prefetch_checkpoint", record_target_prefetch)
    monkeypatch.setattr(
        factory.draft_checkpoint_factory, "prefetch_checkpoint", record_draft_prefetch
    )

    factory.prefetch_checkpoint(force=True, skip_loading_weights=False)

    assert ("target", True, False) in calls
    assert ("draft", True, False) in calls


@torch.inference_mode()
def test_draft_weights_load_strict():
    """Draft loads the real z-lab checkpoint with strict=True (fidelity: 58 tensors match exactly)."""
    factory = _make_factory()
    draft_config = factory._build_draft_config()
    draft_dtype = getattr(draft_config, "torch_dtype", None)
    draft_model = DFlashDrafterForCausalLM(draft_config).to(device="cuda", dtype=draft_dtype).eval()
    # Must not raise (strict=True validates exact key match across all 58 tensors).
    factory._load_draft_weights(draft_model, "cuda")
    assert torch.isfinite(draft_model.model.fc.weight).all()
    assert draft_model.model.hidden_norm.weight.shape[0] == draft_config.hidden_size
    if draft_dtype is not None:
        assert draft_model.model.fc.weight.dtype == draft_dtype


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
