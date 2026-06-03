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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
