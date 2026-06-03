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
"""DFlashWrapper tests (tiny target + draft, no full pipeline / no 8B build).

Validates the export-time prefill path end-to-end: target -> bonus token -> query block
[bonus, MASK, ...] -> draft (emits dflash_attention) -> lm_head -> draft tokens. Eager (cheap); the
kv-cache inference path + full export are validated at E2E (build_and_run_ad).
"""

import pytest
import torch
from transformers import Qwen3Config

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers ops)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_dflash import (
    DFlashDrafterForCausalLM,
    DFlashWrapper,
    DFlashWrapperConfig,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3 import Qwen3ForCausalLM

DEVICE = "cuda"
DTYPE = torch.float16
B, S = 2, 6
MAX_DRAFT_LEN, BLOCK_SIZE, MASK_TOKEN_ID = 4, 8, 5
VOCAB, HIDDEN, N_DRAFT_LAYERS = 256, 32, 2


def _qwen3_config(num_layers):
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_theta=1.0e6,
        attention_bias=False,
        hidden_act="silu",
        tie_word_embeddings=False,
        torch_dtype=DTYPE,  # drives DFlashModel.dtype (the wrapper reads it via _draft_dtype)
    )


def _build_tiny_wrapper():
    torch.manual_seed(0)
    target = Qwen3ForCausalLM(_qwen3_config(4)).to(device=DEVICE, dtype=DTYPE).eval()

    draft_cfg = _qwen3_config(N_DRAFT_LAYERS)
    draft_cfg.dflash_config = {"target_layer_ids": [0, 1], "mask_token_id": MASK_TOKEN_ID}
    draft_cfg.block_size = BLOCK_SIZE
    draft = DFlashDrafterForCausalLM(draft_cfg).to(device=DEVICE, dtype=DTYPE).eval()

    cfg = DFlashWrapperConfig(
        max_draft_len=MAX_DRAFT_LEN, block_size=BLOCK_SIZE, mask_token_id=MASK_TOKEN_ID
    )
    return DFlashWrapper(cfg, target, draft)


@torch.inference_mode()
def test_prefill_only_forward_shapes():
    """Export-path prefill: target+draft run once; output is [B, max_draft_len+1] and finite."""
    wrapper = _build_tiny_wrapper()
    input_ids = torch.randint(0, VOCAB, (B, S), device=DEVICE)
    position_ids = torch.arange(S, device=DEVICE).unsqueeze(0).expand(B, S).contiguous()

    out = wrapper(input_ids=input_ids, position_ids=position_ids)  # prefill dispatch (no cm)
    assert out.new_tokens.shape == (B, MAX_DRAFT_LEN + 1)
    assert out.new_tokens_lens.shape == (B,)
    assert out.new_tokens.dtype == torch.long
    assert torch.all(out.new_tokens >= 0) and torch.all(out.new_tokens < VOCAB)


@torch.inference_mode()
def test_kv_cache_forward_is_stub():
    """The inference (kv-cache) path is the E2E continuation -- currently a documented stub."""
    wrapper = _build_tiny_wrapper()

    class _DummyCSI:
        pass

    with pytest.raises(NotImplementedError):
        wrapper(cache_seq_interface=_DummyCSI())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
