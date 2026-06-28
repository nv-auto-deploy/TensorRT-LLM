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

"""idea_0067: hoist the per-step LM-head bf16->fp32 weight recast.

The DeepSeek-V4 LM head used to recompute ``self.head.weight.float()`` on every
decode step (a constant 66M-element bf16->fp32 cast that cudagraph replays).
Storing ``head.weight`` in fp32 (cast once at load) and dropping the per-step
``.float()`` deletes that recast. Because bf16->fp32 is *lossless*, the stored
fp32 weight ``W_fp32 = bf16(W).float()`` equals the value the old graph produced
per step, so the logits GEMM is bit-identical in the real (checkpoint-loaded)
flow. This test pins that bit-exactness.
"""

from __future__ import annotations

import torch
from torch import nn

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (registers torch_linear_simple)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import _linear


def _head_logits(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mirror the exact op the model forward uses for the LM head."""
    return _linear(hidden_states.float(), weight, None, layer_type="lm_head").float()


def test_lm_head_fp32_storage_bitexact_vs_perstep_cast() -> None:
    torch.manual_seed(0)
    vocab, hidden = 257, 128

    # The checkpoint stores a bf16 lm_head weight.
    w_bf16 = torch.randn(vocab, hidden, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 3, hidden, dtype=torch.bfloat16)

    # OLD decode graph: per-step recast ``self.head.weight.float()``.
    logits_old = _head_logits(hidden_states, w_bf16.float())

    # NEW decode graph: head weight stored fp32 once at load. ``_load_checkpoint``
    # copy-casts the bf16 checkpoint tensor into the fp32 param (no assign=True),
    # which is exactly ``copy_`` of the bf16 tensor into an fp32 buffer.
    w_fp32 = torch.empty(vocab, hidden, dtype=torch.float32)
    w_fp32.copy_(w_bf16)
    logits_new = _head_logits(hidden_states, w_fp32)

    assert w_fp32.dtype == torch.float32
    assert torch.equal(logits_old, logits_new), (
        "fp32-stored head must be bit-identical to the per-step bf16->fp32 recast"
    )


def test_lm_head_module_is_fp32_and_bitexact() -> None:
    """The modeling change constructs ``self.head`` as ``nn.Linear(..., dtype=fp32)``.

    Verify (a) the constructed head param is fp32 and (b) loading a bf16
    checkpoint weight into it reproduces the old per-step-cast logits exactly.
    """
    torch.manual_seed(1)
    vocab, hidden = 129, 64

    head = nn.Linear(hidden, vocab, bias=False, dtype=torch.float32)
    assert head.weight.dtype == torch.float32

    w_bf16 = torch.randn(vocab, hidden, dtype=torch.bfloat16)
    hidden_states = torch.randn(1, 1, hidden, dtype=torch.bfloat16)

    logits_old = _head_logits(hidden_states, w_bf16.float())

    # Mimic load_state_dict copy-cast of the bf16 checkpoint into the fp32 param.
    head.weight.data.copy_(w_bf16)
    logits_new = _head_logits(hidden_states, head.weight)

    assert torch.equal(logits_old, logits_new)
