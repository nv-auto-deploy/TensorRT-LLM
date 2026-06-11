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

"""Unit tests for the fused Step-3.7-Flash head-wise attention gate custom op.

The op ``auto_deploy::step3p7_head_gate`` fuses Step's per-head attention gate
(``attn_output * sigmoid(g_proj(hidden))[..., None]``) -- its sigmoid + per-head
broadcast-multiply -- into one Triton launch (instead of two launch-bound
elementwise kernels per attention layer at batch=1 decode). These tests validate
the op is (1) numerically faithful to the separate-op reference, (2) safe to
capture/replay inside a CUDA graph (the production execution mode), and (3)
traceable by ``torch.export`` (uses ``register_fake``).
"""

import pytest
import torch

# Importing the model module registers the custom op at import time.
from tensorrt_llm._torch.auto_deploy.models.custom import modeling_step3p7  # noqa: F401


def _reference_gate(attn_output, gate_logits):
    """The exact separate-op reference replaced by the fused op."""
    return attn_output * gate_logits.sigmoid().unsqueeze(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "bsz,seq,n_heads,head_dim",
    [
        (1, 1, 8, 128),  # decode, full-attention per-rank head count (64/8)
        (1, 1, 12, 128),  # decode, sliding-attention per-rank head count (96/8)
        (1, 1000, 8, 128),  # prefill
        (2, 4, 16, 64),  # generic shape (non-128 head_dim, batched)
    ],
)
def test_head_gate_matches_reference(bsz, seq, n_heads, head_dim):
    device = "cuda"
    torch.manual_seed(0)
    attn = torch.randn(bsz, seq, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    # Scale logits so sigmoid spans (0, 1) meaningfully.
    gate_logits = torch.randn(bsz, seq, n_heads, dtype=torch.bfloat16, device=device) * 2.0

    ref = _reference_gate(attn, gate_logits)
    out = torch.ops.auto_deploy.step3p7_head_gate(attn, gate_logits)

    assert out.shape == attn.shape
    assert out.dtype == attn.dtype
    # Fused kernel mirrors PyTorch's bf16 sigmoid + multiply rounding (a structural
    # bug would produce O(1) errors far beyond this bf16-rounding tolerance).
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_head_gate_cuda_graph():
    """The op must be capturable and replayable inside a CUDA graph (production mode)."""
    device = "cuda"
    bsz, seq, n_heads, head_dim = 1, 1, 8, 128
    torch.manual_seed(3)

    static_attn = torch.randn(bsz, seq, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    static_gate = torch.randn(bsz, seq, n_heads, dtype=torch.bfloat16, device=device) * 2.0
    op = torch.ops.auto_deploy.step3p7_head_gate

    # Warmup on a side stream (required before capture).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            op(static_attn, static_gate)
    torch.cuda.current_stream().wait_stream(s)

    static_out = torch.empty_like(static_attn)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o = op(static_attn, static_gate)
        static_out.copy_(o)

    # Replay with new data copied into the captured input buffers.
    torch.manual_seed(11)
    new_attn = torch.randn(bsz, seq, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    new_gate = torch.randn(bsz, seq, n_heads, dtype=torch.bfloat16, device=device) * 2.0
    static_attn.copy_(new_attn)
    static_gate.copy_(new_gate)
    g.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        static_out, _reference_gate(new_attn, new_gate), atol=2e-2, rtol=2e-2
    )


def test_head_gate_exports():
    """register_fake must allow torch.export (meta path) with exactly one fused node."""

    class M(torch.nn.Module):
        def forward(self, attn, gate):
            return torch.ops.auto_deploy.step3p7_head_gate(attn, gate)

    m = M()
    ep = torch.export.export(
        m,
        (
            torch.randn(1, 1, 8, 128, dtype=torch.bfloat16),
            torch.randn(1, 1, 8, dtype=torch.bfloat16),
        ),
    )
    assert ep is not None
    target = torch.ops.auto_deploy.step3p7_head_gate.default
    n = sum(1 for node in ep.graph.nodes if node.op == "call_function" and node.target is target)
    assert n == 1, f"expected exactly one fused gate node, found {n}"
