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

"""Byte-exactness tests for the DeepSeek V4 ``topk_is_placeholder`` optimization.

The model emits a cheap width-only ``topk_idxs`` allocation for compressed layers and
passes ``topk_is_placeholder=True`` so the sparse-attention op rebuilds the real
window+compressed selection on the value-reading prefill path (the cached decode path
consumes only the width). These tests pin that the rebuilt indices and the resulting op
output are bit-identical to the pre-optimization explicit placeholder.
"""

from __future__ import annotations

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register custom ops
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _build_placeholder_topk_idxs,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4Config,
    _window_topk_idxs,
)


def _ref_placeholder_topk_idxs(
    window_size: int,
    compress_ratio: int,
    batch_size: int,
    seq_len: int,
    compressed_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Reference: the pre-optimization model-side window + compressed placeholder chain.

    Mirrors ``modeling_deepseek_v4._window_topk_idxs`` (reused directly here) concatenated
    with the removed ``_compress_topk_idxs`` (offset == seq_len) then cast to int64.
    """
    window_idxs = _window_topk_idxs(window_size, batch_size, seq_len, device)
    compressed_positions = torch.arange(compressed_width, device=device)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // compress_ratio
    compressed_idxs = compressed_positions.unsqueeze(0).expand(seq_len, -1)
    compressed_idxs = torch.where(compressed_idxs < valid_lengths, compressed_idxs + seq_len, -1)
    compressed_idxs = compressed_idxs.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat((window_idxs, compressed_idxs), dim=-1).to(torch.int64)


@pytest.mark.parametrize("compress_ratio,compressed_width", [(4, 6), (4, 2), (128, 3)])
@pytest.mark.parametrize("batch_size,seq_len,window_size", [(1, 12, 4), (2, 9, 3), (1, 1, 4)])
def test_build_placeholder_topk_idxs_matches_reference(
    compress_ratio: int,
    compressed_width: int,
    batch_size: int,
    seq_len: int,
    window_size: int,
) -> None:
    device = torch.device("cpu")
    got = _build_placeholder_topk_idxs(
        window_size, compress_ratio, batch_size, seq_len, compressed_width, device
    )
    ref = _ref_placeholder_topk_idxs(
        window_size, compress_ratio, batch_size, seq_len, compressed_width, device
    )
    assert got.dtype == torch.int64
    # The cached op recovers ``index_topk`` as ``topk_idxs.shape[-1] - window_size``.
    assert tuple(got.shape) == (batch_size, seq_len, window_size + compressed_width)
    assert torch.equal(got, ref)


def _compressor_case(compress_ratio: int, seq_len: int, device: torch.device, batch_size: int = 1):
    hidden_size = 16
    head_dim = 8
    rope_dim = 4
    capacity = seq_len
    config = DeepseekV4Config(
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity,
        ad_rope_cache_len=max(capacity, seq_len, 1),
    )
    compressor = DeepseekV4Compressor(config, compress_ratio, head_dim).eval().to(device)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device)
    compressor_kv, compressor_gate = compressor.project(hidden_states)
    positions = torch.arange(max(capacity, seq_len, 1), dtype=torch.float32, device=device)
    positions = positions.unsqueeze(1)
    freqs = torch.linspace(0.05, 0.25, rope_dim // 2, dtype=torch.float32, device=device)
    freqs = freqs.unsqueeze(0)
    angles = positions * freqs
    cos_table, sin_table = angles.cos(), angles.sin()
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.contiguous()
    return compressor, compressor_kv, compressor_gate, cos_table, sin_table, position_ids, head_dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compressor projection uses Triton")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_no_cache_op_placeholder_matches_explicit(compress_ratio: int) -> None:
    """The width-only placeholder + rebuild is bit-identical to the explicit placeholder.

    fp32 + head_dim < 16 keeps the attend on the deterministic reference chunk loop
    (the fused Triton path requires cuda + bf16/fp16 + head_dim >= 16), so the two
    invocations are exactly comparable with ``torch.equal``.
    """
    torch.manual_seed(20260702 + compress_ratio)
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 2 * compress_ratio
    window_size = 4
    num_heads = 2

    (
        compressor,
        compressor_kv,
        compressor_gate,
        cos_table,
        sin_table,
        position_ids,
        head_dim,
    ) = _compressor_case(compress_ratio, seq_len, device, batch_size)

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    kv = torch.randn(batch_size, seq_len, head_dim, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)
    softmax_scale = head_dim**-0.5

    empty_iq = q.new_empty(batch_size, seq_len, 0, 0)
    empty_iw = q.new_empty(batch_size, seq_len, 0)
    empty_ick = q.new_empty(batch_size, seq_len, 0)
    empty_icg = q.new_empty(batch_size, seq_len, 0)
    empty_ica = q.new_empty(0, 0)
    empty_icn = q.new_empty(0)

    compressed_width = compressor.max_compressed_len

    def _run(topk_idxs: torch.Tensor, topk_is_placeholder: bool) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor.ape,
            compressor.norm.weight,
            cos_table,
            sin_table,
            position_ids,
            empty_iq,
            empty_iw,
            empty_ick,
            empty_icg,
            empty_ica,
            empty_icn,
            softmax_scale,
            window_size=window_size,
            compress_ratio=compress_ratio,
            max_compressed_len=compressed_width,
            rope_dim=compressor.rope_head_dim,
            rms_norm_eps=compressor.norm.eps,
            topk_is_placeholder=topk_is_placeholder,
        )

    explicit_topk = _build_placeholder_topk_idxs(
        window_size, compress_ratio, batch_size, seq_len, compressed_width, q.device
    )
    out_explicit = _run(explicit_topk, topk_is_placeholder=False)

    # Model-side emission: a cheap width-only allocation; values are never read because
    # the op rebuilds the real selection when ``topk_is_placeholder`` is set.
    width_only_topk = q.new_empty(
        batch_size, seq_len, window_size + compressed_width, dtype=torch.int64
    )
    out_placeholder = _run(width_only_topk, topk_is_placeholder=True)

    assert torch.equal(out_explicit, out_placeholder)
