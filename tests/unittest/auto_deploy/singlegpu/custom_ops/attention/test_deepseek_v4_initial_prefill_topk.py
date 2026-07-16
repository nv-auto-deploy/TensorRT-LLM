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

"""Regression tests: the initial-prefill placeholder topk is rebuilt PER SEQUENCE.

``torch_deepseek_v4_sparse_attention_with_cache`` used to rebuild the width-only
``topk_idxs`` placeholder once in the GLOBAL flattened (possibly padded) token frame
(window keys at global flattened offsets, compressed slots offset by ``q.shape[1]``)
and then slice it per sequence, while each ``input_pos == 0`` sequence attends against
PER-SEQUENCE LOCAL kv (window rows local, compressed rows appended at ``seq_len_i``).
The two frames coincide only for a single exactly-sized prefill; the mismatch silently
mis-selected rows whenever the forward was padded (piecewise-cudagraph bucket wider
than the actual token count), contained two or more prefill sequences, or mixed
prefill with decode.

These tests pin the fixed per-sequence rebuild with ``torch.equal`` (fp32 keeps the
attend on the deterministic reference chunk loop):

* padded single prefill == exact-width prefill == uncached source op (the source-op
  control also pins that the exactly-sized case is unchanged by the fix — that is the
  one case where old global frame and new local frame coincide);
* two-sequence batched initial prefill == each sequence run alone (compressed ratio-4
  on CUDA and window-only on CPU);
* mixed prefill+decode batch: the prefill half == the same sequence run alone, and the
  decode half is invariant to the prefill partner it is batched with.

The metadata helpers are private copies of the ``_standard_metadata`` /
``_prepare_extra_metadata`` pattern in ``test_deepseek_v4_sparse_attention.py``.
"""

from __future__ import annotations

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register custom ops
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4Indexer,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="compressed sparse-attention path requires CUDA (Triton fused ops)",
)

_HIDDEN_SIZE = 16
_HEAD_DIM = 8
_ROPE_DIM = 4
_NUM_HEADS = 2
_WINDOW_SIZE = 4
_SOFTMAX_SCALE = _HEAD_DIM**-0.5


def _base_meta(
    seq_lens: list[int],
    input_positions: list[int],
    slot_indices: list[int],
    num_prefill: int,
) -> tuple[torch.Tensor, ...]:
    """Base metadata tuple (prefill sequences first, then single-token decodes).

    (batch_info, seq_len, input_pos, slot_idx, cu_seqlen, cu_num_pages, cache_loc);
    one whole-sequence cache page per slot (tokens_per_block == cache.shape[1]).
    """
    num_decode = len(seq_lens) - num_prefill
    num_prefill_tokens = sum(seq_lens[:num_prefill])
    cu_seqlen = [0]
    for seq_len in seq_lens:
        cu_seqlen.append(cu_seqlen[-1] + seq_len)
    batch_info_host = BatchInfo()
    batch_info_host.update([num_prefill, num_prefill_tokens, 0, 0, num_decode, num_decode])
    return (
        batch_info_host.serialize(),
        torch.tensor(seq_lens, dtype=torch.int32),
        torch.tensor(input_positions, dtype=torch.int32),
        torch.tensor(slot_indices, dtype=torch.int64),
        torch.tensor(cu_seqlen, dtype=torch.int32),
        torch.arange(len(seq_lens) + 1, dtype=torch.int32),
        torch.tensor(slot_indices, dtype=torch.int32),
    )


def _prefill_meta(seq_lens: list[int], slot_indices: list[int] | None = None):
    slot_indices = slot_indices if slot_indices is not None else list(range(len(seq_lens)))
    return _base_meta(seq_lens, [0] * len(seq_lens), slot_indices, num_prefill=len(seq_lens))


def _mixed_meta(prefill_len: int, decode_pos: int, slot_indices: list[int]):
    return _base_meta([prefill_len, 1], [0, decode_pos], slot_indices, num_prefill=1)


def _standard_metadata(base_meta: tuple[torch.Tensor, ...], device: torch.device):
    """The 10 standard metadata tensors of the cached op (device tensors + host mirrors)."""
    batch_info_host, seq_len, input_pos, slot_idx, cu_seqlen, cu_num_pages, cache_loc = base_meta
    return (
        batch_info_host,
        input_pos.to(device),
        slot_idx.to(device),
        cu_num_pages.to(device),
        cache_loc.to(device),
        seq_len.cpu(),
        input_pos.cpu(),
        cu_seqlen.cpu(),
        cu_num_pages.cpu(),
        cache_loc.cpu(),
    )


def _prepare_extra_metadata(
    base_meta: tuple[torch.Tensor, ...],
    swa_cache: torch.Tensor,
    *,
    window_size: int | None = None,
    compress_ratio: int = 0,
    max_compressed_len: int | None = None,
    position_ids: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """The 23 hoisted metadata tensors, produced the way production does.

    Their values feed only the pure-decode fastpath; prefill/mixed forwards (the paths
    under test here) never read them, but the fixed 23-output contract must hold.
    """
    _, _, input_pos, _, _, cu_num_pages, cache_loc = base_meta
    device = swa_cache.device
    input_pos = input_pos.to(device)
    position_ids = input_pos if position_ids is None else position_ids.to(device)
    overlap_m = max_compressed_len if compress_ratio == 4 else None
    dense_m = max_compressed_len if compress_ratio == 128 else None
    return torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr(
        input_pos,
        position_ids,
        cu_num_pages.to(device).contiguous(),
        cache_loc.to(device).contiguous(),
        int(swa_cache.shape[1]),
        max(int(overlap_m or 1), 1),
        max(int(dense_m or 1), 1),
        max(int(window_size or 1), 1),
    )


def _rope_tables(max_seq_len: int, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.linspace(0.05, 0.25, rope_dim // 2, dtype=torch.float32).unsqueeze(0)
    angles = positions * freqs
    return angles.cos(), angles.sin()


def _make_compressor(
    compress_ratio: int, capacity_tokens: int, device: torch.device
) -> tuple[DeepseekV4Compressor, torch.Tensor, torch.Tensor]:
    config = DeepseekV4Config(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity_tokens,
        ad_rope_cache_len=capacity_tokens,
    )
    compressor = DeepseekV4Compressor(config, compress_ratio, _HEAD_DIM).eval().to(device)
    cos_table, sin_table = (t.to(device) for t in _rope_tables(capacity_tokens, _ROPE_DIM))
    return compressor, cos_table, sin_table


def _make_caches(
    num_slots: int,
    cache_tokens: int,
    state_dim: int,
    indexer_state_dim: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """[swa, mhc, compressor_kv, compressor_gate, indexer_kv, indexer_gate] caches."""
    return [
        torch.full((num_slots, cache_tokens, _HEAD_DIM), 777.0, device=device),
        torch.full((num_slots, cache_tokens, _HEAD_DIM), 777.0, device=device),
        torch.full((num_slots, cache_tokens, state_dim), 777.0, device=device),
        torch.full((num_slots, cache_tokens, state_dim), 777.0, device=device),
        torch.full((num_slots, cache_tokens, indexer_state_dim), 777.0, device=device),
        torch.full((num_slots, cache_tokens, indexer_state_dim), 777.0, device=device),
    ]


def _placeholder_topk(width: int, num_tokens: int, device: torch.device) -> torch.Tensor:
    """Width-only placeholder exactly like the model emits (values never valid)."""
    return torch.full((1, num_tokens, width), 7, dtype=torch.int64, device=device)


def _run_cached_placeholder(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    base_meta: tuple[torch.Tensor, ...],
    caches: list[torch.Tensor],
    topk_width: int,
    *,
    window_size: int = _WINDOW_SIZE,
    compress_ratio: int = 0,
    compressor: DeepseekV4Compressor | None = None,
    compressor_kv: torch.Tensor | None = None,
    compressor_gate: torch.Tensor | None = None,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    indexer: DeepseekV4Indexer | None = None,
    indexer_q: torch.Tensor | None = None,
    indexer_weights: torch.Tensor | None = None,
    indexer_compressor_kv: torch.Tensor | None = None,
    indexer_compressor_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the cached op with ``topk_is_placeholder=True`` on prefill/mixed metadata."""
    batch, num_tokens = q.shape[0], q.shape[1]
    if compress_ratio:
        assert compressor is not None
        max_compressed_len = compressor.max_compressed_len
        rope_dim = compressor.rope_head_dim
        rms_norm_eps = compressor.norm.eps
        compressor_ape = compressor.ape
        compressor_norm_weight = compressor.norm.weight
    else:
        max_compressed_len = None
        rope_dim = None
        rms_norm_eps = 1e-6
        compressor_kv = q.new_empty(batch, num_tokens, 0)
        compressor_gate = q.new_empty(batch, num_tokens, 0)
        compressor_ape = q.new_empty(0, 0)
        compressor_norm_weight = q.new_empty(0)
        cos_table = q.new_empty(0, 0)
        sin_table = q.new_empty(0, 0)
        position_ids = q.new_zeros(batch, num_tokens)
    if indexer is None:
        indexer_q = q.new_empty(batch, num_tokens, 0, 0)
        indexer_weights = q.new_empty(batch, num_tokens, 0)
        indexer_compressor_kv = q.new_empty(batch, num_tokens, 0)
        indexer_compressor_gate = q.new_empty(batch, num_tokens, 0)
        indexer_compressor_ape = q.new_empty(0, 0)
        indexer_compressor_norm_weight = q.new_empty(0)
    else:
        indexer_compressor_ape = indexer.compressor.ape
        indexer_compressor_norm_weight = indexer.compressor.norm.weight
    metadata = (
        *_standard_metadata(base_meta, q.device),
        *_prepare_extra_metadata(
            base_meta,
            caches[0],
            window_size=window_size,
            compress_ratio=compress_ratio,
            max_compressed_len=max_compressed_len,
        ),
    )
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        q,
        kv,
        attn_sink,
        _placeholder_topk(topk_width, num_tokens, q.device),
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        *metadata,
        *caches,
        _SOFTMAX_SCALE,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
        topk_is_placeholder=True,
    )


def _run_source_placeholder(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_width: int,
    *,
    window_size: int = _WINDOW_SIZE,
    compress_ratio: int = 0,
    compressor: DeepseekV4Compressor | None = None,
    compressor_kv: torch.Tensor | None = None,
    compressor_gate: torch.Tensor | None = None,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Uncached source op with ``topk_is_placeholder=True`` (single-sequence ground truth)."""
    batch, num_tokens = q.shape[0], q.shape[1]
    if compress_ratio:
        assert compressor is not None
        max_compressed_len = compressor.max_compressed_len
        rope_dim = compressor.rope_head_dim
        rms_norm_eps = compressor.norm.eps
        compressor_ape = compressor.ape
        compressor_norm_weight = compressor.norm.weight
    else:
        max_compressed_len = None
        rope_dim = None
        rms_norm_eps = 1e-6
        compressor_kv = q.new_empty(batch, num_tokens, 0)
        compressor_gate = q.new_empty(batch, num_tokens, 0)
        compressor_ape = q.new_empty(0, 0)
        compressor_norm_weight = q.new_empty(0)
        cos_table = q.new_empty(0, 0)
        sin_table = q.new_empty(0, 0)
        position_ids = q.new_zeros(batch, num_tokens)
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
        q,
        kv,
        attn_sink,
        _placeholder_topk(topk_width, num_tokens, q.device),
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        position_ids,
        q.new_empty(batch, num_tokens, 0, 0),
        q.new_empty(batch, num_tokens, 0),
        q.new_empty(batch, num_tokens, 0),
        q.new_empty(batch, num_tokens, 0),
        q.new_empty(0, 0),
        q.new_empty(0),
        _SOFTMAX_SCALE,
        window_size=window_size,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        rope_dim=rope_dim,
        rms_norm_eps=rms_norm_eps,
        topk_is_placeholder=True,
    )


@_requires_cuda
@pytest.mark.parametrize(
    "compress_ratio,actual_len,padded_len,capacity",
    [(4, 12, 16, 16), (128, 128, 160, 256)],
)
def test_padded_initial_prefill_matches_exact_width(
    compress_ratio: int, actual_len: int, padded_len: int, capacity: int
) -> None:
    """Prefill fed at a padded width (cudagraph bucket) == exact width == source op.

    The exact-width == source-op control doubles as the fixed-point check: a single
    exactly-sized prefill is where the old global frame and the new per-sequence
    local frame coincide, so the fix must leave that case bit-identical.
    """
    torch.manual_seed(20260716 + compress_ratio)
    device = torch.device("cuda")
    compressor, cos_table, sin_table = _make_compressor(compress_ratio, capacity, device)
    topk_width = _WINDOW_SIZE + compressor.max_compressed_len

    hidden = torch.randn(1, padded_len, _HIDDEN_SIZE, device=device)
    compressor_kv, compressor_gate = compressor.project(hidden)  # token-wise -> sliceable
    q = torch.randn(1, padded_len, _NUM_HEADS, _HEAD_DIM, device=device)
    kv = torch.randn(1, padded_len, _HEAD_DIM, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)
    position_ids = torch.arange(padded_len, device=device).unsqueeze(0).contiguous()

    def run(width: int) -> torch.Tensor:
        caches = _make_caches(1, capacity, compressor_kv.shape[-1], 0, device)
        out = _run_cached_placeholder(
            q[:, :width].contiguous(),
            kv[:, :width].contiguous(),
            attn_sink,
            _prefill_meta([actual_len]),  # batch info: actual_len active tokens
            caches,
            topk_width,
            compress_ratio=compress_ratio,
            compressor=compressor,
            compressor_kv=compressor_kv[:, :width].contiguous(),
            compressor_gate=compressor_gate[:, :width].contiguous(),
            cos_table=cos_table,
            sin_table=sin_table,
            position_ids=position_ids[:, :width].contiguous(),
        )
        return out

    out_exact = run(actual_len)
    out_padded = run(padded_len)
    ref = _run_source_placeholder(
        q[:, :actual_len].contiguous(),
        kv[:, :actual_len].contiguous(),
        attn_sink,
        topk_width,
        compress_ratio=compress_ratio,
        compressor=compressor,
        compressor_kv=compressor_kv[:, :actual_len].contiguous(),
        compressor_gate=compressor_gate[:, :actual_len].contiguous(),
        cos_table=cos_table,
        sin_table=sin_table,
        position_ids=position_ids[:, :actual_len].contiguous(),
    )

    assert torch.equal(out_exact, ref), "exact-width cached prefill must match the source op"
    assert torch.equal(out_padded[:, :actual_len], out_exact), (
        "padded initial prefill diverges from exact-width prefill: max abs diff "
        f"{(out_padded[:, :actual_len].float() - out_exact.float()).abs().max().item():.6e}"
    )
    assert torch.equal(out_padded[:, actual_len:], torch.zeros_like(out_padded[:, actual_len:])), (
        "output rows beyond the active token count must be zero-filled"
    )


@_requires_cuda
def test_two_sequence_initial_prefill_matches_solo_ratio4() -> None:
    """Two flattened ratio-4 initial-prefill sequences == each sequence run alone."""
    torch.manual_seed(20260717)
    device = torch.device("cuda")
    compress_ratio = 4
    seq_lens = [12, 8]
    total_len = sum(seq_lens)
    capacity = max(seq_lens)
    compressor, cos_table, sin_table = _make_compressor(compress_ratio, capacity, device)
    topk_width = _WINDOW_SIZE + compressor.max_compressed_len

    hidden = torch.randn(1, total_len, _HIDDEN_SIZE, device=device)
    compressor_kv, compressor_gate = compressor.project(hidden)
    q = torch.randn(1, total_len, _NUM_HEADS, _HEAD_DIM, device=device)
    kv = torch.randn(1, total_len, _HEAD_DIM, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)
    position_ids = (
        torch.cat([torch.arange(n, device=device) for n in seq_lens]).unsqueeze(0).contiguous()
    )
    state_dim = compressor_kv.shape[-1]

    out_batched = _run_cached_placeholder(
        q,
        kv,
        attn_sink,
        _prefill_meta(seq_lens),
        _make_caches(len(seq_lens), capacity, state_dim, 0, device),
        topk_width,
        compress_ratio=compress_ratio,
        compressor=compressor,
        compressor_kv=compressor_kv,
        compressor_gate=compressor_gate,
        cos_table=cos_table,
        sin_table=sin_table,
        position_ids=position_ids,
    )

    start = 0
    for i, n in enumerate(seq_lens):
        sl = slice(start, start + n)
        args = (
            q[:, sl].contiguous(),
            kv[:, sl].contiguous(),
            attn_sink,
        )
        kwargs = dict(
            compress_ratio=compress_ratio,
            compressor=compressor,
            compressor_kv=compressor_kv[:, sl].contiguous(),
            compressor_gate=compressor_gate[:, sl].contiguous(),
            cos_table=cos_table,
            sin_table=sin_table,
            position_ids=position_ids[:, sl].contiguous(),
        )
        out_solo = _run_cached_placeholder(
            *args,
            _prefill_meta([n]),
            _make_caches(1, capacity, state_dim, 0, device),
            topk_width,
            **kwargs,
        )
        ref = _run_source_placeholder(*args, topk_width, **kwargs)
        assert torch.equal(out_solo, ref), f"solo cached prefill of seq {i} must match source op"
        assert torch.equal(out_batched[:, sl], out_solo), (
            f"seq {i} (len {n}) diverges in the two-sequence batch: max abs diff "
            f"{(out_batched[:, sl].float() - out_solo.float()).abs().max().item():.6e}"
        )
        start += n


def test_two_sequence_window_only_prefill_matches_solo_cpu() -> None:
    """Two flattened window-only (ratio-0) initial prefills == solo runs (CPU path)."""
    torch.manual_seed(20260718)
    device = torch.device("cpu")
    seq_lens = [10, 6]
    total_len = sum(seq_lens)
    capacity = max(seq_lens)

    q = torch.randn(1, total_len, _NUM_HEADS, _HEAD_DIM, device=device)
    kv = torch.randn(1, total_len, _HEAD_DIM, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)

    out_batched = _run_cached_placeholder(
        q,
        kv,
        attn_sink,
        _prefill_meta(seq_lens),
        _make_caches(len(seq_lens), capacity, 0, 0, device),
        _WINDOW_SIZE,
    )

    start = 0
    for i, n in enumerate(seq_lens):
        sl = slice(start, start + n)
        args = (q[:, sl].contiguous(), kv[:, sl].contiguous(), attn_sink)
        out_solo = _run_cached_placeholder(
            *args,
            _prefill_meta([n]),
            _make_caches(1, capacity, 0, 0, device),
            _WINDOW_SIZE,
        )
        ref = _run_source_placeholder(*args, _WINDOW_SIZE)
        assert torch.equal(out_solo, ref), f"solo window-only prefill of seq {i} != source op"
        assert torch.equal(out_batched[:, sl], out_solo), (
            f"seq {i} (len {n}) diverges in the two-sequence window-only batch"
        )
        start += n


@_requires_cuda
def test_mixed_prefill_decode_batch_prefill_matches_solo_ratio4() -> None:
    """Mixed [initial prefill, decode] batch matches solo runs (ratio 4 + indexer).

    The prefill half must equal the same sequence run alone, and the decode half
    (which drives the learned indexer) must be invariant to the prefill partner it
    is batched with. ``index_head_dim`` must be a multiple of the hadamard-fp4
    block (32) and ``index_topk >= 2`` (the fused top-k select kernel has no
    single-slot config).
    """
    torch.manual_seed(20260719)
    device = torch.device("cuda")
    compress_ratio = 4
    capacity = 16
    decode_prefill_len = 12  # sequence B: prefilled alone, then decoded in a mixed batch
    partner_lens = [12, 8]  # sequence A variants batched in front of B's decode

    compressor, cos_table, sin_table = _make_compressor(compress_ratio, capacity, device)
    indexer_config = DeepseekV4Config(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=_HEAD_DIM,
        q_lora_rank=8,
        qk_rope_head_dim=_ROPE_DIM,
        index_n_heads=1,
        index_head_dim=32,
        index_topk=2,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity,
        ad_rope_cache_len=capacity,
    )
    indexer = DeepseekV4Indexer(indexer_config, compress_ratio).eval().to(device)
    topk_width = _WINDOW_SIZE + indexer.index_topk

    def project(seq_len: int, seed: int):
        torch.manual_seed(seed)
        hidden = torch.randn(1, seq_len, _HIDDEN_SIZE, device=device)
        q_lora = torch.randn(1, seq_len, indexer_config.q_lora_rank, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).contiguous()
        compressor_kv, compressor_gate = compressor.project(hidden)
        cos = cos_table[position_ids]
        sin = sin_table[position_ids]
        iq, iw, ick, icg = indexer.project(hidden, q_lora, cos, sin)
        q = torch.randn(1, seq_len, _NUM_HEADS, _HEAD_DIM, device=device)
        kv = torch.randn(1, seq_len, _HEAD_DIM, device=device)
        return q, kv, compressor_kv, compressor_gate, iq, iw, ick, icg, position_ids

    attn_sink = torch.tensor([-0.25, 0.1], device=device)
    state_dim = compressor.project(torch.randn(1, 1, _HIDDEN_SIZE, device=device))[0].shape[-1]
    istate_dim = indexer.project(
        torch.randn(1, 1, _HIDDEN_SIZE, device=device),
        torch.randn(1, 1, indexer_config.q_lora_rank, device=device),
        cos_table[:1].unsqueeze(0),
        sin_table[:1].unsqueeze(0),
    )[2].shape[-1]

    # Sequence B: full prefill+decode tensors; prefill B alone into slot 1.
    b = project(decode_prefill_len + 1, 1001)
    caches = _make_caches(2, capacity, state_dim, istate_dim, device)
    _run_cached_placeholder(
        b[0][:, :decode_prefill_len],
        b[1][:, :decode_prefill_len],
        attn_sink,
        _prefill_meta([decode_prefill_len], slot_indices=[1]),
        caches,
        topk_width,
        compress_ratio=compress_ratio,
        compressor=compressor,
        compressor_kv=b[2][:, :decode_prefill_len],
        compressor_gate=b[3][:, :decode_prefill_len],
        cos_table=cos_table,
        sin_table=sin_table,
        position_ids=b[8][:, :decode_prefill_len],
        indexer=indexer,
        indexer_q=b[4][:, :decode_prefill_len],
        indexer_weights=b[5][:, :decode_prefill_len],
        indexer_compressor_kv=b[6][:, :decode_prefill_len],
        indexer_compressor_gate=b[7][:, :decode_prefill_len],
    )

    decode_outputs = []
    for partner_idx, partner_len in enumerate(partner_lens):
        a = project(partner_len, 2001 + partner_idx)
        run_caches = [c.clone() for c in caches]  # identical pre-decode state per partner
        d = slice(decode_prefill_len, decode_prefill_len + 1)

        def mixed_arg(a_t: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
            return torch.cat((a_t, b_t[:, d]), dim=1).contiguous()

        out_mixed = _run_cached_placeholder(
            mixed_arg(a[0], b[0]),
            mixed_arg(a[1], b[1]),
            attn_sink,
            _mixed_meta(partner_len, decode_prefill_len, slot_indices=[0, 1]),
            run_caches,
            topk_width,
            compress_ratio=compress_ratio,
            compressor=compressor,
            compressor_kv=mixed_arg(a[2], b[2]),
            compressor_gate=mixed_arg(a[3], b[3]),
            cos_table=cos_table,
            sin_table=sin_table,
            position_ids=mixed_arg(a[8], b[8]),
            indexer=indexer,
            indexer_q=mixed_arg(a[4], b[4]),
            indexer_weights=mixed_arg(a[5], b[5]),
            indexer_compressor_kv=mixed_arg(a[6], b[6]),
            indexer_compressor_gate=mixed_arg(a[7], b[7]),
        )
        assert torch.isfinite(out_mixed).all()

        out_solo = _run_cached_placeholder(
            a[0],
            a[1],
            attn_sink,
            _prefill_meta([partner_len]),
            _make_caches(1, capacity, state_dim, istate_dim, device),
            topk_width,
            compress_ratio=compress_ratio,
            compressor=compressor,
            compressor_kv=a[2],
            compressor_gate=a[3],
            cos_table=cos_table,
            sin_table=sin_table,
            position_ids=a[8],
            indexer=indexer,
            indexer_q=a[4],
            indexer_weights=a[5],
            indexer_compressor_kv=a[6],
            indexer_compressor_gate=a[7],
        )
        assert torch.equal(out_mixed[:, :partner_len], out_solo), (
            f"prefill (len {partner_len}) diverges when batched with a decode: max abs diff "
            f"{(out_mixed[:, :partner_len].float() - out_solo.float()).abs().max().item():.6e}"
        )
        decode_outputs.append(out_mixed[:, partner_len:])

    assert torch.equal(decode_outputs[0], decode_outputs[1]), (
        "decode output must not depend on the prefill partner it is batched with"
    )
