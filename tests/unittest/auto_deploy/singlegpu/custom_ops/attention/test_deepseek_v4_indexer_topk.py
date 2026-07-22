# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 indexer score / top-k selection tests.

Covers the fused top-k select kernel, the fused fullrange index-score fold, the
placeholder top-k rebuild, and the per-sequence initial-prefill top-k paths of
``deepseek_v4_sparse_attention``.
"""

from __future__ import annotations

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register custom ops
import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _build_placeholder_topk_idxs,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4Compressor,
    DeepseekV4Config,
    DeepseekV4Indexer,
    _window_topk_idxs,
)

_requires_cuda_triton = pytest.mark.skipif(
    not (M._HAS_TRITON and torch.cuda.is_available()), reason="requires CUDA + triton"
)
_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_HIDDEN_SIZE = 16
_HEAD_DIM = 8
_ROPE_DIM = 4
_NUM_HEADS = 2
_WINDOW_SIZE = 4
_SOFTMAX_SCALE = _HEAD_DIM**-0.5


# ---------------------------------------------------------------------------
# Fused decode top-k row selection (_fused_topk_select).
# Contract: byte-exact with the eager topk/isfinite/where/pad tail, tie order
# included (+-0.0 folds to ascending index).
# ---------------------------------------------------------------------------
def _eager_topk_tail(index_score: torch.Tensor, index_topk: int, max_compressed_len: int):
    topk_count = min(index_topk, max_compressed_len)
    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(torch.int64), torch.full_like(topk_rows, -1))
    if topk_count < index_topk:
        pad_shape = (index_score.shape[0], index_topk - topk_count)
        row_pad = torch.full(pad_shape, -1, dtype=torch.int64, device=index_score.device)
        valid_pad = torch.zeros(pad_shape, dtype=torch.bool, device=index_score.device)
        topk_rows = torch.cat((topk_rows, row_pad), dim=-1)
        topk_valid = torch.cat((topk_valid, valid_pad), dim=-1)
    return topk_rows, topk_valid


def _check(index_score: torch.Tensor, index_topk: int, case: str):
    c = int(index_score.shape[1])
    ref_rows, ref_valid = _eager_topk_tail(index_score, index_topk, c)
    got_rows, got_valid = M._fused_topk_select(index_score, index_topk, min(index_topk, c))
    assert got_rows.dtype == torch.int64 and got_valid.dtype == torch.bool, case
    assert got_rows.shape == ref_rows.shape and got_valid.shape == ref_valid.shape, case
    assert torch.equal(got_rows, ref_rows), f"{case}: rows mismatch"
    assert torch.equal(got_valid, ref_valid), f"{case}: valid mismatch"


@_requires_cuda_triton
def test_topk_select_matches_eager_tail():
    torch.manual_seed(0)
    dev = "cuda"

    # Distinct scores, k == C.
    s = torch.randn(1, 512, device=dev)
    _check(s, 512, "distinct k==C")

    # -inf visibility tails.
    for visible in (0, 1, 130, 511):
        s = torch.randn(2, 512, device=dev)
        s[:, visible:] = float("-inf")
        _check(s, 512, f"visible={visible}")

    # Heavy identical-bit ties (quantized) + -inf tail, multi-row.
    for t in range(2):
        s = (torch.randn(4, 512, device=dev) * 2).round() / 2
        s[:, 100 + 90 * t :] = float("-inf")
        _check(s, 512, f"ties trial {t}")

    # Mixed +0.0 / -0.0 among distinct values (torch compares them equal).
    s = torch.randn(3, 512, device=dev) + 3.0
    s[:, ::7] = 0.0
    s[:, 3::11] = -0.0
    s[:, 470:] = float("-inf")
    _check(s, 512, "pm-zero mix")

    # Pad path (index_topk > C) and truncating selection (index_topk < C).
    s = torch.randn(3, 512, device=dev)
    s[:, 200:] = float("-inf")
    _check(s, 640, "pad k>C")
    s = torch.randn(4, 512, device=dev)
    s[2, 40:] = float("-inf")
    s[3, :] = float("-inf")
    _check(s, 128, "truncate k<C distinct")

    # Non-power-of-two candidate count (kernel pads the sort width).
    s = torch.randn(2, 384, device=dev)
    s[:, 300:] = float("-inf")
    _check(s, 512, "C=384 non-pow2 + pad")
    _check(torch.randn(2, 384, device=dev), 384, "C=384 k==C")

    # All-negative scores (negative indexer weights).
    s = -torch.rand(2, 512, device=dev) - 1.0
    s[:, 400:] = float("-inf")
    _check(s, 512, "all-negative scores")


def _single_cta_kernel(index_score: torch.Tensor, index_topk: int):
    """Launch the select kernel on its single-CTA full-sort path (NBANDS=1)."""
    import triton

    num_rows, c = int(index_score.shape[0]), int(index_score.shape[1])
    rows = torch.empty(num_rows, index_topk, dtype=torch.int64, device=index_score.device)
    valid = torch.empty(num_rows, index_topk, dtype=torch.uint8, device=index_score.device)
    block_c = triton.next_power_of_2(c)
    M._dsv4_topk_select_kernel[(num_rows, 1)](
        index_score.contiguous(),
        rows,
        valid,
        rows,
        rows,
        rows,  # input_pos dead pointer (HAS_VLEN=False)
        c,
        index_topk,
        min(index_topk, c),
        1,  # RATIO (unused without HAS_VLEN)
        HAS_VLEN=False,
        BLOCK_C=block_c,
        BLOCK_K=triton.next_power_of_2(index_topk),
        TOPK_BLOCK=block_c,
        NBANDS=1,
        LOG_NBANDS=0,
        LOG_TOPK=block_c.bit_length() - 1,
        num_warps=4,
    )
    return rows, valid.view(torch.bool)


def _check_banded(index_score: torch.Tensor, index_topk: int, case: str):
    got_rows, got_valid = M._fused_topk_select(
        index_score, index_topk, min(index_topk, int(index_score.shape[1]))
    )
    ref_rows, ref_valid = _single_cta_kernel(index_score, index_topk)
    assert torch.equal(got_rows, ref_rows), f"{case}: banded rows != single-CTA rows"
    assert torch.equal(got_valid, ref_valid), f"{case}: banded valid != single-CTA valid"
    # Eager equality is only defined without -0.0: torch's large-C gatherTopK ranks
    # +0.0 above -0.0 while the kernel keeps the ascending-index fold for +-0.0 ties.
    if not ((index_score == 0) & torch.signbit(index_score)).any():
        _check(index_score, index_topk, case)


@_requires_cuda_triton
def test_topk_select_banded_matches_eager_tail():
    torch.manual_seed(2)
    dev = "cuda"

    # Traced decode shape (C=2048/K=512 -> 4 band CTAs), distinct scores.
    _check_banded(torch.randn(1, 2048, device=dev), 512, "banded distinct")
    _check_banded(torch.randn(2, 2048, device=dev), 512, "banded distinct N=2")

    # -inf visibility cutoffs at/near band boundaries (bands of 512).
    for visible in (0, 1, 511, 512, 513, 2048):
        s = torch.randn(2, 2048, device=dev)
        s[:, visible:] = float("-inf")
        _check_banded(s, 512, f"banded visible={visible}")

    # Whole-row constant: every candidate ties, selection == smallest indices.
    _check_banded(torch.zeros(1, 2048, device=dev), 512, "banded all-zero ties")
    _check_banded(torch.full((1, 2048), 1.5, device=dev), 512, "banded all-const ties")

    # Quantized heavy ties (rounding can mint -0.0) + tie clusters across band edges.
    for t in range(2):
        s = (torch.randn(2, 2048, device=dev) * 2).round() / 2
        s[:, 400 + 411 * t :] = float("-inf")
        _check_banded(s, 512, f"banded ties trial {t}")
    s = torch.randn(1, 2048, device=dev)
    s[:, 384:640] = 7.0
    s[:, 1400:1600] = 7.0
    _check_banded(s, 512, "banded cross-band tie blocks")

    # Mixed +-0.0 among distinct values.
    s = torch.randn(1, 2048, device=dev) + 3.0
    s[:, ::5] = 0.0
    s[:, 3::7] = -0.0
    _check_banded(s, 512, "banded pm-zero mix")

    # All-negative scores.
    s = -torch.rand(2, 2048, device=dev) - 1.0
    s[:, 1900:] = float("-inf")
    _check_banded(s, 512, "banded all-negative")

    # Non-power-of-two C: pad lanes live inside the last band(s).
    _check_banded(torch.randn(2, 768, device=dev), 512, "banded C=768 pad in band")
    s = torch.randn(1, 1500, device=dev)
    s[:, 1300:] = float("-inf")
    _check_banded(s, 512, "banded C=1500 pad + -inf tail")

    # Deeper fans: K=128 over C=2048 (16 bands); C=4096 (8 bands).
    _check_banded(torch.randn(1, 2048, device=dev), 128, "banded K=128 16 bands")
    _check_banded((torch.randn(1, 2048, device=dev) * 2).round() / 2, 128, "banded K=128 ties")
    _check_banded(torch.randn(1, 4096, device=dev), 512, "banded C=4096 8 bands")

    # Randomized adversarial sweep over every band fan.
    for trial in range(10):
        n = 1 + trial % 3
        c = (512, 768, 1500, 2048, 4096)[trial % 5]
        k = (512, 128)[trial % 2]
        s = (torch.randn(n, c, device=dev) * (10 ** (trial % 3 - 1))).round() / 4
        cut = int(torch.randint(0, c + 1, (1,)).item())
        s[:, cut:] = float("-inf")
        if trial % 4 == 0:
            s[:, :: 3 + trial % 5] = 0.0
        _check_banded(s, k, f"banded property trial={trial} n={n} c={c} k={k} cut={cut}")


@_requires_cuda_triton
def test_topk_select_banded_cudagraph_replay():
    # Arrival tickets are monotonic (never reset): replays and interleaved eager
    # launches must keep emitting identical bytes.
    torch.manual_seed(3)
    s = torch.randn(1, 2048, device="cuda")
    s[:, 1800:] = float("-inf")
    ref_rows, ref_valid = _eager_topk_tail(s, 512, 2048)

    for _ in range(3):
        rows, valid = M._fused_topk_select(s, 512, 512)
        assert torch.equal(rows, ref_rows) and torch.equal(valid, ref_valid)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rows_g, valid_g = M._fused_topk_select(s, 512, 512)
    for _ in range(7):
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(rows_g, ref_rows) and torch.equal(valid_g, ref_valid)
    rows, valid = M._fused_topk_select(s, 512, 512)
    assert torch.equal(rows, ref_rows) and torch.equal(valid, ref_valid)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(rows_g, ref_rows) and torch.equal(valid_g, ref_valid)

    # Capture-time fallback: a shape whose ticket buffer was never warmed up
    # eagerly must route capture to the single-CTA sort and stay exact.
    key = (s.device, 1, 2)
    M._TOPK_SELECT_TICKETS.pop(key, None)
    s2 = torch.randn(1, 1024, device="cuda")
    ref2_rows, ref2_valid = _eager_topk_tail(s2, 512, 1024)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        rows2_g, valid2_g = M._fused_topk_select(s2, 512, 512)
    assert key not in M._TOPK_SELECT_TICKETS
    for _ in range(3):
        g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(rows2_g, ref2_rows) and torch.equal(valid2_g, ref2_valid)


def _masked_scores(n: int, c: int, vlens, dev: str = "cuda", ties: bool = False):
    s = torch.randn(n, c, device=dev)
    if ties:
        s = (s * 2).round() / 2
    input_pos = []
    for row, v in enumerate(vlens):
        s[row, v:] = float("-inf")
        # vlen == min((input_pos + 1) // 4, C): input_pos = 4*vlen - 1 hits it exactly.
        input_pos.append(4 * v - 1 if v > 0 else 0)
    return s, torch.tensor(input_pos, dtype=torch.long, device=dev)


def _check_vlen(index_score, input_pos, index_topk: int, case: str):
    c = int(index_score.shape[1])
    got_rows, got_valid = M._fused_topk_select(
        index_score, index_topk, min(index_topk, c), input_pos, 4
    )
    ref_rows, ref_valid = M._fused_topk_select(index_score, index_topk, min(index_topk, c))
    assert torch.equal(got_rows, ref_rows), f"{case}: vlen rows != no-hint rows"
    assert torch.equal(got_valid, ref_valid), f"{case}: vlen valid != no-hint valid"
    if not ((index_score == 0) & torch.signbit(index_score)).any():
        eager_rows, eager_valid = _eager_topk_tail(index_score, index_topk, c)
        assert torch.equal(got_rows, eager_rows), f"{case}: vlen rows != eager rows"
        assert torch.equal(got_valid, eager_valid), f"{case}: vlen valid != eager valid"


@_requires_cuda_triton
def test_topk_select_visible_prefix_fast_path():
    # input_pos/compress_ratio hint: band 0 emits its own sort while other band CTAs
    # retire whenever vlen <= TOPK_BLOCK (the whole production decode window).
    torch.manual_seed(7)

    # C=1024/K=512 (2 bands): every band-edge transition.
    for vlen in (0, 1, 250, 511, 512):
        s, ip = _masked_scores(2, 1024, [vlen, vlen])
        _check_vlen(s, ip, 512, f"fast vlen={vlen}")
    for vlen in (513, 1024):  # slow path with the hint compiled in
        s, ip = _masked_scores(2, 1024, [vlen, vlen])
        _check_vlen(s, ip, 512, f"slow vlen={vlen}")

    # Heavy ties around the boundary (quantized scores can mint -0.0).
    for vlen in (511, 512, 513):
        s, ip = _masked_scores(2, 1024, [vlen, vlen], ties=True)
        _check_vlen(s, ip, 512, f"ties vlen={vlen}")

    # Mixed fast/slow rows in one launch (per-row branch + ticket parity).
    s, ip = _masked_scores(4, 1024, [250, 513, 0, 1024])
    _check_vlen(s, ip, 512, "mixed fast/slow rows")

    # Deeper band fan: C=2048 -> 4 bands, K=512.
    for vlen in (0, 513, 2048):
        s, ip = _masked_scores(1, 2048, [vlen])
        _check_vlen(s, ip, 512, f"4-band vlen={vlen}")

    # K=128 over C=2048 (TOPK_BLOCK=128, 16 bands): boundary at 128.
    for vlen in (128, 129, 700):
        s, ip = _masked_scores(1, 2048, [vlen])
        _check_vlen(s, ip, 128, f"16-band vlen={vlen}")

    # Short history + pad tail (index_topk > C).
    s, ip = _masked_scores(2, 384, [40, 384])
    _check_vlen(s, ip, 512, "pad k>C with vlen")


@_requires_cuda_triton
def test_topk_select_visible_prefix_ticket_parity_and_replay():
    # Fast-path launches leave the arrival tickets untouched; interleaving fast and
    # slow launches on one ticket buffer must keep the parity arithmetic aligned,
    # eagerly and across replays whose input_pos changes between replays.
    torch.manual_seed(11)
    s_fast, ip_fast = _masked_scores(1, 1024, [250])
    s_slow, ip_slow = _masked_scores(1, 1024, [800])
    ref_fast = M._fused_topk_select(s_fast, 512, 512)
    ref_slow = M._fused_topk_select(s_slow, 512, 512)

    for _ in range(3):
        rf = M._fused_topk_select(s_fast, 512, 512, ip_fast, 4)
        rs = M._fused_topk_select(s_slow, 512, 512, ip_slow, 4)
        assert torch.equal(rf[0], ref_fast[0]) and torch.equal(rf[1], ref_fast[1])
        assert torch.equal(rs[0], ref_slow[0]) and torch.equal(rs[1], ref_slow[1])

    s_buf = s_fast.clone()
    ip_buf = ip_fast.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rows_g, valid_g = M._fused_topk_select(s_buf, 512, 512, ip_buf, 4)
    for src, ref in ((s_slow, ref_slow), (s_fast, ref_fast), (s_slow, ref_slow)):
        s_buf.copy_(src)
        ip_buf.copy_(ip_slow if ref is ref_slow else ip_fast)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(rows_g, ref[0]) and torch.equal(valid_g, ref[1])


# ---------------------------------------------------------------------------
# Placeholder top-k rebuild (_build_placeholder_topk_idxs + topk_is_placeholder).
# ---------------------------------------------------------------------------
def _ref_placeholder_topk_idxs(
    window_size: int,
    compress_ratio: int,
    batch_size: int,
    seq_len: int,
    compressed_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-optimization model-side window + compressed placeholder chain."""
    window_idxs = _window_topk_idxs(window_size, batch_size, seq_len, device)
    compressed_positions = torch.arange(compressed_width, device=device)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // compress_ratio
    compressed_idxs = compressed_positions.unsqueeze(0).expand(seq_len, -1)
    compressed_idxs = torch.where(compressed_idxs < valid_lengths, compressed_idxs + seq_len, -1)
    compressed_idxs = compressed_idxs.unsqueeze(0).expand(batch_size, -1, -1)
    return torch.cat((window_idxs, compressed_idxs), dim=-1).to(torch.int64)


@pytest.mark.parametrize(
    "compress_ratio,compressed_width,batch_size,seq_len,window_size",
    [(4, 6, 1, 12, 4), (4, 2, 2, 9, 3), (128, 3, 1, 1, 4)],
)
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
    # The cached op recovers index_topk as topk_idxs.shape[-1] - window_size.
    assert tuple(got.shape) == (batch_size, seq_len, window_size + compressed_width)
    assert torch.equal(got, ref)


def _linspace_rope_tables(max_seq_len: int, rope_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    freqs = torch.linspace(0.05, 0.25, rope_dim // 2, dtype=torch.float32).unsqueeze(0)
    angles = positions * freqs
    return angles.cos(), angles.sin()


def _compressor_case(compress_ratio: int, seq_len: int, device: torch.device, batch_size: int = 1):
    capacity = seq_len
    table_len = max(capacity, seq_len, 1)
    config = DeepseekV4Config(
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=_HEAD_DIM,
        qk_rope_head_dim=_ROPE_DIM,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity,
        ad_rope_cache_len=table_len,
    )
    compressor = DeepseekV4Compressor(config, compress_ratio, _HEAD_DIM).eval().to(device)
    hidden_states = torch.randn(batch_size, seq_len, _HIDDEN_SIZE, device=device)
    compressor_kv, compressor_gate = compressor.project(hidden_states)
    cos_table, sin_table = (t.to(device) for t in _linspace_rope_tables(table_len, _ROPE_DIM))
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.contiguous()
    return compressor, compressor_kv, compressor_gate, cos_table, sin_table, position_ids


@_requires_cuda
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_no_cache_op_placeholder_matches_explicit(compress_ratio: int) -> None:
    # fp32 + head_dim < 16 keeps the attend on the deterministic reference chunk
    # loop, so the two invocations are exactly comparable with torch.equal.
    torch.manual_seed(20260702 + compress_ratio)
    device = torch.device("cuda")
    batch_size = 1
    seq_len = 2 * compress_ratio

    (
        compressor,
        compressor_kv,
        compressor_gate,
        cos_table,
        sin_table,
        position_ids,
    ) = _compressor_case(compress_ratio, seq_len, device, batch_size)

    q = torch.randn(batch_size, seq_len, _NUM_HEADS, _HEAD_DIM, device=device)
    kv = torch.randn(batch_size, seq_len, _HEAD_DIM, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)

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
            _SOFTMAX_SCALE,
            window_size=_WINDOW_SIZE,
            compress_ratio=compress_ratio,
            max_compressed_len=compressed_width,
            rope_dim=compressor.rope_head_dim,
            rms_norm_eps=compressor.norm.eps,
            topk_is_placeholder=topk_is_placeholder,
        )

    explicit_topk = _build_placeholder_topk_idxs(
        _WINDOW_SIZE, compress_ratio, batch_size, seq_len, compressed_width, q.device
    )
    out_explicit = _run(explicit_topk, topk_is_placeholder=False)

    # Width-only allocation: values never read because the op rebuilds the selection.
    width_only_topk = q.new_empty(
        batch_size, seq_len, _WINDOW_SIZE + compressed_width, dtype=torch.int64
    )
    out_placeholder = _run(width_only_topk, topk_is_placeholder=True)

    assert torch.equal(out_explicit, out_placeholder)


# ---------------------------------------------------------------------------
# Fused fullrange index-score fold (_fused_fullrange_index_score).
# Reference = the retained eager chain of _select_decode_ratio4_indexer_rows.
# ---------------------------------------------------------------------------
def _rope_tables(num_pos: int, rope_dim: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    angles = torch.outer(torch.arange(num_pos, dtype=torch.float32), freqs)
    return angles.cos().to(device), angles.sin().to(device)


def _build_fixture(
    *,
    num_rows: int,
    head_dim: int,
    rope_dim: int,
    m: int,
    tokens_per_block: int,
    input_pos: list,
    pos_delta: int = 0,
    short_pages_row: int = -1,
    constant_caches: bool = False,
    poison_beyond_visible: bool = False,
    device: str = "cuda",
    seed: int = 0,
):
    torch.manual_seed(seed)
    ratio = 4
    state_dim = 2 * head_dim
    pos_count = m * ratio
    full_pages = (pos_count + tokens_per_block - 1) // tokens_per_block
    page_counts = [full_pages - 1 if i == short_pages_row else full_pages for i in range(num_rows)]
    cu_num_pages = torch.tensor(
        [0, *torch.tensor(page_counts).cumsum(0).tolist()], dtype=torch.long, device=device
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=device)

    if constant_caches:
        kv_cache = torch.full((total_pages, tokens_per_block, state_dim), 0.5, device=device)
        gate_cache = torch.full((total_pages, tokens_per_block, state_dim), -0.25, device=device)
    else:
        kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)
        gate_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=device)

    input_pos_t = torch.tensor(input_pos, dtype=torch.long, device=device)
    if poison_beyond_visible:
        for b in range(num_rows):
            start_page = int(cu_num_pages[b].item())
            n_pages = int(cu_num_pages[b + 1].item()) - start_page
            written = int(input_pos_t[b].item()) + 1
            for p in range(n_pages):
                page = int(cache_loc[start_page + p].item())
                lo = p * tokens_per_block
                for t in range(tokens_per_block):
                    if lo + t >= written:
                        kv_cache[page, t] = float("inf")
                        gate_cache[page, t] = float("nan")

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=device)
    position_ids = input_pos_t + pos_delta
    ape = torch.randn(ratio, state_dim, device=device)
    norm_weight = torch.randn(head_dim, device=device)
    cos_table, sin_table = _rope_tables(pos_count + 8, rope_dim, device)

    full_positions = torch.arange(pos_count, dtype=torch.long, device=device)
    full_positions = full_positions.view(1, -1).expand(num_rows, -1)
    full_page_map = M._decode_page_ids_and_offsets(
        kv_cache, seq_idx, full_positions, cu_num_pages, cache_loc
    )
    return dict(
        kv_cache=kv_cache,
        gate_cache=gate_cache,
        seq_idx=seq_idx,
        input_pos=input_pos_t,
        position_ids=position_ids,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
        ape=ape,
        norm_weight=norm_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        full_page_map=full_page_map,
        m=m,
        ratio=ratio,
        head_dim=head_dim,
        rope_dim=rope_dim,
        eps=1e-6,
    )


def _eager_score(fx, q_index, indexer_weights, w_scale: float = 1.0):
    m, ratio = fx["m"], fx["ratio"]
    num_rows = int(fx["seq_idx"].shape[0])
    candidate_rows = torch.arange(m, dtype=torch.long, device=q_index.device)
    candidate_rows = candidate_rows.view(1, -1).expand(num_rows, -1)
    row_position_id = fx["position_ids"].unsqueeze(1) - (
        fx["input_pos"].unsqueeze(1) - candidate_rows * ratio
    )
    index_k = M._batched_overlap_compressed_rows_fullrange(
        fx["kv_cache"],
        fx["gate_cache"],
        fx["seq_idx"],
        row_position_id,
        fx["cu_num_pages"],
        fx["cache_loc"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["eps"],
        fx["rope_dim"],
        ratio,
        fx["head_dim"],
        m,
        q_index.dtype,
        rotate=True,
        full_page_map=fx["full_page_map"],
    )
    visible_len = ((fx["input_pos"] + 1) // ratio).clamp(max=m)
    score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
    score = (score.relu() * (indexer_weights.float() * w_scale).unsqueeze(-1)).sum(dim=1)
    visible = candidate_rows < visible_len.unsqueeze(1)
    return score.masked_fill(~visible, float("-inf"))


def _fused_score(fx, q_index, indexer_weights, w_scale: float = 1.0):
    return M._fused_fullrange_index_score(
        fx["kv_cache"],
        fx["gate_cache"],
        fx["full_page_map"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["input_pos"],
        fx["position_ids"],
        q_index,
        indexer_weights,
        fx["eps"],
        fx["ratio"],
        fx["head_dim"],
        fx["rope_dim"],
        fx["m"],
        w_scale=w_scale,
    )


# (name, fixture kwargs, num_heads, exact). exact=False = the documented
# kernel-vs-eager tolerance class (M>=512 fp32 FMA context / H != 16 head-reduction
# order): <= 4e-6 abs score tail, selection identical either way.
CASES = [
    (
        "prod_shape_mixed_history",
        dict(
            num_rows=3,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 3, 0],
            short_pages_row=0,
            seed=11,
        ),
        16,
        True,
    ),
    (
        "empty_history",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=6,
            tokens_per_block=8,
            input_pos=[0, 2],
            seed=13,
        ),
        16,
        True,
    ),
    (
        "heavy_ties",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 31],
            constant_caches=True,
            seed=19,
        ),
        16,
        True,
    ),
    (
        "shifted_rope_positions",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[27, 15],
            pos_delta=3,
            seed=23,
        ),
        16,
        True,
    ),
    (
        "poisoned_tail",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[13, 5],
            poison_beyond_visible=True,
            seed=29,
        ),
        16,
        True,
    ),
    (
        "nonpow2_heads_h24",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 9],
            seed=31,
        ),
        24,
        False,
    ),
    (
        "full_heads_h64",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=8,
            tokens_per_block=8,
            input_pos=[31, 9],
            seed=37,
        ),
        64,
        False,
    ),
    (
        "prod_scale_m512",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=512,
            tokens_per_block=64,
            input_pos=[1001, 998],
            seed=41,
        ),
        16,
        False,
    ),
    (
        # M=1024: one row deep in the early-exit regime (vlen 250), one fully
        # visible (vlen == M), pinning the visible-prefix split.
        "prod_scale_m1024_visible_split",
        dict(
            num_rows=2,
            head_dim=128,
            rope_dim=64,
            m=1024,
            tokens_per_block=64,
            input_pos=[1001, 4095],
            seed=59,
        ),
        16,
        False,
    ),
]


@_requires_cuda_triton
@pytest.mark.parametrize("name,kwargs,num_heads,exact", CASES, ids=[c[0] for c in CASES])
def test_fused_index_score_matches_two_kernel_chain(name, kwargs, num_heads, exact):
    dtype = torch.bfloat16
    fx = _build_fixture(**kwargs)
    torch.manual_seed(2000 + num_heads)
    num_rows, head_dim = fx["seq_idx"].shape[0], fx["head_dim"]
    q_index = torch.randn(num_rows, num_heads, head_dim, device="cuda", dtype=dtype)
    indexer_weights = torch.randn(num_rows, num_heads, device="cuda", dtype=dtype)

    score_ref = _eager_score(fx, q_index, indexer_weights)
    score_fused = _fused_score(fx, q_index, indexer_weights)
    assert score_fused.shape == score_ref.shape and score_fused.dtype == score_ref.dtype
    if exact:
        assert torch.equal(score_fused, score_ref), f"[{name}] fused index score diverged"
    else:
        finite = torch.isfinite(score_ref)
        assert torch.equal(torch.isfinite(score_fused), finite), f"[{name}] visibility diverged"
        f = score_fused[finite]
        r = score_ref[finite]
        if f.numel() >= 64:
            exact_frac = (f == r).float().mean().item()
            assert exact_frac >= 0.9, f"[{name}] exact-match fraction {exact_frac} too low"
        max_abs = (f - r).abs().max().item()
        assert max_abs <= 4e-6, f"[{name}] score tail moved by {max_abs} > 4e-6"
        torch.testing.assert_close(f, r, rtol=1e-5, atol=1e-4)

    # The selection consuming these scores must be identical: ids, tie order,
    # validity, across narrow and select-all top-k widths.
    for index_topk in {min(fx["m"], 8), fx["m"]}:
        rows_ref, valid_ref = M._fused_topk_select(score_ref, index_topk, min(index_topk, fx["m"]))
        rows_fused, valid_fused = M._fused_topk_select(
            score_fused, index_topk, min(index_topk, fx["m"])
        )
        assert torch.equal(rows_fused, rows_ref), f"[{name}] top-k ids/order diverged"
        assert torch.equal(valid_fused, valid_ref), f"[{name}] top-k validity diverged"


@_requires_cuda_triton
def test_fused_index_score_fp32_weights():
    # fp32 weights shift the fp32 head-reduction rounding pattern: tolerance class.
    fx = _build_fixture(
        num_rows=2,
        head_dim=128,
        rope_dim=64,
        m=8,
        tokens_per_block=8,
        input_pos=[31, 9],
        seed=43,
    )
    torch.manual_seed(4300)
    q_index = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    indexer_weights = torch.randn(2, 16, device="cuda", dtype=torch.float32)
    score_ref = _eager_score(fx, q_index, indexer_weights)
    score_fused = _fused_score(fx, q_index, indexer_weights)
    finite = torch.isfinite(score_ref)
    assert torch.equal(torch.isfinite(score_fused), finite)
    max_abs = (score_fused[finite] - score_ref[finite]).abs().max().item()
    assert max_abs <= 4e-6, f"score tail moved by {max_abs} > 4e-6"
    rows_ref, valid_ref = M._fused_topk_select(score_ref, fx["m"], fx["m"])
    rows_fused, valid_fused = M._fused_topk_select(score_fused, fx["m"], fx["m"])
    assert torch.equal(rows_fused, rows_ref)
    assert torch.equal(valid_fused, valid_ref)


@_requires_cuda_triton
@pytest.mark.parametrize("m,input_pos", [(8, [31, 9]), (512, [1001, 998])])
def test_raw_weight_scale_fold_matches_prescaled(m, input_pos):
    # In-kernel w_scale fold == host pre-scale: identical fp32 multiply on the
    # identical widened value, so bit-equal in BOTH the fused fold and eager chain.
    fx = _build_fixture(
        num_rows=2,
        head_dim=128,
        rope_dim=64,
        m=m,
        tokens_per_block=8 if m == 8 else 64,
        input_pos=input_pos,
        seed=53,
    )
    torch.manual_seed(5300)
    num_heads = 16
    q_index = torch.randn(2, num_heads, 128, device="cuda", dtype=torch.bfloat16)
    raw_weights = torch.randn(2, num_heads, device="cuda", dtype=torch.bfloat16)
    w_scale = fx["head_dim"] ** -0.5 * num_heads**-0.5
    prescaled = raw_weights.float() * w_scale

    fused_folded = _fused_score(fx, q_index, raw_weights, w_scale=w_scale)
    fused_prescaled = _fused_score(fx, q_index, prescaled)
    assert torch.equal(fused_folded, fused_prescaled), "fused w_scale fold diverged"

    eager_folded = _eager_score(fx, q_index, raw_weights, w_scale=w_scale)
    eager_prescaled = _eager_score(fx, q_index, prescaled)
    assert torch.equal(eager_folded, eager_prescaled), "eager w_scale fold diverged"


@_requires_cuda_triton
def test_selection_path_routes_through_fold_and_matches():
    # _select_decode_ratio4_indexer_rows routes H > 8 through the fold and folds the
    # indexer pre-scale (head_dim**-0.5 * num_heads**-0.5) into the kernel.
    fx = _build_fixture(
        num_rows=3,
        head_dim=128,
        rope_dim=64,
        m=8,
        tokens_per_block=8,
        input_pos=[31, 3, 0],
        short_pages_row=0,
        seed=47,
    )
    torch.manual_seed(4700)
    q_index = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    indexer_weights = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    index_topk = 8

    rows, valid = M._select_decode_ratio4_indexer_rows(
        q_index,
        indexer_weights,
        fx["kv_cache"],
        fx["gate_cache"],
        fx["seq_idx"],
        fx["input_pos"],
        fx["position_ids"],
        index_topk,
        fx["cu_num_pages"],
        fx["cache_loc"],
        fx["ape"],
        fx["norm_weight"],
        fx["cos_table"],
        fx["sin_table"],
        fx["eps"],
        fx["rope_dim"],
        fx["m"],
        full_page_map=fx["full_page_map"],
    )
    w_scale = fx["head_dim"] ** -0.5 * int(q_index.shape[1]) ** -0.5
    score_ref = _eager_score(fx, q_index, indexer_weights, w_scale=w_scale)
    rows_ref, valid_ref = M._fused_topk_select(score_ref, index_topk, min(index_topk, fx["m"]))
    assert torch.equal(rows, rows_ref)
    assert torch.equal(valid, valid_ref)


# ---------------------------------------------------------------------------
# Initial-prefill placeholder top-k: rebuilt PER SEQUENCE (padded widths,
# multi-sequence batches, and mixed prefill+decode must match solo runs).
# ---------------------------------------------------------------------------
def _base_meta(
    seq_lens: list[int],
    input_positions: list[int],
    slot_indices: list[int],
    num_prefill: int,
) -> tuple[torch.Tensor, ...]:
    # (batch_info, seq_len, input_pos, slot_idx, cu_seqlen, cu_num_pages, cache_loc);
    # one whole-sequence cache page per slot (tokens_per_block == cache.shape[1]).
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
    # The hoisted metadata tensors, produced the way production does; prefill/mixed
    # forwards never read their values but the fixed output contract must hold.
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
    cos_table, sin_table = (t.to(device) for t in _linspace_rope_tables(capacity_tokens, _ROPE_DIM))
    return compressor, cos_table, sin_table


def _make_caches(
    num_slots: int,
    cache_tokens: int,
    state_dim: int,
    indexer_state_dim: int,
    device: torch.device,
) -> list[torch.Tensor]:
    # [swa, mhc, compressor_kv, compressor_gate, indexer_kv, indexer_gate] caches.
    return [
        torch.full((num_slots, cache_tokens, _HEAD_DIM), 777.0, device=device),
        torch.full((num_slots, cache_tokens, _HEAD_DIM), 777.0, device=device),
        torch.full((num_slots, cache_tokens, state_dim), 777.0, device=device),
        torch.full((num_slots, cache_tokens, state_dim), 777.0, device=device),
        torch.full((num_slots, cache_tokens, indexer_state_dim), 777.0, device=device),
        torch.full((num_slots, cache_tokens, indexer_state_dim), 777.0, device=device),
    ]


def _placeholder_topk(width: int, num_tokens: int, device: torch.device) -> torch.Tensor:
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
    topk_idxs: torch.Tensor | None = None,
    topk_is_placeholder: bool = True,
) -> torch.Tensor:
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
        _placeholder_topk(topk_width, num_tokens, q.device) if topk_idxs is None else topk_idxs,
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
        topk_is_placeholder=topk_is_placeholder,
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
    # Padded width (cudagraph bucket) == exact width == source op; the exact-width
    # control is the fixed point where old global frame and per-sequence frame agree.
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
        return _run_cached_placeholder(
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


@_requires_cuda
def test_long_ratio4_initial_prefill_preserves_learned_topk() -> None:
    # The first token with more visible rows than index_topk must use the learned
    # indexer selection, not the old dense-prefix rebuild.
    torch.manual_seed(20260720)
    device = torch.device("cuda")
    compress_ratio = 4
    seq_len = 12
    capacity = 16
    index_topk = 2

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
        index_topk=index_topk,
        compress_ratios=(compress_ratio,),
        ad_compress_max_seq_len=capacity,
        ad_rope_cache_len=capacity,
    )
    indexer = DeepseekV4Indexer(indexer_config, compress_ratio).eval().to(device)

    hidden = torch.randn(1, seq_len, _HIDDEN_SIZE, device=device)
    q_lora = torch.randn(1, seq_len, indexer_config.q_lora_rank, device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).contiguous()
    compressor_kv, compressor_gate = compressor.project(hidden)
    cos = cos_table[position_ids]
    sin = sin_table[position_ids]
    indexer_q, indexer_weights, indexer_compressor_kv, indexer_compressor_gate = indexer.project(
        hidden, q_lora, cos, sin
    )
    index_k = indexer.compressor.compress_projected(
        indexer_compressor_kv.float(),
        indexer_compressor_gate.float(),
        cos_table,
        sin_table,
        position_ids,
        hidden.dtype,
    )
    # Make the overflow token prefer the newly visible row deterministically.
    indexer_q = indexer_q.clone()
    indexer_q[:, -1] = index_k[:, index_topk].unsqueeze(1)
    indexer_weights = torch.ones_like(indexer_weights)
    fixture_scores = torch.matmul(
        indexer_q[:, -1], index_k[:, : index_topk + 1].transpose(-1, -2)
    ).relu()
    newest_margin = fixture_scores[0, 0, index_topk] - fixture_scores[0, 0, :index_topk].min()
    assert newest_margin.item() > 1.0, (
        f"newest-row score margin too small for a stable fixture: {newest_margin.item():.6f}"
    )
    compressed_topk = indexer.select_topk(indexer_q, index_k, indexer_weights, seq_len, seq_len)
    assert (compressed_topk[0, -1] == seq_len + index_topk).any(), (
        "fixture must select the newest compressed row so the old dense-prefix rebuild fails"
    )
    query_positions = torch.arange(seq_len, device=device).unsqueeze(1)
    window_positions = (
        query_positions - _WINDOW_SIZE + 1 + torch.arange(_WINDOW_SIZE, device=device)
    )
    window_positions = torch.where(
        (window_positions < 0) | (window_positions > query_positions),
        -1,
        window_positions,
    ).unsqueeze(0)
    explicit_topk = torch.cat((window_positions, compressed_topk), dim=-1).to(torch.int64)
    compressed_positions = torch.arange(index_topk, device=device)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // compress_ratio
    dense_compressed_topk = compressed_positions.unsqueeze(0).expand(seq_len, -1)
    dense_compressed_topk = torch.where(
        dense_compressed_topk < valid_lengths,
        dense_compressed_topk + seq_len,
        -1,
    ).unsqueeze(0)
    dense_topk = torch.cat((window_positions, dense_compressed_topk), dim=-1).to(torch.int64)

    q = torch.randn(1, seq_len, _NUM_HEADS, _HEAD_DIM, device=device)
    kv = torch.randn(1, seq_len, _HEAD_DIM, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)
    state_dim = int(compressor_kv.shape[-1])
    indexer_state_dim = int(indexer_compressor_kv.shape[-1])

    def run(topk: torch.Tensor | None, is_placeholder: bool) -> torch.Tensor:
        return _run_cached_placeholder(
            q,
            kv,
            attn_sink,
            _prefill_meta([seq_len]),
            _make_caches(1, capacity, state_dim, indexer_state_dim, device),
            _WINDOW_SIZE + index_topk,
            compress_ratio=compress_ratio,
            compressor=compressor,
            compressor_kv=compressor_kv,
            compressor_gate=compressor_gate,
            cos_table=cos_table,
            sin_table=sin_table,
            position_ids=position_ids,
            indexer=indexer,
            indexer_q=indexer_q,
            indexer_weights=indexer_weights,
            indexer_compressor_kv=indexer_compressor_kv,
            indexer_compressor_gate=indexer_compressor_gate,
            topk_idxs=topk,
            topk_is_placeholder=is_placeholder,
        )

    learned = run(explicit_topk, False)
    dense = run(dense_topk, False)
    rebuilt = run(None, True)
    tail_start = (index_topk + 1) * compress_ratio - 1
    assert torch.equal(rebuilt[:, :tail_start], dense[:, :tail_start])
    assert not torch.equal(dense[:, -1], learned[:, -1]), (
        "fixture must distinguish learned selection from the old dense-prefix output"
    )
    assert torch.equal(rebuilt[:, -1], learned[:, -1])


def test_two_sequence_window_only_prefill_matches_solo_cpu() -> None:
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
    # Mixed [initial prefill, decode] batch: the prefill half must equal the same
    # sequence run alone, and the decode half must be invariant to its prefill
    # partner. index_head_dim must be a multiple of the hadamard-fp4 block (32)
    # and index_topk >= 2 (the fused top-k select kernel has no single-slot config).
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
