# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused exact decode top-k row selection (idea_0046).

The decode ratio-4 selection tail in ``_select_decode_ratio4_indexer_rows`` runs

    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(int64), -1)
    # + -1/False pad up to index_topk when topk_count < index_topk

which costs a fat ``gatherTopK`` + ``radixSortKVInPlace`` pair plus the decomposed
``isfinite``/``where``/pad swarm per ratio-4 layer per decode step. idea_0046 replaces
it with one Triton kernel (``_dsv4_topk_select_kernel`` via ``_fused_topk_select``)
that sorts a packed (float-flip score, index) int64 key per candidate and emits the
padded ``topk_rows`` / ``topk_valid`` directly.

The contract is BYTE-exact equality with the eager tail on the same score tensor,
tie order included: torch's top-k at these shapes orders ties (equal values,
``+0.0`` == ``-0.0`` included) by ascending index, which the packed key reproduces
(``-0.0`` is canonicalized to ``+0.0`` before the bitcast).  This is pinned across
distinct scores, ``-inf`` visibility tails (partial / single-row / all-invalid
history), heavy quantized ties, mixed ``+-0.0``, non-power-of-two candidate counts,
pad (``index_topk > C``) and truncating (``index_topk < C``, distinct scores)
selections, and multi-row batches.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M


def _supported() -> bool:
    return torch.cuda.is_available() and M._HAS_TRITON


def _eager_topk_tail(index_score: torch.Tensor, index_topk: int, max_compressed_len: int):
    """The exact eager chain replaced by ``_fused_topk_select``."""
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


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_matches_eager_tail():
    torch.manual_seed(0)
    dev = "cuda"

    # Proxy shape: distinct scores, k == C.
    s = torch.randn(1, 512, device=dev)
    _check(s, 512, "distinct k==C")

    # -inf visibility tails: partial, single visible row, all-invalid history.
    for visible in (0, 1, 3, 17, 130, 511):
        s = torch.randn(2, 512, device=dev)
        s[:, visible:] = float("-inf")
        _check(s, 512, f"visible={visible}")

    # Heavy identical-bit ties (quantized) + -inf tail, multi-row.
    for t in range(4):
        s = (torch.randn(4, 512, device=dev) * 2).round() / 2
        s[:, 100 + 90 * t :] = float("-inf")
        _check(s, 512, f"ties trial {t}")

    # Mixed +0.0 / -0.0 among distinct values (torch compares them equal).
    s = torch.randn(3, 512, device=dev) + 3.0
    s[:, ::7] = 0.0
    s[:, 3::11] = -0.0
    s[:, 470:] = float("-inf")
    _check(s, 512, "pm-zero mix")

    # Pad path: index_topk > C fills the tail with -1/False.
    s = torch.randn(3, 512, device=dev)
    s[:, 200:] = float("-inf")
    _check(s, 640, "pad k>C")

    # Truncating selection (index_topk < C) on distinct scores == sort prefix.
    s = torch.randn(4, 512, device=dev)
    s[2, 40:] = float("-inf")
    s[3, :] = float("-inf")
    _check(s, 128, "truncate k<C distinct")

    # Non-power-of-two candidate count (kernel pads the sort width).
    s = torch.randn(2, 384, device=dev)
    s[:, 300:] = float("-inf")
    _check(s, 512, "C=384 non-pow2 + pad")
    _check(torch.randn(2, 384, device=dev), 384, "C=384 k==C")

    # Negative scores (negative indexer weights make the whole row negative).
    s = -torch.rand(2, 512, device=dev) - 1.0
    s[:, 400:] = float("-inf")
    _check(s, 512, "all-negative scores")


def _single_cta_kernel(index_score: torch.Tensor, index_topk: int):
    """Launch the select kernel on its single-CTA full-sort path (NBANDS=1).

    This is idea_0046's original layout -- the semantic reference the banded
    multi-CTA layout (idea_0087) must reproduce bit-for-bit on every input.
    """
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
    """Banded output must equal the single-CTA kernel bit-for-bit, always.

    It must also equal the eager torch tail whenever the input has no ``-0.0``:
    torch's large-C ``gatherTopK`` path ranks ``+0.0`` strictly above ``-0.0``
    while the fused kernel keeps idea_0046's documented ascending-index fold
    for +-0.0 ties, so eager equality is only defined without ``-0.0``.
    """
    got_rows, got_valid = M._fused_topk_select(
        index_score, index_topk, min(index_topk, int(index_score.shape[1]))
    )
    ref_rows, ref_valid = _single_cta_kernel(index_score, index_topk)
    assert torch.equal(got_rows, ref_rows), f"{case}: banded rows != single-CTA rows"
    assert torch.equal(got_valid, ref_valid), f"{case}: banded valid != single-CTA valid"
    if not ((index_score == 0) & torch.signbit(index_score)).any():
        _check(index_score, index_topk, case)


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_banded_matches_eager_tail():
    """The banded multi-CTA path (C > topk, idea_0087) stays byte-exact.

    C=2048 -> K=512 is the traced decode shape (4 band CTAs); the adversarial
    patterns pin tie-break order *across* band boundaries (constant rows,
    band-edge tie clusters), ``-inf`` visibility cutoffs at band edges, pad
    lanes inside bands (non-power-of-two C), deeper band fans (K=128 -> 16
    bands; C=4096 -> 8 bands), and multi-row grids.
    """
    torch.manual_seed(2)
    dev = "cuda"

    # Traced decode shape, distinct scores.
    _check_banded(torch.randn(1, 2048, device=dev), 512, "banded distinct")
    _check_banded(torch.randn(2, 2048, device=dev), 512, "banded distinct N=2")

    # -inf visibility cutoffs at/near band boundaries (bands of 512).
    for visible in (0, 1, 511, 512, 513, 1024, 1536, 2047, 2048):
        s = torch.randn(2, 2048, device=dev)
        s[:, visible:] = float("-inf")
        _check_banded(s, 512, f"banded visible={visible}")

    # Whole-row constant: every candidate ties, selection == smallest indices.
    _check_banded(torch.zeros(1, 2048, device=dev), 512, "banded all-zero ties")
    _check_banded(torch.full((1, 2048), 1.5, device=dev), 512, "banded all-const ties")

    # Tie clusters straddling band boundaries, quantized heavy ties (the
    # rounding can mint -0.0, exercising the +-0.0 fold across bands).
    for t in range(4):
        s = (torch.randn(2, 2048, device=dev) * 2).round() / 2
        s[:, 400 + 411 * t :] = float("-inf")
        _check_banded(s, 512, f"banded ties trial {t}")
    s = torch.randn(1, 2048, device=dev)
    s[:, 384:640] = 7.0  # tie block across the band-0/band-1 edge
    s[:, 1400:1600] = 7.0  # second tie block in bands 2/3
    _check_banded(s, 512, "banded cross-band tie blocks")

    # Mixed +-0.0 among distinct values.
    s = torch.randn(1, 2048, device=dev) + 3.0
    s[:, ::5] = 0.0
    s[:, 3::7] = -0.0
    _check_banded(s, 512, "banded pm-zero mix")

    # All-negative scores (negative indexer weights).
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

    # Randomized adversarial property sweep: quantized ties + random -inf
    # cutoffs + +-0 patches over every band fan.
    for trial in range(20):
        n = 1 + trial % 3
        c = (512, 768, 1500, 2048, 4096)[trial % 5]
        k = (512, 128)[trial % 2]
        s = (torch.randn(n, c, device=dev) * (10 ** (trial % 3 - 1))).round() / 4
        cut = int(torch.randint(0, c + 1, (1,)).item())
        s[:, cut:] = float("-inf")
        if trial % 4 == 0:
            s[:, :: 3 + trial % 5] = 0.0
        _check_banded(s, k, f"banded property trial={trial} n={n} c={c} k={k} cut={cut}")


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_inside_select_decode_gate():
    """The integration gate routes fp32 CUDA scores through the fused kernel."""
    torch.manual_seed(1)
    s = torch.randn(1, 512, device="cuda")
    s[:, 490:] = float("-inf")
    rows, valid = M._fused_topk_select(s, 512, 512)
    ref_rows, ref_valid = _eager_topk_tail(s, 512, 512)
    assert torch.equal(rows, ref_rows) and torch.equal(valid, ref_valid)
    # cudagraph-style replay stability: same inputs, repeated launches, same bytes.
    for _ in range(5):
        rows2, valid2 = M._fused_topk_select(s, 512, 512)
        assert torch.equal(rows2, rows) and torch.equal(valid2, valid)


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_banded_cudagraph_replay():
    """Banded arrival tickets stay correct across capture and many replays.

    The tickets are monotonic (never reset), so every replay of a captured
    graph -- and eager relaunches interleaved with it -- must keep emitting
    identical bytes.  Also pins the capture-time fallback: a shape whose
    ticket buffer was never warmed up eagerly must still produce exact
    results when first captured.
    """
    torch.manual_seed(3)
    s = torch.randn(1, 2048, device="cuda")
    s[:, 1800:] = float("-inf")
    ref_rows, ref_valid = _eager_topk_tail(s, 512, 2048)

    # Eager warmup (allocates + exercises the ticket buffer).
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
    # Interleave eager calls with replays (ticket alignment must survive).
    rows, valid = M._fused_topk_select(s, 512, 512)
    assert torch.equal(rows, ref_rows) and torch.equal(valid, ref_valid)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(rows_g, ref_rows) and torch.equal(valid_g, ref_valid)

    # Capture-time fallback: C=1024 -> 2 bands never ran eagerly, so the
    # wrapper must route capture to the single-CTA sort and stay exact.
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
    """Scores masked exactly like the decode score kernels: -inf at c >= vlen."""
    s = torch.randn(n, c, device=dev)
    if ties:
        s = (s * 2).round() / 2
    input_pos = []
    for row, v in enumerate(vlens):
        s[row, v:] = float("-inf")
        # vlen == min((input_pos + 1) // 4, C): input_pos = 4 * vlen - 1 hits it
        # exactly for vlen >= 1; any input_pos in [0, 2] yields vlen == 0.
        input_pos.append(4 * v - 1 if v > 0 else 0)
    return s, torch.tensor(input_pos, dtype=torch.long, device=dev)


def _check_vlen(index_score, input_pos, index_topk: int, case: str):
    """Fast-path output must equal the no-hint kernel AND the eager tail."""
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


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_visible_prefix_fast_path():
    """The banded visible-prefix fast path (idea_0089) stays byte-exact.

    Passing ``input_pos``/``compress_ratio`` lets band 0 emit its own sort
    directly while the other band CTAs retire whenever
    ``vlen = min((input_pos + 1) // 4, C) <= TOPK_BLOCK`` -- the entire
    production decode window.  Pinned across the band-edge transitions
    (vlen 511/512/513), empty history (vlen 0), ties, deeper band fans, mixed
    fast/slow rows in one launch, and the vlen > TOPK_BLOCK slow path with the
    hint still compiled in.
    """
    torch.manual_seed(7)

    # C=1024/K=512 (2 bands, the decode shape): every band-edge transition.
    for vlen in (0, 1, 250, 511, 512):
        s, ip = _masked_scores(2, 1024, [vlen, vlen])
        _check_vlen(s, ip, 512, f"fast vlen={vlen}")
    for vlen in (513, 900, 1024):  # slow path with the hint compiled in
        s, ip = _masked_scores(2, 1024, [vlen, vlen])
        _check_vlen(s, ip, 512, f"slow vlen={vlen}")

    # Heavy ties around the boundary (quantized scores can mint -0.0).
    for vlen in (250, 511, 512, 513):
        s, ip = _masked_scores(2, 1024, [vlen, vlen], ties=True)
        _check_vlen(s, ip, 512, f"ties vlen={vlen}")

    # Mixed fast/slow rows in one launch (per-row branch + ticket parity).
    s, ip = _masked_scores(4, 1024, [250, 513, 0, 1024])
    _check_vlen(s, ip, 512, "mixed fast/slow rows")

    # Deeper band fan: C=2048 -> 4 bands, K=512.
    for vlen in (0, 250, 512, 513, 1500, 2048):
        s, ip = _masked_scores(1, 2048, [vlen])
        _check_vlen(s, ip, 512, f"4-band vlen={vlen}")

    # K=128 over C=2048 (TOPK_BLOCK=128, 16 bands): boundary at 128.
    for vlen in (0, 100, 128, 129, 700):
        s, ip = _masked_scores(1, 2048, [vlen])
        _check_vlen(s, ip, 128, f"16-band vlen={vlen}")

    # Short history + pad tail (index_topk > C).
    s, ip = _masked_scores(2, 384, [40, 384])
    _check_vlen(s, ip, 512, "pad k>C with vlen")


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_topk_select_visible_prefix_ticket_parity_and_replay():
    """Fast-path launches leave the arrival tickets untouched.

    Interleaving fast (vlen <= TOPK_BLOCK, zero ticket arrivals) and slow
    (full merge protocol, two arrivals per node) launches on the same ticket
    buffer must keep the parity arithmetic aligned, eagerly and across CUDA
    graph replays whose input_pos value changes between replays.
    """
    torch.manual_seed(11)
    s_fast, ip_fast = _masked_scores(1, 1024, [250])
    s_slow, ip_slow = _masked_scores(1, 1024, [800])
    ref_fast = M._fused_topk_select(s_fast, 512, 512)
    ref_slow = M._fused_topk_select(s_slow, 512, 512)

    for _ in range(3):  # alternate fast/slow eagerly on one ticket buffer
        rf = M._fused_topk_select(s_fast, 512, 512, ip_fast, 4)
        rs = M._fused_topk_select(s_slow, 512, 512, ip_slow, 4)
        assert torch.equal(rf[0], ref_fast[0]) and torch.equal(rf[1], ref_fast[1])
        assert torch.equal(rs[0], ref_slow[0]) and torch.equal(rs[1], ref_slow[1])

    # Captured graph whose score/input_pos buffers are rewritten per replay:
    # the same launch must flip between fast and slow paths by data alone.
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
