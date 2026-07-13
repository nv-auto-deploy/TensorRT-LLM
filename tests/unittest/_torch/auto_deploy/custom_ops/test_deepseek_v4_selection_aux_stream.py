# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the aux-stream decode ratio-4 selection (decode_selection_aux).

When the flag set by ``set_decode_selection_aux`` is on, the cached decode op
moves the indexer-cache current-token write plus the index-score / top-k
selection onto the CudaStreamManager aux stream and re-joins the main stream
immediately before the assemble kernel.  The aux path launches the exact same
kernels in the same within-stream order as the inline path, so the contract is
BIT-exact equality of the op output and of all six caches -- pinned here over
consecutive decode steps that cross ``(input_pos + 1) % 4 == 0`` (the steps
where the score kernel reads the indexer row written this very step), under
CUDA-graph capture/replay with mutated ``input_pos``, and through the fallback
paths (flag off / ``disable_multi_stream``), which must never enter the aux
helper.
"""

import math

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo

_TPB = 8  # tokens per cache page
_PAGES_PER_SEQ = 8
_CAP = _TPB * _PAGES_PER_SEQ  # per-sequence token capacity
_RATIO = 4
_MAX_COMPRESSED = _CAP // _RATIO
_WINDOW = 8
_INDEX_TOPK = 8
_HEADS = 2
_HEAD_DIM = 16
_IDX_HEADS = 16  # > 8 routes the fused one-kernel index score
_IDX_HEAD_DIM = 32
_IDX_STATE = 2 * _IDX_HEAD_DIM
_STATE = 2 * _HEAD_DIM  # overlap mode reads [:head_dim] and [head_dim:2*head_dim]
_ROPE = 8


def _supported() -> bool:
    return torch.cuda.is_available() and M._HAS_TRITON


def _make_state(seed: int, batch: int) -> dict:
    """Cache universe + step-invariant weights (identical for equal seeds)."""
    torch.manual_seed(seed)
    dev = "cuda"
    pages = batch * _PAGES_PER_SEQ
    return dict(
        swa_cache=torch.randn(pages, _TPB, _HEAD_DIM, device=dev, dtype=torch.bfloat16),
        mhc_cache=torch.randn(pages, _TPB, _HEAD_DIM, device=dev, dtype=torch.bfloat16),
        compressor_kv_cache=torch.randn(pages, _TPB, _STATE, device=dev),
        compressor_gate_cache=torch.randn(pages, _TPB, _STATE, device=dev),
        indexer_kv_cache=torch.randn(pages, _TPB, _IDX_STATE, device=dev),
        indexer_gate_cache=torch.randn(pages, _TPB, _IDX_STATE, device=dev),
        ape=torch.randn(_RATIO, _STATE, device=dev),
        norm_w=torch.rand(_HEAD_DIM, device=dev) + 0.5,
        idx_ape=torch.randn(_RATIO, _IDX_STATE, device=dev),
        idx_norm_w=torch.rand(_IDX_HEAD_DIM, device=dev) + 0.5,
        cos=torch.randn(_CAP, _ROPE // 2, device=dev),
        sin=torch.randn(_CAP, _ROPE // 2, device=dev),
        attn_sink=torch.randn(_HEADS, device=dev),
    )


_CACHE_KEYS = (
    "swa_cache",
    "mhc_cache",
    "compressor_kv_cache",
    "compressor_gate_cache",
    "indexer_kv_cache",
    "indexer_gate_cache",
)


def _step_inputs(seed: int, batch: int) -> dict:
    """Fresh per-step activations (identical for equal seeds)."""
    torch.manual_seed(seed)
    dev = "cuda"
    bf16 = torch.bfloat16
    return dict(
        q=torch.randn(batch, 1, _HEADS, _HEAD_DIM, device=dev, dtype=bf16),
        kv=torch.randn(batch, 1, _HEAD_DIM, device=dev, dtype=bf16),
        compressor_kv=torch.randn(batch, 1, _STATE, device=dev, dtype=bf16),
        compressor_gate=torch.randn(batch, 1, _STATE, device=dev, dtype=bf16),
        indexer_q=torch.randn(batch, 1, _IDX_HEADS, _IDX_HEAD_DIM, device=dev, dtype=bf16),
        indexer_weights=torch.randn(batch, 1, _IDX_HEADS, device=dev, dtype=bf16),
        indexer_kv=torch.randn(batch, 1, _IDX_STATE, device=dev, dtype=bf16),
        indexer_gate=torch.randn(batch, 1, _IDX_STATE, device=dev, dtype=bf16),
    )


def _metadata(input_positions: list[int]):
    """Standard + hoisted decode metadata for the 64-arg cached op."""
    b = len(input_positions)
    bi = BatchInfo()
    bi.update([0, 0, 0, 0, b, b])
    seq_len_host = torch.ones(b, dtype=torch.int32)
    input_pos_host = torch.tensor(input_positions, dtype=torch.int32)
    cu_seqlen_host = torch.arange(b + 1, dtype=torch.int32)
    cu_num_pages_host = torch.arange(b + 1, dtype=torch.int32) * _PAGES_PER_SEQ
    cache_loc_host = torch.arange(b * _PAGES_PER_SEQ, dtype=torch.int32)
    input_pos_dev = input_pos_host.cuda()
    cu_num_pages = cu_num_pages_host.cuda()
    cache_loc = cache_loc_host.cuda()
    position_ids = input_pos_dev.to(torch.int64).view(b, 1).contiguous()
    slot_idx = torch.arange(b, dtype=torch.int64).cuda()
    hoisted = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr(
        input_pos_dev,
        position_ids,
        cu_num_pages,
        cache_loc,
        _TPB,
        _MAX_COMPRESSED,
        1,
        _WINDOW,
    )
    meta = (
        bi.serialize(),
        input_pos_dev,
        slot_idx,
        cu_num_pages,
        cache_loc,
        seq_len_host,
        input_pos_host,
        cu_seqlen_host,
        cu_num_pages_host,
        cache_loc_host,
        *hoisted,
    )
    return meta, position_ids


def _run_step(state: dict, inputs: dict, meta, position_ids, out=None) -> torch.Tensor:
    batch = position_ids.shape[0]
    topk = torch.zeros(batch, 1, _WINDOW + _INDEX_TOPK, dtype=torch.int64, device="cuda")
    return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache(
        inputs["q"],
        inputs["kv"],
        state["attn_sink"],
        topk,
        inputs["compressor_kv"],
        inputs["compressor_gate"],
        state["ape"],
        state["norm_w"],
        state["cos"],
        state["sin"],
        position_ids,
        inputs["indexer_q"],
        inputs["indexer_weights"],
        inputs["indexer_kv"],
        inputs["indexer_gate"],
        state["idx_ape"],
        state["idx_norm_w"],
        *meta,
        state["swa_cache"],
        state["mhc_cache"],
        state["compressor_kv_cache"],
        state["compressor_gate_cache"],
        state["indexer_kv_cache"],
        state["indexer_gate_cache"],
        1.0 / math.sqrt(_HEAD_DIM),
        _WINDOW,
        _RATIO,
        _MAX_COMPRESSED,
        1e-6,
        _ROPE,
        out=out,
    )


def _assert_states_equal(state_a: dict, state_b: dict, msg: str) -> None:
    for key in _CACHE_KEYS:
        assert torch.equal(state_a[key], state_b[key]), f"{msg}: cache {key} diverged"


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
@pytest.mark.parametrize("batch", [1, 2])
def test_selection_aux_bit_exact_over_decode_steps(batch, monkeypatch):
    """Flag-on output/caches match flag-off bit-for-bit across %4 boundaries.

    Steps sweep input_pos 9..16 (sequence 1 shifted by 3), crossing the
    current-token-read steps (input_pos + 1) % 4 == 0 several times for each
    sequence, so the aux-ordered indexer-row write is provably honored.
    """
    aux_calls = 0
    orig_launch = M._launch_ratio4_selection_on_aux

    def counted(*args, **kwargs):
        nonlocal aux_calls
        aux_calls += 1
        return orig_launch(*args, **kwargs)

    monkeypatch.setattr(M, "_launch_ratio4_selection_on_aux", counted)

    state_ref = _make_state(123, batch)
    state_aux = _make_state(123, batch)
    _assert_states_equal(state_ref, state_aux, "seed")

    num_steps = 8
    try:
        for step in range(num_steps):
            positions = [9 + step + 3 * b for b in range(batch)]
            meta, position_ids = _metadata(positions)
            inputs = _step_inputs(1000 + step, batch)

            M.set_decode_selection_aux(False)
            out_ref = _run_step(state_ref, inputs, meta, position_ids)

            M.set_decode_selection_aux(True)
            out_aux = _run_step(state_aux, inputs, meta, position_ids)

            torch.cuda.synchronize()
            assert torch.equal(out_ref, out_aux), f"step {step} (pos={positions}): output"
            _assert_states_equal(state_ref, state_aux, f"step {step} (pos={positions})")
    finally:
        M.set_decode_selection_aux(False)

    assert aux_calls == num_steps, "aux selection helper must run once per flag-on step"


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_selection_aux_cuda_graph_replay_matches_eager_inline():
    """Captured aux-path decode replays bit-exactly against the eager inline path.

    The graph captures prepare-metadata + the cached op with the flag on; the
    input_pos / activation buffers mutate between replays (crossing the
    %4 == 0 current-token-read step), and every replayed step must equal an
    eager flag-off universe stepped with identical inputs.
    """
    batch = 1
    state_g = _make_state(7, batch)
    state_ref = _make_state(7, batch)
    snapshot = {k: state_g[k].clone() for k in _CACHE_KEYS}

    # Static input buffers rewritten per replay.
    input_pos_buf = torch.tensor([9], dtype=torch.int32, device="cuda")
    position_ids_buf = torch.tensor([[9]], dtype=torch.int64, device="cuda")
    inputs_buf = _step_inputs(0, batch)
    out_buf = torch.empty_like(inputs_buf["q"])

    bi = BatchInfo()
    bi.update([0, 0, 0, 0, batch, batch])
    cu_num_pages = (torch.arange(batch + 1, dtype=torch.int32) * _PAGES_PER_SEQ).cuda()
    cache_loc = torch.arange(batch * _PAGES_PER_SEQ, dtype=torch.int32).cuda()
    host_meta = (
        bi.serialize(),
        input_pos_buf,
        torch.arange(batch, dtype=torch.int64).cuda(),
        cu_num_pages,
        cache_loc,
        torch.ones(batch, dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
        torch.arange(batch + 1, dtype=torch.int32),
        (torch.arange(batch + 1, dtype=torch.int32) * _PAGES_PER_SEQ),
        torch.arange(batch * _PAGES_PER_SEQ, dtype=torch.int32),
    )

    def run_captured() -> None:
        hoisted = torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr(
            input_pos_buf,
            position_ids_buf,
            cu_num_pages,
            cache_loc,
            _TPB,
            _MAX_COMPRESSED,
            1,
            _WINDOW,
        )
        _run_step(state_g, inputs_buf, (*host_meta, *hoisted), position_ids_buf, out=out_buf)

    try:
        M.set_decode_selection_aux(True)
        # Warmup on a side stream (mutates caches), then capture.
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            run_captured()
        side.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_captured()

        # Reset the graph universe to the pristine snapshot (warmup mutated it).
        for key in _CACHE_KEYS:
            state_g[key].copy_(snapshot[key])

        M.set_decode_selection_aux(False)
        for step, pos in enumerate([9, 10, 11, 12]):  # 11 hits (pos + 1) % 4 == 0
            step_in = _step_inputs(2000 + step, batch)
            for key, value in step_in.items():
                inputs_buf[key].copy_(value)
            input_pos_buf.fill_(pos)
            position_ids_buf.fill_(pos)
            graph.replay()
            torch.cuda.synchronize()

            meta, position_ids = _metadata([pos])
            out_ref = _run_step(state_ref, step_in, meta, position_ids)
            torch.cuda.synchronize()
            assert torch.equal(out_buf, out_ref), f"replay step {step} (pos={pos}): output"
            _assert_states_equal(state_g, state_ref, f"replay step {step} (pos={pos})")
    finally:
        M.set_decode_selection_aux(False)


@pytest.mark.skipif(not _supported(), reason="requires CUDA + triton")
def test_selection_aux_fallback_paths_never_enter_aux_helper(monkeypatch):
    """Flag off and disable_multi_stream both take today's inline path."""
    from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import disable_multi_stream

    def poisoned(*args, **kwargs):
        raise AssertionError("aux selection helper must not run on fallback paths")

    monkeypatch.setattr(M, "_launch_ratio4_selection_on_aux", poisoned)

    batch = 1
    state_a = _make_state(55, batch)
    state_b = _make_state(55, batch)
    meta, position_ids = _metadata([13])
    inputs = _step_inputs(42, batch)

    # Flag off (default): inline selection.
    out_a = _run_step(state_a, inputs, meta, position_ids)

    # Flag on under disable_multi_stream (piecewise capture): inline selection.
    try:
        M.set_decode_selection_aux(True)
        with disable_multi_stream():
            out_b = _run_step(state_b, inputs, meta, position_ids)
    finally:
        M.set_decode_selection_aux(False)

    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b)
    _assert_states_equal(state_a, state_b, "fallback")
