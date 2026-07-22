# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exactness tests for the decode-tail host fast paths."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.compile.backends.torch_cudagraph import CapturedGraph
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_seq_info(device: str) -> SequenceInfo:
    si = SequenceInfo(max_seq_len=64, max_batch_size=4, max_num_tokens=64)
    si.to(device)
    return si


def _nest_decode_batch(si: SequenceInfo, input_pos, gather_idx, mask_idx, ungathered):
    b = len(input_pos)
    si.nest_sequences(
        input_ids=[-1] * b,
        cu_seqlen=list(range(b + 1)),
        input_pos=list(input_pos),
        slot_idx=list(range(b)),
        _gather_idx=list(gather_idx),
        _mask_scatter_indices=list(mask_idx),
        _ungathered_input_ids=ungathered,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_rescatter_identity_fast_path_matches_triton(batch_size):
    """Identity gather/scatter (the decode steady state) must equal the triton op."""
    device = "cuda"
    torch.manual_seed(0)
    ungathered = torch.randint(0, 1000, (8,), dtype=torch.int, device=device)

    # reference: force the generic triton path with the same indices
    si_ref = _make_seq_info(device)
    b = batch_size
    si_ref.nest_sequences(
        input_ids=[-1] * b,
        cu_seqlen=list(range(b + 1)),
        input_pos=[7] * b,
        slot_idx=list(range(b)),
        _gather_idx=list(range(b)),
        _mask_scatter_indices=list(range(b)),
    )
    out_ref = si_ref.get_arg("input_ids", truncate=True, unflatten=False)
    torch.ops.auto_deploy.triton_utils_fused_gather_scatter(
        ungathered_input=ungathered,
        gather_ids=si_ref.get_arg("_gather_idx", truncate=True),
        mask_indices=si_ref.get_arg("_mask_scatter_indices", truncate=True),
        out=out_ref,
    )

    # candidate: full nest_sequences path (hits the identity fast path internally)
    si = _make_seq_info(device)
    _nest_decode_batch(si, [7] * b, range(b), range(b), ungathered)
    out = si.get_arg("input_ids", truncate=True, unflatten=False)

    torch.cuda.synchronize()
    assert torch.equal(out.cpu(), out_ref.cpu())
    assert torch.equal(out.cpu(), ungathered[:b].to(torch.int).cpu())


def test_rescatter_non_identity_still_generic():
    """Permuted gather indices must bypass the fast path and stay exact."""
    device = "cuda"
    ungathered = torch.tensor([11, 22, 33, 44], dtype=torch.int, device=device)
    si = _make_seq_info(device)
    # gather slots [2, 0] into positions [0, 1]
    _nest_decode_batch(si, [3, 5], [2, 0], [0, 1], ungathered)
    out = si.get_arg("input_ids", truncate=True, unflatten=False)
    torch.cuda.synchronize()
    assert out.cpu().tolist() == [33, 11]


def test_position_ids_generate_only_fast_path():
    """Generate-only batches: position_ids == input_pos, exactly as the general formula."""
    device = "cuda"
    si = _make_seq_info(device)
    input_pos = [3, 17, 0, 41]
    b = len(input_pos)
    si.nest_sequences(
        input_ids=list(range(b)),
        cu_seqlen=list(range(b + 1)),
        input_pos=input_pos,
        slot_idx=list(range(b)),
    )
    pos_host = si.get_arg("position_ids_host", truncate=True, unflatten=False)
    assert pos_host.tolist() == input_pos


def test_position_ids_mixed_batch_general_path():
    """Mixed prefill+decode batches keep the general (repeat + offset) result."""
    device = "cuda"
    si = _make_seq_info(device)
    # seq 0: prefill of 3 tokens from pos 0; seq 1: decode token at pos 9
    si.nest_sequences(
        input_ids=[1, 2, 3, 4],
        cu_seqlen=[0, 3, 4],
        input_pos=[0, 9],
        slot_idx=[0, 1],
        batch_info=[1, 3, 0, 0, 1, 1],
    )
    pos_host = si.get_arg("position_ids_host", truncate=True, unflatten=False)
    assert pos_host.tolist() == [0, 1, 2, 9]


class _AddModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return (x * 2 + y,)


def test_captured_graph_direct_kwargs_and_foreach_copy():
    """Replay via the fast path must equal eager for all captured batch sizes."""
    device = "cuda"
    torch.manual_seed(0)
    model = _AddModel().to(device)
    x_buf = torch.randn(4, 8, device=device)
    y_buf = torch.randn(4, 8, device=device)

    def get_args_kwargs(bs: int):
        return (), {"x": x_buf[:bs], "y": y_buf[:bs]}

    cg = CapturedGraph(model)
    with torch.inference_mode():
        cg.capture_graph(get_args_kwargs, batch_sizes=[4, 2, 1])

        for bs in (1, 2, 4):
            x = torch.randn(bs, 8, device=device)
            y = torch.randn(bs, 8, device=device)
            out = cg(x=x, y=y)[0]
            ref = model(x, y)[0]
            torch.cuda.synchronize()
            assert torch.equal(out.cpu(), ref.cpu()), f"mismatch at bs={bs}"

        # replays must have populated the per-shape input-view cache
        assert len(cg._input_views_cache) >= 1

    # engine graph release must drop the cached views (they alias freed buffers)
    ADEngine._release_cuda_graphs(SimpleNamespace(model=cg))
    assert cg._input_views_cache == {}
    assert cg.cudagraphs == {}
    assert cg._input_buffers == []
