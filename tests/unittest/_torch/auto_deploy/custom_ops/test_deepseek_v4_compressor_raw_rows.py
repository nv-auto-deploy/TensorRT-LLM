# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw bf16 compressor-row contract for the DeepSeek V4 sparse-attention ops.

Since idea_0092 the model hands the compressor kv/gate projections to the
sparse-attention ops in their native activation dtype (bf16) instead of
pre-widening them with a per-layer ``.float()``.  The decode current-token
store converts in-kernel and the compressed-row reconstruction anchors its
compute dtype on the fp32 caches, while the prefill/source paths widen once at
their entry.  Because bf16 -> fp32 widening is exact, every path must produce
byte-identical results for bf16 inputs and their fp32 widenings; these tests
guard that invariance on the source (prefill) op for both compression modes.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention import (
    _build_full_compressed_kv,
)


def _rope_tables(n_pos: int, rope_dim: int, device: torch.device):
    positions = torch.arange(n_pos, dtype=torch.float32, device=device).unsqueeze(1)
    freqs = torch.linspace(0.05, 0.25, rope_dim // 2, dtype=torch.float32, device=device)
    angles = positions * freqs.unsqueeze(0)
    return angles.cos(), angles.sin()


def _source_case(compress_ratio: int, device: torch.device):
    batch_size = 1
    head_dim = 8
    rope_dim = 4
    channels = 2 if compress_ratio == 4 else 1
    state_dim = channels * head_dim
    max_compressed_len = 2
    seq_len = max_compressed_len * compress_ratio
    window_size = 4
    num_heads = 2

    # fp32 q/kv keeps the attend on the deterministic reference chunk loop, so the
    # two invocations differ ONLY in the compressor input dtype.
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
    kv = torch.randn(batch_size, seq_len, head_dim, device=device)
    attn_sink = torch.tensor([-0.25, 0.1], device=device)

    compressor_kv = torch.randn(batch_size, seq_len, state_dim, device=device).to(torch.bfloat16)
    compressor_gate = torch.randn(batch_size, seq_len, state_dim, device=device).to(torch.bfloat16)
    ape = torch.randn(compress_ratio, state_dim, device=device)
    norm_weight = torch.randn(head_dim, device=device)
    cos_table, sin_table = _rope_tables(seq_len + 4, rope_dim, device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).contiguous()

    return dict(
        q=q,
        kv=kv,
        attn_sink=attn_sink,
        compressor_kv=compressor_kv,
        compressor_gate=compressor_gate,
        ape=ape,
        norm_weight=norm_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        position_ids=position_ids,
        head_dim=head_dim,
        rope_dim=rope_dim,
        state_dim=state_dim,
        window_size=window_size,
        max_compressed_len=max_compressed_len,
        seq_len=seq_len,
        batch_size=batch_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compressed pool ops require CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_source_op_bf16_rows_match_fp32_widening(compress_ratio: int) -> None:
    """The source op gives byte-identical output for bf16 rows vs their fp32 widening."""
    torch.manual_seed(20260708 + compress_ratio)
    device = torch.device("cuda")
    case = _source_case(compress_ratio, device)

    empty_iq = case["q"].new_empty(case["batch_size"], case["seq_len"], 0, 0)
    empty_iw = case["q"].new_empty(case["batch_size"], case["seq_len"], 0)
    empty_state = case["q"].new_empty(case["batch_size"], case["seq_len"], 0)
    empty_ape = case["q"].new_empty(0, 0)
    empty_norm = case["q"].new_empty(0)
    width_only_topk = case["q"].new_empty(
        case["batch_size"],
        case["seq_len"],
        case["window_size"] + case["max_compressed_len"],
        dtype=torch.int64,
    )

    def _run(compressor_kv: torch.Tensor, compressor_gate: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention(
            case["q"],
            case["kv"],
            case["attn_sink"],
            width_only_topk,
            compressor_kv,
            compressor_gate,
            case["ape"],
            case["norm_weight"],
            case["cos_table"],
            case["sin_table"],
            case["position_ids"],
            empty_iq,
            empty_iw,
            empty_state,
            empty_state,
            empty_ape,
            empty_norm,
            case["head_dim"] ** -0.5,
            window_size=case["window_size"],
            compress_ratio=compress_ratio,
            max_compressed_len=case["max_compressed_len"],
            rope_dim=case["rope_dim"],
            rms_norm_eps=1e-6,
            topk_is_placeholder=True,
        )

    out_bf16 = _run(case["compressor_kv"], case["compressor_gate"])
    # The exact fp32 widening is what the modeling-side per-layer ``.float()`` used
    # to hand over before idea_0092.
    out_f32 = _run(case["compressor_kv"].float(), case["compressor_gate"].float())

    assert torch.equal(out_bf16, out_f32), (
        f"source op diverged between bf16 rows and their fp32 widening "
        f"(compress_ratio={compress_ratio})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compressed pool ops require CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_build_full_compressed_kv_bf16_rows_match_fp32_widening(compress_ratio: int) -> None:
    """The compressed-kv reference widens raw rows at entry: bf16 in == fp32 in."""
    torch.manual_seed(20260709 + compress_ratio)
    device = torch.device("cuda")
    case = _source_case(compress_ratio, device)

    def _run(compressor_kv: torch.Tensor, compressor_gate: torch.Tensor) -> torch.Tensor:
        return _build_full_compressed_kv(
            compressor_kv,
            compressor_gate,
            case["ape"],
            case["norm_weight"],
            case["cos_table"],
            case["sin_table"],
            case["position_ids"],
            1e-6,
            case["rope_dim"],
            compress_ratio,
            case["max_compressed_len"],
        )

    out_bf16 = _run(case["compressor_kv"], case["compressor_gate"])
    out_f32 = _run(case["compressor_kv"].float(), case["compressor_gate"].float())

    assert out_bf16.dtype == out_f32.dtype
    assert torch.equal(out_bf16, out_f32), (
        f"_build_full_compressed_kv diverged between bf16 rows and their fp32 "
        f"widening (compress_ratio={compress_ratio})"
    )
