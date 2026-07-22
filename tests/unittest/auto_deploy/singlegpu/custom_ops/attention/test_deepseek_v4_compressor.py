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
"""Tests for the DeepSeek-V4 compressor ops and the fused compressed-row update."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fake_fp8_act_quant

DEV = "cuda"

# Frozen pre-fold stage-2 kernel (moved verbatim out of the production module when
# the rope/fp8/masked-store tail was folded into the producing kernels): the
# byte-exact reference the fold is pinned against below.
if M._HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _dsv4_rope_fp8_masked_store_kernel(
        normed_ptr,  # [N, HEAD_DIM] post-rmsnorm rows (activation dtype)
        cos_ptr,  # [n_pos, DH] fp32
        sin_ptr,  # [n_pos, DH] fp32
        row_position_id_ptr,  # [N] int64 (already clamped into [0, n_pos))
        row_valid_ptr,  # [N] bool -- store row iff valid
        mhc_page_ids_ptr,  # [N] int64 write page id per row
        mhc_page_offsets_ptr,  # [N] int64 in-page offset per row
        cache_ptr,  # [P, T, HEAD_DIM] paged mhc cache (mutated in place)
        stride_p,
        stride_t,
        stride_s,
        cossin_row_stride,
        N,
        HEAD_DIM,
        NOPE_DIM,  # HEAD_DIM - ROPE_DIM (multiple of FP8_BLOCK)
        DH,  # ROPE_DIM // 2
        FP8_BLOCK: tl.constexpr,
        NUM_FP8_BLOCKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_DH: tl.constexpr,
        MAX_VAL: tl.constexpr,  # 448.0 (e4m3 absmax)
        MIN_VAL: tl.constexpr,  # 1e-4 (amax floor)
    ):
        """Fake-fp8(nope) + interleaved RoPE(pe) + validity-masked paged store."""
        RD = normed_ptr.dtype.element_ty  # rounding dtype for the fp8/rope math
        CT = cache_ptr.dtype.element_ty  # cache store dtype

        b = tl.program_id(0)
        if b >= N:
            return
        valid = tl.load(row_valid_ptr + b).to(tl.int1)
        pid = tl.load(mhc_page_ids_ptr + b).to(tl.int64)
        poff = tl.load(mhc_page_offsets_ptr + b).to(tl.int64)
        dst_base = cache_ptr + pid * stride_p + poff * stride_t
        nrow = normed_ptr + b.to(tl.int64) * HEAD_DIM

        # fake-fp8 block quant on the nope slice [0, NOPE_DIM)
        d = tl.arange(0, BLOCK_D)
        nmask = d < NOPE_DIM
        nope = tl.load(nrow + d, mask=nmask, other=0.0).to(tl.float32)
        blk = d // FP8_BLOCK
        scale_per_d = tl.full([BLOCK_D], 1.0, tl.float32)
        for j in tl.static_range(NUM_FP8_BLOCKS):
            in_blk = nmask & (blk == j)
            amax_j = tl.max(tl.where(in_blk, tl.abs(nope), 0.0), axis=0)
            scale_j = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax_j, MIN_VAL) / MAX_VAL)))
            scale_per_d = tl.where(in_blk, scale_j, scale_per_d)
        q = nope / scale_per_d
        q = tl.minimum(tl.maximum(q, -MAX_VAL), MAX_VAL)
        q = q.to(RD).to(tl.float32)  # round-trip through the activation dtype
        nope_out = (q * scale_per_d).to(RD)
        tl.store(dst_base + d * stride_s, nope_out.to(CT), mask=nmask & valid)

        # interleaved RoPE on the pe slice [NOPE_DIM, HEAD_DIM)
        k = tl.arange(0, BLOCK_DH)
        kmask = k < DH
        pe_base = nrow + NOPE_DIM
        even = tl.load(pe_base + 2 * k, mask=kmask, other=0.0).to(tl.float32)
        odd = tl.load(pe_base + 2 * k + 1, mask=kmask, other=0.0).to(tl.float32)
        rpid = tl.load(row_position_id_ptr + b).to(tl.int64)
        cos = tl.load(cos_ptr + rpid * cossin_row_stride + k, mask=kmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + rpid * cossin_row_stride + k, mask=kmask, other=0.0).to(tl.float32)
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        pe_out_base = dst_base + NOPE_DIM * stride_s
        tl.store(pe_out_base + (2 * k) * stride_s, out_even.to(RD).to(CT), mask=kmask & valid)
        tl.store(pe_out_base + (2 * k + 1) * stride_s, out_odd.to(RD).to(CT), mask=kmask & valid)

    def _launch_compressed_rope_fp8_store(
        normed: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        row_position_id: torch.Tensor,
        row_valid: torch.Tensor,
        mhc_page_ids: torch.Tensor,
        mhc_page_offsets: torch.Tensor,
        mhc_cache: torch.Tensor,
        head_dim: int,
        rope_dim: int,
    ) -> None:
        n = int(normed.shape[0])
        if n == 0 or head_dim == 0:
            return
        nope_dim = head_dim - rope_dim
        dh = rope_dim // 2
        block_d = triton.next_power_of_2(head_dim)
        grid = (n,)
        _dsv4_rope_fp8_masked_store_kernel[grid](
            normed,
            cos_table,
            sin_table,
            row_position_id,
            row_valid,
            mhc_page_ids,
            mhc_page_offsets,
            mhc_cache,
            mhc_cache.stride(0),
            mhc_cache.stride(1),
            mhc_cache.stride(2),
            int(cos_table.stride(0)),
            n,
            head_dim,
            nope_dim,
            dh,
            FP8_BLOCK=64,
            NUM_FP8_BLOCKS=nope_dim // 64,
            BLOCK_D=block_d,
            BLOCK_DH=triton.next_power_of_2(dh),
            MAX_VAL=448.0,
            MIN_VAL=1.0e-4,
            num_warps=4,
        )


# ---------------------------------------------------------------------------
# auto_deploy::deepseek_v4_compress_pool
# ---------------------------------------------------------------------------


def _pool_ref(kv, gate):
    # The exact eager expression the op replaces (fp32-internal softmax).
    return (kv * gate.softmax(dim=-2)).sum(dim=-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (4, 512),  # rank-2 decode row, small-N BLOCK_D halving
        (2, 130, 8, 128),  # rank-4 overlap, indexer head_dim, odd row count
        (3, 5, 96),  # non-pow2 ratio and channel dims
    ],
)
def test_compress_pool_matches_reference(shape, dtype):
    torch.manual_seed(0)
    kv = torch.randn(shape, device=DEV, dtype=dtype)
    gate = torch.randn(shape, device=DEV, dtype=dtype)

    out = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    ref = _pool_ref(kv, gate)

    assert out.shape == ref.shape
    assert out.dtype == kv.dtype
    atol, rtol = (2e-4, 2e-4) if dtype == torch.float32 else (8e-3, 8e-3)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compress_pool_validity_masking():
    torch.manual_seed(1)
    B, R, D = 2, 8, 512
    dtype = torch.bfloat16
    kv = torch.randn(B, R, D, device=DEV, dtype=dtype)
    gate = torch.randn(B, R, D, device=DEV, dtype=dtype)
    # -1e20 gate == invalid candidate row; row 1 is fully masked.
    gate[0, R // 2 :, :] = -1.0e20
    gate[1, :, :] = -1.0e20

    out = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    ref = _pool_ref(kv, gate)

    torch.testing.assert_close(out.float(), ref.float(), atol=8e-3, rtol=8e-3)
    assert torch.isfinite(out[1]).all()


def test_compress_pool_cpu_empty_and_fake():
    torch.manual_seed(2)
    kv = torch.randn(2, 4, 64)
    gate = torch.randn(2, 4, 64)
    out_cpu = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    torch.testing.assert_close(out_cpu, _pool_ref(kv, gate))

    empty = torch.empty(0, 4, 64)
    out_empty = torch.ops.auto_deploy.deepseek_v4_compress_pool(empty, empty)
    assert out_empty.shape == (0, 64)

    with torch._subclasses.FakeTensorMode():
        fkv = torch.empty(2, 4, 64, dtype=torch.bfloat16)
        fout = torch.ops.auto_deploy.deepseek_v4_compress_pool(fkv, fkv)
    assert fout.shape == (2, 64) and fout.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Rope/quant tail + compressor RMSNorm
# ---------------------------------------------------------------------------


def _old_tail(
    compressed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    rotate: bool,
) -> torch.Tensor:
    """Verbatim copy of the original eager tail implementation."""
    nope_dim = compressed.shape[-1] - rope_dim
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    pe = M._apply_interleaved_rope_ref(pe, cos, sin)
    compressed = torch.cat((nope, pe), dim=-1)
    if rotate:
        return torch.ops.auto_deploy.deepseek_v4_hadamard_fp4(compressed, 32)
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    nope = fake_fp8_act_quant(nope, block_size=64)
    return torch.cat((nope, pe), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,rope_dim",
    [
        ((2, 512), 64),  # decode row: [B, head_dim], nope=448 (7*64)
        ((1, 257, 512), 64),  # context build: [B, max_compressed_len, head_dim]
        ((2, 130), 64),  # nope_dim=66 not %64 -> fake_fp8 is a pass-through
    ],
)
def test_rotate_false_matches_eager(shape, rope_dim):
    torch.manual_seed(0)
    compressed = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    head_dim = shape[-1]
    dh = rope_dim // 2
    cos = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)
    sin = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)

    out = M._apply_compressed_rope_and_quantize(compressed, cos, sin, rope_dim, rotate=False)
    ref = _old_tail(compressed, cos, sin, rope_dim, rotate=False)

    assert out.shape == ref.shape == (*shape[:-1], head_dim)
    assert out.dtype == ref.dtype == torch.bfloat16

    nope_dim = head_dim - rope_dim
    # fp8(nope) half byte-exact; rope(pe) half <=1 ULP (fused FMA vs eager mul/sub).
    torch.testing.assert_close(out[..., :nope_dim], ref[..., :nope_dim], atol=0.0, rtol=0.0)
    torch.testing.assert_close(out[..., nope_dim:], ref[..., nope_dim:], atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(2, 512), (3, 40, 512)])
def test_compressor_rms_norm_byte_identical(shape):
    torch.manual_seed(2)
    head_dim = shape[-1]
    x = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(head_dim, device=DEV, dtype=torch.bfloat16)
    eps = 1e-6
    assert torch.equal(M._compressor_rms_norm(x, w, eps), M._rms_norm_ref(x, w, eps))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressor_rms_norm_falls_back_no_weight():
    x = torch.randn(2, 512, device=DEV, dtype=torch.bfloat16)
    empty = torch.empty(0, device=DEV, dtype=torch.bfloat16)
    assert torch.equal(M._compressor_rms_norm(x, empty, 1e-6), M._rms_norm_ref(x, empty, 1e-6))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,rope_dim",
    [
        ((4, 128), 64),  # indexer decode row
        ((1, 5, 128), 64),  # indexer context rows
    ],
)
def test_rotate_true_byte_identical(shape, rope_dim):
    torch.manual_seed(1)
    compressed = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    dh = rope_dim // 2
    cos = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)
    sin = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)

    out = M._apply_compressed_rope_and_quantize(compressed, cos, sin, rope_dim, rotate=True)
    ref = _old_tail(compressed, cos, sin, rope_dim, rotate=True)
    assert torch.equal(out, ref)


# ---------------------------------------------------------------------------
# Source (prefill) path: raw bf16 compressor rows == their fp32 widening
# ---------------------------------------------------------------------------


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
        window_size=window_size,
        max_compressed_len=max_compressed_len,
        seq_len=seq_len,
        batch_size=batch_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compressed pool ops require CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_source_op_bf16_rows_match_fp32_widening(compress_ratio: int) -> None:
    torch.manual_seed(20260708 + compress_ratio)
    device = torch.device(DEV)
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

    def _run_op(compressor_kv: torch.Tensor, compressor_gate: torch.Tensor) -> torch.Tensor:
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

    out_bf16 = _run_op(case["compressor_kv"], case["compressor_gate"])
    out_f32 = _run_op(case["compressor_kv"].float(), case["compressor_gate"].float())

    assert torch.equal(out_bf16, out_f32)


# ---------------------------------------------------------------------------
# Fused decode compressed-row cache update
# ---------------------------------------------------------------------------


def _build_inputs(compress_ratio, num_rows, seed):
    torch.manual_seed(seed)
    head_dim = 512
    rope_dim = 64
    channels = 2 if compress_ratio == 4 else 1
    state_dim = channels * head_dim
    tokens_per_block = 8
    max_compressed_len = 4
    eps = 1e-6
    dtype = torch.bfloat16

    max_pos = max_compressed_len * compress_ratio + tokens_per_block
    pages_per_seq = (max_pos + tokens_per_block - 1) // tokens_per_block
    page_counts = [pages_per_seq] * num_rows
    cu_num_pages = torch.tensor(
        [0, *torch.tensor(page_counts).cumsum(0).tolist()], dtype=torch.long, device=DEV
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=DEV)

    kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=DEV)
    gate_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=DEV)
    mhc_cache = torch.randn(total_pages, tokens_per_block, head_dim, device=DEV, dtype=dtype)

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=DEV)
    # Mix row-completing (input_pos % ratio == ratio-1) and non-completing steps.
    input_pos = torch.tensor(
        [compress_ratio - 1 + i * compress_ratio + (i % 2) for i in range(num_rows)],
        dtype=torch.long,
        device=DEV,
    )
    position_ids = input_pos.clone()

    compressor_kv_decode = torch.randn(num_rows, state_dim, device=DEV, dtype=dtype)
    compressor_gate_decode = torch.randn(num_rows, state_dim, device=DEV, dtype=dtype)
    ape = torch.randn(compress_ratio, state_dim, device=DEV)
    norm_weight = torch.randn(head_dim, device=DEV, dtype=dtype)
    n_pos = int(input_pos.max().item()) + 4
    cos_table = torch.randn(n_pos, rope_dim // 2, device=DEV)
    sin_table = torch.randn(n_pos, rope_dim // 2, device=DEV)

    # Hoisted maps + update metadata from the production prepare op.
    overlap_m = max_compressed_len if compress_ratio == 4 else 1
    dense_m = max_compressed_len if compress_ratio == 128 else 1
    outs = M.deepseek_v4_sparse_prepare_decode_page_addr(
        input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, overlap_m, dense_m
    )
    overlap_page_map = None
    if compress_ratio == 4:
        overlap_page_map = (outs[2][:num_rows], outs[3][:num_rows], outs[4][:num_rows])
        update_meta = (*(o[:num_rows] for o in outs[8:12]), None, None)
    else:
        update_meta = tuple(o[:num_rows] for o in outs[12:18])

    return dict(
        compressor_kv_decode=compressor_kv_decode,
        compressor_gate_decode=compressor_gate_decode,
        position_ids_decode=position_ids,
        compressor_ape=ape,
        compressor_norm_weight=norm_weight,
        cos_table=cos_table,
        sin_table=sin_table,
        seq_idx=seq_idx,
        input_pos=input_pos,
        cu_num_pages=cu_num_pages,
        cache_loc=cache_loc,
        mhc_cache=mhc_cache,
        compressor_kv_cache=kv_cache,
        compressor_gate_cache=gate_cache,
        rms_norm_eps=eps,
        rope_dim=rope_dim,
        compress_ratio=compress_ratio,
        max_compressed_len=max_compressed_len,
        overlap_page_map=overlap_page_map,
        update_meta=update_meta,
    )


def _run_update(inp, mhc, update_meta=None):
    M._update_decode_compressed_caches(
        inp["compressor_kv_decode"],
        inp["compressor_gate_decode"],
        inp["position_ids_decode"],
        inp["compressor_ape"],
        inp["compressor_norm_weight"],
        inp["cos_table"],
        inp["sin_table"],
        inp["seq_idx"],
        inp["input_pos"],
        inp["cu_num_pages"],
        inp["cache_loc"],
        mhc,
        inp["compressor_kv_cache"],
        inp["compressor_gate_cache"],
        inp["rms_norm_eps"],
        inp["rope_dim"],
        inp["compress_ratio"],
        inp["max_compressed_len"],
        inp["overlap_page_map"],
        update_meta,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_fused_update_matches_eager(monkeypatch, compress_ratio):
    assert M._HAS_TRITON, "test requires triton"
    num_rows = 3 if compress_ratio == 4 else 2
    inp = _build_inputs(compress_ratio, num_rows, seed=100 + compress_ratio + num_rows)

    mhc_fused = inp["mhc_cache"].clone()
    _run_update(inp, mhc_fused)  # real _HAS_TRITON -> fused

    monkeypatch.setattr(M, "_HAS_TRITON", False)
    mhc_eager = inp["mhc_cache"].clone()
    _run_update(inp, mhc_eager)  # forced fallback -> eager op-by-op

    # End-to-end only ULP-close: the rsqrt <=1 ULP propagates into the fp8 quant.
    exact = (mhc_fused == mhc_eager).float().mean().item()
    max_abs = (mhc_fused.float() - mhc_eager.float()).abs().max().item()
    assert exact >= 0.95, f"exact bf16 match ratio {exact} too low (max_abs={max_abs})"
    torch.testing.assert_close(mhc_fused.float(), mhc_eager.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
def test_rope_fp8_store_kernel_byte_exact():
    assert M._HAS_TRITON, "test requires triton"
    num_rows = 3
    torch.manual_seed(7 + num_rows)
    head_dim, rope_dim, tokens_per_block = 512, 64, 8
    nope_dim, dh = head_dim - rope_dim, rope_dim // 2
    dtype = torch.bfloat16
    total_pages = num_rows + 2

    normed = torch.randn(num_rows, head_dim, device=DEV, dtype=dtype)
    n_pos = 32
    cos_table = torch.randn(n_pos, dh, device=DEV)
    sin_table = torch.randn(n_pos, dh, device=DEV)
    row_position_id = torch.randint(0, n_pos, (num_rows,), dtype=torch.long, device=DEV)
    row_valid = torch.tensor([bool((i + 1) % 2) for i in range(num_rows)], device=DEV)
    mhc_page_ids = torch.arange(num_rows, dtype=torch.long, device=DEV)
    mhc_page_offsets = torch.randint(0, tokens_per_block, (num_rows,), dtype=torch.long, device=DEV)
    mhc_cache = torch.randn(total_pages, tokens_per_block, head_dim, device=DEV, dtype=dtype)

    mhc_fused = mhc_cache.clone()
    _launch_compressed_rope_fp8_store(
        normed,
        cos_table,
        sin_table,
        row_position_id,
        row_valid,
        mhc_page_ids,
        mhc_page_offsets,
        mhc_fused,
        head_dim,
        rope_dim,
    )

    # Eager reference: gather cos/sin, run the shipped rope/fp8 tail, masked store.
    cos_g = cos_table[row_position_id]
    sin_g = sin_table[row_position_id]
    compressed = M._apply_compressed_rope_and_quantize(normed, cos_g, sin_g, rope_dim, rotate=False)
    mhc_ref = mhc_cache.clone()
    for r in range(num_rows):
        if bool(row_valid[r]):
            mhc_ref[int(mhc_page_ids[r]), int(mhc_page_offsets[r])] = compressed[r].to(dtype)

    for r in range(num_rows):
        if not bool(row_valid[r]):
            assert torch.equal(
                mhc_fused[int(mhc_page_ids[r]), int(mhc_page_offsets[r])],
                mhc_cache[int(mhc_page_ids[r]), int(mhc_page_offsets[r])],
            ), "invalid row must be left untouched"
    # fake-fp8 nope slice byte-identical; rope pe slice <=1 ULP.
    torch.testing.assert_close(
        mhc_fused[..., :nope_dim], mhc_ref[..., :nope_dim], atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(
        mhc_fused[..., nope_dim:].float(), mhc_ref[..., nope_dim:].float(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize(
    "num_rows,head_dim,rope_dim,dtype",
    [(3, 512, 64, torch.bfloat16), (3, 256, 64, torch.float16)],
)
def test_norm_rope_store_fused_kernel_matches_two_stage(num_rows, head_dim, rope_dim, dtype):
    assert M._HAS_TRITON, "test requires triton"
    torch.manual_seed(11 + num_rows + head_dim)
    tokens_per_block = 8
    nope_dim, dh = head_dim - rope_dim, rope_dim // 2
    eps = 1e-6
    total_pages = num_rows + 2

    pooled = torch.randn(num_rows, head_dim, device=DEV, dtype=dtype)
    norm_weight = torch.randn(head_dim, device=DEV, dtype=dtype)
    n_pos = 32
    cos_table = torch.randn(n_pos, dh, device=DEV)
    sin_table = torch.randn(n_pos, dh, device=DEV)
    row_position_id = torch.randint(0, n_pos, (num_rows,), dtype=torch.long, device=DEV)
    row_valid = torch.tensor([bool((i + 1) % 2) for i in range(num_rows)], device=DEV)
    mhc_page_ids = torch.arange(num_rows, dtype=torch.long, device=DEV)
    mhc_page_offsets = torch.randint(0, tokens_per_block, (num_rows,), dtype=torch.long, device=DEV)
    mhc_cache = torch.randn(total_pages, tokens_per_block, head_dim, device=DEV, dtype=dtype)

    # Reference: the removed two-stage chain (rms_norm launch + frozen stage-2 kernel).
    mhc_ref = mhc_cache.clone()
    normed_ref = M._compressor_rms_norm(pooled, norm_weight, eps)
    _launch_compressed_rope_fp8_store(
        normed_ref,
        cos_table,
        sin_table,
        row_position_id,
        row_valid,
        mhc_page_ids,
        mhc_page_offsets,
        mhc_ref,
        head_dim,
        rope_dim,
    )

    mhc_fused = mhc_cache.clone()
    grid = (num_rows,)
    M._dsv4_norm_rope_fp8_masked_store_kernel[grid](
        pooled,
        norm_weight.contiguous(),
        cos_table,
        sin_table,
        row_position_id,
        row_valid,
        mhc_page_ids,
        mhc_page_offsets,
        mhc_fused,
        mhc_fused.stride(0),
        mhc_fused.stride(1),
        mhc_fused.stride(2),
        int(cos_table.stride(0)),
        num_rows,
        head_dim,
        nope_dim,
        dh,
        float(eps),
        float(1.0 / head_dim),
        FP8_BLOCK=64,
        NUM_FP8_BLOCKS=nope_dim // 64,
        BLOCK_D=triton.next_power_of_2(head_dim),
        MAX_VAL=448.0,
        MIN_VAL=1.0e-4,
        num_warps=4,
    )

    assert torch.equal(mhc_fused, mhc_ref), (
        f"fused rmsnorm+rope+fp8+store differs from the two-stage reference "
        f"(exact={(mhc_fused == mhc_ref).float().mean().item():.6f}, "
        f"max_abs={(mhc_fused.float() - mhc_ref.float()).abs().max().item():.3e})"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
def test_ratio128_paged_pool_matches_gather_pool():
    assert M._HAS_TRITON, "test requires triton"
    num_rows = 3
    torch.manual_seed(300 + num_rows)
    head_dim, ratio, tokens_per_block = 512, 128, 8
    state_dim = head_dim
    dtype = torch.bfloat16

    max_pos = ratio + tokens_per_block
    pages_per_seq = (max_pos + tokens_per_block - 1) // tokens_per_block
    cu_num_pages = torch.tensor(
        [0, *torch.tensor([pages_per_seq] * num_rows).cumsum(0).tolist()],
        dtype=torch.long,
        device=DEV,
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=DEV)
    kv_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=DEV)
    gate_cache = torch.randn(total_pages, tokens_per_block, state_dim, device=DEV)
    ape = torch.randn(ratio, state_dim, device=DEV)

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=DEV)
    positions = torch.arange(ratio, dtype=torch.long, device=DEV).view(1, -1).expand(num_rows, -1)
    page_ids, page_offsets, _ = M._decode_page_ids_and_offsets(
        kv_cache, seq_idx, positions, cu_num_pages, cache_loc
    )

    pooled_fused = M._paged_compress_pool(
        kv_cache, gate_cache, page_ids, page_offsets, ape, ratio, head_dim, dtype
    )

    # Eager reference: gather -> cast -> ape add -> shipped compress_pool op.
    kv_state = kv_cache[page_ids, page_offsets].to(dtype)
    gate_state = gate_cache[page_ids, page_offsets].to(dtype)
    kv = kv_state[..., :head_dim]
    gate = gate_state[..., :head_dim] + ape[:, :head_dim].to(dtype)
    pooled_ref = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)

    # Only the fp32 reduction order differs (fixed vs autotuned num_warps).
    exact = (pooled_fused == pooled_ref).float().mean().item()
    max_abs = (pooled_fused.float() - pooled_ref.float()).abs().max().item()
    assert exact >= 0.99, f"only {exact:.5f} bf16-exact (max_abs={max_abs:.2e})"
    torch.testing.assert_close(pooled_fused.float(), pooled_ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_all_invalid_steps_leave_cache_untouched(compress_ratio):
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio, num_rows=3, seed=400 + compress_ratio)
    # Only non-completing positions, incl. one before / one after a completion.
    input_pos = torch.tensor(
        [compress_ratio - 2, compress_ratio, compress_ratio + compress_ratio // 2],
        dtype=torch.long,
        device=DEV,
    )
    inp["input_pos"] = input_pos
    inp["position_ids_decode"] = input_pos.clone()
    if compress_ratio == 4:
        outs = M.deepseek_v4_sparse_prepare_decode_page_addr(
            inp["input_pos"],
            inp["position_ids_decode"],
            inp["cu_num_pages"],
            inp["cache_loc"],
            int(inp["compressor_kv_cache"].shape[1]),
            inp["max_compressed_len"],
        )
        inp["overlap_page_map"] = (outs[2][:3], outs[3][:3], outs[4][:3])

    old = input_pos // compress_ratio
    new = (input_pos + 1) // compress_ratio
    assert not bool((new > old).any()), "test setup must produce only non-completing steps"

    mhc = inp["mhc_cache"].clone()
    _run_update(inp, mhc)
    assert torch.equal(mhc, inp["mhc_cache"]), "all-invalid decode step mutated the mhc cache"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
def test_ratio128_paged_pool_row_valid_gate():
    assert M._HAS_TRITON, "test requires triton"
    torch.manual_seed(500)
    head_dim, ratio, tokens_per_block = 512, 128, 8
    dtype = torch.bfloat16
    num_rows = 3

    max_pos = ratio + tokens_per_block
    pages_per_seq = (max_pos + tokens_per_block - 1) // tokens_per_block
    cu_num_pages = torch.tensor(
        [0, *torch.tensor([pages_per_seq] * num_rows).cumsum(0).tolist()],
        dtype=torch.long,
        device=DEV,
    )
    total_pages = int(cu_num_pages[-1].item())
    cache_loc = torch.arange(total_pages, dtype=torch.long, device=DEV)
    kv_cache = torch.randn(total_pages, tokens_per_block, head_dim, device=DEV)
    gate_cache = torch.randn(total_pages, tokens_per_block, head_dim, device=DEV)
    ape = torch.randn(ratio, head_dim, device=DEV)

    seq_idx = torch.arange(num_rows, dtype=torch.long, device=DEV)
    positions = torch.arange(ratio, dtype=torch.long, device=DEV).view(1, -1).expand(num_rows, -1)
    page_ids, page_offsets, _ = M._decode_page_ids_and_offsets(
        kv_cache, seq_idx, positions, cu_num_pages, cache_loc
    )
    row_valid = torch.tensor([True, False, True], device=DEV)

    pooled_ungated = M._paged_compress_pool(
        kv_cache, gate_cache, page_ids, page_offsets, ape, ratio, head_dim, dtype
    )
    pooled_gated = M._paged_compress_pool(
        kv_cache,
        gate_cache,
        page_ids,
        page_offsets,
        ape,
        ratio,
        head_dim,
        dtype,
        row_valid=row_valid,
    )
    # Invalid rows of the gated output are unwritten by contract; compare valid only.
    assert torch.equal(pooled_gated[row_valid], pooled_ungated[row_valid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_row_valid_gate_under_cudagraph_replay(compress_ratio):
    # The gate must be data-driven per replay (production captured-decode pattern),
    # not baked in at capture time.
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio, num_rows=2, seed=600 + compress_ratio)
    # Row 0 completes a compressed row (input_pos == ratio-1); row 1 does not.
    input_pos = torch.tensor([compress_ratio - 1, compress_ratio], dtype=torch.long, device=DEV)
    inp["input_pos"] = input_pos
    inp["position_ids_decode"] = input_pos.clone()
    tokens_per_block = int(inp["compressor_kv_cache"].shape[1])
    if compress_ratio == 4:
        outs = M.deepseek_v4_sparse_prepare_decode_page_addr(
            inp["input_pos"],
            inp["position_ids_decode"],
            inp["cu_num_pages"],
            inp["cache_loc"],
            tokens_per_block,
            inp["max_compressed_len"],
        )
        inp["overlap_page_map"] = (outs[2][:2], outs[3][:2], outs[4][:2])

    ratio = compress_ratio
    max_compressed_len = inp["max_compressed_len"]
    old_completed = input_pos // ratio
    new_completed = (input_pos + 1) // ratio
    true_valid = (new_completed > old_completed) & (old_completed < max_compressed_len)
    assert true_valid.tolist() == [True, False]
    row_idx = old_completed.clamp(min=0, max=max_compressed_len - 1)
    row_position_id = inp["position_ids_decode"].to(torch.long) - (input_pos - row_idx * ratio)
    row_logical_pos = row_idx * ratio
    mhc_page_ids, mhc_page_offsets, _ = M._decode_page_ids_and_offsets(
        inp["mhc_cache"], inp["seq_idx"], row_logical_pos, inp["cu_num_pages"], inp["cache_loc"]
    )
    pos_page_ids = None
    pos_page_offsets = None
    if compress_ratio == 128:
        offsets = torch.arange(ratio, dtype=torch.long, device=DEV)
        positions = (row_idx.to(torch.long) * ratio).unsqueeze(1) + offsets.view(1, -1)
        pos_page_ids, pos_page_offsets, _ = M._decode_page_ids_and_offsets(
            inp["compressor_kv_cache"],
            inp["seq_idx"],
            positions,
            inp["cu_num_pages"],
            inp["cache_loc"],
        )

    row_valid_buf = true_valid.clone()  # mutated in place between replays

    def _run_with_meta(mhc):
        _run_update(
            inp,
            mhc,
            update_meta=(
                row_valid_buf,
                row_position_id,
                mhc_page_ids,
                mhc_page_offsets,
                pos_page_ids,
                pos_page_offsets,
            ),
        )

    mhc_orig = inp["mhc_cache"].clone()

    # Uncaptured reference with the true (mixed) validity.
    mhc_ref = mhc_orig.clone()
    _run_with_meta(mhc_ref)
    assert not torch.equal(mhc_ref, mhc_orig), "valid row must have been written"

    # Warm up (triton JIT compile, allocator pools) outside capture.
    mhc_graph = mhc_orig.clone()
    for _ in range(2):
        _run_with_meta(mhc_graph)
    mhc_graph.copy_(mhc_orig)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _run_with_meta(mhc_graph)
    torch.cuda.synchronize()
    assert torch.equal(mhc_graph, mhc_orig)  # capture records, does not execute

    # Replay 1: all-invalid step leaves the cache byte-unchanged.
    row_valid_buf.fill_(False)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(mhc_graph, mhc_orig), "all-invalid cudagraph replay mutated the mhc cache"

    # Replay 2: true validity restores the uncaptured bytes.
    row_valid_buf.copy_(true_valid)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(mhc_graph, mhc_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_update_with_hoisted_meta_byte_exact(compress_ratio):
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio, num_rows=3, seed=403 + compress_ratio)

    mhc_local = inp["mhc_cache"].clone()
    _run_update(inp, mhc_local, update_meta=None)  # per-layer metadata

    mhc_hoisted = inp["mhc_cache"].clone()
    _run_update(inp, mhc_hoisted, update_meta=inp["update_meta"])  # hoisted metadata

    assert torch.equal(mhc_local, mhc_hoisted)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_update_reconstruction_dtype_anchored_on_cache(compress_ratio):
    # bf16 vs fp32 decode rows must not change a byte: the reconstruction reads the
    # current token from the fp32 caches and anchors its compute dtype there.
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio, num_rows=3, seed=920)

    mhc_bf16 = inp["mhc_cache"].clone()
    _run_update(inp, mhc_bf16, update_meta=inp["update_meta"])

    inp_f32 = dict(inp)
    inp_f32["compressor_kv_decode"] = inp["compressor_kv_decode"].float()
    inp_f32["compressor_gate_decode"] = inp["compressor_gate_decode"].float()
    mhc_f32 = inp["mhc_cache"].clone()
    _run_update(inp_f32, mhc_f32, update_meta=inp["update_meta"])

    assert torch.equal(mhc_bf16, mhc_f32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
