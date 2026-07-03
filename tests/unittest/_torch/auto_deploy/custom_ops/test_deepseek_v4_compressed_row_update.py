# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused ratio-4 compressed-row cache update (idea_0007).

``_update_decode_compressed_caches`` reconstructs the just-completed main-compressor
compressed row at decode time and stores it into ``mhc_cache`` on the ~1-in-ratio steps
that complete a row.  idea_0007 collapses the ratio-4 (overlap) reconstruction swarm
(gather / slice / ape-add / where / cat / pool / rmsnorm), the rope/fake-fp8 tail
(``_apply_compressed_rope_and_quantize``, rotate=False), the ``cos``/``sin`` gathers and
the validity-masked store into two Triton kernels
(``_dsv4_compressed_row_r4_front_kernel`` + ``_dsv4_rope_fp8_masked_store_kernel``).

The fused path replicates the eager numerics -- fp32-internal softmax pool and RMSNorm
with bf16 rounding at the same points as the reference, a byte-identical block fake-fp8
on the nope slice and an interleaved RoPE on the pe slice -- so the stored ``mhc_cache``
must match the eager op-by-op path up to the ``rsqrt`` primitive (bf16-absorbed) and the
rope FMA folding (<=1 ULP).  These tests pin that equivalence separately for the ratio-4
(fused) and ratio-128 (untouched fallback) modes so a regression in one cannot hide in the
other, and they exercise both row-completing (valid) and non-completing (invalid, no-write)
decode steps.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M

DEV = "cuda"


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

    # Enough pages per sequence to cover every position the reconstruction reads.
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

    overlap_page_map = None
    if compress_ratio == 4:
        outs = M.deepseek_v4_sparse_prepare_decode_page_addr(
            input_pos, cu_num_pages, cache_loc, tokens_per_block, max_compressed_len
        )
        overlap_page_map = (outs[2][:num_rows], outs[3][:num_rows], outs[4][:num_rows])

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
    )


def _run(inp, mhc):
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
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 3])
def test_ratio4_fused_matches_eager(monkeypatch, num_rows):
    """Ratio-4 (overlap) fused two-kernel path == eager op-by-op reconstruction+store."""
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio=4, num_rows=num_rows, seed=100 + num_rows)

    mhc_fused = inp["mhc_cache"].clone()
    _run(inp, mhc_fused)  # real _HAS_TRITON -> fused

    monkeypatch.setattr(M, "_HAS_TRITON", False)
    mhc_eager = inp["mhc_cache"].clone()
    _run(inp, mhc_eager)  # forced fallback -> eager

    # End-to-end the whole chain (pool -> rmsnorm -> fp8/rope -> store) matches the eager
    # path up to the rsqrt primitive: the rsqrt <=1 ULP in the normed row propagates into
    # the fp8 quant, so even the nope slice is only ULP-close end-to-end (byte-exactness of
    # the fp8 stage GIVEN an identical normed input is pinned separately below).
    exact = (mhc_fused == mhc_eager).float().mean().item()
    max_abs = (mhc_fused.float() - mhc_eager.float()).abs().max().item()
    assert exact >= 0.95, f"exact bf16 match ratio {exact} too low (max_abs={max_abs})"
    torch.testing.assert_close(mhc_fused.float(), mhc_eager.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 3])
def test_rope_fp8_store_kernel_byte_exact(num_rows):
    """Stage-2 kernel vs the eager rope/fp8 tail + masked store, fed an IDENTICAL normed row.

    Isolates the new math from the upstream rsqrt: the fake-fp8 nope slice must be
    byte-identical (op order unchanged) and the rope pe slice equal to <=1 ULP.  Invalid
    rows are left untouched (no-write).
    """
    assert M._HAS_TRITON, "test requires triton"
    import triton

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
    grid = (num_rows,)
    M._dsv4_rope_fp8_masked_store_kernel[grid](
        normed,
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
        FP8_BLOCK=64,
        NUM_FP8_BLOCKS=nope_dim // 64,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_DH=triton.next_power_of_2(dh),
        MAX_VAL=448.0,
        MIN_VAL=1.0e-4,
        num_warps=4,
    )

    # Eager reference: gather cos/sin, run the shipped rope/fp8 tail, masked store.
    cos_g = cos_table[row_position_id]
    sin_g = sin_table[row_position_id]
    compressed = M._apply_compressed_rope_and_quantize(normed, cos_g, sin_g, rope_dim, rotate=False)
    mhc_ref = mhc_cache.clone()
    for r in range(num_rows):
        if bool(row_valid[r]):
            mhc_ref[int(mhc_page_ids[r]), int(mhc_page_offsets[r])] = compressed[r].to(dtype)

    # Invalid rows: byte-identical to the untouched original.
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
@pytest.mark.parametrize("num_rows", [1, 2])
def test_ratio128_unchanged(monkeypatch, num_rows):
    """Ratio-128 (non-overlap) never takes the fused path -> flag on/off must be identical."""
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio=128, num_rows=num_rows, seed=200 + num_rows)

    mhc_on = inp["mhc_cache"].clone()
    _run(inp, mhc_on)

    monkeypatch.setattr(M, "_HAS_TRITON", False)
    mhc_off = inp["mhc_cache"].clone()
    _run(inp, mhc_off)

    # The masked store differs (triton vs eager where/write-back) but both must produce
    # the identical cache contents; ratio-128 reconstruction is byte-for-byte untouched.
    torch.testing.assert_close(mhc_on.float(), mhc_off.float(), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q", "-s"]))
