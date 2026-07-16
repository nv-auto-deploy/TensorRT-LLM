# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the fused compressed-row cache update.

``_update_decode_compressed_caches`` reconstructs the just-completed main-compressor
compressed row at decode time and stores it into ``mhc_cache`` on the ~1-in-ratio steps
that complete a row.  the fused path collapses the ratio-4 (overlap) reconstruction swarm
(gather / slice / ape-add / where / cat / pool / rmsnorm), the rope/fake-fp8 tail
(``_apply_compressed_rope_and_quantize``, rotate=False), the ``cos``/``sin`` gathers and
the validity-masked store into two Triton kernels
(``_dsv4_compressed_row_r4_front_kernel`` + ``_dsv4_rope_fp8_masked_store_kernel``).

the fused path extends the same collapse to the ratio-128 (dense, non-overlap) layers, whose
``[ratio, head_dim]`` pool tile is too large for the ratio-4 one-program-per-row strategy:
the pool is D-tiled (``_dsv4_paged_compress_pool_kernel``), RMSNorm is the shipped fused
``triton_rms_norm`` (``_compressor_rms_norm``), and the rope/fp8/validity-masked-store tail
is the same shared ``_dsv4_rope_fp8_masked_store_kernel`` the fused path introduced.

The fused path folds that rope/fp8/masked-store tail into the producing kernels as a
register-fed final stage (``_dsv4_rope_fp8_store_tail``): the ratio-4 front kernel now
stores the mhc row directly, and the ratio-128 rmsnorm + tail collapse into
``_dsv4_norm_rope_fp8_masked_store_kernel`` -- removing one launch per compressed layer
per decode step plus the ``[N, head_dim]`` normed-row round-trip.  The original
stage-2 kernel (and its ``_launch_compressed_rope_fp8_store`` launcher) lives on
byte-for-byte in THIS file (moved out of the production module) as the frozen
pre-fold reference the fold is pinned against (``torch.equal``, whole cache) below.

The fused path replicates the eager numerics -- fp32-internal softmax pool and RMSNorm
with bf16 rounding at the same points as the reference, a byte-identical block fake-fp8
on the nope slice and an interleaved RoPE on the pe slice -- so the stored ``mhc_cache``
must match the eager op-by-op path up to the ``rsqrt`` primitive (bf16-absorbed) and the
rope FMA folding (<=1 ULP).  The ratio-128 paged pool is even byte-identical to
``gather + ape + deepseek_v4_compress_pool`` (same rounding points; pinned separately).
These tests pin that equivalence for the ratio-4 and ratio-128 modes so a regression in one
cannot hide in the other, and they exercise both row-completing (valid) and non-completing
(invalid, no-write) decode steps across multi-page position spans.

Row gating runs every program of those kernels behind ``row_valid`` at entry: a compressed
row completes only once every ``ratio`` decode steps, and invalid rows were already
no-write, so on the other steps the programs now retire after one scalar load instead of
paying the paged loads / pool / rmsnorm / rope / fp8 math whose result was discarded.
The added tests pin the gate's invariants: all-invalid steps (including the boundary
positions one before / one after a completion) leave the whole cache byte-identical,
the gated pool is byte-identical to the ungated pool on valid rows, and the gate keeps
working under CUDA-graph capture/replay when only the ``row_valid`` buffer content flips
between replays -- the exact production decode pattern.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.attention.deepseek_v4_sparse_attention as M

DEV = "cuda"


# ---------------------------------------------------------------------------
# Frozen pre-fold stage-2 reference, moved verbatim from the production op
# module when the rope/fp8/masked-store tail was folded into the producing
# kernels (`_dsv4_rope_fp8_store_tail`).  Kept byte-for-byte so the bit-exact
# pins below keep their meaning: this kernel IS the launch the fold removed.
# ---------------------------------------------------------------------------
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
        cossin_row_stride,  # cos/sin row stride (== DH)
        N,
        HEAD_DIM,
        NOPE_DIM,  # HEAD_DIM - ROPE_DIM (multiple of FP8_BLOCK)
        DH,  # ROPE_DIM // 2
        FP8_BLOCK: tl.constexpr,  # 64 (fake-fp8 group width)
        NUM_FP8_BLOCKS: tl.constexpr,  # NOPE_DIM // FP8_BLOCK
        BLOCK_D: tl.constexpr,  # next_pow2(HEAD_DIM)
        BLOCK_DH: tl.constexpr,  # next_pow2(DH)
        MAX_VAL: tl.constexpr,  # 448.0 (e4m3 absmax)
        MIN_VAL: tl.constexpr,  # 1e-4 (amax floor)
    ):
        """Fused main-compressor rope + fake-fp8 + validity-masked store (stage 2).

        No longer launched by the production compressed-row updates -- the tail is
        folded into the producing kernels as the register-fed
        ``_dsv4_rope_fp8_store_tail`` epilogue (ratio-4 front kernel /
        ``_dsv4_norm_rope_fp8_masked_store_kernel``).  Retained byte-for-byte as the
        isolated tail-math reference the compressed-row update unit tests pin the
        fold against (via ``_launch_compressed_rope_fp8_store``).

        One program per decode row ``b``.  Reads a post-rmsnorm ``[N, HEAD_DIM]``
        normed row and, only when ``row_valid[b]``, writes the
        fully reconstructed compressed row to ``cache[page_ids[b], page_offsets[b]]``.
        Collapses the rope-tail chain ``_apply_compressed_rope_and_quantize``
        (rotate=False: block fake-fp8 on the nope slice + interleaved RoPE/concat on
        the pe slice) plus the ``cos``/``sin`` gathers plus the ``_masked_paged_store``
        into a single launch.

        Byte-identical to the reference on the nope slice -- the block amax, the
        ``scale = 2**ceil(log2(clamp_min(amax,1e-4)/448))``, the ``clamp -> bf16 ->
        fp32`` round-trip and the ``* scale`` reproduce ``fake_fp8_act_quant`` exactly
        -- and equal to <=1 ULP on the pe slice (FMA folding, as for
        ``deepseek_v4_fused_rope_concat``).  Invalid rows store nothing, leaving the
        slot byte-identical to the prior read-old + write-back no-op.
        """
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

        # --- fake-fp8 block quant on the nope slice [0, NOPE_DIM) ---
        d = tl.arange(0, BLOCK_D)
        nmask = d < NOPE_DIM
        nope = tl.load(nrow + d, mask=nmask, other=0.0).to(tl.float32)  # bf16 -> fp32
        blk = d // FP8_BLOCK
        scale_per_d = tl.full([BLOCK_D], 1.0, tl.float32)  # 1.0 outside the nope slice
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

        # --- interleaved RoPE on the pe slice [NOPE_DIM, HEAD_DIM) ---
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
        normed: torch.Tensor,  # [N, head_dim] post-rmsnorm rows (activation dtype)
        cos_table: torch.Tensor,  # [n_pos, rope_dim//2]
        sin_table: torch.Tensor,  # [n_pos, rope_dim//2]
        row_position_id: torch.Tensor,  # [N] int64, already clamped into [0, n_pos)
        row_valid: torch.Tensor,  # [N] bool -- store the row iff valid
        mhc_page_ids: torch.Tensor,  # [N] int64 write page id per row
        mhc_page_offsets: torch.Tensor,  # [N] int64 in-page offset per row
        mhc_cache: torch.Tensor,  # [P, T, head_dim] paged mhc cache (mutated in place)
        head_dim: int,
        rope_dim: int,
    ) -> None:
        """Standalone stage-2 tail: fp8(nope) + interleaved RoPE(pe) + validity-masked store.

        Launches ``_dsv4_rope_fp8_masked_store_kernel`` over the ``[N, head_dim]``
        post-rmsnorm rows.  No longer called by the production compressed-row updates -- the tail is
        folded into the producing kernels as the register-fed
        ``_dsv4_rope_fp8_store_tail`` epilogue -- but retained (with the kernel, byte-for-byte)
        as the isolated tail-math reference the compressed-row update unit tests pin the fold
        against.  Ratio-agnostic: both the ratio-4 (overlap) and ratio-128 (dense) updates
        produce an identical ``[N, head_dim]`` normed row and write the same ``head_dim``-wide
        mhc row.  Invalid rows write nothing (byte-identical to the prior read-old +
        write-back no-op).
        """
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
            input_pos, position_ids, cu_num_pages, cache_loc, tokens_per_block, max_compressed_len
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
    _dsv4_rope_fp8_masked_store_kernel[grid](
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
@pytest.mark.parametrize("num_rows", [1, 3])
@pytest.mark.parametrize("head_dim,rope_dim", [(512, 64), (256, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_norm_rope_store_fused_kernel_matches_two_stage(num_rows, head_dim, rope_dim, dtype):
    """Byte-pin: the fused rmsnorm+rope+fp8+masked-store kernel == the old chain.

    Feeds an identical pooled row to the two-stage reference
    (``_compressor_rms_norm`` -> ``_launch_compressed_rope_fp8_store``, i.e. the
    ``rms_norm_kernel`` launch plus the byte-for-byte frozen
    ``_dsv4_rope_fp8_masked_store_kernel`` kept in this file) and to the new single
    kernel, then compares
    the ENTIRE mhc cache with ``torch.equal``.  This pins bit-identity of (a) the
    in-kernel RMSNorm replication of ``rms_norm_kernel`` (same ``sum(x*x) * (1/N)``
    mean, ``x / sqrt(var + eps)`` and left-weight multiply, same BLOCK/num_warps
    reduction shape) and (b) the shared register-fed
    ``_dsv4_rope_fp8_store_tail`` epilogue -- the reshape/split pe deinterleave, the
    rope FMA expressions, the fake-fp8 block quant and the validity-masked store --
    which the folded ratio-4 front kernel reuses verbatim.  Untouched cache slots and
    invalid (no-write) rows are covered by the whole-cache compare.
    """
    assert M._HAS_TRITON, "test requires triton"
    import triton

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

    # Reference: the removed two-stage chain (rms_norm_kernel launch + stage-2 kernel).
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

    # Fused single kernel.
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
@pytest.mark.parametrize("num_rows", [1, 2])
def test_ratio128_fused_matches_eager(monkeypatch, num_rows):
    """Ratio-128 (dense) fused three-launch path == eager op-by-op reconstruction+store.

    Ratio-128: the fused path (D-tiled paged pool -> rmsnorm -> rope/fp8/masked-store) must
    match the eager ``_batched_compressed_rows_from_paged_state`` (non-overlap) + eager
    where/write-back store.  The paged pool is bit-identical to ``gather + ape + pool`` and
    RMSNorm is unchanged, so end-to-end the only deviation is the tail rope FMA (<=1 ULP);
    the nope slice is byte-exact.  ``_build_inputs`` mixes a row-completing (valid) row with
    a non-completing (invalid, no-write) row, and the 128-token position span crosses many
    ``tokens_per_block=8`` page boundaries.
    """
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio=128, num_rows=num_rows, seed=200 + num_rows)

    mhc_fused = inp["mhc_cache"].clone()
    _run(inp, mhc_fused)  # real _HAS_TRITON -> fused ratio-128 path

    monkeypatch.setattr(M, "_HAS_TRITON", False)
    mhc_eager = inp["mhc_cache"].clone()
    _run(inp, mhc_eager)  # forced fallback -> eager reconstruction + where/write-back store

    exact = (mhc_fused == mhc_eager).float().mean().item()
    max_abs = (mhc_fused.float() - mhc_eager.float()).abs().max().item()
    assert exact >= 0.95, f"exact bf16 match ratio {exact} too low (max_abs={max_abs})"
    torch.testing.assert_close(mhc_fused.float(), mhc_eager.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("num_rows", [1, 3])
def test_ratio128_paged_pool_matches_gather_pool(num_rows):
    """Isolate the ratio-128 paged pool front kernel from the rsqrt/rope tail.

    ``_paged_compress_pool`` (paged gather + ape add + softmax pool) reproduces the eager
    ``gather + ape + deepseek_v4_compress_pool`` it collapses to <=1 ULP: it rounds the fp32
    cache reads to the activation dtype (== the gather ``.to(dtype)``), adds the ape in the
    activation dtype, and runs the same fp32-internal per-channel softmax pool.  The only
    deviation is the fp32 reduction order over the 128-slot ratio axis (this kernel fixes
    ``num_warps=4`` while ``deepseek_v4_compress_pool`` autotunes it), which flips at most a
    handful of near-zero channels by one bf16 ULP.  The 128-token span crosses many
    ``tokens_per_block`` page boundaries.
    """
    assert M._HAS_TRITON, "test requires triton"
    torch.manual_seed(300 + num_rows)
    head_dim, ratio, tokens_per_block = 512, 128, 8
    state_dim = head_dim  # channels == 1 for the dense (non-overlap) compressor
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
    # Each row reconstructs positions [0, ratio) -> spans ratio/tokens_per_block pages.
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

    exact = (pooled_fused == pooled_ref).float().mean().item()
    max_abs = (pooled_fused.float() - pooled_ref.float()).abs().max().item()
    assert exact >= 0.99, (
        f"paged pool vs gather + ape + compress_pool: only {exact:.5f} bf16-exact "
        f"(max_abs={max_abs:.2e})"
    )
    torch.testing.assert_close(pooled_fused.float(), pooled_ref.float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_all_invalid_steps_leave_cache_untouched(compress_ratio):
    """Row gating: non-completing steps must write nothing, byte-for-byte.

    Overrides ``input_pos`` with only non-completing positions, including both
    validity boundaries -- one step BEFORE a row completes (``ratio-2``, e.g. 2/126)
    and one step AFTER (``ratio``, e.g. 4/128) -- so ``row_valid`` is all-false and
    the early-exited fused path must leave the entire mhc cache byte-identical.
    (The completing boundary ``ratio-1`` is exercised by the mixed-validity tests
    above and the cudagraph test below.)
    """
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio=compress_ratio, num_rows=3, seed=400 + compress_ratio)
    # Boundary non-completing positions: ratio-2 (one before), ratio (one after), and a
    # mid-band position of the next block.
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
    _run(inp, mhc)  # real _HAS_TRITON -> fused, all rows early-exit
    assert torch.equal(mhc, inp["mhc_cache"]), "all-invalid decode step mutated the mhc cache"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
def test_ratio128_paged_pool_row_valid_gate():
    """Row gating: the gated pool is byte-identical to the ungated pool on valid rows.

    Invalid rows of the gated output are unwritten garbage by contract (their only
    consumer early-exits before reading them), so only valid rows are compared.
    """
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
    assert torch.equal(pooled_gated[row_valid], pooled_ungated[row_valid]), (
        "row_valid gate changed the pooled values of valid rows"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused kernel requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
def test_row_valid_gate_under_cudagraph_replay(compress_ratio):
    """Row gating: the early exit is data-gated per replay, not baked in at capture.

    Captures the fused decode update once with ``row_valid`` threaded in via the
    hoisted ``update_meta`` (the production prepare-op contract), then replays the
    SAME graph with different ``row_valid`` buffer contents: an all-false replay must
    leave the cache byte-identical, and a subsequent mixed-validity replay must write
    exactly what an uncaptured run with that validity writes -- including leaving the
    invalid row untouched.  This is the captured-decode production pattern (the launch
    replays every step; only the buffer content changes).
    """
    assert M._HAS_TRITON, "test requires triton"
    inp = _build_inputs(compress_ratio=compress_ratio, num_rows=2, seed=600 + compress_ratio)
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

    # Hoisted update metadata, mirroring the update_meta=None branch of
    # ``_update_decode_compressed_caches`` (and the prepare op).
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
    # Capture records the launches without executing them; the cache is still pristine.
    torch.cuda.synchronize()
    assert torch.equal(mhc_graph, mhc_orig)

    # Replay 1: all-invalid step -> every program early-exits, cache byte-unchanged.
    row_valid_buf.fill_(False)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(mhc_graph, mhc_orig), "all-invalid cudagraph replay mutated the mhc cache"

    # Replay 2: restore the true validity -> identical bytes to the uncaptured run.
    row_valid_buf.copy_(true_valid)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(mhc_graph, mhc_ref), (
        "valid-row cudagraph replay differs from the uncaptured reference"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q", "-s"]))
