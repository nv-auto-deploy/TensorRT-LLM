# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the rowwise direct-store M=1 decode GEMV.

``_w8a8_gemv_rowwise`` replaces both incumbent block-FP8 paths (full-K MMA and
split-K fill + atomic-matmul + cast) for the exact measured DeepSeek-V4-Flash
TP4 per-rank M=1 decode shapes. These tests pin:

* accuracy against an fp64 dequant -> matmul ground truth (strict decode bar),
  including the fused residual epilogue;
* that the funnel actually dispatches the rowwise kernel for every table shape
  (byte-equality with a direct kernel call -- the kernel is deterministic);
* the residual epilogue's exact round-then-add-then-round semantics (matches
  the full-K HAS_RESIDUAL epilogue bit-for-bit);
* run-to-run determinism (the split-K path this replaces is atomic-order
  nondeterministic; the rowwise kernel must not be);
* the dispatch gate: M != 1, off-table shapes, non-128 quant blocks, and
  non-contiguous operands all keep the incumbent paths.
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _W8A8_TP4_M1_ROWWISE_CFG,
    _W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG,
    _safe_act_quant,
    _use_rowwise_gemv,
    _w8a8_block_fp8_matmul_triton,
    _w8a8_gemv_rowwise,
)

ROWWISE_SHAPES = sorted(_W8A8_TP4_M1_ROWWISE_CFG)  # [(N, K), ...]
BLOCK = 128


def _fp8_supported():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def _quant_weight_block_fp8(w: torch.Tensor, block_n: int, block_k: int):
    """Block-quantize a bf16 weight to (fp8 weight, fp32 per-block scale_inv)."""
    N, K = w.shape
    sn, sk = (N + block_n - 1) // block_n, (K + block_k - 1) // block_k
    w_f32 = w.float()
    scale = torch.empty(sn, sk, device=w.device, dtype=torch.float32)
    q = torch.empty_like(w_f32)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    for i in range(sn):
        for j in range(sk):
            blk = w_f32[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k]
            amax = blk.abs().amax().clamp(min=1e-6)
            scale[i, j] = amax / fp8_max
            q[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k] = blk / scale[i, j]
    return q.to(torch.float8_e4m3fn), scale


def _make_case(N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK)
    b_fp8, b_s = _quant_weight_block_fp8(b, BLOCK, BLOCK)
    return a_fp8, a_s, b_fp8, b_s


def _ref_fp64(a_fp8, a_s, b_fp8, b_s):
    K = a_fp8.shape[-1]
    N = b_fp8.shape[0]
    a = a_fp8.double() * a_s.double().repeat_interleave(BLOCK, dim=-1)[:, :K]
    bs_e = b_s.double().repeat_interleave(BLOCK, dim=0)[:N].repeat_interleave(BLOCK, dim=1)[:, :K]
    return (a @ (b_fp8.double() * bs_e).t()).float()


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("N,K", ROWWISE_SHAPES)
def test_rowwise_decode_strict_and_dispatched(N, K):
    """Funnel output at M=1 is the rowwise kernel's and meets the strict bar."""
    a_fp8, a_s, b_fp8, b_s = _make_case(N, K)
    assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a_fp8, b_fp8, a_s, b_s)
    out = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    direct = _w8a8_gemv_rowwise(a_fp8, b_fp8, a_s, b_s, BLOCK, BLOCK, torch.bfloat16, N, K)
    # Deterministic kernel: the funnel must have taken the rowwise path.
    assert torch.equal(out, direct)

    ref = _ref_fp64(a_fp8, a_s, b_fp8, b_s)
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"N={N} K={K}: max_abs_err/amax={max_rel:.4e}"
    # bf16-boundary flips vs the rounded ground truth must stay rare (parity
    # with the incumbent MMA kernel's own behavior on these shapes).
    mism = (out != ref.to(torch.bfloat16)).sum().item()
    assert mism <= max(1, N // 4096), f"N={N} K={K}: {mism} bf16 mismatches vs ref"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("N,K", ROWWISE_SHAPES)
def test_rowwise_deterministic(N, K):
    a_fp8, a_s, b_fp8, b_s = _make_case(N, K, seed=1)
    outs = [
        _w8a8_block_fp8_matmul_triton(
            a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
        )
        for _ in range(3)
    ]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_rowwise_residual_epilogue_exact():
    """residual path == round(acc) -> fp32 add -> round, bit-for-bit."""
    N, K = 4096, 512  # the shared-w2 merge-add site
    a_fp8, a_s, b_fp8, b_s = _make_case(N, K, seed=2)
    residual = torch.randn(1, N, device="cuda", dtype=torch.bfloat16)
    out_res = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16, residual=residual
    )
    out_plain = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    expected = (out_plain.float() + residual.float()).to(torch.bfloat16)
    assert torch.equal(out_res, expected)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_rowwise_dispatch_gate():
    """Only exact table shapes at M=1 with 128x128 blocks and flat rows route."""
    N, K = 1024, 4096
    a_fp8, a_s, b_fp8, b_s = _make_case(N, K)
    assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a_fp8, b_fp8, a_s, b_s)
    # M != 1 keeps the split-K / full-K paths.
    assert not _use_rowwise_gemv(2, N, K, BLOCK, BLOCK, a_fp8, b_fp8, a_s, b_s)
    # Off-table shape.
    assert not _use_rowwise_gemv(1, N + BLOCK, K, BLOCK, BLOCK, a_fp8, b_fp8, a_s, b_s)
    # Non-128 quant blocks.
    assert not _use_rowwise_gemv(1, N, K, 64, 128, a_fp8, b_fp8, a_s, b_s)
    assert not _use_rowwise_gemv(1, N, K, 128, 64, a_fp8, b_fp8, a_s, b_s)
    # Strided (non-row-contiguous) weight falls back.
    b_strided = b_fp8.t().contiguous().t()
    assert b_strided.stride(1) != 1
    assert not _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a_fp8, b_strided, a_s, b_s)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_m2_keeps_incumbent_path():
    """M=2 decode must still produce correct results via the old paths."""
    N, K = 1024, 4096
    torch.manual_seed(3)
    a = torch.randn(2, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK)
    b_fp8, b_s = _quant_weight_block_fp8(b, BLOCK, BLOCK)
    out = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    ref = _ref_fp64(a_fp8, a_s, b_fp8, b_s)
    scale = ref.abs().amax().clamp(min=1e-6)
    assert ((out.float() - ref).abs().amax() / scale).item() < 1.5e-2


# ---------------------------------------------------------------------------
# Fused activation-quant prologue (QUANT_PROLOGUE=True, v6)
# ---------------------------------------------------------------------------
#
# ``As=None`` routes the raw bf16 activation into the rowwise kernel, which
# replicates ``_act_quant_kernel``'s ue8m0 quant (amax -> pow-2 scale -> RTNE
# fp8 cast) per 128-group in its prologue. The quant math is bit-identical by
# construction; the *consumer* reduction, however, runs under a re-tuned
# launch config (``_W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG``) whose different
# rows/CTA / warps change the fp32 summation-tree order, so near-cancellation
# outputs may round to the adjacent bf16 value (<= 1 ULP; measured ~5e-5 of
# outputs). That is the same acceptance class as this kernel's own landing
# (vs the incumbent MMA kernels) and strictly tighter than the split-K path's
# pre-existing atomic-order wobble. Exact-arithmetic inputs (all partial
# products and sums fp32-exact) remove the rounding sensitivity entirely and
# therefore must be torch.equal at every shape and config.


def _standalone_ue8m0(a, b_fp8, b_s, residual=None):
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK, "ue8m0")
    return _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16, residual=residual
    )


def _fused_ue8m0(a, b_fp8, b_s, residual=None):
    return _w8a8_block_fp8_matmul_triton(
        a,
        b_fp8,
        None,
        b_s,
        [BLOCK, BLOCK],
        output_dtype=torch.bfloat16,
        residual=residual,
        input_scale_fmt="ue8m0",
    )


def _assert_bf16_adjacent(out, ref, max_flips):
    """Fused vs standalone may differ only by fp32 reduction order: few flips,
    each either 1 bf16 ULP or a negligible absolute diff at near-cancellation
    outputs (a multi-ULP step there is ULP inflation of a ~0 value)."""
    neq = out != ref
    flips = int(neq.sum())
    assert flips <= max_flips, f"{flips} mismatches (allowed {max_flips})"
    if flips:
        steps = (out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()[neq]
        adiff = (out.float() - ref.float()).abs()[neq]
        scale = float(ref.float().abs().mean()) + 1e-12
        big = steps > 1
        assert bool((adiff[big] <= 1e-4 * scale).all()), (
            f"non-negligible mismatch beyond 1 bf16 ULP: max step {int(steps.max())}, "
            f"max adiff {float(adiff.max()):.3e} vs tensor scale {scale:.3e}"
        )


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("N,K", ROWWISE_SHAPES)
def test_rowwise_quant_prologue_matches_standalone(N, K):
    """Fused-prologue output vs standalone quant+GEMV on random bf16 data."""
    torch.manual_seed(11)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b, BLOCK, BLOCK)
    assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a, b_fp8, None, b_s)
    ref = _standalone_ue8m0(a, b_fp8, b_s)
    out = _fused_ue8m0(a, b_fp8, b_s)
    # Identical launch config does NOT pin FMA scheduling across the two
    # constexpr specializations, so no bit-exact special case on random data;
    # the exact-arithmetic test below is the hard bit-exact gate.
    _assert_bf16_adjacent(out, ref, max_flips=max(2, N // 1024))


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
@pytest.mark.parametrize("N,K", ROWWISE_SHAPES)
def test_rowwise_quant_prologue_exact_arithmetic(N, K):
    """Exact-arithmetic inputs must be torch.equal under ANY reduction order.

    Activations in {0, +-1, +-2, +-4} quantize losslessly (pow-2 scales, fp8-
    representable payloads) and every partial product/sum is fp32-exact, so any
    reassociation/FMA contraction yields the identical result: a hard bit-exact
    gate on the prologue's scale math and scale association.
    """
    torch.manual_seed(12)
    vals = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0], device="cuda")
    a = vals[torch.randint(0, 7, (1, K), device="cuda")].to(torch.bfloat16)
    a[0, :BLOCK] = 0.0  # all-zero group exercises the 1e-4 amax clamp
    w = torch.randint(-2, 3, (N, K), device="cuda").float()
    b_fp8 = w.to(torch.float8_e4m3fn)
    b_s = torch.ones(N // BLOCK, K // BLOCK, device="cuda", dtype=torch.float32)
    ref = _standalone_ue8m0(a, b_fp8, b_s)
    out = _fused_ue8m0(a, b_fp8, b_s)
    assert torch.equal(ref, out)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_rowwise_quant_prologue_nan_propagation():
    """NaN activations must poison both paths identically (cfg-equal shape)."""
    N, K = 16384, 1024  # standalone and prologue configs are identical here
    torch.manual_seed(13)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    a[0, 5] = float("nan")
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b, BLOCK, BLOCK)
    ref = _standalone_ue8m0(a, b_fp8, b_s)
    out = _fused_ue8m0(a, b_fp8, b_s)
    assert ref.isnan().all() and out.isnan().all()
    assert torch.equal(ref.view(torch.int16), out.view(torch.int16))


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_rowwise_quant_prologue_residual_epilogue():
    """Fused-prologue residual epilogue keeps round -> fp32 add -> round."""
    N, K = 4096, 512  # the shared-w2 merge-add site
    torch.manual_seed(14)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b, BLOCK, BLOCK)
    residual = torch.randn(1, N, device="cuda", dtype=torch.bfloat16)
    out_res = _fused_ue8m0(a, b_fp8, b_s, residual=residual)
    out_plain = _fused_ue8m0(a, b_fp8, b_s)
    expected = (out_plain.float() + residual.float()).to(torch.bfloat16)
    assert torch.equal(out_res, expected)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8")
def test_rowwise_quant_prologue_gate_and_cfg_parity():
    """Deferred-quant gating mirrors the fp8 gate; cfg tables cover same keys."""
    assert set(_W8A8_TP4_M1_ROWWISE_CFG) == set(_W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG)
    N, K = 1024, 4096
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    b_fp8, b_s = _quant_weight_block_fp8(b, BLOCK, BLOCK)
    assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a, b_fp8, None, b_s)
    assert not _use_rowwise_gemv(2, N, K, BLOCK, BLOCK, a, b_fp8, None, b_s)
    assert not _use_rowwise_gemv(1, N + BLOCK, K, BLOCK, BLOCK, a, b_fp8, None, b_s)
    assert not _use_rowwise_gemv(1, N, K, 64, 128, a, b_fp8, None, b_s)
