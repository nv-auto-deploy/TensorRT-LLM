# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests for torch_quant's block-FP8 W8A8 stack.

Covers the matmul dispatch funnel (full-K autotune / split-K / rowwise GEMV), the fused
ue8m0 activation-quant prologue, and the standalone activation-quant kernel.
"""

import pytest
import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _W8A8_TP4_M1_ROWWISE_CFG,
    _W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG,
    _safe_act_quant,
    _splitk_schedule,
    _use_rowwise_gemv,
    _use_splitk_decode,
    _w8a8_block_fp8_matmul_splitk,
    _w8a8_block_fp8_matmul_triton,
    _w8a8_gemv_rowwise,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
BLOCK = 128
ROWWISE_SHAPES = sorted(_W8A8_TP4_M1_ROWWISE_CFG)  # [(N, K), ...]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 9),
    reason="Requires CUDA + FP8 (SM89+)",
)


def _quant_weight_block_fp8(w: torch.Tensor, block_n: int = BLOCK, block_k: int = BLOCK):
    N, K = w.shape
    sn, sk = triton.cdiv(N, block_n), triton.cdiv(K, block_k)
    scale = torch.empty(sn, sk, dtype=torch.float32, device=w.device)
    w_fp8 = torch.empty_like(w, dtype=torch.float8_e4m3fn)
    for i in range(sn):
        for j in range(sk):
            blk = w[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k].float()
            s = (blk.abs().amax() / FP8_MAX).clamp(min=1e-12)
            scale[i, j] = s
            w_fp8[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k] = (
                (blk / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return w_fp8, scale


def _dequant_weight(w_fp8, scale, block_n=BLOCK, block_k=BLOCK):
    N, K = w_fp8.shape
    out = w_fp8.float()
    for i in range(triton.cdiv(N, block_n)):
        for j in range(triton.cdiv(K, block_k)):
            out[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k] *= scale[i, j]
    return out


def _dequant_act(a_fp8, a_s, block_k=BLOCK):
    M, K = a_fp8.shape
    blocks = a_fp8.float().reshape(M, K // block_k, block_k)
    return (blocks * a_s.float().unsqueeze(-1)).reshape(M, K)


def _ref_fp64(a_fp8, a_s, b_fp8, b_s):
    return (_dequant_act(a_fp8, a_s).double() @ _dequant_weight(b_fp8, b_s).double().t()).float()


def _make_case(M, N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK)
    b_fp8, b_s = _quant_weight_block_fp8(b)
    return a_fp8, a_s, b_fp8, b_s


def _exact_case(M, N, K, seed):
    # Values in {0, +-1, +-2, +-4} quantize losslessly and sum fp32-exactly, so any
    # reduction order / atomic arrival order yields bit-identical results.
    torch.manual_seed(seed)
    vals = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0], device="cuda")
    a = vals[torch.randint(0, 7, (M, K), device="cuda")].to(torch.bfloat16)
    a[:, :BLOCK] = 0.0  # all-zero group hits the 1e-4 amax clamp
    b_fp8 = torch.randint(-2, 3, (N, K), device="cuda").float().to(torch.float8_e4m3fn)
    b_s = torch.ones(N // BLOCK, K // BLOCK, device="cuda", dtype=torch.float32)
    return a, b_fp8, b_s


def _max_rel(out, ref):
    return ((out.float() - ref).abs().amax() / ref.abs().amax().clamp(min=1e-6)).item()


# ---------------------------------------------------------------------------
# Standalone activation-quant kernel (_safe_act_quant / _act_quant_kernel)
# ---------------------------------------------------------------------------


def _ref_act_quant(x: torch.Tensor, block_size: int, round_scale: bool):
    xb = x.float().reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    amax = xb.abs().amax(dim=-1)
    if round_scale:
        s = torch.exp2(torch.ceil(torch.log2(amax.clamp(min=1e-4) / 448.0)))
    else:
        s = (amax / 448.0).clamp(min=1e-12)
    y = (xb / s.unsqueeze(-1)).reshape(x.shape).to(torch.float8_e4m3fn)
    return y, s.to(x.dtype)


@pytest.mark.parametrize("round_scale", [False, True])
@pytest.mark.parametrize("M", [1, 256])
@pytest.mark.parametrize("K", [512, 7168])
def test_act_quant_matches_reference(M, K, round_scale):
    torch.manual_seed(0)
    x = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    y, s = _safe_act_quant(x, block_size=BLOCK, input_scale_fmt="ue8m0" if round_scale else "")
    y_ref, s_ref = _ref_act_quant(x, BLOCK, round_scale)

    assert y.dtype == torch.float8_e4m3fn and y.shape == x.shape
    assert s.shape == (M, K // BLOCK) and s.dtype == x.dtype
    if not round_scale:
        # fp32 division + RNE fp8 cast -> bit-exact vs the pure-torch reference.
        assert torch.equal(y.float(), y_ref.float())
        assert torch.equal(s.float(), s_ref.float())
    else:
        # Transcendental (log2/exp2) scale path: compare dequantized values.
        deq = y.float() * s.float().repeat_interleave(BLOCK, dim=-1)
        deq_ref = y_ref.float() * s_ref.float().repeat_interleave(BLOCK, dim=-1)
        denom = x.float().abs().amax().clamp(min=1e-6)
        assert ((deq - deq_ref).abs().amax() / denom).item() < 1e-3


@pytest.mark.parametrize("round_scale", [False, True])
def test_act_quant_all_zero_block_is_nan_safe(round_scale):
    x = torch.zeros(2, 256, device="cuda", dtype=torch.bfloat16)
    x[0, :BLOCK] = 0.05
    y, s = _safe_act_quant(
        x.contiguous(), block_size=BLOCK, input_scale_fmt="ue8m0" if round_scale else ""
    )
    assert torch.isfinite(y.float()).all() and torch.isfinite(s.float()).all()
    assert (y.float()[1] == 0).all() and (y.float()[0, BLOCK:] == 0).all()


# ---------------------------------------------------------------------------
# Dispatch funnel (_w8a8_block_fp8_matmul_triton)
# ---------------------------------------------------------------------------


def _run(M, N, K, seed=0):
    a_fp8, a_s, b_fp8, b_s = _make_case(M, N, K, seed)
    out = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    return out, _ref_fp64(a_fp8, a_s, b_fp8, b_s)


@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 512, 512),  # full-K, decode autotune configs
        (1, 7168, 2048),
        (2, 7168, 2048),
        (2, 576, 7168),  # split-K via the funnel, masked non-128 N
        (2, 1024, 4096),  # split-K via the funnel, TP4 shape
    ],
)
def test_decode_dispatch_strict(M, N, K):
    out, ref = _run(M, N, K)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    assert _max_rel(out, ref) < 1.5e-2


@pytest.mark.parametrize("M,N,K", [(16, 7168, 2048), (512, 576, 7168)])
def test_prefill_rmse(M, N, K):
    # RMSE bar: robust to the vendored kernel's pre-existing sparse large-M
    # non-determinism on Blackwell, still catches scale-indexing errors.
    out, ref = _run(M, N, K)
    scale = ref.abs().amax().clamp(min=1e-6)
    assert ((out.float() - ref).pow(2).mean().sqrt() / scale).item() < 2.5e-2


# ---------------------------------------------------------------------------
# Split-K decode path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 256, 7168),  # narrow-N / deep-split schedule
        (2, 256, 7168),  # %M-padded rows
        (1, 2304, 7168),  # wide-N / shallow-split schedule
        (1, 1024, 4096),  # exact M=1 TP4 special-case schedules
        (1, 1536, 4096),
    ],
)
def test_splitk_matches_reference(M, N, K):
    a_fp8, a_s, b_fp8, b_s = _make_case(M, N, K)
    out = _w8a8_block_fp8_matmul_splitk(
        a_fp8, b_fp8, a_s, b_s, BLOCK, BLOCK, torch.bfloat16, M, N, K
    )
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    assert _max_rel(out, _ref_fp64(a_fp8, a_s, b_fp8, b_s)) < 1.5e-2


def test_splitk_schedule_and_gate():
    # (BLOCK_SIZE_N, SPLIT_K, num_warps): M=1 TP4 special cases vs legacy schedules.
    assert _splitk_schedule(1, 1024, 4096) == (64, 32, 2)
    assert _splitk_schedule(1, 1536, 4096) == (64, 24, 2)
    assert _splitk_schedule(2, 1024, 4096) == (128, 24, 4)
    assert _splitk_schedule(1, 1536, 7168) == (128, 24, 4)
    assert _splitk_schedule(1, 256, 7168) == (32, 48, 4)
    assert _splitk_schedule(1, 2304, 7168) == (128, 16, 4)
    # Gate: only small-M + long-K routes to split-K.
    assert _use_splitk_decode(1, 256, 7168)
    assert _use_splitk_decode(2, 1536, 7168)
    assert not _use_splitk_decode(64, 1536, 7168)
    assert not _use_splitk_decode(1, 7168, 2048)


def test_splitk_quant_prologue_exact_arithmetic():
    M, N, K = 4, 1536, 4096
    a, b_fp8, b_s = _exact_case(M, N, K, seed=21)
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK, "ue8m0")
    ref = _w8a8_block_fp8_matmul_splitk(
        a_fp8, b_fp8, a_s, b_s, BLOCK, BLOCK, torch.bfloat16, M, N, K
    )
    out = _w8a8_block_fp8_matmul_splitk(a, b_fp8, None, b_s, BLOCK, BLOCK, torch.bfloat16, M, N, K)
    assert torch.equal(ref, out)


def test_splitk_quant_prologue_random():
    # Random data inherits the split-K atomic-order wobble -> fp64 bar, not torch.equal.
    M, N, K = 2, 1024, 4096
    torch.manual_seed(22)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b)
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK, "ue8m0")
    out = _w8a8_block_fp8_matmul_splitk(a, b_fp8, None, b_s, BLOCK, BLOCK, torch.bfloat16, M, N, K)
    assert _max_rel(out, _ref_fp64(a_fp8, a_s, b_fp8, b_s)) < 1.5e-2


def test_deferred_quant_prefill_falls_back_to_standalone():
    M, N, K = 64, 1536, 4096
    torch.manual_seed(24)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b)
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK, "ue8m0")
    ref = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    out = _w8a8_block_fp8_matmul_triton(
        a, b_fp8, None, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16, input_scale_fmt="ue8m0"
    )
    assert torch.equal(ref, out)


# ---------------------------------------------------------------------------
# Rowwise direct-store M=1 decode GEMV
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,K", ROWWISE_SHAPES)
def test_rowwise_decode_strict_dispatched_deterministic(N, K):
    a_fp8, a_s, b_fp8, b_s = _make_case(1, N, K)
    assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a_fp8, b_fp8, a_s, b_s)
    out = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    direct = _w8a8_gemv_rowwise(a_fp8, b_fp8, a_s, b_s, BLOCK, BLOCK, torch.bfloat16, N, K)
    # Deterministic kernel: funnel == direct proves the rowwise path was taken,
    # and a second funnel call proves run-to-run determinism.
    assert torch.equal(out, direct)
    out2 = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    assert torch.equal(out, out2)

    ref = _ref_fp64(a_fp8, a_s, b_fp8, b_s)
    assert _max_rel(out, ref) < 1.5e-2
    mism = (out != ref.to(torch.bfloat16)).sum().item()
    assert mism <= max(1, N // 4096), f"N={N} K={K}: {mism} bf16 flips vs rounded ref"


def test_rowwise_residual_epilogue_exact():
    # Contract: round(acc) -> fp32 add -> round, bit-for-bit.
    N, K = 4096, 512
    a_fp8, a_s, b_fp8, b_s = _make_case(1, N, K, seed=2)
    residual = torch.randn(1, N, device="cuda", dtype=torch.bfloat16)
    out_res = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16, residual=residual
    )
    out_plain = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16
    )
    assert torch.equal(out_res, (out_plain.float() + residual.float()).to(torch.bfloat16))


def test_rowwise_dispatch_gate():
    N, K = 1024, 4096
    a_fp8, a_s, b_fp8, b_s = _make_case(1, N, K)
    a_raw = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    for a, As in ((a_fp8, a_s), (a_raw, None)):  # pre-quantized and deferred-quant gates
        assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a, b_fp8, As, b_s)
        assert not _use_rowwise_gemv(2, N, K, BLOCK, BLOCK, a, b_fp8, As, b_s)
        assert not _use_rowwise_gemv(1, N + BLOCK, K, BLOCK, BLOCK, a, b_fp8, As, b_s)
        assert not _use_rowwise_gemv(1, N, K, 64, 128, a, b_fp8, As, b_s)
        assert not _use_rowwise_gemv(1, N, K, 128, 64, a, b_fp8, As, b_s)
    b_strided = b_fp8.t().contiguous().t()
    assert not _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a_fp8, b_strided, a_s, b_s)
    a_s_strided = torch.empty(1, 2 * a_s.shape[-1], device="cuda", dtype=a_s.dtype)[:, ::2]
    assert not _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a_fp8, b_fp8, a_s_strided, b_s)
    assert set(_W8A8_TP4_M1_ROWWISE_CFG) == set(_W8A8_TP4_M1_ROWWISE_PROLOGUE_CFG)


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


def _standalone_ue8m0(a, b_fp8, b_s, residual=None):
    a_fp8, a_s = _safe_act_quant(a.contiguous(), BLOCK, "ue8m0")
    return _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16, residual=residual
    )


@pytest.mark.parametrize("N,K", ROWWISE_SHAPES)
def test_rowwise_quant_prologue_exact_arithmetic(N, K):
    # Bit-exact gate on the prologue's scale math per re-swept launch config.
    a, b_fp8, b_s = _exact_case(1, N, K, seed=12)
    assert _use_rowwise_gemv(1, N, K, BLOCK, BLOCK, a, b_fp8, None, b_s)
    assert torch.equal(_standalone_ue8m0(a, b_fp8, b_s), _fused_ue8m0(a, b_fp8, b_s))


def test_rowwise_quant_prologue_random_close():
    # The prologue cfg changes the fp32 summation-tree order: allow only rare
    # <=1-bf16-ULP (or negligible-magnitude near-cancellation) flips.
    N, K = 1024, 4096
    torch.manual_seed(11)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b)
    ref = _standalone_ue8m0(a, b_fp8, b_s)
    out = _fused_ue8m0(a, b_fp8, b_s)
    neq = out != ref
    flips = int(neq.sum())
    assert flips <= max(2, N // 1024), f"{flips} mismatches"
    if flips:
        steps = (out.view(torch.int16).int() - ref.view(torch.int16).int()).abs()[neq]
        adiff = (out.float() - ref.float()).abs()[neq]
        scale = float(ref.float().abs().mean()) + 1e-12
        assert bool((adiff[steps > 1] <= 1e-4 * scale).all())


def test_rowwise_quant_prologue_nan_propagation():
    N, K = 16384, 1024
    torch.manual_seed(13)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    a[0, 5] = float("nan")
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b)
    ref = _standalone_ue8m0(a, b_fp8, b_s)
    out = _fused_ue8m0(a, b_fp8, b_s)
    assert ref.isnan().all() and out.isnan().all()
    assert torch.equal(ref.view(torch.int16), out.view(torch.int16))


def test_rowwise_quant_prologue_residual_epilogue():
    N, K = 4096, 512
    torch.manual_seed(14)
    a = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b)
    residual = torch.randn(1, N, device="cuda", dtype=torch.bfloat16)
    out_res = _fused_ue8m0(a, b_fp8, b_s, residual=residual)
    out_plain = _fused_ue8m0(a, b_fp8, b_s)
    assert torch.equal(out_res, (out_plain.float() + residual.float()).to(torch.bfloat16))
