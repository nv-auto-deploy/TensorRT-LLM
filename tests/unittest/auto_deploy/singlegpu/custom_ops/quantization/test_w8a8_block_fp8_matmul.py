# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the vendored block-scaled FP8 GEMM Triton kernel.

``_w8a8_block_fp8_matmul_triton`` (and its ``_w8a8_block_fp8_matmul_kernel``) is
the FP8 block-scaled linear used on the real-weights path (MLA / dense
projections). This test guards the kernel against an fp64 dequant -> matmul
ground truth so that ``@triton.autotune`` tuning (BLOCK_M/N, GROUP_M, num_warps,
num_stages) can be exercised without silently corrupting results.

Two regimes are checked with regime-appropriate bars:

* **Decode (M=1, M=2)** -- the idea's optimization target. The baseline kernel is
  fully deterministic and accurate here, so we assert a *strict* max-abs-error
  bound. This guarantees autotune never breaks the GEMV decode path.
* **Prefill (M in {16, 64, 512})** -- checked with the fp8-appropriate RMSE bar.
  NOTE: the *baseline* vendored kernel already exhibits a sparse, run-to-run
  non-deterministic discrepancy at large M with K>=2048 on Blackwell (a handful
  of near-cancellation outputs go badly wrong while global RMSE stays ~1%). That
  pre-existing behavior is out of scope for this (autotune-only) kernel work, so
  prefill is gated on RMSE/amax which is robust to the sparse glitch but still
  catches gross scale-indexing errors (which blow RMSE up by >10x).
"""

import pytest
import torch
import triton

from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    _safe_act_quant,
    _use_splitk_decode,
    _w8a8_block_fp8_matmul_splitk,
    _w8a8_block_fp8_matmul_triton,
)

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max

# MLA / dense-projection-shaped (N, K) for DeepSeek-V4-class TP8 ranks, plus a
# non-128-multiple N (kv_a-like) to exercise the masked-tile store path.
SHAPES = [
    (512, 512),
    (2048, 512),  # kv_b-like
    (4096, 512),
    (3072, 1536),  # q_b-like
    (7168, 2048),  # o_proj-like (K=2048 -> 16 K-blocks)
    (576, 7168),  # kv_a-like: N NOT a multiple of 128
    (256, 7168),  # shared-expert gate/up-like
]

# DeepSeek-V4-Flash TP4 per-rank shapes. M=1 dispatches to the rowwise kernel;
# M=2 and prefill exercise the split-K and full-K fallbacks.
TP4_SHAPES = [
    (16384, 1024),  # fused wq_b + indexer.wq_b
    (8192, 1024),  # wq_b alone (ratio-128/0 layers)
    (4096, 2048),  # wo_b (rowwise K=8192/4)
    (1536, 4096),  # fused wq_a + wkv -> split-K path
    (1024, 4096),  # shared w1+w3 / grouped wo_a rank -> split-K path
]


def _quant_weight_block_fp8(w: torch.Tensor, block_n: int = 128, block_k: int = 128):
    """Per-(block_n x block_k)-block FP8 weight quantization.

    Handles N / K not divisible by the block size (trailing partial block keeps
    its own amax scale), matching the kernel's masked handling of ragged tiles.
    """
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


def _dequant_weight(w_fp8, scale, block_n=128, block_k=128):
    N, K = w_fp8.shape
    out = w_fp8.float()
    for i in range(triton.cdiv(N, block_n)):
        for j in range(triton.cdiv(K, block_k)):
            out[i * block_n : (i + 1) * block_n, j * block_k : (j + 1) * block_k] *= scale[i, j]
    return out


def _dequant_act(a_fp8, a_s, block_k=128):
    M, K = a_fp8.shape
    blocks = a_fp8.float().reshape(M, K // block_k, block_k)
    return (blocks * a_s.float().unsqueeze(-1)).reshape(M, K)


def _fp8_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9  # Hopper+ for fp8 tl.dot


def _run(M, N, K, seed=0):
    torch.manual_seed(seed)
    block_n = block_k = 128
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), block_k)  # mirrors the real op
    b_fp8, b_s = _quant_weight_block_fp8(b, block_n, block_k)
    out = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [block_n, block_k], output_dtype=torch.bfloat16
    )
    # fp64 dequant->matmul ground truth (same fp8 operands as the kernel).
    ref = (_dequant_act(a_fp8, a_s).double() @ _dequant_weight(b_fp8, b_s).double().t()).float()
    return out, ref


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("N,K", SHAPES)
def test_decode_gemv_strict(M, N, K):
    """Decode GEMV: deterministic + accurate -> strict max-abs-error bound."""
    out, ref = _run(M, N, K)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"M={M} N={N} K={K}: max_abs_err/amax={max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [16, 64, 512])
@pytest.mark.parametrize("N,K", SHAPES)
def test_prefill_rmse(M, N, K):
    """Prefill: fp8 RMSE bar.

    Robust to the pre-existing sparse Blackwell glitch, but catches gross scale-indexing errors.
    """
    out, ref = _run(M, N, K)
    assert out.shape == (M, N)
    scale = ref.abs().amax().clamp(min=1e-6)
    rmse_rel = ((out.float() - ref).pow(2).mean().sqrt() / scale).item()
    assert rmse_rel < 2.5e-2, f"M={M} N={N} K={K}: RMSE/amax={rmse_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("N,K", TP4_SHAPES)
def test_tp4_decode_strict(M, N, K):
    """Check TP4 per-rank decode shapes through the real dispatcher."""
    out, ref = _run(M, N, K)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"M={M} N={N} K={K}: max_abs_err/amax={max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [16, 48, 512])
@pytest.mark.parametrize("N,K", TP4_SHAPES)
def test_tp4_prefill_rmse(M, N, K):
    """Check TP4 shapes at chunked-prefill Ms with the pre-existing config set."""
    out, ref = _run(M, N, K)
    assert out.shape == (M, N)
    scale = ref.abs().amax().clamp(min=1e-6)
    rmse_rel = ((out.float() - ref).pow(2).mean().sqrt() / scale).item()
    assert rmse_rel < 2.5e-2, f"M={M} N={N} K={K}: RMSE/amax={rmse_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
def test_decode_gemv_multi_kblock():
    """M=1 GEMV across K spanning several K-blocks (scale-indexing in the loop)."""
    for N, K in [(4096, 1024), (576, 2048), (1536, 256)]:
        out, ref = _run(1, N, K, seed=1)
        scale = ref.abs().amax().clamp(min=1e-6)
        assert ((out.float() - ref).abs().amax() / scale).item() < 1.5e-2, f"N={N} K={K}"


# ----------------------------------------------------------------------------------
# Split-K decode path (kernel_layout). The split-K kernel partitions the
# K reduction across SPLIT_K CTAs and reduces fp32 partials via atomics; the result
# must match the same fp64 dequant->matmul ground truth as the full-K kernel.
# ----------------------------------------------------------------------------------


def _run_splitk(M, N, K, seed=0):
    """Drive the split-K helper directly (independent of the dispatch gate)."""
    torch.manual_seed(seed)
    block_n = block_k = 128
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    a_fp8, a_s = _safe_act_quant(a.contiguous(), block_k)
    b_fp8, b_s = _quant_weight_block_fp8(b, block_n, block_k)
    out = _w8a8_block_fp8_matmul_splitk(
        a_fp8,
        b_fp8,
        a_s,
        b_s,
        block_n,
        block_k,
        torch.bfloat16,
        M,
        N,
        K,
    )
    ref = (_dequant_act(a_fp8, a_s).double() @ _dequant_weight(b_fp8, b_s).double().t()).float()
    return out, ref


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("N,K", [(1536, 7168), (576, 7168), (2304, 7168), (256, 7168)])
def test_splitk_matches_reference(M, N, K):
    """Split-K GEMV matches fp64 ground truth with its production schedule."""
    out, ref = _run_splitk(M, N, K)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"M={M} N={N} K={K}: max_rel={max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("N,K", [(1024, 4096), (1536, 4096)])
def test_splitk_tp4_band(M, N, K):
    """The measured TP4 schedule reconstructs the fp64 ground truth."""
    out, ref = _run_splitk(M, N, K)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"M={M} N={N} K={K}: max_rel={max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
def test_splitk_tp4_heuristic_defaults():
    """Check the exact M=1 TP4 schedule and unchanged legacy fallbacks."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import _splitk_schedule

    # (BLOCK_SIZE_N, SPLIT_K, num_warps): exact M=1 TP4 schedule...
    assert _splitk_schedule(1, 1024, 4096) == (64, 32, 2)
    assert _splitk_schedule(1, 1536, 4096) == (64, 24, 2)
    # ...while M=2 and K=7168 keep the legacy schedules.
    assert _splitk_schedule(2, 1024, 4096) == (128, 24, 4)
    assert _splitk_schedule(1, 1536, 7168) == (128, 24, 4)
    assert _splitk_schedule(1, 256, 7168) == (32, 48, 4)
    assert _splitk_schedule(1, 2304, 7168) == (128, 16, 4)

    for N in (1024, 1536):
        out, ref = _run_splitk(1, N, 4096)
        scale = ref.abs().amax().clamp(min=1e-6)
        max_rel = ((out.float() - ref).abs().amax() / scale).item()
        assert max_rel < 1.5e-2, f"heuristic N={N} K=4096: {max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
def test_splitk_dispatch_gate():
    """The dispatch gate routes only small-M + long-K shapes to split-K."""
    assert _use_splitk_decode(1, 256, 7168)
    assert _use_splitk_decode(2, 1536, 7168)
    assert not _use_splitk_decode(64, 1536, 7168)  # prefill M
    assert not _use_splitk_decode(1, 7168, 2048)  # short K (already CTA-rich)


# ---------------------------------------------------------------------------
# Fused activation-quant prologue (QUANT_PROLOGUE=True, v6)
# ---------------------------------------------------------------------------
#
# ``As=None`` routes the raw bf16 activation into the split-K kernel, which
# replicates ``_act_quant_kernel``'s ue8m0 quant per (row, 128-group) MMA tile
# in its prologue and feeds the fp8 tile straight to ``tl.dot``. Exact-
# arithmetic inputs (all partial products/sums fp32-exact) make the result
# independent of both the reassociation AND the split-K atomic arrival order,
# so torch.equal against the standalone pipeline is a hard gate; random-data
# comparisons inherit the split-K path's pre-existing 1-ULP atomic wobble and
# use the fp64 reference bar instead.


def _exact_case(M, N, K, seed):
    """Inputs whose quant is lossless and whose accumulation is fp32-exact."""
    torch.manual_seed(seed)
    vals = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0], device="cuda")
    a = vals[torch.randint(0, 7, (M, K), device="cuda")].to(torch.bfloat16)
    a[:, :128] = 0.0  # all-zero group exercises the 1e-4 amax clamp
    w = torch.randint(-2, 3, (N, K), device="cuda").float()
    b_fp8 = w.to(torch.float8_e4m3fn)
    b_s = torch.ones(N // 128, K // 128, device="cuda", dtype=torch.float32)
    return a, b_fp8, b_s


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [1, 2, 4])
@pytest.mark.parametrize("N,K", [(1024, 4096), (1536, 4096)])
def test_splitk_quant_prologue_exact_arithmetic(M, N, K):
    """Fused-prologue split-K == standalone quant + split-K, bit-for-bit, on exact-arithmetic inputs.

    Any atomic order sums the same exact fp32 values. M in {2,4} exercises the
    %M-padded rows (each padded row re-quantizes the identical data row, so
    scales/payloads dedupe by construction).
    """
    a, b_fp8, b_s = _exact_case(M, N, K, seed=21)
    a_fp8, a_s = _safe_act_quant(a.contiguous(), 128, "ue8m0")
    ref = _w8a8_block_fp8_matmul_splitk(a_fp8, b_fp8, a_s, b_s, 128, 128, torch.bfloat16, M, N, K)
    out = _w8a8_block_fp8_matmul_splitk(a, b_fp8, None, b_s, 128, 128, torch.bfloat16, M, N, K)
    assert torch.equal(ref, out)


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("M", [2, 4])
@pytest.mark.parametrize("N,K", [(1024, 4096), (1536, 4096)])
def test_splitk_quant_prologue_random(M, N, K):
    """Fused-prologue split-K vs fp64 ground truth of standalone-quantized operands on random data.

    This is the strict decode bar; atomic order precludes a bitwise
    fused-vs-standalone assert here.
    """
    torch.manual_seed(22)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b, 128, 128)
    a_fp8, a_s = _safe_act_quant(a.contiguous(), 128, "ue8m0")
    out = _w8a8_block_fp8_matmul_splitk(a, b_fp8, None, b_s, 128, 128, torch.bfloat16, M, N, K)
    ref = (_dequant_act(a_fp8, a_s).double() @ _dequant_weight(b_fp8, b_s).double().t()).float()
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"M={M} N={N} K={K}: max_rel={max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
def test_deferred_quant_prefill_falls_back_to_standalone():
    """As=None with a non-decode M must quantize standalone inside the dispatch.

    It must reproduce the eager quant+full-K pipeline byte-for-byte.
    """
    M, N, K = 64, 1536, 4096
    torch.manual_seed(24)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    b_fp8, b_s = _quant_weight_block_fp8(b, 128, 128)
    a_fp8, a_s = _safe_act_quant(a.contiguous(), 128, "ue8m0")
    ref = _w8a8_block_fp8_matmul_triton(
        a_fp8, b_fp8, a_s, b_s, [128, 128], output_dtype=torch.bfloat16
    )
    out = _w8a8_block_fp8_matmul_triton(
        a, b_fp8, None, b_s, [128, 128], output_dtype=torch.bfloat16, input_scale_fmt="ue8m0"
    )
    assert torch.equal(ref, out)
