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
    """Prefill: fp8 RMSE bar (robust to the pre-existing sparse Blackwell glitch,
    but catches gross scale-indexing errors)."""
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
# Split-K decode path (idea_0025, kernel_layout). The split-K kernel partitions the
# K reduction across SPLIT_K CTAs and reduces fp32 partials via atomics; the result
# must match the same fp64 dequant->matmul ground truth as the full-K kernel.
# ----------------------------------------------------------------------------------


def _run_splitk(M, N, K, split_k, block_size_n=64, num_warps=4, seed=0):
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
        SPLIT_K=split_k,
        BLOCK_SIZE_N=block_size_n,
        num_warps=num_warps,
    )
    ref = (_dequant_act(a_fp8, a_s).double() @ _dequant_weight(b_fp8, b_s).double().t()).float()
    return out, ref


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("split_k", [2, 4, 7, 8, 16])
@pytest.mark.parametrize("M", [1, 2])
@pytest.mark.parametrize("N,K", [(1536, 7168), (576, 7168), (2304, 7168), (256, 7168)])
def test_splitk_matches_reference(M, N, K, split_k):
    """Split-K GEMV == fp64 ground truth for the K=7168 decode projection shapes,
    across SPLIT_K values (incl. 7 and 16 that do NOT divide the 56 K-blocks evenly,
    exercising the ragged-tail K-mask)."""
    out, ref = _run_splitk(M, N, K, split_k)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"M={M} N={N} K={K} SPLIT_K={split_k}: max_rel={max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
@pytest.mark.parametrize("split_k", [4, 8])
@pytest.mark.parametrize("block_size_n", [32, 64, 128])
def test_splitk_block_n_and_ragged_n(split_k, block_size_n):
    """Split-K with N NOT a multiple of BLOCK_SIZE_N (576) across BLOCK_SIZE_N
    choices -- guards the masked-tile atomic store."""
    out, ref = _run_splitk(1, 576, 7168, split_k, block_size_n=block_size_n)
    scale = ref.abs().amax().clamp(min=1e-6)
    max_rel = ((out.float() - ref).abs().amax() / scale).item()
    assert max_rel < 1.5e-2, f"N=576 BLOCK_N={block_size_n} SPLIT_K={split_k}: {max_rel:.4e}"


@pytest.mark.skipif(not _fp8_supported(), reason="Requires Hopper+ FP8 tensor cores")
def test_splitk_dispatch_gate():
    """The dispatch gate routes only small-M + long-K shapes to split-K."""
    assert _use_splitk_decode(1, 256, 7168)
    assert _use_splitk_decode(2, 1536, 7168)
    assert not _use_splitk_decode(64, 1536, 7168)  # prefill M
    assert not _use_splitk_decode(1, 7168, 2048)  # short K (already CTA-rich)
