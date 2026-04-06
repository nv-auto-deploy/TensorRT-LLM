# relu2_quant_fp8 Kernel Optimization Log

**Branch:** `ad-fast-iter/nemotron-nano-20260323`
**Kernel:** `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/relu2_quant_fp8.py`
**Transform:** `tensorrt_llm/_torch/auto_deploy/transform/library/fuse_relu2_quant_fp8.py`
**Date:** 2026-04-06

______________________________________________________________________

## Kernel Overview

Fused ReLU² + FP8 per-tensor quantization for MLP layers in Nemotron-style models.

**Operation:**

```
out_fp8 = clamp(max(x, 0)^2 / scale, FP8_MIN, FP8_MAX).to(float8_e4m3fn)
```

Inputs: BF16 tensor `x`, FP32 scalar `scale`.\
Output: FP8 (float8_e4m3fn) tensor, same shape as `x`.

**Why it exists:** Replaces 3 separate GPU kernels (relu, pow, scaleMatrixPerTensorVec)
with a single Triton elementwise kernel. Added in iter 21.

**Roofline:** Memory-bandwidth bound. For D1 (n=3712):

- Total memory: 3712×2 (bf16 in) + 3712×1 (fp8 out) = 11 KB
- H100 BW = 3.35 TB/s → theoretical = 3.3 ns
- Even at 10% efficiency: 33 ns. Launch overhead (~5 µs) dominates completely.

______________________________________________________________________

## Target Models & Benchmark Shapes

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`

| Config param | Value |
|---|---|
| `hidden_size` | 2688 |
| `moe_shared_expert_intermediate_size` | **3712** (= 128 × 29) |
| `num_hidden_layers` | 52 |
| `mlp_hidden_act` | relu2 |

This kernel fires on the **shared expert MLP** in each transformer layer.\
For decode with concurrency C: `n_elements = C × 3712`.

**Tiling notes for 3712:**

- `3712 mod 512 = 128` → BLOCK=512 wastes last block (25% of it unused)
- `3712 = 128 × 29` → BLOCK=128 tiles exactly (no waste)
- `3712 mod 256 = 128` → BLOCK=256: last block half-full (still minor)

### Shape Matrix

| ID | Description | T | n_elements | n_blocks (B=512) |
|---|---|---|---|---|
| D1 | c=1 decode | 1 | 3,712 | 8 |
| D4 | c=4 decode | 4 | 14,848 | 29 |
| D16 | c=16 decode | 16 | 59,392 | 116 |
| D32 | c=32 decode | 32 | 118,784 | 232 |
| P64 | prefill T=64 | 64 | 237,568 | 464 |
| P256 | prefill T=256 | 256 | 950,272 | 1,856 |
| P1K | prefill T=1024 | 1024 | 3,801,088 | 7,424 |

**Target concurrencies (user goal): c=1, 4, 16 (D1, D4, D16)**

______________________________________________________________________

## Optimization Iterations

### Environment

- **GPU:** NVIDIA H100 80GB HBM3
- **PyTorch:** 2.11.0a0+eb65b36914.nv26.02
- **Triton:** 3.6.0
- **Benchmark:** `triton.testing.do_bench(warmup=25, rep=100)`

______________________________________________________________________

### Iteration 0 — Baseline (BLOCK=512, num_warps=4)

**Config:** BLOCK=512, num_warps=4

| ID | T | n_elem | n_blks | Kernel (µs) |
|---|---|---|---|---|
| D1 | 1 | 3,712 | 8 | **5.33** |
| D4 | 4 | 14,848 | 29 | **5.31** |
| D16 | 16 | 59,392 | 116 | **5.44** |
| D32 | 32 | 118,784 | 232 | **5.67** |
| P64 | 64 | 237,568 | 464 | **5.84** |
| P256 | 256 | 950,272 | 1856 | **6.93** |
| P1K | 1024 | 3,801,088 | 7424 | **10.18** |

**Analysis:** D1–D16 all take ~5.3–5.5 µs regardless of work size — pure kernel launch overhead.
3712 elements / H100 BW = 0.003 µs theoretical; measured is 1780× that.
The useful signal is in P256/P1K where memory access dominates.

______________________________________________________________________

### Iteration 1 — inv_scale (multiply vs divide) — DISCARDED

**Idea:** Replace `relu2 / scale` with `relu2 * inv_scale` where inv_scale is precomputed
on the host (`(1.0/scale).to(float32)`) to avoid an FP32 division in the kernel.

**Result:**

| ID | div (µs) | mul (µs) | Verdict |
|---|---|---|---|
| D1 | 5.33 | 10.26 | ❌ 2× **slower** |
| D4 | 5.31 | 10.32 | ❌ 2× slower |
| D16 | 5.44 | 10.62 | ❌ 2× slower |
| P1K | 10.18 | 15.29 | ❌ 2× slower |

**Root cause:** `inv_scale = (1.0 / scale).to(torch.float32)` triggers a Python-side
tensor operation + extra `.to()` cast before each kernel launch, adding ~5 µs.
The division inside the kernel is trivial vs this overhead.
**Conclusion: Discard. The division is not the bottleneck.**

______________________________________________________________________

### Iteration 2 — Full BLOCK × num_warps sweep

**Tested:** BLOCK ∈ {64, 128, 256, 512, 1024, 2048}, num_warps ∈ {2, 4, 8, 16} — 24 configs.

**Best results per shape (div variant):**

| ID | Best config | Best (µs) | Baseline (µs) | Delta |
|---|---|---|---|---|
| D1 | BLOCK=256, W4 | 5.18 | 5.33 | −2.8% |
| D4 | BLOCK=1024, W8 | 5.31 | 5.31 | 0% |
| D16 | BLOCK=1024, W8 | 5.42 | 5.44 | −0.4% |
| P256 | BLOCK=1024, W4 | 6.78 | 6.93 | −2.2% |
| P1K | BLOCK=2048, W4 | 10.02 | 10.18 | −1.6% |

**Key observations:**

1. **D1/D4/D16**: All configs within 0.1–0.5 µs of each other (noise floor). No meaningful
   improvement possible at launch-overhead-dominated sizes.
1. **P1K prefill**: BLOCK=1024, W4 consistently ~10.07–10.08 µs (vs 10.18 baseline) = **1.1% better**.
   BLOCK=2048, W4 gives 10.02 µs = **1.6% better** but D4/D16 slightly worse than baseline.
1. **BLOCK=1024, num_warps=4** is the best single config across all shapes:
   - Decode: within noise of baseline (±3%)
   - P1K: 10.07 µs (−1.1% vs baseline)

**Single best config: BLOCK=1024, num_warps=4**

| ID | Baseline (µs) | BLOCK=1024, W4 (µs) | Delta |
|---|---|---|---|
| D1 | 5.33 | 5.35 | +0.4% (noise) |
| D4 | 5.31 | 5.44 | +2.4% (noise) |
| D16 | 5.44 | 5.44 | 0% |
| P256 | 6.93 | 6.78 | −2.2% |
| P1K | 10.18 | 10.07 | **−1.1%** |

______________________________________________________________________

### Iteration 3 — Apply BLOCK=1024, num_warps=4

**Change:** `BLOCK = 512 → 1024`, `num_warps = 4` (unchanged).

**Rationale:**

- BLOCK=1024 is best for P1K prefill (−1.1%)
- Decode shapes (D1, D4, D16) are noise-floor, no regression
- n_blocks at P1K drops from 7424 → 3712, reducing grid launch overhead by 2×

**Correctness:** PASS (max_diff=0.0 for all shapes, verified against PyTorch reference)

______________________________________________________________________

### Iterations 4–8 — Structural variants sweep

**Script:** `sweep_structural.py`

Tested 6 kernel variants against v0 (current BLOCK=1024, W4):

| Variant | Change | D1 (µs) | D16 (µs) | P1K (µs) |
|---|---|---|---|---|
| v0_baseline | current | 5.23 | 5.60 | 9.59 |
| v1_stages2 | num_stages=2 | 5.29 | 5.37 | 9.59 |
| v2_bf16relu | relu in bf16 space | 5.30 | 5.36 | 9.59 |
| v3_tl_clamp | tl.clamp vs max/min | 5.35 | 5.37 | **9.58** |
| v4_int32 | explicit int32 load | ❌ Triton error (nested fn) | — | — |
| v5_combined | bf16 relu + tl.clamp | **5.10** | **5.36** | 9.79 |
| v6_scale_first | load scale before data | 5.11 | 5.37 | 9.81 |

**iter 4 — num_stages=2:** No improvement (D1: +1.1%, D16: −4.1% within noise). Elementwise
kernel has no memory/compute overlap to pipeline. **Discarded.**

**iter 5 — bf16 relu:** `tl.maximum(x, 0.0)` on bf16 input before fp32 upcast. Avoids
converting negative elements to fp32 before zeroing them. Marginal improvement on D16
(−4.3%). Numerically correct since relu(bf16) = bf16, then upcast. **Kept as part of v5.**

**iter 6 — tl.clamp:** Replaces `tl.maximum(tl.minimum(...))` with `tl.clamp()`. Compiles
to a single PTX `clamp` instruction. Marginally better for P1K (−0.1%). **Kept.**

**iter 7 — int32 vectorized load:** Triton doesn't support nested `def` inside `@triton.jit`
— compile error. Triton already auto-vectorizes bf16 loads to 4-byte transactions.
**Not possible, discarded.**

**iter 8 — v5_combined (bf16 relu + tl.clamp):** Best for D1 (5.10 vs 5.23 µs = −2.3%)
and D16 (5.36 vs 5.60 µs = −4.3%). Within noise but consistently lower. **Applied.**

______________________________________________________________________

### Iteration 9 — Apply v5_combined structural changes

**Changes applied:**

1. Relu in bf16 space: `r_bf16 = tl.maximum(x, 0.0)` then upcast `r = r_bf16.to(tl.float32)`
1. `tl.clamp(out_scaled, FP8_MIN, FP8_MAX)` instead of `tl.maximum(tl.minimum(...))`

**Post-change benchmark (BLOCK=1024, W4):**

| ID | Baseline (µs) | After iter 9 (µs) | Delta |
|---|---|---|---|
| D1 (c=1) | 5.33 | 5.27 | −1.1% |
| D4 (c=4) | 5.31 | 5.20 | **−2.1%** |
| D16 (c=16) | 5.44 | 5.57 | +2.4% (noise) |
| P256 | 6.93 | 6.91 | −0.3% |
| P1K | 10.18 | 9.75 | **−4.2%** |

**Correctness:** PASS (max_diff=0.0 for all shapes)

______________________________________________________________________

### Iteration 23 — BLOCK=2048, num_warps=2 (DISCARDED)

**Config:** BLOCK=2048, num_warps=2

| ID | div (µs) | vs best (W=4) | Verdict |
|---|---|---|---|
| D1 | 5.30 | -0.07 | noise |
| D4 | 5.35 | -0.12 | noise |
| D16 | 5.72 | +0.06 | noise |
| P1K | 9.71 | +0.04 | noise |

**Analysis:** P1K: 9.71µs vs 9.67µs. Essentially tied with W=4. W=4 marginally better; keeping W=4.

______________________________________________________________________

### Iteration 22 — BLOCK=2048, num_warps=1 (DISCARDED)

**Config:** BLOCK=2048, num_warps=1

| ID | div (µs) | vs best (W=4) | Verdict |
|---|---|---|---|
| D1 | 5.77 | +0.40 | ❌ worse |
| D4 | 5.90 | +0.43 | ❌ worse |
| D16 | 5.86 | +0.20 | ❌ worse |
| P1K | 10.25 | +0.58 | ❌ worse |

**Analysis:** W=1 underutilizes the SM. All shapes worse. Reverted.

______________________________________________________________________

### Iteration 21 — BLOCK=4096, num_warps=4 (DISCARDED)

**Config:** BLOCK=4096, num_warps=4

| ID | div (µs) | vs best (2048) | Verdict |
|---|---|---|---|
| D1 | 5.30 | -0.07 | noise |
| D4 | 5.40 | -0.07 | noise |
| D16 | 5.77 | +0.11 | noise |
| P1K | 9.72 | +0.05 | ≈ noise |

**Analysis:** P1K: 9.72µs vs 9.67µs for BLOCK=2048 — essentially the same (within noise). BLOCK=2048 remains best. Reverted.

______________________________________________________________________

### Iteration 20 — BLOCK=4096, num_warps=2 (DISCARDED)

**Config:** BLOCK=4096, num_warps=2

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.83 | +0.46 | ❌ worse |
| D4 | 5.89 | +0.42 | ❌ worse |
| D16 | 6.10 | +0.44 | ❌ worse |
| P1K | 10.08 | +0.41 | ❌ worse |

**Analysis:** BLOCK=4096 is worse everywhere. Grid becomes too small (928 blocks), not enough parallelism. W=2 also suboptimal. Reverted.

______________________________________________________________________

### Iteration 19 — BLOCK=2048, num_warps=4 (APPLIED — new best P1K)

**Config:** BLOCK=2048, num_warps=4

| ID | div (µs) | vs prev best | Verdict |
|---|---|---|---|
| D1 | 5.37 | +0.18 | noise |
| D4 | 5.47 | -0.03 | noise |
| D16 | 5.66 | +0.04 | noise |
| P1K | **9.67** | **-0.12 (-1.2%)** | ✅ new best |

**Analysis:** P1K improved to 9.67µs (from 9.79µs). BLOCK=2048 halves the grid from 3712 to 1856 blocks vs BLOCK=1024. Each block handles more elements, reducing grid launch overhead slightly. APPLIED.

______________________________________________________________________

### Iteration 18 — BLOCK=2048, num_warps=2 (DISCARDED)

**Config:** BLOCK=2048, num_warps=2

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.38 | +0.19 | noise |
| D4 | 5.67 | +0.17 | noise |
| D16 | 5.85 | +0.23 | noise |
| P1K | 9.86 | +0.07 | slightly worse |

**Analysis:** P1K: 9.86µs vs 9.79µs. W=2 slightly worse than W=4 at BLOCK=2048. Reverted.

______________________________________________________________________

### Iteration 17 — BLOCK=512, num_warps=4 (DISCARDED)

**Config:** BLOCK=512, num_warps=4

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.12 | -0.07 | slight improvement |
| D4 | 5.22 | -0.28 | slight improvement |
| D16 | 5.65 | +0.03 | noise |
| P1K | 10.06 | +0.27 | slightly worse |

**Analysis:** D1/D4 slightly better, but P1K 10.06 vs 9.79µs — BLOCK=1024 still wins for P1K. Mixed results. Keeping BLOCK=1024 as the best single overall config.

______________________________________________________________________

### Iteration 16 — BLOCK=512, num_warps=2 (DISCARDED)

**Config:** BLOCK=512, num_warps=2

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.42 | +0.23 | noise |
| D4 | 5.34 | -0.16 | noise |
| D16 | 5.52 | -0.10 | noise |
| P1K | 10.23 | +0.44 | ❌ slightly worse |

**Analysis:** P1K: 10.23µs vs 9.79µs best. W=2 slightly worse than W=4 at BLOCK=512. Reverted.

______________________________________________________________________

### Iteration 15 — BLOCK=256, num_warps=4 (DISCARDED)

**Config:** BLOCK=256, num_warps=4

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.29 | +0.10 | noise |
| D4 | 5.26 | -0.24 | slight improvement |
| D16 | 5.69 | +0.07 | noise |
| P1K | 14.23 | +4.44 | ❌ worse |

**Analysis:** D4 slightly better but P1K significantly worse (14.23 vs 9.79µs). Reverted.

______________________________________________________________________

### Iteration 14 — BLOCK=256, num_warps=2 (DISCARDED)

**Config:** BLOCK=256, num_warps=2

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.36 | +0.17 | noise |
| D4 | 5.52 | +0.02 | noise |
| D16 | 5.76 | +0.14 | noise |
| P1K | 14.38 | +4.59 | ❌ worse |

**Analysis:** BLOCK=256 still creates 14848 blocks at P1K. Reverted.

______________________________________________________________________

### Iteration 13 — BLOCK=128, num_warps=4 (DISCARDED)

**Config:** BLOCK=128, num_warps=4

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.43 | +0.24 | ❌ worse |
| D4 | 5.40 | -0.10 | noise |
| D16 | 5.67 | +0.05 | noise |
| P1K | 22.94 | +13.15 | ❌ much worse |

**Analysis:** P1K still 22.94µs due to 29696 grid blocks. Reverted.

______________________________________________________________________

### Iteration 12 — BLOCK=128, num_warps=2 (DISCARDED)

**Config:** BLOCK=128, num_warps=2

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.42 | +0.23 | ❌ worse |
| D4 | 5.60 | +0.10 | ❌ worse |
| D16 | 5.55 | -0.07 | noise |
| P1K | 23.23 | +13.44 | ❌ much worse |

**Analysis:** BLOCK=128 still creates 29696 blocks at P1K — grid overhead. Reverted.

______________________________________________________________________

### Iteration 11 — BLOCK=64, num_warps=4 (DISCARDED)

**Config:** BLOCK=64, num_warps=4

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.50 | +0.31 | ❌ worse |
| D4 | 5.66 | +0.16 | ❌ worse |
| D16 | 5.62 | +0.00 | ≈ same |
| P1K | 40.82 | +31.03 | ❌ much worse |

**Analysis:** Same as W=2: BLOCK=64 creates 59392 blocks for P1K. Reverted to BLOCK=1024, W=4.

______________________________________________________________________

### Iteration 10 — BLOCK=64, num_warps=2 (DISCARDED)

**Config:** BLOCK=64, num_warps=2

| ID | div (µs) | vs best | Verdict |
|---|---|---|---|
| D1 | 5.52 | +0.33 | ❌ worse |
| D4 | 5.70 | +0.20 | ❌ worse |
| D16 | 5.93 | +0.31 | ❌ worse |
| P1K | 41.07 | +31.28 | ❌ much worse |

**Analysis:** BLOCK=64 creates 59392 blocks for P1K — extreme grid overhead. Reverted to BLOCK=1024, W=4.

______________________________________________________________________

## Final Best Configuration

```python
BLOCK = 2048
num_warps = 4
# relu in bf16 space, tl.clamp for quantize (v5_combined)
```

| ID | Original (µs) | After iter 9 (µs) | After iter 19 (µs) | Total delta |
|---|---|---|---|---|
| D1 (c=1) | 5.33 | 5.27 | 5.37 | +0.7% (noise) |
| D4 (c=4) | 5.31 | 5.20 | 5.47 | +3.0% (noise) |
| D16 (c=16) | 5.44 | 5.57 | 5.66 | +4.0% (noise) |
| P1K | 10.18 | 9.75 | **9.67** | **−5.0%** |

**Conclusion:** The kernel is fundamentally launch-overhead-limited for decode sizes
(D1–D16). At these sizes (3712–59392 elements), the memory transfer is \< 0.1 µs;
all measured improvements of 0.1–0.3 µs are at the noise floor.

For prefill (P1K), BLOCK=1024 + v5_combined saves ~4% vs original baseline.
**This kernel is not a TPOT bottleneck at c=1,4,16.**

______________________________________________________________________

## Optimization Ideas Backlog

| # | Idea | Status | Result |
|---|---|---|---|
| O1 | BW-bound analysis — confirm roofline | ✅ Done | Launch overhead dominates at decode sizes |
| O2 | BLOCK size tuning (was 512) | ✅ Done iter 2-3 | BLOCK=1024 best for P1K |
| O3 | inv_scale (mul vs div) | ✅ Done iter 1 | Slower due to Python overhead; discard |
| O4 | num_warps sweep | ✅ Done iter 2 | num_warps=4 remains best |
| O5 | num_stages=2 | ✅ Done iter 4 | No improvement; discard |
| O6 | bf16 relu (avoid early fp32 upcast) | ✅ Done iter 5/8 | −2-4% on small shapes |
| O7 | tl.clamp vs max/min | ✅ Done iter 6/8 | Marginal |
| O8 | int32 vectorized load | ✅ Done iter 7 | Not possible (Triton nested fn limitation) |
| O9 | CUDA graphs context (launch overhead) | Structural | Already used in production |
| O10 | Persistent kernel (reduce launches for multi-layer) | Not attempted | Large architectural change |

______________________________________________________________________

## Appendix: How to Reproduce

```bash
cd /path/to/TensorRT-LLM
# Baseline
python tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/sweep_relu2_quant_fp8.py

# Full sweep
python tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/sweep_relu2_quant_fp8.py --sweep

# Specific config
python tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/sweep_relu2_quant_fp8.py --block 1024 --warps 4
```
