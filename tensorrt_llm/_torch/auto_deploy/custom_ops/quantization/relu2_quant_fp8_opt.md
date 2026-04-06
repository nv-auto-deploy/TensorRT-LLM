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

## Final Best Configuration

```python
BLOCK = 1024
num_warps = 4
```

| ID | Baseline (µs) | Final (µs) | Delta |
|---|---|---|---|
| D1 (c=1) | 5.33 | 5.35 | ~0% (noise) |
| D4 (c=4) | 5.31 | 5.44 | ~0% (noise) |
| D16 (c=16) | 5.44 | 5.44 | 0% |
| P1K | 10.18 | 10.07 | **−1.1%** |

**Conclusion:** The kernel is fundamentally launch-overhead-limited for decode sizes
(D1–D16). At these sizes (3712–59392 elements), even with CUDA graphs the memory
transfer is \< 0.1 µs, dwarfed by kernel dispatch overhead. No Triton parameter tuning
can break through this floor.

The only actionable improvement is for large prefill (P1K+): BLOCK=1024, W4 saves ~1%.
**This kernel is not a TPOT bottleneck at c=1,4,16.**

______________________________________________________________________

## Optimization Ideas Backlog

| # | Idea | Status | Result |
|---|---|---|---|
| O1 | BW-bound analysis — confirm roofline | ✅ Done | Launch overhead dominates at decode sizes |
| O2 | BLOCK size tuning (was 512, not power-of-2 aligned) | ✅ Done iter 2-3 | BLOCK=1024 best for P1K |
| O3 | inv_scale (mul vs div) | ✅ Done iter 1 | Slower due to Python overhead; discard |
| O4 | num_warps sweep | ✅ Done iter 2 | num_warps=4 remains best |
| O5 | CUDA graphs context (launch overhead) | Structural | Already used in production via torch-cudagraph |
| O6 | Persistent kernel (reduce launches for multi-layer) | Not attempted | Would require large architectural change |

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
