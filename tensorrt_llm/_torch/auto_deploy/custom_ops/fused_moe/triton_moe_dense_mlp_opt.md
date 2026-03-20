# Triton MoE Dense MLP — Kernel Optimization Log

> **Kernel file:** `triton_moe_dense_mlp.py`
> **Owner:** see git log
> **Started:** 2026-03-20
> **GPU target:** TBD (to be filled after first run)

______________________________________________________________________

## 1. Kernel Overview

The file contains **two Triton kernels** used in the AutoDeploy path for GPT-OSS-style
dense MoE. The end-to-end op (`triton_moe_dense_mlp`) performs:

```
1. gate_up = bmm(hidden, gate_up_w) + gate_up_b          # torch.bmm
2. Interleaved split: gate = gate_up[..., ::2], up = gate_up[..., 1::2]
3. Clamp: gate = clamp(gate, max=limit), up = clamp(up, min=-limit, max=limit)
4. GLU: glu = gate * sigmoid(gate * alpha)                 # ← Triton kernel 1
5. Fused multiply: act_out = (up + 1) * glu                # ← Triton kernel 1
6. down_out = bmm(act_out, down_w) + down_b               # torch.bmm
7. Weighted sum over experts                               # ← Triton kernel 2
```

**Scope of optimization:** Triton kernels only (steps 2-5 and step 7). The BMMs
(steps 1 & 6) are handled by `torch.bmm` and out of scope.

### Kernel 1: `_fused_glu_activation_kernel`

- **Input:** `gate_up` tensor `[E*T, 2*I]` (interleaved)
- **Output:** `act_out` tensor `[E*T, I]`
- **Grid:** `(E*T,)` — one program per row
- **Work per program:** read 2\*I elements (strided: even/odd), compute clamp+GLU+multiply, write I elements

### Kernel 2: `_weighted_expert_sum_kernel`

- **Input:** `expert_out` tensor `[E, T, H]`, `routing_weights` `[T, E]`
- **Output:** `output` tensor `[T, H]`
- **Grid:** `(T,)` — one program per token
- **Work per program:** loop over E experts, accumulate weighted sum over H elements

______________________________________________________________________

## 2. Target Models & Benchmark Shapes

### Models

| Model | H | I | E | top-k | Source |
|-------|---|---|---|-------|--------|
| GPT-OSS-120B | 2880 | 2880 | 128 | 4 | [HF config](https://huggingface.co/openai/gpt-oss-120b) |
| GPT-OSS-20B | 2880 | 2880 | 32 | 4 | [HF config](https://huggingface.co/openai/gpt-oss-20b) |

> **Note:** H == I == 2880 (not a power of 2). `BLOCK_I` / `BLOCK_H` will be 4096.
> The dense MoE pattern (BMM over all experts) is currently GPT-OSS specific.

### Benchmark Configs

| ID | Model | E | T | H | I | Scenario |
|----|-------|---|---|---|---|----------|
| A1 | GPT-OSS-120B | 128 | 1 | 2880 | 2880 | Single-token decode |
| A2 | GPT-OSS-120B | 128 | 8 | 2880 | 2880 | Small batch decode |
| A3 | GPT-OSS-120B | 128 | 32 | 2880 | 2880 | Medium batch decode |
| A4 | GPT-OSS-120B | 128 | 128 | 2880 | 2880 | Large batch decode |
| A5 | GPT-OSS-120B | 128 | 512 | 2880 | 2880 | Prefill |
| B1 | GPT-OSS-20B | 32 | 1 | 2880 | 2880 | Single-token decode |
| B2 | GPT-OSS-20B | 32 | 32 | 2880 | 2880 | Medium batch decode |
| B3 | GPT-OSS-20B | 32 | 512 | 2880 | 2880 | Prefill |

______________________________________________________________________

## 3. Optimization Iterations

### Baseline (v0) — Current Implementation

**Kernel 1 config:** `BLOCK_I = next_power_of_2(2880) = 4096`, `num_warps=4`, `num_stages=3`
**Kernel 2 config:** `BLOCK_H = next_power_of_2(2880) = 4096`, `num_warps=4`, `num_stages=3`

| ID | Kernel 1 (μs) | Kernel 2 (μs) | Total Triton (μs) | E2E (μs) | Triton % of E2E |
|----|--------------|--------------|-------------------|----------|-----------------|
| A1 | 7.7 | 108.6 | 116.3 | 2130.8 | 5.5% |
| A2 | 13.8 | 113.5 | 127.3 | 2192.7 | 5.8% |
| A3 | 33.4 | 121.3 | 154.7 | 2370.9 | 6.5% |
| A4 | 107.4 | 131.1 | 238.5 | 2879.3 | 8.3% |
| A5 | 402.4 | 172.0 | 574.4 | 6751.8 | 8.5% |
| B1 | 7.5 | 31.3 | 38.8 | 562.9 | 6.9% |
| B2 | 13.6 | 34.2 | 47.8 | 617.2 | 7.7% |
| B3 | 107.1 | 56.0 | 163.1 | 1693.1 | 9.6% |

> **GPU:** NVIDIA H100 80GB HBM3
> **Triton:** 3.5.1
> **PyTorch:** 2.10.0a0+b4e4ee81d3.nv25.12
> **Dtype:** bfloat16

______________________________________________________________________

### Iteration 1 — num_warps x num_stages sweep (Phase 1)

**What changed:**

- Swept all 25 combos of num_warps={1,2,4,8,16} x num_stages={1,2,3,4,5} for both kernels.
- No code changes to kernels yet — this identifies the best launch parameters per shape.

**Kernel 1 best configs per shape:**

| ID | Best config | Best (μs) | Baseline (μs) | Δ |
|----|------------|-----------|---------------|---|
| A1 | w=16, s=5 | 6.9 | 8.1 | -14.6% |
| A2 | w=16, s=5 | 13.6 | 13.7 | -0.8% |
| A3 | w=4, s=2 | 33.3 | 33.6 | -0.9% |
| A4 | w=4, s=2 | 107.3 | 107.6 | -0.3% |
| A5 | w=4, s=1 | 402.4 | 402.6 | -0.0% |
| B1 | w=16, s=4 | 6.3 | 7.4 | -14.5% |
| B2 | w=8, s=2 | 13.4 | 14.0 | -4.1% |
| B3 | w=4, s=5 | 107.3 | 107.3 | -0.0% |

**Kernel 2 best configs per shape:**

| ID | Best config | Best (μs) | Baseline (μs) | Δ |
|----|------------|-----------|---------------|---|
| A1 | w=16, s=1 | 59.6 | 109.0 | -45.3% |
| A2 | w=16, s=3 | 62.2 | 113.8 | -45.3% |
| A3 | w=16, s=2 | 68.5 | 121.3 | -43.5% |
| A4 | w=16, s=1 | 78.9 | 130.9 | -39.8% |
| A5 | w=16, s=3 | 145.0 | 172.2 | -15.8% |
| B1 | w=16, s=4 | 18.8 | 31.3 | -39.9% |
| B2 | w=16, s=1 | 20.9 | 34.3 | -39.0% |
| B3 | w=16, s=2 | 47.4 | 56.2 | -15.7% |

**Combined results (using per-shape best for each kernel):**

| ID | K1 (μs) | K2 (μs) | Total (μs) | Baseline Total (μs) | Δ vs baseline |
|----|---------|---------|------------|---------------------|---------------|
| A1 | 6.9 | 59.6 | 66.5 | 116.3 | -42.8% |
| A2 | 13.6 | 62.2 | 75.8 | 127.3 | -40.5% |
| A3 | 33.3 | 68.5 | 101.8 | 154.7 | -34.2% |
| A4 | 107.3 | 78.9 | 186.2 | 238.5 | -21.9% |
| A5 | 402.4 | 145.0 | 547.4 | 574.4 | -4.7% |
| B1 | 6.3 | 18.8 | 25.1 | 38.8 | -35.3% |
| B2 | 13.4 | 20.9 | 34.3 | 47.8 | -28.2% |
| B3 | 107.3 | 47.4 | 154.7 | 163.1 | -5.2% |

**Analysis:**

- **Kernel 2 is the big win:** w=16 is universally best, giving 15-45% improvement. The baseline w=4 severely underutilized parallelism.
- **Kernel 1 is already well-tuned:** baseline w=4,s=3 is near-optimal for large T. Only small-T shapes benefit from w=16 (~14%).
- **Key insight:** Kernel 2 needs more warps because it loops over E experts sequentially — more warps help hide the memory latency of loading expert outputs.
- **Next:** Apply w=16 to kernel 2 as default. For kernel 1, use w=16 for small T, w=4 for large T. Then move to structural changes.

### Iteration 2 — Apply sweep best configs to launcher

**What changed:**

- Kernel 1: adaptive num_warps (16 if total_rows\<=128 else 4), num_stages=2
- Kernel 2: num_warps=16, num_stages=2 (was 4, 3)
- Fixed benchmark script to use launcher-matching params and __main__ guard

**Correctness:** PASS (31/31 tests)

| ID | K1 (us) | K2 (us) | Total (us) | Baseline Total (us) | Delta vs baseline |
|----|---------|---------|------------|---------------------|-------------------|
| A1 | 6.7 | 59.8 | 66.5 | 116.3 | -42.8% |
| A2 | 13.6 | 62.3 | 75.9 | 127.3 | -40.4% |
| A3 | 33.3 | 68.8 | 102.1 | 154.7 | -34.0% |
| A4 | 107.2 | 79.0 | 186.2 | 238.5 | -21.9% |
| A5 | 402.2 | 144.9 | 547.1 | 574.4 | -4.8% |
| B1 | 6.2 | 19.0 | 25.2 | 38.8 | -35.1% |
| B2 | 13.7 | 21.0 | 34.7 | 47.8 | -27.4% |
| B3 | 107.5 | 47.2 | 154.7 | 163.1 | -5.2% |

**Analysis:**

- Confirmed sweep results hold: K2 sees 15-45% improvement from w=16.
- K1 improvement small but consistent at low T.
- Moving to Phase 2: structural changes.

### Iteration 3 — Kernel 2: adaptive 2D/1D grid over H dimension

**What changed:**

- Kernel 2 now uses a 2D grid `(T, cdiv(H, 1024))` when T\<=128 (more parallelism for decode).
- Falls back to 1D grid `(T, 1)` with BLOCK_H=next_pow2(H) when T>128 (prefill has enough parallelism).
- Each H-block processes a slice of the hidden dimension independently.

**Correctness:** PASS (31/31 tests)

| ID | K1 (us) | K2 (us) | Total (us) | Baseline Total (us) | Delta vs baseline |
|----|---------|---------|------------|---------------------|-------------------|
| A1 | 6.6 | 54.0 | 60.6 | 116.3 | -47.9% |
| A2 | 13.8 | 55.6 | 69.4 | 127.3 | -45.5% |
| A3 | 33.6 | 60.8 | 94.4 | 154.7 | -39.0% |
| A4 | 107.4 | 69.8 | 177.2 | 238.5 | -25.7% |
| A5 | 402.3 | 144.9 | 547.2 | 574.4 | -4.7% |
| B1 | 6.4 | 7.8 | 14.2 | 38.8 | -63.4% |
| B2 | 13.6 | 9.5 | 23.1 | 47.8 | -51.7% |
| B3 | 107.3 | 47.3 | 154.6 | 163.1 | -5.2% |

**Analysis:**

- B1/B2 (20B decode) see massive K2 improvement: 7.8/9.5 us vs baseline 31/34 us (-75%/-72%).
- A1-A4 (120B decode) K2 improved ~10% beyond iter 2, now ~50% better than baseline.
- A5/B3 (prefill) use 1D grid, same as iter 2 — no regression.
- The adaptive threshold (T\<=128) works well. Could fine-tune further.

### Iteration 4-5 — K1 coalesced loads + K2 adaptive BLOCK_H

**What changed:**

- K1: Rewrote kernel to read contiguous `[gate | up]` layout instead of stride-2 interleaved.
  Launcher deinterleaves `gate_up` via `gate_up[..., ::2]` / `gate_up[..., 1::2]` + `cat`.
- K2: Adaptive BLOCK_H: 256 for small E+T (B1), 1024 for medium T, next_pow2(H) for high T.
- K1 multi-row variant tested — no benefit, rpp=1 wins everywhere. Dead end.

**Correctness:** PASS (31/31 tests)

| ID | K1 (us) | K2 (us) | Total (us) | Baseline Total (us) | Delta vs baseline |
|----|---------|---------|------------|---------------------|-------------------|
| A1 | 6.6 | 54.4 | 61.0 | 116.3 | -47.5% |
| A2 | 12.8 | 55.5 | 68.3 | 127.3 | -46.3% |
| A3 | 31.3 | 61.0 | 92.3 | 154.7 | -40.3% |
| A4 | 101.9 | 70.0 | 171.9 | 238.5 | -27.9% |
| A5 | 383.7 | 145.2 | 528.9 | 574.4 | -7.9% |
| B1 | 6.0 | 7.6 | 13.6 | 38.8 | -64.9% |
| B2 | 12.6 | 10.1 | 22.7 | 47.8 | -52.5% |
| B3 | 101.9 | 47.3 | 149.2 | 163.1 | -8.5% |

**Analysis:**

- K1 coalesced loads: 5-7% improvement across all shapes. Well worth the deinterleave cost.
- K2 BLOCK_H=256 for B1 reduced from 7.8 to 7.6 us (marginal).
- Net effect is positive across all shapes. Biggest wins at decode (47-65%), modest at prefill (8%).
- E2E cost slightly up for large shapes (deinterleave overhead) but Triton kernel time is lower.

### Iteration 6 — K2: remove dtype_probe load, use OUTPUT_DTYPE constexpr

**What changed:**

- Removed `_dtype_probe = tl.load(expert_out_ptr, ...)` and added `OUTPUT_DTYPE: tl.constexpr`.
- No measurable perf change — cleanup only.

**Correctness:** PASS (31/31). Numbers identical to iter 4-5.

### Iteration 7 — K1: weight rearrange instead of activation deinterleave (REVERTED)

**What changed:**

- Tried rearranging `gate_up_w` from interleaved to `[gate|up]` layout so BMM output is already coalesced.

**Result:** REGRESSION. Weight tensor `[128,2880,5760]` is ~2.5GB — rearranging per forward is ~4x more expensive than activation deinterleave. E2E A1: 2117→8517 us. Reverted.

**Lesson:** Weight rearrange must be done at model load time (outside the op), not per-forward.

### Iteration 8 — K2: skip zero routing weights

**What changed:**

- Added `if w_f != 0.0:` guard in K2 expert loop to skip loading expert outputs
  when routing weight is zero. For GPT-OSS top-4/128 routing, 124/128 are zero.

**Correctness:** PASS (31/31 tests)

**Dense routing (benchmark default, all non-zero):** ~10% slower due to branch overhead.
**Sparse routing (top-4 of 128, real-world):** Massive improvement:

| ID | Dense (us) | Sparse (us) | Delta |
|----|-----------|------------|-------|
| A1 (E=128, T=1) | 59.2 | 12.8 | -78.4% |
| A2 (E=128, T=8) | 61.1 | 13.2 | -78.4% |
| A3 (E=128, T=32) | 65.4 | 14.0 | -78.6% |
| A4 (E=128, T=128) | 73.7 | 19.2 | -73.9% |
| B1 (E=32, T=1) | 18.6 | 7.6 | -58.9% |
| B2 (E=32, T=32) | 20.2 | 8.2 | -59.3% |

**Analysis:**

- For E=128 with top-4 routing, K2 drops from ~60 us to ~13 us (78% faster).
- For E=32 with top-4 routing, K2 drops from ~19 us to ~8 us (59% faster).
- Branch overhead for dense routing is small (~10%) — acceptable tradeoff.
- Combined with all prior optimizations, K2 went from baseline 108 us to 13 us for A1 (88% reduction).

### Iteration 9 — K1: 2D grid with adaptive BLOCK_I

**What changed:**

- K1 now uses 2D grid `(total_rows, cdiv(I, BLOCK_I))` with BLOCK_I=1024 for small total_rows.
- Falls back to 1D grid for large total_rows.
- ~1% K1 improvement at large T (A5: 383.7->379.6 us). Marginal but consistent.

**Correctness:** PASS (31/31)

### Iteration 10 — K1: unified num_warps=16

- Re-sweep revealed w=16 is now optimal for large-T K1 (was w=4 before coalesced loads).
- A3: 31.8->30.5 (-4.1%), A4: 101.4->99.2 (-2.2%), A5: 379.6->375.4 (-1.1%)

### Iteration 11 — Remove deinterleave, revert to stride-2 kernel

- Reverted K1 to stride-2 reads and removed launcher deinterleave (slicing+cat+contiguous).
- K1 kernel ~15% slower, but E2E ~12-16% faster (deinterleave copy was more expensive).
- A4 E2E: 3199->2826 (-12%), A5 E2E: 7982->6735 (-16%)

### Iteration 12 — Remove redundant .contiguous()

- BMM output is already contiguous. Removed no-op call. Also tested K2 routing preload — dead end.

### Iteration 13 — K1: native dtype compute

- Removed explicit fp32 upcast. Triton handles dtype promotion internally.
- Cleaner code, no perf regression. Tested eviction_policy and BLOCK_I=2048 — both dead ends.

### Iteration 14 — Bandwidth analysis

K1 is at **74-79% of H100 peak memory bandwidth** for large shapes — approaching hardware limits.
K2 with sparse routing is **latency-bound** (0.1-6% peak BW) — dominated by launch overhead.

| Kernel | Shape | BW utilization |
|--------|-------|---------------|
| K1 | A5 (65536 rows) | 79.2% |
| K1 | A4 (16384 rows) | 74.0% |
| K1 | A3 (4096 rows) | 58.8% |
| K1 | A1 (128 rows) | 10.0% |
| K2 | B3 sparse (T=512) | 33.6% |
| K2 | A1 sparse (T=1) | 0.1% |

Also tested: K2 w=8 marginally better than w=16 for sparse (~4%), K1 num_stages sweep flat.

______________________________________________________________________

## Current Best Results (iter 14) — Summary vs Baseline

**Dense routing (benchmark default):**

| ID | K1 baseline | K1 best | K2 baseline | K2 best | Total baseline | Total best | Delta |
|----|------------|---------|------------|---------|---------------|-----------|-------|
| A1 | 7.7 | 6.8 | 108.6 | 59.5 | 116.3 | 66.3 | -43.0% |
| A2 | 13.8 | 13.9 | 113.5 | 60.9 | 127.3 | 74.8 | -41.2% |
| A3 | 33.4 | 36.0 | 121.3 | 65.8 | 154.7 | 101.8 | -34.2% |
| A4 | 107.4 | 114.3 | 131.1 | 74.0 | 238.5 | 188.3 | -21.0% |
| A5 | 402.4 | 427.4 | 172.0 | 146.2 | 574.4 | 573.6 | -0.1% |
| B1 | 7.5 | 6.0 | 31.3 | 18.7 | 38.8 | 24.7 | -36.3% |
| B2 | 13.6 | 13.7 | 34.2 | 20.2 | 47.8 | 33.9 | -29.1% |
| B3 | 107.1 | 114.3 | 56.0 | 47.9 | 163.1 | 162.2 | -0.6% |

Note: K1 kernel isolated time is higher than iter 4-5 because iter 11 removed the
deinterleave copy (which was more expensive at E2E level). E2E is what matters.

**With sparse top-4 routing (real-world GPT-OSS):**

| ID | K2 sparse (us) | K1 (us) | Total (us) | vs baseline |
|----|---------------|---------|-----------|-------------|
| A1 | ~12.8 | 6.8 | ~19.6 | **-83.1%** |
| A2 | ~13.2 | 13.9 | ~27.1 | **-78.7%** |
| A3 | ~13.6 | 36.0 | ~49.6 | **-67.9%** |
| A4 | ~19.0 | 114.3 | ~133.3 | **-44.1%** |
| B1 | ~7.6 | 6.0 | ~13.6 | **-64.9%** |
| B2 | ~8.2 | 13.7 | ~21.9 | **-54.2%** |

**E2E improvement (including BMM):**

| ID | E2E baseline (us) | E2E best (us) | Delta |
|----|-------------------|---------------|-------|
| A1 | 2130.8 | 2108.9 | -1.0% |
| A4 | 2879.3 | 2826.3 | -1.8% |
| A5 | 6751.8 | 6701.5 | -0.7% |
| B3 | 1693.1 | 1691.5 | -0.1% |

______________________________________________________________________

## 4. Optimization Ideas Backlog

- \[x\] **num_warps / num_stages sweep:** Done in iter 1-2.
- \[x\] **Kernel 2: 2D grid (T x H_blocks):** Done in iter 3.
- \[x\] **Coalesced loads for K1:** Done in iter 4, reverted in iter 11 (deinterleave cost > kernel gain).
- \[x\] **K2 BLOCK_H sweep:** Done in iter 4.
- \[x\] **K1 persistent kernel:** Dead end (iter 4).
- \[x\] **K2 dtype_probe removal:** Cleanup (iter 6).
- \[x\] **K1 weight rearrange per-forward:** Dead end (iter 7).
- \[x\] **K2 skip zero routing weights:** Done in iter 8. 59-79% K2 speedup with sparse routing.
- \[x\] **K1 2D grid over I dimension:** Done in iter 9. ~1% improvement.
- \[x\] **K1 num_warps re-sweep:** Done in iter 10. w=16 unified.
- \[x\] **K1 remove deinterleave:** Done in iter 11. E2E 12-16% faster.
- \[x\] **K1 native dtype compute:** Done in iter 13. Cleaner, no regression.
- \[x\] **Bandwidth analysis:** Done in iter 14. K1 at 74-79% peak BW. Near HW limits.
- \[x\] **K2 routing preload:** Dead end (iter 12). E2E is BMM-dominated.
- \[x\] **K1 eviction_policy / BLOCK_I=2048:** Dead ends (iter 13).
- \[x\] **K2 num_warps re-sweep with sparse:** Done (iter 14). w=8 marginal, not worth complexity.
- \[ \] **K1: eliminate deinterleave via load-time weight rearrange** (outside the op).
- \[ \] **Fuse kernel 1 + BMM:** Longer-term, fuse activation into GEMM epilogue.
- \[ \] **K2: completely rewrite for sparse-only** (skip the expert loop entirely, use gather).

______________________________________________________________________

## 5. Final Best Configuration

> (To be filled after optimization loop completes)

| Kernel | Config | Notes |
|--------|--------|-------|
| `_fused_glu_activation_kernel` | | |
| `_weighted_expert_sum_kernel` | | |

______________________________________________________________________

## Appendix: How to Reproduce

```bash
cd tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe
python bench_triton_moe_dense_mlp.py
```
