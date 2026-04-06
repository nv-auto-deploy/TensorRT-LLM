# Custom Kernel Optimization Guide — Nemotron Nano v3

Branch: `ad-fast-iter/nemotron-nano-20260323`
Target model: Nemotron Nano v3 30B-A3B-FP8, 4×H100

This document lists every new Triton kernel added in this branch, its current
status, and specific optimization opportunities ranked by expected impact.

______________________________________________________________________

## Kernel 1: `fused_mamba_decode.py` — Fused conv1d + SSM decode

**Added:** iter 19-20
**Status:** Active (re-enabled in this session after prefill activation fix)
**Dispatch:** `fused_cached_conv_ssm.py` — fires when `num_decode > 16`
**Transform:** `fuse_conv_ssm: enabled: true`

### What it does

Fuses `causal_conv1d_update + SiLU + SSM_update` into a single Triton kernel
for decode tokens, eliminating one intermediate HBM round-trip.

### Shapes (Nemotron Nano v3)

| Param | Value |
|-------|-------|
| nheads | 64 |
| dim (head_dim) | 64 |
| dstate | 128 |
| ngroups | 8 |
| nheads_per_group | 8 |
| conv_dim | 6144 |
| kernel_width | 4 |
| BLOCK_DIM | 64 |
| BLOCK_DSTATE | 128 |
| num_warps | **4 (hardcoded)** |

Grid: `(1, batch, 64)` → `batch × 64` thread blocks

### Optimization opportunities

| # | Item | Type | Expected Impact | Effort |
|---|------|------|-----------------|--------|
| O1 | **`dt_clamp` missing** — `softplus(dt+dt_bias)` not clamped to `[time_step_min, time_step_max]`. 94% of real dt values exceed max=0.1, causing wrong SSM states. Add `DT_CLAMP_MIN/MAX: tl.constexpr` + `tl.minimum/maximum` after softplus. | **Correctness fix** | Critical | Low |
| O2 | **`num_warps` heuristic** — hardcoded to 4. SSM state load is \[64×128\]=32KB per block. With 4 warps (128 threads) each thread handles 64 fp32 elements; 8 warps doubles thread count, enables 128-byte coalesced LDG.128. Use `num_warps=8` for `batch>=32`. Previously measured +7% at batch=256. | Perf | +5–10% at batch≥32 | Low |
| O3 | **B/C conv computed 8× redundantly per group** — `nheads_per_group=8` heads all compute identical B/C conv output (same group, same conv state). Lines 193–235 run 8× but produce the same values. Only the first head writes back state (line 243). Fix: compute B/C once per group via shared memory, broadcast to remaining 7 heads. | Perf | +10–15% (saves 7/8 of B/C conv) | Medium |
| O4 | **`num_stages` not set** — SSM state load (32KB, HBM-bound) has high latency. Triton `num_stages=3` or `4` can overlap loads with compute. Currently default (2). | Perf | +3–7% | Low |
| O5 | **Dispatch threshold** — `_FUSED_DECODE_THRESHOLD=16` means FlashInfer for batch≤16, fused Triton for batch>16. First CUDA graph batch using fused kernel is 24. FlashInfer is 3× faster at batch=1 (8µs vs 25µs); crossover point vs fused Triton is unknown. Benchmark batch=8,16,24 to tune threshold. | Perf | Unknown | Low |

**Priority:** O1 (correctness, must fix) → O2 (one-liner, proven) → O4 → O3 → O5

______________________________________________________________________

## Kernel 2: `tuned_ssm_kernel.py` — Tuned SSM state update

**Added:** iter 11
**Status:** Active — used by `flashinfer_ssm` backend for `num_decode > 32`
**Superseded by:** `fused_mamba_decode.py` for the fused path; still active as fallback

### What it does

Replaces FlashInfer's stock `selective_state_update` with a Triton kernel tuned
for Nemotron shapes. Key change vs stock: `BLOCK_SIZE_M=16` (vs 4), reducing
grid size by 4× for dim=64.

### Current params

```
BLOCK_SIZE_M = 16
BLOCK_SIZE_DSTATE = 128
num_warps = 4  (hardcoded for all batch sizes)
```

### Optimization opportunities

| # | Item | Type | Expected Impact | Effort |
|---|------|------|-----------------|--------|
| O1 | **`dt_clamp` missing** — same issue as fused_mamba_decode. No `time_step_limit` clamp after softplus. | **Correctness fix** | Critical | Low |
| O2 | **`num_warps` not swept for batch>32** — hardcoded to 4 regardless of batch. For batch≥64, try num_warps=8. | Perf | +5% at batch≥64 | Low |
| O3 | **This kernel is on the hot path only when `fuse_conv_ssm` is disabled** — with fuse_conv_ssm enabled, this kernel is bypassed for decode. Lower priority. | Structural | — | — |

______________________________________________________________________

## Kernel 3: `relu2_quant_fp8.py` — Fused ReLU² + FP8 quantization

**Added:** iter 21
**Status:** Active (`fuse_relu2_quant_fp8: enabled: true`)
**Transform:** `fuse_relu2_quant_fp8.py`

### What it does

Fuses `x * x * relu(x)` squaring + FP8 quantization into one elementwise kernel
for MLP layers. Eliminates a separate `scaleMatrixPerTensorVec` kernel.

### Current params

```
BLOCK = 512
num_warps = 4
```

### Optimization opportunities

| # | Item | Type | Expected Impact | Effort |
|---|------|------|-----------------|--------|
| O1 | **Bandwidth-bound elementwise** — kernel is memory-bandwidth limited. nvjet already achieves ~85% BW on H100 for elementwise ops at these sizes. Limited headroom. | Analysis | ~0% | Low |
| O2 | **BLOCK size not tuned** — BLOCK=512 is a guess. For `hidden_size=2688` (not power-of-2), try BLOCK=128 or 256 to avoid wasted tail iterations. | Perf | \<2% | Low |

**Assessment: Near-optimal. Not worth further effort.**

______________________________________________________________________

## Kernel 4: `gated_rms_norm_quant_fp8.py` — Fused gated RMSNorm + FP8 quant

**Added:** iter 37
**Status:** Active (`fuse_gated_rmsnorm_quant_fp8: enabled: true`), neutral TPOT result
**Transform:** `fuse_gated_rmsnorm_quant_fp8.py`

### What it does

Fuses gate multiply + RMSNorm + FP8 quantization for Mamba `out_proj` layers.
Fires on 2/2 FP8 Mamba layers.

### Current params

```
BLOCK_N = next_power_of_2(group_size)  (tuned per shape)
num_warps = min(max(BLOCK_N // 256, 1), 8)  (formula-based)
```

### Optimization opportunities

| # | Item | Type | Expected Impact | Effort |
|---|------|------|-----------------|--------|
| O1 | **Result was neutral at c=1** — the fused kernel is correct and saves kernel launches, but TPOT didn't move. This is a norm kernel (2-pass HBM-bound). Already well-structured. | Analysis | ~0% | — |
| O2 | **`num_stages=2` in pass 2** — the second pass uses tiled streaming but no explicit `num_stages`. Already has `num_stages=2` in `tl.range`. Unlikely to help further. | Perf | \<1% | Low |

**Assessment: Already well-optimized for its arithmetic intensity. Not worth further effort.**

______________________________________________________________________

## Kernel 5: `triton_fused_add_rms_norm_quant_fp8.py` — Two-pass RMSNorm + FP8 quant

**Added:** iter 25 (two-pass), extended through iter 33/40
**Status:** Active (`fuse_rmsnorm_quant_fp8: enabled: true`)
**Transform:** `fuse_rmsnorm_quant_fp8.py`

### What it does

Two-pass streaming RMSNorm + FP8 quantization (and fused-add variant). Adaptive
dispatch: FlashInfer for `seq_len≤32` (decode), Triton for `seq_len>32` (prefill/large batch).

### Current params

```
BLOCK_N = 128   (tuned: 2688 = 128×21, zero waste)
num_warps = 4
num_stages = 2  (pass 2 prefetch)
```

### Optimization opportunities

| # | Item | Type | Expected Impact | Effort |
|---|------|------|-----------------|--------|
| O1 | **Bandwidth-bound 2-pass** — pass 1 reads all elements to compute RMS; pass 2 re-reads + writes. Two full HBM passes is the theoretical minimum for online normalization. No algorithmic headroom. | Analysis | ~0% | — |
| O2 | **FlashInfer threshold** — `_RMN_ADAPTIVE_THRESHOLD=32` was tuned empirically. At batch>32 the Triton kernel fires; at batch≤32 FlashInfer is faster (fewer launches). Threshold is already correct. | Analysis | — | — |
| O3 | **`num_warps=4` at large batch** — at seq_len=128+, try num_warps=8 for better HBM utilization. Minor. | Perf | \<3% | Low |

**Assessment: Already well-optimized. BLOCK_N=128 is a key insight for Nemotron hidden_size=2688.**

______________________________________________________________________

## Kernel 6: `bf16_gemv.py` — BF16 GEMV for LM head

**Added:** iter 44
**Status:** Disabled (`fuse_lm_head_gemv: enabled: false`)

### What it does

Custom Triton GEMV for the LM head linear at small decode batch (M≤4).

### Why disabled

nvjet already achieves ~85% memory bandwidth on the TP-sharded weight matrix
`[2688, 32768]`. Triton kernel was neutral vs nvjet. Not worth re-enabling.

**Assessment: Discard. nvjet wins.**

______________________________________________________________________

## Summary Table

| Kernel | File | Status | Perf Headroom | Priority |
|--------|------|--------|---------------|----------|
| Fused conv+SSM | `fused_mamba_decode.py` | ✅ active | **High** — dt_clamp bug + redundant B/C + num_warps | **1** |
| Tuned SSM | `tuned_ssm_kernel.py` | ✅ active (fallback) | Low — superseded by fused kernel | 3 |
| ReLU² + FP8 | `relu2_quant_fp8.py` | ✅ active | None — BW-bound, nvjet-level | 5 |
| Gated RMSNorm + FP8 | `gated_rms_norm_quant_fp8.py` | ✅ active | None — already well-optimized | 4 |
| RMSNorm + FP8 | `triton_fused_add_rms_norm_quant_fp8.py` | ✅ active | None — 2-pass is optimal | 4 |
| BF16 GEMV | `bf16_gemv.py` | ❌ disabled | N/A — nvjet wins | discard |

**The only kernel worth further optimization work is `fused_mamba_decode.py`.**
Fix O1 (dt_clamp correctness) first, then O2 (num_warps), then O3 (B/C redundancy).
