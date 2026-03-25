# `_mla_attention_kernel` Optimization Log

**File:** `tensorrt_llm/_torch/auto_deploy/custom_ops/mla/triton_mla.py`
**Started:** 2026-03-24
**GPU:** NVIDIA H100 80GB HBM3
**PyTorch:** 2.10.0a0+b4e4ee81d3.nv25.12
**Triton:** 3.5.1
**Dtype:** bfloat16 (cache + model), float32 (kernel accumulators)

______________________________________________________________________

## Kernel Overview

### `_mla_attention_kernel` (single kernel, shared by decode and prefill)

**What it computes:**
MLA (Multi-head Latent Attention) in compressed KV space with online softmax.
Each program handles one `(token, head)` pair and iterates over the KV cache
in blocks of `SEQ_BLOCK` positions.

```
score[t] = (q_absorbed[h] · compressed_kv[t] + q_pe[h] · kpe[t]) * scale
attn     = softmax(score[:kv_len])
output   = Σ attn[t] * compressed_kv[t]        # [KV_LORA_RANK]
```

The surrounding weight absorptions (`q_absorbed = q_nope @ W_kn^T`) and value
projection (`out = weighted_kv @ W_v^T`) are PyTorch einsum ops **outside** the
Triton kernel — they are out of scope for this optimization (but are fusion candidates).

**Signature (constexpr dims):**

```python
Grid: (num_tokens, N_HEADS)   # one program per (token, head)
KV_BLOCK = 256   # next_power_of_2(KV_LORA_RANK=256)
PE_BLOCK = 64    # next_power_of_2(QK_ROPE_HEAD_DIM=64)
```

**Launch parameters:**
| Path    | SEQ_BLOCK | num_warps | num_stages |
|---------|-----------|-----------|------------|
| Decode  | 8         | 2         | 2          |
| Prefill | 16        | 4         | 2          |

**Tiling strategy:**

- Grid dimension 0 = token index; dimension 1 = head index.
- Inner loop over `kv_len // SEQ_BLOCK` blocks of the KV cache sequence.
- Per block: loads `[SEQ_BLOCK, KV_BLOCK]` ckv and `[SEQ_BLOCK, PE_BLOCK]` kpe from HBM.

**Memory access pattern:**

- q_absorbed and q_pe loaded once per program into registers.
- ckv/kpe loaded from HBM in `SEQ_BLOCK`-row tiles — stride-1 across CACHE_DIM
  (coalesced in the innermost dim, contiguous layout `[max_batch, max_seq, CACHE_DIM]`).
- Output `weighted_kv` written once per program.

**CRITICAL ISSUE — Redundant cache reads across heads:**
For a single token, all `N_HEADS=32` programs load **identical** ckv and kpe data
from the cache. Per-token redundancy:

- Data unique to one token: `kv_len × CACHE_DIM × 2 bytes`
- Data actually loaded:     `kv_len × CACHE_DIM × 2 bytes × 32 heads`
- Redundancy factor: **32×**

For `kv_len=2048`: 2048 × 320 × 2 = 1.31 MB unique data, but 41.9 MB loaded.
At H100 BW of 3350 GB/s: theoretical minimum ≈ **0.39 µs**; baseline: **491 µs** — ~1260× over roofline.
Even accounting for the 32 heads loading in parallel, if we could share the cache load, we'd save ~32× in HBM traffic.

**Surrounding torch ops (out of scope, but fusion candidates):**

- Decode: `torch.einsum("bnd,ndk->bnk", q_nope, w_k_nope)` → weight absorption (GPU compute)
- Decode: `torch.einsum("bnk,nvk->bnv", weighted_kv, w_v)` → value projection
- Prefill: same pattern, plus vectorized cache scatter using `index_select`

______________________________________________________________________

## Bottleneck Classification

| Metric | Value |
|--------|-------|
| CACHE_DIM | 320 (256 ckv + 64 kpe) |
| Bytes per (token, head, kv_pos) | 320 × 2 = 640 B |
| FLOPS per (token, head, kv_pos) | ~1152 (dot products + softmax update + weighted sum) |
| Arithmetic intensity per head | ~1.8 FLOP/byte |
| H100 ridge point (fp16) | ~591 FLOP/byte |
| Classification | **Heavily memory-bound** |

**Roofline analysis (A4: T=1, kv=1024):**

| Scenario | HBM traffic | Theoretical min | Actual | Overhead |
|----------|------------|-----------------|--------|----------|
| All 32 heads load independently | 32 × 655 KB = 20.9 MB | 6.2 µs | 249 µs | ~40× |
| Ideal (cache loaded once, reused) | 655 KB | 0.20 µs | — | — |

**Root causes of overhead:**

1. 32× redundant HBM loads (biggest factor) — each head re-loads the same cache rows
1. Small `SEQ_BLOCK=8` → 128 inner loop iterations for kv=1024 → loop overhead dominates
1. Low SM utilization for small batch/short sequences (grid=(1,32) = 32 programs on 132 SMs)

______________________________________________________________________

## Target Models & Benchmark Shapes

### Model A: Mistral-Small-4-119B-2603

| Dim | Value |
|-----|-------|
| `kv_lora_rank` | 256 |
| `qk_rope_head_dim` | 64 |
| `qk_nope_head_dim` | 64 |
| `v_head_dim` | 128 |
| `num_attention_heads` | 32 |
| `CACHE_DIM` | 320 |
| `qk_head_dim` (for scale) | 128 → scale = 0.0884 |

### Shape Matrix

| ID | Model | B | T | kv_len | Description |
|----|-------|---|---|--------|-------------|
| A1 | MS4   | 1 | 1 | 64     | decode, short ctx |
| A2 | MS4   | 1 | 1 | 256    | decode, medium ctx |
| A3 | MS4   | 1 | 1 | 512    | decode, medium ctx |
| A4 | MS4   | 1 | 1 | 1024   | decode, long ctx |
| A5 | MS4   | 1 | 1 | 2048   | decode, very long ctx |
| A6 | MS4   | 8 | 8 | 256    | batched decode |
| A7 | MS4   | 8 | 8 | 512    | batched decode |
| A8 | MS4   | 16| 16| 512    | batched decode |
| A9 | MS4   | 32| 32| 256    | batched decode |
| A10| MS4   | 32| 32| 512    | batched decode |
| B1 | MS4   | 1 |128| 128    | prefill, short |
| B2 | MS4   | 1 |512| 512    | prefill, medium |
| B3 | MS4   | 1 |1024|1024   | prefill, long |
| B4 | MS4   | 1 |2048|2048   | prefill, very long |

______________________________________________________________________

## Current Best Summary

*Updated after every iteration.*

| ID  | Best kernel µs | Config                                          | Iter | vs Baseline |
| --- | -------------- | ----------------------------------------------- | ---- | ----------- |
| A1  | 9.4            | SEQ_BLOCK=64, warps=4, stages=4                 | 4    | **2.2×**    |
| A2  | 17.0           | SEQ_BLOCK=128, warps=8, stages=2                | 4    | **3.9×**    |
| A3  | 28.1           | SEQ_BLOCK=128, warps=8, stages=4                | 4    | **4.5×**    |
| A4  | 50.4           | SEQ_BLOCK=128, warps=8, stages=4                | 4    | **4.9×**    |
| A5  | 94.6           | SEQ_BLOCK=128, warps=8, stages=5                | 4    | **5.2×**    |
| A6  | 26.4           | SEQ_BLOCK=64, warps=4, stages=2                 | 4    | **2.6×**    |
| A7  | 45.7           | SEQ_BLOCK=64, warps=4, stages=4                 | 4    | **2.9×**    |
| A8  | 53.4           | multihead HB=4, SEQ_BLOCK=16, warps=4, stgs=2   | 5    | **2.8×**    |
| A9  | 32.5           | multihead HB=4, SEQ_BLOCK=16, warps=4, stgs=2   | 5    | **4.0×**    |
| A10 | 56.0           | multihead HB=4, SEQ_BLOCK=16, warps=4, stgs=2   | 5    | **4.5×**    |
| B1  | 25.6           | multihead HB=8, SEQ_BLOCK=16, warps=4, stgs=4   | 6    | **15.5×**   |
| B2  | 177.5          | multihead HB=16, SEQ_BLOCK=16, warps=4, stgs=4  | 6    | **34.4×**   |
| B3  | 515.3          | multihead HB=32, SEQ_BLOCK=16, warps=8, stgs=4  | 6    | **46.9×**   |
| B4  | 1836.7         | multihead HB=32, SEQ_BLOCK=16, warps=8, stgs=4  | 6    | **52.3×**   |

______________________________________________________________________

## Optimization Iterations

### Iteration 0 — Baseline

**Config:** Decode: SEQ_BLOCK=8, num_warps=2, num_stages=2 | Prefill: SEQ_BLOCK=16, num_warps=4, num_stages=2

| ID  | kernel µs  | vs baseline |
|-----|-----------|-------------|
| A1  | 20.9      | —           |
| A2  | 66.6      | —           |
| A3  | 127.1     | —           |
| A4  | 249.0     | —           |
| A5  | 491.1     | —           |
| A6  | 68.5      | —           |
| A7  | 130.2     | —           |
| A8  | 149.2     | —           |
| A9  | 130.3     | —           |
| A10 | 252.6     | —           |
| B1  | 396.4     | —           |
| B2  | 6107.9    | —           |
| B3  | 24183.3   | —           |
| B4  | 96220.1   | —           |

**Commit:** iter 0 — benchmark script + baseline

______________________________________________________________________

### Iteration 1 — Per-shape config lookup tables (decode + prefill)

**Change:** Replace hard-coded `SEQ_BLOCK=8, warps=2, stages=2` (decode) and
`SEQ_BLOCK=16, warps=4, stages=2` (prefill) with runtime-selected configs from
`_get_mla_decode_config(num_tokens)` and `_get_mla_prefill_config(total_tokens)`.
Analysis-driven initial values; to be refined after full sweep.

| num_tokens | decode SEQ_BLOCK | warps | stages |
| ---------- | ---------------- | ----- | ------ |
| ≤4         | 128              | 8     | 4      |
| ≤12        | 64               | 4     | 4      |
| ≤24        | 32               | 2     | 2      |
| >24        | 16               | 1     | 2      |

| total_tokens | prefill SEQ_BLOCK | warps | stages |
| ------------ | ----------------- | ----- | ------ |
| ≤512         | 8                 | 1     | 5      |
| >512         | 16                | 1     | 5      |

Correctness: PASS (all 14 shapes). Benchmark pending full sweep run.

**Commit:** iter 1 — per-shape lookup tables for decode and prefill

______________________________________________________________________

### Iteration 2 — HEAD_BLOCK tiling: `_mla_attention_kernel_multihead`

**Change:** New `@triton.jit` kernel `_mla_attention_kernel_multihead` with grid
`(num_tokens, N_HEADS // HEAD_BLOCK)`. Each program processes `HEAD_BLOCK`
consecutive heads, loading ckv+kpe ONCE per SEQ_BLOCK (shared across all
HEAD_BLOCK heads). Uses `tl.dot(bf16 × bf16 → fp32)` for tensor-core
acceleration. Added `_get_mla_multihead_config(num_tokens, is_prefill)` lookup.

Key constraints: `SEQ_BLOCK >= 16` (tl.dot minimum K); `HEAD_BLOCK` must divide
`N_HEADS=32`; valid HEAD_BLOCK values: 1, 2, 4, 8, 16, 32.

**Prefill results (benchmark on single GPU H100):**

| ID  | baseline µs | HB=2  | HB=4  | HB=8  | HB=16  | HB=32   | best     | speedup  |
| --- | ----------- | ----- | ----- | ----- | ------ | ------- | -------- | -------- |
| B1  | 396.4       | 73.6  | 41.5  | 25.6  | 29.0   | 47.2    | **25.6** | **15.5×** |
| B2  | 6107.9      | 872.0 | 444.0 | 243.0 | 177.5  | 220.0   | **177.5** | **34.4×** |
| B3  | 24183.3     | 3420  | 1730  | 944   | 614    | 515.0   | **515.0** | **46.9×** |
| B4  | 96220.1     | 13640 | 6870  | 3720  | 2210   | 1842.0  | **1842.0** | **52.2×** |

Correctness: PASS for all HEAD_BLOCK variants across all 14 shapes.

**Commit:** iter 2 — \_mla_attention_kernel_multihead with HEAD_BLOCK tiling

______________________________________________________________________

### Iteration 3 — Multi-GPU benchmark infrastructure

**Change:** Added `--gpu-id N` flag to `sweep_triton_mla.py` and
`parallel_bench_mla.sh` to orchestrate 8 GPU jobs simultaneously. Modes:
`sweep`, `head_block`, `correctness`, `benchmark`. Also fixed `tl.dot`
SEQ_BLOCK≥16 constraint in `bench_multihead_kernel` (`assert SEQ_BLOCK >= 16`,
`max(params["SEQ_BLOCK"], 16)` in HEAD_BLOCK mode). No kernel logic changes.

**Commit:** iter 3 — parallel_bench_mla.sh + multi-GPU infra + SEQ_BLOCK≥16 fix

______________________________________________________________________

### Iteration 4 — Full 150-config parameter sweep; update decode/prefill lookup tables

**Change:** Ran full `--sweep` (SEQ_BLOCK×warps×stages grid, 150 configs × 14 shapes).
Updated `_get_mla_decode_config` to take `max_kv_len` as second parameter — for
single-token decode, context length is the key sweep dimension. Updated
`_get_mla_prefill_config` with sweep-optimal defaults (fallback path; multihead
kernel preferred for prefill).

**Sweep results (best original kernel per shape):**

| ID  | baseline µs | best µs | speedup | Config                         |
| --- | ----------- | ------- | ------- | ------------------------------ |
| A1  | 20.9        | 9.4     | 2.2×    | SEQ_BLOCK=64,  warps=4, stgs=4 |
| A2  | 66.6        | 17.0    | 3.9×    | SEQ_BLOCK=128, warps=8, stgs=2 |
| A3  | 127.1       | 28.1    | 4.5×    | SEQ_BLOCK=128, warps=8, stgs=4 |
| A4  | 249.0       | 50.4    | 4.9×    | SEQ_BLOCK=128, warps=8, stgs=4 |
| A5  | 491.1       | 94.6    | 5.2×    | SEQ_BLOCK=128, warps=8, stgs=5 |
| A6  | 68.5        | 26.4    | 2.6×    | SEQ_BLOCK=64,  warps=4, stgs=2 |
| A7  | 130.2       | 45.7    | 2.9×    | SEQ_BLOCK=64,  warps=4, stgs=4 |
| A8  | 149.2       | 62.0    | 2.4×    | SEQ_BLOCK=32,  warps=2, stgs=2 |
| A9  | 130.3       | 43.6    | 3.0×    | SEQ_BLOCK=16,  warps=1, stgs=3 |
| A10 | 252.6       | 79.5    | 3.2×    | SEQ_BLOCK=16,  warps=1, stgs=2 |
| B1  | 396.4       | 82.5    | 4.8×    | SEQ_BLOCK=16,  warps=1, stgs=1 |
| B2  | 6107.9      | 1402.3  | 4.4×    | SEQ_BLOCK=8,   warps=1, stgs=5 |
| B3  | 24183.3     | 8152.5  | 3.0×    | SEQ_BLOCK=16,  warps=1, stgs=5 |
| B4  | 96220.1     | 31584.2 | 3.0×    | SEQ_BLOCK=16,  warps=1, stgs=1 |

Note: B-shapes are better served by the multihead kernel (iter 2): B1=25.6µs, B2=177.5µs,
B3=515µs, B4=1842µs — these are 3-17× faster than the original kernel's best.

**Commit:** iter 4 — full sweep results; update decode/prefill lookup tables

______________________________________________________________________

### Iteration 5 — Decode HEAD_BLOCK sweep; dispatch multihead for B≥16

**Change:** Ran decode HEAD_BLOCK sweep (HB=1,2,4,8,16,32) across all shapes on 8 GPUs.
Finding: multihead kernel is better than original for `num_tokens >= 16` (A8-A10 shapes).
For smaller batches, the original kernel with SB=64-128 remains faster.

Updated `_triton_mla_decode` to dispatch based on batch size:

- `num_tokens < 16`: original `_mla_attention_kernel` with lookup-table configs
- `num_tokens >= 16`: `_mla_attention_kernel_multihead` with HB=4

**Decode HEAD_BLOCK sweep (all shapes, SB=16, H100):**

| ID  | orig-best | HB=1  | HB=2  | HB=4  | HB=8  | HB=16 | HB=32 | dispatch |
| --- | --------- | ----- | ----- | ----- | ----- | ----- | ----- | -------- |
| A1  | 9.4       | 13.4  | 13.8  | 13.9  | 14.1  | 16.7  | 22.1  | original |
| A2  | 17.0      | 29.1  | 30.4  | 31.1  | 32.0  | 33.8  | 40.1  | original |
| A3  | 28.1      | 49.8  | 52.2  | 53.9  | 55.7  | 56.0  | 63.9  | original |
| A4  | 50.4      | 91.3  | 95.2  | 99.0  | 102.4 | 100.3 | 111.0 | original |
| A5  | 94.6      | 173.1 | 180.0 | 187.6 | 193.9 | 187.5 | 202.4 | original |
| A6  | 26.4      | 29.7  | 30.9  | 30.7  | 31.2  | 33.5  | 39.8  | original |
| A7  | 45.7      | 51.3  | 53.3  | 52.5  | 53.2  | 54.8  | 63.0  | original |
| A8  | 62.0      | 55.5  | 54.4  | **53.4** | 53.9 | 55.1 | 65.4 | mhead HB=4 |
| A9  | 43.6      | 40.7  | 35.2  | **32.5** | 32.6 | 34.4 | 41.2 | mhead HB=4 |
| A10 | 79.5      | 70.0  | 59.8  | **56.0** | 56.1 | 56.9 | 65.2 | mhead HB=4 |
| B1  | 82.5      | 118.0 | 65.4  | 42.2  | 25.6  | 26.9  | 28.1  | (prefill) |
| B2  | 1402.3    | 2103.9 | 1026.5 | 523.8 | 282.2 | 178.1 | 188.1 | (prefill) |
| B3  | 8152.5    | 7989.2 | 3958.3 | 2143.8 | 1151.8 | 630.3 | 515.3 | (prefill) |
| B4  | 31584.2   | 30957.5 | 14818.6 | 7645.2 | 4053.0 | 2373.2 | 1836.7 | (prefill) |

Correctness: PASS all 14 shapes. A8-A10 confirmed better with multihead HB=4.

**Commit:** iter 5 — decode HEAD_BLOCK sweep; multihead dispatch for B≥16

______________________________________________________________________

### Iteration 6 — Dispatch prefill to multihead kernel

**Change:** Updated `_triton_mla_prefill` to always use `_mla_attention_kernel_multihead`
via `_get_mla_multihead_config(total_tokens, is_prefill=True)`. The original kernel
fallback config is retained for reference but no longer dispatched.

Expected prefill improvements (from iter 2+5 standalone benchmarks):

| ID  | orig-best µs | multihead µs | speedup |
| --- | ------------ | ------------ | ------- |
| B1  | 82.5         | 25.6         | 3.2×    |
| B2  | 1402.3       | 178.1        | 7.9×    |
| B3  | 8152.5       | 515.3        | 15.8×   |
| B4  | 31584.2      | 1836.7       | 17.2×   |

Combined with iter 5 (decode B≥16 multihead), all shapes now see multihead dispatch.
Correctness: PASS all 14 shapes.

**Commit:** iter 6 — dispatch prefill to multihead kernel (always)

______________________________________________________________________

### Iteration 7 — `evict_first` hints on cache loads (both kernels)

**Change:** Added `eviction_policy="evict_first"` to all `tl.load` calls that read
from `mla_cache` in both `_mla_attention_kernel` and `_mla_attention_kernel_multihead`.
Each cache position is read exactly once per program; hinting L2 to evict early frees
capacity for subsequent blocks. No kernel logic change.

Correctness: PASS all 14 shapes. Effect on original kernel (SB=128, warps=8): negligible
(A2=17.0µs, A4=50.7µs — within noise of iter4 sweep). Effect on multihead kernel
not isolated (SB=64 results from iter8 sweep are with evict_first).

**Commit:** iter 7 — evict_first hints on all cache loads

______________________________________________________________________

## Optimization Ideas Backlog

### A.2 Tiling & SEQ_BLOCK \[HIGHEST PRIORITY\]

- \[x\] **SEQ_BLOCK sweep (4,8,16,32,64,128)** — Done (iter 1): lookup table implemented, full GPU sweep pending.
- \[x\] **Shape-conditional SEQ_BLOCK** — Done (iter 1): `_get_mla_decode_config` and `_get_mla_prefill_config` implemented.
- \[x\] **Separate decode vs prefill SEQ_BLOCK** — Done (iter 1): separate functions for decode vs prefill.

### A.1 Memory Access Patterns

- \[x\] **Reduce head-redundant loads via HEAD_BLOCK tiling** — Done (iter 2): `_mla_attention_kernel_multihead` with HEAD_BLOCK tiling. Prefill: 15-52× speedup. Decode: pending sweep.
- \[x\] **Cache load eviction hint `evict_first`** — Done (iter 7): added to all cache loads in both kernels. Benefit pending benchmark.
- \[ \] **Wider loads via `other` alignment** — ensure KV_BLOCK=256 and PE_BLOCK=64 loads are 128-byte aligned for vectorized HBM transactions. **Impact: Low** | All shapes | Correctness risk: No

### A.5 Parallelism & Occupancy

- \[ \] **num_warps sweep (1,2,4,8,16)** — Why: current warps=2 for decode may underutilize warp-level parallelism. **Impact: Medium** | All shapes | Correctness risk: No
- \[ \] **num_stages sweep (1,2,3,4,5)** — Why: more stages pipeline the inner HBM loads with compute. **Impact: Medium** | All shapes | Correctness risk: No
- \[ \] **Increase decode parallelism for small batches** — For T=1 grid=(1,32)=32 programs; only 24% SM utilization on H100. Consider parallelizing over kv_len (split-K style). **Impact: High for small-batch** | A1-A5 | Correctness risk: Yes (needs final reduction)

### A.3 Compute Optimizations

- \[ \] **`tl.dot` for inner products** — Replace `tl.sum(q_abs[None,:] * ckv, axis=1)` with `tl.dot(ckv, q_abs)` to use tensor cores. **Impact: Low** (memory-bound, not compute-bound) | Prefill large shapes | Correctness risk: No
- \[ \] **Static range unrolling for small SEQ_BLOCK** — If SEQ_BLOCK is small and kv_len/SEQ_BLOCK is known, unroll with `tl.static_range`. **Impact: Low-Medium** | Decode shapes | Correctness risk: No

### A.4 Kernel Fusion (Larger scope)

- \[ \] **Fuse weight absorption into kernel** — Absorb the `q_absorbed = q_nope @ W_kn^T` einsum into the kernel (load W_kn from global, compute on the fly). Saves one global memory roundtrip for q_absorbed. **Impact: Medium** | All shapes | Correctness risk: Yes (major rewrite)
- \[ \] **Fuse value projection into kernel** — After weighted_kv accumulation, multiply by W_v inside the kernel. Saves one global roundtrip. **Impact: Medium** | All shapes | Correctness risk: Yes (major rewrite)

______________________________________________________________________

## Final Best Configuration

*To be filled after Phase 3.*

______________________________________________________________________

## Appendix: How to Reproduce

```bash
# Environment
GPU: NVIDIA H100 80GB HBM3
torch: 2.10.0a0+b4e4ee81d3.nv25.12
triton: 3.5.1
dtype: bfloat16

cd tensorrt_llm/_torch/auto_deploy/custom_ops/mla/

# Correctness check
python sweep_triton_mla.py --correctness

# Baseline benchmark
python sweep_triton_mla.py

# Full parameter sweep
python sweep_triton_mla.py --sweep --output sweep_results.json

# Override params
python sweep_triton_mla.py --seq-block 32 --num-warps 4 --num-stages 3
```
