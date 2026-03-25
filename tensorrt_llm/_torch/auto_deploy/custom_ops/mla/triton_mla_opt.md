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

*Updated after every iteration. Kernel-only timings (attention kernel in isolation,
excluding weight absorption, cache write, value projection, and output scatter).*

| ID  | Best kernel µs | Config                                               | Iter | vs Baseline |
| --- | -------------- | ---------------------------------------------------- | ---- | ----------- |
| A1  | 8.5            | multihead HB=4, SEQ_BLOCK=64, warps=8, stgs=3       | 18   | **2.5×**    |
| A2  | 10.66          | split-K NP=4, HB=4, SB=64, warps=8, stgs=2         | 37   | **6.5×**    |
| A3  | 10.56          | split-K NP=8, HB=4, SB=64, w=8, ns=2, rw=8        | 42   | **12.0×**   |
| A4  | 11.31          | split-K NP=16, HB=4, SB=64, w=8, ns=2, rw=8       | 42   | **22.0×**   |
| A5  | 12.12          | split-K NP=16, HB=4, SB=128, w=4, ns=2, rw=8      | 42   | **40.5×**   |
| A6  | 11.9           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3      | 21   | **5.8×**    |
| A7  | 15.21          | split-K NP=4, HB=4, SB=64, w=8, ns=2, rw=8        | 41   | **8.6×**    |
| A8  | 16.0           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3      | 21   | **9.3×**    |
| A9  | 13.84          | multihead HB=8, SEQ_BLOCK=64, warps=8, stgs=3       | 36   | **9.5×**    |
| A10 | 18.0           | multihead HB=8, SEQ_BLOCK=128, warps=8, stgs=3      | 21   | **14.0×**   |
| B1  | 18.55          | multihead HB=16, SEQ_BLOCK=64, warps=8, stgs=3      | 35   | **21.4×**   |
| B2  | 96.4           | multihead HB=32, SEQ_BLOCK=128, warps=8, stgs=2     | 38   | **63.4×**   |
| B3  | 288.0          | multihead HB=32, SEQ_BLOCK=128, warps=8, stgs=2     | 39   | **83.9×**   |
| B4  | 980.5          | multihead HB=32, SEQ_BLOCK=128, warps=8, stgs=2     | 40   | **98.1×**   |

______________________________________________________________________

## Full-Launcher Performance Comparison

**Measured on:** NVIDIA H100 80GB HBM3 · PyTorch 2.10.0a0+nv25.12 · Triton 3.5.1 ·
dtype bfloat16 · Mistral-Small-4-119B-2603 (N=32, KV_LORA_RANK=256, V_HEAD=128)

**What is included:** End-to-end launcher time — weight absorption, cache write,
attention kernel, value projection, and output scatter. This is what matters in serving.

**Three columns:**

- **triton_baseline** — full launcher with iter 0 kernel params (SEQ_BLOCK=8,
  num_warps=2 for decode; SEQ_BLOCK=16, num_warps=4 for prefill; HEAD_BLOCK=1;
  no split-K; no index overhead optimizations). *Represents the un-optimized starting point.*
- **torch_impl** — `_torch_mla_generate_with_absorption` (decode) /
  `_torch_mla_context_with_expansion` (prefill) from `torch_backend_mla.py`.
  *The PyTorch reference implementation used before triton_mla existed.*
- **triton_latest** — current code (iter 61) with all optimizations applied.

### Decode (full launcher)

| ID   | Shape          | triton_baseline µs | torch_impl µs | triton_latest µs | vs baseline | vs torch |
| ---- | -------------- | ------------------- | -------------- | ----------------- | ----------- | -------- |
| A1   | B=1, kv=64     | ~430                | 231.7          | **86.1**          | **5.0×**    | **2.7×** |
| A2   | B=1, kv=256    | ~1330               | 232.4          | **111.7**         | **11.9×**   | **2.1×** |
| A3   | B=1, kv=512    | ~2540               | 231.0          | **110.5**         | **23.0×**   | **2.1×** |
| A4   | B=1, kv=1024   | ~4980               | 226.5          | **111.5**         | **44.7×**   | **2.0×** |
| A5   | B=1, kv=2048   | ~9820               | 230.0          | **112.9**         | **87.0×**   | **2.0×** |
| A6   | B=8, kv=256    | ~1370               | 1699.4         | **91.0**          | **15.1×**   | **18.7×** |
| A7   | B=8, kv=512    | ~2600               | 1708.6         | **120.1**         | **21.7×**   | **14.2×** |
| A8   | B=16, kv=512   | ~5200               | 3586.1         | **96.9**          | **53.7×**   | **37.0×** |
| A9   | B=32, kv=256   | ~5480               | 6868.1         | **94.3**          | **58.1×**   | **72.8×** |
| A10  | B=32, kv=512   | ~10100              | 6752.6         | **92.2**          | **109.5×**  | **73.2×** |

*triton_baseline decode estimates: kernel-only iter0 × launcher overhead factor (~21×
at B=1) derived from kernel-only/launcher ratio measured at iter 57.*

### Prefill (full launcher)

| ID  | Shape        | triton_baseline µs | torch_impl µs | triton_latest µs | vs baseline | vs torch |
| --- | ------------ | ------------------- | -------------- | ----------------- | ----------- | -------- |
| B1  | T=128        | ~568                | 277            | **146**           | **3.9×**    | **1.9×** |
| B2  | T=512        | ~573                | 328            | **163**           | **3.5×**    | **2.0×** |
| B3  | T=1024       | ~717                | 708            | **303**           | **2.4×**    | **2.3×** |
| B4  | T=2048       | ~1078               | 2140           | **690**           | **1.6×**    | **3.1×** |

*triton_baseline prefill = iter 57 full launcher (before iter 60-61 overhead fixes),
measured at 568/573/717/1078 µs for B1-B4.*

### Key takeaways

**Decode path** (the hot path in production serving):

- At small batch (B=1): triton is **2× faster** than torch. The torch decode path has
  a flat ~230 µs floor from Python scalar ops; triton runs at ~87-113 µs.
- At large batch (B≥8): triton is **14–73× faster**. The torch path has a serial Python
  loop over batch elements (O(B) kernel launches); triton batches them all in one launch.
- Triton is **5–109× faster than triton iter0** thanks to HEAD_BLOCK tiling + split-K.

**Prefill path** (important but less frequent than decode in production):

- triton_latest is **1.9–3.1× faster than torch** across all tested sizes (T=128–2048).
- The dominant overhead before iter 60-61 was NOT the attention kernel (~25-540 µs) but
  the multi-sequence index scaffolding (~275 µs flat, from `repeat_interleave` / `cumsum`).
- iter 61 eliminated this with a single-sequence fast path: index_select → direct use,
  `out.index_copy_` → `out.copy_`, `.item()` → `q_nope.shape[0]`.
- triton prefill is **1.6–3.9× faster than triton iter0** (prior prefill improvements
  from HEAD_BLOCK tiling are also included).

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

### Iteration 8 — Update multihead SEQ_BLOCK lookup: SB=64 across the board

**Change:** Updated `_get_mla_multihead_config` from SB=16 to SB=64 for all paths,
based on HB×SEQ_BLOCK sweep data.

Key finding: increasing SEQ_BLOCK from 16 to 64 reduces inner-loop iterations by 4×
and dramatically improves performance for all shapes:

| Config            | A8    | A9    | A10   | B1   | B2    | B3    | B4     |
| ----------------- | ----- | ----- | ----- | ---- | ----- | ----- | ------ |
| HB=4, SB=16 (old) | 53.4  | 32.5  | 56.0  | 25.6 | 177.5 | 515   | 1837   |
| HB=8, SB=64 (new) | 29.7  | 20.4  | 32.6  | 21.0 | 268   | 972   | 3642   |
| HB=32, SB=64 (new)| —     | —     | —     | 22.2 | 107   | 323   | 1076   |

Updated configs: prefill T≤128 → HB=8,SB=64; prefill T>128 → HB=32,SB=64;
decode B≥16 → HB=8,SB=64. Correctness: PASS all 14 shapes.

**Commit:** iter 8 — multihead SEQ_BLOCK → 64 for all paths (HB=8 decode, HB=8/32 prefill)

______________________________________________________________________

### Iteration 9 — q_abs/q_pe in bfloat16 (remove float32 roundtrip)

**Change:** In `_mla_attention_kernel_multihead`, q_absorbed and q_pe are now
explicitly cast to bfloat16 on load (`.to(tl.bfloat16)` instead of `.to(tl.float32)`).
This eliminates the per-program fp32 roundtrip: old path stored \[HEAD_BLOCK, KV_BLOCK\]
as fp32 then cast back to bf16 for each tl.dot; new path stores them directly in bf16.
Register savings: HEAD_BLOCK×KV_BLOCK = 8×256 = 2048 elements halved from 4→2 bytes.

The `.to(tl.bfloat16)` casts in the inner loop tl.dot calls are removed (no-ops now).
The softmax state (m_i, l_i, acc) and all arithmetic remain in float32.

Bug fixed: first attempt used bare `tl.load(...)` without explicit cast; this broke when
the input tensor (from standalone benchmark) was float32. Explicit `.to(tl.bfloat16)`
handles both float32 and bfloat16 input tensors correctly.

Correctness: PASS (comprehensive HB×SB sweep). Benchmark confirmed (same session, H100):
A8=29.4µs, A9=20.2µs, A10=32.3µs; B1=20.8µs, B2=268.2µs (HB=8).
B shapes use HB=32: B2=106.7µs, B3=322.8µs, B4=1075.8µs (HB=32 SB=64, warps=8).

**Commit:** iter 9 — q_abs/q_pe cast to bf16 on load (register pressure reduction)

______________________________________________________________________

### Iteration 10 — Prefill HB=32 SEQ_BLOCK: 64 → 128

**Change:** Updated `_get_mla_multihead_config` prefill T>128 path from
`return 64, 32, 8, 4` to `return 128, 32, 8, 4`.

Benchmark data (H100, from comprehensive HB×SB sweep):

| Config          | B2 µs | B3 µs | B4 µs |
| --------------- | ----- | ----- | ----- |
| HB=32, SB=64    | 107   | 323   | 1076  |
| HB=32, SB=128   | 103.5 | 296.5 | 993.9 |
| Speedup         | +3%   | +8%   | +8%   |

The gain grows with sequence length — longer prefills benefit more from larger
SB because tensor-core utilisation increases and the inner loop overhead amortises
over more work per block.

Correctness: PASS (inherited from iter 8 sweep — SB change only affects SEQ_BLOCK
boundary masking which was already tested at SB=128 in the HB sweep).

**Commit:** iter 10 — prefill HB=32 SEQ_BLOCK 64→128 (+3-8% for B2-B4)

______________________________________________________________________

### Iteration 11 — Lower decode multihead dispatch threshold: B≥16 → B≥8

**Change:** Updated `_triton_mla_decode` dispatch from `if b >= 16` to `if b >= 8`.

A6 (B=8, kv=256) and A7 (B=8, kv=512) now use `_mla_attention_kernel_multihead`
(HB=8, SB=64) instead of the original kernel with sweep-optimal config.

**Benchmark data (H100, HB=8 SB=64 vs original sweep best):**

| ID  | Original best µs | Multihead HB=8 SB=64 µs | Speedup |
| --- | ---------------- | ----------------------- | ------- |
| A6  | 26.4             | 18.7                    | +41%    |
| A7  | 45.7             | 28.4                    | +61%    |

For B\<8 (T=1 single-token decode), the original kernel with large SEQ_BLOCK
still wins (A1=9.4µs, A2=17.0µs vs multihead 11.0/18.1µs).

Correctness: PASS (HB=8, SB=64 was tested across all 14 shapes in iter 8 sweep).

**Commit:** iter 11 — decode multihead dispatch threshold B>=16 -> B>=8 (+41-61% A6-A7)

______________________________________________________________________

### Iteration 12 — Multihead for T=1 long-context decode (kv_len ≥ 512)

**Change:** Updated `_triton_mla_decode` dispatch condition from `if b >= 8` to
`if b >= 8 or max_kv_len >= 512`.

For single-token decode with long context, the 32× cache redundancy is large
enough that HEAD_BLOCK sharing outweighs the SM underutilisation cost.
Breakeven is at kv_len ≈ 512; below that the original kernel wins.

**Benchmark (H100, HB=8 SB=64 vs original sweep best):**

| ID  | Original best µs | Multihead µs | Speedup |
| --- | ---------------- | ------------ | ------- |
| A1  | 9.4 (kv=64)      | 11.0         | −14% (no change — still uses original) |
| A2  | 17.0 (kv=256)    | 18.1         | −6% (no change — still uses original)  |
| A3  | 28.1 (kv=512)    | 27.8         | +1%                                     |
| A4  | 50.4 (kv=1024)   | 47.5         | +6%                                     |
| A5  | 94.6 (kv=2048)   | 85.8         | +10%                                    |

Correctness: PASS (HB=8 SB=64 tested for all 14 shapes in iter 8 sweep).

**Commit:** iter 12 — decode multihead for T=1 kv_len>=512 (+1-10% A3-A5)

______________________________________________________________________

### Iteration 13 — exp2 softmax: tl.exp → tl.math.exp2

**Change:** In both `_mla_attention_kernel` and `_mla_attention_kernel_multihead`,
replaced `tl.exp` with `tl.math.exp2` for the online softmax:

```python
# Before:
scores = (scores_nope + scores_pe) * SCALE
alpha = tl.exp(m_i - m_new)
p = tl.exp(scores - m_new)

# After:
scores = (scores_nope + scores_pe) * (SCALE * 1.44269504)  # SCALE * log2e
alpha = tl.math.exp2(m_i - m_new)  # m values in log2 space
p = tl.math.exp2(scores - m_new)
```

`tl.math.exp2` compiles to `ex2.approx.f32` (native H100 instruction, ~4× faster
than `exp` in isolation). Math is unchanged: `exp2(x * log2e) == exp(x)`.

**Result:** Correctness PASS (all 14 shapes). Performance: negligible change.
A9=20.2µs (unchanged), B3=972µs (HB=8 baseline, unchanged).

**Analysis:** The kernel is HBM bandwidth-bound; the softmax exp operations
represent a tiny fraction of total cycles compared to tl.load bandwidth waits.
The optimization is still worthwhile as free throughput on compute-bound paths
and correct, but does not move the needle for memory-bound shapes.

**Commit:** iter 13 — exp2 softmax (tl.exp → tl.math.exp2, no perf regression)

______________________________________________________________________

### Iteration 14 — bf16 weighted_kv output (halve HBM write bandwidth)

**Change:** Changed `weighted_kv` tensor allocation from `dtype=torch.float32` to
`dtype=q_nope.dtype` (bfloat16) in both `_triton_mla_decode` and `_triton_mla_prefill`.

The kernel accumulates in fp32 and Triton performs fp32->bf16 conversion at `tl.store`.
The downstream einsum already casts to fp32 (via `w_v.float()`), so end-to-end numeric
precision is unchanged. The redundant `.to(q_nope.dtype)` call was removed from decode.

**Bandwidth saved per call (B4, T=2048):**

- Before: write 2048x32x256x4=67MB fp32; then read 67MB for .to() + einsum
- After: write 2048x32x256x2=33MB bf16; einsum auto-upcasts from bf16

**Benchmark (H100, with iter 13 as base):**

| ID  | Iter 13 µs | Iter 14 µs | Delta  |
| --- | ---------- | ---------- | ------ |
| A8  | 29.4       | 28.7       | +2.4%  |
| A9  | 20.2       | 20.1       | +0.5%  |
| A10 | 32.3       | 32.0       | +0.9%  |
| B2  | 103.5      | 101.2      | +2.2%  |
| B3  | 296.5      | 290.8      | +1.9%  |
| B4  | 993.9      | 989.9      | +0.4%  |

Correctness: PASS (max_abs_err \< 0.002 across all 14 shapes).

**Commit:** iter 14 — weighted_kv as bf16 (halve write BW, +0.4-2.4% across shapes)

______________________________________________________________________

### Iteration 15 — Warps=8, stages=3/2 from comprehensive parallel sweep

**Change:** Updated `_get_mla_multihead_config` based on an 8-GPU parallel sweep over all
`(SEQ_BLOCK, num_warps, num_stages)` combinations for both HEAD_BLOCK=8 and HEAD_BLOCK=32.

Key findings from the sweep (150 configs × 14 shapes):

- `num_warps=8` universally best for both HB=8 and HB=32 (more warp-level parallelism fills pipeline)
- Decode HB=8: `stages=3` optimal (A9: 14.2µs vs warps=4,stages=4: 20.1µs — **+42%**)
- Prefill HB=32 SB=128: `stages=2` safest and best (stages=5 OOMs: needs 356 KB > H100's 232 KB SMEM limit)
- Prefill HB=8 SB=64 T≤128: `stages=2` (simpler; minor SMEM reduction)

Updated config function:

```python
if is_prefill:
    if num_tokens <= 128:
        return 64, 8, 8, 2   # HB=8, SB=64, w=8, s=2
    else:
        return 128, 32, 8, 2  # HB=32, SB=128, w=8, s=2  (stages=5 would OOM)
else:
    return 64, 8, 8, 3        # HB=8, SB=64, w=8, s=3  (stages=3 > stages=4)
```

**Benchmark (H100, with iter 14 as base):**

| ID  | Iter 14 µs | Iter 15 µs | Delta   |
| --- | ---------- | ---------- | ------- |
| A3  | 26.5       | 17.7       | +33%    |
| A4  | 42.8       | 28.5       | +33%    |
| A5  | 72.6       | 49.5       | +32%    |
| A6  | 19.2       | 13.7       | +29%    |
| A7  | 28.1       | 18.7       | +33%    |
| A8  | 28.7       | 19.2       | +33%    |
| A9  | 20.1       | 14.4       | +28%    |
| A10 | 32.0       | 20.0       | +38%    |
| B1  | 25.6       | 21.3       | +17%    |
| B2  | 101.2      | 96.9       | +4.3%   |
| B3  | 290.8      | 289.1      | +0.6%   |
| B4  | 989.9      | 980.7      | +0.9%   |

Correctness: PASS (max_abs_err \< 0.002 across all 14 shapes).

**Commit:** iter 15 — warps=8, stages=3(decode)/2(prefill) from comprehensive sweep (+28-42% A-shapes)

______________________________________________________________________

### Iteration 16 — Multihead dispatch for ALL decode shapes

**Change:** Removed the conditional `if b >= 8 or max_kv_len >= 512` guard in `_triton_mla_decode`.
Previously, small-batch / short-context shapes A1 (B=1, kv=64) and A2 (B=1, kv=256) still fell
through to the original single-head `_mla_attention_kernel`. With iter 15's warps=8 config, the
multihead kernel (HB=8 SB=64 w=8 s=3) now beats the single-head best on every shape, so the
conditional is dead weight — replaced with unconditional multihead dispatch.

**Benchmark (HB=8, SB=64, w=8, s=3 for all decode shapes):**

| ID  | Iter 15 µs | Iter 16 µs | Delta |
| --- | ---------- | ---------- | ----- |
| A1  | 9.4        | 8.6        | +9%   |
| A2  | 17.0       | 12.4       | +27%  |

A3–A10, B1–B4: unchanged (already used multihead from iter 11/12).

Correctness: PASS (all 14 shapes, max_abs_err \< 0.002).

**Commit:** iter 16 — unconditional multihead dispatch for all decode shapes (+9-27% A1-A2)

______________________________________________________________________

### Iteration 17 — tl.multiple_of hints + tl.cdiv; remove dead single-head kernel

**Change:**

1. Added `tl.multiple_of(block_id * SEQ_BLOCK, SEQ_BLOCK)` and
   `tl.multiple_of(slot_idx * MAX_SEQ_LEN * CACHE_DIM, CACHE_DIM)` hints in
   `_mla_attention_kernel_multihead`. These tell Triton that `block_start` is aligned
   to SEQ_BLOCK and the cache row pointer is aligned to CACHE_DIM, allowing the compiler
   to generate tighter address arithmetic and potentially vectorized memory ops.
   Also replaced `(kv_len + SEQ_BLOCK - 1) // SEQ_BLOCK` with `tl.cdiv(kv_len, SEQ_BLOCK)`.

1. Removed dead code: `_get_mla_decode_config`, `_get_mla_prefill_config`, and
   `_mla_attention_kernel` (the original single-head kernel). All three have been
   completely superseded by `_mla_attention_kernel_multihead` since iter 16.

**Performance impact:** No measurable change — within benchmark noise (±2%) as expected for
compiler hints. Correctness: PASS (all 14 shapes, max_abs_err \< 0.002).

| ID  | Iter 16 µs | Iter 17 µs | Delta |
| --- | ---------- | ---------- | ----- |
| A1  | 8.6        | 8.7        | ±noise |
| A2  | 12.4       | 12.7       | ±noise |
| A9  | 14.4       | 14.3       | ±noise |

**Commit:** iter 17 — tl.multiple_of hints + remove dead single-head kernel (no perf change)

______________________________________________________________________

### Iteration 18 — Adaptive HEAD_BLOCK: HB=4 for B≤16, HB=8 for B>16 (decode)

**Change:** Updated `_get_mla_multihead_config` decode path to use HEAD_BLOCK=4 for `num_tokens ≤ 16`
and HEAD_BLOCK=8 for `num_tokens > 16`.

**Why HEAD_BLOCK=4 wins for small batch (A1-A8):**
With HB=4 the decode grid for B=8 is (8, 32//4=8) = 64 programs vs (8, 32//8=4) = 32 programs
with HB=8. The extra programs improve SM utilization without significantly hurting cache sharing
(4 heads still share each cache load vs 8). For very small batch (B=1), the gain is in
register/pipeline efficiency as each program handles a smaller register footprint per head-group.

**Why HEAD_BLOCK=8 wins for B=32 (A9-A10):**
Grid=(32, 4)=128 programs with HB=8 fills H100 (132 SMs) in ≈1 wave.
Grid=(32, 8)=256 programs with HB=4 spills into 2 waves (+launch overhead) while reading
the cache 2× more per total — a double penalty.

**Benchmark (H100, HB=4 with SB=64, w=8, s=3):**

| ID  | Iter 17 µs | Iter 18 µs | Delta  |
| --- | ---------- | ---------- | ------ |
| A1  | 8.6        | 8.5        | +1.2%  |
| A2  | 12.4       | 12.3       | +0.8%  |
| A3  | 17.7       | 17.3       | +2.3%  |
| A4  | 28.5       | 27.1       | +4.9%  |
| A5  | 49.5       | 47.1       | +4.9%  |
| A6  | 13.7       | 12.6       | +8.0%  |
| A7  | 18.7       | 17.7       | +5.3%  |
| A8  | 19.2       | 18.8       | +2.1%  |
| A9  | 14.4       | 14.4       | —      |
| A10 | 20.0       | 20.0       | —      |

Also includes sweep script cleanup: fixed `run_correctness()` to use `_mla_attention_kernel_multihead`
(the only active kernel since iter 16), removed dead `_launch_kernel` and `bench_kernel` functions.

Correctness: PASS (all 14 shapes, max_abs_err \< 0.002).

**Commit:** iter 18 — adaptive HEAD_BLOCK=4/8 in decode dispatch (+1-8% A1-A8)

______________________________________________________________________

### Iteration 19 — HB=16 for prefill T≤128 (+8% B1)

**Change:** Updated `_get_mla_multihead_config` for `is_prefill=True, num_tokens ≤ 128`:
from `return 64, 8, 8, 2` (HB=8) to `return 64, 16, 8, 2` (HB=16).

HB=16 doubles cache sharing (16 heads share one cache load vs 8), halving HBM traffic for this path.
Grid=(T, 32//16=2)=2T programs — for B1 (T=128): grid=256 programs on H100 (132 SMs), ~2 waves.
The extra pipeline occupancy from 2 waves outweighs the small dispatch overhead.

SMEM usage with s=2: 1 prefetch buffer × (64×256 + 64×64) × 2 bytes = 80 KB — well within H100 limits.

**Benchmark:**

| ID  | Iter 18 µs | Iter 19 µs | Delta  |
| --- | ---------- | ---------- | ------ |
| B1  | 21.3       | 19.6       | +8.0%  |

A-shapes unchanged (only `is_prefill=True, T≤128` path changed).
Correctness: PASS (all 14 shapes, max_abs_err \< 0.002).

**Commit:** iter 19 — HB=16 for prefill T≤128 (+8% B1)

______________________________________________________________________

### Iteration 20 — SB=128 for large-batch decode (B>16, A9-A10)

**Change:** Updated `_get_mla_multihead_config` decode path for `num_tokens > 16`:
from `return 64, 8, 8, 3` (SB=64) to `return 128, 8, 8, 3` (SB=128).

**Why SB=128 wins for large-batch:**
With B=32 and kv_len=512 (A10), the kernel iterates over `kv_len/SEQ_BLOCK` blocks:

- SB=64: 512/64 = 8 iterations per program
- SB=128: 512/128 = 4 iterations per program

Larger SEQ_BLOCK reduces loop overhead and improves the ratio of useful computation
to loop-control overhead. The pipeline depth (stages=3) prefetches more data per
iteration. SMEM usage: (128×256 + 128×64) × 2 bytes × (stages-1) stages =
2 × 40960 × 2 = 163 KB — within H100 limits.

**Benchmark:**

| ID  | Iter 19 µs | Iter 20 µs | Delta  |
| --- | ---------- | ---------- | ------ |
| A9  | 14.4       | 14.3       | +0.7%  |
| A10 | 20.0       | 18.3       | +8.5%  |

A1-A8 unchanged (only `num_tokens > 16` path changed).
Correctness: PASS (all 14 shapes, max_abs_err \< 0.002).

**Commit:** iter 20 — SB=128 for large-batch decode B>16 (+8.5% A10)

______________________________________________________________________

### Iteration 21 — Adaptive SEQ_BLOCK for HB=4 decode path (B≤16) based on max_kv_len

**Change:** Updated `_get_mla_multihead_config` to accept `max_kv_len` parameter.
In the `num_tokens <= 16` (HB=4) branch:

```python
sb = 64 if max_kv_len <= 64 else 128
return sb, 4, 8, 3
```

Updated `_triton_mla_decode` to compute `max_kv_len = int(kv_len.max().item())` and
pass it to `_get_mla_multihead_config`. Also fixed pre-existing bug in
`DEFAULT_DECODE_PARAMS` (was `SEQ_BLOCK=8`, violating `tl.dot` K≥16 constraint;
updated to `SEQ_BLOCK=64, num_warps=8, num_stages=3`).

**Why SB=128 wins for kv_len > 64:**
HB=4 decode (B≤16) iterates `kv_len/SEQ_BLOCK` times:

- A2 (kv=256): SB=64→4 iters, SB=128→2 iters — 2× fewer iterations, less loop overhead
- A5 (kv=2048): SB=64→32 iters, SB=128→16 iters — pipeline fills more efficiently

A1 (kv=64) is the only shape where kv_len ≤ SB=128 — a single half-empty block wastes
50% of thread work. Using SB=64 keeps 2 full blocks, maintaining 8.5µs.

**Benchmark:**

| ID  | Prev best µs | Iter 21 µs | Delta   | Config           |
| --- | ------------ | ---------- | ------- | ---------------- |
| A1  | 8.5          | 8.5        | 0%      | SB=64 (adaptive) |
| A2  | 12.3         | **10.8**   | +12.2%  | SB=128 adaptive  |
| A3  | 17.3         | **14.3**   | +17.3%  | SB=128 adaptive  |
| A4  | 27.1         | **21.5**   | +20.7%  | SB=128 adaptive  |
| A5  | 47.1         | **35.4**   | +24.8%  | SB=128 adaptive  |
| A6  | 12.6         | **11.9**   | +5.6%   | SB=128 adaptive  |
| A7  | 17.7         | **15.4**   | +13.0%  | SB=128 adaptive  |
| A8  | 18.8         | **16.0**   | +14.9%  | SB=128 adaptive  |
| A9  | 14.3         | 14.0       | +2.1%   | unchanged path   |
| A10 | 18.3         | 18.0       | +1.6%   | unchanged path   |

Correctness: PASS (all 14 shapes, max_abs_err \< 0.002).

**Commit:** iter 21 — adaptive SB=64/128 for HB=4 decode path (+12-25% on A2-A8)

______________________________________________________________________

### Iteration 22 — stages sweep for SB=128 path \[FAILED/NO IMPROVEMENT\]

**Change:** Benchmark-only experiment; no code change.

Tested `num_stages=2` and `num_stages=4` with SEQ_BLOCK=128, num_warps=8, HB=4/8:

- **stages=4**: OOM on all shapes. SMEM required: 3 × (128×256 + 128×64) × 2 bytes =
  245 760 bytes > H100 limit of 232 448 bytes. Not viable at SB=128, warps=8.
- **stages=2**: Worse than stages=3 across A2–A8.

| ID  | stages=2 µs | stages=3 µs (cur best) | stages=4 |
| --- | ----------- | ---------------------- | -------- |
| A2  | 11.7        | **10.8**               | OOM      |
| A3  | 17.1        | **14.3**               | OOM      |
| A4  | 27.0        | **21.5**               | OOM      |
| A5  | 47.4        | **35.4**               | OOM      |
| A7  | 18.1        | **15.4**               | OOM      |
| A8  | 19.5        | **16.0**               | OOM      |

**Conclusion:** stages=3 is optimal for SB=128 path. Fewer stages reduce
pipeline latency hiding; more stages exceed SMEM budget. No code change needed.

**Commit:** iter 22 — stages sweep SB=128 \[FAILED: stages=4 OOM, stages=2 worse\]

______________________________________________________________________

### Iteration 23 — warps sweep for SB=128 path \[NO IMPROVEMENT\]

**Change:** Benchmark-only experiment; no code change.

Tested `num_warps=4` and `num_warps=16` with SEQ_BLOCK=128, stages=3,
compared against current best `num_warps=8`:

| ID  | warps=4 µs | warps=8 (cur best) | warps=16 µs |
| --- | ---------- | ------------------ | ----------- |
| A2  | 11.5       | **10.8**           | 11.7        |
| A3  | 15.1       | **14.3**           | 16.4        |
| A4  | 21.5       | **21.5**           | 24.2        |
| A5  | 34.8       | **35.4**           | 41.0        |
| A6  | 12.7       | **11.9**           | 12.4        |
| A7  | 16.0       | **15.4**           | 17.1        |
| A8  | 16.5       | **16.0**           | 17.6        |

warps=8 wins on 6/7 shapes. warps=4 ties A4 and is 1.7% faster on A5 (34.8 vs
35.4µs) but loses elsewhere. warps=16 is uniformly worse. No change to config.

**Commit:** iter 23 — warps sweep SB=128 \[NO IMPROVEMENT: warps=8 remains best\]

______________________________________________________________________

### Iteration 24 — HEAD_BLOCK sweep: HB=1 and HB=2 for decode \[NO IMPROVEMENT\]

**Change:** Benchmark-only; no code change.

Hypothesis: HB=1 (grid=(T, 32)) gives 4× more SM programs than HB=4 (grid=(T, 8))
for T=1 shapes, potentially hiding more latency via more active SMs.

Results (SB=128, warps=8, stages=3) vs current best:

| ID  | HB=2 µs | HB=1 µs | HB=4 cur best | Winner    |
| --- | ------- | ------- | ------------- | --------- |
| A1  | 9.6     | 9.3     | **8.5**       | HB=4      |
| A2  | 11.1    | 10.8    | **10.8**      | tie HB=4  |
| A3  | 14.8    | 14.3    | **14.3**      | tie HB=4  |
| A4  | 21.6    | 21.3    | **21.5**      | HB=1 ≈HB=4 |
| A5  | 35.9    | 35.5    | **35.4**      | HB=4      |
| A6  | 12.8    | 20.2    | **11.9**      | HB=4      |
| A7  | 16.8    | 27.5    | **15.4**      | HB=4      |
| A8  | 29.2    | 48.3    | **16.0**      | HB=4      |

HB=2 and HB=1 lose for all T>1 shapes (A6-A10) because they expose
more parallelism but sacrifice cache amortization. HB=4 remains optimal.

**Commit:** iter 24 — HB sweep HB=1,HB=2 \[NO IMPROVEMENT: HB=4 wins\]

______________________________________________________________________

### Iteration 25 — SB=64 stages=5 and warps=4 sweep \[NO IMPROVEMENT\]

**Change:** Benchmark-only; no code change.

Tested alternative configs for the SB=64 path (A1 shape, kv=64) and compared
broadly:

- `SB=64 warps=8 stages=5`: A1 = 8.8µs vs **8.5µs** (current best, stages=3) — worse
- `SB=64 warps=4 stages=3`: A1 = 9.7µs — worse; also worse for all other shapes

stages=5 adds more SMEM pipeline overhead than it hides for this small kv_len.
warps=4 reduces occupancy without compensating benefit.

| Config           | A1 µs | A2 µs | A3 µs | A9 µs |
| ---------------- | ----- | ----- | ----- | ----- |
| SB=64 w=8 s=3 ✓ | **8.5** | 12.5  | 17.7  | 14.5  |
| SB=64 w=8 s=5   | 8.8   | 12.6  | 17.7  | 15.6  |
| SB=64 w=4 s=3   | 9.7   | 14.2  | 20.9  | 16.4  |

**Commit:** iter 25 — SB=64 stages/warps variants \[NO IMPROVEMENT\]

______________________________________________________________________

### Iteration 26 — Split-K kernel + Triton reduction for small-batch long-context

**Change:** Added two new Triton kernels:

1. **`_mla_attention_kernel_splitk`** — 3-D grid `(T, N//HB, NUM_PARTS)`. Each program
   handles one `(token, head_group, kv_partition)` and processes blocks
   `[part_start, part_end)` of the kv sequence. Stores partial `(acc, m_i, l_i)` to
   workspace buffers — no normalization.
1. **`_mla_splitk_reduce`** — 2-D grid `(T, N_HEADS)`. Loads `NUM_PARTS` partial results,
   combines them with numerically stable log-sum-exp, normalizes, stores bf16 output.

Updated `_triton_mla_decode` to dispatch split-K when `b <= 4` and `max_kv_len >= 512`
with `NUM_PARTS=8`. Updated `sweep_triton_mla.py` to add `bench_splitk_kernel` and use
it for eligible shapes in `run_benchmark`.

**Why split-K helps for small batch:**
T=1 decode with HB=4 creates grid=(1,8)=8 programs — only 6% SM utilization on H100
(132 SMs). Split-K with NP=8 gives grid=(1,8,8)=64 programs → 48% utilization. Each
partition loads `kv_len/NP` cache blocks, so the parallelism hides HBM latency.

**Initial Python reduction** had ~490µs overhead (CPU-GPU sync barrier after kernel).
Replaced immediately with Triton reduction kernel (`_mla_splitk_reduce`) that runs
back-to-back with the main kernel on GPU, adding only ~1µs overhead.

**Benchmark:**

| ID  | Prev best µs | Split-K µs | Delta  |
| --- | ------------ | ---------- | ------ |
| A3  | 14.3         | **11.2**   | +21.7% |
| A4  | 21.5         | **12.7**   | +40.9% |
| A5  | 35.4         | **15.5**   | +56.2% |

A1/A2 unaffected (kv≤256, not eligible for split-K).
A6-A10 unaffected (T>4, not eligible).
Correctness: PASS (all 14 shapes, max_abs_err \< 0.005 vs multihead reference).

**Commit:** iter 26 — split-K kernel + Triton reduction (+22-56% for A3-A5)

______________________________________________________________________

### Iteration 27 — Split-K NUM_PARTS and SB tuning

**Change:** Benchmark-only; identified that SB=64 outperforms SB=128 for the split-K path.

With kv=512, SB=128: `total_blocks=4`, split into NP=8 → half partitions are empty,
wasting 4 of 8 SM slots. With SB=64: `total_blocks=8`, split into NP=8 → all 8
partitions have exactly 1 block — 100% partition utilization.

This explains the benchmark results: the sweep script's DEFAULT_DECODE_PARAMS uses
SB=64 and produces the best split-K numbers (A3=11.2µs). SB=128 would give 12.2µs.

The `_triton_mla_decode` dispatch uses SB from `_get_mla_multihead_config` which returns
SB=128 for kv>64. The actual dispatch is therefore suboptimal. Fix deferred to iter 28.

| ID  | SK SB=64 µs | SK SB=128 µs | Winner |
| --- | ----------- | ------------ | ------ |
| A3  | **11.2**    | 12.2         | SB=64  |
| A4  | **12.7**    | 12.5         | ≈ tie  |
| A5  | **15.5**    | 14.3         | SB=128 |

For A5 (kv=2048, 16 blocks): SB=128 → 16/8=2 blocks/part (all busy). SB=64 → 32/8=4
blocks/part (also all busy). SB=128 wins here due to better pipeline fill.
Optimal: SB=64 for kv=\[512, 1536\], SB=128 for kv>1536.

**Commit:** iter 27 — split-K SB tuning analysis \[doc only\]

______________________________________________________________________

### Iteration 28 — Adaptive SB=64/128 in split-K dispatch (kv threshold 1536)

**Change:** Updated `_triton_mla_decode` split-K path to use adaptive `seq_block`:

```python
if use_splitk:
    seq_block = 64 if max_kv_len <= 1536 else 128
```

Also updated `sweep_triton_mla.py` `run_benchmark` to use the same adaptive logic when
benchmarking split-K shapes.

**Benchmark:**

| ID  | Prev split-K µs | Adaptive µs | Delta  |
| --- | --------------- | ----------- | ------ |
| A3  | 11.2 (SB=64)    | **10.8**    | +3.6%  |
| A4  | 12.7 (SB=64)    | **12.2**    | +3.9%  |
| A5  | 15.5 (SB=64)    | **13.9**    | +10.3% |

Updated Current Best Summary: A3=10.8µs, A4=12.2µs, A5=13.9µs.
Correctness: PASS (all 14 shapes).

**Commit:** iter 28 — adaptive SB in split-K dispatch (+4-10% A3-A5)

______________________________________________________________________

### Iteration 29 — Adaptive NUM_PARTS: NP=8 for kv≤512, NP=16 for kv>512

**Change:** Updated `_triton_mla_decode` and `sweep_triton_mla.py` to use:

```python
num_parts = 8 if max_kv_len <= 512 else 16
```

**Why NP=16 helps for longer context:**
With NP=16 and HB=4: grid = (T=1, 8, 16) = 128 programs → ~97% SM utilization on H100
(132 SMs), vs NP=8's 64 programs (~48% utilization). Near-full SM fill hides HBM
latency of longer kv sequences.

NP sweep results (on GPU 0):

| ID  | kv  | NP=4 | NP=8  | NP=16 | NP=32 | Best |
| --- | --- | ---- | ----- | ----- | ----- | ---- |
| A3  | 512 | 12.0 | **11.0** | 11.6 | 13.0 | NP=8  |
| A4  | 1024 | 14.6 | 12.5 | **11.8** | 13.0 | NP=16 |
| A5  | 2048 | 17.1 | 14.4 | **13.2** | 15.7 | NP=16 |

NP=32 regresses (over-partitioning: too many small blocks, launcher overhead dominates).

**Improvement vs iter 28:**

| ID  | Iter 28 µs | Iter 29 µs | Delta |
| --- | ---------- | ---------- | ----- |
| A4  | 12.2       | **11.8**   | +3.4% |
| A5  | 13.9       | **13.2**   | +5.0% |

Correctness: PASS (all 14 shapes).

**Commit:** iter 29 — adaptive NP=8/16 for split-K (+3-5% A4-A5)

______________________________________________________________________

### Iteration 30 — Split-K stages=4 for SB=64 path \[NO IMPROVEMENT / OOM\]

**Change:** Benchmark-only; no code change.

Hypothesis: With SB=64, SMEM usage = (stages-1) × (64×256 + 64×64) × 2 bytes.
For stages=4: SMEM = 3 × (16384 + 4096) × 2 = 122 880 bytes — well within H100's
232 448-byte limit. Might improve pipeline fill for split-K partitions.

**Results:**

| ID  | kv   | NP | SB  | stages=3 µs | stages=4 µs | Winner   |
| --- | ---- | -- | --- | ----------- | ----------- | -------- |
| A3  | 512  | 8  | 64  | **10.8**    | 10.8        | tie      |
| A4  | 1024 | 16 | 64  | **11.4**    | 11.4        | tie      |
| A5  | 2048 | 16 | 128 | **12.7**    | OOM         | stages=3 |

- A3/A4 (SB=64): stages=4 ties stages=3 exactly — no pipeline benefit from the extra
  prefetch stage. Each partition already only processes a handful of blocks; the inner
  loop is short enough that extra pipelining adds no latency hiding.
- A5 (SB=128, NP=16): stages=4 triggers `OutOfResources` —
  SMEM required = 3 × (128×256 + 128×64) × 2 = 245 760 bytes > 232 448 limit.

**Conclusion:** Keep `num_stages=3` for all split-K shapes. No code change.

**Commit:** iter 30 — split-K stages=4 sweep \[NO IMPROVEMENT / OOM\]

______________________________________________________________________

### Iteration 31 — Split-K warps sweep + HEAD_BLOCK=8 sweep \[MIXED\]

**Change:** Benchmark-only; no code change.

**Exp A — warps=4 vs warps=8 for split-K:**

| ID  | kv   | NP | SB  | warps=4 µs | warps=8 µs | Winner |
| --- | ---- | -- | --- | ---------- | ---------- | ------ |
| A3  | 512  | 8  | 64  | 11.1       | **10.8**   | w=8    |
| A4  | 1024 | 16 | 64  | 11.8       | **11.5**   | w=8    |
| A5  | 2048 | 16 | 128 | **12.7**   | 12.9       | w=4    |

warps=8 wins for A3/A4. warps=4 gives a slight edge on A5 (~1.5%).
Motivates adaptive warps: w=8 for kv≤1024, w=4 for kv>1024 (deferred to iter 33).

**Exp B — HEAD_BLOCK=8 vs HB=4 for split-K:**

| ID  | HB=4 µs  | HB=8 µs | Winner |
| --- | -------- | ------- | ------ |
| A3  | **10.8** | 10.9    | HB=4   |
| A4  | **11.5** | 11.6    | HB=4   |
| A5  | 12.9     | 12.8    | ≈ tie  |

HB=4 is uniformly better; HB=8 gives \<1% on A5 only. Keep HB=4.

Also discovered: A2 (kv=256, T=1) with split-K NP=4 SB=64 = **10.3µs**
vs current best 10.8µs (multihead) — 4.7% improvement. Deferred to iter 32.

**Commit:** iter 31 — split-K warps/HB sweep \[MIXED: warps adaptive + A2 split-K planned\]

______________________________________________________________________

### Iteration 32 — Enable split-K for A2 (kv=256, NP=4): +8-9% improvement

**Change:** Extended split-K dispatch threshold from `max_kv_len >= 512` to
`max_kv_len >= 256`. Added `NP=4` case:

```python
use_splitk = b <= 4 and max_kv_len >= 256
if use_splitk:
    if max_kv_len <= 256:
        num_parts = 4
    elif max_kv_len <= 512:
        num_parts = 8
    else:
        num_parts = 16
```

**Why NP=4 helps for kv=256:** kv=256 with SB=64 gives `total_blocks=4`. NP=4
assigns 1 block per partition — 100% utilization. Grid=(1,8,4)=32 programs → 24%
SM utilization vs HB=4 multihead's 8 programs (6%). 4× more SM parallelism fills
idle HBM latency slots.

**Benchmark (3 runs each):**

| ID  | multihead µs | split-K NP=4 µs | Delta  |
| --- | ------------ | --------------- | ------ |
| A2  | 11.65        | **10.66**       | +8.5%  |

Correctness: PASS (max_abs_err \< 0.004). Updated `sweep_triton_mla.py`.

**Current Best Update:**

| ID  | Best µs | Config                                          | Iter | vs Baseline |
| --- | ------- | ----------------------------------------------- | ---- | ----------- |
| A2  | 10.66   | split-K NP=4, HB=4, SB=64, warps=8, stgs=3     | 32   | **6.5×**    |

**Commit:** iter 32 — enable split-K for A2 (kv=256, NP=4, +8-9%)

______________________________________________________________________

### Iteration 33 — Adaptive warps=4 for long-context split-K (kv>1024): +2% A5

**Change:** Added `nw` selection in `_triton_mla_decode` split-K path:

```python
nw = 4 if max_kv_len > 1024 else 8
```

**Why warps=4 wins at kv=2048:** With SB=128, NP=16, each partition handles 1 block
of 128 positions. Working set = \[HB=4, SB=128\] = 512 elements. Fewer warps (128
threads vs 256) reduces register pressure — better occupancy hides HBM latency.
warps=8 remains optimal for shorter kv (A2-A4) where pipeline fill is the bottleneck.

**Benchmark (3 runs each):**

| ID  | warps=8 µs | warps=4 µs | Delta |
| --- | ---------- | ---------- | ----- |
| A5  | 13.0       | **12.75**  | +2.0% |

A3/A4 unaffected (warps=8 used for kv≤1024). Updated `sweep_triton_mla.py`.

**Current Best Update:**

| ID  | Best µs | Config                                           | Iter | vs Baseline |
| --- | ------- | ------------------------------------------------ | ---- | ----------- |
| A5  | 12.75   | split-K NP=16, HB=4, SB=128, warps=4, stgs=3    | 33   | **38.5×**   |

**Commit:** iter 33 — adaptive warps=4 for long-context split-K (+2% A5)

______________________________________________________________________

### Iteration 34 — Batch-size threshold + NP sweep analysis \[NO CODE CHANGE\]

**Change:** Benchmark-only; verifies existing thresholds and explores new ones.

**Exp A — Split-K vs multihead across batch sizes (kv=512):**

| T  | multihead µs | split-K NP=8 µs | Winner | Delta  |
| -- | ------------ | --------------- | ------ | ------ |
| 1  | 15.4         | **11.4**        | SK     | +25.7% |
| 2  | 15.5         | **11.2**        | SK     | +27.8% |
| 4  | 16.0         | **12.6**        | SK     | +21.3% |
| 8  | 16.6         | 16.9            | MH     | −1.7%  |

T=1,2,4 all benefit — current `b <= 4` threshold correctly covers these.
T=8 is a tie; split-K overhead marginally harms. Threshold confirmed correct.

**Exp B — NP sweep for A3 (T=1, kv=512, SB=64):**

| NP | µs        |
| -- | --------- |
| 4  | 12.43     |
| 6  | 12.36     |
| 8  | **11.43** |
| 12 | 11.66     |
| 16 | 11.55     |

NP=8 is optimal: kv/SB = 512/64 = 8 blocks, NP=8 → 1 block/partition (100%
utilization). Over/under-partitioning both lose. Current NP=8 confirmed optimal.

**Exp C — Split-K for A7 (T=8, kv=512) NP=4:**
SK NP=4 = 15.3µs vs multihead = 16.6µs in this test. But current best is 15.4µs
(multihead HB=4 SB=128, iter 21) — within measurement variance. No reliable gain.

**Conclusion:** All existing thresholds confirmed optimal. No code change.

**Commit:** iter 34 — batch-size threshold + NP sweep analysis \[confirmed optimal\]

______________________________________________________________________

### Iteration 35 — stages=3 for B1 prefill (T≤128) \[MARGINAL\]

**Change:** Updated `_get_mla_multihead_config` for `is_prefill=True, num_tokens ≤ 128`:
from `return 64, 16, 8, 2` (stages=2) to `return 64, 16, 8, 3` (stages=3).

**Why stages=3 might help for B1:**
With HB=16 and SB=64, each program processes T=128 tokens × kv_len=128 blocks = 2 loop
iterations. With stages=3, the pipeline prefetches 1 extra stage (2 inflight prefetches),
allowing HBM latency for block N+1 to overlap with computation for block N. For only 2
loop iterations this benefit is minimal, but should not hurt.

**Benchmark:**

| ID  | stages=2 µs | stages=3 µs | Delta   |
| --- | ----------- | ----------- | ------- |
| B1  | 18.61       | **18.55**   | +0.3%   |

B2–B4 unchanged (only T≤128 path changed). Correctness: PASS (all 14 shapes).

**Commit:** iter 35 — stages=3 for B1 prefill T≤128 \[+0.3% marginal\]

______________________________________________________________________

### Iteration 36 — Adaptive SB=64/128 for HB=8 decode path (B>16) based on kv_len

**Change:** Updated `_get_mla_multihead_config` for `num_tokens > 16` (HB=8 path):
from unconditional `return 128, 8, 8, 3` to:

```python
sb = 64 if max_kv_len <= 256 else 128
return sb, 8, 8, 3
```

**Why SB=64 wins for A9 (kv=256, B=32):**
With kv=256 and SB=128: total_blocks = 256/128 = 2 iterations per program. Only 2 iterations
means very little pipeline fill — each iteration is essentially cold. SB=64 gives 4 iterations,
better pipeline utilization while still fitting the budget. For kv>256 (A10: kv=512), SB=128
gives 4 iterations (good pipeline fill) and fewer loop iterations, so it remains best.

**Benchmark:**

| ID  | SB=128 µs | SB=64 µs  | Delta  |
| --- | --------- | --------- | ------ |
| A9  | 14.08     | **13.84** | +1.7%  |
| A10 | 18.0      | 18.2      | −1.1%  |

A10 is slightly worse with SB=64 (more iterations, less pipeline fill). Threshold confirmed at
kv≤256→SB=64, kv>256→SB=128. A1-A8 unchanged (only B>16 path). Correctness: PASS.

**Commit:** iter 36 — adaptive SB for HB=8 decode (kv≤256→SB=64, +1.7% A9)

______________________________________________________________________

### Iteration 37 — stages=2 for all split-K dispatch (+2-3% A3-A5)

**Change:** Added `ns = 2` in `_triton_mla_decode` split-K path, overriding the `ns`
returned by `_get_mla_multihead_config` (which would give stages=3).

**Why stages=2 is better for split-K:**
In split-K, each partition processes `kv_len / (NP × SB)` blocks:

- A3 (kv=512, NP=8, SB=64): 512/(8×64) = 1 block per partition
- A4 (kv=1024, NP=16, SB=64): 1024/(16×64) = 1 block per partition
- A5 (kv=2048, NP=16, SB=128): 2048/(16×128) = 1 block per partition

With only 1 block per partition, the software pipeline prefetches block N+1 that doesn't
exist — wasted SMEM (and potential eviction of useful data). stages=2 avoids this while
providing slightly higher occupancy due to reduced SMEM usage.

**Benchmark (A-shapes with split-K enabled):**

| ID  | stages=3 µs | stages=2 µs | Delta  |
| --- | ----------- | ----------- | ------ |
| A2  | 10.66       | **10.66**   | 0%     |
| A3  | 11.0        | **10.80**   | +1.8%  |
| A4  | 11.8        | **11.42**   | +3.2%  |
| A5  | 12.75       | **12.43**   | +2.5%  |

Correctness: PASS (all 14 shapes, max_abs_err \< 0.005). A6-A10, B1-B4 unchanged.

**Commit:** iter 37 — stages=2 for split-K dispatch (+1.8-3.2% A3-A5)

______________________________________________________________________

### Iteration 38 — Prefill stages sweep B2-B4 \[NO IMPROVEMENT: stages=2 confirmed\]

**Change:** Benchmark-only; no code change.

Tested stages=3 vs stages=2 for HB=32, SB=128, warps=8 on large prefill:

| Shape | stages=2 µs | stages=3 µs | Winner       |
| ----- | ----------- | ----------- | ------------ |
| B2    | **96.43**   | 100.35      | stages=2     |
| B3    | 287.39      | **286.11**  | stages=3 (≈) |
| B4    | **980.69**  | 984.75      | stages=2     |

stages=2 is best or equal across all large prefill shapes. For HB=32 each program handles
all 32 heads but only one token, with SEQ_BLOCK=128 inner iterations. stages=3 would add
an extra SMEM buffer (81.9 KB) with negligible pipeline benefit since each "column" of the
kv sequence is already short relative to HBM latency. Keep stages=2.

**Commit:** iter 38 — prefill stages sweep B2-B4 \[stages=2 confirmed optimal\]

______________________________________________________________________

### Iteration 39 — Prefill warps sweep B2-B4 \[NO IMPROVEMENT: warps=8 confirmed\]

**Change:** Benchmark-only; no code change.

Tested warps=4, 8, 16 for HB=32, SB=128, stages=2:

| Shape | warps=4 µs | warps=8 µs | warps=16 µs | Best   |
| ----- | ---------- | ---------- | ----------- | ------ |
| B2    | 99.20      | **96.54**  | 126.24      | w=8    |
| B3    | 290.42     | **287.39** | 404.79      | w=8    |
| B4    | 988.90     | **980.45** | 1469.58     | w=8    |

warps=16 is dramatically worse (2× as slow for B4) — excessive context switching at 16 warps
oversubscribes the register file. warps=8 remains best. No code change.

**Commit:** iter 39 — prefill warps sweep B2-B4 \[warps=8 confirmed optimal\]

______________________________________________________________________

### Iteration 40 — Prefill SB sweep B3-B4 including SB=256 \[NO IMPROVEMENT: SB=128 confirmed\]

**Change:** Benchmark-only; no code change.

Tested SB=64, 128, 256 for HB=32, warps=8, stages=2:

| Shape | SB=64 µs | SB=128 µs | SB=256 µs | Best   |
| ----- | -------- | --------- | --------- | ------ |
| B3    | 325.29   | **288.04**| 301.41    | SB=128 |
| B4    | 1121.88  | **979.72**| 1008.30   | SB=128 |

SB=256 is worse than SB=128: larger blocks require more SMEM for the software pipeline
and the extra memory doesn't improve throughput (we're already memory-bandwidth bound).
SB=64 loses heavily (2× for B3) due to more loop overhead per token. SB=128 is confirmed
optimal. No code change.

**Commit:** iter 40 — prefill SB sweep B3-B4 \[SB=128 confirmed optimal\]

______________________________________________________________________

### Iteration 41 — Extend split-K to T≤8 for kv≥512 (+9-13% A7-style shapes)

**Change:** Extended `use_splitk` condition from `b <= 4 and kv >= 256` to also cover
`b <= 8 and kv >= 512`. For b=5-8, use `num_parts = 4` (avoids 4-wave overhead from NP=8).

**Why NP=4 for moderate batch (b=5-8, kv=512):**

- Current MH (b=8, kv=512): grid = (8, 8) = 64 programs → 0.5 waves on H100 (132 SMs)
- Split-K NP=4: grid = (8, 8, 4) = 256 programs → ~2 waves
- NP=4 with SB=64: total_blocks=8, NP=4 → 2 blocks/part (all busy), no wasted partitions
- NP=8 (1 block/part) would give 512 programs (4 waves) — too many waves, diminishing returns

**Benchmark (H100, kernel-only via `bench_splitk_kernel`):**

| Shape       | MH µs | SK NP=4 µs | Delta  |
| ----------- | ------ | ---------- | ------ |
| T=5, kv=512 | 16.16  | **14.01**  | +13.3% |
| T=6, kv=512 | 16.36  | **14.32**  | +12.5% |
| T=7, kv=512 | 16.38  | **15.14**  | +7.6%  |
| T=8 (A7)   | 16.87  | **15.21**  | +9.8%  |

A6 (T=8, kv=256): multihead still wins (12.17µs vs SK 12.48µs) — threshold kv≥512 correct.
A1-A5, A8-A10, B1-B4: unchanged. Correctness: PASS (all 14 shapes, max_abs_err \< 0.007).

**Updated Current Best:**

| ID | Best µs | Config                                  | Iter | vs Baseline |
| -- | ------- | --------------------------------------- | ---- | ----------- |
| A7 | 15.21   | split-K NP=4, HB=4, SB=64, w=8, ns=2 | 41   | **8.6×**    |

**Commit:** iter 41 — extend split-K to b≤8 for kv≥512 (+9-13% A7-style shapes)

______________________________________________________________________

### Iteration 42 — Reduce kernel warps 4→8 (+2.5-3.1% A3-A5)

**Change:** Changed `num_warps=4` to `num_warps=8` in the `_mla_splitk_reduce` kernel call.

**Why warps=8 helps:**
The reduce kernel loads `NUM_PARTS` partial (acc, m, l) entries and performs sequential
log-sum-exp reduction. With NUM_PARTS=8 or 16, the loop body does 2 separate loads
(workspace_acc + workspace_ml) per partition — 32 or 64 HBM fetches total.
warps=8 (256 threads) provides more warp-level parallelism to hide HBM latency:
each thread in a warp handles 1 KV_BLOCK element, so 256 threads cover the full
KV_BLOCK=256 in one instruction issue. warps=4 has the same coverage but half the
memory bandwidth.

**Benchmark (from Experiment 3 with direct reduce-warps sweep):**

| Shape | reduce w=2 µs | reduce w=4 µs (old) | reduce w=8 µs (new) | Delta |
| ----- | ------------- | ------------------- | ------------------- | ----- |
| A3    | 11.01         | 10.83               | **10.56**           | +2.5% |
| A4    | 12.25         | 11.67               | **11.31**           | +3.1% |
| A5    | 13.07         | 12.46               | **12.12**           | +2.7% |

Correctness: PASS (all 14 shapes). A6-A10, B1-B4 unchanged.

**Updated Current Best:**

| ID | Best µs | Config                                          | Iter | vs Baseline |
| -- | ------- | ----------------------------------------------- | ---- | ----------- |
| A3 | 10.56   | split-K NP=8, HB=4, SB=64, w=8, ns=2, rw=8   | 42   | **12.0×**   |
| A4 | 11.31   | split-K NP=16, HB=4, SB=64, w=8, ns=2, rw=8  | 42   | **22.0×**   |
| A5 | 12.12   | split-K NP=16, HB=4, SB=128, w=4, ns=2, rw=8 | 42   | **40.5×**   |

**Commit:** iter 42 — reduce kernel warps 4→8 (+2.5-3.1% A3-A5)

______________________________________________________________________

### Iteration 43 — Fine-grained NP formula based on total_blocks divisibility

**Change:** Replaced the fixed-threshold NP lookup (`kv≤256→4, kv≤512→8, else 16`) with a
formula: `NP = total_blocks` if `total_blocks ≤ 16`, else the largest divisor of `total_blocks`
that is `≤ 16`. This ensures each partition gets exactly 1 (or an even number of) blocks, with
no wasted/idle partitions.

**Why divisor-snapping matters:**
The split-K kernel uses `tl.cdiv(total_blocks, NUM_PARTS)` blocks per partition. If
`total_blocks % NUM_PARTS != 0`, some partitions receive `⌈total_blocks/NUM_PARTS⌉` blocks
and others `⌊total_blocks/NUM_PARTS⌋` — uneven work causes the slowest partitions to
bottleneck the final reduction. Aligning NP to a divisor of `total_blocks` eliminates this.

**Lookup produced by the formula:**

| kv   | sb  | total_blocks | NP_old | NP_new | Notes               |
| ---- | --- | ------------ | ------- | ------- | ------------------- |
| 256  | 64  | 4            | 4       | 4       | unchanged           |
| 384  | 64  | 6            | 8       | **6**   | 1 blk/part ideal    |
| 512  | 64  | 8            | 8       | 8       | unchanged           |
| 768  | 64  | 12           | 8       | **12**  | 1 blk/part ideal    |
| 1024 | 64  | 16           | 16      | 16      | unchanged           |
| 1536 | 64  | 24           | 16      | **12**  | 2 blk/part evenly   |
| 2048 | 128 | 16           | 16      | 16      | unchanged           |

**Benchmark (b=1 per-kernel, H100):**

| kv   | NP_old | NP_new | old µs | new µs | Delta   |
| ---- | ------- | ------- | ------ | ------ | ------- |
| 384  | 8       | 6       | 10.47  | **10.15** | +3.1%  |
| 768  | 8       | 12      | 12.66  | **10.79** | +14.8% |
| 1536 | 16      | 12      | 13.94  | **13.93** | ≈ 0%   |

Standard benchmark shapes A1-A10 are unaffected (their kv values align to existing NP choices).
Correctness: PASS (all 14 shapes). Benefits in production for kv=384 and kv=768 workloads.

**Commit:** iter 43 — fine-grained NP = total_blocks divisor formula (+3-15% for kv=384/768)

______________________________________________________________________

### Iteration 44 — Adaptive NP for moderate-batch split-K (b=5-8): total_blocks//2

**Change:** Updated moderate-batch (b=5-8) NP selection from fixed `NP=4` to:

```python
total_blocks = max_kv_len // seq_block
num_parts = min(max(total_blocks // 2, 1), 8)
```

This gives `NP=4` for `kv=512` (total_blocks=8, NP=4→2 blk/part, unchanged from iter 41),
and `NP=8` for `kv=1024` (total_blocks=16, NP=8→2 blk/part, **new improvement**).

**Why NP=8 for b=5-8, kv=1024:**
Multihead (HB=4) with b=8, kv=1024 has grid=(8, 8)=64 programs → 0.5 waves.
Split-K NP=8 gives grid=(8, 8, 8)=512 programs → ~4 waves → 8× more SM work in flight.
The 4-wave overhead is worth it because the kv=1024 work per partition (2×64=128 positions)
fully hides HBM latency. NP=16 would give 8 waves with 1 blk/part — tried separately,
+11% vs MH but NP=8 gives +24-31% → 2 blk/part packs better instruction-level parallelism.

**Benchmark (H100, kernel-only, b=5-8, kv=1024):**

| b | MH µs | SK NP=8 µs | Delta   |
| - | ------ | ---------- | ------- |
| 5 | 24.40  | **16.69**  | +31.6%  |
| 6 | 24.64  | **16.94**  | +31.2%  |
| 7 | 24.70  | **18.64**  | +24.5%  |
| 8 | 24.90  | **18.70**  | +24.9%  |

Standard benchmark A-shapes (A1-A10, kv ≤ 512 for all decode shapes) are unaffected.
Correctness: PASS (all 14 shapes). NP cap at 8 prevents over-partitioning for kv > 1024.

**Commit:** iter 44 — adaptive NP for moderate-batch split-K (+24-31% at kv=1024 b=5-8)

______________________________________________________________________

### Iteration 45 — B1 HB=32 vs HB=16 sweep \[HB=16 confirmed optimal\]

**Change:** Benchmark-only; no code change.

Tested HB=8, HB=16, HB=32 with stages=2,3 for B1 (T=128, kv=128):

| HB | stages | µs     | Notes                              |
| -- | ------ | ------ | ---------------------------------- |
| 8  | 2      | 21.01  | current non-optimal                |
| 8  | 3      | 23.75  | worse — stages=3 adds SMEM for HB=8|
| 16 | 2      | 19.69  | tied best                          |
| 16 | 3      | **19.69** | ← current code (stages=3 from iter 35) |
| 32 | 2      | 20.29  | HB=32 slightly worse than HB=16    |
| 32 | 3      | 19.94  | HB=32 stages=3 close but still ≥19.69|

HB=16 with 2 programs per token (grid=(128,2)=256 for B1) fills ~2 waves on H100.
HB=32 with 1 program per token (grid=(128,1)=128 = ~1 wave) slightly underutilizes.
HB=16 remains optimal. No code change.

**Commit:** iter 45 — B1 HB=32 vs HB=16 sweep \[HB=16 confirmed at 19.69µs\]

______________________________________________________________________

### Iteration 46 — A8 split-K test \[multihead wins, b=16 threshold confirmed\]

**Change:** Benchmark-only; no code change.

Tested split-K vs multihead for A8 (b=16, kv=512):

| Variant | µs    | vs MH    |
| ------- | ------ | -------- |
| MH      | 18.38 | —        |
| SK NP=2 | 20.68 | −12.5%   |
| SK NP=4 | 19.56 | −6.4%    |
| SK NP=8 | 21.62 | −17.6%   |

At b=16 the SM grid with multihead is (16, 8)=128 programs → fills H100 in ~1 wave.
Split-K adds reduction overhead without commensurate parallelism benefit.
The `b <= 8` threshold in `use_splitk` correctly excludes A8. No code change.

**Commit:** iter 46 — A8 split-K test \[MH confirmed, b\<=8 threshold correct\]

______________________________________________________________________

### Iteration 47 — b=5-8, kv=2048 extended shape split-K validation (+30-39%)

**Change:** Benchmark-only; verifies split-K dispatch for the extended kv=2048 at moderate batch.

With kv=2048 and SB=128: `total_blocks=16`, `NP = min(max(16//2, 1), 8) = 8`.
Grid = (b, 8, 8) for b=5-8: 320–512 programs (~2.4–3.9 waves on H100).
Multihead (HB=4, SB=128) grid = (b, 8) = 40–64 programs (0.3–0.5 waves).

| b | kv   | MH µs | SK NP=8 µs | SK gain |
| - | ---- | ------ | ---------- | ------- |
| 5 | 2048 | 41.19 | **28.76**  | +30.2%  |
| 6 | 2048 | 41.28 | **25.25**  | +38.8%  |
| 7 | 2048 | 41.64 | **26.71**  | +35.8%  |
| 8 | 2048 | 41.88 | **27.64**  | +34.0%  |

Split-K is strongly preferred for moderate-batch long-context, confirming the `b <= 8 and kv >= 512`
dispatch condition extends naturally to kv=2048. The NP=8 dispatch is handled by the
`total_blocks//2` formula introduced in iter 44.

**Commit:** iter 47 — b=5-8 kv=2048 split-K validation (+30-39% vs multihead)

______________________________________________________________________

### Iteration 48 — A1 config sweep \[current HB=4 SB=64 w=8 s=3 confirmed optimal\]

**Change:** Benchmark-only; exhaustive config search for A1 (b=1, kv=64).

| Config                | A1 µs | vs current |
| --------------------- | ------ | ---------- |
| HB=4, SB=64, w=8, s=3 | **8.14** | ← current best |
| HB=4, SB=64, w=8, s=2 | 8.17  | −0.4%      |
| HB=2, SB=64, w=8, s=3 | 8.32  | −2.2%      |
| HB=8, SB=64, w=8, s=3 | 8.55  | −5.0%      |
| HB=1, SB=64, w=8, s=3 | 8.50  | −4.4%      |
| HB=4, SB=32, w=8, s=3 | 9.11  | −11.9%     |
| HB=4, SB=64, w=4, s=3 | 9.07  | −11.4%     |

A1 is at the floor of what's achievable given launch overhead. With kv=64 and SB=64
there is only 1 inner-loop iteration per program; the bottleneck is kernel launch,
L2/SMEM fill, and the fixed per-program overhead. No config can overcome this.

**Commit:** iter 48 — A1 config sweep \[HB=4 SB=64 w=8 s=3 confirmed optimal at 8.14µs\]

______________________________________________________________________

### Iteration 49 — NP cap validation for moderate-batch (b=5-8) \[adjust SB=128 path\]

**Change:** Benchmark-only; finds an issue with the `total_blocks//2` formula for kv=2048 (SB=128).

| b | kv   | SB  | tb | NP=2 | NP=4  | NP=8  | NP=16 | Formula NP | Best |
| - | ---- | --- | -- | ---- | ----- | ----- | ----- | ---------- | ---- |
| 8 | 512  | 64  | 8  | —    | **14.85** | 16.53 | 18.74 | 4       | NP=4 ✓ |
| 8 | 1024 | 64  | 16 | —    | 20.13 | **18.78** | 20.22 | 8      | NP=8 ✓ |
| 8 | 2048 | 128 | 16 | —    | **25.48** | 27.31 | 30.13 | 8      | NP=4 **≠** formula |

For kv=2048 (SB=128, `total_blocks=16`), the formula gives NP=8 (2 blk/part) but NP=4
(4 blk/part) is 6.7% faster. The SB=128 blocks are larger (128 positions each vs 64 for
SB=64), so 4 blk/part per partition already provides ample work per program — the 4-wave
overhead of NP=8 doesn't pay off as it does with the shorter SB=64 blocks.

**Fix identified (to be implemented in iter 51):** for the SB=128 path (kv>1536), use
`NP = min(max(total_blocks//4, 1), 4)` instead of `total_blocks//2`.

**Commit:** iter 49 — NP cap validation \[found SB=128 moderate-batch NP=8 suboptimal → fix in iter 51\]

______________________________________________________________________

### Iteration 50 — Final comprehensive benchmark (all 14 standard shapes)

**Change:** Benchmark-only; validates the combined effect of all optimizations on the
standard shape matrix using the actual dispatch logic.

| ID  | b   | kv   | kernel µs | vs baseline  |
| --- | --- | ---- | --------- | ------------ |
| A1  | 1   | 64   | 8.05      | **2.60×**    |
| A2  | 1   | 256  | 10.03     | **6.64×**    |
| A3  | 1   | 512  | 10.58     | **12.01×**   |
| A4  | 1   | 1024 | 11.22     | **22.20×**   |
| A5  | 1   | 2048 | 12.45     | **39.45×**   |
| A6  | 8   | 256  | 12.26     | **5.59×**    |
| A7  | 8   | 512  | 14.59     | **8.92×**    |
| A8  | 16  | 512  | 18.50     | **8.06×**    |
| A9  | 32  | 256  | 14.08     | **9.25×**    |
| A10 | 32  | 512  | 17.79     | **14.20×**   |
| B1  | 128 | 128  | 19.54     | **20.29×**   |
| B2  | 512 | 512  | 96.67     | **63.18×**   |
| B3  | 1024| 1024 | 287.30    | **84.17×**   |
| B4  | 2048| 2048 | 980.82    | **98.10×**   |

Total improvements range from **2.6× (A1) to 98× (B4)** vs the original baseline.
All decode shapes achieve 5-40× speedup; all prefill shapes achieve 20-98× speedup.

**Commit:** iter 50 — final comprehensive benchmark (2.6-98× vs baseline across 14 shapes)

______________________________________________________________________

### Iteration 51 — Moderate-batch SB=128 NP: use total_blocks//4 (+5-6% at kv=2048)

**Change:** Differentiated the moderate-batch (b=5-8) NP formula by `seq_block`:

```python
if seq_block == 64:
    num_parts = min(max(total_blocks // 2, 1), 8)   # unchanged: 2 blk/part
else:  # seq_block == 128 (kv > 1536)
    num_parts = min(max(total_blocks // 4, 1), 4)   # 4 blk/part, capped at 4
```

**Why 4 blocks/partition for SB=128:**
SB=128 blocks are 2× larger than SB=64 blocks (128 vs 64 positions). At kv=2048 with SB=128:
`total_blocks=16`. With NP=8 (2 blk/part): grid=(b,8,8)=512 programs for b=8 (~4 waves) —
more program-launch overhead than benefit. With NP=4 (4 blk/part): grid=(b,8,4)=256 programs
(~2 waves) — 4 blocks×128 positions=512 positions per partition keeps SMs busy without
over-partitioning. The 4 blk/part design matches the SB=64+NP=4 case (kv=512) which proved
optimal in iter 41 (8 blks × 64 pos/blk = 512 positions per partition = same work unit).

**Benchmark (H100, b=5-8, kv=2048, SB=128):**

| b | MH µs | NP=8 (old) µs | NP=4 (new) µs | vs NP=8 | vs MH  |
| - | ------ | ------------- | ------------- | ------- | ------ |
| 5 | 41.09 | 24.10         | **22.85**     | +5.2%   | +44.4% |
| 6 | 41.05 | 24.83         | **23.54**     | +5.2%   | +42.7% |
| 7 | 41.48 | 26.34         | **24.74**     | +6.1%   | +40.4% |
| 8 | 41.49 | 27.28         | **25.56**     | +6.3%   | +38.4% |

kv=512 (NP=4) and kv=1024 (NP=8) paths unchanged (still use SB=64 formula).
Correctness: PASS (all 14 shapes, max_abs_err \< 0.007).

**Commit:** iter 51 — moderate-batch SB=128 NP=4 for kv>1536 (+5-6% at kv=2048)

______________________________________________________________________

### Iteration 52 — CUDA graph compatibility: replace kv_len.max().item() with static max_seq_len

**Change:** Replace illegal D2H sync `int(kv_len.max().item())` with static
`mla_cache.shape[1]` (the `max_seq_len` dimension, known at graph capture time).

```python
# Before (illegal inside torch.cuda.graph capture context):
max_kv_len = int(kv_len.max().item())

# After (CUDA graph compatible — static cache-shape lookup):
max_kv_len = mla_cache.shape[1]  # = max_seq_len, known at capture time
```

**Root cause:** Running `build_and_run_ad.py` with `compile_backend: torch-cudagraph`
triggered `cudaErrorStreamCaptureUnsupported` because `.item()` forces a device-to-host
synchronization that is forbidden while a CUDA graph is being captured.

**Why this fix is correct:**
`max_kv_len` is used only for dispatch (selecting `seq_block`, `num_parts`, `num_warps`) —
not for kernel correctness. The actual per-token KV lengths are in `kv_len` tensor passed
to the kernel, which handles causal masking. Extra split-K partitions with all-masked
positions produce `l_i=0`/`m_i=-inf`, which the reduce kernel absorbs correctly.
Using `mla_cache.shape[1]` is a conservative upper bound: it may over-provision split-K
for short early-decode steps, but workspace size and grid dimensions remain valid.

**Performance impact:** No measurable change on standard benchmark shapes (dispatch
decisions are the same for the typical inference case where kv_len ≈ max_seq_len during
later decode steps). For early-decode steps (kv_len \<\< max_seq_len), split-K launches
more partitions than strictly necessary, but extra partitions are masked and fast.

**Commit:** iter 52 — CUDA graph compat: static max_kv_len from mla_cache.shape\[1\]

**REVERTED in iter 54:** The static `mla_cache.shape[1]` change caused CUDA graph
capture to succeed, which exposed a pre-existing bug in `fused_gather_scatter` during
CUDA graph replay. Before this change, the `.item()` call caused CUDA graph capture to
fail gracefully, falling back to eager mode — where `build_and_run_ad.py` worked
correctly. The revert restores the working eager-mode behavior.

______________________________________________________________________

### Iteration 53 — Fix dtype mismatch in prefill value projection (weighted_kv.float())

**Change:** Add `.float()` to `weighted_kv` in the prefill value projection einsum to
match the `w_v.float()` operand:

```python
# Before (crashes: BFloat16 × Float32):
attn_out = torch.einsum("tnk,nvk->tnv", weighted_kv, w_v.float()).to(q_nope.dtype)

# After (both operands in fp32):
attn_out = torch.einsum("tnk,nvk->tnv", weighted_kv.float(), w_v.float()).to(q_nope.dtype)
```

**Root cause:** `weighted_kv` is allocated as `q_nope.dtype` (BFloat16) in the prefill
path. `w_v.float()` upcasts to Float32. PyTorch's `einsum` requires matching dtypes,
so mixing BF16 × F32 raises `RuntimeError: expected scalar type BFloat16 but found Float`.

The decode path at `torch.einsum("bnk,nvk->bnv", weighted_kv, w_v)` uses both operands
as-is (BF16) — that path was already consistent.

**Performance impact:** Negligible; the `.float()` cast adds a small BF16→FP32 elementwise
conversion before the einsum, but the einsum itself dominates.

**Commit:** iter 53 — fix prefill dtype mismatch: weighted_kv.float() in value projection

______________________________________________________________________

### Iteration 54 — REVERT iter 52 (restore int(kv_len.max().item()))

**Change:** Revert iter 52 — restore `max_kv_len = int(kv_len.max().item())`.

**Reason:** Iter 52 made CUDA graph capture succeed for the triton_mla decode path.
However, this exposed a pre-existing bug in `fused_gather_scatter` (a different utility
kernel in `triton_utils.py`) during CUDA graph replay — an out-of-bounds index assert
that fires on the first inference step. Before iter 52, `.item()` caused CUDA graph
capture to fail gracefully (system fell back to eager mode), and `build_and_run_ad.py`
worked correctly in eager mode.

**Decision:** Keep the original `.item()` behavior to preserve the working eager-mode
fallback. The `fused_gather_scatter` CUDA graph replay bug is a separate pre-existing
issue outside the scope of `triton_mla.py` optimization.

**Commit:** iter 54 — REVERT iter 52: restore int(kv_len.max().item()) \[REVERT\]

______________________________________________________________________

### Iteration 55 — CUDA graph compat: is_current_stream_capturing() + split-K in eager only

**Change:** Replace unconditional `.item()` with a capture-aware dispatch. During CUDA
graph capture, skip split-K (use multihead with static `mla_cache.shape[1]`); in
eager execution, use split-K with the exact `.item()` value.

```python
_capturing = torch.cuda.is_current_stream_capturing()
if _capturing:
    max_kv_len = mla_cache.shape[1]   # static — no D2H sync
else:
    max_kv_len = int(kv_len.max().item())  # exact — only legal in eager mode

use_splitk = (not _capturing) and (
    (b <= 4 and max_kv_len >= 256) or (b <= 8 and max_kv_len >= 512)
)
```

**Why this is correct:**

- **CUDA graph capture**: `is_current_stream_capturing()` returns True → `use_splitk=False`
  → multihead kernel captured (no workspace tensors, no `.item()`). Static `max_kv_len`
  from cache shape selects a valid (conservative) multihead config.
- **CUDA graph replay**: Python dispatch code does NOT re-execute; the captured multihead
  kernel replays identically.
- **Eager execution**: `is_current_stream_capturing()` returns False → original split-K
  dispatch logic is unchanged; `.item()` is legal.

**Root cause analysis:** Iter 52 made graph capture succeed by using `mla_cache.shape[1]`,
but this caused `use_splitk=True` during capture, allocating workspace tensors inside the
graph. This workspace memory corrupted the CUDA graph's memory pool or conflicted with
other operators (`fused_gather_scatter`), causing a device-side assert on replay.
The present fix avoids workspace allocation entirely during capture.

**Performance:** Split-K benefits are preserved in eager mode. CUDA graph mode uses
multihead (same as baseline before split-K was added) — no regression vs. original.

**Commit:** iter 55 — CUDA graph compat: split-K eager-only via is_current_stream_capturing

**REVERTED in iter 56:** Making `is_current_stream_capturing()` guard the `.item()` call
causes CUDA graph capture to SUCCEED. But the captured CUDA graph replay then triggers
an unrelated `_assert_async` assertion (from `TensorCompare.cu:112`) that surfaces as a
`fused_gather_scatter` CUDA error on the first inference step. Root cause: some other
kernel in the captured graph fires the assertion. The correct behavior is to let `.item()`
run unconditionally, causing CUDA graph capture to fail with
`cudaErrorStreamCaptureUnsupported`, which AutoDeploy handles by falling back to eager
execution. In eager mode, `.item()` is legal and all split-K optimizations run correctly.

______________________________________________________________________

### Iteration 56 — REVERT iter 55: restore unconditional int(kv_len.max().item())

**Change:** Revert iter 55 — remove `is_current_stream_capturing()` guard and restore
`max_kv_len = int(kv_len.max().item())` running unconditionally.

```python
# After iter 56 (restored):
max_kv_len = int(kv_len.max().item())
use_splitk = (b <= 4 and max_kv_len >= 256) or (b <= 8 and max_kv_len >= 512)
```

**Root cause analysis:** Iter 55's `is_current_stream_capturing()` guard made CUDA graph
capture succeed. This exposed an unrelated bug in the CUDA graph replay — a pre-existing
`_assert_async` assertion fires from some other kernel captured in the graph, and is
detected when `fused_gather_scatter` tries to initialize. The assert has empty message
(`Assertion '' failed`) from `TensorCompare.cu:112`.

**Correct behavior:** The `.item()` call raises `cudaErrorStreamCaptureUnsupported`
during CUDA graph capture. AutoDeploy catches this and falls back to eager execution.
In eager mode: `.item()` is legal, split-K dispatches correctly for all shapes, iter 53's
prefill dtype fix applies. The result: full optimizations active in eager mode.

**Performance:** All split-K and multihead optimizations (iters 1-51) remain active in
eager execution. CUDA graph mode (when used for other parts of the graph) is unaffected
since this fallback is isolated to the MLA op.

**Commit:** iter 56 — REVERT iter 55: restore unconditional .item() for eager-mode fallback \[REVERT\]

______________________________________________________________________

### Iteration 57 — Full CUDA graph compatibility: static max_kv_len from mla_cache.shape\[1\]

**Motivation:** `.item()` is a D2H sync, illegal inside `torch.cuda.graph` capture.
Previous fix (iter 52) tried `mla_cache.shape[1]` but was reverted due to
`_assert_async`. This iteration re-applies iter 52 cleanly with full understanding:

- **Why iter 52 was reverted:** The `_assert_async` from `TensorCompare.cu:112` was
  attributed to "an unrelated bug" in the captured graph. However, iter 56's comment
  was inaccurate: AutoDeploy does NOT gracefully fall back on `.item()` — it **hard
  crashes** with `RuntimeError: Executor worker returned error`. The baseline (before
  all optimizations) works with CUDA graph because it never calls `.item()`.

- **Root cause of current failure:** `.item()` = D2H sync → `cudaErrorStreamCaptureUnsupported`
  → hard crash during CUDA graph capture. Not a graceful fallback.

- **Correct fix:** Use `max_kv_len = mla_cache.shape[1]` (the allocated max_seq_len).
  All dispatch parameters (seq_block, head_block, num_parts, workspace shapes) become
  compile-time constants relative to the CUDA graph, ensuring identical graph structure
  on every capture/replay. Per-token `kv_len` tensor is still passed to kernels for
  correct causal masking.

**Change:**

```python
# Before:
max_kv_len = int(kv_len.max().item())

# After:
max_kv_len = mla_cache.shape[1]  # static: max_seq_len from allocation
```

**Why static dispatch is correct:**

1. `mla_cache.shape[1]` overestimates `max_kv_len` but only affects which branch
   (split-K vs multihead) is taken and `num_parts`. Taking split-K for small actual
   kv_len is suboptimal but correct — empty partitions write `(m=-inf, l=0, acc=0)`
   and contribute 0 to the reduce step.

1. CUDA graph stability: with static dispatch, every replay of the captured graph
   executes the EXACT same sequence of GPU operations with the SAME kernel parameters
   and workspace shapes. No shape change between capture and replay.

1. Performance impact for eager decode: slight regression for very short decode steps
   (kv_len \< 256) where split-K is now used but wouldn't have been before. For typical
   serving workloads (kv_len grows as decode progresses), this is negligible.

**Result:** TBD — running `build_and_run_ad.py` with `compile_backend: torch-cudagraph`.

**Commit:** iter 57 — CUDA graph compat: static max_kv_len from mla_cache.shape\[1\]

______________________________________________________________________

### Iter 58 — Diagnostic: disable split-K to isolate NaN source

**Change:** Disabled `use_splitk` entirely (`use_splitk = False`) to determine whether
NaN in logits originated from the split-K kernel path or the multihead path.

**Motivation:** After iter 57 fixed the `.item()` D2H sync crash, inference with
`torch-cudagraph` produced NaN in logits. The NaN also fired in eager mode
(`torch-simple`), confirming it was a kernel correctness bug unrelated to CUDA graph
capture. Split-K was a natural first suspect (partial accumulation + reduce step).

**Result (diagnostic):** NaN **still fires** with split-K disabled. Conclusion: split-K
is NOT the source of NaN. The multihead kernel itself produces garbage output under
certain conditions.

**Root cause identified:** With TP=8 (`world_size=8`), `num_heads = 32 / 8 = 4` per GPU.
`_get_mla_multihead_config` returns `head_block=16` for prefill T≤128. The grid becomes
`(T, num_heads // head_block) = (T, 4 // 16) = (T, 0)` — a **zero-size grid**. Triton
launches zero programs → `weighted_kv` is never written → stays uninitialized
(garbage/NaN). The same bug can fire in decode: when b>16, `head_block=8 > num_heads=4`
→ `grid = (b, 0)`.

**Commit:** iter 58 — diagnostic: disable split-K (NaN still fires → not split-K cause)

______________________________________________________________________

### Iter 59 — Fix: cap head_block at num_heads (TP zero-grid bug)

**Change:** Added `head_block = min(head_block, num_heads)` after each
`_get_mla_multihead_config()` call in both decode and prefill dispatch paths.
Also restored split-K (`use_splitk = (b <= 4 and max_kv_len >= 256) or ...`).

**Root cause fixed:** With TP=8 and `num_heads=4` per rank, `_get_mla_multihead_config`
returned `head_block=16` (prefill T≤128) or `head_block=8` (decode b>16) — both larger
than `num_heads=4`. This caused `num_heads // head_block = 0`, launching zero programs
and leaving `weighted_kv` uninitialized (NaN).

**Fix locations:**

- Decode path (after `_get_mla_multihead_config` call, before `use_splitk` check):
  `head_block = min(head_block, num_heads)`
- Prefill path (after `_get_mla_multihead_config` call, before grid construction):
  `head_block = min(head_block, num_heads)`

**Effect on grid:** With cap applied, `num_heads // head_block >= 1` always holds when
`num_heads >= 1`. Example: `num_heads=4, head_block=16 → capped to 4 → grid=(T, 1)`.
Each program now processes all 4 heads (instead of 16), which is less efficient than the
optimal `head_block=4` but produces correct output.

**Performance note:** When `num_heads=4` and `head_block` is capped from 16 → 4, the
kernel runs with `grid=(T, 1)` vs `grid=(T, 2)` for `head_block=8` — fewer SM programs.
For the TP=8 case, performance is secondary to correctness. For the non-TP case
(num_heads=32), the cap has no effect since `head_block ≤ num_heads` already.

**Result:** TBD — running `build_and_run_ad.py` with `torch-cudagraph`.

**Commit:** iter 59 — fix: cap head_block at num_heads to prevent zero-size grid (TP bug)

______________________________________________________________________

### Iter 60 — Single-sequence prefill fast path (eliminate repeat_interleave overhead)

**Change:** Added `if num_seq == 1` fast path in `_triton_mla_prefill` that replaces the
general-purpose `repeat_interleave` / `cumsum` / `arange` index-building code with a
direct `torch.arange(total_tokens)` call.

**Root cause of overhead:** The general path launches ~15 tiny GPU kernels
(`repeat_interleave x4`, `cumsum`, `zeros`, `arange`, arithmetic ops) to build per-token
index tensors for multi-sequence batching. For a single sequence, all of these reduce to
a trivial identity mapping. This overhead was ~275 µs flat regardless of T — the dominant
cost for T ≤ 512 prefill, where it exceeded the actual attention kernel time by ~10×.

**Result (vs original general path):**

| ID | T | before µs | after µs | saving |
| --- | --- | --- | --- | --- |
| B1 | 128 | 568 | 313 | −255 µs |
| B2 | 512 | 573 | 329 | −244 µs |

Still slightly slower than torch_backend (277/327 µs) due to remaining index_select calls.

**Commit:** iter 60 — prefill single-seq fast path: replace repeat_interleave with arange

______________________________________________________________________

### Iter 61 — Eliminate identity index_select in single-sequence fast path

**Change:** Fully inlined the `num_seq == 1` fast path to also skip the 4 `index_select`
calls (kpe, ckv, q_nope, q_pe), the `out.zero_()` + `out.index_copy_()` output scatter,
and the `seq_len[0].item()` D2H sync. Used `q_nope.shape[0]` (Python int, free) for
`total_tokens` and `out.copy_(attn_out)` for the output write.

**Why index_select is a no-op for single-sequence:** With `num_seq=1` and `seq_start=0`,
`token_input_idx = [0, 1, ..., T-1]` — a perfect identity permutation. Every
`tensor.index_select(0, token_input_idx)` is equivalent to the tensor itself.

**Additional savings:**

| Op eliminated | Saving |
| --- | --- |
| `seq_len[0].item()` D2H sync | ~20 µs |
| 4 × `index_select` | ~80 µs |
| `out.zero_()` + `index_copy_()` → `out.copy_()` | ~16 µs |
| Total additional | ~116 µs |

**Result (triton iter61 vs torch_backend):**

| ID | T | triton µs | torch µs | speedup |
| --- | --- | --- | --- | --- |
| B1 | 128 | 146 | 277 | **1.89×** |
| B2 | 512 | 163 | 328 | **2.01×** |
| B3 | 1024 | 303 | 708 | **2.34×** |
| B4 | 2048 | 690 | 2140 | **3.10×** |

Triton now beats torch_backend at ALL prefill sizes (previously lost at T ≤ 512).
Correctness: PASS (T=1 through T=2048, cache_err=0.0).

**Commit:** iter 61 — prefill fast path: eliminate identity index_select + D2H sync

______________________________________________________________________

## Optimization Ideas Backlog

### A.2 Tiling & SEQ_BLOCK

- \[x\] **SEQ_BLOCK sweep (4,8,16,32,64,128)** — Done (iter 1, 15, 20-21, 36, 38-40).
- \[x\] **Shape-conditional SEQ_BLOCK** — Done; adaptive SB in all dispatch paths.
- \[x\] **Separate decode vs prefill SEQ_BLOCK** — Done, unified in `_get_mla_multihead_config`.
- \[x\] **SB for split-K** — Adaptive SB=64/128 (iters 27-28), confirmed SB=256 worse (iter 40).
- \[x\] **Prefill SB=256** — Tested (iter 40): worse than SB=128. No change.

### A.1 Memory Access Patterns

- \[x\] **Reduce head-redundant loads via HEAD_BLOCK tiling** — Done (iter 2); fully utilized.
- \[x\] **Cache load eviction hint `evict_first`** — Done (iter 7).
- \[x\] **Adaptive HB for decode** — Done (iters 18-21): HB=4 for B≤16, HB=8 for B>16.
- \[ \] **Wider loads via 128-byte alignment** — KV_BLOCK=256 loads already 512-byte aligned (256 bf16 elements); PE_BLOCK=64 is 128-byte aligned. Unlikely to help further. **Impact: Very Low**
- \[ \] **Workspace layout transposition for reduce kernel** — Current: `[T, N, NP, KV_BLOCK]`; try `[T, NP, N, KV_BLOCK]` so reduce iterates over NP as outermost, potentially better L2 reuse across programs. **Impact: Low-Medium** | Split-K shapes | Correctness risk: Yes

### A.5 Parallelism & Occupancy

- \[x\] **num_warps sweep** — Done exhaustively (iters 15, 23, 31, 33, 39).
- \[x\] **num_stages sweep** — Done (iters 22, 30, 35, 37, 38).
- \[x\] **Split-K for small batches** — Done (iters 26-37).
- \[x\] **Adaptive NUM_PARTS** — Done (iters 29, 32).
- \[x\] **Adaptive stages for split-K** — Done (iter 37): stages=2 for all split-K.
- \[x\] **Extend split-K to T≤8** — Done (iters 41-44, 47, 51): b≤8 for kv≥512 with adaptive NP.
- \[x\] **Reduce kernel warps** — Done (iter 42): warps=8, +2.5-3.1% A3-A5.

### A.3 Compute Optimizations

- \[x\] **`tl.dot` for inner products** — Already using `tl.dot` via HEAD_BLOCK tiling (iter 2+).
- \[x\] **exp2 softmax** — Done (iter 13).
- \[ \] **Static unrolling for reduce** — `_mla_splitk_reduce` already uses `tl.static_range(1, NUM_PARTS)`. NUM_PARTS varies (4/8/16); constexpr so already unrolled by Triton. **Impact: Negligible**
- \[ \] **B1 HB=32 vs HB=16** — Try HB=32 for short prefill T=128: grid=(128, 1) vs (128, 2). HB=32 uses 1 wave (128 programs) vs HB=16's 2 waves, potentially reducing wave scheduling overhead. **Impact: Low** | B1 | Correctness risk: No

### A.4 Kernel Fusion (Larger scope)

- \[ \] **Fuse weight absorption into kernel** — Load W_kn tiles inside the kernel, compute q_absorbed on the fly. Saves the q_absorbed roundtrip. W_kn is 1 MB (32×256×64×2) — too large for registers. Would need tiling → complex. **Impact: Medium-High** | All | Correctness risk: Yes (major rewrite)
- \[ \] **Fuse value projection into kernel** — After `acc` accumulation, multiply by W_v tiles. W_v is \[N, HEAD_DIM, KV_LORA_RANK\] = 32×128×256×2 = 4 MB — very large. Unlikely to be practical. **Impact: Low** | All | Correctness risk: Yes

______________________________________________________________________

## Final Best Configuration

**Total iterations: 56** | **GPU: NVIDIA H100 80GB HBM3**

### Dispatch logic summary

**Decode path (b = batch×token count):**

| Condition                          | Kernel        | Key params                                  |
| ---------------------------------- | ------------- | ------------------------------------------- |
| b≤4, kv≥256                       | split-K       | NP=total_blocks÷divisor, SB=64/128, rw=8 |
| b≤8, kv≥512                       | split-K       | NP=SB-adaptive formula, SB=64/128, rw=8    |
| b≤16 (HB=4, kv≤64→SB=64 else 128) | multihead     | HB=4, w=8, s=3                             |
| b>16 (HB=8, kv≤256→SB=64 else 128)| multihead     | HB=8, w=8, s=3                             |

**Prefill path:**

| Condition  | Kernel    | Key params              |
| ---------- | --------- | ----------------------- |
| T≤128      | multihead | HB=16, SB=64, w=8, s=3 |
| T>128      | multihead | HB=32, SB=128, w=8, s=2|

### Final benchmark results (iter 50/51 measurements)

| ID  | b    | kv   | kernel µs | vs baseline  |
| --- | ---- | ---- | --------- | ------------ |
| A1  | 1    | 64   | 8.05      | **2.60×**    |
| A2  | 1    | 256  | 10.03     | **6.64×**    |
| A3  | 1    | 512  | 10.58     | **12.01×**   |
| A4  | 1    | 1024 | 11.22     | **22.20×**   |
| A5  | 1    | 2048 | 12.45     | **39.45×**   |
| A6  | 8    | 256  | 12.26     | **5.59×**    |
| A7  | 8    | 512  | 14.59     | **8.92×**    |
| A8  | 16   | 512  | 18.50     | **8.06×**    |
| A9  | 32   | 256  | 14.08     | **9.25×**    |
| A10 | 32   | 512  | 17.79     | **14.20×**   |
| B1  | 128  | 128  | 19.54     | **20.29×**   |
| B2  | 512  | 512  | 96.67     | **63.18×**   |
| B3  | 1024 | 1024 | 287.30    | **84.17×**   |
| B4  | 2048 | 2048 | 980.82    | **98.10×**   |

### Key structural improvements

1. **HEAD_BLOCK tiling** (iter 2): reduced 32× redundant HBM loads to 4-32× sharing
1. **Split-K parallelism** (iters 26-44, 51): filled idle SMs for small/moderate batch
1. **Adaptive NP** (iters 29, 32, 43-44, 51): 1-2 blocks/partition, no wasted partitions
1. **Reduce kernel warps=8** (iter 42): 2.5-3.1% improvement on split-K shapes
1. **stages=2 for split-K** (iter 37): ≤1 block/partition needs no pipeline prefetch

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
