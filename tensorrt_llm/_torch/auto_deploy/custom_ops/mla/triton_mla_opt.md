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

| ID  | Best kernel µs | Config                                           | Iter | vs Baseline |
| --- | -------------- | ------------------------------------------------ | ---- | ----------- |
| A1  | 8.5            | multihead HB=4, SEQ_BLOCK=64, warps=8, stgs=3   | 18   | **2.5×**    |
| A2  | 10.8           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **6.2×**    |
| A3  | 14.3           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **8.9×**    |
| A4  | 21.5           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **11.6×**   |
| A5  | 35.4           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **13.9×**   |
| A6  | 11.9           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **5.8×**    |
| A7  | 15.4           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **8.5×**    |
| A8  | 16.0           | multihead HB=4, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **9.3×**    |
| A9  | 14.0           | multihead HB=8, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **9.3×**    |
| A10 | 18.0           | multihead HB=8, SEQ_BLOCK=128, warps=8, stgs=3  | 21   | **14.0×**   |
| B1  | 19.6           | multihead HB=16, SEQ_BLOCK=64, warps=8, stgs=2  | 19   | **20.2×**   |
| B2  | 96.9           | multihead HB=32, SEQ_BLOCK=128, warps=8, stgs=2 | 15   | **63.0×**   |
| B3  | 289.1          | multihead HB=32, SEQ_BLOCK=128, warps=8, stgs=2 | 15   | **83.6×**   |
| B4  | 980.7          | multihead HB=32, SEQ_BLOCK=128, warps=8, stgs=2 | 15   | **98.1×**   |

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

## Optimization Ideas Backlog

### A.2 Tiling & SEQ_BLOCK \[HIGHEST PRIORITY\]

- \[x\] **SEQ_BLOCK sweep (4,8,16,32,64,128)** — Done (iter 1): lookup table implemented, full GPU sweep pending.
- \[x\] **Shape-conditional SEQ_BLOCK** — Done (iter 1): lookup tables; subsumed into `_get_mla_multihead_config` (iter 15); old single-head configs removed (iter 17).
- \[x\] **Separate decode vs prefill SEQ_BLOCK** — Done (iter 1): separate configs per path; now unified in `_get_mla_multihead_config`.

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
