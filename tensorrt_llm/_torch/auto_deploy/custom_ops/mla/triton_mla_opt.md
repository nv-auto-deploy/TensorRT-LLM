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

| ID | Best kernel µs | Config | Iter | vs Baseline |
|----|---------------|--------|------|-------------|
| A1  | 20.9  | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A2  | 66.6  | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A3  | 127.1 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A4  | 249.0 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A5  | 491.1 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A6  | 68.5  | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A7  | 130.2 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A8  | 149.2 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A9  | 130.3 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| A10 | 252.6 | SEQ_BLOCK=8,  warps=2, stages=2 | 0 | baseline |
| B1  | 396.4 | SEQ_BLOCK=16, warps=4, stages=2 | 0 | baseline |
| B2  | 6107.9| SEQ_BLOCK=16, warps=4, stages=2 | 0 | baseline |
| B3  | 24183.3| SEQ_BLOCK=16, warps=4, stages=2 | 0 | baseline |
| B4  | 96220.1| SEQ_BLOCK=16, warps=4, stages=2 | 0 | baseline |

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

## Optimization Ideas Backlog

### A.2 Tiling & SEQ_BLOCK \[HIGHEST PRIORITY\]

- \[ \] **SEQ_BLOCK sweep (4,8,16,32,64,128)** — Why: current SEQ_BLOCK=8 creates 128 inner-loop iterations for kv=1024. Larger blocks reduce iteration count and improve L2 reuse within each block. **Impact: High** | All shapes | Correctness risk: No (with masking already correct)
- \[ \] **Shape-conditional SEQ_BLOCK** — Why: short kv (64) benefits from small SEQ_BLOCK (less masking waste); long kv (2048) benefits from large SEQ_BLOCK (fewer iters). **Impact: High** | All shapes | Correctness risk: No
- \[ \] **Separate decode vs prefill SEQ_BLOCK** — already distinct; verify optimal values differ by path

### A.1 Memory Access Patterns

- \[ \] **Reduce head-redundant loads via HEAD_BLOCK tiling** — Why: 32 programs per token each load identical cache data. Restructure grid to `(num_tokens, N_HEADS//HEAD_BLOCK)` and process HEAD_BLOCK heads per program, loading ckv+kpe once per SEQ_BLOCK. **Impact: Very High (up to HEAD_BLOCK×)** | Large kv shapes | Correctness risk: Yes
- \[ \] **Cache load eviction hint `evict_first`** — Why: each cache position is read once per program; hint L2 to evict early to make room for next block. **Impact: Low-Medium** | All shapes | Correctness risk: No
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
