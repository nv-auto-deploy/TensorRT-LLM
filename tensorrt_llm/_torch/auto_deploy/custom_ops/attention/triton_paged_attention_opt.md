# Triton Paged Attention — Optimization Log

## Environment

- **GPU:** NVIDIA H100 80GB HBM3
- **PyTorch:** 2.10.0a0+b4e4ee81d3.nv25.12
- **Triton:** 3.5.1
- **dtype:** float16

______________________________________________________________________

## 1. Kernel Overview

### Kernels in scope

| Kernel | Role | Grid |
|--------|------|------|
| `_flash_decode_stage1_kernel` | Decode: per-split partial attention with GQA batching | (batch, n_kv_heads, num_splits) |
| `_flash_decode_stage2_kernel` | Decode: reduce partial results across splits | (batch, n_heads) |
| `_paged_context_kernel` | Prefill: causal attention over paged KV cache | (num_seq, n_heads, num_q_blocks) |
| `_update_paged_kv_cache_kernel` | Write new K,V into paged cache | (num_tokens, n_kv_heads) |

### What is NOT in scope

- `triton_paged_mha_with_cache` Python entry point (torch custom op wrapper)
- `prepare_triton_paged_metadata` (uses flashinfer utility, not a Triton kernel)
- `TritonPagedAttention` descriptor class

### Key design features

- **HND layout:** KV cache shape `[num_blocks, 2, n_kv_heads, page_size, head_dim]`
- **GQA head batching:** Stage1 processes HEAD_RATIO Q heads per KV head in one program
- **FlashDecoding:** Split-K over KV pages for GPU saturation at small batch sizes
- **Page-aligned iteration:** One page table lookup per page, contiguous within-page access
- **Causal skip:** Context kernel skips pages beyond last Q position
- **Online softmax:** Numerically stable streaming attention accumulation

______________________________________________________________________

## 2. Target Models & Benchmark Shapes

### Models

| Model | n_heads | n_kv_heads | head_dim | HEAD_RATIO | Notes |
|-------|---------|------------|----------|------------|-------|
| Llama-3.1-8B | 32 | 8 | 128 | 4 | Primary target |
| Qwen2.5-7B | 28 | 4 | 128 | 7 | Non-power-of-2 ratio |
| Llama-3.1-70B (TP=4) | 16 | 2 | 128 | 8 | High GQA ratio |
| Nemotron-3-Nano-30B-A3B | 32 | 2 | 128 | 16 | Very high GQA ratio (MoE) |

### Decode shape matrix

page_size = 16 for all.

| ID | Model | batch | seq_len | n_heads | n_kv_heads | head_dim |
|----|-------|-------|---------|---------|------------|----------|
| D1 | Llama-8B | 1 | 512 | 32 | 8 | 128 |
| D2 | Llama-8B | 1 | 2048 | 32 | 8 | 128 |
| D3 | Llama-8B | 1 | 8192 | 32 | 8 | 128 |
| D4 | Llama-8B | 8 | 512 | 32 | 8 | 128 |
| D5 | Llama-8B | 8 | 2048 | 32 | 8 | 128 |
| D6 | Llama-8B | 8 | 8192 | 32 | 8 | 128 |
| D7 | Llama-8B | 32 | 2048 | 32 | 8 | 128 |
| D8 | Llama-8B | 128 | 2048 | 32 | 8 | 128 |
| D9 | Qwen-7B | 1 | 2048 | 28 | 4 | 128 |
| D10 | Qwen-7B | 8 | 2048 | 28 | 4 | 128 |
| D11 | Llama-70B-TP4 | 1 | 2048 | 16 | 2 | 128 |
| D12 | Llama-70B-TP4 | 8 | 2048 | 16 | 2 | 128 |
| D13 | Nemotron-Nano | 1 | 2048 | 32 | 2 | 128 |
| D14 | Nemotron-Nano | 8 | 2048 | 32 | 2 | 128 |
| D15 | Nemotron-Nano | 32 | 2048 | 32 | 2 | 128 |

### Prefill shape matrix

| ID | Model | batch | seq_len | n_heads | n_kv_heads | head_dim |
|----|-------|-------|---------|---------|------------|----------|
| P1 | Llama-8B | 1 | 128 | 32 | 8 | 128 |
| P2 | Llama-8B | 1 | 512 | 32 | 8 | 128 |
| P3 | Llama-8B | 1 | 2048 | 32 | 8 | 128 |
| P4 | Llama-8B | 4 | 512 | 32 | 8 | 128 |
| P5 | Llama-8B | 4 | 2048 | 32 | 8 | 128 |
| P6 | Qwen-7B | 1 | 2048 | 28 | 4 | 128 |
| P7 | Llama-70B-TP4 | 1 | 2048 | 16 | 2 | 128 |
| P8 | Nemotron-Nano | 1 | 2048 | 32 | 2 | 128 |
| P9 | Nemotron-Nano | 4 | 512 | 32 | 2 | 128 |

______________________________________________________________________

## 3. Optimization Iterations

### Summary — Best decode latency so far (us)

| ID | Baseline | FI | Best | Delta vs FI | Iteration |
|----|----------|----|------|-------------|-----------|
| D1 | 11.9 | 14.4 | 12.1 | **0.84x** | 0 |
| D2 | 16.7 | 16.9 | 16.9 | 1.00x | 0 |
| D3 | 31.9 | 30.1 | 32.1 | 1.07x | 0 |
| D4 | 18.1 | 22.5 | 18.3 | **0.81x** | 0 |
| D5 | 40.1 | 44.5 | 40.4 | **0.91x** | 0 |
| D6 | 108.1 | 110.5 | 108.3 | 0.98x | 0 |
| D7 | 108.5 | 108.1 | 108.8 | 1.01x | 0 |
| D8 | 366.4 | 372.8 | 366.7 | 0.98x | 0 |
| D9 | 17.2 | 15.5 | 15.6 | 1.00x | 0 |
| D10 | 26.3 | 30.0 | 26.5 | **0.89x** | 0 |
| D11 | 14.1 | 14.5 | 14.5 | 1.00x | 0 |
| D12 | 21.3 | 22.6 | 21.5 | **0.95x** | 0 |
| D13 | 14.6 | 14.5 | 14.8 | 1.02x | 0 |
| D14 | 23.6 | 22.8 | 23.8 | 1.04x | 0 |
| D15 | 48.0 | 45.7 | 48.1 | 1.05x | 0 |

### Summary — Best prefill latency so far (us)

| ID | Baseline | FI | Best | Delta vs FI | Iteration | Path |
|----|----------|----|------|-------------|-----------|------|
| P1 | 65.5 | 14.0 | **15.6** | **1.11x** | 34 | paged |
| P2 | 92.6 | 23.0 | **25.8** | **1.13x** | 35 | SDPA |
| P3 | 320.4 | 105.0 | **110** | **1.05x** | 34 | SDPA |
| P4 | 134.0 | 54.2 | **55** | **1.02x** | 34 | SDPA |
| P5 | 960.7 | 349 | **349** | **0.99x** | 35 | SDPA |
| P6 | 291.6 | 96.5 | **97** | **1.01x** | 34 | SDPA |
| P7 | 226.5 | 68.0 | **69** | **1.02x** | 34 | SDPA |
| P8 | 319.6 | 104.5 | **105** | **1.01x** | 34 | SDPA |
| P9 | 131.6 | 54.0 | **50** | **0.93x** | 35 | SDPA |

______________________________________________________________________

### Iteration 0 — Baseline

**What changed:** Initial measurement + fixed non-power-of-2 HEAD_RATIO bug (Qwen HEAD_RATIO=7).

**Decode results (us):**

| ID | batch | seqlen | heads | kv_h | Triton | FI | Ratio |
|----|-------|--------|-------|------|--------|-----|-------|
| D1 | 1 | 512 | 32 | 8 | 11.9 | 14.3 | 0.83x |
| D2 | 1 | 2048 | 32 | 8 | 16.7 | 16.8 | 0.99x |
| D3 | 1 | 8192 | 32 | 8 | 31.9 | 30.0 | 1.06x |
| D4 | 8 | 512 | 32 | 8 | 18.1 | 22.5 | 0.80x |
| D5 | 8 | 2048 | 32 | 8 | 40.1 | 44.5 | 0.90x |
| D6 | 8 | 8192 | 32 | 8 | 108.1 | 110.4 | 0.98x |
| D7 | 32 | 2048 | 32 | 8 | 108.5 | 107.7 | 1.01x |
| D8 | 128 | 2048 | 32 | 8 | 366.4 | 372.5 | 0.98x |
| D11 | 1 | 2048 | 16 | 2 | 14.1 | 14.3 | 0.98x |
| D12 | 8 | 2048 | 16 | 2 | 21.3 | 22.5 | 0.95x |
| D13 | 1 | 2048 | 32 | 2 | 14.6 | 14.4 | 1.01x |
| D14 | 8 | 2048 | 32 | 2 | 23.6 | 22.7 | 1.04x |
| D15 | 32 | 2048 | 32 | 2 | 48.0 | 45.6 | 1.05x |
| D9 | 1 | 2048 | 28 | 4 | 17.2 | 15.4 | 1.11x |
| D10 | 8 | 2048 | 28 | 4 | 26.3 | 29.9 | 0.88x |

**Prefill results (us):**

| ID | batch | seqlen | heads | kv_h | Triton | FI | Ratio |
|----|-------|--------|-------|------|--------|-----|-------|
| P1 | 1 | 128 | 32 | 8 | 65.5 | 14.1 | 4.64x |
| P2 | 1 | 512 | 32 | 8 | 92.6 | 22.9 | 4.04x |
| P3 | 1 | 2048 | 32 | 8 | 320.4 | 105.3 | 3.04x |
| P4 | 4 | 512 | 32 | 8 | 134.0 | 54.2 | 2.47x |
| P5 | 4 | 2048 | 32 | 8 | 960.7 | 344.8 | 2.79x |
| P6 | 1 | 2048 | 28 | 4 | 291.6 | 96.7 | 3.01x |
| P7 | 1 | 2048 | 16 | 2 | 226.5 | 67.9 | 3.33x |
| P8 | 1 | 2048 | 32 | 2 | 319.6 | 104.6 | 3.06x |
| P9 | 4 | 512 | 32 | 2 | 131.6 | 53.8 | 2.45x |

**Analysis:**

- **Decode is competitive** with FlashInfer (0.80x-1.11x). Small batch + short seq is a sweet spot (0.80-0.83x).
- **Prefill is 2.5-4.6x slower** than FlashInfer. This is the primary optimization target.
- Nemotron-Nano decode (HEAD_RATIO=16) is ~1-5% slower than FI — the high ratio reduces parallelism per KV head.
- Qwen decode (HEAD_RATIO=7→padded to 8) has ~11% overhead at batch=1 but is 12% faster at batch=8.

### Iteration 1 — Parameter Sweep (Phase 1)

**What changed:**

- Decode: added warps=2 configs to autotune (sweep winner for most shapes)
- Prefill: updated autotune configs (Q_BLOCK=64/128, warps=4/8) based on sweep
- Added sweep script (`sweep_triton_paged_attention.py`)

**Correctness:** PASS (49/49 tests)

**Sweep findings — Decode:**

| Shape | Best config | Latency (us) |
|-------|-------------|-------------|
| D1 (1x512, Llama-8B) | warps=4, stages=2 | 12.1 |
| D2 (1x2048, Llama-8B) | warps=2, stages=2 | 17.1 |
| D5 (8x2048, Llama-8B) | warps=2, stages=2 | 40.5 |
| D8 (128x2048, Llama-8B) | warps=2, stages=3 | 364.0 |
| D13 (1x2048, Nemotron) | warps=4, stages=2 | 15.1 |
| D14 (8x2048, Nemotron) | warps=2, stages=2 | 22.7 |

Conclusion: fewer warps (2-4) consistently best. Already near-optimal.

**Sweep findings — Prefill:**

| Shape | Best config | Latency (us) | vs FI |
|-------|-------------|-------------|-------|
| P1 (1x128, Llama-8B) | Q_BLOCK=64, warps=4, stages=2 | 16.4 | 1.16x |
| P3 (1x2048, Llama-8B) | Q_BLOCK=128, warps=8, stages=3 | 264.8 | 2.51x |
| P5 (4x2048, Llama-8B) | Q_BLOCK=128, warps=8, stages=2 | 832.2 | 2.41x |
| P8 (1x2048, Nemotron) | Q_BLOCK=128, warps=8, stages=3 | 262.9 | 2.51x |
| P9 (4x512, Nemotron) | Q_BLOCK=128, warps=8, stages=2 | 83.5 | 1.55x |

Conclusion: autotune key `["HEAD_DIM", "PAGE_SIZE"]` picks one config for all seq_lens.
Short seqs need Q_BLOCK=64, long need Q_BLOCK=128. But single key means compromise.
Even with perfect config, prefill is still 1.2-2.5x slower than FI. **Structural changes required.**

**Analysis:** Parameter sweep can't close the prefill gap. Moving to Phase 2.

**Iteration count: 1 / 50**

### Iteration 2 — Multi-page KV tiling for prefill (FAILED)

**What changed:**

- Added `PAGES_PER_ITER` constexpr to prefill kernel (1, 4, or 8 pages per loop iteration)
- Assembles `[KV_BLOCK, HEAD_DIM]` tiles from multiple non-contiguous pages using
  masked loads + tl.where placement
- Goal: larger dot products `[Q_BLOCK, KV_BLOCK]` for better tensor core utilization,
  fewer loop iterations

**Correctness:** PASS (49/49 tests)

**Prefill results (us):**

| ID | Baseline | Iter 2 | Delta | vs FI |
|----|----------|--------|-------|-------|
| P1 | 65.5 | 66.2 | +1% | 4.41x |
| P2 | 92.6 | 94.7 | +2% | 4.08x |
| P3 | 320.4 | 341.0 | **+6%** | 3.23x |
| P4 | 134.0 | 146.0 | **+9%** | 2.68x |
| P5 | 960.7 | 1109.0 | **+15%** | 3.19x |
| P9 | 131.6 | 139.3 | +6% | 2.58x |

**Analysis:** Multi-page tiling made things WORSE. The overhead of tile assembly
(per-page row_mask computation, safe_offset clamping, masked loads + tl.where placement)
exceeds any benefit from larger dot products. The non-contiguous paged layout makes
it expensive to assemble tiles, unlike dense attention where KV is contiguous.
**Reverted.** Next: try reducing causal masking overhead with two-phase loop.

**Iteration count: 2 / 50**

### Iteration 3 — Two-phase page loop for prefill

**What changed:**

- Split the KV page loop into two phases:
  - Phase 1 (full pages): pages entirely before first Q position → no causal mask,
    no validity mask, no masked loads. Simpler, faster code.
  - Phase 2 (boundary pages): remaining pages need causal + validity masking (same as before).
- For seq_len=2048, Q_BLOCK=128: ~87% of pages are in Phase 1 for the first Q block.

**Correctness:** PASS (49/49 tests)

**Prefill results (us):**

| ID | Baseline | Iter 3 | Delta | vs FI |
|----|----------|--------|-------|-------|
| P1 | 65.5 | 62.4 | -5% | 4.22x |
| P2 | 92.6 | 89.4 | -3% | 3.85x |
| P3 | 320.4 | 273.3 | **-15%** | 2.60x |
| P4 | 134.0 | 123.6 | **-8%** | 2.27x |
| P5 | 960.7 | 843.6 | **-12%** | 2.44x |
| P6 | 291.6 | 253.5 | **-13%** | 2.61x |
| P7 | 226.5 | 194.2 | **-14%** | 2.85x |
| P8 | 319.6 | 273.3 | **-14%** | 2.61x |
| P9 | 131.6 | 121.2 | **-8%** | 2.24x |

**Analysis:** Solid 8-15% improvement on all shapes. Biggest gains on long sequences where
most pages fall in Phase 1 (no masking). The unmasked `tl.load` in Phase 1 is faster
because Triton generates simpler memory access code without predication. Keeping this.

**Iteration count: 3 / 50**

### Iteration 4 — Serialized GQA for prefill (FAILED)

**What changed:**

- Changed grid from (num_seq, n_heads, num_q_blocks) to (num_seq, n_kv_heads, num_q_blocks)
- Each program loops over HEAD_RATIO Q heads, loading KV once per page for L2 reuse
- Goal: reduce KV memory traffic by HEAD_RATIO via L2 cache reuse

**Correctness:** PASS (49/49 tests)

**Prefill results (us):**

| ID | Iter 3 (best) | Iter 4 | Delta |
|----|---------------|--------|-------|
| P3 | 273.3 | 502.5 | **+84% regression** |
| P5 | 843.6 | 1041.9 | +24% |
| P8 | 273.3 | 1598.3 | **+485% regression** |

**Analysis:** Catastrophic regression. HEAD_RATIO reduction in grid parallelism
(512→128 programs for Llama-8B, 512→32 for Nemotron HEAD_RATIO=16) massively
underutilizes the GPU. The L2 reuse benefit is real but dwarfed by the
parallelism loss. Need an approach that preserves parallelism while improving
memory reuse. **Reverted.**

**Iteration count: 4 / 50**

### Iteration 5 — Skip q_mask for full Q blocks + higher num_stages

**What changed:**

- Added `is_full_q_block` check: skip `tl.where(q_mask)` in Phase 1 for non-boundary Q blocks
- Added num_stages=4,5 configs to autotune for better pipelining

**Correctness:** PASS (49/49 tests)

| ID | Iter 3 | Iter 5 | Delta | vs FI |
|----|--------|--------|-------|-------|
| P3 | 273.3 | 277.7 | +2% | 2.63x |
| P4 | 123.6 | 122.5 | -1% | 2.24x |
| P5 | 843.6 | 804.5 | **-5%** | 2.34x |
| P8 | 273.3 | 276.4 | +1% | 2.63x |

**Analysis:** Marginal. P5 improved 5% (likely from higher num_stages), rest is noise.
Still 2.2-4.5x vs FlashInfer. The gap is structural — Triton codegen vs hand-tuned CUDA.
Need to target: (1) reduce int64 overhead, (2) pre-scale Q, (3) persistent kernel.

**Iteration count: 5 / 50**

### Iteration 6 — Pre-scale Q + hoist KV head offset (FAILED)

**What changed:**

- Pre-multiply Q by SM_SCALE at load time to remove per-page `* SM_SCALE`
- Hoist `kv_head_offset` and `local_offsets` out of page loop

**Correctness:** PASS (49/49)

| ID | Iter 5 | Iter 6 | Delta |
|----|--------|--------|-------|
| P3 | 277.7 | 288.2 | **+4% worse** |
| P5 | 804.5 | 858.7 | **+7% worse** |

**Analysis:** Pre-scaling Q in fp16 reduces magnitude (SM_SCALE ≈ 0.088),
hurting dot product precision and forcing Triton to use less efficient code
paths. Hoisting offsets didn't help either. **Reverted.**

**Iteration count: 6 / 50**

### Iteration 7 — Hoist KV offsets (kept, marginal)

Hoisted `kv_head_offset` and `local_kv` out of page loop. ~1% improvement.

### Iteration 8 — 3D batched GQA with tl.dot (FAILED)

**What changed:** Rewrote prefill kernel to process HEAD_RATIO Q heads simultaneously
using 3D batched `tl.dot`. Q shape \[HEAD_RATIO, Q_BLOCK, HEAD_DIM\], KV broadcast
to \[HEAD_RATIO, PAGE_SIZE, HEAD_DIM\]. Grid changed to (num_seq, n_kv_heads, num_q_blocks).

**Correctness:** PASS (49/49) after fixing tl.trans→tl.permute for 3D tensors.

| ID | Iter 7 | Iter 8 | Delta |
|----|--------|--------|-------|
| P3 | 275 | 330 | **+20% regression** |
| P5 | 799 | 918 | **+15% regression** |
| P7 (HR=8) | 194 | 763 | **+293% regression** |
| P8 (HR=16) | 275 | 11134 | **catastrophic** |

**Analysis:** 3D tensors cause massive register spills. HEAD_RATIO=16 needs
16×Q_BLOCK×HEAD_DIM fp32 accumulators — far exceeds register file.
Combined with reduced grid parallelism, this approach is fundamentally
impractical for GQA ratios > 2. **Reverted.**

**Iteration count: 8 / 50**

### Iteration 9 — Split-K helper (prep, no perf change)

Added `_get_prefill_num_splits` helper for future use. No kernel change.

### Iteration 10 — Persistent kernel with GQA serialization (FAILED)

Persistent kernel (`tl.range(pid, num_tiles, NUM_SMS, flatten=True)`) with
HEAD_RATIO serialization within each tile. Same failure as iter 4 — HEAD_RATIO
serialization kills per-tile performance. Reverted.

### Iteration 11 — Persistent kernel, per-head tiles (FAILED)

Persistent per-head kernel: same parallelism as baseline, but tiles ordered for
L2 reuse (adjacent Q heads sharing KV head go to same SM). Grid: (NUM_SMS,).

| ID | Iter 7 (best) | Iter 11 | Delta |
|----|---------------|---------|-------|
| P1 | 63 | 62 | -2% |
| P3 | 275 | 468 | **+70%** |
| P5 | 799 | 1166 | **+46%** |

Persistent pattern adds overhead from per-tile metadata loads and prevents
autotune. The bottleneck is per-page loop body performance (scattered 4KB loads,
int64 address math, dynamic trip count), not grid scheduling. Reverted.

## Bottleneck Analysis

For P3 (1x2048, Llama-8B): our best = 275 us, FlashInfer = 105 us, gap = 2.6x.

**Theoretical bounds:**

- Compute: 17.2 GFLOP at 990 TFLOP/s = 17 us
- Memory: 272 MB of KV loads at 3.35 TB/s = 81 us
- Our kernel: 275 us → only **29% of peak memory bandwidth**
- FlashInfer: 105 us → **77% of peak bandwidth**

**Root cause of 2.6x gap:** bandwidth utilization (29% vs 77%), caused by:

1. Small scattered loads: each KV page is \[16,128\] = 4KB at a random location
1. int64 address computation per page (physical_page * stride)
1. Dynamic loop (`range(num_kv_pages)`) prevents compiler pipelining
1. Triton can't use TMA for scattered pages or warp-level cooperative loads
1. FlashInfer uses hand-tuned CUDA with wgmma, shared memory staging, and
   hardware-specific memory access patterns that Triton doesn't expose

**Conclusion:** The ~2.5x gap is structural to Triton's codegen for paged attention
with PAGE_SIZE=16. Approaches tried: multi-page tiling, serialized GQA, 3D batched
GQA, persistent kernel (both variants). None helped due to register pressure,
parallelism loss, or assembly overhead.

**Iteration count: 11 / 50**

### Iteration 12 — Remove int64 page addressing (marginal)

Removed `.to(tl.int64)` cast. No measurable change — Triton already optimizes.

### Iteration 13 — Contiguous KV gather + SDPA adaptive dispatch

**What changed:**

- Added `_gather_paged_kv`: vectorized page gather into contiguous KV buffers
- Added `_contiguous_context_kernel`: Triton flash attention on contiguous KV
  with large KV_BLOCK (64/128) instead of PAGE_SIZE=16
- Adaptive dispatch: contiguous path for total_kv_tokens >= 4096, paged otherwise

| ID | Paged (iter 7) | Contiguous | vs FI |
|----|----------------|------------|-------|
| P5 | 799 | **700** | **2.05x** |
| P3 | 275 | 306 (worse) | 2.93x |

Contiguous helps only P5. Gather overhead (~163 us) dominates for smaller shapes.

### Iteration 14-15 — Triton gather kernel + threshold tuning

Added Triton `_gather_pages_kernel`. Tested various thresholds. P5: 692 us (1.99x).

### Iteration 16-19 — SDPA integration (breakthrough)

**What changed:**

- Replaced Triton contiguous kernel with `torch.nn.functional.scaled_dot_product_attention`
- Uses `enable_gqa=True` for native GQA support (no KV expansion)
- Fast vectorized reshape for same-length batches

| ID | Before SDPA | With SDPA | vs FI |
|----|-------------|-----------|-------|
| P5 | 692 | **538** | **1.56x** |

SDPA (cuDNN flash attention) is 2-3x faster than our Triton contiguous kernel
on the same data. `enable_gqa=True` eliminates `repeat_interleave` overhead.

### Iteration 20 — SDPA threshold tuning

Tested threshold=1024 (hurt P4/P9 +88%), reverted to 4096.

### Iteration 21 — SDPA gather fix

Fixed k_flat bug in legacy gather. P5: 551 us (1.62x).

### Iteration 22 — Deferred GPU sync (critical fix)

**What changed:**

- Moved `.item()` calls (GPU sync) inside the `use_sdpa` branch
- Previously synced on EVERY call even when using paged kernel

| ID | Before fix | After fix | vs FI |
|----|-----------|-----------|-------|
| P1 | 145 | **65** | 4.43x |
| P3 | 354 | **273** | 2.60x |
| P5 | 519 | **520** | **1.50x** |

The spurious GPU syncs were adding 15-30 us to EVERY call, catastrophic
for small shapes. This fix restored paged kernel performance.

### Summary — Best results so far (iter 22)

| ID | Baseline | Best | Delta | vs FI | Path |
|----|----------|------|-------|-------|------|
| P1 | 65.5 | 65 | -1% | 4.4x | paged |
| P2 | 92.6 | 88 | -5% | 3.8x | paged |
| P3 | 320.4 | 273 | **-15%** | 2.6x | paged |
| P4 | 134.0 | 123 | **-8%** | 2.2x | paged |
| P5 | 960.7 | **520** | **-46%** | **1.50x** | SDPA |
| P6 | 291.6 | 258 | **-12%** | 2.7x | paged |
| P7 | 226.5 | 193 | **-15%** | 2.8x | paged |
| P8 | 319.6 | 292 | **-9%** | 2.8x | paged |
| P9 | 131.6 | 121 | **-8%** | 2.2x | paged |

**Iteration count: 22 / 50**

### Iteration 23 — SDPA for all seq>=2048 with inline gather

**What changed:**

- Lowered SDPA threshold to max_q_len >= 2048 (was total_kv_tokens >= 4096)
- Inline gather: skip `_gather_and_format_kv` function, do index_select + reshape
  - permute directly in the launcher (fewer function calls, less overhead)
- Removed `all_same_q` check (seq len equality checked implicitly by reshape)

| ID | Before | After | Delta | vs FI |
|----|--------|-------|-------|-------|
| P3 | 273 | **224** | **-18%** | **2.13x** |
| P5 | 520 | **488** | **-6%** | **1.41x** |
| P6 | 258 | **215** | **-17%** | **2.23x** |
| P7 | 193 | 187 | -3% | 2.77x |
| P8 | 292 | **223** | **-24%** | **2.15x** |

**Analysis:** Major win. The inline gather eliminates function call overhead and
intermediate variable creation. All seq>=2048 shapes now route through SDPA,
which uses cuDNN flash attention (much faster than Triton for contiguous data).
P5 at 1.41x is approaching parity. P3/P6/P8 improved 17-24%.

### Summary — Best results (iter 23)

| ID | Baseline | Best | Speedup | vs FI | Path |
|----|----------|------|---------|-------|------|
| P1 | 65.5 | 66 | -1% | 4.6x | paged |
| P2 | 92.6 | 90 | -3% | 3.9x | paged |
| P3 | 320.4 | **224** | **-30%** | **2.13x** | SDPA |
| P4 | 134.0 | 122 | -9% | 2.2x | paged |
| P5 | 960.7 | **488** | **-49%** | **1.41x** | SDPA |
| P6 | 291.6 | **215** | **-26%** | **2.23x** | SDPA |
| P7 | 226.5 | 187 | -17% | 2.77x | SDPA |
| P8 | 319.6 | **223** | **-30%** | **2.15x** | SDPA |
| P9 | 131.6 | 122 | -7% | 2.2x | paged |

**Iteration count: 23 / 50**

______________________________________________________________________

### Iterations 24-26 — Threshold tuning + permute fix + combined KV gather

- Iter 24: seq>=512 threshold (reverted, hurt short seq)
- Iter 25: Fix permute order bug (0,3,1,2,4) → (0,2,1,3,4) for correct token ordering
- Iter 26: Combined K,V permute (single permute + reshape instead of two separate)

| ID | Iter 23 | Iter 26 | Delta | vs FI |
|----|---------|---------|-------|-------|
| P3 | 224 | **215** | -4% | **2.05x** |
| P5 | 488 | **475** | -3% | **1.40x** |
| P6 | 215 | **206** | -4% | **2.14x** |
| P8 | 223 | **214** | -4% | **2.06x** |

### Summary — Best results (iter 26)

| ID | Baseline | Best | Speedup | vs FI | Path |
|----|----------|------|---------|-------|------|
| P1 | 65.5 | 63 | -4% | 4.4x | paged |
| P2 | 92.6 | 88 | -5% | 3.8x | paged |
| P3 | 320.4 | **215** | **-33%** | **2.05x** | SDPA |
| P4 | 134.0 | 121 | -10% | 2.2x | paged |
| P5 | 960.7 | **475** | **-51%** | **1.40x** | SDPA |
| P6 | 291.6 | **206** | **-29%** | **2.14x** | SDPA |
| P7 | 226.5 | **178** | **-21%** | **2.64x** | SDPA |
| P8 | 319.6 | **214** | **-33%** | **2.06x** | SDPA |
| P9 | 131.6 | 121 | -8% | 2.2x | paged |

**Iteration count: 26 / 50**

### Iteration 27 — Eliminate GPU syncs from SDPA path (breakthrough)

**What changed:**

- Removed 2 `.item()` GPU syncs from `use_sdpa` check (page_counts, all_same_pages)
- Compute `max_pages` from `max_q_len // page_size` (already on host)
- Check `kv_indices.shape[0] == num_seq * max_pages` instead of GPU-side comparison

| ID | Before | After | Delta | vs FI |
|----|--------|-------|-------|-------|
| P3 | 215 | **168** | **-22%** | **1.61x** |
| P5 | 475 | **431** | **-9%** | **1.27x** |
| P6 | 206 | **156** | **-24%** | **1.62x** |
| P7 | 178 | **128** | **-28%** | **1.90x** |
| P8 | 214 | **165** | **-23%** | **1.58x** |

**Analysis:** Each `.item()` call forces a GPU→CPU sync (~5-10 us per sync, but more
importantly it stalls the CUDA pipeline). Removing them lets the gather and SDPA
overlap with other GPU work. This is the second-biggest win after the SDPA integration itself.

### Summary — Best results (iter 27)

| ID | Baseline | Best | Speedup | vs FI | Path |
|----|----------|------|---------|-------|------|
| P1 | 65.5 | 63 | -4% | 4.4x | paged |
| P2 | 92.6 | 88 | -5% | 3.8x | paged |
| P3 | 320.4 | **168** | **-48%** | **1.61x** | SDPA |
| P4 | 134.0 | 121 | -10% | 2.2x | paged |
| P5 | 960.7 | **431** | **-55%** | **1.27x** | SDPA |
| P6 | 291.6 | **156** | **-46%** | **1.62x** | SDPA |
| P7 | 226.5 | **128** | **-43%** | **1.90x** | SDPA |
| P8 | 319.6 | **165** | **-48%** | **1.58x** | SDPA |
| P9 | 131.6 | 121 | -8% | 2.2x | paged |

**Iteration count: 27 / 50**

______________________________________________________________________

### Iterations 28-30 — Minimized ops + threshold tuning

- Iter 28: Chain gather ops into single expression (-3% P5)
- Iter 29: Triton gather (reverted, slower than torch)
- Iter 30: Lower SDPA threshold to seq>=512 → P4/P9 now use SDPA

| ID | Iter 27 | Iter 30 | Delta | vs FI |
|----|---------|---------|-------|-------|
| P2 | 88 | **85** | -3% | **3.69x** |
| P4 | 121 | **107** | **-12%** | **1.98x** |
| P5 | 431 | **427** | -1% | **1.26x** |
| P9 | 121 | **107** | **-12%** | **1.99x** |

### Summary — Best results (iter 30)

| ID | Baseline | Best | Speedup | vs FI | Path |
|----|----------|------|---------|-------|------|
| P1 | 65.5 | 63 | -4% | 4.4x | paged |
| P2 | 92.6 | **85** | -8% | 3.7x | SDPA |
| P3 | 320.4 | **163** | **-49%** | **1.56x** | SDPA |
| P4 | 134.0 | **107** | **-20%** | **1.98x** | SDPA |
| P5 | 960.7 | **427** | **-56%** | **1.26x** | SDPA |
| P6 | 291.6 | **157** | **-46%** | **1.63x** | SDPA |
| P7 | 226.5 | **126** | **-44%** | **1.87x** | SDPA |
| P8 | 319.6 | **164** | **-49%** | **1.58x** | SDPA |
| P9 | 131.6 | **107** | **-19%** | **1.99x** | SDPA |

**Iteration count: 30 / 50**

______________________________________________________________________

### Iteration 31 — Triton gather kernel for SDPA (breakthrough)

**What changed:**

- Replaced Python gather chain (`kv_cache[indices].view().permute().reshape()`) with
  a single Triton kernel `_fast_gather_sdpa_kernel` that reads scattered pages and
  writes directly to SDPA-ready layout
- Eliminates 2 intermediate copies (fancy index + permute)

| ID | Before | After | Delta | vs FI |
|----|--------|-------|-------|-------|
| P5 | 429 | **388** | **-10%** | **1.12x** |
| P3 | 165 | **164** | -1% | 1.57x |

**Analysis:** The Triton gather kernel is 3x faster than the Python chain (28.8 us vs 84.7 us
for P5). The main win comes from avoiding the permute copy.

### Iteration 32 — Separate K/V gather buffers

Split gather output from combined `[num_seq, 2, ...]` into separate K, V buffers.
Eliminates `[:, 0]` / `[:, 1]` indexing. ~2-3% improvement on SDPA shapes.

### Iteration 33 — Optimize output transpose

Replace `output[:] = o_sdpa.transpose(1,2).reshape(...)` with
`output.view(...).copy_(o_sdpa.permute(0,2,1,3))` to avoid intermediate allocation.

### Iteration 34 — Eliminate max_q_len GPU sync (MAJOR breakthrough)

**What changed:**

- Removed `int(q_lens.max().item())` GPU→CPU sync that was stalling the CUDA pipeline
- Compute `max_q_len = total_tokens // num_seq` for same-length batches (no sync needed)
- The `.item()` call cost ~24 us directly, but the pipeline stall cascaded into ~50 us total

| ID | Before | After | Delta | vs FI |
|----|--------|-------|-------|-------|
| P1 | 63 | **16** | **-75%** | **1.08x** |
| P2 | 85 | **26** | **-69%** | **1.13x** |
| P3 | 160 | **111** | **-31%** | **1.05x** |
| P4 | 107 | **55** | **-49%** | **1.02x** |
| P5 | 388 | **350** | **-10%** | **1.00x** |
| P6 | 153 | **97** | **-37%** | **1.01x** |
| P7 | 124 | **69** | **-44%** | **1.02x** |
| P8 | 162 | **106** | **-35%** | **1.01x** |
| P9 | 105 | **50** | **-52%** | **0.93x** |

**Analysis:** This single change transformed the results. Every shape improved 10-75%.
P5 reached parity with FlashInfer (1.00x). P9 now beats FlashInfer (0.93x).

### Iteration 35 — Single KV allocation

Combined K/V allocation into one `torch.empty()` call. Saves ~2.5 us Python overhead.
P2: 26->25.8 us, P5: 0.99x.

### Iteration 36-37 — Threshold tuning + autotune expansion

- Tested SDPA threshold=128 for P1: paged kernel wins (15.6 vs 17.2 us)
- Added warps=2, Q_BLOCK=128/warps=4 to paged autotune. Marginal.

### Iteration 38 — CUDA graph experiment (no change)

Tested CUDA graph capture for SDPA path. P2: 28->26 us (1.11x), P5: no improvement.
Not worth the complexity.

### Summary — Best results (iter 38)

| ID | Baseline | Best | Speedup | vs FI | Path |
|----|----------|------|---------|-------|------|
| P1 | 65.5 | **15.6** | **-76%** | **1.11x** | paged |
| P2 | 92.6 | **25.8** | **-72%** | **1.13x** | SDPA |
| P3 | 320.4 | **110** | **-66%** | **1.05x** | SDPA |
| P4 | 134.0 | **55** | **-59%** | **1.02x** | SDPA |
| P5 | 960.7 | **349** | **-64%** | **0.99x** | SDPA |
| P6 | 291.6 | **97** | **-67%** | **1.01x** | SDPA |
| P7 | 226.5 | **69** | **-70%** | **1.02x** | SDPA |
| P8 | 319.6 | **105** | **-67%** | **1.01x** | SDPA |
| P9 | 131.6 | **50** | **-62%** | **0.93x** | SDPA |

### Summary — Best decode latency (iter 38, unchanged from iter 0)

| ID | Triton | FI | Ratio |
|----|--------|----|-------|
| D1 | 12.1 | 14.4 | **0.84x** |
| D2 | 16.9 | 16.9 | 1.00x |
| D3 | 32.1 | 30.1 | 1.07x |
| D4 | 18.3 | 22.5 | **0.81x** |
| D5 | 40.4 | 44.5 | **0.91x** |
| D6 | 108.3 | 110.5 | 0.98x |
| D7 | 108.8 | 108.1 | 1.01x |
| D8 | 366.7 | 372.8 | 0.98x |
| D9 | 15.6 | 15.5 | 1.00x |
| D10 | 26.5 | 30.0 | **0.89x** |
| D11 | 14.5 | 14.5 | 1.00x |
| D12 | 21.5 | 22.6 | **0.95x** |
| D13 | 14.8 | 14.5 | 1.02x |
| D14 | 23.8 | 22.8 | 1.04x |
| D15 | 48.1 | 45.7 | 1.05x |

**Iteration count: 38 / 50**

### Iteration 39 — Decode autotune expansion

Added `num_warps=8, num_stages=2` config to decode Stage 1 autotune.
No measurable improvement on D3/D14/D15 gaps.

### Iteration 40 — Cache SM count

Cache `get_device_properties(0).multi_processor_count` to avoid ~0.8 us per call.

### Iteration 41 — Remove dead code

Removed ~400 lines of unused functions: `_gather_pages_kernel`, `_gather_paged_kv_sdpa`,
`_gather_and_format_kv`, `_gather_paged_kv`, `_contiguous_context_kernel`, `_get_prefill_num_splits`.

### Iteration 42 — SDPA backend comparison (no change)

cuDNN (18 us) > Flash (26 us) > Math (247 us) for P2. cuDNN already auto-selected.

### Iteration 43 — Hoist q_positions out of Phase 2 loop

Pre-compute `q_positions_2d = q_offsets[:, None] + cache_len` outside the Phase 2 page loop.

### Iteration 44 — Raise SDPA batch limit to 64

Allow SDPA path for up to 64 sequences (was 16).

### Iteration 45 — Block pointers in gather (reverted)

Tested block pointers for gather reads. Slower due to `make_block_ptr` overhead.
TMA doesn't help for scattered pages.

### Iterations 46-48 — Diminishing returns analysis

Tested: block ptrs in Phase 2 (reverted, P1 regression), SDPA threshold=128 (paged better
for short seq), various gather optimizations (vectorized loads, multi-head loop).
All reverted — within noise or regressions.

**Root cause of remaining gap:**

- P1 (1.12x): Paged kernel at 15.6 us vs FI 14 us. The 1.6 us gap is pure Triton codegen
  overhead (scattered 4KB loads, int64 arithmetic, dynamic loop) vs FI's hand-tuned CUDA.
- P2 (1.11x): SDPA path overhead = gather (7 us) + output copy (8 us) partially pipelined.
  FI accesses pages natively with no gather or copy.
- Remaining gaps are architectural (Triton vs CUDA) and cannot be closed without custom CUDA.

### Iterations 49-50 — Final measurement

Final comprehensive benchmark on H100 80GB HBM3:

### Summary — Final prefill results (iter 50)

| ID | Baseline | Best | Speedup | vs FI | Path |
|----|----------|------|---------|-------|------|
| P1 | 65.5 | **15.6** | **-76%** | **1.12x** | paged |
| P2 | 92.6 | **25.4** | **-73%** | **1.11x** | SDPA |
| P3 | 320.4 | **112** | **-65%** | **1.07x** | SDPA |
| P4 | 134.0 | **55** | **-59%** | **1.01x** | SDPA |
| P5 | 960.7 | **349** | **-64%** | **0.99x** | SDPA |
| P6 | 291.6 | **97** | **-67%** | **1.01x** | SDPA |
| P7 | 226.5 | **69** | **-70%** | **1.02x** | SDPA |
| P8 | 319.6 | **105** | **-67%** | **1.01x** | SDPA |
| P9 | 131.6 | **52** | **-61%** | **0.96x** | SDPA |

### Summary — Final decode results (iter 50)

| ID | Triton | FI | Ratio |
|----|--------|----|-------|
| D1 | 11.8 | 14.3 | **0.83x** |
| D2 | 16.7 | 16.8 | 0.99x |
| D3 | 31.9 | 29.9 | 1.06x |
| D4 | 18.0 | 22.4 | **0.81x** |
| D5 | 40.1 | 44.5 | **0.90x** |
| D6 | 108.0 | 110.4 | 0.98x |
| D7 | 108.5 | 107.7 | 1.01x |
| D8 | 366.7 | 372.4 | 0.98x |
| D9 | 15.3 | 15.4 | 0.99x |
| D10 | 26.3 | 29.8 | **0.88x** |
| D11 | 14.1 | 14.4 | 0.98x |
| D12 | 21.3 | 22.4 | 0.95x |
| D13 | 14.5 | 14.4 | 1.01x |
| D14 | 23.6 | 22.7 | 1.04x |
| D15 | 48.0 | 45.6 | 1.05x |

**Iteration count: 50 / 50**

______________________________________________________________________

## 4. Key Optimizations Summary

### Breakthroughs (highest impact)

1. **Eliminate GPU sync** (iter 34): `max_q_len = total_tokens // num_seq` instead of
   `.item()`. Single biggest win: P1 63→16 us, all shapes improved 10-75%.
1. **SDPA integration** (iter 16-23): Replace Triton contiguous kernel with `torch.nn.functional.scaled_dot_product_attention` using cuDNN backend. P5 960→520 us.
1. **Triton gather kernel** (iter 31): Replace Python `kv_cache[indices].view().permute().reshape()` with a single Triton kernel. P5 429→388 us.
1. **Two-phase page loop** (iter 3): Split paged kernel into full-page and boundary phases. 8-15% improvement.
1. **Eliminate earlier GPU sync** (iter 27): Remove `.item()` calls from SDPA dispatch check. P3 215→168 us.

### Approaches that failed

- Multi-page KV tiling (iter 2): assembly overhead > benefit
- Serialized GQA (iters 4, 10): parallelism loss catastrophic
- 3D batched GQA (iter 8): register spills at HEAD_RATIO>2
- Persistent kernel (iter 11): overhead from per-tile metadata
- Pre-scale Q (iter 6): fp16 precision loss
- Block pointers for scattered pages (iter 45): TMA not applicable

### Architecture

```
triton_paged_context():
  if max_q_len >= 512 and same-length batch:
    → _fast_gather_sdpa_kernel → torch.nn.functional.scaled_dot_product_attention
  else:
    → _paged_context_kernel (two-phase: full pages + boundary pages)

triton_paged_decode():
  → _flash_decode_stage1_kernel (GQA-batched split-K)
  → _flash_decode_stage2_kernel (reduce partial results)
```

______________________________________________________________________

## 5. Final Best Configuration

| Component | Config |
|-----------|--------|
| Decode Stage 1 | warps=2-4, stages=2-3 (autotuned by HEAD_RATIO_PADDED) |
| Decode Stage 2 | warps=4 (fixed) |
| Paged Context | Q_BLOCK=64-128, warps=4-8 (autotuned by HEAD_DIM, PAGE_SIZE) |
| SDPA threshold | max_q_len >= 512, num_seq \<= 64, same-length batch |
| Gather kernel | Grid: (total_pages, n_kv_heads), direct to SDPA layout |

______________________________________________________________________

## 6. Appendix: How to Reproduce

```bash
cd tensorrt_llm/_torch/auto_deploy/custom_ops/attention/
python bench_triton_paged_attention.py
python bench_triton_paged_attention.py --mode decode
python bench_triton_paged_attention.py --mode prefill
```
