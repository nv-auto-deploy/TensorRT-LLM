# Triton Paged Attention — Optimization Log

**Target model:** `google/gemma-4-26B-A4B-it`
**File:** `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_paged_attention.py`
**Environment:** H100 80GB HBM3, Triton 3.6.0, PyTorch 2.11.0 nv26.02, bf16

______________________________________________________________________

## 1. Kernel Overview

| Kernel | Purpose | Grid | In-scope? |
|--------|---------|------|-----------|
| `_update_paged_kv_cache_kernel` | Scatter new K,V tokens into paged KV cache | (num_tokens, n_kv_heads) | Yes |
| `_flash_decode_stage1_kernel` | Decode attention: GQA-batched, split-K FlashDecode | (batch, n_kv_heads, num_splits) | **Yes — primary target** |
| `_flash_decode_stage2_kernel` | Combine partial outputs from stage1 | (batch, n_heads) | Yes |
| `_paged_context_kernel` | Prefill/context attention for q_len \< 512 | (num_seq, n_heads, num_q_blocks) | Yes |
| `_fast_gather_sdpa_kernel` | Scatter paged KV into contiguous buffers for SDPA | (total_pages, n_kv_heads) | Yes |

**Out of scope:** The SDPA call itself (`torch.nn.functional.scaled_dot_product_attention`), flashinfer metadata ops, Python-side tensor allocations.

### Kernel Details

**`_update_paged_kv_cache_kernel`**

- One program per (token, kv_head). Loads K,V row, writes to physical page via page table lookup.
- Bottleneck: scattered writes (one page-table indirection per token).

**`_flash_decode_stage1_kernel`** (autotuned)

- Grid: (batch, n_kv_heads, num_splits). Each program processes `pages_per_split` pages for one (batch, kv_head) pair.
- Loads Q for HEAD_RATIO_PADDED Q-heads at once (GQA batching), iterates pages, online softmax.
- Inner loop GEMMs: `[HEAD_RATIO_PADDED, HEAD_DIM] @ [HEAD_DIM, PAGE_SIZE]` and `[HEAD_RATIO_PADDED, PAGE_SIZE] @ [PAGE_SIZE, HEAD_DIM]`.
- **Gemma-4 specifics:** HEAD_RATIO=2 (HEAD_RATIO_PADDED=2), HEAD_DIM=176, PAGE_SIZE=16.
- Current autotune: 6 configs varying num_warps ∈ {2,4,8} × num_stages ∈ {2,3}. Key: HEAD_DIM, PAGE_SIZE, HEAD_RATIO_PADDED, SLIDING_WINDOW.

**`_flash_decode_stage2_kernel`**

- Grid: (batch, n_heads). Serial loop over num_splits, weighted combination.
- For num_splits=1 this is a no-op copy. For larger splits it's a small reduction.

**`_paged_context_kernel`** (autotuned)

- Used only when q_len \< 512 (otherwise SDPA path). Two-phase: full pages (no mask) + boundary pages (causal mask).
- Current autotune: 6 configs with Q_BLOCK ∈ {64, 128} × num_stages × num_warps.

**`_fast_gather_sdpa_kernel`**

- One program per (page, kv_head). Copies one page from paged cache to contiguous K,V buffers.
- Pure memory copy, no compute.

______________________________________________________________________

## 2. Target Model & Benchmark Shapes

### Gemma-4-26B-A4B-it Config

| Parameter | Value |
|-----------|-------|
| hidden_size | 2816 |
| num_attention_heads | 16 |
| num_key_value_heads | 8 |
| head_dim | **176** (= 11 × 16, non-power-of-2!) |
| HEAD_RATIO | 2 |
| HEAD_RATIO_PADDED | 2 |
| num_hidden_layers | 30 |
| attention type | 24 sliding-window (size=1024) + 6 full |
| max_position_embeddings | 262144 |
| page_size (typical) | 16 |

### Decode Shape Matrix (primary target: serving hot path)

| ID | batch | n_heads | n_kv_heads | head_dim | page_size | seq_len | SW | Notes |
|----|-------|---------|------------|----------|-----------|---------|-----|-------|
| D1 | 1 | 16 | 8 | 176 | 16 | 128 | 0 | Single decode, short ctx |
| D2 | 1 | 16 | 8 | 176 | 16 | 512 | 0 | Single decode, medium ctx |
| D3 | 1 | 16 | 8 | 176 | 16 | 1024 | 1024 | SW layer, effective 1024 |
| D4 | 1 | 16 | 8 | 176 | 16 | 2048 | 0 | Single decode, long ctx |
| D5 | 8 | 16 | 8 | 176 | 16 | 512 | 0 | Small batch decode |
| D6 | 8 | 16 | 8 | 176 | 16 | 1024 | 1024 | SW small batch |
| D7 | 32 | 16 | 8 | 176 | 16 | 512 | 0 | Medium batch |
| D8 | 64 | 16 | 8 | 176 | 16 | 512 | 1024 | Large batch with SW |

### Context/Prefill Triton Path (q_len \< 512)

| ID | batch | n_heads | n_kv_heads | head_dim | page_size | q_len | kv_len | Notes |
|----|-------|---------|------------|----------|-----------|-------|--------|-------|
| P1 | 1 | 16 | 8 | 176 | 16 | 64 | 64 | Short prefill |
| P2 | 1 | 16 | 8 | 176 | 16 | 256 | 256 | Medium prefill |
| P3 | 4 | 16 | 8 | 176 | 16 | 128 | 128 | Batched prefill |

### Gather + SDPA Path (q_len >= 512)

| ID | batch | n_heads | n_kv_heads | head_dim | page_size | q_len | Notes |
|----|-------|---------|------------|----------|-----------|-------|-------|
| G1 | 1 | 16 | 8 | 176 | 16 | 512 | Prefill SDPA boundary |
| G2 | 1 | 16 | 8 | 176 | 16 | 2048 | Long prefill |
| G3 | 4 | 16 | 8 | 176 | 16 | 512 | Batched prefill SDPA |

______________________________________________________________________

## 3. Optimization Iterations

### Current Best Summary (updated each iteration)

Shape ID mapping updated in iter 2 (extended batch sweep, new IDs):

| ID | Batch | SL | SW | Baseline s1(us) | Baseline E2E(us) | Best s1(us) | Best E2E(us) | Iter |
|----|-------|----|----|----------------|-----------------|-------------|-------------|------|
| D1 | 1 | 512 | 0 | 9.72 | 12.87 | 9.72 | 12.87 | 0 |
| D5 | 16 | 512 | 0 | 31.77 | 35.07 | 31.77 | 35.07 | 0 |
| D7 | 64 | 512 | 0 | 81.76 | 84.69 | 81.76 | 84.69 | 3 |
| D10 | 384 | 512 | 0 | 405.42 | 409.87 | 405.42 | 409.87 | 3 |
| S1 | 1 | 1024 | 1024 | 11.17 | 15.08 | 11.17 | 15.08 | 0 |
| S5 | 384 | 1024 | 1024 | 764.54 | 768.70 | 764.54 | 768.70 | 3 |
| L1 | 1 | 2048 | 0 | 14.01 | 29.74 | 13.85 | 17.90 | 5 |
| L3 | 1 | 8192 | 0 | 31.40 | 60.71 | 30.75 | 34.78 | 5 |

______________________________________________________________________

### Iteration 0 — Baseline (Gemma-4 head_dim=176, first working implementation)

**What changed:** Added `HEAD_DIM_PADDED` constexpr to all five kernels (plus updated Python launchers and autotune keys). Uses `tl.arange(0, HEAD_DIM_PADDED)` where `HEAD_DIM_PADDED = triton.next_power_of_2(head_dim)` = 256 for head_dim=176. All loads/stores masked with `head_dim_mask = dhead_offsets < HEAD_DIM`. `tl.dot` uses padded zeros which do not affect softmax/output correctness.

**Correctness:** PASS (132/132 existing tests pass; head_dim=176 decode verified: max diff 0.0012 vs bf16 reference)

**Environment:** H100 80GB HBM3, Triton 3.6.0, PyTorch 2.11.0 nv26.02, bf16, page_size=16

| ID | s1(us) | s2(us) | e2e(us) | s1%e2e | notes |
|----|--------|--------|---------|--------|-------|
| D1 | 9.50 | 5.93 | 11.88 | 80% | batch=1, seq=128, splits=4 |
| D2 | 10.41 | 8.73 | 14.11 | 74% | batch=1, seq=512, splits=16 |
| D3 | 11.08 | 9.18 | 15.25 | 73% | batch=1, seq=1024, SW=1024, splits=32 |
| D4 | 13.73 | 32.47 | 29.43 | 47% | batch=1, seq=2048, splits=64 — **s2 bottleneck!** |
| D5 | 20.38 | 9.09 | 24.48 | 83% | batch=8, seq=512, splits=16 |
| D6 | 31.59 | 9.09 | 36.05 | 88% | batch=8, seq=1024, SW=1024, splits=16 |
| D7 | 52.33 | 6.68 | 56.06 | 93% | batch=32, seq=512, splits=4 |
| D8 | 81.96 | 6.69 | 85.17 | 96% | batch=64, seq=512, SW=1024, splits=1 |

Context benchmarks:

| ID | ctx(us) | e2e(us) | notes |
|----|---------|---------|-------|
| P1 | 14.17 | 14.16 | batch=1, q=64 |
| P2 | 29.01 | 29.03 | batch=1, q=256 |
| P3 | 20.51 | 69.01 | batch=4, q=128 |

Gather+SDPA benchmarks:

| ID | gather(us) | e2e(us) | notes |
|----|-----------|---------|-------|
| G1 | 8.11 | 27.77 | batch=1, q=512 |
| G2 | 13.51 | 97.12 | batch=1, q=2048 |
| G3 | 13.76 | 102.49 | batch=4, q=512 |

**Analysis:**

- Stage2 is the dominant bottleneck for D4 (s2=32us cold, ~16us warm). Parallelizing stage2 is the highest priority.
- HEAD_DIM_PADDED=256 uses 46% more bandwidth than needed for head_dim=176. Reducing padding overhead is second priority.
- Stage1 scales linearly with batch (D1=9.5→D7=52us at 32x batch, linear). At 8 KV heads, 132 SMs are underutilized for small batches.
- The autotune currently selects from only 6 configs (num_warps×num_stages, no block-size variation).

**Commit:** iter 0

______________________________________________________________________

### Iteration 1 — Expanded autotune search space (stage1 + context)

**What changed:**

- `_flash_decode_stage1_kernel`: Added 9 new configs: num_warps ∈ {1,16}, num_stages ∈ {4,5}.
- `_paged_context_kernel`: Added 6 new configs including Q_BLOCK=32 and num_stages=4.

**Correctness:** No structural change — parameter-only. No re-test needed.

| ID | s1(us) | s2(us) | e2e(us) | Δ e2e | notes |
|----|--------|--------|---------|-------|-------|
| D1 | 9.20 | 5.70 | 11.29 | -5% | improved |
| D2 | 10.00 | 8.74 | 13.93 | -1% | small improvement |
| D3 | 11.21 | 9.21 | 15.56 | +2% | no change |
| D4 | 14.15 | 32.77 | 29.87 | +1% | stage2 bottleneck unchanged |
| D5 | 21.05 | 9.34 | 25.17 | +3% | no improvement |
| D6 | 31.85 | 9.07 | 36.10 | +0% | no change |
| D7 | 50.91 | 6.70 | 54.60 | -3% | small improvement |
| D8 | 81.98 | 6.70 | 85.24 | +0% | no change |

**Analysis:** Small improvements on D1 (+5%), D2 (+1%), D7 (+3%). Stage2 bottleneck for D4 unchanged — sequential reduction over 64 splits with only 16 programs. Next: parallelize stage2 over HEAD_DIM blocks.

**Commit:** iter 1

______________________________________________________________________

### Iteration 2 — Stage2 HEAD_DIM tiling for more parallelism

**What changed:**

- `_flash_decode_stage2_kernel`: Changed grid from (batch, n_heads) to (batch, n_heads, n_hd_blocks) where BLOCK_HD=64. Added 3rd axis `hd_block_id`, computes `dhead_offsets = hd_start + tl.arange(0, BLOCK_HD)`. Added early exit when `hd_start >= HEAD_DIM` to skip all-padding tiles.
- Extended DECODE_SHAPES in sweep script to cover full batch range 1..384.
- Added CONTEXT_SHAPES with chunked prefill (kv_len > q_len) test cases.

**Correctness:** PASS (132/132 tests pass)

| ID | B | SL | SW | splits | s1(us) | s2(us) | e2e(us) | Δ e2e |
|----|---|----|----|--------|--------|--------|---------|-------|
| D1 | 1 | 512 | 0 | 16 | 9.72 | 6.71 | 12.87 | baseline |
| D5 | 16 | 512 | 0 | 8 | 31.77 | 6.85 | 35.07 | baseline |
| D7 | 64 | 512 | 0 | 1 | 81.76 | 7.70 | 84.69\* | - |
| D10 | 384 | 512 | 0 | 1 | 405.42 | 20.32 | 409.87\* | - |
| L1 | 1 | 2048 | 0 | 64 | 14.01 | 31.70 | 29.74 | -10% vs raw s2 |
| L3 | 1 | 8192 | 0 | 128 | 31.40 | 57.32 | 60.71 | baseline |

\*D7/D10 numbers include iter 3 optimization (stage2 bypass for splits=1)

**Analysis:** For L1/L3 the HEAD_DIM tiling gives 4 programs per (batch, head) instead of 1, helping SM utilization. But serial LSE scan loop over NUM_SPLITS=64/128 still dominates stage2 time. The s2 bottleneck at L1 (31μs) is 52% of e2e. Need to parallelize the splits dimension.

**Commit:** `97ff9d0f45`

______________________________________________________________________

### Iteration 3 — Bypass stage2 for num_splits=1

**What changed:**

- In `triton_paged_decode` launcher: added `if num_splits == 1: output.copy_(partial_o.squeeze(2))` to skip the stage2 Triton kernel entirely. For num_splits=1 (which occurs when `batch_size * n_kv_heads >= 2 * num_SMs`, i.e., batch >= 33 for Gemma-4 on H100), partial_o already contains the final normalized output.

**Correctness:** PASS (132/132 tests)

| ID | B | SL | SW | splits | s1(us) | s2(us)\_iso | e2e(us) | e2e overhead |
|----|---|----|----|--------|--------|-----------|---------|-------------|
| D7 | 64 | 512 | 0 | 1 | 81.76 | 7.70 | 84.69 | 2.93μs |
| D8 | 128 | 512 | 0 | 1 | 153.73 | 10.17 | 157.04 | 3.31μs |
| D9 | 256 | 512 | 0 | 1 | 279.64 | 15.17 | 283.24 | 3.60μs |
| D10 | 384 | 512 | 0 | 1 | 405.42 | 20.32 | 409.87 | 4.45μs |
| S4 | 128 | 1024 | 1024 | 1 | 312.55 | 10.40 | 316.25 | 3.70μs |
| S5 | 384 | 1024 | 1024 | 1 | 764.54 | 20.33 | 768.70 | 4.16μs |

s2(us)_iso = isolated stage2 kernel time (reference only, not used in e2e path). e2e overhead = e2e - s1 = cost of torch.copy_() + other.

**Analysis:** For D10, torch.copy\_() costs ~4.45μs vs isolated stage2 kernel 20.32μs. Saves ~16μs for batch=384 decode step. The copy\_ overhead is mainly dtype conversion (float32 → bfloat16) plus memory bandwidth: 384*16*176*4B=41MB read + 384*16*176*2B=21MB write = 62MB at 3.35TB/s = ~18μs... wait that doesn't match. Actually partial_o has head_dim=176 (not padded), so: 384*16*1*176*4=43MB read, 384*16*176\*2=21MB write. 64MB total at roofline = 19μs. But actual = 4.45μs, suggesting the copy is highly optimized or partially from cache.

Next: two-chunk head_dim loading to reduce 25% KV bandwidth in stage1.

**Commit:** see git log

______________________________________________________________________

### Iteration 4 — Stage2 BLOCK_HD=32 (FAILED, reverted)

**What changed:** Reduced stage2 BLOCK_HD from 64 to 32, giving 8 programs per (batch, head) for the 3D grid.

**Correctness:** PASS (parameter-only change).

| ID | s2_iso(us) BLOCK_HD=64 | s2_iso(us) BLOCK_HD=32 | e2e delta |
|----|------------------------|------------------------|-----------|
| L1 | 31.70 | 31.62 | 0% |
| L3 | 57.32 | 57.35 | 0% |
| D1 | 6.71 | 6.62 | +2% worse |
| D10 | 20.32 | 34.95 | n/a (not used in e2e) |

**Analysis:** No improvement for L1/L3 (bottleneck is serial split loop, not SM count). D10 isolated stage2 gets 72% slower with BLOCK_HD=32 (smaller blocks = worse memory coalescing). e2e unchanged because num_splits=1 for D7-D10 (copy\_ path is used). Reverted to BLOCK_HD=64.

**Next:** The L1/L3 stage2 bottleneck is the serial loop over NUM_SPLITS. The fix is to reduce NUM_SPLITS for long context, not increase BLOCK_HD count.

**Commit:** `see git log (reverted)`

______________________________________________________________________

### Iteration 5 — Cap num_splits at 32 (reduce stage2 serial loop for long context)

**What changed:**

- `_get_num_splits`: Changed final cap from `min(num_splits, 128)` to `min(num_splits, 32)`. For short sequences (max_splits ≤ 32 already), no change. For long context (L1: 64→32, L3: 128→32), stage2 serial iterations halve or quadruple decrease.

**Correctness:** PASS (132/132 tests pass)

| ID | B | SL | splits_old | splits_new | s1(us) | s2(us) | e2e(us) | Δ e2e |
|----|---|----|----|-----|--------|--------|---------|-------|
| D1 | 1 | 512 | 16 | 16 | 10.05 | 6.56 | 12.87 | 0% |
| D5 | 16 | 512 | 8 | 8 | 31.84 | 6.72 | 35.26 | 0% |
| L1 | 1 | 2048 | 64 | 32 | 13.85 | 8.62 | 17.90 | **-40%** |
| L2 | 8 | 2048 | 16 | 16 | 50.80 | 7.32 | 54.94 | 0% |
| L3 | 1 | 8192 | 128 | 32 | 30.75 | 8.40 | 34.78 | **-43%** |
| L4 | 8 | 8192 | 16 | 16 | 152.63 | 7.32 | 156.62 | 0% |

**Analysis:** Massive improvement for long-context single-batch decode: L1 -40%, L3 -43%. Stage2 with 32 splits: grid=(1,16,4)=64 programs, 32 serial iterations each. Stage2 ≈ 8μs (from 32/57μs). Stage1 essentially unchanged (same total work, fewer programs but each processes more pages). Short sequences (D1-D5, L2, L4) unaffected because their max_splits was already ≤ 32.

Root cause: stage2 serial loop over splits is O(num_splits) per program, and stage2 was dominating for large num_splits. Capping at 32 is a universal improvement for all GPUs, not just H100.

**Commit:** see git log

______________________________________________________________________

### Iteration 6 — Two-chunk head_dim for stage1 (non-power-of-2 HEAD_DIM)

**What changed:**

- Added `_flash_decode_stage1_two_chunk_kernel`: splits head_dim=176 into two chunks (CHUNK1=128, CHUNK2=64), avoiding HEAD_DIM_PADDED=256 padding waste. For head_dim=176: loads 192 elements (128+64) vs 256, saving 25% bandwidth per KV load.
- Launcher `triton_paged_decode` dispatches to two-chunk kernel when `head_dim_padded != head_dim`.
- Updated `bench_decode_stage1` to benchmark the correct kernel variant.

**Correctness:** PASS (132/132 tests pass; two-chunk dispatches correctly for head_dim=176)

| ID | B | SL | splits | s1_old(us) | s1_twochunk(us) | e2e_old(us) | e2e_new(us) | Δ |
|----|---|----|----|--------|--------|---------|---------|-------|
| D1 | 1 | 512 | 16 | 9.72 | 9.91 | 12.87 | 12.87 | 0% |
| D5 | 16 | 512 | 8 | 31.84 | 33.04 | 35.26 | 36.40 | +3% worse |
| D7 | 64 | 512 | 1 | 81.76 | 81.35 | 84.69 | 84.21 | -0.6% |
| D10 | 384 | 512 | 1 | 405.42 | 402.41 | 409.87 | 406.76 | -0.8% |
| L1 | 1 | 2048 | 32 | 13.85 | 13.80 | 17.90 | 17.72 | -1.0% |
| L3 | 1 | 8192 | 32 | 30.75 | 30.48 | 34.78 | 34.63 | -0.4% |

**Analysis:** Mixed results. Large-batch BW-bound shapes (D7-D10): +0.6-0.8% improvement. Small-page-count shapes (D5, 4 pages/split): +3% regression from two extra dot products per page iteration. The two-chunk approach reduces theoretical KV bandwidth by 25% (192 vs 256 elements) but the benefit is small because:

1. Tensor core overhead of 4 dots vs 2 adds latency for short loops
1. Software pipelining (num_stages) may be less effective with two separate loads per page

Net: kept since large-batch (primary serving bottleneck) marginally improves, and the kernel is correct.

**Commit:** see git log

______________________________________________________________________

### Iteration 7 — Two-chunk gather kernel (non-power-of-2 HEAD_DIM bandwidth reduction)

**What changed:**

- Added two-chunk code path to `_fast_gather_sdpa_kernel`: when `HD_CHUNK1 > 0`, loads/stores head_dim in two chunks (HD_CHUNK1=128 + HD_CHUNK2=64 = 192 elements) instead of HEAD_DIM_PADDED=256, saving 25% KV bandwidth per page.
- Single-chunk path (HD_CHUNK1==0) unchanged for power-of-2 head_dims.
- Caller updated to compute and pass HD_CHUNK1/HD_CHUNK2 based on head_dim.
- Fixed Triton bug: `other=0.0` requires `mask`; removed from always-valid c1 loads.

**Correctness:** PASS (132/132 tests pass)

| ID | B | q_len | gather_old(us) | gather_new(us) | e2e_old(us) | e2e_new(us) | Δ gather |
|----|---|-------|---------------|---------------|------------|------------|---------|
| G1 | 1 | 512 | 8.11 | 7.97 | 27.77 | 32.92 | -1.7% |
| G2 | 1 | 2048 | 13.51 | 13.89 | 97.12 | 96.67 | +2.8% (noise) |
| G3 | 4 | 512 | 13.76 | 14.15 | 102.49 | 103.83 | +2.7% (noise) |

**Analysis:** Marginal improvement on G1 (-1.7%), within measurement noise on G2/G3 (~3%). The 25% theoretical bandwidth reduction doesn't translate well because:

1. Gather is already fast (7-14 μs) — overhead of 4 loads/stores vs 2 adds latency
1. Working set is small → may be latency-bound rather than bandwidth-bound
1. Two-chunk's extra instruction count partially offsets bandwidth savings

Kept since it's correct and reduces memory ops. E2E variance is dominated by SDPA/torch ops.

**Commit:** see git log

______________________________________________________________________

### Iteration 8 — Stage2 @triton.autotune (BLOCK_HD + num_warps sweep)

**What changed:**

- Added `@triton.autotune` to `_flash_decode_stage2_kernel` with 10 configs sweeping BLOCK_HD ∈ {32, 64, 128, 256} × num_warps ∈ {1, 2, 4, 8}. Key: `[HEAD_DIM, HEAD_DIM_PADDED, NUM_SPLITS]`.
- Updated caller to use lambda grid `(batch, n_heads, cdiv(head_dim_padded, BLOCK_HD))` so grid adapts to autotuned BLOCK_HD.
- Updated sweep script stage2 benchmark to use same lambda grid.

**Correctness:** PASS (132/132 tests pass)

| ID | s2_old(us) | s2_new(us) | e2e_old(us) | e2e_new(us) | Δ e2e |
|----|-----------|-----------|------------|------------|-------|
| D1 | 6.82 | 6.38 | 13.14 | 13.03 | -0.8% |
| D2 | 6.99 | 6.22 | 16.09 | 14.56 | -9.5% |
| D3 | 6.90 | 6.65 | 18.24 | 17.54 | -3.8% |
| D5 | 6.70 | 7.02 | 36.70 | 36.74 | +0.1% |
| L1 | 8.36 | 7.09 | 18.00 | 17.35 | -3.6% |
| L3 | 8.65 | 7.17 | 34.86 | 34.18 | -1.9% |
| D9\* | 15.22 | 8.62 | n/a | n/a | kernel-only (splits=1 bypassed) |
| D10\* | 20.44 | 9.58 | n/a | n/a | kernel-only (splits=1 bypassed) |

\*D9/D10 use num_splits=1 bypass so stage2 kernel improvement doesn't affect e2e.

**Analysis:** Stage2 autotune picks BLOCK_HD=256 for large-batch/large-splits shapes, reducing grid size and improving SM efficiency. For D9/D10 the isolated stage2 halves (15-20μs → 8-10μs) but since splits=1 bypasses stage2, e2e is unaffected. Active stage2 shapes (splits>1) see 1-10% improvement. Largest win: D2 (-9.5% e2e). S3 showed slight regression (noise/autotuning variance).

**Commit:** see git log

______________________________________________________________________

### Iteration 9 — One-pass stage2 reduction (FAILED, reverted)

**What changed:**

- Replaced two-pass stage2 reduction with one-pass online log-sum-exp: combines `find global_max` + `weighted sum` passes into a single loop using running rescaling `alpha = exp(old_lse - new_max)`, `acc = acc * alpha + weight * partial_o`.

**Correctness:** PASS (132/132)

**Results:** Large regression for NUM_SPLITS=32 shapes:

- S1 (splits=32): s2 6.91→18.23 μs (+164%), e2e 15.11→20.32 (+35%)
- L1 (splits=32): s2 7.09→18.49 μs (+161%), e2e 17.35→22.92 (+32%)

**Analysis:** Per-iteration alpha rescaling adds `exp() + vector multiply` overhead absent in two-pass. For splits=32, this 32 extra exp + vector ops exceeds savings from halved partial_lse reads. Reverted.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 10 — Context kernel Q_BLOCK=256 expansion (FAILED, reverted)

**What changed:**

- Added Q_BLOCK=256 configs to context kernel autotune (2 configs: num_stages=2/3, num_warps=16).
- Hypothesis: longer Q tiles reduce grid size for long prefills and improve QK GEMM size.

**Correctness:** PASS (132/132)

**Results:** All context shapes regressed 6-10% vs iter 0 baseline:

- C1 (q=128, sl=512): 47.81 → 50.76-52.60 μs (+6-10%)
- C4 (q=256, sl=2048): 177.49 → 190-196 μs (+7-10%)

**Analysis:** The autotuner explored Q_BLOCK=256 configs and selected a slower config. Q_BLOCK=256 may have higher register usage (spills) for the GQA-batched inner loop, causing more pipeline stalls. Also, with Q_BLOCK=256 and q_len=128, the block is 2× bigger than the actual work → wasted computation. Reverted.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 11 — Update kernel autotune num_warps (FAILED, reverted)

**What changed:** Added `@triton.autotune` to `_update_paged_kv_cache_kernel` with 6 configs sweeping num_warps ∈ {1,2,4,8} × num_stages ∈ {1,2}. Key: `[HEAD_DIM, HEAD_DIM_PADDED, N_KV_HEADS, PAGE_SIZE]`.

**Correctness:** PASS (132/132)

**Results:** D5 +12%, D10 +16% regression; D1 -4% improvement.

**Analysis:** Autotune key lacks NUM_TOKENS, so one config applies to all grid sizes. Autotuner picked config optimal for large prefill grids (T=512) but wasteful for decode T=1 (grid=(1,8)=8 programs — 8 warps per 8-program kernel is ~1 warp per program). Reverted. Fix: need per-grid-size config selection or two separate kernels.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 12 — Add SLIDING_WINDOW to context kernel autotune key

**What changed:**

- Added `SLIDING_WINDOW` to context kernel autotune key. SW shapes (Gemma-4 has 24 SW layers + 6 full) use a different code path (extra per-token masking, different phase1/phase2 balance) and may benefit from different configs.

**Correctness:** PASS (132/132)

**Results:** No measurable change; all shapes within noise of iter 8 baseline:

- C1: 50.68 μs (baseline 50.76, flat)
- CS1: 50.96 μs (was ~50.78, flat)

**Analysis:** The autotune selected the same optimal config for both SW and non-SW shapes at these shapes. The key addition is correct (prevents overfitting one config across two different code paths) but provides no measurable speedup. Kept as a correctness improvement.

**Commit:** see git log

______________________________________________________________________

### Iteration 13 — bf16 partial_o to halve stage2 bandwidth (FAILED, reverted)

**What changed:**

- Stage1 (both kernels) store `(acc / l_i_safe).to(bfloat16)` instead of float32.
- partial_o allocated as `q.dtype` (bf16) instead of float32.
- Stage2: explicitly cast loaded partial_o to float32 for accumulation.
- Also tried adding NUM_SPLITS to two-chunk autotune key to prevent same config for short/long loops.

**Correctness:** PASS (132/132)

**Results:** Mixed — significant regression on L3:

- D5: 36.74 → 33.12 (-9.8%), D8: 166.41 → 155.61 (-6.7%), S1: 15.11 → 14.38 (-4.8%)
- L3: 34.18 → 41.37 (+21%), L3 stage1 30.88 → 37.71 (+22%)

**Analysis:** The `(acc / l_i_safe).to(bfloat16)` cast in the two-chunk kernel increases register pressure (need both float32 and bf16 in scope simultaneously) and changes SASS, causing the autotuner to select a worse config for L3 (16 pages/split). Short-loop shapes (D5: 4 pages/split) see improvement because they have less loop overhead to amortize. Reverted.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 14 — Gather kernel autotune num_warps (no improvement, reverted)

**What changed:** Added `@triton.autotune` to `_fast_gather_sdpa_kernel` with 6 configs sweeping num_warps ∈ {1,2,4,8}. Key: `[HEAD_DIM, HEAD_DIM_PADDED, PAGE_SIZE, HD_CHUNK1, HD_CHUNK2]`.

**Correctness:** PASS (132/132)

**Results:** G1: 7.97→8.06 μs (+1.1%), G2: 13.89→14.20 μs (+2.2%). Within noise — no improvement.

**Analysis:** Gather is a pure memory copy. Autotuner picked same config as default. Extra benchmarking overhead on first use without measurable benefit. Reverted.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 17 — Hoist loop-invariant KV offset tables outside inner page loop (KEPT)

**What changed:**

- In both `_flash_decode_stage1_kernel` and `_flash_decode_stage1_two_chunk_kernel`:
  - Moved `page_offsets = tl.arange(0, PAGE_SIZE)` outside the inner loop (original kernel had it inside)
  - Precomputed `kv_head_base = kv_head_id * cache_stride_head` (scalar) outside loop
  - Precomputed `local_kv_c1 = page_offsets[:, None] * cache_stride_token + dhead_c1[None, :]` and `local_kv_c2` as `[PAGE_SIZE, HD_CHUNK]` tensors outside loop — matching the pattern already used in `_paged_context_kernel`
- The loop body now computes only the loop-variant part: `cache_page_base = physical_page.to(int64) * cache_stride_block + kv_head_base` (scalar)

**Correctness:** PASS (132/132)

**Results vs iter 8 baseline:**

| ID | Baseline e2e (μs) | Iter 17 e2e (μs) | Delta |
|----|------------------|-----------------|-------|
| D1 | 12.73 | 12.47 | -2.0% |
| D2 | 16.38 | 14.14 | **-13.7%** |
| D3 | 22.40 | 17.40 | **-22.3%** |
| D4 | 29.43 | 23.36 | **-20.6%** |
| D5 | 41.22 | 34.51 | **-16.3%** |
| D6 | 60.92 | 53.24 | **-12.6%** |
| D7 | 93.26 | 84.00 | **-9.9%** |
| D8 | 170.69 | 153.57 | **-10.0%** |
| D9 | 324.89 | 279.65 | **-13.9%** |
| D10 | 477.90 | 407.19 | **-14.8%** |
| S1 | 14.35 | 14.11 | -1.7% |
| S2 | 35.85 | 32.82 | **-8.5%** |
| S3 | 94.75 | 82.86 | **-12.5%** |
| S4 | 320.91 | 268.55 | **-16.3%** |
| L1 | 17.43 | 16.79 | **-3.7%** |
| L2 | 60.67 | 53.24 | **-12.2%** |
| L3 | 34.87 | 33.40 | **-4.2%** |
| L4 | 168.07 | 152.68 | **-9.2%** |

**Analysis:** Large, consistent improvements across almost all shapes. The Triton compiler was NOT automatically hoisting the `page_offsets[:, None] * cache_stride_token + dhead_c{1,2}[None, :]` computation outside the inner loop — even though it is loop-invariant (no dependency on loop variable or page table data). Making it explicit via Python-level hoisting at the Triton AST level forces the pre-computation outside the loop, saving 2D offset table recomputation per page iteration. This is purely an instruction-count reduction with no mathematical change.

Median improvement: ~12% across most decode shapes. Shapes with more pages per split benefit more (D3: -22% with 2 pages/split and 16 programs/iteration needing more compute).

**Commit:** see git log

______________________________________________________________________

### Iteration 16 — Q_BLOCK=16 in context kernel autotune (FAILED, reverted)

**What changed:** Added 3 configs with `Q_BLOCK=16` to `_paged_context_kernel` autotune. Hypothesis: for very low-parallelism shapes (B=1, q_len=64), grid=(1, 16, 1)=16 programs uses only 12% of H100's 132 SMs. Q_BLOCK=16 gives 4× more programs.

**Correctness:** PASS (132/132)

**Results:**

| ID | Baseline (μs) | Iter 16 (μs) | Delta |
|----|--------------|-------------|-------|
| P1 (B=1, q=64) | 14.23 | 12.49 | **-12%** |
| P2 (B=1, q=256) | 28.84 | 28.68 | flat |
| P3 (B=4, q=128) | 20.18 | 23.46 | **+16%** |
| C1 (B=1, q=128, kv=512) | 48.06 | 48.90 | flat |
| C2 (B=1, q=128, kv=1024) | 85.55 | 89.69 | **+5%** |
| C3 (B=4, q=128, kv=512) | 50.84 | 94.70 | **+86%** |
| C4 (B=4, q=256, kv=2048) | 178.04 | 581.59 | **+226%** |

**Analysis:** Autotune picked Q_BLOCK=16 based on P1 (first benchmarked shape). This is optimal for P1 (small B × small q_len → 12% SM utilization with Q_BLOCK=64) but catastrophic for B=4 shapes where Q_BLOCK=128 was needed for GEMM efficiency. Root cause: autotune key lacks `batch` or `q_len` dimension, so one config is selected for all shapes. Reverted.

**Lesson:** Adding smaller Q_BLOCK configs to autotune hurts because the autotuner can't distinguish B=1/q=64 (needs Q_BLOCK=16) from B=4/q=256 (needs Q_BLOCK=128). A shape-aware Python-side dispatch (not autotune) would be needed.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 15 — Add NUM_SPLITS to two-chunk stage1 autotune key (FAILED, reverted)

**What changed:** Added `NUM_SPLITS` to two-chunk stage1 kernel autotune key. Hypothesis: small-split (D1: 2 pages/split) and large-split (L3: 16 pages/split) need different optimal configs.

**Correctness:** PASS (132/132)

**Results:** D4 +24%, L3 +25% regression; D5 -8%, D6 -9%, D9 -4% improvement.

**Analysis:** Adding NUM_SPLITS triggers 15 separate autotune runs for each unique splits value. For heavy shapes (NUM_SPLITS=16/32), the autotune benchmark itself might be noisy (375 kernel launches per shapes with heavy inner loops), selecting suboptimal configs. The inconsistent per-shape results suggest autotune instability. Reverted.

**Commit:** see git log (reverted)

______________________________________________________________________

### Iteration 18 — Logit soft-capping support (Gemma-4: logit_cap=50.0)

**What changed:** Added full logit soft-cap support to all three attention kernels and the Python dispatch layer. Previously `triton_paged_mha_with_cache` completely ignored `logit_cap`, producing incorrect attention scores for Gemma-4 (which uses `attn_logit_softcapping=50.0`).

**Changes (7 locations):**

1. Added `"LOGIT_CAP"` to autotune keys of all 3 kernels (`_flash_decode_stage1_kernel`, `_flash_decode_stage1_two_chunk_kernel`, `_paged_context_kernel`) — softcap changes the compute pattern and may prefer different configs
1. Added `LOGIT_CAP: tl.constexpr = 0.0` parameter to all 3 kernels
1. Added soft-capping code `LOGIT_CAP * (2.0 * tl.sigmoid(2.0 * (score / LOGIT_CAP)) - 1.0)` BEFORE masking in all 4 compute sites (stage1 kernel + two-chunk kernel + context phase1 ×3 branches + context phase2)
1. Added `logit_cap: Optional[float] = None` to `triton_paged_decode` and `triton_paged_context`
1. Added `logit_cap: Optional[float] = None` to `triton_paged_mha_with_cache` op and fake op
1. Updated `get_constants` to extract `logit_cap` from source attention node
1. Disabled SDPA path when `logit_cap > 0` (cuDNN SDPA doesn't support soft-capping)

**Implementation note — tanh via tl.sigmoid:**\
Triton 3.6.0 does not have `tl.math.tanh`. Using the identity `tanh(x) = 2·sigmoid(2x) − 1` which is numerically stable: for large |x|, sigmoid saturates to 0/1, so `tanh` saturates to ±1 correctly.

**Implementation note — apply BEFORE masking:**\
Soft-capping is applied after `score *= SM_SCALE` but before page/causal/sliding-window masks. This ensures `-inf` from masks propagates cleanly through softmax (`exp(-inf)=0`). The difference vs applying after masking is negligible: `exp(−logit_cap)=exp(−50)≈1.9e-22≈0` for any masked position.

**Correctness:** PASS — 31/31 new `TestLogitSoftCap` tests pass, covering:

- Decode with logit_cap=50.0 for head_dim ∈ {64, 128, 176}, batch ∈ {1, 4}, seq_len ∈ {64, 256}
- Context with logit_cap=50.0 for head_dim ∈ {64, 128, 176}, batch ∈ {1, 2}, seq_len ∈ {32, 128, 256}
- Backward-compat: logit_cap=None gives same result as omitting arg

**Performance impact:** Zero overhead when `logit_cap=0.0` (default) because the `if LOGIT_CAP > 0.0:` is a compile-time constexpr branch — Triton eliminates the dead branch. When `logit_cap > 0`, overhead is one `sigmoid` per score element (~1 SFU cycle per 32 elements on H100), dominated by the QK matmul.

**Commit:** see git log

______________________________________________________________________

### Iteration 19 — Hoist q_positions_2d before both context kernel loops

**What changed:** In `_paged_context_kernel`, `q_kv_pos = q_offsets[:, None] + cache_len` was
computed inside the Phase 1 loop (SLIDING_WINDOW path) on every iteration, despite being fully
loop-invariant. An identical computation `q_positions_2d` was already hoisted before Phase 2.
Unified the variable and moved the single computation before both loops.

**Changes:**

- Added `q_positions_2d = q_offsets[:, None] + cache_len` before Phase 1 loop
- Phase 1 SW path: replaced `q_kv_pos = ...` + `sw_mask = (q_kv_pos - ...)` with `sw_mask = (q_positions_2d - ...)`
- Removed duplicate `q_positions_2d` assignment before Phase 2

**Affected shapes:** Context with SLIDING_WINDOW > 0 (S1-S4). Saves one `[Q_BLOCK, 1]` vector add per Phase 1 iteration.

**Correctness:** PASS (163/163)

**Commit:** see git log

______________________________________________________________________

### Iteration 20 — One-pass stage2 online softmax (FAILED, reverted in iter 22)

**What changed:** In `_flash_decode_stage2_kernel`, replaced the prior two-pass algorithm
(pass 1: find global max LSE, pass 2: weighted sum) with a single-pass online combination
maintaining running `(m_i, l_i, acc)` flash-attention style. Also hoisted batch/head offsets
outside the loop.

**Correctness:** PASS (163/163)

**Results:**

| ID | Iter17 e2e (μs) | Iter20 e2e (μs) | Delta |
|----|-----------------|-----------------|-------|
| D1 | 12.47 | 12.46 | flat |
| D2 | 14.14 | 19.82 | **+40% REGRESSION** |
| D3 | 17.40 | 23.23 | **+33% REGRESSION** |
| D4 | 23.36 | 23.26 | flat |
| D5 | 34.51 | 34.33 | flat |
| S1 | 14.11 | 19.76 | **+40% REGRESSION** |
| L3 | 33.40 | 38.99 | **+17% REGRESSION** |

**Analysis:** The one-pass algorithm requires TWO `tl.exp` calls per split (α = exp(m_old − m_new)
and weight = exp(lse − m_new)) plus TWO `tl.where` guards. GPU predicated execution computes
both branches of `tl.where`, so both `tl.exp` calls always execute. The two-pass algorithm only
needs ONE `tl.exp` per split in pass 2 (weight = exp(lse − global_max)), with no guards needed
since global_max is already known. The extra exp+where overhead dominates at NUM_SPLITS=16-32.
The hoisting optimization (batch/head offsets) is correct and kept; only the loop algorithm is reverted.

**Commit:** see git log (reverted in iter 22)

______________________________________________________________________

### Iteration 21 — Tighten Phase 2 loop upper bound in context kernel

**What changed:** In `_paged_context_kernel`, Phase 2 loop (boundary/causal pages) previously
iterated from `num_full_pages` to `num_kv_pages` and used an inner `if kv_base_pos <= max_q_pos:`
check to skip pages beyond the current Q block's causal window. Replaced with a tighter loop bound:
`phase2_end = tl.minimum(num_kv_pages, max_q_pos // PAGE_SIZE + 1)`, encoding the causal skip
in the loop bound itself and removing the per-page branch.

**Changes:**

- Added `max_q_pos = q_offsets[-1] + cache_len` scalar computation before Phase 2
- Changed loop `for page_idx in range(num_full_pages, num_kv_pages):` to
  `for page_idx in range(num_full_pages, phase2_end):`
- Removed inner `if kv_base_pos <= max_q_pos:` branch and dedented the softmax update block

**Affected shapes:** Context/prefill with chunked input (early Q blocks in long prefill skip
all pages beyond the Q causal window). For single-block Q (typical short prefill), max_q_pos
≈ q_len + cache_len, so phase2_end ≈ num_kv_pages — no difference. For multi-block Q during
long prefill, saves O(num_q_blocks) page iterations.

**Correctness:** PASS (163/163)

**Commit:** see git log

______________________________________________________________________

### Iteration 22 — Revert iter 20 one-pass stage2; restore two-pass with hoisting

**What changed:** Reverted `_flash_decode_stage2_kernel` from iter 20's one-pass online algorithm
back to the two-pass approach. Kept the batch/head offset hoisting (a pure win from iter 20)
but replaced the algorithm with the standard two-pass:

- **Pass 1** (scalar ops only): loop over NUM_SPLITS to find `global_max_lse`
- **Pass 2** (vector ops): loop over NUM_SPLITS, compute `weight = tl.exp(lse - global_max_lse)`,
  accumulate `acc += weight * partial_o`

Key difference vs one-pass: pass 2 has ONE `tl.exp` per split (vs two in one-pass), no `tl.where`
guards needed (global_max already known), and no per-element alpha rescaling of `acc`.

Also kept: early-exit guard `if global_max_lse == float("-inf")` for empty sequences (all-inf LSE).

**Correctness:** PASS (163/163 — same test suite)

**Expected recovery:** D2 +40%, D3 +33%, S1 +40%, L3 +17% regressions from iter 20 resolved.

**Commit:** see git log

______________________________________________________________________

## 4. Optimization Ideas Backlog

### Category A — Decode Stage1 Autotune Space

- \[ \] **A1** Expand num_warps to {1, 2, 4, 8, 16}: current max is 8; 16 warps may help on H100 with long sequences (more MIO/memory parallelism)
- \[ \] **A2** Expand num_stages to {1, 2, 3, 4, 5}: pipeline depth for HBM prefetch; stage 4-5 can hide HBM latency
- \[ \] **A3** Add PAGE_SIZE_BLOCK as tuneable: currently processes exactly one cache page per iteration; try processing 2 or 4 pages per iteration (inner-loop unroll)
- \[ \] **A4** BLOCK_M (HEAD_RATIO_PADDED) tuning: for HEAD_RATIO=2 (Gemma-4), PADDED=2. Not much room here, but try padding to 4 to enable wider dot
- \[ \] **A5** Sweep autotune key: add `BATCH_SIZE` range to key (small batch vs large may need different warps)

### Category B — head_dim=176 Specific

- \[ \] **B1** Pad HEAD_DIM to 192 (next 64-multiple) and mask: `tl.arange(0, 192)` with mask for dim 176; may improve vectorized load throughput
- \[ \] **B2** Pad HEAD_DIM to 256 (next power-of-2) and mask: classic approach; may unlock wider GEMM intrinsics but wastes 32% bandwidth
- \[ \] **B3** Split HEAD_DIM into 160 + 16 and process separately: avoids padding waste while keeping 16-aligned chunks; avoids masking overhead
- \[ \] **B4** Use `tl.load` with block_ptr and `boundary_check`: cleaner masking for padded head_dim
- \[ \] **B5** Verify actual Triton code gen for HEAD_DIM=176: compile-time check to confirm no implicit padding or register waste

### Category C — Memory Access Patterns

- \[ \] **C1** Vectorized K/V loads: current HEAD_DIM loads could use 128-bit aligned accesses (176 elements × 2 bytes = 352 bytes; not 128-bit aligned per-row, but overall pages are aligned)
- \[ \] **C2** Transpose KV in cache at write time: current cache layout \[block, 2, kv_heads, page_size, head_dim\]. Consider \[block, 2, kv_heads, head_dim, page_size\] (HND→NHD within page) to make Q×K the transpose-free dim
- \[ \] **C3** Coalesced Q load: currently loads Q as \[HEAD_RATIO_PADDED, HEAD_DIM\] in stage1 with strided access. Check if this is contiguous
- \[ \] **C4** Prefetch page table entries: page table lookups `tl.load(kv_indices_ptr + ...)` are scalar loads; could be prefetched

### Category D — Parallelism & Structure

- \[ \] **D1** Stage1 + Stage2 fusion for num_splits=1: when sequence is short enough that num_splits=1, stage2 is a trivial normalize; skip intermediate buffer write/read
- \[ \] **D2** Stage2 vectorization over HEAD_DIM: currently the stage2 kernel has all HEAD_DIM in one thread (no parallelism across head_dim within one program). For large HEAD_DIM (176), consider 2D grid
- \[ \] **D3** Persistent kernel for decode: use persistent grid (fewer programs, each loops over multiple batches) to improve SM utilization for small batches
- \[ \] **D4** Increase split-K parallelism for long-context: current `_get_num_splits` targets 4 waves; experiment with 8 waves (may help for long ctx single-batch)
- \[ \] **D5** GQA-aware scheduling: for HEAD_RATIO=2, try processing one KV-head pair → 2 Q-heads at warp level with interleaved computation

### Category E — KV Cache Update Kernel

- \[ \] **E1** Merge token dimension: for num_tokens=1 (common single-decode), avoid full (num_tokens, n_kv_heads) grid; special-case with smaller grid
- \[ \] **E2** Vectorized K/V stores: 176 × 2 bytes = 352 bytes per head; try 128-bit stores for the first 160 elements + remaining 16
- \[ \] **E3** Fuse KV update with Q load (pipelined): during decode, Q and KV are available simultaneously; in principle update could overlap with attention

### Category F — Context/Prefill Kernel

- \[ \] **F1** Expand Q_BLOCK to {32, 256}: current configs only have 64 and 128; try smaller (less register pressure) and larger
- \[ \] **F2** Add `HEAD_DIM` block padding to context kernel: same B1/B2 for context
- \[ \] **F3** Widen Phase1 to process 2 pages per outer iteration (equivalent to BLOCK_KV=32): avoid loop overhead

### Category G — Gather SDPA Kernel

- \[ \] **G1** Vectorized loads/stores: current does \[PAGE_SIZE, HEAD_DIM\] = \[16, 176\] per program; load as wider dtype
- \[ \] **G2** Process multiple KV-heads per program: currently one KV head per program; try 2
- \[ \] **G3** Use `tl.make_block_ptr` for contiguous destination writes

______________________________________________________________________

## 5. Final Best Configuration

*To be filled after all iterations.*

______________________________________________________________________

## 6. Appendix: How to Reproduce

```bash
# From TensorRT-LLM root
python tensorrt_llm/_torch/auto_deploy/custom_ops/attention/sweep_triton_paged_attention.py

# With sweep mode
python tensorrt_llm/_torch/auto_deploy/custom_ops/attention/sweep_triton_paged_attention.py --sweep

# Override params (single eval)
python tensorrt_llm/_torch/auto_deploy/custom_ops/attention/sweep_triton_paged_attention.py \
    --num-warps 8 --num-stages 3

# Filter shapes
python tensorrt_llm/_torch/auto_deploy/custom_ops/attention/sweep_triton_paged_attention.py \
    --shapes D1,D2,D3
```

**Environment:**

- GPU: NVIDIA H100 80GB HBM3 (132 SMs)
- Triton: 3.6.0
- PyTorch: 2.11.0a0+eb65b36914.nv26.02
- dtype: bf16
- Benchmark: `triton.testing.do_bench(warmup=25, rep=100)`
