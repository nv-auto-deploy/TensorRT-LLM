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

| ID | Baseline s1(us) | Baseline E2E(us) | Best s1(us) | Best E2E(us) | Iter |
|----|----------------|-----------------|-------------|-------------|------|
| D1 | 9.50 | 11.88 | 9.50 | 11.88 | 0 |
| D2 | 10.41 | 14.11 | 10.41 | 14.11 | 0 |
| D3 | 11.08 | 15.25 | 11.08 | 15.25 | 0 |
| D4 | 13.73 | 29.43 | 13.73 | 29.43 | 0 |
| D5 | 20.38 | 24.48 | 20.38 | 24.48 | 0 |
| D6 | 31.59 | 36.05 | 31.59 | 36.05 | 0 |
| D7 | 52.33 | 56.06 | 52.33 | 56.06 | 0 |
| D8 | 81.96 | 85.17 | 81.96 | 85.17 | 0 |

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
