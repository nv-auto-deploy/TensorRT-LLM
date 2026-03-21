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
| D1 | 11.9 | 14.3 | 11.9 | 0.83x | 0 |
| D2 | 16.7 | 16.8 | 16.7 | 0.99x | 0 |
| D3 | 31.9 | 30.0 | 31.9 | 1.06x | 0 |
| D4 | 18.1 | 22.5 | 18.1 | 0.80x | 0 |
| D5 | 40.1 | 44.5 | 40.1 | 0.90x | 0 |
| D6 | 108.1 | 110.4 | 108.1 | 0.98x | 0 |
| D7 | 108.5 | 107.7 | 108.5 | 1.01x | 0 |
| D8 | 366.4 | 372.5 | 366.4 | 0.98x | 0 |
| D11 | 14.1 | 14.3 | 14.1 | 0.98x | 0 |
| D12 | 21.3 | 22.5 | 21.3 | 0.95x | 0 |
| D13 | 14.6 | 14.4 | 14.6 | 1.01x | 0 |
| D14 | 23.6 | 22.7 | 23.6 | 1.04x | 0 |
| D15 | 48.0 | 45.6 | 48.0 | 1.05x | 0 |
| D9 | 17.2 | 15.4 | 17.2 | 1.11x | 0 |
| D10 | 26.3 | 29.9 | 26.3 | 0.88x | 0 |

### Summary — Best prefill latency so far (us)

| ID | Baseline | FI | Best | Delta vs FI | Iteration |
|----|----------|----|------|-------------|-----------|
| P1 | 65.5 | 14.1 | 65.5 | 4.64x | 0 |
| P2 | 92.6 | 22.9 | 92.6 | 4.04x | 0 |
| P3 | 320.4 | 105.3 | 320.4 | 3.04x | 0 |
| P4 | 134.0 | 54.2 | 134.0 | 2.47x | 0 |
| P5 | 960.7 | 344.8 | 960.7 | 2.79x | 0 |
| P6 | 291.6 | 96.7 | 291.6 | 3.01x | 0 |
| P7 | 226.5 | 67.9 | 226.5 | 3.33x | 0 |
| P8 | 319.6 | 104.6 | 319.6 | 3.06x | 0 |
| P9 | 131.6 | 53.8 | 131.6 | 2.45x | 0 |

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

______________________________________________________________________

## 4. Optimization Ideas Backlog

### Decode Stage 1

- \[ \] Sweep num_warps {1,2,4,8,16} x num_stages {1,2,3,4,5}
- \[ \] Vectorized KV loads (load float4 / use tl.load with wider access)
- \[ \] Process multiple pages per loop iteration to increase arithmetic intensity
- \[ \] Precompute page indices into shared memory to avoid repeated page-table loads
- \[ \] Try different HEAD_RATIO tiling (process fewer heads per program but more pages)
- \[ \] Persistent kernel: assign multiple (batch, kv_head) pairs to one program
- \[ \] Use tl.dot with tf32 accumulation for higher throughput on H100

### Decode Stage 2

- \[ \] Unroll split loop for common NUM_SPLITS values
- \[ \] Load all LSE values first, compute global max, then load partial_o (better pipelining)
- \[ \] Vectorized partial_o loads

### Context/Prefill Kernel

- \[ \] Sweep Q_BLOCK {16,32,64,128,256} x num_warps x num_stages
- \[ \] Process multiple KV pages per loop iteration
- \[ \] 2D tiling over Q and KV dimensions
- \[ \] Skip fully-masked pages more efficiently (precompute boundary)
- \[ \] Persistent kernel: process multiple sequences per program

### Cache Update Kernel

- \[ \] Vectorized K,V stores
- \[ \] Fuse cache update into stage1 decode / context kernel

______________________________________________________________________

## 5. Final Best Configuration

(to be filled at end)

______________________________________________________________________

## 6. Appendix: How to Reproduce

```bash
cd tensorrt_llm/_torch/auto_deploy/custom_ops/attention/
python bench_triton_paged_attention.py
python bench_triton_paged_attention.py --mode decode
python bench_triton_paged_attention.py --mode prefill
```
