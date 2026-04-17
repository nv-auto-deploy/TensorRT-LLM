# Megakernel A: Status & Future Work

## What's Done

| Feature | Status | Commit |
|---------|--------|--------|
| Persistent kernel framework (132 SMs, instruction dispatch) | DONE | c92a70b1 |
| GEMV_QKV (distributed, uint4 loads, all 20 warps) | DONE | c92a70b1 |
| QKV_POST (per-head RMSNorm + RoPE + paged cache write) | DONE | c92a70b1 |
| PAGED_ATTN (multi-SM with page distribution) | DONE | c92a70b1 |
| ATTN_REDUCE (LSE-corrected partial combination) | DONE | c92a70b1 |
| GEMV_OPROJ (distributed O-projection) | DONE | c92a70b1 |
| OPROJ_POST (all-warp norms + residual + pre-FFN norm) | DONE | c92a70b1 |
| Instruction preload into shared memory | DONE | c92a70b1 |
| Sliding window attention | DONE | 03aaa59d |
| K=V sharing (full-attention layers) | DONE | 2a303a3c |
| Proportional RoPE (full-attention layers) | DONE | 2a303a3c |

## Current Performance (April 2026)

| seq_len | Latency | vs Decomposed (~125 µs) |
|---------|---------|------------------------|
| 33-257 | 84-88 µs | 1.4-1.5x faster |
| 513 | 106 µs | 1.2x faster |
| 1025 | 136 µs | comparable |

## What's Left

### Next: AutoDeploy Integration (Phase 4)

Register as `@torch.library.custom_op`, FX graph transform to replace attention
subgraph in Gemma4 model, benchmark on real model inference. This is where we
prove the megakernel speeds up end-to-end serving. Est. 3-5 days.

### Next: Batch Size > 1

The CUDA kernel already indexes per-token (`token_id`), but the Python
instruction scheduler assumes 1 token. For batched decode:

- **GEMV**: Input becomes `[batch, HIDDEN_SIZE]`. Each SM handles a subset of
  `(token, output_row)` pairs. The shared memory input load needs to either
  iterate over tokens or use a larger smem buffer.
- **QKV_POST**: Each head instruction includes a token_id — works as-is for
  batch>1 if we issue separate instructions per token.
- **Attention**: Each SM handles one `(token, kv_head, page_range)` triple.
  Different tokens have different page tables and sequence lengths.
  The instruction builder needs per-token metadata.
- **Barriers**: Expected count stays at NUM_SMS (all SMs participate in every
  barrier regardless of batch size).

The kernel-side changes are moderate (GEMV batching). The scheduler complexity
is the main effort. Est. 1-2 weeks.

### Performance: TMA-Pipelined GEMV (est. 25-28 µs savings → ~56-60 µs total)

**The single biggest remaining performance optimization.**

Current GEMV loads weight data synchronously. TMA on H100 enables
hardware-managed async copies with multi-stage pipelining.

| Component | Current | With TMA | Savings |
|-----------|---------|----------|---------|
| QKV GEMV | 27 µs | 15-17 µs | 10-12 µs |
| O-proj GEMV | 16 µs | 8-10 µs | 6-8 µs |
| Fused QKV_POST (eliminate opcode + barrier) | 8 µs | 0 µs | 8 µs |
| **Total** | **51 µs** | **23-27 µs** | **24-28 µs** |

Why it can beat cuBLAS (22 µs for QKV):

- Zero kernel launch overhead (megakernel is already resident)
- Fused epilogue: norms/RoPE/cache-write in registers after dot product
- cuBLAS pays ~5 µs launch + can't fuse post-processing

Implementation: Integrate ThunderKittens, `tma::load_async` with 3-stage
pipeline, warp specialization (loader + consumer warps). Est. 1-2 weeks.

### Performance: Barrier Reduction (~10 µs savings)

Current: 4-5 barriers × ~4 µs = 16-20 µs.

Options:

- Embed barriers into opcodes (non-participating SMs get NOOP-with-barrier)
- Tree-reduction barrier (O(log N) instead of O(N) contention)
- Est. 2-3 days

### Performance: Register-Tiled Attention

Pre-load KV block into shared memory for short sequences (seq \< 128
with HEAD_DIM=256 fits in 192 KB smem). Eliminates repeated global
memory loads. Est. 1 week.

## Key Lessons Learned

1. **Unbalanced \_\_syncthreads() is UB**: Non-consumer warps must participate
   in all syncthreads inside opcode handlers before early-returning. Caused
   warp misalignment that broke OPROJ_POST (took hours to debug).

1. **Instruction preload >> per-instruction fetch**: Preloading all instructions
   into shared memory at kernel start reduced dispatch overhead from 5.7 µs
   to 0.2 µs per instruction (29x improvement).

1. **Barrier overhead was mostly dispatch overhead**: The 12 µs "per barrier"
   was actually 4 µs barrier + 8 µs dispatch. Fixing dispatch fixed barriers.

1. **Lamport barriers don't help on GPU**: Per-SM generation counters require
   O(N) reads per barrier (132 L2 reads). atomicAdd on single slot is faster
   despite serialized writes, because the spin-wait is O(1) reads.

1. **Multi-SM attention eliminates seq-len scaling**: Distributing pages across
   SMs makes attention time nearly constant regardless of sequence length.
