# Fused Mamba Decode Kernel Optimization Log

## 1. Kernel Overview

**File:** `fused_mamba_decode.py`
**Kernel:** `_fused_conv_ssm_kernel`
**Caller:** `fused_cached_conv_ssm.py` → `fused_conv_ssm_decode()`

### What the kernel does

Fuses two operations that were previously separate CUDA kernels + HBM round-trips:

1. **Conv1d update** (depthwise, width=4) for hidden channels + B/C channels
1. **SiLU activation** on hidden channels and B/C
1. **SSM state update**: `state = state * dA + dB * x; out = sum(state * C) + x * D`

### Grid layout

```
Grid: (cdiv(dim, BLOCK_DIM), batch, nheads)
```

- `pid_d` = dim tile (0..cdiv(dim,BLOCK_DIM)-1)
- `pid_b` = batch index
- `pid_h` = head index

### Work per program instance

- Loads conv state for `BLOCK_DIM` hidden channels × 3 past timesteps
- Loads conv weights for those channels × 4 timesteps (kernel_width=4)
- Loads conv state + weights for `BLOCK_DSTATE` B and C channels × 4 timesteps (all heads in a group)
- Computes SSM state update: loads `[BLOCK_DIM, BLOCK_DSTATE]` state tile, updates, stores back
- Only `pid_h % nheads_per_group == 0` writes B/C conv state (race condition avoidance)

### Inputs / Outputs

| Tensor | Shape | Notes |
|--------|-------|-------|
| conv_input | \[batch, conv_dim\] | bf16 |
| conv_state | \[max_batch, conv_dim, kw-1\] | slot-indexed |
| conv_weight | \[conv_dim, kernel_width\] | |
| conv_bias | \[conv_dim\] | |
| dt | \[batch, nheads\] | |
| dt_bias | \[nheads\] | |
| A | \[nheads\] | |
| D | \[nheads\] | |
| ssm_state | \[max_batch, nheads, dim, dstate\] | slot-indexed, large |
| out | \[batch, nheads, dim\] | bf16, written |

______________________________________________________________________

## 2. Target Models & Benchmark Shapes

### Nemotron Nano v3 (30B MoE, Mamba-hybrid)

| Param | Value |
|-------|-------|
| nheads | 64 |
| dim | 64 |
| dstate | 128 |
| ngroups | 8 |
| nheads_per_group | 8 |
| kernel_width | 4 |
| intermediate_size | nheads×dim = 4096 |
| conv_dim | 4096 + 2×8×128 = 6144 |
| BLOCK_DIM (default) | next_power_of_2(64) = 64 |
| BLOCK_DSTATE (default) | next_power_of_2(128) = 128 |

Grid shape per layer (decode): `(1, batch, 64)` — only 1 dim tile since BLOCK_DIM=dim=64.

### Decode batch sizes from cuda_graph_batch_sizes

`[1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 320, 384]`

______________________________________________________________________

## 3. Optimization Iterations

### Iter 0: Baseline

- Default params: `num_warps=4, BLOCK_DIM=64, BLOCK_DSTATE=128`
- Grid: `(1, batch, 64)` for Nano v3 (dim=64 so only 1 dim tile)

| batch | kernel_us | e2e_us | kernel_pct |
|-------|-----------|--------|------------|
|     1 |      8.78 |   9.00 |      97.5% |
|     2 |     10.23 |  10.02 |     102.1% |
|     4 |     12.50 |  12.52 |      99.9% |
|     8 |     19.58 |  19.57 |     100.0% |
|    16 |     32.60 |  32.35 |     100.8% |
|    32 |     58.75 |  58.78 |      99.9% |
|    64 |    109.88 | 109.91 |     100.0% |
|   128 |    212.57 | 212.73 |      99.9% |
|   256 |    421.69 | 421.63 |     100.0% |
|   384 |    630.83 | 631.03 |     100.0% |

Key observations:

- kernel_pct ~100%: the Triton kernel IS the bottleneck (almost all e2e time is the kernel).
- Launch overhead is negligible (~0.2us overhead).
- Performance scales linearly with batch: batch=1→8us, batch=384→631us (~79x for 384x batch).
- At batch=1, ideal linear scaling would be 1/384 * 631us = 1.64us — we're at 8.78us (5.3x overhead), suggesting significant fixed cost per kernel launch for small batch.
- B=1 fixed cost analysis: 8.78us for 64 heads × 64 dim × 128 dstate = ~537K FP32 state elements = 2.15MB state, plus 6144 conv_dim × 3 state elements = 73KB conv state. HBM bandwidth: 3.35TB/s. Minimum BW-roofline for B=1: ~(2.15MB + 73KB) × 2 (R+W) / 3.35TB/s ≈ 1.3us. So we're ~6.7x over roofline at B=1.

**B/C conv state race condition note:**
The kernel has a pre-existing write-read race for B/C conv state when `nheads_per_group > 1`. For Nano v3 (nheads_per_group=8), only head-0 per group writes the B/C state shift, but all 8 heads read it concurrently. This makes multi-batch results non-deterministic. The correctness benchmark only tests batch=1,2 (deterministic cases).

______________________________________________________________________

## 4. Optimization Ideas Backlog

### Memory Access

- \[ \] **Vectorized loads for conv state**: Conv state is \[max_batch, conv_dim, kw-1\]; access pattern is strided (per channel). Load BLOCK_DIM channels in a single vectorized transaction instead of element-by-element.
- \[ \] **Vectorized conv weight loads**: weight\[channel, k\] — same pattern, all channels for a given k are contiguous in memory if stride_cw_d=1, stride_cw_w=conv_dim. Load as block vector.
- \[ \] **Eviction hints on SSM state**: State is \[max_batch, nheads, dim, dstate\] — large tensor. Use `eviction_policy="evict_last"` to avoid thrashing L2 with state data and keep other data cached.
- \[ \] **Eviction hints on conv state**: Similar argument — conv state is large. Try evict_last on state reads.

### Compute

- \[ \] **Precompute dA once per head** (currently every pid_d in the same head computes the same scalar dA). In the current grid, each pid_h already maps to one head, so dA is computed once per CTA — already fine. No issue here.
- \[ \] **Static loop unrolling**: Use `tl.static_range(kernel_width - 1)` instead of `range(kernel_width - 1)`. With kernel_width as constexpr, Triton may already unroll, but explicit static range makes it guaranteed.
- \[ \] **Precompute B/C channel base pointers**: Reduce redundant address arithmetic in inner loops.

### Parallelism / Occupancy

- \[ \] **num_warps tuning**: Default is 4. For decode (small batch), 2 warps may be better (less register spilling). For large batch, 4-8 warps may improve occupancy. Sweep: {1, 2, 4, 8}.
- \[ \] **BLOCK_DIM variants**: 32 (more tiles, more parallelism) vs 64 (default) vs 128 (fewer tiles, more work per CTA). With dim=64 and BLOCK_DIM=64, only 1 tile per head already — going to 32 doubles the grid.
- \[ \] **BLOCK_DSTATE variants**: 64 (halved; smaller state tile), 128 (default), 256 (oversized, padded). With dstate=128, BLOCK_DSTATE=128 is exact. Try 64 with two passes (if it reduces register pressure).

### Structural

- \[ \] **Persistent kernel**: For small batch (1-8), try a persistent kernel that loops over multiple batch elements within one CTA. Amortizes launch overhead and may improve L2 hit rate for weights.
- \[ \] **Separate B/C conv state update kernel**: The `if pid_h % nheads_per_group == 0` guard means 7/8 heads skip the B/C store — wasted divergence. Alternative: separate small kernel just for B/C conv state update (ngroups × BLOCK_DSTATE threads per batch).
- \[ \] **Grid reorder**: Current order is (dim_tile, batch, nhead). Reorder to (batch, nhead, dim_tile) so consecutive CTAs share the same batch element → better L2 locality for conv_state and ssm_state.
- \[ \] **Load B/C conv values once per group**: Currently every head in a group independently loads and computes B/C convolution. Only the first head stores the result. Optimization: compute B/C conv only for the first head, broadcast via shared memory or restructure grid.
- \[ \] **num_stages tuning**: Software pipelining in loops. Try num_stages=2 vs default (1).

### Precision

- \[ \] **bf16 intermediate**: State loads are float32; try keeping state in bf16 and cast only for FMA. (May reduce accuracy — needs correctness check.)

______________________________________________________________________

## 5. Final Best Configuration

TBD after iterations complete.

______________________________________________________________________

## 6. Appendix: How to Reproduce

### Environment

- GPU: NVIDIA H100 80GB HBM3
- PyTorch: 2.11.0a0+eb65b36914.nv26.02
- Triton: 3.6.0
- Branch: `ad-fast-iter/nemotron-nano-20260323`

### Benchmark

```bash
cd /path/to/TensorRT-LLM
python tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/sweep_fused_mamba_decode.py
```

### Correctness check

```bash
python tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/sweep_fused_mamba_decode.py --correctness
```

### Custom tuning params

```bash
python tensorrt_llm/_torch/auto_deploy/custom_ops/mamba/sweep_fused_mamba_decode.py \
    --num_warps 2 --block_dim 64 --block_dstate 128
```
