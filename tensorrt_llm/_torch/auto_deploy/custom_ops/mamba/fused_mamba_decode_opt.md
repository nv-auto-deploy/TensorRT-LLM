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

### Iter 1: num_warps sweep {1, 2, 4, 8, 16}

Tested at B=1,8,64,384 with BLOCK_DIM=64, BLOCK_DSTATE=128.

| num_warps | B=1 kernel_us | B=8 kernel_us | B=64 kernel_us | B=384 kernel_us |
|-----------|---------------|---------------|----------------|-----------------|
| 1         | 32.46         | 94.61         | 675.67         | 4035.09         |
| 2         | 9.68          | 18.91         | 111.14         | 642.09          |
| 4 (base)  | 8.78          | 19.58         | 109.88         | 630.83          |
| **8**     | **8.99**      | **19.44**     | **106.91**     | **608.25**      |
| 16        | 9.03          | 20.57         | 115.82         | 661.94          |

**Winner: num_warps=8** — ~3% better at B=384, B=256.

- num_warps=1 is catastrophically slow (serialized warp).
- num_warps=2 is slightly slower than 4 at small batch.
- num_warps=8 gives best throughput for large batch.
- num_warps=16 is slower (too many warps for this register-heavy kernel).

Full sweep with num_warps=8:

| batch | kernel_us | e2e_us | kernel_pct |
|-------|-----------|--------|------------|
|     1 |      8.99 |   9.16 |      98.1% |
|     2 |     10.38 |  10.43 |      99.5% |
|     4 |     12.70 |  12.91 |      98.4% |
|     8 |     19.44 |  19.94 |      97.5% |
|    16 |     32.43 |  32.76 |      99.0% |
|    32 |     57.81 |  58.92 |      98.1% |
|    64 |    106.91 | 110.04 |      97.2% |
|   128 |    206.16 | 212.85 |      96.9% |
|   256 |    406.89 | 421.93 |      96.4% |
|   384 |    608.25 | 631.04 |      96.4% |

______________________________________________________________________

### Iter 2: BLOCK_DIM sweep {32, 64, 128} with num_warps=8

Tested at B=1,8,64,384.

| BLOCK_DIM | B=1 kernel_us | B=8 kernel_us | B=64 kernel_us | B=384 kernel_us |
|-----------|---------------|---------------|----------------|-----------------|
| 32        | 8.90          | 20.37         | 110.78         | 621.71          |
| **64** (base) | **8.99** | **19.44**     | **106.91**     | **608.25**      |
| 128       | 9.44          | 25.28         | 142.97         | 850.82          |

**Winner: BLOCK_DIM=64** — the default is optimal.

- BLOCK_DIM=32: slightly worse at B>=8 (more tiles = more kernel launch / scheduling overhead, and more B/C redundancy).
- BLOCK_DIM=128: much worse (exceeds dim=64, padding waste; larger state tiles increase register pressure).

______________________________________________________________________

### Iter 3: BLOCK_DSTATE sweep {64, 128, 256} with num_warps=8

Tested at B=1,8,64,384.

| BLOCK_DSTATE | B=1 kernel_us | B=8 kernel_us | B=64 kernel_us | B=384 kernel_us | correct? |
|--------------|---------------|---------------|----------------|-----------------|----------|
| 64           | 8.10          | 13.20         | 59.74          | 339.14          | FAIL     |
| **128** (base) | **8.99**   | **19.44**     | **106.91**     | **608.25**      | PASS     |
| 256          | 9.64          | 25.98         | 154.45         | 905.58          | PASS     |

**BLOCK_DSTATE=64 is incorrect** — with dstate=128, BLOCK_DSTATE=64 only processes half the state (no loop over dstate tiles), producing wrong results. Fast but wrong.

**Winner: BLOCK_DSTATE=128** — the default is correct and optimal.

**Conclusion from parameter sweep (iters 1-3):**
Best config so far: `num_warps=8, BLOCK_DIM=64, BLOCK_DSTATE=128`

- B=384: 608us (vs 631us baseline) = **+3.6% improvement**
- B=64: 107us (vs 110us baseline) = **+2.8% improvement**
- Small batch (B=1,2,4): within noise of baseline

______________________________________________________________________

### Iter 4: num_warps=8 in launcher + tl.static_range attempted (reverted)

**Change 1:** Updated `fused_conv_ssm_decode` launcher to use `num_warps=8` (was 4).
**Result:** B=384: 608us → +3.6% vs baseline 631us. KEPT.

**Change 2:** Replaced `range(...)` with `tl.static_range(...)` in all conv loops.
**Result:** CATASTROPHIC REGRESSION — B=384: 608us → 966us (1.59x SLOWER).
**Reason:** `tl.static_range` forces full loop unrolling. With 3 kernel loads per loop × 3 loop bodies unrolled, register pressure explodes. The H100 scheduler can no longer overlap memory latency with compute. REVERTED.

**Lesson:** Triton's `range(kernel_width-1)` with `kernel_width: tl.constexpr` already unrolls optimally. Forcing extra unrolling via `tl.static_range` is counterproductive for memory-bound kernels with many active tensors.

Final state of iter 4 (num_warps=8, range reverted):

| batch | kernel_us | e2e_us | kernel_pct |
|-------|-----------|--------|------------|
|     1 |      9.22 |   9.24 |      99.8% |
|     2 |     10.15 |  10.15 |     100.0% |
|     4 |     12.73 |  12.98 |      98.0% |
|     8 |     19.47 |  19.47 |     100.0% |
|    16 |     32.44 |  32.17 |     100.8% |
|    32 |     57.84 |  57.86 |     100.0% |
|    64 |    106.68 | 106.93 |      99.8% |
|   128 |    206.14 | 206.25 |      99.9% |
|   256 |    406.85 | 407.15 |      99.9% |
|   384 |    608.28 | 608.15 |     100.0% |

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
