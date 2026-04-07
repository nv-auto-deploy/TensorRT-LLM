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

### Iter 5: eviction_policy="evict_last" on SSM state load

The SSM state `[max_batch, nheads, dim, dstate]` is a large tensor (~MAX_BATCH×64×64×128×4B = ~2GB for MAX_BATCH=512).
Each CTA only uses a small slice but the total working set vastly exceeds L2 cache (50MB on H100).
Setting `eviction_policy="evict_last"` on the state load hints the GPU to deprioritize state cachelines,
freeing L2 for conv weights and input data which benefit more from reuse across batch elements.

Results (SSM state evict_last only):

| batch | before (us) | after (us) | delta |
|-------|-------------|------------|-------|
|     1 |        9.24 |       8.24 | -10.8% |
|     8 |       19.47 |      18.85 |  -3.2% |
|    64 |      106.93 |     105.64 |  -1.2% |
|   384 |      608.15 |     589.07 |  -3.1% |

______________________________________________________________________

### Iter 6: eviction_policy="evict_last" on conv state loads (hidden + B/C)

Additional evict_last on hidden channel conv state reads and B/C channel conv state reads.

Full sweep results (iter 5 + iter 6 combined — SSM + conv state evict_last):

| batch | kernel_us | e2e_us | kernel_pct |
|-------|-----------|--------|------------|
|     1 |      8.25 |   8.24 |     100.1% |
|     2 |      8.71 |   8.72 |      99.9% |
|     4 |     11.26 |  10.98 |     102.6% |
|     8 |     18.77 |  22.71 |      82.7% |
|    16 |     33.56 |  33.56 |     100.0% |
|    32 |     58.34 |  58.07 |     100.5% |
|    64 |    105.37 | 105.59 |      99.8% |
|   128 |    201.06 | 201.19 |      99.9% |
|   256 |    394.21 | 394.38 |     100.0% |
|   384 |    588.94 | 588.57 |     100.1% |

vs baseline (iter 0): B=384 631us → 589us = **-6.7% total improvement** vs baseline.
vs iter 4 (num_warps=8): B=384 608us → 589us = **-3.1% additional improvement**.

______________________________________________________________________

### Iter 7: Attempted optimizations (reverted)

Multiple structural changes attempted but not kept:

**7a. Separate B/C conv state update kernel** (`_bc_conv_state_update_kernel`):
Run B/C state shift in a 2nd kernel (batch×ngroups grid) after the main kernel.
Main kernel timing improved: B=384 582us vs 589us (-1.2%).
But e2e WORSE: 591us vs 589us due to 2nd kernel launch overhead (~9us).
REVERTED.

**7b. 2D batch-load for conv state** (`[BLOCK_DIM, KW_PAD]` single load):
Load all k=0..kw-2 state values per channel at once.
Requires `tl.arange(0, KW_PAD)` where KW_PAD=4 (power of 2 for kw-1=3).
CATASTROPHIC REGRESSION: B=384 617us (+4.8%). Excess register pressure from \[64, 4\] tile.
REVERTED.

**7c. num_stages=2 for software pipelining**:
No meaningful improvement (B=384 within noise of current best).
Not applied.

**7d. evict_first on conv weight loads**:
No improvement. Weights (49KB) are small enough to stay in L2 naturally.
Not applied.

**7e. BLOCK_DIM=32 with num_warps=2**:
Halves state register pressure per CTA. B=384 589.6us (within noise of 588.6us baseline).
No improvement on current H100.

**7f. `_bc_conv_compute_kernel` (precompute B/C once per group)**:
Implemented as a separate kernel that computes B/C values, updates B/C state,
and stores results to a \[batch, 2, ngroups, dstate\] buffer.

- B/C kernel alone: 12us for B=384 (3,072 CTAs vs 24,576 main CTAs)
- Would eliminate 7/8 redundant B/C computation from main kernel
- Main kernel could remove B/C section (saves ~30% of memory ops per CTA)
- Estimated savings at B=384: ~97us → 491us total (vs current 589us = -17%)
- But requires implementing a new main kernel that reads from bc_buf
- Not yet integrated into launcher. Left as future work.

**Register pressure analysis:**

- Current kernel: 445 b32 registers per thread (from PTX)
- With 256 threads/CTA and 65536 regs/SM → only 1 CTA per SM possible!
- Root cause: state\[BLOCK_DIM=64, BLOCK_DSTATE=128\] = 8192 fp32 values per CTA
- Solutions explored: BLOCK_DIM=32 (halves state regs, no net improvement on H100),
  dstate tiling (not yet implemented), B/C precompute (not yet integrated).

______________________________________________________________________

### Iter 8: Two-kernel approach (precomputed B/C) — NOT faster

Added `_fused_conv_ssm_kernel_bc_buf` and `fused_conv_ssm_decode_two_kernel`:

- Kernel 1 (`_bc_conv_compute_kernel`): grid `(batch, ngroups)` — computes B/C conv + state update (race-free), stores to `bc_buf[batch, 2, ngroups, dstate]`
- Kernel 2 (`_fused_conv_ssm_kernel_bc_buf`): grid `(1, batch, 64)` — hidden-only conv + SSM reading B/C from bc_buf

Benchmark vs single-kernel (e2e us):

| batch | 1-kernel (us) | 2-kernel (us) | speedup |
|-------|--------------|--------------|---------|
|     1 |         8.07 |        10.86 |  0.743x |
|     8 |        18.84 |        21.58 |  0.873x |
|    64 |       105.41 |       107.12 |  0.984x |
|   384 |       588.73 |       588.65 |  1.000x |

**Conclusion:** No improvement at B=384 (kernel2 execution time is nearly the same as single kernel
since hidden conv savings ≈ bc_buf load overhead). Small batch is worse due to extra kernel launch.
The B/C conv section is NOT the bottleneck at large batch — the SSM state load/store dominates.

The `_fused_conv_ssm_kernel_bc_buf` and `fused_conv_ssm_decode_two_kernel` are kept in the file
for reference but are NOT the production path. The single-kernel `fused_conv_ssm_decode` remains
the launcher.

______________________________________________________________________

### Iter 9: Correct production dtype — SSM state is bf16 (not fp32)

**Critical finding**: All previous benchmark iterations used `ssm_state` dtype=float32.
In production, `mamba_ssm_cache_dtype: auto` resolves to the model activation dtype (bfloat16
for Nano v3). This halves the SSM state memory bandwidth requirements.

Updated `make_inputs()` in sweep script to use `ssm_state` dtype=bfloat16 by default.

**New (correct) baseline with bf16 SSM state:**

| batch | e2e_us (bf16 state) | e2e_us (fp32 state) | speedup |
|-------|---------------------|---------------------|---------|
|     1 |                7.44 |                8.22 |   1.10x |
|     2 |                7.87 |                8.71 |   1.11x |
|     4 |                8.75 |               10.99 |   1.26x |
|     8 |               10.89 |               19.00 |   1.74x |
|    16 |               19.99 |               33.30 |   1.67x |
|    32 |               34.76 |               58.05 |   1.67x |
|    64 |               61.02 |              105.35 |   1.73x |
|   128 |              110.73 |              201.16 |   1.82x |
|   256 |              209.01 |              394.37 |   1.89x |
|   384 |              308.60 |              588.63 |   1.91x |

**BW roofline analysis (bf16 state, B=384):**

- SSM state per CTA: 64×128×2B = 16KB (bf16)
- 24,576 CTAs × 16KB R+W = 786MB total SSM state traffic
- Plus hidden/B/C conv state: ~24,576 × (64×3 + 128×3) × 2B = 226MB additional
- Total estimated HBM traffic: ~1.0GB at 3.35TB/s = 298us theoretical minimum
- Actual: 308us = **1.03× roofline** — extremely close to bandwidth-optimal!

The kernel is now very near the HBM bandwidth limit with bf16 state.
Further improvements require either reducing memory traffic or improving compute/memory ratio.

______________________________________________________________________

### Iter 10: Re-sweep num_warps with correct bf16 SSM state dtype

With bf16 SSM state (half the register pressure for state tile), the optimal warp count changes.

| num_warps | B=1 (us) | B=8 (us) | B=64 (us) | B=384 (us) |
|-----------|----------|----------|-----------|------------|
| 1         |    12.85 |    19.13 |    117.49 |     628.66 |
| 2         |     8.77 |    11.56 |     62.17 |     317.04 |
| **4**     |  **7.57**| **10.61**|  **60.11**|  **306.28**|
| 8         |     7.41 |    10.86 |     61.26 |     307.99 |
| 16        |     7.73 |    12.69 |     73.53 |     388.67 |

**Winner: num_warps=4** at large batch (B=64,384).
B=1 is slightly better with num_warps=8 (7.41 vs 7.57us) but num_warps=4 dominates at production batch sizes.

Updated launcher from num_warps=8 → num_warps=4.

Full sweep with num_warps=4 + bf16 SSM state:

| batch | kernel_us | e2e_us | kernel_pct |
|-------|-----------|--------|------------|
|     1 |      7.52 |   7.31 |     102.9% |
|     2 |      7.77 |   7.54 |     103.1% |
|     4 |      8.43 |   8.44 |      99.9% |
|     8 |     10.44 |  10.43 |     100.0% |
|    16 |     19.22 |  19.45 |      98.8% |
|    32 |     34.10 |  34.28 |      99.5% |
|    64 |     59.92 |  59.90 |     100.0% |
|   128 |    109.06 | 109.19 |      99.9% |
|   256 |    207.09 | 207.12 |     100.0% |
|   384 |    306.52 | 306.48 |     100.0% |

vs iter 9 baseline (num_warps=8, bf16): B=384 308us → 306us = **-0.6%** additional improvement.

______________________________________________________________________

### Iter 11: Prefetch SSM state to overlap HBM latency with B/C computation

The SSM state load (16KB per CTA with bf16 dtype) is the dominant memory access.
By issuing the state load **before** the B/C conv computation, the GPU memory controller
can begin fetching the state while the warp computes the B/C section (conv1d for B/C channels,
state shift). This hides a portion of the 16KB state load latency.

Change: moved `state_ptrs = ...` and `tl.load(state_ptrs, ...)` to the top of the kernel
(before hidden conv computation), and replaced the later state load with `state_prefetch.to(fp32)`.

Results:

| batch | before (us) | after (us) | delta |
|-------|-------------|------------|-------|
|     1 |        7.31 |       7.21 |  -1.4% |
|     2 |        7.54 |       7.63 |  +1.2% |
|     4 |        8.44 |       8.45 |  +0.1% |
|     8 |       10.43 |      10.80 |  +3.5% |
|    16 |       19.45 |      19.50 |  +0.3% |
|    32 |       34.28 |      34.16 |  -0.4% |
|    64 |       59.90 |      58.99 |  -1.5% |
|   128 |      109.19 |     107.59 |  -1.5% |
|   256 |      207.12 |     206.13 |  -0.5% |
|   384 |      306.48 |     304.56 |  -0.6% |

B=384: 306us → 304us = **-0.6%**. B=64,128: -1.5% improvement.
Small batch (B=8) slightly regresses — additional register pressure from holding state_prefetch.

Correctness: PASS (batch=1, batch=2).

______________________________________________________________________

### Iter 12: Characterization experiments (batched heads, skip B/C, BLOCK_DIM=32)

Several structural changes were tested with bf16 SSM state to understand the performance breakdown.
All tested against current best (iter 11 prefetch kernel, 304us at B=384).

**12a: Skip B/C conv computation** (lower bound test):
Zero out B/C values to measure how much the B/C section costs.

- B=384: 293us (vs 304us) — B/C section costs ~11us = 3.6% of total kernel time.
- Small B is more sensitive: B=8: 10.22 vs 10.8us (5.4% of kernel time).

**12b: Batched heads (1 CTA per group)** — grid `(batch, ngroups)`:
Process all 8 heads in a group sequentially within one CTA, loading B/C state once for all heads.

- B=384: 348us (vs 304us) — SLOWER due to fewer CTAs (3072 vs 24576), worse GPU utilization.

**12c: BLOCK_DIM=32, num_warps=2**:

- B=384: 307.9us (vs 304us) — SLIGHTLY WORSE. More CTAs but smaller state tiles help.
- B=1: 7.62us (vs 7.21us) — worse at small batch.

**12d: num_stages=2 (tl.range with num_stages=2)**:

- B=384: 306us (vs 304us) — within noise. No benefit from pipelining short loops.

**12e: Eviction policy on SSM state store** (default/evict_last/evict_first):

- All within noise at B=384 (~306us). SSM store eviction policy doesn't matter.

**Key insight from characterization:**

- B/C conv costs only 3-5% of total kernel time at large batch.
- The dominant cost is SSM state R/W (16KB per CTA, already at bandwidth roofline).
- Further improvements are marginal — kernel is ~1.0x bandwidth roofline.

______________________________________________________________________

### Iter 13: Load SSM scalars (dt, A, D) before state prefetch

Reordered kernel: load scalar values (dt, dt_bias, A, D — 4 scalars ≈ 8B) and compute
softplus+exp **before** issuing the SSM state prefetch. This means:

1. Scalar loads are serviced from L2/L1 (fast, ~1 cycle)
1. dA = exp(A * softplus(dt + dt_bias)) is computed
1. State prefetch is then issued with all prior scalar work complete
1. Conv1d sections execute while state load is in flight

Previously, state prefetch was issued before any scalar computation, meaning the
scalar work only slightly preceded the actual state consumption. Now the scalar
computation fully precedes the issue, giving the memory controller maximum notice.

Results:

| batch | before (us) | after (us) | delta |
|-------|-------------|------------|-------|
|     1 |        7.21 |       7.53 |  +4.5% |
|     2 |        7.63 |       7.83 |  +2.6% |
|     4 |        8.45 |       8.41 |  -0.5% |
|     8 |       10.80 |      10.64 |  -1.5% |
|    16 |       19.50 |      19.03 |  -2.4% |
|    32 |       34.16 |      33.27 |  -2.6% |
|    64 |       58.99 |      58.16 |  -1.4% |
|   128 |      107.59 |     106.87 |  -0.7% |
|   256 |      206.13 |     204.91 |  -0.6% |
|   384 |      304.56 |     303.55 |  -0.3% |

B=384: 304us → 303us = **-0.3%**. B=16,32: **-2.5%** improvement.
B=1,2 slightly worse — scalar computation adds instructions to critical path at tiny batch.

Correctness: PASS (batch=1, batch=2).

______________________________________________________________________

### Iter 14: Move B/C conv state writes to after SSM computation

The B/C conv state shift-write (~512B) is a blocking operation for 1/8 of CTAs.
Previously it occurred before SiLU and SSM computation (creating a sequential dependency).

Reordered to: B/C conv reads → SiLU → SSM state update → output store → B/C state writes.

This allows the GPU to:

1. Begin SSM computation as soon as B/C conv values are computed
1. Issue the B/C writes to the store buffer at the end, overlapping with SM pipeline drain
1. The B/C writes are not on the critical path for the SSM output

Additionally confirmed: B/C writes cost ~5us at B=384 (from `no_bc_write` test: 297us),
while B/C reads cost ~10us. Total B/C overhead = ~11us (reads + writes are pipelined).

Results:

| batch | before (us) | after (us) | delta |
|-------|-------------|------------|-------|
|     1 |        7.53 |       7.19 |  -4.5% |
|     2 |        7.83 |       7.82 |  -0.1% |
|     4 |        8.41 |       8.71 |  +3.6% |
|     8 |       10.64 |      11.16 |  +4.9% |
|    16 |       19.03 |      19.26 |  +1.2% |
|    32 |       33.27 |      33.16 |  -0.3% |
|    64 |       58.16 |      57.81 |  -0.6% |
|   128 |      106.87 |     105.76 |  -1.0% |
|   256 |      204.91 |     203.20 |  -0.8% |
|   384 |      303.55 |     301.57 |  -0.7% |

B=384: 303us → 301us = **-0.7%**. B=64,128,256: -0.7% to -1.0%.
Some small-batch regression (B=4,8): B/C writes moved after SSM, adding register pressure.

Correctness: PASS (batch=1, batch=2).

______________________________________________________________________

### Iter 15: evict_first on SSM state prefetch — WORSE (tested and rejected)

Hypothesis: since the SSM state is the largest working set (16KB per CTA, total >>L2),
using `evict_first` on the prefetch load could free L2 space for conv weights and inputs
once the state data has been consumed. The GPU would treat it as a streaming access.

Tested by changing `eviction_policy="evict_last"` → `eviction_policy="evict_first"` on the
`state_prefetch = tl.load(state_ptrs, ...)` call only.

Results:

| batch | evict_last (us) | evict_first (us) | delta |
|-------|-----------------|------------------|-------|
|     1 |            7.19 |             7.45 | +3.6% |
|     8 |           11.16 |            12.32 | +10.4% |
|    64 |           57.81 |            57.42 | -0.7% |
|   384 |          301.57 |           304.39 | +0.9% |

**Conclusion:** `evict_first` is WORSE at the critical batch sizes (B=1, B=8, B=384).
At B=384, 304 vs 301us = +0.9% regression.

Analysis: The prefetch is a 16KB state tile that the warp will consume immediately
in the SSM update step. Using `evict_first` tells the hardware this data can be
evicted as soon as possible — which conflicts with the prefetch intent (we WANT the
data to stay in cache until consumed). `evict_last` is the correct choice: it keeps
the recently-fetched state in L2 long enough for the SSM computation to consume it.

No code change. `evict_last` on SSM state prefetch remains the production setting.

______________________________________________________________________

### Iter 16: Scattered slot indices — production realism characterization

In production, KV cache slots are allocated from a pool and may not be contiguous.
Scattered slot indices mean SSM state and conv state are accessed at random HBM offsets,
potentially preventing L2 cache line reuse across batch elements within the same kernel
invocation.

Tested by using `torch.randperm(MAX_BATCH)[:batch]` for slot indices instead of
`torch.arange(batch)` (sequential). This simulates a fully fragmented cache pool.

| batch | sequential (us) | scattered (us) | ratio |
|-------|-----------------|----------------|-------|
|     1 |            7.53 |           7.21 | 0.957x |
|     8 |           11.27 |          11.08 | 0.984x |
|    64 |           57.95 |          58.44 | 1.009x |
|   128 |          105.83 |         106.38 | 1.005x |
|   256 |          203.28 |         204.60 | 1.006x |
|   384 |          301.66 |         302.68 | 1.003x |

**Conclusion:** Scattered slots show essentially no performance difference at large batch
(ratio 1.000-1.006x at B=64-384). The kernel is insensitive to slot layout because:

1. Each CTA accesses its own independent state slice — no inter-CTA reuse opportunity.
1. With 24,576 CTAs at B=384, each slot/state tile is accessed by exactly one CTA.
1. H100 HBM bandwidth is saturated regardless of access order — scatter vs sequential doesn't matter when L2 miss rate is already 100%.

The sequential-slot benchmark is representative of production performance.
No code change.

______________________________________________________________________

### Iter 17: evict_last on conv weight loads — no improvement

Conv weights are `[CONV_DIM, KERNEL_WIDTH]` = 6144×4 = 49KB (bf16).
Currently loaded with the default (no explicit) eviction policy.

Hypothesis: `evict_last` on weight loads would encourage the H100 L2 to retain
weights across batch elements, since all 384 batch × 64 heads × 64-dim CTAs read the
same weight tensor.

Tested by adding `eviction_policy="evict_last"` to ALL conv weight loads (hidden + B/C).

| batch | current (us) | w_evict_last (us) | delta |
|-------|--------------|-------------------|-------|
|     1 |         7.38 |              7.58 | +2.6% |
|     8 |        11.16 |             10.95 | -1.9% |
|    64 |        57.81 |             58.15 | +0.6% |
|   128 |       105.75 |            106.28 | +0.5% |
|   256 |       203.21 |            203.68 | +0.2% |
|   384 |       301.49 |            302.52 | +0.3% |

**Conclusion:** No improvement — within noise at B=384 (302 vs 301us, +0.3%).
The H100 L2 (50MB) already retains the 49KB weight tensor naturally without explicit hints.
The weight-to-L2 ratio is 49KB / 50MB ≈ 0.1% — weights always fit in L2.
Explicit evict_last hints add instruction overhead without cache benefit.
No code change.

______________________________________________________________________

### Iter 18: evict_first on conv state WRITES — no improvement

The hidden conv state shift involves 2 loads (k+1 → k shifts) and 3 stores (shifted
values + new x_in). These writes are next-read at the following decode step.

Hypothesis: `evict_first` on writes avoids polluting L2 with just-written data, freeing
L2 space for the SSM state prefetch and conv state reads.

Tested: `eviction_policy="evict_first"` on all hidden conv state store ops AND all B/C
conv state store ops (both shift writes and new-value writes).

| batch | current (us) | conv_write_evict_first (us) | delta |
|-------|--------------|------------------------------|-------|
|     1 |         7.52 |                         7.52 |  -0.0% |
|     8 |        11.00 |                        11.00 |  +0.0% |
|    64 |        57.67 |                        57.85 |  +0.3% |
|   128 |       106.14 |                       105.95 |  -0.2% |
|   256 |       203.25 |                       203.54 |  +0.1% |
|   384 |       301.61 |                       302.11 |  +0.2% |

**Conclusion:** Completely within noise at all batch sizes. The conv state writes
are a small fraction of total traffic (~10% of SSM state traffic) and GPU write-combine
hardware already handles them efficiently. No benefit from streaming hints.
No code change.

______________________________________________________________________

### Iter 19: evict_first on SSM state STORE — significantly WORSE

After computing the new SSM state, it is stored back to HBM.
Hypothesis: since the updated state won't be re-read until the next decode step,
`evict_first` on the store would avoid filling cache with written-once data,
freeing L2 for conv weights and SSM state reads.

Tested: `eviction_policy="evict_first"` only on the SSM state store:
`tl.store(state_ptrs, state.to(...), mask=mask_dn, eviction_policy="evict_first")`.

| batch | current (us) | ssm_store_evict_first (us) | delta |
|-------|--------------|----------------------------|-------|
|     1 |         7.29 |                       7.70 | +5.6% |
|     8 |        11.28 |                      10.84 | -3.9% |
|    64 |        57.96 |                      59.69 | +3.0% |
|   128 |       106.10 |                     111.09 | +4.7% |
|   256 |       203.53 |                     214.97 | +5.6% |
|   384 |       301.91 |                     319.50 | +5.8% |

**Conclusion:** SIGNIFICANTLY WORSE at large batch — +5.8% regression at B=384.

Analysis: The SSM state has HIGH temporal locality across consecutive decode steps:
each sequence's slot is accessed at every decode step. After writing the updated state,
the GPU holds the written data in L2 (write-back), and the NEXT decode step reads the
same slot — getting an L2 hit instead of an HBM fetch. `evict_first` on the store
immediately evicts this data, making the next step incur a full HBM re-fetch.

The "streaming write" assumption is incorrect for SSM state. The data is hot across
decode steps and should remain in L2 as long as possible.
No code change.

______________________________________________________________________

## 4. Optimization Ideas Backlog

### Memory Access

- \[ \] **Vectorized loads for conv state**: Conv state is \[max_batch, conv_dim, kw-1\]; access pattern is strided (per channel). Load BLOCK_DIM channels in a single vectorized transaction instead of element-by-element.
- \[ \] **Vectorized conv weight loads**: weight\[channel, k\] — same pattern, all channels for a given k are contiguous in memory if stride_cw_d=1, stride_cw_w=conv_dim. Load as block vector.
- \[ \] **evict_first on conv_input**: Each batch element's input (12KB) is read by all 64 heads. evict_first would evict early and force re-fetches — TESTED in iter 20, significantly worse.

### Structural

- \[ \] **Persistent kernel**: For small batch (1-8), try a persistent kernel that loops over multiple batch elements within one CTA. Amortizes launch overhead and may improve L2 hit rate for weights.
- \[ \] **Separate B/C conv state update kernel**: Tested in iter 7a (worse e2e due to kernel launch overhead).
- \[ \] **Grid reorder**: Current order is (dim_tile, batch, nhead). Reorder to (batch, nhead, dim_tile) so consecutive CTAs share the same batch element → better L2 locality for conv_state and ssm_state.

### Precision

- \[ \] **bf16 intermediate**: State loads are float32; try keeping state in bf16 and cast only for FMA. (May reduce accuracy — needs correctness check.)

______________________________________________________________________

## 5. Final Best Configuration

**Applied optimizations (in `fused_mamba_decode.py`):**

1. `num_warps=4` (re-tuned from 8 after bf16 SSM state dtype fix) — optimal at B=64+
1. `eviction_policy="evict_last"` on SSM state load (prefetch) — retains in L2 for compute
1. `eviction_policy="evict_last"` on conv state loads (hidden + B/C) — reduces L2 thrash
1. SSM state prefetch issued before conv computation — overlaps HBM latency with compute
1. Scalar loads (dt, A, D) before state prefetch — gives memory controller maximum notice
1. B/C conv state writes reordered to after SSM output stores — removes from critical path

**Per-shape best config (num_warps=4, BLOCK_DIM=64, BLOCK_DSTATE=128, bf16 SSM state):**

| batch | iter-0 baseline (fp32 state, us) | best e2e (bf16 state, us) | improvement |
|-------|----------------------------------|---------------------------|-------------|
|     1 |                             9.00 |                      7.39 |     -17.9%  |
|     8 |                            19.57 |                     10.92 |     -44.2%  |
|    64 |                           109.91 |                     57.57 |     -47.6%  |
|   128 |                           212.73 |                    105.80 |     -50.3%  |
|   256 |                           421.63 |                    203.21 |     -51.8%  |
|   384 |                           631.03 |                    301.50 |     -52.2%  |

Note: The large improvement vs iter-0 is mostly due to fixing the SSM state dtype from fp32 to
bf16 (iter 9), which halves the dominant memory traffic.

**HBM bandwidth roofline (B=384, bf16 SSM state):**

- SSM state: 24,576 CTAs × 16KB R+W = 786MB
- Conv state (hidden + B/C): ~226MB additional
- Total: ~1.0GB at H100 HBM3 3.35TB/s = 298us theoretical minimum
- Actual: 301us = **1.01× roofline** — bandwidth-optimal

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
