# Triton Kernel Optimization: Patterns & Catalog

---

## Bottleneck Classification

Classify the kernel before optimizing. Different bottlenecks require different strategies.

### How to classify

**Step 1: Estimate arithmetic intensity (AI)**
```
AI = FLOP_per_element / bytes_per_element_of_memory_traffic
```

For a simple elementwise kernel (e.g., RMSNorm scale):
- FLOPs per element: ~3–5 (mul, add, rsqrt)
- Bytes: 4 (fp32 load) + 4 (fp32 store) = 8
- AI ≈ 0.5 FLOP/byte → very memory-bound

For a matmul (M=N=K=4096, fp16):
- FLOPs: 2*M*N*K = 137B
- Bytes: (M*K + K*N)*2 + M*N*2 = ~134MB
- AI ≈ 1024 FLOP/byte → compute-bound on H100, memory-bound on A10

**Step 2: Compare to GPU roofline**

| GPU | Mem BW (GB/s) | TF32 TFLOPS | FP16 TFLOPS | Ridge point (FP16) |
|-----|---------------|-------------|-------------|-------------------|
| H100 SXM | 3350 | 989 | 1979 | ~591 FLOP/byte |
| A100 80G | 2000 | 312 | 624 | ~312 FLOP/byte |
| A10 | 600 | 125 | 250 | ~417 FLOP/byte |
| RTX 4090 | 1008 | 330 | 660 | ~654 FLOP/byte |

If `AI < ridge_point`: **memory-bound** — focus on A.1, A.2, A.6 categories.
If `AI > ridge_point`: **compute-bound** — focus on A.3, A.5 categories.
If kernel is tiny (< 1µs): likely **launch-overhead-bound** — focus on A.4 (fusion) or A.5 (persistent kernels).

### Symptoms by bottleneck type

| Bottleneck | Symptoms | Primary fixes |
|------------|----------|---------------|
| Memory bandwidth | Increasing `num_warps` beyond 4 gives no benefit; larger BLOCK sizes help modestly | Coalesced access, vectorized loads, layout transforms |
| Compute | Increasing `num_stages` helps a lot (hides compute latency); FP16 vs FP32 matters | Mixed precision, fast math, loop unrolling |
| Register pressure | High occupancy hurts (compiler report shows >64 regs/thread); weird perf cliffs | Smaller tiles, fewer live variables, split kernel |
| Launch overhead | Profiling shows kernel is < 5µs but E2E is much slower | Persistent kernels, kernel fusion |
| Shared memory | SMEM utilization near 100%; reducing `num_stages` helps | Smaller tiles, fewer stages, avoid SMEM where possible |

---

## Optimization Catalog

Use this to build the per-kernel backlog in Step 0.5. Mark ideas as applicable/not-applicable for your kernel.

### A.1 — Memory Access Patterns

- [ ] **Coalesced loads/stores**
  Thread `i` in a warp should access address `base + i`, not `base + i*stride`. Check all `tl.load`/`tl.store` index expressions.
  *Relevant when:* Input tensor has non-unit innermost stride, or grid maps threads to non-contiguous locations.
  *Quick check:* Print the address for lane 0 and lane 1 — they should differ by exactly 1 element.

- [ ] **Vectorized loads (`tl.load` with `eviction_policy`)**
  Wider loads (e.g., loading 8 fp16 values at once) amortize address computation and increase memory throughput.
  *How:* Use `BLOCK_SIZE` that is a multiple of 8 (for fp16) or 4 (for fp32); ensure base pointer is aligned.
  *Relevant when:* Memory-bound kernel with simple stride-1 access.

- [ ] **Reduce redundant global loads**
  Cache frequently reused values in registers or shared memory. Triton handles register allocation, but you must structure the code to keep values alive across iterations.
  *Pattern:* If the same global tensor row is accessed in multiple loop iterations, load it once before the loop.

- [ ] **Software pipelining (`num_stages > 1`)**
  Overlaps async memory loads with computation. Most effective when each loop iteration loads data and then computes with it.
  *How:* Increase `num_stages` in `tl.constexpr` or the autotune config. Usually 2–4 is optimal.
  *Relevant when:* Kernel has a loop with `tl.load` followed by arithmetic — the prototypical matmul pattern.

- [ ] **L2 cache-friendly tile ordering (swizzle)**
  Reorder which CTA processes which tile to maximize L2 reuse across thread blocks.
  *Pattern:*
  ```python
  # Swizzled 2D tile mapping (standard for matmul-like kernels)
  GROUP_M = 8
  pid = tl.program_id(0)
  num_pid_m = tl.cdiv(M, BLOCK_M)
  num_pid_n = tl.cdiv(N, BLOCK_N)
  num_pid_in_group = GROUP_M * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * GROUP_M
  group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  ```
  *Relevant when:* 2D tiling with large M and N; standard for fused attention, MoE, and GEMM-like kernels.

### A.2 — Tiling & Grid Strategies

- [ ] **Tile size tuning**
  Larger tiles → more reuse per launch, less grid overhead, more registers (may reduce occupancy).
  Smaller tiles → higher occupancy, less register pressure, less reuse.
  *Try:* BLOCK_M/N powers of 2 from 16 to 256. Profile occupancy at each.

- [ ] **2D or 3D program ID grid**
  If the kernel uses a flat 1D grid over a 2D/3D problem, a 2D grid can expose more parallelism and simplify index math.
  *When to try:* Grid size < number of SMs (low occupancy / parallelism).

- [ ] **Persistent kernels**
  Instead of one program per tile, launch N_SM programs that each iterate over multiple tiles. Eliminates grid launch overhead and improves L2 reuse within a program.
  *Tradeoff:* More complex code; wave quantization artifacts become explicit.
  *Pattern:*
  ```python
  pid = tl.program_id(0)
  num_programs = tl.num_programs(0)
  for tile_id in range(pid, total_tiles, num_programs):
      # compute tile_id → (tile_m, tile_n)
      ...
  ```

- [ ] **Split-K / split-reduction**
  Split the reduction dimension across multiple programs. Each program computes a partial reduction; a final kernel sums them.
  *When to try:* Small M, N but large K (reduction-bound); common in decode attention and small GEMM.

- [ ] **Wave quantization awareness**
  If `total_tiles % num_SMs != 0`, the last wave is under-utilized. Try adjusting BLOCK sizes so that `total_tiles` is a multiple of `num_SMs` (or as close as possible).

### A.3 — Compute Optimizations

- [ ] **Mixed precision compute**
  Load in fp16/bf16, upcast accumulators to fp32, downcast outputs. This is the default pattern for correctness, but ensure inputs are not unnecessarily upcast too early.
  *Pattern:* `acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)` inside the kernel, then `output = acc.to(tl.float16)` before storing.

- [ ] **Fast math approximations**
  `tl.math.fast_expf`, `tl.math.fast_dividef`, `tl.extra.cuda.libdevice.fast_rsqrt` etc.
  *When:* Softmax, sigmoid, LayerNorm, and any exp/log/sigmoid in inner loops. Check accuracy requirements first.

- [ ] **Strength reduction**
  Replace `a / b` with `a * tl.math.fast_dividef(1.0, b)` or precompute `inv_b = 1.0 / b` outside the loop.
  Replace modulo with bitwise `&` when divisor is power-of-2.

- [ ] **Static range unrolling**
  ```python
  for i in tl.static_range(UNROLL_FACTOR):
      ...
  ```
  Forces the compiler to unroll a fixed-trip-count loop. Eliminates loop overhead and enables better instruction scheduling.
  *When:* Known small trip count (≤ 8 iterations). Be careful with register pressure.

- [ ] **Instruction-level parallelism (ILP)**
  Process multiple independent elements per thread (increase work per program without changing the grid). The compiler can then overlap independent instructions.
  *Pattern:* Double or quadruple the BLOCK size and let a single program handle more elements.

- [ ] **Avoid unnecessary intermediate materialization**
  Fuse `x = tl.load(...)`, `y = some_op(x)`, `tl.store(..., y)` into one sequence. Do not write intermediates back to global memory unless needed for multi-kernel communication.

### A.4 — Kernel Fusion & Fission

- [ ] **Fuse adjacent kernels**
  If two kernels run back-to-back on the same data (e.g., elementwise scale followed by bias add), combine into one to eliminate the intermediate global memory round-trip.
  *When:* Both kernels are memory-bound; the fused computation fits in registers; no shape-incompatibility.

- [ ] **Absorb surrounding torch ops into the kernel**
  Look at the Python launcher for small ops done before/after the Triton kernel: `reshape`, `transpose`, `.contiguous()`, small elementwise ops. These can often be absorbed into the kernel's index math.
  *Example:* A `.transpose(1, 2)` before the kernel can be replaced by swapping two stride variables in the index computation.

- [ ] **Epilogue fusion**
  Common pattern: after a matmul-like kernel, add bias, apply activation, or scale. Fuse these into the kernel's output writeback loop. Already done in most good GEMM kernels; verify this kernel does it.

- [ ] **Split an overly complex kernel**
  If a single kernel has poor occupancy (> 96 registers/thread) or does logically distinct passes, split it. The two kernels may each be faster than one monolithic kernel.

### A.5 — Parallelism & Occupancy

- [ ] **Expose more parallelism**
  If `total_tiles < num_SMs`, the GPU is underutilized. Increase the number of programs by splitting a dimension further, adding batch parallelism, or using Split-K.

- [ ] **Reduce register pressure**
  Each register saved potentially allows one more warp to reside on the SM (increasing occupancy).
  *Techniques:* Smaller intermediates (fp16 instead of fp32 where safe), fewer live variables at any point, avoid keeping tensor pointers alive across branches.

- [ ] **Tune `num_warps` for occupancy**
  Fewer warps per block → more blocks per SM (occupancy). More warps per block → better latency hiding within a block.
  *Rule of thumb:* Start at 4. For memory-bound kernels, 2–4 is often best. For compute-bound, 8 may be better.

- [ ] **Warp specialization (Triton 3.x+)**
  Use `tl.async_commit_group` / `tl.async_wait` for explicit pipelining with TMA (on H100). Producer warp loads data; consumer warp computes. Significant speedup for large GEMM-like kernels.
  *Requires:* Triton 3.0+; H100 or newer for TMA.

- [ ] **Cooperative launch for reductions**
  Instead of two-pass reductions (partial sum then final reduction kernel), use cooperative groups to synchronize across blocks in a single launch.

### A.6 — Data Layout & Preprocessing

- [ ] **Input layout transformation**
  Change the storage order of a frequently-accessed tensor (e.g., transpose [T, H] to [H, T]) so the kernel's hot access path is stride-1.
  *Cost:* One-time transpose overhead; amortized if the kernel runs many times on the same weights (e.g., attention weights in a model).

- [ ] **Eliminate dynamic shapes with `tl.constexpr`**
  If a dimension is always the same value at runtime (e.g., `head_dim=128` in all DeepSeek-V3 variants), declare it as `tl.constexpr`. The compiler can then optimize branches away and use known-size tile computations.

- [ ] **Precompute host-side derived values**
  If the kernel computes something complex from its inputs that could be precomputed (e.g., `1 / sqrt(head_dim)`, index lookup tables), move it to the Python launcher.

- [ ] **Pack small values**
  For integer index arrays or small-range values, pack multiple values into a single wider word to reduce load count.

---

## Common Numerical Stability Pitfalls (that affect which optimizations are safe)

- **Always upcast reduction accumulators to fp32** before summing fp16/bf16 values. Failure to do so causes catastrophic cancellation in LayerNorm, Softmax, etc.
- **Be careful with `tl.math.fast_expf`** — it has ~1 ULP error but can compound in softmax denominators. Check max error against the reference.
- **Mixed-precision matmul**: Triton does not automatically use tensor cores — you must use `tl.dot` with compatible dtypes. `tl.dot` on fp16/bf16 inputs uses tensor cores on Ampere+.
- **Masked loads must initialize with safe values**: use `other=0.0` for loads that may go out of bounds, and `other=-float('inf')` for attention masks.
- **Integer overflow in index math**: use `tl.int64` for strides/offsets when `M * N * K` could exceed 2^31.
