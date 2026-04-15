# Mirage PK: Static Schedule + Matvec for Low-Batch Decode

## Problem Statement

The Mirage PersistentKernel achieves excellent fusion for large-batch workloads
but exhibits **5000x overhead vs cuBLAS** for single-token decode (batch=1).

Profiled per-layer breakdown for Gemma4 MoE decode at batch=1:

| Operation | Mirage PK | cuBLAS/torch | Ratio |
|-----------|-----------|-------------|-------|
| QKV linear `(1,2816)×(2816,8192)` | 100ms | 0.020ms | 5000x |
| O-proj linear `(1,4096)×(4096,2816)` | 50ms | 0.020ms | 2500x |
| Dense FFN (norm+gateup+gelu+down) | 246ms | 0.367ms | 670x |
| Router linear | 4ms | 0.010ms | 400x |
| Fused MoE (128 experts, topk=8) | 301ms | ~1.5ms | 200x |
| **Layer total** | **710ms** | **~2ms** | **355x** |

At 30 layers × 32 tokens, this means **682 seconds** vs a cuBLAS baseline of
~1.9 seconds. The target is **< 2 minutes** for 32 output tokens.

### Root Causes

Two independent root causes contribute roughly equally:

**1. Dynamic scheduling overhead (~30% of gap)**

The PK's event-driven task dispatch adds per-task overhead:
- Spin-wait on event counter: `ld_acquire_sys_u64` + `__nanosleep(10)` loop
- TaskDesc load from global memory: 128-byte `cp_async` per task
- Dependent-event barrier: atomic spin per task transition
- Event trigger: `atomicAdd` on completion

With 20+ tasks per layer, this overhead accumulates even at `num_workers=1`.

**2. Tiled GEMM on M=1 workloads (~70% of gap)**

All linear tasks use `linear_swapAB_kernel_hopper` with WGMMA 64×64 or
128×128 tiles. At batch=1 (M=1):
- TMA loads a 64-row tile when only 1 row has data (63/64 wasted)
- WGMMA computes a 64×64 output tile when only 1×64 is useful (1/64 utilization)
- Shared memory is partitioned for 64-row tiles (63/64 wasted)
- The `actual_batch` early-exit (from the dynamic batch feature) helps but still
  launches the full tile infrastructure before exiting

cuBLAS handles this by dispatching a **matvec kernel** (warp-per-row dot product)
instead of a tiled GEMM when M is small. Mirage's codegen always emits the tiled
path regardless of M.

### Goal

Reduce per-layer decode latency from 710ms to **< 4ms** at batch=1 through:
1. Static task scheduling (eliminate per-task dispatch overhead)
2. Matvec GEMM path for small M (eliminate tiled GEMM waste)

Together these close the 355x gap to within ~2x of cuBLAS, while preserving the
single-PK-launch-per-layer advantage and fused data paths.

---

## Requirement 1: Static Schedule Mode

### Current behavior

```
execute_worker():
  while (true):
    spin_wait_for_queue_entry()          # ld_acquire_gpu_u64 loop
    load_task_descs_from_gmem()          # cp_async 128B per task
    wait_for_dependent_event()           # ld_acquire_sys_u64 loop
    __syncthreads()
    _execute_task(task_desc, config)     # actual compute
    tma::store_async_wait()
    __threadfence()
    __syncthreads()
    trigger_event()                      # atomicAdd
```

Per-task overhead: ~5-60μs (event spin + gmem load + barrier + fence).
With 20 tasks per layer: ~100-1200μs of pure scheduling overhead.

### Required behavior

New compilation mode: `PersistentKernel(schedule_mode="static")` (default remains
`"dynamic"` for backward compatibility).

In static mode, the codegen emits a hardcoded task sequence with
`cooperative_groups::this_grid().sync()` barriers between dependent phases:

```cpp
__device__ void _execute_layer_static(RuntimeConfig const &config) {
    auto grid = cooperative_groups::this_grid();

    // Phase 1: Input norm + QKV projection
    _execute_task(&preloaded_tasks[0], config);  // rmsnorm
    _execute_task(&preloaded_tasks[1], config);  // linear (QKV)
    grid.sync();

    // Phase 2: Paged attention
    _execute_task(&preloaded_tasks[2], config);  // paged_attention
    grid.sync();

    // Phase 3: O-proj + post-attn residual + FFN
    _execute_task(&preloaded_tasks[3], config);  // linear (O-proj)
    _execute_task(&preloaded_tasks[4], config);  // rmsnorm + residual
    _execute_task(&preloaded_tasks[5], config);  // linear (FFN gate_up)
    _execute_task(&preloaded_tasks[6], config);  // gelu_mul
    _execute_task(&preloaded_tasks[7], config);  // moe_w2_linear (FFN down)
    _execute_task(&preloaded_tasks[8], config);  // moe_mul_sum_add
    _execute_task(&preloaded_tasks[9], config);  // rmsnorm (FFN norm)
    grid.sync();

    // Phase 4: Router + MoE
    _execute_task(&preloaded_tasks[10], config); // rmsnorm (router norm)
    _execute_task(&preloaded_tasks[11], config); // linear (router)
    _execute_task(&preloaded_tasks[12], config); // topk_softmax
    _execute_task(&preloaded_tasks[13], config); // rmsnorm (MoE input)
    _execute_task(&preloaded_tasks[14], config); // moe_w13_gelu_mul
    _execute_task(&preloaded_tasks[15], config); // moe_w2_linear
    _execute_task(&preloaded_tasks[16], config); // moe_mul_sum_add (merge)
    grid.sync();

    // Phase 5: Final merge + residual + scale
    _execute_task(&preloaded_tasks[17], config); // rmsnorm + residual
    _execute_task(&preloaded_tasks[18], config); // linear_with_residual (scale)
}
```

### Key design points

**Task descriptors preloaded at init, not per-dispatch.** In dynamic mode, each
task descriptor is loaded from global memory via `cp_async` on every dispatch.
In static mode, the task array is fixed — descriptors are loaded once during
`init_persistent_kernel` and stored in a `__constant__` or `__shared__` buffer
that persists across the kernel lifetime.

**`grid.sync()` replaces per-task events.** The dependency graph for a single
Gemma4 layer has 4-5 barrier points where all preceding tasks must complete
before subsequent tasks can start. The codegen analyzes the task graph's
dependency edges and inserts `grid.sync()` at these natural phase boundaries.
Tasks within a phase have no cross-dependencies and execute in sequence on
each worker (or in parallel across workers for independent tasks like per-expert
MoE).

**Cooperative launch required.** `grid.sync()` requires the kernel to be launched
via `cudaLaunchCooperativeKernel`. This is a one-line change in
`launch_persistent_kernel`:
```cpp
void *args[] = { &global_runtime_config };
cudaLaunchCooperativeKernel(
    (void*)persistent_kernel, grid_dim, block_dim, args, smem_size, stream);
```

**Worker count = SM count.** For maximum parallelism within each phase, launch
one worker CTA per SM. Each worker handles a slice of the output rows for
linear tasks (like Luce's approach). For the current bridge which uses
`num_workers=1`, this is also a perf improvement — going from 1 CTA to 132 CTAs
(H100) means linear tasks are distributed across all SMs.

### Estimated savings

| Component | Dynamic mode | Static mode | Saved |
|-----------|-------------|-------------|-------|
| Event spin-wait | ~5-60μs/task | 0 | ~100-1200μs/layer |
| TaskDesc gmem load | ~2-3μs/task | 0 (preloaded) | ~40-60μs/layer |
| Event trigger atomic | ~0.5μs/task | 0 | ~10μs/layer |
| `__threadfence` per task | ~1-5μs/task | 0 (only at grid.sync) | ~20-100μs/layer |
| Grid sync overhead | 0 | ~5μs × 4 syncs | +20μs/layer |
| **Net scheduling overhead** | **~170-1370μs** | **~20μs** | **~150-1350μs** |

This alone doesn't close the 710ms gap — the GEMM kernel itself is the
dominant cost. But it eliminates one of the two root causes and is required
for the matvec path to reach its potential.

---

## Requirement 2: Matvec GEMM Path for Small Batch

### Current behavior

All linear tasks emit `linear_swapAB_kernel_hopper` with WGMMA tiled GEMM
regardless of batch size:

```cpp
// Generated code for (1, 8192) = (1, 2816) × (8192, 2816)^T
kernel::linear_swapAB_kernel_hopper<bfloat16, 1, 8192, 2816, 5,
    TMA_A, TMA_B, TMA_OUT, void, 8192, false>(
    tma_a, tma_b, tma_out, nullptr, false,
    runtime_config.actual_batch_size);
```

At M=1, this kernel:
- Loads 64-row TMA tiles (63 rows wasted)
- Executes 64×64 WGMMA (1/64 utilization)
- Runs on 1 CTA for a 1×8192 output (most SMs idle)

Result: **100ms** for a computation that should take **0.020ms**.

### Required behavior

The codegen should emit a **dual-path dispatch** for linear tasks:

```cpp
if (runtime_config.actual_batch_size <= MATVEC_THRESHOLD) {
    // Warp-per-row matvec: each warp computes one output row
    kernel::matvec_kernel<bfloat16, K_DIM>(
        input_ptr, weight_ptr, output_ptr,
        runtime_config.actual_batch_size, N_DIM);
} else {
    // Tiled WGMMA GEMM: existing path
    kernel::linear_swapAB_kernel_hopper<...>(...);
}
```

Where `MATVEC_THRESHOLD` is a compile-time constant (suggested: 4 or 8).

### Matvec kernel specification

**Algorithm**: Warp-cooperative horizontal reduction.

```
Grid: (N_DIM, 1, 1) — one program/warp-group per output row
Block: (128, 1, 1) — 4 warps per block, each handles one row

For each output row m (assigned to this warp):
  acc = 0.0f
  for k_start in range(0, K_DIM, WARP_SIZE * VEC_SIZE):
    // 32 lanes × 8 elements = 256 bf16 values per step
    x_vec = load_128bit(input + k_start + lane * VEC_SIZE)
    w_vec = load_128bit(weight[m] + k_start + lane * VEC_SIZE)
    acc += dot(x_vec, w_vec)   // bf16 → f32 accumulate
  // Warp-level reduction
  acc = warp_reduce_sum(acc)
  if lane == 0:
    for batch_row in range(actual_batch):  // typically 1
      output[batch_row * N + m] = bf16(acc)
```

**For batch > 1 but ≤ threshold**: Each warp processes one output row for ALL
batch elements simultaneously. The inner loop loads `actual_batch` rows of input
and accumulates `actual_batch` separate accumulators. This is still a matvec
per batch element, but amortizes the weight load across batch elements.

**Memory access pattern**:
- Input: sequential reads, fits in L1 after first access
- Weight: streaming reads with `ld.global.L1::no_allocate` (like Luce) —
  each row read once, no need to cache
- Output: one store per warp per batch element

### Performance targets

| Shape (M, K, N) | cuBLAS | Matvec target | Current Mirage |
|-----------------|--------|---------------|----------------|
| (1, 2816, 8192) | 0.020ms | 0.030ms | 100ms |
| (1, 4096, 2816) | 0.020ms | 0.025ms | 50ms |
| (1, 2816, 4224) | 0.015ms | 0.025ms | ~50ms |
| (1, 2816, 128) | 0.005ms | 0.008ms | 4ms |
| (4, 2816, 8192) | 0.040ms | 0.050ms | 100ms |
| (8, 2816, 8192) | 0.080ms | 0.100ms | 100ms |
| (16, 2816, 8192) | 0.150ms | — use tiled | 100ms |

The matvec path should be within **1.5x of cuBLAS** for M≤8. Beyond that,
the tiled WGMMA path is appropriate.

### Affected task types

All linear task types need the dual-path dispatch:

| Task Type | Matvec variant needed |
|-----------|----------------------|
| `TASK_LINEAR_SWAPAB_HOPPER` | Yes — QKV, O-proj, router, FFN gate_up |
| `TASK_LINEAR_WITH_RESIDUAL_HOPPER` | Yes — final residual + scale |
| `TASK_RMS_NORM_LINEAR` | Yes — fused norm + linear (Ampere path) |
| `TASK_MOE_W13_LINEAR_SM90` | Yes — but per-expert, so M is always small |
| `TASK_MOE_W2_LINEAR_SM90` | Yes — same |
| `TASK_LINEAR_SM100` / `TASK_LINEAR_WITH_RESIDUAL_SM100` | Yes (Blackwell path) |

The `TASK_MOE_TOPK_SOFTMAX_SM100` and `TASK_MOE_MUL_SUM_ADD_SM100` are not
GEMMs and don't need a matvec path.

---

## Requirement 3: Shared Memory Intermediate Passing (Phase 2)

### Current behavior

Every task writes its output to global memory (HBM) and the next task reads
it back from HBM. For a norm→linear sequence:

```
rmsnorm writes (1, 2816) bf16 to HBM    [5.6KB write, ~1μs]
linear reads (1, 2816) bf16 from HBM    [5.6KB read, ~1μs]
```

### Required behavior (lower priority)

In static schedule mode, consecutive tasks that run on the same SM can pass
data through shared memory:

```
rmsnorm writes (1, 2816) bf16 to SMEM   [5.6KB, ~0.1μs]
__syncthreads()
linear reads (1, 2816) bf16 from SMEM   [5.6KB, ~0.1μs]
```

This saves ~2μs per transition × ~15 transitions per layer = ~30μs per layer.
At batch=1 this is marginal; at batch=8+ the data sizes grow and bandwidth
savings compound.

**Implementation**: The codegen allocates a shared memory buffer for each
intermediate tensor that flows between consecutive tasks within a phase
(between grid syncs). Tasks write to SMEM output instead of GMEM, and the
next task reads from SMEM input. Only works for tasks assigned to the same
CTA — cross-CTA data still goes through GMEM.

**This is a Phase 2 optimization.** Requirements 1 and 2 deliver the bulk of
the speedup. SMEM passing is a polish that improves bandwidth efficiency at
larger batch sizes.

---

## Combined Performance Estimate

Per-layer decode latency at batch=1, Gemma4 MoE (30 layers):

| Configuration | Per-layer | 30L × 32T |
|---------------|-----------|-----------|
| Current Mirage (dynamic + tiled GEMM) | 710ms | 682s |
| Static schedule only (R1) | ~700ms | ~672s |
| Matvec only (R2, keep dynamic sched) | ~5ms | ~4.8s |
| **Static + matvec (R1 + R2)** | **~3.5ms** | **~3.4s** |
| Static + matvec + SMEM (R1+R2+R3) | ~3ms | ~2.9s |
| cuBLAS baseline (no fusion) | ~2ms | ~1.9s |

The matvec path (R2) delivers **99% of the speedup**. Static scheduling (R1)
adds the remaining benefit and is the architecturally correct foundation for
the matvec path to work efficiently (single cooperative launch, all SMs
participating in each matvec, grid.sync between phases).

---

## Implementation Roadmap

### Phase 1: Matvec kernel + dual dispatch (highest impact)

1. Implement `kernel::matvec_kernel<T, K>` in a new task header
   (`tasks/common/matvec.cuh` or `tasks/hopper/matvec_hopper.cuh`)
2. Add `TASK_MATVEC_HOPPER` / `TASK_MATVEC_SM100` task types to `runtime_header.h`
3. Modify `TaskRegister::register_linear_*` to emit dual-path dispatch based
   on `runtime_config.actual_batch_size <= MATVEC_THRESHOLD`
4. Wire the matvec variant into `_execute_task` switch

This can be done **without** the static schedule — the dynamic scheduler will
dispatch matvec tasks the same way it dispatches tiled GEMM tasks. The per-task
scheduling overhead (~12μs) is now small relative to the matvec compute time
(~0.03ms), so the dynamic scheduler is acceptable.

### Phase 2: Static schedule mode

1. Add `schedule_mode` parameter to `PersistentKernel`
2. Codegen emits `_execute_layer_static` function when `schedule_mode="static"`
3. Replace `execute_worker` loop with static sequence + `grid.sync()`
4. Use `cudaLaunchCooperativeKernel` for launch
5. Set `num_workers = SM_count` for full-GPU parallel matvecs

### Phase 3: SMEM intermediate passing

1. Codegen analyzes consecutive same-CTA tasks within a phase
2. Allocates SMEM buffers for intermediate tensors
3. Emits SMEM write/read instead of GMEM for qualifying pairs

---

## Consumer Contract (Bridge Side)

Once Mirage implements the matvec path (Phase 1), the bridge changes are minimal:

1. **No bridge changes needed for Phase 1.** The `runtime_config.actual_batch_size`
   is already set by `update_dynamic_dims()`. The codegen's dual dispatch reads
   it at runtime and selects matvec vs tiled GEMM. Existing executor classes
   and dispatch methods work as-is.

2. **For Phase 2 (static schedule):** The bridge passes `schedule_mode="static"`
   when constructing `PersistentKernel` for single-PK-per-layer executors.
   The `_MirageGemmaLayerSinglePkExecutor` sets this flag during construction.

3. **No config changes needed.** The `max_batch_size` / `max_seq_length` plumbing
   from the dynamic batch feature continues to work. The matvec threshold is
   a Mirage-side compile constant, not a bridge config.

---

## Reference: Luce Megakernel Architecture

The design draws from [Luce Megakernel](https://github.com/Luce-Org/luce-megakernel)
which achieves 413 tok/s decode for Qwen 3.5-0.8B on RTX 3090 by:

- Single CUDA dispatch for all 24 layers
- Warp-per-row matvec for all GEMMs at batch=1
- `AtomicGridSync` (custom atomic barrier) between phases
- Weight streaming with `ld.global.L1::no_allocate`
- Shared memory for activation buffers between phases
- Static control flow (hardcoded layer loop, no dynamic dispatch)

The key adaptation for Mirage: instead of hand-writing the entire kernel per
model (like Luce), use Mirage's task graph + codegen to **generate** the static
schedule and matvec dispatch automatically for any model registered through the
`PersistentKernel` API. This preserves Mirage's generality while matching
hand-written megakernel performance.

---

## Test Plan

### T1: Matvec correctness

```python
@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("K,N", [(2816, 8192), (4096, 2816), (2816, 128)])
def test_matvec_matches_cublas(M, K, N):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 8.0
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 16.0
    # Build PK with linear_layer, compile, run at actual_batch=M
    pk_result = run_pk_linear(x, w, actual_batch=M)
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    assert (pk_result - ref).abs().max() < 0.5
```

### T2: Matvec vs tiled GEMM threshold

```python
@pytest.mark.parametrize("M", [1, 4, 8, 16, 32])
def test_matvec_threshold_crossover(M):
    """Verify matvec is used below threshold, tiled above."""
    # Profile both paths, verify matvec is faster below threshold
    # and tiled is faster above threshold
```

### T3: Static schedule correctness (full layer)

```python
def test_static_schedule_layer_matches_dynamic():
    """Same layer with static vs dynamic schedule produces identical output."""
    pk_dynamic = build_gemma_layer_pk(schedule_mode="dynamic")
    pk_static = build_gemma_layer_pk(schedule_mode="static")
    hidden = torch.randn(1, 2816, ...)
    out_dynamic = run_pk(pk_dynamic, hidden)
    out_static = run_pk(pk_static, hidden)
    assert torch.allclose(out_dynamic, out_static, atol=1e-3)
```

### T4: Performance regression test

```python
def test_matvec_within_2x_cublas():
    """Matvec at M=1 should be within 2x of cuBLAS."""
    cublas_time = benchmark_cublas(M=1, K=2816, N=8192)
    pk_time = benchmark_pk_linear(M=1, K=2816, N=8192)
    assert pk_time < cublas_time * 2.0

def test_tiled_gemm_not_regressed():
    """Tiled GEMM at M=64 should not regress from adding matvec dispatch."""
    pk_time_before = benchmark_pk_linear(M=64, K=2816, N=8192)  # baseline
    pk_time_after = benchmark_pk_linear(M=64, K=2816, N=8192)   # with matvec codepath
    assert pk_time_after < pk_time_before * 1.05  # <5% regression
```

### T5: End-to-end decode latency

```python
def test_decode_latency_target():
    """30-layer Gemma4 decode at batch=1 should complete 32 tokens in < 2 min."""
    # Full model e2e test
    total_time = run_gemma4_decode(num_layers=30, num_tokens=32, batch=1)
    assert total_time < 120.0  # seconds
```
