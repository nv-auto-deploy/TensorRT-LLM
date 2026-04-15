# Mirage PK: Dynamic Batch Size and Sequence Length Support

## Problem Statement

Today, Mirage's code generator emits CUDA kernels with **batch size (num_tokens) and
sequence-length-derived dimensions hardcoded as C++ template parameters**. This means any
change to the number of active tokens or the maximum sequence length requires a full
recompilation (nvcc + link), which takes seconds per kernel. In an LLM serving context,
batch sizes change every iteration and sequence lengths vary per request — making the
current compile-per-shape model impractical without extensive caching and pre-warming.

### Goal

Compile a PersistentKernel **once** for a (max_batch, max_seq_length) envelope.
At runtime, handle any (actual_batch <= max_batch, actual_seq_len <= max_seq_length)
**without recompilation** — only through metadata updates and (where needed) runtime
TMA descriptor re-creation.

All other dimensions (hidden_size, head_dim, num_experts, intermediate_size, etc.)
remain static compile-time template parameters. No changes needed there.

---

## Scope: What Needs to Change

### Dimensions that must become dynamic

| Dimension | Where it appears today | Runtime source |
|-----------|----------------------|----------------|
| **batch_size / num_tokens / capacity** | Template param in `linear_swapAB_kernel_hopper`, `topk_softmax_task_impl`, `mul_sum_add_sm100_task_impl`, `norm_linear_task_impl`; TMA descriptor shapes; task_graph.json tensor dims/strides | `RuntimeConfig` (already has `qo_indptr_buffer` which encodes per-request token counts) |
| **max_seq_length** | `RuntimeConfig.max_seq_length`, page table sizes | `RuntimeConfig.max_seq_length` (already runtime) |

### Dimensions that stay static (no changes needed)

hidden_size, head_dim, intermediate_size, num_experts, topk, num_kv_heads,
num_q_heads, page_size, weight matrix dimensions (M, K, N for model-architecture
matrices).

---

## Detailed Requirements

### R1: Code Generator — Decouple batch dim from template params

**Current behavior** (from actual generated code):

```cpp
// Batch=1 hardcoded in 3 places: template param, TMA type, and layout
kernel::linear_swapAB_kernel_hopper<bfloat16, 1, 8192, 2816, 5,
    TMA_A, TMA_B, TMA_OUT, void, 8192, false>(...);

// TMA descriptor type encodes batch in the tile shape
using TMA_OUT = kernel::tma::tma_2d<bfloat16, 3,3,3,
    1, 8192,    // <-- batch=1 baked into global shape
    1, 64, 8192, 1, 1, 1, 512, true>;

// task_graph.json encodes concrete dims
"dims": [1, 2816], "strides": [2816, 1]  // batch=1
```

**Required behavior:**

The code generator should emit kernels compiled for `MAX_BATCH` (the envelope) but
with runtime dispatch based on `actual_batch` read from `RuntimeConfig` or
`TaskDesc::task_metadata`.

Concretely, for each affected kernel family:

#### R1a: `linear_swapAB_kernel_hopper`

Current template: `<T, batch, M, K, stages, TMA_A, TMA_B, TMA_OUT, Residual, stride, has_residual>`

The `batch` parameter controls the number of token-rows in the GEMM. Two options:

Keep the template compiled for `MAX_BATCH`, pass `actual_batch` through
`TaskDesc::task_metadata` or a new field, and have the kernel's inner loop only
process `actual_batch` rows. The GEMM tile loop already iterates over M-tiles;
it just needs an early-exit condition when `tile_row >= actual_batch`. TMA loads
for out-of-bounds tiles can be skipped or masked.

#### R1b: `topk_softmax_task_impl`

Current template: `<T, num_experts, capacity, topk, tile_size>`

The `capacity` param (= num_tokens) should become runtime. The kernel processes
one token per thread-group; with `MAX_CAPACITY` compiled, fewer active tokens
means fewer thread-groups do real work.

#### R1c: `mul_sum_add_sm100_task_impl`

Current template: `<T, batch, hidden, batch2, hidden2>`

Both `batch` and `batch2` should be `MAX_BATCH` at compile time, with `actual_batch`
read at runtime to bound the loop.

#### R1d: `norm_linear_task_impl` (Ampere path)

Current template: `<T, num_rows, out_dim, hidden, head_dim>`

`num_rows` = batch. Same approach: compile for MAX, loop only over actual.

#### R1e: `moe_linear_sm90_task_impl`

The batch dimension here is implicit in the routing_indices and expert_mask layouts.
These are already partially dynamic (expert_offset from task_metadata). The change
needed is that the cute::Layout shapes for `mInput`, `mOutput`, `mRoutingIndices`
should use `MAX_BATCH` for allocation but `actual_batch` for the compute extent.

### R2: TMA Descriptors — Runtime re-creation

**Current behavior:**

TMA descriptors are created as compile-time template types:
```cpp
using TMA_A = kernel::tma::tma_2d<bfloat16, 3,3,3,
    8192, 2816,   // global_shape: (M=8192, K=2816) — M encodes batch
    64, 64,       // tile_shape
    2816, 1,      // global_stride
    1, 2,         // swizzle
    4096, true>;  // smem
```

**Required behavior:**

TMA descriptors that include the batch dimension in their global shape must be
re-created at runtime when `actual_batch` changes. On Hopper/Blackwell, this is done via
`cuTensorMapEncode*` (driver API) or the `cute::make_tma_copy` runtime path.

Specifically:
- `TMA_A` (input activation): global shape row count = `actual_batch`
- `TMA_OUT` (output): global shape row count = `actual_batch`
- `TMA_B` (weight): **no change needed** — weight shape is static

The TMA descriptor pointers are already stored in `TaskDesc::input_tma_desc_ptrs[]` and
`TaskDesc::output_tma_desc_ptrs[]`. The runtime can update these between PK launches
using pre-allocated `CUtensorMap` objects that get re-encoded with the new batch dim.

**Suggested implementation:**

Add a host-side helper (called between launches, not on the GPU):
```cpp
void update_tma_descriptors_for_batch(
    RuntimeConfig &config,
    int actual_batch,
    std::vector<FullTaskDesc> &all_tasks);
```

This iterates over all task descriptors, identifies TMA-backed tensors whose batch
dimension differs from the current encoding, and calls `cuTensorMapEncode` to
re-create them. This is a host-side operation (~microseconds) and can be done
as part of `prepare_next_batch` or before `launch_persistent_kernel`.

### R3: Task Graph Tensor Dims — Runtime patching

**Current behavior:**

`construct_task_graph` reads tensor dims from `task_graph.json`:
```json
"dims": [1, 2816], "strides": [2816, 1]
```

These get stored in `FullTaskTemplate::inputs[].dim[]` and propagated to
`TaskDesc::input_ptrs[]` via `BindingSlotDesc`.

**Required behavior:**

The task graph should be generated for `MAX_BATCH`. At runtime, when `actual_batch`
changes, the dims/strides in the binding slots and task descriptors need updating.
Specifically:

- `BindingSlotDesc::tensor.dim[0]` for batch-dimension tensors → set to `actual_batch`
- `BindingSlotDesc::tensor.stride[]` — if the batch dim is dim[0] with stride = inner_dim,
  the stride doesn't change, only the extent does
- Any `FullTaskDesc` input/output dims that encode batch → update

This is similar to how `bind_tensor_func` already allows rebinding data pointers —
extend it to also accept dim overrides.

**Suggested API addition to PersistentKernel Python class:**

```python
pk.update_dynamic_dims(actual_batch=N)
```

Which internally calls a C++ function that patches all batch-dependent tensor
descriptors and re-creates affected TMA descriptors.

### R4: RuntimeConfig — Add `actual_batch_size` field

Add to `RuntimeConfig`:
```cpp
struct RuntimeConfig {
    // ... existing fields ...
    int actual_batch_size;    // NEW: actual tokens this launch (<= MPK_MAX_NUM_BATCHED_TOKENS)
};
```

This field is set by the host before each PK launch and is readable by `_execute_task`
to pass to kernels that need the runtime batch dim. It's already partially available
via `qo_indptr_buffer`, but having it as a first-class field is cleaner for the
non-attention kernels (linear, MoE, etc.) that don't use the indptr scheme.

### R5: Shared Memory Allocation

Compile-time shared memory is sized for `MAX_BATCH`. With `actual_batch < MAX_BATCH`,
the excess shared memory is simply unused. No functional change needed — just a
minor memory overhead that's acceptable since:
- Shared memory is per-SM, not global
- MAX_BATCH is small (e.g., 16 or 32 for decode serving)
- The overhead is `(MAX_BATCH - actual_batch) * tile_bytes` per SM

### R6: max_seq_length — Already Mostly Runtime

`RuntimeConfig.max_seq_length` is already a runtime field (line 287 of runtime_header.h).
The KV cache page table (`paged_kv_indptr_buffer`, `paged_kv_indices_buffer`,
`paged_kv_last_page_len_buffer`) already supports variable sequence lengths at runtime.

**One remaining issue:** `MPK_MAX_NUM_BATCHED_TOKENS` appears to be a compile-time define
(line 48 of persistent_kernel.cuh, currently commented out). If it gates any buffer
allocation, ensure it's set to the envelope max, not the per-launch actual.

No code generator changes needed for max_seq_length — it's already dynamic. Just ensure
the bridge compiles with `max_seq_length = serving_config.max_seq_len` (the envelope)
rather than per-request values.

---

## End-to-End Integration Flow

### Current flow (recompiles per shape)

```
AD serving config (max_batch_size, max_seq_len)
        │
        ▼
  GemmaMpkRuntimeWrapper.forward()
        │
        ├─ prefill/mixed batch → original GraphModule (no MPK)
        │
        └─ generate-only batch → _GemmaMirageRuntime.__call__()
                │
                │  batch_size = input_ids.shape[0]     ← changes every iteration
                │
                ▼
           per-layer loop: layer_block(hidden=...)
                │
                ├─ _linear(input_tensor)
                │     capacity = input_tensor.shape[0]
                │     key = (capacity, in_dim, out_dim)  ← exact shape in cache key
                │     cache miss? → _MirageLinearExecutor(capacity=capacity, ...)
                │                      └─ create_test_persistent_kernel(
                │                             max_num_batched_tokens=capacity)
                │                      └─ pk.linear_layer(...)
                │                      └─ compile_persistent_kernel_with_patches(pk)  ← NVCC!
                │     cache hit?  → reuse compiled executor
                │     executor(input, weight) → pk.run()
                │
                ├─ _router(...)       ← same: shape → cache key → compile if miss
                ├─ _dense_ffn(...)    ← same
                ├─ _moe_w13(...)      ← same
                └─ _moe_w2_reduce(...)← same
```

Each new batch size triggers nvcc compilation for every executor type that hasn't
seen that exact capacity before.

### Proposed flow (compile once, dynamic at runtime)

```
AD serving config
  ├─ max_batch_size (e.g., 16)       ─┐
  └─ max_seq_len (e.g., 8192)        ─┤
                                       │
  LowerToMpkConfig                     │  plumbed via SharedConfig or
    .max_batch_size                  ◄──┘  LowerToMpkConfig fields
    .max_seq_length
        │
        ▼
  build_gemma_mirage_runtime_callable(plan, source_model,
                                      max_batch=16, max_seq_len=8192)
        │
        ▼
  _GemmaMirageRuntime.__init__()              ← ONE TIME at model load
        │
        │  For each unique (in_dim, out_dim) / (hidden, experts, topk) combo:
        │    _MirageLinearExecutor(capacity=MAX_BATCH, ...)
        │    _MirageRouterExecutor(capacity=MAX_BATCH, ...)
        │    _MirageDenseFfnExecutor(capacity=MAX_BATCH, ...)
        │    _MirageMoeW13Executor(capacity=MAX_BATCH, ...)
        │    _MirageMoeW2ReduceExecutor(capacity=MAX_BATCH, ...)
        │    _MirageGemmaLayerSinglePkExecutor(max_seq_length=MAX_SEQ, ...)
        │
        │  Each PK compiled with max_num_batched_tokens=MAX_BATCH
        │  Cache keys: (in_dim, out_dim) — capacity removed from key
        │  Total compiles: one per unique executor shape (cached to disk)
        │
        ▼
  GemmaMpkRuntimeWrapper.forward()            ← EVERY ITERATION
        │
        └─ generate-only → _GemmaMirageRuntime.__call__()
                │
                │  actual_batch = input_ids.shape[0]  (e.g., 3)
                │
                ▼
           per-layer loop:
                │
                ├─ _linear(input_tensor)
                │     executor = self._linear_cache[(in_dim, out_dim)]    ← always cache hit
                │     if actual_batch != executor._last_batch:            ← skip if unchanged
                │         executor.update_batch(actual_batch=3)           ← host-side, ~μs
                │           └─ pk.update_dynamic_dims(actual_batch=3)
                │               ├─ patch BindingSlotDesc dims
                │               ├─ cuTensorMapEncode for batch-dep TMA descriptors
                │               └─ set RuntimeConfig.actual_batch_size = 3
                │     executor(input[:3], weight) → pk.run()
                │       └─ kernel: tiles with row >= 3 early-exit
                │     return output[:3]                                   ← slice to actual
                │
                ├─ _router(...)       ← same: update_batch → run → slice
                ├─ _dense_ffn(...)    ← same
                └─ _moe_*(...)        ← same
```

### Runtime overhead of `update_dynamic_dims`

The `update_dynamic_dims` call is **host-side work executed before kernel launch**,
not during kernel execution. The per-iteration timeline looks like:

```
Host:   [update_dynamic_dims] [pk.run() launch]  [... next layer ...]
         ~0-50μs               ~5-10μs
GPU:                           [====== kernel execution ======]
                                ~100s μs to ms
```

**Cost breakdown:**
- `cuTensorMapEncode`: ~1-2μs per TMA descriptor. Worst case per layer:
  ~4 batch-dependent TMA descriptors × ~6 task types = ~24 re-encodes = ~25-50μs
- Tensor dim patching: trivial integer writes, <1μs
- `RuntimeConfig.actual_batch_size` update: single int write, <1μs

**Amortization strategies:**
- **Skip if unchanged:** In steady-state decode, batch size is often constant across
  iterations. A `if self._last_batch == actual_batch: return` guard makes the common
  case zero-cost.
- **Batch updates per layer:** All executor descriptor updates for a layer can be
  batched into a single pass rather than per-executor calls.
- **Overlap with GPU:** The host-side patching for layer N+1 can overlap with GPU
  execution of layer N (the PK launch is async). With proper stream management,
  the descriptor update cost is fully hidden behind GPU compute.

**Worst case (batch changes every iteration):** ~50μs additional host overhead per
layer, which is <5% of typical layer GPU execution time (~1ms+). In practice, the
skip-if-unchanged guard means this cost is hit only on the iteration where batch
size actually changes (e.g., when a request completes or arrives).

---

## Consumer Contract (Bridge Side)

Once Mirage implements R1-R5, the TRT-LLM bridge will:

1. **At model load time:** Create one PK per layer compiled with:
   - `max_num_batched_tokens = MAX_BATCH` (e.g., 16 or 32)
   - `max_seq_length = serving_config.max_seq_len`
   - All static model dims (hidden, heads, experts, etc.)
   - `max_batch` and `max_seq_len` plumbed from `LowerToMpkConfig` →
     `build_gemma_mirage_runtime_callable` → `_GemmaMirageRuntime.__init__`

2. **Before each forward pass:** Call `pk.update_dynamic_dims(actual_batch=N)`
   which patches tensor descriptors + re-encodes TMA descriptors on the host.
   Skipped entirely when `actual_batch` is unchanged from the previous launch.

3. **Launch:** Call `pk()` as today. The PK scheduler + kernels read
   `RuntimeConfig.actual_batch_size` for loop bounds / tile early-exit.

4. **After launch:** Slice output tensors to `[:actual_batch, :]`.

This eliminates all per-shape recompilation. The only compile events are:
- First load (or cache miss): one nvcc compile per PK variant
- Model architecture change: recompile (different static dims)

---

## Affected Kernel Task Types (from generated code analysis)

| Task Type | Template Params with batch dim | Priority |
|-----------|-------------------------------|----------|
| `TASK_LINEAR_SWAPAB_HOPPER` | param 2 (batch/M), TMA_A/TMA_OUT global shape | **P0** — most frequent |
| `TASK_MOE_TOPK_SOFTMAX_SM100` | param 3 (capacity) | P0 |
| `TASK_MOE_MUL_SUM_ADD_SM100` | params 2,4 (batch, batch2) | P0 |
| `TASK_MOE_W13_LINEAR_SM90` | mInput/mOutput layout batch dim | P1 |
| `TASK_MOE_W2_LINEAR_SM90` | mInput/mOutput layout batch dim | P1 |
| `TASK_NORM_LINEAR` (Ampere) | param 2 (num_rows) | P1 |
| Attention tasks | Already dynamic via indptr metadata | No change |

---

## Incremental Delivery Suggestion

**Phase 1 — Runtime batch via kernel early-exit + runtime TMA (the real fix):**

This is the primary deliverable. Modify kernel task implementations to:
1. Compile once for `MAX_BATCH` (single template instantiation per task type)
2. Accept `actual_batch` as a runtime argument (via `RuntimeConfig.actual_batch_size`
   or `TaskDesc::task_metadata`)
3. Skip tiles / loop iterations where `tile_row >= actual_batch`
4. Re-create TMA descriptors on the host between launches using `cuTensorMapEncode`
   when `actual_batch` changes — this is ~microseconds, done once per launch

This is a single compile per PK variant. Handles arbitrary batch sizes up to the
compiled max.

**Phase 2 — Optimized tile scheduling for partial batches:**

Phase 1's early-exit means some thread blocks launch but immediately return when
their tile range is beyond `actual_batch`. For small actual_batch relative to
MAX_BATCH, this wastes SM occupancy. Phase 2 would optimize the PK scheduler to
only dispatch tasks whose tile range overlaps the active batch, avoiding the wasted
launches entirely. This is a scheduler-level optimization, not a kernel change.

Phase 1 alone fully solves the recompilation problem and is sufficient for serving.
Phase 2 is a performance polish for the case where MAX_BATCH >> actual_batch.

---

## Test Plan

### T1: Correctness — Single executor, varying batch within compiled max

For each executor type (`linear`, `router`, `dense_ffn`, `moe_w13`, `moe_w2_reduce`):

1. Compile with `MAX_BATCH=16`, Gemma4 model dims (hidden=2816, intermediate=2112,
   num_experts=128, topk=8)
2. Run with `actual_batch` in `{1, 2, 3, 7, 8, 15, 16}` (including non-power-of-2)
3. Compare output `[:actual_batch]` against PyTorch reference (fp32 matmul downcast)
4. Assert `max_abs_error < threshold` per executor type (use existing tolerances:
   linear < 1.0, router topk_weight < 0.005, moe_w2 < 0.03, dense_ffn < 0.04)

**Key checks:**
- Output rows beyond `actual_batch` must NOT corrupt the valid output region
- `actual_batch=MAX_BATCH` must match the existing non-dynamic path exactly (bitwise)
- `actual_batch=1` (decode steady-state) must maintain existing accuracy

```python
@pytest.mark.parametrize("actual_batch", [1, 2, 3, 7, 8, 15, 16])
def test_dynamic_batch_linear_correctness(actual_batch):
    MAX_BATCH = 16
    in_dim, out_dim = 2816, 8192
    executor = _MirageLinearExecutor(capacity=MAX_BATCH, in_dim=in_dim, out_dim=out_dim, ...)
    input_tensor = torch.randn(actual_batch, in_dim, device="cuda", dtype=torch.bfloat16) / 8.0
    weight = torch.randn(out_dim, in_dim, device="cuda", dtype=torch.bfloat16) / 16.0

    executor.update_batch(actual_batch=actual_batch)
    output = executor(input_tensor, weight)
    result = output[:actual_batch]

    ref = (input_tensor.float() @ weight.float().T).to(torch.bfloat16)
    assert (result.float() - ref.float()).abs().max() < 1.0
```

### T2: Correctness — Batch size transitions within a sequence of launches

Simulate a serving loop where batch size changes mid-session:

```python
def test_dynamic_batch_transitions():
    MAX_BATCH = 16
    executor = _MirageLinearExecutor(capacity=MAX_BATCH, ...)

    # Simulate: 8 requests → one finishes → 7 → new arrives → 8 → burst → 16
    for actual_batch in [8, 7, 8, 16, 1, 4, 16, 1]:
        executor.update_batch(actual_batch=actual_batch)
        input_t = torch.randn(actual_batch, in_dim, ...)
        output = executor(input_t, weight)[:actual_batch]
        ref = (input_t.float() @ weight.float().T).to(torch.bfloat16)
        assert (output.float() - ref.float()).abs().max() < 1.0
```

**Key checks:**
- Transitioning from large→small batch doesn't leave stale data in output
- Transitioning from small→large batch doesn't read uninitialized TMA regions
- Repeated same-batch launches (the skip-if-unchanged path) stay correct

### T3: Correctness — Full layer with dynamic batch

Run the full `_MirageGemmaLayerSinglePkExecutor` or blockwise
`_GemmaDecodeLayerBlock` with varying batch sizes against the original
GraphModule reference:

```python
@pytest.mark.parametrize("actual_batch", [1, 4, 8, 16])
def test_dynamic_batch_full_layer(actual_batch):
    MAX_BATCH = 16
    # Build _GemmaMirageRuntime with max_batch=MAX_BATCH
    runtime = _GemmaMirageRuntime(source_model, plan, max_batch=MAX_BATCH)

    # Run one layer with actual_batch tokens
    hidden = torch.randn(actual_batch, 1, 2816, device="cuda", dtype=torch.bfloat16) / 8.0
    mpk_result = runtime.run_single_layer(layer_idx=0, hidden=hidden, ...)
    ref_result = original_model.run_single_layer(layer_idx=0, hidden=hidden, ...)

    assert (mpk_result.float() - ref_result.float()).abs().max() < 0.06
```

### T4: Correctness — Varying sequence lengths within max_seq_length

```python
@pytest.mark.parametrize("seq_len", [1, 64, 128, 512, 2048, 8192])
def test_dynamic_seq_length_attention(seq_len):
    MAX_SEQ = 8192
    # Build single-PK executor with max_seq_length=MAX_SEQ
    executor = _MirageGemmaLayerSinglePkExecutor(
        layer_spec=spec, kv_cache=kv_cache, max_seq_length=MAX_SEQ, ...
    )

    # Fill KV cache with seq_len tokens worth of pages
    # Run decode step (1 new token attending to seq_len cached tokens)
    hidden = torch.randn(1, 2816, device="cuda", dtype=torch.bfloat16) / 8.0
    output = executor(hidden=hidden, kv_cache=kv_cache, ...)

    ref = torch_reference_attention(hidden, kv_cache, seq_len=seq_len, ...)
    assert (output.float() - ref.float()).abs().max() < 0.06
```

### T5: No-recompile verification

Verify that changing batch size does NOT trigger nvcc:

```python
def test_no_recompile_on_batch_change():
    MAX_BATCH = 16
    executor = _MirageLinearExecutor(capacity=MAX_BATCH, ...)

    # Record compile count (or mock compile_persistent_kernel_with_patches)
    compile_count_before = get_compile_count()

    for actual_batch in [1, 4, 8, 3, 16, 7, 1]:
        executor.update_batch(actual_batch=actual_batch)
        executor(torch.randn(actual_batch, in_dim, ...), weight)

    compile_count_after = get_compile_count()
    assert compile_count_after == compile_count_before  # zero new compiles
```

### T6: update_dynamic_dims overhead measurement

Benchmark the host-side cost to validate the μs-scale claim:

```python
def test_update_dynamic_dims_latency():
    MAX_BATCH = 16
    executor = _MirageLinearExecutor(capacity=MAX_BATCH, ...)

    # Warm up
    executor.update_batch(actual_batch=1)

    # Measure batch-change case
    latencies_change = []
    for i in range(100):
        batch = (i % 15) + 1  # varies 1-15
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        executor.update_batch(actual_batch=batch)
        t1 = time.perf_counter()
        latencies_change.append((t1 - t0) * 1e6)  # μs

    # Measure skip-if-unchanged case
    executor.update_batch(actual_batch=8)
    latencies_skip = []
    for _ in range(100):
        t0 = time.perf_counter()
        executor.update_batch(actual_batch=8)  # same batch, should skip
        t1 = time.perf_counter()
        latencies_skip.append((t1 - t0) * 1e6)

    p99_change = sorted(latencies_change)[98]
    p99_skip = sorted(latencies_skip)[98]

    assert p99_change < 200  # < 200μs even worst case
    assert p99_skip < 1      # < 1μs when skipped (just an int comparison)
```

### T7: Edge cases

```python
def test_actual_batch_equals_max():
    """actual_batch == MAX_BATCH: no early-exit, full kernel utilization."""
    # Should produce identical output to a statically-compiled MAX_BATCH kernel

def test_actual_batch_one():
    """actual_batch == 1: decode steady-state, most tiles early-exit."""
    # Verify output matches existing capacity=1 compiled kernel

def test_actual_batch_zero():
    """actual_batch == 0: should be rejected or produce empty output."""
    # Decide on semantics: error, or no-op with empty output slice

def test_max_batch_exceeds_compiled():
    """actual_batch > MAX_BATCH: must raise clear error, not silent corruption."""
    executor = _MirageLinearExecutor(capacity=16, ...)
    with pytest.raises(ValueError, match="exceeds compiled max"):
        executor.update_batch(actual_batch=17)

def test_non_contiguous_input():
    """Input tensor from a larger buffer (e.g., sliced view) still works."""
    big_buffer = torch.randn(32, 2816, device="cuda", dtype=torch.bfloat16)
    executor.update_batch(actual_batch=4)
    output = executor(big_buffer[:4], weight)  # non-contiguous possible
    # verify correctness
```

### T8: Multi-layer serving simulation

End-to-end test simulating realistic serving workload:

```python
def test_serving_simulation():
    """Simulate 100 decode iterations with varying batch from request arrivals/completions."""
    MAX_BATCH = 16
    runtime = _GemmaMirageRuntime(source_model, plan, max_batch=MAX_BATCH)

    active_requests = 4
    for iteration in range(100):
        # Simulate request lifecycle
        if iteration % 20 == 0:
            active_requests = min(MAX_BATCH, active_requests + 3)  # burst arrival
        if iteration % 7 == 0 and active_requests > 1:
            active_requests -= 1  # completion

        hidden = torch.randn(active_requests, 1, 2816, device="cuda", dtype=torch.bfloat16)
        output = runtime(build_serving_inputs(hidden, active_requests, ...))
        logits = output["logits"]

        assert logits.shape[0] == active_requests
        assert torch.isfinite(logits).all(), f"NaN/Inf at iteration {iteration}, batch={active_requests}"
```
