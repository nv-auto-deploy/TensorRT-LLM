# Intra-Bucket Finalize: Implementation Overview

## Goal

Reduce piecewise CUDA-graph capture peak memory for the largest token bucket by
releasing no-longer-needed static runner outputs during capture, while keeping
replay correctness.

This document describes the implementation architecture, runtime invariants,
and current operational behavior.

---

## Architecture

### Components

- `PiecewiseCapturedGraph` (`torch_cudagraph.py`)
  - Splits the FX graph into static and dynamic partitions.
  - Wraps static partitions in `ADPiecewiseRunner`.
  - Orchestrates warmup, capture, finalize/prune scheduling, and replay setup.

- `ADPiecewiseRunner` (`piecewise_runner.py`)
  - Owns per-bucket capture/replay state (`SegmentEntry`).
  - Captures/replays one static partition.
  - Tracks dynamic inputs vs static (weight/buffer) inputs.
  - Finalizes/materializes captured entries.

- `IntraBucketFinalizeInterpreter` (`torch_cudagraph.py`)
  - Subclasses `torch.fx.Interpreter` with `garbage_collect_values=True`.
  - Executes capture graph node-by-node.
  - Triggers finalize and registry-prune hooks according to schedules.

---

## Core Data Structures

### `SegmentEntry`

Per static runner per bucket:

- `cuda_graph`
- `static_inputs`
- `dynamic_indices`
- `static_output`
- `_reconstructable_input_indices`
- `_finalized` (`_FinalizedState`)

### `_FinalizedState`

Metadata stash used between finalize and materialize:

- `input_metadata`
- `output_metadata`
- `output_non_tensors`
- `output_spec`

### Class-level registries

- `_static_output_registry[(num_tokens, data_ptr)] -> tensor`
  - Used for capture-time zero-copy chaining across static runners.
- `_registry_prune_hits[num_tokens]`
  - Debug counter for successful registry removals.

---

## Capture-Phase Execution

For each bucket (largest to smallest):

1. Set runner context (`num_tokens`, phase).
2. Warmup (largest bucket uses configured warmup; smaller buckets can be reduced).
3. Enter capture phase with `IntraBucketFinalizeInterpreter`.
4. Interpreter runs FX nodes and applies two schedules:
   - **Finalize schedule**: drop runner-held refs when execution liveness allows.
   - **Registry-prune schedule**: remove `_static_output_registry` entries only
     after downstream capture-time lookup users have run.
5. Cross-bucket cleanup (`clear_static_output_registry`, `empty_cache`) before next bucket.
6. Materialize finalized entries after capture loop (reconstruct replay handles).

---

## Scheduling

### Finalize schedule

`compute_finalize_schedule(split_gm)`:

- For each static runner node, find last direct consumer in topo order.
- Trigger `finalize_entry()` after that node.

### Registry-prune schedule

`compute_registry_prune_schedule(split_gm)`:

- Frontier traversal from producer runner:
  - walk through non-runner nodes,
  - stop at first downstream runner on each path,
  - prune after latest frontier runner executes.

This preserves capture-time lookup correctness while enabling earlier reclaim
than fully deferring prune to end-of-bucket.

---

## Dynamic Input Classification

Inputs are classified using:

- local submodule parameter/buffer pointers, plus
- graph-level parameter/buffer pointer set.

This avoids misclassifying forwarded graph weights as dynamic inputs.

Only dynamic indices in `_reconstructable_input_indices` are eligible for
drop+reconstruct flow; other dynamic inputs remain pinned.

---

## Finalize and Materialize Semantics

### `finalize_entry(...)`

- Captures metadata for reconstructable dynamic inputs and outputs.
- Drops strong refs according to mode flags.
- Optionally prunes registry by `(num_tokens, data_ptr)`.
- Produces `_FinalizedState`.

### `materialize_entry(...)`

- Reconstructs tensors from `_FinalizedState` metadata using PyTorch internal
APIs for pointer-backed tensor handles.
- Restores replay-ready `static_inputs`/`static_output`.

---

## Replay Invariant

Replay correctness requires:

1. producer/consumer buffer alias assumptions made during capture remain valid,
2. `_prepare_replay_inputs` only copies into valid static buffers,
3. registry prune timing does not remove capture-time lookup dependencies too early.

---

## Observability / Debugging

The implementation logs:

- CUDA memory snapshots per bucket:
  - `before_warmup`
  - `after_warmup`
  - `after_warmup_empty_cache`
  - `after_capture`
  - `after_finalize`
  - `after_cross_bucket_cleanup`
- capture stats per bucket:
  - dynamic/reconstructable/pinned input counts
  - byte totals (raw + unique by data_ptr)
  - output tensor count
  - finalized runner count
  - registry prune hit count

---

## Current Behavior Summary

- Intra-bucket finalize path is active in capture orchestration.
- Interpreter-driven finalize/prune schedules are active.
- Registry keying is consistent with capture (`(num_tokens, data_ptr)`).
- Weight-forwarding dynamic misclassification is addressed via global pointer set.
- Correctness and memory are coupled to schedule timing and reconstructability
  coverage of dynamic inputs.

---

## Implementation Files

- `tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py`
  - `IntraBucketFinalizeInterpreter`
  - `PiecewiseCapturedGraph.prepare()`
  - `PiecewiseCapturedGraph.warmup_and_capture()`
  - memory + capture telemetry logging

- `tensorrt_llm/_torch/auto_deploy/compile/piecewise_runner.py`
  - `SegmentEntry`, `_FinalizedState`
  - `finalize_entry()`, `materialize_entry()`, `prune_registry_for_entry()`
  - `compute_finalize_schedule()`, `compute_registry_prune_schedule()`
  - dynamic input classification + replay preparation

---

## Next Engineering Steps

1. Keep correctness invariant tests around mixed/prefill + decode transitions.
2. Continue reducing capture peak by improving reconstructability coverage and/or
   partitioning hot tail segments with large temporary allocations.
3. Validate 8192 bucket capture peak and cross-bucket OOM safety under production
   KV-cache settings.
