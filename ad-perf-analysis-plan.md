<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Plan: `ad-perf-analysis` Skill for AutoDeploy Performance Analysis

## Summary

Create a new repo-local skill, `ad-perf-analysis`, that automates the AutoDeploy
performance-analysis loop for models like MiniMax M2, with a generic AutoDeploy design
from day one and MiniMax M2 as the first validated target.

V1 is analysis-first, not a fully autonomous optimizer. It will:

- collect or import `trtllm-bench` / `trtllm-serve` profiling artifacts
- export and parse `nsys` traces programmatically
- scope to a representative decode-layer window using NVTX plus AutoDeploy graph dumps
- correlate host CUDA runtime calls to GPU kernels, memcpys, memsets, and sync events
- classify H2D / D2H / D2D / P2P movement and host waiting behavior
- inspect serving config choices against AutoDeploy registry defaults and comparable model configs
- produce evidence-backed findings, one isolated change suggestion per iteration, and detailed iteration logs
- rerun benchmark/profile when the workflow includes a proposed experiment
- compare iterations and suggest the next highest-value step

The default policy is one change per iteration to preserve attribution.

## Goals

- Replace manual Nsight GUI-only analysis with a reproducible CLI-driven workflow.
- Make layer/op-level analysis possible from `.nsys-rep` plus optional AD graph dump.
- Add host-runtime correlation so the workflow can distinguish GPU bottlenecks from host
  enqueue/sync bottlenecks.
- Make every iteration resumable, auditable, and explainable.
- Support `nsys`-only analysis, but strongly prefer a matching AutoDeploy graph dump.

## Non-Goals for V1

- No automatic source-code edits.
- No multi-change batch tuning by default.
- No requirement to support non-AutoDeploy backends.
- No dependence on Nsight GUI for core functionality.
- No requirement that deep host stack capture always be present.

## Inputs and Artifact Contracts

### Required Inputs

- model identifier
- workload command template for `trtllm-bench` or `trtllm-serve`
- serving config YAML
- benchmark log or permission to generate one

### Preferred Inputs

- `.nsys-rep` file from the exact run
- matching AutoDeploy graph dump directory from the same run or same config pipeline

### Optional Inputs

- PyTorch profiler `trace.json`
- prior iteration root
- one-layer run artifacts via `TLLM_OVERRIDE_LAYER_NUM=1`
- manually curated notes from Nsight GUI inspection

### Graph Dump Contract

Primary source is `AD_DUMP_GRAPHS_DIR`, which already exists in-tree.

Expected contents:

- ordered transform dumps like `001_<stage>_<transform>.txt`
- SSA-style FX graph text with node targets, args, shapes, and dtypes

The skill should request this artifact when available, but continue in degraded
mode without it.

## Artifact Layout

Use a stable working root:

- `artifacts/ad-perf-analysis/<model_slug>/<timestamp>/`

Structure:

- `session.json`
- `commands/`
- `inputs/`
- `iterations/iter_000_baseline/`
- `iterations/iter_001_<slug>/`

Per iteration:

- `iteration.json`
- `report.md`
- `bench.log`
- `metrics.json`
- `trace/trace.nsys-rep`
- `trace/trace.sqlite`
- `trace/trace.json`
- `trace/range_tree.json`
- `trace/kernel_timeline.json`
- `trace/layer_windows.json`
- `graph_dump/`
- `graph/graph_summary.json`
- `graph/node_index.json`
- `graph/stage_index.json`
- `graph/joined_op_timeline.json`
- `config/config_before.yaml`
- `config/config_after.yaml`
- `config/config_diff.patch`
- `analysis/findings.json`
- `analysis/hypotheses.json`
- `analysis/recommendations.json`
- `analysis/runtime_kernel_map.json`
- `analysis/memcpy_summary.json`
- `analysis/host_waits.json`
- `analysis/host_hotpaths.json`

## Skill Deliverables

### New Skill

- `.codex/skills/ad-perf-analysis/SKILL.md`
- `.codex/skills/ad-perf-analysis/agents/openai.yaml`

### Helper Scripts

- `.codex/skills/ad-perf-analysis/scripts/run_capture.py`
- `.codex/skills/ad-perf-analysis/scripts/export_nsys_sqlite.py`
- `.codex/skills/ad-perf-analysis/scripts/parse_nsys_ranges.py`
- `.codex/skills/ad-perf-analysis/scripts/parse_ad_graph_dump.py`
- `.codex/skills/ad-perf-analysis/scripts/join_trace_and_graph.py`
- `.codex/skills/ad-perf-analysis/scripts/analyze_layer_window.py`
- `.codex/skills/ad-perf-analysis/scripts/analyze_serving_config.py`
- `.codex/skills/ad-perf-analysis/scripts/parse_runtime_cuda_correlation.py`
- `.codex/skills/ad-perf-analysis/scripts/analyze_memcpy_timeline.py`
- `.codex/skills/ad-perf-analysis/scripts/analyze_host_waits.py`
- `.codex/skills/ad-perf-analysis/scripts/compare_iterations.py`
- `.codex/skills/ad-perf-analysis/scripts/render_iteration_report.py`

### References

- `.codex/skills/ad-perf-analysis/references/artifact-layout.md`
- `.codex/skills/ad-perf-analysis/references/heuristics.md`
- `.codex/skills/ad-perf-analysis/references/minimax-m2-profile-notes.md`
- `.codex/skills/ad-perf-analysis/references/host-correlation.md`

## Existing Local Knowledge To Reuse

The new skill should explicitly reuse patterns from:

- `.claude/skills/perf-nsight-systems/SKILL.md`
- `.claude/skills/perf-host-analysis/SKILL.md`
- `.claude/skills/perf-host-analysis/scripts/analyze_host_overhead.py`

Specifically borrow:

- structured `nsys stats` / `nsys analyze` usage
- host-bottleneck framing
- iteration isolation ideas
- GPU idle vs exposed host-prep reasoning

## Workflow

## Phase 1: Normalize Inputs

Capture:

- git SHA
- hostname
- GPU inventory
- CUDA version
- `nsys` version
- environment variables relevant to profiling and AutoDeploy
- workload command, dataset, model id, serving config, TP/PP/EP, backend

Persist in `session.json`.

## Phase 2: Collect or Import Artifacts

If the user provides artifacts, reuse them.

Otherwise generate them via one of two capture modes.

### Capture Mode A: `standard_trace`

Purpose:

- layer/op analysis
- runtime-to-kernel/memcpy/sync correlation
- lower overhead

Template:

- `TLLM_PROFILE_START_STOP`
- `TLLM_LLMAPI_ENABLE_NVTX=1`
- `TLLM_TORCH_PROFILE_TRACE=<path>`
- optional `AD_DUMP_GRAPHS_DIR=<path>`
- `nsys profile -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node`

### Capture Mode B: `deep_host_trace`

Purpose:

- host stack attribution
- blocking wait diagnosis
- richer memcpy/sync analysis

Template:

```bash
nsys profile \
  -t cuda,nvtx,osrt,python-gil \
  -s process-tree \
  --cpuctxsw=process-tree \
  --cudabacktrace=kernel:50000,memory:50000,sync:10000 \
  --cuda-graph-trace=node \
  -c cudaProfilerApi \
  ...
```

Notes:

- Requires CPU sampling for `--cudabacktrace`
- Higher overhead, use only when host bottleneck evidence exists or when the user asks
  for deep host analysis

## Phase 3: Export and Parse `nsys`

Primary machine-readable source is SQLite:

```bash
nsys export --type sqlite --output <trace.sqlite> <trace.nsys-rep>
```

### Core Tables

- `NVTX_EVENTS`
- `CUPTI_ACTIVITY_KIND_RUNTIME`
- `CUPTI_ACTIVITY_KIND_KERNEL`
- `CUPTI_ACTIVITY_KIND_MEMCPY`
- `CUPTI_ACTIVITY_KIND_MEMSET`
- `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`
- `StringIds`
- enum tables such as `ENUM_CUDA_MEMCPY_OPER`, `ENUM_CUDA_MEM_KIND`,
  `ENUM_CUPTI_SYNC_TYPE`

### Optional Deep Host Table

- `CUDA_CALLCHAINS`

Only available when captured with CPU sampling plus CUDA backtraces.

### Parsing Outputs

- `trace/range_tree.json`
- `trace/kernel_timeline.json`
- `trace/runtime_timeline.json`
- `trace/memcpy_timeline.json`
- `trace/sync_timeline.json`
- `trace/layer_windows.json`

## Phase 4: Parse AutoDeploy Graph Dump

Parse `AD_DUMP_GRAPHS_DIR` dumps to recover:

- transform order
- stage names
- graph modules
- node names and targets
- shapes and dtypes
- likely model-semantic groupings vs later fused/kernel-ish stages

Outputs:

- `graph/graph_summary.json`
- `graph/node_index.json`
- `graph/stage_index.json`

## Phase 5: Join Trace and Graph

### Join Priorities

1. NVTX ranges provide temporal/module boundaries
2. AD graph provides semantic structure
3. kernel names provide executed GPU work

### Matching Strategy

Use this precedence:

1. exact or normalized name match between AD graph nodes and NVTX/kernel names
2. layer/module-path alignment
3. ordered motif matching within a layer window
4. kernel-family fallback classification

### Classified GPU Families

- `qkv_gemm`
- `attn`
- `o_proj`
- `moe_router`
- `moe_dispatch`
- `moe_fc1`
- `moe_fc2`
- `moe_combine`
- `norm`
- `allreduce`
- `allgather`
- `reduce_scatter`
- `memcpy`
- `misc`

### Confidence Levels

Every semantic join result carries:

- `high`
- `medium`
- `low`

`high` requires both semantic and temporal support.

Outputs:

- `graph/joined_op_timeline.json`
- `analysis/scoped_layers.json`

## Phase 6: Host CUDA Runtime Correlation

This is a first-class analysis axis in V1.

### Authoritative Join Mechanism

Use `correlationId` to join:

- runtime API call -> kernel
- runtime API call -> memcpy
- runtime API call -> memset
- runtime API call -> synchronization record

This is the core host-to-GPU mapping.

### Host Stack Join

If `CUDA_CALLCHAINS` exists:

- join `CUPTI_ACTIVITY_KIND_RUNTIME.callchainId` -> `CUDA_CALLCHAINS.id`
- recover stack-attributed hot launch and wait paths

If not:

- still report runtime API correlation
- mark stack-level attribution unavailable

### Outputs

- `analysis/runtime_kernel_map.json`
- `analysis/host_hotpaths.json`

## Phase 7: Data Movement Analysis

Use `CUPTI_ACTIVITY_KIND_MEMCPY` plus enum tables.

### Required Classifications

By transfer kind:

- H2D
- D2H
- D2D
- P2P
- managed / unified-memory-related where present

By memory kind:

- pageable
- pinned
- device
- managed
- unknown

### Findings To Support

- unexpected steady-state D2H
- unnecessary H2D on hot path
- D2D traffic concentrated inside a layer window
- pageable-memory copy risk
- copy immediately followed by host sync
- memcpy serialized on a hot stream

Outputs:

- `analysis/memcpy_summary.json`

## Phase 8: Host Wait Analysis

Use:

- long runtime API durations
- matching records in `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`
- GPU idle/gap intervals
- optional OS runtime/backtrace data when available

### Wait Classes

- host blocked on stream sync
- host blocked on device sync
- host blocked on event wait
- host-side enqueue delay without GPU work
- GPU starvation caused by host-side gap

Outputs:

- `analysis/host_waits.json`

## Phase 9: Scope To One Representative Layer

Default target is one representative decode-layer window.

### Selection Policy

1. Prefer decode-only or generation-dominant windows when the goal is steady-state
   decode analysis.
2. Choose a stable iteration in the middle of the capture window.
3. Select a layer with a full motif:
   - qkv GEMM
   - attention
   - MoE if applicable
   - transition to next-layer boundary
4. If heterogeneous layers exist, allow layer-type-specific selection.

Per selected layer, compute:

- total layer latency
- per-op latency
- inter-kernel gaps
- communication overlap
- memcpy presence inside layer
- host API calls immediately preceding the layer
- any host waits that expose GPU idle before the layer starts

## Phase 10: Analyze Findings

### GPU-Side Findings

- kernel gaps
- missing overlap
- synchronization barriers
- cudagraph fragmentation
- excessive small kernels
- missing fusion opportunities
- MoE communication dominance
- backend mismatch
- scheduling underfill

### Host-Side Findings

- host launch bottleneck
- host blocked on stream sync
- host blocked on device sync
- excessive runtime API overhead
- stack-attributed hot launch path
- Python/OS-runtime contribution when deep host capture exists

### Data-Movement Findings

- H2D in steady-state decode
- D2H in hot path
- D2D concentration in one op family
- pageable-memory copy risk
- memcpy followed by sync
- memcpy on critical stream causing serialization

Every finding includes:

- statement
- evidence
- confidence
- likely cause
- recommended next action

## Phase 11: Analyze Serving Config

Compare the active config against:

- `examples/auto_deploy/model_registry/configs/`
- comparable models in the same size or architecture family

Scrub:

- `max_seq_len`
- `max_num_tokens`
- `max_batch_size`
- chunked prefill
- cudagraph usage
- piecewise cudagraph
- multistream MoE
- MoE communication settings
- attention backend
- compile backend
- transform toggles

Outputs:

- `analysis/config_findings.json`
- `analysis/comparable_configs.json`

## Phase 12: Propose Exactly One Change

Default policy:

- one isolated change per iteration

Change classes:

- serving config tweak
- diagnostic capture tweak
- benchmark-shape tweak

V1 does not auto-edit source code. Code-level opportunities are logged as
recommendations only.

Priority order:

1. highest-confidence, lowest-risk config improvement
2. highest-value observability improvement if current data is insufficient
3. code-change recommendation only when config space is exhausted

## Phase 13: Re-run and Compare

If an iteration includes an experiment:

- record config diff
- rerun benchmark and optionally reprofile
- compare metrics before/after

Compare:

- req/s
- output tok/s
- total tok/s
- latency
- scoped layer latency
- host wait totals
- memcpy totals by class
- targeted op-family latency
- GPU idle totals

Classify:

- improved
- regressed
- inconclusive

## Iteration Journal Schema

### `iteration.json`

Top-level:

- `iteration_id`
- `timestamp`
- `model`
- `objective`
- `inputs`
- `artifacts`
- `baseline_ref`
- `analysis_scope`
- `findings`
- `change`
- `execution`
- `results`
- `classification`
- `next_steps`

### Additional Result Groups

- `host_runtime_deltas`
- `memcpy_deltas`
- `host_wait_deltas`

### Finding Schema

- `id`
- `title`
- `category`
- `severity`
- `confidence`
- `evidence`
- `layer_scope`
- `op_scope`
- `host_scope`
- `impact_guess`
- `recommended_action`

## Human-Readable Report Format

Each `report.md` must include:

- `Context`
- `Artifacts Used`
- `Scope Chosen`
- `What I Inspected`
- `Layer and Op Findings`
- `Host CUDA API To GPU Correlation`
- `Data Movement Summary`
- `Host Wait Analysis`
- `Serving Config Findings`
- `Change Made`
- `Measured Impact`
- `Confidence and Caveats`
- `Recommended Next Experiments`

## Public Interfaces and Types

This introduces no TensorRT-LLM runtime API change, but it does define stable
skill/script interfaces.

### Skill Interface

Canonical inputs:

- `model`
- `workload_command`
- `config_yaml`
- `dataset`
- `nsys_rep` optional
- `ad_graph_dump_dir` optional
- `bench_log` optional
- `iteration_root` optional
- `analysis_mode` default `decode_layer`
- `target_layer` optional
- `capture_mode` default `standard_trace`
- `apply_change` boolean default `false`

### Script Interfaces

#### `run_capture.py`

Inputs:

- command template
- env vars
- output dir
- capture mode

Outputs:

- artifact paths JSON

#### `export_nsys_sqlite.py`

Inputs:

- `.nsys-rep`

Outputs:

- sqlite path JSON

#### `parse_nsys_ranges.py`

Inputs:

- sqlite path

Outputs:

- range tree
- kernel timeline
- layer windows

#### `parse_ad_graph_dump.py`

Inputs:

- graph dump dir

Outputs:

- graph summary
- node/stage indices

#### `join_trace_and_graph.py`

Inputs:

- parsed nsys
- parsed graph

Outputs:

- joined semantic timeline
- confidence stats

#### `parse_runtime_cuda_correlation.py`

Inputs:

- sqlite path

Outputs:

- runtime->kernel map
- runtime->memcpy map
- runtime->sync map
- optional runtime->callchain map

#### `analyze_memcpy_timeline.py`

Inputs:

- runtime correlation JSON
- memcpy timeline JSON

Outputs:

- transfer classification
- hot-path movement findings

#### `analyze_host_waits.py`

Inputs:

- runtime correlation JSON
- sync timeline JSON
- GPU idle intervals

Outputs:

- wait classification
- exposed host-wait findings

#### `analyze_layer_window.py`

Inputs:

- joined semantic timeline
- selected layer

Outputs:

- per-layer findings
- op metrics
- gap analysis

#### `analyze_serving_config.py`

Inputs:

- active config
- comparable config roots

Outputs:

- config findings
- candidate changes

#### `compare_iterations.py`

Inputs:

- previous iteration dir
- current iteration dir

Outputs:

- metric deltas
- verdict

#### `render_iteration_report.py`

Inputs:

- `iteration.json`

Outputs:

- `report.md`

## MiniMax M2 Heuristics Pack

V1 includes a MiniMax M2 heuristics pack describing:

- expected layer naming patterns
- expected qkv/attention/moe motifs
- likely MoE kernel families
- comparable registry configs to inspect first

The pack should be data-driven where possible.

## Tests

## Unit Tests

### `parse_nsys_ranges`

- nested NVTX range parsing
- name resolution through `text` and `textId`
- kernel containment and gap calculation

### `parse_ad_graph_dump`

- transform-order parsing
- SSA node extraction
- shape/dtype extraction
- tolerance to partial dump files

### `join_trace_and_graph`

- exact-name join
- fuzzy-name join
- motif fallback join
- confidence downgrade without graph dump

### `parse_runtime_cuda_correlation`

- runtime->kernel join by `correlationId`
- runtime->memcpy join by `correlationId`
- runtime->sync join by `correlationId`
- optional callchain join when present

### `analyze_memcpy_timeline`

- H2D/D2H/D2D/P2P classification
- src/dst memory-kind decoding
- pageable-memory warning logic

### `analyze_host_waits`

- stream-sync classification
- device-sync classification
- exposed wait vs hidden wait classification
- degradation when deep host capture is absent

### `analyze_layer_window`

- qkv->attention->moe motif detection
- gap reporting
- dense-layer handling

### `analyze_serving_config`

- comparable-config analysis
- missing chunked prefill
- oversized `max_seq_len`
- disabled multistream MoE detection

## Integration Tests

### Synthetic Nsight Smoke Test

- nested NVTX around CUDA ops
- export to SQLite
- verify kernel-to-range slicing

### Runtime Correlation Smoke Test

- synthetic profile with launches, memcpys, and syncs
- verify runtime->kernel/memcpy/sync joins

### One-Layer Oracle Test

- `TLLM_OVERRIDE_LAYER_NUM=1`
- verify exactly one layer motif is selected

### Graph-Dump Join Test

- capture `AD_DUMP_GRAPHS_DIR`
- verify at least one joined layer timeline reaches `high` confidence

### Deep Host Trace Test

- capture with CPU sampling plus `--cudabacktrace`
- verify stack-attributed runtime hotpaths when `CUDA_CALLCHAINS` is present

### End-to-End Dry Run

- import existing artifacts
- generate complete iteration report without mutating source code

## Acceptance Criteria

1. A user can run the skill on an AutoDeploy model with existing artifacts and
   receive a complete iteration report.
2. The skill works in `nsys`-only mode and clearly marks reduced semantic confidence.
3. When a matching AD graph dump is present, the skill produces layer/op-aware mapping.
4. The skill produces runtime-to-kernel/memcpy/sync correlation.
5. The skill classifies H2D, D2H, D2D, and P2P transfers when present.
6. The skill distinguishes host enqueue overhead from host blocking waits.
7. When deep host capture is present, the skill reports stack-attributed host hotpaths.
8. The skill logs one machine-readable record and one markdown report per iteration.
9. The skill proposes exactly one config-level next step by default.
10. The implementation is validated on MiniMax M2.

## Assumptions and Defaults

- Skill location: `.codex/skills/ad-perf-analysis/`
- Scope: generic AutoDeploy skill, MiniMax M2 first validated target
- V1 mode: analysis-first
- Change policy: one change per iteration
- AD graph policy: strongly recommended, not required
- Default target: representative decode-layer window
- Primary trace format: `nsys` SQLite export
- Preferred graph artifact source: `AD_DUMP_GRAPHS_DIR`
- Standard capture is the default
- Deep host capture is opt-in or triggered by host-bottleneck evidence
- If callchains are absent, stack-level findings are reported as unavailable, not inferred
- No automatic source-code edits in V1

## Sources Incorporated Into The Plan

- NVIDIA Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/UserGuide/>
- Nsight Systems SQLite schema reference:
  <https://archive.docs.nvidia.com/nsight-systems/2022.2/nsys-exporter/exported_data.html>
- Nsight Systems SQLite examples:
  <https://archive.docs.nvidia.com/nsight-systems/2022.4/nsys-exporter/examples.html>
- NVIDIA CUPTI activity docs:
  <https://docs.nvidia.com/cupti/12.9/api/structCUpti__ActivityAPI.html>
- Local skill references listed above

## Implementation Order

1. Create skill skeleton and `SKILL.md`.
2. Define session and iteration schemas plus artifact layout.
3. Implement SQLite export and base `nsys` parsers.
4. Implement runtime->kernel/memcpy/sync correlation.
5. Implement AD graph-dump parser.
6. Implement trace+graph join engine.
7. Implement layer-window selection and per-layer analysis.
8. Implement memcpy and host-wait analysis.
9. Implement serving-config scrubber.
10. Implement iteration comparator and markdown renderer.
11. Add MiniMax M2 heuristics pack.
12. Add unit and integration tests.
13. Validate on a real MiniMax M2 trace with and without graph dump, then with deep host trace.
