---
name: ad-perf-analysis
description: Analyze AutoDeploy performance using Nsight Systems traces, AutoDeploy graph dumps, host CUDA runtime correlation, data-movement analysis, and iterative serving-config recommendations.
---

# AutoDeploy Performance Analysis

Use this skill when the user wants to analyze or optimize an AutoDeploy model with
`nsys`, a graph dump, or both.

This skill is analysis-first. It focuses on:

- importing or collecting profiling artifacts
- exporting `.nsys-rep` to SQLite
- mapping kernels to layer/op scopes with NVTX and AD graph dumps
- correlating host CUDA API calls to kernels, memcpys, memsets, and syncs
- classifying H2D / D2H / D2D / P2P transfers
- finding host waits that expose GPU idle time
- inspecting serving configs and suggesting one isolated next experiment
- producing a structured iteration record and a markdown report

## Inputs

Preferred inputs:

- `.nsys-rep`
- AutoDeploy graph dump directory from `AD_DUMP_GRAPHS_DIR`
- benchmark log
- serving config YAML

Optional:

- PyTorch `trace.json`
- previous iteration record
- one-layer debug run captured with `TLLM_OVERRIDE_LAYER_NUM=1`

## Core Workflow

1. Normalize the user inputs and artifact paths.
2. Export `.nsys-rep` to SQLite with `export_nsys_sqlite.py`.
3. Parse the trace with:
   - `parse_nsys_ranges.py`
   - `parse_runtime_cuda_correlation.py`
4. If an AD graph dump is present, parse it with `parse_ad_graph_dump.py`.
5. Join trace semantics and graph semantics with `join_trace_and_graph.py`.
6. Pick a representative decode layer and analyze it with `analyze_layer_window.py`.
7. Analyze data movement with `analyze_memcpy_timeline.py`.
8. Analyze host waits with `analyze_host_waits.py`.
9. Analyze the serving config with `analyze_serving_config.py`.
10. Assemble an iteration record and render it with `render_iteration_report.py`.

## Capture Guidance

Default trace mode:

```bash
TLLM_PROFILE_START_STOP=500-1000 \
TLLM_LLMAPI_ENABLE_NVTX=1 \
TLLM_TORCH_PROFILE_TRACE=trace.json \
AD_DUMP_GRAPHS_DIR=ad_graphs \
nsys profile \
  -o trace_output \
  -f true \
  -t cuda,nvtx,python-gil \
  -c cudaProfilerApi \
  --cuda-graph-trace node \
  --trace-fork-before-exec=true \
  ...
```

Deep host trace mode:

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

Use deep host tracing only when host bottleneck evidence exists or when the user
specifically wants stack-attributed host analysis.

## Output Expectations

Each iteration should produce:

- a machine-readable iteration JSON record
- a markdown report
- enough supporting JSON to explain:
  - layer-level findings
  - host CUDA API to GPU correlation
  - data movement
  - host waits
  - serving-config findings

## References

- Artifact layout: `references/artifact-layout.md`
- Host correlation notes: `references/host-correlation.md`
- MiniMax M2 notes: `references/minimax-m2-profile-notes.md`
