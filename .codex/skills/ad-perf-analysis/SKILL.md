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
- producing a compact SQLite-backed summary for very large traces
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
3. Run `summarize_nsys_sqlite.py` first to get a cheap summary of trace scale,
   top kernels, top runtime calls, memcpy totals, GPU gaps, and capture-log
   transform timings.
4. Decide whether the full JSON parser path is appropriate:
   - If the trace is moderate, parse the trace with:
     - `parse_nsys_ranges.py`
     - `parse_runtime_cuda_correlation.py`
   - If the trace is very large, keep the compact SQLite summary as the primary
     artifact and only run the full JSON parsers if layer-level semantic joining
     is worth the cost.
5. If an AD graph dump is present, parse it with `parse_ad_graph_dump.py`.
6. Join trace semantics and graph semantics with `join_trace_and_graph.py`.
7. Pick a representative decode layer and analyze it with `analyze_layer_window.py`.
8. Analyze data movement with `analyze_memcpy_timeline.py`.
9. Analyze host waits with `analyze_host_waits.py`.
10. Analyze the serving config with `analyze_serving_config.py`.
11. Assemble an iteration record and render it with `render_iteration_report.py`.

## Scaling Guidance

Start with the compact summary whenever the trace is large enough that the full
JSON parser path may explode in size. Practical signs include:

- multi-million kernels or runtime calls
- parser RSS climbing into tens of GB
- intermediate JSON files growing into multi-GB artifacts
- the need to answer "what dominates?" before doing layer-level semantic joins

For large traces:

- do not run `parse_nsys_ranges.py` and `parse_runtime_cuda_correlation.py` in
  parallel by default
- prefer `summarize_nsys_sqlite.py` plus targeted SQLite queries first
- treat giant parser outputs as optional artifacts, not mandatory outputs
- only pay the cost of full layer-window reconstruction when the next decision
  truly depends on it

## Capture Guidance

Default `nsys` mode:

```bash
TLLM_PROFILE_START_STOP=500-1000 \
TLLM_LLMAPI_ENABLE_NVTX=1 \
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

Only enable `TLLM_TORCH_PROFILE_TRACE=trace.json` when the PyTorch trace is the
primary goal. During `nsys` runs it can contend for CUPTI ownership, produce
partial data, or fail to materialize the trace file.

If the run has a very long tail after the capture window closes, it is acceptable
to stop it after the application reports that the capture range ended, as long as
the `.nsys-rep` generation has started. In that mode, expect the benchmark report
or auxiliary trace files to be absent.

When the first profiling attempt does not reach the capture window because the
model path is unstable, create a minimal config override that only unblocks the
capture. Keep that override narrow and record exactly which knobs changed.

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

## Operational Notes

- Distinguish startup, steady-state, and teardown before drawing conclusions from
  the largest GPU gaps. Very large idle windows are often outside the hot decode
  region.
- If the graph dump is present but full parsing is too expensive, keep the raw
  dump and compact SQLite summary first; semantic graph joins can be deferred.
- Preserve fresh artifacts even when the benchmark is interrupted. A complete
  `.nsys-rep` plus SQLite export is often more valuable than a finished benchmark
  report.
- If profiling leaves stale MPI or worker processes behind, clean them up before
  the next attempt so the next capture is not polluted by OOM or lingering GPU
  allocations.

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
- Heuristics: `references/heuristics.md`
- Host correlation notes: `references/host-correlation.md`
- MiniMax M2 notes: `references/minimax-m2-profile-notes.md`
