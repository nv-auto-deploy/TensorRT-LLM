<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Artifact Layout

Suggested root:

- `artifacts/ad-perf-analysis/<model_slug>/<timestamp>/`

Per iteration:

- `iteration.json`
- `report.md`
- `trace/trace.nsys-rep`
- `trace/trace.sqlite`
- `graph_dump/`
- `analysis/sqlite_summary.json`
- `analysis/runtime_kernel_map.json`
- `analysis/memcpy_summary.json`
- `analysis/host_waits.json`

Notes:

- `analysis/sqlite_summary.json` is the preferred first-pass artifact for large
  traces because it stays compact while still capturing top kernels, top runtime
  calls, memcpy totals, GPU gap summaries, and capture-log transform timings.
- `analysis/runtime_kernel_map.json` and other full JSON parser outputs are
  optional for very large traces. They may grow to multi-GB artifacts and are
  not required when the immediate goal is to identify dominant costs.
