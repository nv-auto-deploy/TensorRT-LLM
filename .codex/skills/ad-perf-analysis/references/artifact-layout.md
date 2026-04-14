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
- `analysis/runtime_kernel_map.json`
- `analysis/memcpy_summary.json`
- `analysis/host_waits.json`
