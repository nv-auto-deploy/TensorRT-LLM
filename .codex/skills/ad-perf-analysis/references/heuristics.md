<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Heuristics

- Prefer one change per iteration.
- Prefer reusing existing artifacts before recollecting them.
- If the graph dump is absent, continue analysis but mark semantic confidence lower.
- If host waits dominate GPU gaps, collect a deep host trace before recommending deeper GPU work.
- If `nsys` is the primary tool, disable `TLLM_TORCH_PROFILE_TRACE` unless the
  PyTorch trace is explicitly required.
- If the capture window has already closed and the run tail is very long, prefer
  harvesting the finished `.nsys-rep` over waiting for the full benchmark to end.
- For traces with multi-million kernels or runtime calls, start with the compact
  SQLite summary and avoid parallel full-parser runs by default.
- Treat very large GPU idle gaps as phase-boundary candidates first. Prove they
  are in steady-state before turning them into decode-path recommendations.
- If the first runnable path requires a temporary config override, keep the
  override minimal and record each changed knob with the reason it was needed.
