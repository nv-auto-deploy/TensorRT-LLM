# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse NVTX ranges, kernels, and layer windows from an Nsight SQLite export."""

from __future__ import annotations

import argparse

from _common import write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import load_nsys_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace_data = load_nsys_trace(args.sqlite)
    payload = {
        "sqlite_path": trace_data["sqlite_path"],
        "range_tree": trace_data["range_tree"],
        "ranges": trace_data["ranges"],
        "kernels": trace_data["kernels"],
        "layer_windows": trace_data["layer_windows"],
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
