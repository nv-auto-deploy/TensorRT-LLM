# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join parsed trace artifacts with parsed AutoDeploy graph artifacts."""

from __future__ import annotations

import argparse

from _common import read_json, write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import join_trace_and_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", required=True)
    parser.add_argument("--graph-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trace_payload = read_json(args.trace_json)
    graph_payload = read_json(args.graph_json) if args.graph_json else None
    joined = join_trace_and_graph(trace_payload, graph_payload)
    write_json(args.output, joined)


if __name__ == "__main__":
    main()
