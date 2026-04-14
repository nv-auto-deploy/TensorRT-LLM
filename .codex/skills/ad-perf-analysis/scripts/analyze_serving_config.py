# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze one serving config against heuristic checks and comparable configs."""

from __future__ import annotations

import argparse

from _common import write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import analyze_serving_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--comparable-root", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = analyze_serving_config(args.config, comparable_roots=args.comparable_root)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
