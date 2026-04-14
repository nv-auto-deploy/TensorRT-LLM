# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze memcpy activity from correlated runtime records."""

from __future__ import annotations

import argparse

from _common import read_json, write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import analyze_memcpy_timeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime_payload = read_json(args.runtime_json)
    payload = analyze_memcpy_timeline(runtime_payload)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
