# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two iteration records and classify the result."""

from __future__ import annotations

import argparse

from _common import read_json, write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import compare_iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    previous_payload = read_json(args.previous)
    current_payload = read_json(args.current)
    payload = compare_iterations(previous_payload, current_payload)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
