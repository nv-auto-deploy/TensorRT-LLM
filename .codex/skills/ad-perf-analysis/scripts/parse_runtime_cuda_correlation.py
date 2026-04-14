# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correlate host CUDA runtime calls with GPU-side events."""

from __future__ import annotations

import argparse

from _common import write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import parse_runtime_cuda_correlation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = parse_runtime_cuda_correlation(args.sqlite)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
