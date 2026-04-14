# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render one iteration JSON record into Markdown."""

from __future__ import annotations

import argparse

from _common import read_json, write_text

from tensorrt_llm.tools.profiler.ad_perf_analysis import render_iteration_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    record = read_json(args.iteration_json)
    report = render_iteration_report(record)
    write_text(args.output, report)


if __name__ == "__main__":
    main()
