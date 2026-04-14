# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export an `.nsys-rep` file to SQLite."""

from __future__ import annotations

import argparse

from tensorrt_llm.tools.profiler.ad_perf_analysis import export_nsys_sqlite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsys-rep", required=True)
    parser.add_argument("--output")
    parser.add_argument("--nsys-cmd", default="nsys")
    args = parser.parse_args()

    sqlite_path = export_nsys_sqlite(args.nsys_rep, output_path=args.output, nsys_cmd=args.nsys_cmd)
    print(str(sqlite_path))


if __name__ == "__main__":
    main()
