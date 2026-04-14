# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse an AutoDeploy graph dump directory."""

from __future__ import annotations

import argparse

from _common import write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import parse_ad_graph_dump_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dump-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = parse_ad_graph_dump_dir(args.graph_dump_dir)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
