# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select and analyze a representative layer window."""

from __future__ import annotations

import argparse

from _common import read_json, write_json

from tensorrt_llm.tools.profiler.ad_perf_analysis import (
    analyze_layer_window,
    select_representative_layer_window,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined-json", required=True)
    parser.add_argument("--runtime-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    joined_payload = read_json(args.joined_json)
    runtime_payload = read_json(args.runtime_json) if args.runtime_json else None
    selected = select_representative_layer_window(joined_payload["joined_timeline"])
    analysis = analyze_layer_window(selected, runtime_payload)
    write_json(args.output, analysis)


if __name__ == "__main__":
    main()
