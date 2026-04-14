# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a capture command for AutoDeploy performance analysis."""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", required=True, help="Shell command to execute.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env", action="append", default=[], help="KEY=VALUE pairs.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for env_entry in args.env:
        key, value = env_entry.split("=", maxsplit=1)
        env[key] = value

    subprocess.run(args.command, shell=True, check=True, cwd=output_dir, env=env)  # nosec B602
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "command": args.command,
                "env_overrides": args.env,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
