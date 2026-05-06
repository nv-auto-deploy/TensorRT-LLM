# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TIMESTAMP_UTILS_SMOKE_TARGET = "//cpp:timestamp_utils_smoke_test"


def _find_bazel() -> str | None:
    return shutil.which("bazel") or shutil.which("bazelisk")


def test_timestamp_utils_smoke_target_builds_with_bazel() -> None:
    bazel = _find_bazel()
    if bazel is None:
        pytest.skip("bazel or bazelisk is not available")

    result = subprocess.run(
        [
            bazel,
            "test",
            TIMESTAMP_UTILS_SMOKE_TARGET,
            "--test_output=errors",
            "--cache_test_results=no",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"{TIMESTAMP_UTILS_SMOKE_TARGET} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
