# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TIMESTAMP_UTILS_SMOKE_TARGET = "//cpp:timestamp_utils_smoke_test"
COMMON_HOST_SMOKE_TARGET = "//cpp:common_host_smoke_test"
GLOBAL_TIMER_KERNEL_TARGET = "//cpp:global_timer_kernel_cuda"


def _find_bazel() -> str | None:
    return shutil.which("bazel") or shutil.which("bazelisk")


def _run_bazel(args: list[str]) -> subprocess.CompletedProcess[str]:
    bazel = _find_bazel()
    if bazel is None:
        pytest.skip("bazel or bazelisk is not available")

    return subprocess.run(
        [bazel, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_host_smoke_targets_build_with_bazel() -> None:
    result = _run_bazel(
        [
            "test",
            TIMESTAMP_UTILS_SMOKE_TARGET,
            COMMON_HOST_SMOKE_TARGET,
            "--test_output=errors",
            "--cache_test_results=no",
        ]
    )

    assert result.returncode == 0, (
        "host Bazel smoke targets failed "
        f"with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_cuda_kernel_seed_builds_with_bazel() -> None:
    result = _run_bazel(["build", GLOBAL_TIMER_KERNEL_TARGET])

    assert result.returncode == 0, (
        f"{GLOBAL_TIMER_KERNEL_TARGET} failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
