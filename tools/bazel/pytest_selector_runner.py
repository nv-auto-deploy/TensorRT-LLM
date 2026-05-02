# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a pytest selector from a resolved TensorRT-LLM repository root."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pytest selector from the repository root.")
    parser.add_argument("--selector", required=True, help="Pytest selector to execute.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        help=(
            "TensorRT-LLM repository root. Defaults to TRTLLM_BAZEL_REPO_ROOT, "
            "then BUILD_WORKSPACE_DIRECTORY."
        ),
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional pytest argument. Repeat once per argument.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pytest command without executing it.",
    )
    args = parser.parse_args()

    try:
        repo_root = _resolve_repo_root(args.repo_root)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    selector = _normalize_selector(args.selector, repo_root)
    command = [sys.executable, "-m", "pytest", selector, *args.pytest_arg]
    if args.dry_run:
        print(f"cwd: {repo_root}")
        print(shlex.join(command))
        return 0

    completed = subprocess.run(command, cwd=repo_root)
    return completed.returncode


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return _validated_repo_root(repo_root)

    for env_name in ("TRTLLM_BAZEL_REPO_ROOT", "BUILD_WORKSPACE_DIRECTORY"):
        env_value = os.environ.get(env_name)
        if env_value:
            return _validated_repo_root(Path(env_value))

    raise ValueError(
        "repository root is required; pass --repo-root or set TRTLLM_BAZEL_REPO_ROOT "
        "or BUILD_WORKSPACE_DIRECTORY. Bazel tests can pass "
        "--test_env=TRTLLM_BAZEL_REPO_ROOT=<checkout>."
    )


def _normalize_selector(selector: str, repo_root: Path) -> str:
    path_part, separator, suffix = selector.partition("::")
    if not path_part:
        return selector

    if (repo_root / path_part).exists():
        return selector

    integration_path = Path("tests/integration/defs") / path_part
    if (repo_root / integration_path).exists():
        normalized_path = integration_path.as_posix()
        if separator:
            return f"{normalized_path}{separator}{suffix}"
        return normalized_path

    return selector


def _validated_repo_root(repo_root: Path) -> Path:
    resolved = repo_root.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"repository root does not exist or is not a directory: {resolved}")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
