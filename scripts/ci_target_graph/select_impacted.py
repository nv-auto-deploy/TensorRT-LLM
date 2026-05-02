# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select Bazel CI targets impacted by a source diff."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Protocol

SCHEMA_VERSION = 1
DEFAULT_TEST_UNIVERSES = ("//ci/bazel/...",)
GIT_DIFF_FILTER = "ACMRTUXBD"

_AUTODEPLOY_RUNTIME_LABEL = "//tensorrt_llm/_torch/auto_deploy:runtime"
_ACCURACY_TESTS_LABEL = "//tests/integration/defs/accuracy:accuracy_tests"
_CI_TARGET_GRAPH_LABEL = "//scripts/ci_target_graph:ci_target_graph_lib"

_BROAD_FALLBACK_EXACT_PATHS = {
    ".bazelrc",
    ".bazelversion",
    "MODULE.bazel",
    "MODULE.bazel.lock",
    "requirements.txt",
    "requirements_bazel_lock.txt",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
}
_BROAD_FALLBACK_PREFIXES = (
    ".github/",
    "ci/bazel/",
    "jenkins/",
    "platforms/",
    "tests/integration/test_lists/",
    "tools/bazel/",
)
_BROAD_FALLBACK_BASENAMES = {
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
}
_PLATFORM_CONSTRAINTS = {
    "//platforms:b200_1gpu": (
        "//platforms/gpu:b200",
        "//platforms/gpu_count:one",
    ),
    "//platforms:b200_4gpu": (
        "//platforms/gpu:b200",
        "//platforms/gpu_count:four",
    ),
    "//platforms:h100_1gpu": (
        "//platforms/gpu:h100",
        "//platforms/gpu_count:one",
    ),
    "//platforms:h100_4gpu": (
        "//platforms/gpu:h100",
        "//platforms/gpu_count:four",
    ),
}
_CONSTRAINT_GROUPS = (
    (
        "//platforms/gpu:b200",
        "//platforms/gpu:h100",
        "//platforms/gpu:none",
    ),
    (
        "//platforms/gpu_count:four",
        "//platforms/gpu_count:one",
    ),
)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:/")
_CQUERY_CONFIGURATION_SUFFIX_RE = re.compile(r"\s+\([^)]*\)$")


class BazelQueryError(RuntimeError):
    """Raised when a Bazel query command fails."""


class QueryClient(Protocol):
    """Minimal query interface used by the selector and tests."""

    def query(self, expression: str) -> list[str]:
        """Run a Bazel query expression and return label lines."""

    def cquery(self, expression: str, platform: str) -> list[str]:
        """Run a Bazel cquery expression for a target platform and return label lines."""


@dataclass(frozen=True)
class ChangedFile:
    """One changed-file evidence record from git or explicit CLI input."""

    status: str
    paths: tuple[str, ...]
    raw_paths: tuple[str, ...]
    source: str
    valid: bool = True
    reason: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "paths": list(self.paths),
            "raw_paths": list(self.raw_paths),
            "source": self.source,
            "status": self.status,
            "valid": self.valid,
        }
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class ChangedFilesResult:
    """Changed files plus fallback/warning evidence discovered while reading inputs."""

    changed_files: tuple[ChangedFile, ...]
    fallback_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectionResult:
    """Stable output for the impacted-target selector."""

    base: str
    head: str
    platform: str | None
    fallback_reasons: tuple[str, ...]
    changed_files: tuple[ChangedFile, ...]
    owner_labels: tuple[str, ...]
    candidate_targets: tuple[str, ...]
    selected_targets: tuple[str, ...]
    smoke_targets: tuple[str, ...]
    warnings: tuple[str, ...]
    selection_available: bool = True

    @property
    def fallback_used(self) -> bool:
        return bool(self.fallback_reasons)

    def changed_path_count(self) -> int:
        return len(
            _dedupe_sorted(
                path for changed_file in self.changed_files for path in changed_file.paths
            )
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "base": self.base,
            "candidate_targets": list(self.candidate_targets),
            "changed_files": [
                changed_file.to_json_dict()
                for changed_file in sorted(self.changed_files, key=_changed_file_sort_key)
            ],
            "fallback": {
                "reasons": list(self.fallback_reasons),
                "used": self.fallback_used,
            },
            "head": self.head,
            "owner_labels": list(self.owner_labels),
            "platform": self.platform,
            "schema_version": SCHEMA_VERSION,
            "selection_available": self.selection_available,
            "selected_targets": list(self.selected_targets),
            "smoke_targets": list(self.smoke_targets),
            "warnings": list(self.warnings),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)

    def to_human_text(self) -> str:
        lines = [
            "CI target impact selection",
            f"Base: {self.base}",
            f"Head: {self.head}",
            f"Platform: {self.platform or '(none)'}",
            (
                f"Changed files: {self.changed_path_count()} paths from "
                f"{len(self.changed_files)} records"
            ),
            f"Selection available: {'yes' if self.selection_available else 'no'}",
            f"Fallback used: {'yes' if self.fallback_used else 'no'}",
        ]

        lines.append("Fallback reasons:")
        lines.extend(_format_bullets(self.fallback_reasons))
        lines.append("Owner labels:")
        lines.extend(_format_bullets(self.owner_labels))
        lines.append(f"Candidate targets: {len(self.candidate_targets)}")
        lines.append("Selected targets:")
        lines.extend(_format_bullets(self.selected_targets))
        lines.append("Smoke targets:")
        lines.extend(_format_bullets(self.smoke_targets))
        lines.append("Warnings:")
        lines.extend(_format_bullets(self.warnings))
        return "\n".join(lines)


class BazelQueryClient:
    """Subprocess-backed Bazel query client."""

    def __init__(self, bazel_binary: str, repo_root: Path):
        command = shlex.split(bazel_binary)
        if not command:
            raise ValueError("--bazel-binary must not be empty")
        self._bazel_command = command
        self._repo_root = repo_root

    def query(self, expression: str) -> list[str]:
        return self._run(["query", expression, "--output=label"])

    def cquery(self, expression: str, platform: str) -> list[str]:
        return self._run(["cquery", expression, f"--platforms={platform}", "--output=label"])

    def _run(self, args: list[str]) -> list[str]:
        command = [*self._bazel_command, *args]
        try:
            result = subprocess.run(
                command,
                cwd=self._repo_root,
                check=False,
                capture_output=True,
                shell=False,
                text=True,
            )
        except OSError as error:
            raise BazelQueryError(f"{command[0]} failed to start: {error}") from error

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if detail:
                raise BazelQueryError(
                    f"{' '.join(command[:2])} failed with exit code {result.returncode}: {detail}"
                )
            raise BazelQueryError(
                f"{' '.join(command[:2])} failed with exit code {result.returncode}"
            )

        return _normalize_bazel_labels(result.stdout.splitlines())


def normalize_changed_path(raw_path: str) -> tuple[str | None, str | None]:
    """Normalize a changed path to POSIX repo-relative form."""
    if raw_path == "":
        return None, "empty path"

    candidate = raw_path.replace("\\", "/")
    if _WINDOWS_DRIVE_RE.match(candidate):
        return None, "absolute path"

    path = PurePosixPath(candidate)
    if path.is_absolute():
        return None, "absolute path"

    parts: list[str] = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            return None, "path escapes the repository"
        parts.append(part)

    if not parts:
        return None, "empty path"
    return PurePosixPath(*parts).as_posix(), None


def parse_name_status_z(payload: bytes | str, source: str = "git diff") -> ChangedFilesResult:
    """Parse ``git diff --name-status -z`` output."""
    if isinstance(payload, bytes):
        text = payload.decode("utf-8", errors="surrogateescape")
    else:
        text = payload

    fields = text.split("\0")
    if fields and fields[-1] == "":
        fields.pop()

    changed_files: list[ChangedFile] = []
    fallback_reasons: list[str] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status:
            raise ValueError("empty name-status record")

        path_count = 2 if status[0] in ("C", "R") else 1
        if index + path_count > len(fields):
            raise ValueError(f"malformed name-status record for status {status!r}")

        raw_paths = tuple(fields[index : index + path_count])
        index += path_count
        changed_file = changed_file_from_raw(status=status, raw_paths=raw_paths, source=source)
        changed_files.append(changed_file)
        if not changed_file.valid and changed_file.reason:
            fallback_reasons.append(changed_file.reason)

    return ChangedFilesResult(
        changed_files=tuple(changed_files),
        fallback_reasons=tuple(_dedupe_sorted(fallback_reasons)),
    )


def changed_file_from_raw(status: str, raw_paths: Iterable[str], source: str) -> ChangedFile:
    """Build a normalized changed-file record from raw path evidence."""
    raw_paths_tuple = tuple(raw_paths)
    paths: list[str] = []
    invalid_reasons: list[str] = []
    for raw_path in raw_paths_tuple:
        normalized_path, reason = normalize_changed_path(raw_path)
        if normalized_path is not None:
            paths.append(normalized_path)
        if reason is not None:
            invalid_reasons.append(f"{raw_path!r}: {reason}")

    if invalid_reasons:
        reason = f"invalid changed path in {source}: {', '.join(invalid_reasons)}"
        return ChangedFile(
            status=status,
            paths=tuple(paths),
            raw_paths=raw_paths_tuple,
            source=source,
            valid=False,
            reason=reason,
        )

    return ChangedFile(
        status=status,
        paths=tuple(_dedupe_preserving_order(paths)),
        raw_paths=raw_paths_tuple,
        source=source,
    )


def changed_files_from_paths(paths: Iterable[str], source: str) -> ChangedFilesResult:
    """Build changed-file records from explicit path-only inputs."""
    changed_files: list[ChangedFile] = []
    fallback_reasons: list[str] = []
    for raw_path in paths:
        changed_file = changed_file_from_raw(status="M", raw_paths=(raw_path,), source=source)
        changed_files.append(changed_file)
        if not changed_file.valid and changed_file.reason:
            fallback_reasons.append(changed_file.reason)
    return ChangedFilesResult(
        changed_files=tuple(changed_files),
        fallback_reasons=tuple(_dedupe_sorted(fallback_reasons)),
    )


def collect_changed_files(
    *,
    repo_root: Path,
    base: str,
    head: str,
    changed_files: Iterable[str],
    changed_files_files: Iterable[Path],
) -> ChangedFilesResult:
    """Collect changed files from explicit CLI inputs or git diff."""
    explicit_changed_files = list(changed_files)
    changed_files_file_paths = list(changed_files_files)
    if explicit_changed_files or changed_files_file_paths:
        return _collect_explicit_changed_files(explicit_changed_files, changed_files_file_paths)

    return _collect_git_changed_files(repo_root=repo_root, base=base, head=head)


def owner_labels_for_path(path: str) -> tuple[str, ...]:
    """Return modeled owner labels for a normalized repo-relative path."""
    labels: list[str] = []
    if path.startswith("tensorrt_llm/_torch/auto_deploy/") and path.endswith(".py"):
        labels.append(_AUTODEPLOY_RUNTIME_LABEL)

    accuracy_prefix = "tests/integration/defs/accuracy/"
    if path.startswith(accuracy_prefix) and path.endswith(".py"):
        remainder = path.removeprefix(accuracy_prefix)
        if "/" not in remainder:
            labels.append(_ACCURACY_TESTS_LABEL)

    graph_prefix = "scripts/ci_target_graph/"
    if path.startswith(graph_prefix) and path.endswith(".py"):
        remainder = path.removeprefix(graph_prefix)
        if "/" not in remainder:
            labels.append(_CI_TARGET_GRAPH_LABEL)

    return tuple(_dedupe_preserving_order(labels))


def broad_fallback_reason_for_path(path: str) -> str | None:
    """Return a broad fallback reason for CI/build/dependency policy paths."""
    if path in _BROAD_FALLBACK_EXACT_PATHS:
        return f"broad fallback for build/dependency policy change: {path}"
    if path.startswith("requirements") and path.endswith(".txt"):
        return f"broad fallback for Python dependency change: {path}"
    if path.endswith(".bzl") or PurePosixPath(path).name in _BROAD_FALLBACK_BASENAMES:
        return f"broad fallback for Bazel build metadata change: {path}"
    for prefix in _BROAD_FALLBACK_PREFIXES:
        if path.startswith(prefix):
            return f"broad fallback for CI/build/test policy path: {path}"
    return None


def select_impacted(
    *,
    repo_root: Path | str,
    base: str,
    head: str = "HEAD",
    platform: str | None = None,
    bazel_binary: str = "bazel",
    changed_files: Iterable[ChangedFile] | None = None,
    initial_fallback_reasons: Iterable[str] = (),
    initial_warnings: Iterable[str] = (),
    test_universes: Iterable[str] = DEFAULT_TEST_UNIVERSES,
    smoke_targets: Iterable[str] = (),
    manual_policy: str = "include",
    include_tags: Iterable[str] = (),
    exclude_tags: Iterable[str] = (),
    query_client: QueryClient | None = None,
) -> SelectionResult:
    """Select impacted targets from normalized changed-file records."""
    repo_root_path = Path(repo_root).resolve()
    if changed_files is None:
        changed_files_result = _collect_git_changed_files(
            repo_root=repo_root_path,
            base=base,
            head=head,
        )
        changed_files_tuple = changed_files_result.changed_files
        fallback_reasons = list(initial_fallback_reasons) + list(
            changed_files_result.fallback_reasons
        )
        warnings = list(initial_warnings) + list(changed_files_result.warnings)
    else:
        changed_files_tuple = tuple(changed_files)
        fallback_reasons = list(initial_fallback_reasons)
        warnings = list(initial_warnings)

    test_universes_tuple = tuple(test_universes) or DEFAULT_TEST_UNIVERSES
    include_tags_tuple = tuple(_dedupe_sorted(include_tags))
    exclude_tags_tuple = tuple(_dedupe_sorted(exclude_tags))
    smoke_targets_tuple = tuple(_dedupe_sorted(smoke_targets))

    if manual_policy not in ("include", "exclude", "only"):
        raise ValueError(f"unsupported manual policy: {manual_policy}")

    owner_labels = _owner_labels_and_fallback_reasons(changed_files_tuple, fallback_reasons)
    owner_label_tuple = tuple(_dedupe_sorted(owner_labels))

    if query_client is None:
        query_client = BazelQueryClient(bazel_binary=bazel_binary, repo_root=repo_root_path)

    candidate_targets: tuple[str, ...] = ()
    selected_base_targets: tuple[str, ...] = ()
    selection_available = True

    if fallback_reasons:
        candidate_targets, warnings, selection_available = _query_broad_fallback_targets(
            query_client=query_client,
            test_universes=test_universes_tuple,
            manual_policy=manual_policy,
            include_tags=include_tags_tuple,
            exclude_tags=exclude_tags_tuple,
            warnings=warnings,
        )
        selected_base_targets = candidate_targets
    elif owner_label_tuple:
        (
            candidate_targets,
            warnings,
            fallback_reasons,
            selection_available,
        ) = _query_impacted_or_fallback_targets(
            query_client=query_client,
            owner_labels=owner_label_tuple,
            test_universes=test_universes_tuple,
            manual_policy=manual_policy,
            include_tags=include_tags_tuple,
            exclude_tags=exclude_tags_tuple,
            warnings=warnings,
            fallback_reasons=fallback_reasons,
        )
        selected_base_targets = candidate_targets

    if platform and candidate_targets and selection_available:
        (
            selected_base_targets,
            candidate_targets,
            warnings,
            fallback_reasons,
            selection_available,
        ) = _apply_platform_filter(
            query_client=query_client,
            candidate_targets=candidate_targets,
            test_universes=test_universes_tuple,
            platform=platform,
            manual_policy=manual_policy,
            include_tags=include_tags_tuple,
            exclude_tags=exclude_tags_tuple,
            warnings=warnings,
            fallback_reasons=fallback_reasons,
        )

    selected_targets = (
        tuple(_dedupe_sorted([*selected_base_targets, *smoke_targets_tuple]))
        if selection_available
        else ()
    )
    return SelectionResult(
        base=base,
        head=head,
        platform=platform,
        fallback_reasons=tuple(_dedupe_sorted(fallback_reasons)),
        changed_files=tuple(sorted(changed_files_tuple, key=_changed_file_sort_key)),
        owner_labels=owner_label_tuple,
        candidate_targets=tuple(_dedupe_sorted(candidate_targets)),
        selected_targets=selected_targets,
        smoke_targets=smoke_targets_tuple,
        warnings=tuple(_dedupe_sorted(warnings)),
        selection_available=selection_available,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select Bazel CI targets impacted by changed files.",
    )
    parser.add_argument("--base", required=True, help="Base git ref for the diff.")
    parser.add_argument("--head", default="HEAD", help="Head git ref for the diff.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the TensorRT-LLM repository root.",
    )
    parser.add_argument(
        "--platform",
        help="Optional Bazel platform label used to cquery-filter candidate targets.",
    )
    parser.add_argument(
        "--bazel-binary",
        default="bazel",
        help="Bazel/Bazelisk command. Startup flags may be included and are split safely.",
    )
    parser.add_argument(
        "--changed-file",
        action="append",
        default=[],
        help="Explicit changed repo-relative path. May be repeated.",
    )
    parser.add_argument(
        "--changed-files-file",
        action="append",
        type=Path,
        default=[],
        help="File containing changed paths, newline-separated or git name-status NUL data.",
    )
    parser.add_argument(
        "--test-universe",
        action="append",
        default=[],
        help="Bazel test universe expression. May be repeated. Defaults to //ci/bazel/...",
    )
    parser.add_argument(
        "--smoke-target",
        action="append",
        default=[],
        help="Bazel target label always included in selected targets. May be repeated.",
    )
    parser.add_argument(
        "--manual-policy",
        choices=["include", "exclude", "only"],
        default="include",
        help="How to handle Bazel targets tagged manual. Defaults to include.",
    )
    parser.add_argument(
        "--include-tag",
        action="append",
        default=[],
        help="Only include targets matching this Bazel tag regex. May be repeated.",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="Exclude targets matching this Bazel tag regex. May be repeated.",
    )
    parser.add_argument("--json-output", type=Path, help="Write stable JSON result to this path.")
    parser.add_argument(
        "--targets-output",
        type=Path,
        help="Write selected target labels, one per line, to this path.",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "both"],
        default="text",
        help="Output format printed to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    changed_files_result = collect_changed_files(
        repo_root=repo_root,
        base=args.base,
        head=args.head,
        changed_files=args.changed_file,
        changed_files_files=args.changed_files_file,
    )
    result = select_impacted(
        repo_root=repo_root,
        base=args.base,
        head=args.head,
        platform=args.platform,
        bazel_binary=args.bazel_binary,
        changed_files=changed_files_result.changed_files,
        initial_fallback_reasons=changed_files_result.fallback_reasons,
        initial_warnings=changed_files_result.warnings,
        test_universes=args.test_universe or DEFAULT_TEST_UNIVERSES,
        smoke_targets=args.smoke_target,
        manual_policy=args.manual_policy,
        include_tags=args.include_tag,
        exclude_tags=args.exclude_tag,
    )

    if args.targets_output:
        args.targets_output.parent.mkdir(parents=True, exist_ok=True)
        args.targets_output.write_text(
            "".join(f"{target}\n" for target in result.selected_targets),
            encoding="utf-8",
        )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{result.to_json()}\n", encoding="utf-8")

    if args.output in ("text", "both"):
        sys.stdout.write(f"{result.to_human_text()}\n")
    if args.output in ("json", "both"):
        if args.output == "both":
            sys.stdout.write("\n")
        sys.stdout.write(f"{result.to_json()}\n")

    if not result.selection_available:
        return 4

    return 0


def _collect_explicit_changed_files(
    changed_files: list[str],
    changed_files_files: list[Path],
) -> ChangedFilesResult:
    records: list[ChangedFile] = []
    fallback_reasons: list[str] = []
    warnings: list[str] = []

    direct_result = changed_files_from_paths(changed_files, source="--changed-file")
    records.extend(direct_result.changed_files)
    fallback_reasons.extend(direct_result.fallback_reasons)

    for changed_files_file in changed_files_files:
        try:
            payload = changed_files_file.read_bytes()
        except OSError as error:
            reason = f"could not read changed-files file {changed_files_file}: {error}"
            fallback_reasons.append(reason)
            warnings.append(reason)
            continue

        source = str(changed_files_file)
        if b"\0" in payload:
            try:
                parsed = parse_name_status_z(payload, source=source)
            except ValueError as error:
                reason = f"could not parse changed-files file {changed_files_file}: {error}"
                fallback_reasons.append(reason)
                warnings.append(reason)
                continue
            records.extend(parsed.changed_files)
            fallback_reasons.extend(parsed.fallback_reasons)
            warnings.extend(parsed.warnings)
            continue

        text = payload.decode("utf-8", errors="surrogateescape")
        paths = [line.strip() for line in text.splitlines() if line.strip()]
        parsed = changed_files_from_paths(paths, source=source)
        records.extend(parsed.changed_files)
        fallback_reasons.extend(parsed.fallback_reasons)
        warnings.extend(parsed.warnings)

    return ChangedFilesResult(
        changed_files=tuple(records),
        fallback_reasons=tuple(_dedupe_sorted(fallback_reasons)),
        warnings=tuple(_dedupe_sorted(warnings)),
    )


def _collect_git_changed_files(repo_root: Path, base: str, head: str) -> ChangedFilesResult:
    diff_ref = f"{base}...{head}"
    command = [
        "git",
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        f"--diff-filter={GIT_DIFF_FILTER}",
        diff_ref,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            shell=False,
        )
    except OSError as error:
        reason = f"git diff failed to start for {diff_ref}: {error}"
        return ChangedFilesResult(fallback_reasons=(reason,), warnings=(reason,))

    if result.returncode != 0:
        detail = (
            (result.stderr or result.stdout or b"")
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )
        reason = f"git diff failed for {diff_ref}"
        if detail:
            reason = f"{reason}: {detail}"
        return ChangedFilesResult(fallback_reasons=(reason,), warnings=(reason,))

    try:
        return parse_name_status_z(result.stdout, source=f"git diff {diff_ref}")
    except ValueError as error:
        reason = f"could not parse git diff name-status output for {diff_ref}: {error}"
        return ChangedFilesResult(fallback_reasons=(reason,), warnings=(reason,))


def _owner_labels_and_fallback_reasons(
    changed_files: tuple[ChangedFile, ...],
    fallback_reasons: list[str],
) -> list[str]:
    owner_labels: list[str] = []
    for changed_file in changed_files:
        if not changed_file.valid:
            if changed_file.reason:
                fallback_reasons.append(changed_file.reason)
            continue

        if not changed_file.paths:
            fallback_reasons.append(f"changed-file record has no normalized paths: {changed_file}")
            continue

        for path in changed_file.paths:
            broad_reason = broad_fallback_reason_for_path(path)
            if broad_reason is not None:
                fallback_reasons.append(broad_reason)
                continue

            labels = owner_labels_for_path(path)
            if labels:
                owner_labels.extend(labels)
                continue

            fallback_reasons.append(f"unmodeled changed path: {path}")

    return owner_labels


def _query_impacted_or_fallback_targets(
    *,
    query_client: QueryClient,
    owner_labels: tuple[str, ...],
    test_universes: tuple[str, ...],
    manual_policy: str,
    include_tags: tuple[str, ...],
    exclude_tags: tuple[str, ...],
    warnings: list[str],
    fallback_reasons: list[str],
) -> tuple[tuple[str, ...], list[str], list[str], bool]:
    expression = build_impacted_query_expression(
        owner_labels=owner_labels,
        test_universes=test_universes,
        manual_policy=manual_policy,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )
    try:
        impacted_targets = tuple(_normalize_bazel_labels(query_client.query(expression)))
        if impacted_targets:
            return (
                impacted_targets,
                warnings,
                fallback_reasons,
                True,
            )

        reason = "impacted Bazel query returned no targets for modeled owners; using broad fallback"
        fallback_reasons.append(reason)
        fallback_targets, warnings, selection_available = _query_broad_fallback_targets(
            query_client=query_client,
            test_universes=test_universes,
            manual_policy=manual_policy,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            warnings=warnings,
        )
        return (
            fallback_targets,
            warnings,
            fallback_reasons,
            selection_available,
        )
    except BazelQueryError as error:
        reason = f"impacted Bazel query failed; using broad fallback: {error}"
        fallback_reasons.append(reason)
        warnings.append(reason)
        fallback_targets, warnings, selection_available = _query_broad_fallback_targets(
            query_client=query_client,
            test_universes=test_universes,
            manual_policy=manual_policy,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            warnings=warnings,
        )
        return fallback_targets, warnings, fallback_reasons, selection_available


def _query_broad_fallback_targets(
    *,
    query_client: QueryClient,
    test_universes: tuple[str, ...],
    manual_policy: str,
    include_tags: tuple[str, ...],
    exclude_tags: tuple[str, ...],
    warnings: list[str],
) -> tuple[tuple[str, ...], list[str], bool]:
    expression = build_fallback_query_expression(
        test_universes=test_universes,
        manual_policy=manual_policy,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )
    try:
        return tuple(_normalize_bazel_labels(query_client.query(expression))), warnings, True
    except BazelQueryError as error:
        warning = f"broad fallback Bazel query failed; target selection unavailable: {error}"
        warnings.append(warning)
        return (), warnings, False


def _apply_platform_filter(
    *,
    query_client: QueryClient,
    candidate_targets: tuple[str, ...],
    test_universes: tuple[str, ...],
    platform: str,
    manual_policy: str,
    include_tags: tuple[str, ...],
    exclude_tags: tuple[str, ...],
    warnings: list[str],
    fallback_reasons: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...], list[str], list[str], bool]:
    try:
        compatible_targets = _platform_compatible_targets(
            query_client=query_client,
            candidate_targets=candidate_targets,
            platform=platform,
        )
        return compatible_targets, candidate_targets, warnings, fallback_reasons, True
    except BazelQueryError as error:
        reason = f"platform cquery failed for {platform}; using broad unfiltered fallback: {error}"
        fallback_reasons.append(reason)
        warnings.append(reason)
        broad_targets, warnings, selection_available = _query_broad_fallback_targets(
            query_client=query_client,
            test_universes=test_universes,
            manual_policy=manual_policy,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            warnings=warnings,
        )
        if broad_targets:
            return broad_targets, broad_targets, warnings, fallback_reasons, True
        if not selection_available:
            return (), candidate_targets, warnings, fallback_reasons, False
        return candidate_targets, candidate_targets, warnings, fallback_reasons, True


def _platform_compatible_targets(
    *,
    query_client: QueryClient,
    candidate_targets: tuple[str, ...],
    platform: str,
) -> tuple[str, ...]:
    expression = _set_expression(candidate_targets, always_set=True)
    cquery_labels = set(_normalize_bazel_labels(query_client.cquery(expression, platform)))
    incompatible_labels = _platform_incompatible_targets(
        query_client=query_client,
        candidate_expression=expression,
        platform=platform,
    )
    compatible_labels = cquery_labels - incompatible_labels
    return tuple(label for label in candidate_targets if label in compatible_labels)


def _platform_incompatible_targets(
    *,
    query_client: QueryClient,
    candidate_expression: str,
    platform: str,
) -> set[str]:
    incompatible_constraints = _incompatible_constraints_for_platform(platform)
    incompatible_labels: set[str] = set()
    for constraint in incompatible_constraints:
        expression = (
            f'attr("target_compatible_with", {json.dumps(re.escape(constraint))}, '
            f"{candidate_expression})"
        )
        incompatible_labels.update(_normalize_bazel_labels(query_client.query(expression)))
    return incompatible_labels


def _incompatible_constraints_for_platform(platform: str) -> tuple[str, ...]:
    platform_constraints = set(_PLATFORM_CONSTRAINTS.get(platform, ()))
    if not platform_constraints:
        return ()

    incompatible_constraints: list[str] = []
    for constraint_group in _CONSTRAINT_GROUPS:
        selected_constraints = platform_constraints.intersection(constraint_group)
        if not selected_constraints:
            continue
        incompatible_constraints.extend(
            constraint for constraint in constraint_group if constraint not in selected_constraints
        )
    return tuple(sorted(incompatible_constraints))


def build_impacted_query_expression(
    *,
    owner_labels: Iterable[str],
    test_universes: Iterable[str],
    manual_policy: str,
    include_tags: Iterable[str],
    exclude_tags: Iterable[str],
) -> str:
    universe_expression = _set_expression(test_universes)
    owner_expression = _set_expression(owner_labels, always_set=True)
    expression = f'kind(".*_test rule", rdeps({universe_expression}, {owner_expression}))'
    return _apply_query_filters(
        expression,
        manual_policy=manual_policy,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )


def build_fallback_query_expression(
    *,
    test_universes: Iterable[str],
    manual_policy: str,
    include_tags: Iterable[str],
    exclude_tags: Iterable[str],
) -> str:
    universe_expression = _set_expression(test_universes)
    expression = f'kind(".*_test rule", tests({universe_expression}))'
    return _apply_query_filters(
        expression,
        manual_policy=manual_policy,
        include_tags=include_tags,
        exclude_tags=exclude_tags,
    )


def _apply_query_filters(
    expression: str,
    *,
    manual_policy: str,
    include_tags: Iterable[str],
    exclude_tags: Iterable[str],
) -> str:
    filtered_expression = expression
    if manual_policy == "exclude":
        filtered_expression = _except_tag_expression(filtered_expression, "manual")
    elif manual_policy == "only":
        filtered_expression = _tag_attr_expression(filtered_expression, "manual")

    for tag in _dedupe_sorted(include_tags):
        filtered_expression = _tag_attr_expression(filtered_expression, tag)
    for tag in _dedupe_sorted(exclude_tags):
        filtered_expression = _except_tag_expression(filtered_expression, tag)
    return filtered_expression


def _tag_attr_expression(expression: str, tag: str) -> str:
    return f'attr("tags", {json.dumps(re.escape(tag))}, {expression})'


def _except_tag_expression(expression: str, tag: str) -> str:
    return f"({expression}) except {_tag_attr_expression(expression, tag)}"


def _set_expression(values: Iterable[str], always_set: bool = False) -> str:
    labels = _dedupe_sorted(values)
    if not labels:
        raise ValueError("query set must not be empty")
    if len(labels) == 1 and not always_set:
        return labels[0]
    return f"set({' '.join(labels)})"


def _normalize_bazel_labels(labels: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for label in labels:
        cleaned = _CQUERY_CONFIGURATION_SUFFIX_RE.sub("", label.strip())
        if cleaned:
            normalized.append(cleaned)
    return _dedupe_sorted(normalized)


def _changed_file_sort_key(
    changed_file: ChangedFile,
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    return (changed_file.paths, changed_file.raw_paths, changed_file.source, changed_file.status)


def _format_bullets(values: Iterable[str]) -> list[str]:
    sorted_values = _dedupe_sorted(values)
    if not sorted_values:
        return ["  (none)"]
    return [f"  - {value}" for value in sorted_values]


def _dedupe_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}))


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


if __name__ == "__main__":
    raise SystemExit(main())
