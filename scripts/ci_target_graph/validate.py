# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the shadow CI target graph manifest against its JSON schema."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ci_target_graph.generate import build_manifest

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("manifest_schema_v2.json")


def schema_errors(
    manifest: dict[str, Any],
    schema_path: Path = DEFAULT_SCHEMA_PATH,
) -> list[str]:
    """Return schema validation errors for ``manifest``."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    try:
        from jsonschema.validators import validator_for
    except ImportError as error:
        raise RuntimeError(
            "jsonschema is required for manifest validation; install requirements.txt."
        ) from error

    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator = validator_class(schema)
    errors = sorted(validator.iter_errors(manifest), key=lambda error: list(error.path))
    return [_format_schema_error(error) for error in errors]


def targets_without_jenkins_stage(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return manifest targets that have no matching Jenkins stage candidate."""
    targets = manifest.get("targets", [])
    if not isinstance(targets, list):
        return []

    missing_targets: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        source = target.get("source")
        if not isinstance(source, dict):
            continue
        if not source.get("jenkins_stages"):
            missing_targets.append(target)
    return missing_targets


def summarize_missing_jenkins_stage_targets(
    targets: list[dict[str, Any]],
) -> list[tuple[int, str, str, str]]:
    """Summarize missing Jenkins stage candidates by YAML, stage, and backend."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for target in targets:
        source = target.get("source", {})
        constraints = target.get("constraints", {})
        yaml_path = str(source.get("yaml") or "<unknown>")
        stage = str(constraints.get("stage") or "<unset>")
        backend = str(constraints.get("backend") or "<unset>")
        counts[(yaml_path, stage, backend)] += 1

    return [
        (count, yaml_path, stage, backend)
        for (yaml_path, stage, backend), count in sorted(counts.items())
    ]


def targets_with_incomplete_runtime_metadata(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return manifest targets whose runtime metadata is explicitly incomplete."""
    targets = manifest.get("targets", [])
    if not isinstance(targets, list):
        return []

    incomplete_targets: list[dict[str, Any]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        runtime = target.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("metadata_complete") is not True:
            incomplete_targets.append(target)
    return incomplete_targets


def summarize_incomplete_runtime_metadata_targets(
    targets: list[dict[str, Any]],
) -> list[tuple[int, str, str, str]]:
    """Summarize incomplete runtime metadata by YAML, backend, and missing fields."""
    counts: Counter[tuple[str, str, str]] = Counter()
    for target in targets:
        source = target.get("source", {})
        runtime = target.get("runtime", {})
        yaml_path = str(source.get("yaml") or "<unknown>")
        backend = str(runtime.get("backend") or "<unset>")
        missing = runtime.get("missing") if isinstance(runtime, dict) else None
        if isinstance(missing, list) and missing:
            missing_key = ",".join(str(item) for item in missing)
        else:
            missing_key = "<runtime object missing>"
        counts[(yaml_path, backend, missing_key)] += 1

    return [
        (count, yaml_path, backend, missing_key)
        for (yaml_path, backend, missing_key), count in sorted(counts.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate and validate the shadow CI target graph manifest."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the TensorRT-LLM repository root.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="JSON schema path for validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the regenerated manifest after validation.",
    )
    parser.add_argument(
        "--fail-on-missing-jenkins-stage",
        action="store_true",
        help="Return a non-zero exit code when targets have no Jenkins stage candidate.",
    )
    parser.add_argument(
        "--fail-on-incomplete-runtime-metadata",
        action="store_true",
        help="Return a non-zero exit code when targets have incomplete runtime metadata.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.repo_root)
    try:
        errors = schema_errors(manifest, args.schema)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if errors:
        print("manifest schema validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    missing_stage_targets = targets_without_jenkins_stage(manifest)
    incomplete_runtime_targets = targets_with_incomplete_runtime_metadata(manifest)
    print(
        f"validated {len(manifest.get('targets', []))} manifest targets against {args.schema}",
        file=sys.stderr,
    )
    _report_missing_jenkins_stage_targets(missing_stage_targets)
    _report_incomplete_runtime_metadata_targets(incomplete_runtime_targets)

    if args.output:
        output = json.dumps(manifest, indent=2, sort_keys=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")

    if args.fail_on_missing_jenkins_stage and missing_stage_targets:
        return 3
    if args.fail_on_incomplete_runtime_metadata and incomplete_runtime_targets:
        return 4
    return 0


def _format_schema_error(error: Any) -> str:
    path = "$"
    for part in error.absolute_path:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return f"{path}: {error.message}"


def _report_missing_jenkins_stage_targets(targets: list[dict[str, Any]]) -> None:
    if not targets:
        print("all manifest targets have at least one Jenkins stage candidate", file=sys.stderr)
        return

    print(
        "warning: "
        f"{len(targets)} manifest targets have no Jenkins stage candidate; "
        "treat these as conservative fallback inputs until the Jenkins mapping is modeled.",
        file=sys.stderr,
    )
    for count, yaml_path, stage, backend in summarize_missing_jenkins_stage_targets(targets):
        print(
            f"  - {count}: {yaml_path} stage={stage} backend={backend}",
            file=sys.stderr,
        )


def _report_incomplete_runtime_metadata_targets(targets: list[dict[str, Any]]) -> None:
    if not targets:
        print("all manifest targets have complete runtime metadata", file=sys.stderr)
        return

    print(
        "warning: "
        f"{len(targets)} manifest targets have incomplete runtime metadata; "
        "use metadata:runtime_incomplete tags as conservative query inputs.",
        file=sys.stderr,
    )
    for count, yaml_path, backend, missing_key in summarize_incomplete_runtime_metadata_targets(
        targets
    ):
        print(
            f"  - {count}: {yaml_path} backend={backend} missing={missing_key}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    raise SystemExit(main())
