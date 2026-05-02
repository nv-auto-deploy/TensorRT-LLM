# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a shadow-mode CI target graph manifest from Jenkins/test-db metadata."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .selector_parser import ParsedSelector, parse_pytest_selector

SCHEMA_VERSION = 1
TEST_DB_RELATIVE_DIR = Path("tests/integration/test_lists/test-db")
JENKINS_L0_TEST_RELATIVE_PATH = Path("jenkins/L0_Test.groovy")

_DIRECT_STAGE_RE = re.compile(r'"(?P<stage>(?:\\.|[^"\\])*)"\s*:\s*\[(?P<values>[^\]]*)\]')
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")

_BACKEND_STAGE_KEYWORDS = {
    "autodeploy": ["AUTODEPLOY"],
    "cpp": ["CPP"],
    "fmha": ["FMHA"],
    "pytorch": ["PYTORCH"],
    "tensorrt": ["TENSORRT", "TRT"],
    "triton": ["TRITON"],
    "verl": ["VERL"],
}
_AUTO_TRIGGER_STAGE_KEYWORDS = {
    "deepseek": ["DEEPSEEK"],
    "gpt_oss": ["GPTOSS", "GPT_OSS", "GPT-OSS"],
    "others": ["OTHERS"],
}


@dataclass(frozen=True)
class StageConfig:
    """Jenkins stage mapping extracted from ``jenkins/L0_Test.groovy``."""

    name: str
    platform: str
    yaml_stem: str
    split_id: int
    split_count: int
    gpu_count: int = 1
    node_count: int = 1
    run_with_sbatch: bool = False


def build_manifest(repo_root: Path | str) -> dict[str, Any]:
    """Build the versioned shadow CI target manifest."""
    repo_root = Path(repo_root).resolve()
    stage_configs = parse_jenkins_stage_configs(repo_root / JENKINS_L0_TEST_RELATIVE_PATH)
    stages_by_yaml: dict[str, list[StageConfig]] = defaultdict(list)
    for stage_config in stage_configs:
        stages_by_yaml[stage_config.yaml_stem].append(stage_config)

    targets = []
    for yaml_path in _iter_test_db_yaml_paths(repo_root):
        targets.extend(_targets_from_yaml(repo_root, yaml_path, stages_by_yaml))

    targets.sort(key=lambda target: target["target_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "targets": targets,
    }


def parse_jenkins_stage_configs(groovy_path: Path) -> list[StageConfig]:
    """Parse direct map entries and simple ``buildStageConfigs`` calls from L0_Test.groovy."""
    text = _strip_groovy_comments(groovy_path.read_text(encoding="utf-8"))
    stage_configs: dict[str, StageConfig] = {}

    for match in _DIRECT_STAGE_RE.finditer(text):
        values = _parse_groovy_list(match.group("values"))
        if values is None:
            continue
        stage_config = _stage_config_from_values(match.group("stage"), values)
        if stage_config is not None:
            stage_configs[stage_config.name] = stage_config

    for stage_config in _parse_build_stage_config_calls(text):
        stage_configs[stage_config.name] = stage_config

    return [stage_configs[name] for name in sorted(stage_configs)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a shadow-mode CI target graph manifest.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the TensorRT-LLM repository root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path. If omitted, the manifest is written to stdout.",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.repo_root)
    output = json.dumps(manifest, indent=2, sort_keys=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        sys.stdout.write(f"{output}\n")

    return 0


def _targets_from_yaml(
    repo_root: Path,
    yaml_path: Path,
    stages_by_yaml: dict[str, list[StageConfig]],
) -> list[dict[str, Any]]:
    rel_yaml = yaml_path.relative_to(repo_root).as_posix()
    with yaml_path.open("r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file) or {}

    if not isinstance(data, dict):
        return []

    yaml_stem = yaml_path.stem
    targets: list[dict[str, Any]] = []
    for context, entries in data.items():
        if context == "version" or entries is None:
            continue
        if not isinstance(entries, list):
            continue

        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            terms = _extract_terms(entry)
            condition = entry.get("condition") if isinstance(entry.get("condition"), dict) else {}
            constraints = _constraints_from_entry(condition, terms)
            jenkins_stages = _matching_jenkins_stages(
                stages_by_yaml.get(yaml_stem, []), constraints
            )
            tests = entry.get("tests") or []
            if not isinstance(tests, list):
                continue
            for test_index, raw_selector in enumerate(tests):
                if not isinstance(raw_selector, str):
                    continue
                parsed_selector = parse_pytest_selector(raw_selector)
                target = _target_from_selector(
                    rel_yaml=rel_yaml,
                    yaml_stem=yaml_stem,
                    context=str(context),
                    entry_index=entry_index,
                    test_index=test_index,
                    parsed_selector=parsed_selector,
                    constraints=constraints,
                    jenkins_stages=jenkins_stages,
                )
                targets.append(target)

    return targets


def _target_from_selector(
    *,
    rel_yaml: str,
    yaml_stem: str,
    context: str,
    entry_index: int,
    test_index: int,
    parsed_selector: ParsedSelector,
    constraints: dict[str, Any],
    jenkins_stages: list[str],
) -> dict[str, Any]:
    return {
        "target_id": _target_id(
            yaml_stem=yaml_stem,
            context=context,
            entry_index=entry_index,
            test_index=test_index,
            parsed_selector=parsed_selector,
        ),
        "kind": "pytest_selector",
        "source": {
            "yaml": rel_yaml,
            "context": context,
            "entry_index": entry_index,
            "test_index": test_index,
            "jenkins_stages": jenkins_stages,
            "jenkins_stage_scope": "pre_shard_candidates",
        },
        "selector": parsed_selector.to_dict(),
        "constraints": constraints,
        "tags": _tags_for_target(parsed_selector, constraints),
        "component_hints": _component_hints(parsed_selector.paths),
    }


def _extract_terms(entry: dict[str, Any]) -> dict[str, Any]:
    terms = entry.get("terms")
    if isinstance(terms, dict) and terms:
        return terms

    condition = entry.get("condition")
    if not isinstance(condition, dict):
        return {}
    condition_terms = condition.get("terms")
    if isinstance(condition_terms, dict):
        return condition_terms
    return {}


def _constraints_from_entry(condition: dict[str, Any], terms: dict[str, Any]) -> dict[str, Any]:
    ranges = condition.get("ranges") if isinstance(condition, dict) else {}
    wildcards = condition.get("wildcards") if isinstance(condition, dict) else {}
    if not isinstance(ranges, dict):
        ranges = {}
    if not isinstance(wildcards, dict):
        wildcards = {}

    system_gpu_count = ranges.get("system_gpu_count")
    return {
        "stage": _string_or_none(terms.get("stage")),
        "backend": _string_or_none(terms.get("backend")),
        "orchestrator": _string_or_none(terms.get("orchestrator")),
        "auto_trigger": _string_or_none(terms.get("auto_trigger")),
        "gpu_wildcards": _string_list(wildcards.get("gpu")),
        "system_gpu_count": _primitive_mapping(system_gpu_count),
    }


def _matching_jenkins_stages(
    stage_configs: list[StageConfig], constraints: dict[str, Any]
) -> list[str]:
    matching_stages = [
        stage_config.name
        for stage_config in stage_configs
        if _stage_matches_constraints(stage_config, constraints)
    ]
    return sorted(matching_stages)


def _stage_matches_constraints(stage_config: StageConfig, constraints: dict[str, Any]) -> bool:
    stage_name_upper = stage_config.name.upper()

    stage_term = constraints.get("stage")
    if stage_term == "post_merge" and "POST-MERGE" not in stage_name_upper:
        return False
    if stage_term == "pre_merge" and "POST-MERGE" in stage_name_upper:
        return False

    backend = constraints.get("backend")
    if backend and not _stage_name_has_keyword(stage_name_upper, backend, _BACKEND_STAGE_KEYWORDS):
        return False

    orchestrator = constraints.get("orchestrator")
    if orchestrator == "ray" and "RAY" not in stage_name_upper:
        return False
    if orchestrator == "mpi" and "RAY" in stage_name_upper:
        return False

    auto_trigger = constraints.get("auto_trigger")
    if auto_trigger:
        normalized_auto_trigger = auto_trigger.lower()
        if normalized_auto_trigger == "others":
            if _stage_name_has_specific_auto_trigger(stage_name_upper):
                return False
        elif not _stage_name_has_keyword(
            stage_name_upper, auto_trigger, _AUTO_TRIGGER_STAGE_KEYWORDS
        ):
            return False

    system_gpu_count = constraints.get("system_gpu_count")
    if isinstance(system_gpu_count, dict) and not _gpu_count_matches(
        stage_config.gpu_count, system_gpu_count
    ):
        return False

    return True


def _stage_name_has_keyword(
    stage_name_upper: str,
    value: str,
    keyword_map: dict[str, list[str]],
) -> bool:
    normalized = value.lower()
    keywords = keyword_map.get(normalized)
    if keywords is None:
        keywords = [normalized.replace("_", "").replace("-", "").upper()]
    compact_stage_name = stage_name_upper.replace("_", "").replace("-", "")
    return any(
        keyword.replace("_", "").replace("-", "") in compact_stage_name for keyword in keywords
    )


def _stage_name_has_specific_auto_trigger(stage_name_upper: str) -> bool:
    return any(
        _stage_name_has_keyword(stage_name_upper, auto_trigger, _AUTO_TRIGGER_STAGE_KEYWORDS)
        for auto_trigger in _AUTO_TRIGGER_STAGE_KEYWORDS
        if auto_trigger != "others"
    )


def _gpu_count_matches(stage_gpu_count: int, system_gpu_count: dict[str, Any]) -> bool:
    for key, value in system_gpu_count.items():
        if not isinstance(value, int):
            continue
        if key == "eq" and stage_gpu_count != value:
            return False
        if key == "gte" and stage_gpu_count < value:
            return False
        if key == "lte" and stage_gpu_count > value:
            return False
        if key == "gt" and stage_gpu_count <= value:
            return False
        if key == "lt" and stage_gpu_count >= value:
            return False
    return True


def _tags_for_target(parsed_selector: ParsedSelector, constraints: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in ("stage", "backend", "orchestrator", "auto_trigger"):
        value = constraints.get(key)
        if value:
            tags.append(f"{key}:{value}")

    for wildcard in constraints.get("gpu_wildcards", []):
        tags.append(f"gpu:{wildcard}")

    system_gpu_count = constraints.get("system_gpu_count")
    if isinstance(system_gpu_count, dict) and system_gpu_count:
        tags.append(f"system_gpu_count:{_range_tag_value(system_gpu_count)}")
        lower_bound = system_gpu_count.get("gte") or system_gpu_count.get("gt")
        upper_bound = system_gpu_count.get("lte") or system_gpu_count.get("lt")
        if (lower_bound and lower_bound > 1) or (upper_bound and upper_bound > 1):
            tags.append("multi_gpu")

    if parsed_selector.timeout_minutes is not None:
        tags.append(f"timeout:{parsed_selector.timeout_minutes}m")
    if parsed_selector.isolation:
        tags.append("isolation")

    return tags


def _range_tag_value(value: dict[str, Any]) -> str:
    return "_".join(f"{key}{value[key]}" for key in sorted(value))


def _component_hints(selector_paths: list[str]) -> list[str]:
    hints: list[str] = []
    for selector_path in selector_paths:
        hints.extend(_component_hints_for_path(selector_path))
    return _dedupe_preserving_order(hints)


def _component_hints_for_path(selector_path: str) -> list[str]:
    if not selector_path:
        return []

    path_part = selector_path.split("::", 1)[0]
    if path_part.startswith("unittest/"):
        repo_path = f"tests/{path_part}"
    elif path_part.startswith("tests/"):
        repo_path = path_part
    else:
        repo_path = f"tests/integration/defs/{path_part}"

    parts = [part for part in repo_path.split("/") if part]
    if repo_path.endswith(".py"):
        component_dir = "/".join(parts[:-1])
    else:
        component_dir = "/".join(parts)

    hints: list[str] = []
    if component_dir.startswith("tests/unittest/"):
        component_parts = component_dir.split("/")
        if len(component_parts) >= 3:
            hints.append("/".join(component_parts[:3]))
        if len(component_parts) >= 4:
            hints.append("/".join(component_parts[:4]))
    elif component_dir.startswith("tests/integration/defs/"):
        component_parts = component_dir.split("/")
        if len(component_parts) >= 4:
            hints.append("/".join(component_parts[:4]))
        if len(component_parts) >= 5:
            hints.append("/".join(component_parts[:5]))

    if component_dir and component_dir not in hints:
        hints.append(component_dir)

    return hints


def _target_id(
    *,
    yaml_stem: str,
    context: str,
    entry_index: int,
    test_index: int,
    parsed_selector: ParsedSelector,
) -> str:
    selector_identity = parsed_selector.path or parsed_selector.raw or "selector"
    selector_slug = _slug(selector_identity, max_length=72)
    return (
        f"//ci_target_graph/{_slug(yaml_stem)}:"
        f"{_slug(context)}__e{entry_index:03d}__t{test_index:04d}__{selector_slug}"
    )


def _slug(value: str, max_length: int = 96) -> str:
    slug = _SANITIZE_RE.sub("_", value).strip("_").lower()
    if not slug:
        slug = "item"
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{slug[: max_length - 11]}_{digest}"


def _iter_test_db_yaml_paths(repo_root: Path) -> list[Path]:
    test_db_dir = repo_root / TEST_DB_RELATIVE_DIR
    paths = list(test_db_dir.glob("*.yml")) + list(test_db_dir.glob("*.yaml"))
    return sorted(paths, key=lambda path: path.as_posix())


def _strip_groovy_comments(text: str) -> str:
    return "\n".join(_strip_groovy_line_comment(line) for line in text.splitlines())


def _strip_groovy_line_comment(line: str) -> str:
    output = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote is not None:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if line.startswith("//", index):
            break
        if char in {"'", '"'}:
            quote = char
        output.append(char)
        index += 1

    return "".join(output)


def _parse_groovy_list(values_text: str) -> list[Any] | None:
    python_text = re.sub(r"\btrue\b", "True", values_text)
    python_text = re.sub(r"\bfalse\b", "False", python_text)
    python_text = re.sub(r"\bnull\b", "None", python_text)
    try:
        value = ast.literal_eval(f"[{python_text}]")
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, list):
        return value
    return None


def _stage_config_from_values(stage_name: str, values: list[Any]) -> StageConfig | None:
    if len(values) < 4:
        return None
    if not isinstance(values[0], str) or not isinstance(values[1], str):
        return None
    if not _is_int(values[2]) or not _is_int(values[3]):
        return None

    return StageConfig(
        name=stage_name,
        platform=values[0],
        yaml_stem=values[1],
        split_id=int(values[2]),
        split_count=int(values[3]),
        gpu_count=int(values[4]) if len(values) > 4 and _is_int(values[4]) else 1,
        node_count=int(values[5]) if len(values) > 5 and _is_int(values[5]) else 1,
        run_with_sbatch=(
            bool(values[6]) if len(values) > 6 and isinstance(values[6], bool) else False
        ),
    )


def _parse_build_stage_config_calls(text: str) -> list[StageConfig]:
    stage_configs: list[StageConfig] = []
    search_from = 0
    function_name = "buildStageConfigs"
    while True:
        function_index = text.find(function_name, search_from)
        if function_index < 0:
            break
        open_index = text.find("(", function_index + len(function_name))
        if open_index < 0:
            break
        content, close_index = _extract_parenthesized_content(text, open_index)
        if content is None:
            search_from = function_index + len(function_name)
            continue
        stage_configs.extend(_stage_configs_from_build_stage_args(content))
        search_from = close_index + 1
    return stage_configs


def _extract_parenthesized_content(text: str, open_index: int) -> tuple[str | None, int]:
    depth = 0
    quote: str | None = None
    escaped = False
    content_start = open_index + 1
    index = open_index
    while index < len(text):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index
        index += 1

    return None, len(text)


def _stage_configs_from_build_stage_args(content: str) -> list[StageConfig]:
    args = _split_top_level_commas(content)
    if len(args) < 5:
        return []

    parsed_args = [_parse_groovy_scalar(arg) for arg in args]
    stage_base, platform, yaml_stem, test_count, gpu_count = parsed_args[:5]
    if not (
        isinstance(stage_base, str)
        and isinstance(platform, str)
        and isinstance(yaml_stem, str)
        and _is_int(test_count)
        and _is_int(gpu_count)
    ):
        return []

    node_count = parsed_args[5] if len(parsed_args) > 5 and _is_int(parsed_args[5]) else 1
    run_with_sbatch = (
        parsed_args[6] if len(parsed_args) > 6 and isinstance(parsed_args[6], bool) else False
    )

    return [
        StageConfig(
            name=f"{stage_base}-{split_id}",
            platform=platform,
            yaml_stem=yaml_stem,
            split_id=split_id,
            split_count=int(test_count),
            gpu_count=int(gpu_count),
            node_count=int(node_count),
            run_with_sbatch=bool(run_with_sbatch),
        )
        for split_id in range(1, int(test_count) + 1)
    ]


def _split_top_level_commas(content: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(content):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(content[start:index].strip())
            start = index + 1
    final_arg = content[start:].strip()
    if final_arg:
        args.append(final_arg)
    return args


def _parse_groovy_scalar(text: str) -> Any:
    normalized = text.strip()
    normalized = re.sub(r"\btrue\b", "True", normalized)
    normalized = re.sub(r"\bfalse\b", "False", normalized)
    normalized = re.sub(r"\bnull\b", "None", normalized)
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return None


def _primitive_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in sorted(value):
        candidate = value[key]
        if isinstance(candidate, (str, int, float, bool)) or candidate is None:
            result[str(key)] = candidate
    return result


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


if __name__ == "__main__":
    raise SystemExit(main())
