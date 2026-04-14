# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parsers for AutoDeploy graph dumps captured with `AD_DUMP_GRAPHS_DIR`."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

GRAPH_MODULE_RE = re.compile(r"^# GraphModule: (?P<name>.+)$")
TRANSFORM_RE = re.compile(r"^# Transform: (?P<value>.+)$")
STAGE_RE = re.compile(r"^# Stage: (?P<value>.+)$")
CALL_RE = re.compile(r"^%(?P<name>\S+) = (?P<target>.+?)\((?P<inputs>.*)\) : (?P<output>.+)$")
PLACEHOLDER_RE = re.compile(r"^%(?P<name>\S+) : (?P<output>.+)$")
OUTPUT_RE = re.compile(r"^output (?P<value>.+)$")


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def parse_graph_dump_file(path: str | Path) -> dict[str, Any]:
    """Parse one AutoDeploy graph dump text file."""
    graph_modules = []
    current_module: dict[str, Any] | None = None
    transform_name = None
    stage_name = None

    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        transform_match = TRANSFORM_RE.match(line)
        if transform_match:
            transform_name = transform_match.group("value")
            continue
        stage_match = STAGE_RE.match(line)
        if stage_match:
            stage_name = stage_match.group("value")
            continue
        module_match = GRAPH_MODULE_RE.match(line)
        if module_match:
            current_module = {"name": module_match.group("name"), "nodes": []}
            graph_modules.append(current_module)
            continue
        if current_module is None:
            continue

        call_match = CALL_RE.match(line)
        if call_match:
            target = call_match.group("target")
            current_module["nodes"].append(
                {
                    "kind": "call",
                    "name": call_match.group("name"),
                    "target": target,
                    "inputs": [value.strip() for value in call_match.group("inputs").split(",") if value.strip()],
                    "output": call_match.group("output"),
                    "normalized_name": _normalize_name(call_match.group("name")),
                    "normalized_target": _normalize_name(target),
                    "raw_line": raw_line,
                }
            )
            continue

        placeholder_match = PLACEHOLDER_RE.match(line)
        if placeholder_match:
            current_module["nodes"].append(
                {
                    "kind": "placeholder",
                    "name": placeholder_match.group("name"),
                    "output": placeholder_match.group("output"),
                    "normalized_name": _normalize_name(placeholder_match.group("name")),
                    "raw_line": raw_line,
                }
            )
            continue

        output_match = OUTPUT_RE.match(line)
        if output_match:
            current_module["nodes"].append(
                {
                    "kind": "output",
                    "name": "output",
                    "value": output_match.group("value"),
                    "normalized_name": "output",
                    "raw_line": raw_line,
                }
            )

    return {
        "path": str(path),
        "transform": transform_name,
        "stage": stage_name,
        "graph_modules": graph_modules,
    }


def parse_ad_graph_dump_dir(graph_dump_dir: str | Path) -> dict[str, Any]:
    """Parse every graph dump file in an AutoDeploy graph dump directory."""
    graph_dir = Path(graph_dump_dir)
    dump_files = sorted(graph_dir.glob("*.txt"))
    parsed_files = [parse_graph_dump_file(path) for path in dump_files]

    node_index = {}
    stage_index: dict[str, list[str]] = {}
    transform_index: list[dict[str, Any]] = []
    for parsed_file in parsed_files:
        transform_index.append(
            {
                "path": parsed_file["path"],
                "transform": parsed_file["transform"],
                "stage": parsed_file["stage"],
                "graph_module_count": len(parsed_file["graph_modules"]),
            }
        )
        stage_index.setdefault(parsed_file["stage"] or "<unknown>", []).append(parsed_file["path"])
        for module in parsed_file["graph_modules"]:
            for node in module["nodes"]:
                node_key = f"{Path(parsed_file['path']).name}:{module['name']}:{node['name']}"
                node_index[node_key] = {
                    **node,
                    "graph_module": module["name"],
                    "transform": parsed_file["transform"],
                    "stage": parsed_file["stage"],
                    "path": parsed_file["path"],
                }

    return {
        "graph_dump_dir": str(graph_dir),
        "files": parsed_files,
        "graph_summary": {
            "file_count": len(parsed_files),
            "graph_module_count": sum(len(parsed_file["graph_modules"]) for parsed_file in parsed_files),
            "node_count": len(node_index),
        },
        "node_index": node_index,
        "stage_index": stage_index,
        "transform_index": transform_index,
    }
