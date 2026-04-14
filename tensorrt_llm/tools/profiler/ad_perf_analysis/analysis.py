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

"""Core analysis helpers for AutoDeploy performance iterations."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .nsys_sqlite import compute_gpu_gap_intervals

FAMILY_PATTERNS = {
    "qkv_gemm": ("qkv", "q_proj", "k_proj", "v_proj", "gemm", "cutlass"),
    "attn": ("attn", "attention", "flash", "fmha", "mla"),
    "o_proj": ("o_proj", "out_proj"),
    "moe_router": ("router", "gate"),
    "moe_dispatch": ("dispatch", "permute"),
    "moe_fc1": ("moe", "fc1", "w1", "up_proj"),
    "moe_fc2": ("fc2", "w2", "down_proj"),
    "moe_combine": ("combine", "unpermute"),
    "norm": ("norm", "rms"),
    "allreduce": ("allreduce", "ncclallreduce", "ar_fusion"),
    "allgather": ("allgather",),
    "reduce_scatter": ("reduce_scatter",),
    "memcpy": ("memcpy",),
}

CONFIG_KEY_HEURISTICS = {
    "chunked_prefill": {
        "recommended_when_missing": True,
        "message": "chunked_prefill is missing; compare against model-registry defaults.",
    },
    "max_num_tokens": {
        "message": "max_num_tokens should be validated against comparable AD configs.",
    },
    "max_seq_len": {
        "message": "max_seq_len should be checked against the benchmarked workload and comparable configs.",
    },
    "enable_multistream_moe": {
        "recommended_when_missing": True,
        "message": "enable_multistream_moe is missing; verify whether the model family uses it.",
    },
}


def classify_kernel_family(name: str) -> str:
    """Map a kernel or graph-node name to a coarse family."""
    lowered = name.lower()
    for family, patterns in FAMILY_PATTERNS.items():
        if all(pattern in lowered for pattern in patterns[:1]) and any(
            pattern in lowered for pattern in patterns
        ):
            return family
    return "misc"


def _normalize_name(name: str) -> set[str]:
    return {token for token in "".join(ch if ch.isalnum() else " " for ch in name.lower()).split() if token}


def join_trace_and_graph(trace_data: dict[str, Any], graph_data: dict[str, Any] | None) -> dict[str, Any]:
    """Join kernel executions and layer windows with AD graph semantics."""
    graph_nodes = graph_data["node_index"] if graph_data is not None else {}
    joined_timeline = []
    confidence_counts = Counter()

    for layer_window in trace_data["layer_windows"]:
        layer_entry = {
            **layer_window,
            "kernels": [],
        }
        for kernel in layer_window["kernels"]:
            family = classify_kernel_family(kernel["display_name"])
            kernel_tokens = _normalize_name(kernel["display_name"])
            best_match = None
            best_overlap = 0
            for node in graph_nodes.values():
                node_tokens = _normalize_name(node.get("target", "")) | _normalize_name(node.get("name", ""))
                overlap = len(kernel_tokens & node_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = node

            if best_match is not None and best_overlap >= 2:
                confidence = "high"
            elif best_match is not None and best_overlap == 1:
                confidence = "medium"
            elif graph_data is None:
                confidence = "low"
            else:
                confidence = "low"
            confidence_counts[confidence] += 1
            layer_entry["kernels"].append(
                {
                    **kernel,
                    "family": family,
                    "graph_match": {
                        "target": best_match.get("target"),
                        "name": best_match.get("name"),
                        "stage": best_match.get("stage"),
                        "transform": best_match.get("transform"),
                    }
                    if best_match is not None
                    else None,
                    "confidence": confidence,
                }
            )
        joined_timeline.append(layer_entry)

    return {
        "joined_timeline": joined_timeline,
        "confidence_counts": dict(confidence_counts),
    }


def select_representative_layer_window(joined_timeline: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select a representative layer window for detailed analysis."""
    if not joined_timeline:
        return None

    scored_windows = []
    for index, layer_window in enumerate(joined_timeline):
        families = {kernel["family"] for kernel in layer_window["kernels"]}
        score = len(families)
        if "attn" in families:
            score += 2
        if "qkv_gemm" in families:
            score += 2
        if any(family.startswith("moe_") for family in families):
            score += 1
        distance_from_middle = abs(index - (len(joined_timeline) // 2))
        scored_windows.append((score, -distance_from_middle, layer_window))

    scored_windows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored_windows[0][2]


def _build_gap_records(kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    for previous_kernel, next_kernel in zip(kernels, kernels[1:]):
        gap_duration = next_kernel["start"] - previous_kernel["end"]
        if gap_duration <= 0:
            continue
        gaps.append(
            {
                "start": previous_kernel["end"],
                "end": next_kernel["start"],
                "duration_ns": gap_duration,
                "previous_kernel": previous_kernel["display_name"],
                "next_kernel": next_kernel["display_name"],
            }
        )
    return gaps


def analyze_layer_window(
    selected_window: dict[str, Any] | None,
    runtime_correlation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a layer-level summary and evidence-backed findings."""
    if selected_window is None:
        return {"summary": {"selected": False}, "findings": []}

    kernels = sorted(selected_window["kernels"], key=lambda kernel: kernel["start"])
    gaps = _build_gap_records(kernels)
    family_ns = defaultdict(int)
    for kernel in kernels:
        family_ns[kernel["family"]] += kernel["duration_ns"]

    findings = []
    if gaps:
        largest_gap = max(gaps, key=lambda gap: gap["duration_ns"])
        findings.append(
            {
                "id": "layer-gap",
                "title": "Largest intra-layer GPU gap",
                "category": "gpu_gap",
                "severity": "medium",
                "confidence": "medium",
                "evidence": largest_gap,
                "layer_scope": selected_window["name"],
                "recommended_action": "Inspect host runtime calls and syncs immediately preceding the gap.",
            }
        )

    if family_ns:
        dominant_family = max(family_ns.items(), key=lambda item: item[1])
        findings.append(
            {
                "id": "dominant-family",
                "title": f"Dominant kernel family is {dominant_family[0]}",
                "category": "kernel_family",
                "severity": "info",
                "confidence": "medium",
                "evidence": {
                    "family": dominant_family[0],
                    "duration_ns": dominant_family[1],
                },
                "layer_scope": selected_window["name"],
                "recommended_action": "Use this family as the first target for optimization work.",
            }
        )

    host_runtime_calls = []
    if runtime_correlation is not None:
        for record in runtime_correlation["records"]:
            runtime_call = record["runtime_call"]
            if runtime_call["start"] <= selected_window["end"] and runtime_call["end"] >= selected_window["start"]:
                host_runtime_calls.append(runtime_call)

    return {
        "summary": {
            "selected": True,
            "layer_name": selected_window["name"],
            "layer_id": selected_window["layer_id"],
            "duration_ns": selected_window["duration_ns"],
            "kernel_count": len(kernels),
            "family_ns": dict(family_ns),
            "gap_count": len(gaps),
        },
        "gaps": gaps,
        "host_runtime_calls": host_runtime_calls,
        "findings": findings,
    }


def analyze_memcpy_timeline(runtime_correlation: dict[str, Any]) -> dict[str, Any]:
    """Summarize memcpy traffic and highlight suspicious movement patterns."""
    bytes_by_kind = defaultdict(int)
    duration_by_kind = defaultdict(int)
    findings = []
    records = []

    for record in runtime_correlation["records"]:
        runtime_call = record["runtime_call"]
        for memcpy in record["memcpys"]:
            bytes_by_kind[memcpy["copy_kind"]] += memcpy["bytes"]
            duration_by_kind[memcpy["copy_kind"]] += memcpy["duration_ns"]
            records.append(
                {
                    "runtime_name": runtime_call["name"],
                    "copy_kind": memcpy["copy_kind"],
                    "src_kind": memcpy["src_kind"],
                    "dst_kind": memcpy["dst_kind"],
                    "bytes": memcpy["bytes"],
                    "duration_ns": memcpy["duration_ns"],
                }
            )
            if "Pageable" in memcpy["src_kind"] or "Pageable" in memcpy["dst_kind"]:
                findings.append(
                    {
                        "id": f"pageable-{len(findings)}",
                        "title": "Memcpy touches pageable memory",
                        "category": "memcpy",
                        "severity": "medium",
                        "confidence": "high",
                        "evidence": {
                            "runtime_name": runtime_call["name"],
                            "copy_kind": memcpy["copy_kind"],
                            "src_kind": memcpy["src_kind"],
                            "dst_kind": memcpy["dst_kind"],
                        },
                        "recommended_action": "Consider pinned memory for transfers on the hot path.",
                    }
                )
            if "Device-to-Host" in memcpy["copy_kind"]:
                findings.append(
                    {
                        "id": f"d2h-{len(findings)}",
                        "title": "Device-to-host transfer on the analyzed path",
                        "category": "memcpy",
                        "severity": "medium",
                        "confidence": "high",
                        "evidence": {
                            "runtime_name": runtime_call["name"],
                            "copy_kind": memcpy["copy_kind"],
                            "bytes": memcpy["bytes"],
                        },
                        "recommended_action": "Verify whether D2H traffic is expected in steady-state decode.",
                    }
                )

    return {
        "summary": {
            "copy_count": len(records),
            "bytes_by_kind": dict(bytes_by_kind),
            "duration_ns_by_kind": dict(duration_by_kind),
        },
        "records": records,
        "findings": findings,
    }


def analyze_host_waits(trace_data: dict[str, Any], runtime_correlation: dict[str, Any]) -> dict[str, Any]:
    """Classify host waits and detect waits that expose GPU idle time."""
    gpu_gaps = compute_gpu_gap_intervals(trace_data["kernels"])
    wait_records = []
    findings = []

    for record in runtime_correlation["records"]:
        runtime_call = record["runtime_call"]
        if not record["syncs"] and "Synchronize" not in runtime_call["name"]:
            continue

        exposed_gap = None
        for gap in gpu_gaps:
            overlap_start = max(runtime_call["start"], gap["start"])
            overlap_end = min(runtime_call["end"], gap["end"])
            if overlap_end > overlap_start:
                exposed_gap = {
                    **gap,
                    "overlap_ns": overlap_end - overlap_start,
                }
                break
        wait_record = {
            "runtime_name": runtime_call["name"],
            "runtime_duration_ns": runtime_call["duration_ns"],
            "syncs": record["syncs"],
            "callchain": record.get("callchain"),
            "exposed_gpu_gap": exposed_gap,
        }
        wait_records.append(wait_record)
        if exposed_gap is not None:
            findings.append(
                {
                    "id": f"host-wait-{len(findings)}",
                    "title": "Host wait overlaps a GPU idle gap",
                    "category": "host_wait",
                    "severity": "high",
                    "confidence": "high",
                    "evidence": {
                        "runtime_name": runtime_call["name"],
                        "runtime_duration_ns": runtime_call["duration_ns"],
                        "exposed_gpu_gap": exposed_gap,
                    },
                    "recommended_action": "Inspect whether this synchronization is avoidable or can be delayed.",
                }
            )

    return {
        "summary": {
            "wait_count": len(wait_records),
            "gpu_gap_count": len(gpu_gaps),
            "gpu_gap_ns_total": sum(gap["duration_ns"] for gap in gpu_gaps),
        },
        "waits": wait_records,
        "findings": findings,
        "gpu_gaps": gpu_gaps,
    }


def _flatten_config(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_config(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}, found {type(loaded).__name__}.")
    return loaded


def analyze_serving_config(
    config_path: str | Path,
    comparable_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Compare one serving config against heuristic checks and comparable configs."""
    active_config = _load_yaml(config_path)
    active_flat = _flatten_config(active_config)
    comparable_configs = []
    comparable_flattened = []
    for root in comparable_roots or []:
        for path in sorted(Path(root).glob("*.yaml")):
            if Path(path) == Path(config_path):
                continue
            try:
                comparable_payload = _load_yaml(path)
            except ValueError:
                continue
            comparable_configs.append({"path": str(path), "config": comparable_payload})
            comparable_flattened.append((str(path), _flatten_config(comparable_payload)))

    findings = []
    suggestions = []
    for key, rule in CONFIG_KEY_HEURISTICS.items():
        matching_keys = [candidate for candidate in active_flat if candidate.endswith(key)]
        if not matching_keys:
            findings.append(
                {
                    "id": f"config-missing-{key}",
                    "title": f"Config key `{key}` is missing",
                    "category": "config",
                    "severity": "medium",
                    "confidence": "medium",
                    "evidence": {
                        "message": rule["message"],
                    },
                    "recommended_action": f"Inspect `{key}` against comparable AutoDeploy configs.",
                }
            )
            if rule.get("recommended_when_missing"):
                suggestions.append({"key": key, "action": "evaluate_enablement"})

    for compared_path, compared_flat in comparable_flattened:
        for key in ("max_seq_len", "max_num_tokens"):
            active_key = next((candidate for candidate in active_flat if candidate.endswith(key)), None)
            compared_key = next((candidate for candidate in compared_flat if candidate.endswith(key)), None)
            if active_key is None or compared_key is None:
                continue
            active_value = active_flat[active_key]
            compared_value = compared_flat[compared_key]
            if isinstance(active_value, int) and isinstance(compared_value, int):
                if compared_value > 0 and active_value > compared_value * 4:
                    findings.append(
                        {
                            "id": f"config-outlier-{key}-{len(findings)}",
                            "title": f"`{key}` is much larger than a comparable config",
                            "category": "config",
                            "severity": "low",
                            "confidence": "low",
                            "evidence": {
                                "active_value": active_value,
                                "compared_value": compared_value,
                                "compared_path": compared_path,
                            },
                            "recommended_action": "Validate whether the larger value is intentional for this workload.",
                        }
                    )

    return {
        "active_config_path": str(config_path),
        "active_config": active_config,
        "comparable_configs": comparable_configs,
        "findings": findings,
        "suggestions": suggestions,
    }


def build_iteration_record(
    *,
    model: str,
    objective: str,
    trace_data: dict[str, Any],
    graph_data: dict[str, Any] | None,
    runtime_correlation: dict[str, Any],
    layer_analysis: dict[str, Any],
    memcpy_analysis: dict[str, Any],
    host_wait_analysis: dict[str, Any],
    config_analysis: dict[str, Any],
    change: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one machine-readable iteration record."""
    findings = []
    for analysis_payload in (
        layer_analysis,
        memcpy_analysis,
        host_wait_analysis,
        config_analysis,
    ):
        findings.extend(analysis_payload.get("findings", []))

    return {
        "model": model,
        "objective": objective,
        "artifacts": {
            "sqlite_path": trace_data.get("sqlite_path"),
            "graph_dump_dir": graph_data.get("graph_dump_dir") if graph_data is not None else None,
        },
        "analysis_scope": {
            "selected_layer": layer_analysis.get("summary", {}).get("layer_name"),
        },
        "findings": findings,
        "change": change,
        "results": {
            "layer_summary": layer_analysis.get("summary"),
            "memcpy_summary": memcpy_analysis.get("summary"),
            "host_wait_summary": host_wait_analysis.get("summary"),
            "runtime_summary": runtime_correlation.get("summary"),
        },
        "supporting_data": {
            "layer_analysis": layer_analysis,
            "memcpy_analysis": memcpy_analysis,
            "host_wait_analysis": host_wait_analysis,
            "config_analysis": config_analysis,
        },
    }


def compare_iterations(previous_iteration: dict[str, Any], current_iteration: dict[str, Any]) -> dict[str, Any]:
    """Compare two iteration records and classify the direction of change."""
    deltas = {}
    verdict = "inconclusive"
    for key in ("duration_ns",):
        previous_value = previous_iteration["results"]["layer_summary"].get(key)
        current_value = current_iteration["results"]["layer_summary"].get(key)
        if previous_value is None or current_value is None:
            continue
        deltas[f"layer_summary.{key}"] = current_value - previous_value
    if deltas:
        duration_delta = deltas.get("layer_summary.duration_ns")
        if duration_delta is not None:
            if duration_delta < 0:
                verdict = "improved"
            elif duration_delta > 0:
                verdict = "regressed"

    return {
        "verdict": verdict,
        "deltas": deltas,
    }


def render_iteration_report(iteration_record: dict[str, Any]) -> str:
    """Render a human-readable Markdown report for one iteration."""
    findings = iteration_record.get("findings", [])
    layer_summary = iteration_record["results"].get("layer_summary", {})
    memcpy_summary = iteration_record["results"].get("memcpy_summary", {})
    host_wait_summary = iteration_record["results"].get("host_wait_summary", {})
    runtime_summary = iteration_record["results"].get("runtime_summary", {})

    lines = [
        f"# AutoDeploy Performance Iteration: {iteration_record['model']}",
        "",
        "## Context",
        f"- Objective: {iteration_record['objective']}",
        f"- SQLite trace: `{iteration_record['artifacts']['sqlite_path']}`",
        f"- Graph dump: `{iteration_record['artifacts']['graph_dump_dir']}`",
        "",
        "## Scope Chosen",
        f"- Selected layer: `{layer_summary.get('layer_name')}`",
        f"- Layer duration (ns): `{layer_summary.get('duration_ns')}`",
        f"- Kernel count: `{layer_summary.get('kernel_count')}`",
        "",
        "## Layer and Op Findings",
    ]
    layer_findings = [
        finding for finding in findings if finding.get("category") in {"gpu_gap", "kernel_family"}
    ]
    if layer_findings:
        for finding in layer_findings:
            lines.append(f"- {finding['title']}: `{json.dumps(finding['evidence'], sort_keys=True)}`")
    else:
        lines.append("- No layer-level findings were generated.")

    lines.extend(
        [
            "",
            "## Host CUDA API To GPU Correlation",
            f"- Runtime calls analyzed: `{runtime_summary.get('runtime_count')}`",
            f"- Runtime calls with kernels: `{runtime_summary.get('runtime_with_kernels')}`",
            f"- Runtime calls with memcpys: `{runtime_summary.get('runtime_with_memcpys')}`",
            f"- Runtime calls with syncs: `{runtime_summary.get('runtime_with_syncs')}`",
            "",
            "## Data Movement Summary",
            f"- Copy count: `{memcpy_summary.get('copy_count')}`",
            f"- Bytes by kind: `{json.dumps(memcpy_summary.get('bytes_by_kind', {}), sort_keys=True)}`",
            "",
            "## Host Wait Analysis",
            f"- Wait count: `{host_wait_summary.get('wait_count')}`",
            f"- GPU gap count: `{host_wait_summary.get('gpu_gap_count')}`",
            f"- GPU gap total (ns): `{host_wait_summary.get('gpu_gap_ns_total')}`",
            "",
            "## Recommended Next Experiments",
        ]
    )
    if findings:
        for finding in findings[:5]:
            lines.append(f"- {finding['recommended_action']}")
    else:
        lines.append("- Collect a graph dump or deep host trace for richer findings.")
    return "\n".join(lines) + "\n"
