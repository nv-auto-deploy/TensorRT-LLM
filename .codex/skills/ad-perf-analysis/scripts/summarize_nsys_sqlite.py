# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce a compact summary from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from _common import write_json

TRANSFORM_RE = re.compile(
    r"\[stage=(?P<stage>[^,]+), transform=(?P<transform>[^\]]+)\].*\[SUMMARY\].*time: (?P<time>[0-9.]+)s"
)
TOTAL_TRANSFORM_RE = re.compile(r"Total time for all transforms: (?P<time>[0-9.]+)s")


def _load_label_table(connection: sqlite3.Connection, table_name: str) -> dict[int, str]:
    cursor = connection.cursor()
    try:
        return dict(cursor.execute(f"SELECT id, label FROM {table_name}").fetchall())
    except sqlite3.OperationalError:
        return {}


def _build_capture_summary(
    capture_log: Path | None, graph_dump_dir: Path | None
) -> dict[str, object]:
    summary: dict[str, object] = {
        "cupti_multiple_subscribers": 0,
        "torch_trace_rename_failures": 0,
        "hang_detected": 0,
        "total_transform_time_s": None,
        "top_rank0_transforms": [],
        "graph_dump_file_count": len(list(graph_dump_dir.glob("*.txt"))) if graph_dump_dir else 0,
    }
    if capture_log is None or not capture_log.exists():
        return summary

    transform_stats: dict[tuple[str, str], list[float]] = defaultdict(list)
    for line in capture_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED" in line:
            summary["cupti_multiple_subscribers"] = int(summary["cupti_multiple_subscribers"]) + 1
        if "Failed to rename" in line and "trace.json.tmp" in line:
            summary["torch_trace_rename_failures"] = int(summary["torch_trace_rename_failures"]) + 1
        if "Hang detected" in line:
            summary["hang_detected"] = int(summary["hang_detected"]) + 1

        total_match = TOTAL_TRANSFORM_RE.search(line)
        if total_match:
            summary["total_transform_time_s"] = float(total_match.group("time"))

        transform_match = TRANSFORM_RE.search(line)
        if transform_match and "[RANK 0]" in line:
            key = (transform_match.group("stage"), transform_match.group("transform"))
            transform_stats[key].append(float(transform_match.group("time")))

    top_rank0_transforms = []
    for (stage, transform), values in transform_stats.items():
        total_time_s = sum(values)
        top_rank0_transforms.append(
            {
                "stage": stage,
                "transform": transform,
                "count": len(values),
                "total_time_s": total_time_s,
                "max_time_s": max(values),
                "avg_time_s": total_time_s / len(values),
            }
        )
    top_rank0_transforms.sort(key=lambda item: item["total_time_s"], reverse=True)
    summary["top_rank0_transforms"] = top_rank0_transforms[:25]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-log")
    parser.add_argument("--graph-dump-dir")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    capture_log = Path(args.capture_log) if args.capture_log else None
    graph_dump_dir = Path(args.graph_dump_dir) if args.graph_dump_dir else None

    connection = sqlite3.connect(sqlite_path)
    cursor = connection.cursor()
    try:
        string_ids = dict(cursor.execute("SELECT id, value FROM StringIds").fetchall())
        memcpy_kinds = _load_label_table(connection, "ENUM_CUDA_MEMCPY_OPER")
        mem_kinds = _load_label_table(connection, "ENUM_CUDA_MEM_KIND")

        trace_counts = {}
        for key, table_name in {
            "nvtx_range_count": "NVTX_EVENTS",
            "kernel_count": "CUPTI_ACTIVITY_KIND_KERNEL",
            "runtime_call_count": "CUPTI_ACTIVITY_KIND_RUNTIME",
            "memcpy_count": "CUPTI_ACTIVITY_KIND_MEMCPY",
            "sync_count": "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
        }.items():
            trace_counts[key] = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        kernel_rows = cursor.execute(
            "SELECT start, end, streamId, shortName, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        ).fetchall()
        runtime_rows = cursor.execute(
            "SELECT start, end, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
        ).fetchall()
        memcpy_rows = cursor.execute(
            "SELECT bytes, copyKind, srcKind, dstKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY"
        ).fetchall()

        kernel_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"duration_ns": 0, "count": 0}
        )
        stream_counts: dict[int | None, int] = defaultdict(int)
        intervals: list[tuple[int, int]] = []
        for start, end, stream_id, short_name_id, demangled_name_id in kernel_rows:
            display_name = (
                string_ids.get(demangled_name_id) if demangled_name_id is not None else None
            )
            if not display_name:
                display_name = string_ids.get(short_name_id, "<unknown>")
            duration_ns = end - start
            kernel_totals[display_name]["duration_ns"] += duration_ns
            kernel_totals[display_name]["count"] += 1
            stream_counts[stream_id] += 1
            intervals.append((start, end))

        intervals.sort()
        merged_intervals: list[list[int]] = []
        for start, end in intervals:
            if not merged_intervals or start > merged_intervals[-1][1]:
                merged_intervals.append([start, end])
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

        active_gpu_time_ns = sum(end - start for start, end in merged_intervals)
        gpu_gap_count = 0
        gpu_gap_total_ns = 0
        largest_gpu_gaps = []
        for (_, previous_end), (next_start, _) in zip(merged_intervals, merged_intervals[1:]):
            if next_start <= previous_end:
                continue
            duration_ns = next_start - previous_end
            gpu_gap_count += 1
            gpu_gap_total_ns += duration_ns
            largest_gpu_gaps.append(
                {
                    "start": previous_end,
                    "end": next_start,
                    "duration_ns": duration_ns,
                }
            )
        largest_gpu_gaps.sort(key=lambda item: item["duration_ns"], reverse=True)

        runtime_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"duration_ns": 0, "count": 0}
        )
        runtime_name_by_correlation: dict[int, str] = {}
        for start, end, correlation_id, name_id in runtime_rows:
            runtime_name = string_ids.get(name_id, "<unknown>")
            duration_ns = end - start
            runtime_totals[runtime_name]["duration_ns"] += duration_ns
            runtime_totals[runtime_name]["count"] += 1
            runtime_name_by_correlation[correlation_id] = runtime_name

        sync_runtime_totals = {
            name: stats
            for name, stats in runtime_totals.items()
            if "Synchronize" in name or "EventQuery" in name or "StreamWait" in name
        }

        memcpy_summary: dict[str, dict[str, int]] = defaultdict(lambda: {"bytes": 0, "count": 0})
        memcpy_samples = []
        for num_bytes, copy_kind_id, src_kind_id, dst_kind_id, correlation_id in memcpy_rows:
            copy_kind = memcpy_kinds.get(copy_kind_id, f"Unknown({copy_kind_id})")
            src_kind = mem_kinds.get(src_kind_id, f"Unknown({src_kind_id})")
            dst_kind = mem_kinds.get(dst_kind_id, f"Unknown({dst_kind_id})")
            key = f"{copy_kind} | {src_kind} -> {dst_kind}"
            memcpy_summary[key]["bytes"] += num_bytes
            memcpy_summary[key]["count"] += 1
            if len(memcpy_samples) < 20:
                memcpy_samples.append(
                    {
                        "runtime_name": runtime_name_by_correlation.get(
                            correlation_id, "<unknown>"
                        ),
                        "copy_kind": copy_kind,
                        "src_kind": src_kind,
                        "dst_kind": dst_kind,
                        "bytes": num_bytes,
                    }
                )
    finally:
        connection.close()

    payload = {
        "artifact_root": str(sqlite_path.parent.parent),
        "trace": {
            **trace_counts,
            "active_gpu_time_ns": active_gpu_time_ns,
            "gpu_gap_count": gpu_gap_count,
            "gpu_gap_total_ns": gpu_gap_total_ns,
            "largest_gpu_gaps": largest_gpu_gaps[:20],
            "stream_count": len(stream_counts),
            "busiest_streams": [
                {"stream_id": stream_id, "kernel_count": count}
                for stream_id, count in sorted(
                    stream_counts.items(), key=lambda item: item[1], reverse=True
                )[:10]
            ],
            "top_kernels_by_total_duration_ns": [
                {"name": name, **stats}
                for name, stats in sorted(
                    kernel_totals.items(), key=lambda item: item[1]["duration_ns"], reverse=True
                )[:25]
            ],
            "top_runtime_calls_by_total_duration_ns": [
                {"name": name, **stats}
                for name, stats in sorted(
                    runtime_totals.items(), key=lambda item: item[1]["duration_ns"], reverse=True
                )[:25]
            ],
            "top_sync_runtime_calls_by_total_duration_ns": [
                {"name": name, **stats}
                for name, stats in sorted(
                    sync_runtime_totals.items(),
                    key=lambda item: item[1]["duration_ns"],
                    reverse=True,
                )[:25]
            ],
            "memcpy_by_kind": [
                {"kind": name, **stats}
                for name, stats in sorted(
                    memcpy_summary.items(), key=lambda item: item[1]["bytes"], reverse=True
                )[:25]
            ],
            "memcpy_samples": memcpy_samples,
        },
        "capture_log": _build_capture_summary(capture_log, graph_dump_dir),
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
