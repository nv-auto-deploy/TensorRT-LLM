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

"""Helpers for parsing Nsight Systems SQLite exports."""

from __future__ import annotations

import sqlite3
import subprocess  # nosec B404
from collections import defaultdict
from pathlib import Path
from typing import Any

from .models import (
    CorrelatedRuntimeRecord,
    KernelRecord,
    MemcpyRecord,
    NvtxRange,
    RuntimeCall,
    SyncRecord,
)

STRING_FALLBACK = "<unknown>"


def _load_string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    cursor = connection.cursor()
    cursor.execute("SELECT id, value FROM StringIds")
    return {row[0]: row[1] for row in cursor.fetchall()}


def _load_enum_table(connection: sqlite3.Connection, table_name: str) -> dict[int, str]:
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT id, label FROM {table_name}")
    except sqlite3.OperationalError:
        return {}
    return {row[0]: row[1] for row in cursor.fetchall()}


def export_nsys_sqlite(
    nsys_rep_path: str | Path,
    output_path: str | Path | None = None,
    nsys_cmd: str = "nsys",
) -> Path:
    """Export an `.nsys-rep` file to SQLite using `nsys export`.

    Args:
        nsys_rep_path: Path to the input Nsight Systems report.
        output_path: Desired SQLite path. Defaults to replacing the suffix.
        nsys_cmd: Nsight Systems binary to invoke.

    Returns:
        Path to the generated SQLite file.
    """
    nsys_rep = Path(nsys_rep_path)
    sqlite_path = Path(output_path) if output_path is not None else nsys_rep.with_suffix(".sqlite")

    command = [
        nsys_cmd,
        "export",
        "--force-overwrite=true",
        "--type",
        "sqlite",
        "--output",
        str(sqlite_path),
        str(nsys_rep),
    ]
    subprocess.run(command, check=True)  # nosec B603
    return sqlite_path


def _fetch_nvtx_ranges(connection: sqlite3.Connection, string_ids: dict[int, str]) -> list[NvtxRange]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT
            start,
            end,
            COALESCE(text, StringIds.value, jsonText, ?),
            globalTid,
            category,
            domainId
        FROM NVTX_EVENTS
        LEFT JOIN StringIds
            ON StringIds.id = NVTX_EVENTS.textId
        WHERE end > start
        ORDER BY start, end DESC
        """,
        (STRING_FALLBACK,),
    )
    return [
        NvtxRange(
            start=row[0],
            end=row[1],
            name=row[2] or STRING_FALLBACK,
            global_tid=row[3],
            category=row[4],
            domain_id=row[5],
        )
        for row in cursor.fetchall()
    ]


def _fetch_kernels(connection: sqlite3.Connection, string_ids: dict[int, str]) -> list[KernelRecord]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT start, end, correlationId, streamId, shortName, demangledName, graphNodeId
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        ORDER BY start
        """
    )
    kernels = []
    for row in cursor.fetchall():
        kernels.append(
            KernelRecord(
                start=row[0],
                end=row[1],
                correlation_id=row[2],
                stream_id=row[3],
                short_name=string_ids.get(row[4], STRING_FALLBACK),
                demangled_name=string_ids.get(row[5]) if row[5] is not None else None,
                graph_node_id=row[6],
            )
        )
    return kernels


def _fetch_runtime_calls(connection: sqlite3.Connection, string_ids: dict[int, str]) -> list[RuntimeCall]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT start, end, correlationId, nameId, globalTid, callchainId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        ORDER BY start
        """
    )
    return [
        RuntimeCall(
            start=row[0],
            end=row[1],
            correlation_id=row[2],
            name=string_ids.get(row[3], STRING_FALLBACK),
            global_tid=row[4],
            callchain_id=row[5],
        )
        for row in cursor.fetchall()
    ]


def _fetch_memcpys(
    connection: sqlite3.Connection,
    memcpy_kinds: dict[int, str],
    mem_kinds: dict[int, str],
) -> list[MemcpyRecord]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT start, end, correlationId, bytes, copyKind, srcKind, dstKind, streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        ORDER BY start
        """
    )
    memcpys = []
    for row in cursor.fetchall():
        memcpys.append(
            MemcpyRecord(
                start=row[0],
                end=row[1],
                correlation_id=row[2],
                bytes=row[3],
                copy_kind=memcpy_kinds.get(row[4], f"Unknown({row[4]})"),
                src_kind=mem_kinds.get(row[5], f"Unknown({row[5]})"),
                dst_kind=mem_kinds.get(row[6], f"Unknown({row[6]})"),
                stream_id=row[7],
            )
        )
    return memcpys


def _fetch_syncs(connection: sqlite3.Connection, sync_types: dict[int, str]) -> list[SyncRecord]:
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT start, end, correlationId, syncType, streamId
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
        ORDER BY start
        """
    )
    return [
        SyncRecord(
            start=row[0],
            end=row[1],
            correlation_id=row[2],
            sync_type=sync_types.get(row[3], f"Unknown({row[3]})"),
            stream_id=row[4],
        )
        for row in cursor.fetchall()
    ]


def _fetch_callchains(connection: sqlite3.Connection) -> dict[int, list[str]]:
    cursor = connection.cursor()
    tables = {
        row[0]
        for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    if "CUDA_CALLCHAINS" not in tables:
        return {}

    try:
        cursor.execute("SELECT id, stackFrames FROM CUDA_CALLCHAINS")
    except sqlite3.OperationalError:
        return {}

    callchains: dict[int, list[str]] = {}
    for callchain_id, stack_frames in cursor.fetchall():
        if stack_frames is None:
            callchains[callchain_id] = []
            continue
        if isinstance(stack_frames, str):
            frames = [frame.strip() for frame in stack_frames.split("\n") if frame.strip()]
        else:
            frames = [str(stack_frames)]
        callchains[callchain_id] = frames
    return callchains


def build_range_tree(ranges: list[NvtxRange]) -> list[NvtxRange]:
    """Build a nested NVTX tree by timestamp containment."""
    roots: list[NvtxRange] = []
    stack: list[NvtxRange] = []

    for current_range in sorted(ranges, key=lambda item: (item.start, -item.end)):
        while stack and current_range.start >= stack[-1].end:
            stack.pop()
        if stack and current_range.end <= stack[-1].end:
            current_range.depth = stack[-1].depth + 1
            stack[-1].children.append(current_range)
        else:
            current_range.depth = 0
            roots.append(current_range)
        stack.append(current_range)
    return roots


def _serialize_range_tree(ranges: list[NvtxRange]) -> list[dict[str, Any]]:
    return [current_range.to_dict() for current_range in ranges]


def _extract_layer_id(name: str) -> str | None:
    lowered = name.lower()
    markers = ["layers.", "layer ", "layer_", "decoder.layers.", "model.layers."]
    for marker in markers:
        start = lowered.find(marker)
        if start < 0:
            continue
        suffix = lowered[start + len(marker) :]
        digits = []
        for character in suffix:
            if character.isdigit():
                digits.append(character)
            else:
                break
        if digits:
            return "".join(digits)
    return None


def _layer_windows_from_ranges(ranges: list[NvtxRange], kernels: list[KernelRecord]) -> list[dict[str, Any]]:
    layer_windows = []
    for current_range in ranges:
        layer_id = _extract_layer_id(current_range.name)
        if layer_id is None:
            continue
        enclosed_kernels = [
            kernel for kernel in kernels if kernel.start >= current_range.start and kernel.end <= current_range.end
        ]
        layer_windows.append(
            {
                "layer_id": layer_id,
                "name": current_range.name,
                "start": current_range.start,
                "end": current_range.end,
                "duration_ns": current_range.duration_ns,
                "kernel_count": len(enclosed_kernels),
                "kernels": [kernel.to_dict() for kernel in enclosed_kernels],
            }
        )
    return layer_windows


def _flatten_range_tree(ranges: list[NvtxRange]) -> list[NvtxRange]:
    flattened = []
    stack = list(reversed(ranges))
    while stack:
        current_range = stack.pop()
        flattened.append(current_range)
        stack.extend(reversed(current_range.children))
    return flattened


def load_nsys_trace(sqlite_path: str | Path) -> dict[str, Any]:
    """Load the relevant Nsight Systems trace tables into Python structures."""
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        string_ids = _load_string_ids(connection)
        memcpy_kinds = _load_enum_table(connection, "ENUM_CUDA_MEMCPY_OPER")
        mem_kinds = _load_enum_table(connection, "ENUM_CUDA_MEM_KIND")
        sync_types = _load_enum_table(connection, "ENUM_CUPTI_SYNC_TYPE")

        ranges = _fetch_nvtx_ranges(connection, string_ids)
        kernels = _fetch_kernels(connection, string_ids)
        runtime_calls = _fetch_runtime_calls(connection, string_ids)
        memcpys = _fetch_memcpys(connection, memcpy_kinds, mem_kinds)
        syncs = _fetch_syncs(connection, sync_types)
        callchains = _fetch_callchains(connection)

        roots = build_range_tree(ranges)
        flattened_ranges = _flatten_range_tree(roots)
        layer_windows = _layer_windows_from_ranges(flattened_ranges, kernels)

        return {
            "sqlite_path": str(sqlite_path),
            "range_tree": _serialize_range_tree(roots),
            "ranges": [current_range.to_dict() for current_range in flattened_ranges],
            "kernels": [kernel.to_dict() for kernel in kernels],
            "runtime_calls": [runtime_call.to_dict() for runtime_call in runtime_calls],
            "memcpys": [memcpy.to_dict() for memcpy in memcpys],
            "syncs": [sync.to_dict() for sync in syncs],
            "callchains": callchains,
            "layer_windows": layer_windows,
        }
    finally:
        connection.close()


def parse_runtime_cuda_correlation(sqlite_path: str | Path) -> dict[str, Any]:
    """Correlate host CUDA runtime calls with GPU-side activity records."""
    connection = sqlite3.connect(sqlite_path)
    try:
        string_ids = _load_string_ids(connection)
        memcpy_kinds = _load_enum_table(connection, "ENUM_CUDA_MEMCPY_OPER")
        mem_kinds = _load_enum_table(connection, "ENUM_CUDA_MEM_KIND")
        sync_types = _load_enum_table(connection, "ENUM_CUPTI_SYNC_TYPE")
        runtime_calls = _fetch_runtime_calls(connection, string_ids)
        kernels = _fetch_kernels(connection, string_ids)
        memcpys = _fetch_memcpys(connection, memcpy_kinds, mem_kinds)
        syncs = _fetch_syncs(connection, sync_types)
        callchains = _fetch_callchains(connection)

        kernels_by_correlation: dict[int, list[KernelRecord]] = defaultdict(list)
        for kernel in kernels:
            kernels_by_correlation[kernel.correlation_id].append(kernel)

        memcpys_by_correlation: dict[int, list[MemcpyRecord]] = defaultdict(list)
        for memcpy in memcpys:
            memcpys_by_correlation[memcpy.correlation_id].append(memcpy)

        syncs_by_correlation: dict[int, list[SyncRecord]] = defaultdict(list)
        for sync in syncs:
            syncs_by_correlation[sync.correlation_id].append(sync)

        correlated_records = []
        for runtime_call in runtime_calls:
            correlated_records.append(
                CorrelatedRuntimeRecord(
                    runtime_call=runtime_call,
                    kernels=kernels_by_correlation.get(runtime_call.correlation_id, []),
                    memcpys=memcpys_by_correlation.get(runtime_call.correlation_id, []),
                    syncs=syncs_by_correlation.get(runtime_call.correlation_id, []),
                    callchain=callchains.get(runtime_call.callchain_id)
                    if runtime_call.callchain_id is not None
                    else None,
                )
            )

        summary = {
            "runtime_count": len(runtime_calls),
            "runtime_with_kernels": sum(1 for record in correlated_records if record.kernels),
            "runtime_with_memcpys": sum(1 for record in correlated_records if record.memcpys),
            "runtime_with_syncs": sum(1 for record in correlated_records if record.syncs),
            "runtime_with_callchains": sum(1 for record in correlated_records if record.callchain),
        }
        return {
            "summary": summary,
            "records": [record.to_dict() for record in correlated_records],
        }
    finally:
        connection.close()


def merge_kernel_intervals(kernels: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Merge overlapping kernel intervals to compute active GPU periods."""
    if not kernels:
        return []

    intervals = sorted((kernel["start"], kernel["end"]) for kernel in kernels)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_gpu_gap_intervals(kernels: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Return idle gaps between merged kernel intervals."""
    merged = merge_kernel_intervals(kernels)
    gaps = []
    for (_, previous_end), (next_start, _) in zip(merged, merged[1:]):
        if next_start > previous_end:
            gaps.append(
                {
                    "start": previous_end,
                    "end": next_start,
                    "duration_ns": next_start - previous_end,
                }
            )
    return gaps
