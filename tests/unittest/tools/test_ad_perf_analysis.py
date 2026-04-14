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

"""Unit tests for AutoDeploy performance-analysis helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tensorrt_llm.tools.profiler.ad_perf_analysis import (
    analyze_host_waits,
    analyze_layer_window,
    analyze_memcpy_timeline,
    analyze_serving_config,
    build_iteration_record,
    join_trace_and_graph,
    load_nsys_trace,
    parse_ad_graph_dump_dir,
    parse_runtime_cuda_correlation,
    render_iteration_report,
    select_representative_layer_window,
)


def _build_test_sqlite(sqlite_path: Path) -> None:
    connection = sqlite3.connect(sqlite_path)
    try:
        cursor = connection.cursor()
        cursor.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        cursor.execute(
            """
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,
                end INTEGER,
                eventType INTEGER,
                rangeId INTEGER,
                category INTEGER,
                color INTEGER,
                text TEXT,
                globalTid INTEGER,
                endGlobalTid INTEGER,
                textId INTEGER,
                domainId INTEGER,
                uint64Value INTEGER,
                int64Value INTEGER,
                doubleValue REAL,
                uint32Value INTEGER,
                int32Value INTEGER,
                floatValue REAL,
                jsonTextId INTEGER,
                jsonText TEXT,
                binaryData TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                contextId INTEGER,
                greenContextId INTEGER,
                streamId INTEGER,
                correlationId INTEGER,
                globalPid INTEGER,
                demangledName INTEGER,
                shortName INTEGER,
                mangledName INTEGER,
                launchType INTEGER,
                cacheConfig INTEGER,
                registersPerThread INTEGER,
                gridX INTEGER,
                gridY INTEGER,
                gridZ INTEGER,
                blockX INTEGER,
                blockY INTEGER,
                blockZ INTEGER,
                staticSharedMemory INTEGER,
                dynamicSharedMemory INTEGER,
                localMemoryPerThread INTEGER,
                localMemoryTotal INTEGER,
                gridId INTEGER,
                sharedMemoryExecuted INTEGER,
                graphNodeId INTEGER,
                sharedMemoryLimitConfig INTEGER,
                qmdBulkReleaseDone INTEGER,
                qmdPreexitDone INTEGER,
                qmdLastCtaDone INTEGER,
                graphId INTEGER,
                clusterX INTEGER,
                clusterY INTEGER,
                clusterZ INTEGER,
                clusterSchedulingPolicy INTEGER,
                maxPotentialClusterSize INTEGER,
                maxActiveClusters INTEGER,
                sharedMemoryRequestedPercentage INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER,
                end INTEGER,
                eventClass INTEGER,
                globalTid INTEGER,
                correlationId INTEGER,
                nameId INTEGER,
                returnValue INTEGER,
                callchainId INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                contextId INTEGER,
                greenContextId INTEGER,
                streamId INTEGER,
                correlationId INTEGER,
                globalPid INTEGER,
                bytes INTEGER,
                copyKind INTEGER,
                deprecatedSrcId INTEGER,
                srcKind INTEGER,
                dstKind INTEGER,
                srcDeviceId INTEGER,
                srcContextId INTEGER,
                dstDeviceId INTEGER,
                dstContextId INTEGER,
                migrationCause INTEGER,
                graphNodeId INTEGER,
                virtualAddress INTEGER,
                copyCount INTEGER
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION (
                start INTEGER,
                end INTEGER,
                deviceId INTEGER,
                contextId INTEGER,
                greenContextId INTEGER,
                streamId INTEGER,
                correlationId INTEGER,
                globalPid INTEGER,
                deprecatedSyncType INTEGER,
                syncType INTEGER,
                eventId INTEGER,
                eventSyncId INTEGER
            )
            """
        )
        cursor.execute("CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id INTEGER, name TEXT, label TEXT)")
        cursor.execute("CREATE TABLE ENUM_CUDA_MEM_KIND (id INTEGER, name TEXT, label TEXT)")
        cursor.execute("CREATE TABLE ENUM_CUPTI_SYNC_TYPE (id INTEGER, name TEXT, label TEXT)")
        cursor.execute("CREATE TABLE CUDA_CALLCHAINS (id INTEGER, stackFrames TEXT)")

        cursor.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [
                (1, "Model.layers.0"),
                (2, "cudaLaunchKernel_v7000"),
                (3, "trtllm_qkv_gemm"),
                (4, "trtllm_attention_kernel"),
                (5, "cudaMemcpyAsync_v3020"),
                (6, "cudaStreamSynchronize_v3020"),
                (7, "trtllm_moe_fc1"),
                (8, "Model.layers.1"),
            ],
        )
        cursor.executemany(
            "INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (?, ?, ?)",
            [
                (1, "CUDA_MEMCPY_KIND_HTOD", "Host-to-Device"),
                (2, "CUDA_MEMCPY_KIND_DTOH", "Device-to-Host"),
                (8, "CUDA_MEMCPY_KIND_DTOD", "Device-to-Device"),
            ],
        )
        cursor.executemany(
            "INSERT INTO ENUM_CUDA_MEM_KIND VALUES (?, ?, ?)",
            [
                (0, "CUDA_MEMOPR_MEMORY_KIND_PAGEABLE", "Pageable"),
                (1, "CUDA_MEMOPR_MEMORY_KIND_PINNED", "Pinned"),
                (2, "CUDA_MEMOPR_MEMORY_KIND_DEVICE", "Device"),
            ],
        )
        cursor.executemany(
            "INSERT INTO ENUM_CUPTI_SYNC_TYPE VALUES (?, ?, ?)",
            [
                (3, "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE", "Stream sync"),
                (4, "CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE", "Context sync"),
            ],
        )
        cursor.executemany(
            "INSERT INTO CUDA_CALLCHAINS VALUES (?, ?)",
            [
                (1, "frame_a\nframe_b"),
                (2, "sync_frame"),
            ],
        )

        cursor.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, 0, 0, 0, 0, NULL, 0, 0, ?, 0, 0, 0, 0, 0, 0, 0, 0, NULL, NULL)",
            [
                (100, 700, 1),
                (800, 1100, 8),
            ],
        )
        cursor.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 0, 0, ?, ?, 0, ?)",
            [
                (110, 150, 100, 2, 1),
                (250, 290, 101, 2, 1),
                (320, 350, 102, 5, 1),
                (360, 470, 103, 6, 2),
                (480, 520, 104, 2, 1),
            ],
        )
        cursor.executemany(
            """
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (
                ?, ?, 0, 0, 0, ?, ?, 0, NULL, ?, NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            )
            """,
            [
                (151, 200, 7, 100, 3),
                (291, 310, 7, 101, 4),
                (521, 560, 7, 104, 7),
                (820, 900, 7, 999, 4),
            ],
        )
        cursor.executemany(
            """
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
            (?, ?, 0, 0, 0, ?, ?, 0, ?, ?, 0, ?, ?, 0, 0, 0, 0, 0, NULL, 0, 1)
            """,
            [
                (351, 359, 7, 102, 4096, 2, 2, 0),
            ],
        )
        cursor.executemany(
            """
            INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES
            (?, ?, 0, 0, 0, ?, ?, 0, 0, ?, 0, 0)
            """,
            [
                (360, 470, 7, 103, 3),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _build_graph_dump(graph_dir: Path) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graph_dir / "001_runtime_export.txt"
    graph_path.write_text(
        "\n".join(
            [
                "# Transform: export",
                "# Stage: final",
                "",
                "================================================================================",
                "# GraphModule: (root)",
                "================================================================================",
                "",
                "%input_ids : 4x1 : torch.int32",
                "%qkv = trtllm_qkv_gemm(%input_ids : 4x1 : torch.int32) : 4x64 : torch.bfloat16",
                "%attn = trtllm_attention_kernel(%qkv : 4x64 : torch.bfloat16) : 4x64 : torch.bfloat16",
                "%moe = trtllm_moe_fc1(%attn : 4x64 : torch.bfloat16) : 4x64 : torch.bfloat16",
                "output %moe",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_load_nsys_trace_extracts_layer_windows(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    _build_test_sqlite(sqlite_path)

    trace_data = load_nsys_trace(sqlite_path)

    assert len(trace_data["ranges"]) == 2
    assert len(trace_data["layer_windows"]) == 2
    assert trace_data["layer_windows"][0]["layer_id"] == "0"
    assert trace_data["layer_windows"][0]["kernel_count"] == 3


def test_runtime_correlation_and_memcpy_analysis(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    _build_test_sqlite(sqlite_path)

    runtime_correlation = parse_runtime_cuda_correlation(sqlite_path)
    memcpy_analysis = analyze_memcpy_timeline(runtime_correlation)

    assert runtime_correlation["summary"]["runtime_with_kernels"] == 3
    assert runtime_correlation["summary"]["runtime_with_memcpys"] == 1
    assert runtime_correlation["summary"]["runtime_with_syncs"] == 1
    assert runtime_correlation["summary"]["runtime_with_callchains"] >= 1
    assert memcpy_analysis["summary"]["bytes_by_kind"]["Device-to-Host"] == 4096
    assert any(finding["category"] == "memcpy" for finding in memcpy_analysis["findings"])


def test_graph_join_layer_analysis_and_report(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "trace.sqlite"
    graph_dir = tmp_path / "graphs"
    config_path = tmp_path / "serving.yaml"
    comparable_dir = tmp_path / "comparables"
    _build_test_sqlite(sqlite_path)
    _build_graph_dump(graph_dir)
    comparable_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text("max_seq_len: 8192\n", encoding="utf-8")
    (comparable_dir / "baseline.yaml").write_text(
        "max_seq_len: 1024\nchunked_prefill: true\nenable_multistream_moe: true\n",
        encoding="utf-8",
    )

    trace_data = load_nsys_trace(sqlite_path)
    runtime_correlation = parse_runtime_cuda_correlation(sqlite_path)
    graph_data = parse_ad_graph_dump_dir(graph_dir)
    joined = join_trace_and_graph(trace_data, graph_data)
    selected_layer = select_representative_layer_window(joined["joined_timeline"])
    layer_analysis = analyze_layer_window(selected_layer, runtime_correlation)
    host_wait_analysis = analyze_host_waits(trace_data, runtime_correlation)
    memcpy_analysis = analyze_memcpy_timeline(runtime_correlation)
    config_analysis = analyze_serving_config(config_path, comparable_roots=[comparable_dir])
    iteration_record = build_iteration_record(
        model="MiniMaxAI/MiniMax-M2",
        objective="decode analysis",
        trace_data=trace_data,
        graph_data=graph_data,
        runtime_correlation=runtime_correlation,
        layer_analysis=layer_analysis,
        memcpy_analysis=memcpy_analysis,
        host_wait_analysis=host_wait_analysis,
        config_analysis=config_analysis,
        change={"type": "config", "description": "evaluate chunked_prefill"},
    )
    report = render_iteration_report(iteration_record)

    assert joined["confidence_counts"]["high"] >= 1
    assert layer_analysis["summary"]["selected"] is True
    assert layer_analysis["summary"]["layer_id"] == "0"
    assert any(finding["category"] == "host_wait" for finding in host_wait_analysis["findings"])
    assert any(finding["category"] == "config" for finding in config_analysis["findings"])
    assert "Host CUDA API To GPU Correlation" in report
    assert "Data Movement Summary" in report
    assert "Host Wait Analysis" in report
