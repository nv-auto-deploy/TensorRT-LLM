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

"""Shared data models for AutoDeploy performance-analysis utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class NvtxRange:
    """One NVTX range record exported by Nsight Systems."""

    start: int
    end: int
    name: str
    global_tid: int | None = None
    category: int | None = None
    domain_id: int | None = None
    depth: int = 0
    children: list["NvtxRange"] = field(default_factory=list)

    @property
    def duration_ns(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ns"] = self.duration_ns
        return payload


@dataclass(slots=True)
class KernelRecord:
    """One GPU kernel execution record."""

    start: int
    end: int
    correlation_id: int
    stream_id: int | None
    short_name: str
    demangled_name: str | None = None
    graph_node_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end - self.start

    def display_name(self) -> str:
        return self.demangled_name or self.short_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ns"] = self.duration_ns
        payload["display_name"] = self.display_name()
        return payload


@dataclass(slots=True)
class RuntimeCall:
    """One CUDA runtime or driver API call."""

    start: int
    end: int
    correlation_id: int
    name: str
    global_tid: int | None = None
    callchain_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ns"] = self.duration_ns
        return payload


@dataclass(slots=True)
class MemcpyRecord:
    """One CUDA memcpy activity record."""

    start: int
    end: int
    correlation_id: int
    bytes: int
    copy_kind: str
    src_kind: str
    dst_kind: str
    stream_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ns"] = self.duration_ns
        return payload


@dataclass(slots=True)
class SyncRecord:
    """One synchronization activity record."""

    start: int
    end: int
    correlation_id: int
    sync_type: str
    stream_id: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_ns"] = self.duration_ns
        return payload


@dataclass(slots=True)
class CorrelatedRuntimeRecord:
    """One runtime call annotated with joined GPU-side activities."""

    runtime_call: RuntimeCall
    kernels: list[KernelRecord] = field(default_factory=list)
    memcpys: list[MemcpyRecord] = field(default_factory=list)
    syncs: list[SyncRecord] = field(default_factory=list)
    callchain: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_call": self.runtime_call.to_dict(),
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "memcpys": [memcpy.to_dict() for memcpy in self.memcpys],
            "syncs": [sync.to_dict() for sync in self.syncs],
            "callchain": self.callchain,
        }
