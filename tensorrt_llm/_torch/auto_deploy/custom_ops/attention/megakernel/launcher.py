# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""JIT compilation and launch wrapper for the Gemma4 megakernel."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import torch

_MODULE_CACHE: dict[str, Any] = {}


@functools.lru_cache(maxsize=1)
def _source_dir() -> Path:
    return Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def _deep_gemm_include_dir() -> Path:
    repo_root = _source_dir().parents[5]
    return repo_root / "tensorrt_llm" / "deep_gemm" / "include"


def load_megakernel_module() -> Any:
    """JIT-compile and load the megakernel CUDA extension."""
    if "gemma4_megakernel" in _MODULE_CACHE:
        return _MODULE_CACHE["gemma4_megakernel"]

    import hashlib

    from torch.utils.cpp_extension import load

    src_dir = _source_dir()
    deep_gemm_include_dir = _deep_gemm_include_dir()
    build_dir = src_dir / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Hash CUDA source files (excluding _build) to detect changes and force recompilation
    src_files = [
        f
        for f in sorted(src_dir.glob("**/*.cuh")) + sorted(src_dir.glob("**/*.cu"))
        if "_build" not in str(f)
    ]
    hasher = hashlib.md5()
    for f in src_files:
        hasher.update(f.read_bytes())
    src_hash = hasher.hexdigest()[:8]

    module = load(
        name=f"gemma4_megakernel_{src_hash}",
        sources=[str(src_dir / "megakernel_host.cu")],
        extra_include_paths=[str(src_dir), str(deep_gemm_include_dir)],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )
    _MODULE_CACHE["gemma4_megakernel"] = module
    return module


def choose_attention_num_partials(total_pages: int) -> int:
    """Choose the multi-SM attention partition count for one KV head.

    Empirically on H100 after the shared-memory page reuse and `cp.async`
    pipeline work, aggressive partitioning helps only up to a point. Short
    decode windows still prefer one partial per page, but longer windows now
    hit a reduction-overhead knee before the old 16-partial cap.
    """
    if total_pages <= 2:
        return 1
    if total_pages <= 16:
        return total_pages
    if total_pages <= 48:
        return 11
    return 11


def partition_attention_pages(total_pages: int, num_partials: int) -> list[tuple[int, int]]:
    """Split pages into contiguous non-empty ranges with near-even sizes."""
    if total_pages <= 0:
        return []
    num_partials = max(1, min(num_partials, total_pages))
    base = total_pages // num_partials
    remainder = total_pages % num_partials
    ranges = []
    start = 0
    for pi in range(num_partials):
        size = base + (1 if pi < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


class InstructionBuilder:
    """Build per-SM instruction queues for the megakernel.

    Instructions are 32 x int32 (128 bytes each).
    The builder maintains a [num_sms, max_instructions, 32] tensor
    and a per-SM instruction counter.
    """

    def __init__(self, num_sms: int, max_instructions: int = 512, device: str = "cuda"):
        self.num_sms = num_sms
        self.max_instructions = max_instructions
        self.device = device

        # Instruction queue: [num_sms, max_instructions, 32]
        self.instructions = torch.zeros(
            num_sms, max_instructions, 32, dtype=torch.int32, device=device
        )
        # Per-SM instruction count
        self.counts = torch.zeros(num_sms, dtype=torch.int32, device=device)
        # Host-side counts for building
        self._host_counts = [0] * num_sms

    def _add_instruction(
        self, sm_id: int, words: list[int], barrier_count: int = 0, barrier_id: int = 0
    ) -> None:
        idx = self._host_counts[sm_id]
        if idx >= self.max_instructions:
            raise RuntimeError(f"SM {sm_id} instruction queue full ({self.max_instructions} max)")
        padded = words + [0] * (32 - len(words))
        # Embed barrier in last two words: [30] = count, [31] = id
        padded[30] = barrier_count
        padded[31] = barrier_id
        self.instructions[sm_id, idx] = torch.tensor(
            padded[:32], dtype=torch.int32, device=self.device
        )
        self._host_counts[sm_id] = idx + 1
        self.counts[sm_id] = idx + 1

    def add_noop(
        self,
        sm_ids: list[int] | range,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add a no-op instruction to specified SMs."""
        bc, bid = barrier if barrier else (0, 0)
        for sm_id in sm_ids:
            self._add_instruction(sm_id, [0], barrier_count=bc, barrier_id=bid)  # OP_NOOP = 0

    def add_barrier(self, sm_ids: list[int] | range, barrier_id: int) -> None:
        """Add a Lamport barrier instruction to specified SMs.

        barrier_id is used as the generation counter value. Each successive
        barrier in the instruction stream should use an incrementing generation.
        All SMs in sm_ids must execute this instruction.
        The expected_count = total num_sms (all SMs participate in the generation).
        """
        expected_count = len(list(sm_ids))
        for sm_id in sm_ids:
            self._add_instruction(sm_id, [1, barrier_id, expected_count])  # OP_BARRIER = 1

    def add_gemv_qkv(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add QKV GEMV instruction to specified SMs.

        Args:
            sm_ids: which SMs to assign
            row_ranges: (row_start, row_end) per SM — which output rows to compute
            token_id: which token to process
        """
        sm_list = list(sm_ids)
        assert len(sm_list) == len(row_ranges), (
            f"sm_ids ({len(sm_list)}) and row_ranges ({len(row_ranges)}) must match"
        )
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, (row_start, row_end) in zip(sm_list, row_ranges):
            self._add_instruction(
                sm_id, [2, row_start, row_end, token_id], barrier_count=bc, barrier_id=bid
            )

    def add_qkv_post(
        self,
        sm_ids: list[int] | range,
        head_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
        shared_kv: bool = False,
        rope_dim: int = 0,
    ) -> None:
        """Add QKV post-processing instruction (norms + RoPE + cache write).

        Args:
            shared_kv: If True, K values are copied to V cache slot (K=V sharing
                       for Gemma4 full-attention layers).
            rope_dim: Number of dimensions per half to rotate. 0 = full rotation
                      (standard RoPE). >0 = proportional RoPE (e.g., 32 for 25%
                      of head_dim=256, meaning rotate first 32 of 128 per half).
        """
        sm_list = list(sm_ids)
        assert len(sm_list) == len(head_ranges)
        bc, bid = barrier if barrier else (0, 0)
        skv = 1 if shared_kv else 0
        for sm_id, (h_start, h_end) in zip(sm_list, head_ranges):
            self._add_instruction(
                sm_id,
                [6, h_start, h_end, token_id, skv, rope_dim],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_paged_attn(
        self,
        sm_ids: list[int] | range,
        kv_head_indices: list[int],
        token_id: int = 0,
        page_ranges: list[tuple[int, int]] | None = None,
        partial_indices: list[int] | None = None,
        is_single: bool = True,
        barrier: tuple[int, int] | None = None,
        sliding_window: int = 0,
    ) -> None:
        """Add paged attention instruction.

        For single-partial mode (default): one SM per KV head, processes all pages.
        For multi-partial mode: multiple SMs per KV head, each processing a page range.
        """
        sm_list = list(sm_ids)
        n = len(sm_list)
        assert len(kv_head_indices) == n
        if page_ranges is None:
            page_ranges = [(0, 9999)] * n  # sentinel: handler clips to actual page count
        if partial_indices is None:
            partial_indices = list(range(n))
        assert len(page_ranges) == n and len(partial_indices) == n
        is_single_val = 1 if is_single else 0
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, kv_head, (ps, pe), pidx in zip(
            sm_list, kv_head_indices, page_ranges, partial_indices
        ):
            self._add_instruction(
                sm_id,
                [3, kv_head, token_id, ps, pe, pidx, is_single_val, sliding_window],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_attn_reduce(
        self,
        sm_ids: list[int] | range,
        kv_head_indices: list[int],
        num_partials: int,
        partial_starts: list[int] | None = None,
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add attention reduction instruction — one SM per KV head."""
        sm_list = list(sm_ids)
        assert len(sm_list) == len(kv_head_indices)
        if partial_starts is None:
            partial_starts = [kv * num_partials for kv in range(len(sm_list))]
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, kv_head, ps in zip(sm_list, kv_head_indices, partial_starts):
            self._add_instruction(
                sm_id, [4, kv_head, token_id, num_partials, ps], barrier_count=bc, barrier_id=bid
            )

    def add_gemv_oproj(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add O-proj GEMV instruction."""
        sm_list = list(sm_ids)
        assert len(sm_list) == len(row_ranges)
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, (rs, re) in zip(sm_list, row_ranges):
            self._add_instruction(sm_id, [5, rs, re, token_id], barrier_count=bc, barrier_id=bid)

    def add_oproj_post(self, sm_id: int, token_id: int = 0) -> None:
        """Add O-proj post-processing (norms + residual). Single SM."""
        self._add_instruction(sm_id, [7, token_id, 0])  # mode 0 = legacy single-SM path

    def _add_oproj_post_mode(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int,
        mode: int,
        num_partials: int,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        sm_list = list(sm_ids)
        assert len(sm_list) == len(row_ranges)
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, (row_start, row_end) in zip(sm_list, row_ranges):
            self._add_instruction(
                sm_id,
                [7, token_id, mode, row_start, row_end, num_partials],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_oproj_post_stats(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add distributed O-proj RMS-stat accumulation."""
        self._add_oproj_post_mode(
            sm_ids,
            row_ranges,
            token_id=token_id,
            mode=1,
            num_partials=len(list(sm_ids)),
            barrier=barrier,
        )

    def add_oproj_post_apply(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add distributed post-attention RMSNorm + residual phase."""
        self._add_oproj_post_mode(
            sm_ids,
            row_ranges,
            token_id=token_id,
            mode=2,
            num_partials=len(list(sm_ids)),
            barrier=barrier,
        )

    def add_pre_ffn_post(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
    ) -> None:
        """Add distributed pre-FFN RMSNorm phase."""
        self._add_oproj_post_mode(
            sm_ids,
            row_ranges,
            token_id=token_id,
            mode=3,
            num_partials=len(list(sm_ids)),
        )

    def add_ffn_gateup(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        layer_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Reserve Kernel B gate/up FFN work in the instruction stream."""
        sm_list = list(sm_ids)
        assert len(sm_list) == len(row_ranges)
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, (rs, re) in zip(sm_list, row_ranges):
            self._add_instruction(
                sm_id,
                [8, layer_id, rs, re, token_id],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_ffn_down(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        layer_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Reserve Kernel B down-projection work in the instruction stream."""
        sm_list = list(sm_ids)
        assert len(sm_list) == len(row_ranges)
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, (rs, re) in zip(sm_list, row_ranges):
            self._add_instruction(
                sm_id,
                [9, layer_id, rs, re, token_id],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_router_topk(
        self,
        sm_ids: list[int] | range,
        token_id: int = 0,
        layer_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Reserve Kernel B router/top-k work in the instruction stream."""
        bc, bid = barrier if barrier else (0, 0)
        for sm_id in sm_ids:
            self._add_instruction(
                sm_id,
                [10, layer_id, token_id],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_moe(
        self,
        sm_ids: list[int] | range,
        expert_ranges: list[tuple[int, int]],
        token_id: int = 0,
        layer_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Reserve Kernel B MoE expert work in the instruction stream."""
        sm_list = list(sm_ids)
        assert len(sm_list) == len(expert_ranges)
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, (es, ee) in zip(sm_list, expert_ranges):
            self._add_instruction(
                sm_id,
                [11, layer_id, es, ee, token_id],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_moe_sharded(
        self,
        sm_ids: list[int] | range,
        slot_indices: list[int],
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        mode: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Add sharded Kernel B MoE work.

        mode 0: expert gate/up rows
        mode 1: expert down-proj rows
        """
        sm_list = list(sm_ids)
        assert len(sm_list) == len(slot_indices) == len(row_ranges)
        bc, bid = barrier if barrier else (0, 0)
        for sm_id, slot_idx, (rs, re) in zip(sm_list, slot_indices, row_ranges):
            self._add_instruction(
                sm_id,
                [11, mode, slot_idx, rs, re, token_id],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_b_post(
        self,
        sm_ids: list[int] | range,
        token_id: int = 0,
        layer_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        """Reserve Kernel B post-processing work in the instruction stream."""
        bc, bid = barrier if barrier else (0, 0)
        for sm_id in sm_ids:
            self._add_instruction(
                sm_id,
                [12, token_id, 0],
                barrier_count=bc,
                barrier_id=bid,
            )

    def _add_b_post_mode(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int,
        mode: int,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        sm_list = list(sm_ids)
        assert len(sm_list) == len(row_ranges)
        bc, bid = barrier if barrier else (0, 0)
        num_partials = len(sm_list)
        for sm_id, (row_start, row_end) in zip(sm_list, row_ranges):
            self._add_instruction(
                sm_id,
                [12, token_id, mode, row_start, row_end, num_partials],
                barrier_count=bc,
                barrier_id=bid,
            )

    def add_b_post_stats(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        self._add_b_post_mode(sm_ids, row_ranges, token_id=token_id, mode=1, barrier=barrier)

    def add_b_post_merge(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        self._add_b_post_mode(sm_ids, row_ranges, token_id=token_id, mode=2, barrier=barrier)

    def add_b_post_hidden(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        self._add_b_post_mode(sm_ids, row_ranges, token_id=token_id, mode=3, barrier=barrier)

    def add_b_post_next_norm(
        self,
        sm_ids: list[int] | range,
        row_ranges: list[tuple[int, int]],
        token_id: int = 0,
        barrier: tuple[int, int] | None = None,
    ) -> None:
        self._add_b_post_mode(sm_ids, row_ranges, token_id=token_id, mode=4, barrier=barrier)

    def add_done(self, sm_ids: list[int] | range) -> None:
        """Add a done instruction (kernel exit) to specified SMs."""
        for sm_id in sm_ids:
            self._add_instruction(sm_id, [255])  # OP_DONE = 255


class MegakernelLauncher:
    """Manage state and launch the persistent megakernel."""

    def __init__(self, num_sms: int = 132, device: str = "cuda"):
        self.num_sms = num_sms
        self.max_barrier_slots = 256
        self.device = device
        self._module = load_megakernel_module()
        self._config = self._module.get_config()

    @property
    def config(self) -> dict:
        return self._config

    def launch(self, builder: InstructionBuilder) -> torch.Tensor:
        """Launch the megakernel with the given instruction queue.

        Returns the debug_output tensor [num_sms] which is non-zero
        for each SM that completed execution.
        """
        barrier_slots = torch.zeros(
            self.max_barrier_slots,
            dtype=torch.int32,
            device=self.device,
        )
        debug_output = torch.zeros(self.num_sms, dtype=torch.int32, device=self.device)

        self._module.launch_megakernel(
            builder.instructions,
            builder.counts,
            barrier_slots,
            debug_output,
            self.num_sms,
        )
        return debug_output

    def launch_preallocated(
        self,
        builder: InstructionBuilder,
        barrier_slots: torch.Tensor,
        debug_output: torch.Tensor,
    ) -> torch.Tensor:
        """Launch with pre-allocated buffers (zero them before calling).

        Use this for benchmarking to avoid torch.zeros allocation overhead.
        Caller must zero barrier_slots and debug_output before each call.
        """
        self._module.launch_megakernel(
            builder.instructions,
            builder.counts,
            barrier_slots,
            debug_output,
            self.num_sms,
        )
        return debug_output

    def launch_gemv_qkv(
        self,
        builder: InstructionBuilder,
        attn_normed: torch.Tensor,
        qkv_weight: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        v_norm_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_loc: torch.Tensor,
        cu_num_pages: torch.Tensor,
        triton_positions: torch.Tensor,
        triton_batch_indices: torch.Tensor,
        last_page_len: torch.Tensor,
        qkv_scratch: torch.Tensor,
        attn_scratch: torch.Tensor,
        eps: float = 1e-6,
        page_size: int = 16,
        attn_scale: float = 0.0,
        # Phase 3 (optional — pass empty tensors to skip)
        o_proj_weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        post_attn_norm_weight: torch.Tensor | None = None,
        pre_ffn_norm_weight: torch.Tensor | None = None,
        o_proj_scratch: torch.Tensor | None = None,
        post_attn_out: torch.Tensor | None = None,
        pre_ffn_out: torch.Tensor | None = None,
        partial_attn_scratch: torch.Tensor | None = None,
        # Pre-allocated barrier/debug tensors (avoids CUDA alloc in hot path)
        barrier_slots: torch.Tensor | None = None,
        debug_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch the megakernel with Gemma4 globals."""
        import math

        if attn_scale == 0.0:
            attn_scale = 1.0 / math.sqrt(256)  # HEAD_DIM

        empty = torch.empty(0, device=self.device)
        if barrier_slots is None:
            barrier_slots = torch.zeros(
                self.max_barrier_slots, dtype=torch.int32, device=self.device
            )
        else:
            barrier_slots.zero_()
        if debug_output is None:
            debug_output = torch.zeros(self.num_sms, dtype=torch.int32, device=self.device)
        else:
            debug_output.zero_()
        self._module.launch_gemv_qkv(
            builder.instructions,
            builder.counts,
            barrier_slots,
            debug_output,
            self.num_sms,
            attn_normed,
            qkv_weight,
            q_norm_weight,
            k_norm_weight,
            v_norm_weight,
            cos_sin_cache,
            kv_cache,
            cache_loc,
            cu_num_pages,
            triton_positions,
            triton_batch_indices,
            last_page_len,
            qkv_scratch,
            attn_scratch,
            eps,
            page_size,
            attn_scale,
            o_proj_weight if o_proj_weight is not None else empty,
            residual if residual is not None else empty,
            post_attn_norm_weight if post_attn_norm_weight is not None else empty,
            pre_ffn_norm_weight if pre_ffn_norm_weight is not None else empty,
            o_proj_scratch if o_proj_scratch is not None else empty,
            post_attn_out if post_attn_out is not None else empty,
            pre_ffn_out if pre_ffn_out is not None else empty,
            partial_attn_scratch if partial_attn_scratch is not None else empty,
        )
        return debug_output

    def launch_and_sync(self, builder: InstructionBuilder) -> torch.Tensor:
        """Launch and wait for completion. Returns debug_output."""
        result = self.launch(builder)
        torch.cuda.synchronize()
        return result

    def launch_dense_b(
        self,
        builder: InstructionBuilder,
        post_attn_in: torch.Tensor,
        pre_ffn_in: torch.Tensor,
        ffn_gate_up_weight: torch.Tensor,
        ffn_down_weight: torch.Tensor,
        post_ffn1_norm_weight: torch.Tensor,
        post_ffn_norm_weight: torch.Tensor,
        layer_scalar: torch.Tensor,
        next_input_norm_weight: torch.Tensor,
        ffn_gate_scratch: torch.Tensor,
        ffn_up_scratch: torch.Tensor,
        ffn_down_scratch: torch.Tensor,
        hidden_out: torch.Tensor,
        next_attn_normed_out: torch.Tensor,
        router_proj_weight: torch.Tensor | None = None,
        router_scale: torch.Tensor | None = None,
        router_root_size: torch.Tensor | None = None,
        pre_ffn2_norm_weight: torch.Tensor | None = None,
        router_topk_weights: torch.Tensor | None = None,
        router_topk_indices: torch.Tensor | None = None,
        moe_input_scratch: torch.Tensor | None = None,
        post_ffn2_norm_weight: torch.Tensor | None = None,
        moe_w13_stacked_weight: torch.Tensor | None = None,
        moe_w2_weight: torch.Tensor | None = None,
        moe_gate_scratch: torch.Tensor | None = None,
        moe_up_scratch: torch.Tensor | None = None,
        moe_scratch: torch.Tensor | None = None,
        eps: float = 1e-6,
        barrier_slots: torch.Tensor | None = None,
        debug_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Launch the dense Kernel B slice on the persistent megakernel."""
        empty = torch.empty(0, device=self.device)
        if barrier_slots is None:
            barrier_slots = torch.zeros(
                self.max_barrier_slots, dtype=torch.int32, device=self.device
            )
        else:
            barrier_slots.zero_()
        if debug_output is None:
            debug_output = torch.zeros(self.num_sms * 5, dtype=torch.int32, device=self.device)
        else:
            debug_output.zero_()
        if moe_scratch is not None:
            moe_scratch.zero_()

        self._module.launch_dense_b(
            builder.instructions,
            builder.counts,
            barrier_slots,
            debug_output,
            self.num_sms,
            post_attn_in,
            pre_ffn_in,
            ffn_gate_up_weight,
            ffn_down_weight,
            post_ffn1_norm_weight,
            post_ffn_norm_weight,
            layer_scalar,
            next_input_norm_weight,
            ffn_gate_scratch,
            ffn_up_scratch,
            ffn_down_scratch,
            hidden_out,
            next_attn_normed_out,
            router_proj_weight if router_proj_weight is not None else empty,
            router_scale if router_scale is not None else empty,
            router_root_size if router_root_size is not None else empty,
            pre_ffn2_norm_weight if pre_ffn2_norm_weight is not None else empty,
            router_topk_weights if router_topk_weights is not None else empty,
            router_topk_indices if router_topk_indices is not None else empty,
            moe_input_scratch if moe_input_scratch is not None else empty,
            post_ffn2_norm_weight if post_ffn2_norm_weight is not None else empty,
            moe_w13_stacked_weight if moe_w13_stacked_weight is not None else empty,
            moe_w2_weight if moe_w2_weight is not None else empty,
            moe_gate_scratch if moe_gate_scratch is not None else empty,
            moe_up_scratch if moe_up_scratch is not None else empty,
            moe_scratch if moe_scratch is not None else empty,
            eps,
        )
        return debug_output
