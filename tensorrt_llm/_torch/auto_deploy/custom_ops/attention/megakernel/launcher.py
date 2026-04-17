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


def load_megakernel_module() -> Any:
    """JIT-compile and load the megakernel CUDA extension."""
    if "gemma4_megakernel" in _MODULE_CACHE:
        return _MODULE_CACHE["gemma4_megakernel"]

    from torch.utils.cpp_extension import load

    src_dir = _source_dir()
    build_dir = src_dir / "_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    module = load(
        name="gemma4_megakernel",
        sources=[str(src_dir / "megakernel_host.cu")],
        extra_include_paths=[str(src_dir)],
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


class InstructionBuilder:
    """Build per-SM instruction queues for the megakernel.

    Instructions are 32 x int32 (128 bytes each).
    The builder maintains a [num_sms, max_instructions, 32] tensor
    and a per-SM instruction counter.
    """

    def __init__(self, num_sms: int, max_instructions: int = 64, device: str = "cuda"):
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

    def add_noop(self, sm_ids: list[int] | range) -> None:
        """Add a no-op instruction to specified SMs."""
        for sm_id in sm_ids:
            self._add_instruction(sm_id, [0])  # OP_NOOP = 0

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
        self._add_instruction(sm_id, [7, token_id])  # OP_OPROJ_POST = 7

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
    ) -> torch.Tensor:
        """Launch the megakernel with Gemma4 globals."""
        import math

        if attn_scale == 0.0:
            attn_scale = 1.0 / math.sqrt(256)  # HEAD_DIM

        empty = torch.empty(0, device=self.device)
        barrier_slots = torch.zeros(
            self.max_barrier_slots,
            dtype=torch.int32,
            device=self.device,
        )
        debug_output = torch.zeros(self.num_sms, dtype=torch.int32, device=self.device)
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
