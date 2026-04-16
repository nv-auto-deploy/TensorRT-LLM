#!/usr/bin/env python3
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

"""Benchmark the current Triton Gemma Kernel A decode op.

This benchmark compares:
- the packaged ``auto_deploy::triton_gemma_kernel_a_decode`` op
- the equivalent explicit decomposition using the same underlying building blocks

The decomposition baseline gives us the current "no micro-pipeline" reference.
Once the op internals are replaced with a true micro-pipeline, we can rerun this
script and compare against the same baseline numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
    update_paged_kv_cache,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.triton_rms_norm import rms_norm
from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    batch_size: int
    past_length: int
    page_size: int


def make_decode_metadata(
    *,
    batch_size: int,
    seq_len: int,
    page_size: int,
    seq_len_with_cache: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_indices: torch.Tensor,
    last_page_len_host: torch.Tensor,
    triton_batch_indices: torch.Tensor,
    triton_positions: torch.Tensor,
    device: str = "cuda",
):
    batch_info = BatchInfo()
    batch_info.update([0, 0, 0, 0, batch_size, batch_size * seq_len])
    batch_info_host = batch_info.serialize()

    cu_seqlen_host = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, pin_memory=prefer_pinned()
    )[: batch_size + 1]
    cu_num_pages_host = maybe_pin_memory(kv_indptr_host.to(dtype=torch.int32, device="cpu"))

    cu_num_pages = cu_num_pages_host.to(device)
    cache_loc = kv_indices.to(dtype=torch.int32, device=device)
    last_page_len = last_page_len_host.to(dtype=torch.int32, device=device)
    triton_batch_indices = triton_batch_indices.to(dtype=torch.int32, device=device)
    triton_positions = triton_positions.to(dtype=torch.int32, device=device)

    return {
        "batch_info_host": batch_info_host,
        "cu_seqlen_host": cu_seqlen_host,
        "cu_num_pages": cu_num_pages,
        "cu_num_pages_host": cu_num_pages_host,
        "cache_loc": cache_loc,
        "last_page_len": last_page_len,
        "last_page_len_host": maybe_pin_memory(
            last_page_len_host.to(dtype=torch.int32, device="cpu")
        ),
        "seq_len_with_cache_host": maybe_pin_memory(
            seq_len_with_cache.to(dtype=torch.int32, device="cpu")
        ),
        "triton_batch_indices": triton_batch_indices,
        "triton_positions": triton_positions,
    }


def split_qkv(qkv: torch.Tensor, o_proj_weight: torch.Tensor, head_dim: int):
    q_width = int(o_proj_weight.shape[1])
    kv_width = (int(qkv.shape[-1]) - q_width) // 2
    q, k, v = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
    q = q.reshape(*q.shape[:-1], q_width // head_dim, head_dim)
    k = k.reshape(*k.shape[:-1], kv_width // head_dim, head_dim)
    v = v.reshape(*v.shape[:-1], kv_width // head_dim, head_dim)
    return q, k, v


def explicit_kernel_a(
    residual_in: torch.Tensor,
    attn_normed_in: torch.Tensor,
    qkv_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    v_norm_weight: torch.Tensor,
    post_attention_layernorm_weight: torch.Tensor,
    pre_feedforward_layernorm_weight: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    metadata: dict,
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = int(q_norm_weight.numel())
    attn_normed_bsh = attn_normed_in.unsqueeze(1)

    qkv = torch.ops.aten.linear(attn_normed_bsh, qkv_weight, None)
    q, k, v = split_qkv(qkv, o_proj_weight, head_dim)

    q_normed = rms_norm(q, q_norm_weight, eps=1e-6)
    k_normed = rms_norm(k, k_norm_weight, eps=1e-6)
    v_normed = rms_norm(v, v_norm_weight, eps=1e-6)
    q_rope, k_rope = torch.ops.auto_deploy.flashinfer_rope.default(
        q_normed, k_normed, position_ids, cos_sin_cache, True
    )
    attn_out = torch.ops.auto_deploy.triton_paged_mha_with_cache.default(
        q_rope,
        k_rope,
        v_normed,
        metadata["batch_info_host"],
        metadata["cu_seqlen_host"],
        metadata["cu_num_pages"],
        metadata["cu_num_pages_host"],
        metadata["cache_loc"],
        metadata["last_page_len"],
        metadata["last_page_len_host"],
        metadata["seq_len_with_cache_host"],
        metadata["triton_batch_indices"],
        metadata["triton_positions"],
        kv_cache,
        None,
        None,
    )
    o_proj = torch.ops.aten.linear(attn_out.reshape(attn_out.shape[0], -1), o_proj_weight, None)
    attn_branch = rms_norm(o_proj, post_attention_layernorm_weight, eps=1e-6)
    post_attn_residual = residual_in + attn_branch
    pre_ffn_normed = rms_norm(post_attn_residual, pre_feedforward_layernorm_weight, eps=1e-6)
    return post_attn_residual, pre_ffn_normed


def benchmark_fn(fn: Callable[[], object], *, warmup: int = 20, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(start_events, end_events)]
    times_us.sort()
    return times_us[len(times_us) // 2]


def build_inputs(case: BenchmarkCase) -> dict:
    hidden_size = 2816
    head_dim = 256
    num_q_heads = 16
    num_kv_heads = 8
    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    dtype = torch.bfloat16
    device = "cuda"

    torch.manual_seed(1234 + case.batch_size * 31 + case.past_length * 7 + case.page_size)

    residual_in = torch.randn(case.batch_size, hidden_size, device=device, dtype=dtype)
    attn_normed_in = torch.randn(case.batch_size, hidden_size, device=device, dtype=dtype)
    position_ids = torch.tensor(
        [[case.past_length] for _ in range(case.batch_size)], device=device, dtype=torch.int64
    )
    cos_sin_cache = torch.randn(4096, head_dim, device=device, dtype=torch.float32)

    qkv_weight = torch.randn(q_width + 2 * kv_width, hidden_size, device=device, dtype=dtype)
    o_proj_weight = torch.randn(hidden_size, q_width, device=device, dtype=dtype)
    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    v_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    post_attention_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    pre_feedforward_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    past_lengths = [case.past_length for _ in range(case.batch_size)]
    total_lengths = [past + 1 for past in past_lengths]
    pages_per_seq = [(total + case.page_size - 1) // case.page_size for total in total_lengths]
    num_total_pages = sum(pages_per_seq)
    kv_indptr_values = [0]
    for num_pages in pages_per_seq:
        kv_indptr_values.append(kv_indptr_values[-1] + num_pages)
    kv_indptr_host = torch.tensor(kv_indptr_values, dtype=torch.int32)
    kv_indices = torch.arange(num_total_pages, dtype=torch.int32, device=device)
    seq_len_with_cache = torch.tensor(total_lengths, dtype=torch.int32)
    last_page_len_host = torch.tensor(
        [((total - 1) % case.page_size) + 1 for total in total_lengths], dtype=torch.int32
    )
    triton_batch_indices = torch.arange(case.batch_size, dtype=torch.int32)
    triton_positions = torch.tensor(past_lengths, dtype=torch.int32)

    metadata = make_decode_metadata(
        batch_size=case.batch_size,
        seq_len=1,
        page_size=case.page_size,
        seq_len_with_cache=seq_len_with_cache,
        kv_indptr_host=kv_indptr_host,
        kv_indices=kv_indices,
        last_page_len_host=last_page_len_host,
        triton_batch_indices=triton_batch_indices,
        triton_positions=triton_positions,
    )

    num_blocks = num_total_pages + 8
    kv_cache_template = torch.zeros(
        num_blocks, 2, num_kv_heads, case.page_size, head_dim, device=device, dtype=dtype
    )

    for batch_idx in range(case.batch_size):
        prefix_hidden = torch.randn(case.past_length, hidden_size, device=device, dtype=dtype)
        prefix_pos = torch.arange(case.past_length, device=device, dtype=torch.int64)
        prefix_qkv = torch.ops.aten.linear(prefix_hidden.unsqueeze(0), qkv_weight, None)
        _, prefix_k, prefix_v = torch.split(prefix_qkv, [q_width, kv_width, kv_width], dim=-1)
        prefix_k = prefix_k.reshape(1, case.past_length, num_kv_heads, head_dim)
        prefix_v = prefix_v.reshape(1, case.past_length, num_kv_heads, head_dim)
        prefix_k = rms_norm(prefix_k, k_norm_weight, eps=1e-6)
        prefix_v = rms_norm(prefix_v, v_norm_weight, eps=1e-6)
        _, prefix_k = torch.ops.auto_deploy.flashinfer_rope.default(
            torch.zeros(1, case.past_length, num_q_heads, head_dim, device=device, dtype=dtype),
            prefix_k,
            prefix_pos.view(1, -1),
            cos_sin_cache,
            True,
        )
        update_paged_kv_cache(
            prefix_k.squeeze(0),
            prefix_v.squeeze(0),
            torch.full((case.past_length,), batch_idx, dtype=torch.int32, device=device),
            prefix_pos.to(dtype=torch.int32),
            kv_cache_template,
            metadata["cache_loc"],
            metadata["cu_num_pages"],
        )

    return {
        "residual_in": residual_in,
        "attn_normed_in": attn_normed_in,
        "qkv_weight": qkv_weight,
        "o_proj_weight": o_proj_weight,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "v_norm_weight": v_norm_weight,
        "post_attention_layernorm_weight": post_attention_layernorm_weight,
        "pre_feedforward_layernorm_weight": pre_feedforward_layernorm_weight,
        "position_ids": position_ids,
        "cos_sin_cache": cos_sin_cache,
        "metadata": metadata,
        "kv_cache_template": kv_cache_template,
    }


def run_case(case: BenchmarkCase) -> None:
    inputs = build_inputs(case)
    kv_cache_op = inputs["kv_cache_template"].clone()
    kv_cache_decomp = inputs["kv_cache_template"].clone()

    ref_post, ref_pre = explicit_kernel_a(
        inputs["residual_in"],
        inputs["attn_normed_in"],
        inputs["qkv_weight"],
        inputs["o_proj_weight"],
        inputs["q_norm_weight"],
        inputs["k_norm_weight"],
        inputs["v_norm_weight"],
        inputs["post_attention_layernorm_weight"],
        inputs["pre_feedforward_layernorm_weight"],
        inputs["position_ids"],
        inputs["cos_sin_cache"],
        inputs["metadata"],
        kv_cache_decomp,
    )
    op_post, op_pre = torch.ops.auto_deploy.triton_gemma_kernel_a_decode.default(
        inputs["residual_in"],
        inputs["attn_normed_in"],
        inputs["qkv_weight"],
        inputs["o_proj_weight"],
        inputs["q_norm_weight"],
        inputs["k_norm_weight"],
        inputs["v_norm_weight"],
        inputs["post_attention_layernorm_weight"],
        inputs["pre_feedforward_layernorm_weight"],
        inputs["position_ids"],
        inputs["cos_sin_cache"],
        inputs["metadata"]["batch_info_host"],
        inputs["metadata"]["cu_seqlen_host"],
        inputs["metadata"]["cu_num_pages"],
        inputs["metadata"]["cu_num_pages_host"],
        inputs["metadata"]["cache_loc"],
        inputs["metadata"]["last_page_len"],
        inputs["metadata"]["last_page_len_host"],
        inputs["metadata"]["seq_len_with_cache_host"],
        inputs["metadata"]["triton_batch_indices"],
        inputs["metadata"]["triton_positions"],
        kv_cache_op,
        None,
        None,
    )
    torch.testing.assert_close(op_post.float(), ref_post.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(op_pre.float(), ref_pre.float(), rtol=1e-2, atol=1e-2)

    def run_op():
        return torch.ops.auto_deploy.triton_gemma_kernel_a_decode.default(
            inputs["residual_in"],
            inputs["attn_normed_in"],
            inputs["qkv_weight"],
            inputs["o_proj_weight"],
            inputs["q_norm_weight"],
            inputs["k_norm_weight"],
            inputs["v_norm_weight"],
            inputs["post_attention_layernorm_weight"],
            inputs["pre_feedforward_layernorm_weight"],
            inputs["position_ids"],
            inputs["cos_sin_cache"],
            inputs["metadata"]["batch_info_host"],
            inputs["metadata"]["cu_seqlen_host"],
            inputs["metadata"]["cu_num_pages"],
            inputs["metadata"]["cu_num_pages_host"],
            inputs["metadata"]["cache_loc"],
            inputs["metadata"]["last_page_len"],
            inputs["metadata"]["last_page_len_host"],
            inputs["metadata"]["seq_len_with_cache_host"],
            inputs["metadata"]["triton_batch_indices"],
            inputs["metadata"]["triton_positions"],
            kv_cache_op,
            None,
            None,
        )

    def run_decomposition():
        return explicit_kernel_a(
            inputs["residual_in"],
            inputs["attn_normed_in"],
            inputs["qkv_weight"],
            inputs["o_proj_weight"],
            inputs["q_norm_weight"],
            inputs["k_norm_weight"],
            inputs["v_norm_weight"],
            inputs["post_attention_layernorm_weight"],
            inputs["pre_feedforward_layernorm_weight"],
            inputs["position_ids"],
            inputs["cos_sin_cache"],
            inputs["metadata"],
            kv_cache_decomp,
        )

    op_us = benchmark_fn(run_op)
    decomp_us = benchmark_fn(run_decomposition)
    ratio = decomp_us / op_us if op_us > 0 else float("inf")
    print(
        f"{case.name:>18s}  bs={case.batch_size:<2d}  past={case.past_length:<4d}  "
        f"page={case.page_size:<2d}  op={op_us:>9.2f}us  "
        f"decomp={decomp_us:>9.2f}us  decomp/op={ratio:>6.3f}x"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cases = [
        BenchmarkCase(name="gemma_decode_b1", batch_size=1, past_length=128, page_size=16),
        BenchmarkCase(name="gemma_decode_b4", batch_size=4, past_length=128, page_size=16),
        BenchmarkCase(name="gemma_decode_b1_l", batch_size=1, past_length=1024, page_size=16),
        BenchmarkCase(name="gemma_decode_b4_l", batch_size=4, past_length=1024, page_size=16),
        BenchmarkCase(name="gemma_decode_b1_p32", batch_size=1, past_length=1024, page_size=32),
        BenchmarkCase(name="gemma_decode_b4_p32", batch_size=4, past_length=1024, page_size=32),
        BenchmarkCase(name="gemma_decode_b1_p64", batch_size=1, past_length=1024, page_size=64),
        BenchmarkCase(name="gemma_decode_b4_p64", batch_size=4, past_length=1024, page_size=64),
    ]

    print("=" * 96)
    print("Triton Gemma Kernel A Decode Benchmark")
    print("Shapes: hidden=2816, q_heads=16, kv_heads=8, head_dim=256, dtype=bf16")
    print("=" * 96)
    print(
        f"{'case':>18s}  {'batch':>6s}  {'past':>8s}  {'page':>6s}  "
        f"{'kernel_a_op':>14s}  {'decomposition':>14s}  {'decomp/op':>10s}"
    )
    print("-" * 96)
    for case in cases:
        run_case(case)
    print("=" * 96)


if __name__ == "__main__":
    main()
