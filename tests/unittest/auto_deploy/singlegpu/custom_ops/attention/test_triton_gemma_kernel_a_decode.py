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

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.attention.triton_paged_attention import (
    update_paged_kv_cache,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import BatchInfo
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.triton_rms_norm import rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

_REFERENCE_SWEEP_CASES = [
    (1, 128, 64, 4, 4, 16),
    (2, 128, 64, 4, 2, 16),
    (2, 192, 64, 8, 4, 32),
    (3, 256, 64, 8, 2, 16),
]

_PREFILLED_SWEEP_CASES = [
    (2, 128, 64, 4, 4, 16),
    (2, 128, 64, 8, 4, 32),
    (3, 192, 64, 8, 2, 16),
]


def _make_decode_metadata(
    *,
    batch_size: int,
    seq_len: int,
    page_size: int,
    seq_len_with_cache: torch.Tensor | None = None,
    kv_indptr_host: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    last_page_len_host: torch.Tensor | None = None,
    triton_batch_indices: torch.Tensor | None = None,
    triton_positions: torch.Tensor | None = None,
    device: str = "cuda",
):
    batch_info = BatchInfo()
    batch_info.update([0, 0, 0, 0, batch_size, batch_size * seq_len])
    batch_info_host = batch_info.serialize()

    cu_seqlen_host = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, pin_memory=True
    )[: batch_size + 1]
    if seq_len_with_cache is None:
        seq_len_with_cache_host = torch.full(
            (batch_size,), seq_len, dtype=torch.int32, pin_memory=True
        )
    else:
        seq_len_with_cache_host = seq_len_with_cache.to(
            dtype=torch.int32, device="cpu"
        ).pin_memory()

    if kv_indptr_host is None:
        cu_num_pages_host = torch.arange(0, batch_size + 1, dtype=torch.int32, pin_memory=True)
    else:
        cu_num_pages_host = kv_indptr_host.to(dtype=torch.int32, device="cpu").pin_memory()

    if last_page_len_host is None:
        last_page_len_host = ((seq_len_with_cache_host - 1) % page_size + 1).to(dtype=torch.int32)
        if not last_page_len_host.is_pinned():
            last_page_len_host = last_page_len_host.pin_memory()

    cu_seqlen = cu_seqlen_host.to(device)
    cu_num_pages = cu_num_pages_host.to(device)
    if kv_indices is None:
        cache_loc = torch.arange(0, batch_size, dtype=torch.int32, device=device)
    else:
        cache_loc = kv_indices.to(dtype=torch.int32, device=device)
    last_page_len = last_page_len_host.to(device)

    if triton_batch_indices is None or triton_positions is None:
        triton_batch_indices, triton_positions = (
            torch.ops.auto_deploy.triton_paged_prepare_metadata.default(
                torch.zeros((batch_size, seq_len), dtype=torch.int64, device=device),
                batch_info_host,
                cu_seqlen,
                seq_len_with_cache_host.to(device),
            )
        )
    else:
        triton_batch_indices = triton_batch_indices.to(dtype=torch.int32, device=device)
        triton_positions = triton_positions.to(dtype=torch.int32, device=device)

    return (
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
    )


def _extract_sequence_from_paged_cache(
    kv_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    last_page_len: torch.Tensor,
    batch_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(kv_indptr[batch_idx].item())
    end = int(kv_indptr[batch_idx + 1].item())
    pages = kv_indices[start:end]
    tokens = []
    for page_offset, page in enumerate(pages.tolist()):
        page_tokens = kv_cache[page]
        valid_tokens = int(
            kv_cache.shape[3] if page_offset < len(pages) - 1 else last_page_len[batch_idx].item()
        )
        tokens.append(page_tokens[:, :, :valid_tokens, :])
    stacked = torch.cat(tokens, dim=2)
    return stacked[0].transpose(0, 1).contiguous(), stacked[1].transpose(0, 1).contiguous()


def _manual_kernel_a_reference(
    *,
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
    batch_info_host: torch.Tensor,
    cu_seqlen_host: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    last_page_len_host: torch.Tensor,
    seq_len_with_cache_host: torch.Tensor,
    triton_batch_indices: torch.Tensor,
    triton_positions: torch.Tensor,
    kv_cache: torch.Tensor,
):
    head_dim = q_norm_weight.numel()
    q_width = o_proj_weight.shape[1]
    kv_width = (qkv_weight.shape[0] - q_width) // 2

    if residual_in.ndim == 2:
        batch_size, seq_len = position_ids.shape
        residual_bsh = residual_in.reshape(batch_size, seq_len, residual_in.shape[-1])
        attn_normed_bsh = attn_normed_in.reshape(batch_size, seq_len, attn_normed_in.shape[-1])
    else:
        residual_bsh = residual_in
        attn_normed_bsh = attn_normed_in

    qkv = torch.ops.aten.linear(attn_normed_bsh, qkv_weight, None)
    q, k, v = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
    q = q.reshape(*q.shape[:-1], q_width // head_dim, head_dim)
    k = k.reshape(*k.shape[:-1], kv_width // head_dim, head_dim)
    v = v.reshape(*v.shape[:-1], kv_width // head_dim, head_dim)

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
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache,
        None,
        None,
    )

    o_proj = torch.ops.aten.linear(attn_out.reshape(*attn_out.shape[:-2], -1), o_proj_weight, None)
    attn_branch = rms_norm(o_proj, post_attention_layernorm_weight, eps=1e-6)
    post_attn_residual = residual_bsh + attn_branch
    pre_ffn_normed = rms_norm(post_attn_residual, pre_feedforward_layernorm_weight, eps=1e-6)

    if residual_in.ndim == 2:
        return (
            post_attn_residual.reshape_as(residual_in),
            pre_ffn_normed.reshape_as(residual_in),
        )
    return post_attn_residual, pre_ffn_normed


def _run_reference_case(
    *,
    batch_size: int,
    hidden_size: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    flatten_inputs: bool,
) -> None:
    seq_len = 1
    num_blocks = batch_size + 4
    dtype = torch.float16
    device = "cuda"

    residual_in = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    attn_normed_in = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    position_ids = torch.zeros((batch_size, seq_len), device=device, dtype=torch.int64)
    cos_sin_cache = torch.randn(64, head_dim, device=device, dtype=torch.float32)

    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    qkv_weight = torch.randn(q_width + 2 * kv_width, hidden_size, device=device, dtype=dtype)
    o_proj_weight = torch.randn(hidden_size, q_width, device=device, dtype=dtype)

    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    v_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    post_attention_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    pre_feedforward_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    metadata = _make_decode_metadata(batch_size=batch_size, seq_len=seq_len, page_size=page_size)
    (
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
    ) = metadata

    kv_cache_op = torch.zeros(
        num_blocks, 2, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
    )
    kv_cache_ref = kv_cache_op.clone()

    if flatten_inputs:
        residual_input = residual_in.reshape(batch_size * seq_len, hidden_size)
        attn_normed_input = attn_normed_in.reshape(batch_size * seq_len, hidden_size)
    else:
        residual_input = residual_in
        attn_normed_input = attn_normed_in

    post_attn_residual, pre_ffn_normed = torch.ops.auto_deploy.triton_gemma_kernel_a_decode.default(
        residual_input,
        attn_normed_input,
        qkv_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight,
        position_ids,
        cos_sin_cache,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache_op,
        None,
        None,
    )

    ref_post_attn_residual, ref_pre_ffn_normed = _manual_kernel_a_reference(
        residual_in=residual_input,
        attn_normed_in=attn_normed_input,
        qkv_weight=qkv_weight,
        o_proj_weight=o_proj_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        v_norm_weight=v_norm_weight,
        post_attention_layernorm_weight=post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight=pre_feedforward_layernorm_weight,
        position_ids=position_ids,
        cos_sin_cache=cos_sin_cache,
        batch_info_host=batch_info_host,
        cu_seqlen_host=cu_seqlen_host,
        cu_num_pages=cu_num_pages,
        cu_num_pages_host=cu_num_pages_host,
        cache_loc=cache_loc,
        last_page_len=last_page_len,
        last_page_len_host=last_page_len_host,
        seq_len_with_cache_host=seq_len_with_cache_host,
        triton_batch_indices=triton_batch_indices,
        triton_positions=triton_positions,
        kv_cache=kv_cache_ref,
    )

    torch.testing.assert_close(
        post_attn_residual.float(),
        ref_post_attn_residual.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        pre_ffn_normed.float(),
        ref_pre_ffn_normed.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(kv_cache_op.float(), kv_cache_ref.float(), rtol=0, atol=2e-7)


def _run_prefilled_sdpa_case(
    *,
    batch_size: int,
    hidden_size: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> None:
    seq_len = 1
    dtype = torch.float16
    device = "cuda"
    past_lengths = [page_size - 1 + i for i in range(batch_size - 1)] + [page_size + 1]
    past_lengths = past_lengths[:batch_size]
    total_lengths = [past_len + 1 for past_len in past_lengths]
    pages_per_seq = [(total_len + page_size - 1) // page_size for total_len in total_lengths]
    num_total_pages = sum(pages_per_seq)
    num_blocks = num_total_pages + 4

    residual_in = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    attn_normed_in = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    position_ids = torch.tensor(
        [[past_len] for past_len in past_lengths], device=device, dtype=torch.int64
    )
    cos_sin_cache = torch.randn(128, head_dim, device=device, dtype=torch.float32)

    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    qkv_weight = torch.randn(q_width + 2 * kv_width, hidden_size, device=device, dtype=dtype)
    o_proj_weight = torch.randn(hidden_size, q_width, device=device, dtype=dtype)

    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    v_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    post_attention_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    pre_feedforward_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    prefix_hidden = [
        torch.randn(past_len, hidden_size, device=device, dtype=dtype) for past_len in past_lengths
    ]
    prefix_position_ids = [
        torch.arange(past_len, device=device, dtype=torch.int64) for past_len in past_lengths
    ]

    kv_indptr_values = [0]
    for num_pages in pages_per_seq:
        kv_indptr_values.append(kv_indptr_values[-1] + num_pages)
    kv_indptr_host = torch.tensor(kv_indptr_values, dtype=torch.int32)
    kv_indices = torch.arange(num_total_pages, dtype=torch.int32, device=device)
    seq_len_with_cache = torch.tensor(total_lengths, dtype=torch.int32)
    last_page_len_host = torch.tensor(
        [((total_len - 1) % page_size) + 1 for total_len in total_lengths], dtype=torch.int32
    )
    triton_batch_indices = torch.arange(batch_size, dtype=torch.int32)
    triton_positions = torch.tensor(past_lengths, dtype=torch.int32)

    metadata = _make_decode_metadata(
        batch_size=batch_size,
        seq_len=seq_len,
        page_size=page_size,
        seq_len_with_cache=seq_len_with_cache,
        kv_indptr_host=kv_indptr_host,
        kv_indices=kv_indices,
        last_page_len_host=last_page_len_host,
        triton_batch_indices=triton_batch_indices,
        triton_positions=triton_positions,
    )
    (
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
    ) = metadata

    kv_cache = torch.zeros(
        num_blocks, 2, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
    )

    prefix_k_expected = []
    prefix_v_expected = []
    for batch_idx, (prefix_hidden_seq, prefix_pos) in enumerate(
        zip(prefix_hidden, prefix_position_ids)
    ):
        prefix_attn_normed = prefix_hidden_seq.unsqueeze(0)
        prefix_qkv = torch.ops.aten.linear(prefix_attn_normed, qkv_weight, None)
        _, prefix_k, prefix_v = torch.split(prefix_qkv, [q_width, kv_width, kv_width], dim=-1)
        prefix_k = prefix_k.reshape(1, prefix_hidden_seq.shape[0], num_kv_heads, head_dim)
        prefix_v = prefix_v.reshape(1, prefix_hidden_seq.shape[0], num_kv_heads, head_dim)
        prefix_k = rms_norm(prefix_k, k_norm_weight, eps=1e-6)
        prefix_v = rms_norm(prefix_v, v_norm_weight, eps=1e-6)
        _, prefix_k = torch.ops.auto_deploy.flashinfer_rope.default(
            torch.zeros(
                1, prefix_hidden_seq.shape[0], num_q_heads, head_dim, device=device, dtype=dtype
            ),
            prefix_k,
            prefix_pos.view(1, -1),
            cos_sin_cache,
            True,
        )
        prefix_k_expected.append(prefix_k.squeeze(0))
        prefix_v_expected.append(prefix_v.squeeze(0))

        update_paged_kv_cache(
            prefix_k.squeeze(0),
            prefix_v.squeeze(0),
            torch.full((prefix_hidden_seq.shape[0],), batch_idx, dtype=torch.int32, device=device),
            prefix_pos.to(dtype=torch.int32),
            kv_cache,
            cache_loc,
            cu_num_pages,
        )

    post_attn_residual, pre_ffn_normed = torch.ops.auto_deploy.triton_gemma_kernel_a_decode.default(
        residual_in,
        attn_normed_in,
        qkv_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight,
        position_ids,
        cos_sin_cache,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache,
        None,
        None,
    )

    current_qkv = torch.ops.aten.linear(attn_normed_in.unsqueeze(1), qkv_weight, None)
    current_q, current_k, current_v = torch.split(
        current_qkv, [q_width, kv_width, kv_width], dim=-1
    )
    current_q = current_q.reshape(batch_size, 1, num_q_heads, head_dim)
    current_k = current_k.reshape(batch_size, 1, num_kv_heads, head_dim)
    current_v = current_v.reshape(batch_size, 1, num_kv_heads, head_dim)

    current_q = rms_norm(current_q, q_norm_weight, eps=1e-6)
    current_k = rms_norm(current_k, k_norm_weight, eps=1e-6)
    current_v = rms_norm(current_v, v_norm_weight, eps=1e-6)
    current_q, current_k = torch.ops.auto_deploy.flashinfer_rope.default(
        current_q, current_k, position_ids, cos_sin_cache, True
    )

    expected_attn = []
    for batch_idx in range(batch_size):
        full_k = (
            torch.cat([prefix_k_expected[batch_idx], current_k[batch_idx]], dim=0)
            .transpose(0, 1)
            .contiguous()
        )
        full_v = (
            torch.cat([prefix_v_expected[batch_idx], current_v[batch_idx]], dim=0)
            .transpose(0, 1)
            .contiguous()
        )

        q_ref = current_q[batch_idx].transpose(0, 1).unsqueeze(0)
        k_ref = full_k.unsqueeze(0)
        v_ref = full_v.unsqueeze(0)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_ref,
            k_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
            v_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
            scale=1.0 / (head_dim**0.5),
            is_causal=False,
        )
        expected_attn.append(attn.squeeze(0).transpose(0, 1))

    expected_attn = torch.stack(expected_attn, dim=0)
    o_proj = torch.ops.aten.linear(expected_attn.reshape(batch_size, -1), o_proj_weight, None)
    attn_branch = rms_norm(o_proj, post_attention_layernorm_weight, eps=1e-6)
    expected_post_attn_residual = residual_in + attn_branch
    expected_pre_ffn_normed = rms_norm(
        expected_post_attn_residual, pre_feedforward_layernorm_weight, eps=1e-6
    )

    torch.testing.assert_close(
        post_attn_residual.float(), expected_post_attn_residual.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pre_ffn_normed.float(), expected_pre_ffn_normed.float(), rtol=2e-2, atol=2e-2
    )

    for batch_idx, total_length in enumerate(total_lengths):
        cached_k, cached_v = _extract_sequence_from_paged_cache(
            kv_cache, cache_loc, cu_num_pages, last_page_len, batch_idx
        )
        expected_k = torch.cat([prefix_k_expected[batch_idx], current_k[batch_idx]], dim=0)
        expected_v = torch.cat([prefix_v_expected[batch_idx], current_v[batch_idx]], dim=0)
        torch.testing.assert_close(
            cached_k[:total_length].float(), expected_k[:total_length].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            cached_v[:total_length].float(),
            expected_v[:total_length].float(),
            rtol=0,
            atol=2e-7,
        )


@pytest.mark.parametrize("flatten_inputs", [False, True])
def test_triton_gemma_kernel_a_decode_matches_reference(flatten_inputs: bool):
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 1
    hidden_size = 128
    head_dim = 64
    num_q_heads = 4
    num_kv_heads = 2
    page_size = 16
    num_blocks = batch_size + 4
    dtype = torch.float16
    device = "cuda"

    residual_in = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    attn_normed_in = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    position_ids = torch.zeros((batch_size, seq_len), device=device, dtype=torch.int64)
    cos_sin_cache = torch.randn(32, head_dim, device=device, dtype=torch.float32)

    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    qkv_weight = torch.randn(q_width + 2 * kv_width, hidden_size, device=device, dtype=dtype)
    o_proj_weight = torch.randn(hidden_size, q_width, device=device, dtype=dtype)

    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    v_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    post_attention_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    pre_feedforward_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    metadata = _make_decode_metadata(batch_size=batch_size, seq_len=seq_len, page_size=page_size)
    (
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
    ) = metadata

    kv_cache_op = torch.zeros(
        num_blocks, 2, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
    )
    kv_cache_ref = kv_cache_op.clone()

    if flatten_inputs:
        residual_input = residual_in.reshape(batch_size * seq_len, hidden_size)
        attn_normed_input = attn_normed_in.reshape(batch_size * seq_len, hidden_size)
    else:
        residual_input = residual_in
        attn_normed_input = attn_normed_in

    post_attn_residual, pre_ffn_normed = torch.ops.auto_deploy.triton_gemma_kernel_a_decode.default(
        residual_input,
        attn_normed_input,
        qkv_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight,
        position_ids,
        cos_sin_cache,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache_op,
        None,
        None,
    )

    ref_post_attn_residual, ref_pre_ffn_normed = _manual_kernel_a_reference(
        residual_in=residual_input,
        attn_normed_in=attn_normed_input,
        qkv_weight=qkv_weight,
        o_proj_weight=o_proj_weight,
        q_norm_weight=q_norm_weight,
        k_norm_weight=k_norm_weight,
        v_norm_weight=v_norm_weight,
        post_attention_layernorm_weight=post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight=pre_feedforward_layernorm_weight,
        position_ids=position_ids,
        cos_sin_cache=cos_sin_cache,
        batch_info_host=batch_info_host,
        cu_seqlen_host=cu_seqlen_host,
        cu_num_pages=cu_num_pages,
        cu_num_pages_host=cu_num_pages_host,
        cache_loc=cache_loc,
        last_page_len=last_page_len,
        last_page_len_host=last_page_len_host,
        seq_len_with_cache_host=seq_len_with_cache_host,
        triton_batch_indices=triton_batch_indices,
        triton_positions=triton_positions,
        kv_cache=kv_cache_ref,
    )

    torch.testing.assert_close(
        post_attn_residual.float(),
        ref_post_attn_residual.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        pre_ffn_normed.float(),
        ref_pre_ffn_normed.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(kv_cache_op.float(), kv_cache_ref.float(), rtol=0, atol=2e-7)


def test_triton_gemma_kernel_a_decode_matches_sdpa_with_prefilled_cache():
    torch.manual_seed(1)

    batch_size = 2
    seq_len = 1
    hidden_size = 128
    head_dim = 64
    num_q_heads = 4
    num_kv_heads = 2
    page_size = 16
    past_lengths = [3, 17]
    total_lengths = [past_len + 1 for past_len in past_lengths]
    num_blocks = 8
    dtype = torch.float16
    device = "cuda"

    residual_in = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    attn_normed_in = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    position_ids = torch.tensor(
        [[past_lengths[0]], [past_lengths[1]]], device=device, dtype=torch.int64
    )
    cos_sin_cache = torch.randn(32, head_dim, device=device, dtype=torch.float32)

    q_width = num_q_heads * head_dim
    kv_width = num_kv_heads * head_dim
    qkv_weight = torch.randn(q_width + 2 * kv_width, hidden_size, device=device, dtype=dtype)
    o_proj_weight = torch.randn(hidden_size, q_width, device=device, dtype=dtype)

    q_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    k_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    v_norm_weight = torch.randn(head_dim, device=device, dtype=torch.float32)
    post_attention_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    pre_feedforward_layernorm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    prefix_hidden = [
        torch.randn(past_len, hidden_size, device=device, dtype=dtype) for past_len in past_lengths
    ]
    prefix_position_ids = [
        torch.arange(past_len, device=device, dtype=torch.int64) for past_len in past_lengths
    ]

    kv_indptr_host = torch.tensor([0, 1, 3], dtype=torch.int32)
    kv_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    seq_len_with_cache = torch.tensor(total_lengths, dtype=torch.int32)
    last_page_len_host = torch.tensor([4, 2], dtype=torch.int32)
    triton_batch_indices = torch.arange(batch_size, dtype=torch.int32)
    triton_positions = torch.tensor(past_lengths, dtype=torch.int32)

    metadata = _make_decode_metadata(
        batch_size=batch_size,
        seq_len=seq_len,
        page_size=page_size,
        seq_len_with_cache=seq_len_with_cache,
        kv_indptr_host=kv_indptr_host,
        kv_indices=kv_indices,
        last_page_len_host=last_page_len_host,
        triton_batch_indices=triton_batch_indices,
        triton_positions=triton_positions,
    )
    (
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
    ) = metadata

    kv_cache = torch.zeros(
        num_blocks, 2, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
    )

    prefix_k_expected = []
    prefix_v_expected = []
    for batch_idx, (prefix_hidden_seq, prefix_pos) in enumerate(
        zip(prefix_hidden, prefix_position_ids)
    ):
        prefix_attn_normed = prefix_hidden_seq.unsqueeze(0)
        prefix_qkv = torch.ops.aten.linear(prefix_attn_normed, qkv_weight, None)
        _, prefix_k, prefix_v = torch.split(prefix_qkv, [q_width, kv_width, kv_width], dim=-1)
        prefix_k = prefix_k.reshape(1, prefix_hidden_seq.shape[0], num_kv_heads, head_dim)
        prefix_v = prefix_v.reshape(1, prefix_hidden_seq.shape[0], num_kv_heads, head_dim)
        prefix_k = rms_norm(prefix_k, k_norm_weight, eps=1e-6)
        prefix_v = rms_norm(prefix_v, v_norm_weight, eps=1e-6)
        _, prefix_k = torch.ops.auto_deploy.flashinfer_rope.default(
            torch.zeros(
                1, prefix_hidden_seq.shape[0], num_q_heads, head_dim, device=device, dtype=dtype
            ),
            prefix_k,
            prefix_pos.view(1, -1),
            cos_sin_cache,
            True,
        )
        prefix_k_expected.append(prefix_k.squeeze(0))
        prefix_v_expected.append(prefix_v.squeeze(0))

        update_paged_kv_cache(
            prefix_k.squeeze(0),
            prefix_v.squeeze(0),
            torch.full((prefix_hidden_seq.shape[0],), batch_idx, dtype=torch.int32, device=device),
            prefix_pos.to(dtype=torch.int32),
            kv_cache,
            cache_loc,
            cu_num_pages,
        )

    post_attn_residual, pre_ffn_normed = torch.ops.auto_deploy.triton_gemma_kernel_a_decode.default(
        residual_in,
        attn_normed_in,
        qkv_weight,
        o_proj_weight,
        q_norm_weight,
        k_norm_weight,
        v_norm_weight,
        post_attention_layernorm_weight,
        pre_feedforward_layernorm_weight,
        position_ids,
        cos_sin_cache,
        batch_info_host,
        cu_seqlen_host,
        cu_num_pages,
        cu_num_pages_host,
        cache_loc,
        last_page_len,
        last_page_len_host,
        seq_len_with_cache_host,
        triton_batch_indices,
        triton_positions,
        kv_cache,
        None,
        None,
    )

    current_qkv = torch.ops.aten.linear(attn_normed_in.unsqueeze(1), qkv_weight, None)
    current_q, current_k, current_v = torch.split(
        current_qkv, [q_width, kv_width, kv_width], dim=-1
    )
    current_q = current_q.reshape(batch_size, 1, num_q_heads, head_dim)
    current_k = current_k.reshape(batch_size, 1, num_kv_heads, head_dim)
    current_v = current_v.reshape(batch_size, 1, num_kv_heads, head_dim)

    current_q = rms_norm(current_q, q_norm_weight, eps=1e-6)
    current_k = rms_norm(current_k, k_norm_weight, eps=1e-6)
    current_v = rms_norm(current_v, v_norm_weight, eps=1e-6)
    current_q, current_k = torch.ops.auto_deploy.flashinfer_rope.default(
        current_q, current_k, position_ids, cos_sin_cache, True
    )

    expected_attn = []
    for batch_idx in range(batch_size):
        full_k = (
            torch.cat([prefix_k_expected[batch_idx], current_k[batch_idx]], dim=0)
            .transpose(0, 1)
            .contiguous()
        )
        full_v = (
            torch.cat([prefix_v_expected[batch_idx], current_v[batch_idx]], dim=0)
            .transpose(0, 1)
            .contiguous()
        )

        q_ref = current_q[batch_idx].transpose(0, 1).unsqueeze(0)
        k_ref = full_k.unsqueeze(0)
        v_ref = full_v.unsqueeze(0)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q_ref,
            k_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
            v_ref.repeat_interleave(num_q_heads // num_kv_heads, dim=1),
            scale=1.0 / (head_dim**0.5),
            is_causal=False,
        )
        expected_attn.append(attn.squeeze(0).transpose(0, 1))

    expected_attn = torch.stack(expected_attn, dim=0)
    o_proj = torch.ops.aten.linear(expected_attn.reshape(batch_size, -1), o_proj_weight, None)
    attn_branch = rms_norm(o_proj, post_attention_layernorm_weight, eps=1e-6)
    expected_post_attn_residual = residual_in + attn_branch
    expected_pre_ffn_normed = rms_norm(
        expected_post_attn_residual, pre_feedforward_layernorm_weight, eps=1e-6
    )

    torch.testing.assert_close(
        post_attn_residual.float(), expected_post_attn_residual.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pre_ffn_normed.float(), expected_pre_ffn_normed.float(), rtol=2e-2, atol=2e-2
    )

    for batch_idx, total_length in enumerate(total_lengths):
        cached_k, cached_v = _extract_sequence_from_paged_cache(
            kv_cache, cache_loc, cu_num_pages, last_page_len, batch_idx
        )
        expected_k = torch.cat([prefix_k_expected[batch_idx], current_k[batch_idx]], dim=0)
        expected_v = torch.cat([prefix_v_expected[batch_idx], current_v[batch_idx]], dim=0)
        torch.testing.assert_close(
            cached_k[:total_length].float(), expected_k[:total_length].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            cached_v[:total_length].float(),
            expected_v[:total_length].float(),
            rtol=0,
            atol=2e-7,
        )


@pytest.mark.parametrize(
    "batch_size,hidden_size,head_dim,num_q_heads,num_kv_heads,page_size", _REFERENCE_SWEEP_CASES
)
@pytest.mark.parametrize("flatten_inputs", [False, True])
def test_triton_gemma_kernel_a_decode_reference_shape_sweep(
    batch_size: int,
    hidden_size: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    flatten_inputs: bool,
):
    torch.manual_seed(
        1000
        + batch_size
        + hidden_size
        + head_dim
        + num_q_heads * 10
        + num_kv_heads * 100
        + page_size
    )
    _run_reference_case(
        batch_size=batch_size,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        flatten_inputs=flatten_inputs,
    )


@pytest.mark.parametrize(
    "batch_size,hidden_size,head_dim,num_q_heads,num_kv_heads,page_size", _PREFILLED_SWEEP_CASES
)
def test_triton_gemma_kernel_a_decode_prefilled_shape_sweep(
    batch_size: int,
    hidden_size: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
):
    torch.manual_seed(
        2000
        + batch_size
        + hidden_size
        + head_dim
        + num_q_heads * 10
        + num_kv_heads * 100
        + page_size
    )
    _run_prefilled_sdpa_case(
        batch_size=batch_size,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
    )
