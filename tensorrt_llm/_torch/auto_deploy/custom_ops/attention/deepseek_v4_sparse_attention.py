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

"""DeepSeek V4 sparse attention source and cached reference ops."""

from typing import List, NamedTuple, Optional, Tuple

import torch
from torch._ops import OpOverloadPacket
from torch._subclasses import FakeTensor
from torch.fx import Node

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton always present on CUDA builds
    _HAS_TRITON = False

from ..._compat import KvCacheConfig
from ...utils.node_utils import extract_op_args, is_op
from ...utils.quantization_utils import fake_fp8_act_quant as _fake_fp8_act_quant
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    BatchInfo,
    Constant,
    MHACallable,
    PagedResourceHandler,
    PrepareMetadataCallable,
    ResourceHandlerDict,
)

__all__ = [
    "DeepSeekV4SparseAttention",
    "torch_deepseek_v4_sparse_attention",
    "torch_deepseek_v4_sparse_attention_with_cache",
]

_SPARSE_ATTENTION_CHUNK_TARGET_BYTES = 512 * 1024 * 1024
_SPARSE_ATTENTION_MAX_CHUNK_TOKENS = 64
_COMPRESS_RATIO_DISABLED = 0
_COMPRESS_RATIO_OVERLAP_INDEXER = 4
_COMPRESS_RATIO_DENSE = 128


class _CompressionMode(NamedTuple):
    ratio: int
    enabled: bool
    overlap: bool
    uses_indexer: bool
    channels: int


_COMPRESSION_MODES = {
    _COMPRESS_RATIO_DISABLED: _CompressionMode(
        ratio=_COMPRESS_RATIO_DISABLED,
        enabled=False,
        overlap=False,
        uses_indexer=False,
        channels=1,
    ),
    _COMPRESS_RATIO_OVERLAP_INDEXER: _CompressionMode(
        ratio=_COMPRESS_RATIO_OVERLAP_INDEXER,
        enabled=True,
        overlap=True,
        uses_indexer=True,
        channels=2,
    ),
    _COMPRESS_RATIO_DENSE: _CompressionMode(
        ratio=_COMPRESS_RATIO_DENSE,
        enabled=True,
        overlap=False,
        uses_indexer=False,
        channels=1,
    ),
}
_SUPPORTED_COMPRESS_RATIOS = tuple(_COMPRESSION_MODES)
_SOURCE_TENSOR_ARG_NAMES = (
    "q",
    "kv",
    "attn_sink",
    "topk_idxs",
    "compressor_kv",
    "compressor_gate",
    "compressor_ape",
    "compressor_norm_weight",
    "cos_table",
    "sin_table",
    "position_ids",
    "indexer_q",
    "indexer_weights",
    "indexer_compressor_kv",
    "indexer_compressor_gate",
    "indexer_compressor_ape",
    "indexer_compressor_norm_weight",
)


def _validate_rank(name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(f"{name} must have rank {rank}, got rank {tensor.dim()}")


def _compression_mode(compress_ratio: int) -> _CompressionMode:
    mode = _COMPRESSION_MODES.get(compress_ratio)
    if mode is None:
        raise ValueError(
            "DeepSeek V4 cached sparse attention supports "
            f"compress_ratio in {_SUPPORTED_COMPRESS_RATIOS}, got {compress_ratio}"
        )
    return mode


def _validate_compress_ratio(compress_ratio: int) -> None:
    _compression_mode(compress_ratio)


def _validate_deepseek_v4_sparse_attention_inputs(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
) -> None:
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)

    if not q.is_floating_point():
        raise TypeError(f"q must be floating point, got {q.dtype}")
    if not kv.is_floating_point():
        raise TypeError(f"kv must be floating point, got {kv.dtype}")
    if not attn_sink.is_floating_point():
        raise TypeError(f"attn_sink must be floating point, got {attn_sink.dtype}")
    if q.dtype != kv.dtype:
        raise TypeError(f"q and kv must have the same dtype, got {q.dtype} and {kv.dtype}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_idxs must be int32 or int64, got {topk_idxs.dtype}")

    if kv.device != q.device:
        raise ValueError(f"kv must be on {q.device}, got {kv.device}")
    if attn_sink.device != q.device:
        raise ValueError(f"attn_sink must be on {q.device}, got {attn_sink.device}")
    if topk_idxs.device != q.device:
        raise ValueError(f"topk_idxs must be on {q.device}, got {topk_idxs.device}")

    batch_size, seq_len, num_heads, head_dim = q.shape
    kv_batch_size, _, kv_head_dim = kv.shape
    topk_batch_size, topk_seq_len, _ = topk_idxs.shape

    if kv_batch_size != batch_size:
        raise ValueError(f"kv batch dimension must be {batch_size}, got {kv_batch_size}")
    if topk_batch_size != batch_size:
        raise ValueError(f"topk_idxs batch dimension must be {batch_size}, got {topk_batch_size}")
    if topk_seq_len != seq_len:
        raise ValueError(f"topk_idxs sequence dimension must be {seq_len}, got {topk_seq_len}")
    if kv_head_dim != head_dim:
        raise ValueError(f"kv head dimension must be {head_dim}, got {kv_head_dim}")
    if attn_sink.shape[0] != num_heads:
        raise ValueError(f"attn_sink length must be {num_heads}, got {attn_sink.shape[0]}")


def _validate_swa_cache_inputs(q: torch.Tensor, kv: torch.Tensor, swa_cache: torch.Tensor) -> None:
    _validate_rank("swa_cache", swa_cache, 3)
    if not swa_cache.is_floating_point():
        raise TypeError(f"swa_cache must be floating point, got {swa_cache.dtype}")
    if swa_cache.device != q.device:
        raise ValueError(f"swa_cache must be on {q.device}, got {swa_cache.device}")
    if swa_cache.shape[-1] != kv.shape[-1]:
        raise ValueError(
            f"swa_cache head dimension must be {kv.shape[-1]}, got {swa_cache.shape[-1]}"
        )


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    compute = x.to(torch.float32)
    output = compute * torch.rsqrt(compute.square().mean(dim=-1, keepdim=True) + eps)
    if weight.numel() != 0:
        output = output * weight.to(device=x.device, dtype=torch.float32)
    return output.to(x.dtype)


def _compressor_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm for the *main-compressor* pooled states (idea_0014).

    Routes through the fused ``auto_deploy::triton_rms_norm`` op -- one kernel that does
    the fp32 cast, square, mean, rsqrt and weighted scale, replacing the ~6 eager kernels
    ``_rms_norm_ref`` emits. The Triton kernel forces fp32 internals and an fp32 weight
    multiply, so it is *byte-identical* to ``_rms_norm_ref`` for the compressor head_dim
    shapes (validated). Falls back to the eager reference on non-CUDA or when no per-channel
    norm weight is present (the op requires a 1-D weight of width ``head_dim``).

    Only the main-compressor (rotate=False) sites use this; the lightning-indexer norm
    (which feeds top-k selection) is intentionally left on ``_rms_norm_ref``.
    """
    if x.device.type == "cuda" and weight.dim() == 1 and weight.numel() == x.shape[-1]:
        return torch.ops.auto_deploy.triton_rms_norm(x, weight, eps)
    return _rms_norm_ref(x, weight, eps)


def _apply_interleaved_rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x.contiguous()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(x.dtype)


def _apply_compressed_rope_and_quantize(
    compressed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    rotate: bool = False,
) -> torch.Tensor:
    if rope_dim < 0 or rope_dim > compressed.shape[-1]:
        raise ValueError(f"rope_dim must be in [0, {compressed.shape[-1]}], got {rope_dim}")
    nope_dim = compressed.shape[-1] - rope_dim
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    if rotate:
        # Indexer RoPE->Hadamard tail (rotate=True) -- left intact (out of scope per
        # idea_0014: "without revisiting the discarded RoPE-to-Hadamard sites").
        pe = _apply_interleaved_rope_ref(pe, cos, sin)
        compressed = torch.cat((nope, pe), dim=-1)
        return torch.ops.auto_deploy.deepseek_v4_hadamard_fp4(compressed, 32)
    # Main-compressor rotate=False tail. fp8-quantize the nope slice, then collapse the
    # interleaved RoPE on the pe slice AND the final concat into ONE fused Triton kernel
    # (auto_deploy::deepseek_v4_fused_rope_concat -- the same op the main q/kv/out paths
    # already use). This removes the eager rope's ~6-7 elementwise muls + stack, the
    # redundant intermediate ``cat((nope, pe))`` + re-``split`` (the round-trip returns
    # the identical nope/rope'd-pe it just built), and the final ``cat`` -- ~9 launches
    # per call collapse to 1. fp8(nope) is byte-identical to before; the rope differs by
    # <=1 ULP (FMA folding; see test_deepseek_v4_fused_rope_concat.py).
    nope = _fake_fp8_act_quant(nope, block_size=64)
    if rope_dim == 0:
        return torch.cat((nope, pe), dim=-1)
    return torch.ops.auto_deploy.deepseek_v4_fused_rope_concat(nope, pe, cos, sin, False)


def _overlap_transform_projected(
    tensor: torch.Tensor,
    head_dim: int,
    value: float,
) -> torch.Tensor:
    batch_size, compressed_len, ratio, _ = tensor.shape
    previous = tensor[:, :, :, :head_dim]
    current = tensor[:, :, :, head_dim:]
    prefix = tensor.new_full((batch_size, 1, ratio, head_dim), value)
    previous = torch.cat((prefix, previous[:, :-1]), dim=1)
    return torch.cat((previous, current), dim=2)


def _build_full_compressed_kv(
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
    rotate: bool = False,
) -> torch.Tensor:
    mode = _compression_mode(compress_ratio)
    if not mode.enabled:
        return compressor_kv.new_empty(compressor_kv.shape[0], 0, compressor_kv.shape[-1])

    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("cos_table", cos_table, 2)
    _validate_rank("sin_table", sin_table, 2)
    _validate_rank("position_ids", position_ids, 2)
    if compressor_kv.shape != compressor_gate.shape:
        raise ValueError(
            "compressor_kv and compressor_gate must have matching shapes, "
            f"got {tuple(compressor_kv.shape)} and {tuple(compressor_gate.shape)}"
        )
    if max_compressed_len <= 0:
        raise ValueError(f"max_compressed_len must be positive, got {max_compressed_len}")

    batch_size, seq_len, state_dim = compressor_kv.shape
    if state_dim % mode.channels != 0:
        raise ValueError(f"compressor state dim {state_dim} is not divisible by {mode.channels}")
    head_dim = state_dim // mode.channels
    max_compressed_tokens = max_compressed_len * compress_ratio
    if seq_len > max_compressed_tokens:
        raise ValueError(f"seq_len {seq_len} exceeds compressed capacity {max_compressed_tokens}")
    if seq_len == 0:
        return compressor_kv.new_empty(batch_size, max_compressed_len, head_dim)

    row_offsets = torch.arange(max_compressed_len, device=compressor_kv.device)
    token_offsets = torch.arange(compress_ratio, device=compressor_kv.device)
    gather_idxs = row_offsets.unsqueeze(1) * compress_ratio + token_offsets
    valid = gather_idxs < seq_len
    gather_idxs = torch.where(valid, gather_idxs, torch.zeros_like(gather_idxs))
    flat_idxs = gather_idxs.reshape(-1)

    kv = compressor_kv[:, flat_idxs].view(batch_size, max_compressed_len, compress_ratio, state_dim)
    gate = compressor_gate[:, flat_idxs].view(
        batch_size, max_compressed_len, compress_ratio, state_dim
    )
    gate = gate + compressor_ape.to(device=gate.device, dtype=gate.dtype)
    gate = torch.where(
        valid.view(1, max_compressed_len, compress_ratio, 1),
        gate,
        gate.new_full((), -1.0e20),
    )
    if mode.overlap:
        kv = _overlap_transform_projected(kv, head_dim, 0.0)
        gate = _overlap_transform_projected(gate, head_dim, -1.0e20)

    compressed = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    compressed = _compressor_rms_norm(compressed, compressor_norm_weight, rms_norm_eps)

    row_start = row_offsets * compress_ratio
    row_start = torch.minimum(row_start, torch.full_like(row_start, seq_len - 1))
    compressed_position_ids = position_ids[:, row_start]
    cos = cos_table[compressed_position_ids]
    sin = sin_table[compressed_position_ids]
    return _apply_compressed_rope_and_quantize(compressed, cos, sin, rope_dim, rotate=rotate)


def _gather_selected_kv(
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    batch_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kv_rows = kv.shape[1]
    if kv_rows == 0:
        return kv.new_zeros(*topk_idxs.shape, kv.shape[-1])

    gather_topk_idxs = topk_idxs.to(torch.long).clamp(min=0, max=kv_rows - 1)
    if batch_idxs is not None:
        return kv[batch_idxs.to(torch.long).unsqueeze(1), gather_topk_idxs]

    batch_size, seq_len, k_select = topk_idxs.shape
    head_dim = kv.shape[-1]
    gather_idx = gather_topk_idxs.unsqueeze(-1).expand(batch_size, seq_len, k_select, head_dim)
    expanded_kv = kv.unsqueeze(1).expand(batch_size, seq_len, kv.shape[1], head_dim)
    return torch.gather(expanded_kv, dim=2, index=gather_idx)


def _to_host_long(name: str, tensor: torch.Tensor, length: int) -> torch.Tensor:
    flat = tensor.detach().cpu().to(torch.long).flatten()
    if flat.numel() < length:
        raise ValueError(f"{name} must have at least {length} elements, got {flat.numel()}")
    return flat[:length]


def _host_page_id_and_offset(
    cache: torch.Tensor,
    seq_idx: int,
    logical_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
) -> tuple[int, int]:
    if logical_pos < 0:
        raise ValueError(f"logical_pos must be non-negative, got {logical_pos}")
    tokens_per_block = int(cache.shape[1])
    page_ordinal = logical_pos // tokens_per_block
    page_offset = logical_pos % tokens_per_block
    page_start = int(cu_num_pages_host[seq_idx].item())
    page_end = int(cu_num_pages_host[seq_idx + 1].item())
    page_table_idx = page_start + page_ordinal
    if page_table_idx >= page_end:
        raise ValueError(
            f"Sequence {seq_idx} logical position {logical_pos} needs page ordinal "
            f"{page_ordinal}, but only {page_end - page_start} page(s) are active"
        )
    return int(cache_loc_host[page_table_idx].item()), page_offset


def _host_position_is_valid(
    cache: torch.Tensor,
    seq_idx: int,
    logical_pos: int,
    cu_num_pages_host: torch.Tensor,
) -> bool:
    if logical_pos < 0:
        return False
    tokens_per_block = int(cache.shape[1])
    page_ordinal = logical_pos // tokens_per_block
    page_start = int(cu_num_pages_host[seq_idx].item())
    page_end = int(cu_num_pages_host[seq_idx + 1].item())
    return page_start + page_ordinal < page_end


def _write_paged_cache_rows(
    values: torch.Tensor,
    cache: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
) -> None:
    if input_pos < 0:
        raise ValueError(f"input_pos must be non-negative, got {input_pos}")
    if values.numel() == 0:
        return

    cursor = 0
    logical_pos = input_pos
    tokens_per_block = int(cache.shape[1])
    while cursor < values.shape[0]:
        page_id, page_offset = _host_page_id_and_offset(
            cache, seq_idx, logical_pos, cu_num_pages_host, cache_loc_host
        )
        write_len = min(values.shape[0] - cursor, tokens_per_block - page_offset)
        cache[page_id, page_offset : page_offset + write_len].copy_(
            values[cursor : cursor + write_len].to(cache.dtype)
        )
        cursor += write_len
        logical_pos += write_len


def _write_paged_cache_rows_at_positions(
    values: torch.Tensor,
    cache: torch.Tensor,
    seq_idx: int,
    logical_positions: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
) -> None:
    if values.numel() == 0:
        return
    positions_host = logical_positions.detach().cpu().to(torch.long).flatten()
    for row_idx, logical_pos_tensor in enumerate(positions_host):
        page_id, page_offset = _host_page_id_and_offset(
            cache,
            seq_idx,
            int(logical_pos_tensor.item()),
            cu_num_pages_host,
            cache_loc_host,
        )
        cache[page_id, page_offset].copy_(values[row_idx].to(cache.dtype))


def _slice_sequence_tokens(
    tensor: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.new_empty(seq_len, *tensor.shape[2:])
    if tensor.shape[0] > seq_idx and tensor.shape[0] != 1:
        return tensor[seq_idx, :seq_len]
    return tensor.reshape(-1, *tensor.shape[2:])[flat_start : flat_start + seq_len]


def _prefill_kv_source(
    kv: torch.Tensor,
    kv_seq: torch.Tensor,
    seq_idx: int,
    num_seq: int,
) -> torch.Tensor:
    if kv.shape[0] > seq_idx and kv.shape[0] != 1:
        return kv[seq_idx : seq_idx + 1]
    if num_seq == 1:
        return kv
    return kv_seq.unsqueeze(0)


def _slice_sequence_positions(
    position_ids: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
) -> torch.Tensor:
    if position_ids.shape[0] > seq_idx and position_ids.shape[0] != 1:
        return position_ids[seq_idx : seq_idx + 1, :seq_len]
    return position_ids.reshape(1, -1)[:, flat_start : flat_start + seq_len]


def _slice_sequence_kv_rows(
    kv: torch.Tensor,
    seq_idx: int,
    flat_start: int,
    seq_len: int,
    num_seq: int,
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio == 0:
        return _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len)
    if kv.shape[0] > seq_idx and kv.shape[0] != 1:
        return kv[seq_idx]
    if num_seq == 1:
        return kv.reshape(-1, kv.shape[-1])
    raise ValueError(
        "Flattened compressed DeepSeek V4 sparse attention KV rows are not supported; "
        f"pass batched kv for compress_ratio={compress_ratio}."
    )


def _cached_sparse_attention_from_positions(
    q_token: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    seq_idx: int,
    positions: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    selected_kv, valid_rows = _gather_paged_rows_from_positions(
        swa_cache,
        seq_idx,
        positions,
        cu_num_pages_host,
        cache_loc_host,
        q_token.dtype,
    )
    selected_kv = selected_kv.unsqueeze(0)
    local_topk = torch.arange(positions.numel(), dtype=torch.long, device=q_token.device).view(
        1, 1, -1
    )
    local_topk = torch.where(valid_rows.view(1, 1, -1), local_topk, torch.full_like(local_topk, -1))
    return _deepseek_v4_sparse_attention(
        q_token.view(1, 1, *q_token.shape),
        selected_kv,
        attn_sink,
        local_topk,
        softmax_scale,
    ).view(*q_token.shape)


def _cached_local_window_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    window_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        start_pos = max(0, query_pos - window_size + 1)
        positions = torch.arange(start_pos, query_pos + 1, device=q_seq.device)
        outputs.append(
            _cached_sparse_attention_from_positions(
                q_seq[token_offset],
                attn_sink,
                swa_cache,
                seq_idx,
                positions,
                cu_num_pages_host,
                cache_loc_host,
                softmax_scale,
            )
        )
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _gather_paged_rows_from_positions(
    cache: torch.Tensor,
    seq_idx: int,
    positions: torch.Tensor,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    dtype: torch.dtype,
    width: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Vectorized gather (prefill path). Mirrors ``_host_page_id_and_offset`` /
    # ``_host_position_is_valid`` for every position at once, with no per-position
    # host sync (``.cpu()``/``.item()``) or Python loop. ``page_start``/``page_end``
    # are the only host reads -- two O(1) scalar loads for this sequence.
    row_width = cache.shape[-1] if width is None else width
    tokens_per_block = int(cache.shape[1])
    page_start = int(cu_num_pages_host[seq_idx].item())
    page_end = int(cu_num_pages_host[seq_idx + 1].item())

    positions_flat = positions.detach().to(device=cache.device, dtype=torch.long).flatten()
    if positions_flat.numel() == 0:
        gathered = cache.new_empty(0, row_width, dtype=dtype)
        valid = torch.empty(0, dtype=torch.bool, device=cache.device)
        return gathered.view(*positions.shape, row_width), valid.view(positions.shape)

    safe_positions = positions_flat.clamp(min=0)
    page_ordinal = safe_positions // tokens_per_block
    page_offset = safe_positions % tokens_per_block
    page_table_idx = page_start + page_ordinal
    valid = (positions_flat >= 0) & (page_table_idx < page_end)

    # Mask invalid page-table indices to a safe in-range slot before the physical
    # page lookup, then zero the corresponding rows out below.
    safe_page_table_idx = torch.where(
        valid, page_table_idx, page_table_idx.new_full((), page_start)
    )
    safe_page_table_idx = safe_page_table_idx.clamp(min=0, max=cache_loc_host.numel() - 1)
    phys_page = cache_loc_host.to(device=cache.device, dtype=torch.long)[safe_page_table_idx]

    gathered = cache[phys_page, page_offset]
    if width is not None:
        gathered = gathered[..., :width]
    gathered = gathered.to(dtype)
    gathered = torch.where(valid.unsqueeze(-1), gathered, gathered.new_zeros(()))

    valid = valid.to(device=positions.device)
    return gathered.view(*positions.shape, row_width), valid.view(positions.shape)


def _gather_paged_rows(
    cache: torch.Tensor,
    seq_idx: int,
    start_pos: int,
    end_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    dtype: torch.dtype,
    width: Optional[int] = None,
) -> torch.Tensor:
    if start_pos < 0 or end_pos < start_pos:
        raise ValueError(f"Invalid cache slice [{start_pos}, {end_pos})")
    positions = torch.arange(start_pos, end_pos, dtype=torch.long, device=cache.device)
    rows, _ = _gather_paged_rows_from_positions(
        cache, seq_idx, positions, cu_num_pages_host, cache_loc_host, dtype, width=width
    )
    return rows


def _compressed_row_from_paged_state(
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    seq_idx: int,
    row_idx: int,
    row_position_id: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    head_dim: int,
    state_dim: int,
    dtype: torch.dtype,
    rotate: bool = False,
) -> torch.Tensor:
    anchor = row_idx * compress_ratio
    kv_rows = []
    gate_rows = []
    mode = _compression_mode(compress_ratio)
    if mode.overlap:
        for offset in range(compress_ratio):
            position = anchor - compress_ratio + offset
            if position < 0:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=dtype, device=compressor_kv_cache.device)
                )
                gate_rows.append(
                    torch.full(
                        (head_dim,),
                        -1.0e20,
                        dtype=dtype,
                        device=compressor_gate_cache.device,
                    )
                )
                continue
            kv_state = _gather_paged_rows(
                compressor_kv_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages_host,
                cache_loc_host,
                dtype,
            ).squeeze(0)
            gate_state = _gather_paged_rows(
                compressor_gate_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages_host,
                cache_loc_host,
                dtype,
            ).squeeze(0)
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + compressor_ape[offset, :head_dim].to(dtype))

        for offset in range(compress_ratio):
            position = anchor + offset
            kv_state = _gather_paged_rows(
                compressor_kv_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages_host,
                cache_loc_host,
                dtype,
            ).squeeze(0)
            gate_state = _gather_paged_rows(
                compressor_gate_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages_host,
                cache_loc_host,
                dtype,
            ).squeeze(0)
            kv_rows.append(kv_state[head_dim : 2 * head_dim])
            gate_rows.append(
                gate_state[head_dim : 2 * head_dim]
                + compressor_ape[offset, head_dim : 2 * head_dim].to(dtype)
            )
    else:
        for offset in range(compress_ratio):
            position = anchor + offset
            kv_state = _gather_paged_rows(
                compressor_kv_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages_host,
                cache_loc_host,
                dtype,
            ).squeeze(0)
            gate_state = _gather_paged_rows(
                compressor_gate_cache,
                seq_idx,
                position,
                position + 1,
                cu_num_pages_host,
                cache_loc_host,
                dtype,
            ).squeeze(0)
            kv_rows.append(kv_state[:head_dim])
            gate_rows.append(gate_state[:head_dim] + compressor_ape[offset, :head_dim].to(dtype))

    kv = torch.stack(kv_rows, dim=0)
    gate = torch.stack(gate_rows, dim=0)
    pooled = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    pooled = _rms_norm_ref(pooled.unsqueeze(0), compressor_norm_weight, rms_norm_eps).squeeze(0)
    del state_dim
    row_position_id = max(0, min(row_position_id, cos_table.shape[0] - 1))
    cos = cos_table[row_position_id].unsqueeze(0)
    sin = sin_table[row_position_id].unsqueeze(0)
    return _apply_compressed_rope_and_quantize(
        pooled.unsqueeze(0),
        cos,
        sin,
        rope_dim,
        rotate=rotate,
    ).squeeze(0)


def _update_compressed_paged_caches(
    compressor_kv_seq: torch.Tensor,
    compressor_gate_seq: torch.Tensor,
    position_ids_seq: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
) -> None:
    mode = _compression_mode(compress_ratio)
    if not mode.enabled or compressor_kv_seq.numel() == 0:
        return

    if compressor_kv_seq.shape != compressor_gate_seq.shape:
        raise ValueError(
            "compressor_kv and compressor_gate sequence slices must have matching shapes, "
            f"got {tuple(compressor_kv_seq.shape)} and {tuple(compressor_gate_seq.shape)}"
        )
    state_dim = int(compressor_kv_seq.shape[-1])
    head_dim = state_dim // mode.channels
    _write_paged_cache_rows(
        compressor_kv_seq,
        compressor_kv_cache,
        seq_idx,
        input_pos,
        cu_num_pages_host,
        cache_loc_host,
    )
    _write_paged_cache_rows(
        compressor_gate_seq,
        compressor_gate_cache,
        seq_idx,
        input_pos,
        cu_num_pages_host,
        cache_loc_host,
    )

    old_completed = min(input_pos // compress_ratio, max_compressed_len)
    new_completed = min(
        (input_pos + compressor_kv_seq.shape[0]) // compress_ratio, max_compressed_len
    )
    compressed_rows = []
    flat_position_ids = position_ids_seq.reshape(-1)
    first_position_id = int(flat_position_ids[0].item())
    for row_idx in range(old_completed, new_completed):
        row_token_offset = row_idx * compress_ratio - input_pos
        if 0 <= row_token_offset < flat_position_ids.numel():
            row_position_id = int(flat_position_ids[row_token_offset].item())
        else:
            row_position_id = first_position_id + row_token_offset
        compressed_rows.append(
            _compressed_row_from_paged_state(
                compressor_kv_cache,
                compressor_gate_cache,
                seq_idx,
                row_idx,
                row_position_id,
                cu_num_pages_host,
                cache_loc_host,
                compressor_ape,
                compressor_norm_weight,
                cos_table,
                sin_table,
                rms_norm_eps,
                rope_dim,
                compress_ratio,
                head_dim,
                state_dim,
                compressor_kv_seq.dtype,
            )
        )
    if compressed_rows:
        logical_positions = torch.arange(
            old_completed,
            old_completed + len(compressed_rows),
            dtype=torch.long,
            device=mhc_cache.device,
        )
        logical_positions = logical_positions * compress_ratio
        _write_paged_cache_rows_at_positions(
            torch.stack(compressed_rows, dim=0),
            mhc_cache,
            seq_idx,
            logical_positions,
            cu_num_pages_host,
            cache_loc_host,
        )


def _update_raw_paged_caches(
    compressor_kv_seq: torch.Tensor,
    compressor_gate_seq: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
) -> None:
    if compressor_kv_seq.numel() == 0:
        return
    if compressor_kv_seq.shape != compressor_gate_seq.shape:
        raise ValueError(
            "compressor_kv and compressor_gate sequence slices must have matching shapes, "
            f"got {tuple(compressor_kv_seq.shape)} and {tuple(compressor_gate_seq.shape)}"
        )
    _write_paged_cache_rows(
        compressor_kv_seq,
        compressor_kv_cache,
        seq_idx,
        input_pos,
        cu_num_pages_host,
        cache_loc_host,
    )
    _write_paged_cache_rows(
        compressor_gate_seq,
        compressor_gate_cache,
        seq_idx,
        input_pos,
        cu_num_pages_host,
        cache_loc_host,
    )


def _select_ratio4_indexer_rows(
    q_index: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    seq_idx: int,
    query_pos: int,
    query_position_id: int,
    index_topk: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    max_compressed_len: int,
) -> torch.Tensor:
    if index_topk <= 0:
        return torch.empty(0, dtype=torch.int64, device=q_index.device)

    visible_len = min((query_pos + 1) // 4, max_compressed_len)
    if visible_len <= 0:
        return torch.full((index_topk,), -1, dtype=torch.int64, device=q_index.device)

    index_head_dim = int(q_index.shape[-1])
    # Vectorized replacement for the per-row ``_compressed_row_from_paged_state``
    # stack: one call to the batched decode helper over ``row_idx = arange(visible_len)``.
    # ``seq_idx`` is broadcast to every row and ``row_position_id`` reproduces the loop's
    # scalar ``query_position_id - (query_pos - row_idx * 4)`` as a tensor. The batched
    # helper returns rows in the same ``[visible_len, index_head_dim]`` order/shape the
    # stack produced (validated bit-exact in the op unit tests).
    row_idx = torch.arange(visible_len, dtype=torch.long, device=q_index.device)
    seq_idx_rows = torch.full((visible_len,), seq_idx, dtype=torch.long, device=q_index.device)
    row_position_id_rows = query_position_id - (query_pos - row_idx * 4)
    # The batched decode helper indexes ``cu_num_pages``/``cache_loc`` with device-side
    # tensors (row_idx/seq_idx live on ``q_index.device``), so move the host page tables
    # onto that device first -- the naive loop used the ``_host_*`` scalar path instead.
    cu_num_pages_dev = cu_num_pages_host.to(q_index.device)
    cache_loc_dev = cache_loc_host.to(q_index.device)
    index_k = _batched_compressed_rows_from_paged_state(
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        seq_idx_rows,
        row_idx,
        row_position_id_rows,
        cu_num_pages_dev,
        cache_loc_dev,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        4,
        index_head_dim,
        q_index.dtype,
        rotate=True,
    )
    index_score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
    index_score = (index_score.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=0)
    # ``q_index``/``indexer_weights`` are replicated across TP ranks (the indexer
    # index-score projection is no longer head-sharded; see DeepseekV4Indexer),
    # so ``index_score`` already sums over all index heads -- no all_reduce needed.

    topk_count = min(index_topk, visible_len)
    selected = index_score.topk(topk_count, dim=-1).indices.to(torch.int64)
    if topk_count < index_topk:
        pad = torch.full(
            (index_topk - topk_count,),
            -1,
            dtype=selected.dtype,
            device=selected.device,
        )
        selected = torch.cat((selected, pad), dim=0)
    return selected


def _cached_compressed_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    seq_idx: int,
    input_pos: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    position_ids_seq: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    softmax_scale: float,
    topk_seq: Optional[torch.Tensor] = None,
    indexer_q_seq: Optional[torch.Tensor] = None,
    indexer_weights_seq: Optional[torch.Tensor] = None,
    indexer_compressor_kv_cache: Optional[torch.Tensor] = None,
    indexer_compressor_gate_cache: Optional[torch.Tensor] = None,
    indexer_compressor_ape: Optional[torch.Tensor] = None,
    indexer_compressor_norm_weight: Optional[torch.Tensor] = None,
    cos_table: Optional[torch.Tensor] = None,
    sin_table: Optional[torch.Tensor] = None,
    rms_norm_eps: float = 1e-6,
    rope_dim: Optional[int] = None,
) -> torch.Tensor:
    outputs = []
    flat_position_ids = position_ids_seq.reshape(-1)
    for token_offset in range(q_seq.shape[0]):
        query_pos = input_pos + token_offset
        query_position_id = int(flat_position_ids[token_offset].item())
        local_start = max(0, query_pos - window_size + 1)
        local_kv = _gather_paged_rows(
            swa_cache,
            seq_idx,
            local_start,
            query_pos + 1,
            cu_num_pages_host,
            cache_loc_host,
            q_seq.dtype,
        )
        local_idxs = torch.arange(local_kv.shape[0], dtype=torch.int64, device=q_seq.device)
        mode = _compression_mode(compress_ratio)
        if mode.uses_indexer:
            if (
                topk_seq is None
                or indexer_q_seq is None
                or indexer_weights_seq is None
                or indexer_compressor_kv_cache is None
                or indexer_compressor_gate_cache is None
                or indexer_compressor_ape is None
                or indexer_compressor_norm_weight is None
                or cos_table is None
                or sin_table is None
                or rope_dim is None
            ):
                raise ValueError(
                    "Overlap/indexer cached decode requires indexer tensors and caches."
                )
            index_topk = max(int(topk_seq.shape[-1]) - int(window_size), 0)
            selected_rows = _select_ratio4_indexer_rows(
                indexer_q_seq[token_offset],
                indexer_weights_seq[token_offset],
                indexer_compressor_kv_cache,
                indexer_compressor_gate_cache,
                seq_idx,
                query_pos,
                query_position_id,
                index_topk,
                cu_num_pages_host,
                cache_loc_host,
                indexer_compressor_ape,
                indexer_compressor_norm_weight,
                cos_table,
                sin_table,
                rms_norm_eps,
                rope_dim,
                max_compressed_len,
            )
            compressed_positions = selected_rows.clamp(min=0) * compress_ratio
            compressed_kv, compressed_valid = _gather_paged_rows_from_positions(
                mhc_cache,
                seq_idx,
                compressed_positions,
                cu_num_pages_host,
                cache_loc_host,
                q_seq.dtype,
            )
            compressed_valid = compressed_valid & (selected_rows >= 0)
            compressed_idxs = torch.where(
                compressed_valid,
                torch.arange(selected_rows.numel(), dtype=torch.int64, device=q_seq.device)
                + local_kv.shape[0],
                torch.full_like(selected_rows, -1),
            )
        else:
            compressed_len = min((query_pos + 1) // compress_ratio, max_compressed_len)
            compressed_positions = (
                torch.arange(compressed_len, dtype=torch.long, device=q_seq.device) * compress_ratio
            )
            compressed_kv, compressed_valid = _gather_paged_rows_from_positions(
                mhc_cache,
                seq_idx,
                compressed_positions,
                cu_num_pages_host,
                cache_loc_host,
                q_seq.dtype,
            )
            compressed_idxs = torch.arange(
                compressed_kv.shape[0], dtype=torch.int64, device=q_seq.device
            )
            compressed_idxs = compressed_idxs + local_kv.shape[0]
            compressed_idxs = torch.where(
                compressed_valid,
                compressed_idxs,
                torch.full_like(compressed_idxs, -1),
            )
        topk = torch.cat((local_idxs, compressed_idxs), dim=0).view(1, 1, -1)
        kv = torch.cat((local_kv, compressed_kv), dim=0)
        out = _deepseek_v4_sparse_attention(
            q_seq[token_offset : token_offset + 1].unsqueeze(0),
            kv.unsqueeze(0),
            attn_sink,
            topk,
            softmax_scale,
        )
        outputs.append(out.squeeze(0).squeeze(0))
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _cached_topk_attention(
    q_seq: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_seq: torch.Tensor,
    swa_cache: torch.Tensor,
    seq_idx: int,
    cu_num_pages_host: torch.Tensor,
    cache_loc_host: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    outputs = []
    for token_offset in range(q_seq.shape[0]):
        outputs.append(
            _cached_sparse_attention_from_positions(
                q_seq[token_offset],
                attn_sink,
                swa_cache,
                seq_idx,
                topk_seq[token_offset],
                cu_num_pages_host,
                cache_loc_host,
                softmax_scale,
            )
        )
    if not outputs:
        return q_seq.new_empty(q_seq.shape)
    return torch.stack(outputs, dim=0)


def _flatten_decode_tokens(tensor: torch.Tensor, num_decode: int) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.new_empty(num_decode, *tensor.shape[2:])
    return tensor.reshape(-1, *tensor.shape[2:])[:num_decode]


def _page_ids_and_offsets_from_tpb(
    tokens_per_block: int,
    seq_idx: torch.Tensor,
    positions: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged ``(page_ids, page_offsets, valid)`` translation for ``positions``.

    Single source of truth for the DeepSeek-V4 sparse-attention page-map math.
    ``_decode_page_ids_and_offsets`` (per-layer, reads ``tokens_per_block`` from a
    cache tensor) and ``deepseek_v4_sparse_prepare_decode_page_addr`` (once-per-forward
    hoist, receives ``tokens_per_block`` as a constant) both call this so the hoisted
    addresses are bit-identical to the per-layer translation they replace.
    """
    if cache_loc.numel() == 0:
        raise ValueError("cache_loc must contain at least one page id")

    positions_long = positions.to(torch.long)
    safe_positions = positions_long.clamp(min=0)
    page_ordinals = safe_positions // tokens_per_block
    page_offsets = safe_positions % tokens_per_block

    seq_idx_long = seq_idx.to(torch.long)
    while seq_idx_long.dim() < positions_long.dim():
        seq_idx_long = seq_idx_long.unsqueeze(-1)
    page_start = cu_num_pages[seq_idx_long].to(torch.long)
    page_end = cu_num_pages[seq_idx_long + 1].to(torch.long)
    page_table_idx = page_start + page_ordinals
    valid = (positions_long >= 0) & (page_table_idx < page_end)
    safe_page_table_idx = torch.where(valid, page_table_idx, page_start)
    safe_page_table_idx = safe_page_table_idx.clamp(min=0, max=cache_loc.numel() - 1)
    page_ids = cache_loc[safe_page_table_idx].to(torch.long)
    return page_ids, page_offsets, valid


def _decode_page_ids_and_offsets(
    cache: torch.Tensor,
    seq_idx: torch.Tensor,
    positions: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _page_ids_and_offsets_from_tpb(
        int(cache.shape[1]), seq_idx, positions, cu_num_pages, cache_loc
    )


def _write_decode_cache_rows(
    cache: torch.Tensor,
    values: torch.Tensor,
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    page_ids: Optional[torch.Tensor] = None,
    page_offsets: Optional[torch.Tensor] = None,
) -> None:
    if values.numel() == 0:
        return
    # ``page_ids``/``page_offsets`` are the precomputed current-token write
    # address (see ``deepseek_v4_sparse_prepare_decode_page_addr``). When given,
    # the per-layer page-map translation is skipped (bit-identical addresses).
    if page_ids is None:
        page_ids, page_offsets, _ = _decode_page_ids_and_offsets(
            cache, seq_idx, input_pos, cu_num_pages, cache_loc
        )
    cache[page_ids, page_offsets] = values.to(cache.dtype)


def _decode_cache_rows_from_positions(
    cache: torch.Tensor,
    seq_idx: torch.Tensor,
    positions: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    dtype: torch.dtype,
    page_map: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ``page_map`` is a precomputed ``(page_ids, page_offsets, valid)`` triple from
    # ``_decode_page_ids_and_offsets``. Every DeepSeek-V4 sparse cache shares one
    # page table (``cu_num_pages``/``cache_loc``) and one ``tokens_per_block``, so
    # the translation for a given ``positions`` is identical regardless of which
    # cache it indexes. Callers that read the same positions from a kv/gate pair
    # (or the same logical row for a paired read+write) compute the map once and
    # reuse it here, skipping a redundant ~16-kernel integer translation chain.
    if page_map is None:
        page_ids, page_offsets, valid = _decode_page_ids_and_offsets(
            cache, seq_idx, positions, cu_num_pages, cache_loc
        )
    else:
        page_ids, page_offsets, valid = page_map
    return cache[page_ids, page_offsets].to(dtype), valid


def _decode_attention_from_selected(
    q_decode: torch.Tensor,
    selected_kv: torch.Tensor,
    rel_topk: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Attend ``q_decode`` over ``selected_kv`` given precomputed relative row indices.

    ``rel_topk`` has shape ``[N, kv_rows]``: the slot's own id for a kept row and
    ``-1`` for a masked slot. Shared by ``_decode_attention_from_rows`` (which
    derives ``rel_topk`` from a boolean ``valid_rows`` mask via arange + where) and
    the fused paged-assemble path, which emits ``rel_topk`` directly so the
    arange/where pair is dropped from the decode graph.
    """
    output = _deepseek_v4_sparse_attention(
        q_decode.unsqueeze(1),
        selected_kv,
        attn_sink,
        rel_topk.unsqueeze(1),
        softmax_scale,
    )
    return output.squeeze(1)


def _decode_attention_from_rows(
    q_decode: torch.Tensor,
    selected_kv: torch.Tensor,
    valid_rows: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    rel_topk = torch.arange(selected_kv.shape[1], dtype=torch.int64, device=q_decode.device)
    rel_topk = rel_topk.view(1, -1).expand(q_decode.shape[0], -1)
    rel_topk = torch.where(valid_rows, rel_topk, torch.full_like(rel_topk, -1))
    return _decode_attention_from_selected(
        q_decode, selected_kv, rel_topk, attn_sink, softmax_scale
    )


def _decode_local_cache_rows(
    swa_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    window_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The local-window position generation + page-address translation is a fixed
    # ~20-kernel element-wise chain that runs once per layer per decode step. Fuse
    # it into one Triton kernel (page_ids/page_offsets/valid), leaving only the
    # final cache-row gather as a separate op. Byte-identical addresses + mask.
    if _HAS_TRITON and swa_cache.is_cuda:
        page_ids, page_offsets, valid = _fused_local_window_pagemap(
            input_pos,
            seq_idx,
            cu_num_pages,
            cache_loc,
            window_size,
            int(swa_cache.shape[1]),
        )
        rows = swa_cache[page_ids, page_offsets].to(dtype)
        return rows, valid

    offsets = torch.arange(window_size, dtype=torch.long, device=input_pos.device)
    positions = input_pos.unsqueeze(1) - window_size + 1 + offsets.view(1, -1)
    valid = (positions >= 0) & (positions <= input_pos.unsqueeze(1))
    rows, page_valid = _decode_cache_rows_from_positions(
        swa_cache, seq_idx, positions, cu_num_pages, cache_loc, dtype
    )
    valid = valid & page_valid
    return rows, valid


def _decode_topk_cache_attention(
    q_decode: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_decode: torch.Tensor,
    swa_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int],
) -> torch.Tensor:
    if window_size is not None:
        selected_kv, valid_rows = _decode_local_cache_rows(
            swa_cache,
            seq_idx,
            input_pos,
            cu_num_pages,
            cache_loc,
            window_size,
            q_decode.dtype,
        )
        return _decode_attention_from_rows(
            q_decode,
            selected_kv,
            valid_rows,
            attn_sink,
            softmax_scale,
        )

    positions = topk_decode.to(torch.long)
    selected_kv, page_valid = _decode_cache_rows_from_positions(
        swa_cache, seq_idx, positions, cu_num_pages, cache_loc, q_decode.dtype
    )
    valid_rows = (positions >= 0) & page_valid
    return _decode_attention_from_rows(
        q_decode,
        selected_kv,
        valid_rows,
        attn_sink,
        softmax_scale,
    )


def _batched_compressed_rows_from_paged_state(
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    row_idx: torch.Tensor,
    row_position_id: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    head_dim: int,
    dtype: torch.dtype,
    rotate: bool = False,
    overlap_page_map: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    mode = _compression_mode(compress_ratio)
    if mode.overlap:
        if overlap_page_map is not None:
            # Hoisted once-per-forward ratio-4 page map covering the contiguous
            # ``[anchor - ratio, anchor + ratio)`` band: the first ``ratio`` columns
            # are the ``previous`` block, the last ``ratio`` the ``current`` block.
            # Every ratio-4 layer resolves the identical (seq_idx, positions)
            # addresses, so this ``_decode_page_ids_and_offsets`` chain runs once in
            # ``deepseek_v4_sparse_prepare_decode_page_addr`` instead of twice per
            # layer (addresses bit-identical to the per-layer translation below).
            ovl_page_ids, ovl_page_offsets, ovl_valid = overlap_page_map
            previous_positions = None
            current_positions = None
            previous_map = (
                ovl_page_ids[:, :compress_ratio],
                ovl_page_offsets[:, :compress_ratio],
                ovl_valid[:, :compress_ratio],
            )
            current_map = (
                ovl_page_ids[:, compress_ratio:],
                ovl_page_offsets[:, compress_ratio:],
                ovl_valid[:, compress_ratio:],
            )
            # ``valid`` already encodes ``positions >= 0`` (see helper), so
            # ``previous_map[2]`` equals ``(previous_positions >= 0) & page_ok`` and
            # the ``previous_valid & previous_page_valid`` below stays bit-identical.
            previous_valid = previous_map[2]
        else:
            offsets = torch.arange(compress_ratio, dtype=torch.long, device=row_idx.device)
            anchor = row_idx.to(torch.long) * compress_ratio
            previous_positions = anchor.unsqueeze(1) - compress_ratio + offsets.view(1, -1)
            current_positions = anchor.unsqueeze(1) + offsets.view(1, -1)
            previous_valid = previous_positions >= 0

            # kv and gate caches share one page table + tokens_per_block, so the reads
            # at ``previous_positions`` (resp. ``current_positions``) resolve to the same
            # page map across both caches. Compute each distinct map once and reuse it
            # for the kv and gate gather -- 4 page-map chains -> 2.
            previous_map = _decode_page_ids_and_offsets(
                compressor_kv_cache, seq_idx, previous_positions, cu_num_pages, cache_loc
            )
            current_map = _decode_page_ids_and_offsets(
                compressor_kv_cache, seq_idx, current_positions, cu_num_pages, cache_loc
            )
        previous_kv_state, previous_page_valid = _decode_cache_rows_from_positions(
            compressor_kv_cache,
            seq_idx,
            previous_positions,
            cu_num_pages,
            cache_loc,
            dtype,
            previous_map,
        )
        previous_gate_state, _ = _decode_cache_rows_from_positions(
            compressor_gate_cache,
            seq_idx,
            previous_positions,
            cu_num_pages,
            cache_loc,
            dtype,
            previous_map,
        )
        current_kv_state, _ = _decode_cache_rows_from_positions(
            compressor_kv_cache,
            seq_idx,
            current_positions,
            cu_num_pages,
            cache_loc,
            dtype,
            current_map,
        )
        current_gate_state, _ = _decode_cache_rows_from_positions(
            compressor_gate_cache,
            seq_idx,
            current_positions,
            cu_num_pages,
            cache_loc,
            dtype,
            current_map,
        )
        previous_valid = previous_valid & previous_page_valid

        previous_kv = previous_kv_state[..., :head_dim]
        previous_gate = previous_gate_state[..., :head_dim]
        previous_gate = previous_gate + compressor_ape[:, :head_dim].to(
            device=previous_gate.device, dtype=dtype
        )
        previous_kv = torch.where(
            previous_valid.unsqueeze(-1), previous_kv, previous_kv.new_zeros(())
        )
        previous_gate = torch.where(
            previous_valid.unsqueeze(-1),
            previous_gate,
            previous_gate.new_full((), -1.0e20),
        )

        current_kv = current_kv_state[..., head_dim : 2 * head_dim]
        current_gate = current_gate_state[..., head_dim : 2 * head_dim]
        current_gate = current_gate + compressor_ape[:, head_dim : 2 * head_dim].to(
            device=current_gate.device, dtype=dtype
        )
        kv = torch.cat((previous_kv, current_kv), dim=1)
        gate = torch.cat((previous_gate, current_gate), dim=1)
    else:
        offsets = torch.arange(compress_ratio, dtype=torch.long, device=row_idx.device)
        anchor = row_idx.to(torch.long) * compress_ratio
        positions = anchor.unsqueeze(1) + offsets.view(1, -1)
        # kv and gate share the page map for these positions (see overlap branch).
        positions_map = _decode_page_ids_and_offsets(
            compressor_kv_cache, seq_idx, positions, cu_num_pages, cache_loc
        )
        kv_state, _ = _decode_cache_rows_from_positions(
            compressor_kv_cache,
            seq_idx,
            positions,
            cu_num_pages,
            cache_loc,
            dtype,
            positions_map,
        )
        gate_state, _ = _decode_cache_rows_from_positions(
            compressor_gate_cache,
            seq_idx,
            positions,
            cu_num_pages,
            cache_loc,
            dtype,
            positions_map,
        )
        kv = kv_state[..., :head_dim]
        gate = gate_state[..., :head_dim]
        gate = gate + compressor_ape[:, :head_dim].to(device=gate.device, dtype=dtype)

    pooled = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    pooled = _compressor_rms_norm(pooled, compressor_norm_weight, rms_norm_eps)
    row_position_id = row_position_id.to(torch.long).clamp(min=0, max=cos_table.shape[0] - 1)
    cos = cos_table[row_position_id]
    sin = sin_table[row_position_id]
    return _apply_compressed_rope_and_quantize(
        pooled,
        cos,
        sin,
        rope_dim,
        rotate=rotate,
    )


def _batched_overlap_compressed_rows_fullrange(
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    row_position_id: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    head_dim: int,
    max_compressed_len: int,
    dtype: torch.dtype,
    rotate: bool = False,
    full_page_map: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Decode-time overlap compressed rows for the *full* candidate set [0, max_compressed_len).

    This is a launch/bandwidth-reduced specialization of
    ``_batched_compressed_rows_from_paged_state`` (overlap branch) for the case where every
    candidate row 0..max_compressed_len-1 is requested for every sequence (the lightning
    indexer decode path). In that case the per-row ``previous`` block is exactly the
    ``current`` block of the preceding row, so instead of four scattered ``index_select``
    gathers over ``[B, M, ratio]`` grids (previous/current x kv/gate, each token fetched
    twice) we issue a single contiguous gather per cache over ``[B, M*ratio]`` and derive
    the ``previous`` states by a one-row shift. The pool/rmsnorm/rope math is identical to
    the generic helper, so outputs match bit-for-bit (validated in the op unit tests).

    Args:
        seq_idx: ``[B]`` sequence/slot index per decode row.
        row_position_id: ``[B, max_compressed_len]`` query-relative rope position per row.
    Returns:
        ``[B, max_compressed_len, head_dim]`` compressed (post-rope/quant) index rows.
    """
    num_rows = int(seq_idx.shape[0])
    m = int(max_compressed_len)
    device = seq_idx.device

    if full_page_map is not None:
        # Hoisted once-per-forward ratio-4 full-range page map for positions
        # ``[0, m * ratio)``. This is the decode-dominant page-map chain and resolves
        # to identical addresses across every ratio-4 layer, so it is computed once in
        # ``deepseek_v4_sparse_prepare_decode_page_addr`` (bit-identical addresses).
        full_positions = None
        full_map = full_page_map
    else:
        full_positions = torch.arange(m * compress_ratio, dtype=torch.long, device=device)
        full_positions = full_positions.view(1, -1).expand(num_rows, -1)
        # kv and gate share the page map for ``full_positions`` (caches share one page
        # table + tokens_per_block); compute it once instead of once per cache. This is
        # the decode-dominant gather, so the duplicate chain is the largest to remove.
        full_map = _decode_page_ids_and_offsets(
            compressor_kv_cache, seq_idx, full_positions, cu_num_pages, cache_loc
        )

    # Fused single-kernel path (idea_0075): collapse the gather / row-shift /
    # concat / where / pool / rmsnorm swarm into one launch that emits the
    # pooled+normed candidate rows directly from the paged caches and the hoisted
    # page maps.  Requires the ratio-4 overlap layout (>= 2*head_dim state
    # channels) and a per-channel rms weight; falls back to the eager chain below
    # otherwise.  Only the pool/norm result is produced here -- the rope/quantize
    # tail is shared with the eager path.
    if (
        _HAS_TRITON
        and compressor_kv_cache.is_cuda
        and compress_ratio == 4
        and full_map[2] is not None
        and compressor_norm_weight.dim() == 1
        and compressor_norm_weight.numel() == head_dim
        and int(compressor_kv_cache.shape[-1]) >= 2 * head_dim
        and compressor_ape.dim() == 2
        and int(compressor_ape.shape[1]) >= 2 * head_dim
    ):
        pooled = _fused_fullrange_candidate_rows(
            compressor_kv_cache,
            compressor_gate_cache,
            full_map[0],
            full_map[1],
            full_map[2],
            compressor_ape,
            compressor_norm_weight,
            rms_norm_eps,
            compress_ratio,
            head_dim,
            m,
            dtype,
        )
        row_position_id = row_position_id.to(torch.long).clamp(min=0, max=cos_table.shape[0] - 1)
        cos = cos_table[row_position_id]
        sin = sin_table[row_position_id]
        return _apply_compressed_rope_and_quantize(pooled, cos, sin, rope_dim, rotate=rotate)

    current_kv_state, current_page_valid = _decode_cache_rows_from_positions(
        compressor_kv_cache, seq_idx, full_positions, cu_num_pages, cache_loc, dtype, full_map
    )
    current_gate_state, _ = _decode_cache_rows_from_positions(
        compressor_gate_cache, seq_idx, full_positions, cu_num_pages, cache_loc, dtype, full_map
    )
    state_dim = int(current_kv_state.shape[-1])
    current_kv_state = current_kv_state.view(num_rows, m, compress_ratio, state_dim)
    current_gate_state = current_gate_state.view(num_rows, m, compress_ratio, state_dim)
    current_page_valid = current_page_valid.view(num_rows, m, compress_ratio)

    # previous block of row r == current block of row r-1; row 0 has no previous block.
    zero_state = current_kv_state.new_zeros(num_rows, 1, compress_ratio, state_dim)
    previous_kv_state = torch.cat((zero_state, current_kv_state[:, :-1]), dim=1)
    previous_gate_state = torch.cat((zero_state, current_gate_state[:, :-1]), dim=1)
    false_valid = current_page_valid.new_zeros(num_rows, 1, compress_ratio)
    previous_page_valid = torch.cat((false_valid, current_page_valid[:, :-1]), dim=1)
    row_has_previous = (
        torch.arange(m, device=device).view(1, m, 1) >= 1
    )  # previous_positions >= 0 iff row >= 1
    previous_valid = row_has_previous & previous_page_valid

    previous_kv = previous_kv_state[..., :head_dim]
    previous_gate = previous_gate_state[..., :head_dim]
    previous_gate = previous_gate + compressor_ape[:, :head_dim].to(
        device=previous_gate.device, dtype=dtype
    )
    previous_kv = torch.where(previous_valid.unsqueeze(-1), previous_kv, previous_kv.new_zeros(()))
    previous_gate = torch.where(
        previous_valid.unsqueeze(-1),
        previous_gate,
        previous_gate.new_full((), -1.0e20),
    )

    current_kv = current_kv_state[..., head_dim : 2 * head_dim]
    current_gate = current_gate_state[..., head_dim : 2 * head_dim]
    current_gate = current_gate + compressor_ape[:, head_dim : 2 * head_dim].to(
        device=current_gate.device, dtype=dtype
    )
    kv = torch.cat((previous_kv, current_kv), dim=2)
    gate = torch.cat((previous_gate, current_gate), dim=2)

    pooled = torch.ops.auto_deploy.deepseek_v4_compress_pool(kv, gate)
    pooled = _rms_norm_ref(pooled, compressor_norm_weight, rms_norm_eps)
    row_position_id = row_position_id.to(torch.long).clamp(min=0, max=cos_table.shape[0] - 1)
    cos = cos_table[row_position_id]
    sin = sin_table[row_position_id]
    return _apply_compressed_rope_and_quantize(pooled, cos, sin, rope_dim, rotate=rotate)


def _compressed_row_update_metadata(
    input_pos: torch.Tensor,
    position_ids: torch.Tensor,
    seq_idx: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    max_compressed_len: int,
    want_pos_map: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Per-decode-row compressed-cache update metadata for one compression ratio.

    Returns ``(row_valid, row_position_id, mhc_page_ids, mhc_page_offsets,
    pos_page_ids, pos_page_offsets)``.  The first four -- whether this decode step
    completes a compressed row, the query-relative RoPE position of that row, and
    the ``(page_id, page_offset)`` write address of the completed row in ``mhc_cache``
    -- are identical for every layer of the given ``compress_ratio`` because they
    depend only on ``input_pos`` / ``position_ids`` and the shared page table.  The
    dense (ratio-128, non-overlap) update additionally reads the ``[num_seq, ratio]``
    compressor-token page map (``want_pos_map=True``); the overlap (ratio-4) update
    reads through its separately hoisted overlap band map instead, so it requests
    ``want_pos_map=False`` and the pos map is ``None``.

    This is the single source of truth shared by the per-layer
    ``_update_decode_compressed_caches`` (when the metadata is not hoisted) and the
    once-per-forward ``deepseek_v4_sparse_prepare_decode_page_addr`` hoist, so the
    hoisted values are bit-identical to the per-layer computation they replace.
    """
    input_pos = input_pos.to(torch.long)
    position_ids = position_ids.to(torch.long)
    old_completed = input_pos // compress_ratio
    new_completed = (input_pos + 1) // compress_ratio
    row_valid = (new_completed > old_completed) & (old_completed < max_compressed_len)
    row_idx = old_completed.clamp(min=0, max=max_compressed_len - 1)
    row_position_id = position_ids - (input_pos - row_idx * compress_ratio)
    row_logical_pos = row_idx * compress_ratio
    mhc_page_ids, mhc_page_offsets, _ = _page_ids_and_offsets_from_tpb(
        tokens_per_block, seq_idx, row_logical_pos, cu_num_pages, cache_loc
    )
    pos_page_ids = None
    pos_page_offsets = None
    if want_pos_map:
        offsets = torch.arange(compress_ratio, dtype=torch.long, device=input_pos.device)
        positions = row_logical_pos.unsqueeze(1) + offsets.view(1, -1)  # [num_seq, ratio]
        pos_page_ids, pos_page_offsets, _ = _page_ids_and_offsets_from_tpb(
            tokens_per_block, seq_idx, positions, cu_num_pages, cache_loc
        )
    return (
        row_valid,
        row_position_id,
        mhc_page_ids,
        mhc_page_offsets,
        pos_page_ids,
        pos_page_offsets,
    )


def _update_decode_compressed_caches(
    compressor_kv_decode: torch.Tensor,
    compressor_gate_decode: torch.Tensor,
    position_ids_decode: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    compress_ratio: int,
    max_compressed_len: int,
    overlap_page_map: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    update_meta: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]
    ] = None,
) -> None:
    mode = _compression_mode(compress_ratio)
    if not mode.enabled or compressor_kv_decode.numel() == 0:
        return

    state_dim = int(compressor_kv_decode.shape[-1])
    head_dim = state_dim // mode.channels
    # The compressor kv/gate current-token rows are written by the caller's fused
    # ``_fused_current_token_store`` (idea_0006) -- together with the SWA and
    # indexer-compressor rows -- before this helper runs, so they are already in
    # ``compressor_kv_cache`` / ``compressor_gate_cache`` when the compressed-row
    # reconstruction below reads them. The mhc_cache write further down targets a
    # compressed row (``row_logical_pos``), a different logical position, and keeps
    # its own per-layer translation + validity-masked store.

    # Per-row update metadata (row_valid / query-relative rope position / mhc write
    # address; and, for the dense ratio-128 layers, the [num_seq, ratio] compressor
    # read page map). Every layer of this compression ratio resolves identical values,
    # so ``deepseek_v4_sparse_prepare_decode_page_addr`` hoists them once per forward
    # and threads them in via ``update_meta`` (idea_0044); when absent (eager/CPU, or no
    # prepare op) they are computed here. ``row_idx`` / ``row_logical_pos`` are cheap and
    # consumed only by the eager fallback below, so they stay local to that branch.
    if update_meta is not None:
        (
            row_valid,
            row_position_id,
            mhc_page_ids,
            mhc_page_offsets,
            hoisted_pos_page_ids,
            hoisted_pos_page_offsets,
        ) = update_meta
    else:
        old_completed = input_pos // compress_ratio
        new_completed = (input_pos + 1) // compress_ratio
        row_valid = (new_completed > old_completed) & (old_completed < max_compressed_len)
        row_idx = old_completed.clamp(min=0, max=max_compressed_len - 1)
        row_position_id = position_ids_decode.to(torch.long) - (
            input_pos - row_idx * compress_ratio
        )
        row_logical_pos = row_idx * compress_ratio
        # The mhc read (previous_rows, eager fallback only) and the write below both target
        # ``row_logical_pos`` of mhc_cache, so they resolve to one page map. Compute it once
        # and feed both the read and the write -- the second translation chain is removed.
        mhc_page_ids, mhc_page_offsets, _ = _decode_page_ids_and_offsets(
            mhc_cache, seq_idx, row_logical_pos, cu_num_pages, cache_loc
        )
        hoisted_pos_page_ids = None
        hoisted_pos_page_offsets = None

    # Fused ratio-4 path (idea_0007): the overlap reconstruction (gather / slice /
    # ape-add / where / cat / pool / rmsnorm), the rope/fp8-quant tail
    # (``_apply_compressed_rope_and_quantize``, rotate=False), the ``cos``/``sin``
    # gathers and the validity-masked store collapse into two kernels reading the
    # hoisted overlap band map. Requires the fp8 nope slice to be block_size=64
    # aligned and a non-empty even rope dim; otherwise fall back to the op-by-op
    # path below. Ratio-128 (non-overlap) always takes the fallback -- its
    # ``[ratio, head_dim]`` pool tile is too large for one program.
    nope_dim = head_dim - rope_dim
    if (
        mode.overlap
        and _HAS_TRITON
        and mhc_cache.is_cuda
        and overlap_page_map is not None
        and rope_dim > 0
        and rope_dim % 2 == 0
        and nope_dim > 0
        and nope_dim % 64 == 0
    ):
        row_position_id_clamped = row_position_id.to(torch.long).clamp(
            min=0, max=cos_table.shape[0] - 1
        )
        _fused_compressed_row_update_r4(
            compressor_kv_cache,
            compressor_gate_cache,
            overlap_page_map,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            row_position_id_clamped,
            row_valid,
            mhc_page_ids,
            mhc_page_offsets,
            mhc_cache,
            rms_norm_eps,
            compress_ratio,
            head_dim,
            rope_dim,
            compressor_kv_decode.dtype,
        )
        return

    # Fused ratio-128 (dense, non-overlap) path (idea_0039): the ratio-4 kernel fuses its
    # whole reconstruction in one program per row, but the dense ``[ratio, head_dim]`` pool
    # tile is too large for that, so the pool is D-tiled (``_paged_compress_pool``), RMSNorm
    # is a separate ``head_dim`` reduction (``_compressor_rms_norm``) and the rope/fp8/store
    # tail is the shared ``_launch_compressed_rope_fp8_store``.  Requires the fp8 nope slice
    # to be block_size=64 aligned, a non-empty even rope dim, and contiguous paged compressor
    # caches (the kernel indexes them with the contiguous ``T*S`` / ``S`` strides); otherwise
    # fall back to the op-by-op path below.  The dense reconstruction never validity-masks the
    # gate (the reference discards ``page_valid``), so the pooled row matches
    # ``gather + ape + deepseek_v4_compress_pool`` to <=1 ULP (fp32 ratio-axis reduction order).
    if (
        not mode.overlap
        and _HAS_TRITON
        and mhc_cache.is_cuda
        and compressor_kv_cache.is_contiguous()
        and compressor_gate_cache.is_contiguous()
        and rope_dim > 0
        and rope_dim % 2 == 0
        and nope_dim > 0
        and nope_dim % 64 == 0
    ):
        if hoisted_pos_page_ids is not None:
            # Hoisted once per forward by the prepare op and shared by every ratio-128
            # layer (they all read the identical [num_seq, ratio] compressor positions).
            pos_page_ids = hoisted_pos_page_ids
            pos_page_offsets = hoisted_pos_page_offsets
        else:
            offsets = torch.arange(compress_ratio, dtype=torch.long, device=row_idx.device)
            anchor = row_idx.to(torch.long) * compress_ratio
            positions = anchor.unsqueeze(1) + offsets.view(1, -1)  # [N, ratio]
            # kv and gate caches share one page table + tokens_per_block, so the reads at
            # these positions resolve to one page map -- bit-identical to the fallback's
            # translation.
            pos_page_ids, pos_page_offsets, _ = _decode_page_ids_and_offsets(
                compressor_kv_cache, seq_idx, positions, cu_num_pages, cache_loc
            )
        row_position_id_clamped = row_position_id.to(torch.long).clamp(
            min=0, max=cos_table.shape[0] - 1
        )
        _fused_compressed_row_update_r128(
            compressor_kv_cache,
            compressor_gate_cache,
            pos_page_ids,
            pos_page_offsets,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            row_position_id_clamped,
            row_valid,
            mhc_page_ids,
            mhc_page_offsets,
            mhc_cache,
            rms_norm_eps,
            compress_ratio,
            head_dim,
            rope_dim,
            compressor_kv_decode.dtype,
        )
        return

    # Eager fallback (CPU / non-Triton / unsupported shapes). The hoisted metadata does
    # not carry ``row_idx`` / ``row_logical_pos`` (only this path consumes them and they
    # are cheap), so recompute them locally when the metadata was hoisted.
    if update_meta is not None:
        old_completed = input_pos // compress_ratio
        row_idx = old_completed.clamp(min=0, max=max_compressed_len - 1)
        row_logical_pos = row_idx * compress_ratio
    compressed_rows = _batched_compressed_rows_from_paged_state(
        compressor_kv_cache,
        compressor_gate_cache,
        seq_idx,
        row_idx,
        row_position_id,
        cu_num_pages,
        cache_loc,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        compress_ratio,
        head_dim,
        compressor_kv_decode.dtype,
        overlap_page_map=overlap_page_map,
    )
    if _HAS_TRITON and mhc_cache.is_cuda:
        # Validity-masked paged store. ``row_valid`` is true only on the ~1-in-4 decode
        # steps that complete a compressed row (and is identical across every layer within
        # a step, since it depends only on ``input_pos``). The prior code unconditionally
        # gathered the previous mhc row and ``torch.where``-selected it back for the other
        # ~3-in-4 steps -- a per-layer read + where + write-back that runs every captured
        # cudagraph step regardless of ``row_valid``. Fold the conditional into the store:
        # valid rows write ``compressed_rows`` exactly as the prior index_put; invalid rows
        # store nothing, leaving the slot byte-identical to reading and writing it back.
        _masked_write_decode_cache_rows(
            mhc_cache,
            compressed_rows.to(mhc_cache.dtype),
            row_valid,
            mhc_page_ids,
            mhc_page_offsets,
        )
    else:
        previous_rows, _ = _decode_cache_rows_from_positions(
            mhc_cache,
            seq_idx,
            row_logical_pos,
            cu_num_pages,
            cache_loc,
            mhc_cache.dtype,
            (mhc_page_ids, mhc_page_offsets, None),
        )
        rows_to_write = torch.where(
            row_valid.unsqueeze(-1),
            compressed_rows.to(mhc_cache.dtype),
            previous_rows,
        )
        _write_decode_cache_rows(
            mhc_cache,
            rows_to_write,
            seq_idx,
            row_logical_pos,
            cu_num_pages,
            cache_loc,
            mhc_page_ids,
            mhc_page_offsets,
        )


def _select_decode_ratio4_indexer_rows(
    q_index: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    position_ids_decode: torch.Tensor,
    index_topk: int,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rms_norm_eps: float,
    rope_dim: int,
    max_compressed_len: int,
    full_page_map: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_topk <= 0:
        empty_rows = torch.empty(q_index.shape[0], 0, dtype=torch.int64, device=q_index.device)
        empty_valid = torch.empty(q_index.shape[0], 0, dtype=torch.bool, device=q_index.device)
        return empty_rows, empty_valid

    candidate_rows = torch.arange(max_compressed_len, dtype=torch.long, device=q_index.device)
    candidate_rows = candidate_rows.view(1, -1).expand(q_index.shape[0], -1)
    row_position_id = position_ids_decode.unsqueeze(1) - (
        input_pos.unsqueeze(1) - candidate_rows * 4
    )
    index_head_dim = int(q_index.shape[-1])
    # Every candidate row 0..max_compressed_len-1 is requested for each decode row, so the
    # overlap "previous" block of a row is the "current" block of the preceding row. Gather
    # the full contiguous token range once per cache and derive previous via a row shift,
    # instead of four scattered index_select gathers (the decode-dominant kernel).
    index_k = _batched_overlap_compressed_rows_fullrange(
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        seq_idx,
        row_position_id,
        cu_num_pages,
        cache_loc,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        cos_table,
        sin_table,
        rms_norm_eps,
        rope_dim,
        4,
        index_head_dim,
        max_compressed_len,
        q_index.dtype,
        rotate=True,
        full_page_map=full_page_map,
    )
    # ``q_index``/``indexer_weights`` are replicated across TP ranks (the indexer
    # index-score projection is no longer head-sharded; see DeepseekV4Indexer),
    # so the head reduction already sums over all index heads -- no all_reduce needed.
    visible_len = ((input_pos + 1) // 4).clamp(max=max_compressed_len)
    if (
        _HAS_TRITON
        and q_index.is_cuda
        and q_index.dtype in (torch.float16, torch.bfloat16)
        and index_k.dtype == q_index.dtype
        and index_head_dim >= 16
    ):
        # Fused matmul + relu + weighted head reduction + visibility mask (idea_0004):
        # the [N, H, C] head-by-candidate score and the separate masked [N, C] tensor
        # are never materialized -- one kernel emits the masked score row fed straight
        # into the top-k below. Scores match the eager chain to within one ULP so the
        # selected rows / sort order are preserved.
        index_score = _fused_index_score(
            q_index, index_k, indexer_weights, visible_len, max_compressed_len
        )
    else:
        index_score = torch.matmul(q_index, index_k.transpose(-1, -2)).float()
        index_score = (index_score.relu() * indexer_weights.float().unsqueeze(-1)).sum(dim=1)
        visible = candidate_rows < visible_len.unsqueeze(1)
        index_score = index_score.masked_fill(~visible, float("-inf"))
    topk_count = min(index_topk, max_compressed_len)
    if _HAS_TRITON and index_score.is_cuda and index_score.dtype == torch.float32:
        # One-launch exact top-k select (idea_0046): replaces the fat
        # gatherTopK / radixSortKVInPlace pair, the decomposed isfinite/where
        # fixups and the short-history pad path with a single kernel emitting the
        # padded rows/validity directly.  Byte-identical to the eager tail below,
        # tie order included (see _dsv4_topk_select_kernel + the op unit test).
        return _fused_topk_select(index_score, index_topk, topk_count)
    topk_values, topk_rows = index_score.topk(topk_count, dim=-1)
    topk_valid = torch.isfinite(topk_values)
    topk_rows = torch.where(topk_valid, topk_rows.to(torch.int64), torch.full_like(topk_rows, -1))
    if topk_count < index_topk:
        pad_shape = (q_index.shape[0], index_topk - topk_count)
        row_pad = torch.full(pad_shape, -1, dtype=torch.int64, device=q_index.device)
        valid_pad = torch.zeros(pad_shape, dtype=torch.bool, device=q_index.device)
        topk_rows = torch.cat((topk_rows, row_pad), dim=-1)
        topk_valid = torch.cat((topk_valid, valid_pad), dim=-1)
    return topk_rows, topk_valid


def _decode_compressed_cache_attention(
    q_decode: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_decode: torch.Tensor,
    indexer_q_decode: torch.Tensor,
    indexer_weights_decode: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    position_ids_decode: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    max_compressed_len: int,
    softmax_scale: float,
    rms_norm_eps: float,
    rope_dim: int,
    full_page_map: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    # Select the compressed candidate rows (and their pre-page validity) without
    # touching the paged caches yet: the indexer top-k for ratio-4, the full
    # ``arange`` range for the ratio-128 dense case.
    mode = _compression_mode(compress_ratio)
    if mode.uses_indexer:
        index_topk = max(int(topk_decode.shape[-1]) - int(window_size), 0)
        selected_rows, compressed_valid = _select_decode_ratio4_indexer_rows(
            indexer_q_decode,
            indexer_weights_decode,
            indexer_compressor_kv_cache,
            indexer_compressor_gate_cache,
            seq_idx,
            input_pos,
            position_ids_decode,
            index_topk,
            cu_num_pages,
            cache_loc,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            cos_table,
            sin_table,
            rms_norm_eps,
            rope_dim,
            max_compressed_len,
            full_page_map=full_page_map,
        )
    else:
        candidate_rows = torch.arange(
            max_compressed_len,
            dtype=torch.long,
            device=q_decode.device,
        )
        selected_rows = candidate_rows.view(1, -1).expand(q_decode.shape[0], -1)
        compressed_len = ((input_pos + 1) // compress_ratio).clamp(max=max_compressed_len)
        compressed_valid = selected_rows < compressed_len.unsqueeze(1)

    if _HAS_TRITON and swa_cache.is_cuda and mhc_cache.is_cuda:
        # Fold the local/compressed page-map translation, the two paged row gathers,
        # the selected_kv / valid_rows concatenation and the attend arange/where into
        # one paged assemble kernel, then attend directly on the emitted rel_topk.
        selected_kv, rel_topk = _fused_assemble_selected_kv(
            swa_cache,
            mhc_cache,
            selected_rows,
            compressed_valid,
            input_pos,
            seq_idx,
            cu_num_pages,
            cache_loc,
            window_size,
            compress_ratio,
            q_decode.dtype,
        )
        return _decode_attention_from_selected(
            q_decode,
            selected_kv,
            rel_topk,
            attn_sink,
            softmax_scale,
        )

    # Eager fallback (CPU / no-Triton): materialize selected_kv via row gathers + cat.
    local_kv, local_valid = _decode_local_cache_rows(
        swa_cache,
        seq_idx,
        input_pos,
        cu_num_pages,
        cache_loc,
        window_size,
        q_decode.dtype,
    )
    compressed_positions = selected_rows.clamp(min=0) * compress_ratio
    compressed_kv, page_valid = _decode_cache_rows_from_positions(
        mhc_cache,
        seq_idx,
        compressed_positions,
        cu_num_pages,
        cache_loc,
        q_decode.dtype,
    )
    compressed_valid = compressed_valid & page_valid & (selected_rows >= 0)
    selected_kv = torch.cat((local_kv, compressed_kv), dim=1)
    valid_rows = torch.cat((local_valid, compressed_valid), dim=1)
    return _decode_attention_from_rows(
        q_decode,
        selected_kv,
        valid_rows,
        attn_sink,
        softmax_scale,
    )


def _deepseek_v4_sparse_attention_decode_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    cur_page_ids: Optional[torch.Tensor],
    cur_page_offsets: Optional[torch.Tensor],
    ovl_page_ids: Optional[torch.Tensor],
    ovl_page_offsets: Optional[torch.Tensor],
    ovl_valid: Optional[torch.Tensor],
    full_page_ids: Optional[torch.Tensor],
    full_page_offsets: Optional[torch.Tensor],
    full_valid: Optional[torch.Tensor],
    r4_row_valid: Optional[torch.Tensor],
    r4_row_position_id: Optional[torch.Tensor],
    r4_mhc_page_ids: Optional[torch.Tensor],
    r4_mhc_page_offsets: Optional[torch.Tensor],
    r128_row_valid: Optional[torch.Tensor],
    r128_row_position_id: Optional[torch.Tensor],
    r128_mhc_page_ids: Optional[torch.Tensor],
    r128_mhc_page_offsets: Optional[torch.Tensor],
    r128_pos_page_ids: Optional[torch.Tensor],
    r128_pos_page_offsets: Optional[torch.Tensor],
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    num_decode: int,
    softmax_scale: float,
    window_size: Optional[int],
    compress_ratio: int,
    max_compressed_len: Optional[int],
    rms_norm_eps: float,
    rope_dim: Optional[int],
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    q_flat = q.reshape(-1, *q.shape[2:])
    q_decode = q_flat[:num_decode]
    kv_decode = _flatten_decode_tokens(kv, num_decode)
    topk_decode = _flatten_decode_tokens(topk_idxs, num_decode)
    del slot_idx
    seq_idx_decode = torch.arange(num_decode, dtype=torch.long, device=input_pos.device)
    input_pos_decode = input_pos.reshape(-1)[:num_decode].to(torch.long)
    position_ids_decode = position_ids.reshape(-1)[:num_decode].to(torch.long)
    # Current-token write address, hoisted once per forward by
    # ``deepseek_v4_sparse_prepare_decode_page_addr`` and shared by every layer.
    # Slice to the active decode sequences; ``None`` falls back to per-layer
    # translation inside ``_write_decode_cache_rows``.
    if cur_page_ids is not None:
        cur_page_ids = cur_page_ids[:num_decode]
        cur_page_offsets = cur_page_offsets[:num_decode]

    # Ratio-4 compressed-row / full-range page maps, hoisted once per forward by
    # ``deepseek_v4_sparse_prepare_decode_page_addr`` and shared by every ratio-4
    # layer (they resolve the identical (seq_idx, positions) addresses). Slice to
    # the active decode sequences; ``None`` falls back to per-layer translation
    # inside the batched compressor helpers.
    overlap_page_map = None
    full_page_map = None
    if compress_ratio == 4 and ovl_page_ids is not None and full_page_ids is not None:
        overlap_page_map = (
            ovl_page_ids[:num_decode],
            ovl_page_offsets[:num_decode],
            ovl_valid[:num_decode],
        )
        full_page_map = (
            full_page_ids[:num_decode],
            full_page_offsets[:num_decode],
            full_valid[:num_decode],
        )

    # Compressed-cache UPDATE metadata (idea_0044), hoisted once per forward by
    # ``deepseek_v4_sparse_prepare_decode_page_addr`` and shared by every layer of this
    # ratio class. Select the bundle matching this layer's ratio and slice to the active
    # decode sequences; ``None`` falls back to per-layer computation inside
    # ``_update_decode_compressed_caches`` (the R4 bundle carries no compressor read
    # page map -- that path reads through ``overlap_page_map`` instead).
    update_meta = None
    if compress_ratio == _COMPRESS_RATIO_OVERLAP_INDEXER and r4_row_valid is not None:
        update_meta = (
            r4_row_valid[:num_decode],
            r4_row_position_id[:num_decode],
            r4_mhc_page_ids[:num_decode],
            r4_mhc_page_offsets[:num_decode],
            None,
            None,
        )
    elif compress_ratio == _COMPRESS_RATIO_DENSE and r128_row_valid is not None:
        update_meta = (
            r128_row_valid[:num_decode],
            r128_row_position_id[:num_decode],
            r128_mhc_page_ids[:num_decode],
            r128_mhc_page_offsets[:num_decode],
            r128_pos_page_ids[:num_decode],
            r128_pos_page_offsets[:num_decode],
        )

    # Current-token cache writes (idea_0006). Every DeepSeek-V4 cache listed below
    # stores the freshly produced current decode-token row at logical position
    # ``input_pos`` and shares the hoisted ``(cur_page_ids, cur_page_offsets)`` write
    # address, so a single fused paged store replaces the per-cache ``index_put``
    # scatters (SWA kv, main-compressor kv/gate, and the ratio-4 indexer-compressor
    # kv/gate). The stored rows are byte-identical to the prior per-cache writes.
    store_caches = [swa_cache]
    store_values = [kv_decode]
    compressor_kv_decode = None
    compressor_gate_decode = None
    mode = _compression_mode(compress_ratio) if compress_ratio else None
    if compress_ratio:
        assert window_size is not None
        assert max_compressed_len is not None
        assert rope_dim is not None
        compressor_kv_decode = _flatten_decode_tokens(compressor_kv, num_decode)
        compressor_gate_decode = _flatten_decode_tokens(compressor_gate, num_decode)
        # ``_update_decode_compressed_caches`` gates the whole compressed-row update
        # (both kv/gate current-token writes included) on ``compressor_kv_decode``, so
        # add the pair to the fused store under the identical condition -- keeping the
        # gate write tied to the kv presence, not filtered independently.
        if compressor_kv_decode.numel() > 0:
            store_caches += [compressor_kv_cache, compressor_gate_cache]
            store_values += [compressor_kv_decode, compressor_gate_decode]
        if mode.uses_indexer:
            indexer_compressor_kv_decode = _flatten_decode_tokens(indexer_compressor_kv, num_decode)
            indexer_compressor_gate_decode = _flatten_decode_tokens(
                indexer_compressor_gate, num_decode
            )
            store_caches += [indexer_compressor_kv_cache, indexer_compressor_gate_cache]
            store_values += [indexer_compressor_kv_decode, indexer_compressor_gate_decode]

    _fused_current_token_store(
        store_caches,
        store_values,
        seq_idx_decode,
        input_pos_decode,
        cu_num_pages,
        cache_loc,
        cur_page_ids,
        cur_page_offsets,
    )

    if compress_ratio:
        _update_decode_compressed_caches(
            compressor_kv_decode,
            compressor_gate_decode,
            position_ids_decode,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            seq_idx_decode,
            input_pos_decode,
            cu_num_pages,
            cache_loc,
            mhc_cache,
            compressor_kv_cache,
            compressor_gate_cache,
            rms_norm_eps,
            rope_dim,
            compress_ratio,
            max_compressed_len,
            overlap_page_map,
            update_meta,
        )
        indexer_q_decode = _flatten_decode_tokens(indexer_q, num_decode)
        indexer_weights_decode = _flatten_decode_tokens(indexer_weights, num_decode)
        decode_output = _decode_compressed_cache_attention(
            q_decode,
            attn_sink,
            topk_decode,
            indexer_q_decode,
            indexer_weights_decode,
            swa_cache,
            mhc_cache,
            indexer_compressor_kv_cache,
            indexer_compressor_gate_cache,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            cos_table,
            sin_table,
            seq_idx_decode,
            input_pos_decode,
            cu_num_pages,
            cache_loc,
            position_ids_decode,
            window_size,
            compress_ratio,
            max_compressed_len,
            softmax_scale,
            rms_norm_eps,
            rope_dim,
            full_page_map=full_page_map,
        )
    else:
        decode_output = _decode_topk_cache_attention(
            q_decode,
            attn_sink,
            topk_decode,
            swa_cache,
            seq_idx_decode,
            input_pos_decode,
            cu_num_pages,
            cache_loc,
            softmax_scale,
            window_size,
        )

    output_flat = torch.zeros_like(q_flat)
    output_flat[:num_decode].copy_(decode_output)
    output = output_flat.view_as(q)
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


def _cached_decode_topk_positions(
    topk_seq: torch.Tensor,
    input_pos: int,
    window_size: Optional[int],
    compress_ratio: int,
) -> torch.Tensor:
    if compress_ratio == 0 or window_size is None:
        return topk_seq

    local_window_cols = min(window_size, topk_seq.shape[-1])
    if local_window_cols == 0:
        return topk_seq

    token_offsets = torch.arange(topk_seq.shape[0], device=topk_seq.device).unsqueeze(1)
    query_positions = input_pos + token_offsets
    window_offsets = torch.arange(local_window_cols, device=topk_seq.device)
    local_topk = query_positions - local_window_cols + 1 + window_offsets
    local_topk = torch.where(local_topk < 0, -1, local_topk)
    local_topk = local_topk.to(topk_seq.dtype)
    if local_window_cols == topk_seq.shape[-1]:
        return local_topk
    return torch.cat((local_topk, topk_seq[..., local_window_cols:]), dim=-1)


def _sparse_attention_query_chunk_size(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    k_select: int,
    compute_dtype: torch.dtype,
) -> int:
    compute_element_size = torch.empty((), dtype=compute_dtype).element_size()
    logits_element_size = torch.empty((), dtype=torch.float32).element_size()
    bytes_per_token = (
        k_select * head_dim * compute_element_size
        + 3 * num_heads * (k_select + 1) * logits_element_size
        + num_heads * head_dim * compute_element_size
    )
    if bytes_per_token <= 0:
        return 1
    chunk_size = _SPARSE_ATTENTION_CHUNK_TARGET_BYTES // bytes_per_token
    chunk_size = max(1, int(chunk_size))
    chunk_size = min(chunk_size, _SPARSE_ATTENTION_MAX_CHUNK_TOKENS)
    return min(num_tokens, chunk_size)


_LOG2E = 1.4426950408889634
# Scores for masked / out-of-range / padded key slots are floored to this value
# *before* the exp2.  A large finite negative (rather than -inf) keeps the online
# softmax running-max arithmetic NaN-free when an entire query has no valid key
# (the sink-only case) — see _fused_sparse_attention_kernel.
_SPARSE_ATTN_NEG = -1.0e30

# Fused-kernel launch heuristics (tuned for the decode tpot path on Blackwell).
_SPARSE_ATTN_SM_TARGET = 132  # min SM count across supported GPUs (H100/B200)
_SPARSE_ATTN_SPLITK_MAX_TOKENS = 8  # use split-K only for decode / small batches
_SPARSE_ATTN_DECODE_HEAD_BLOCK = 8  # balances CTA count vs MMA tile efficiency (swept)
_SPARSE_ATTN_DECODE_NUM_WARPS = 8  # warps for the split-K partial kernel (swept)
_SPARSE_ATTN_REDUCE_NUM_WARPS = 8  # warps for the split-K reduce kernel (swept; 4->8
# nearly halves its tail at the D=512 partial width — the reduce loads NUM_PARTS
# fp32 acc rows of width D_BLOCK, so more warps cover the row in fewer steps)
_SPARSE_ATTN_MAX_PARTS = 32  # cap on split-K key partitions
_SPARSE_ATTN_DECODE_CTA_TARGET = 80  # split-K grid CTA budget (head_groups*num_parts):
# ~one wave of SMs on H100(132)/B200(148).  Swept: the decode attend plateaus in
# latency from ~40-80 CTAs and regresses sharply past ~160 (a 2nd SM wave), so
# HEAD_BLOCK is sized to land head_groups*num_parts near this, not above it.

# Small-head decode (head-parallelism-starved: per-rank head count <= HEAD_BLOCK, e.g.
# the DSV4-Flash TP8 per-rank H=8 shape).  Here the CTA-target heuristic above floors
# HEAD_BLOCK to 1, leaving thin M=1 split-K tiles.  Halving SEQ_BLOCK (more split-K
# parts), packing a few heads per CTA (fatter MMA M-tile), using only 4 split-K warps
# (8 over-subscribes the small tile) and 16 reduce warps (cover the D-wide partial row
# in one pass) cut the H=8/L=640/D=512 decode attend by ~4.5% (swept; idea_0020).
# Larger H (e.g. full H=64) already has enough head-parallel CTAs and is *not* routed
# here — it regresses under this config, so the predicate is strictly H <= MAX_HEADS.
_SPARSE_ATTN_DECODE_SMALL_H_MAX_HEADS = 8
_SPARSE_ATTN_DECODE_SMALL_H_SEQ_BLOCK = 32
_SPARSE_ATTN_DECODE_SMALL_H_HEAD_BLOCK = 4
_SPARSE_ATTN_DECODE_SMALL_H_NUM_WARPS = 4  # split-K partial warps for the small M-tile
_SPARSE_ATTN_DECODE_SMALL_H_REDUCE_NUM_WARPS = 16


if _HAS_TRITON:

    @triton.jit
    def _fused_sparse_attention_kernel(
        q_ptr,  # [num_tokens, num_heads, D]
        kv_ptr,  # [batch_size, kv_rows, D]
        topk_ptr,  # [num_tokens, k_select] int
        sink_ptr,  # [num_heads]
        batch_ptr,  # [num_tokens] int -> batch row of kv
        out_ptr,  # [num_tokens, num_heads, D]
        num_heads,
        kv_rows,
        k_select,
        SCALE_LOG2: tl.constexpr,  # softmax_scale * log2(e)
        D: tl.constexpr,
        D_BLOCK: tl.constexpr,  # next_pow2(D)
        SEQ_BLOCK: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
    ):
        """Fused on-the-fly selected-KV sparse attention (flash-MQA style).

        Grid: (num_tokens, cdiv(num_heads, HEAD_BLOCK)).  Each program handles one
        query token and HEAD_BLOCK heads (which all attend the *same* gathered KV
        rows — K==V), so the selected/compressed KV rows are read from HBM exactly
        once and reused across heads and across the score/output matmuls.  No fp32
        [num_tokens, k_select, D] tensor is ever materialized.  fp32 online softmax
        matches the reference reduction; the per-head ``attn_sink`` logit is folded
        into the denominator (no value contribution) after the key loop.
        """
        token_id = tl.program_id(0)
        head_group = tl.program_id(1)
        head_start = head_group * HEAD_BLOCK

        # Inlined literals: jit kernels cannot read module-level globals.
        NEG: tl.constexpr = -1.0e30  # masked-key score floor (NaN-safe vs -inf)
        LOG2E: tl.constexpr = 1.4426950408889634

        batch_id = tl.load(batch_ptr + token_id).to(tl.int64)
        head_offsets = tl.arange(0, HEAD_BLOCK)
        heads = head_start + head_offsets
        head_mask = heads < num_heads
        d_offsets = tl.arange(0, D_BLOCK)
        d_mask = d_offsets < D

        q_base = token_id.to(tl.int64) * num_heads * D
        q_ptrs = q_ptr + q_base + heads[:, None] * D + d_offsets[None, :]
        q = tl.load(q_ptrs, mask=head_mask[:, None] & d_mask[None, :], other=0.0)

        m_i = tl.full([HEAD_BLOCK], NEG, dtype=tl.float32)
        l_i = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
        acc = tl.zeros([HEAD_BLOCK, D_BLOCK], dtype=tl.float32)

        kv_batch_base = batch_id * kv_rows
        topk_base = token_id.to(tl.int64) * k_select

        for start in range(0, k_select, SEQ_BLOCK):
            kcol = start + tl.arange(0, SEQ_BLOCK)
            kcol_mask = kcol < k_select
            idx = tl.load(topk_ptr + topk_base + kcol, mask=kcol_mask, other=-1).to(tl.int64)
            valid = (idx >= 0) & (idx < kv_rows) & kcol_mask
            idx_c = tl.minimum(tl.maximum(idx, 0), kv_rows - 1)
            kv_ptrs = kv_ptr + (kv_batch_base + idx_c)[:, None] * D + d_offsets[None, :]
            kvb = tl.load(kv_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0)

            scores = tl.dot(q, tl.trans(kvb)).to(tl.float32) * SCALE_LOG2  # [HB, SB]
            scores = tl.where(valid[None, :], scores, NEG)

            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(scores - m_new[:, None])
            p = tl.where(valid[None, :], p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kvb.dtype), kvb).to(tl.float32)
            m_i = m_new

        # Fold the per-head sink logit (raw, not scaled) into the denominator only.
        sink = tl.load(sink_ptr + heads, mask=head_mask, other=0.0).to(tl.float32) * LOG2E
        m_new = tl.maximum(m_i, sink)
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.math.exp2(sink - m_new)
        acc = acc * alpha[:, None]

        out = acc / tl.maximum(l_i, 1e-38)[:, None]
        out_ptrs = out_ptr + q_base + heads[:, None] * D + d_offsets[None, :]
        tl.store(
            out_ptrs,
            out.to(out_ptr.dtype.element_ty),
            mask=head_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _fused_sparse_attention_splitk_kernel(
        q_ptr,  # [num_tokens, num_heads, D]
        kv_ptr,  # [batch_size, kv_rows, D]
        topk_ptr,  # [num_tokens, k_select] int
        batch_ptr,  # [num_tokens] int -> batch row of kv
        ws_acc_ptr,  # [num_tokens, num_heads, NUM_PARTS, D_BLOCK] fp32
        ws_ml_ptr,  # [num_tokens, num_heads, NUM_PARTS, 2] fp32 (m, l)
        num_heads,
        kv_rows,
        k_select,
        SCALE_LOG2: tl.constexpr,
        D: tl.constexpr,
        D_BLOCK: tl.constexpr,
        SEQ_BLOCK: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
        NUM_PARTS: tl.constexpr,
    ):
        """Split-K partial of the fused sparse attend (no sink, no normalization).

        Grid: (num_tokens, cdiv(num_heads, HEAD_BLOCK), NUM_PARTS).  Each program
        scores a contiguous slice of the selected-key columns and writes its partial
        (acc, m, l) to workspace.  Splitting the key reduction across NUM_PARTS CTAs
        fills the GPU at decode (few tokens, many heads sharing one KV) where pure
        head/token parallelism leaves most SMs idle.  Reduction + sink fold happen in
        _fused_sparse_attention_reduce_kernel.
        """
        NEG: tl.constexpr = -1.0e30

        token_id = tl.program_id(0)
        head_group = tl.program_id(1)
        part_id = tl.program_id(2)
        head_start = head_group * HEAD_BLOCK

        batch_id = tl.load(batch_ptr + token_id).to(tl.int64)
        head_offsets = tl.arange(0, HEAD_BLOCK)
        heads = head_start + head_offsets
        head_mask = heads < num_heads
        d_offsets = tl.arange(0, D_BLOCK)
        d_mask = d_offsets < D

        q_base = token_id.to(tl.int64) * num_heads * D
        q_ptrs = q_ptr + q_base + heads[:, None] * D + d_offsets[None, :]
        q = tl.load(q_ptrs, mask=head_mask[:, None] & d_mask[None, :], other=0.0)

        m_i = tl.full([HEAD_BLOCK], NEG, dtype=tl.float32)
        l_i = tl.zeros([HEAD_BLOCK], dtype=tl.float32)
        acc = tl.zeros([HEAD_BLOCK, D_BLOCK], dtype=tl.float32)

        kv_batch_base = batch_id * kv_rows
        topk_base = token_id.to(tl.int64) * k_select

        total_blocks = tl.cdiv(k_select, SEQ_BLOCK)
        blocks_per_part = tl.cdiv(total_blocks, NUM_PARTS)
        part_start_block = part_id * blocks_per_part
        part_end_block = tl.minimum(part_start_block + blocks_per_part, total_blocks)

        for block_id in range(part_start_block, part_end_block):
            kcol = block_id * SEQ_BLOCK + tl.arange(0, SEQ_BLOCK)
            kcol_mask = kcol < k_select
            idx = tl.load(topk_ptr + topk_base + kcol, mask=kcol_mask, other=-1).to(tl.int64)
            valid = (idx >= 0) & (idx < kv_rows) & kcol_mask
            idx_c = tl.minimum(tl.maximum(idx, 0), kv_rows - 1)
            kv_ptrs = kv_ptr + (kv_batch_base + idx_c)[:, None] * D + d_offsets[None, :]
            kvb = tl.load(kv_ptrs, mask=valid[:, None] & d_mask[None, :], other=0.0)

            scores = tl.dot(q, tl.trans(kvb)).to(tl.float32) * SCALE_LOG2
            scores = tl.where(valid[None, :], scores, NEG)
            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(scores - m_new[:, None])
            p = tl.where(valid[None, :], p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(kvb.dtype), kvb).to(tl.float32)
            m_i = m_new

        acc_ptrs = (
            ws_acc_ptr
            + (token_id.to(tl.int64) * num_heads + heads[:, None]) * NUM_PARTS * D_BLOCK
            + part_id * D_BLOCK
            + d_offsets[None, :]
        )
        tl.store(acc_ptrs, acc, mask=head_mask[:, None] & d_mask[None, :])
        ml_base = (
            ws_ml_ptr + (token_id.to(tl.int64) * num_heads + heads) * NUM_PARTS * 2 + part_id * 2
        )
        tl.store(ml_base, m_i, mask=head_mask)
        tl.store(ml_base + 1, l_i, mask=head_mask)

    @triton.jit
    def _fused_sparse_attention_reduce_kernel(
        ws_acc_ptr,  # [num_tokens, num_heads, NUM_PARTS, D_BLOCK] fp32
        ws_ml_ptr,  # [num_tokens, num_heads, NUM_PARTS, 2] fp32
        sink_ptr,  # [num_heads]
        out_ptr,  # [num_tokens, num_heads, D]
        num_heads,
        D: tl.constexpr,
        D_BLOCK: tl.constexpr,
        NUM_PARTS: tl.constexpr,
    ):
        """Combine NUM_PARTS partials, fold the sink logit, normalize, write output.

        Grid: (num_tokens, num_heads).
        """
        token_id = tl.program_id(0)
        head_id = tl.program_id(1)
        LOG2E: tl.constexpr = 1.4426950408889634

        d_offsets = tl.arange(0, D_BLOCK)
        d_mask = d_offsets < D
        base = token_id.to(tl.int64) * num_heads + head_id

        ml0 = ws_ml_ptr + base * NUM_PARTS * 2
        m_cur = tl.load(ml0)
        l_cur = tl.load(ml0 + 1)
        acc_cur = tl.load(
            ws_acc_ptr + base * NUM_PARTS * D_BLOCK + d_offsets, mask=d_mask, other=0.0
        )

        for p in tl.static_range(1, NUM_PARTS):
            mlp = ws_ml_ptr + base * NUM_PARTS * 2 + p * 2
            m_p = tl.load(mlp)
            l_p = tl.load(mlp + 1)
            acc_p = tl.load(
                ws_acc_ptr + base * NUM_PARTS * D_BLOCK + p * D_BLOCK + d_offsets,
                mask=d_mask,
                other=0.0,
            )
            m_new = tl.maximum(m_cur, m_p)
            a = tl.math.exp2(m_cur - m_new)
            b = tl.math.exp2(m_p - m_new)
            l_cur = l_cur * a + l_p * b
            acc_cur = acc_cur * a + acc_p * b
            m_cur = m_new

        sink = tl.load(sink_ptr + head_id).to(tl.float32) * LOG2E
        m_new = tl.maximum(m_cur, sink)
        a = tl.math.exp2(m_cur - m_new)
        l_cur = l_cur * a + tl.math.exp2(sink - m_new)
        acc_cur = acc_cur * a

        out = acc_cur / tl.maximum(l_cur, 1e-38)
        out_ptrs = out_ptr + base * D + d_offsets
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=d_mask)

    @triton.jit
    def _decode_local_window_pagemap_kernel(
        input_pos_ptr,  # [N] int -- current decode position per sequence
        seq_idx_ptr,  # [N] int -- sequence row per decode token
        cu_num_pages_ptr,  # [num_seq + 1] -- prefix-sum page-table offsets
        cache_loc_ptr,  # [total_pages] -- physical page id per page-table slot
        page_ids_ptr,  # [N, W] int64 out
        page_offsets_ptr,  # [N, W] int64 out
        valid_ptr,  # [N, W] bool out
        n_elements,  # N * W
        W,  # window_size (row stride of the [N, W] grid)
        TOKENS_PER_BLOCK,
        CACHE_LOC_MAX,  # cache_loc.numel() - 1
        BLOCK: tl.constexpr,
    ):
        """One-launch local-window page map: position gen + validity + translate.

        Fuses the per-decode-step local-window integer chain -- the position
        generation of ``_decode_local_cache_rows`` (arange / sub / add / two
        compares / boolean-and) followed by the page-address translation of
        ``_decode_page_ids_and_offsets`` (clamp / floordiv / remainder / the two
        ``cu_num_pages`` lookups / add / compare / boolean-and / where / clamp /
        ``cache_loc`` lookup) -- into a single kernel.  The final
        ``swa_cache[page_ids, page_offsets]`` row gather is left to the caller
        unchanged.  All divisions/remainders act on non-negative operands, so the
        produced ``(page_ids, page_offsets)`` addresses and the combined validity
        mask are byte-identical to the reference element-wise chain.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        n = offs // W
        w = offs % W
        input_pos = tl.load(input_pos_ptr + n, mask=mask, other=0).to(tl.int64)
        seq_idx = tl.load(seq_idx_ptr + n, mask=mask, other=0).to(tl.int64)

        # Local-window position generation (mirror _decode_local_cache_rows):
        # positions = input_pos - window_size + 1 + arange(window_size).
        pos = input_pos - W + 1 + w
        valid_pos = (pos >= 0) & (pos <= input_pos)

        # Page-address translation (mirror _decode_page_ids_and_offsets).
        safe_pos = tl.maximum(pos, 0)
        page_ordinal = safe_pos // TOKENS_PER_BLOCK
        page_offset = safe_pos % TOKENS_PER_BLOCK
        page_start = tl.load(cu_num_pages_ptr + seq_idx, mask=mask, other=0).to(tl.int64)
        page_end = tl.load(cu_num_pages_ptr + seq_idx + 1, mask=mask, other=0).to(tl.int64)
        page_table_idx = page_start + page_ordinal
        page_valid = (pos >= 0) & (page_table_idx < page_end)
        safe_idx = tl.where(page_valid, page_table_idx, page_start)
        safe_idx = tl.minimum(tl.maximum(safe_idx, 0), CACHE_LOC_MAX)
        page_id = tl.load(cache_loc_ptr + safe_idx, mask=mask, other=0).to(tl.int64)

        valid = valid_pos & page_valid
        tl.store(page_ids_ptr + offs, page_id, mask=mask)
        tl.store(page_offsets_ptr + offs, page_offset, mask=mask)
        tl.store(valid_ptr + offs, valid, mask=mask)

    @triton.jit
    def _masked_paged_store_kernel(
        src_ptr,  # [N, S] contiguous, already cache dtype -- rows to conditionally store
        row_valid_ptr,  # [N] bool -- store row r iff row_valid[r]
        page_ids_ptr,  # [N] int64 -- physical page id per row
        page_offsets_ptr,  # [N] int64 -- in-page offset per row
        cache_ptr,  # [P, T, S] paged cache (mutated in place)
        stride_p,  # cache.stride(0)
        stride_t,  # cache.stride(1)
        stride_s,  # cache.stride(2)
        S,  # state_dim (row width)
        BLOCK_S: tl.constexpr,
    ):
        """Read-free validity-masked paged store.

        Replaces the decode ``mhc_cache`` compressed-row update's previous-row gather +
        ``torch.where`` + unconditional index_put write-back with a single masked store.
        For each row ``r`` the freshly compressed ``src[r]`` is written to
        ``cache[page_ids[r], page_offsets[r], :]`` only when ``row_valid[r]`` is true;
        invalid rows (the ~3-in-4 decode steps that do not complete a compressed row, and
        which are identical across every layer within a step) store nothing, leaving the
        slot byte-identical to reading the old row and writing it back. ``src`` is already
        cast to the cache dtype by the caller, so the store is a pure copy -- byte-identical
        to the prior ``cache[page_ids, page_offsets] = compressed_rows.to(cache.dtype)``.
        """
        row = tl.program_id(0)
        sblk = tl.program_id(1)
        valid = tl.load(row_valid_ptr + row).to(tl.int1)
        pid = tl.load(page_ids_ptr + row).to(tl.int64)
        poff = tl.load(page_offsets_ptr + row).to(tl.int64)
        col = sblk * BLOCK_S + tl.arange(0, BLOCK_S)
        smask = col < S
        vals = tl.load(src_ptr + row * S + col, mask=smask, other=0)
        dst = cache_ptr + pid * stride_p + poff * stride_t + col * stride_s
        tl.store(dst, vals.to(cache_ptr.dtype.element_ty), mask=smask & valid)

    @triton.jit
    def _store_current_token_row(
        row,
        sblk,
        pid,  # int64 physical page id (shared across caches)
        poff,  # int64 in-page offset (shared across caches)
        src_ptr,  # [N, S] contiguous, already cache dtype
        cache_ptr,  # [P, T, S] contiguous paged cache (mutated in place)
        S,  # this cache's row width (state_dim)
        T,  # tokens_per_block (shared cache.shape[1])
        BLOCK_S: tl.constexpr,
    ):
        """Copy ``src[row]`` into ``cache[pid, poff, :]`` for one paged cache.

        Every current-token cache shares the ``(pid, poff)`` write address, so the
        multi-cache kernel below computes it once and dispatches to this helper per
        cache. The store is a pure copy (``src`` pre-cast to the cache dtype), so it
        is byte-identical to ``cache[page_ids, page_offsets] = values.to(dtype)``.
        """
        col = sblk * BLOCK_S + tl.arange(0, BLOCK_S)
        smask = col < S
        vals = tl.load(src_ptr + row * S + col, mask=smask, other=0)
        dst = cache_ptr + pid * (T * S) + poff * S + col
        tl.store(dst, vals.to(cache_ptr.dtype.element_ty), mask=smask)

    @triton.jit
    def _multi_current_token_store_kernel(
        page_ids_ptr,  # [N] int64 -- shared current-token page id per decode row
        page_offsets_ptr,  # [N] int64 -- shared current-token in-page offset per row
        src0,
        cache0,
        S0,
        src1,
        cache1,
        S1,
        src2,
        cache2,
        S2,
        src3,
        cache3,
        S3,
        src4,
        cache4,
        S4,
        T,  # tokens_per_block, shared by every cache (cache.shape[1])
        N_CACHES: tl.constexpr,  # number of active caches (2..5)
        BLOCK_S: tl.constexpr,  # >= next_pow2(min(max_S, 1024))
    ):
        """Write the current decode token into up to 5 heterogeneous paged caches.

        Every DeepSeek-V4 current-token cache write (SWA kv, main-compressor
        kv/gate, and the ratio-4 indexer-compressor kv/gate) stores the fresh row
        at logical position ``input_pos`` of a cache that shares one page table and
        one ``tokens_per_block``, so they all resolve to the identical hoisted
        ``(page_ids, page_offsets)`` address.  This kernel folds those per-cache
        ``index_put`` scatters into one launch: grid ``(N, N_CACHES, cdiv(max_S,
        BLOCK_S))``, program ``(row, c, sblk)`` copies one row-block of cache ``c``.
        The caches differ in dtype (SWA is the activation dtype, the compressor
        caches are fp32) and row width, so each is dispatched to its own pointer;
        unused slots (``c >= N_CACHES``) are never launched.  Byte-identical to the
        prior per-cache ``cache[page_ids, page_offsets] = values.to(cache.dtype)``.
        """
        row = tl.program_id(0)
        c = tl.program_id(1)
        sblk = tl.program_id(2)
        pid = tl.load(page_ids_ptr + row).to(tl.int64)
        poff = tl.load(page_offsets_ptr + row).to(tl.int64)
        if c == 0:
            _store_current_token_row(row, sblk, pid, poff, src0, cache0, S0, T, BLOCK_S)
        elif c == 1:
            _store_current_token_row(row, sblk, pid, poff, src1, cache1, S1, T, BLOCK_S)
        elif c == 2:
            _store_current_token_row(row, sblk, pid, poff, src2, cache2, S2, T, BLOCK_S)
        elif c == 3:
            _store_current_token_row(row, sblk, pid, poff, src3, cache3, S3, T, BLOCK_S)
        else:
            _store_current_token_row(row, sblk, pid, poff, src4, cache4, S4, T, BLOCK_S)

    @triton.jit
    def _dsv4_fullrange_candidate_rows_kernel(
        kv_cache_ptr,  # [P, T, S] fp32 paged compressor kv cache
        gate_cache_ptr,  # [P, T, S] fp32 paged compressor gate cache
        page_ids_ptr,  # [B, M*R] int64 (hoisted full-range page map)
        page_offsets_ptr,  # [B, M*R] int64
        valid_ptr,  # [B, M*R] bool
        ape_ptr,  # [R, 2*head_dim] fp32
        norm_weight_ptr,  # [head_dim]
        out_ptr,  # [B, M, head_dim] (dtype = q_index.dtype)
        M,  # max_compressed_len
        RATIO,  # compress_ratio (4)
        HEAD_DIM,
        T,  # tokens_per_block (paged cache dim-1 stride)
        S,  # state_dim (paged cache dim-2 stride)
        APE_STRIDE,  # ape.shape[1] (== 2 * head_dim)
        PAGEMAP_STRIDE,  # M * R (row stride of the [B, M*R] page maps)
        eps,
        TWO_R: tl.constexpr,  # 2 * compress_ratio (8)
        BLOCK_D: tl.constexpr,  # next_pow2(head_dim)
    ):
        """Fused ratio-4 full-range candidate-row reconstruction (idea_0075).

        One program per (batch ``b``, candidate row ``r``).  It gathers the ``2*R``
        paged compressor kv/gate slots for the overlap window (the previous block
        of row ``r`` == the current block of row ``r-1``, so the full-range slot
        index is uniformly ``(r-1)*R + s`` for ``s`` in ``[0, 2R)``), adds the ape
        bias, applies the previous-block validity mask, softmax-pools over the
        ``2*R`` axis and RMS-norms over ``head_dim`` -- collapsing the
        gather / row-shift / concat / where / pool / rmsnorm swarm of
        ``_batched_overlap_compressed_rows_fullrange`` into a single launch.  It
        emits the post-pool, post-rmsnorm candidate rows (pre-rope); the caller
        keeps the rope / quantize tail unchanged.  All reductions are fp32-internal
        with bf16 rounding at the same points as the eager reference
        (gather ``.to(dtype)``, the ape add, the ``compress_pool`` output, the
        ``_rms_norm_ref`` output), so the result matches bit-for-bit up to the
        ``rsqrt`` primitive (validated in the op unit test).
        """
        ROUND = out_ptr.dtype.element_ty
        NEG = -1.0e20  # masked previous-gate floor (matches new_full(-1e20))

        prog = tl.program_id(0)
        b = prog // M
        r = prog % M

        s = tl.arange(0, TWO_R)
        is_prev = s < RATIO
        # previous block of row r == current block of row r-1, so the full-range
        # index is uniformly (r-1)*R + s: the first R slots are the previous block,
        # the last R the current block.  Row 0 has no previous block (masked below).
        idx = (r - 1) * RATIO + s
        idx_safe = tl.maximum(idx, 0)
        pm_base = b * PAGEMAP_STRIDE
        pid = tl.load(page_ids_ptr + pm_base + idx_safe).to(tl.int64)
        poff = tl.load(page_offsets_ptr + pm_base + idx_safe).to(tl.int64)
        pvalid = tl.load(valid_ptr + pm_base + idx_safe).to(tl.int1)

        ratio_slot = tl.where(is_prev, s, s - RATIO)  # ape row per slot
        channel_offset = tl.where(is_prev, 0, HEAD_DIM)  # previous / current channels

        d = tl.arange(0, BLOCK_D)
        dmask = d < HEAD_DIM
        cmask = dmask[None, :]

        base = pid * (T * S) + poff * S + channel_offset  # [TWO_R]
        offs = base[:, None] + d[None, :]  # [TWO_R, BLOCK_D]
        raw_kv = tl.load(kv_cache_ptr + offs, mask=cmask, other=0.0)
        raw_gate = tl.load(gate_cache_ptr + offs, mask=cmask, other=0.0)
        # Round the fp32 cache reads to the activation dtype (== gather ``.to(dtype)``).
        kv_b = raw_kv.to(ROUND)
        gate_b = raw_gate.to(ROUND)

        ape_off = ratio_slot[:, None] * APE_STRIDE + channel_offset[:, None] + d[None, :]
        ape_b = tl.load(ape_ptr + ape_off, mask=cmask, other=0.0).to(ROUND)
        # bf16 add: round(bf16(gate) + bf16(ape)) -- the eager path adds in the
        # activation dtype before masking, so keep the intermediate rounding.
        gate_sum = (gate_b.to(tl.float32) + ape_b.to(tl.float32)).to(ROUND)

        # Only the previous block is validity-masked; the current block is read
        # as-is (the reference never masks it).  Row 0 has no previous block.
        keep_prev = (r >= 1) & pvalid  # [TWO_R]
        mask_prev_invalid = is_prev & (keep_prev == 0)  # true only for masked prev slots

        g = gate_sum.to(tl.float32)
        k = kv_b.to(tl.float32)
        g = tl.where(mask_prev_invalid[:, None], NEG, g)  # exp(-1e20 - m) -> 0
        k = tl.where(mask_prev_invalid[:, None], 0.0, k)

        # Softmax-weighted pool over the 2R axis (matches deepseek_v4_compress_pool:
        # max -> exp -> normalize per channel -> weighted sum).
        m_max = tl.max(g, axis=0)  # [BLOCK_D]
        e = tl.exp(g - m_max[None, :])
        ssum = tl.sum(e, axis=0)  # [BLOCK_D]
        w = e / ssum[None, :]
        pooled = tl.sum(k * w, axis=0)  # [BLOCK_D] fp32
        pooled_b = pooled.to(ROUND)  # compress_pool returns the activation dtype

        # RMSNorm over head_dim (matches _rms_norm_ref: fp32 internal + fp32 weight).
        c = pooled_b.to(tl.float32)
        sq = tl.where(dmask, c * c, 0.0)
        ms = tl.sum(sq, axis=0) / HEAD_DIM
        rinv = tl.rsqrt(ms + eps)
        wgt = tl.load(norm_weight_ptr + d, mask=dmask, other=0.0).to(tl.float32)
        o = c * rinv * wgt
        tl.store(out_ptr + prog.to(tl.int64) * HEAD_DIM + d, o.to(ROUND), mask=dmask)

    @triton.jit
    def _dsv4_index_score_kernel(
        q_ptr,  # [N, H, D] index queries (fp16/bf16)
        k_ptr,  # [N, C, D] compressed candidate index keys (same dtype as q)
        w_ptr,  # [N, H] per-head indexer weights (fp32)
        vis_ptr,  # [N] int visible candidate count per decode row
        out_ptr,  # [N, C] fp32 masked index score
        H,
        C,
        D: tl.constexpr,  # index_head_dim
        D_BLOCK: tl.constexpr,  # next_pow2(D)
        H_BLOCK: tl.constexpr,  # next_pow2(H)
        BLOCK_C: tl.constexpr,
    ):
        """Fused ratio-4 lightning-indexer score for one (decode row, candidate block).

        Collapses ``(matmul(q, k^T).float().relu() * w.float()).sum(dim=1)`` followed by
        ``masked_fill(~visible, -inf)`` into a single launch, so the ``[N, H, C]``
        head-by-candidate score and the separate masked ``[N, C]`` tensor are never
        materialized -- only the fused masked score row consumed by the decode top-k is
        written. The bf16/fp16 matmul output is rounded to the input dtype (matching the
        reference ``matmul(...).float()``) and the relu / per-head weighting / head
        reduction run in fp32, so the score values (and therefore the top-k sort order)
        stay within one ULP of the reference and the selected rows are preserved.
        """
        n = tl.program_id(0)
        cb = tl.program_id(1)
        offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        offs_d = tl.arange(0, D_BLOCK)
        offs_h = tl.arange(0, H_BLOCK)
        c_mask = offs_c < C
        d_mask = offs_d < D
        h_mask = offs_h < H

        # k tile [BLOCK_C, D_BLOCK]; padded candidate/dim lanes load 0 (add nothing).
        k = tl.load(
            k_ptr + n.to(tl.int64) * C * D + offs_c[:, None] * D + offs_d[None, :],
            mask=c_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        # q tile [H_BLOCK, D_BLOCK]; padded head/dim lanes load 0.
        q = tl.load(
            q_ptr + n.to(tl.int64) * H * D + offs_h[:, None] * D + offs_d[None, :],
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        # scores[BLOCK_C, H_BLOCK] = k @ q^T ; padded heads/candidates contribute 0.
        sc = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
        # Match the reference bf16/fp16 matmul output rounding before ``.float()``.
        sc = sc.to(k.dtype).to(tl.float32)
        sc = tl.maximum(sc, 0.0)  # relu
        w = tl.load(w_ptr + n.to(tl.int64) * H + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        sc = sc * w[None, :]
        score = tl.sum(sc, axis=1)  # [BLOCK_C] fp32 weighted head reduction
        vlen = tl.load(vis_ptr + n)
        score = tl.where(offs_c < vlen, score, float("-inf"))  # visibility mask
        tl.store(out_ptr + n.to(tl.int64) * C + offs_c, score, mask=c_mask)

    @triton.jit
    def _dsv4_topk_select_kernel(
        score_ptr,  # [N, C] fp32 masked index score
        rows_ptr,  # [N, K_OUT] int64 out: selected candidate row per slot (-1 = invalid)
        valid_ptr,  # [N, K_OUT] uint8 out: 1 iff the slot's score is finite
        C,  # candidate count (score row width)
        K_OUT,  # index_topk (output row width)
        TOPK_COUNT,  # min(index_topk, C): slots filled from the sort
        BLOCK_C: tl.constexpr,  # next_pow2(C)
        BLOCK_K: tl.constexpr,  # next_pow2(K_OUT)
    ):
        """Exact decode top-k row selection for the ratio-4 indexer (idea_0046).

        One program per decode row.  It replaces ``index_score.topk(topk_count)``
        (the fat ``gatherTopK`` + ``radixSortKVInPlace`` pair), the decomposed
        ``isfinite`` chain, the ``where``-to-``-1`` fixup and the short-history
        -1/False pad path with a single launch that emits the padded
        ``topk_rows`` / ``topk_valid`` directly.

        The score row is packed into one sortable int64 key per candidate:
        ``inv_u`` is the IEEE-754 float-flip (ascending ``inv_u`` == descending
        float total order, the transform CUDA's radix top-k uses) in the high
        bits, the candidate index in the low 31 bits.  ``-0.0`` is canonicalized
        to ``+0.0`` first because torch's top-k compares them equal (ties break
        by ascending index, which the low bits reproduce).  A bitonic
        ``tl.sort`` of the keys therefore yields exactly torch's value order,
        tie order included; non-finite scores (the ``-inf`` visibility mask)
        decode to ``valid == 0`` / ``row == -1`` just like the eager
        ``isfinite``/``where`` tail.  NaN scores sort first (torch's "NaN is
        largest") and also emit ``-1``; only the relative order *among* multiple
        differently-signed NaNs may differ, where every affected slot is ``-1``
        either way.
        """
        n = tl.program_id(0)
        c = tl.arange(0, BLOCK_C)
        cmask = c < C
        s = tl.load(score_ptr + n.to(tl.int64) * C + c, mask=cmask, other=float("-inf"))
        # torch's top-k orders +-0.0 as equal keys; distinct bit patterns would
        # rank +0.0 above -0.0, so fold both onto the +0.0 pattern.
        s = tl.where(s == 0.0, 0.0, s)
        u = s.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        # Float-flip: negative floats (sign bit set) already ascend toward -inf as
        # raw bits; positive floats are mirrored below them.
        inv_u = tl.where(u >= 0x80000000, u, 0x7FFFFFFF - u)
        # Padding lanes sort after every real candidate (-inf included).  A real
        # 0xFFFFFFFF (negative-NaN payload) key ties with padding and wins the tie
        # via its smaller low-bits index, so it is never displaced out of the row.
        inv_u = tl.where(cmask, inv_u, 0xFFFFFFFF)
        key = (inv_u << 31) | c.to(tl.int64)
        key = tl.sort(key)
        idx_s = key & 0x7FFFFFFF
        inv_s = key >> 31
        # Strictly-finite window == torch.isfinite: -inf flips to 0xFF800000, +inf
        # to 0x007FFFFF, NaN payloads fall outside on either side.
        valid = (inv_s > 0x007FFFFF) & (inv_s < 0xFF800000)
        rows = tl.where(valid, idx_s, tl.full((BLOCK_C,), -1, tl.int64))
        out_base = n.to(tl.int64) * K_OUT
        smask = c < TOPK_COUNT
        tl.store(rows_ptr + out_base + c, rows, mask=smask)
        tl.store(valid_ptr + out_base + c, valid.to(tl.uint8), mask=smask)
        # Short-history pad tail (topk_count < index_topk): -1 rows, False validity.
        p = tl.arange(0, BLOCK_K)
        pmask = (p >= TOPK_COUNT) & (p < K_OUT)
        tl.store(rows_ptr + out_base + p, tl.full((BLOCK_K,), -1, tl.int64), mask=pmask)
        tl.store(valid_ptr + out_base + p, tl.zeros((BLOCK_K,), tl.uint8), mask=pmask)

    @triton.jit
    def _dsv4_assemble_selected_kv_kernel(
        swa_cache_ptr,  # [P, T, D] paged local (sliding-window) kv cache
        mhc_cache_ptr,  # [P, T, D] paged compressed kv cache (same P/T/D/dtype as swa)
        selected_rows_ptr,  # [B, TOPK] int64 selected compressed row ids (-1 = pad)
        comp_valid_ptr,  # [B, TOPK] bool indexer/candidate row validity
        input_pos_ptr,  # [B] int current decode position per sequence
        seq_idx_ptr,  # [B] int sequence row per decode token
        cu_num_pages_ptr,  # [num_seq + 1] prefix-sum page-table offsets
        cache_loc_ptr,  # [total_pages] physical page id per page-table slot
        out_kv_ptr,  # [B, KV_ROWS, D] out (activation dtype)
        out_relidx_ptr,  # [B, KV_ROWS] int64 out (slot id if kept else -1)
        B,
        KV_ROWS,  # W + TOPK (row stride of the [B, KV_ROWS, ...] outputs)
        W,  # window_size (local slots occupy [0, W))
        TOPK,  # number of compressed slots (compressed slots occupy [W, KV_ROWS))
        RATIO,  # compress_ratio
        T,  # tokens_per_block (paged cache dim-1)
        D,  # head_dim (paged cache dim-2)
        CACHE_LOC_MAX,  # cache_loc.numel() - 1
        BLOCK_D: tl.constexpr,  # next_pow2(head_dim)
    ):
        """Fold the decode selected-KV assembly into one paged gather (idea_0001).

        One program per (decode row ``b``, output slot ``slot``).  It replaces the
        per-ratio-4/128-layer tail of ``_decode_compressed_cache_attention`` -- the
        local-window page-map + ``swa_cache`` gather, the dynamic compressed page-map
        translation + ``mhc_cache`` gather, the two ``torch.cat``s (selected_kv /
        valid_rows) and the ``arange``/``where`` that builds the attend's relative
        indices -- by reading the paged local/compressed cache rows directly and
        emitting the contiguous ``selected_kv`` block plus the attend's ``rel_topk``.

        Local slots (``slot < W``) mirror ``_decode_local_cache_rows``' position
        generation; compressed slots (``slot >= W``) read the caller-selected
        ``selected_rows`` (``clamp(min=0) * RATIO`` -> paged position).  The page
        translation mirrors ``_page_ids_and_offsets_from_tpb`` exactly (all
        divisions/remainders act on non-negative, identically-clamped operands), so
        the gathered rows, the per-slot validity, and ``rel_topk`` are byte-identical
        to the eager gather/cat/where chain -- including the clamped rows of masked
        slots, which the attend ignores (``rel_topk == -1``).
        """
        prog = tl.program_id(0)
        b = prog // KV_ROWS
        slot = prog % KV_ROWS
        is_local = slot < W
        is_comp = slot >= W

        input_pos = tl.load(input_pos_ptr + b).to(tl.int64)
        seq_idx = tl.load(seq_idx_ptr + b).to(tl.int64)

        # Local-window position (mirror _decode_local_cache_rows):
        # pos = input_pos - W + 1 + slot.
        lpos = input_pos - W + 1 + slot
        lvalid_pos = (lpos >= 0) & (lpos <= input_pos)

        # Compressed position: selected_rows.clamp(min=0) * RATIO.  Guard the load so
        # TOPK == 0 (no compressed slots) never dereferences the empty selection.
        c = slot - W
        c_safe = tl.minimum(tl.maximum(c, 0), TOPK - 1)
        csel = tl.load(selected_rows_ptr + b * TOPK + c_safe, mask=is_comp, other=0).to(tl.int64)
        cvalid_in = tl.load(comp_valid_ptr + b * TOPK + c_safe, mask=is_comp, other=0).to(tl.int1)
        cpos = tl.maximum(csel, 0) * RATIO

        pos = tl.where(is_local, lpos, cpos)

        # Page-address translation (mirror _page_ids_and_offsets_from_tpb).
        safe_pos = tl.maximum(pos, 0)
        page_ordinal = safe_pos // T
        page_offset = safe_pos % T
        page_start = tl.load(cu_num_pages_ptr + seq_idx).to(tl.int64)
        page_end = tl.load(cu_num_pages_ptr + seq_idx + 1).to(tl.int64)
        page_table_idx = page_start + page_ordinal
        page_valid = (pos >= 0) & (page_table_idx < page_end)
        safe_idx = tl.where(page_valid, page_table_idx, page_start)
        safe_idx = tl.minimum(tl.maximum(safe_idx, 0), CACHE_LOC_MAX)
        pid = tl.load(cache_loc_ptr + safe_idx).to(tl.int64)

        local_valid = lvalid_pos & page_valid
        # ratio-4 masks pad rows via (selected_rows >= 0); ratio-128 uses arange (>=0
        # always), so the (csel >= 0) term is uniform and byte-identical to both.
        comp_valid = cvalid_in & page_valid & (csel >= 0)
        valid = tl.where(is_local, local_valid, comp_valid)

        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        row_off = (pid * T + page_offset) * D + d
        swa_row = tl.load(swa_cache_ptr + row_off, mask=is_local & dmask, other=0.0)
        mhc_row = tl.load(mhc_cache_ptr + row_off, mask=is_comp & dmask, other=0.0)
        row = tl.where(is_local, swa_row, mhc_row)

        out_base = (b * KV_ROWS + slot).to(tl.int64) * D + d
        tl.store(out_kv_ptr + out_base, row.to(out_kv_ptr.dtype.element_ty), mask=dmask)
        relidx = tl.where(valid, slot, -1).to(tl.int64)
        tl.store(out_relidx_ptr + b * KV_ROWS + slot, relidx)

    @triton.jit
    def _dsv4_compressed_row_r4_front_kernel(
        kv_cache_ptr,  # [P, T, S=2*HEAD_DIM] fp32 paged compressor kv cache
        gate_cache_ptr,  # [P, T, S] fp32 paged compressor gate cache
        ovl_page_ids_ptr,  # [N, 2*RATIO] int64 (hoisted overlap band page map)
        ovl_page_offsets_ptr,  # [N, 2*RATIO] int64
        ovl_valid_ptr,  # [N, 2*RATIO] bool
        ape_ptr,  # [RATIO, 2*HEAD_DIM] fp32
        norm_weight_ptr,  # [HEAD_DIM]
        out_ptr,  # [N, HEAD_DIM] normed rows (activation dtype)
        N,
        RATIO,  # compress_ratio (4)
        HEAD_DIM,
        T,  # tokens_per_block (paged cache dim-1 stride)
        S,  # compressor state_dim (paged cache dim-2 stride, == 2*HEAD_DIM)
        APE_STRIDE,  # ape.shape[1] (== 2 * HEAD_DIM)
        PAGEMAP_STRIDE,  # 2 * RATIO (row stride of the [N, 2*RATIO] page maps)
        eps,
        TWO_R: tl.constexpr,  # 2 * compress_ratio (8)
        BLOCK_D: tl.constexpr,  # next_pow2(HEAD_DIM)
    ):
        """Fused ratio-4 main-compressor decode-row front (idea_0007, stage 1).

        One program per decode row ``b``.  Reconstructs the single just-completed
        compressed row exactly as ``_batched_compressed_rows_from_paged_state``'s
        overlap branch does: it reads the ``2*RATIO`` paged compressor kv/gate slots
        of the hoisted overlap band (the first ``RATIO`` columns are the previous
        block, the last ``RATIO`` the current block), adds the ape bias, applies the
        previous-block validity mask, softmax-pools over the ``2*RATIO`` axis and
        RMS-norms over ``HEAD_DIM`` -- collapsing that gather / slice / ape-add /
        where / cat / pool / rmsnorm swarm into one launch and emitting the
        post-rmsnorm (pre-rope) row.  The rope / fp8-quant / masked store tail runs in
        ``_dsv4_rope_fp8_masked_store_kernel`` (stage 2).  All reductions are
        fp32-internal with bf16 rounding at the same points as the eager reference
        (gather ``.to(dtype)``, the ape add, the ``compress_pool`` output, the
        ``_rms_norm_ref`` output), so the row matches bit-for-bit up to the ``rsqrt``
        primitive.  Mirrors ``_dsv4_fullrange_candidate_rows_kernel`` but reads the
        ``[N, 2*RATIO]`` band map by slot index ``s`` directly (no ``(r-1)*R+s``
        fullrange remap) and masks the previous block with the band ``valid`` flag.
        """
        ROUND = out_ptr.dtype.element_ty
        NEG = -1.0e20  # masked previous-gate floor (matches new_full(-1e20))

        b = tl.program_id(0)
        if b >= N:
            return

        s = tl.arange(0, TWO_R)
        is_prev = s < RATIO
        pm_base = b * PAGEMAP_STRIDE
        pid = tl.load(ovl_page_ids_ptr + pm_base + s).to(tl.int64)
        poff = tl.load(ovl_page_offsets_ptr + pm_base + s).to(tl.int64)
        pvalid = tl.load(ovl_valid_ptr + pm_base + s).to(tl.int1)

        ratio_slot = tl.where(is_prev, s, s - RATIO)  # ape row per slot
        channel_offset = tl.where(is_prev, 0, HEAD_DIM)  # previous / current channels

        d = tl.arange(0, BLOCK_D)
        dmask = d < HEAD_DIM
        cmask = dmask[None, :]

        base = pid * (T * S) + poff * S + channel_offset  # [TWO_R]
        offs = base[:, None] + d[None, :]  # [TWO_R, BLOCK_D]
        raw_kv = tl.load(kv_cache_ptr + offs, mask=cmask, other=0.0)
        raw_gate = tl.load(gate_cache_ptr + offs, mask=cmask, other=0.0)
        # Round the fp32 cache reads to the activation dtype (== gather ``.to(dtype)``).
        kv_b = raw_kv.to(ROUND)
        gate_b = raw_gate.to(ROUND)

        ape_off = ratio_slot[:, None] * APE_STRIDE + channel_offset[:, None] + d[None, :]
        ape_b = tl.load(ape_ptr + ape_off, mask=cmask, other=0.0).to(ROUND)
        # bf16 add: round(bf16(gate) + bf16(ape)) -- the eager path adds in the
        # activation dtype before masking, so keep the intermediate rounding.
        gate_sum = (gate_b.to(tl.float32) + ape_b.to(tl.float32)).to(ROUND)

        # Only the previous block is validity-masked; the current block is read as-is
        # (the reference never masks it).  ``ovl_valid`` already encodes ``position>=0``
        # AND page validity, so ``previous_valid = ovl_valid[:, :RATIO]`` bit-exactly.
        mask_prev_invalid = is_prev & (pvalid == 0)  # [TWO_R]

        g = gate_sum.to(tl.float32)
        k = kv_b.to(tl.float32)
        g = tl.where(mask_prev_invalid[:, None], NEG, g)  # exp(-1e20 - m) -> 0
        k = tl.where(mask_prev_invalid[:, None], 0.0, k)

        # Softmax-weighted pool over the 2R axis (matches deepseek_v4_compress_pool).
        m_max = tl.max(g, axis=0)  # [BLOCK_D]
        e = tl.exp(g - m_max[None, :])
        ssum = tl.sum(e, axis=0)  # [BLOCK_D]
        w = e / ssum[None, :]
        pooled = tl.sum(k * w, axis=0)  # [BLOCK_D] fp32
        pooled_b = pooled.to(ROUND)  # compress_pool returns the activation dtype

        # RMSNorm over head_dim (matches _rms_norm_ref: fp32 internal + fp32 weight).
        c = pooled_b.to(tl.float32)
        sq = tl.where(dmask, c * c, 0.0)
        ms = tl.sum(sq, axis=0) / HEAD_DIM
        rinv = tl.rsqrt(ms + eps)
        wgt = tl.load(norm_weight_ptr + d, mask=dmask, other=0.0).to(tl.float32)
        o = c * rinv * wgt
        tl.store(out_ptr + b.to(tl.int64) * HEAD_DIM + d, o.to(ROUND), mask=dmask)

    @triton.jit
    def _dsv4_rope_fp8_masked_store_kernel(
        normed_ptr,  # [N, HEAD_DIM] post-rmsnorm rows (activation dtype)
        cos_ptr,  # [n_pos, DH] fp32
        sin_ptr,  # [n_pos, DH] fp32
        row_position_id_ptr,  # [N] int64 (already clamped into [0, n_pos))
        row_valid_ptr,  # [N] bool -- store row iff valid
        mhc_page_ids_ptr,  # [N] int64 write page id per row
        mhc_page_offsets_ptr,  # [N] int64 in-page offset per row
        cache_ptr,  # [P, T, HEAD_DIM] paged mhc cache (mutated in place)
        stride_p,
        stride_t,
        stride_s,
        cossin_row_stride,  # cos/sin row stride (== DH)
        N,
        HEAD_DIM,
        NOPE_DIM,  # HEAD_DIM - ROPE_DIM (multiple of FP8_BLOCK)
        DH,  # ROPE_DIM // 2
        FP8_BLOCK: tl.constexpr,  # 64 (fake-fp8 group width)
        NUM_FP8_BLOCKS: tl.constexpr,  # NOPE_DIM // FP8_BLOCK
        BLOCK_D: tl.constexpr,  # next_pow2(HEAD_DIM)
        BLOCK_DH: tl.constexpr,  # next_pow2(DH)
        MAX_VAL: tl.constexpr,  # 448.0 (e4m3 absmax)
        MIN_VAL: tl.constexpr,  # 1e-4 (amax floor)
    ):
        """Fused main-compressor rope + fake-fp8 + validity-masked store (idea_0007, stage 2).

        One program per decode row ``b``.  Reads the post-rmsnorm row emitted by
        ``_dsv4_compressed_row_r4_front_kernel`` (or, for the eager fallback, any
        ``[N, HEAD_DIM]`` normed row) and, only when ``row_valid[b]``, writes the
        fully reconstructed compressed row to ``cache[page_ids[b], page_offsets[b]]``.
        Collapses the rope-tail chain ``_apply_compressed_rope_and_quantize``
        (rotate=False: block fake-fp8 on the nope slice + interleaved RoPE/concat on
        the pe slice) plus the ``cos``/``sin`` gathers plus the ``_masked_paged_store``
        into a single launch.

        Byte-identical to the reference on the nope slice -- the block amax, the
        ``scale = 2**ceil(log2(clamp_min(amax,1e-4)/448))``, the ``clamp -> bf16 ->
        fp32`` round-trip and the ``* scale`` reproduce ``fake_fp8_act_quant`` exactly
        -- and equal to <=1 ULP on the pe slice (FMA folding, as for
        ``deepseek_v4_fused_rope_concat``).  Invalid rows store nothing, leaving the
        slot byte-identical to the prior read-old + write-back no-op.
        """
        RD = normed_ptr.dtype.element_ty  # rounding dtype for the fp8/rope math
        CT = cache_ptr.dtype.element_ty  # cache store dtype

        b = tl.program_id(0)
        if b >= N:
            return
        valid = tl.load(row_valid_ptr + b).to(tl.int1)
        pid = tl.load(mhc_page_ids_ptr + b).to(tl.int64)
        poff = tl.load(mhc_page_offsets_ptr + b).to(tl.int64)
        dst_base = cache_ptr + pid * stride_p + poff * stride_t
        nrow = normed_ptr + b.to(tl.int64) * HEAD_DIM

        # --- fake-fp8 block quant on the nope slice [0, NOPE_DIM) ---
        d = tl.arange(0, BLOCK_D)
        nmask = d < NOPE_DIM
        nope = tl.load(nrow + d, mask=nmask, other=0.0).to(tl.float32)  # bf16 -> fp32
        blk = d // FP8_BLOCK
        scale_per_d = tl.full([BLOCK_D], 1.0, tl.float32)  # 1.0 outside the nope slice
        for j in tl.static_range(NUM_FP8_BLOCKS):
            in_blk = nmask & (blk == j)
            amax_j = tl.max(tl.where(in_blk, tl.abs(nope), 0.0), axis=0)
            scale_j = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax_j, MIN_VAL) / MAX_VAL)))
            scale_per_d = tl.where(in_blk, scale_j, scale_per_d)
        q = nope / scale_per_d
        q = tl.minimum(tl.maximum(q, -MAX_VAL), MAX_VAL)
        q = q.to(RD).to(tl.float32)  # round-trip through the activation dtype
        nope_out = (q * scale_per_d).to(RD)
        tl.store(dst_base + d * stride_s, nope_out.to(CT), mask=nmask & valid)

        # --- interleaved RoPE on the pe slice [NOPE_DIM, HEAD_DIM) ---
        k = tl.arange(0, BLOCK_DH)
        kmask = k < DH
        pe_base = nrow + NOPE_DIM
        even = tl.load(pe_base + 2 * k, mask=kmask, other=0.0).to(tl.float32)
        odd = tl.load(pe_base + 2 * k + 1, mask=kmask, other=0.0).to(tl.float32)
        rpid = tl.load(row_position_id_ptr + b).to(tl.int64)
        cos = tl.load(cos_ptr + rpid * cossin_row_stride + k, mask=kmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + rpid * cossin_row_stride + k, mask=kmask, other=0.0).to(tl.float32)
        out_even = even * cos - odd * sin
        out_odd = even * sin + odd * cos
        pe_out_base = dst_base + NOPE_DIM * stride_s
        tl.store(pe_out_base + (2 * k) * stride_s, out_even.to(RD).to(CT), mask=kmask & valid)
        tl.store(pe_out_base + (2 * k + 1) * stride_s, out_odd.to(RD).to(CT), mask=kmask & valid)

    @triton.jit
    def _dsv4_paged_compress_pool_kernel(
        kv_cache_ptr,  # [P, T, S] fp32 paged compressor kv cache
        gate_cache_ptr,  # [P, T, S] fp32 paged compressor gate cache
        page_ids_ptr,  # [N, R] int64 (per-row page id of each ratio slot)
        page_offsets_ptr,  # [N, R] int64 (per-row in-page offset of each ratio slot)
        ape_ptr,  # [R, APE_STRIDE] fp32; column d in [0, HEAD_DIM) used
        out_ptr,  # [N, HEAD_DIM] pooled row (activation dtype)
        N,
        R,  # compress_ratio (128)
        HEAD_DIM,
        T,  # tokens_per_block (paged cache dim-1)
        S,  # compressor state_dim (paged cache dim-2 stride, == HEAD_DIM here)
        APE_STRIDE,  # ape.shape[1]
        PAGEMAP_STRIDE,  # R (row stride of the [N, R] page maps)
        BLOCK_R: tl.constexpr,  # next_pow2(R)
        BLOCK_D: tl.constexpr,
    ):
        """Fused ratio-128 (dense) main-compressor pool front (idea_0039, stage 1).

        D-tiled companion of ``_dsv4_compressed_row_r4_front_kernel``: one program per
        ``(decode row, HEAD_DIM block)``.  The ratio-128 pool tile ``[R, HEAD_DIM]`` is
        too large for the ratio-4 kernel's single-program strategy, so the HEAD_DIM axis
        is fanned across ``cdiv(HEAD_DIM, BLOCK_D)`` programs (recovering occupancy at the
        small decode ``N``).  Reads the ``R`` paged compressor kv/gate slots of the
        just-completed block via the precomputed ``[N, R]`` page map, adds the ape bias
        and softmax-pools over the ratio axis -- collapsing the two paged gathers, the two
        ``.to(dtype)`` casts, the ape add and the ``deepseek_v4_compress_pool`` launch into
        one kernel emitting the pooled (pre-rmsnorm) row.  The non-overlap branch performs
        NO validity masking (the reference discards ``page_valid`` and never masks the
        gate for the dense path), so every slot participates.  RMSNorm
        (``_compressor_rms_norm``) and the rope/fp8/masked-store tail
        (``_dsv4_rope_fp8_masked_store_kernel``) run as the subsequent stages.  All
        reductions are fp32-internal with bf16 rounding at the same points as the eager
        reference (gather ``.to(dtype)``, the ape add, the ``compress_pool`` output), so the
        pooled row matches ``gather + ape + _dsv4_compress_pool_kernel`` to <=1 ULP -- the
        only deviation is the fp32 reduction order over the ratio axis (this kernel fixes
        ``num_warps=4`` while ``deepseek_v4_compress_pool`` autotunes it), which flips at
        most a handful of near-zero channels by one bf16 ULP.
        """
        ROUND = out_ptr.dtype.element_ty

        n = tl.program_id(0)
        if n >= N:
            return
        d0 = tl.program_id(1) * BLOCK_D
        d = d0 + tl.arange(0, BLOCK_D)
        dmask = d < HEAD_DIM

        r = tl.arange(0, BLOCK_R)
        rmask = r < R
        pm_base = n * PAGEMAP_STRIDE
        pid = tl.load(page_ids_ptr + pm_base + r, mask=rmask, other=0).to(tl.int64)
        poff = tl.load(page_offsets_ptr + pm_base + r, mask=rmask, other=0).to(tl.int64)

        base = pid * (T * S) + poff * S  # [BLOCK_R]; channel offset 0 (non-overlap)
        offs = base[:, None] + d[None, :]  # [BLOCK_R, BLOCK_D]
        cmask = rmask[:, None] & dmask[None, :]
        raw_kv = tl.load(kv_cache_ptr + offs, mask=cmask, other=0.0)
        raw_gate = tl.load(gate_cache_ptr + offs, mask=cmask, other=0.0)
        # Round the fp32 cache reads to the activation dtype (== gather ``.to(dtype)``).
        kv_b = raw_kv.to(ROUND)
        gate_b = raw_gate.to(ROUND)

        ape_off = r[:, None] * APE_STRIDE + d[None, :]
        ape_b = tl.load(ape_ptr + ape_off, mask=cmask, other=0.0).to(ROUND)
        # bf16 add: round(bf16(gate) + bf16(ape)) -- the eager path adds ``gate + ape`` in
        # the activation dtype, so keep the intermediate rounding.
        gate_sum = (gate_b.to(tl.float32) + ape_b.to(tl.float32)).to(ROUND)

        g = gate_sum.to(tl.float32)
        k = kv_b.to(tl.float32)
        # Padded ratio rows (r >= R) never contribute: -inf gate -> zero softmax weight
        # (R == BLOCK_R for ratio-128, so this is a no-op safety net).
        g = tl.where(rmask[:, None], g, float("-inf"))
        k = tl.where(rmask[:, None], k, 0.0)

        # Per-channel softmax over the ratio axis, then weighted sum -- same op order as
        # ``(kv * gate.softmax(dim=-2)).sum(dim=-2)`` / ``_dsv4_compress_pool_kernel``.
        m = tl.max(g, axis=0)  # [BLOCK_D]
        e = tl.exp(g - m[None, :])
        ssum = tl.sum(e, axis=0)  # [BLOCK_D]
        w = e / ssum[None, :]
        pooled = tl.sum(k * w, axis=0)  # [BLOCK_D] fp32
        tl.store(out_ptr + n.to(tl.int64) * HEAD_DIM + d, pooled.to(ROUND), mask=dmask)


def _masked_write_decode_cache_rows(
    cache: torch.Tensor,  # [P, T, S] paged cache (mutated in place)
    values: torch.Tensor,  # [N, S] rows to conditionally store (already cache dtype)
    row_valid: torch.Tensor,  # [N] bool
    page_ids: torch.Tensor,  # [N] int64 write page id per row
    page_offsets: torch.Tensor,  # [N] int64 in-page offset per row
) -> None:
    """Store ``values[r]`` into ``cache[page_ids[r], page_offsets[r]]`` iff ``row_valid[r]``.

    Triton wrapper for the read-free validity-masked paged store. ``page_ids``/
    ``page_offsets`` are the precomputed write address (the same map fed to the removed
    read + write-back). ``values`` must already be cast to ``cache.dtype`` so the store is
    a pure copy -- byte-identical to the prior
    ``cache[page_ids, page_offsets] = compressed_rows.to(cache.dtype)`` for valid rows,
    and a no-op (slot untouched) for invalid rows, which is byte-identical to gathering the
    previous row and writing it back unchanged.
    """
    n_rows = int(values.shape[0])
    if n_rows == 0:
        return
    state_dim = int(values.shape[-1])
    values = values.contiguous()
    BLOCK_S = min(triton.next_power_of_2(state_dim), 1024)
    grid = (n_rows, triton.cdiv(state_dim, BLOCK_S))
    _masked_paged_store_kernel[grid](
        values,
        row_valid,
        page_ids,
        page_offsets,
        cache,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        state_dim,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )


def _fused_current_token_store(
    caches: List[torch.Tensor],
    values: List[torch.Tensor],
    seq_idx: torch.Tensor,
    input_pos: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    page_ids: Optional[torch.Tensor],
    page_offsets: Optional[torch.Tensor],
) -> None:
    """Write the current decode token into every listed cache in one launch.

    Each ``(cache, value)`` pair stores the freshly produced current-token row at
    logical position ``input_pos``. Every DeepSeek-V4 current-token cache -- SWA kv,
    main-compressor kv/gate, and the ratio-4 indexer-compressor kv/gate -- shares one
    page table and one ``tokens_per_block``, so they all resolve to the identical
    hoisted ``(page_ids, page_offsets)`` write address (see
    ``deepseek_v4_sparse_prepare_decode_page_addr``). Folding the per-cache
    ``index_put`` scatters into a single ``_multi_current_token_store_kernel`` launch
    removes ~4 (ratio-4) / ~2 (ratio-128) gather/scatter kernels per layer per decode
    step. Values are pre-cast to the cache dtype in torch -- exactly the cast the prior
    per-cache index_put performed -- so the kernel is a pure copy, byte-identical to the
    prior ``cache[page_ids, page_offsets] = values.to(cache.dtype)`` for each cache.

    Falls back to the per-cache ``_write_decode_cache_rows`` (identical semantics) when
    Triton/CUDA is unavailable, the hoisted address is missing, a cache is not a
    contiguous 3-D paged tensor, or fewer than two caches would be written.
    """
    # Skip empty value tensors -- matches the per-cache ``_write_decode_cache_rows``
    # ``numel() == 0`` guard so a degenerate (state_dim 0) cache is never written.
    pairs = [(c, v) for c, v in zip(caches, values) if v.numel() > 0]
    if not pairs:
        return
    caches = [c for c, _ in pairs]
    values = [v for _, v in pairs]

    use_fused = (
        _HAS_TRITON
        and page_ids is not None
        and page_offsets is not None
        and len(caches) >= 2
        and all(c.is_cuda and c.dim() == 3 and c.is_contiguous() for c in caches)
    )
    if not use_fused:
        # Byte-identical per-cache path (the original write, one index_put each).
        for cache, value in zip(caches, values):
            _write_decode_cache_rows(
                cache,
                value,
                seq_idx,
                input_pos,
                cu_num_pages,
                cache_loc,
                page_ids,
                page_offsets,
            )
        return

    n_cache = len(caches)
    n_rows = int(page_ids.shape[0])
    # Pre-cast each value to its cache dtype (no-op when already matching) and make it
    # row-contiguous ``[N, S]``. This is exactly the cast the per-cache index_put did
    # (e.g. the compressor kv/gate bf16 -> fp32 cast), so no extra copy_cast kernel is
    # introduced; ``.contiguous()`` is a no-op on the already-contiguous decode rows.
    srcs = [v.to(c.dtype).contiguous() for c, v in zip(caches, values)]
    dims = [int(c.shape[-1]) for c in caches]
    tokens_per_block = int(caches[0].shape[1])
    max_dim = max(dims)
    BLOCK_S = min(triton.next_power_of_2(max_dim), 1024)

    # The kernel has 5 fixed pointer slots; unused ones reuse the last real cache and
    # are never launched (grid dim 1 == ``n_cache``).
    while len(srcs) < 5:
        srcs.append(srcs[-1])
        caches.append(caches[-1])
        dims.append(dims[-1])

    grid = (n_rows, n_cache, triton.cdiv(max_dim, BLOCK_S))
    _multi_current_token_store_kernel[grid](
        page_ids,
        page_offsets,
        srcs[0],
        caches[0],
        dims[0],
        srcs[1],
        caches[1],
        dims[1],
        srcs[2],
        caches[2],
        dims[2],
        srcs[3],
        caches[3],
        dims[3],
        srcs[4],
        caches[4],
        dims[4],
        tokens_per_block,
        N_CACHES=n_cache,
        BLOCK_S=BLOCK_S,
        num_warps=4,
    )


def _fused_local_window_pagemap(
    input_pos: torch.Tensor,  # [N]
    seq_idx: torch.Tensor,  # [N]
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    window_size: int,
    tokens_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton wrapper for the fused local-window page map.

    Returns ``(page_ids, page_offsets, valid)`` -- both addresses ``int64`` of
    shape ``[N, window_size]`` and the combined (position && page) validity mask,
    bit-identical to ``_decode_local_cache_rows``' position gen + the
    ``_decode_page_ids_and_offsets`` translate.  The caller does the final
    ``swa_cache[page_ids, page_offsets]`` gather.
    """
    if cache_loc.numel() == 0:
        raise ValueError("cache_loc must contain at least one page id")
    num_decode = int(input_pos.shape[0])
    device = input_pos.device
    page_ids = torch.empty(num_decode, window_size, dtype=torch.long, device=device)
    page_offsets = torch.empty(num_decode, window_size, dtype=torch.long, device=device)
    valid = torch.empty(num_decode, window_size, dtype=torch.bool, device=device)
    n_elements = num_decode * window_size
    if n_elements == 0:
        return page_ids, page_offsets, valid
    BLOCK = 256
    grid = (triton.cdiv(n_elements, BLOCK),)
    _decode_local_window_pagemap_kernel[grid](
        input_pos.contiguous(),
        seq_idx.contiguous(),
        cu_num_pages,
        cache_loc,
        page_ids,
        page_offsets,
        valid,
        n_elements,
        window_size,
        tokens_per_block,
        cache_loc.numel() - 1,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return page_ids, page_offsets, valid


def _fused_assemble_selected_kv(
    swa_cache: torch.Tensor,  # [P, T, D] paged local kv cache
    mhc_cache: torch.Tensor,  # [P, T, D] paged compressed kv cache (same P/T/D/dtype)
    selected_rows: torch.Tensor,  # [B, TOPK] int64 selected compressed row ids (-1 = pad)
    compressed_valid: torch.Tensor,  # [B, TOPK] bool indexer/candidate validity
    input_pos: torch.Tensor,  # [B] int
    seq_idx: torch.Tensor,  # [B] int
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch paged assembly of the decode ``selected_kv`` block + ``rel_topk``.

    Reads the local-window rows from ``swa_cache`` and the caller-selected
    compressed rows from ``mhc_cache`` directly (folding the two page-map
    translations, the two row gathers, the selected_kv/valid_rows ``torch.cat``s and
    the attend's arange/where), returning the contiguous ``[B, window+TOPK, D]``
    selected-KV tensor and the ``[B, window+TOPK]`` relative row indices consumed by
    ``_decode_attention_from_selected``.  Byte-identical to the eager
    gather/cat/where chain it replaces (see ``_dsv4_assemble_selected_kv_kernel``).
    """
    if cache_loc.numel() == 0:
        raise ValueError("cache_loc must contain at least one page id")
    num_decode = int(input_pos.shape[0])
    topk = int(selected_rows.shape[1])
    kv_rows = int(window_size) + topk
    head_dim = int(swa_cache.shape[-1])
    tokens_per_block = int(swa_cache.shape[1])
    out_kv = torch.empty(num_decode, kv_rows, head_dim, dtype=dtype, device=swa_cache.device)
    out_relidx = torch.empty(num_decode, kv_rows, dtype=torch.int64, device=swa_cache.device)
    n_programs = num_decode * kv_rows
    if n_programs == 0:
        return out_kv, out_relidx
    BLOCK_D = triton.next_power_of_2(head_dim)
    grid = (n_programs,)
    _dsv4_assemble_selected_kv_kernel[grid](
        swa_cache,
        mhc_cache,
        selected_rows.contiguous(),
        compressed_valid.contiguous(),
        input_pos.contiguous(),
        seq_idx.contiguous(),
        cu_num_pages,
        cache_loc,
        out_kv,
        out_relidx,
        num_decode,
        kv_rows,
        int(window_size),
        topk,
        int(compress_ratio),
        tokens_per_block,
        head_dim,
        cache_loc.numel() - 1,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out_kv, out_relidx


def _fused_fullrange_candidate_rows(
    kv_cache: torch.Tensor,  # [P, T, S] paged compressor kv cache
    gate_cache: torch.Tensor,  # [P, T, S] paged compressor gate cache
    page_ids: torch.Tensor,  # [B, M*R] int64 (hoisted full-range page map)
    page_offsets: torch.Tensor,  # [B, M*R] int64
    valid: torch.Tensor,  # [B, M*R] bool
    ape: torch.Tensor,  # [R, 2*head_dim]
    norm_weight: torch.Tensor,  # [head_dim]
    rms_norm_eps: float,
    compress_ratio: int,
    head_dim: int,
    max_compressed_len: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One-launch ratio-4 full-range candidate-row reconstruction (idea_0075).

    Replaces the gather / row-shift / concat / where / pool / rmsnorm swarm of
    ``_batched_overlap_compressed_rows_fullrange`` with a single kernel that emits
    the post-pool, post-rmsnorm candidate rows ``[B, max_compressed_len, head_dim]``
    (pre-rope) directly from the paged caches and the hoisted page maps.  The
    caller keeps the rope / quantize tail.
    """
    num_rows = int(page_ids.shape[0])
    m = int(max_compressed_len)
    out = torch.empty(num_rows, m, head_dim, device=kv_cache.device, dtype=dtype)
    if num_rows == 0 or m == 0 or head_dim == 0:
        return out
    two_r = 2 * compress_ratio
    grid = (num_rows * m,)
    _dsv4_fullrange_candidate_rows_kernel[grid](
        kv_cache,
        gate_cache,
        page_ids.contiguous(),
        page_offsets.contiguous(),
        valid.contiguous(),
        ape.contiguous(),
        norm_weight.contiguous(),
        out,
        m,
        compress_ratio,
        head_dim,
        int(kv_cache.shape[1]),
        int(kv_cache.shape[2]),
        int(ape.shape[1]),
        m * compress_ratio,
        float(rms_norm_eps),
        TWO_R=two_r,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out


def _fused_compressed_row_update_r4(
    kv_cache: torch.Tensor,  # [P, T, S=2*head_dim] fp32 paged compressor kv cache
    gate_cache: torch.Tensor,  # [P, T, S] fp32 paged compressor gate cache
    overlap_page_map: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],  # each [N, 2*ratio]
    ape: torch.Tensor,  # [ratio, 2*head_dim]
    norm_weight: torch.Tensor,  # [head_dim]
    cos_table: torch.Tensor,  # [n_pos, rope_dim//2]
    sin_table: torch.Tensor,  # [n_pos, rope_dim//2]
    row_position_id: torch.Tensor,  # [N] int64, already clamped into [0, n_pos)
    row_valid: torch.Tensor,  # [N] bool -- store the row iff valid
    mhc_page_ids: torch.Tensor,  # [N] int64 write page id per row
    mhc_page_offsets: torch.Tensor,  # [N] int64 in-page offset per row
    mhc_cache: torch.Tensor,  # [P, T, head_dim] paged mhc cache (mutated in place)
    rms_norm_eps: float,
    compress_ratio: int,
    head_dim: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> None:
    """Two-launch ratio-4 main-compressor compressed-row update (idea_0007).

    Replaces the ``_batched_compressed_rows_from_paged_state`` (overlap) reconstruction
    swarm + ``_apply_compressed_rope_and_quantize`` rope/fp8 tail + ``cos``/``sin``
    gathers + ``_masked_write_decode_cache_rows`` store with two kernels: stage 1
    reconstructs the post-rmsnorm rows from the paged caches and the hoisted overlap
    band map; stage 2 fp8-quantizes the nope slice, RoPE-rotates the pe slice and
    validity-masked-stores the row into ``mhc_cache``.  Invalid rows write nothing
    (byte-identical to the prior read-old + write-back no-op).
    """
    ovl_page_ids, ovl_page_offsets, ovl_valid = overlap_page_map
    n = int(ovl_page_ids.shape[0])
    if n == 0 or head_dim == 0:
        return
    two_r = 2 * compress_ratio
    block_d = triton.next_power_of_2(head_dim)

    # Stage 1: gather + ape + mask + softmax-pool + rmsnorm -> post-rmsnorm rows.
    normed = torch.empty(n, head_dim, device=kv_cache.device, dtype=dtype)
    grid = (n,)
    _dsv4_compressed_row_r4_front_kernel[grid](
        kv_cache,
        gate_cache,
        ovl_page_ids.contiguous(),
        ovl_page_offsets.contiguous(),
        ovl_valid.contiguous(),
        ape.contiguous(),
        norm_weight.contiguous(),
        normed,
        n,
        compress_ratio,
        head_dim,
        int(kv_cache.shape[1]),
        int(kv_cache.shape[2]),
        int(ape.shape[1]),
        two_r,
        float(rms_norm_eps),
        TWO_R=two_r,
        BLOCK_D=block_d,
        num_warps=4,
    )

    # Stage 2: fp8(nope) + rope(pe) + validity-masked store into mhc_cache (shared tail).
    _launch_compressed_rope_fp8_store(
        normed,
        cos_table,
        sin_table,
        row_position_id,
        row_valid,
        mhc_page_ids,
        mhc_page_offsets,
        mhc_cache,
        head_dim,
        rope_dim,
    )


def _launch_compressed_rope_fp8_store(
    normed: torch.Tensor,  # [N, head_dim] post-rmsnorm rows (activation dtype)
    cos_table: torch.Tensor,  # [n_pos, rope_dim//2]
    sin_table: torch.Tensor,  # [n_pos, rope_dim//2]
    row_position_id: torch.Tensor,  # [N] int64, already clamped into [0, n_pos)
    row_valid: torch.Tensor,  # [N] bool -- store the row iff valid
    mhc_page_ids: torch.Tensor,  # [N] int64 write page id per row
    mhc_page_offsets: torch.Tensor,  # [N] int64 in-page offset per row
    mhc_cache: torch.Tensor,  # [P, T, head_dim] paged mhc cache (mutated in place)
    head_dim: int,
    rope_dim: int,
) -> None:
    """Shared stage-2 tail: fp8(nope) + interleaved RoPE(pe) + validity-masked store.

    Launches ``_dsv4_rope_fp8_masked_store_kernel`` (idea_0007) over the ``[N, head_dim]``
    post-rmsnorm rows.  Ratio-agnostic: the same tail serves both the ratio-4 (overlap,
    idea_0007) and ratio-128 (dense, idea_0039) main-compressor compressed-row updates,
    since both produce an identical ``[N, head_dim]`` normed row and write the same
    ``head_dim``-wide mhc row.  Invalid rows write nothing (byte-identical to the prior
    read-old + write-back no-op).
    """
    n = int(normed.shape[0])
    if n == 0 or head_dim == 0:
        return
    nope_dim = head_dim - rope_dim
    dh = rope_dim // 2
    block_d = triton.next_power_of_2(head_dim)
    grid = (n,)
    _dsv4_rope_fp8_masked_store_kernel[grid](
        normed,
        cos_table,
        sin_table,
        row_position_id,
        row_valid,
        mhc_page_ids,
        mhc_page_offsets,
        mhc_cache,
        mhc_cache.stride(0),
        mhc_cache.stride(1),
        mhc_cache.stride(2),
        int(cos_table.stride(0)),
        n,
        head_dim,
        nope_dim,
        dh,
        FP8_BLOCK=64,
        NUM_FP8_BLOCKS=nope_dim // 64,
        BLOCK_D=block_d,
        BLOCK_DH=triton.next_power_of_2(dh),
        MAX_VAL=448.0,
        MIN_VAL=1.0e-4,
        num_warps=4,
    )


def _paged_compress_pool(
    kv_cache: torch.Tensor,  # [P, T, S] fp32 paged compressor kv cache
    gate_cache: torch.Tensor,  # [P, T, S] fp32 paged compressor gate cache
    page_ids: torch.Tensor,  # [N, ratio] int64 page id per ratio slot
    page_offsets: torch.Tensor,  # [N, ratio] int64 in-page offset per ratio slot
    ape: torch.Tensor,  # [ratio, S] fp32 (column d in [0, head_dim) used)
    ratio: int,
    head_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Paged softmax-weighted pool for the dense (ratio-128) compressed-row front.

    Fuses the two paged compressor kv/gate gathers, the ``.to(dtype)`` casts, the ape add
    and the ``deepseek_v4_compress_pool`` launch into one D-tiled kernel that reads the
    ``[N, ratio]`` page map directly.  Mirrors ``deepseek_v4_compress_pool``'s BLOCK_D
    occupancy heuristic (start at the maximal D-block and halve while the grid is below the
    ~512-CTA machine-fill target, floor 16) so the small decode ``N`` is not
    occupancy-starved.  Returns the pooled ``[N, head_dim]`` row in ``dtype``.
    """
    n = int(page_ids.shape[0])
    pooled = torch.empty(n, head_dim, device=kv_cache.device, dtype=dtype)
    if n == 0 or head_dim == 0:
        return pooled
    cap = min(128, triton.next_power_of_2(head_dim))
    block_d = cap
    while block_d > 16 and n * triton.cdiv(head_dim, block_d) < 512:
        block_d //= 2
    grid = (n, triton.cdiv(head_dim, block_d))
    _dsv4_paged_compress_pool_kernel[grid](
        kv_cache,
        gate_cache,
        page_ids.contiguous(),
        page_offsets.contiguous(),
        ape.contiguous(),
        pooled,
        n,
        ratio,
        head_dim,
        int(kv_cache.shape[1]),
        int(kv_cache.shape[2]),
        int(ape.shape[1]),
        ratio,
        BLOCK_R=triton.next_power_of_2(ratio),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return pooled


def _fused_compressed_row_update_r128(
    kv_cache: torch.Tensor,  # [P, T, S=head_dim] fp32 paged compressor kv cache
    gate_cache: torch.Tensor,  # [P, T, S] fp32 paged compressor gate cache
    positions_page_ids: torch.Tensor,  # [N, ratio] int64 page id per ratio slot
    positions_page_offsets: torch.Tensor,  # [N, ratio] int64 in-page offset per ratio slot
    ape: torch.Tensor,  # [ratio, head_dim]
    norm_weight: torch.Tensor,  # [head_dim]
    cos_table: torch.Tensor,  # [n_pos, rope_dim//2]
    sin_table: torch.Tensor,  # [n_pos, rope_dim//2]
    row_position_id: torch.Tensor,  # [N] int64, already clamped into [0, n_pos)
    row_valid: torch.Tensor,  # [N] bool -- store the row iff valid
    mhc_page_ids: torch.Tensor,  # [N] int64 write page id per row
    mhc_page_offsets: torch.Tensor,  # [N] int64 in-page offset per row
    mhc_cache: torch.Tensor,  # [P, T, head_dim] paged mhc cache (mutated in place)
    rms_norm_eps: float,
    compress_ratio: int,
    head_dim: int,
    rope_dim: int,
    dtype: torch.dtype,
) -> None:
    """Three-launch ratio-128 (dense) main-compressor compressed-row update (idea_0039).

    The dense-path analogue of ``_fused_compressed_row_update_r4``.  Ratio-4 fused its
    reconstruction in a single one-program-per-row front kernel because its ``[2*ratio,
    head_dim]`` pool tile fits one program; the ratio-128 ``[ratio, head_dim]`` tile does
    not, so the pool is D-tiled instead (``_paged_compress_pool``) and RMSNorm is a
    separate reduction over ``head_dim`` (``_compressor_rms_norm``, the shipped fused
    ``triton_rms_norm``).  The rope/fp8/validity-masked-store tail is the shared
    ``_launch_compressed_rope_fp8_store``.  This replaces the dense-branch
    ``_batched_compressed_rows_from_paged_state`` reconstruction (2 paged gathers, 2 casts,
    the ape add, the pool), the ``_apply_compressed_rope_and_quantize`` rope/fp8 tail, the
    ``cos``/``sin`` gathers and the ``_masked_write_decode_cache_rows`` store.  The pooled
    row matches ``gather + ape + deepseek_v4_compress_pool`` to <=1 ULP (same rounding
    points; only the fp32 ratio-axis reduction order differs) and the stored row matches the
    eager path up to the rsqrt (bf16-absorbed) and the rope FMA (<=1 ULP).  Invalid rows
    write nothing.
    """
    n = int(positions_page_ids.shape[0])
    if n == 0 or head_dim == 0:
        return
    # Stage 1: paged gather(kv/gate) + ape-add + softmax-pool -> pooled [N, head_dim].
    pooled = _paged_compress_pool(
        kv_cache,
        gate_cache,
        positions_page_ids,
        positions_page_offsets,
        ape,
        compress_ratio,
        head_dim,
        dtype,
    )
    # Stage 1b: RMSNorm over head_dim (reuse the shipped fused triton_rms_norm).
    normed = _compressor_rms_norm(pooled, norm_weight, rms_norm_eps)
    # Stage 2: fp8(nope) + rope(pe) + validity-masked store (shared tail).
    _launch_compressed_rope_fp8_store(
        normed,
        cos_table,
        sin_table,
        row_position_id,
        row_valid,
        mhc_page_ids,
        mhc_page_offsets,
        mhc_cache,
        head_dim,
        rope_dim,
    )


def _fused_index_score(
    q_index: torch.Tensor,  # [N, H, D] index queries (fp16/bf16)
    index_k: torch.Tensor,  # [N, C, D] compressed candidate keys (same dtype as q)
    indexer_weights: torch.Tensor,  # [N, H] per-head indexer weights
    visible_len: torch.Tensor,  # [N] int visible candidate count per decode row
    max_compressed_len: int,
) -> torch.Tensor:
    """One-launch fused ratio-4 lightning-indexer score (idea_0004).

    Replaces ``(matmul(q_index, index_k^T).float().relu() * indexer_weights.float()
    .unsqueeze(-1)).sum(dim=1)`` + ``masked_fill(~visible, -inf)`` with a single kernel
    that emits the ``[N, max_compressed_len]`` masked score consumed by the decode
    top-k, without ever materializing the ``[N, H, C]`` head-by-candidate score or the
    separate masked ``[N, C]`` tensor. The matmul is rounded to the input dtype and the
    head reduction runs in fp32, so scores match the reference to within one ULP and the
    top-k selection/order is preserved (validated in the op unit tests).
    """
    num_rows = int(q_index.shape[0])
    h = int(q_index.shape[1])
    d = int(q_index.shape[2])
    c = int(max_compressed_len)
    out = torch.empty(num_rows, c, device=q_index.device, dtype=torch.float32)
    if num_rows == 0 or c == 0:
        return out
    block_c = 128 if c >= 128 else triton.next_power_of_2(c)
    grid = (num_rows, triton.cdiv(c, block_c))
    _dsv4_index_score_kernel[grid](
        q_index.contiguous(),
        index_k.contiguous(),
        indexer_weights.contiguous().to(torch.float32),
        visible_len.contiguous(),
        out,
        h,
        c,
        D=d,
        D_BLOCK=triton.next_power_of_2(d),
        H_BLOCK=triton.next_power_of_2(h),
        BLOCK_C=block_c,
        num_warps=4,
    )
    return out


def _fused_topk_select(
    index_score: torch.Tensor,  # [N, C] fp32 masked index score
    index_topk: int,
    topk_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch exact decode top-k row selection (idea_0046).

    Replaces the ``index_score.topk(topk_count)`` + ``isfinite`` + ``where`` +
    pad tail of ``_select_decode_ratio4_indexer_rows`` with a single kernel that
    emits the padded ``[N, index_topk]`` rows / validity directly.  Byte-identical
    to the eager chain for finite and ``-inf`` scores, tie order included
    (validated in the op unit test); see ``_dsv4_topk_select_kernel``.
    """
    num_rows, c = int(index_score.shape[0]), int(index_score.shape[1])
    device = index_score.device
    rows = torch.empty(num_rows, index_topk, dtype=torch.int64, device=device)
    valid = torch.empty(num_rows, index_topk, dtype=torch.uint8, device=device)
    if num_rows == 0 or index_topk == 0:
        return rows, valid.view(torch.bool)
    if c == 0 or topk_count <= 0:
        rows.fill_(-1)
        valid.zero_()
        return rows, valid.view(torch.bool)
    _dsv4_topk_select_kernel[(num_rows,)](
        index_score.contiguous(),
        rows,
        valid,
        c,
        index_topk,
        min(topk_count, c),
        BLOCK_C=triton.next_power_of_2(c),
        BLOCK_K=triton.next_power_of_2(index_topk),
        num_warps=4,
    )
    return rows, valid.view(torch.bool)


def _can_use_fused_sparse_attention(
    q: torch.Tensor, kv: torch.Tensor, topk_idxs: torch.Tensor
) -> bool:
    """Whether the fused Triton attend kernel supports these inputs.

    The pure-torch chunk loop remains the fallback for CPU, fp32, empty-kv, and
    tiny-head-dim shapes (the latter cannot use ``tl.dot``).
    """
    return (
        _HAS_TRITON
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and kv.shape[1] > 0  # kv_rows
        and q.shape[-1] >= 16  # head_dim large enough for tl.dot contraction
        and topk_idxs.shape[-1] > 0  # k_select
    )


def _fused_sparse_attention_triton(
    q_flat: torch.Tensor,  # [num_tokens, num_heads, D]
    kv: torch.Tensor,  # [batch_size, kv_rows, D]
    attn_sink: torch.Tensor,  # [num_heads]
    topk_flat: torch.Tensor,  # [num_tokens, k_select]
    softmax_scale: float,
    batch_idxs: torch.Tensor,  # [num_tokens]
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = q_flat.shape
    _, kv_rows, _ = kv.shape
    k_select = topk_flat.shape[1]

    q_flat = q_flat.contiguous()
    kv = kv.contiguous()
    topk_flat = topk_flat.contiguous()
    sink = attn_sink.contiguous()
    # Keep batch_idxs in its native int dtype (the kernels upcast to int64); an
    # explicit .to(int32) here would add a cast kernel into the captured cudagraph.
    batch_idxs = batch_idxs.contiguous()
    out = torch.empty_like(q_flat)

    scale_log2 = softmax_scale * _LOG2E
    d_block = triton.next_power_of_2(head_dim)

    is_decode = num_tokens <= _SPARSE_ATTN_SPLITK_MAX_TOKENS
    # Head-parallelism-starved decode (per-rank head count <= HEAD_BLOCK, e.g. TP8
    # per-rank H=8): take the small-head split-K config (smaller SEQ_BLOCK / fewer
    # split-K warps / more reduce warps).  See _SPARSE_ATTN_DECODE_SMALL_H_* above.
    small_head = is_decode and num_heads <= _SPARSE_ATTN_DECODE_SMALL_H_MAX_HEADS

    sk_num_warps = _SPARSE_ATTN_DECODE_NUM_WARPS
    rd_num_warps = _SPARSE_ATTN_REDUCE_NUM_WARPS
    if small_head:
        seq_block = _SPARSE_ATTN_DECODE_SMALL_H_SEQ_BLOCK
        sk_num_warps = _SPARSE_ATTN_DECODE_SMALL_H_NUM_WARPS
        rd_num_warps = _SPARSE_ATTN_DECODE_SMALL_H_REDUCE_NUM_WARPS
    else:
        seq_block = 64
    total_blocks = triton.cdiv(k_select, seq_block)
    # Split-K key partitions (used iff use_splitk below).  Computed up front so the
    # decode HEAD_BLOCK can be sized against the resulting head_groups*num_parts.
    num_parts = min(total_blocks, _SPARSE_ATTN_MAX_PARTS)

    # Pick head grouping.  At decode (few query tokens) the GPU is occupancy-starved,
    # so the split-K path also shrinks HEAD_BLOCK to expose more head-groups, sizing
    # head_groups*num_parts to ~one wave of SMs (_SPARSE_ATTN_DECODE_CTA_TARGET).
    # HEAD_BLOCK is kept as large as possible (better MMA M-utilization, smaller fp32
    # acc footprint) subject to that CTA budget: at full H=64 this is the original
    # HEAD_BLOCK=8 (8 head-groups).  Under TP sharding (per-rank H<=8) the default
    # would floor HEAD_BLOCK to 1; the small-head path instead packs a few heads per
    # CTA (fatter M-tile) over more split-K parts.  At prefill the simple kernel
    # already saturates the GPU with token*head parallelism.
    if small_head:
        head_block = min(_SPARSE_ATTN_DECODE_SMALL_H_HEAD_BLOCK, num_heads)
    elif is_decode:
        if total_blocks > 1:
            target_groups = max(1, _SPARSE_ATTN_DECODE_CTA_TARGET // num_parts)
            head_block = max(
                1, min(_SPARSE_ATTN_DECODE_HEAD_BLOCK, triton.cdiv(num_heads, target_groups))
            )
        else:
            head_block = min(_SPARSE_ATTN_DECODE_HEAD_BLOCK, num_heads)
    else:
        head_block = 16 if num_heads >= 16 else triton.next_power_of_2(num_heads)
    head_groups = triton.cdiv(num_heads, head_block)
    base_programs = num_tokens * head_groups
    use_splitk = base_programs < _SPARSE_ATTN_SM_TARGET and total_blocks > 1

    if use_splitk:
        # Split the key reduction across NUM_PARTS CTAs to fill idle SMs, then
        # combine + fold sink + normalize in a cheap reduction kernel.
        ws_acc = torch.empty(
            num_tokens, num_heads, num_parts, d_block, device=q_flat.device, dtype=torch.float32
        )
        ws_ml = torch.empty(
            num_tokens, num_heads, num_parts, 2, device=q_flat.device, dtype=torch.float32
        )
        grid = (num_tokens, head_groups, num_parts)
        _fused_sparse_attention_splitk_kernel[grid](
            q_flat,
            kv,
            topk_flat,
            batch_idxs,
            ws_acc,
            ws_ml,
            num_heads,
            kv_rows,
            k_select,
            SCALE_LOG2=scale_log2,
            D=head_dim,
            D_BLOCK=d_block,
            SEQ_BLOCK=seq_block,
            HEAD_BLOCK=head_block,
            NUM_PARTS=num_parts,
            num_warps=sk_num_warps,
            num_stages=2,
        )
        _fused_sparse_attention_reduce_kernel[(num_tokens, num_heads)](
            ws_acc,
            ws_ml,
            sink,
            out,
            num_heads,
            D=head_dim,
            D_BLOCK=d_block,
            NUM_PARTS=num_parts,
            num_warps=rd_num_warps,
        )
        return out

    grid = (num_tokens, head_groups)
    _fused_sparse_attention_kernel[grid](
        q_flat,
        kv,
        topk_flat,
        sink,
        batch_idxs,
        out,
        num_heads,
        kv_rows,
        k_select,
        SCALE_LOG2=scale_log2,
        D=head_dim,
        D_BLOCK=d_block,
        SEQ_BLOCK=seq_block,
        HEAD_BLOCK=head_block,
        num_warps=4,
        num_stages=2,
    )
    return out


def _deepseek_v4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference DeepSeek V4 sparse attention implementation.

    Args:
        q: Query states with shape ``[batch, seq_len, num_heads, head_dim]``.
        kv: Shared sparse key/value rows with shape ``[batch, kv_rows, head_dim]``.
        attn_sink: Per-head sink logits with shape ``[num_heads]``.
        topk_idxs: Selected row indices into ``kv`` with shape
            ``[batch, seq_len, k_select]``. Duplicate indices are preserved and
            receive independent probability mass. Negative indices are masked
            slots and receive zero probability.
        softmax_scale: Scale applied to query/key logits before adding the sink
            logit.

    Returns:
        Attention output with shape ``[batch, seq_len, num_heads, head_dim]``.
        The sink participates in softmax normalization but contributes no value
        vector.
    """
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)

    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    batch_size, seq_len, num_heads, q_head_dim = q.shape
    _, _, k_select = topk_idxs.shape
    num_tokens = batch_size * seq_len
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if num_tokens == 0:
        return output

    chunk_size = _sparse_attention_query_chunk_size(
        num_tokens, num_heads, q_head_dim, k_select, compute_dtype
    )

    q_flat = q.reshape(num_tokens, num_heads, q_head_dim)
    topk_flat = topk_idxs.reshape(num_tokens, k_select)
    batch_idxs = torch.arange(batch_size, device=q.device).view(batch_size, 1)
    batch_idxs = batch_idxs.expand(batch_size, seq_len).reshape(num_tokens)

    if _can_use_fused_sparse_attention(q, kv, topk_idxs):
        # Fused on-the-fly gather+attend kernel: reads selected/compressed KV by
        # index inside the matmul instead of materializing the fp32 selected-KV
        # tensor and round-tripping it through HBM for the two matmuls + softmax.
        out_flat = _fused_sparse_attention_triton(
            q_flat, kv, attn_sink, topk_flat, softmax_scale, batch_idxs
        )
        return out_flat.reshape(q.shape)

    output_flat = output.reshape(num_tokens, num_heads, q_head_dim)
    sink_logits = attn_sink.to(dtype=compute_dtype).reshape(1, num_heads, 1)

    for start in range(0, num_tokens, chunk_size):
        end = min(start + chunk_size, num_tokens)
        topk_chunk = topk_flat[start:end]
        valid_topk = (topk_chunk >= 0) & (topk_chunk < kv.shape[1])
        selected_kv_compute = _gather_selected_kv(kv, topk_chunk, batch_idxs[start:end]).to(
            compute_dtype
        )
        q_compute = q_flat[start:end].to(compute_dtype)

        logits = torch.matmul(q_compute, selected_kv_compute.transpose(-1, -2))
        logits = logits * softmax_scale
        logits = logits.masked_fill((~valid_topk).unsqueeze(1), float("-inf"))
        chunk_sink_logits = sink_logits.expand(end - start, num_heads, 1)
        logits_with_sink = torch.cat([logits, chunk_sink_logits], dim=-1)

        weights_with_sink = torch.softmax(logits_with_sink, dim=-1, dtype=torch.float32)
        weights = weights_with_sink[..., :-1].to(compute_dtype)
        chunk_output = torch.matmul(weights, selected_kv_compute)
        output_flat[start:end].copy_(chunk_output.to(q.dtype))

    return output


def _build_placeholder_topk_idxs(
    window_size: int,
    compress_ratio: int,
    batch_size: int,
    seq_len: int,
    compressed_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Rebuild the DeepSeek-V4 compressed sparse-attention selection placeholder.

    Mirrors the model-side ``_window_topk_idxs`` + ``_compress_topk_idxs`` chain in
    ``modeling_deepseek_v4.py`` (kept in sync; the modeling test carries independent
    ``_ref_window_topk_idxs`` / ``_ref_compress_topk_idxs`` copies). The value-reading
    initial-prefill gather consumes these indices, but the cached decode path reads only
    their static width (``index_topk = topk_idxs.shape[-1] - window_size``). The model
    therefore emits a cheap width-only allocation and passes ``topk_is_placeholder=True``
    so the op rebuilds the real window+compressed selection on the eager prefill path,
    which keeps the per-layer arange/where/expand/cat/cast index chain out of the decode
    graph while leaving prefill/decode outputs bit-identical.
    """
    query_positions = torch.arange(seq_len, device=device).unsqueeze(1)
    key_positions = query_positions - window_size + 1 + torch.arange(window_size, device=device)
    key_positions = torch.where(
        (key_positions < 0) | (key_positions > query_positions),
        -1,
        key_positions,
    )
    window_idxs = key_positions.unsqueeze(0).expand(batch_size, -1, -1)

    compressed_positions = torch.arange(compressed_width, device=device)
    valid_lengths = torch.arange(1, seq_len + 1, device=device).unsqueeze(1) // compress_ratio
    compressed_idxs = compressed_positions.unsqueeze(0).expand(seq_len, -1)
    compressed_idxs = torch.where(compressed_idxs < valid_lengths, compressed_idxs + seq_len, -1)
    compressed_idxs = compressed_idxs.unsqueeze(0).expand(batch_size, -1, -1)

    return torch.cat((window_idxs, compressed_idxs), dim=-1).to(torch.int64)


@torch.library.custom_op("auto_deploy::torch_deepseek_v4_sparse_attention", mutates_args=())
def torch_deepseek_v4_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    topk_is_placeholder: bool = False,
) -> torch.Tensor:
    """DeepSeek V4 sparse source op with explicit compressor projections.

    ``topk_is_placeholder`` signals that ``topk_idxs`` is a width-only allocation (the
    model emits one for compressed layers, whose values the cached decode path never
    reads). When set for a compressed layer, the real window+compressed selection is
    rebuilt here from ``window_size``/``compress_ratio`` and ``topk_idxs``' width so the
    value-reading prefill gather stays bit-identical while the per-layer index chain is
    kept out of the model graph.
    """
    del (
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        enable_sharding,
        layer_type,
        layer_idx,
        head_dim,
    )
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_compress_ratio(compress_ratio)
    if compress_ratio:
        if max_compressed_len is None:
            raise ValueError("max_compressed_len is required for compressed attention.")
        if rope_dim is None:
            raise ValueError("rope_dim is required for compressed attention.")
        if topk_is_placeholder:
            if window_size is None:
                raise ValueError(
                    "window_size is required to rebuild the compressed topk placeholder."
                )
            compressed_width = int(topk_idxs.shape[-1]) - int(window_size)
            topk_idxs = _build_placeholder_topk_idxs(
                window_size, compress_ratio, q.shape[0], q.shape[1], compressed_width, q.device
            )
        compressed_kv = _build_full_compressed_kv(
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            rms_norm_eps,
            rope_dim,
            compress_ratio,
            max_compressed_len,
        ).to(kv.dtype)
        kv = torch.cat((kv, compressed_kv), dim=1)
    return _deepseek_v4_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale)


@torch_deepseek_v4_sparse_attention.register_fake
def torch_deepseek_v4_sparse_attention_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    softmax_scale: float,
    enable_sharding: bool = False,
    layer_type: str = "mha_sparse",
    layer_idx: Optional[int] = None,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    head_dim: Optional[int] = None,
    rope_dim: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    topk_is_placeholder: bool = False,
) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    del (
        softmax_scale,
        enable_sharding,
        layer_type,
        layer_idx,
        window_size,
        compress_ratio,
        max_compressed_len,
        head_dim,
        rope_dim,
        rms_norm_eps,
        topk_is_placeholder,
    )
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("cos_table", cos_table, 2)
    _validate_rank("sin_table", sin_table, 2)
    _validate_rank("position_ids", position_ids, 2)
    _validate_rank("indexer_q", indexer_q, 4)
    _validate_rank("indexer_weights", indexer_weights, 3)
    _validate_rank("indexer_compressor_kv", indexer_compressor_kv, 3)
    _validate_rank("indexer_compressor_gate", indexer_compressor_gate, 3)
    _validate_rank("indexer_compressor_ape", indexer_compressor_ape, 2)
    _validate_rank("indexer_compressor_norm_weight", indexer_compressor_norm_weight, 1)
    return q.new_empty(q.shape).contiguous()


@torch.library.custom_op(
    "auto_deploy::deepseek_v4_sparse_prepare_decode_page_addr", mutates_args=()
)
def deepseek_v4_sparse_prepare_decode_page_addr(
    input_pos: torch.Tensor,
    position_ids: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    tokens_per_block: int,
    overlap_max_compressed_len: int = 0,
    dense_max_compressed_len: int = 0,
) -> List[torch.Tensor]:
    """Precompute the current-token paged write address once per forward.

    Every DeepSeek V4 sparse-attention layer writes the freshly produced
    KV / compressor / indexer rows for the *current* decode token to the same
    logical position ``input_pos`` of caches that share one page table
    (``cu_num_pages`` / ``cache_loc``) and one ``tokens_per_block``.  The
    ``(page_id, page_offset)`` translation is therefore identical across all
    layers and across all of those caches, yet the per-layer op currently
    recomputes it once per current-token write (5x per layer).  Hoisting it
    into this prepare op -- a single graph node whose result feeds every layer
    -- removes that redundant, launch-bound page-map work while leaving the
    stored values byte-identical.

    Returns ``[page_ids, page_offsets]`` (both int64, shape ``[num_seq]``); the
    decode op slices ``[:num_decode]``.  Mirrors ``_decode_page_ids_and_offsets``
    for ``seq_idx = arange(num_seq)`` and ``positions = input_pos`` so the
    produced addresses are bit-identical to the per-layer computation.

    When ``overlap_max_compressed_len > 0`` (the production contract) it also
    emits the ratio-4 overlap band / full-range page maps (6 tensors) and the
    compressed-cache UPDATE metadata for the ratio-4 and ratio-128 layers
    (idea_0044): ``row_valid`` / query-relative rope position / ``mhc_cache``
    write address for each ratio (4 + 4 tensors), plus the ratio-128
    ``[num_seq, ratio]`` compressor read page map (2 tensors) -- 18 outputs
    total.  ``dense_max_compressed_len`` is the ratio-128 ``max_compressed_len``
    discovered from the graph; it selects the completed dense row.
    """
    num_seq = int(cu_num_pages.shape[0]) - 1
    positions_long = input_pos.reshape(-1)[:num_seq].to(torch.long)
    seq_idx_long = torch.arange(num_seq, dtype=torch.long, device=positions_long.device)
    safe_positions = positions_long.clamp(min=0)
    page_ordinals = safe_positions // tokens_per_block
    page_offsets = safe_positions % tokens_per_block
    page_start = cu_num_pages[seq_idx_long].to(torch.long)
    page_end = cu_num_pages[seq_idx_long + 1].to(torch.long)
    page_table_idx = page_start + page_ordinals
    valid = (positions_long >= 0) & (page_table_idx < page_end)
    safe_page_table_idx = torch.where(valid, page_table_idx, page_start)
    safe_page_table_idx = safe_page_table_idx.clamp(min=0, max=cache_loc.numel() - 1)
    page_ids = cache_loc[safe_page_table_idx].to(torch.long)
    outs = [page_ids, page_offsets]
    if overlap_max_compressed_len > 0:
        # Ratio-4 (overlap+indexer) compressed-row and full-range page maps. Every
        # ratio-4 layer resolves these identical (seq_idx, positions) addresses, so
        # hoisting them here drops the per-layer ``_decode_page_ids_and_offsets``
        # chain from both ``_batched_compressed_rows_from_paged_state`` (overlap
        # previous/current) and ``_batched_overlap_compressed_rows_fullrange``.
        # ``_page_ids_and_offsets_from_tpb`` is the shared translation, so the
        # produced addresses are bit-identical to the per-layer computation.
        ratio = _COMPRESS_RATIO_OVERLAP_INDEXER
        m = int(overlap_max_compressed_len)
        device = positions_long.device
        # Overlap band ``[anchor - ratio, anchor + ratio)``: the first ``ratio``
        # columns are the previous block, the last ``ratio`` the current block.
        row_idx = (positions_long // ratio).clamp(min=0, max=m - 1)
        anchor = row_idx * ratio
        band_offsets = torch.arange(2 * ratio, dtype=torch.long, device=device) - ratio
        ovl_positions = anchor.unsqueeze(1) + band_offsets.view(1, -1)
        ovl_page_ids, ovl_page_offsets, ovl_valid = _page_ids_and_offsets_from_tpb(
            tokens_per_block, seq_idx_long, ovl_positions, cu_num_pages, cache_loc
        )
        # Full candidate range ``[0, m * ratio)`` for the lightning-indexer path.
        full_positions = torch.arange(m * ratio, dtype=torch.long, device=device)
        full_positions = full_positions.view(1, -1).expand(num_seq, -1)
        full_page_ids, full_page_offsets, full_valid = _page_ids_and_offsets_from_tpb(
            tokens_per_block, seq_idx_long, full_positions, cu_num_pages, cache_loc
        )
        outs += [
            ovl_page_ids,
            ovl_page_offsets,
            ovl_valid,
            full_page_ids,
            full_page_offsets,
            full_valid,
        ]
        # Compressed-cache UPDATE metadata (idea_0044). Every layer of a given
        # compression ratio resolves the identical row_valid / query-relative rope
        # position / mhc write address (they depend only on input_pos / position_ids
        # and the shared page table); the dense ratio-128 layers additionally read the
        # identical [num_seq, ratio] compressor page map. Compute each ratio's bundle
        # once here via the shared metadata helper so it is bit-identical to the
        # per-layer ``_update_decode_compressed_caches`` computation, and thread it into
        # every matching layer. Fixed contract: the ratio-4 bundle (4 tensors) then the
        # ratio-128 bundle (6 tensors) -> 18 outputs total.
        position_ids_long = position_ids.reshape(-1)[:num_seq].to(torch.long)
        r4_valid, r4_pos, r4_mhc_pid, r4_mhc_poff, _, _ = _compressed_row_update_metadata(
            positions_long,
            position_ids_long,
            seq_idx_long,
            cu_num_pages,
            cache_loc,
            tokens_per_block,
            _COMPRESS_RATIO_OVERLAP_INDEXER,
            m,
            want_pos_map=False,
        )
        dense_m = max(int(dense_max_compressed_len), 1)
        (
            r128_valid,
            r128_pos,
            r128_mhc_pid,
            r128_mhc_poff,
            r128_pos_pid,
            r128_pos_poff,
        ) = _compressed_row_update_metadata(
            positions_long,
            position_ids_long,
            seq_idx_long,
            cu_num_pages,
            cache_loc,
            tokens_per_block,
            _COMPRESS_RATIO_DENSE,
            dense_m,
            want_pos_map=True,
        )
        outs += [
            r4_valid,
            r4_pos,
            r4_mhc_pid,
            r4_mhc_poff,
            r128_valid,
            r128_pos,
            r128_mhc_pid,
            r128_mhc_poff,
            r128_pos_pid,
            r128_pos_poff,
        ]
    return outs


@deepseek_v4_sparse_prepare_decode_page_addr.register_fake
def deepseek_v4_sparse_prepare_decode_page_addr_fake(
    input_pos: torch.Tensor,
    position_ids: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    tokens_per_block: int,
    overlap_max_compressed_len: int = 0,
    dense_max_compressed_len: int = 0,
) -> List[torch.Tensor]:
    num_seq = cu_num_pages.shape[0] - 1
    device = input_pos.device
    outs = [
        torch.empty(num_seq, dtype=torch.long, device=device),
        torch.empty(num_seq, dtype=torch.long, device=device),
    ]
    if overlap_max_compressed_len > 0:
        ratio = _COMPRESS_RATIO_OVERLAP_INDEXER
        m = int(overlap_max_compressed_len)
        dense_ratio = _COMPRESS_RATIO_DENSE
        outs += [
            torch.empty(num_seq, 2 * ratio, dtype=torch.long, device=device),
            torch.empty(num_seq, 2 * ratio, dtype=torch.long, device=device),
            torch.empty(num_seq, 2 * ratio, dtype=torch.bool, device=device),
            torch.empty(num_seq, m * ratio, dtype=torch.long, device=device),
            torch.empty(num_seq, m * ratio, dtype=torch.long, device=device),
            torch.empty(num_seq, m * ratio, dtype=torch.bool, device=device),
            # Compressed-cache UPDATE metadata (idea_0044): ratio-4 bundle (4) then
            # ratio-128 bundle (6). The [num_seq, dense_ratio] pos map shape is fixed
            # by the dense compression ratio (independent of dense_max_compressed_len).
            torch.empty(num_seq, dtype=torch.bool, device=device),  # r4_row_valid
            torch.empty(num_seq, dtype=torch.long, device=device),  # r4_row_position_id
            torch.empty(num_seq, dtype=torch.long, device=device),  # r4_mhc_page_ids
            torch.empty(num_seq, dtype=torch.long, device=device),  # r4_mhc_page_offsets
            torch.empty(num_seq, dtype=torch.bool, device=device),  # r128_row_valid
            torch.empty(num_seq, dtype=torch.long, device=device),  # r128_row_position_id
            torch.empty(num_seq, dtype=torch.long, device=device),  # r128_mhc_page_ids
            torch.empty(num_seq, dtype=torch.long, device=device),  # r128_mhc_page_offsets
            torch.empty(num_seq, dense_ratio, dtype=torch.long, device=device),  # r128_pos_page_ids
            torch.empty(
                num_seq, dense_ratio, dtype=torch.long, device=device
            ),  # r128_pos_page_offsets
        ]
    return outs


@torch.library.custom_op(
    "auto_deploy::torch_deepseek_v4_sparse_attention_with_cache",
    mutates_args=(
        "swa_cache",
        "mhc_cache",
        "compressor_kv_cache",
        "compressor_gate_cache",
        "indexer_compressor_kv_cache",
        "indexer_compressor_gate_cache",
    ),
)
def torch_deepseek_v4_sparse_attention_with_cache(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    cur_page_ids: torch.Tensor,
    cur_page_offsets: torch.Tensor,
    ovl_page_ids: torch.Tensor,
    ovl_page_offsets: torch.Tensor,
    ovl_valid: torch.Tensor,
    full_page_ids: torch.Tensor,
    full_page_offsets: torch.Tensor,
    full_valid: torch.Tensor,
    r4_row_valid: torch.Tensor,
    r4_row_position_id: torch.Tensor,
    r4_mhc_page_ids: torch.Tensor,
    r4_mhc_page_offsets: torch.Tensor,
    r128_row_valid: torch.Tensor,
    r128_row_position_id: torch.Tensor,
    r128_mhc_page_ids: torch.Tensor,
    r128_mhc_page_offsets: torch.Tensor,
    r128_pos_page_ids: torch.Tensor,
    r128_pos_page_offsets: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    rope_dim: Optional[int] = None,
    topk_is_placeholder: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference paged cached DeepSeek V4 sparse attention with compressor state."""
    _validate_deepseek_v4_sparse_attention_inputs(q, kv, attn_sink, topk_idxs)
    _validate_swa_cache_inputs(q, kv, swa_cache)
    _validate_swa_cache_inputs(q, kv, mhc_cache)
    _validate_rank("compressor_kv_cache", compressor_kv_cache, 3)
    _validate_rank("compressor_gate_cache", compressor_gate_cache, 3)
    _validate_rank("indexer_q", indexer_q, 4)
    _validate_rank("indexer_weights", indexer_weights, 3)
    _validate_rank("indexer_compressor_kv", indexer_compressor_kv, 3)
    _validate_rank("indexer_compressor_gate", indexer_compressor_gate, 3)
    _validate_rank("indexer_compressor_ape", indexer_compressor_ape, 2)
    _validate_rank("indexer_compressor_norm_weight", indexer_compressor_norm_weight, 1)
    _validate_rank("indexer_compressor_kv_cache", indexer_compressor_kv_cache, 3)
    _validate_rank("indexer_compressor_gate_cache", indexer_compressor_gate_cache, 3)
    _validate_compress_ratio(compress_ratio)
    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be positive when provided, got {window_size}")
    if compress_ratio:
        if window_size is None:
            raise ValueError("window_size is required for compressed cached attention.")
        if max_compressed_len is None or max_compressed_len <= 0:
            raise ValueError(
                "max_compressed_len must be positive for compressed cached attention, "
                f"got {max_compressed_len}"
            )
        if rope_dim is None:
            raise ValueError("rope_dim is required for compressed cached attention.")

    batch_info = BatchInfo(batch_info_host)
    num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()
    num_seq = num_prefill + num_decode
    active_tokens = num_prefill_tokens + num_decode
    q_flat = q.reshape(-1, *q.shape[2:])
    if active_tokens > q_flat.shape[0]:
        raise ValueError(
            f"active token count {active_tokens} exceeds flattened q tokens {q_flat.shape[0]}"
        )
    if num_prefill == 0 and num_decode > 0:
        return _deepseek_v4_sparse_attention_decode_with_cache(
            q,
            kv,
            attn_sink,
            topk_idxs,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm_weight,
            indexer_q,
            indexer_weights,
            indexer_compressor_kv,
            indexer_compressor_gate,
            indexer_compressor_ape,
            indexer_compressor_norm_weight,
            cos_table,
            sin_table,
            position_ids,
            input_pos,
            slot_idx,
            cu_num_pages,
            cache_loc,
            cur_page_ids,
            cur_page_offsets,
            ovl_page_ids,
            ovl_page_offsets,
            ovl_valid,
            full_page_ids,
            full_page_offsets,
            full_valid,
            r4_row_valid,
            r4_row_position_id,
            r4_mhc_page_ids,
            r4_mhc_page_offsets,
            r128_row_valid,
            r128_row_position_id,
            r128_mhc_page_ids,
            r128_mhc_page_offsets,
            r128_pos_page_ids,
            r128_pos_page_offsets,
            swa_cache,
            mhc_cache,
            compressor_kv_cache,
            compressor_gate_cache,
            indexer_compressor_kv_cache,
            indexer_compressor_gate_cache,
            num_decode,
            softmax_scale,
            window_size,
            compress_ratio,
            max_compressed_len,
            rms_norm_eps,
            rope_dim,
            out,
        )

    if compress_ratio and topk_is_placeholder:
        # The model emits a width-only placeholder for ``topk_idxs`` on compressed
        # layers; rebuild the real window+compressed selection here (once per prefill,
        # eager) for the value-reading initial-prefill gather. The pure-decode path
        # above returns before this and reads only the placeholder width, so it never
        # needs the rebuilt indices.
        if window_size is None:
            raise ValueError("window_size is required to rebuild the compressed topk placeholder.")
        compressed_width = int(topk_idxs.shape[-1]) - int(window_size)
        topk_idxs = _build_placeholder_topk_idxs(
            window_size, compress_ratio, q.shape[0], q.shape[1], compressed_width, q.device
        )

    seq_len_host = _to_host_long("seq_len", seq_len, num_seq)
    input_pos_host = _to_host_long("input_pos", input_pos, num_seq)
    cu_seqlen_host = _to_host_long("cu_seqlen", cu_seqlen, num_seq + 1)
    cu_num_pages_host = _to_host_long("cu_num_pages", cu_num_pages, num_seq + 1)
    num_page_entries = int(cu_num_pages_host[-1].item())
    cache_loc_host = _to_host_long("cache_loc", cache_loc, num_page_entries)
    del slot_idx, last_page_len

    output_flat = torch.zeros_like(q_flat)
    compressed_capacity = int(max_compressed_len) if compress_ratio else 0

    for seq_idx in range(num_seq):
        seq_len_i = int(seq_len_host[seq_idx].item())
        if seq_len_i == 0:
            continue
        flat_start = int(cu_seqlen_host[seq_idx].item())
        input_pos_i = int(input_pos_host[seq_idx].item())

        q_seq = q_flat[flat_start : flat_start + seq_len_i]
        kv_seq = _slice_sequence_tokens(kv, seq_idx, flat_start, seq_len_i)
        topk_seq = _slice_sequence_tokens(topk_idxs, seq_idx, flat_start, seq_len_i)
        position_ids_seq = _slice_sequence_positions(position_ids, seq_idx, flat_start, seq_len_i)
        indexer_q_seq = _slice_sequence_tokens(indexer_q, seq_idx, flat_start, seq_len_i)
        indexer_weights_seq = _slice_sequence_tokens(
            indexer_weights, seq_idx, flat_start, seq_len_i
        )
        if q_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} q slice has length {q_seq.shape[0]}, expected {seq_len_i}"
            )
        if kv_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} kv slice has length {kv_seq.shape[0]}, expected {seq_len_i}"
            )
        if topk_seq.shape[0] != seq_len_i:
            raise ValueError(
                f"Sequence {seq_idx} topk_idxs slice has length {topk_seq.shape[0]}, "
                f"expected {seq_len_i}"
            )

        _write_paged_cache_rows(
            kv_seq, swa_cache, seq_idx, input_pos_i, cu_num_pages_host, cache_loc_host
        )

        if compress_ratio:
            compressor_kv_seq = _slice_sequence_tokens(
                compressor_kv, seq_idx, flat_start, seq_len_i
            )
            compressor_gate_seq = _slice_sequence_tokens(
                compressor_gate, seq_idx, flat_start, seq_len_i
            )
            indexer_compressor_kv_seq = _slice_sequence_tokens(
                indexer_compressor_kv, seq_idx, flat_start, seq_len_i
            )
            indexer_compressor_gate_seq = _slice_sequence_tokens(
                indexer_compressor_gate, seq_idx, flat_start, seq_len_i
            )
            _update_compressed_paged_caches(
                compressor_kv_seq,
                compressor_gate_seq,
                position_ids_seq,
                compressor_ape,
                compressor_norm_weight,
                cos_table,
                sin_table,
                seq_idx,
                input_pos_i,
                cu_num_pages_host,
                cache_loc_host,
                mhc_cache,
                compressor_kv_cache,
                compressor_gate_cache,
                rms_norm_eps,
                rope_dim,
                compress_ratio,
                compressed_capacity,
            )
            mode = _compression_mode(compress_ratio)
            if mode.uses_indexer:
                _update_raw_paged_caches(
                    indexer_compressor_kv_seq,
                    indexer_compressor_gate_seq,
                    indexer_compressor_kv_cache,
                    indexer_compressor_gate_cache,
                    seq_idx,
                    input_pos_i,
                    cu_num_pages_host,
                    cache_loc_host,
                )

            if input_pos_i == 0:
                output_flat[flat_start : flat_start + seq_len_i] = (
                    torch_deepseek_v4_sparse_attention(
                        q_seq.unsqueeze(0),
                        kv_seq.unsqueeze(0),
                        attn_sink,
                        topk_seq.unsqueeze(0),
                        compressor_kv_seq.unsqueeze(0),
                        compressor_gate_seq.unsqueeze(0),
                        compressor_ape,
                        compressor_norm_weight,
                        cos_table,
                        sin_table,
                        position_ids_seq,
                        indexer_q_seq.unsqueeze(0),
                        indexer_weights_seq.unsqueeze(0),
                        indexer_compressor_kv_seq.unsqueeze(0),
                        indexer_compressor_gate_seq.unsqueeze(0),
                        indexer_compressor_ape,
                        indexer_compressor_norm_weight,
                        softmax_scale,
                        window_size=window_size,
                        compress_ratio=compress_ratio,
                        max_compressed_len=max_compressed_len,
                        rope_dim=rope_dim,
                        rms_norm_eps=rms_norm_eps,
                    ).squeeze(0)
                )
            else:
                output_flat[flat_start : flat_start + seq_len_i] = _cached_compressed_attention(
                    q_seq,
                    attn_sink,
                    swa_cache,
                    mhc_cache,
                    seq_idx,
                    input_pos_i,
                    cu_num_pages_host,
                    cache_loc_host,
                    position_ids_seq,
                    window_size,
                    compress_ratio,
                    compressed_capacity,
                    softmax_scale,
                    topk_seq=topk_seq,
                    indexer_q_seq=indexer_q_seq,
                    indexer_weights_seq=indexer_weights_seq,
                    indexer_compressor_kv_cache=indexer_compressor_kv_cache,
                    indexer_compressor_gate_cache=indexer_compressor_gate_cache,
                    indexer_compressor_ape=indexer_compressor_ape,
                    indexer_compressor_norm_weight=indexer_compressor_norm_weight,
                    cos_table=cos_table,
                    sin_table=sin_table,
                    rms_norm_eps=rms_norm_eps,
                    rope_dim=rope_dim,
                )
        elif input_pos_i == 0:
            kv_source = _prefill_kv_source(kv, kv_seq, seq_idx, num_seq)
            output_flat[flat_start : flat_start + seq_len_i] = _deepseek_v4_sparse_attention(
                q_seq.unsqueeze(0),
                kv_source,
                attn_sink,
                topk_seq.unsqueeze(0),
                softmax_scale,
            ).squeeze(0)
        elif window_size is not None:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_local_window_attention(
                q_seq,
                attn_sink,
                swa_cache,
                seq_idx,
                input_pos_i,
                cu_num_pages_host,
                cache_loc_host,
                window_size,
                softmax_scale,
            )
        else:
            output_flat[flat_start : flat_start + seq_len_i] = _cached_topk_attention(
                q_seq,
                attn_sink,
                topk_seq,
                swa_cache,
                seq_idx,
                cu_num_pages_host,
                cache_loc_host,
                softmax_scale,
            )

    if active_tokens < q_flat.shape[0]:
        output_flat[active_tokens:].zero_()
    output = output_flat.view_as(q)
    if out is not None:
        out.copy_(output)
        return out.new_empty(0)
    return output


@torch_deepseek_v4_sparse_attention_with_cache.register_fake
def torch_deepseek_v4_sparse_attention_with_cache_fake(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    compressor_kv: torch.Tensor,
    compressor_gate: torch.Tensor,
    compressor_ape: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    position_ids: torch.Tensor,
    indexer_q: torch.Tensor,
    indexer_weights: torch.Tensor,
    indexer_compressor_kv: torch.Tensor,
    indexer_compressor_gate: torch.Tensor,
    indexer_compressor_ape: torch.Tensor,
    indexer_compressor_norm_weight: torch.Tensor,
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    input_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlen: torch.Tensor,
    cu_num_pages: torch.Tensor,
    cache_loc: torch.Tensor,
    last_page_len: torch.Tensor,
    cur_page_ids: torch.Tensor,
    cur_page_offsets: torch.Tensor,
    ovl_page_ids: torch.Tensor,
    ovl_page_offsets: torch.Tensor,
    ovl_valid: torch.Tensor,
    full_page_ids: torch.Tensor,
    full_page_offsets: torch.Tensor,
    full_valid: torch.Tensor,
    r4_row_valid: torch.Tensor,
    r4_row_position_id: torch.Tensor,
    r4_mhc_page_ids: torch.Tensor,
    r4_mhc_page_offsets: torch.Tensor,
    r128_row_valid: torch.Tensor,
    r128_row_position_id: torch.Tensor,
    r128_mhc_page_ids: torch.Tensor,
    r128_mhc_page_offsets: torch.Tensor,
    r128_pos_page_ids: torch.Tensor,
    r128_pos_page_offsets: torch.Tensor,
    swa_cache: torch.Tensor,
    mhc_cache: torch.Tensor,
    compressor_kv_cache: torch.Tensor,
    compressor_gate_cache: torch.Tensor,
    indexer_compressor_kv_cache: torch.Tensor,
    indexer_compressor_gate_cache: torch.Tensor,
    softmax_scale: float,
    window_size: Optional[int] = None,
    compress_ratio: int = 0,
    max_compressed_len: Optional[int] = None,
    rms_norm_eps: float = 1e-6,
    rope_dim: Optional[int] = None,
    topk_is_placeholder: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out.new_empty(0)
    del topk_is_placeholder
    _validate_compress_ratio(compress_ratio)
    _validate_rank("q", q, 4)
    _validate_rank("kv", kv, 3)
    _validate_rank("attn_sink", attn_sink, 1)
    _validate_rank("topk_idxs", topk_idxs, 3)
    _validate_rank("compressor_kv", compressor_kv, 3)
    _validate_rank("compressor_gate", compressor_gate, 3)
    _validate_rank("compressor_ape", compressor_ape, 2)
    _validate_rank("compressor_norm_weight", compressor_norm_weight, 1)
    _validate_rank("cos_table", cos_table, 2)
    _validate_rank("sin_table", sin_table, 2)
    _validate_rank("position_ids", position_ids, 2)
    _validate_rank("indexer_q", indexer_q, 4)
    _validate_rank("indexer_weights", indexer_weights, 3)
    _validate_rank("indexer_compressor_kv", indexer_compressor_kv, 3)
    _validate_rank("indexer_compressor_gate", indexer_compressor_gate, 3)
    _validate_rank("indexer_compressor_ape", indexer_compressor_ape, 2)
    _validate_rank("indexer_compressor_norm_weight", indexer_compressor_norm_weight, 1)
    _validate_rank("swa_cache", swa_cache, 3)
    _validate_rank("mhc_cache", mhc_cache, 3)
    _validate_rank("compressor_kv_cache", compressor_kv_cache, 3)
    _validate_rank("compressor_gate_cache", compressor_gate_cache, 3)
    _validate_rank("indexer_compressor_kv_cache", indexer_compressor_kv_cache, 3)
    _validate_rank("indexer_compressor_gate_cache", indexer_compressor_gate_cache, 3)
    del (
        kv,
        attn_sink,
        topk_idxs,
        compressor_kv,
        compressor_gate,
        compressor_ape,
        compressor_norm_weight,
        cos_table,
        sin_table,
        position_ids,
        indexer_q,
        indexer_weights,
        indexer_compressor_kv,
        indexer_compressor_gate,
        indexer_compressor_ape,
        indexer_compressor_norm_weight,
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        cu_num_pages,
        cache_loc,
        last_page_len,
        cur_page_ids,
        cur_page_offsets,
        ovl_page_ids,
        ovl_page_offsets,
        ovl_valid,
        full_page_ids,
        full_page_offsets,
        full_valid,
        swa_cache,
        mhc_cache,
        compressor_kv_cache,
        compressor_gate_cache,
        indexer_compressor_kv_cache,
        indexer_compressor_gate_cache,
        softmax_scale,
        window_size,
        compress_ratio,
        max_compressed_len,
        rms_norm_eps,
        rope_dim,
        out,
    )
    return q.new_empty(q.shape).contiguous()


@AttentionRegistry.register("deepseek_v4_sparse")
class DeepSeekV4SparseAttention(AttentionDescriptor):
    """Cached DeepSeek V4 sparse attention descriptor for reference validation."""

    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        return len(_SOURCE_TENSOR_ARG_NAMES)

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return torch.ops.auto_deploy.torch_deepseek_v4_sparse_attention_with_cache.default

    @classmethod
    def get_standard_metadata_args(cls) -> list[str]:
        return [
            "batch_info_host",
            "seq_len",
            "input_pos",
            "slot_idx",
            "cu_seqlen",
            "cu_num_pages",
            "cache_loc",
            "last_page_len",
        ]

    @classmethod
    def get_prepare_extra_metadata_info(
        cls, any_source_attn_node: Node, sequence_info=None
    ) -> Tuple[Optional[PrepareMetadataCallable], int, List[Constant]]:
        """Hoist the current-token paged write address out of the per-layer op.

        Every layer writes the current decode token to logical position
        ``input_pos`` of caches that share one page table (``cu_num_pages`` /
        ``cache_loc``) and one ``tokens_per_block``, so the
        ``(page_id, page_offset)`` translation is identical across all layers
        and is recomputed redundantly today.  This registers
        ``deepseek_v4_sparse_prepare_decode_page_addr`` as a once-per-forward
        prepare op whose two outputs (``cur_page_ids`` / ``cur_page_offsets``)
        are wired as extra metadata into every cached-attention invocation.

        ``sequence_info`` is forwarded by the cache-insertion transform and
        supplies ``tokens_per_block`` (the cache page size) as a constant arg.
        """
        if sequence_info is None:
            raise RuntimeError(
                "DeepSeek V4 sparse attention requires sequence_info to hoist the "
                "current-token page address; the cache-insertion transform must "
                "forward it to get_prepare_extra_metadata_info."
            )
        tokens_per_block = int(sequence_info.tokens_per_block)
        # Discover the ratio-4 (overlap+indexer) and ratio-128 (dense) max_compressed_len
        # shared by every layer of the respective ratio so the prepare op can hoist their
        # compressed-row / full-range page maps and their compressed-cache update metadata
        # (idea_0044). Every layer of a given ratio shares one max_compressed_len (it is
        # derived only from compress_ratio and a global config), so the per-ratio max is
        # that uniform value. This method only receives one source node, so scan the shared
        # graph for all sparse-attn nodes.
        overlap_m = 0
        dense_m = 0
        source_op = cls.get_source_attention_op()
        for n in any_source_attn_node.graph.nodes:
            if not is_op(n, source_op):
                continue
            cr, mcl = extract_op_args(n, "compress_ratio", "max_compressed_len")
            if not isinstance(mcl, int) or mcl <= 0:
                continue
            if cr == _COMPRESS_RATIO_OVERLAP_INDEXER:
                overlap_m = max(overlap_m, mcl)
            elif cr == _COMPRESS_RATIO_DENSE:
                dense_m = max(dense_m, mcl)
        # Fixed 18-output contract: 2 current-token addresses + 6 ratio-4 map tensors
        # (overlap page_ids/page_offsets/valid + full-range page_ids/page_offsets/valid)
        # + 10 update-metadata tensors (ratio-4 row_valid/row_position_id/mhc page_ids/
        # page_offsets, then the same 4 for ratio-128, then the ratio-128 [N, ratio]
        # compressor read page_ids/page_offsets). Keep ``overlap_m`` / ``dense_m`` >= 1 so
        # the map shapes stay valid and the per-layer argument alignment is invariant even
        # when a ratio class is absent (the dummy maps are then never consumed).
        overlap_m = max(overlap_m, 1)
        dense_m = max(dense_m, 1)
        return (
            torch.ops.auto_deploy.deepseek_v4_sparse_prepare_decode_page_addr.default,
            18,
            [tokens_per_block, overlap_m, dense_m],
        )

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        kv_node, compressor_kv_node, indexer_compressor_kv_node = extract_op_args(
            source_attn_node,
            "kv",
            "compressor_kv",
            "indexer_compressor_kv",
        )
        kv_fake: FakeTensor = kv_node.meta["val"]
        head_dim = int(kv_fake.shape[-1])
        compressor_kv_fake: FakeTensor = compressor_kv_node.meta["val"]
        compressor_state_dim = int(compressor_kv_fake.shape[-1])
        if compressor_state_dim <= 0:
            compressor_state_dim = head_dim
        indexer_compressor_kv_fake: FakeTensor = indexer_compressor_kv_node.meta["val"]
        indexer_compressor_state_dim = int(indexer_compressor_kv_fake.shape[-1])
        dtype = cls.resolve_cache_dtype(cache_config.dtype, kv_fake.dtype)
        return {
            "swa_cache": PagedResourceHandler(
                head_dim,
                dtype=dtype,
            ),
            "mhc_cache": PagedResourceHandler(head_dim, dtype=dtype),
            "compressor_kv_cache": PagedResourceHandler(
                compressor_state_dim,
                dtype=torch.float32,
            ),
            "compressor_gate_cache": PagedResourceHandler(
                compressor_state_dim,
                dtype=torch.float32,
            ),
            "indexer_compressor_kv_cache": PagedResourceHandler(
                indexer_compressor_state_dim,
                dtype=torch.float32,
            ),
            "indexer_compressor_gate_cache": PagedResourceHandler(
                indexer_compressor_state_dim,
                dtype=torch.float32,
            ),
        }

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> list[Constant]:
        (
            softmax_scale,
            window_size,
            compress_ratio,
            max_compressed_len,
            rms_norm_eps,
            rope_dim,
            topk_is_placeholder,
        ) = extract_op_args(
            source_attn_node,
            "softmax_scale",
            "window_size",
            "compress_ratio",
            "max_compressed_len",
            "rms_norm_eps",
            "rope_dim",
            "topk_is_placeholder",
        )
        if not isinstance(softmax_scale, float):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"float softmax_scale, got {softmax_scale!r}."
            )
        if window_size is not None and not isinstance(window_size, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int window_size or None, got {window_size!r}."
            )
        if not isinstance(compress_ratio, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int compress_ratio, got {compress_ratio!r}."
            )
        try:
            _compression_mode(compress_ratio)
        except ValueError as exc:
            raise RuntimeError(
                "DeepSeek V4 sparse attention cache insertion supports "
                f"compress_ratio in {_SUPPORTED_COMPRESS_RATIOS}, got {compress_ratio}."
            ) from exc
        if max_compressed_len is not None and not isinstance(max_compressed_len, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int max_compressed_len or None, got {max_compressed_len!r}."
            )
        if not isinstance(rms_norm_eps, float):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"float rms_norm_eps, got {rms_norm_eps!r}."
            )
        if rope_dim is not None and not isinstance(rope_dim, int):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"int rope_dim or None, got {rope_dim!r}."
            )
        if not isinstance(topk_is_placeholder, bool):
            raise RuntimeError(
                "DeepSeek V4 sparse attention source node must carry a literal "
                f"bool topk_is_placeholder, got {topk_is_placeholder!r}."
            )
        return [
            softmax_scale,
            window_size,
            compress_ratio,
            max_compressed_len,
            rms_norm_eps,
            rope_dim,
            topk_is_placeholder,
        ]
