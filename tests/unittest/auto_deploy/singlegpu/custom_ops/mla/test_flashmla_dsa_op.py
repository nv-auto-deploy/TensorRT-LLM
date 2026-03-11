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

"""Tests for FlashMLA DSA backend (flash_mla_dsa_with_cache) and FlashMLADSAAttention descriptor.

FlashMLA requires SM90+ (Hopper/Blackwell). All tests are skipped on older hardware.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FlashMLA requires SM90+ (Hopper/Blackwell)",
)

import tensorrt_llm._torch.auto_deploy  # noqa: F401

# ---------------------------------------------------------------------------
# Helper: build paged metadata from simple contiguous-page layout
# ---------------------------------------------------------------------------


def _make_paged_data(
    batch_size: int,
    seq_len_per_seq: int,  # tokens to write in this batch (1 for decode)
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    page_size: int = 16,
    cache_offset: int = 0,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Build tensors for flash_mla_dsa_with_cache.

    Uses a simple paged layout where each sequence gets its own contiguous set of pages.
    """
    B = batch_size
    S = seq_len_per_seq
    total_tokens = B * S
    kv_head_dim = qk_nope_head_dim + v_head_dim

    # --- Input tensors (BSND layout for decode, or flat for prefill) ---
    if S == 1:
        # Decode: [B, 1, ...]
        q_nope = torch.randn(B, 1, num_heads, qk_nope_head_dim, dtype=dtype, device=device)
        q_pe = torch.randn(B, 1, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        compressed_kv = torch.randn(B, 1, kv_lora_rank, dtype=dtype, device=device)
        kpe = torch.randn(B, 1, 1, qk_rope_head_dim, dtype=dtype, device=device)
        index_q = torch.randn(B, 1, index_n_heads, index_head_dim, dtype=dtype, device=device)
        index_k = torch.randn(B, 1, index_head_dim, dtype=dtype, device=device)
        index_weights = torch.randn(B, 1, index_n_heads, dtype=dtype, device=device)
    else:
        # Prefill: [1, B*S, ...] (flat batch)
        q_nope = torch.randn(
            1, total_tokens, num_heads, qk_nope_head_dim, dtype=dtype, device=device
        )
        q_pe = torch.randn(1, total_tokens, num_heads, qk_rope_head_dim, dtype=dtype, device=device)
        compressed_kv = torch.randn(1, total_tokens, kv_lora_rank, dtype=dtype, device=device)
        kpe = torch.randn(1, total_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device)
        index_q = torch.randn(
            1, total_tokens, index_n_heads, index_head_dim, dtype=dtype, device=device
        )
        index_k = torch.randn(1, total_tokens, index_head_dim, dtype=dtype, device=device)
        index_weights = torch.randn(1, total_tokens, index_n_heads, dtype=dtype, device=device)

    kv_b_proj_weight = torch.randn(
        num_heads * kv_head_dim, kv_lora_rank, dtype=dtype, device=device
    )

    # --- Paged cache metadata ---
    # Each sequence needs ceil((cache_offset + S) / page_size) pages
    max_pos = cache_offset + S
    pages_per_seq = (max_pos + page_size - 1) // page_size
    total_pages = B * pages_per_seq

    # Simple contiguous page layout: seq i gets pages [i*pages_per_seq, (i+1)*pages_per_seq)
    # cache_loc: maps from (seq, page_k) index to actual block index
    # We use a simple identity mapping: page i in seq j is stored at block j*pages_per_seq + i
    page_table = torch.arange(total_pages, device=device, dtype=torch.int32)
    cache_loc = page_table  # [total_pages]

    # cu_num_pages[i] = i * pages_per_seq
    cu_num_pages = torch.arange(B + 1, device=device, dtype=torch.int32) * pages_per_seq

    # Paged caches (zero-initialized, pre-filled at cache_offset if requested)
    mla_cache = torch.zeros(
        total_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )
    index_k_cache = torch.zeros(total_pages, page_size, index_head_dim, dtype=dtype, device=device)

    # Fill cache at cache_offset positions with random data
    if cache_offset > 0:
        for b in range(B):
            for pos in range(cache_offset):
                page_k = pos // page_size
                page_off = pos % page_size
                blk = int(cache_loc[cu_num_pages[b].item() + page_k].item())
                mla_cache[blk, page_off, 0, :] = torch.randn(
                    kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
                )
                index_k_cache[blk, page_off, :] = torch.randn(
                    index_head_dim, dtype=dtype, device=device
                )

    # last_page_len: valid tokens in last page after writing
    last_page_len = torch.full(
        (B,), (max_pos - 1) % page_size + 1, device=device, dtype=torch.int32
    )

    # seq_len and input_pos per sequence
    seq_len_tensor = torch.full((B,), S, device=device, dtype=torch.int32)
    input_pos_tensor = torch.full((B,), cache_offset, device=device, dtype=torch.int32)

    if S == 1:
        # Decode
        batch_info_host = torch.tensor([0, 0, B], device=device, dtype=torch.int32)
        cu_seqlen = torch.arange(B + 1, device=device, dtype=torch.int32)
    else:
        # Prefill
        batch_info_host = torch.tensor([B, B * S, 0], device=device, dtype=torch.int32)
        cu_seqlen = torch.arange(0, B * S + 1, S, device=device, dtype=torch.int32)

    return {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "compressed_kv": compressed_kv,
        "kpe": kpe,
        "kv_b_proj_weight": kv_b_proj_weight,
        "index_q": index_q,
        "index_k": index_k,
        "index_weights": index_weights,
        "batch_info_host": batch_info_host,
        "seq_len": seq_len_tensor,
        "input_pos": input_pos_tensor,
        "cu_seqlen": cu_seqlen,
        "cache_loc": cache_loc,
        "cu_num_pages": cu_num_pages,
        "last_page_len": last_page_len,
        "mla_cache": mla_cache,
        "index_k_cache": index_k_cache,
        "kv_lora_rank": kv_lora_rank,
        "v_head_dim": v_head_dim,
        "num_heads": num_heads,
    }


def _run_flash_mla_dsa(data, scale=None, index_topk=64):
    """Call flash_mla_dsa_with_cache with the given data dict."""
    return torch.ops.auto_deploy.flash_mla_dsa_with_cache(
        data["q_nope"],
        data["q_pe"],
        data["compressed_kv"],
        data["kpe"],
        data["kv_b_proj_weight"],
        data["index_q"],
        data["index_k"],
        data["index_weights"],
        data["batch_info_host"],
        data["seq_len"],
        data["input_pos"],
        data["cu_seqlen"],
        data["cache_loc"],
        data["cu_num_pages"],
        data["last_page_len"],
        data["mla_cache"],
        data["index_k_cache"],
        scale,
        data["kv_lora_rank"],
        index_topk,
    )


def _run_torch_dsa(data):
    """Call the torch reference cached DSA (unpaged) with equivalent inputs.

    Builds an unpaged cache from the paged mla_cache/index_k_cache contents,
    then calls torch_cached_dsa_with_cache.
    """
    B = data["seq_len"].shape[0]
    kv_lora_rank = data["kv_lora_rank"]
    index_head_dim = data["index_k_cache"].shape[-1]
    qk_rope_head_dim = data["q_pe"].shape[-1]
    num_heads = data["num_heads"]
    v_head_dim = data["v_head_dim"]
    device = data["q_nope"].device
    dtype = data["q_nope"].dtype

    seq_len = data["seq_len"][:B]
    input_pos = data["input_pos"][:B]
    cache_loc_torch = data["cache_loc"]
    cu_num_pages = data["cu_num_pages"]

    # Compute cache_seqlens (after write, same as flash_mla_dsa does)
    cache_seqlens = (input_pos + seq_len).to(torch.int32)

    # Build unpaged mla_cache and index_k_cache for the torch reference op
    max_seq_len = int(cache_seqlens.max().item())
    mla_cache_unpaged = torch.zeros(
        B, max_seq_len, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
    )
    index_k_cache_unpaged = torch.zeros(B, max_seq_len, index_head_dim, dtype=dtype, device=device)

    page_size = data["mla_cache"].shape[1]
    for b in range(B):
        for t in range(int(cache_seqlens[b].item())):
            page_k = t // page_size
            page_off = t % page_size
            blk = int(cache_loc_torch[int(cu_num_pages[b].item()) + page_k].item())
            mla_cache_unpaged[b, t] = data["mla_cache"][blk, page_off, 0, :]
            index_k_cache_unpaged[b, t] = data["index_k_cache"][blk, page_off, :]

    # slot_idx: contiguous for this simple layout
    slot_idx = torch.arange(B, device=device, dtype=torch.int32)
    S = data["q_nope"].shape[1]
    cu_seqlen = data["cu_seqlen"]

    batch_info_host = data["batch_info_host"]

    return torch.ops.auto_deploy.torch_cached_dsa_with_cache(
        data["q_nope"],
        data["q_pe"],
        data["compressed_kv"],
        data["kpe"],
        data["kv_b_proj_weight"],
        data["index_q"],
        data["index_k"],
        data["index_weights"],
        batch_info_host,
        seq_len,
        input_pos,
        slot_idx,
        cu_seqlen,
        mla_cache_unpaged,
        index_k_cache_unpaged,
        None,
        kv_lora_rank,
        # Use large topk to match full-attend behavior for numerical comparison
        int(cache_seqlens.max().item()),
    )


# ---------------------------------------------------------------------------
# Class 1: TestFlashMLADSAWithCache
# ---------------------------------------------------------------------------


class TestFlashMLADSAWithCache:
    """Tests for flash_mla_dsa_with_cache cached op."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        self.dtype = torch.bfloat16
        self.device = "cuda"
        self.atol = 5e-2

    # Standard DSA dims matching DeepSeek/GLM usage
    B, S_prefill = 2, 4
    N, nope, rope = 4, 32, 64
    kv_lora_rank = 512
    v_head_dim = 128
    idx_H, idx_D, topk = 1, 16, 4
    page_size = 64  # FlashMLA requires page_block_size == 64

    def test_decode_shape_and_finite(self):
        """Output shape [B, 1, N, v] and all-finite."""
        data = _make_paged_data(
            self.B,
            1,
            self.N,
            self.nope,
            self.rope,
            self.kv_lora_rank,
            self.v_head_dim,
            self.idx_H,
            self.idx_D,
            page_size=self.page_size,
            cache_offset=5,
            dtype=self.dtype,
            device=self.device,
        )
        out = _run_flash_mla_dsa(data, index_topk=self.topk)

        assert out.shape == (self.B, 1, self.N, self.v_head_dim), (
            f"Expected ({self.B}, 1, {self.N}, {self.v_head_dim}), got {out.shape}"
        )
        assert torch.isfinite(out).all(), "Decode output contains NaN or Inf"

    def test_prefill_shape_and_finite(self):
        """Output shape [1, B*S, N, v] and all-finite."""
        data = _make_paged_data(
            self.B,
            self.S_prefill,
            self.N,
            self.nope,
            self.rope,
            self.kv_lora_rank,
            self.v_head_dim,
            self.idx_H,
            self.idx_D,
            page_size=self.page_size,
            cache_offset=0,
            dtype=self.dtype,
            device=self.device,
        )
        out = _run_flash_mla_dsa(data, index_topk=self.topk)

        expected_shape = (1, self.B * self.S_prefill, self.N, self.v_head_dim)
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
        assert torch.isfinite(out).all(), "Prefill output contains NaN or Inf"

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="Sparse FlashMLA decode requires SM100+ (Blackwell)",
    )
    def test_decode_vs_torch_equivalence(self):
        """FlashMLA sparse decode ≈ TorchBackendDSA (atol=5e-2) using large topk."""
        data = _make_paged_data(
            self.B,
            1,
            self.N,
            self.nope,
            self.rope,
            self.kv_lora_rank,
            self.v_head_dim,
            self.idx_H,
            self.idx_D,
            page_size=self.page_size,
            cache_offset=5,
            dtype=self.dtype,
            device=self.device,
        )
        out_flash = _run_flash_mla_dsa(data, index_topk=self.topk)
        out_torch = _run_torch_dsa(data)

        # Shapes may differ slightly; align to compare
        out_flash_cmp = out_flash.reshape(-1, self.N, self.v_head_dim).float()
        out_torch_cmp = out_torch.reshape(-1, self.N, self.v_head_dim).float()

        max_diff = (out_flash_cmp - out_torch_cmp).abs().max().item()
        assert torch.allclose(out_flash_cmp, out_torch_cmp, atol=self.atol), (
            f"Decode: FlashMLA vs TorchDSA max diff = {max_diff:.4f} > atol={self.atol}"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="Sparse FlashMLA prefill requires SM100+ (Blackwell)",
    )
    def test_prefill_vs_torch_equivalence(self):
        """FlashMLA sparse prefill ≈ TorchBackendDSA (atol=5e-2) using large topk."""
        data = _make_paged_data(
            self.B,
            self.S_prefill,
            self.N,
            self.nope,
            self.rope,
            self.kv_lora_rank,
            self.v_head_dim,
            self.idx_H,
            self.idx_D,
            page_size=self.page_size,
            cache_offset=0,
            dtype=self.dtype,
            device=self.device,
        )
        out_flash = _run_flash_mla_dsa(data, index_topk=self.topk)
        out_torch = _run_torch_dsa(data)

        out_flash_cmp = out_flash.reshape(-1, self.N, self.v_head_dim).float()
        out_torch_cmp = out_torch.reshape(-1, self.N, self.v_head_dim).float()

        max_diff = (out_flash_cmp - out_torch_cmp).abs().max().item()
        assert torch.allclose(out_flash_cmp, out_torch_cmp, atol=self.atol), (
            f"Prefill: FlashMLA vs TorchDSA max diff = {max_diff:.4f} > atol={self.atol}"
        )

    def test_both_caches_updated(self):
        """mla_cache and index_k_cache must be non-zero after forward."""
        data = _make_paged_data(
            self.B,
            1,
            self.N,
            self.nope,
            self.rope,
            self.kv_lora_rank,
            self.v_head_dim,
            self.idx_H,
            self.idx_D,
            page_size=self.page_size,
            cache_offset=0,
            dtype=self.dtype,
            device=self.device,
        )
        mla_before = data["mla_cache"].clone()
        idx_before = data["index_k_cache"].clone()

        _run_flash_mla_dsa(data, index_topk=self.topk)

        assert not torch.allclose(data["mla_cache"], mla_before, atol=1e-6), (
            "mla_cache was not updated"
        )
        assert not torch.allclose(data["index_k_cache"], idx_before, atol=1e-6), (
            "index_k_cache was not updated"
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtype_preservation(self, dtype):
        """Output dtype should match input dtype."""
        data = _make_paged_data(
            self.B,
            1,
            self.N,
            self.nope,
            self.rope,
            self.kv_lora_rank,
            self.v_head_dim,
            self.idx_H,
            self.idx_D,
            page_size=self.page_size,
            cache_offset=3,
            dtype=dtype,
            device=self.device,
        )
        out = _run_flash_mla_dsa(data, index_topk=self.topk)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# ---------------------------------------------------------------------------
# Class 2: TestFlashMLADSADescriptor
# ---------------------------------------------------------------------------


class TestFlashMLADSADescriptor:
    """Tests for FlashMLADSAAttention descriptor configuration."""

    def _get_descriptor(self):
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        return AttentionRegistry.get("flashmla_dsa")

    def test_descriptor_registration(self):
        """FlashMLADSAAttention should be registered under 'flashmla_dsa'."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        assert AttentionRegistry.has("flashmla_dsa"), (
            "'flashmla_dsa' not found in AttentionRegistry"
        )

    def test_descriptor_layout(self):
        """Descriptor should return 'bsnd' layout."""
        desc = self._get_descriptor()
        assert desc.get_attention_layout() == "bsnd"

    def test_descriptor_num_qkv_args(self):
        """Descriptor should expect 8 tensor args."""
        desc = self._get_descriptor()
        assert desc.get_num_qkv_args() == 8, f"Expected 8, got {desc.get_num_qkv_args()}"

    def test_descriptor_source_op(self):
        """Source op should be torch_dsa."""
        desc = self._get_descriptor()
        assert desc.get_source_attention_op() == torch.ops.auto_deploy.torch_dsa

    def test_descriptor_cached_op(self):
        """Cached op should be flash_mla_dsa_with_cache.default."""
        desc = self._get_descriptor()
        assert (
            desc.get_cached_attention_op() == torch.ops.auto_deploy.flash_mla_dsa_with_cache.default
        )

    def test_descriptor_standard_metadata(self):
        """Standard metadata should include paged-cache args."""
        desc = self._get_descriptor()
        meta = desc.get_standard_metadata_args()
        for required in ["cache_loc", "cu_num_pages", "last_page_len"]:
            assert required in meta, f"'{required}' missing from standard_metadata_args: {meta}"
