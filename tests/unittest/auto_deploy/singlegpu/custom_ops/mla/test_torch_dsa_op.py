# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Comprehensive test suite for torch DSA (DeepSeek Sparse Attention) backend operations.

Tests the torch_dsa source op and torch_cached_dsa_with_cache cached op.
DSA extends MLA with an Indexer that selects top-k KV positions per query token.

Key features:
- 8 tensor arguments: q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight,
                      index_q, index_k, index_weights
- Two caches: mla_cache [max_batch, max_seq, kv_lora_rank + rope_dim]
              index_k_cache [max_batch, max_seq, index_head_dim]
- Prefill: Expand compressed_kv, compute sparse index mask
- Generate: Weight absorption for MLA + sparse index mask from cached index_k
"""

import math

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def numpy_dsa_reference(
    q_nope: np.ndarray,
    q_pe: np.ndarray,
    compressed_kv: np.ndarray,
    kpe: np.ndarray,
    kv_b_proj_weight: np.ndarray,
    index_q: np.ndarray,
    index_k: np.ndarray,
    index_weights: np.ndarray,
    mla_cache: np.ndarray,
    index_k_cache: np.ndarray,
    seq_len: np.ndarray,
    input_pos: np.ndarray,
    cache_loc: np.ndarray,
    seq_start: np.ndarray,
    scale: float = None,
    kv_lora_rank: int = None,
    index_topk: int = 64,
    is_generate: bool = False,
):
    """Numpy reference implementation of DSA attention with KV cache.

    Mirrors numpy_mla_reference_with_expansion but adds the Indexer step:
    per-head scores are computed from index_q / index_k, weighted-summed,
    and converted to a sparse mask that is added to MLA attention scores.
    """
    if is_generate:
        batch_size = q_nope.shape[0]
        num_heads = q_nope.shape[2]
        qk_nope_head_dim = q_nope.shape[3]
        qk_rope_head_dim = q_pe.shape[3]
    else:
        batch_size = len(seq_len)
        num_heads = q_nope.shape[2]
        qk_nope_head_dim = q_nope.shape[3]
        qk_rope_head_dim = q_pe.shape[3]

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if kv_lora_rank is None:
        kv_lora_rank = compressed_kv.shape[-1]

    out_features = kv_b_proj_weight.shape[0]
    kv_head_dim = out_features // num_heads
    v_head_dim = kv_head_dim - qk_nope_head_dim
    index_head_dim = index_q.shape[-1]
    index_softmax_scale = 1.0 / math.sqrt(index_head_dim)

    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Update both caches first
    if is_generate:
        for i in range(batch_size):
            cache_idx = cache_loc[i]
            pos = input_pos[i]
            mla_cache[cache_idx, pos, :kv_lora_rank] = compressed_kv[i, 0]
            mla_cache[cache_idx, pos, kv_lora_rank:] = kpe[i, 0, 0]
            index_k_cache[cache_idx, pos] = index_k[i, 0]
    else:
        for i in range(batch_size):
            cache_idx = cache_loc[i]
            pos = input_pos[i]
            seq_len_i = seq_len[i]
            seq_start_i = seq_start[i]
            for j in range(seq_len_i):
                mla_cache[cache_idx, pos + j, :kv_lora_rank] = compressed_kv[seq_start_i + j]
                mla_cache[cache_idx, pos + j, kv_lora_rank:] = kpe[seq_start_i + j, 0]
                index_k_cache[cache_idx, pos + j] = index_k[seq_start_i + j]

    outputs = []

    for i in range(batch_size):
        cache_idx = cache_loc[i]
        pos = input_pos[i]
        seq_len_i = seq_len[i]
        seq_start_i = seq_start[i]

        if seq_len_i == 0:
            continue

        if is_generate:
            q_nope_seq = q_nope[i, 0]  # [N, qk_nope_head_dim]
            q_pe_seq = q_pe[i, 0]  # [N, qk_rope_head_dim]
            index_q_seq = index_q[i, 0]  # [H, index_head_dim]
            index_w_seq = index_weights[i, 0]  # [H]
        else:
            q_nope_seq = q_nope[seq_start_i : seq_start_i + seq_len_i]  # [S, N, nope]
            q_pe_seq = q_pe[seq_start_i : seq_start_i + seq_len_i]  # [S, N, rope]
            index_q_seq = index_q[seq_start_i : seq_start_i + seq_len_i]  # [S, H, D]
            index_w_seq = index_weights[seq_start_i : seq_start_i + seq_len_i]  # [S, H]

        kv_seq_len = pos + seq_len_i

        # Get cached MLA data
        cached_data = mla_cache[cache_idx, :kv_seq_len]
        compressed_kv_cached = cached_data[:, :kv_lora_rank]  # [T, kv_lora_rank]
        kpe_cached = cached_data[:, kv_lora_rank:]  # [T, rope_dim]

        # Get cached index_k
        index_k_cached = index_k_cache[cache_idx, :kv_seq_len]  # [T, D]

        # Expand compressed_kv
        kv_expanded = np.matmul(compressed_kv_cached, kv_b_proj_weight.T)
        kv_expanded = kv_expanded.reshape(kv_seq_len, num_heads, kv_head_dim)
        k_nope = kv_expanded[:, :, :qk_nope_head_dim]  # [T, N, nope]
        v = kv_expanded[:, :, qk_nope_head_dim:]  # [T, N, v]

        kpe_expanded = np.broadcast_to(
            kpe_cached[:, None, :], (kv_seq_len, num_heads, qk_rope_head_dim)
        )  # [T, N, rope]

        # ======================================================================
        # Indexer: compute sparse mask
        # ======================================================================
        # per_head_scores: scores for each (query_token, index_head, kv_token)
        if is_generate:
            # index_q_seq: [H, D], index_k_cached: [T, D]
            per_head_scores = np.einsum("hd,td->ht", index_q_seq, index_k_cached)  # [H, T]
            # index_score: weighted sum over index heads -> [T]
            index_score = np.einsum("ht,h->t", per_head_scores, index_w_seq) * index_softmax_scale
            # No causal mask needed for generate (all cached positions are past)
            effective_topk = min(index_topk, kv_seq_len)
            topk_indices = np.argpartition(index_score, -effective_topk)[-effective_topk:]
            index_mask = np.full(kv_seq_len, float("-inf"))
            index_mask[topk_indices] = 0.0  # [T]
        else:
            # index_q_seq: [S, H, D], index_k_cached: [T, D]
            per_head_scores = np.einsum("shd,td->sht", index_q_seq, index_k_cached)  # [S, H, T]
            index_score = (
                np.einsum("sht,sh->st", per_head_scores, index_w_seq) * index_softmax_scale
            )  # [S, T]
            # Apply causal mask: future positions (relative to current seq) get -inf
            causal_mask = np.triu(np.ones((seq_len_i, kv_seq_len)), k=kv_seq_len - seq_len_i + 1)
            index_score = np.where(causal_mask, -np.inf, index_score)
            # Build sparse mask: -inf everywhere except top-k positions per query token
            effective_topk = min(index_topk, kv_seq_len)
            index_mask = np.full_like(index_score, float("-inf"))
            for s in range(seq_len_i):
                valid = np.where(index_score[s] > float("-inf"))[0]
                if len(valid) > 0:
                    scores_valid = index_score[s, valid]
                    k = min(effective_topk, len(valid))
                    topk_local = np.argpartition(scores_valid, -k)[-k:]
                    index_mask[s, valid[topk_local]] = 0.0

        # ======================================================================
        # MLA: compute attention scores with sparse mask
        # ======================================================================
        if is_generate:
            # query_full: [N, qk_head_dim]
            query_full = np.concatenate([q_nope_seq, q_pe_seq], axis=-1)
            # key_full: [T, N, qk_head_dim]
            key_full = np.concatenate([k_nope, kpe_expanded], axis=-1)
            attn_scores = np.einsum("nh,tnh->nt", query_full, key_full) * scale  # [N, T]
            # Add index_mask (broadcast [T] -> [N, T])
            attn_scores = attn_scores + index_mask[None, :]
        else:
            # query_full: [S, N, qk_head_dim]
            query_full = np.concatenate([q_nope_seq, q_pe_seq], axis=-1)
            # key_full: [T, N, qk_head_dim]
            key_full = np.concatenate([k_nope, kpe_expanded], axis=-1)
            attn_scores = np.einsum("snh,tnh->snt", query_full, key_full) * scale  # [S, N, T]
            # Causal mask for MLA
            causal_mask_mla = np.triu(
                np.ones((seq_len_i, kv_seq_len)), k=kv_seq_len - seq_len_i + 1
            )
            attn_scores = np.where(causal_mask_mla[:, None, :], -np.inf, attn_scores)
            # Add index_mask: [S, T] -> [S, 1, T] -> broadcast over N
            attn_scores = attn_scores + index_mask[:, None, :]

        # Softmax + output
        attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
        attn_scores_exp = np.exp(attn_scores - attn_scores_max)
        attn_weights = attn_scores_exp / np.sum(attn_scores_exp, axis=-1, keepdims=True)

        if is_generate:
            attn_out = np.einsum("nt,tnh->nh", attn_weights, v)  # [N, v]
        else:
            attn_out = np.einsum("snt,tnh->snh", attn_weights, v)  # [S, N, v]

        outputs.append(attn_out)

    if len(outputs) == 0:
        return np.zeros((1, 0, num_heads, v_head_dim), dtype=np.float32)
    elif is_generate:
        result = np.stack(outputs, axis=0)
        return result[:, None, :, :]  # [B, 1, N, v]
    else:
        result = np.concatenate(outputs, axis=0)
        return result[None, :, :, :]  # [1, B*S, N, v]


class TestTorchDSASourceOp:
    """Test torch_dsa source op (without cache)."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test configuration."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.atol = 1e-2
        self.rtol = 1e-2

        torch.cuda.empty_cache()
        torch.manual_seed(42)
        np.random.seed(42)

    def _create_dsa_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
    ):
        """Create test data for DSA source op (bsnd layout)."""
        kv_head_dim = qk_nope_head_dim + v_head_dim

        q_nope = torch.randn(
            batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=self.dtype, device=self.device
        )
        q_pe = torch.randn(
            batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )
        compressed_kv = torch.randn(
            batch_size, seq_len, kv_lora_rank, dtype=self.dtype, device=self.device
        )
        kpe = torch.randn(
            batch_size, seq_len, 1, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=self.dtype, device=self.device
        )
        index_q = torch.randn(
            batch_size, seq_len, index_n_heads, index_head_dim, dtype=self.dtype, device=self.device
        )
        index_k = torch.randn(
            batch_size, seq_len, index_head_dim, dtype=self.dtype, device=self.device
        )
        index_weights = torch.randn(
            batch_size, seq_len, index_n_heads, dtype=self.dtype, device=self.device
        )

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "compressed_kv": compressed_kv,
            "kpe": kpe,
            "kv_b_proj_weight": kv_b_proj_weight,
            "index_q": index_q,
            "index_k": index_k,
            "index_weights": index_weights,
        }

    def test_basic_functionality(self):
        """Test basic DSA source op functionality: shape and finiteness."""
        B, S, N = 1, 4, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 2

        data = self._create_dsa_data(B, S, N, nope, rope, kv_lora, v, idx_H, idx_D)

        output = torch.ops.auto_deploy.torch_dsa(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["index_q"],
            data["index_k"],
            data["index_weights"],
            topk,
            True,
            None,
            "bsnd",
        )

        expected_shape = (B, S, N, v)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_sparse_masking_changes_output(self):
        """Test that sparse masking (topk < S) produces different output than MLA (full attend)."""
        B, S, N = 1, 4, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D = 2, 20

        data = self._create_dsa_data(B, S, N, nope, rope, kv_lora, v, idx_H, idx_D)

        # DSA with topk < S: mask is applied
        output_dsa = torch.ops.auto_deploy.torch_dsa(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["index_q"],
            data["index_k"],
            data["index_weights"],
            2,  # topk=2 < S=4 → mask is non-trivial
            True,
            None,
            "bsnd",
        )

        # MLA-only (no sparse mask): use torch_mla
        output_mla = torch.ops.auto_deploy.torch_mla(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            True,
            None,
            "bsnd",
        )

        # DSA output should differ from MLA output when topk < S
        assert not torch.allclose(output_dsa, output_mla, atol=1e-3), (
            "DSA with topk < S should produce different output than full MLA"
        )

    def test_topk_full_equals_dense(self):
        """Test that DSA with index_topk=S (attend all) ≈ MLA (mask all zeros).

        When every position is selected, the sparse mask is all zeros (no masking),
        so DSA should produce the same result as plain MLA.
        """
        B, S, N = 1, 4, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D = 2, 20

        # Use float32 for this equality check
        self.dtype = torch.float32
        data = self._create_dsa_data(B, S, N, nope, rope, kv_lora, v, idx_H, idx_D)

        output_dsa = torch.ops.auto_deploy.torch_dsa(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["index_q"],
            data["index_k"],
            data["index_weights"],
            S,  # topk == S → select all → mask is all zeros
            True,
            None,
            "bsnd",
        )

        output_mla = torch.ops.auto_deploy.torch_mla(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            True,
            None,
            "bsnd",
        )

        max_diff = (output_dsa - output_mla).abs().max().item()
        assert torch.allclose(output_dsa, output_mla, atol=1e-5), (
            f"DSA with topk=S should equal MLA. Max diff: {max_diff:.2e}"
        )

    def test_custom_scale(self):
        """Test that custom scale changes DSA output."""
        B, S, N = 1, 4, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 2

        data = self._create_dsa_data(B, S, N, nope, rope, kv_lora, v, idx_H, idx_D)

        output_default = torch.ops.auto_deploy.torch_dsa(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["index_q"],
            data["index_k"],
            data["index_weights"],
            topk,
            True,
            None,
            "bsnd",
        )

        output_custom = torch.ops.auto_deploy.torch_dsa(
            data["q_nope"],
            data["q_pe"],
            data["compressed_kv"],
            data["kpe"],
            data["kv_b_proj_weight"],
            data["index_q"],
            data["index_k"],
            data["index_weights"],
            topk,
            True,
            0.5,
            "bsnd",
        )

        assert not torch.allclose(output_default, output_custom, atol=1e-3), (
            "Custom scale should affect output"
        )


class TestTorchBackendDSAWithCache:
    """Test torch_cached_dsa_with_cache cached op."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Setup test configuration."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.atol = 5e-2
        self.rtol = 5e-2

        torch.cuda.empty_cache()
        torch.manual_seed(42)
        np.random.seed(42)

    def _create_cached_dsa_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        kv_lora_rank: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        max_seq_len: int,
        cache_offset: int = 0,
    ):
        """Create test data for cached DSA op, mirroring _create_cached_mla_data."""
        kv_head_dim = qk_nope_head_dim + v_head_dim

        # Create input tensors (BSND layout)
        q_nope = torch.randn(
            batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=self.dtype, device=self.device
        )
        q_pe = torch.randn(
            batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )
        compressed_kv = torch.randn(
            batch_size, seq_len, kv_lora_rank, dtype=self.dtype, device=self.device
        )
        kpe = torch.randn(
            batch_size, seq_len, 1, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )
        kv_b_proj_weight = torch.randn(
            num_heads * kv_head_dim, kv_lora_rank, dtype=self.dtype, device=self.device
        )
        index_q = torch.randn(
            batch_size, seq_len, index_n_heads, index_head_dim, dtype=self.dtype, device=self.device
        )
        index_k = torch.randn(
            batch_size, seq_len, index_head_dim, dtype=self.dtype, device=self.device
        )
        index_weights = torch.randn(
            batch_size, seq_len, index_n_heads, dtype=self.dtype, device=self.device
        )

        # MLA cache: [max_batch, max_seq, kv_lora_rank + qk_rope_head_dim]
        mla_cache = torch.zeros(
            batch_size,
            max_seq_len,
            kv_lora_rank + qk_rope_head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        # Index key cache: [max_batch, max_seq, index_head_dim]
        index_k_cache = torch.zeros(
            batch_size,
            max_seq_len,
            index_head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        if cache_offset > 0:
            mla_cache[:, :cache_offset, :] = torch.randn(
                batch_size,
                cache_offset,
                kv_lora_rank + qk_rope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            index_k_cache[:, :cache_offset, :] = torch.randn(
                batch_size,
                cache_offset,
                index_head_dim,
                dtype=self.dtype,
                device=self.device,
            )

        seq_len_tensor = torch.full((batch_size,), seq_len, device=self.device, dtype=torch.int32)
        input_pos = torch.full((batch_size,), cache_offset, device=self.device, dtype=torch.int32)
        cache_loc = torch.arange(batch_size, device=self.device, dtype=torch.int32)

        if seq_len == 1:
            # Generate phase
            batch_info_host = torch.tensor(
                [0, 0, batch_size], device=self.device, dtype=torch.int32
            )
            cu_seqlen = torch.arange(batch_size, device=self.device, dtype=torch.int32)
        else:
            # Context phase: flatten inputs
            batch_info_host = torch.tensor(
                [batch_size, batch_size * seq_len, 0], device=self.device, dtype=torch.int32
            )
            cu_seqlen = torch.arange(
                0, batch_size * seq_len, seq_len, device=self.device, dtype=torch.int32
            )
            q_nope = q_nope.view(1, batch_size * seq_len, num_heads, qk_nope_head_dim)
            q_pe = q_pe.view(1, batch_size * seq_len, num_heads, qk_rope_head_dim)
            compressed_kv = compressed_kv.view(1, batch_size * seq_len, kv_lora_rank)
            kpe = kpe.view(1, batch_size * seq_len, 1, qk_rope_head_dim)
            index_q = index_q.view(1, batch_size * seq_len, index_n_heads, index_head_dim)
            index_k = index_k.view(1, batch_size * seq_len, index_head_dim)
            index_weights = index_weights.view(1, batch_size * seq_len, index_n_heads)

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
            "input_pos": input_pos,
            "cache_loc": cache_loc,
            "cu_seqlen": cu_seqlen,
            "mla_cache": mla_cache,
            "index_k_cache": index_k_cache,
            "kv_lora_rank": kv_lora_rank,
            "index_head_dim": index_head_dim,
        }

    def _run_cached_dsa(self, data, scale=None, index_topk=64):
        """Run cached DSA operation."""
        return torch.ops.auto_deploy.torch_cached_dsa_with_cache(
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
            data["cache_loc"],
            data["cu_seqlen"],
            data["mla_cache"],
            data["index_k_cache"],
            scale,
            data["kv_lora_rank"],
            index_topk,
        )

    def test_context_phase_basic(self):
        """Test context (prefill) phase: shape and finiteness."""
        B, S, N = 2, 4, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 2
        max_seq_len = 64

        data = self._create_cached_dsa_data(
            B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len
        )
        output = self._run_cached_dsa(data, index_topk=topk)

        expected_shape = (1, B * S, N, v)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_generate_phase_basic(self):
        """Test generate phase (single token): shape and finiteness."""
        B, S, N = 2, 1, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 2
        max_seq_len = 64
        cache_offset = 5

        data = self._create_cached_dsa_data(
            B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len, cache_offset
        )
        output = self._run_cached_dsa(data, index_topk=topk)

        expected_shape = (B, S, N, v)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_source_vs_cached_equivalence_prefill(self):
        """Test that source op == cached prefill op (single batch, no prior cache).

        With cache_offset=0 and batch_size=1, both ops attend to exactly the same
        tokens, so their outputs should match numerically in float32.
        """
        B, S, N = 1, 4, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 4  # topk = S → full attend for exact equality
        max_seq_len = 64

        self.dtype = torch.float32
        data = self._create_cached_dsa_data(
            B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len, cache_offset=0
        )

        # Source op (operates on [B, S] layout directly)
        # Reconstruct original [B, S] views from the flattened context tensors
        q_nope_src = data["q_nope"].view(B, S, N, nope)
        q_pe_src = data["q_pe"].view(B, S, N, rope)
        compressed_kv_src = data["compressed_kv"].view(B, S, kv_lora)
        kpe_src = data["kpe"].view(B, S, 1, rope)
        index_q_src = data["index_q"].view(B, S, idx_H, idx_D)
        index_k_src = data["index_k"].view(B, S, idx_D)
        index_weights_src = data["index_weights"].view(B, S, idx_H)

        output_src = torch.ops.auto_deploy.torch_dsa(
            q_nope_src,
            q_pe_src,
            compressed_kv_src,
            kpe_src,
            data["kv_b_proj_weight"],
            index_q_src,
            index_k_src,
            index_weights_src,
            topk,
            True,
            None,
            "bsnd",
        )  # [B, S, N, v]

        output_cached = self._run_cached_dsa(data, index_topk=topk)  # [1, B*S, N, v]

        # Reshape for comparison
        output_src_flat = output_src.view(1, B * S, N, v)
        max_diff = (output_src_flat - output_cached).abs().max().item()
        assert torch.allclose(output_src_flat, output_cached, atol=1e-5), (
            f"Source op and cached prefill op differ. Max diff: {max_diff:.2e}"
        )

    def test_both_caches_updated(self):
        """Test that after a forward pass, both mla_cache and index_k_cache are updated."""
        B, S, N = 1, 1, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 2
        max_seq_len = 32
        cache_offset = 5

        data = self._create_cached_dsa_data(
            B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len, cache_offset
        )

        # Record original cache values at target position
        orig_mla = data["mla_cache"][0, cache_offset].clone()
        orig_idx = data["index_k_cache"][0, cache_offset].clone()

        _ = self._run_cached_dsa(data, index_topk=topk)

        updated_mla = data["mla_cache"][0, cache_offset]
        updated_idx = data["index_k_cache"][0, cache_offset]

        assert not torch.allclose(orig_mla, updated_mla, atol=1e-6), (
            "mla_cache should have been updated at the target position"
        )
        assert not torch.allclose(orig_idx, updated_idx, atol=1e-6), (
            "index_k_cache should have been updated at the target position"
        )

    def test_generate_with_numpy_reference(self):
        """Test generate phase against numpy_dsa_reference within tolerance."""
        # Use small dims + float32 to keep bfloat16 accumulated error below atol
        B, S, N = 2, 1, 4
        nope, rope, kv_lora, v = 32, 16, 64, 32
        idx_H, idx_D, topk = 2, 20, 4
        max_seq_len = 64
        cache_offset = 3

        self.dtype = torch.float32

        data = self._create_cached_dsa_data(
            B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len, cache_offset
        )

        # Disable TF32 so float32 matmuls use full precision and match numpy
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            output = self._run_cached_dsa(data, index_topk=topk)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

        reference = numpy_dsa_reference(
            data["q_nope"].cpu().float().numpy(),
            data["q_pe"].cpu().float().numpy(),
            data["compressed_kv"].cpu().float().numpy(),
            data["kpe"].cpu().float().numpy(),
            data["kv_b_proj_weight"].cpu().float().numpy(),
            data["index_q"].cpu().float().numpy(),
            data["index_k"].cpu().float().numpy(),
            data["index_weights"].cpu().float().numpy(),
            data["mla_cache"].cpu().float().numpy(),
            data["index_k_cache"].cpu().float().numpy(),
            data["seq_len"].cpu().numpy(),
            data["input_pos"].cpu().numpy(),
            data["cache_loc"].cpu().numpy(),
            data["cu_seqlen"].cpu().numpy(),
            scale=None,
            kv_lora_rank=kv_lora,
            index_topk=topk,
            is_generate=True,
        )

        reference_torch = torch.from_numpy(reference).to(output.device, output.dtype)
        max_diff = (output - reference_torch).abs().max().item()
        # float32 inputs → tight tolerance; bfloat16 would need ~5e-2
        assert torch.allclose(output, reference_torch, atol=1e-4, rtol=1e-4), (
            f"Generate phase output doesn't match numpy reference. Max diff: {max_diff:.6f}"
        )

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype for float16 and bfloat16."""
        B, S, N = 1, 1, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D, topk = 2, 20, 2
        max_seq_len = 32

        for dtype in [torch.float16, torch.bfloat16]:
            self.dtype = dtype
            data = self._create_cached_dsa_data(
                B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len
            )
            output = self._run_cached_dsa(data, index_topk=topk)
            assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"

    def test_cache_shapes(self):
        """Test that both caches have the expected shapes."""
        B, S, N = 2, 1, 4
        nope, rope, kv_lora, v = 32, 16, 128, 32
        idx_H, idx_D = 2, 20
        max_seq_len = 64

        data = self._create_cached_dsa_data(
            B, S, N, nope, rope, kv_lora, v, idx_H, idx_D, max_seq_len
        )

        expected_mla_shape = (B, max_seq_len, kv_lora + rope)
        expected_idx_shape = (B, max_seq_len, idx_D)

        assert data["mla_cache"].shape == expected_mla_shape, (
            f"mla_cache shape {data['mla_cache'].shape} != {expected_mla_shape}"
        )
        assert data["index_k_cache"].shape == expected_idx_shape, (
            f"index_k_cache shape {data['index_k_cache'].shape} != {expected_idx_shape}"
        )


class TestDSADescriptor:
    """Test TorchBackendDSAAttention descriptor configuration."""

    def _get_dsa_descriptor(self):
        """Get DSA descriptor from registry."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        return AttentionRegistry.get("torch_dsa")

    def test_descriptor_registration(self):
        """Test that DSA descriptor is registered under 'torch_dsa'."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        assert AttentionRegistry.has("torch_dsa"), "torch_dsa should be registered"

    def test_descriptor_layout(self):
        """Test that DSA descriptor returns 'bsnd' layout."""
        dsa_descriptor = self._get_dsa_descriptor()
        assert dsa_descriptor.get_attention_layout() == "bsnd", "DSA should use bsnd layout"

    def test_descriptor_num_qkv_args(self):
        """Test that DSA descriptor expects 8 tensor args."""
        dsa_descriptor = self._get_dsa_descriptor()
        assert dsa_descriptor.get_num_qkv_args() == 8, (
            "DSA should expect 8 tensor args "
            "(q_nope, q_pe, compressed_kv, kpe, kv_b_proj_weight, index_q, index_k, index_weights)"
        )

    def test_descriptor_source_op(self):
        """Test that DSA descriptor points to torch_dsa source op."""
        dsa_descriptor = self._get_dsa_descriptor()
        source_op = dsa_descriptor.get_source_attention_op()
        assert source_op == torch.ops.auto_deploy.torch_dsa, "DSA should use torch_dsa as source op"

    def test_descriptor_cached_op(self):
        """Test that DSA descriptor points to torch_cached_dsa_with_cache cached op."""
        dsa_descriptor = self._get_dsa_descriptor()
        cached_op = dsa_descriptor.get_cached_attention_op()
        assert cached_op == torch.ops.auto_deploy.torch_cached_dsa_with_cache.default, (
            "DSA should use torch_cached_dsa_with_cache as cached op"
        )

    def test_descriptor_standard_metadata(self):
        """Test that DSA descriptor returns the standard metadata arg names."""
        dsa_descriptor = self._get_dsa_descriptor()
        expected_args = ["batch_info_host", "seq_len", "input_pos", "slot_idx", "cu_seqlen"]
        actual_args = dsa_descriptor.get_standard_metadata_args()
        assert actual_args == expected_args, (
            f"Expected standard metadata {expected_args}, got {actual_args}"
        )
