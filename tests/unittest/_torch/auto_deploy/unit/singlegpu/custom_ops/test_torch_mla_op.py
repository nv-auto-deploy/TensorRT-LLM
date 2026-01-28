"""Comprehensive test suite for torch MLA backend operations.

Tests the new torch_mla source op and torch_backend_mla_with_cache cached op
with unified FlashInfer cache layout.
"""

import math

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401


def numpy_mla_reference(
    q_nope: np.ndarray,
    q_pe: np.ndarray,
    ckv: np.ndarray,
    kpe: np.ndarray,
    mla_cache: np.ndarray,
    seq_len: np.ndarray,
    input_pos: np.ndarray,
    cache_loc: np.ndarray,
    seq_start: np.ndarray,
    scale: float = None,
    head_dim_ckv: int = None,
    is_generate: bool = False,
):
    """Numpy reference implementation of MLA attention with FlashInfer cache layout."""
    # Get dimensions
    if is_generate:
        batch_size = q_nope.shape[0]
        num_heads = q_nope.shape[2]
        qk_nope_head_dim = q_nope.shape[3]
        qk_rope_head_dim = q_pe.shape[3]
    else:
        # Context phase: flattened [1, total_tokens, ...]
        batch_size = len(seq_len)
        num_heads = q_nope.shape[2]
        qk_nope_head_dim = q_nope.shape[3]
        qk_rope_head_dim = q_pe.shape[3]

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    # head_dim_kpe = kpe.shape[-1]

    if head_dim_ckv is None:
        head_dim_ckv = ckv.shape[-1]

    v_head_dim = head_dim_ckv - qk_nope_head_dim

    # Set default scale
    if scale is None:
        scale = 1.0 / math.sqrt(qk_head_dim)

    # Update MLA cache first
    if is_generate:
        for i in range(batch_size):
            cache_idx = cache_loc[i]
            pos = input_pos[i]
            # Update ckv portion
            mla_cache[cache_idx, pos, :head_dim_ckv] = ckv[i, 0, 0]
            # Update kpe portion
            mla_cache[cache_idx, pos, head_dim_ckv:] = kpe[i, 0, 0]
    else:
        for i in range(batch_size):
            cache_idx = cache_loc[i]
            pos = input_pos[i]
            seq_len_i = seq_len[i]
            seq_start_i = seq_start[i]

            for j in range(seq_len_i):
                mla_cache[cache_idx, pos + j, :head_dim_ckv] = ckv[seq_start_i + j, 0]
                mla_cache[cache_idx, pos + j, head_dim_ckv:] = kpe[seq_start_i + j, 0]

    # Compute attention for each sequence
    outputs = []

    for i in range(batch_size):
        cache_idx = cache_loc[i]
        pos = input_pos[i]
        seq_len_i = seq_len[i]
        seq_start_i = seq_start[i]

        if seq_len_i == 0:
            continue

        # Get query for this sequence
        if is_generate:
            q_nope_seq = q_nope[i, 0]  # [N, qk_nope_head_dim]
            q_pe_seq = q_pe[i, 0]  # [N, qk_rope_head_dim]
        else:
            q_nope_seq = q_nope[
                seq_start_i : seq_start_i + seq_len_i
            ]  # [seq_len_i, N, qk_nope_head_dim]
            q_pe_seq = q_pe[
                seq_start_i : seq_start_i + seq_len_i
            ]  # [seq_len_i, N, qk_rope_head_dim]

        # Get cached ckv and kpe
        kv_seq_len = pos + seq_len_i
        cached_data = mla_cache[cache_idx, :kv_seq_len]  # [kv_seq_len, head_dim_ckv + head_dim_kpe]
        ckv_cached = cached_data[:, :head_dim_ckv]  # [kv_seq_len, head_dim_ckv]
        kpe_cached = cached_data[:, head_dim_ckv:]  # [kv_seq_len, head_dim_kpe]

        # Split ckv into k_nope and value
        k_nope_cached = ckv_cached[:, :qk_nope_head_dim]  # [kv_seq_len, qk_nope_head_dim]
        v_cached = ckv_cached[:, qk_nope_head_dim:]  # [kv_seq_len, v_head_dim]

        # Construct full query
        if is_generate:
            query_full = np.concatenate([q_nope_seq, q_pe_seq], axis=-1)  # [N, qk_head_dim]
        else:
            query_full = np.concatenate(
                [q_nope_seq, q_pe_seq], axis=-1
            )  # [seq_len_i, N, qk_head_dim]

        # Construct full key (expand to num_heads)
        k_nope_expanded = np.broadcast_to(
            k_nope_cached[:, None, :], (kv_seq_len, num_heads, qk_nope_head_dim)
        )
        kpe_expanded = np.broadcast_to(
            kpe_cached[:, None, :], (kv_seq_len, num_heads, qk_rope_head_dim)
        )
        key_full = np.concatenate(
            [k_nope_expanded, kpe_expanded], axis=-1
        )  # [kv_seq_len, N, qk_head_dim]

        # Compute attention scores
        if is_generate:
            # query_full: [N, qk_head_dim], key_full: [kv_seq_len, N, qk_head_dim]
            attn_scores = np.einsum("nh,knh->nk", query_full, key_full) * scale  # [N, kv_seq_len]
        else:
            # query_full: [seq_len_i, N, qk_head_dim], key_full: [kv_seq_len, N, qk_head_dim]
            attn_scores = (
                np.einsum("snh,knh->snk", query_full, key_full) * scale
            )  # [seq_len_i, N, kv_seq_len]

            # Apply causal mask
            causal_mask = np.triu(np.ones((seq_len_i, kv_seq_len)), k=kv_seq_len - seq_len_i + 1)
            attn_scores = np.where(causal_mask[:, None, :], -np.inf, attn_scores)

        # Apply softmax
        attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
        attn_scores_exp = np.exp(attn_scores - attn_scores_max)
        attn_weights = attn_scores_exp / np.sum(attn_scores_exp, axis=-1, keepdims=True)

        # Compute output
        v_expanded = np.broadcast_to(
            v_cached[:, None, :], (kv_seq_len, num_heads, v_head_dim)
        )  # [kv_seq_len, N, v_head_dim]

        if is_generate:
            # attn_weights: [N, kv_seq_len], v_expanded: [kv_seq_len, N, v_head_dim]
            attn_out = np.einsum("nk,knh->nh", attn_weights, v_expanded)  # [N, v_head_dim]
        else:
            # attn_weights: [seq_len_i, N, kv_seq_len], v_expanded: [kv_seq_len, N, v_head_dim]
            attn_out = np.einsum(
                "snk,knh->snh", attn_weights, v_expanded
            )  # [seq_len_i, N, v_head_dim]

        outputs.append(attn_out)

    # Concatenate outputs
    if len(outputs) == 0:
        return np.zeros((1, 0, num_heads, v_head_dim), dtype=np.float32)
    elif is_generate:
        # Generate phase: outputs is a list of [N, v_head_dim] tensors
        result = np.stack(outputs, axis=0)  # [batch_size, N, v_head_dim]
        return result[:, None, :, :]  # [batch_size, 1, N, v_head_dim]
    else:
        # Context phase: outputs is a list of [seq_len_i, N, v_head_dim] tensors
        result = np.concatenate(outputs, axis=0)  # [total_seq, N, v_head_dim]
        return result[None, :, :, :]  # [1, total_seq, N, v_head_dim]


class TestTorchMLASourceOp:
    """Test torch_mla source op (without cache)."""

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

    def _create_mla_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        head_dim_ckv: int,
        layout: str = "bsnd",
    ):
        """Create test data for MLA source op."""
        head_dim_kpe = qk_rope_head_dim

        if layout == "bsnd":
            q_nope = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                qk_nope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            q_pe = torch.randn(
                batch_size,
                seq_len,
                num_heads,
                qk_rope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            ckv = torch.randn(
                batch_size, seq_len, 1, head_dim_ckv, dtype=self.dtype, device=self.device
            )
            kpe = torch.randn(
                batch_size, seq_len, 1, head_dim_kpe, dtype=self.dtype, device=self.device
            )
        else:  # bnsd
            q_nope = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                qk_nope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            q_pe = torch.randn(
                batch_size,
                num_heads,
                seq_len,
                qk_rope_head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            ckv = torch.randn(
                batch_size, 1, seq_len, head_dim_ckv, dtype=self.dtype, device=self.device
            )
            kpe = torch.randn(
                batch_size, 1, seq_len, head_dim_kpe, dtype=self.dtype, device=self.device
            )

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "ckv": ckv,
            "kpe": kpe,
        }

    def test_basic_functionality(self):
        """Test basic MLA source op functionality."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 128, 64
        head_dim_ckv = qk_nope_head_dim + 128  # v_head_dim = 128

        data = self._create_mla_data(
            batch_size, seq_len, num_heads, qk_nope_head_dim, qk_rope_head_dim, head_dim_ckv
        )

        output = torch.ops.auto_deploy.torch_mla(
            data["q_nope"],
            data["q_pe"],
            data["ckv"],
            data["kpe"],
            True,  # is_causal
            None,  # scale
            "bsnd",  # layout
        )

        # Verify output shape: [B, S, N, v_head_dim]
        v_head_dim = head_dim_ckv - qk_nope_head_dim
        expected_shape = (batch_size, seq_len, num_heads, v_head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_both_layouts(self):
        """Test MLA source op with both bsnd and bnsd layouts."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        head_dim_ckv = qk_nope_head_dim + 64  # v_head_dim = 64

        for layout in ["bsnd", "bnsd"]:
            data = self._create_mla_data(
                batch_size,
                seq_len,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                head_dim_ckv,
                layout,
            )

            output = torch.ops.auto_deploy.torch_mla(
                data["q_nope"],
                data["q_pe"],
                data["ckv"],
                data["kpe"],
                True,
                None,
                layout,
            )

            v_head_dim = head_dim_ckv - qk_nope_head_dim
            if layout == "bsnd":
                expected_shape = (batch_size, seq_len, num_heads, v_head_dim)
            else:
                expected_shape = (batch_size, num_heads, seq_len, v_head_dim)

            assert output.shape == expected_shape, (
                f"Layout {layout}: Expected {expected_shape}, got {output.shape}"
            )

    def test_custom_scale(self):
        """Test MLA source op with custom scale."""
        batch_size, seq_len, num_heads = 1, 2, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        head_dim_ckv = qk_nope_head_dim + 32

        data = self._create_mla_data(
            batch_size, seq_len, num_heads, qk_nope_head_dim, qk_rope_head_dim, head_dim_ckv
        )

        # Test with default scale
        output_default = torch.ops.auto_deploy.torch_mla(
            data["q_nope"], data["q_pe"], data["ckv"], data["kpe"], True, None, "bsnd"
        )

        # Test with custom scale
        custom_scale = 0.5
        output_custom = torch.ops.auto_deploy.torch_mla(
            data["q_nope"], data["q_pe"], data["ckv"], data["kpe"], True, custom_scale, "bsnd"
        )

        # Outputs should be different
        assert not torch.allclose(output_default, output_custom, atol=1e-3), (
            "Custom scale should affect output"
        )


class TestTorchBackendMLAWithCache:
    """Test torch_backend_mla_with_cache cached op."""

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

    def _create_cached_mla_data(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        head_dim_ckv: int,
        max_seq_len: int,
        cache_offset: int = 0,
    ):
        """Create test data for cached MLA op with FlashInfer layout."""
        head_dim_kpe = qk_rope_head_dim

        # Create input tensors (BSND layout)
        q_nope = torch.randn(
            batch_size, seq_len, num_heads, qk_nope_head_dim, dtype=self.dtype, device=self.device
        )
        q_pe = torch.randn(
            batch_size, seq_len, num_heads, qk_rope_head_dim, dtype=self.dtype, device=self.device
        )
        ckv = torch.randn(
            batch_size, seq_len, 1, head_dim_ckv, dtype=self.dtype, device=self.device
        )
        kpe = torch.randn(
            batch_size, seq_len, 1, head_dim_kpe, dtype=self.dtype, device=self.device
        )

        # Create unified MLA cache (FlashInfer layout)
        # Shape: [max_batch, max_seq, head_dim_ckv + head_dim_kpe]
        mla_cache = torch.zeros(
            batch_size,
            max_seq_len,
            head_dim_ckv + head_dim_kpe,
            dtype=self.dtype,
            device=self.device,
        )

        # Pre-fill cache with random data if cache_offset > 0
        if cache_offset > 0:
            mla_cache[:, :cache_offset, :] = torch.randn(
                batch_size,
                cache_offset,
                head_dim_ckv + head_dim_kpe,
                dtype=self.dtype,
                device=self.device,
            )

        # Setup metadata
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
            # Context phase
            batch_info_host = torch.tensor(
                [batch_size, batch_size * seq_len, 0], device=self.device, dtype=torch.int32
            )
            cu_seqlen = torch.arange(
                0, batch_size * seq_len, seq_len, device=self.device, dtype=torch.int32
            )
            # Flatten inputs for context phase
            q_nope = q_nope.view(1, batch_size * seq_len, num_heads, qk_nope_head_dim)
            q_pe = q_pe.view(1, batch_size * seq_len, num_heads, qk_rope_head_dim)
            ckv = ckv.view(1, batch_size * seq_len, 1, head_dim_ckv)
            kpe = kpe.view(1, batch_size * seq_len, 1, head_dim_kpe)

        return {
            "q_nope": q_nope,
            "q_pe": q_pe,
            "ckv": ckv,
            "kpe": kpe,
            "batch_info_host": batch_info_host,
            "seq_len": seq_len_tensor,
            "input_pos": input_pos,
            "cache_loc": cache_loc,
            "cu_seqlen": cu_seqlen,
            "mla_cache": mla_cache,
            "head_dim_ckv": head_dim_ckv,
        }

    def _run_cached_mla(self, data, scale=None):
        """Run cached MLA operation."""
        return torch.ops.auto_deploy.torch_cached_mla_with_cache(
            data["q_nope"],
            data["q_pe"],
            data["ckv"],
            data["kpe"],
            data["batch_info_host"],
            data["seq_len"],
            data["input_pos"],
            data["cache_loc"],
            data["cu_seqlen"],
            data["mla_cache"],
            scale,
            data["head_dim_ckv"],
        )

    def test_generate_phase_basic(self):
        """Test generate phase (single token) basic functionality."""
        batch_size, seq_len, num_heads = 2, 1, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        head_dim_ckv = qk_nope_head_dim + 64
        max_seq_len = 128
        cache_offset = 5

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            head_dim_ckv,
            max_seq_len,
            cache_offset,
        )

        output = self._run_cached_mla(data)

        # Verify output shape
        v_head_dim = head_dim_ckv - qk_nope_head_dim
        expected_shape = (batch_size, seq_len, num_heads, v_head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_context_phase_basic(self):
        """Test context phase (multi-token) basic functionality."""
        batch_size, seq_len, num_heads = 2, 4, 8
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        head_dim_ckv = qk_nope_head_dim + 64
        max_seq_len = 128

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            head_dim_ckv,
            max_seq_len,
        )

        output = self._run_cached_mla(data)

        # Verify output shape
        v_head_dim = head_dim_ckv - qk_nope_head_dim
        expected_shape = (1, batch_size * seq_len, num_heads, v_head_dim)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        # Verify output is finite
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_cache_update_correctness(self):
        """Test that cache is updated correctly during forward pass."""
        batch_size, seq_len, num_heads = 1, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        head_dim_ckv = qk_nope_head_dim + 32
        # head_dim_kpe = qk_rope_head_dim
        max_seq_len = 32
        cache_offset = 5

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            head_dim_ckv,
            max_seq_len,
            cache_offset,
        )

        # Store original cache values at target position
        original_cache_at_pos = data["mla_cache"][0, cache_offset].clone()

        # Run forward pass
        _ = self._run_cached_mla(data)

        # Check cache was updated at the correct position
        updated_cache_at_pos = data["mla_cache"][0, cache_offset]

        # The cache should have been updated (values should be different from zeros/original)
        # since we're writing ckv and kpe to that position
        assert not torch.allclose(original_cache_at_pos, updated_cache_at_pos, atol=1e-6), (
            "Cache should have been updated at the target position"
        )

    def test_cache_layout_flashinfer_compatible(self):
        """Test that cache layout matches FlashInfer spec."""
        batch_size, seq_len, num_heads = 2, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 64, 32
        head_dim_ckv = 512  # DeepSeek-style kv_lora_rank
        head_dim_kpe = qk_rope_head_dim
        max_seq_len = 64

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            head_dim_ckv,
            max_seq_len,
        )

        # Verify cache shape matches FlashInfer layout
        expected_cache_shape = (batch_size, max_seq_len, head_dim_ckv + head_dim_kpe)
        assert data["mla_cache"].shape == expected_cache_shape, (
            f"Cache shape {data['mla_cache'].shape} doesn't match FlashInfer layout {expected_cache_shape}"
        )

        # Verify zero-copy slicing works
        ckv_slice = data["mla_cache"][:, :, :head_dim_ckv]
        kpe_slice = data["mla_cache"][:, :, head_dim_ckv:]

        assert ckv_slice.shape == (batch_size, max_seq_len, head_dim_ckv)
        assert kpe_slice.shape == (batch_size, max_seq_len, head_dim_kpe)

        # Verify slices share memory (zero-copy)
        assert ckv_slice.data_ptr() == data["mla_cache"].data_ptr(), "ckv slice should be zero-copy"

    def test_generate_with_reference(self):
        """Test generate phase against numpy reference."""
        batch_size, seq_len, num_heads = 2, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        head_dim_ckv = qk_nope_head_dim + 32
        max_seq_len = 64
        cache_offset = 3

        data = self._create_cached_mla_data(
            batch_size,
            seq_len,
            num_heads,
            qk_nope_head_dim,
            qk_rope_head_dim,
            head_dim_ckv,
            max_seq_len,
            cache_offset,
        )

        # Run backend
        output = self._run_cached_mla(data)

        # Run numpy reference (convert to float32 first since numpy doesn't support bfloat16)
        reference = numpy_mla_reference(
            data["q_nope"].cpu().float().numpy(),
            data["q_pe"].cpu().float().numpy(),
            data["ckv"].cpu().float().numpy(),
            data["kpe"].cpu().float().numpy(),
            data["mla_cache"].cpu().float().numpy(),
            data["seq_len"].cpu().numpy(),
            data["input_pos"].cpu().numpy(),
            data["cache_loc"].cpu().numpy(),
            data["cu_seqlen"].cpu().numpy(),
            None,
            head_dim_ckv,
            is_generate=True,
        )

        reference_torch = torch.from_numpy(reference).to(output.device, output.dtype)
        assert torch.allclose(output, reference_torch, atol=self.atol, rtol=self.rtol), (
            f"Generate phase output doesn't match reference. Max diff: {(output - reference_torch).abs().max():.6f}"
        )

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        batch_size, seq_len, num_heads = 1, 1, 4
        qk_nope_head_dim, qk_rope_head_dim = 32, 16
        head_dim_ckv = qk_nope_head_dim + 32
        max_seq_len = 32

        for dtype in [torch.float16, torch.bfloat16]:
            self.dtype = dtype
            data = self._create_cached_mla_data(
                batch_size,
                seq_len,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                head_dim_ckv,
                max_seq_len,
            )

            output = self._run_cached_mla(data)
            assert output.dtype == dtype, f"Expected dtype {dtype}, got {output.dtype}"


class TestMLADescriptor:
    """Test MultiHeadLatentAttention descriptor configuration."""

    def test_descriptor_registration(self):
        """Test that MLA descriptor is properly registered."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import AttentionRegistry

        assert AttentionRegistry.has("MultiHeadLatentAttention"), (
            "MultiHeadLatentAttention should be registered"
        )

    def test_descriptor_layout(self):
        """Test that MLA descriptor uses correct layout."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.mla import MultiHeadLatentAttention

        assert MultiHeadLatentAttention.get_attention_layout() == "bsnd", (
            "MLA should use bsnd layout"
        )

    def test_descriptor_num_qkv_args(self):
        """Test that MLA descriptor expects 4 qkv args."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.mla import MultiHeadLatentAttention

        assert MultiHeadLatentAttention.get_num_qkv_args() == 4, (
            "MLA should expect 4 qkv args (q_nope, q_pe, ckv, kpe)"
        )

    def test_descriptor_source_op(self):
        """Test that MLA descriptor points to correct source op."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.mla import MultiHeadLatentAttention

        source_op = MultiHeadLatentAttention.get_source_attention_op()
        assert source_op == torch.ops.auto_deploy.torch_mla, "MLA should use torch_mla as source op"

    def test_descriptor_cached_op(self):
        """Test that MLA descriptor points to correct cached op."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.mla import MultiHeadLatentAttention

        cached_op = MultiHeadLatentAttention.get_cached_attention_op()
        assert cached_op == torch.ops.auto_deploy.torch_cached_mla_with_cache.default, (
            "MLA should use torch_cached_mla_with_cache as cached op"
        )

    def test_descriptor_standard_metadata(self):
        """Test that MLA descriptor uses standard metadata args."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.mla import MultiHeadLatentAttention

        expected_args = ["batch_info_host", "seq_len", "input_pos", "cache_loc", "cu_seqlen"]
        actual_args = MultiHeadLatentAttention.get_standard_metadata_args()
        assert actual_args == expected_args, (
            f"Expected standard metadata {expected_args}, got {actual_args}"
        )
