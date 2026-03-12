"""Tests for DeepSeekV32 custom model implementation for auto_deploy export.

Covers the DeepSeek-V3.2 model family (model_type="deepseek_v32"):
- deepseek-ai/DeepSeek-V3.2
- deepseek-ai/DeepSeek-V3.2-Speciale
- nvidia/DeepSeek-V3.2-NVFP4

The core forward path (MLA + MoE + RMSNorm + RoPE) is identical to DeepSeek-V3.
V3.2 adds an Indexer module (for sparse attention) and MTP layers (multi-token
prediction), both included for weight loading but not used in the prefill forward path.
Numerical equivalence is tested against the HF DeepSeek-V3 reference since the
forward math is identical.
"""

import pytest
import torch
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v32 import (
    DeepSeekV32Attention,
    DeepseekV32Config,
    DeepSeekV32DecoderLayer,
    DeepSeekV32ForCausalLM,
    DeepSeekV32Indexer,
    DeepSeekV32MLP,
    DeepSeekV32MoE,
    DeepSeekV32MTPDecoderLayer,
    DeepSeekV32MTPSharedHead,
    DeepSeekV32RMSNorm,
    DeepSeekV32RotaryEmbedding,
    DeepSeekV32YarnRotaryEmbedding,
)
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


# =============================================================================
# Config helpers
# =============================================================================


def _create_config(**kwargs):
    """Create a small DeepSeekV32 config for testing."""
    defaults = dict(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        pad_token_id=0,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=64,
        num_nextn_predict_layers=1,
    )
    defaults.update(kwargs)
    return DeepseekV32Config(**defaults)


def _create_config_no_mtp(**kwargs):
    """Create config without MTP layers."""
    return _create_config(num_nextn_predict_layers=0, **kwargs)


def _make_cos_sin(config, B, S, device, dtype):
    """Create pre-sliced cos/sin for testing attention/layer directly."""
    rope = DeepSeekV32RotaryEmbedding(
        config.qk_rope_head_dim,
        max_position_embeddings=config.max_position_embeddings,
        base=config.rope_theta,
    ).to(device)
    x_dummy = torch.randn(1, 1, 1, config.qk_rope_head_dim, dtype=dtype, device=device)
    cos, sin = rope(x_dummy)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    return cos[position_ids].to(dtype), sin[position_ids].to(dtype)


# =============================================================================
# RMSNorm Tests
# =============================================================================


class TestDeepSeekV32RMSNorm:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def test_forward_shape(self):
        hidden_size = 64
        norm = DeepSeekV32RMSNorm(hidden_size).to(self.device, self.dtype)
        x = torch.randn(2, 4, hidden_size, dtype=self.dtype, device=self.device)
        output = norm(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_output_normalized(self):
        hidden_size = 64
        norm = DeepSeekV32RMSNorm(hidden_size).to(self.device, torch.float32)
        x = torch.randn(2, 4, hidden_size, dtype=torch.float32, device=self.device)
        output = norm(x)
        rms = torch.sqrt((output**2).mean(-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


# =============================================================================
# Rotary Embedding Tests
# =============================================================================


class TestDeepSeekV32RotaryEmbedding:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def test_base_rope_shape(self):
        dim = 8
        max_pos = 512
        rope = DeepSeekV32RotaryEmbedding(dim, max_pos).to(self.device)
        x = torch.randn(2, 4, 4, dim, dtype=self.dtype, device=self.device)
        cos, sin = rope(x)
        assert cos.shape == (max_pos, dim)
        assert sin.shape == (max_pos, dim)

    def test_yarn_rope_shape(self):
        dim = 8
        max_pos = 512
        rope = DeepSeekV32YarnRotaryEmbedding(
            dim,
            max_pos,
            scaling_factor=2.0,
            original_max_position_embeddings=256,
        ).to(self.device)
        x = torch.randn(2, 4, 4, dim, dtype=self.dtype, device=self.device)
        cos, sin = rope(x)
        assert cos.shape == (max_pos, dim)
        assert sin.shape == (max_pos, dim)


# =============================================================================
# MLP Tests
# =============================================================================


class TestDeepSeekV32MLP:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def test_forward_shape(self):
        config = _create_config()
        mlp = DeepSeekV32MLP(config).to(self.device, self.dtype)
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = mlp(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))


# =============================================================================
# MoE Tests
# =============================================================================


class TestDeepSeekV32MoE:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def _create_moe(self, config):
        moe = DeepSeekV32MoE(config).to(self.device, self.dtype)
        moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
        return moe

    def test_forward_shape(self):
        config = _create_config()
        moe = self._create_moe(config)
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_with_shared_experts(self):
        config = _create_config(n_shared_experts=2)
        moe = self._create_moe(config)
        assert moe.shared_experts is not None
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_without_shared_experts(self):
        config = _create_config(n_shared_experts=None)
        moe = self._create_moe(config)
        assert moe.shared_experts is None
        x = torch.randn(2, 4, config.hidden_size, dtype=self.dtype, device=self.device)
        output = moe(x)
        assert output.shape == x.shape
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_expert_structure(self):
        config = _create_config()
        moe = DeepSeekV32MoE(config)
        assert isinstance(moe.experts, torch.nn.ModuleList)
        assert len(moe.experts) == config.n_routed_experts
        for i, expert in enumerate(moe.experts):
            assert hasattr(expert, "gate_proj"), f"Expert {i} missing gate_proj"
            assert hasattr(expert, "up_proj"), f"Expert {i} missing up_proj"
            assert hasattr(expert, "down_proj"), f"Expert {i} missing down_proj"


# =============================================================================
# Attention Tests
# =============================================================================


class TestDeepSeekV32Attention:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def test_forward_shape_with_qlora(self):
        config = _create_config()
        assert config.q_lora_rank is not None
        attn = DeepSeekV32Attention(config, layer_idx=0).to(self.device, self.dtype)

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        cos, sin = _make_cos_sin(config, B, S, self.device, self.dtype)

        output = attn(hidden_states, cos, sin)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_forward_shape_no_qlora(self):
        config = _create_config(q_lora_rank=None)
        attn = DeepSeekV32Attention(config, layer_idx=0).to(self.device, self.dtype)

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        cos, sin = _make_cos_sin(config, B, S, self.device, self.dtype)

        output = attn(hidden_states, cos, sin)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_indexer_present(self):
        config = _create_config()
        attn = DeepSeekV32Attention(config, layer_idx=0)
        assert isinstance(attn.indexer, DeepSeekV32Indexer)
        # Verify indexer has expected sub-modules
        assert hasattr(attn.indexer, "wq_b")
        assert hasattr(attn.indexer, "wk")
        assert hasattr(attn.indexer, "k_norm")
        assert hasattr(attn.indexer, "weights_proj")

    def test_different_batch_sizes(self):
        config = _create_config()
        attn = DeepSeekV32Attention(config, layer_idx=0).to(self.device, self.dtype)
        for B in [1, 2, 4]:
            S = 4
            hidden_states = torch.randn(
                B, S, config.hidden_size, dtype=self.dtype, device=self.device
            )
            cos, sin = _make_cos_sin(config, B, S, self.device, self.dtype)
            output = attn(hidden_states, cos, sin)
            assert output.shape == (B, S, config.hidden_size)

    def test_different_sequence_lengths(self):
        config = _create_config()
        attn = DeepSeekV32Attention(config, layer_idx=0).to(self.device, self.dtype)
        for S in [1, 4, 16]:
            B = 2
            hidden_states = torch.randn(
                B, S, config.hidden_size, dtype=self.dtype, device=self.device
            )
            cos, sin = _make_cos_sin(config, B, S, self.device, self.dtype)
            output = attn(hidden_states, cos, sin)
            assert output.shape == (B, S, config.hidden_size)


# =============================================================================
# Decoder Layer Tests
# =============================================================================


class TestDeepSeekV32DecoderLayer:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def test_dense_layer(self):
        config = _create_config()
        layer = DeepSeekV32DecoderLayer(config, layer_idx=0).to(self.device, self.dtype)
        assert isinstance(layer.mlp, DeepSeekV32MLP)

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        cos, sin = _make_cos_sin(config, B, S, self.device, self.dtype)
        output = layer(hidden_states, cos, sin)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_moe_layer(self):
        config = _create_config()
        layer = DeepSeekV32DecoderLayer(config, layer_idx=1).to(self.device, self.dtype)
        assert isinstance(layer.mlp, DeepSeekV32MoE)
        layer.mlp.gate.weight = torch.nn.Parameter(torch.randn_like(layer.mlp.gate.weight))

        B, S = 2, 4
        hidden_states = torch.randn(B, S, config.hidden_size, dtype=self.dtype, device=self.device)
        cos, sin = _make_cos_sin(config, B, S, self.device, self.dtype)
        output = layer(hidden_states, cos, sin)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        assert not torch.allclose(output, torch.zeros_like(output))


# =============================================================================
# MTP Layer Tests
# =============================================================================


class TestDeepSeekV32MTPLayer:
    def test_mtp_layer_structure(self):
        config = _create_config()
        mtp_layer = DeepSeekV32MTPDecoderLayer(config, layer_idx=config.num_hidden_layers)

        # MTP-specific modules
        assert hasattr(mtp_layer, "embed_tokens")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "enorm")
        assert hasattr(mtp_layer, "hnorm")
        assert hasattr(mtp_layer, "shared_head")
        assert isinstance(mtp_layer.shared_head, DeepSeekV32MTPSharedHead)
        assert hasattr(mtp_layer.shared_head, "head")
        assert hasattr(mtp_layer.shared_head, "norm")

        # Standard decoder layer modules
        assert hasattr(mtp_layer, "self_attn")
        assert hasattr(mtp_layer, "mlp")
        assert hasattr(mtp_layer, "input_layernorm")
        assert hasattr(mtp_layer, "post_attention_layernorm")

    def test_mtp_layer_weight_keys(self):
        """Verify MTP layer state dict keys match checkpoint expectations."""
        config = _create_config()
        mtp_layer = DeepSeekV32MTPDecoderLayer(config, layer_idx=config.num_hidden_layers)
        state_dict = mtp_layer.state_dict()

        expected_keys = [
            "embed_tokens.weight",
            "eh_proj.weight",
            "enorm.weight",
            "hnorm.weight",
            "shared_head.head.weight",
            "shared_head.norm.weight",
            "self_attn.indexer.wq_b.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.k_norm.weight",
            "self_attn.indexer.k_norm.bias",
            "self_attn.indexer.weights_proj.weight",
        ]
        for key in expected_keys:
            assert key in state_dict, f"Missing key: {key}"


# =============================================================================
# Full Model Tests
# =============================================================================


class TestDeepSeekV32ForCausalLM:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16

    def test_forward(self):
        config = _create_config()
        model = DeepSeekV32ForCausalLM(config).to(self.device, self.dtype)

        B, S = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        output = model(input_ids=input_ids, position_ids=position_ids)
        assert output.logits.shape == (B, S, config.vocab_size)
        assert torch.isfinite(output.logits).all()
        assert not torch.allclose(output.logits, torch.zeros_like(output.logits))

    def test_output_dtype(self):
        config = _create_config()
        model = DeepSeekV32ForCausalLM(config).to(self.device, self.dtype)

        B, S = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=self.device)
        position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        output = model(input_ids=input_ids, position_ids=position_ids)
        assert output.logits.dtype == torch.float32
        assert torch.isfinite(output.logits).all()

    def test_position_ids_required(self):
        config = _create_config()
        model = DeepSeekV32ForCausalLM(config).to(self.device, self.dtype)

        B, S = 2, 4
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=self.device)

        with pytest.raises(AssertionError, match="position_ids is required"):
            model(input_ids=input_ids)

    def test_layer_types(self):
        config = _create_config()
        model = DeepSeekV32ForCausalLM(config)
        # Layer 0 should be dense
        assert type(model.model.layers[0].mlp).__name__ == "DeepSeekV32MLP"
        # Layer 1+ should be MoE
        for i in range(1, config.num_hidden_layers):
            assert type(model.model.layers[i].mlp).__name__ == "DeepSeekV32MoE"

    def test_mtp_layers_present(self):
        config = _create_config()
        model = DeepSeekV32ForCausalLM(config)
        total_layers = config.num_hidden_layers + config.num_nextn_predict_layers
        assert len(model.model.layers) == total_layers
        # Last layer should be MTP
        assert isinstance(model.model.layers[-1], DeepSeekV32MTPDecoderLayer)

    def test_no_mtp_layers(self):
        config = _create_config_no_mtp()
        model = DeepSeekV32ForCausalLM(config)
        assert len(model.model.layers) == config.num_hidden_layers

    def test_indexer_in_all_layers(self):
        config = _create_config()
        model = DeepSeekV32ForCausalLM(config)
        for i, layer in enumerate(model.model.layers):
            assert hasattr(layer.self_attn, "indexer"), f"Layer {i} missing indexer"
            assert isinstance(layer.self_attn.indexer, DeepSeekV32Indexer)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_full_model(B, S, dtype):
    """Test full model produces valid output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _create_config()
    model = DeepSeekV32ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    output = model(input_ids=input_ids, position_ids=position_ids)

    assert output.logits.shape == (B, S, config.vocab_size)
    assert not torch.isnan(output.logits).any()
    assert not torch.isinf(output.logits).any()
    assert not torch.allclose(output.logits, torch.zeros_like(output.logits))


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_moe_layer(B, S, dtype):
    """Test MoE layer produces valid output."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = _create_config()

    moe = DeepSeekV32MoE(config).to(device=device, dtype=dtype)
    moe.gate.weight = torch.nn.Parameter(torch.randn_like(moe.gate.weight))
    moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    output = moe(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert not torch.allclose(output, torch.zeros_like(output))


# =============================================================================
# Export Test
# =============================================================================


def test_deepseek_v32_model_can_be_exported():
    """Test that the custom model can be exported with torch_export_to_gm."""
    device = "cuda"
    dtype = torch.bfloat16
    config = _create_config_no_mtp()

    model = DeepSeekV32ForCausalLM(config)
    model.to(device=device, dtype=dtype)
    model.eval()

    B, S = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    gm = torch_export_to_gm(
        model,
        args=tuple(),
        kwargs={"input_ids": input_ids, "position_ids": position_ids},
        dynamic_shapes=dynamic_shapes,
    )

    move_to_device(gm, device)

    with torch.inference_mode():
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    assert "logits" in out_gm
    logits = out_gm["logits"]
    assert logits.shape == (B, S, config.vocab_size)
    assert torch.isfinite(logits).all()

    # Test with different input shape (dynamic shapes)
    B2, S2 = 1, 4
    input_ids2 = torch.randint(0, config.vocab_size, (B2, S2), device=device)
    position_ids2 = torch.arange(S2, device=device).unsqueeze(0).expand(B2, -1)

    with torch.inference_mode():
        out_gm2 = gm(input_ids=input_ids2, position_ids=position_ids2)

    logits2 = out_gm2["logits"]
    assert logits2.shape == (B2, S2, config.vocab_size)
    assert torch.isfinite(logits2).all()


# =============================================================================
# Registration Test
# =============================================================================


def test_deepseek_v32_model_registration():
    """Test that DeepSeekV32ForCausalLM is registered with the factory."""
    from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory

    assert "DeepseekV32Config" in AutoModelForCausalLMFactory._custom_model_mapping
    assert (
        AutoModelForCausalLMFactory._custom_model_mapping["DeepseekV32Config"]
        == DeepSeekV32ForCausalLM
    )


def test_deepseek_v32_config_registration():
    """Test that DeepseekV32Config is registered with AutoConfig."""
    from transformers import AutoConfig

    config = AutoConfig.for_model("deepseek_v32")
    assert config.__class__.__name__ == "DeepseekV32Config"


# =============================================================================
# Numerical Equivalence Tests (against HF DeepSeek-V3 reference)
# =============================================================================


def _get_hf_v3_model_class():
    """Get the HF DeepseekV3ForCausalLM class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3ForCausalLM as HFDeepseekV3ForCausalLM,
        )

        return HFDeepseekV3ForCausalLM
    except ImportError:
        return None


def _get_hf_v3_config_class():
    """Get the HF DeepseekV3Config class."""
    try:
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
            DeepseekV3Config as HFDeepseekV3Config,
        )

        return HFDeepseekV3Config
    except ImportError:
        return None


def _get_hf_v3_mlp_class():
    """Get the HF DeepseekV3MLP class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3MLP as HFDeepseekV3MLP,
        )

        return HFDeepseekV3MLP
    except ImportError:
        return None


def _get_hf_v3_moe_class():
    """Get the HF DeepseekV3MoE class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3MoE as HFDeepseekV3MoE,
        )

        return HFDeepseekV3MoE
    except ImportError:
        return None


def _get_hf_v3_rmsnorm_class():
    """Get the HF DeepseekV3RMSNorm class."""
    try:
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
            DeepseekV3RMSNorm as HFDeepseekV3RMSNorm,
        )

        return HFDeepseekV3RMSNorm
    except ImportError:
        return None


def _create_hf_v3_config():
    """Create HF V3 config matching our small test config."""
    HFConfig = _get_hf_v3_config_class()
    if HFConfig is None:
        return None

    return HFConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=0,
    )


def _filter_hf_state_dict(hf_sd):
    """Filter out HF-specific buffers/keys not present in custom model."""
    return {k: v for k, v in hf_sd.items() if "ep_rank" not in k and "experts_per_rank" not in k}


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_rmsnorm_numerical_equivalence(B, S, dtype):
    """Test RMSNorm produces numerically equivalent output to HF V3."""
    HFRMSNorm = _get_hf_v3_rmsnorm_class()
    if HFRMSNorm is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    hidden_size = 64

    hf_norm = HFRMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    hf_norm.eval()

    custom_norm = DeepSeekV32RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=dtype)
    custom_norm.load_state_dict(hf_norm.state_dict())
    custom_norm.eval()

    x = torch.randn(B, S, hidden_size, device=device, dtype=dtype)

    hf_out = hf_norm(x)
    custom_out = custom_norm(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_mlp_numerical_equivalence(B, S, dtype):
    """Test MLP produces numerically equivalent output to HF V3."""
    HFMLP = _get_hf_v3_mlp_class()
    if HFMLP is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    config = _create_config_no_mtp()
    hf_config = _create_hf_v3_config()

    hf_mlp = HFMLP(hf_config).to(device=device, dtype=dtype)
    hf_mlp.eval()

    custom_mlp = DeepSeekV32MLP(config).to(device=device, dtype=dtype)
    custom_mlp.load_state_dict(hf_mlp.state_dict())
    custom_mlp.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_mlp(x)
    custom_out = custom_mlp(x)

    torch.testing.assert_close(custom_out, hf_out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_moe_numerical_equivalence(B, S, dtype):
    """Test MoE produces numerically equivalent output to HF V3."""
    HFMoE = _get_hf_v3_moe_class()
    if HFMoE is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    config = _create_config_no_mtp()
    hf_config = _create_hf_v3_config()

    hf_moe = HFMoE(hf_config).to(device=device, dtype=dtype)
    hf_moe.eval()
    hf_moe.gate.weight = torch.nn.Parameter(torch.randn_like(hf_moe.gate.weight))

    custom_moe = DeepSeekV32MoE(config).to(device=device, dtype=dtype)
    hf_sd = _filter_hf_state_dict(hf_moe.state_dict())
    custom_moe.load_state_dict(hf_sd)
    custom_moe.eval()

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)

    hf_out = hf_moe(x)
    if isinstance(hf_out, tuple):
        hf_out = hf_out[0]
    custom_out = custom_moe(x)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.02, msg="MoE: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_attention_numerical_equivalence(B, S, dtype):
    """Test attention (MLA) produces numerically equivalent output to HF V3.

    Uses the full model to load weights (via the model-level de-interleaving hook),
    then compares attention layer outputs directly.
    """
    HFModel = _get_hf_v3_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    config = _create_config_no_mtp()
    hf_config = _create_hf_v3_config()

    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    hf_attn = hf_model.model.layers[0].self_attn
    hf_rope = hf_model.model.rotary_emb

    custom_model = DeepSeekV32ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = _filter_hf_state_dict(hf_model.state_dict())
    # Custom model has extra indexer keys, load with strict=False
    custom_model.load_state_dict(hf_sd, strict=False)
    custom_model.eval()
    custom_attn = custom_model.model.layers[0].self_attn

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    pos_emb = hf_rope(x, position_ids)
    hf_out, _ = hf_attn(x, position_embeddings=pos_emb, position_ids=position_ids)
    cos, sin = _make_cos_sin(config, B, S, device, dtype)
    custom_out = custom_attn(x, cos, sin)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.10, msg="Attention: ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_decoder_layer_numerical_equivalence(B, S, dtype):
    """Test decoder layer (dense, layer_idx=0) produces numerically equivalent output to HF V3."""
    HFModel = _get_hf_v3_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    config = _create_config_no_mtp()
    hf_config = _create_hf_v3_config()

    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    hf_layer = hf_model.model.layers[0]
    hf_rope = hf_model.model.rotary_emb

    custom_model = DeepSeekV32ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = _filter_hf_state_dict(hf_model.state_dict())
    custom_model.load_state_dict(hf_sd, strict=False)
    custom_model.eval()
    custom_layer = custom_model.model.layers[0]

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    pos_emb = hf_rope(x, position_ids)
    hf_out = hf_layer(x, position_ids=position_ids, position_embeddings=pos_emb)
    cos, sin = _make_cos_sin(config, B, S, device, dtype)
    custom_out = custom_layer(x, cos, sin)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer (dense): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_moe_decoder_layer_numerical_equivalence(B, S, dtype):
    """Test MoE decoder layer (layer_idx=1) produces numerically equivalent output to HF V3."""
    HFModel = _get_hf_v3_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    config = _create_config_no_mtp()
    hf_config = _create_hf_v3_config()

    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    for module in hf_model.modules():
        if hasattr(module, "gate") and hasattr(module.gate, "weight"):
            module.gate.weight = torch.nn.Parameter(torch.randn_like(module.gate.weight))

    hf_layer = hf_model.model.layers[1]
    hf_rope = hf_model.model.rotary_emb

    custom_model = DeepSeekV32ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = _filter_hf_state_dict(hf_model.state_dict())
    custom_model.load_state_dict(hf_sd, strict=False)
    custom_model.eval()
    custom_layer = custom_model.model.layers[1]

    x = torch.randn(B, S, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    pos_emb = hf_rope(x, position_ids)
    hf_out = hf_layer(x, position_ids=position_ids, position_embeddings=pos_emb)
    cos, sin = _make_cos_sin(config, B, S, device, dtype)
    custom_out = custom_layer(x, cos, sin)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(custom_out, hf_out, rmse_ratio_tol=0.05, msg="Decoder layer (MoE): ")


@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
def test_deepseek_v32_full_model_numerical_equivalence(B, S, dtype):
    """Test full model produces numerically equivalent output to HF V3."""
    HFModel = _get_hf_v3_model_class()
    if HFModel is None:
        pytest.skip("transformers doesn't have deepseek_v3 modeling")

    device = "cuda"
    config = _create_config_no_mtp()
    hf_config = _create_hf_v3_config()

    hf_model = HFModel(hf_config).to(device=device, dtype=dtype)
    hf_model.eval()
    for module in hf_model.modules():
        if hasattr(module, "gate") and hasattr(module.gate, "weight"):
            module.gate.weight = torch.nn.Parameter(torch.randn_like(module.gate.weight))

    custom_model = DeepSeekV32ForCausalLM(config).to(device=device, dtype=dtype)
    hf_sd = _filter_hf_state_dict(hf_model.state_dict())
    custom_model.load_state_dict(hf_sd, strict=False)
    custom_model.eval()

    input_ids = torch.randint(0, config.vocab_size, (B, S), device=device)
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    hf_out = hf_model(input_ids=input_ids, position_ids=position_ids)
    custom_out = custom_model(input_ids=input_ids, position_ids=position_ids)

    from _model_test_utils import assert_rmse_close

    assert_rmse_close(
        custom_out.logits.float(),
        hf_out.logits.float(),
        rmse_ratio_tol=0.05,
        msg="Full model: ",
    )
