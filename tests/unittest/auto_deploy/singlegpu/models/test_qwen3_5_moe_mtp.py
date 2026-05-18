# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re

import pytest
import torch
import torch.nn.functional as F
from torch.export import Dim

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import (
    EagleConfig,
    EagleDrafterForCausalLM,
    EagleWrapper,
    get_eagle_layers,
)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeEagleLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeTopKRouter,
    apply_rotary_pos_emb,
)
from tensorrt_llm._torch.auto_deploy.models.eagle import (
    DraftModelExportInfo,
    EagleDrafterFactory,
    EagleOneModelFactory,
    TargetModelExportInfo,
)
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.hidden_states import (
    DetectHiddenStatesForCapture,
)
from tensorrt_llm.llmapi import MTPDecodingConfig


def _make_small_config(**overrides) -> Qwen3_5MoeTextConfig:
    defaults = dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=16,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 2, 2],
        },
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        layer_types=["full_attention"] * 4,
        mtp_num_hidden_layers=1,
    )
    defaults.update(overrides)
    return Qwen3_5MoeTextConfig(**defaults)


def _write_qwen35_mtp_config(tmp_path, **overrides):
    config_dir = tmp_path / "qwen3_5_moe_mtp"
    config_dir.mkdir()
    composite_config = Qwen3_5MoeConfig(text_config=_make_small_config(**overrides).to_dict())
    with (config_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(composite_config.to_dict(), config_file)
    return config_dir


def _build_qwen35_mtp_one_model_factory(config_dir) -> EagleOneModelFactory:
    return EagleOneModelFactory(
        model=str(config_dir),
        skip_loading_weights=True,
        max_seq_len=64,
        speculative_config=MTPDecodingConfig(
            max_draft_len=1,
            mtp_eagle_one_model=True,
            speculative_model=str(config_dir),
        ),
    )


def _dynamic_shapes_for_eagle_target_export():
    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    return {
        "inputs_embeds": {0: batch_size_dynamic, 1: seq_len_dynamic},
        "position_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
    }


def _dynamic_shapes_for_eagle_draft_export():
    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    return {
        "inputs_embeds": {0: batch_size_dynamic, 1: seq_len_dynamic},
        "position_ids": {0: batch_size_dynamic, 1: seq_len_dynamic},
        "hidden_states": {0: batch_size_dynamic, 1: seq_len_dynamic},
    }


def _export_qwen35_mtp_target_for_capture(num_hidden_layers: int = 3):
    config = _make_small_config(
        num_hidden_layers=num_hidden_layers,
        layer_types=["full_attention"] * num_hidden_layers,
    )
    model = Qwen3_5MoeForCausalLM(config).eval()

    inputs_embeds = torch.randn(1, 4, config.hidden_size)
    position_ids = torch.arange(4).view(1, 4)
    gm = torch_export_to_gm(
        model,
        args=(),
        kwargs={
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        },
        dynamic_shapes=_dynamic_shapes_for_eagle_target_export(),
        strict=False,
        num_moe_experts_for_export=2,
    )
    return gm, config


def _init_module_weights(module: torch.nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            submodule.weight.data.normal_(mean=0.0, std=0.02)
            if submodule.bias is not None:
                submodule.bias.data.zero_()
        elif isinstance(submodule, Qwen3_5MoeTopKRouter):
            submodule.weight.data.normal_(mean=0.0, std=0.5)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    output = x.float()
    output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * output).to(input_dtype)


def _mlp(module, x: torch.Tensor) -> torch.Tensor:
    return module.down_proj(F.silu(module.gate_proj(x)) * module.up_proj(x))


def _manual_moe(moe, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(hidden_states_flat, moe.gate.weight)
    routing_weights = F.softmax(router_logits, dtype=torch.float, dim=-1)
    routing_weights, selected_experts = torch.topk(routing_weights, moe.gate.top_k, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    expert_output = torch.zeros_like(hidden_states_flat)
    for slot in range(moe.gate.top_k):
        slot_experts = selected_experts[:, slot]
        slot_weights = routing_weights[:, slot]
        for expert_idx, expert in enumerate(moe.experts):
            mask = slot_experts == expert_idx
            if mask.any():
                expert_output[mask] += _mlp(expert, hidden_states_flat[mask]) * slot_weights[
                    mask
                ].unsqueeze(-1)

    shared_output = _mlp(moe.shared_expert, hidden_states_flat)
    shared_output = torch.sigmoid(moe.shared_expert_gate(hidden_states_flat)) * shared_output
    return (expert_output + shared_output).view(batch_size, sequence_length, hidden_dim)


def _manual_attention(layer: Qwen3_5MoeEagleLayer, hidden_states: torch.Tensor, inputs_embeds):
    batch_size, sequence_length, _ = hidden_states.shape
    position_ids = torch.arange(sequence_length).view(1, sequence_length)
    position_ids = position_ids.expand(batch_size, -1)
    position_ids = position_ids[None, ...].expand(3, batch_size, -1)
    position_embeddings = layer.rotary_emb(inputs_embeds, position_ids)

    attention = layer.self_attn
    qg = attention.q_proj(hidden_states).view(
        batch_size, sequence_length, -1, attention.head_dim * 2
    )
    query_states, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(batch_size, sequence_length, -1)
    key_states = attention.k_proj(hidden_states).view(
        batch_size, sequence_length, attention.num_key_value_heads, attention.head_dim
    )
    value_states = attention.v_proj(hidden_states).view(
        batch_size, sequence_length, attention.num_key_value_heads, attention.head_dim
    )

    query_states = _rms_norm(query_states, attention.q_norm.weight, attention.q_norm.eps)
    key_states = _rms_norm(key_states, attention.k_norm.weight, attention.k_norm.eps)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, *position_embeddings, unsqueeze_dim=2
    )

    repeats = attention.num_heads // attention.num_key_value_heads
    key_states = key_states.repeat_interleave(repeats, dim=2)
    value_states = value_states.repeat_interleave(repeats, dim=2)
    attn_output = F.scaled_dot_product_attention(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        is_causal=True,
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, sequence_length, -1)
    return attention.o_proj(attn_output * torch.sigmoid(gate))


def _manual_qwen_mtp_layer(
    layer: Qwen3_5MoeEagleLayer,
    hidden_states: torch.Tensor,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor:
    inputs_embeds_norm = _rms_norm(
        inputs_embeds, layer.pre_fc_norm_embedding.weight, layer.pre_fc_norm_embedding.eps
    )
    hidden_states_norm = _rms_norm(
        hidden_states, layer.pre_fc_norm_hidden.weight, layer.pre_fc_norm_hidden.eps
    )
    hidden_states = layer.fc(torch.cat([inputs_embeds_norm, hidden_states_norm], dim=-1))

    residual = hidden_states
    hidden_states = _rms_norm(
        hidden_states, layer.input_layernorm.weight, layer.input_layernorm.eps
    )
    hidden_states = residual + _manual_attention(layer, hidden_states, inputs_embeds)

    residual = hidden_states
    hidden_states = _rms_norm(
        hidden_states, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps
    )
    hidden_states = residual + _manual_moe(layer.mlp, hidden_states)
    return _rms_norm(hidden_states, layer.norm.weight, layer.norm.eps)


def _remap_key(mapping: dict[str, str], key: str) -> str:
    for pattern, replacement in mapping.items():
        key = re.sub(pattern, replacement, key)
    return key


def _to_hf_mtp_key(key: str) -> str:
    if key.startswith("model.layers.fc."):
        return key.replace("model.layers.fc.", "mtp.fc.", 1)
    if key.startswith("model.layers.pre_fc_norm_embedding."):
        return key.replace("model.layers.pre_fc_norm_embedding.", "mtp.pre_fc_norm_embedding.", 1)
    if key.startswith("model.layers.pre_fc_norm_hidden."):
        return key.replace("model.layers.pre_fc_norm_hidden.", "mtp.pre_fc_norm_hidden.", 1)
    if key.startswith("model.layers.norm."):
        return key.replace("model.layers.norm.", "mtp.norm.", 1)
    if key.startswith("model.layers."):
        return key.replace("model.layers.", "mtp.layers.0.", 1)
    return key


def test_qwen35_mtp_eagle_config_defaults_and_checkpoint_mapping():
    config = _make_small_config()
    eagle_config = EagleConfig.from_base_config(config, config.model_type)

    assert eagle_config.load_embedding_from_target is True
    assert eagle_config.load_lm_head_from_target is True
    assert eagle_config.num_capture_layers == 1
    assert eagle_config.normalize_target_hidden_state is True
    assert eagle_config.layers_handle_final_norm is True

    mapping = eagle_config._checkpoint_conversion_mapping
    assert _remap_key(mapping, "mtp.fc.weight") == "model.layers.fc.weight"
    assert (
        _remap_key(mapping, "mtp.pre_fc_norm_embedding.weight")
        == "model.layers.pre_fc_norm_embedding.weight"
    )
    assert (
        _remap_key(mapping, "mtp.layers.0.self_attn.q_proj.weight")
        == "model.layers.self_attn.q_proj.weight"
    )
    assert _remap_key(mapping, "mtp.norm.weight") == "model.layers.norm.weight"


def test_qwen35_mtp_eagle_layer_matches_manual_reference():
    torch.manual_seed(0)
    config = _make_small_config()
    layer = Qwen3_5MoeEagleLayer(config, layer_idx=0)
    _init_module_weights(layer)

    hidden_states = torch.randn(2, 5, config.hidden_size)
    inputs_embeds = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.arange(5).view(1, 5).expand(2, -1)

    actual = layer(hidden_states, inputs_embeds, position_ids)
    expected = _manual_qwen_mtp_layer(layer, hidden_states, inputs_embeds)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    actual_with_1d_position_ids = layer(hidden_states, inputs_embeds, torch.arange(5))
    torch.testing.assert_close(actual_with_1d_position_ids, expected, rtol=2e-4, atol=2e-4)


def test_qwen35_mtp_eagle_layer_components_match_manual_reference():
    torch.manual_seed(1)
    config = _make_small_config()
    layer = Qwen3_5MoeEagleLayer(config, layer_idx=0)
    _init_module_weights(layer)

    hidden_states = torch.randn(2, 5, config.hidden_size)
    inputs_embeds = torch.randn(2, 5, config.hidden_size)
    position_ids = torch.arange(5).view(1, 5).expand(2, -1)

    actual_prologue = layer.fc(
        torch.cat(
            [
                layer.pre_fc_norm_embedding(inputs_embeds),
                layer.pre_fc_norm_hidden(hidden_states),
            ],
            dim=-1,
        )
    )
    expected_prologue = layer.fc(
        torch.cat(
            [
                _rms_norm(
                    inputs_embeds,
                    layer.pre_fc_norm_embedding.weight,
                    layer.pre_fc_norm_embedding.eps,
                ),
                _rms_norm(
                    hidden_states,
                    layer.pre_fc_norm_hidden.weight,
                    layer.pre_fc_norm_hidden.eps,
                ),
            ],
            dim=-1,
        )
    )
    torch.testing.assert_close(actual_prologue, expected_prologue)

    normed = layer.input_layernorm(actual_prologue)
    position_embeddings = layer.rotary_emb(
        inputs_embeds, position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    )
    actual_attention = layer.self_attn(normed, position_embeddings=position_embeddings)
    expected_attention = _manual_attention(layer, normed, inputs_embeds)
    torch.testing.assert_close(actual_attention, expected_attention, rtol=2e-4, atol=2e-4)

    normed_after_attention = layer.post_attention_layernorm(actual_prologue + actual_attention)
    actual_mlp = layer.mlp(normed_after_attention)
    expected_mlp = _manual_moe(layer.mlp, normed_after_attention)
    torch.testing.assert_close(actual_mlp, expected_mlp, rtol=2e-4, atol=2e-4)


def test_qwen35_mtp_builder_rejects_unsupported_layer_count():
    config = _make_small_config(mtp_num_hidden_layers=2)
    eagle_config = EagleConfig.from_base_config(config, config.model_type)

    with pytest.raises(ValueError, match="exactly one MTP layer"):
        get_eagle_layers(eagle_config, eagle_config.model_type)


def test_qwen35_causallm_unwraps_composite_config_and_exposes_eagle_hooks():
    text_config = _make_small_config()
    composite_config = Qwen3_5MoeConfig(text_config=text_config.to_dict())
    model = Qwen3_5MoeForCausalLM(composite_config)

    assert isinstance(model.config, Qwen3_5MoeTextConfig)
    assert model.get_input_embeddings() is model.model.embed_tokens
    assert model.get_output_embeddings() is model.lm_head
    assert model.get_final_normalization() is model.model.norm


def test_qwen35_mtp_drafter_tolerates_unused_hf_kwargs():
    config = _make_small_config()
    eagle_config = EagleConfig.from_base_config(config, config.model_type)

    model = Qwen3_5MoeForCausalLM._from_config(config)
    assert model.config.model_type == "qwen3_5_moe_text"

    drafter = EagleDrafterForCausalLM._from_config(eagle_config, use_cache=False)
    assert drafter.config.model_type == "qwen3_5_moe_text"


def test_qwen35_mtp_drafter_forward_outputs_are_finite():
    torch.manual_seed(2)
    config = _make_small_config()
    eagle_config = EagleConfig.from_base_config(config, config.model_type)
    drafter = EagleDrafterForCausalLM._from_config(eagle_config, use_cache=False)
    _init_module_weights(drafter)

    inputs_embeds = torch.randn(2, 4, config.hidden_size)
    hidden_states = torch.randn(2, 4, config.hidden_size)
    position_ids = torch.arange(4).view(1, 4).expand(2, -1)

    outputs = drafter(inputs_embeds, position_ids, hidden_states=hidden_states)

    assert outputs.logits is None
    assert outputs.norm_hidden_state.shape == (2, 4, config.hidden_size)
    assert outputs.last_hidden_state.shape == (2, 4, config.hidden_size)
    assert torch.isfinite(outputs.norm_hidden_state).all()
    assert torch.isfinite(outputs.last_hidden_state).all()


def test_qwen35_mtp_drafter_export_uses_eagle_io_contract():
    config = _make_small_config()
    eagle_config = EagleConfig.from_base_config(config, config.model_type)
    drafter = EagleDrafterForCausalLM._from_config(eagle_config, use_cache=False).eval()

    inputs_embeds = torch.randn(1, 4, config.hidden_size)
    hidden_states = torch.randn(1, 4, config.hidden_size)
    position_ids = torch.arange(4).view(1, 4)

    gm = torch_export_to_gm(
        drafter,
        args=(),
        kwargs={
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
        },
        dynamic_shapes=_dynamic_shapes_for_eagle_draft_export(),
        strict=False,
        num_moe_experts_for_export=2,
    )

    placeholder_names = {node.target for node in gm.graph.nodes if node.op == "placeholder"}
    assert {"inputs_embeds", "position_ids", "hidden_states"}.issubset(placeholder_names)

    with torch.inference_mode():
        outputs = gm(
            inputs_embeds=torch.randn(2, 3, config.hidden_size),
            position_ids=torch.arange(3).view(1, 3).expand(2, -1),
            hidden_states=torch.randn(2, 3, config.hidden_size),
        )

    assert outputs["norm_hidden_state"].shape == (2, 3, config.hidden_size)
    assert outputs["last_hidden_state"].shape == (2, 3, config.hidden_size)
    assert torch.isfinite(outputs["norm_hidden_state"]).all()
    assert torch.isfinite(outputs["last_hidden_state"]).all()


def test_qwen35_mtp_drafter_factory_load_or_random_init_contract(tmp_path):
    config_dir = _write_qwen35_mtp_config(tmp_path)
    factory = EagleDrafterFactory(
        model=str(config_dir),
        skip_loading_weights=True,
        max_seq_len=64,
    )

    drafter = factory.build_model("cpu")
    factory.load_or_random_init(drafter, "cpu")

    assert isinstance(drafter, EagleDrafterForCausalLM)
    assert isinstance(drafter.model.layers[0], Qwen3_5MoeEagleLayer)
    assert drafter.config.load_embedding_from_target is True
    assert drafter.config.load_lm_head_from_target is True
    assert drafter.config.normalize_target_hidden_state is True
    assert all(not parameter.is_meta for parameter in drafter.state_dict().values())


def test_qwen35_mtp_one_model_factory_builds_expected_wrapper(tmp_path):
    config_dir = _write_qwen35_mtp_config(tmp_path)
    factory = _build_qwen35_mtp_one_model_factory(config_dir)

    wrapper = factory.build_model("meta")

    assert isinstance(wrapper, EagleWrapper)
    assert isinstance(wrapper.target_model, Qwen3_5MoeForCausalLM)
    assert isinstance(wrapper.draft_model, EagleDrafterForCausalLM)
    assert isinstance(wrapper.draft_model.model.layers[0], Qwen3_5MoeEagleLayer)
    assert wrapper.max_draft_len == 1
    assert wrapper.load_embedding_from_target is True
    assert wrapper.load_lm_head_from_target is True
    assert wrapper.normalize_target_hidden_state is True


def test_qwen35_mtp_one_model_factory_export_infos_use_eagle_io_contract(tmp_path):
    config_dir = _write_qwen35_mtp_config(tmp_path)
    factory = _build_qwen35_mtp_one_model_factory(config_dir)
    wrapper = factory.build_model("meta")

    export_infos = factory.get_export_infos(wrapper)

    assert len(export_infos) == 2
    assert isinstance(export_infos[0], TargetModelExportInfo)
    assert isinstance(export_infos[1], DraftModelExportInfo)
    assert export_infos[0].submodule_name == "target_model"
    assert set(export_infos[0].dynamic_shape_lookup) == {"inputs_embeds", "position_ids"}
    assert export_infos[1].submodule_name == "draft_model"
    assert set(export_infos[1].dynamic_shape_lookup) == {
        "hidden_states",
        "inputs_embeds",
        "position_ids",
    }


@pytest.mark.parametrize("num_hidden_layers", [1, 3])
def test_qwen35_mtp_target_hidden_state_capture_finds_expected_layers(num_hidden_layers):
    gm, config = _export_qwen35_mtp_target_for_capture(num_hidden_layers)
    transform = DetectHiddenStatesForCapture(
        config=TransformConfig(
            stage="pattern_matcher",
            eagle3_layers_to_capture={-1},
        )
    )

    residual_nodes = transform.collect_residual_add_nodes(gm)

    assert sorted(residual_nodes) == list(range(num_hidden_layers))
    for residual_node in residual_nodes.values():
        assert residual_node.meta["val"].shape[-1] == config.hidden_size

    last_layer = max(residual_nodes)
    expected_arg_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else arg
        for arg in residual_nodes[last_layer].args
    )

    gm, info = transform._apply(gm, None, None, None)
    capture_nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.auto_deploy.residual_add_for_capture.default
    ]

    assert info.num_matches == 1
    assert len(capture_nodes) == 1
    capture_arg_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else arg for arg in capture_nodes[0].args
    )
    assert capture_arg_names == expected_arg_names


def test_qwen35_mtp_target_export_uses_inputs_embeds_as_live_seed():
    gm, _ = _export_qwen35_mtp_target_for_capture(num_hidden_layers=1)

    placeholder_names = {node.target for node in gm.graph.nodes if node.op == "placeholder"}

    assert "inputs_embeds" in placeholder_names
    assert "position_ids" in placeholder_names
    assert "input_ids" not in placeholder_names
    assert not any(
        node.op == "call_module" and "embed_tokens" in str(node.target) for node in gm.graph.nodes
    )


def test_qwen35_mtp_hf_checkpoint_keys_load_strictly():
    config = _make_small_config()
    eagle_config = EagleConfig.from_base_config(config, config.model_type)

    drafter = EagleDrafterForCausalLM._from_config(eagle_config)
    state_dict = {
        _to_hf_mtp_key(key): torch.randn_like(value) for key, value in drafter.state_dict().items()
    }

    factory = object.__new__(AutoModelForCausalLMFactory)
    factory._checkpoint_conversion_mapping = eagle_config._checkpoint_conversion_mapping
    hook = drafter.register_load_state_dict_pre_hook(factory._remap_param_names_load_hook)
    try:
        incompatible_keys = drafter.load_state_dict(state_dict, strict=True)
    finally:
        hook.remove()

    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []
