# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main
from test_common.llm_data import hf_id_to_local_model_dir

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_eagle import EagleWrapper
from tensorrt_llm._torch.auto_deploy.models.eagle import (
    DraftModelExportInfo,
    EagleOneModelFactory,
    TargetModelExportInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.hidden_states import (
    DetectHiddenStatesForCapture,
)
from tensorrt_llm._torch.speculative import get_num_extra_kv_tokens
from tensorrt_llm.llmapi import Eagle3DecodingConfig, MTPDecodingConfig


def get_extra_seq_len_for_kv_cache(llm_args) -> int:
    """Mirror the current extra-KV sizing logic used by the runtime."""
    extra = 0
    spec_config = llm_args.speculative_config
    if not llm_args.disable_overlap_scheduler:
        extra += 1
        if spec_config is not None:
            extra += spec_config.tokens_per_gen_step - 1

    if spec_config is not None:
        extra += spec_config.tokens_per_gen_step - 1
        extra += get_num_extra_kv_tokens(spec_config)

    return extra


def _make_graph_module_with_placeholders(*names):
    graph = torch.fx.Graph()
    output = None
    for name in names:
        placeholder = graph.placeholder(name)
        if output is None:
            output = placeholder
    graph.output(output)
    return torch.fx.GraphModule(nn.Module(), graph)


def test_eagle_wrapper_filters_kwargs_for_direct_graph_module():
    gm = _make_graph_module_with_placeholders("inputs_embeds", "position_ids")
    kwargs = {
        "inputs_embeds": torch.empty(1),
        "position_ids": torch.empty(1),
        "not_in_graph": torch.empty(1),
    }

    filtered = EagleWrapper._filter_kwargs_for_submodule(kwargs, gm)

    assert set(filtered) == {"inputs_embeds", "position_ids"}
    for name, value in filtered.items():
        assert value is kwargs[name]


def test_eagle_wrapper_filters_kwargs_using_nested_target_graph():
    inner_gm = _make_graph_module_with_placeholders(
        "inputs_embeds",
        "position_ids",
        "layer_0_hidden_states_cache",
        "kv_cache",
    )

    eager_target = nn.Module()
    eager_target.model = nn.Module()
    eager_target.model.language_model = inner_gm
    kwargs = {
        "inputs_embeds": torch.empty(1),
        "position_ids": torch.empty(1),
        "layer_0_hidden_states_cache": torch.empty(1),
        "kv_cache": torch.empty(1),
        "not_in_graph": torch.empty(1),
    }

    filtered = EagleWrapper._filter_kwargs_for_submodule(kwargs, eager_target)

    assert set(filtered) == {
        "inputs_embeds",
        "position_ids",
        "layer_0_hidden_states_cache",
        "kv_cache",
    }
    for name, value in filtered.items():
        assert value is kwargs[name]


def test_eagle_wrapper_filters_kwargs_rejects_ambiguous_graph_modules():
    eager_target = nn.Module()
    eager_target.first = _make_graph_module_with_placeholders("inputs_embeds")
    eager_target.second = _make_graph_module_with_placeholders("position_ids")

    with pytest.raises(ValueError, match="unwrapping is ambiguous"):
        EagleWrapper._filter_kwargs_for_submodule({"inputs_embeds": torch.empty(1)}, eager_target)


def test_eagle_one_model_factory_export_infos_use_eagle_io_contract(monkeypatch):
    factory = EagleOneModelFactory(
        model="test-model",
        skip_loading_weights=True,
        max_seq_len=64,
        speculative_config=MTPDecodingConfig(
            max_draft_len=1,
            mtp_eagle_one_model=True,
            speculative_model="test-model",
        ),
    )
    draft_model = nn.Module()
    draft_model.config = SimpleNamespace(
        load_embedding_from_target=True,
        load_lm_head_from_target=True,
    )
    wrapper = SimpleNamespace(target_model=nn.Module(), draft_model=draft_model)
    monkeypatch.setattr(factory.target_factory, "get_export_infos", lambda _model: [])

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


def test_super_mtp_smoke():
    """Test one-model MTP/Eagle runtime with a tiny Nemotron SuperV3 target."""
    test_prompt = "What is the capital of France?"
    model_hub_id = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
    model_path = hf_id_to_local_model_dir(model_hub_id)

    experiment_config = get_small_model_config(
        model_hub_id,
        transforms={
            "insert_cached_causal_conv": {"backend": "triton_causal_conv"},
            "insert_cached_ssm_attention": {"backend": "triton_ssm"},
        },
    )
    experiment_config["args"]["model"] = model_path
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["speculative_config"] = MTPDecodingConfig(
        num_nextn_predict_layers=3,
        mtp_eagle_one_model=True,
        speculative_model=model_path,
    )
    # Shrink the Eagle/MTP drafter model to match the target's reduced dimensions.
    experiment_config["args"]["speculative_model_kwargs"] = experiment_config["args"][
        "model_kwargs"
    ]
    # NOTE: trtllm attention backend fails on B200 (likely illegal memory access); use flashinfer.
    experiment_config["args"]["attn_backend"] = "flashinfer"
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["compile_backend"] = "torch-simple"
    experiment_config["args"]["max_num_tokens"] = 256
    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    cfg = ExperimentConfig(**experiment_config)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 64,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    results = main(cfg)

    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1


def test_qwen3_5_moe_mtp_smoke():
    """Test one-model MTP/Eagle runtime with a tiny Qwen3.5 MoE VLM target."""
    test_prompt = "What is the capital of France?"
    model_hub_id = "Qwen/Qwen3.5-35B-A3B"
    model_path = hf_id_to_local_model_dir(model_hub_id)

    experiment_config = get_small_model_config(
        model_hub_id,
        transforms={
            "insert_cached_causal_conv": {"backend": "triton_causal_conv"},
            "initialize_mrope_delta_cache": {"enabled": True},
        },
    )
    experiment_config["args"]["model"] = model_path
    experiment_config["args"]["tokenizer"] = model_path
    experiment_config["args"]["runtime"] = "trtllm"
    experiment_config["args"]["world_size"] = 1
    experiment_config["args"]["model_factory"] = "Qwen3_5MoeForConditionalGeneration"
    experiment_config["args"]["speculative_config"] = MTPDecodingConfig(
        max_draft_len=1,
        mtp_eagle_one_model=True,
        speculative_model=model_path,
    )
    experiment_config["args"]["speculative_model_kwargs"] = experiment_config["args"][
        "model_kwargs"
    ]
    experiment_config["args"]["attn_backend"] = "flashinfer"
    experiment_config["args"]["disable_overlap_scheduler"] = True
    experiment_config["args"]["compile_backend"] = "torch-simple"
    experiment_config["args"]["max_num_tokens"] = 256
    experiment_config["prompt"]["batch_size"] = 1
    experiment_config["prompt"]["queries"] = test_prompt

    cfg = ExperimentConfig(**experiment_config)
    cfg.prompt.sp_kwargs = {
        "max_tokens": 64,
        "top_k": None,
        "temperature": 0.0,
        "seed": 42,
    }

    results = main(cfg)

    prompts_and_outputs = results["prompts_and_outputs"]
    assert len(prompts_and_outputs) == 1


def test_kv_cache_extra_seq_len_for_spec_dec():
    """Test that get_extra_seq_len_for_kv_cache computes correct extra capacity."""
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    # Case 1: No spec config, no overlap
    args_no_spec = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        disable_overlap_scheduler=True,
    )
    assert get_extra_seq_len_for_kv_cache(args_no_spec) == 0

    # Case 2: No spec config, with overlap
    args_overlap = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        disable_overlap_scheduler=False,
    )
    assert get_extra_seq_len_for_kv_cache(args_overlap) == 1  # overlap adds +1

    # Case 3: Eagle3 one-model, overlap disabled
    spec_config = Eagle3DecodingConfig(
        max_draft_len=3,
        speculative_model="some/model",
        eagle3_one_model=True,
    )
    args_eagle = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        speculative_config=spec_config,
        disable_overlap_scheduler=True,
    )
    extra = get_extra_seq_len_for_kv_cache(args_eagle)
    # Should include max_total_draft_tokens + get_num_extra_kv_tokens (max_draft_len - 1)
    assert extra > 0
    assert extra == spec_config.max_total_draft_tokens + (spec_config.max_draft_len - 1)

    # Case 4: Eagle3 one-model, overlap enabled
    args_eagle_overlap = LlmArgs(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        speculative_config=spec_config,
        disable_overlap_scheduler=False,
    )
    extra_overlap = get_extra_seq_len_for_kv_cache(args_eagle_overlap)
    # Should be more than without overlap
    assert extra_overlap > extra


def test_mtp_autodeploy_uses_eagle_one_model_capture():
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    args = LlmArgs(
        model=model,
        speculative_config=MTPDecodingConfig(
            num_nextn_predict_layers=3,
            mtp_eagle_one_model=True,
        ),
    )

    assert isinstance(args.speculative_config, MTPDecodingConfig)
    assert args.model_factory == "eagle_one_model"
    assert args.target_model_factory == "AutoModelForCausalLM"
    assert args.transforms["detect_hidden_states_for_capture"]["enabled"] is True
    assert args.transforms["detect_hidden_states_for_capture"]["eagle3_layers_to_capture"] == {-1}


def test_detect_hidden_states_capture_last_layer_for_mtp_eagle_one_model():
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    config = get_small_model_config("meta-llama/Meta-Llama-3.1-8B-Instruct")

    args = LlmArgs(
        **config["args"],
        speculative_config=MTPDecodingConfig(
            num_nextn_predict_layers=3,
            mtp_eagle_one_model=True,
            speculative_model=config["args"]["model"],
        ),
    )

    factory = args.create_factory()
    assert isinstance(factory, EagleOneModelFactory)

    model = factory.target_factory.build_model("meta")
    input_ids = torch.ones((1, 8), dtype=torch.int64)
    position_ids = torch.arange(8, dtype=torch.int64).unsqueeze(0)
    gm = torch_export_to_gm(
        model,
        args=(input_ids, position_ids),
    )

    transform = DetectHiddenStatesForCapture(
        config=TransformConfig(
            stage="pattern_matcher",
            eagle3_layers_to_capture={-1},
        )
    )

    original_residual_nodes = transform.collect_residual_add_nodes(gm)
    assert original_residual_nodes
    last_layer = max(original_residual_nodes)
    last_layer_residual = original_residual_nodes[last_layer]
    expected_arg_names = tuple(
        arg.name if isinstance(arg, torch.fx.Node) else arg for arg in last_layer_residual.args
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
