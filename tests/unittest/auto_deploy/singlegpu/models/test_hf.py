import copy
import json
from unittest.mock import MagicMock, patch

import pytest
import safetensors.torch
import torch
import torch.nn as nn
from accelerate.utils import modeling
from transformers import AutoModelForCausalLM
from transformers.models.llama4.configuration_llama4 import Llama4Config

from tensorrt_llm._torch.auto_deploy.models.hf import (
    AutoModelForCausalLMFactory,
    hf_load_state_dict_with_device,
)
from tensorrt_llm._torch.auto_deploy.models.quant_config_reader import HFQuantConfigReader


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


@pytest.fixture(autouse=True)
def restore_custom_model_mapping():
    old_mapping = copy.copy(AutoModelForCausalLMFactory._custom_model_mapping)
    yield
    AutoModelForCausalLMFactory._custom_model_mapping = old_mapping


def test_hf_load_state_dict_with_device():
    """Test that hf_load_state_dict_with_device correctly patches modeling.load_state_dict."""
    # Create mock for original load_state_dict
    original_load_state_dict = MagicMock()

    # Test with CPU device
    with patch.object(modeling, "load_state_dict", original_load_state_dict):
        with hf_load_state_dict_with_device(device="cpu"):
            # Call the patched function
            modeling.load_state_dict("dummy_checkpoint")

            # Check that device was set correctly
            original_load_state_dict.assert_called_once_with(
                "dummy_checkpoint", device_map={"": "cpu"}
            )

            # Reset mock for next test
            original_load_state_dict.reset_mock()

        # Check that original behavior is restored
        modeling.load_state_dict("dummy_checkpoint", device_map="original_device_map")
        original_load_state_dict.assert_called_once_with(
            "dummy_checkpoint", device_map="original_device_map"
        )
        original_load_state_dict.reset_mock()

    # Test with CUDA device (if available)
    if torch.cuda.is_available():
        with patch.object(modeling, "load_state_dict", original_load_state_dict):
            with hf_load_state_dict_with_device(device="cuda"):
                # Call the patched function
                modeling.load_state_dict("dummy_checkpoint")

                # Check that device was set correctly
                original_load_state_dict.assert_called_once_with(
                    "dummy_checkpoint", device_map={"": "cuda"}
                )


def test_hf_load_state_dict_with_device_streams_expected_safetensors_keys(tmp_path):
    class RenamedStateDictModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(2, 2))

    def rename_weight_hook(module, state_dict, prefix, local_metadata):
        state_dict[prefix + "checkpoint_weight"] = state_dict.pop(prefix + "weight")

    model = RenamedStateDictModel()
    model.register_state_dict_post_hook(rename_weight_hook)
    checkpoint_path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(
        {
            "checkpoint_weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "unused_expert_weight": torch.ones(2, 2),
        },
        checkpoint_path,
    )

    with hf_load_state_dict_with_device(device="cpu", model=model):
        loaded = modeling.load_state_dict(str(checkpoint_path))

    assert set(loaded) == {"checkpoint_weight"}
    assert loaded["checkpoint_weight"].device.type == "cpu"
    torch.testing.assert_close(
        loaded["checkpoint_weight"],
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    )


def test_hf_quant_config_reader_normalizes_ignore_aliases():
    reader = HFQuantConfigReader()

    reader.read_config(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "ignored_layers": ["model.layers.0.self_attn.o_proj"],
                "modules_to_not_convert": ["model.layers.1.self_attn.o_proj"],
                "exclude_modules": ["model.layers.2.self_attn.o_proj"],
            }
        }
    )

    qconfig = reader.get_config()
    expected_excludes = [
        "model.layers.2.self_attn.o_proj",
        "model.layers.1.self_attn.o_proj",
        "model.layers.0.self_attn.o_proj",
        "lm_head",
        "model.embed_tokens",
    ]
    assert qconfig["exclude_modules"] == expected_excludes
    assert qconfig["modules_to_not_convert"] == expected_excludes


@pytest.fixture
def mock_factory():
    with (
        patch.object(AutoModelForCausalLMFactory, "prefetch_checkpoint"),
        patch.object(AutoModelForCausalLMFactory, "_load_quantization_config"),
    ):
        # Create factory instance with mocked methods to avoid HTTP requests
        factory = AutoModelForCausalLMFactory(model="dummy_model")
        # Set model path directly to avoid prefetch
        factory._prefetched_model_path = "/dummy/path"
        yield factory


def test_recursive_update_config(mock_factory):
    """Test that _recursive_update_config correctly updates a config object recursively."""
    # Get the mocked factory instance
    factory = mock_factory

    # Create a Llama4Config instance
    config = Llama4Config()

    # Create an update dictionary with both simple and nested values
    update_dict = {
        "bos_token_id": 42,  # Simple value at root level
        "text_config": {  # Nested config update
            "hidden_size": 4096,
            "num_attention_heads": 32,
        },
        "vision_config": {  # Another nested config update
            "hidden_size": 1024,
            "image_size": 224,
        },
        "non_existent_key": "this should be ignored",  # This key doesn't exist in the config
    }

    # Apply the recursive update
    updated_config, nested_unused = factory._recursive_update_config(config, update_dict)

    # Check that it returns the same object
    assert updated_config is config

    # Check root level updates
    assert config.bos_token_id == 42

    # Check nested updates in text_config
    assert config.text_config.hidden_size == 4096
    assert config.text_config.num_attention_heads == 32

    # Check nested updates in vision_config
    assert config.vision_config.hidden_size == 1024
    assert config.vision_config.image_size == 224

    # Check that non-existent keys were ignored
    assert not hasattr(config, "non_existent_key")
    # Check that nested_unused contains the non-existent key
    assert nested_unused == {"non_existent_key": "this should be ignored"}

    # Create a more complex update with deeper nesting
    complex_update = {"text_config": {"rope_scaling": {"factor": 2.0, "type": "linear"}}}

    # Apply the recursive update again
    factory._recursive_update_config(config, complex_update)

    # Check that complex nested updates were applied correctly
    assert config.text_config.rope_scaling["factor"] == 2.0
    assert config.text_config.rope_scaling["type"] == "linear"


def test_get_checkpoint_file_repairs_stale_safetensors_index(mock_factory, tmp_path):
    shard_path = tmp_path / "actual_shard.safetensors"
    safetensors.torch.save_file({"linear.weight": torch.ones(2, 2)}, shard_path)
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"linear.weight": "missing.safetensors"},
            }
        )
    )

    repaired_index_path = mock_factory._get_checkpoint_file(tmp_path)

    assert repaired_index_path == str(tmp_path / "model.safetensors.auto_deploy.index.json")
    with open(repaired_index_path) as f:
        repaired_index = json.load(f)
    assert repaired_index["weight_map"]["linear.weight"] == "actual_shard.safetensors"


def test_get_checkpoint_file_reuses_valid_generated_safetensors_index(mock_factory, tmp_path):
    shard_path = tmp_path / "actual_shard.safetensors"
    safetensors.torch.save_file({"linear.weight": torch.ones(2, 2)}, shard_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"linear.weight": "missing.safetensors"},
            }
        )
    )
    generated_index_path = tmp_path / "model.safetensors.auto_deploy.index.json"
    generated_index_path.write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"linear.weight": "actual_shard.safetensors"},
            }
        )
    )

    repaired_index_path = mock_factory._get_checkpoint_file(tmp_path)

    assert repaired_index_path == str(generated_index_path)


def test_register_custom_model_cls():
    config_cls_name = "FooConfig"
    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=config_cls_name, custom_model_cls=custom_model_cls
    )

    assert AutoModelForCausalLMFactory._custom_model_mapping[config_cls_name] == custom_model_cls


class MyError(Exception):
    pass


# Needed for `type(config)` calls.
class FooConfig:
    pass


def test_build_model_raises_when_custom_model_cls_does_not_have_from_config(mock_factory):
    custom_model_cls = MagicMock(spec=AutoModelForCausalLM, __name__="FooModel")
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    with (
        patch.object(
            AutoModelForCausalLMFactory,
            "_get_model_config",
            return_value=(FooConfig(), {}),
        ),
        pytest.raises(ValueError, match=r"from_config"),
    ):
        mock_factory.build_model(device="meta")


def test_build_model_uses_custom_model_cls_from_config(mock_factory):
    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    custom_model_cls.configure_mock(_from_config=MagicMock(side_effect=MyError))
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    with (
        patch.object(
            AutoModelForCausalLMFactory,
            "_get_model_config",
            return_value=(FooConfig(), {}),
        ),
        pytest.raises(MyError),
    ):
        mock_factory.build_model(device="meta")


def test_custom_model_mapping_in_parent_does_not_affect_children():
    class Child(AutoModelForCausalLMFactory):
        pass

    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    custom_model_cls.configure_mock(_from_config=MagicMock(side_effect=MyError))
    AutoModelForCausalLMFactory.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    assert Child._custom_model_mapping == {}


def test_custom_model_mapping_in_parent_does_not_affect_parent():
    class Child(AutoModelForCausalLMFactory):
        pass

    parent_mapping = copy.copy(AutoModelForCausalLMFactory._custom_model_mapping)

    custom_model_cls = MagicMock(spec=AutoModelForCausalLM)
    custom_model_cls.configure_mock(_from_config=MagicMock(side_effect=MyError))
    Child.register_custom_model_cls(
        config_cls_name=FooConfig.__name__, custom_model_cls=custom_model_cls
    )

    assert AutoModelForCausalLMFactory._custom_model_mapping == parent_mapping
