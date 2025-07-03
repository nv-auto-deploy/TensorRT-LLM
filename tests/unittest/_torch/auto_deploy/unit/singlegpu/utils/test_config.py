"""Test suite for DynamicYamlMixInForSettings utility class."""

import os
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_settings import BaseSettings

from tensorrt_llm._torch.auto_deploy.utils._config import DynamicYamlMixInForSettings


class SimpleModel(BaseModel):
    """Simple model for testing."""

    value: int
    name: str
    flag: bool = False


class OptionModel(BaseModel):
    """Model with literal options."""

    name: str
    option: Literal["on", "off"] = "off"


class BasicSettings(DynamicYamlMixInForSettings, BaseSettings):
    """Basic settings class for testing."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields

    simple: SimpleModel
    option: OptionModel


class SettingsWithDefaultYaml(DynamicYamlMixInForSettings, BaseSettings):
    """Settings class with default yaml file."""

    model_config = ConfigDict(yaml_file="default.yaml", extra="allow")  # Allow extra fields

    simple: SimpleModel
    option: OptionModel


class NestedSettings(DynamicYamlMixInForSettings, BaseSettings):
    """Nested settings class for testing precedence."""

    model_config = ConfigDict(yaml_file="nested_default.yaml", extra="allow")  # Allow extra fields

    args: BasicSettings
    extra_field: str = "default"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def basic_yaml_files(temp_dir):
    """Create basic yaml test files."""
    files = {}

    # Default config
    files["default"] = temp_dir / "default.yaml"
    files["default"].write_text("""
simple:
  value: 100
  name: "default"
  flag: true
option:
  name: "default_option"
  option: "on"
""")

    # Override config 1
    files["config1"] = temp_dir / "config1.yaml"
    files["config1"].write_text("""
simple:
  value: 200
  name: "config1"
option:
  name: "config1_option"
""")

    # Override config 2
    files["config2"] = temp_dir / "config2.yaml"
    files["config2"].write_text("""
simple:
  flag: false
  name: "config2"
option:
  option: "off"
""")

    # Partial config
    files["partial"] = temp_dir / "partial.yaml"
    files["partial"].write_text("""
simple:
  value: 999
""")

    return files


@pytest.fixture
def nested_yaml_files(temp_dir):
    """Create nested yaml test files."""
    files = {}

    # Nested default
    files["nested_default"] = temp_dir / "nested_default.yaml"
    files["nested_default"].write_text("""
args:
  simple:
    value: 50
    name: "nested_default"
    flag: true
  option:
    name: "nested_default_option"
    option: "on"
extra_field: "nested_default_extra"
""")

    # Nested override 1
    files["nested_override1"] = temp_dir / "nested_override1.yaml"
    files["nested_override1"].write_text("""
args:
  simple:
    value: 150
    name: "nested_override1"
  option:
    name: "nested_override1_option"
extra_field: "nested_override1_extra"
""")

    # Nested override 2
    files["nested_override2"] = temp_dir / "nested_override2.yaml"
    files["nested_override2"].write_text("""
args:
  simple:
    flag: false
    name: "nested_override2"
  option:
    option: "off"
""")

    # Inner config (for args.yaml_configs)
    files["inner_config"] = temp_dir / "inner_config.yaml"
    files["inner_config"].write_text("""
simple:
  value: 300
  name: "inner_config"
option:
  name: "inner_config_option"
  option: "on"
""")

    return files


class TestBasicYamlLoading:
    """Test basic YAML loading functionality."""

    def test_no_yaml_configs(self, temp_dir):
        """Test settings without any yaml configs."""
        with pytest.raises(ValidationError):
            # Should fail because required fields are missing
            BasicSettings()

    def test_single_yaml_config(self, basic_yaml_files, temp_dir):
        """Test loading a single yaml config file."""
        os.chdir(temp_dir)

        settings = BasicSettings(yaml_configs=[basic_yaml_files["config1"]])

        assert settings.simple.value == 200
        assert settings.simple.name == "config1"
        assert settings.simple.flag is False  # default value
        assert settings.option.name == "config1_option"
        assert settings.option.option == "off"  # default value

    def test_multiple_yaml_configs_merging(self, basic_yaml_files, temp_dir):
        """Test merging multiple yaml configs in order."""
        os.chdir(temp_dir)

        # Order: config1, config2 (config2 should override config1)
        settings = BasicSettings(
            yaml_configs=[basic_yaml_files["config1"], basic_yaml_files["config2"]]
        )

        assert settings.simple.value == 200  # from config1
        assert settings.simple.name == "config2"  # overridden by config2
        assert settings.simple.flag is False  # from config2
        assert settings.option.name == "config1_option"  # from config1
        assert settings.option.option == "off"  # from config2

    def test_partial_yaml_config(self, basic_yaml_files, temp_dir):
        """Test partial yaml config with some missing fields."""
        os.chdir(temp_dir)

        with pytest.raises(ValidationError):
            # Should fail because 'name' is missing from simple
            BasicSettings(yaml_configs=[basic_yaml_files["partial"]])


class TestDefaultYamlFile:
    """Test settings with default yaml file specified."""

    def test_default_yaml_file_loading(self, basic_yaml_files, temp_dir):
        """Test loading default yaml file from model_config."""
        os.chdir(temp_dir)

        settings = SettingsWithDefaultYaml()

        assert settings.simple.value == 100
        assert settings.simple.name == "default"
        assert settings.simple.flag is True
        assert settings.option.name == "default_option"
        assert settings.option.option == "on"

    def test_default_yaml_with_additional_configs(self, basic_yaml_files, temp_dir):
        """Test default yaml file with additional configs."""
        os.chdir(temp_dir)

        settings = SettingsWithDefaultYaml(yaml_configs=[basic_yaml_files["config1"]])

        # Additional configs should override default
        assert settings.simple.value == 200  # from config1
        assert settings.simple.name == "config1"  # from config1
        assert settings.simple.flag is True  # from default
        assert settings.option.name == "config1_option"  # from config1
        assert settings.option.option == "on"  # from default

    def test_multiple_additional_configs_with_default(self, basic_yaml_files, temp_dir):
        """Test multiple additional configs with default yaml file."""
        os.chdir(temp_dir)

        settings = SettingsWithDefaultYaml(
            yaml_configs=[basic_yaml_files["config1"], basic_yaml_files["config2"]]
        )

        # Order: default.yaml, config1.yaml, config2.yaml
        assert settings.simple.value == 200  # from config1
        assert settings.simple.name == "config2"  # from config2 (last override)
        assert settings.simple.flag is False  # from config2
        assert settings.option.name == "config1_option"  # from config1
        assert settings.option.option == "off"  # from config2


class TestNestedSettings:
    """Test nested settings with yaml configs."""

    def test_nested_default_yaml(self, nested_yaml_files, temp_dir):
        """Test nested settings with default yaml file."""
        os.chdir(temp_dir)

        settings = NestedSettings()

        assert settings.args.simple.value == 50
        assert settings.args.simple.name == "nested_default"
        assert settings.args.simple.flag is True
        assert settings.args.option.name == "nested_default_option"
        assert settings.args.option.option == "on"
        assert settings.extra_field == "nested_default_extra"

    def test_nested_with_outer_yaml_configs(self, nested_yaml_files, temp_dir):
        """Test nested settings with yaml configs at outer level."""
        os.chdir(temp_dir)

        settings = NestedSettings(yaml_configs=[nested_yaml_files["nested_override1"]])

        # Outer config should override inner defaults
        assert settings.args.simple.value == 150
        assert settings.args.simple.name == "nested_override1"
        assert settings.args.simple.flag is True  # from default
        assert settings.args.option.name == "nested_override1_option"
        assert settings.args.option.option == "on"  # from default
        assert settings.extra_field == "nested_override1_extra"

    def test_nested_with_inner_yaml_configs(self, nested_yaml_files, temp_dir):
        """Test nested settings with yaml configs at inner level."""
        os.chdir(temp_dir)

        # Create nested settings with inner yaml configs
        settings = NestedSettings(
            args=BasicSettings(yaml_configs=[nested_yaml_files["inner_config"]])
        )

        # Inner yaml configs should be processed
        assert settings.args.simple.value == 300
        assert settings.args.simple.name == "inner_config"
        assert settings.args.simple.flag is False  # default
        assert settings.args.option.name == "inner_config_option"
        assert settings.args.option.option == "on"
        assert settings.extra_field == "nested_default_extra"  # from outer default

    def test_nested_precedence_outer_over_inner(self, nested_yaml_files, temp_dir):
        """Test precedence: outer yaml configs override inner yaml configs."""
        os.chdir(temp_dir)

        # Both outer and inner yaml configs
        settings = NestedSettings(
            yaml_configs=[nested_yaml_files["nested_override1"]],
            args=BasicSettings(yaml_configs=[nested_yaml_files["inner_config"]]),
        )

        # Outer should take precedence over inner
        assert settings.args.simple.value == 150  # from outer (nested_override1)
        assert settings.args.simple.name == "nested_override1"  # from outer
        assert settings.args.simple.flag is True  # from outer default
        assert settings.args.option.name == "nested_override1_option"  # from outer
        assert settings.args.option.option == "on"  # from outer default
        assert settings.extra_field == "nested_override1_extra"


class TestPrecedenceOrder:
    """Test precedence order of different setting sources."""

    def test_init_overrides_yaml(self, basic_yaml_files, temp_dir):
        """Test that init values override yaml configs."""
        os.chdir(temp_dir)

        init_simple = SimpleModel(value=999, name="init_value", flag=True)
        init_option = OptionModel(name="init_option", option="on")

        settings = BasicSettings(
            simple=init_simple, option=init_option, yaml_configs=[basic_yaml_files["config1"]]
        )

        # Init values should override yaml
        assert settings.simple.value == 999
        assert settings.simple.name == "init_value"
        assert settings.simple.flag is True
        assert settings.option.name == "init_option"
        assert settings.option.option == "on"

    def test_env_overrides_yaml(self, basic_yaml_files, temp_dir):
        """Test that environment variables override yaml configs."""
        os.chdir(temp_dir)

        with patch.dict(
            os.environ,
            {"SIMPLE__VALUE": "888", "SIMPLE__NAME": "env_value", "OPTION__NAME": "env_option"},
        ):
            settings = BasicSettings(yaml_configs=[basic_yaml_files["config1"]])

            # Environment should override yaml
            assert settings.simple.value == 888
            assert settings.simple.name == "env_value"
            assert settings.simple.flag is False  # from yaml (no env override)
            assert settings.option.name == "env_option"
            assert settings.option.option == "off"  # from yaml default

    def test_partial_env_override(self, basic_yaml_files, temp_dir):
        """Test partial environment variable override."""
        os.chdir(temp_dir)

        with patch.dict(os.environ, {"SIMPLE__FLAG": "true", "OPTION__OPTION": "on"}):
            settings = BasicSettings(yaml_configs=[basic_yaml_files["config1"]])

            # Mix of env and yaml values
            assert settings.simple.value == 200  # from yaml
            assert settings.simple.name == "config1"  # from yaml
            assert settings.simple.flag is True  # from env
            assert settings.option.name == "config1_option"  # from yaml
            assert settings.option.option == "on"  # from env


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_missing_yaml_file(self, temp_dir):
        """Test handling of missing yaml file."""
        os.chdir(temp_dir)

        missing_file = temp_dir / "missing.yaml"

        # Should not raise error for missing file (gracefully ignored)
        with pytest.raises(ValidationError):
            # But should still fail validation for missing required fields
            BasicSettings(yaml_configs=[missing_file])

    def test_invalid_yaml_syntax(self, temp_dir):
        """Test handling of invalid yaml syntax."""
        os.chdir(temp_dir)

        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("""
simple:
  value: 100
  name: "test"
  flag: true
option:
  name: "test_option"
  option: invalid_option  # This should cause validation error
""")

        with pytest.raises(ValidationError):
            BasicSettings(yaml_configs=[invalid_yaml])

    def test_malformed_yaml_file(self, temp_dir):
        """Test handling of malformed yaml file."""
        os.chdir(temp_dir)

        malformed_yaml = temp_dir / "malformed.yaml"
        malformed_yaml.write_text("""
simple:
  value: 100
  name: "test"
  flag: true
option:
  name: "test_option"
  option: "on"
  invalid_structure: {
    missing_close_brace: "value"
""")

        with pytest.raises(Exception):  # Should raise yaml parsing error
            BasicSettings(yaml_configs=[malformed_yaml])


class TestDeepMerging:
    """Test deep merging functionality."""

    def test_deep_merge_nested_dicts(self, temp_dir):
        """Test deep merging of nested dictionaries."""
        os.chdir(temp_dir)

        base_yaml = temp_dir / "base.yaml"
        base_yaml.write_text("""
simple:
  value: 100
  name: "base"
  flag: true
option:
  name: "base_option"
  option: "on"
""")

        override_yaml = temp_dir / "override.yaml"
        override_yaml.write_text("""
simple:
  value: 200
  # name should remain from base
  # flag should remain from base
option:
  option: "off"
  # name should remain from base
""")

        settings = BasicSettings(yaml_configs=[base_yaml, override_yaml])

        # Deep merge should preserve non-overridden values
        assert settings.simple.value == 200  # overridden
        assert settings.simple.name == "base"  # preserved
        assert settings.simple.flag is True  # preserved
        assert settings.option.name == "base_option"  # preserved
        assert settings.option.option == "off"  # overridden

    def test_complex_deep_merge_order(self, temp_dir):
        """Test complex deep merge with multiple files."""
        os.chdir(temp_dir)

        # Create three files with overlapping but different fields
        yaml1 = temp_dir / "yaml1.yaml"
        yaml1.write_text("""
simple:
  value: 100
  name: "yaml1"
  flag: true
option:
  name: "yaml1_option"
  option: "on"
""")

        yaml2 = temp_dir / "yaml2.yaml"
        yaml2.write_text("""
simple:
  value: 200
  name: "yaml2"
  # flag not specified, should remain from yaml1
option:
  name: "yaml2_option"
  # option not specified, should remain from yaml1
""")

        yaml3 = temp_dir / "yaml3.yaml"
        yaml3.write_text("""
simple:
  # value not specified, should remain from yaml2
  # name not specified, should remain from yaml2
  flag: false
option:
  # name not specified, should remain from yaml2
  option: "off"
""")

        settings = BasicSettings(yaml_configs=[yaml1, yaml2, yaml3])

        # Final result should be deep merge of all three
        assert settings.simple.value == 200  # from yaml2
        assert settings.simple.name == "yaml2"  # from yaml2
        assert settings.simple.flag is False  # from yaml3
        assert settings.option.name == "yaml2_option"  # from yaml2
        assert settings.option.option == "off"  # from yaml3


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_cli_like_usage(self, temp_dir):
        """Test CLI-like usage with multiple config levels."""
        os.chdir(temp_dir)

        # Create a realistic scenario with default config and user overrides
        default_config = temp_dir / "default.yaml"
        default_config.write_text("""
simple:
  value: 42
  name: "default_model"
  flag: false
option:
  name: "default_option"
  option: "off"
""")

        user_config = temp_dir / "user.yaml"
        user_config.write_text("""
simple:
  value: 100
  flag: true
option:
  option: "on"
""")

        experiment_config = temp_dir / "experiment.yaml"
        experiment_config.write_text("""
simple:
  value: 999
  name: "experiment_model"
""")

        # Simulate CLI usage: default + user + experiment configs
        settings = SettingsWithDefaultYaml(yaml_configs=[user_config, experiment_config])

        # Should have proper precedence
        assert settings.simple.value == 999  # from experiment (highest priority)
        assert settings.simple.name == "experiment_model"  # from experiment
        assert settings.simple.flag is True  # from user
        assert settings.option.name == "default_option"  # from default
        assert settings.option.option == "on"  # from user

    def test_empty_yaml_configs_list(self, basic_yaml_files, temp_dir):
        """Test with empty yaml_configs list."""
        os.chdir(temp_dir)

        # Should behave same as no yaml_configs
        with pytest.raises(ValidationError):
            BasicSettings(yaml_configs=[])

    def test_relative_and_absolute_paths(self, basic_yaml_files, temp_dir):
        """Test with both relative and absolute paths."""
        os.chdir(temp_dir)

        # Mix of relative and absolute paths
        relative_path = "config1.yaml"
        basic_yaml_files["config1"].rename(temp_dir / relative_path)

        settings = BasicSettings(
            yaml_configs=[
                relative_path,  # relative
                basic_yaml_files["config2"],  # absolute
            ]
        )

        # Should work with both path types
        assert settings.simple.value == 200  # from config1
        assert settings.simple.name == "config2"  # from config2
