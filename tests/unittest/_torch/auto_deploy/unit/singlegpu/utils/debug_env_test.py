import os
import tempfile
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings

from tensorrt_llm._torch.auto_deploy.utils._config import DynamicYamlMixInForSettings


class SimpleModel(BaseModel):
    value: int
    name: str
    flag: bool = False


class OptionModel(BaseModel):
    name: str
    option: Literal["on", "off"] = "off"


class BasicSettings(DynamicYamlMixInForSettings, BaseSettings):
    model_config = ConfigDict(extra="allow")
    simple: SimpleModel
    option: OptionModel


def test_env_precedence():
    temp_dir = Path(tempfile.mkdtemp())
    os.chdir(temp_dir)

    config_yaml = temp_dir / "config1.yaml"
    config_yaml.write_text("""
simple:
  value: 200
  name: "config1"
option:
  name: "config1_option"
""")

    print("Testing environment variable precedence...")

    # Test normal pydantic settings without yaml_configs
    print("\n1. Testing BaseSettings without mixin:")

    class PlainSettings(BaseSettings):
        simple: SimpleModel
        option: OptionModel

    with patch.dict(os.environ, {"SIMPLE__VALUE": "888", "SIMPLE__NAME": "env_value"}):
        try:
            plain_settings = PlainSettings()
            print(f"   simple.value = {plain_settings.simple.value}")
            print(f"   simple.name = {plain_settings.simple.name}")
        except Exception as e:
            print(f"   Expected error: {e}")

    # Test with our mixin and yaml configs
    print("\n2. Testing with mixin and yaml configs:")
    with patch.dict(os.environ, {"SIMPLE__VALUE": "888", "SIMPLE__NAME": "env_value"}):
        settings = BasicSettings(yaml_configs=[config_yaml])
        print(f"   simple.value = {settings.simple.value} (should be 888 from env)")
        print(f"   simple.name = {settings.simple.name} (should be env_value from env)")
        print(f"   option.name = {settings.option.name} (should be config1_option from yaml)")

    # Test just yaml configs without env vars
    print("\n3. Testing just yaml configs:")
    settings = BasicSettings(yaml_configs=[config_yaml])
    print(f"   simple.value = {settings.simple.value} (should be 200 from yaml)")
    print(f"   simple.name = {settings.simple.name} (should be config1 from yaml)")


if __name__ == "__main__":
    test_env_precedence()
