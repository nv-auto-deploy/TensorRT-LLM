"""Test deferred YAML settings loading in pydantic's settings."""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, CliApp

from tensorrt_llm._torch.auto_deploy.utils._config import \
    DynamicYamlMixInForSettings


class Foo(BaseModel):
    """Foo model."""

    value: int
    name: str
    flag: bool


class Bar(BaseModel):
    """Bar model."""

    name: str
    option: Literal["on", "off"]


class MySettings(DynamicYamlMixInForSettings, BaseSettings):
    """Main settings class of the CLI app."""

    model_config = ConfigDict(yaml_file="config.yaml", )

    foo: Foo
    bar: Bar

    def cli_cmd(self) -> None:
        """Run the subcommand."""
        print(repr(self))


class NestedCLI(DynamicYamlMixInForSettings, BaseSettings):
    """Main settings class of the CLI app."""

    model_config = ConfigDict(
        cli_parse_args=True,
        yaml_file="config_nested.yaml",
        cli_kebab_case=True,
    )

    args: MySettings

    def cli_cmd(self) -> None:
        """Run the subcommand."""
        print(repr(self))


def main() -> None:
    """Run the CLI app."""
    CliApp.run(NestedCLI)


if __name__ == "__main__":
    main()
