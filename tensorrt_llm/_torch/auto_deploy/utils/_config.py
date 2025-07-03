"""Helper functions for config-related settings."""

import os
from pathlib import Path
from typing import Any, List

from pydantic import Field
from pydantic._internal._utils import deep_update
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, YamlConfigSettingsSource
from pydantic_settings.sources.types import PathType


class DynamicYamlWithDeepMergeSettingsSource(YamlConfigSettingsSource):
    """YAML config settings source that can reload config files and merges them via deep update."""

    def _read_files(self, files: PathType | None) -> dict[str, Any]:
        if files is None:
            return {}
        if isinstance(files, (str, os.PathLike)):
            files = [files]
        vars: dict[str, Any] = {}
        for file in files:
            file_path = Path(file).expanduser()
            if file_path.is_file():
                new_vars = self._read_file(file_path)
                vars = deep_update(vars, new_vars)
        return vars

    def __call__(self):
        """Call additional config files based on current state."""
        yaml_data = self.yaml_data  # this points to the default yaml data now
        # update and return from additional files
        final_data = deep_update(
            yaml_data,
            self._read_files(self.current_state.get("yaml_configs", [])),
        )
        print(f"\n{final_data=}\n")
        return final_data


# TODO: think a little more about the order of precedence with recursive yaml file sources....
# TODO: maybe it's helpful to understand how default settings source precedence works...
# TODO: let's review the Cursor chat history again as well...


class DynamicYamlMixInForSettings:
    """Mix-in class class for settings providing dynamic yaml loading as lowest priority source.

    NOTE: This class must come FIRST in the MRO such that `yaml_configs` can be processed before
    since otherwise we cannot load default values from the `yaml_configs` first.
    """

    yaml_configs: List[PathType] = Field(
        default_factory=list,
        description="Additional yaml config files to load.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise settings sources."""
        deferred_yaml_settings = DynamicYamlWithDeepMergeSettingsSource(settings_cls)
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            deferred_yaml_settings,  # yaml files have lowest priority just before default values
        )
