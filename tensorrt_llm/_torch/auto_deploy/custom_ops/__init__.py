"""AutoDeploy's custom ops library.

This file ensures that all publicly listed files/custom ops in the custom_ops folder are
auto-imported and the corresponding custom ops are registered.
"""

import importlib
import pkgutil

__all__ = []

for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if module_name.startswith("_"):
        continue
    __all__.append(module_name)
    importlib.import_module(f"{__name__}.{module_name}")
