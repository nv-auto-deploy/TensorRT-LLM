# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared utilities for custom op kernel benchmarks."""

import argparse
from typing import Dict, List, Set

import torch

# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

STR_DTYPE_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "half": torch.float16,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "float32": torch.float32,
}


# ---------------------------------------------------------------------------
# Provider dtype filtering
# ---------------------------------------------------------------------------


def filter_providers_by_dtype(
    provider_dtype_support: Dict[str, Set[torch.dtype]],
    dtype: torch.dtype,
) -> List[str]:
    """Return the list of provider names that support *dtype*.

    Each benchmark script defines its own ``provider_dtype_support`` mapping,
    e.g.::

        PROVIDER_DTYPE_SUPPORT = {
            "flashinfer": {torch.float16, torch.bfloat16},
            "triton": {torch.float16, torch.bfloat16, torch.float32},
            "torch": {torch.float16, torch.bfloat16, torch.float32},
        }

    Calling ``filter_providers_by_dtype(PROVIDER_DTYPE_SUPPORT, torch.float32)``
    returns ``["triton", "torch"]``.
    """
    return [name for name, dtypes in provider_dtype_support.items() if dtype in dtypes]


# ---------------------------------------------------------------------------
# Common argparse
# ---------------------------------------------------------------------------


def make_arg_parser(description: str = "Benchmark custom op kernels.") -> argparse.ArgumentParser:
    """Create an :class:`ArgumentParser` with common flags shared across benchmarks."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(STR_DTYPE_TO_TORCH_DTYPE.keys()),
        default="bfloat16",
        help="Data type for benchmark inputs (default: bfloat16).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to save benchmark plot images.",
    )
    return parser
