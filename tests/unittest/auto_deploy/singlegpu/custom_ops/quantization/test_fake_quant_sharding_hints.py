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

"""Tests for sharding hint kwargs on fake quant linear ops.

AD-SHARD-002: Verify that all four fake quant linear ops accept TP sharding
hints (tp_mode, tp_fused_group_sizes, tp_min_local_shape) as keyword
arguments without changing their schema or behavior.

Tests at the schema level: the custom op schema must include the hint
parameters with correct defaults.
"""

import importlib.util
import os

import pytest
import torch

_quant_path = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        *([".."] * 8),
        "tensorrt_llm",
        "_torch",
        "auto_deploy",
        "custom_ops",
        "quantization",
        "torch_quant.py",
    )
)
_spec = importlib.util.spec_from_file_location("_ad_torch_quant", _quant_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

FAKE_QUANT_OPS = [
    "torch_fake_quant_fp8_linear",
    "torch_fake_quant_nvfp4_linear",
    "torch_fake_quant_int4_linear",
    "torch_fake_quant_int4_gptq_linear",
]


def _get_op_schema(op_name: str) -> str:
    """Get the string schema for a given auto_deploy op."""
    op = getattr(torch.ops.auto_deploy, op_name)
    return str(op.default._schema)


# ---------------------------------------------------------------------------
# Test 1: Schema includes TP hint parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op_name", FAKE_QUANT_OPS)
def test_schema_includes_tp_mode(op_name):
    schema = _get_op_schema(op_name)
    assert "tp_mode" in schema, f"{op_name} schema missing tp_mode: {schema}"


@pytest.mark.parametrize("op_name", FAKE_QUANT_OPS)
def test_schema_includes_tp_fused_group_sizes(op_name):
    schema = _get_op_schema(op_name)
    assert "tp_fused_group_sizes" in schema, (
        f"{op_name} schema missing tp_fused_group_sizes: {schema}"
    )


@pytest.mark.parametrize("op_name", FAKE_QUANT_OPS)
def test_schema_includes_tp_min_local_shape(op_name):
    schema = _get_op_schema(op_name)
    assert "tp_min_local_shape" in schema, f"{op_name} schema missing tp_min_local_shape: {schema}"


# ---------------------------------------------------------------------------
# Test 2: Defaults are correct (none, None, 1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op_name", FAKE_QUANT_OPS)
def test_schema_defaults(op_name):
    schema = _get_op_schema(op_name)
    assert 'tp_mode="none"' in schema, f"{op_name}: tp_mode default wrong in {schema}"
    assert "tp_min_local_shape=1" in schema, f"{op_name}: tp_min_local_shape default wrong"


# ---------------------------------------------------------------------------
# Test 3: Backward compatibility -- schema still has original params
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op_name", FAKE_QUANT_OPS)
def test_backward_compat_params(op_name):
    schema = _get_op_schema(op_name)
    for required in ["input", "weight_quantized", "bias", "input_scale", "weight_scale"]:
        assert required in schema, f"{op_name} schema missing original param {required}: {schema}"
