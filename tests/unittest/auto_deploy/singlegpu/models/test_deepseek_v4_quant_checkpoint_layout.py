# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DeepSeek V4 quantized-checkpoint-layout detection and validation tests."""

import json
import struct
from pathlib import Path

import pytest

from tensorrt_llm._torch.auto_deploy.models.checkpoint_metadata import has_safetensors_metadata
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (
    QuantizedCheckpointLayoutError,
)
from tensorrt_llm._torch.auto_deploy.models.quant_config_reader import (
    HFQuantConfigReader,
    autodetect_quant_config_reader,
)

_DSV4_CONFIG = {
    "model_type": "deepseek_v4",
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
}


def _default_tensor_metadata() -> dict:
    return {
        "layers.0.attn.wq_a.weight": {"dtype": "F8_E4M3", "shape": [1024, 4096]},
        "layers.0.attn.wq_a.scale": {"dtype": "F8_E8M0", "shape": [8, 32]},
        "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 2048]},
        "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [2048, 128]},
        "layers.0.ffn.experts.0.w2.weight": {"dtype": "I8", "shape": [4096, 1024]},
        "layers.0.ffn.experts.0.w2.scale": {"dtype": "F8_E8M0", "shape": [4096, 64]},
        "layers.0.ffn.experts.0.w3.weight": {"dtype": "I8", "shape": [2048, 2048]},
        "layers.0.ffn.experts.0.w3.scale": {"dtype": "F8_E8M0", "shape": [2048, 128]},
    }


def _write_deepseek_v4_checkpoint_fixture(
    tmp_path: Path, tensor_metadata: dict | None = None
) -> None:
    tensor_metadata = tensor_metadata if tensor_metadata is not None else _default_tensor_metadata()
    safetensors_name = "model-00001-of-00001.safetensors"
    header = {
        name: {
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
            "data_offsets": [0, 0],
        }
        for name, metadata in tensor_metadata.items()
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    (tmp_path / "config.json").write_text(json.dumps(_DSV4_CONFIG), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: safetensors_name for name in tensor_metadata},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / safetensors_name).write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)


def test_deepseek_v4_hf_reader_selects_checkpoint_layout_and_quant_config(
    tmp_path: Path,
) -> None:
    _write_deepseek_v4_checkpoint_fixture(tmp_path)

    result = HFQuantConfigReader.from_file(str(tmp_path))

    assert result is not None
    reader, extra_model_kwargs = result
    qcfg = reader.get_config()
    assert extra_model_kwargs == {}
    assert qcfg["checkpoint_layout"] is not None
    assert qcfg["expert_quant_method"] == "mxfp4"
    assert qcfg["expert_block_size"] == 32
    assert qcfg["scale_fmt"] == "ue8m0"
    assert qcfg["weight_block_size"] == [128, 128]
    assert "*.ffn.gate" in qcfg["exclude_modules"]
    assert "mtp.*" in qcfg["exclude_modules"]


def test_non_deepseek_fp8_config_uses_generic_hf_behavior() -> None:
    reader = HFQuantConfigReader()
    reader.read_config(
        {
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "fp8",
                "exclude_modules": ["custom"],
            },
        }
    )

    qcfg = reader.get_config()
    assert qcfg["quant_method"] == "fp8"
    assert qcfg["exclude_modules"] == ["custom", "lm_head", "model.embed_tokens"]
    assert "checkpoint_layout" not in qcfg


def test_safetensors_metadata_probe(tmp_path: Path) -> None:
    _write_deepseek_v4_checkpoint_fixture(tmp_path)
    assert has_safetensors_metadata(tmp_path)

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "model.safetensors").write_bytes(b"")
    assert has_safetensors_metadata(bare)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert not has_safetensors_metadata(empty)
    assert not has_safetensors_metadata(tmp_path / "missing")


def test_layout_detection_without_metadata_defers_when_not_required(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_DSV4_CONFIG), encoding="utf-8")
    assert HFQuantConfigReader.from_file(str(tmp_path), require_checkpoint_metadata=False) is None


def _drop_expert_weight(metadata: dict) -> None:
    del metadata["layers.0.ffn.experts.0.w1.weight"]


def _drop_expert_projection(metadata: dict) -> None:
    del metadata["layers.0.ffn.experts.0.w1.weight"]
    del metadata["layers.0.ffn.experts.0.w1.scale"]


def _bad_expert_scale_dtype(metadata: dict) -> None:
    metadata["layers.0.ffn.experts.0.w1.scale"]["dtype"] = "F32"


def _bad_fp8_scale_shape(metadata: dict) -> None:
    metadata["layers.0.attn.wq_a.scale"]["shape"] = [9, 32]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(_drop_expert_weight, "missing companion weight", id="scale-without-weight"),
        pytest.param(_drop_expert_projection, "missing required projection", id="no-projection"),
        pytest.param(_bad_expert_scale_dtype, "should be F8_E8M0 scale", id="bad-scale-dtype"),
        pytest.param(_bad_fp8_scale_shape, "does not match expected", id="bad-fp8-scale-shape"),
    ],
)
def test_layout_validation_errors_surface_through_autodetect(tmp_path: Path, mutate, match) -> None:
    metadata = _default_tensor_metadata()
    mutate(metadata)
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata=metadata)

    with pytest.raises(QuantizedCheckpointLayoutError, match=match):
        autodetect_quant_config_reader(str(tmp_path))


def test_invalid_safetensors_index_raises(tmp_path: Path) -> None:
    _write_deepseek_v4_checkpoint_fixture(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text("not json", encoding="utf-8")

    with pytest.raises(QuantizedCheckpointLayoutError, match="Invalid safetensors index JSON"):
        autodetect_quant_config_reader(str(tmp_path))
