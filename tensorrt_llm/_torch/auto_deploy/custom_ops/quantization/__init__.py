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

"""Quantization operations.

This module provides quantization utilities and operations:
- quant: Quantization operations (FP8, FP4, INT4, INT8)
- torch_quant: PyTorch-based quantization implementations
- relu2_quant_fp8: Fused ReLU² + FP8 per-tensor quantization (Triton)
- gated_rms_norm_quant_fp8: Fused gated-RMSNorm + FP8 per-tensor quantization (Triton)
"""

from . import (
    gated_rms_norm_quant_fp8,  # noqa: F401 — registers auto_deploy::gated_rms_norm_quant_fp8
    relu2_quant_fp8,  # noqa: F401 — registers auto_deploy::relu2_quant_fp8
)

__all__ = [
    "quant",
    "torch_quant",
    "relu2_quant_fp8",
    "gated_rms_norm_quant_fp8",
]
