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

"""Mirage/MPK bridge utilities for the Gemma4MoE dry-run translator.

This module intentionally keeps the first live integration small and explicit:
- resolve planned MPK method names against a real Mirage ``PersistentKernel``
- exercise a tiny set of task registrations on an actual H100-backed kernel
- optionally generate the Mirage task graph without compiling the generated CUDA

The goal is to prove that our translation-layer vocabulary is compatible with
the installed Mirage Python surface before full artifact emission is added.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from .types import GemmaLayerLoweringPlan, GemmaLoweringStatus

_COMPOSED_STEP_METHODS: Dict[str, tuple[str, ...]] = {
    "attn_rmsnorm_linear": ("rmsnorm_layer", "linear_layer"),
    "dense_ffn_gate_up": ("rmsnorm_layer", "linear_layer"),
    "dense_ffn_activation": ("moe_w2_linear_layer", "moe_mul_sum_add_layer"),
}

_MIRAGE_RUNTIME_EXTENSION_CACHE: Dict[str, Any] = {}
_MIRAGE_LINEAR_MAX_CAPACITY = 16
_MIRAGE_MOE_W2_MAX_CAPACITY = 16

_NORM_LINEAR_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "norm_linear.cuh"
#include "norm_linear_new.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void norm_linear_kernel_wrapper(
    void const* input_ptr,
    void const* norm_weight_ptr,
    void const* weight_ptr,
    float eps,
    void* output_ptr) {
  kernel::norm_linear_task_impl<T, 2, 64, 4096, 64>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

void norm_linear(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor weight,
    torch::Tensor output,
    float eps) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 1024 * 150;
  cudaFuncSetAttribute(
      norm_linear_kernel_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  norm_linear_kernel_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      input.data_ptr(),
      norm_weight.data_ptr(),
      weight.data_ptr(),
      eps,
      output.data_ptr());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_linear", &norm_linear, "norm linear kernel");
}
"""

_LINEAR_WITH_RESIDUAL_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "linear.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void linear_kernel_wrapper(
    void const* input_ptr,
    void const* weight_ptr,
    void const* residual_ptr,
    void* output_ptr) {
  kernel::linear_kernel<T, 4, 512, 512>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, 4, true);
}

void linear_with_residual(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor residual,
    torch::Tensor output) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(
      linear_kernel_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  linear_kernel_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      input.data_ptr(),
      weight.data_ptr(),
      residual.data_ptr(),
      output.data_ptr());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_with_residual",
      &linear_with_residual,
      "linear with residual kernel");
}
"""

_ATTN_QKV_NORM_LINEAR_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "norm_linear.cuh"
#include "norm_linear_new.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void norm_linear_kernel_wrapper(
    void const* input_ptr,
    void const* norm_weight_ptr,
    void const* weight_ptr,
    float eps,
    void* output_ptr) {
  kernel::norm_linear_task_impl<T, 4, 768, 512, 768>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

void norm_linear(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor weight,
    torch::Tensor output,
    float eps) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 1024 * 150;
  cudaFuncSetAttribute(
      norm_linear_kernel_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  norm_linear_kernel_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      input.data_ptr(),
      norm_weight.data_ptr(),
      weight.data_ptr(),
      eps,
      output.data_ptr());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_linear", &norm_linear, "attention qkv norm linear kernel");
}
"""

_ATTN_QKV_NORM_LINEAR_SINGLE_TOKEN_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "norm_linear.cuh"
#include "norm_linear_new.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void norm_linear_kernel_wrapper(
    void const* input_ptr,
    void const* norm_weight_ptr,
    void const* weight_ptr,
    float eps,
    void* output_ptr) {
  kernel::norm_linear_task_impl<T, 1, 768, 512, 768>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

void norm_linear(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor weight,
    torch::Tensor output,
    float eps) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 1024 * 150;
  cudaFuncSetAttribute(
      norm_linear_kernel_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  norm_linear_kernel_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      input.data_ptr(),
      norm_weight.data_ptr(),
      weight.data_ptr(),
      eps,
      output.data_ptr());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_linear", &norm_linear, "single-token attention qkv norm linear kernel");
}
"""

_ATTENTION_SUBLAYER_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "linear.cuh"
#include "multitoken_paged_attention.cuh"
#include "norm_linear.cuh"
#include "norm_linear_new.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void qkv_norm_linear_kernel_wrapper(
    void const* input_ptr,
    void const* norm_weight_ptr,
    void const* weight_ptr,
    float eps,
    void* output_ptr) {
  kernel::norm_linear_task_impl<T, 4, 768, 512, 768>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

template <typename T>
__global__ void multitoken_paged_attention_wrapper(
    void const* qkv_ptr,
    void* paged_k_cache_ptr,
    void* paged_v_cache_ptr,
    void* output_ptr,
    int const* qo_indptr_buffer_ptr,
    int const* paged_kv_indptr_buffer_ptr,
    int const* paged_kv_indices_buffer_ptr,
    int const* paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    void const* q_norm_weight_ptr,
    void const* k_norm_weight_ptr,
    void const* cos_ptr,
    void const* sin_ptr,
    float q_eps,
    float k_eps) {
  kernel::multitoken_paged_attention_task_impl<T, 4, 1, 128, 768, 512, 128, 512, 64, 4>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);
}

template <typename T>
__global__ void linear_with_residual_kernel_wrapper(
    void const* input_ptr,
    void const* weight_ptr,
    void const* residual_ptr,
    void* output_ptr) {
  kernel::linear_kernel<T, 4, 512, 512>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, 4, true);
}

void qkv_norm_linear(
    torch::Tensor input,
    torch::Tensor norm_weight,
    torch::Tensor weight,
    torch::Tensor output,
    float eps) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 1024 * 150;
  cudaFuncSetAttribute(
      qkv_norm_linear_kernel_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  qkv_norm_linear_kernel_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      input.data_ptr(),
      norm_weight.data_ptr(),
      weight.data_ptr(),
      eps,
      output.data_ptr());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

void multitoken_paged_attention(
    torch::Tensor qkv,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor output,
    torch::Tensor qo_indptr_buffer,
    torch::Tensor paged_kv_indptr_buffer,
    torch::Tensor paged_kv_indices_buffer,
    torch::Tensor paged_kv_last_page_len_buffer,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 88888;
  void const* q_norm_weight_ptr = qk_norm ? q_norm_weight->data_ptr() : nullptr;
  void const* k_norm_weight_ptr = qk_norm ? k_norm_weight->data_ptr() : nullptr;
  void const* cos_ptr = rope ? cos->data_ptr() : nullptr;
  void const* sin_ptr = rope ? sin->data_ptr() : nullptr;
  cudaFuncSetAttribute(
      multitoken_paged_attention_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  multitoken_paged_attention_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      qkv.data_ptr(),
      paged_k_cache.data_ptr(),
      paged_v_cache.data_ptr(),
      output.data_ptr(),
      qo_indptr_buffer.data_ptr<int>(),
      paged_kv_indptr_buffer.data_ptr<int>(),
      paged_kv_indices_buffer.data_ptr<int>(),
      paged_kv_last_page_len_buffer.data_ptr<int>(),
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

void linear_with_residual(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor residual,
    torch::Tensor output) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(
      linear_with_residual_kernel_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  linear_with_residual_kernel_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      input.data_ptr(),
      weight.data_ptr(),
      residual.data_ptr(),
      output.data_ptr());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qkv_norm_linear", &qkv_norm_linear, "attention qkv norm linear kernel");
  m.def("multitoken_paged_attention", &multitoken_paged_attention, "paged attention kernel");
  m.def(
      "linear_with_residual",
      &linear_with_residual,
      "linear with residual kernel");
}
"""

_PAGED_ATTENTION_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "multitoken_paged_attention.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void multitoken_paged_attention_wrapper(
    void const* qkv_ptr,
    void* paged_k_cache_ptr,
    void* paged_v_cache_ptr,
    void* output_ptr,
    int const* qo_indptr_buffer_ptr,
    int const* paged_kv_indptr_buffer_ptr,
    int const* paged_kv_indices_buffer_ptr,
    int const* paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    void const* q_norm_weight_ptr,
    void const* k_norm_weight_ptr,
    void const* cos_ptr,
    void const* sin_ptr,
    float q_eps,
    float k_eps) {
  kernel::multitoken_paged_attention_task_impl<T, 4, 1, 128, 768, 512, 128, 512, 64, 4>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);
}

void multitoken_paged_attention(
    torch::Tensor qkv,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor output,
    torch::Tensor qo_indptr_buffer,
    torch::Tensor paged_kv_indptr_buffer,
    torch::Tensor paged_kv_indices_buffer,
    torch::Tensor paged_kv_last_page_len_buffer,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 88888;
  void const* q_norm_weight_ptr = qk_norm ? q_norm_weight->data_ptr() : nullptr;
  void const* k_norm_weight_ptr = qk_norm ? k_norm_weight->data_ptr() : nullptr;
  void const* cos_ptr = rope ? cos->data_ptr() : nullptr;
  void const* sin_ptr = rope ? sin->data_ptr() : nullptr;
  cudaFuncSetAttribute(
      multitoken_paged_attention_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  multitoken_paged_attention_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      qkv.data_ptr(),
      paged_k_cache.data_ptr(),
      paged_v_cache.data_ptr(),
      output.data_ptr(),
      qo_indptr_buffer.data_ptr<int>(),
      paged_kv_indptr_buffer.data_ptr<int>(),
      paged_kv_indices_buffer.data_ptr<int>(),
      paged_kv_last_page_len_buffer.data_ptr<int>(),
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multitoken_paged_attention", &multitoken_paged_attention, "paged attention kernel");
}
"""

_PAGED_ATTENTION_SINGLE_TOKEN_WRAPPER_SOURCE = """
#include "bfloat16.h"
#include "multitoken_paged_attention.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void multitoken_paged_attention_wrapper(
    void const* qkv_ptr,
    void* paged_k_cache_ptr,
    void* paged_v_cache_ptr,
    void* output_ptr,
    int const* qo_indptr_buffer_ptr,
    int const* paged_kv_indptr_buffer_ptr,
    int const* paged_kv_indices_buffer_ptr,
    int const* paged_kv_last_page_len_buffer_ptr,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    void const* q_norm_weight_ptr,
    void const* k_norm_weight_ptr,
    void const* cos_ptr,
    void const* sin_ptr,
    float q_eps,
    float k_eps) {
  kernel::multitoken_paged_attention_task_impl<T, 4, 1, 128, 768, 512, 128, 512, 64, 1>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);
}

void multitoken_paged_attention(
    torch::Tensor qkv,
    torch::Tensor paged_k_cache,
    torch::Tensor paged_v_cache,
    torch::Tensor output,
    torch::Tensor qo_indptr_buffer,
    torch::Tensor paged_kv_indptr_buffer,
    torch::Tensor paged_kv_indices_buffer,
    torch::Tensor paged_kv_last_page_len_buffer,
    int16_t request_id,
    bool qk_norm,
    bool rope,
    torch::optional<torch::Tensor> q_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> k_norm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 88888;
  void const* q_norm_weight_ptr = qk_norm ? q_norm_weight->data_ptr() : nullptr;
  void const* k_norm_weight_ptr = qk_norm ? k_norm_weight->data_ptr() : nullptr;
  void const* cos_ptr = rope ? cos->data_ptr() : nullptr;
  void const* sin_ptr = rope ? sin->data_ptr() : nullptr;
  cudaFuncSetAttribute(
      multitoken_paged_attention_wrapper<bfloat16>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
  multitoken_paged_attention_wrapper<bfloat16><<<grid_dim, block_dim, smem_size>>>(
      qkv.data_ptr(),
      paged_k_cache.data_ptr(),
      paged_v_cache.data_ptr(),
      output.data_ptr(),
      qo_indptr_buffer.data_ptr<int>(),
      paged_kv_indptr_buffer.data_ptr<int>(),
      paged_kv_indices_buffer.data_ptr<int>(),
      paged_kv_last_page_len_buffer.data_ptr<int>(),
      request_id,
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      q_eps,
      k_eps);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\\n", cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multitoken_paged_attention", &multitoken_paged_attention, "single-token paged attention kernel");
}
"""


def _find_mirage_package_dir() -> Optional[Path]:
    for path_entry in sys.path:
        if not path_entry:
            continue
        candidate = Path(path_entry) / "mirage"
        if (candidate / "mpk" / "persistent_kernel.py").is_file():
            return candidate

    fallback = Path(
        "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/"
        "bmarimuthu/common/mirage/python/mirage"
    )
    if (fallback / "mpk" / "persistent_kernel.py").is_file():
        return fallback
    return None


def _bootstrap_mirage_namespace(package_dir: Path) -> None:
    existing_pkg = sys.modules.get("mirage")
    if existing_pkg is None or not hasattr(existing_pkg, "new_kernel_graph"):
        sys.modules.pop("mirage", None)
        init_py = package_dir / "__init__.py"
        if not init_py.is_file():
            raise RuntimeError(f"Mirage package init not found: {init_py}")
        spec = importlib.util.spec_from_file_location(
            "mirage",
            init_py,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load Mirage package spec from {init_py}")
        mirage_pkg = importlib.util.module_from_spec(spec)
        sys.modules["mirage"] = mirage_pkg
        spec.loader.exec_module(mirage_pkg)

    if "mirage.mpk" not in sys.modules:
        mirage_mpk_pkg = types.ModuleType("mirage.mpk")
        mirage_mpk_pkg.__path__ = [str(package_dir / "mpk")]
        sys.modules["mirage.mpk"] = mirage_mpk_pkg


def _preload_z3_shared_library() -> None:
    try:
        import z3
    except ImportError:
        return

    z3_lib_dir = Path(z3.__file__).resolve().parent / "lib"
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if str(z3_lib_dir) not in current_ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{z3_lib_dir}:{current_ld_library_path}"
            if current_ld_library_path
            else str(z3_lib_dir)
        )

    for lib_name in ("libz3.so.4.16", "libz3.so"):
        lib_path = z3_lib_dir / lib_name
        if not lib_path.is_file():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        break


def _require_mirage():
    _preload_z3_shared_library()
    try:
        from mirage.mpk.persistent_kernel import PersistentKernel
    except ImportError as exc:  # pragma: no cover - exercised only in Mirage-enabled envs
        package_dir = _find_mirage_package_dir()
        if package_dir is not None:
            _bootstrap_mirage_namespace(package_dir)
            module = importlib.import_module("mirage.mpk.persistent_kernel")
            return getattr(module, "PersistentKernel")

        raise RuntimeError(
            "Mirage is not importable. Ensure the mirage Python package is installed "
            "or /lustre/.../common/mirage/python is on PYTHONPATH."
        ) from exc
    return PersistentKernel


def _require_mirage_compile_support():
    _preload_z3_shared_library()
    try:
        persistent_kernel_mod = importlib.import_module("mirage.mpk.persistent_kernel")
        kernel_mod = importlib.import_module("mirage.kernel")
    except ImportError as exc:  # pragma: no cover - exercised only in Mirage-enabled envs
        package_dir = _find_mirage_package_dir()
        if package_dir is not None:
            _bootstrap_mirage_namespace(package_dir)
            persistent_kernel_mod = importlib.import_module("mirage.mpk.persistent_kernel")
            kernel_mod = importlib.import_module("mirage.kernel")
            return persistent_kernel_mod, kernel_mod
        raise RuntimeError(
            "Mirage compile helpers are not importable. Ensure mirage.mpk.persistent_kernel "
            "and mirage.kernel are available on PYTHONPATH."
        ) from exc
    return persistent_kernel_mod, kernel_mod


def _mpk_debug_enabled() -> bool:
    value = os.environ.get("AD_MPK_DEBUG", "")
    return value.lower() in {"1", "true", "yes", "on"}


_MPK_DEBUG_START_TIME = time.perf_counter()


def _mpk_debug(message: str) -> None:
    if _mpk_debug_enabled():
        elapsed_s = time.perf_counter() - _MPK_DEBUG_START_TIME
        print(f"[AD_MPK_DEBUG +{elapsed_s:8.3f}s] {message}", flush=True)


def _mpk_profile_blocks_enabled() -> bool:
    value = os.environ.get("AD_MPK_PROFILE_BLOCKS", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _mpk_sync_each_block_enabled() -> bool:
    value = os.environ.get("AD_MPK_SYNC_EVERY_BLOCK", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _mpk_profile_executors_enabled() -> bool:
    value = os.environ.get("AD_MPK_PROFILE_EXECUTORS", "")
    return value.lower() in {"1", "true", "yes", "on"}


def _mpk_profile(message: str) -> None:
    if _mpk_profile_blocks_enabled():
        elapsed_s = time.perf_counter() - _MPK_DEBUG_START_TIME
        print(f"[AD_MPK_PROFILE +{elapsed_s:8.3f}s] {message}", flush=True)


_T = TypeVar("_T")


def _mpk_run_profiled(label: str, fn: Callable[[], _T]) -> _T:
    if not _mpk_profile_blocks_enabled():
        return fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    _mpk_profile(f"{label} elapsed_s={time.perf_counter() - start:.6f}")
    return result


def _mpk_profile_executor(label: str, fn: Callable[[], _T]) -> _T:
    if not _mpk_profile_executors_enabled():
        return fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    _mpk_profile(f"{label} elapsed_s={time.perf_counter() - start:.6f}")
    return result


def _mirage_repo_root() -> Path:
    package_dir = _find_mirage_package_dir()
    if package_dir is None:
        raise RuntimeError(
            "Mirage package directory could not be located. Ensure mirage is installed "
            "or /lustre/.../common/mirage/python is on PYTHONPATH."
        )
    return package_dir.parent.parent


def _mirage_runtime_extension_build_dir(name: str) -> Path:
    return Path.cwd() / ".tmp" / "mirage_runtime_extensions" / name


def _mirage_runtime_include_paths(root: Path) -> list[str]:
    return [
        str(root / "include/mirage/persistent_kernel/tasks/ampere"),
        str(root / "include"),
        str(root / "include/mirage/transpiler"),
        str(root / "include/mirage/persistent_kernel/tasks/common"),
        str(root / "include/mirage/persistent_kernel"),
        str(root / "deps/json/include"),
        str(root / "deps/cutlass/include"),
        str(root / "deps/cutlass/tools/util/include"),
    ]


def _load_mirage_runtime_extension(
    *,
    name: str,
    source_text: str,
) -> Any:
    if name in _MIRAGE_RUNTIME_EXTENSION_CACHE:
        return _MIRAGE_RUNTIME_EXTENSION_CACHE[name]

    from torch.utils.cpp_extension import load

    root = _mirage_repo_root()
    build_dir = _mirage_runtime_extension_build_dir(name)
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{name}.cu"
    source_path.write_text(source_text, encoding="utf-8")

    module = load(
        name=name,
        sources=[str(source_path)],
        extra_include_paths=_mirage_runtime_include_paths(root),
        extra_cflags=["-DMIRAGE_BACKEND_USE_CUDA", "-DMIRAGE_FINGERPRINT_USE_CUDA"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-DMIRAGE_BACKEND_USE_CUDA",
            "-DMIRAGE_FINGERPRINT_USE_CUDA",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        build_directory=str(build_dir),
        verbose=False,
    )
    _MIRAGE_RUNTIME_EXTENSION_CACHE[name] = module
    return module


def _load_mirage_norm_linear_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_norm_linear_rt",
        source_text=_NORM_LINEAR_WRAPPER_SOURCE,
    )


def _load_mirage_attn_qkv_norm_linear_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_attn_qkv_norm_linear_rt",
        source_text=_ATTN_QKV_NORM_LINEAR_WRAPPER_SOURCE,
    )


def _load_mirage_attn_qkv_norm_linear_single_token_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_attn_qkv_norm_linear_single_token_rt",
        source_text=_ATTN_QKV_NORM_LINEAR_SINGLE_TOKEN_WRAPPER_SOURCE,
    )


def _load_mirage_attention_sublayer_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_attention_sublayer_rt",
        source_text=_ATTENTION_SUBLAYER_WRAPPER_SOURCE,
    )


def _load_mirage_linear_with_residual_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_linear_with_residual_rt",
        source_text=_LINEAR_WITH_RESIDUAL_WRAPPER_SOURCE,
    )


def _load_mirage_attention_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_attention_rt",
        source_text=_PAGED_ATTENTION_WRAPPER_SOURCE,
    )


def _load_mirage_attention_single_token_extension() -> Any:
    return _load_mirage_runtime_extension(
        name="ad_mirage_attention_single_token_rt",
        source_text=_PAGED_ATTENTION_SINGLE_TOKEN_WRAPPER_SOURCE,
    )


def patch_generated_mirage_cuda_source(cuda_code: str) -> str:
    """Patch known Mirage-generated CUDA source compatibility gaps.

    Current Mirage task registration can emit calls to task implementations on
    the Ampere runtime path without including the corresponding task header in
    ``task_header.cuh``. Until that is fixed upstream, patch the generated CUDA
    source locally before invoking ``nvcc``.
    """

    anchor = '#include "persistent_kernel.cuh"\n'
    missing_headers: list[str] = []
    if "norm_linear_task_impl" in cuda_code and "norm_linear_new.cuh" not in cuda_code:
        missing_headers.extend(
            [
                '#include "tasks/ampere/norm_linear.cuh"\n',
                '#include "tasks/ampere/norm_linear_new.cuh"\n',
            ]
        )
    if "single_batch_extend_kernel" in cuda_code and "single_batch_extend.cuh" not in cuda_code:
        missing_headers.append('#include "tasks/ampere/single_batch_extend.cuh"\n')
    if missing_headers and anchor in cuda_code:
        compat_headers = anchor + "".join(missing_headers)
        cuda_code = cuda_code.replace(anchor, compat_headers, 1)

    # Mirage's SM90 MoE linear task registration currently emits the expert-mask
    # layout with ``num_experts`` entries even though the kernel reads the final
    # ``num_experts`` slot as the "num activated experts" sentinel. Patch the
    # generated SM90 callsites so the mask layout is sized to ``num_experts + 1``.
    def _fix_sm90_moe_mask(match: re.Match[str]) -> str:
        num_experts = int(match.group(2))
        return f"{match.group(1)}{num_experts + 1}{match.group(3)}"

    sm90_moe_mask_pattern = re.compile(
        r"(cute::Layout layout_expert_mask =\s*cute::make_layout\(cute::make_shape\()"
        r"(\d+)"
        r"(\),\s*cute::make_stride\(cute::Int<1>\{\}\)\);\s*"
        r"cute::Tensor mMask.*?kernel::moe_linear_sm90_task_impl)",
        flags=re.DOTALL,
    )
    cuda_code = sm90_moe_mask_pattern.sub(_fix_sm90_moe_mask, cuda_code)
    return cuda_code


def _normalize_mpk_cache_source(cuda_code: str) -> str:
    """Normalize generated CUDA text for cache-key stability.

    Mirage can emit concrete device pointer literals into the generated launcher
    source for ``attach_input``-bound tensors. Those pointers are runtime
    instance details rather than structural compile inputs, so they should not
    force a cache miss.
    """

    return re.sub(r"0x[0-9a-fA-F]+", "0x__MIRAGE_PTR__", cuda_code)


def _rebind_compiled_pk_named_bindings(pk) -> None:
    """Rebind current named tensors/meta after loading or compiling a launcher."""

    bind_tensor = getattr(pk, "bind_tensor_func", None)
    if bind_tensor is not None:
        for name, tensor in getattr(pk, "_bound_tensors", {}).items():
            bind_tensor(name, tensor.data_ptr())

    bind_meta = getattr(pk, "bind_meta_func", None)
    if bind_meta is not None:
        for name, tensor in getattr(pk, "_bound_meta_tensors", {}).items():
            bind_meta(name, tensor.data_ptr())


def _load_cached_launcher_isolated(
    pk,
    cache_paths: dict[str, str],
    output_dir: Optional[str],
    export_compile_artifacts_fn: Callable[[str, Optional[str], int], None],
) -> bool:
    """Load a cached launcher through a per-instance copied module path.

    Mirage launcher modules hold process-global runtime state. Loading the same
    cached ``.so`` into multiple fresh PK objects in one interpreter can cause
    teardown collisions on shutdown. Copying the cached launcher to a unique
    per-instance path keeps the compile cache benefit while isolating module
    state for each PK instance.
    """

    if not os.path.isfile(cache_paths["module"]):
        return False

    ext_suffix = os.path.splitext(cache_paths["module"])[1]
    isolated_dir = tempfile.mkdtemp(prefix="mirage_pk_cached_load_")
    isolated_module_path = os.path.join(isolated_dir, f"launcher{ext_suffix}")
    shutil.copy2(cache_paths["module"], isolated_module_path)
    pk._bridge_cached_module_dir = isolated_dir
    pk._load_launcher_from_path(isolated_module_path)
    export_compile_artifacts_fn(cache_paths["dir"], output_dir, pk.mpi_rank)
    print(f"Loaded cached megakernel from {cache_paths['dir']}")
    return True


def compile_persistent_kernel_with_patches(
    pk,
    *,
    output_dir: Optional[str] = None,
):
    """Compile a Mirage ``PersistentKernel`` via the patched bridge cache path."""

    persistent_kernel_mod, kernel_mod = _require_mirage_compile_support()
    hard_code = getattr(persistent_kernel_mod, "HARD_CODE")
    mpk_version = getattr(persistent_kernel_mod, "__version__")
    mpk_cache_version = getattr(persistent_kernel_mod, "MPK_CACHE_VERSION")
    mpk_cache_source_name = getattr(persistent_kernel_mod, "MPK_CACHE_SOURCE_NAME")
    mpk_cache_task_graph_name = getattr(persistent_kernel_mod, "MPK_CACHE_TASK_GRAPH_NAME")
    mpk_cache_module_stem = getattr(persistent_kernel_mod, "MPK_CACHE_MODULE_STEM")
    get_compile_command = getattr(persistent_kernel_mod, "get_compile_command")
    export_compile_artifacts = getattr(persistent_kernel_mod, "_export_compile_artifacts")
    is_mpk_cache_enabled = getattr(persistent_kernel_mod, "_is_mpk_cache_enabled")
    get_mpk_cache_root = getattr(persistent_kernel_mod, "_get_mpk_cache_root")
    build_cache_key = getattr(persistent_kernel_mod, "_build_cache_key")
    cache_entry_paths = getattr(persistent_kernel_mod, "_cache_entry_paths")
    cache_entry_complete = getattr(persistent_kernel_mod, "_cache_entry_complete")
    readable_nvcc_version = getattr(persistent_kernel_mod, "_readable_nvcc_version")
    sha256_text = getattr(persistent_kernel_mod, "_sha256_text")
    get_key_paths = getattr(kernel_mod, "get_key_paths")

    if pk._is_compiled:
        return pk

    results = pk.kn_graph.generate_task_graph(num_gpus=pk.world_size, my_gpu_id=pk.mpi_rank)
    task_graph_json = results["json_file"]
    cuda_code = patch_generated_mirage_cuda_source(results["cuda_code"] + hard_code)

    cc = shutil.which("nvcc")
    if cc is None:
        raise RuntimeError("nvcc not found. Please make sure you have installed CUDA.")

    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

    mirage_root, include_path, deps_path = get_key_paths()
    mirage_home_path = os.environ.get("MIRAGE_HOME", mirage_root)

    normalized_cc_cmd = get_compile_command(
        mpk=pk,
        target_cc=pk.target_cc,
        cc=cc,
        file_name="__MIRAGE_MPK_SOURCE__",
        py_include_dir=py_include_dir,
        mirage_home_path=mirage_home_path,
        mirage_inc_path=include_path,
        mirage_deps_path=deps_path,
        nvshmem_inc_path=None,
        nvshmem_lib_path=None,
        mpi_inc_path=None,
        mpi_lib_path=None,
        py_so_path="__MIRAGE_MPK_OUTPUT__" + ext_suffix,
        profiling=True if pk.profiler_tensor is not None else False,
        use_nvshmem=pk.use_nvshmem,
        num_workers=pk.num_workers,
        num_local_schedulers=pk.num_local_schedulers,
        num_remote_schedulers=pk.num_remote_schedulers,
        use_cutlass_kernel=pk.use_cutlass_kernel,
    )
    if pk.target_cc == 90:
        normalized_cc_cmd = [arg for arg in normalized_cc_cmd if arg != "-arch=sm_90a"]

    cache_key_material = {
        "cache_version": mpk_cache_version,
        "mirage_version": mpk_version,
        "ext_suffix": ext_suffix,
        "target_cc": pk.target_cc,
        "mode": pk.mode,
        "use_nvshmem": pk.use_nvshmem,
        "nvcc_path": cc,
        "nvcc_version": readable_nvcc_version(cc),
        "compile_command": normalized_cc_cmd,
        "bridge_patch_version": 2,
    }
    cache_key = build_cache_key(
        key_material=cache_key_material,
        cuda_code=_normalize_mpk_cache_source(cuda_code),
        task_graph_json=task_graph_json,
    )

    cache_paths = None
    tempdir = None
    use_cache = is_mpk_cache_enabled()
    if use_cache:
        try:
            cache_root = get_mpk_cache_root()
            os.makedirs(cache_root, exist_ok=True)
            cache_paths = cache_entry_paths(os.path.join(cache_root, cache_key), ext_suffix)
            if cache_entry_complete(cache_paths):
                try:
                    if _load_cached_launcher_isolated(
                        pk,
                        cache_paths,
                        output_dir,
                        export_compile_artifacts,
                    ):
                        pk._initialize_compiled_launcher()
                        _rebind_compiled_pk_named_bindings(pk)
                        pk._is_compiled = True
                        return pk
                except Exception as exc:
                    print(
                        f"Failed to load cached MPK entry at {cache_paths['dir']} ({exc}); recompiling."
                    )
                    shutil.rmtree(cache_paths["dir"], ignore_errors=True)
            if os.path.isdir(cache_paths["dir"]) and not cache_entry_complete(cache_paths):
                shutil.rmtree(cache_paths["dir"], ignore_errors=True)
            tempdir = cache_paths["dir"]
            os.makedirs(tempdir, exist_ok=True)
        except OSError:
            cache_paths = None
            use_cache = False

    if tempdir is None:
        tempdir = tempfile.mkdtemp(prefix="mirage_pk_")

    cuda_code_path = os.path.join(tempdir, mpk_cache_source_name)
    so_path = os.path.join(tempdir, mpk_cache_module_stem + ext_suffix)
    json_file_path = os.path.join(tempdir, mpk_cache_task_graph_name)

    with open(json_file_path, "w", encoding="utf-8") as f:
        f.write(task_graph_json)
    with open(cuda_code_path, "w", encoding="utf-8") as f:
        f.write(cuda_code)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{pk.mpi_rank}.cu"))
        shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{pk.mpi_rank}.json"))

    cc_cmd = get_compile_command(
        mpk=pk,
        target_cc=pk.target_cc,
        cc=cc,
        file_name=cuda_code_path,
        py_include_dir=py_include_dir,
        mirage_home_path=mirage_home_path,
        mirage_inc_path=include_path,
        mirage_deps_path=deps_path,
        nvshmem_inc_path=None,
        nvshmem_lib_path=None,
        mpi_inc_path=None,
        mpi_lib_path=None,
        py_so_path=so_path,
        profiling=True if pk.profiler_tensor is not None else False,
        use_nvshmem=pk.use_nvshmem,
        num_workers=pk.num_workers,
        num_local_schedulers=pk.num_local_schedulers,
        num_remote_schedulers=pk.num_remote_schedulers,
        use_cutlass_kernel=pk.use_cutlass_kernel,
    )

    # On Hopper, Mirage's default compile command includes both
    # ``-arch=sm_90a`` and the explicit ``-gencode=arch=compute_90a,code=sm_90a``.
    # With CUDA 13.1, that combination emits an extra ``compute_90`` PTX stream
    # in addition to the desired ``compute_90a`` stream. Hopper-only WGMMA code
    # in Mirage then fails during ptxas on the unintended ``sm_90`` PTX path.
    # Keeping only the explicit ``compute_90a -> sm_90a`` gencode avoids the
    # stray PTX target while preserving the intended Hopper cubin.
    if pk.target_cc == 90:
        cc_cmd = [arg for arg in cc_cmd if arg != "-arch=sm_90a"]

    extra_include_dirs = [
        os.path.join(include_path, "mirage", "persistent_kernel", "tasks", "common"),
        os.path.join(include_path, "mirage", "persistent_kernel", "tasks", "ampere"),
        os.path.join(include_path, "mirage", "transpiler"),
    ]
    insertion_index = 0
    for i, arg in enumerate(cc_cmd):
        if arg.startswith("-D"):
            insertion_index = i
            break
    else:
        insertion_index = len(cc_cmd)
    include_args = [f"-I{path}" for path in extra_include_dirs if f"-I{path}" not in cc_cmd]
    cc_cmd[insertion_index:insertion_index] = include_args
    subprocess.check_call(cc_cmd)

    artifact_dir = tempdir
    if use_cache and cache_paths is not None:
        metadata = {
            "cache_key": cache_key,
            "cache_version": mpk_cache_version,
            "mirage_version": mpk_version,
            "target_cc": pk.target_cc,
            "mode": pk.mode,
            "use_nvshmem": pk.use_nvshmem,
            "nvcc_path": cc,
            "nvcc_version": cache_key_material["nvcc_version"],
            "compile_command": normalized_cc_cmd,
            "source_sha256": sha256_text(cuda_code),
            "task_graph_sha256": sha256_text(task_graph_json),
            "hard_code_sha256": sha256_text(hard_code),
            "bridge_patch_version": cache_key_material["bridge_patch_version"],
        }
        metadata_path = os.path.join(artifact_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

    spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Mirage launcher module from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pk.init_func = getattr(mod, "init_func")
    pk.launch_func = getattr(mod, "launch_func")
    pk.init_request_func = getattr(mod, "init_request_func")
    pk.bind_tensor_func = getattr(mod, "bind_tensor_func", None)
    pk.bind_meta_func = getattr(mod, "bind_meta_func", None)
    pk.update_dynamic_dims_func = getattr(mod, "update_dynamic_dims_func", None)
    pk.finalize_func = getattr(mod, "finalize_func")
    export_compile_artifacts(artifact_dir, output_dir, pk.mpi_rank)

    meta_tensors = [
        pk.meta_tensors["step"],
        pk.meta_tensors["tokens"],
        pk.meta_tensors["input_tokens"],
        pk.meta_tensors["output_tokens"],
        pk.meta_tensors["num_new_tokens"],
        pk.meta_tensors["prompt_lengths"],
        pk.meta_tensors["qo_indptr_buffer"],
        pk.meta_tensors["paged_kv_indptr_buffer"],
        pk.meta_tensors["paged_kv_indices_buffer"],
        pk.meta_tensors["paged_kv_last_page_len_buffer"],
    ]
    meta_tensors_ptr = [tensor.data_ptr() for tensor in meta_tensors]
    profiler_buffer_ptr = pk.profiler_tensor.data_ptr() if pk.profiler_tensor is not None else 0
    pk.init_func(
        meta_tensors_ptr,
        profiler_buffer_ptr,
        pk.mpi_rank,
        pk.num_workers,
        pk.num_local_schedulers,
        pk.num_remote_schedulers,
        pk.max_seq_length,
        pk.total_num_requests,
        pk.eos_token_id,
        pk.allocate_nvshmem_teams,
    )
    _rebind_compiled_pk_named_bindings(pk)
    pk._is_compiled = True
    return pk


@dataclass
class MirageBindingResult:
    step_name: str
    requested_method: Optional[str]
    status: str
    resolved: bool
    notes: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "requested_method": self.requested_method,
            "status": self.status,
            "resolved": self.resolved,
            "notes": list(self.notes),
        }


def _rms_norm(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply a small RMSNorm helper for reference execution."""

    output_dtype = input_tensor.dtype
    input_fp32 = input_tensor.float()
    weight_fp32 = weight.float()
    variance = input_fp32.square().mean(dim=-1, keepdim=True)
    normalized = input_fp32 * torch.rsqrt(variance + eps)
    return (normalized * weight_fp32).to(dtype=output_dtype)


def _apply_rope(input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply a simple pairwise RoPE rotation for reference execution."""

    hidden_size = input_tensor.shape[-1]
    rotary_dim = hidden_size - (hidden_size % 2)
    if rotary_dim == 0:
        return input_tensor

    input_rotary = input_tensor[..., :rotary_dim]
    input_tail = input_tensor[..., rotary_dim:]
    half_rotary = rotary_dim // 2

    x0 = input_rotary[..., :half_rotary]
    x1 = input_rotary[..., half_rotary:]
    cos_slice = cos[..., :half_rotary].float()
    sin_slice = sin[..., :half_rotary].float()

    rotated = torch.cat([x0 * cos_slice - x1 * sin_slice, x0 * sin_slice + x1 * cos_slice], dim=-1)
    if input_tail.numel() == 0:
        return rotated
    return torch.cat([rotated, input_tail.float()], dim=-1)


def _expert_gated_activation(input_tensor: torch.Tensor, *, act_fn: str) -> torch.Tensor:
    gate, up = torch.chunk(input_tensor.float(), 2, dim=-1)
    if act_fn == "silu":
        activated = F.silu(gate)
    else:
        activated = F.gelu(gate)
    return activated * up


def _extract_ranked_experts(routing_indices: torch.Tensor, topk: int) -> list[int]:
    selected_experts = []
    for expert_index in range(int(routing_indices.shape[0])):
        rank = int(routing_indices[expert_index, 0].item())
        if rank != 0:
            selected_experts.append((expert_index, rank))
    selected_experts = sorted(selected_experts, key=lambda item: item[1])
    return [expert for expert, _ in selected_experts[:topk]]


def execute_layer_plan_reference(
    layer_plan: GemmaLayerLoweringPlan,
    *,
    hidden_in: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Execute one Gemma layer plan numerically with torch reference kernels.

    This is intentionally a plan-driven reference executor rather than a Mirage
    runtime path. It lets us validate the full translated layer semantics end to
    end, including the steps that remain backend gaps in today's Mirage task
    surface.
    """

    buffers: Dict[str, torch.Tensor] = {layer_plan.input_buffer: hidden_in.float()}

    for step in layer_plan.mpk_steps:
        if step.name == "attn_rmsnorm_linear":
            hidden = buffers[step.inputs[0]]
            normed = _rms_norm(hidden, weights["attn_norm_weight"], eps=eps)
            buffers[step.outputs[0]] = normed @ weights["qkv_weight"].float().transpose(0, 1)
        elif step.name == "paged_attention":
            qkv = buffers[step.inputs[0]]
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            q_ready = _apply_rope(
                _rms_norm(q, weights["q_norm_weight"], eps=eps),
                weights["cos"],
                weights["sin"],
            )
            k_ready = _apply_rope(
                _rms_norm(k, weights["k_norm_weight"], eps=eps),
                weights["cos"],
                weights["sin"],
            )
            v_ready = _rms_norm(v, weights["v_norm_weight"], eps=eps)
            k_cache_bias = weights["k_cache"].float().mean(dim=0, keepdim=True)
            v_cache_bias = weights["v_cache"].float().mean(dim=0, keepdim=True)
            buffers[step.outputs[0]] = q_ready + k_ready + v_ready + k_cache_bias + v_cache_bias
        elif step.name == "attn_out_proj":
            attn_out = buffers[step.inputs[0]]
            residual = buffers[step.inputs[1]]
            projected = attn_out @ weights["o_proj_weight"].float().transpose(0, 1)
            buffers[step.outputs[0]] = projected + residual
        elif step.name == "dense_ffn_gate_up":
            hidden = buffers[step.inputs[0]]
            normed = _rms_norm(hidden, weights["ffn_norm_weight"], eps=eps)
            buffers[step.outputs[0]] = normed @ weights["ffn_gate_up_weight"].float().transpose(
                0, 1
            )
        elif step.name == "dense_ffn_activation":
            gate_up = buffers[step.inputs[0]]
            activated = _expert_gated_activation(gate_up, act_fn="gelu")
            buffers[step.outputs[0]] = activated @ weights["ffn_down_weight"].float().transpose(
                0, 1
            )
        elif step.name == "router_projection":
            hidden = buffers[step.inputs[0]]
            buffers[step.outputs[0]] = hidden @ weights["router_weight"].float().transpose(0, 1)
        elif step.name == "router_topk_softmax":
            logits = buffers[step.inputs[0]]
            top_k = int(step.params["top_k"])
            topk_values, topk_indices = torch.topk(logits, k=top_k, dim=-1)
            topk_weights = torch.softmax(topk_values, dim=-1)
            routing_mask = torch.zeros_like(logits, dtype=torch.float32)
            routing_mask.scatter_(1, topk_indices, 1.0)
            buffers[step.outputs[0]] = topk_weights
            buffers[step.outputs[1]] = topk_indices
            buffers[step.outputs[2]] = routing_mask
        elif step.name == "moe_w13_linear":
            hidden = buffers[step.inputs[0]]
            routing_indices = buffers[step.inputs[1]].long()
            token_outputs = []
            for token_idx in range(hidden.shape[0]):
                expert_outputs = []
                for route_idx in range(routing_indices.shape[1]):
                    expert_index = int(routing_indices[token_idx, route_idx].item())
                    expert_weight = weights["moe_w13_weight"][expert_index].float()
                    expert_outputs.append(hidden[token_idx].float() @ expert_weight.transpose(0, 1))
                token_outputs.append(torch.stack(expert_outputs, dim=0))
            buffers[step.outputs[0]] = torch.stack(token_outputs, dim=0)
        elif step.name == "moe_activation":
            act_name = str(step.params.get("act_fn", "gelu"))
            buffers[step.outputs[0]] = _expert_gated_activation(
                buffers[step.inputs[0]], act_fn=act_name
            )
        elif step.name == "moe_w2_linear":
            activated = buffers[step.inputs[0]]
            routing_indices = buffers[step.inputs[1]].long()
            token_outputs = []
            for token_idx in range(activated.shape[0]):
                expert_outputs = []
                for route_idx in range(routing_indices.shape[1]):
                    expert_index = int(routing_indices[token_idx, route_idx].item())
                    expert_weight = weights["moe_w2_weight"][expert_index].float()
                    expert_outputs.append(
                        activated[token_idx, route_idx].float() @ expert_weight.transpose(0, 1)
                    )
                token_outputs.append(torch.stack(expert_outputs, dim=0))
            buffers[step.outputs[0]] = torch.stack(token_outputs, dim=0)
        elif step.name == "moe_reduce":
            moe_out = buffers[step.inputs[0]]
            router_weights = buffers[step.inputs[1]]
            dense_residual = buffers[step.inputs[2]]
            reduced = (moe_out * router_weights.unsqueeze(-1)).sum(dim=1)
            buffers[step.outputs[0]] = reduced + dense_residual
        else:
            raise ValueError(f"Unsupported reference step: {step.name}")

    return buffers


def run_mirage_norm_linear_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a real Mirage norm-linear kernel and compare against torch."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    module = _load_mirage_norm_linear_extension()

    x = torch.randn((2, 4096), device="cuda", dtype=torch.bfloat16)
    g = torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((64, 4096), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((2, 64), device="cuda", dtype=torch.bfloat16)

    module.norm_linear(x, g, w, out, eps)

    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    ref = (x.float() * torch.rsqrt(variance + eps)) * g.float()
    ref = ref @ w.float().transpose(0, 1)
    diff = (out.float() - ref).abs()

    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def _reference_multitoken_paged_attention(
    *,
    q: torch.Tensor,
    paged_k_cache: torch.Tensor,
    paged_v_cache: torch.Tensor,
    qo_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    num_tokens: int,
    paged_kv_indptr_buffer: torch.Tensor,
    paged_kv_indices_buffer: torch.Tensor,
    paged_kv_last_page_len_buffer: torch.Tensor,
) -> torch.Tensor:
    device = q.device
    first_page_pos = int(paged_kv_indptr_buffer[0].item())
    last_page_pos = int(paged_kv_indptr_buffer[1].item())
    page_indices = [
        int(paged_kv_indices_buffer[i].item()) for i in range(first_page_pos, last_page_pos)
    ]
    seq_len = (last_page_pos - first_page_pos - 1) * page_size + int(
        paged_kv_last_page_len_buffer[0].item()
    )

    k_cache = torch.cat([paged_k_cache[idx] for idx in page_indices], dim=0)
    v_cache = torch.cat([paged_v_cache[idx] for idx in page_indices], dim=0)
    norm_q = q.reshape(num_tokens * qo_heads, head_dim)
    k = k_cache[:seq_len, :]
    v = v_cache[:seq_len, :].view(seq_len * kv_heads, head_dim)

    scores = torch.matmul(norm_q.float(), k.float().transpose(-2, -1))
    seq_cols = scores.shape[1]
    base = seq_cols - num_tokens
    cols = torch.arange(seq_cols, device=device)
    mask = (cols[None, :] < base) | (
        cols[None, :] < (base + torch.arange(1, num_tokens + 1, device=device)[:, None])
    )
    mask = mask.repeat_interleave(qo_heads, dim=0)
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores / (float(head_dim) ** 0.5), dim=-1)
    return torch.matmul(attn, v.float())


def run_mirage_paged_attention_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a real Mirage paged-attention kernel and compare against torch."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    attn_module = _load_mirage_attention_extension()
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    max_tokens = 4

    paged_k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads * head_dim), device="cuda", dtype=torch.bfloat16
    )
    paged_v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads * head_dim), device="cuda", dtype=torch.bfloat16
    )
    torch_paged_k_cache = paged_k_cache.clone()
    torch_paged_v_cache = paged_v_cache.clone()

    qo_indptr_buffer = torch.tensor([0, max_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr_buffer = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices_buffer = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len_buffer = torch.tensor([8 + max_tokens], device="cuda", dtype=torch.int32)

    qkv = 0.3 + 0.1 * torch.randn(
        (max_tokens, (qo_heads + 2 * kv_heads) * head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = qkv[:, : qo_heads * head_dim].view(max_tokens, qo_heads, head_dim)
    k = qkv[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv[:, qo_heads * head_dim + kv_heads * head_dim :]

    page_idx = int(paged_kv_indices_buffer[0].item())
    page_offset = int(paged_kv_last_page_len_buffer[0].item()) - max_tokens
    torch_paged_k_cache[page_idx, page_offset : page_offset + max_tokens] = k
    torch_paged_v_cache[page_idx, page_offset : page_offset + max_tokens] = v

    mirage_output = torch.empty(
        (max_tokens * qo_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    q_norm_weight = torch.ones((1, head_dim), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((1, head_dim), device="cuda", dtype=torch.bfloat16)
    all_cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    all_sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)

    attn_module.multitoken_paged_attention(
        qkv,
        paged_k_cache,
        paged_v_cache,
        mirage_output,
        qo_indptr_buffer,
        paged_kv_indptr_buffer,
        paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer,
        0,
        False,
        False,
        q_norm_weight,
        k_norm_weight,
        all_cos,
        all_sin,
        1e-5,
        1e-5,
    )

    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_paged_k_cache,
        paged_v_cache=torch_paged_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=max_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr_buffer,
        paged_kv_indices_buffer=paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
    )
    attn_diff = (mirage_output.float() - ref_attn.float()).abs()
    return {
        "attn_max_abs": float(attn_diff.max().item()),
        "attn_mean_abs": float(attn_diff.mean().item()),
    }


def run_mirage_linear_with_residual_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a real Mirage residual-linear kernel and compare against torch."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    linear_module = _load_mirage_linear_with_residual_extension()

    linear_input = (torch.randn((4, 512), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    residual = (torch.randn((4, 512), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    weight = (torch.randn((512, 512), device="cuda", dtype=torch.bfloat16) / 16.0).contiguous()
    out = torch.zeros_like(linear_input)

    linear_module.linear_with_residual(linear_input, weight, residual, out)
    if not torch.isfinite(out.float()).all():
        return {"linear_max_abs": float("inf"), "linear_mean_abs": float("inf")}

    ref_linear = linear_input.float() @ weight.float().transpose(0, 1)
    ref_linear = ref_linear + residual.float()
    linear_diff = (out.float() - ref_linear).abs()
    return {
        "linear_max_abs": float(linear_diff.max().item()),
        "linear_mean_abs": float(linear_diff.mean().item()),
    }


def run_mirage_linear_with_residual_pk_forward_correctness(
    *,
    seed: int = 0,
    repeats: int = 2,
) -> Dict[str, float]:
    """Run a tiny compiled Mirage PersistentKernel linear block repeatedly."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=4,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    linear_input = (torch.randn((4, 512), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    residual = (torch.randn((4, 512), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    weight = (torch.randn((512, 512), device="cuda", dtype=torch.bfloat16) / 16.0).contiguous()
    output = torch.zeros((4, 512), device="cuda", dtype=torch.bfloat16)

    input_dt = pk.attach_input(linear_input, name="pk_linear_input")
    weight_dt = pk.attach_input(weight, name="pk_linear_weight")
    residual_dt = pk.attach_input(residual, name="pk_linear_residual")
    output_dt = pk.attach_input(output, name="pk_linear_output")
    pk.linear_with_residual_layer(
        input=input_dt,
        weight=weight_dt,
        residual=residual_dt,
        output=output_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)

    ref_linear = linear_input.float() @ weight.float().transpose(0, 1)
    ref_linear = ref_linear + residual.float()

    metrics: Dict[str, float] = {}
    for repeat_idx in range(repeats):
        _reset_pk_runtime_state(pk, active_tokens=linear_input.shape[0])
        output.zero_()
        pk()
        torch.cuda.synchronize()
        if not torch.isfinite(output.float()).all():
            metrics[f"repeat_{repeat_idx}_max_abs"] = float("inf")
            metrics[f"repeat_{repeat_idx}_mean_abs"] = float("inf")
            continue
        linear_diff = (output.float() - ref_linear).abs()
        metrics[f"repeat_{repeat_idx}_max_abs"] = float(linear_diff.max().item())
        metrics[f"repeat_{repeat_idx}_mean_abs"] = float(linear_diff.mean().item())
    return metrics


def run_mirage_rmsnorm_linear_pk_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
    repeats: int = 2,
) -> Dict[str, float]:
    """Run a tiny compiled Mirage PersistentKernel rmsnorm->linear block repeatedly."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    hidden_in = (torch.randn((1, 512), device="cuda", dtype=torch.bfloat16) / 8.0).contiguous()
    norm_weight = torch.ones((512,), device="cuda", dtype=torch.bfloat16)
    linear_weight = (
        torch.randn((768, 512), device="cuda", dtype=torch.bfloat16) / 16.0
    ).contiguous()
    rmsnorm_out = torch.zeros((1, 512), device="cuda", dtype=torch.bfloat16)
    linear_out = torch.zeros((1, 768), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in, name="pk_rmsnorm_linear_hidden")
    norm_weight_dt = pk.attach_input(norm_weight, name="pk_rmsnorm_linear_weight")
    linear_weight_dt = pk.attach_input(linear_weight, name="pk_rmsnorm_linear_proj_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="pk_rmsnorm_linear_norm_out")
    linear_out_dt = pk.attach_input(linear_out, name="pk_rmsnorm_linear_out")
    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=norm_weight_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=linear_weight_dt,
        output=linear_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)

    variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    rmsnorm_ref = (hidden_in.float() * torch.rsqrt(variance + eps)) * norm_weight.float()
    linear_ref = rmsnorm_ref @ linear_weight.float().transpose(0, 1)

    metrics: Dict[str, float] = {}
    for repeat_idx in range(repeats):
        _reset_pk_runtime_state(pk, active_tokens=hidden_in.shape[0])
        rmsnorm_out.zero_()
        linear_out.zero_()
        pk()
        torch.cuda.synchronize()
        rmsnorm_diff = (rmsnorm_out.float() - rmsnorm_ref).abs()
        linear_diff = (linear_out.float() - linear_ref).abs()
        metrics[f"repeat_{repeat_idx}_rmsnorm_max_abs"] = float(rmsnorm_diff.max().item())
        metrics[f"repeat_{repeat_idx}_rmsnorm_mean_abs"] = float(rmsnorm_diff.mean().item())
        metrics[f"repeat_{repeat_idx}_linear_max_abs"] = float(linear_diff.max().item())
        metrics[f"repeat_{repeat_idx}_linear_mean_abs"] = float(linear_diff.mean().item())
    return metrics


def run_mirage_attention_sublayer_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a live Mirage attention sublayer and compare each stage against torch.

    The tested composition is:
    ``rmsnorm+qkv linear -> paged attention -> o_proj + residual``.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    sublayer_module = _load_mirage_attention_sublayer_extension()

    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((1, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    qkv_out = torch.empty(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    sublayer_module.qkv_norm_linear(
        hidden_in.contiguous(), attn_norm_weight, qkv_weight, qkv_out, eps
    )

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    qkv_diff = (qkv_out.float() - qkv_ref).abs()

    paged_k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads * head_dim), device="cuda", dtype=torch.bfloat16
    )
    paged_v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads * head_dim), device="cuda", dtype=torch.bfloat16
    )
    torch_paged_k_cache = paged_k_cache.clone()
    torch_paged_v_cache = paged_v_cache.clone()

    qo_indptr_buffer = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr_buffer = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices_buffer = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len_buffer = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)

    k = qkv_out[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_out[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices_buffer[0].item())
    page_offset = int(paged_kv_last_page_len_buffer[0].item()) - num_tokens
    torch_paged_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_paged_v_cache[page_idx, page_offset : page_offset + num_tokens] = v

    attn_out = torch.empty((num_tokens * qo_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    q_norm_weight = torch.ones((1, head_dim), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((1, head_dim), device="cuda", dtype=torch.bfloat16)
    all_cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    all_sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)

    sublayer_module.multitoken_paged_attention(
        qkv_out.contiguous(),
        paged_k_cache,
        paged_v_cache,
        attn_out,
        qo_indptr_buffer,
        paged_kv_indptr_buffer,
        paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer,
        0,
        False,
        False,
        q_norm_weight,
        k_norm_weight,
        all_cos,
        all_sin,
        eps,
        eps,
    )

    q = qkv_out[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_paged_k_cache,
        paged_v_cache=torch_paged_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr_buffer,
        paged_kv_indices_buffer=paged_kv_indices_buffer,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
    )
    attn_diff = (attn_out.float() - ref_attn.float()).abs()

    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    post_attn_residual = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    linear_input = attn_out.reshape(num_tokens, hidden_size).contiguous()
    sublayer_module.linear_with_residual(
        linear_input,
        o_proj_weight.contiguous(),
        hidden_in.contiguous(),
        post_attn_residual,
    )

    if not torch.isfinite(post_attn_residual.float()).all():
        return {
            "qkv_max_abs": float(qkv_diff.max().item()),
            "qkv_mean_abs": float(qkv_diff.mean().item()),
            "attn_max_abs": float(attn_diff.max().item()),
            "attn_mean_abs": float(attn_diff.mean().item()),
            "linear_max_abs": float("inf"),
            "linear_mean_abs": float("inf"),
        }

    ref_linear = ref_attn.reshape(num_tokens, hidden_size)
    ref_linear = ref_linear @ o_proj_weight.float().transpose(0, 1)
    ref_linear = ref_linear + hidden_in.float()
    linear_diff = (post_attn_residual.float() - ref_linear).abs()

    return {
        "qkv_max_abs": float(qkv_diff.max().item()),
        "qkv_mean_abs": float(qkv_diff.mean().item()),
        "attn_max_abs": float(attn_diff.max().item()),
        "attn_mean_abs": float(attn_diff.mean().item()),
        "linear_max_abs": float(linear_diff.max().item()),
        "linear_mean_abs": float(linear_diff.mean().item()),
    }


def run_mirage_attention_sublayer_pk_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
    repeats: int = 2,
) -> Dict[str, float]:
    """Run a tiny compiled Mirage PersistentKernel attention sublayer repeatedly."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64

    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)
    pk.meta_tensors["qo_indptr_buffer"].zero_()
    pk.meta_tensors["qo_indptr_buffer"][:2] = qo_indptr
    pk.meta_tensors["paged_kv_indptr_buffer"].zero_()
    pk.meta_tensors["paged_kv_indptr_buffer"][:2] = paged_kv_indptr
    pk.meta_tensors["paged_kv_indices_buffer"].copy_(paged_kv_indices)
    pk.meta_tensors["paged_kv_last_page_len_buffer"].fill_(8 + num_tokens)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    block_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="pk_attn_hidden")
    attn_norm_dt = pk.attach_input(attn_norm_weight.contiguous(), name="pk_attn_norm_weight")
    qkv_weight_dt = pk.attach_input(qkv_weight.contiguous(), name="pk_attn_qkv_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="pk_attn_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="pk_attn_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="pk_attn_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="pk_attn_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="pk_attn_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="pk_attn_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="pk_attn_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="pk_attn_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="pk_attn_out")
    o_proj_weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="pk_attn_o_proj_weight")
    block_out_dt = pk.attach_input(block_out, name="pk_attn_block_out")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=hidden_dt,
        output=block_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)

    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    ref_block = ref_attn @ o_proj_weight.float().transpose(0, 1)
    ref_block = ref_block + hidden_in.float()

    metrics: Dict[str, float] = {}
    for repeat_idx in range(repeats):
        _reset_pk_runtime_state(
            pk,
            active_tokens=num_tokens,
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
        )
        rmsnorm_out.zero_()
        qkv_out.zero_()
        attn_out.zero_()
        block_out.zero_()
        pk()
        torch.cuda.synchronize()
        qkv_diff = (qkv_out.float() - qkv_ref).abs()
        post_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
        post_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
        ref_attn_live = _reference_multitoken_paged_attention(
            q=qkv_out.float()[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim),
            paged_k_cache=post_k_cache,
            paged_v_cache=post_v_cache,
            qo_heads=qo_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            num_tokens=num_tokens,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
        ).reshape(num_tokens, hidden_size)
        attn_diff = (attn_out.float() - ref_attn_live).abs()
        if not torch.isfinite(block_out.float()).all():
            metrics[f"repeat_{repeat_idx}_qkv_max_abs"] = float(qkv_diff.max().item())
            metrics[f"repeat_{repeat_idx}_qkv_mean_abs"] = float(qkv_diff.mean().item())
            metrics[f"repeat_{repeat_idx}_attn_max_abs"] = float(attn_diff.max().item())
            metrics[f"repeat_{repeat_idx}_attn_mean_abs"] = float(attn_diff.mean().item())
            metrics[f"repeat_{repeat_idx}_block_max_abs"] = float("inf")
            metrics[f"repeat_{repeat_idx}_block_mean_abs"] = float("inf")
            continue
        ref_block_live = ref_attn_live @ o_proj_weight.float().transpose(0, 1)
        ref_block_live = ref_block_live + hidden_in.float()
        block_diff = (block_out.float() - ref_block_live).abs()
        metrics[f"repeat_{repeat_idx}_qkv_max_abs"] = float(qkv_diff.max().item())
        metrics[f"repeat_{repeat_idx}_qkv_mean_abs"] = float(qkv_diff.mean().item())
        metrics[f"repeat_{repeat_idx}_attn_max_abs"] = float(attn_diff.max().item())
        metrics[f"repeat_{repeat_idx}_attn_mean_abs"] = float(attn_diff.mean().item())
        metrics[f"repeat_{repeat_idx}_block_max_abs"] = float(block_diff.max().item())
        metrics[f"repeat_{repeat_idx}_block_mean_abs"] = float(block_diff.mean().item())
    return metrics


def run_mirage_attention_block_pk_forward_correctness(
    *,
    seed: int = 0,
    repeats: int = 2,
) -> Dict[str, float]:
    """Run a tiny compiled Mirage PersistentKernel attention->linear block repeatedly."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64

    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)
    pk.meta_tensors["qo_indptr_buffer"].zero_()
    pk.meta_tensors["qo_indptr_buffer"][:2] = qo_indptr
    pk.meta_tensors["paged_kv_indptr_buffer"].zero_()
    pk.meta_tensors["paged_kv_indptr_buffer"][:2] = paged_kv_indptr
    pk.meta_tensors["paged_kv_indices_buffer"].copy_(paged_kv_indices)
    pk.meta_tensors["paged_kv_last_page_len_buffer"].fill_(8 + num_tokens)

    qkv_in = (
        torch.randn(
            (num_tokens, (qo_heads + 2 * kv_heads) * head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    block_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    qkv_dt = pk.attach_input(qkv_in.contiguous(), name="pk_attn_block_qkv")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="pk_attn_block_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="pk_attn_block_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="pk_attn_block_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="pk_attn_block_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="pk_attn_block_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="pk_attn_block_v_cache")
    residual_dt = pk.attach_input(residual.contiguous(), name="pk_attn_block_residual")
    o_proj_weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="pk_attn_block_o_proj")
    attn_out_dt = pk.attach_input(attn_out, name="pk_attn_block_attn_out")
    block_out_dt = pk.attach_input(block_out, name="pk_attn_block_out")

    pk.paged_attention_layer(
        input=qkv_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=residual_dt,
        output=block_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)

    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_in[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_in[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_in[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    ref_block = ref_attn @ o_proj_weight.float().transpose(0, 1)
    ref_block = ref_block + residual.float()

    metrics: Dict[str, float] = {}
    for repeat_idx in range(repeats):
        _reset_pk_runtime_state(
            pk,
            active_tokens=num_tokens,
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
        )
        attn_out.zero_()
        block_out.zero_()
        pk()
        torch.cuda.synchronize()
        post_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
        post_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
        ref_attn_live = _reference_multitoken_paged_attention(
            q=qkv_in.float()[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim),
            paged_k_cache=post_k_cache,
            paged_v_cache=post_v_cache,
            qo_heads=qo_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            num_tokens=num_tokens,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
        ).reshape(num_tokens, hidden_size)
        attn_diff = (attn_out.float() - ref_attn_live).abs()
        if not torch.isfinite(block_out.float()).all():
            metrics[f"repeat_{repeat_idx}_attn_max_abs"] = float(attn_diff.max().item())
            metrics[f"repeat_{repeat_idx}_attn_mean_abs"] = float(attn_diff.mean().item())
            metrics[f"repeat_{repeat_idx}_block_max_abs"] = float("inf")
            metrics[f"repeat_{repeat_idx}_block_mean_abs"] = float("inf")
            continue
        ref_block_live = ref_attn_live @ o_proj_weight.float().transpose(0, 1)
        ref_block_live = ref_block_live + residual.float()
        block_diff = (block_out.float() - ref_block_live).abs()
        metrics[f"repeat_{repeat_idx}_attn_max_abs"] = float(attn_diff.max().item())
        metrics[f"repeat_{repeat_idx}_attn_mean_abs"] = float(attn_diff.mean().item())
        metrics[f"repeat_{repeat_idx}_block_max_abs"] = float(block_diff.max().item())
        metrics[f"repeat_{repeat_idx}_block_mean_abs"] = float(block_diff.mean().item())
    return metrics


def run_mirage_moe_silu_block_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a live Mirage SiLU-style MoE block and compare against torch."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_experts = 128
    topk = 8
    batch = 1
    hidden = 256
    intermediate = 64

    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=batch,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )

    hidden_in = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16) / 8
    router_weight = torch.randn((num_experts, hidden), device="cuda", dtype=torch.bfloat16) / 16
    router_logits = torch.zeros((batch, num_experts), device="cuda", dtype=torch.bfloat16)
    topk_weight = torch.zeros((batch, topk), device="cuda", dtype=torch.float32)
    routing_indices = torch.zeros((num_experts, batch), device="cuda", dtype=torch.int32)
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)
    moe_w13_weight = (
        torch.randn((num_experts, 2 * intermediate, hidden), device="cuda", dtype=torch.bfloat16)
        / 16
    )
    moe_w13_out = torch.zeros((batch, topk, 2 * intermediate), device="cuda", dtype=torch.bfloat16)
    moe_act_out = torch.zeros((batch, topk, intermediate), device="cuda", dtype=torch.bfloat16)
    moe_w2_weight = (
        torch.randn((num_experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16) / 16
    )
    moe_w2_out = torch.zeros((batch, topk, hidden), device="cuda", dtype=torch.bfloat16)
    hidden_out = torch.zeros((batch, hidden), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="dbg_moe_hidden")
    router_weight_dt = pk.attach_input(router_weight.contiguous(), name="dbg_moe_router_weight")
    router_logits_dt = pk.attach_input(router_logits, name="dbg_moe_router_logits")
    topk_weight_dt = pk.attach_input(topk_weight, name="dbg_moe_topk_weight")
    routing_indices_dt = pk.attach_input(routing_indices, name="dbg_moe_routing_indices")
    routing_mask_dt = pk.attach_input(routing_mask, name="dbg_moe_routing_mask")
    moe_w13_weight_dt = pk.attach_input(moe_w13_weight.contiguous(), name="dbg_moe_w13_weight")
    moe_w13_out_dt = pk.attach_input(moe_w13_out, name="dbg_moe_w13_out")
    moe_act_out_dt = pk.attach_input(moe_act_out, name="dbg_moe_act_out")
    moe_w2_weight_dt = pk.attach_input(moe_w2_weight.contiguous(), name="dbg_moe_w2_weight")
    moe_w2_out_dt = pk.attach_input(moe_w2_out, name="dbg_moe_w2_out")
    hidden_out_dt = pk.attach_input(hidden_out, name="dbg_moe_hidden_out")

    pk.linear_layer(
        input=hidden_dt,
        weight=router_weight_dt,
        output=router_logits_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_topk_softmax_routing_layer(
        input=router_logits_dt,
        output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_w13_linear_layer(
        input=hidden_dt,
        weight=moe_w13_weight_dt,
        moe_routing_indices=routing_indices_dt,
        moe_mask=routing_mask_dt,
        output=moe_w13_out_dt,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=True),
        block_dim=(128, 1, 1),
    )
    pk.moe_silu_mul_layer(
        input=moe_w13_out_dt,
        output=moe_act_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_w2_linear_layer(
        input=moe_act_out_dt,
        weight=moe_w2_weight_dt,
        moe_routing_indices=routing_indices_dt,
        moe_mask=routing_mask_dt,
        output=moe_w2_out_dt,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    pk.moe_mul_sum_add_layer(
        input=moe_w2_out_dt,
        weight=topk_weight_dt,
        residual=hidden_dt,
        output=hidden_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(pk, active_tokens=batch)
    pk()
    torch.cuda.synchronize()

    if not all(
        torch.isfinite(t.float()).all()
        for t in (router_logits, topk_weight, moe_w13_out, moe_act_out, moe_w2_out, hidden_out)
    ):
        return {
            "topk_weight_max_abs": float("inf"),
            "topk_weight_mean_abs": float("inf"),
            "routing_prefix_matches": -1.0,
            "routing_overlap_count": -1.0,
            "w2_max_abs": float("inf"),
            "w2_mean_abs": float("inf"),
            "out_max_abs": float("inf"),
            "out_mean_abs": float("inf"),
        }

    ref_logits = hidden_in.float() @ router_weight.float().transpose(0, 1)
    ref_topk_values, ref_topk_indices = torch.topk(ref_logits, k=topk, dim=-1)
    ref_topk_weight = torch.softmax(ref_topk_values, dim=-1)

    ref_moe_w13 = []
    ref_moe_w2 = []
    for batch_idx in range(batch):
        w13_rows = []
        w2_rows = []
        for route_idx in range(topk):
            expert_index = int(ref_topk_indices[batch_idx, route_idx].item())
            expert_w13 = moe_w13_weight[expert_index].float()
            expert_w2 = moe_w2_weight[expert_index].float()
            w13_row = hidden_in[batch_idx].float() @ expert_w13.transpose(0, 1)
            act_row = _expert_gated_activation(w13_row.unsqueeze(0), act_fn="silu").squeeze(0)
            w2_row = act_row @ expert_w2.transpose(0, 1)
            w13_rows.append(w13_row)
            w2_rows.append(w2_row)
        ref_moe_w13.append(torch.stack(w13_rows, dim=0))
        ref_moe_w2.append(torch.stack(w2_rows, dim=0))
    ref_moe_w13 = torch.stack(ref_moe_w13, dim=0)
    ref_moe_act = _expert_gated_activation(ref_moe_w13, act_fn="silu")
    ref_moe_w2 = torch.stack(ref_moe_w2, dim=0)
    ref_hidden_out = (ref_moe_w2 * ref_topk_weight.unsqueeze(-1)).sum(dim=1) + hidden_in.float()

    selected_experts = []
    for expert_index in range(num_experts):
        rank = int(routing_indices[expert_index, 0].item())
        if rank != 0:
            selected_experts.append((expert_index, rank))
    selected_experts = sorted(selected_experts, key=lambda item: item[1])
    selected_ids = [expert for expert, _ in selected_experts[:topk]]
    ref_ids = ref_topk_indices[0].cpu().tolist()
    prefix_matches = sum(
        1 for actual_expert, ref_expert in zip(selected_ids, ref_ids) if actual_expert == ref_expert
    )
    overlap_count = len(set(selected_ids) & set(ref_ids))

    topk_weight_diff = (topk_weight.float() - ref_topk_weight).abs()
    moe_w2_diff = (moe_w2_out.float() - ref_moe_w2).abs()
    hidden_out_diff = (hidden_out.float() - ref_hidden_out).abs()
    return {
        "topk_weight_max_abs": float(topk_weight_diff.max().item()),
        "topk_weight_mean_abs": float(topk_weight_diff.mean().item()),
        "routing_prefix_matches": float(prefix_matches),
        "routing_overlap_count": float(overlap_count),
        "w13_max_abs": float((moe_w13_out.float() - ref_moe_w13).abs().max().item()),
        "w13_mean_abs": float((moe_w13_out.float() - ref_moe_w13).abs().mean().item()),
        "act_max_abs": float((moe_act_out.float() - ref_moe_act).abs().max().item()),
        "act_mean_abs": float((moe_act_out.float() - ref_moe_act).abs().mean().item()),
        "w2_max_abs": float(moe_w2_diff.max().item()),
        "w2_mean_abs": float(moe_w2_diff.mean().item()),
        "out_max_abs": float(hidden_out_diff.max().item()),
        "out_mean_abs": float(hidden_out_diff.mean().item()),
    }


def run_mirage_moe_gelu_fused_block_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run the fused Gemma-style GELU MoE block and compare against torch."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    batch = 1
    hidden = 2816
    intermediate = 256
    num_experts = 128
    topk = 8

    hidden_in = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16) / 8
    moe_up_weight = (
        torch.randn((num_experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16) / 16
    )
    moe_gate_weight = (
        torch.randn((num_experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16) / 16
    )
    moe_w13_weight = torch.cat((moe_up_weight, moe_gate_weight), dim=1).contiguous()
    moe_w2_weight = (
        torch.randn((num_experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16) / 16
    )

    ref_logits = torch.randn((batch, num_experts), device="cuda", dtype=torch.float32)
    ref_topk_values, ref_topk_indices = torch.topk(ref_logits, k=topk, dim=-1)
    topk_weight = torch.softmax(ref_topk_values, dim=-1).contiguous()
    routing_indices = torch.zeros((num_experts, batch), device="cuda", dtype=torch.int32)
    selected_experts = ref_topk_indices[0].tolist()
    for route_idx, expert_idx in enumerate(selected_experts):
        routing_indices[expert_idx, 0] = route_idx + 1
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)
    for mask_idx, expert_idx in enumerate(sorted(selected_experts)):
        routing_mask[mask_idx] = expert_idx
    routing_mask[num_experts] = len(selected_experts)

    executor = _MirageFusedMoeExecutor(
        capacity=batch,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=num_experts,
        topk=topk,
    )
    hidden_out = executor(
        hidden_in,
        moe_w13_weight,
        routing_indices,
        routing_mask,
        moe_w2_weight,
        topk_weight,
    )
    torch.cuda.synchronize()

    ref_moe_act = []
    ref_moe_act_swapped = []
    ref_moe_w2 = []
    ref_moe_w2_swapped = []
    for expert_idx in selected_experts:
        fused = hidden_in[0].float() @ moe_w13_weight[expert_idx].float().transpose(0, 1)
        up, gate = torch.chunk(fused, 2, dim=-1)
        act = F.gelu(gate) * up
        swapped_act = F.gelu(up) * gate
        ref_moe_act.append(act)
        ref_moe_act_swapped.append(swapped_act)
        ref_moe_w2.append(act @ moe_w2_weight[expert_idx].float().transpose(0, 1))
        ref_moe_w2_swapped.append(swapped_act @ moe_w2_weight[expert_idx].float().transpose(0, 1))
    ref_moe_act = torch.stack(ref_moe_act, dim=0).unsqueeze(0)
    ref_moe_act_swapped = torch.stack(ref_moe_act_swapped, dim=0).unsqueeze(0)
    ref_moe_w2 = torch.stack(ref_moe_w2, dim=0).unsqueeze(0)
    ref_moe_w2_swapped = torch.stack(ref_moe_w2_swapped, dim=0).unsqueeze(0)
    ref_hidden_out = (ref_moe_w2 * topk_weight.unsqueeze(-1)).sum(dim=1)
    ref_hidden_out_swapped = (ref_moe_w2_swapped * topk_weight.unsqueeze(-1)).sum(dim=1)
    sorted_experts = sorted(selected_experts)
    expert_to_rank = {expert_idx: rank for rank, expert_idx in enumerate(selected_experts)}
    sorted_rank_indices = [expert_to_rank[expert_idx] for expert_idx in sorted_experts]
    ref_moe_act_sorted = ref_moe_act[:, sorted_rank_indices]
    ref_moe_w2_sorted = ref_moe_w2[:, sorted_rank_indices]

    act_diff = (executor.moe_act.float() - ref_moe_act).abs()
    w2_diff = (executor.moe_w2_out.float() - ref_moe_w2).abs()
    out_diff = (hidden_out.float() - ref_hidden_out).abs()
    swapped_out_diff = (hidden_out.float() - ref_hidden_out_swapped).abs()
    return {
        "act_max_abs": float(act_diff.max().item()),
        "act_mean_abs": float(act_diff.mean().item()),
        "act_swapped_max_abs": float(
            (executor.moe_act.float() - ref_moe_act_swapped).abs().max().item()
        ),
        "act_swapped_mean_abs": float(
            (executor.moe_act.float() - ref_moe_act_swapped).abs().mean().item()
        ),
        "act_sorted_max_abs": float(
            (executor.moe_act.float() - ref_moe_act_sorted).abs().max().item()
        ),
        "act_sorted_mean_abs": float(
            (executor.moe_act.float() - ref_moe_act_sorted).abs().mean().item()
        ),
        "w2_max_abs": float(w2_diff.max().item()),
        "w2_mean_abs": float(w2_diff.mean().item()),
        "w2_sorted_max_abs": float(
            (executor.moe_w2_out.float() - ref_moe_w2_sorted).abs().max().item()
        ),
        "w2_sorted_mean_abs": float(
            (executor.moe_w2_out.float() - ref_moe_w2_sorted).abs().mean().item()
        ),
        "out_max_abs": float(out_diff.max().item()),
        "out_mean_abs": float(out_diff.mean().item()),
        "out_swapped_max_abs": float(swapped_out_diff.max().item()),
        "out_swapped_mean_abs": float(swapped_out_diff.mean().item()),
    }


def run_mirage_moe_gelu_split_block_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a live Gemma-style GELU-gated MoE block via split expert linears."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    import mirage

    torch.manual_seed(seed)
    num_experts = 128
    topk = 8
    batch = 1
    hidden = 256
    intermediate = 64

    def _run_gelu_mul_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        graph = mirage.new_kernel_graph()
        gate_in = graph.new_input(tuple(gate.shape), dtype=mirage.bfloat16)
        up_in = graph.new_input(tuple(up.shape), dtype=mirage.bfloat16)
        out = graph.mul(graph.gelu(gate_in), up_in)
        graph.mark_output(out)
        return graph(inputs=[gate, up], target_cc=90)[0]

    hidden_in = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16) / 8
    router_weight = torch.randn((num_experts, hidden), device="cuda", dtype=torch.bfloat16) / 16
    topk_weight = torch.zeros((batch, topk), device="cuda", dtype=torch.float32)
    routing_indices = torch.zeros((num_experts, batch), device="cuda", dtype=torch.int32)
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)
    gate_weight = (
        torch.randn((num_experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16) / 16
    )
    up_weight = (
        torch.randn((num_experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16) / 16
    )
    w2_weight = (
        torch.randn((num_experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16) / 16
    )

    gate_out = torch.zeros((batch, topk, intermediate), device="cuda", dtype=torch.bfloat16)
    up_out = torch.zeros((batch, topk, intermediate), device="cuda", dtype=torch.bfloat16)
    router_logits = torch.zeros((batch, num_experts), device="cuda", dtype=torch.bfloat16)

    # Phase 1: routing plus separate gate/up expert projections.
    phase1 = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=batch,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    hidden_dt = phase1.attach_input(hidden_in.contiguous(), name="dbg_moe_gelu_hidden")
    router_weight_dt = phase1.attach_input(
        router_weight.contiguous(), name="dbg_moe_gelu_router_weight"
    )
    router_logits_dt = phase1.attach_input(router_logits, name="dbg_moe_gelu_router_logits")
    topk_weight_dt = phase1.attach_input(topk_weight, name="dbg_moe_gelu_topk_weight")
    routing_indices_dt = phase1.attach_input(routing_indices, name="dbg_moe_gelu_routing_indices")
    routing_mask_dt = phase1.attach_input(routing_mask, name="dbg_moe_gelu_routing_mask")
    gate_weight_dt = phase1.attach_input(gate_weight.contiguous(), name="dbg_moe_gelu_gate_w")
    up_weight_dt = phase1.attach_input(up_weight.contiguous(), name="dbg_moe_gelu_up_w")
    gate_out_dt = phase1.attach_input(gate_out, name="dbg_moe_gelu_gate_out")
    up_out_dt = phase1.attach_input(up_out, name="dbg_moe_gelu_up_out")
    phase1.linear_layer(
        input=hidden_dt,
        weight=router_weight_dt,
        output=router_logits_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    phase1.moe_topk_softmax_routing_layer(
        input=router_logits_dt,
        output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    phase1.moe_w13_linear_layer(
        input=hidden_dt,
        weight=gate_weight_dt,
        moe_routing_indices=routing_indices_dt,
        moe_mask=routing_mask_dt,
        output=gate_out_dt,
        grid_dim=_moe_expert_grid_dim(phase1, w13_linear=True),
        block_dim=(128, 1, 1),
    )
    phase1.moe_w13_linear_layer(
        input=hidden_dt,
        weight=up_weight_dt,
        moe_routing_indices=routing_indices_dt,
        moe_mask=routing_mask_dt,
        output=up_out_dt,
        grid_dim=_moe_expert_grid_dim(phase1, w13_linear=True),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(phase1)
    _reset_pk_runtime_state(phase1, active_tokens=batch)
    phase1()
    torch.cuda.synchronize()

    gelu_out = _run_gelu_mul_kernel(gate_out.contiguous(), up_out.contiguous()).contiguous()

    # Phase 2: expert output projection and reduction.
    phase2 = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=batch,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    gelu_out_dt = phase2.attach_input(gelu_out, name="dbg_moe_gelu_act_out")
    topk_weight_dt_2 = phase2.attach_input(topk_weight, name="dbg_moe_gelu_topk_weight_2")
    routing_indices_dt_2 = phase2.attach_input(
        routing_indices, name="dbg_moe_gelu_routing_indices_2"
    )
    routing_mask_dt_2 = phase2.attach_input(routing_mask, name="dbg_moe_gelu_routing_mask_2")
    w2_weight_dt = phase2.attach_input(w2_weight.contiguous(), name="dbg_moe_gelu_w2_weight")
    w2_out = torch.zeros((batch, topk, hidden), device="cuda", dtype=torch.bfloat16)
    hidden_out = torch.zeros((batch, hidden), device="cuda", dtype=torch.bfloat16)
    w2_out_dt = phase2.attach_input(w2_out, name="dbg_moe_gelu_w2_out")
    hidden_residual_dt = phase2.attach_input(
        hidden_in.contiguous(), name="dbg_moe_gelu_hidden_residual"
    )
    hidden_out_dt = phase2.attach_input(hidden_out, name="dbg_moe_gelu_hidden_out")
    phase2.moe_w2_linear_layer(
        input=gelu_out_dt,
        weight=w2_weight_dt,
        moe_routing_indices=routing_indices_dt_2,
        moe_mask=routing_mask_dt_2,
        output=w2_out_dt,
        grid_dim=_moe_expert_grid_dim(phase2, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    phase2.moe_mul_sum_add_layer(
        input=w2_out_dt,
        weight=topk_weight_dt_2,
        residual=hidden_residual_dt,
        output=hidden_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(phase2)
    _reset_pk_runtime_state(phase2, active_tokens=batch)
    phase2()
    torch.cuda.synchronize()

    ref_logits = hidden_in.float() @ router_weight.float().transpose(0, 1)
    ref_topk_values, ref_topk_indices = torch.topk(ref_logits, k=topk, dim=-1)
    ref_topk_weight = torch.softmax(ref_topk_values, dim=-1)

    ref_gate = []
    ref_up = []
    ref_gelu = []
    ref_w2 = []
    for batch_idx in range(batch):
        gate_rows = []
        up_rows = []
        gelu_rows = []
        w2_rows = []
        for route_idx in range(topk):
            expert_index = int(ref_topk_indices[batch_idx, route_idx].item())
            gate_row = hidden_in[batch_idx].float() @ gate_weight[expert_index].float().transpose(
                0, 1
            )
            up_row = hidden_in[batch_idx].float() @ up_weight[expert_index].float().transpose(0, 1)
            gelu_row = F.gelu(gate_row) * up_row
            w2_row = gelu_row @ w2_weight[expert_index].float().transpose(0, 1)
            gate_rows.append(gate_row)
            up_rows.append(up_row)
            gelu_rows.append(gelu_row)
            w2_rows.append(w2_row)
        ref_gate.append(torch.stack(gate_rows, dim=0))
        ref_up.append(torch.stack(up_rows, dim=0))
        ref_gelu.append(torch.stack(gelu_rows, dim=0))
        ref_w2.append(torch.stack(w2_rows, dim=0))
    ref_gate = torch.stack(ref_gate, dim=0)
    ref_up = torch.stack(ref_up, dim=0)
    ref_gelu = torch.stack(ref_gelu, dim=0)
    ref_w2 = torch.stack(ref_w2, dim=0)
    ref_hidden_out = (ref_w2 * ref_topk_weight.unsqueeze(-1)).sum(dim=1) + hidden_in.float()

    selected_experts = []
    for expert_index in range(num_experts):
        rank = int(routing_indices[expert_index, 0].item())
        if rank != 0:
            selected_experts.append((expert_index, rank))
    selected_experts = sorted(selected_experts, key=lambda item: item[1])
    selected_ids = [expert for expert, _ in selected_experts[:topk]]
    ref_ids = ref_topk_indices[0].cpu().tolist()
    prefix_matches = sum(
        1 for actual_expert, ref_expert in zip(selected_ids, ref_ids) if actual_expert == ref_expert
    )
    overlap_count = len(set(selected_ids) & set(ref_ids))

    return {
        "topk_weight_max_abs": float((topk_weight.float() - ref_topk_weight).abs().max().item()),
        "topk_weight_mean_abs": float((topk_weight.float() - ref_topk_weight).abs().mean().item()),
        "routing_prefix_matches": float(prefix_matches),
        "routing_overlap_count": float(overlap_count),
        "gate_max_abs": float((gate_out.float() - ref_gate).abs().max().item()),
        "gate_mean_abs": float((gate_out.float() - ref_gate).abs().mean().item()),
        "up_max_abs": float((up_out.float() - ref_up).abs().max().item()),
        "up_mean_abs": float((up_out.float() - ref_up).abs().mean().item()),
        "act_max_abs": float((gelu_out.float() - ref_gelu).abs().max().item()),
        "act_mean_abs": float((gelu_out.float() - ref_gelu).abs().mean().item()),
        "w2_max_abs": float((w2_out.float() - ref_w2).abs().max().item()),
        "w2_mean_abs": float((w2_out.float() - ref_w2).abs().mean().item()),
        "out_max_abs": float((hidden_out.float() - ref_hidden_out).abs().max().item()),
        "out_mean_abs": float((hidden_out.float() - ref_hidden_out).abs().mean().item()),
    }


def run_mirage_moe_gelu_split_dense_projection_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run Gemma-style expert gate/up projections via dense Mirage linear tasks.

    This targets the exact regime where ``moe_w13_linear_layer`` is currently
    numerically unstable for Gemma-like attention-scale inputs. The purpose is
    to validate a live Mirage fallback composition using the dense
    ``linear_layer`` task, one expert at a time, in top-k rank order.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    num_experts = 128
    topk = 8
    intermediate = 64

    # Build one attention-like residual stream so we test the actual regime that
    # breaks the fused MoE expert kernel, rather than only tiny random inputs.
    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)
    pk.meta_tensors["qo_indptr_buffer"].zero_()
    pk.meta_tensors["qo_indptr_buffer"][:2] = qo_indptr
    pk.meta_tensors["paged_kv_indptr_buffer"].zero_()
    pk.meta_tensors["paged_kv_indptr_buffer"][:2] = paged_kv_indptr
    pk.meta_tensors["paged_kv_indices_buffer"].copy_(paged_kv_indices)
    pk.meta_tensors["paged_kv_last_page_len_buffer"].fill_(8 + num_tokens)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="dbg_split_dense_hidden")
    attn_norm_dt = pk.attach_input(attn_norm_weight.contiguous(), name="dbg_split_dense_attn_norm")
    qkv_weight_dt = pk.attach_input(qkv_weight.contiguous(), name="dbg_split_dense_qkv_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="dbg_split_dense_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="dbg_split_dense_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="dbg_split_dense_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="dbg_split_dense_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="dbg_split_dense_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="dbg_split_dense_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="dbg_split_dense_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="dbg_split_dense_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="dbg_split_dense_attn_out")
    o_proj_weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="dbg_split_dense_o_proj")
    post_attn_dt = pk.attach_input(post_attn, name="dbg_split_dense_post_attn")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=hidden_dt,
        output=post_attn_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(
        pk,
        active_tokens=num_tokens,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
    pk()
    torch.cuda.synchronize()

    # Use the reference attention residual so we isolate expert-projection
    # correctness from the already-accepted attention approximation.
    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + 1e-5)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    post_attn_ref = (ref_attn @ o_proj_weight.float().transpose(0, 1) + hidden_in.float()).to(
        torch.bfloat16
    )

    router_weight = (
        torch.randn((num_experts, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    gate_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    up_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    logits = post_attn_ref.float() @ router_weight.float().transpose(0, 1)
    _, topk_indices = torch.topk(logits, k=topk, dim=-1)
    experts = topk_indices[0].tolist()

    expert_pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    expert_hidden_dt = expert_pk.attach_input(
        post_attn_ref.contiguous(), name="dbg_split_dense_expert_hidden"
    )
    gate_outs = []
    up_outs = []
    for rank, expert in enumerate(experts):
        gate_weight_dt = expert_pk.attach_input(
            gate_weight[expert].contiguous(), name=f"dbg_split_dense_gate_w_{rank}"
        )
        up_weight_dt = expert_pk.attach_input(
            up_weight[expert].contiguous(), name=f"dbg_split_dense_up_w_{rank}"
        )
        gate_out = torch.zeros((1, intermediate), device="cuda", dtype=torch.bfloat16)
        up_out = torch.zeros((1, intermediate), device="cuda", dtype=torch.bfloat16)
        gate_out_dt = expert_pk.attach_input(gate_out, name=f"dbg_split_dense_gate_out_{rank}")
        up_out_dt = expert_pk.attach_input(up_out, name=f"dbg_split_dense_up_out_{rank}")
        expert_pk.linear_layer(
            input=expert_hidden_dt,
            weight=gate_weight_dt,
            output=gate_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        expert_pk.linear_layer(
            input=expert_hidden_dt,
            weight=up_weight_dt,
            output=up_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        gate_outs.append(gate_out)
        up_outs.append(up_out)

    compile_persistent_kernel_with_patches(expert_pk)
    _reset_pk_runtime_state(expert_pk, active_tokens=1)
    expert_pk()
    torch.cuda.synchronize()

    gate_live = torch.stack([tensor[0].float() for tensor in gate_outs], dim=0).unsqueeze(0)
    up_live = torch.stack([tensor[0].float() for tensor in up_outs], dim=0).unsqueeze(0)
    gate_ref = torch.stack(
        [
            post_attn_ref[0].float() @ gate_weight[expert].float().transpose(0, 1)
            for expert in experts
        ],
        dim=0,
    ).unsqueeze(0)
    up_ref = torch.stack(
        [
            post_attn_ref[0].float() @ up_weight[expert].float().transpose(0, 1)
            for expert in experts
        ],
        dim=0,
    ).unsqueeze(0)
    return {
        "post_attn_max_abs": float(post_attn_ref.float().abs().max().item()),
        "post_attn_mean_abs": float(post_attn_ref.float().abs().mean().item()),
        "gate_max_abs": float((gate_live - gate_ref).abs().max().item()),
        "gate_mean_abs": float((gate_live - gate_ref).abs().mean().item()),
        "up_max_abs": float((up_live - up_ref).abs().max().item()),
        "up_mean_abs": float((up_live - up_ref).abs().mean().item()),
    }


def run_mirage_moe_split_dense_w2_reduce_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run the second half of a split-dense Gemma-style MoE block live.

    This validates the practical workaround path after the gate/up projections:
    per-expert dense ``linear_layer`` calls for ``w2`` followed by the live
    Mirage ``moe_mul_sum_add_layer`` reduction.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    batch = 1
    topk = 8
    hidden = 512
    intermediate = 64
    num_experts = 128

    experts = [3, 7, 12, 18, 44, 79, 95, 111]
    act_in = torch.randn((batch, topk, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0
    topk_weight = torch.softmax(
        torch.randn((batch, topk), device="cuda", dtype=torch.float32), dim=-1
    )
    residual = torch.randn((batch, hidden), device="cuda", dtype=torch.bfloat16) / 8.0
    w2_weight = (
        torch.randn((num_experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    phase = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    w2_outs = []
    for rank, expert in enumerate(experts):
        act_rank = torch.empty((batch, intermediate), device="cuda", dtype=torch.bfloat16)
        act_rank.copy_(act_in[:, rank, :])
        act_dt = phase.attach_input(act_rank, name=f"dbg_split_dense_w2_act_{rank}")
        w2_weight_dt = phase.attach_input(
            w2_weight[expert].contiguous(), name=f"dbg_split_dense_w2_w_{rank}"
        )
        w2_out = torch.zeros((batch, hidden), device="cuda", dtype=torch.bfloat16)
        w2_out_dt = phase.attach_input(w2_out, name=f"dbg_split_dense_w2_out_{rank}")
        phase.linear_layer(
            input=act_dt,
            weight=w2_weight_dt,
            output=w2_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        w2_outs.append(w2_out)

    compile_persistent_kernel_with_patches(phase)
    _reset_pk_runtime_state(phase, active_tokens=1)
    phase()
    torch.cuda.synchronize()
    w2_live = (
        torch.stack([tensor[0].float() for tensor in w2_outs], dim=0).unsqueeze(0).contiguous()
    )

    reduce_pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    w2_live_dt = reduce_pk.attach_input(
        w2_live.to(torch.bfloat16), name="dbg_split_dense_reduce_w2"
    )
    topk_weight_dt = reduce_pk.attach_input(topk_weight, name="dbg_split_dense_reduce_topk")
    residual_dt = reduce_pk.attach_input(residual.contiguous(), name="dbg_split_dense_reduce_res")
    out = torch.zeros((batch, hidden), device="cuda", dtype=torch.bfloat16)
    out_dt = reduce_pk.attach_input(out, name="dbg_split_dense_reduce_out")
    reduce_pk.moe_mul_sum_add_layer(
        input=w2_live_dt,
        weight=topk_weight_dt,
        residual=residual_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(reduce_pk)
    _reset_pk_runtime_state(reduce_pk, active_tokens=1)
    reduce_pk()
    torch.cuda.synchronize()

    w2_ref = torch.stack(
        [
            act_in[0, rank].float() @ w2_weight[expert].float().transpose(0, 1)
            for rank, expert in enumerate(experts)
        ],
        dim=0,
    ).unsqueeze(0)
    out_ref = (w2_ref * topk_weight.unsqueeze(-1)).sum(dim=1) + residual.float()
    return {
        "w2_max_abs": float((w2_live - w2_ref).abs().max().item()),
        "w2_mean_abs": float((w2_live - w2_ref).abs().mean().item()),
        "out_max_abs": float((out.float() - out_ref).abs().max().item()),
        "out_mean_abs": float((out.float() - out_ref).abs().mean().item()),
    }


def run_mirage_ffn_down_projection_forward_correctness(
    *,
    seed: int = 0,
    grid_dim_x: int = 1,
    use_generic_activation_input: bool = True,
    repack_after_activation: bool = True,
) -> Dict[str, float]:
    """Run the Gemma FFN down projection as a standalone live PK linear stage."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    intermediate = 64
    hidden = 512

    if use_generic_activation_input:
        import mirage

        graph = mirage.new_kernel_graph()
        gate = torch.randn((num_tokens, 1, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0
        up = torch.randn((num_tokens, 1, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0
        gate_in = graph.new_input(tuple(gate.shape), dtype=mirage.bfloat16)
        up_in = graph.new_input(tuple(up.shape), dtype=mirage.bfloat16)
        out = graph.mul(graph.gelu(gate_in), up_in)
        graph.mark_output(out)
        act_in = graph(inputs=[gate, up], target_cc=90)[0].squeeze(1).contiguous()
    else:
        act_in = torch.randn((num_tokens, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0

    if repack_after_activation:
        packed = torch.empty_like(act_in)
        packed.copy_(act_in)
        act_in = packed

    weight = torch.randn((hidden, intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    out = torch.zeros((num_tokens, hidden), device="cuda", dtype=torch.bfloat16)
    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    act_dt = pk.attach_input(act_in, name="dbg_ffn_down_act")
    weight_dt = pk.attach_input(weight.contiguous(), name="dbg_ffn_down_weight")
    out_dt = pk.attach_input(out, name="dbg_ffn_down_out")
    pk.linear_layer(
        input=act_dt,
        weight=weight_dt,
        output=out_dt,
        grid_dim=(grid_dim_x, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(pk, active_tokens=1)
    pk()
    torch.cuda.synchronize()

    ref = act_in.float() @ weight.float().transpose(0, 1)
    return {
        "out_max_abs": float((out.float() - ref).abs().max().item()),
        "out_mean_abs": float((out.float() - ref).abs().mean().item()),
    }


def run_mirage_ffn_down_via_moe_w2_forward_correctness(
    *,
    seed: int = 0,
    use_generic_activation_input: bool = True,
    repack_after_activation: bool = True,
) -> Dict[str, float]:
    """Run FFN down projection through live ``moe_w2`` + reduction with top-k 1."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    topk = 1
    num_experts = 1
    intermediate = 64
    hidden = 512

    if use_generic_activation_input:
        import mirage

        graph = mirage.new_kernel_graph()
        gate = torch.randn((num_tokens, 1, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0
        up = torch.randn((num_tokens, 1, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0
        gate_in = graph.new_input(tuple(gate.shape), dtype=mirage.bfloat16)
        up_in = graph.new_input(tuple(up.shape), dtype=mirage.bfloat16)
        out = graph.mul(graph.gelu(gate_in), up_in)
        graph.mark_output(out)
        act_in = graph(inputs=[gate, up], target_cc=90)[0].contiguous()
    else:
        act_in = (
            torch.randn((num_tokens, topk, intermediate), device="cuda", dtype=torch.bfloat16) / 8.0
        )

    if repack_after_activation:
        packed = torch.empty_like(act_in)
        packed.copy_(act_in)
        act_in = packed

    weight = (
        torch.randn((num_experts, hidden, intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    routing_indices = torch.zeros((num_experts, num_tokens), device="cuda", dtype=torch.int32)
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)
    routing_indices[0, 0] = 1
    routing_mask[0] = 0
    routing_mask[num_experts] = 1
    topk_weight = torch.ones((num_tokens, topk), device="cuda", dtype=torch.float32)
    residual = torch.zeros((num_tokens, hidden), device="cuda", dtype=torch.bfloat16)
    w2_out = torch.zeros((num_tokens, topk, hidden), device="cuda", dtype=torch.bfloat16)
    out = torch.zeros((num_tokens, hidden), device="cuda", dtype=torch.bfloat16)

    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    act_dt = pk.attach_input(act_in, name="dbg_ffn_moe_w2_act")
    weight_dt = pk.attach_input(weight.contiguous(), name="dbg_ffn_moe_w2_weight")
    routing_indices_dt = pk.attach_input(routing_indices, name="dbg_ffn_moe_w2_routing_indices")
    routing_mask_dt = pk.attach_input(routing_mask, name="dbg_ffn_moe_w2_routing_mask")
    topk_weight_dt = pk.attach_input(topk_weight, name="dbg_ffn_moe_w2_topk_weight")
    residual_dt = pk.attach_input(residual, name="dbg_ffn_moe_w2_residual")
    w2_out_dt = pk.attach_input(w2_out, name="dbg_ffn_moe_w2_out")
    out_dt = pk.attach_input(out, name="dbg_ffn_moe_w2_hidden_out")

    pk.moe_w2_linear_layer(
        input=act_dt,
        weight=weight_dt,
        moe_routing_indices=routing_indices_dt,
        moe_mask=routing_mask_dt,
        output=w2_out_dt,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    pk.moe_mul_sum_add_layer(
        input=w2_out_dt,
        weight=topk_weight_dt,
        residual=residual_dt,
        output=out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(pk, active_tokens=1)
    pk()
    torch.cuda.synchronize()

    ref = act_in[:, 0, :].float() @ weight[0].float().transpose(0, 1)
    return {
        "w2_max_abs": float((w2_out[:, 0, :].float() - ref).abs().max().item()),
        "w2_mean_abs": float((w2_out[:, 0, :].float() - ref).abs().mean().item()),
        "out_max_abs": float((out.float() - ref).abs().max().item()),
        "out_mean_abs": float((out.float() - ref).abs().mean().item()),
    }


def run_mirage_gemma_decode_ffn_down_direct_matmul_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run the exact Gemma decode FFN-down shape through direct Mirage matmul.

    This mirrors the historical direct decode path: ``(1, 2112) x (2112, 2816)``.
    It is intended to be exercised in a subprocess because the current runtime
    behavior may terminate the process instead of returning a clean numerical
    result.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    act = torch.randn((1, 2112), device="cuda", dtype=torch.bfloat16) / 8.0
    weight = torch.randn((2816, 2112), device="cuda", dtype=torch.bfloat16) / 16.0
    ref = act.float() @ weight.float().transpose(0, 1)

    executor = _MirageMatmulExecutor(m=1, k=2112, n=2816)
    out = executor(act, weight.transpose(0, 1).contiguous())
    torch.cuda.synchronize()

    diff = (out.float() - ref).abs()
    return {
        "out_max_abs": float(diff.max().item()),
        "out_mean_abs": float(diff.mean().item()),
    }


def run_mirage_gemma_decode_ffn_down_via_moe_w2_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run the exact Gemma decode FFN-down shape through the ``moe_w2`` workaround."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    act_flat = torch.randn((1, 2112), device="cuda", dtype=torch.bfloat16) / 8.0
    weight_flat = torch.randn((2816, 2112), device="cuda", dtype=torch.bfloat16) / 16.0
    act = act_flat.view(1, 1, 2112).contiguous()
    weight = weight_flat.unsqueeze(0).contiguous()
    routing_indices = torch.ones((1, 1), device="cuda", dtype=torch.int32)
    routing_mask = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    topk_weight = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    residual = torch.zeros((1, 2816), device="cuda", dtype=torch.bfloat16)
    ref = act_flat.float() @ weight_flat.float().transpose(0, 1)

    executor = _MirageMoeW2ReduceExecutor(
        capacity=1,
        topk=1,
        hidden_size=2816,
        intermediate_size=2112,
        num_experts=1,
    )
    out = executor(act, weight, routing_indices, routing_mask, topk_weight, residual)
    torch.cuda.synchronize()

    diff = (out.float() - ref).abs()
    return {
        "out_max_abs": float(diff.max().item()),
        "out_mean_abs": float(diff.mean().item()),
    }


def run_mirage_moe_gelu_split_dense_block_forward_correctness(
    *,
    seed: int = 0,
) -> Dict[str, float]:
    """Run a live Gemma-style MoE block with split-dense gate/up only.

    This is the current best live Mirage composition for the Gemma decode path:
    live routing, dense per-expert gate/up projections, live GELU*mul,
    fused live ``moe_w2_linear_layer``, and live ``moe_mul_sum_add_layer``.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    import mirage

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    num_experts = 128
    topk = 8
    intermediate = 64

    def _run_gelu_mul_kernel(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        graph = mirage.new_kernel_graph()
        gate_in = graph.new_input(tuple(gate.shape), dtype=mirage.bfloat16)
        up_in = graph.new_input(tuple(up.shape), dtype=mirage.bfloat16)
        out = graph.mul(graph.gelu(gate_in), up_in)
        graph.mark_output(out)
        return graph(inputs=[gate, up], target_cc=90)[0]

    # Recreate the attention-scale residual stream used in the dense projection
    # test so the MoE block sees the exact regime that breaks fused ``w13``.
    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)
    pk.meta_tensors["qo_indptr_buffer"].zero_()
    pk.meta_tensors["qo_indptr_buffer"][:2] = qo_indptr
    pk.meta_tensors["paged_kv_indptr_buffer"].zero_()
    pk.meta_tensors["paged_kv_indptr_buffer"][:2] = paged_kv_indptr
    pk.meta_tensors["paged_kv_indices_buffer"].copy_(paged_kv_indices)
    pk.meta_tensors["paged_kv_last_page_len_buffer"].fill_(8 + num_tokens)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="dbg_split_dense_block_hidden")
    attn_norm_dt = pk.attach_input(
        attn_norm_weight.contiguous(), name="dbg_split_dense_block_attn_norm"
    )
    qkv_weight_dt = pk.attach_input(
        qkv_weight.contiguous(), name="dbg_split_dense_block_qkv_weight"
    )
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="dbg_split_dense_block_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="dbg_split_dense_block_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="dbg_split_dense_block_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="dbg_split_dense_block_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="dbg_split_dense_block_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="dbg_split_dense_block_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="dbg_split_dense_block_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="dbg_split_dense_block_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="dbg_split_dense_block_attn_out")
    o_proj_weight_dt = pk.attach_input(
        o_proj_weight.contiguous(), name="dbg_split_dense_block_o_proj"
    )
    post_attn_dt = pk.attach_input(post_attn, name="dbg_split_dense_block_post_attn")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=hidden_dt,
        output=post_attn_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(
        pk,
        active_tokens=num_tokens,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
    pk()
    torch.cuda.synchronize()

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + 1e-5)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    post_attn_ref = (ref_attn @ o_proj_weight.float().transpose(0, 1) + hidden_in.float()).to(
        torch.bfloat16
    )

    router_weight = (
        torch.randn((num_experts, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    gate_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    up_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    w2_weight = (
        torch.randn((num_experts, hidden_size, intermediate), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    router_logits = torch.zeros((num_tokens, num_experts), device="cuda", dtype=torch.bfloat16)
    topk_weight = torch.zeros((num_tokens, topk), device="cuda", dtype=torch.float32)
    routing_indices = torch.zeros((num_experts, num_tokens), device="cuda", dtype=torch.int32)
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)

    # Phase 1: live routing.
    router_pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    router_hidden_dt = router_pk.attach_input(
        post_attn_ref.contiguous(), name="dbg_split_dense_block_router_hidden"
    )
    router_weight_dt = router_pk.attach_input(
        router_weight.contiguous(), name="dbg_split_dense_block_router_weight"
    )
    router_logits_dt = router_pk.attach_input(
        router_logits, name="dbg_split_dense_block_router_logits"
    )
    topk_weight_dt = router_pk.attach_input(topk_weight, name="dbg_split_dense_block_topk_weight")
    routing_indices_dt = router_pk.attach_input(
        routing_indices, name="dbg_split_dense_block_routing_indices"
    )
    routing_mask_dt = router_pk.attach_input(
        routing_mask, name="dbg_split_dense_block_routing_mask"
    )
    router_pk.linear_layer(
        input=router_hidden_dt,
        weight=router_weight_dt,
        output=router_logits_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    router_pk.moe_topk_softmax_routing_layer(
        input=router_logits_dt,
        output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(router_pk)
    _reset_pk_runtime_state(router_pk, active_tokens=1)
    router_pk()
    torch.cuda.synchronize()

    selected_experts = []
    for expert_index in range(num_experts):
        rank = int(routing_indices[expert_index, 0].item())
        if rank != 0:
            selected_experts.append((expert_index, rank))
    selected_experts = sorted(selected_experts, key=lambda item: item[1])
    experts = [expert for expert, _ in selected_experts[:topk]]

    # Phase 2: dense gate/up expert projections.
    expert_pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    expert_hidden_dt = expert_pk.attach_input(
        post_attn_ref.contiguous(), name="dbg_split_dense_block_expert_hidden"
    )
    gate_outs = []
    up_outs = []
    for rank, expert in enumerate(experts):
        gate_weight_dt = expert_pk.attach_input(
            gate_weight[expert].contiguous(), name=f"dbg_split_dense_block_gate_w_{rank}"
        )
        up_weight_dt = expert_pk.attach_input(
            up_weight[expert].contiguous(), name=f"dbg_split_dense_block_up_w_{rank}"
        )
        gate_out = torch.zeros((1, intermediate), device="cuda", dtype=torch.bfloat16)
        up_out = torch.zeros((1, intermediate), device="cuda", dtype=torch.bfloat16)
        gate_out_dt = expert_pk.attach_input(
            gate_out, name=f"dbg_split_dense_block_gate_out_{rank}"
        )
        up_out_dt = expert_pk.attach_input(up_out, name=f"dbg_split_dense_block_up_out_{rank}")
        expert_pk.linear_layer(
            input=expert_hidden_dt,
            weight=gate_weight_dt,
            output=gate_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        expert_pk.linear_layer(
            input=expert_hidden_dt,
            weight=up_weight_dt,
            output=up_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        gate_outs.append(gate_out)
        up_outs.append(up_out)

    compile_persistent_kernel_with_patches(expert_pk)
    _reset_pk_runtime_state(expert_pk, active_tokens=1)
    expert_pk()
    torch.cuda.synchronize()

    gate_live = torch.stack([tensor[0].float() for tensor in gate_outs], dim=0).unsqueeze(0)
    up_live = torch.stack([tensor[0].float() for tensor in up_outs], dim=0).unsqueeze(0)
    gelu_out = _run_gelu_mul_kernel(
        gate_live.to(torch.bfloat16), up_live.to(torch.bfloat16)
    ).contiguous()

    # Phase 3: fused live w2 and reduction.
    phase3 = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    gelu_out_dt = phase3.attach_input(gelu_out, name="dbg_split_dense_block_act_out")
    topk_weight_dt_3 = phase3.attach_input(topk_weight, name="dbg_split_dense_block_topk_weight_3")
    routing_indices_dt_3 = phase3.attach_input(
        routing_indices, name="dbg_split_dense_block_routing_indices_3"
    )
    routing_mask_dt_3 = phase3.attach_input(
        routing_mask, name="dbg_split_dense_block_routing_mask_3"
    )
    w2_weight_dt = phase3.attach_input(
        w2_weight.contiguous(), name="dbg_split_dense_block_w2_weight"
    )
    w2_out = torch.zeros((num_tokens, topk, hidden_size), device="cuda", dtype=torch.bfloat16)
    hidden_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    w2_out_dt = phase3.attach_input(w2_out, name="dbg_split_dense_block_w2_out")
    hidden_residual_dt = phase3.attach_input(
        post_attn_ref.contiguous(), name="dbg_split_dense_block_hidden_residual"
    )
    hidden_out_dt = phase3.attach_input(hidden_out, name="dbg_split_dense_block_hidden_out")
    phase3.moe_w2_linear_layer(
        input=gelu_out_dt,
        weight=w2_weight_dt,
        moe_routing_indices=routing_indices_dt_3,
        moe_mask=routing_mask_dt_3,
        output=w2_out_dt,
        grid_dim=_moe_expert_grid_dim(phase3, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    phase3.moe_mul_sum_add_layer(
        input=w2_out_dt,
        weight=topk_weight_dt_3,
        residual=hidden_residual_dt,
        output=hidden_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(phase3)
    _reset_pk_runtime_state(phase3, active_tokens=1)
    phase3()
    torch.cuda.synchronize()

    live_router_logits = router_logits.float()
    live_topk_values, live_topk_indices = torch.topk(live_router_logits, k=topk, dim=-1)
    live_topk_weight = torch.softmax(live_topk_values, dim=-1)

    ref_logits = post_attn_ref.float() @ router_weight.float().transpose(0, 1)
    ref_topk_values, ref_topk_indices = torch.topk(ref_logits, k=topk, dim=-1)
    ref_topk_weight = torch.softmax(ref_topk_values, dim=-1)
    ref_ids = ref_topk_indices[0].cpu().tolist()

    gate_ref = torch.stack(
        [
            post_attn_ref[0].float() @ gate_weight[expert].float().transpose(0, 1)
            for expert in experts
        ],
        dim=0,
    ).unsqueeze(0)
    up_ref = torch.stack(
        [
            post_attn_ref[0].float() @ up_weight[expert].float().transpose(0, 1)
            for expert in experts
        ],
        dim=0,
    ).unsqueeze(0)
    live_gelu_ref = (F.gelu(gate_live) * up_live).contiguous()
    ref_gelu = (F.gelu(gate_ref) * up_ref).contiguous()
    ref_w2 = torch.stack(
        [
            ref_gelu[0, rank].float() @ w2_weight[expert].float().transpose(0, 1)
            for rank, expert in enumerate(experts)
        ],
        dim=0,
    ).unsqueeze(0)
    ref_hidden_out = (ref_w2 * topk_weight.unsqueeze(-1)).sum(dim=1) + post_attn_ref.float()

    selected_ids = experts
    prefix_matches = sum(
        1 for actual_expert, ref_expert in zip(selected_ids, ref_ids) if actual_expert == ref_expert
    )
    overlap_count = len(set(selected_ids) & set(ref_ids))

    return {
        "post_attn_max_abs": float(post_attn_ref.float().abs().max().item()),
        "post_attn_mean_abs": float(post_attn_ref.float().abs().mean().item()),
        "topk_weight_live_logits_max_abs": float(
            (topk_weight.float() - live_topk_weight).abs().max().item()
        ),
        "topk_weight_live_logits_mean_abs": float(
            (topk_weight.float() - live_topk_weight).abs().mean().item()
        ),
        "routing_live_logits_prefix_matches": float(
            sum(
                1
                for actual_expert, live_expert in zip(
                    selected_ids, live_topk_indices[0].cpu().tolist()
                )
                if actual_expert == live_expert
            )
        ),
        "routing_live_logits_overlap_count": float(
            len(set(selected_ids) & set(live_topk_indices[0].cpu().tolist()))
        ),
        "topk_weight_max_abs": float((topk_weight.float() - ref_topk_weight).abs().max().item()),
        "topk_weight_mean_abs": float((topk_weight.float() - ref_topk_weight).abs().mean().item()),
        "routing_prefix_matches": float(prefix_matches),
        "routing_overlap_count": float(overlap_count),
        "gate_max_abs": float((gate_live - gate_ref).abs().max().item()),
        "gate_mean_abs": float((gate_live - gate_ref).abs().mean().item()),
        "up_max_abs": float((up_live - up_ref).abs().max().item()),
        "up_mean_abs": float((up_live - up_ref).abs().mean().item()),
        "act_live_inputs_max_abs": float((gelu_out.float() - live_gelu_ref).abs().max().item()),
        "act_live_inputs_mean_abs": float((gelu_out.float() - live_gelu_ref).abs().mean().item()),
        "act_max_abs": float((gelu_out.float() - ref_gelu).abs().max().item()),
        "act_mean_abs": float((gelu_out.float() - ref_gelu).abs().mean().item()),
        "w2_max_abs": float((w2_out.float() - ref_w2).abs().max().item()),
        "w2_mean_abs": float((w2_out.float() - ref_w2).abs().mean().item()),
        "out_max_abs": float((hidden_out.float() - ref_hidden_out).abs().max().item()),
        "out_mean_abs": float((hidden_out.float() - ref_hidden_out).abs().mean().item()),
    }


def run_mirage_gemma_full_layer_split_dense_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Run a synthetic Gemma-style layer live through attention, FFN, and MoE."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    import mirage

    torch.manual_seed(seed)
    stage_timings: Dict[str, float] = {}
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    num_experts = 128
    topk = 8
    intermediate = 64
    ffn_intermediate = 256

    def _timed_call(
        stage_name: str,
        fn: Callable[[], Any],
        *,
        cuda_sync: bool = False,
    ) -> Any:
        start = time.perf_counter()
        result = fn()
        if cuda_sync:
            torch.cuda.synchronize()
        stage_timings[stage_name] = time.perf_counter() - start
        if verbose:
            print(f"{stage_name}={stage_timings[stage_name]:.3f}s", flush=True)
        return result

    def _run_gelu_mul_kernel(
        gate: torch.Tensor,
        up: torch.Tensor,
        *,
        stage_prefix: str,
    ) -> Dict[str, torch.Tensor]:
        kernel_mod = importlib.import_module("mirage.kernel")
        graph = mirage.new_kernel_graph()
        gate_in = graph.new_input(tuple(gate.shape), dtype=mirage.bfloat16)
        up_in = graph.new_input(tuple(up.shape), dtype=mirage.bfloat16)
        out = graph.mul(graph.gelu(gate_in), up_in)
        graph.mark_output(out)

        original_generate_cuda_program = kernel_mod.generate_cuda_program
        original_check_call = subprocess.check_call
        codegen_s: Optional[float] = None
        nvcc_s: Optional[float] = None

        def _timed_generate_cuda_program(*args, **kwargs):
            nonlocal codegen_s
            start = time.perf_counter()
            result = original_generate_cuda_program(*args, **kwargs)
            codegen_s = time.perf_counter() - start
            return result

        def _timed_check_call(*args, **kwargs):
            nonlocal nvcc_s
            start = time.perf_counter()
            result = original_check_call(*args, **kwargs)
            nvcc_s = time.perf_counter() - start
            return result

        compile_start = time.perf_counter()
        try:
            kernel_mod.generate_cuda_program = _timed_generate_cuda_program
            subprocess.check_call = _timed_check_call
            graph.compile(inputs=[gate, up], target_cc=90)
        finally:
            kernel_mod.generate_cuda_program = original_generate_cuda_program
            subprocess.check_call = original_check_call

        stage_timings[f"{stage_prefix}_compile_s"] = time.perf_counter() - compile_start
        if codegen_s is not None:
            stage_timings[f"{stage_prefix}_codegen_s"] = codegen_s
        if nvcc_s is not None:
            stage_timings[f"{stage_prefix}_nvcc_s"] = nvcc_s
        if verbose:
            print(
                f"{stage_prefix}_compile_s={stage_timings[f'{stage_prefix}_compile_s']:.3f}s "
                f"(codegen={codegen_s if codegen_s is not None else float('nan'):.3f}s, "
                f"nvcc={nvcc_s if nvcc_s is not None else float('nan'):.3f}s)",
                flush=True,
            )

        launch_start = time.perf_counter()
        outputs = graph(inputs=[gate, up], target_cc=90)
        torch.cuda.synchronize()
        stage_timings[f"{stage_prefix}_launch_s"] = time.perf_counter() - launch_start
        if verbose:
            print(
                f"{stage_prefix}_launch_s={stage_timings[f'{stage_prefix}_launch_s']:.3f}s",
                flush=True,
            )
        return outputs[0]

    # Attention sublayer.
    attn_pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = attn_pk.attach_input(hidden_in.contiguous(), name="full_layer_hidden")
    attn_norm_dt = attn_pk.attach_input(attn_norm_weight.contiguous(), name="full_layer_attn_norm")
    qkv_weight_dt = attn_pk.attach_input(qkv_weight.contiguous(), name="full_layer_qkv_weight")
    rmsnorm_out_dt = attn_pk.attach_input(rmsnorm_out, name="full_layer_rmsnorm_out")
    qkv_out_dt = attn_pk.attach_input(qkv_out, name="full_layer_qkv_out")
    q_norm_dt = attn_pk.attach_input(q_norm_weight.contiguous(), name="full_layer_q_norm")
    k_norm_dt = attn_pk.attach_input(k_norm_weight.contiguous(), name="full_layer_k_norm")
    cos_dt = attn_pk.attach_input(cos.contiguous(), name="full_layer_cos")
    sin_dt = attn_pk.attach_input(sin.contiguous(), name="full_layer_sin")
    k_cache_dt = attn_pk.attach_input(k_cache.contiguous(), name="full_layer_k_cache")
    v_cache_dt = attn_pk.attach_input(v_cache.contiguous(), name="full_layer_v_cache")
    attn_out_dt = attn_pk.attach_input(attn_out, name="full_layer_attn_out")
    o_proj_weight_dt = attn_pk.attach_input(o_proj_weight.contiguous(), name="full_layer_o_proj")
    post_attn_dt = attn_pk.attach_input(post_attn, name="full_layer_post_attn")

    attn_pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    attn_pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    attn_pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    attn_pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=hidden_dt,
        output=post_attn_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    _timed_call("attn_pk_compile_s", lambda: compile_persistent_kernel_with_patches(attn_pk))
    _reset_pk_runtime_state(
        attn_pk,
        active_tokens=num_tokens,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
    _timed_call("attn_pk_launch_s", attn_pk, cuda_sync=True)

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    post_attn_ref = ref_attn @ o_proj_weight.float().transpose(0, 1) + hidden_in.float()
    post_attn_live = post_attn.float()

    # Dense FFN branch from live attention output.
    ffn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_weight = (
        torch.randn((2 * ffn_intermediate, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_down_weight = (
        torch.randn((hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    ffn_phase1 = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    ffn_hidden_dt = ffn_phase1.attach_input(
        post_attn.detach().contiguous(), name="full_layer_ffn_hidden"
    )
    ffn_norm_dt = ffn_phase1.attach_input(ffn_norm_weight.contiguous(), name="full_layer_ffn_norm")
    ffn_gate_up_weight_dt = ffn_phase1.attach_input(
        ffn_gate_up_weight.contiguous(), name="full_layer_ffn_gate_up_weight"
    )
    ffn_normed = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up = torch.zeros(
        (num_tokens, 2 * ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_normed_dt = ffn_phase1.attach_input(ffn_normed, name="full_layer_ffn_normed")
    ffn_gate_up_dt = ffn_phase1.attach_input(ffn_gate_up, name="full_layer_ffn_gate_up")
    ffn_phase1.rmsnorm_layer(
        input=ffn_hidden_dt,
        weight=ffn_norm_dt,
        output=ffn_normed_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    ffn_phase1.linear_layer(
        input=ffn_normed_dt,
        weight=ffn_gate_up_weight_dt,
        output=ffn_gate_up_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    _timed_call("ffn_phase1_compile_s", lambda: compile_persistent_kernel_with_patches(ffn_phase1))
    _reset_pk_runtime_state(ffn_phase1, active_tokens=1)
    _timed_call("ffn_phase1_launch_s", ffn_phase1, cuda_sync=True)

    ffn_gate_live, ffn_up_live = torch.chunk(ffn_gate_up, 2, dim=-1)
    packed_ffn_gate = torch.empty(
        (num_tokens, ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    packed_ffn_up = torch.empty((num_tokens, ffn_intermediate), device="cuda", dtype=torch.bfloat16)
    packed_ffn_gate.copy_(ffn_gate_live)
    packed_ffn_up.copy_(ffn_up_live)
    ffn_gate_live = packed_ffn_gate
    ffn_up_live = packed_ffn_up
    # The 3D MoE-shaped activation path compiles much faster than the 2D
    # variant for this small FFN case, so route the FFN activation through the
    # same kernel shape and squeeze it back.
    ffn_act = _run_gelu_mul_kernel(
        ffn_gate_live.unsqueeze(1),
        ffn_up_live.unsqueeze(1),
        stage_prefix="ffn_activation",
    )
    ffn_act = ffn_act.squeeze(1).contiguous()

    ffn_phase2 = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    ffn_act_moe = torch.empty(
        (num_tokens, 1, ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_act_moe[:, 0, :].copy_(ffn_act)
    ffn_routing_indices = torch.zeros((1, num_tokens), device="cuda", dtype=torch.int32)
    ffn_routing_mask = torch.zeros((2,), device="cuda", dtype=torch.int32)
    ffn_topk_weight = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    ffn_residual_zero = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_w2_weight = torch.empty(
        (1, hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_w2_weight[0].copy_(ffn_down_weight)
    ffn_intermediate_out = torch.zeros(
        (num_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16
    )

    ffn_routing_indices[0, 0] = 1
    ffn_routing_mask[0] = 0
    ffn_routing_mask[1] = 1

    ffn_act_dt = ffn_phase2.attach_input(ffn_act_moe, name="full_layer_ffn_act")
    ffn_down_weight_dt = ffn_phase2.attach_input(
        ffn_w2_weight.contiguous(), name="full_layer_ffn_down_weight"
    )
    ffn_routing_indices_dt = ffn_phase2.attach_input(
        ffn_routing_indices, name="full_layer_ffn_routing_indices"
    )
    ffn_routing_mask_dt = ffn_phase2.attach_input(
        ffn_routing_mask, name="full_layer_ffn_routing_mask"
    )
    ffn_topk_weight_dt = ffn_phase2.attach_input(ffn_topk_weight, name="full_layer_ffn_topk_weight")
    ffn_residual_zero_dt = ffn_phase2.attach_input(
        ffn_residual_zero, name="full_layer_ffn_zero_residual"
    )
    ffn_down = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_intermediate_out_dt = ffn_phase2.attach_input(
        ffn_intermediate_out, name="full_layer_ffn_w2_out"
    )
    ffn_down_dt = ffn_phase2.attach_input(ffn_down, name="full_layer_ffn_down")
    ffn_phase2.moe_w2_linear_layer(
        input=ffn_act_dt,
        weight=ffn_down_weight_dt,
        moe_routing_indices=ffn_routing_indices_dt,
        moe_mask=ffn_routing_mask_dt,
        output=ffn_intermediate_out_dt,
        grid_dim=_moe_expert_grid_dim(ffn_phase2, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    ffn_phase2.moe_mul_sum_add_layer(
        input=ffn_intermediate_out_dt,
        weight=ffn_topk_weight_dt,
        residual=ffn_residual_zero_dt,
        output=ffn_down_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    _timed_call("ffn_phase2_compile_s", lambda: compile_persistent_kernel_with_patches(ffn_phase2))
    _reset_pk_runtime_state(ffn_phase2, active_tokens=1)
    _timed_call("ffn_phase2_launch_s", ffn_phase2, cuda_sync=True)

    live_ffn_gate, live_ffn_up = torch.chunk(ffn_gate_up.float(), 2, dim=-1)
    ffn_down_ref = (F.gelu(live_ffn_gate) * live_ffn_up) @ ffn_down_weight.float().transpose(0, 1)

    # MoE branch from the same live attention output.
    router_weight = (
        torch.randn((num_experts, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    gate_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    up_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    w2_weight = (
        torch.randn((num_experts, hidden_size, intermediate), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    router_logits = torch.zeros((num_tokens, num_experts), device="cuda", dtype=torch.bfloat16)
    topk_weight = torch.zeros((num_tokens, topk), device="cuda", dtype=torch.float32)
    routing_indices = torch.zeros((num_experts, num_tokens), device="cuda", dtype=torch.int32)
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)

    router_pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    router_hidden_dt = router_pk.attach_input(
        post_attn.detach().contiguous(), name="full_layer_router_hidden"
    )
    router_weight_dt = router_pk.attach_input(
        router_weight.contiguous(), name="full_layer_router_weight"
    )
    router_logits_dt = router_pk.attach_input(router_logits, name="full_layer_router_logits")
    topk_weight_dt = router_pk.attach_input(topk_weight, name="full_layer_topk_weight")
    routing_indices_dt = router_pk.attach_input(routing_indices, name="full_layer_routing_indices")
    routing_mask_dt = router_pk.attach_input(routing_mask, name="full_layer_routing_mask")
    router_pk.linear_layer(
        input=router_hidden_dt,
        weight=router_weight_dt,
        output=router_logits_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    router_pk.moe_topk_softmax_routing_layer(
        input=router_logits_dt,
        output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    _timed_call("router_pk_compile_s", lambda: compile_persistent_kernel_with_patches(router_pk))
    _reset_pk_runtime_state(router_pk, active_tokens=1)
    _timed_call("router_pk_launch_s", router_pk, cuda_sync=True)

    selected_experts = []
    for expert_index in range(num_experts):
        rank = int(routing_indices[expert_index, 0].item())
        if rank != 0:
            selected_experts.append((expert_index, rank))
    selected_experts = sorted(selected_experts, key=lambda item: item[1])
    experts = [expert for expert, _ in selected_experts[:topk]]

    expert_pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    expert_hidden_dt = expert_pk.attach_input(
        post_attn.detach().contiguous(), name="full_layer_expert_hidden"
    )
    gate_outs = []
    up_outs = []
    for rank, expert in enumerate(experts):
        gate_weight_dt = expert_pk.attach_input(
            gate_weight[expert].contiguous(), name=f"full_layer_gate_w_{rank}"
        )
        up_weight_dt = expert_pk.attach_input(
            up_weight[expert].contiguous(), name=f"full_layer_up_w_{rank}"
        )
        gate_out = torch.zeros((1, intermediate), device="cuda", dtype=torch.bfloat16)
        up_out = torch.zeros((1, intermediate), device="cuda", dtype=torch.bfloat16)
        gate_out_dt = expert_pk.attach_input(gate_out, name=f"full_layer_gate_out_{rank}")
        up_out_dt = expert_pk.attach_input(up_out, name=f"full_layer_up_out_{rank}")
        expert_pk.linear_layer(
            input=expert_hidden_dt,
            weight=gate_weight_dt,
            output=gate_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        expert_pk.linear_layer(
            input=expert_hidden_dt,
            weight=up_weight_dt,
            output=up_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        gate_outs.append(gate_out)
        up_outs.append(up_out)

    _timed_call("expert_pk_compile_s", lambda: compile_persistent_kernel_with_patches(expert_pk))
    _reset_pk_runtime_state(expert_pk, active_tokens=1)
    _timed_call("expert_pk_launch_s", expert_pk, cuda_sync=True)

    gate_live = torch.stack([tensor[0].float() for tensor in gate_outs], dim=0).unsqueeze(0)
    up_live = torch.stack([tensor[0].float() for tensor in up_outs], dim=0).unsqueeze(0)
    moe_gate_in = torch.empty((num_tokens, topk, intermediate), device="cuda", dtype=torch.bfloat16)
    moe_up_in = torch.empty((num_tokens, topk, intermediate), device="cuda", dtype=torch.bfloat16)
    moe_gate_in.copy_(gate_live.to(torch.bfloat16))
    moe_up_in.copy_(up_live.to(torch.bfloat16))
    moe_act = _run_gelu_mul_kernel(
        moe_gate_in,
        moe_up_in,
        stage_prefix="moe_activation",
    ).contiguous()

    phase3 = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    moe_act_dt = phase3.attach_input(moe_act, name="full_layer_moe_act")
    topk_weight_dt_3 = phase3.attach_input(topk_weight, name="full_layer_topk_weight_3")
    routing_indices_dt_3 = phase3.attach_input(routing_indices, name="full_layer_routing_indices_3")
    routing_mask_dt_3 = phase3.attach_input(routing_mask, name="full_layer_routing_mask_3")
    w2_weight_dt = phase3.attach_input(w2_weight.contiguous(), name="full_layer_w2_weight")
    w2_out = torch.zeros((num_tokens, topk, hidden_size), device="cuda", dtype=torch.bfloat16)
    hidden_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    w2_out_dt = phase3.attach_input(w2_out, name="full_layer_w2_out")
    ffn_down_dt = phase3.attach_input(
        ffn_down.detach().contiguous(), name="full_layer_ffn_residual"
    )
    hidden_out_dt = phase3.attach_input(hidden_out, name="full_layer_hidden_out")
    phase3.moe_w2_linear_layer(
        input=moe_act_dt,
        weight=w2_weight_dt,
        moe_routing_indices=routing_indices_dt_3,
        moe_mask=routing_mask_dt_3,
        output=w2_out_dt,
        grid_dim=_moe_expert_grid_dim(phase3, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    phase3.moe_mul_sum_add_layer(
        input=w2_out_dt,
        weight=topk_weight_dt_3,
        residual=ffn_down_dt,
        output=hidden_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    _timed_call("phase3_compile_s", lambda: compile_persistent_kernel_with_patches(phase3))
    _reset_pk_runtime_state(phase3, active_tokens=1)
    _timed_call("phase3_launch_s", phase3, cuda_sync=True)

    ref_logits = post_attn_live @ router_weight.float().transpose(0, 1)
    ref_topk_values, ref_topk_indices = torch.topk(ref_logits, k=topk, dim=-1)
    ref_topk_weight = torch.softmax(ref_topk_values, dim=-1)
    ref_ids = ref_topk_indices[0].cpu().tolist()
    gate_ref = torch.stack(
        [post_attn_live[0] @ gate_weight[expert].float().transpose(0, 1) for expert in experts],
        dim=0,
    ).unsqueeze(0)
    up_ref = torch.stack(
        [post_attn_live[0] @ up_weight[expert].float().transpose(0, 1) for expert in experts],
        dim=0,
    ).unsqueeze(0)
    ref_moe_act = (F.gelu(gate_ref) * up_ref).contiguous()
    ref_w2 = torch.stack(
        [
            ref_moe_act[0, rank] @ w2_weight[expert].float().transpose(0, 1)
            for rank, expert in enumerate(experts)
        ],
        dim=0,
    ).unsqueeze(0)
    ref_hidden_out = (ref_w2.float() * topk_weight.float().unsqueeze(-1)).sum(dim=1) + ffn_down_ref
    live_moe_act_ref = (F.gelu(gate_live) * up_live).contiguous()

    return {
        **stage_timings,
        "post_attn_max_abs": float((post_attn_live - post_attn_ref).abs().max().item()),
        "post_attn_mean_abs": float((post_attn_live - post_attn_ref).abs().mean().item()),
        "ffn_down_max_abs": float((ffn_down.float() - ffn_down_ref).abs().max().item()),
        "ffn_down_mean_abs": float((ffn_down.float() - ffn_down_ref).abs().mean().item()),
        "topk_weight_max_abs": float((topk_weight.float() - ref_topk_weight).abs().max().item()),
        "topk_weight_mean_abs": float((topk_weight.float() - ref_topk_weight).abs().mean().item()),
        "routing_overlap_count": float(len(set(experts) & set(ref_ids))),
        "moe_act_live_inputs_max_abs": float(
            (moe_act.float() - live_moe_act_ref).abs().max().item()
        ),
        "moe_act_live_inputs_mean_abs": float(
            (moe_act.float() - live_moe_act_ref).abs().mean().item()
        ),
        "moe_act_max_abs": float((moe_act.float() - ref_moe_act).abs().max().item()),
        "moe_act_mean_abs": float((moe_act.float() - ref_moe_act).abs().mean().item()),
        "w2_max_abs": float((w2_out.float() - ref_w2).abs().max().item()),
        "w2_mean_abs": float((w2_out.float() - ref_w2).abs().mean().item()),
        "hidden_out_max_abs": float((hidden_out.float() - ref_hidden_out).abs().max().item()),
        "hidden_out_mean_abs": float((hidden_out.float() - ref_hidden_out).abs().mean().item()),
    }


def run_mirage_gemma_full_layer_single_pk_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
    output_dir: Optional[str] = None,
    launch: bool = True,
    include_moe_tail: bool = True,
    use_patched_compile: bool = True,
) -> Dict[str, float]:
    """Run a synthetic Gemma-style decode layer through one PersistentKernel.

    This mirrors the working decode runtime semantics closely while still using
    the proven top-k=1 MoE path for the dense FFN down projection.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    num_experts = 128
    topk = 8
    intermediate = 64
    ffn_intermediate = 256

    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )

    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    post_attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    pre_ffn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_weight = (
        torch.randn((2 * ffn_intermediate, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_down_weight = (
        torch.randn((hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    post_ffn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    router_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    router_root_size = (
        0.9 + 0.2 * torch.rand((hidden_size,), device="cuda", dtype=torch.float32)
    ).to(torch.bfloat16)
    router_scale = (0.9 + 0.2 * torch.rand((hidden_size,), device="cuda", dtype=torch.float32)).to(
        torch.bfloat16
    )
    router_weight = (
        torch.randn((num_experts, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    router_weight_scaled = (
        router_weight.float()
        * router_root_size.float().view(1, -1)
        * router_scale.float().view(1, -1)
    ).to(torch.bfloat16)
    pre_moe_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    gate_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    up_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    moe_w13_weight = torch.cat((up_weight, gate_weight), dim=1).contiguous()
    w2_weight = (
        torch.randn((num_experts, hidden_size, intermediate), device="cuda", dtype=torch.bfloat16)
        / 16.0
    )
    post_moe_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    post_merge_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    layer_scalar = (0.9 + 0.2 * torch.rand((hidden_size,), device="cuda", dtype=torch.float32)).to(
        torch.bfloat16
    )
    identity_weight = torch.eye(hidden_size, device="cuda", dtype=torch.bfloat16)
    layer_scale_weight = torch.diag(layer_scalar.float()).to(torch.bfloat16)

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    o_proj = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    post_attn_norm = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_normed = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_storage = torch.zeros(
        (num_tokens, 1, 2 * ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_gate_up = ffn_gate_up_storage.view(num_tokens, 2 * ffn_intermediate)
    ffn_act = torch.zeros((num_tokens, 1, ffn_intermediate), device="cuda", dtype=torch.bfloat16)
    ffn_routing_indices = torch.ones((1, num_tokens), device="cuda", dtype=torch.int32)
    ffn_routing_mask = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    ffn_topk_weight = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    zero_residual = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_w2_weight = torch.empty(
        (1, hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_w2_weight[0].copy_(ffn_down_weight)
    ffn_w2_out = torch.zeros((num_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_down = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_norm = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    router_in = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    router_logits = torch.zeros((num_tokens, num_experts), device="cuda", dtype=torch.bfloat16)
    topk_weight = torch.zeros((num_tokens, topk), device="cuda", dtype=torch.float32)
    routing_indices = torch.zeros((num_experts, num_tokens), device="cuda", dtype=torch.int32)
    routing_mask = torch.zeros((num_experts + 1,), device="cuda", dtype=torch.int32)
    moe_in = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    moe_act = torch.zeros((num_tokens, topk, intermediate), device="cuda", dtype=torch.bfloat16)
    moe_w2_out = torch.zeros((num_tokens, topk, hidden_size), device="cuda", dtype=torch.bfloat16)
    moe_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    moe_norm = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_moe_add = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_moe_norm = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    hidden_pre_scale = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    scaled_post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    hidden_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="single_pk_hidden")
    attn_norm_dt = pk.attach_input(attn_norm_weight.contiguous(), name="single_pk_attn_norm")
    qkv_weight_dt = pk.attach_input(qkv_weight.contiguous(), name="single_pk_qkv_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="single_pk_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="single_pk_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="single_pk_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="single_pk_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="single_pk_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="single_pk_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="single_pk_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="single_pk_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="single_pk_attn_out")
    o_proj_weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="single_pk_o_proj_weight")
    o_proj_dt = pk.attach_input(o_proj, name="single_pk_o_proj")
    post_attn_norm_weight_dt = pk.attach_input(
        post_attn_norm_weight.contiguous(), name="single_pk_post_attn_norm_weight"
    )
    post_attn_norm_dt = pk.attach_input(post_attn_norm, name="single_pk_post_attn_norm")
    identity_weight_dt = pk.attach_input(
        identity_weight.contiguous(), name="single_pk_identity_weight"
    )
    post_attn_dt = pk.attach_input(post_attn, name="single_pk_post_attn")
    pre_ffn_norm_weight_dt = pk.attach_input(
        pre_ffn_norm_weight.contiguous(), name="single_pk_pre_ffn_norm_weight"
    )
    ffn_normed_dt = pk.attach_input(ffn_normed, name="single_pk_ffn_normed")
    ffn_gate_up_weight_dt = pk.attach_input(
        ffn_gate_up_weight.contiguous(), name="single_pk_ffn_gate_up_weight"
    )
    ffn_gate_up_dt = pk.attach_input(ffn_gate_up, name="single_pk_ffn_gate_up")
    ffn_gate_up_moe_dt = pk.attach_input(ffn_gate_up_storage, name="single_pk_ffn_gate_up_moe")
    ffn_act_dt = pk.attach_input(ffn_act, name="single_pk_ffn_act")
    ffn_routing_indices_dt = pk.attach_input(
        ffn_routing_indices, name="single_pk_ffn_routing_indices"
    )
    ffn_routing_mask_dt = pk.attach_input(ffn_routing_mask, name="single_pk_ffn_routing_mask")
    ffn_topk_weight_dt = pk.attach_input(ffn_topk_weight, name="single_pk_ffn_topk_weight")
    zero_residual_dt = pk.attach_input(zero_residual, name="single_pk_zero_residual")
    ffn_w2_weight_dt = pk.attach_input(ffn_w2_weight.contiguous(), name="single_pk_ffn_w2_weight")
    ffn_w2_out_dt = pk.attach_input(ffn_w2_out, name="single_pk_ffn_w2_out")
    ffn_down_dt = pk.attach_input(ffn_down, name="single_pk_ffn_down")
    post_ffn_norm_weight_dt = pk.attach_input(
        post_ffn_norm_weight.contiguous(), name="single_pk_post_ffn_norm_weight"
    )
    ffn_norm_dt = pk.attach_input(ffn_norm, name="single_pk_ffn_norm")

    router_norm_weight_dt = None
    router_in_dt = None
    router_weight_dt = None
    router_logits_dt = None
    moe_in_norm_weight_dt = None
    moe_in_dt = None
    topk_weight_dt = None
    routing_indices_dt = None
    routing_mask_dt = None
    moe_w13_weight_dt = None
    moe_act_dt = None
    moe_w2_weight_dt = None
    moe_w2_out_dt = None
    moe_out_dt = None
    post_moe_norm_weight_dt = None
    moe_norm_dt = None
    ffn_moe_add_dt = None
    post_merge_norm_weight_dt = None
    ffn_moe_norm_dt = None
    hidden_pre_scale_dt = None
    scaled_post_attn_dt = None
    layer_scale_weight_dt = None
    hidden_out_dt = None

    if include_moe_tail:
        router_norm_weight_dt = pk.attach_input(
            router_norm_weight.contiguous(), name="single_pk_router_norm_weight"
        )
        router_in_dt = pk.attach_input(router_in, name="single_pk_router_in")
        router_weight_dt = pk.attach_input(
            router_weight_scaled.contiguous(), name="single_pk_router_weight"
        )
        router_logits_dt = pk.attach_input(router_logits, name="single_pk_router_logits")
        moe_in_norm_weight_dt = pk.attach_input(
            pre_moe_norm_weight.contiguous(), name="single_pk_moe_in_norm_weight"
        )
        moe_in_dt = pk.attach_input(moe_in, name="single_pk_moe_in")
        topk_weight_dt = pk.attach_input(topk_weight, name="single_pk_topk_weight")
        routing_indices_dt = pk.attach_input(routing_indices, name="single_pk_routing_indices")
        routing_mask_dt = pk.attach_input(routing_mask, name="single_pk_routing_mask")
        moe_w13_weight_dt = pk.attach_input(
            moe_w13_weight.contiguous(), name="single_pk_moe_w13_weight"
        )
        moe_act_dt = pk.attach_input(moe_act, name="single_pk_moe_act")
        moe_w2_weight_dt = pk.attach_input(w2_weight.contiguous(), name="single_pk_moe_w2_weight")
        moe_w2_out_dt = pk.attach_input(moe_w2_out, name="single_pk_moe_w2_out")
        moe_out_dt = pk.attach_input(moe_out, name="single_pk_moe_out")
        post_moe_norm_weight_dt = pk.attach_input(
            post_moe_norm_weight.contiguous(), name="single_pk_post_moe_norm_weight"
        )
        moe_norm_dt = pk.attach_input(moe_norm, name="single_pk_moe_norm")
        ffn_moe_add_dt = pk.attach_input(ffn_moe_add, name="single_pk_ffn_moe_add")
        post_merge_norm_weight_dt = pk.attach_input(
            post_merge_norm_weight.contiguous(), name="single_pk_post_merge_norm_weight"
        )
        ffn_moe_norm_dt = pk.attach_input(ffn_moe_norm, name="single_pk_ffn_moe_norm")
        hidden_pre_scale_dt = pk.attach_input(hidden_pre_scale, name="single_pk_hidden_pre_scale")
        scaled_post_attn_dt = pk.attach_input(scaled_post_attn, name="single_pk_scaled_post_attn")
        layer_scale_weight_dt = pk.attach_input(
            layer_scale_weight.contiguous(), name="single_pk_layer_scale_weight"
        )
        hidden_out_dt = pk.attach_input(hidden_out, name="single_pk_hidden_out")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        output=o_proj_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.rmsnorm_layer(
        input=o_proj_dt,
        weight=post_attn_norm_weight_dt,
        output=post_attn_norm_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=post_attn_norm_dt,
        weight=identity_weight_dt,
        residual=hidden_dt,
        output=post_attn_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.rmsnorm_layer(
        input=post_attn_dt,
        weight=pre_ffn_norm_weight_dt,
        output=ffn_normed_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=ffn_normed_dt,
        weight=ffn_gate_up_weight_dt,
        output=ffn_gate_up_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_gelu_mul_layer(
        input=ffn_gate_up_moe_dt,
        output=ffn_act_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_w2_linear_layer(
        input=ffn_act_dt,
        weight=ffn_w2_weight_dt,
        moe_routing_indices=ffn_routing_indices_dt,
        moe_mask=ffn_routing_mask_dt,
        output=ffn_w2_out_dt,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    pk.moe_mul_sum_add_layer(
        input=ffn_w2_out_dt,
        weight=ffn_topk_weight_dt,
        residual=zero_residual_dt,
        output=ffn_down_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.rmsnorm_layer(
        input=ffn_down_dt,
        weight=post_ffn_norm_weight_dt,
        output=ffn_norm_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    if include_moe_tail:
        pk.rmsnorm_layer(
            input=post_attn_dt,
            weight=router_norm_weight_dt,
            output=router_in_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.linear_layer(
            input=router_in_dt,
            weight=router_weight_dt,
            output=router_logits_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.moe_topk_softmax_routing_layer(
            input=router_logits_dt,
            output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.rmsnorm_layer(
            input=post_attn_dt,
            weight=moe_in_norm_weight_dt,
            output=moe_in_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.moe_w13_gelu_mul_layer(
            input=moe_in_dt,
            weight=moe_w13_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=moe_act_dt,
            grid_dim=_moe_expert_grid_dim(pk, w13_linear=True),
            block_dim=(128, 1, 1),
        )
        pk.moe_w2_linear_layer(
            input=moe_act_dt,
            weight=moe_w2_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=moe_w2_out_dt,
            grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
            block_dim=(128, 1, 1),
        )
        pk.moe_mul_sum_add_layer(
            input=moe_w2_out_dt,
            weight=topk_weight_dt,
            residual=zero_residual_dt,
            output=moe_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.rmsnorm_layer(
            input=moe_out_dt,
            weight=post_moe_norm_weight_dt,
            output=moe_norm_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.linear_with_residual_layer(
            input=ffn_norm_dt,
            weight=identity_weight_dt,
            residual=moe_norm_dt,
            output=ffn_moe_add_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.rmsnorm_layer(
            input=ffn_moe_add_dt,
            weight=post_merge_norm_weight_dt,
            output=ffn_moe_norm_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.linear_layer(
            input=ffn_moe_norm_dt,
            weight=layer_scale_weight_dt,
            output=hidden_pre_scale_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.linear_layer(
            input=post_attn_dt,
            weight=layer_scale_weight_dt,
            output=scaled_post_attn_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        pk.linear_with_residual_layer(
            input=hidden_pre_scale_dt,
            weight=identity_weight_dt,
            residual=scaled_post_attn_dt,
            output=hidden_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )

    if use_patched_compile:
        compile_persistent_kernel_with_patches(pk, output_dir=output_dir)
    else:
        pk.compile(output_dir=output_dir)
    _reset_pk_runtime_state(
        pk,
        active_tokens=num_tokens,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
    if not launch:
        return {"compiled_only": 1.0}

    pk()
    torch.cuda.synchronize()

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)

    o_proj_ref = ref_attn @ o_proj_weight.float().transpose(0, 1)
    o_proj_var = o_proj_ref.pow(2).mean(dim=-1, keepdim=True)
    post_attn_ref = (
        (o_proj_ref * torch.rsqrt(o_proj_var + eps)) * post_attn_norm_weight.float()
    ) + hidden_in.float()
    ffn_in_var = post_attn_ref.pow(2).mean(dim=-1, keepdim=True)
    ffn_in_ref = (post_attn_ref * torch.rsqrt(ffn_in_var + eps)) * pre_ffn_norm_weight.float()
    ffn_gate_up_ref = ffn_in_ref @ ffn_gate_up_weight.float().transpose(0, 1)
    ffn_gate_ref, ffn_up_ref = torch.chunk(ffn_gate_up_ref, 2, dim=-1)
    ffn_down_ref = (F.gelu(ffn_gate_ref) * ffn_up_ref) @ ffn_down_weight.float().transpose(0, 1)
    ffn_down_var = ffn_down_ref.pow(2).mean(dim=-1, keepdim=True)
    ffn_norm_ref = (ffn_down_ref * torch.rsqrt(ffn_down_var + eps)) * post_ffn_norm_weight.float()

    results = {
        "post_attn_max_abs": float((post_attn.float() - post_attn_ref).abs().max().item()),
        "post_attn_mean_abs": float((post_attn.float() - post_attn_ref).abs().mean().item()),
        "ffn_down_max_abs": float((ffn_down.float() - ffn_down_ref).abs().max().item()),
        "ffn_down_mean_abs": float((ffn_down.float() - ffn_down_ref).abs().mean().item()),
        "ffn_gate_up_max_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().max().item()),
        "ffn_gate_up_mean_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().mean().item()),
        "include_moe_tail": float(include_moe_tail),
    }
    if not include_moe_tail:
        return results

    router_in_var = post_attn_ref.pow(2).mean(dim=-1, keepdim=True)
    router_in_ref = (
        (post_attn_ref * torch.rsqrt(router_in_var + 1e-6))
        * router_root_size.float().view(1, -1)
        * router_scale.float().view(1, -1)
    )
    ref_logits = router_in_ref @ router_weight.float().transpose(0, 1)
    ref_topk_values, ref_topk_indices = torch.topk(ref_logits, k=topk, dim=-1)
    ref_topk_weight = torch.softmax(ref_topk_values, dim=-1)
    ref_ids = ref_topk_indices[0].cpu().tolist()
    selected_experts = []
    for expert_index in range(num_experts):
        rank = int(routing_indices[expert_index, 0].item())
        if rank != 0:
            selected_experts.append((expert_index, rank))
    selected_experts = sorted(selected_experts, key=lambda item: item[1])
    experts = [expert for expert, _ in selected_experts[:topk]]

    moe_in_var = post_attn_ref.pow(2).mean(dim=-1, keepdim=True)
    moe_in_ref = (post_attn_ref * torch.rsqrt(moe_in_var + eps)) * pre_moe_norm_weight.float()
    gate_ref = torch.stack(
        [moe_in_ref[0] @ gate_weight[expert].float().transpose(0, 1) for expert in experts],
        dim=0,
    ).unsqueeze(0)
    up_ref = torch.stack(
        [moe_in_ref[0] @ up_weight[expert].float().transpose(0, 1) for expert in experts],
        dim=0,
    ).unsqueeze(0)
    ref_moe_act = (F.gelu(gate_ref) * up_ref).contiguous()
    ref_w2 = torch.stack(
        [
            ref_moe_act[0, rank] @ w2_weight[expert].float().transpose(0, 1)
            for rank, expert in enumerate(experts)
        ],
        dim=0,
    ).unsqueeze(0)
    ref_moe_out = (ref_w2 * ref_topk_weight.unsqueeze(-1)).sum(dim=1)
    ref_moe_var = ref_moe_out.pow(2).mean(dim=-1, keepdim=True)
    ref_moe_norm = (ref_moe_out * torch.rsqrt(ref_moe_var + eps)) * post_moe_norm_weight.float()
    ref_ffn_moe_add = ffn_norm_ref + ref_moe_norm
    ref_ffn_moe_var = ref_ffn_moe_add.pow(2).mean(dim=-1, keepdim=True)
    ref_ffn_moe_norm = (
        ref_ffn_moe_add * torch.rsqrt(ref_ffn_moe_var + eps)
    ) * post_merge_norm_weight.float()
    ref_hidden_out = (post_attn_ref + ref_ffn_moe_norm) * layer_scalar.float().view(1, -1)

    results.update(
        {
            "topk_weight_max_abs": float(
                (topk_weight.float() - ref_topk_weight).abs().max().item()
            ),
            "topk_weight_mean_abs": float(
                (topk_weight.float() - ref_topk_weight).abs().mean().item()
            ),
            "routing_overlap_count": float(len(set(experts) & set(ref_ids))),
            "moe_act_max_abs": float((moe_act.float() - ref_moe_act).abs().max().item()),
            "moe_act_mean_abs": float((moe_act.float() - ref_moe_act).abs().mean().item()),
            "w2_max_abs": float((moe_w2_out.float() - ref_w2).abs().max().item()),
            "w2_mean_abs": float((moe_w2_out.float() - ref_w2).abs().mean().item()),
            "hidden_out_max_abs": float((hidden_out.float() - ref_hidden_out).abs().max().item()),
            "hidden_out_mean_abs": float((hidden_out.float() - ref_hidden_out).abs().mean().item()),
        }
    )
    return results


def run_mirage_attention_ffn_fused_single_pk_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
) -> Dict[str, float]:
    """Run attention + dense FFN fused-down in one PersistentKernel."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    ffn_intermediate = 256

    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_weight = (
        torch.randn((2 * ffn_intermediate, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_down_weight = (
        torch.randn((hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_normed = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up = torch.zeros(
        (num_tokens, 2 * ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_down = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_down_residual = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="attn_ffn_hidden")
    attn_norm_dt = pk.attach_input(attn_norm_weight.contiguous(), name="attn_ffn_attn_norm")
    qkv_weight_dt = pk.attach_input(qkv_weight.contiguous(), name="attn_ffn_qkv_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="attn_ffn_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="attn_ffn_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="attn_ffn_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="attn_ffn_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="attn_ffn_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="attn_ffn_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="attn_ffn_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="attn_ffn_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="attn_ffn_attn_out")
    o_proj_weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="attn_ffn_o_proj_weight")
    post_attn_dt = pk.attach_input(post_attn, name="attn_ffn_post_attn")
    ffn_norm_dt = pk.attach_input(ffn_norm_weight.contiguous(), name="attn_ffn_ffn_norm")
    ffn_normed_dt = pk.attach_input(ffn_normed, name="attn_ffn_ffn_normed")
    ffn_gate_up_weight_dt = pk.attach_input(
        ffn_gate_up_weight.contiguous(), name="attn_ffn_gate_up_weight"
    )
    ffn_gate_up_dt = pk.attach_input(ffn_gate_up, name="attn_ffn_gate_up")
    ffn_down_weight_dt = pk.attach_input(ffn_down_weight.contiguous(), name="attn_ffn_down_weight")
    ffn_down_residual_dt = pk.attach_input(ffn_down_residual, name="attn_ffn_down_residual")
    ffn_down_dt = pk.attach_input(ffn_down, name="attn_ffn_down")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=hidden_dt,
        output=post_attn_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.rmsnorm_layer(
        input=post_attn_dt,
        weight=ffn_norm_dt,
        output=ffn_normed_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=ffn_normed_dt,
        weight=ffn_gate_up_weight_dt,
        output=ffn_gate_up_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.gelu_mul_linear_with_residual_layer(
        input=ffn_gate_up_dt,
        weight=ffn_down_weight_dt,
        residual=ffn_down_residual_dt,
        output=ffn_down_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(
        pk,
        active_tokens=num_tokens,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
    pk()
    torch.cuda.synchronize()

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    post_attn_ref = ref_attn @ o_proj_weight.float().transpose(0, 1) + hidden_in.float()
    post_attn_live = post_attn.float()
    ffn_variance = post_attn_live.pow(2).mean(dim=-1, keepdim=True)
    ffn_norm_ref = (post_attn_live * torch.rsqrt(ffn_variance + eps)) * ffn_norm_weight.float()
    ffn_gate_up_ref = ffn_norm_ref @ ffn_gate_up_weight.float().transpose(0, 1)
    live_ffn_gate, live_ffn_up = torch.chunk(ffn_gate_up.float(), 2, dim=-1)
    ffn_down_ref = (F.gelu(live_ffn_gate) * live_ffn_up) @ ffn_down_weight.float().transpose(0, 1)
    ffn_down_silu_ref = (F.silu(live_ffn_gate) * live_ffn_up) @ ffn_down_weight.float().transpose(
        0, 1
    )
    ffn_down_swapped_ref = (
        F.gelu(live_ffn_up) * live_ffn_gate
    ) @ ffn_down_weight.float().transpose(0, 1)

    return {
        "post_attn_max_abs": float((post_attn_live - post_attn_ref).abs().max().item()),
        "post_attn_mean_abs": float((post_attn_live - post_attn_ref).abs().mean().item()),
        "ffn_gate_up_max_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().max().item()),
        "ffn_gate_up_mean_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().mean().item()),
        "ffn_down_max_abs": float((ffn_down.float() - ffn_down_ref).abs().max().item()),
        "ffn_down_mean_abs": float((ffn_down.float() - ffn_down_ref).abs().mean().item()),
        "ffn_down_silu_max_abs": float((ffn_down.float() - ffn_down_silu_ref).abs().max().item()),
        "ffn_down_silu_mean_abs": float((ffn_down.float() - ffn_down_silu_ref).abs().mean().item()),
        "ffn_down_swapped_max_abs": float(
            (ffn_down.float() - ffn_down_swapped_ref).abs().max().item()
        ),
        "ffn_down_swapped_mean_abs": float(
            (ffn_down.float() - ffn_down_swapped_ref).abs().mean().item()
        ),
    }


def run_mirage_linear_residual_ffn_fused_single_pk_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
) -> Dict[str, float]:
    """Run linear-with-residual + dense FFN fused-down in one PersistentKernel."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    ffn_intermediate = 256

    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )

    input_hidden = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    residual = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    post_linear = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    ffn_normed = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_weight = (
        torch.randn((2 * ffn_intermediate, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_gate_up = torch.zeros(
        (num_tokens, 2 * ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_down_weight = (
        torch.randn((hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_down_residual = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_down = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    input_dt = pk.attach_input(input_hidden.contiguous(), name="linear_ffn_input")
    proj_weight_dt = pk.attach_input(proj_weight.contiguous(), name="linear_ffn_proj_weight")
    residual_dt = pk.attach_input(residual.contiguous(), name="linear_ffn_residual")
    post_linear_dt = pk.attach_input(post_linear, name="linear_ffn_post_linear")
    ffn_norm_weight_dt = pk.attach_input(
        ffn_norm_weight.contiguous(), name="linear_ffn_norm_weight"
    )
    ffn_normed_dt = pk.attach_input(ffn_normed, name="linear_ffn_normed")
    ffn_gate_up_weight_dt = pk.attach_input(
        ffn_gate_up_weight.contiguous(), name="linear_ffn_gate_up_weight"
    )
    ffn_gate_up_dt = pk.attach_input(ffn_gate_up, name="linear_ffn_gate_up")
    ffn_down_weight_dt = pk.attach_input(
        ffn_down_weight.contiguous(), name="linear_ffn_down_weight"
    )
    ffn_down_residual_dt = pk.attach_input(ffn_down_residual, name="linear_ffn_down_residual")
    ffn_down_dt = pk.attach_input(ffn_down, name="linear_ffn_down")

    pk.linear_with_residual_layer(
        input=input_dt,
        weight=proj_weight_dt,
        residual=residual_dt,
        output=post_linear_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.rmsnorm_layer(
        input=post_linear_dt,
        weight=ffn_norm_weight_dt,
        output=ffn_normed_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=ffn_normed_dt,
        weight=ffn_gate_up_weight_dt,
        output=ffn_gate_up_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.gelu_mul_linear_with_residual_layer(
        input=ffn_gate_up_dt,
        weight=ffn_down_weight_dt,
        residual=ffn_down_residual_dt,
        output=ffn_down_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(pk, active_tokens=num_tokens)
    pk()
    torch.cuda.synchronize()

    post_linear_ref = input_hidden.float() @ proj_weight.float().transpose(0, 1) + residual.float()
    variance = post_linear.float().pow(2).mean(dim=-1, keepdim=True)
    ffn_norm_ref = (post_linear.float() * torch.rsqrt(variance + eps)) * ffn_norm_weight.float()
    ffn_gate_up_ref = ffn_norm_ref @ ffn_gate_up_weight.float().transpose(0, 1)
    live_ffn_gate, live_ffn_up = torch.chunk(ffn_gate_up.float(), 2, dim=-1)
    ffn_down_ref = (F.gelu(live_ffn_gate) * live_ffn_up) @ ffn_down_weight.float().transpose(0, 1)

    return {
        "post_linear_max_abs": float((post_linear.float() - post_linear_ref).abs().max().item()),
        "post_linear_mean_abs": float((post_linear.float() - post_linear_ref).abs().mean().item()),
        "ffn_gate_up_max_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().max().item()),
        "ffn_gate_up_mean_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().mean().item()),
        "ffn_down_max_abs": float((ffn_down.float() - ffn_down_ref).abs().max().item()),
        "ffn_down_mean_abs": float((ffn_down.float() - ffn_down_ref).abs().mean().item()),
    }


def run_mirage_attention_ffn_moew2_single_pk_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
) -> Dict[str, float]:
    """Run attention + dense FFN-down via top-k=1 MoE kernels in one PersistentKernel."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    num_tokens = 1
    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64
    ffn_intermediate = 256

    pk = create_test_persistent_kernel(
        max_seq_length=512,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )
    qo_indptr = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    q_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device="cuda", dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_norm_weight = torch.ones((hidden_size,), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_weight = (
        torch.randn((2 * ffn_intermediate, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_down_weight = (
        torch.randn((hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    ffn_w2_weight = torch.empty(
        (1, hidden_size, ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_w2_weight[0].copy_(ffn_down_weight)

    rmsnorm_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )
    attn_out_size = qo_heads * head_dim
    attn_out = torch.zeros((num_tokens, attn_out_size), device="cuda", dtype=torch.bfloat16)
    post_attn = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_normed = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_gate_up_storage = torch.zeros(
        (num_tokens, 1, 2 * ffn_intermediate), device="cuda", dtype=torch.bfloat16
    )
    ffn_gate_up = ffn_gate_up_storage.view(num_tokens, 2 * ffn_intermediate)
    ffn_act = torch.zeros((num_tokens, 1, ffn_intermediate), device="cuda", dtype=torch.bfloat16)
    ffn_routing_indices = torch.tensor([[1]], device="cuda", dtype=torch.int32)
    ffn_routing_mask = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    ffn_topk_weight = torch.ones((num_tokens, 1), device="cuda", dtype=torch.float32)
    ffn_residual_zero = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_w2_out = torch.zeros((num_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16)
    ffn_down = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)

    hidden_dt = pk.attach_input(hidden_in.contiguous(), name="attn_moew2_hidden")
    attn_norm_dt = pk.attach_input(attn_norm_weight.contiguous(), name="attn_moew2_attn_norm")
    qkv_weight_dt = pk.attach_input(qkv_weight.contiguous(), name="attn_moew2_qkv_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="attn_moew2_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="attn_moew2_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight.contiguous(), name="attn_moew2_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight.contiguous(), name="attn_moew2_k_norm")
    cos_dt = pk.attach_input(cos.contiguous(), name="attn_moew2_cos")
    sin_dt = pk.attach_input(sin.contiguous(), name="attn_moew2_sin")
    k_cache_dt = pk.attach_input(k_cache.contiguous(), name="attn_moew2_k_cache")
    v_cache_dt = pk.attach_input(v_cache.contiguous(), name="attn_moew2_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="attn_moew2_attn_out")
    o_proj_weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="attn_moew2_o_proj_weight")
    post_attn_dt = pk.attach_input(post_attn, name="attn_moew2_post_attn")
    ffn_norm_dt = pk.attach_input(ffn_norm_weight.contiguous(), name="attn_moew2_ffn_norm")
    ffn_normed_dt = pk.attach_input(ffn_normed, name="attn_moew2_ffn_normed")
    ffn_gate_up_weight_dt = pk.attach_input(
        ffn_gate_up_weight.contiguous(), name="attn_moew2_gate_up_weight"
    )
    ffn_gate_up_dt = pk.attach_input(ffn_gate_up, name="attn_moew2_gate_up")
    ffn_gate_up_moe_dt = pk.attach_input(ffn_gate_up_storage, name="attn_moew2_gate_up_moe")
    ffn_act_dt = pk.attach_input(ffn_act, name="attn_moew2_act")
    ffn_w2_weight_dt = pk.attach_input(ffn_w2_weight.contiguous(), name="attn_moew2_w2_weight")
    ffn_routing_indices_dt = pk.attach_input(ffn_routing_indices, name="attn_moew2_routing_indices")
    ffn_routing_mask_dt = pk.attach_input(ffn_routing_mask, name="attn_moew2_routing_mask")
    ffn_topk_weight_dt = pk.attach_input(ffn_topk_weight, name="attn_moew2_topk_weight")
    ffn_residual_zero_dt = pk.attach_input(ffn_residual_zero, name="attn_moew2_zero_residual")
    ffn_w2_out_dt = pk.attach_input(ffn_w2_out, name="attn_moew2_w2_out")
    ffn_down_dt = pk.attach_input(ffn_down, name="attn_moew2_down")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.paged_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out_dt,
        weight=o_proj_weight_dt,
        residual=hidden_dt,
        output=post_attn_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.rmsnorm_layer(
        input=post_attn_dt,
        weight=ffn_norm_dt,
        output=ffn_normed_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=ffn_normed_dt,
        weight=ffn_gate_up_weight_dt,
        output=ffn_gate_up_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_gelu_mul_layer(
        input=ffn_gate_up_moe_dt,
        output=ffn_act_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.moe_w2_linear_layer(
        input=ffn_act_dt,
        weight=ffn_w2_weight_dt,
        moe_routing_indices=ffn_routing_indices_dt,
        moe_mask=ffn_routing_mask_dt,
        output=ffn_w2_out_dt,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
        block_dim=(128, 1, 1),
    )
    pk.moe_mul_sum_add_layer(
        input=ffn_w2_out_dt,
        weight=ffn_topk_weight_dt,
        residual=ffn_residual_zero_dt,
        output=ffn_down_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)
    _reset_pk_runtime_state(
        pk,
        active_tokens=num_tokens,
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
    )
    pk()
    torch.cuda.synchronize()

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)
    torch_k_cache = k_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    torch_v_cache = v_cache.reshape(max_num_pages, page_size, kv_heads * head_dim).clone()
    q = qkv_ref[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim)
    k = qkv_ref[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
    v = qkv_ref[:, qo_heads * head_dim + kv_heads * head_dim :]
    page_idx = int(paged_kv_indices[0].item())
    page_offset = int(paged_kv_last_page_len[0].item()) - num_tokens
    torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
    torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v
    ref_attn = _reference_multitoken_paged_attention(
        q=q,
        paged_k_cache=torch_k_cache,
        paged_v_cache=torch_v_cache,
        qo_heads=qo_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_tokens=num_tokens,
        paged_kv_indptr_buffer=paged_kv_indptr,
        paged_kv_indices_buffer=paged_kv_indices,
        paged_kv_last_page_len_buffer=paged_kv_last_page_len,
    ).reshape(num_tokens, hidden_size)
    post_attn_ref = ref_attn @ o_proj_weight.float().transpose(0, 1) + hidden_in.float()
    post_attn_live = post_attn.float()
    ffn_variance = post_attn_live.pow(2).mean(dim=-1, keepdim=True)
    ffn_norm_ref = (post_attn_live * torch.rsqrt(ffn_variance + eps)) * ffn_norm_weight.float()
    ffn_gate_up_ref = ffn_norm_ref @ ffn_gate_up_weight.float().transpose(0, 1)
    live_ffn_gate, live_ffn_up = torch.chunk(ffn_gate_up.float(), 2, dim=-1)
    ffn_down_ref = (F.gelu(live_ffn_gate) * live_ffn_up) @ ffn_down_weight.float().transpose(0, 1)

    return {
        "post_attn_max_abs": float((post_attn_live - post_attn_ref).abs().max().item()),
        "post_attn_mean_abs": float((post_attn_live - post_attn_ref).abs().mean().item()),
        "ffn_gate_up_max_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().max().item()),
        "ffn_gate_up_mean_abs": float((ffn_gate_up.float() - ffn_gate_up_ref).abs().mean().item()),
        "ffn_down_max_abs": float((ffn_down.float() - ffn_down_ref).abs().max().item()),
        "ffn_down_mean_abs": float((ffn_down.float() - ffn_down_ref).abs().mean().item()),
    }


def profile_mirage_gemma_full_layer_split_dense_compile_stages(
    *,
    eps: float = 1e-5,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Return only timing metrics for the synthetic full live-layer helper."""

    results = run_mirage_gemma_full_layer_split_dense_forward_correctness(
        eps=eps,
        seed=seed,
        verbose=verbose,
    )
    return {
        key: value
        for key, value in results.items()
        if key.endswith("_compile_s")
        or key.endswith("_codegen_s")
        or key.endswith("_nvcc_s")
        or key.endswith("_launch_s")
    }


def run_mirage_hybrid_attention_sublayer_forward_correctness(
    *,
    eps: float = 1e-5,
    seed: int = 0,
    repeats: int = 2,
    num_tokens: int = 1,
) -> Dict[str, float]:
    """Run a hybrid live attention sublayer using proven reusable Mirage pieces.

    Composition:
    - direct Mirage qkv norm-linear
    - direct Mirage paged attention
    - compiled PersistentKernel linear_with_residual
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    if num_tokens == 1:
        qkv_module = _load_mirage_attn_qkv_norm_linear_single_token_extension()
        attn_module = _load_mirage_attention_single_token_extension()
    else:
        qkv_module = _load_mirage_attn_qkv_norm_linear_extension()
        attn_module = _load_mirage_attention_extension()

    hidden_size = 512
    qo_heads = 4
    kv_heads = 1
    head_dim = 128
    page_size = 64
    max_num_pages = 64

    hidden_in = torch.randn((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16) / 8.0
    attn_norm_weight = torch.ones((1, hidden_size), device="cuda", dtype=torch.bfloat16)
    qkv_weight = (
        torch.randn(
            ((qo_heads + 2 * kv_heads) * head_dim, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        / 16.0
    )
    qkv_out = torch.zeros(
        (num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device="cuda", dtype=torch.bfloat16
    )

    paged_k_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads * head_dim), device="cuda", dtype=torch.bfloat16
    )
    paged_v_cache = 0.2 + 0.1 * torch.randn(
        (max_num_pages, page_size, kv_heads * head_dim), device="cuda", dtype=torch.bfloat16
    )
    qo_indptr_buffer = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)
    paged_kv_indptr_buffer = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    paged_kv_indices_buffer = torch.arange(max_num_pages, device="cuda", dtype=torch.int32)
    paged_kv_last_page_len_buffer = torch.tensor([8 + num_tokens], device="cuda", dtype=torch.int32)
    attn_out_flat = torch.zeros(
        (num_tokens * qo_heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    q_norm_weight = torch.ones((1, head_dim), device="cuda", dtype=torch.bfloat16)
    k_norm_weight = torch.ones((1, head_dim), device="cuda", dtype=torch.bfloat16)
    all_cos = torch.ones((513, head_dim), device="cuda", dtype=torch.bfloat16)
    all_sin = torch.zeros((513, head_dim), device="cuda", dtype=torch.bfloat16)
    page_idx = int(paged_kv_indices_buffer[0].item())
    page_offset = int(paged_kv_last_page_len_buffer[0].item()) - num_tokens

    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=num_tokens,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
    )
    linear_input = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    residual = hidden_in.clone().contiguous()
    o_proj_weight = (
        torch.randn((hidden_size, hidden_size), device="cuda", dtype=torch.bfloat16) / 16.0
    )
    block_out = torch.zeros((num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16)
    input_dt = pk.attach_input(linear_input, name="hybrid_attn_linear_input")
    weight_dt = pk.attach_input(o_proj_weight.contiguous(), name="hybrid_attn_linear_weight")
    residual_dt = pk.attach_input(residual, name="hybrid_attn_linear_residual")
    output_dt = pk.attach_input(block_out, name="hybrid_attn_linear_output")
    pk.linear_with_residual_layer(
        input=input_dt,
        weight=weight_dt,
        residual=residual_dt,
        output=output_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk)

    qkv_variance = hidden_in.float().pow(2).mean(dim=-1, keepdim=True)
    qkv_ref = (hidden_in.float() * torch.rsqrt(qkv_variance + eps)) * attn_norm_weight.float()
    qkv_ref = qkv_ref @ qkv_weight.float().transpose(0, 1)

    metrics: Dict[str, float] = {}
    for repeat_idx in range(repeats):
        _reset_pk_runtime_state(pk, active_tokens=num_tokens)
        qkv_out.zero_()
        attn_out_flat.zero_()
        linear_input.zero_()
        block_out.zero_()

        qkv_module.norm_linear(hidden_in.contiguous(), attn_norm_weight, qkv_weight, qkv_out, eps)
        qkv_diff = (qkv_out.float() - qkv_ref).abs()

        torch_k_cache = paged_k_cache.clone()
        torch_v_cache = paged_v_cache.clone()
        k = qkv_out[:, qo_heads * head_dim : qo_heads * head_dim + kv_heads * head_dim]
        v = qkv_out[:, qo_heads * head_dim + kv_heads * head_dim :]
        torch_k_cache[page_idx, page_offset : page_offset + num_tokens] = k
        torch_v_cache[page_idx, page_offset : page_offset + num_tokens] = v

        attn_module.multitoken_paged_attention(
            qkv_out.contiguous(),
            paged_k_cache,
            paged_v_cache,
            attn_out_flat,
            qo_indptr_buffer,
            paged_kv_indptr_buffer,
            paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer,
            0,
            False,
            False,
            q_norm_weight,
            k_norm_weight,
            all_cos,
            all_sin,
            eps,
            eps,
        )

        ref_attn = _reference_multitoken_paged_attention(
            q=qkv_out[:, : qo_heads * head_dim].view(num_tokens, qo_heads, head_dim),
            paged_k_cache=torch_k_cache,
            paged_v_cache=torch_v_cache,
            qo_heads=qo_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            num_tokens=num_tokens,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
        )
        attn_diff = (attn_out_flat.float() - ref_attn.float()).abs()

        linear_input.copy_(attn_out_flat.reshape(num_tokens, hidden_size))
        residual.copy_(hidden_in)
        pk()
        torch.cuda.synchronize()

        if not torch.isfinite(block_out.float()).all():
            metrics[f"repeat_{repeat_idx}_qkv_max_abs"] = float(qkv_diff.max().item())
            metrics[f"repeat_{repeat_idx}_qkv_mean_abs"] = float(qkv_diff.mean().item())
            metrics[f"repeat_{repeat_idx}_attn_max_abs"] = float(attn_diff.max().item())
            metrics[f"repeat_{repeat_idx}_attn_mean_abs"] = float(attn_diff.mean().item())
            metrics[f"repeat_{repeat_idx}_block_max_abs"] = float("inf")
            metrics[f"repeat_{repeat_idx}_block_mean_abs"] = float("inf")
            continue

        ref_block = ref_attn.reshape(num_tokens, hidden_size)
        ref_block = ref_block @ o_proj_weight.float().transpose(0, 1)
        ref_block = ref_block + hidden_in.float()
        block_diff = (block_out.float() - ref_block).abs()

        metrics[f"repeat_{repeat_idx}_qkv_max_abs"] = float(qkv_diff.max().item())
        metrics[f"repeat_{repeat_idx}_qkv_mean_abs"] = float(qkv_diff.mean().item())
        metrics[f"repeat_{repeat_idx}_attn_max_abs"] = float(attn_diff.max().item())
        metrics[f"repeat_{repeat_idx}_attn_mean_abs"] = float(attn_diff.mean().item())
        metrics[f"repeat_{repeat_idx}_block_max_abs"] = float(block_diff.max().item())
        metrics[f"repeat_{repeat_idx}_block_mean_abs"] = float(block_diff.mean().item())
    return metrics


def resolve_layer_plan_against_mirage(
    layer_plan: GemmaLayerLoweringPlan,
) -> list[MirageBindingResult]:
    """Resolve one layer plan against the live Mirage ``PersistentKernel`` API."""

    PersistentKernel = _require_mirage()
    results: list[MirageBindingResult] = []
    for step in layer_plan.mpk_steps:
        requested_method = step.mpk_method
        resolved = requested_method is not None and hasattr(PersistentKernel, requested_method)
        composed_methods = _COMPOSED_STEP_METHODS.get(step.name, ())
        composed_resolved = bool(composed_methods) and all(
            hasattr(PersistentKernel, method_name) for method_name in composed_methods
        )
        notes = list(step.notes)
        if requested_method is None:
            notes.append("No Mirage method requested for this step.")
        elif not resolved:
            notes.append("Requested Mirage method is not present on PersistentKernel.")
        else:
            notes.append("Resolved against installed Mirage PersistentKernel.")
        if composed_resolved:
            notes.append(
                "Bridge can lower this step via supported Mirage composition: "
                + ", ".join(composed_methods)
            )
        results.append(
            MirageBindingResult(
                step_name=step.name,
                requested_method=requested_method,
                status=step.status.value
                if isinstance(step.status, GemmaLoweringStatus)
                else str(step.status),
                resolved=resolved or composed_resolved,
                notes=notes,
            )
        )
    return results


def _compose_rmsnorm_linear_layer(
    pk,
    *,
    input_tensor,
    weight_norm_tensor,
    weight_linear_tensor,
    output_tensor,
    intermediate_name: str,
    grid_dim: tuple[int, int, int],
    block_dim: tuple[int, int, int],
):
    hidden_size = input_tensor.dim(1)
    intermediate = pk.new_tensor(
        (input_tensor.dim(0), hidden_size),
        name=intermediate_name,
    )
    pk.rmsnorm_layer(
        input=input_tensor,
        weight=weight_norm_tensor,
        output=intermediate,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )
    pk.linear_layer(
        input=intermediate,
        weight=weight_linear_tensor,
        output=output_tensor,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )


def _reset_pk_runtime_state(
    pk,
    *,
    active_tokens: int = 1,
    qo_indptr: Optional[torch.Tensor] = None,
    paged_kv_indptr: Optional[torch.Tensor] = None,
    paged_kv_indices: Optional[torch.Tensor] = None,
    paged_kv_last_page_len: Optional[torch.Tensor] = None,
    reinit_request_resources: bool = True,
) -> None:
    pk.meta_tensors["step"].zero_()
    pk.meta_tensors["tokens"].zero_()
    pk.meta_tensors["input_tokens"].zero_()
    pk.meta_tensors["output_tokens"].zero_()
    pk.meta_tensors["num_new_tokens"].fill_(1)
    pk.meta_tensors["prompt_lengths"].zero_()
    pk.meta_tensors["prompt_lengths"][0] = active_tokens
    pk.meta_tensors["tokens"][0, :active_tokens] = torch.arange(
        active_tokens, device=pk.meta_tensors["tokens"].device, dtype=torch.int64
    )
    pk.meta_tensors["qo_indptr_buffer"].zero_()
    pk.meta_tensors["paged_kv_indptr_buffer"].zero_()
    pk.meta_tensors["paged_kv_indices_buffer"].zero_()
    pk.meta_tensors["paged_kv_last_page_len_buffer"].zero_()
    if qo_indptr is not None:
        pk.meta_tensors["qo_indptr_buffer"][: qo_indptr.numel()] = qo_indptr
    if paged_kv_indptr is not None:
        pk.meta_tensors["paged_kv_indptr_buffer"][: paged_kv_indptr.numel()] = paged_kv_indptr
    if paged_kv_indices is not None:
        pk.meta_tensors["paged_kv_indices_buffer"][: paged_kv_indices.numel()] = paged_kv_indices
    if paged_kv_last_page_len is not None:
        pk.meta_tensors["paged_kv_last_page_len_buffer"][: paged_kv_last_page_len.numel()] = (
            paged_kv_last_page_len
        )
    if (
        reinit_request_resources
        and hasattr(pk, "init_request_func")
        and pk.init_request_func is not None
    ):
        pk.init_request_func()


def _moe_expert_grid_dim(pk, *, w13_linear: bool) -> tuple[int, int, int]:
    """Return the grid shape needed to cover all expert-offset shards.

    Mirage's MoE kernels shard activated experts by ``task_metadata.expert_offset``
    using a target-specific expert stride. Launch ``grid_dim.x`` to cover that
    full stride so a single logical MoE op computes all expert shards.
    """

    if getattr(pk, "target_cc", None) == 90:
        expert_stride = 5 if w13_linear else 4
    else:
        expert_stride = 10 if w13_linear else 8
    return (expert_stride, 1, 1)


def _node_arg_getattr_target(node: Node, arg_idx: int) -> str:
    arg = node.args[arg_idx]
    if not isinstance(arg, Node) or arg.op != "get_attr":
        raise ValueError(
            f"Expected get_attr argument at index {arg_idx} for node {node.name}, got {type(arg)}"
        )
    return str(arg.target)


def _lookup_tensor_attr(source_model: GraphModule, attr_name: str) -> torch.Tensor:
    candidate_names = [attr_name]
    flattened_attr_name = attr_name.replace(".", "_")
    if flattened_attr_name != attr_name:
        candidate_names.append(flattened_attr_name)

    for candidate_name in candidate_names:
        tensor = getattr(source_model, candidate_name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor

    for candidate_name in candidate_names:
        try:
            tensor = source_model.get_parameter(candidate_name)
        except AttributeError:
            tensor = None
        if isinstance(tensor, torch.Tensor):
            return tensor

        try:
            tensor = source_model.get_buffer(candidate_name)
        except AttributeError:
            tensor = None
        if isinstance(tensor, torch.Tensor):
            return tensor

    raise AttributeError(f"Missing expected GraphModule attribute: {attr_name}")


def _find_tensor_attr_by_substrings(
    source_model: GraphModule,
    *substrings: str,
) -> torch.Tensor:
    normalized = tuple(substrings)
    for name, tensor in source_model.named_buffers():
        flat_name = name.replace(".", "_")
        if all(substr in name or substr in flat_name for substr in normalized):
            return tensor
    for name, tensor in source_model.named_parameters():
        flat_name = name.replace(".", "_")
        if all(substr in name or substr in flat_name for substr in normalized):
            return tensor
    for candidate_name in dir(source_model):
        flat_name = candidate_name.replace(".", "_")
        if not all(substr in candidate_name or substr in flat_name for substr in normalized):
            continue
        tensor = getattr(source_model, candidate_name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor
    raise AttributeError(
        "Missing GraphModule tensor attribute containing substrings: "
        + ", ".join(repr(substr) for substr in normalized)
    )


def _node_target_name(node: Node) -> str:
    return getattr(node.target, "__name__", str(node.target))


def _node_target_matches_any(node: Node, *target_substrs: str) -> bool:
    target_name = _node_target_name(node)
    return any(target_substr in target_name for target_substr in target_substrs)


def _find_direct_user(node: Node, target_substr: str) -> Node:
    for user in node.users:
        if target_substr in _node_target_name(user):
            return user
    raise ValueError(
        f"Could not find direct user containing '{target_substr}' for node {node.name}"
    )


def _find_user_through_passthrough(node: Node, target_substr: str, max_depth: int = 4) -> Node:
    frontier = [node]
    visited = {id(node)}
    for _ in range(max_depth):
        next_frontier = []
        for current in frontier:
            for user in current.users:
                if id(user) in visited:
                    continue
                visited.add(id(user))
                if target_substr in _node_target_name(user):
                    return user
                next_frontier.append(user)
        frontier = next_frontier
    raise ValueError(f"Could not find user containing '{target_substr}' reachable from {node.name}")


def _extract_router_aux_attr_names(router_proj_node: Node) -> tuple[str, str]:
    router_scale_mul = router_proj_node.args[0]
    if not isinstance(router_scale_mul, Node):
        raise ValueError("Expected router projection input to be a node")
    router_scale_to = router_scale_mul.args[1]
    if not isinstance(router_scale_to, Node):
        raise ValueError("Expected router scale input conversion node")
    router_scale_name = _node_arg_getattr_target(router_scale_to, 0)

    router_root_mul = router_scale_mul.args[0]
    if not isinstance(router_root_mul, Node):
        raise ValueError("Expected router root-size multiply node")
    router_root_to = router_root_mul.args[1]
    if not isinstance(router_root_to, Node):
        raise ValueError("Expected router root-size conversion node")
    router_root_size_name = _node_arg_getattr_target(router_root_to, 0)
    return router_root_size_name, router_scale_name


@dataclass
class _GemmaRuntimeLayerSpec:
    layer_index: int
    q_heads: int
    kv_heads: int
    head_dim: int
    topk: int
    sliding_window: Optional[int]
    qkv_weight: torch.Tensor
    qkv_shared_kv: bool
    input_layernorm_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    v_norm_weight: torch.Tensor
    o_proj_weight: torch.Tensor
    post_attention_layernorm_weight: torch.Tensor
    pre_feedforward_layernorm_weight: torch.Tensor
    ffn_gate_up_weight: torch.Tensor
    ffn_down_weight: torch.Tensor
    post_feedforward_layernorm_1_weight: torch.Tensor
    router_proj_weight: torch.Tensor
    router_root_size: torch.Tensor
    router_scale: torch.Tensor
    pre_feedforward_layernorm_2_weight: torch.Tensor
    moe_w13_stacked_weight: torch.Tensor
    moe_gate_weight: torch.Tensor
    moe_up_weight: torch.Tensor
    moe_w2_weight: torch.Tensor
    post_feedforward_layernorm_2_weight: torch.Tensor
    post_feedforward_layernorm_weight: torch.Tensor
    layer_scalar: torch.Tensor


def _build_gemma_runtime_specs(
    source_model: GraphModule,
    translation_plan: Dict[str, Any],
) -> list[_GemmaRuntimeLayerSpec]:
    node_map = {node.name: node for node in source_model.graph.nodes}
    layer_infos = translation_plan["graph_info"]["layer_infos"]
    layer_specs: list[_GemmaRuntimeLayerSpec] = []

    for layer_info in layer_infos:
        layer_idx = int(layer_info["layer_index"])
        anchors = layer_info["anchors"]

        qkv_node = node_map[anchors["qkv_linear"]["name"]]
        ffn_gate_up_node = node_map[anchors["ffn_gate_up"]["name"]]
        ffn_down_node = node_map[anchors["ffn_down"]["name"]]
        o_proj_node = node_map[anchors["o_proj"]["name"]]
        topk_node = node_map[anchors["topk"]["name"]]
        router_proj_node = node_map[anchors["router_proj"]["name"]]
        moe_fused_node = node_map[anchors["moe_fused"]["name"]]
        cached_attention_node = node_map[anchors["cached_attention"]["name"]]
        q_norm_node = node_map[anchors["q_norm"]["name"]]
        k_norm_node = node_map[anchors["k_norm"]["name"]]
        v_norm_node = node_map[anchors["v_norm"]["name"]]

        qkv_weight_name = _node_arg_getattr_target(qkv_node, 1)
        ffn_gate_up_weight_name = _node_arg_getattr_target(ffn_gate_up_node, 1)
        ffn_down_weight_name = _node_arg_getattr_target(ffn_down_node, 1)
        o_proj_weight_name = _node_arg_getattr_target(o_proj_node, 1)
        router_proj_weight_name = _node_arg_getattr_target(router_proj_node, 1)
        q_norm_weight_name = _node_arg_getattr_target(q_norm_node, 1)
        k_norm_weight_name = _node_arg_getattr_target(k_norm_node, 1)
        v_norm_weight_name = _node_arg_getattr_target(v_norm_node, 1)
        topk = int(topk_node.args[1])
        sliding_window = None
        if len(cached_attention_node.args) >= 16 and cached_attention_node.args[15] is not None:
            sliding_window = int(cached_attention_node.args[15])

        input_layernorm_node = qkv_node.args[0]
        if not isinstance(input_layernorm_node, Node):
            raise ValueError("Expected qkv linear input to be a norm node")
        input_layernorm_weight_name = _node_arg_getattr_target(input_layernorm_node, 1)

        post_attention_layernorm_node = _find_direct_user(o_proj_node, "flashinfer_rms_norm")
        post_attention_layernorm_weight_name = _node_arg_getattr_target(
            post_attention_layernorm_node, 1
        )

        pre_feedforward_layernorm_node = ffn_gate_up_node.args[0]
        if not isinstance(pre_feedforward_layernorm_node, Node):
            raise ValueError("Expected FFN gate/up input to be a norm node")
        pre_feedforward_layernorm_weight_name = _node_arg_getattr_target(
            pre_feedforward_layernorm_node, 1
        )

        post_feedforward_layernorm_1_node = _find_direct_user(ffn_down_node, "flashinfer_rms_norm")
        post_feedforward_layernorm_1_weight_name = _node_arg_getattr_target(
            post_feedforward_layernorm_1_node, 1
        )

        router_root_size_name, router_scale_name = _extract_router_aux_attr_names(router_proj_node)

        pre_feedforward_layernorm_2_to = moe_fused_node.args[0]
        if not isinstance(pre_feedforward_layernorm_2_to, Node):
            raise ValueError("Expected MoE fused input to be a cast node")
        pre_feedforward_layernorm_2_node = pre_feedforward_layernorm_2_to.args[0]
        if not isinstance(pre_feedforward_layernorm_2_node, Node):
            raise ValueError("Expected MoE fused cast input to be a norm node")
        pre_feedforward_layernorm_2_weight_name = _node_arg_getattr_target(
            pre_feedforward_layernorm_2_node, 1
        )

        moe_w13_name = _node_arg_getattr_target(moe_fused_node, 3)
        moe_w2_name = _node_arg_getattr_target(moe_fused_node, 4)
        post_feedforward_layernorm_2_node = _find_user_through_passthrough(
            moe_fused_node, "flashinfer_rms_norm"
        )
        post_feedforward_layernorm_2_weight_name = _node_arg_getattr_target(
            post_feedforward_layernorm_2_node, 1
        )

        ffn_moe_add_node = next(
            user
            for user in post_feedforward_layernorm_1_node.users
            if user in post_feedforward_layernorm_2_node.users
        )
        post_feedforward_layernorm_node = _find_direct_user(ffn_moe_add_node, "flashinfer_rms_norm")
        post_feedforward_layernorm_weight_name = _node_arg_getattr_target(
            post_feedforward_layernorm_node, 1
        )
        post_attention_residual_node = next(
            user
            for user in post_attention_layernorm_node.users
            if _node_target_matches_any(user, "aten.add", "add.Tensor", "add")
        )
        final_add_node = next(
            user
            for user in post_feedforward_layernorm_node.users
            if _node_target_matches_any(user, "aten.add", "add.Tensor", "add")
            and any(
                isinstance(arg, Node) and arg is post_attention_residual_node for arg in user.args
            )
        )
        layer_scalar_mul_node = next(
            user
            for user in final_add_node.users
            if _node_target_matches_any(user, "aten.mul", "mul.Tensor", "mul")
        )
        layer_scalar_name = _node_arg_getattr_target(layer_scalar_mul_node, 1)

        q_heads = int(layer_info["q_heads"])
        kv_heads = int(layer_info["kv_heads"])
        head_dim = int(layer_info["head_dim"])
        q_size = q_heads * head_dim
        kv_size = kv_heads * head_dim
        qkv_weight = _lookup_tensor_attr(source_model, qkv_weight_name)
        qkv_shared_kv = int(qkv_weight.shape[0]) == q_size + kv_size

        moe_w13_stacked = _lookup_tensor_attr(source_model, moe_w13_name)
        # The fused MoE stack uses TRT-LLM's ``w3_w1`` layout:
        # first half is the up projection (w3), second half is the gate
        # projection (w1). Gemma's expert activation is ``gelu(gate) * up``,
        # so preserve that semantic mapping here instead of reusing the
        # physical stack order directly.
        up_weight, gate_weight = torch.chunk(moe_w13_stacked, 2, dim=1)

        layer_specs.append(
            _GemmaRuntimeLayerSpec(
                layer_index=layer_idx,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                topk=topk,
                sliding_window=sliding_window,
                qkv_weight=qkv_weight,
                qkv_shared_kv=qkv_shared_kv,
                input_layernorm_weight=_lookup_tensor_attr(
                    source_model, input_layernorm_weight_name
                ),
                q_norm_weight=_lookup_tensor_attr(source_model, q_norm_weight_name),
                k_norm_weight=_lookup_tensor_attr(source_model, k_norm_weight_name),
                v_norm_weight=_lookup_tensor_attr(source_model, v_norm_weight_name),
                o_proj_weight=_lookup_tensor_attr(source_model, o_proj_weight_name),
                post_attention_layernorm_weight=_lookup_tensor_attr(
                    source_model, post_attention_layernorm_weight_name
                ),
                pre_feedforward_layernorm_weight=_lookup_tensor_attr(
                    source_model, pre_feedforward_layernorm_weight_name
                ),
                ffn_gate_up_weight=_lookup_tensor_attr(source_model, ffn_gate_up_weight_name),
                ffn_down_weight=_lookup_tensor_attr(source_model, ffn_down_weight_name),
                post_feedforward_layernorm_1_weight=_lookup_tensor_attr(
                    source_model, post_feedforward_layernorm_1_weight_name
                ),
                router_proj_weight=_lookup_tensor_attr(source_model, router_proj_weight_name),
                router_root_size=_lookup_tensor_attr(source_model, router_root_size_name),
                router_scale=_lookup_tensor_attr(source_model, router_scale_name),
                pre_feedforward_layernorm_2_weight=_lookup_tensor_attr(
                    source_model, pre_feedforward_layernorm_2_weight_name
                ),
                moe_w13_stacked_weight=moe_w13_stacked,
                moe_gate_weight=gate_weight,
                moe_up_weight=up_weight,
                moe_w2_weight=_lookup_tensor_attr(source_model, moe_w2_name),
                post_feedforward_layernorm_2_weight=_lookup_tensor_attr(
                    source_model, post_feedforward_layernorm_2_weight_name
                ),
                post_feedforward_layernorm_weight=_lookup_tensor_attr(
                    source_model, post_feedforward_layernorm_weight_name
                ),
                layer_scalar=_lookup_tensor_attr(source_model, layer_scalar_name),
            )
        )

    return layer_specs


def _should_use_single_pk_layer(
    *,
    batch_size: int,
    seq_len: int,
    num_prefill: int,
) -> bool:
    return (
        os.getenv("AD_MPK_ENABLE_SINGLE_PK_LAYER", "0") == "1"
        and batch_size == 1
        and seq_len == 1
        and num_prefill == 0
    )


@dataclass
class _GemmaQkvBlockResult:
    qkv_packed_view: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass
class _GemmaAttentionBlockResult:
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    attn_out_view: torch.Tensor
    attn_out_flat: torch.Tensor


@dataclass
class _GemmaAttentionOutputBlockResult:
    o_proj: torch.Tensor
    post_attn: torch.Tensor


@dataclass
class _GemmaDenseFfnBlockResult:
    ffn_down: torch.Tensor
    ffn_norm: torch.Tensor


@dataclass
class _GemmaMoeBlockResult:
    moe_out: torch.Tensor
    moe_norm: torch.Tensor
    ffn_moe_add: torch.Tensor
    ffn_moe_norm: torch.Tensor
    hidden: torch.Tensor
    debug_tensors: Dict[str, torch.Tensor]


@dataclass
class _GemmaLayerBlockResult:
    hidden: torch.Tensor
    debug_tensors: Dict[str, torch.Tensor]


class _GemmaQkvBlock:
    def __init__(self, runtime: "_GemmaMirageRuntime", layer_spec: _GemmaRuntimeLayerSpec) -> None:
        self.runtime = runtime
        self.layer_spec = layer_spec

    def __call__(self, *, hidden: torch.Tensor) -> _GemmaQkvBlockResult:
        batch_size, seq_len = [int(dim) for dim in hidden.shape[:2]]
        attn_in = _rms_norm(hidden, self.layer_spec.input_layernorm_weight).reshape(
            batch_size * seq_len, -1
        )
        qkv_packed = self.runtime._linear(
            name=f"layer_{self.layer_spec.layer_index}_qkv",
            input_tensor=attn_in,
            weight_tensor=self.layer_spec.qkv_weight,
        )
        qkv_packed_view = qkv_packed.view(batch_size, seq_len, -1)

        q_size = self.layer_spec.q_heads * self.layer_spec.head_dim
        kv_size = self.layer_spec.kv_heads * self.layer_spec.head_dim
        if self.layer_spec.qkv_shared_kv:
            q_flat = qkv_packed[:, :q_size]
            shared_kv = qkv_packed[:, q_size : q_size + kv_size]
            k_flat = shared_kv
            v_flat = shared_kv
        else:
            q_flat = qkv_packed[:, :q_size]
            k_flat = qkv_packed[:, q_size : q_size + kv_size]
            v_flat = qkv_packed[:, q_size + kv_size : q_size + 2 * kv_size]

        q = q_flat.view(batch_size, seq_len, self.layer_spec.q_heads, self.layer_spec.head_dim)
        k = k_flat.view(batch_size, seq_len, self.layer_spec.kv_heads, self.layer_spec.head_dim)
        v = v_flat.view(batch_size, seq_len, self.layer_spec.kv_heads, self.layer_spec.head_dim)
        q = _rms_norm(q, self.layer_spec.q_norm_weight)
        k = _rms_norm(k, self.layer_spec.k_norm_weight)
        v = _rms_norm(v, self.layer_spec.v_norm_weight)
        return _GemmaQkvBlockResult(qkv_packed_view=qkv_packed_view, q=q, k=k, v=v)


class _GemmaAttentionBlock:
    def __init__(self, runtime: "_GemmaMirageRuntime", layer_spec: _GemmaRuntimeLayerSpec) -> None:
        self.runtime = runtime
        self.layer_spec = layer_spec

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: torch.Tensor,
        batch_info_host: torch.Tensor,
        cu_seqlen_host: torch.Tensor,
        cu_num_pages: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        last_page_len: torch.Tensor,
        last_page_len_host: torch.Tensor,
        seq_len_with_cache_host: torch.Tensor,
        triton_batch_indices: torch.Tensor,
        triton_positions: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> _GemmaAttentionBlockResult:
        batch_size, seq_len = [int(dim) for dim in q.shape[:2]]
        rope_positions, cos_sin_cache = self.runtime._rope_inputs(
            position_ids=position_ids,
            head_dim=self.layer_spec.head_dim,
            batch_size=batch_size,
            seq_len=seq_len,
        )
        q_rope, k_rope = torch.ops.auto_deploy.flashinfer_rope.default(
            q,
            k,
            rope_positions,
            cos_sin_cache,
            True,
        )
        attn_out = torch.ops.auto_deploy.triton_paged_mha_with_cache.default(
            q_rope,
            k_rope,
            v,
            batch_info_host,
            cu_seqlen_host,
            cu_num_pages,
            cu_num_pages_host,
            cache_loc,
            last_page_len,
            last_page_len_host,
            seq_len_with_cache_host,
            triton_batch_indices,
            triton_positions,
            kv_cache,
            1.0,
            self.layer_spec.sliding_window,
        )
        attn_out_flat = attn_out.reshape(batch_size * seq_len, -1).contiguous()
        return _GemmaAttentionBlockResult(
            q_rope=q_rope,
            k_rope=k_rope,
            attn_out_view=attn_out_flat.view(batch_size, seq_len, -1),
            attn_out_flat=attn_out_flat,
        )


class _GemmaAttentionOutputBlock:
    def __init__(self, runtime: "_GemmaMirageRuntime", layer_spec: _GemmaRuntimeLayerSpec) -> None:
        self.runtime = runtime
        self.layer_spec = layer_spec

    def __call__(
        self,
        *,
        hidden: torch.Tensor,
        attn_out_flat: torch.Tensor,
    ) -> _GemmaAttentionOutputBlockResult:
        batch_size, seq_len = [int(dim) for dim in hidden.shape[:2]]
        o_proj = self.runtime._linear(
            name=f"layer_{self.layer_spec.layer_index}_o_proj",
            input_tensor=attn_out_flat,
            weight_tensor=self.layer_spec.o_proj_weight,
        ).view(batch_size, seq_len, -1)
        post_attn = _rms_norm(o_proj, self.layer_spec.post_attention_layernorm_weight) + hidden
        return _GemmaAttentionOutputBlockResult(o_proj=o_proj, post_attn=post_attn)


class _GemmaDenseFfnBlock:
    def __init__(self, runtime: "_GemmaMirageRuntime", layer_spec: _GemmaRuntimeLayerSpec) -> None:
        self.runtime = runtime
        self.layer_spec = layer_spec

    def __call__(self, *, post_attn: torch.Tensor) -> _GemmaDenseFfnBlockResult:
        batch_size, seq_len = [int(dim) for dim in post_attn.shape[:2]]
        post_attn_flat = post_attn.reshape(batch_size * seq_len, -1)
        if self.runtime._disable_fused_dense_ffn:
            ffn_normed = _rms_norm(post_attn_flat, self.layer_spec.pre_feedforward_layernorm_weight)
            ffn_gate_up = self.runtime._linear(
                name=f"layer_{self.layer_spec.layer_index}_ffn_gate_up_unfused",
                input_tensor=ffn_normed,
                weight_tensor=self.layer_spec.ffn_gate_up_weight,
            )
            ffn_gate, ffn_up = torch.chunk(ffn_gate_up, 2, dim=-1)
            ffn_act = self.runtime._gelu_mul(ffn_gate, ffn_up)
            ffn_down = self.runtime._ffn_down_decode(
                act_tensor=ffn_act,
                weight_tensor=self.layer_spec.ffn_down_weight,
            ).view(batch_size, seq_len, -1)
        else:
            ffn_down = self.runtime._dense_ffn(
                post_attn_tensor=post_attn_flat,
                norm_weight_tensor=self.layer_spec.pre_feedforward_layernorm_weight,
                gate_up_weight_tensor=self.layer_spec.ffn_gate_up_weight,
                down_weight_tensor=self.layer_spec.ffn_down_weight,
            ).view(batch_size, seq_len, -1)
        ffn_norm = _rms_norm(ffn_down, self.layer_spec.post_feedforward_layernorm_1_weight)
        return _GemmaDenseFfnBlockResult(ffn_down=ffn_down, ffn_norm=ffn_norm)


class _GemmaMoeBlock:
    def __init__(self, runtime: "_GemmaMirageRuntime", layer_spec: _GemmaRuntimeLayerSpec) -> None:
        self.runtime = runtime
        self.layer_spec = layer_spec

    def __call__(
        self,
        *,
        post_attn: torch.Tensor,
        ffn_norm: torch.Tensor,
        capture_debug: bool,
    ) -> _GemmaMoeBlockResult:
        batch_size, seq_len = [int(dim) for dim in post_attn.shape[:2]]
        post_attn_flat = post_attn.reshape(batch_size * seq_len, -1).contiguous()
        moe_token_outputs = []
        debug_tensors: Dict[str, torch.Tensor] = {}

        for token_idx in range(batch_size * seq_len):
            token_hidden = post_attn_flat[token_idx : token_idx + 1].contiguous()
            router_in = token_hidden.float()
            router_mean = router_in.pow(2).mean(dim=-1, keepdim=True)
            router_in = router_in * torch.rsqrt(router_mean + 1e-6)
            router_in = router_in.to(torch.bfloat16)
            router_in = router_in * self.layer_spec.router_root_size.to(dtype=router_in.dtype)
            router_in = router_in * self.layer_spec.router_scale.to(dtype=router_in.dtype)
            topk_weight, routing_indices, routing_mask = self.runtime._router(
                hidden_tensor=router_in,
                weight_tensor=self.layer_spec.router_proj_weight,
                topk=self.layer_spec.topk,
            )

            moe_in = _rms_norm(token_hidden, self.layer_spec.pre_feedforward_layernorm_2_weight)
            experts = _extract_ranked_experts(routing_indices, self.layer_spec.topk)
            if capture_debug and token_idx == 0:
                router_logits = self.runtime._linear(
                    name=f"layer_{self.layer_spec.layer_index}_router_proj_debug",
                    input_tensor=router_in,
                    weight_tensor=self.layer_spec.router_proj_weight,
                )
                debug_tensors["layer_0_router_in"] = (
                    router_in.detach().clone().view(batch_size, seq_len, -1)
                )
                debug_tensors["layer_0_router_logits"] = (
                    router_logits.detach().clone().view(batch_size, seq_len, -1)
                )
                debug_tensors["layer_0_router_topk_weight"] = (
                    topk_weight.detach().clone().view(batch_size, seq_len, -1)
                )
                debug_tensors["layer_0_router_topk_indices"] = torch.tensor(
                    experts,
                    device=routing_indices.device,
                    dtype=torch.int32,
                ).view(batch_size, seq_len, -1)
                debug_tensors["layer_0_moe_in"] = (
                    moe_in.detach().clone().view(batch_size, seq_len, -1)
                )

            if self.runtime._disable_fused_moe:
                moe_token_out = self.runtime._split_moe(
                    hidden_tensor=moe_in,
                    w13_weight_tensor=self.layer_spec.moe_w13_stacked_weight,
                    routing_indices=routing_indices,
                    routing_mask=routing_mask,
                    w2_weight_tensor=self.layer_spec.moe_w2_weight,
                    topk_weight_tensor=topk_weight,
                )
            else:
                moe_token_out = self.runtime._fused_moe(
                    hidden_tensor=moe_in,
                    w13_weight_tensor=self.layer_spec.moe_w13_stacked_weight,
                    routing_indices=routing_indices,
                    routing_mask=routing_mask,
                    topk_weight_tensor=topk_weight,
                    w2_weight_tensor=self.layer_spec.moe_w2_weight,
                )
            moe_token_outputs.append(moe_token_out[0])

        moe_out = torch.stack(moe_token_outputs, dim=0).view(batch_size, seq_len, -1)
        moe_norm = _rms_norm(moe_out, self.layer_spec.post_feedforward_layernorm_2_weight)
        ffn_moe_add = ffn_norm + moe_norm
        ffn_moe_norm = _rms_norm(ffn_moe_add, self.layer_spec.post_feedforward_layernorm_weight)
        hidden = post_attn + ffn_moe_norm
        hidden = hidden * self.layer_spec.layer_scalar.view(1, 1, -1)
        return _GemmaMoeBlockResult(
            moe_out=moe_out,
            moe_norm=moe_norm,
            ffn_moe_add=ffn_moe_add,
            ffn_moe_norm=ffn_moe_norm,
            hidden=hidden,
            debug_tensors=debug_tensors,
        )


class _GemmaDecodeLayerBlock:
    def __init__(self, runtime: "_GemmaMirageRuntime", layer_spec: _GemmaRuntimeLayerSpec) -> None:
        self.layer_spec = layer_spec
        self.qkv_block = _GemmaQkvBlock(runtime, layer_spec)
        self.attention_block = _GemmaAttentionBlock(runtime, layer_spec)
        self.attention_output_block = _GemmaAttentionOutputBlock(runtime, layer_spec)
        self.dense_ffn_block = _GemmaDenseFfnBlock(runtime, layer_spec)
        self.moe_block = _GemmaMoeBlock(runtime, layer_spec)

    def __call__(
        self,
        *,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        batch_info_host: torch.Tensor,
        cu_seqlen_host: torch.Tensor,
        cu_num_pages: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        last_page_len: torch.Tensor,
        last_page_len_host: torch.Tensor,
        seq_len_with_cache_host: torch.Tensor,
        triton_batch_indices: torch.Tensor,
        triton_positions: torch.Tensor,
        kv_cache: torch.Tensor,
        capture_debug: bool,
    ) -> _GemmaLayerBlockResult:
        layer_label = f"layer_{self.layer_spec.layer_index}"
        qkv_outputs = _mpk_run_profiled(
            f"{layer_label}.qkv",
            lambda: self.qkv_block(hidden=hidden),
        )
        attention_outputs = _mpk_run_profiled(
            f"{layer_label}.attention",
            lambda: self.attention_block(
                q=qkv_outputs.q,
                k=qkv_outputs.k,
                v=qkv_outputs.v,
                position_ids=position_ids,
                batch_info_host=batch_info_host,
                cu_seqlen_host=cu_seqlen_host,
                cu_num_pages=cu_num_pages,
                cu_num_pages_host=cu_num_pages_host,
                cache_loc=cache_loc,
                last_page_len=last_page_len,
                last_page_len_host=last_page_len_host,
                seq_len_with_cache_host=seq_len_with_cache_host,
                triton_batch_indices=triton_batch_indices,
                triton_positions=triton_positions,
                kv_cache=kv_cache,
            ),
        )
        attention_output = _mpk_run_profiled(
            f"{layer_label}.attn_out",
            lambda: self.attention_output_block(
                hidden=hidden,
                attn_out_flat=attention_outputs.attn_out_flat,
            ),
        )
        dense_ffn_output = _mpk_run_profiled(
            f"{layer_label}.dense_ffn",
            lambda: self.dense_ffn_block(post_attn=attention_output.post_attn),
        )
        moe_output = _mpk_run_profiled(
            f"{layer_label}.moe",
            lambda: self.moe_block(
                post_attn=attention_output.post_attn,
                ffn_norm=dense_ffn_output.ffn_norm,
                capture_debug=capture_debug,
            ),
        )

        debug_tensors: Dict[str, torch.Tensor] = {}
        if capture_debug:
            debug_tensors["layer_0_qkv_packed"] = qkv_outputs.qkv_packed_view.detach().clone()
            debug_tensors["layer_0_q_norm"] = qkv_outputs.q.detach().clone()
            debug_tensors["layer_0_k_norm"] = qkv_outputs.k.detach().clone()
            debug_tensors["layer_0_v_norm"] = qkv_outputs.v.detach().clone()
            debug_tensors["layer_0_q_rope"] = attention_outputs.q_rope.detach().clone()
            debug_tensors["layer_0_k_rope"] = attention_outputs.k_rope.detach().clone()
            debug_tensors["layer_0_attn_out"] = attention_outputs.attn_out_view.detach().clone()
            debug_tensors["layer_0_o_proj"] = attention_output.o_proj.detach().clone()
            debug_tensors["layer_0_post_attn"] = attention_output.post_attn.detach().clone()
            debug_tensors["layer_0_ffn_down"] = dense_ffn_output.ffn_down.detach().clone()
            debug_tensors["layer_0_ffn_norm"] = dense_ffn_output.ffn_norm.detach().clone()
            debug_tensors["layer_0_moe_out"] = moe_output.moe_out.detach().clone()
            debug_tensors["layer_0_moe_norm"] = moe_output.moe_norm.detach().clone()
            debug_tensors["layer_0_ffn_moe_add"] = moe_output.ffn_moe_add.detach().clone()
            debug_tensors["layer_0_ffn_moe_norm"] = moe_output.ffn_moe_norm.detach().clone()
            debug_tensors["layer_0_hidden"] = moe_output.hidden.detach().clone()
            debug_tensors.update(moe_output.debug_tensors)

        return _GemmaLayerBlockResult(hidden=moe_output.hidden, debug_tensors=debug_tensors)


class _MirageLinearExecutor:
    def __init__(self, *, capacity: int, in_dim: int, out_dim: int, name: str):
        self.capacity = capacity
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._last_actual_batch: Optional[int] = None
        self.input_name = f"{name}_input"
        self.weight_name = f"{name}_weight"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.input = torch.empty((capacity, in_dim), device="cuda", dtype=torch.bfloat16)
        self.weight = torch.empty((out_dim, in_dim), device="cuda", dtype=torch.bfloat16)
        self.output = torch.empty((capacity, out_dim), device="cuda", dtype=torch.bfloat16)
        input_dt = self.pk.attach_input(self.input, name=self.input_name)
        weight_dt = self.pk.attach_input(self.weight, name=self.weight_name)
        output_dt = self.pk.attach_input(self.output, name=f"{name}_output")
        self.pk.linear_layer(
            input=input_dt,
            weight=weight_dt,
            output=output_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        """Patch the PK for a new actual batch size if it changed."""
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> torch.Tensor:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            input_binding = torch.zeros(
                (cap, self.in_dim), device=input_tensor.device, dtype=input_tensor.dtype
            )
            input_binding[:actual_batch] = input_tensor.contiguous().view(actual_batch, self.in_dim)
        else:
            input_binding = input_tensor.contiguous().view(actual_batch, self.in_dim)
        weight_binding = weight_tensor.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        _mpk_profile_executor(
            f"linear.launch capacity={self.capacity} actual={actual_batch} in_dim={self.in_dim} out_dim={self.out_dim}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.input_name: input_binding,
                    self.weight_name: weight_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return self.output[:actual_batch]


class _MirageRouterExecutor:
    def __init__(self, *, capacity: int, hidden_size: int, num_experts: int, topk: int):
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk
        self._last_actual_batch: Optional[int] = None
        self.hidden_name = "router_hidden"
        self.weight_name = "router_weight"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.hidden = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.weight = torch.empty((num_experts, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.logits = torch.empty((capacity, num_experts), device="cuda", dtype=torch.bfloat16)
        self.topk_weight = torch.empty((capacity, topk), device="cuda", dtype=torch.float32)
        self.routing_indices = torch.empty(
            (num_experts, capacity), device="cuda", dtype=torch.int32
        )
        self.routing_mask = torch.empty((num_experts + 1,), device="cuda", dtype=torch.int32)
        hidden_dt = self.pk.attach_input(self.hidden, name=self.hidden_name)
        weight_dt = self.pk.attach_input(self.weight, name=self.weight_name)
        logits_dt = self.pk.attach_input(self.logits, name="router_logits")
        topk_weight_dt = self.pk.attach_input(self.topk_weight, name="router_topk_weight")
        routing_indices_dt = self.pk.attach_input(
            self.routing_indices, name="router_routing_indices"
        )
        routing_mask_dt = self.pk.attach_input(self.routing_mask, name="router_routing_mask")
        self.pk.linear_layer(
            input=hidden_dt,
            weight=weight_dt,
            output=logits_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_topk_softmax_routing_layer(
            input=logits_dt,
            output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        hidden_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            hidden_binding = torch.zeros(
                (cap, self.hidden_size), device=hidden_tensor.device, dtype=hidden_tensor.dtype
            )
            hidden_binding[:actual_batch] = hidden_tensor.contiguous().view(
                actual_batch, self.hidden_size
            )
        else:
            hidden_binding = hidden_tensor.contiguous().view(actual_batch, self.hidden_size)
        weight_binding = weight_tensor.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        _mpk_profile_executor(
            f"router.launch capacity={self.capacity} actual={actual_batch} "
            f"hidden={self.hidden_size} experts={self.num_experts}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.hidden_name: hidden_binding,
                    self.weight_name: weight_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return (
            self.topk_weight[:actual_batch],
            self.routing_indices[:, :actual_batch],
            self.routing_mask,
        )


class _MirageMoeW13Executor:
    def __init__(
        self,
        *,
        capacity: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
    ):
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk
        self._last_actual_batch: Optional[int] = None
        self.hidden_name = "moe_w13_hidden"
        self.gate_weight_name = "moe_w13_gate_weight"
        self.up_weight_name = "moe_w13_up_weight"
        self.routing_indices_name = "moe_w13_routing_indices"
        self.routing_mask_name = "moe_w13_routing_mask"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.hidden = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.gate_weight = torch.empty(
            (num_experts, intermediate_size, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.up_weight = torch.empty_like(self.gate_weight)
        self.routing_indices = torch.empty(
            (num_experts, capacity), device="cuda", dtype=torch.int32
        )
        self.routing_mask = torch.empty((num_experts + 1,), device="cuda", dtype=torch.int32)
        self.gate_out = torch.zeros(
            (capacity, topk, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.up_out = torch.empty_like(self.gate_out)
        hidden_dt = self.pk.attach_input(self.hidden, name=self.hidden_name)
        gate_weight_dt = self.pk.attach_input(self.gate_weight, name=self.gate_weight_name)
        up_weight_dt = self.pk.attach_input(self.up_weight, name=self.up_weight_name)
        routing_indices_dt = self.pk.attach_input(
            self.routing_indices, name=self.routing_indices_name
        )
        routing_mask_dt = self.pk.attach_input(self.routing_mask, name=self.routing_mask_name)
        gate_out_dt = self.pk.attach_input(self.gate_out, name="moe_w13_gate_out")
        up_out_dt = self.pk.attach_input(self.up_out, name="moe_w13_up_out")
        self.pk.moe_w13_linear_layer(
            input=hidden_dt,
            weight=gate_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=gate_out_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=True),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_w13_linear_layer(
            input=hidden_dt,
            weight=up_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=up_out_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=True),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        hidden_tensor: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            hidden_binding = torch.zeros(
                (cap, self.hidden_size), device=hidden_tensor.device, dtype=hidden_tensor.dtype
            )
            hidden_binding[:actual_batch] = hidden_tensor.contiguous().view(
                actual_batch, self.hidden_size
            )
            ri_padded = torch.zeros(
                (self.num_experts, cap), device=routing_indices.device, dtype=routing_indices.dtype
            )
            ri_padded[:, :actual_batch] = routing_indices[:, :actual_batch]
            routing_indices_binding = ri_padded
        else:
            hidden_binding = hidden_tensor.contiguous().view(actual_batch, self.hidden_size)
            routing_indices_binding = routing_indices.contiguous()
        gate_weight_binding = gate_weight.contiguous()
        up_weight_binding = up_weight.contiguous()
        routing_mask_binding = routing_mask.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        self.gate_out.zero_()
        self.up_out.zero_()
        _mpk_profile_executor(
            "moe_w13.launch "
            f"capacity={self.capacity} actual={actual_batch} hidden={self.hidden_size} "
            f"intermediate={self.intermediate_size} experts={self.num_experts} topk={self.topk}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.hidden_name: hidden_binding,
                    self.gate_weight_name: gate_weight_binding,
                    self.up_weight_name: up_weight_binding,
                    self.routing_indices_name: routing_indices_binding,
                    self.routing_mask_name: routing_mask_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return self.gate_out[:actual_batch], self.up_out[:actual_batch]


class _MirageDenseFfnExecutor:
    def __init__(self, *, capacity: int, hidden_size: int, intermediate_size: int):
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._last_actual_batch: Optional[int] = None
        self.input_name = "dense_ffn_input"
        self.norm_weight_name = "dense_ffn_norm_weight"
        self.gate_up_weight_name = "dense_ffn_gate_up_weight"
        self.down_weight_name = "dense_ffn_down_weight"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.input = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.norm_weight = torch.empty((hidden_size,), device="cuda", dtype=torch.bfloat16)
        self.gate_up_weight = torch.empty(
            (2 * intermediate_size, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.down_weight = torch.empty(
            (hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.gate_up = torch.empty(
            (capacity, 2 * intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.zero_residual = torch.zeros(
            (capacity, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.output = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        input_dt = self.pk.attach_input(self.input, name=self.input_name)
        norm_weight_dt = self.pk.attach_input(self.norm_weight, name=self.norm_weight_name)
        gate_up_weight_dt = self.pk.attach_input(self.gate_up_weight, name=self.gate_up_weight_name)
        down_weight_dt = self.pk.attach_input(self.down_weight, name=self.down_weight_name)
        gate_up_dt = self.pk.attach_input(self.gate_up, name="dense_ffn_gate_up")
        zero_residual_dt = self.pk.attach_input(self.zero_residual, name="dense_ffn_zero_residual")
        output_dt = self.pk.attach_input(self.output, name="dense_ffn_output")
        self.pk.rmsnorm_linear_layer(
            input=input_dt,
            weight_norm=norm_weight_dt,
            weight_linear=gate_up_weight_dt,
            output=gate_up_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.pk.gelu_mul_linear_with_residual_layer(
            input=gate_up_dt,
            weight=down_weight_dt,
            residual=zero_residual_dt,
            output=output_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        input_tensor: torch.Tensor,
        norm_weight_tensor: torch.Tensor,
        gate_up_weight_tensor: torch.Tensor,
        down_weight_tensor: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> torch.Tensor:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            input_binding = torch.zeros(
                (cap, self.hidden_size), device=input_tensor.device, dtype=input_tensor.dtype
            )
            input_binding[:actual_batch] = input_tensor.contiguous().view(
                actual_batch, self.hidden_size
            )
        else:
            input_binding = input_tensor.contiguous().view(actual_batch, self.hidden_size)
        norm_weight_binding = norm_weight_tensor.contiguous().view(self.hidden_size)
        gate_up_weight_binding = gate_up_weight_tensor.contiguous()
        down_weight_binding = down_weight_tensor.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        _mpk_profile_executor(
            "dense_ffn.launch "
            f"capacity={self.capacity} actual={actual_batch} "
            f"hidden={self.hidden_size} intermediate={self.intermediate_size}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.input_name: input_binding,
                    self.norm_weight_name: norm_weight_binding,
                    self.gate_up_weight_name: gate_up_weight_binding,
                    self.down_weight_name: down_weight_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return self.output[:actual_batch]


class _MirageMoeW2ReduceExecutor:
    def __init__(
        self,
        *,
        capacity: int,
        topk: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        self.capacity = capacity
        self.topk = topk
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self._last_actual_batch: Optional[int] = None
        self.act_name = "moe_w2_act"
        self.weight_name = "moe_w2_weight"
        self.routing_indices_name = "moe_w2_indices"
        self.routing_mask_name = "moe_w2_mask"
        self.topk_weight_name = "moe_w2_topk_weight"
        self.residual_name = "moe_w2_residual"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.act = torch.empty(
            (capacity, topk, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.weight = torch.empty(
            (num_experts, hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.routing_indices = torch.empty(
            (num_experts, capacity), device="cuda", dtype=torch.int32
        )
        self.routing_mask = torch.empty((num_experts + 1,), device="cuda", dtype=torch.int32)
        self.topk_weight = torch.empty((capacity, topk), device="cuda", dtype=torch.float32)
        self.residual = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.w2_out = torch.zeros(
            (capacity, topk, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.output = torch.zeros((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        act_dt = self.pk.attach_input(self.act, name=self.act_name)
        weight_dt = self.pk.attach_input(self.weight, name=self.weight_name)
        routing_indices_dt = self.pk.attach_input(
            self.routing_indices, name=self.routing_indices_name
        )
        routing_mask_dt = self.pk.attach_input(self.routing_mask, name=self.routing_mask_name)
        topk_weight_dt = self.pk.attach_input(self.topk_weight, name=self.topk_weight_name)
        residual_dt = self.pk.attach_input(self.residual, name=self.residual_name)
        w2_out_dt = self.pk.attach_input(self.w2_out, name="moe_w2_out")
        output_dt = self.pk.attach_input(self.output, name="moe_w2_output")
        self.pk.moe_w2_linear_layer(
            input=act_dt,
            weight=weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=w2_out_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=False),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_mul_sum_add_layer(
            input=w2_out_dt,
            weight=topk_weight_dt,
            residual=residual_dt,
            output=output_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        act_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        topk_weight: torch.Tensor,
        residual: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> torch.Tensor:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            act_binding = torch.zeros(
                (cap, self.topk, self.intermediate_size),
                device=act_tensor.device,
                dtype=act_tensor.dtype,
            )
            act_binding[:actual_batch] = act_tensor[:actual_batch]
            ri_padded = torch.zeros(
                (self.num_experts, cap), device=routing_indices.device, dtype=routing_indices.dtype
            )
            ri_padded[:, :actual_batch] = routing_indices[:, :actual_batch]
            routing_indices_binding = ri_padded
            tw_padded = torch.zeros(
                (cap, self.topk), device=topk_weight.device, dtype=topk_weight.dtype
            )
            tw_padded[:actual_batch] = topk_weight[:actual_batch]
            topk_weight_binding = tw_padded
            residual_binding = torch.zeros(
                (cap, self.hidden_size), device=residual.device, dtype=residual.dtype
            )
            residual_binding[:actual_batch] = residual.contiguous().view(
                actual_batch, self.hidden_size
            )
        else:
            act_binding = act_tensor.contiguous()
            routing_indices_binding = routing_indices.contiguous()
            topk_weight_binding = topk_weight.contiguous()
            residual_binding = residual.contiguous().view(actual_batch, self.hidden_size)
        weight_binding = weight_tensor.contiguous()
        routing_mask_binding = routing_mask.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        self.w2_out.zero_()
        self.output.zero_()
        _mpk_profile_executor(
            "moe_w2.launch "
            f"capacity={self.capacity} actual={actual_batch} topk={self.topk} hidden={self.hidden_size} "
            f"intermediate={self.intermediate_size} experts={self.num_experts}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.act_name: act_binding,
                    self.weight_name: weight_binding,
                    self.routing_indices_name: routing_indices_binding,
                    self.routing_mask_name: routing_mask_binding,
                    self.topk_weight_name: topk_weight_binding,
                    self.residual_name: residual_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return self.output[:actual_batch]


class _MirageSplitMoeExecutor:
    def __init__(
        self,
        *,
        capacity: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
    ):
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk
        self._last_actual_batch: Optional[int] = None
        self.hidden_name = "split_moe_hidden"
        self.w13_weight_name = "split_moe_w13_weight"
        self.routing_indices_name = "split_moe_indices"
        self.routing_mask_name = "split_moe_mask"
        self.w2_weight_name = "split_moe_w2_weight"
        self.topk_weight_name = "split_moe_topk_weight"
        self.residual_name = "split_moe_residual"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.hidden = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.w13_weight = torch.empty(
            (num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.routing_indices = torch.empty(
            (num_experts, capacity), device="cuda", dtype=torch.int32
        )
        self.routing_mask = torch.empty((num_experts + 1,), device="cuda", dtype=torch.int32)
        self.w2_weight = torch.empty(
            (num_experts, hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.topk_weight = torch.empty((capacity, topk), device="cuda", dtype=torch.float32)
        self.zero_residual = torch.zeros(
            (capacity, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.moe_w13_out = torch.empty(
            (capacity, topk, 2 * intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.moe_act = torch.empty(
            (capacity, topk, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.moe_w2_out = torch.zeros(
            (capacity, topk, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.output = torch.zeros((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        hidden_dt = self.pk.attach_input(self.hidden, name=self.hidden_name)
        w13_weight_dt = self.pk.attach_input(self.w13_weight, name=self.w13_weight_name)
        routing_indices_dt = self.pk.attach_input(
            self.routing_indices, name=self.routing_indices_name
        )
        routing_mask_dt = self.pk.attach_input(self.routing_mask, name=self.routing_mask_name)
        w2_weight_dt = self.pk.attach_input(self.w2_weight, name=self.w2_weight_name)
        topk_weight_dt = self.pk.attach_input(self.topk_weight, name=self.topk_weight_name)
        residual_dt = self.pk.attach_input(self.zero_residual, name=self.residual_name)
        moe_w13_out_dt = self.pk.attach_input(self.moe_w13_out, name="split_moe_w13_out")
        moe_act_dt = self.pk.attach_input(self.moe_act, name="split_moe_act")
        moe_w2_out_dt = self.pk.attach_input(self.moe_w2_out, name="split_moe_w2_out")
        output_dt = self.pk.attach_input(self.output, name="split_moe_output")
        self.pk.moe_w13_linear_layer(
            input=hidden_dt,
            weight=w13_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=moe_w13_out_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=True),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_gelu_mul_swapped_layer(
            input=moe_w13_out_dt,
            output=moe_act_dt,
            grid_dim=(capacity, topk, 1),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_w2_linear_layer(
            input=moe_act_dt,
            weight=w2_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=moe_w2_out_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=False),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_mul_sum_add_layer(
            input=moe_w2_out_dt,
            weight=topk_weight_dt,
            residual=residual_dt,
            output=output_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        hidden_tensor: torch.Tensor,
        w13_weight_tensor: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        w2_weight_tensor: torch.Tensor,
        topk_weight_tensor: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> torch.Tensor:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            hidden_binding = torch.zeros(
                (cap, self.hidden_size), device=hidden_tensor.device, dtype=hidden_tensor.dtype
            )
            hidden_binding[:actual_batch] = hidden_tensor.contiguous().view(
                actual_batch, self.hidden_size
            )
            ri_padded = torch.zeros(
                (self.num_experts, cap), device=routing_indices.device, dtype=routing_indices.dtype
            )
            ri_padded[:, :actual_batch] = routing_indices[:, :actual_batch]
            routing_indices_binding = ri_padded
            tw_padded = torch.zeros(
                (cap, self.topk), device=topk_weight_tensor.device, dtype=topk_weight_tensor.dtype
            )
            tw_padded[:actual_batch] = topk_weight_tensor[:actual_batch]
            topk_weight_binding = tw_padded
        else:
            hidden_binding = hidden_tensor.contiguous().view(actual_batch, self.hidden_size)
            routing_indices_binding = routing_indices.contiguous()
            topk_weight_binding = topk_weight_tensor.contiguous()
        w13_weight_binding = w13_weight_tensor.contiguous()
        routing_mask_binding = routing_mask.contiguous()
        w2_weight_binding = w2_weight_tensor.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        self.moe_w13_out.zero_()
        self.moe_w2_out.zero_()
        self.output.zero_()
        _mpk_profile_executor(
            "split_moe.launch "
            f"capacity={self.capacity} actual={actual_batch} hidden={self.hidden_size} "
            f"intermediate={self.intermediate_size} experts={self.num_experts} topk={self.topk}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.hidden_name: hidden_binding,
                    self.w13_weight_name: w13_weight_binding,
                    self.routing_indices_name: routing_indices_binding,
                    self.routing_mask_name: routing_mask_binding,
                    self.w2_weight_name: w2_weight_binding,
                    self.topk_weight_name: topk_weight_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return self.output[:actual_batch]


class _MirageFusedMoeExecutor:
    def __init__(
        self,
        *,
        capacity: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
    ):
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk
        self._last_actual_batch: Optional[int] = None
        self.hidden_name = "fused_moe_hidden"
        self.w13_weight_name = "fused_moe_w13_weight"
        self.routing_indices_name = "fused_moe_indices"
        self.routing_mask_name = "fused_moe_mask"
        self.w2_weight_name = "fused_moe_w2_weight"
        self.topk_weight_name = "fused_moe_topk_weight"
        self.residual_name = "fused_moe_residual"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, capacity),
            max_num_batched_requests=1,
            max_num_batched_tokens=capacity,
            max_num_pages=8,
            page_size=64,
            use_cutlass_kernel=False,
        )
        self.hidden = torch.empty((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        self.w13_weight = torch.empty(
            (num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.routing_indices = torch.empty(
            (num_experts, capacity), device="cuda", dtype=torch.int32
        )
        self.routing_mask = torch.empty((num_experts + 1,), device="cuda", dtype=torch.int32)
        self.w2_weight = torch.empty(
            (num_experts, hidden_size, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.topk_weight = torch.empty((capacity, topk), device="cuda", dtype=torch.float32)
        self.zero_residual = torch.zeros(
            (capacity, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.moe_act = torch.empty(
            (capacity, topk, intermediate_size), device="cuda", dtype=torch.bfloat16
        )
        self.moe_w2_out = torch.zeros(
            (capacity, topk, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        self.output = torch.zeros((capacity, hidden_size), device="cuda", dtype=torch.bfloat16)
        hidden_dt = self.pk.attach_input(self.hidden, name=self.hidden_name)
        w13_weight_dt = self.pk.attach_input(self.w13_weight, name=self.w13_weight_name)
        routing_indices_dt = self.pk.attach_input(
            self.routing_indices, name=self.routing_indices_name
        )
        routing_mask_dt = self.pk.attach_input(self.routing_mask, name=self.routing_mask_name)
        w2_weight_dt = self.pk.attach_input(self.w2_weight, name=self.w2_weight_name)
        topk_weight_dt = self.pk.attach_input(self.topk_weight, name=self.topk_weight_name)
        residual_dt = self.pk.attach_input(self.zero_residual, name=self.residual_name)
        moe_act_dt = self.pk.attach_input(self.moe_act, name="fused_moe_act")
        moe_w2_out_dt = self.pk.attach_input(self.moe_w2_out, name="fused_moe_w2_out")
        output_dt = self.pk.attach_input(self.output, name="fused_moe_output")
        self.pk.moe_w13_gelu_mul_layer(
            input=hidden_dt,
            weight=w13_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=moe_act_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=True),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_w2_linear_layer(
            input=moe_act_dt,
            weight=w2_weight_dt,
            moe_routing_indices=routing_indices_dt,
            moe_mask=routing_mask_dt,
            output=moe_w2_out_dt,
            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=False),
            block_dim=(128, 1, 1),
        )
        self.pk.moe_mul_sum_add_layer(
            input=moe_w2_out_dt,
            weight=topk_weight_dt,
            residual=residual_dt,
            output=output_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        compile_persistent_kernel_with_patches(self.pk)

    def _update_dynamic_batch(self, actual_batch: int) -> None:
        if actual_batch == self._last_actual_batch:
            return
        if hasattr(self.pk, "update_dynamic_dims"):
            self.pk.update_dynamic_dims(actual_batch=actual_batch)
        self._last_actual_batch = actual_batch

    def __call__(
        self,
        hidden_tensor: torch.Tensor,
        w13_weight_tensor: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        w2_weight_tensor: torch.Tensor,
        topk_weight_tensor: torch.Tensor,
        actual_batch: Optional[int] = None,
    ) -> torch.Tensor:
        if actual_batch is None:
            actual_batch = self.capacity
        # Pad batch-dependent inputs to compiled capacity for dynamic_inputs binding.
        # The PK's update_dynamic_dims handles kernel-level early-exit, but
        # dynamic_inputs must match the compiled tensor shapes.
        cap = self.capacity
        if actual_batch < cap:
            hidden_binding = torch.zeros(
                (cap, self.hidden_size), device=hidden_tensor.device, dtype=hidden_tensor.dtype
            )
            hidden_binding[:actual_batch] = hidden_tensor.contiguous().view(
                actual_batch, self.hidden_size
            )
            ri_padded = torch.zeros(
                (self.num_experts, cap), device=routing_indices.device, dtype=routing_indices.dtype
            )
            ri_padded[:, :actual_batch] = routing_indices[:, :actual_batch]
            routing_indices_binding = ri_padded
            tw_padded = torch.zeros(
                (cap, self.topk), device=topk_weight_tensor.device, dtype=topk_weight_tensor.dtype
            )
            tw_padded[:actual_batch] = topk_weight_tensor[:actual_batch]
            topk_weight_binding = tw_padded
        else:
            hidden_binding = hidden_tensor.contiguous().view(actual_batch, self.hidden_size)
            routing_indices_binding = routing_indices.contiguous()
            topk_weight_binding = topk_weight_tensor.contiguous()
        w13_weight_binding = w13_weight_tensor.contiguous()
        routing_mask_binding = routing_mask.contiguous()
        w2_weight_binding = w2_weight_tensor.contiguous()
        self._update_dynamic_batch(actual_batch)
        _reset_pk_runtime_state(self.pk, active_tokens=actual_batch)
        self.moe_w2_out.zero_()
        self.output.zero_()
        _mpk_profile_executor(
            "fused_moe.launch "
            f"capacity={self.capacity} actual={actual_batch} hidden={self.hidden_size} "
            f"intermediate={self.intermediate_size} experts={self.num_experts} topk={self.topk}",
            lambda: self.pk.run(
                dynamic_inputs={
                    self.hidden_name: hidden_binding,
                    self.w13_weight_name: w13_weight_binding,
                    self.routing_indices_name: routing_indices_binding,
                    self.routing_mask_name: routing_mask_binding,
                    self.w2_weight_name: w2_weight_binding,
                    self.topk_weight_name: topk_weight_binding,
                },
                reset_runtime_state=False,
            ),
        )
        if _mpk_sync_each_block_enabled():
            torch.cuda.synchronize()
        return self.output[:actual_batch]


class _MirageGeluMulExecutor:
    def __init__(self, *, shape: tuple[int, ...]):
        _require_mirage()
        import mirage

        self.shape = shape
        self.graph = mirage.new_kernel_graph()
        gate_in = self.graph.new_input(shape, dtype=mirage.bfloat16)
        up_in = self.graph.new_input(shape, dtype=mirage.bfloat16)
        out = self.graph.mul(self.graph.gelu(gate_in), up_in)
        self.graph.mark_output(out)
        self._compiled = False

    def __call__(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if not self._compiled:
            self.graph.compile(inputs=[gate, up], target_cc=90)
            self._compiled = True
        return self.graph(inputs=[gate, up], target_cc=90)[0]


class _MirageMatmulExecutor:
    def __init__(self, *, m: int, k: int, n: int):
        _require_mirage()
        import mirage

        self.shape_a = (m, k)
        self.shape_b = (k, n)
        self.graph = mirage.new_kernel_graph()
        a_in = self.graph.new_input(self.shape_a, dtype=mirage.bfloat16)
        b_in = self.graph.new_input(self.shape_b, dtype=mirage.bfloat16)
        out = self.graph.matmul(a_in, b_in)
        self.graph.mark_output(out)
        self._compiled = False

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._compiled:
            self.graph.compile(inputs=[a, b], target_cc=90)
            self._compiled = True
        return self.graph(inputs=[a, b], target_cc=90)[0]


def _expand_hidden_param(param: torch.Tensor, hidden_size: int) -> torch.Tensor:
    flat = param.detach().reshape(-1).to(torch.bfloat16)
    if flat.numel() == 1:
        return flat.expand(hidden_size).clone()
    if flat.numel() != hidden_size:
        raise ValueError(
            f"Expected hidden-sized parameter with {hidden_size} elements, got {tuple(param.shape)}."
        )
    return flat.contiguous()


class _MirageGemmaLayerSinglePkExecutor:
    def __init__(
        self,
        *,
        layer_spec: _GemmaRuntimeLayerSpec,
        kv_cache: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        identity_weight: torch.Tensor,
        max_seq_length: Optional[int] = None,
    ) -> None:
        disable_moe_tail = os.getenv("AD_MPK_SINGLE_PK_DISABLE_MOE_TAIL", "0") == "1"
        disable_ffn_tail = os.getenv("AD_MPK_SINGLE_PK_DISABLE_FFN_TAIL", "0") == "1"
        disable_post_attn_tail = os.getenv("AD_MPK_SINGLE_PK_DISABLE_POST_ATTN_TAIL", "0") == "1"
        disable_moe_exec = os.getenv("AD_MPK_SINGLE_PK_DISABLE_MOE_EXEC", "0") == "1"
        disable_moe_merge = os.getenv("AD_MPK_SINGLE_PK_DISABLE_MOE_MERGE", "0") == "1"
        disable_post_merge_norm = os.getenv("AD_MPK_SINGLE_PK_DISABLE_POST_MERGE_NORM", "0") == "1"
        disable_final_residual = os.getenv("AD_MPK_SINGLE_PK_DISABLE_FINAL_RESIDUAL", "0") == "1"
        disable_final_scale = os.getenv("AD_MPK_SINGLE_PK_DISABLE_FINAL_SCALE", "0") == "1"

        def _constructor_probe(stage: str) -> None:
            if os.getenv("AD_MPK_SINGLE_PK_CONSTRUCTOR_DEBUG", "0") != "1":
                return
            free_mem, total_mem = torch.cuda.mem_get_info(device=layer_spec.o_proj_weight.device)
            _mpk_debug(
                "single-pk ctor "
                f"stage={stage} free_gb={free_mem / (1024**3):.2f} total_gb={total_mem / (1024**3):.2f}"
            )
            torch.cuda.synchronize()
            torch.zeros((1,), device=layer_spec.o_proj_weight.device, dtype=torch.float32)
            torch.cuda.synchronize()

        if kv_cache.dim() != 5 or int(kv_cache.shape[1]) != 2:
            raise ValueError(
                "Expected combined HND KV cache with shape [pages, 2, kv_heads, page_size, head_dim], "
                f"got {tuple(kv_cache.shape)}."
            )

        self.layer_spec = layer_spec
        self.hidden_size = int(layer_spec.o_proj_weight.shape[0])
        self.attn_out_size = int(layer_spec.q_heads) * int(layer_spec.head_dim)
        self.ffn_intermediate = int(layer_spec.ffn_down_weight.shape[1])
        self.intermediate = int(layer_spec.moe_w2_weight.shape[2])
        self.num_experts = int(layer_spec.moe_w2_weight.shape[0])
        self.topk = int(layer_spec.topk)
        self.external_max_num_pages = int(kv_cache.shape[0])
        self.external_page_size = int(kv_cache.shape[3])
        page_size_override = os.getenv("AD_MPK_SINGLE_PK_PAGE_SIZE", "")
        if page_size_override:
            self.page_size = int(page_size_override)
        else:
            self.page_size = 64
        rope_max_seq_length = max(1, int(cos.shape[0]) - 1)
        if max_seq_length is None:
            env_limit = os.getenv("AD_MPK_SINGLE_PK_MAX_SEQ_LENGTH", "")
            if env_limit:
                requested_max_seq_length = int(env_limit)
            else:
                requested_max_seq_length = rope_max_seq_length
        else:
            requested_max_seq_length = int(max_seq_length)
        self.max_seq_length = max(1, min(rope_max_seq_length, requested_max_seq_length))
        self.max_num_pages = max(
            1,
            (self.max_seq_length + self.page_size - 1) // self.page_size,
        )
        self.kv_heads = int(kv_cache.shape[2])
        self.head_dim = int(kv_cache.shape[4])
        self.kv_cache_shape = tuple(int(dim) for dim in kv_cache.shape)
        self.kv_cache_dtype = kv_cache.dtype
        self.use_single_batch_extend_attention = bool(
            layer_spec.qkv_shared_kv and int(layer_spec.head_dim) == 512
        )
        self.attn_kv_expand_factor = 1
        self.attn_kv_heads = self.kv_heads
        self.attn_qkv_weight = layer_spec.qkv_weight.contiguous()
        if layer_spec.qkv_shared_kv and self.use_single_batch_extend_attention:
            q_size = int(layer_spec.q_heads) * int(layer_spec.head_dim)
            kv_size = int(layer_spec.kv_heads) * int(layer_spec.head_dim)
            q_per_kv = int(layer_spec.q_heads) // int(layer_spec.kv_heads)
            q_weight = layer_spec.qkv_weight[:q_size].view(
                int(layer_spec.kv_heads),
                q_per_kv,
                int(layer_spec.head_dim),
                self.hidden_size,
            )
            shared_kv_weight = layer_spec.qkv_weight[q_size : q_size + kv_size].view(
                int(layer_spec.kv_heads),
                int(layer_spec.head_dim),
                self.hidden_size,
            )
            grouped_rows = []
            for kv_idx in range(int(layer_spec.kv_heads)):
                grouped_rows.append(
                    q_weight[kv_idx].reshape(q_per_kv * int(layer_spec.head_dim), self.hidden_size)
                )
                grouped_rows.append(shared_kv_weight[kv_idx])
                grouped_rows.append(shared_kv_weight[kv_idx])
            self.attn_qkv_weight = torch.cat(grouped_rows, dim=0).contiguous()
        elif layer_spec.qkv_shared_kv and not self.use_single_batch_extend_attention:
            q_per_kv = int(layer_spec.q_heads) // int(layer_spec.kv_heads)
            max_q_per_kv_override = os.getenv("AD_MPK_SINGLE_PK_MAX_Q_PER_KV", "")
            if max_q_per_kv_override:
                target_q_per_kv = min(q_per_kv, int(max_q_per_kv_override))
            else:
                # Hopper paged attention overflows shared memory for Gemma's
                # global layers when q-heads-per-kv stays too large at head_dim=512.
                # Keep the per-KV query grouping roughly bounded by 1024 features.
                target_q_per_kv = max(1, min(q_per_kv, 1024 // int(layer_spec.head_dim)))
            while q_per_kv % target_q_per_kv != 0 and target_q_per_kv > 1:
                target_q_per_kv -= 1
            if q_per_kv % target_q_per_kv != 0:
                raise ValueError(
                    f"Unsupported shared-KV q-head grouping: q_heads={layer_spec.q_heads}, "
                    f"kv_heads={layer_spec.kv_heads}, target_q_per_kv={target_q_per_kv}."
                )
            self.attn_kv_expand_factor = q_per_kv // target_q_per_kv
            self.attn_kv_heads = self.kv_heads * self.attn_kv_expand_factor
            q_size = int(layer_spec.q_heads) * int(layer_spec.head_dim)
            kv_size = int(layer_spec.kv_heads) * int(layer_spec.head_dim)
            q_weight = layer_spec.qkv_weight[:q_size].view(
                int(layer_spec.kv_heads),
                q_per_kv,
                int(layer_spec.head_dim),
                self.hidden_size,
            )
            shared_kv_weight = layer_spec.qkv_weight[q_size : q_size + kv_size].view(
                int(layer_spec.kv_heads),
                int(layer_spec.head_dim),
                self.hidden_size,
            )
            grouped_rows = []
            for kv_idx in range(int(layer_spec.kv_heads)):
                for split_idx in range(self.attn_kv_expand_factor):
                    start = split_idx * target_q_per_kv
                    end = start + target_q_per_kv
                    grouped_rows.append(
                        q_weight[kv_idx, start:end].reshape(
                            target_q_per_kv * int(layer_spec.head_dim), self.hidden_size
                        )
                    )
                    grouped_rows.append(shared_kv_weight[kv_idx])
                    grouped_rows.append(shared_kv_weight[kv_idx])
            self.attn_qkv_weight = torch.cat(grouped_rows, dim=0).contiguous()
        disable_split_kv = os.getenv("AD_MPK_SINGLE_PK_DISABLE_SPLIT_KV", "0") == "1"
        force_split_kv = os.getenv("AD_MPK_SINGLE_PK_FORCE_SPLIT_KV", "0") == "1"
        self.use_split_kv_attention = force_split_kv and not disable_split_kv
        split_kv_chunks_override = os.getenv("AD_MPK_SINGLE_PK_NUM_KV_CHUNKS", "")
        if split_kv_chunks_override:
            self.num_kv_cache_chunks = max(1, int(split_kv_chunks_override))
        else:
            self.num_kv_cache_chunks = max(1, self.max_seq_length // 256)
        self.num_kv_cache_chunks = min(self.num_kv_cache_chunks, self.max_num_pages)

        router_root_size = _expand_hidden_param(layer_spec.router_root_size, self.hidden_size)
        router_scale = _expand_hidden_param(layer_spec.router_scale, self.hidden_size)
        router_weight_scaled = (
            layer_spec.router_proj_weight.float()
            * router_root_size.float().view(1, -1)
            * router_scale.float().view(1, -1)
        ).to(torch.bfloat16)
        layer_scalar = _expand_hidden_param(layer_spec.layer_scalar, self.hidden_size)
        layer_scale_weight = torch.diag(layer_scalar.float()).to(torch.bfloat16)
        router_norm_weight = torch.ones(
            (self.hidden_size,), device=layer_spec.o_proj_weight.device, dtype=torch.bfloat16
        )

        use_static = os.getenv("AD_MPK_SINGLE_PK_STATIC_SCHEDULE", "1") == "1"
        self.pk = create_test_persistent_kernel(
            max_seq_length=max(128, self.max_num_pages * self.page_size),
            max_num_batched_requests=1,
            max_num_batched_tokens=1,
            max_num_pages=self.max_num_pages,
            page_size=self.page_size,
            use_cutlass_kernel=False,
            schedule_mode="static" if use_static else "dynamic",
        )

        device = layer_spec.o_proj_weight.device
        self.hidden_in = torch.empty((1, self.hidden_size), device=device, dtype=torch.bfloat16)
        self.rmsnorm_out = torch.empty_like(self.hidden_in)
        self.qkv_out = torch.empty(
            (1, int(self.attn_qkv_weight.shape[0])), device=device, dtype=torch.bfloat16
        )
        self.k_cache = torch.empty(
            (self.max_num_pages, self.page_size, self.attn_kv_heads, self.head_dim),
            device=device,
            dtype=kv_cache.dtype,
        )
        self.v_cache = torch.empty_like(self.k_cache)
        self.attn_out = torch.empty((1, self.attn_out_size), device=device, dtype=torch.bfloat16)
        self.o_proj = torch.empty_like(self.hidden_in)
        self.post_attn_norm = torch.empty_like(self.hidden_in)
        self.post_attn = torch.empty_like(self.hidden_in)
        self.ffn_normed = torch.empty_like(self.hidden_in)
        self.ffn_gate_up_storage = torch.empty(
            (1, 1, 2 * self.ffn_intermediate), device=device, dtype=torch.bfloat16
        )
        self.ffn_gate_up = self.ffn_gate_up_storage.view(1, 2 * self.ffn_intermediate)
        self.ffn_act = torch.empty(
            (1, 1, self.ffn_intermediate), device=device, dtype=torch.bfloat16
        )
        self.ffn_routing_indices = torch.ones((1, 1), device=device, dtype=torch.int32)
        self.ffn_routing_mask = torch.tensor([0, 1], device=device, dtype=torch.int32)
        self.ffn_topk_weight = torch.ones((1, 1), device=device, dtype=torch.float32)
        self.zero_residual = torch.zeros((1, self.hidden_size), device=device, dtype=torch.bfloat16)
        self.ffn_w2_weight = torch.empty(
            (1, self.hidden_size, self.ffn_intermediate),
            device=device,
            dtype=torch.bfloat16,
        )
        self.ffn_w2_weight[0].copy_(layer_spec.ffn_down_weight)
        self.ffn_w2_out = torch.empty((1, 1, self.hidden_size), device=device, dtype=torch.bfloat16)
        self.ffn_down = torch.empty_like(self.hidden_in)
        self.ffn_norm = torch.empty_like(self.hidden_in)
        self.router_in = torch.empty_like(self.hidden_in)
        self.router_logits = torch.empty((1, self.num_experts), device=device, dtype=torch.bfloat16)
        self.topk_weight = torch.empty((1, self.topk), device=device, dtype=torch.float32)
        self.routing_indices = torch.empty((self.num_experts, 1), device=device, dtype=torch.int32)
        self.routing_mask = torch.empty((self.num_experts + 1,), device=device, dtype=torch.int32)
        self.moe_in = torch.empty_like(self.hidden_in)
        self.moe_act = torch.empty(
            (1, self.topk, self.intermediate), device=device, dtype=torch.bfloat16
        )
        self.moe_w2_out = torch.empty(
            (1, self.topk, self.hidden_size), device=device, dtype=torch.bfloat16
        )
        self.moe_out = torch.empty_like(self.hidden_in)
        self.moe_norm = torch.empty_like(self.hidden_in)
        self.ffn_moe_add = torch.empty_like(self.hidden_in)
        self.ffn_moe_norm = torch.empty_like(self.hidden_in)
        self.hidden_pre_scale = torch.empty_like(self.hidden_in)
        self.scaled_post_attn = torch.empty_like(self.hidden_in)
        self.hidden_out = torch.empty_like(self.hidden_in)
        _constructor_probe("post_alloc")

        hidden_dt = self.pk.attach_input(
            self.hidden_in, name=f"layer_{layer_spec.layer_index}_hidden"
        )
        attn_norm_dt = self.pk.attach_input(
            layer_spec.input_layernorm_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_attn_norm_weight",
        )
        qkv_weight_dt = self.pk.attach_input(
            self.attn_qkv_weight, name=f"layer_{layer_spec.layer_index}_qkv_weight"
        )
        rmsnorm_out_dt = self.pk.attach_input(
            self.rmsnorm_out, name=f"layer_{layer_spec.layer_index}_rmsnorm_out"
        )
        qkv_out_dt = self.pk.attach_input(
            self.qkv_out, name=f"layer_{layer_spec.layer_index}_qkv_out"
        )
        q_norm_dt = self.pk.attach_input(
            layer_spec.q_norm_weight.contiguous(), name=f"layer_{layer_spec.layer_index}_q_norm"
        )
        k_norm_dt = self.pk.attach_input(
            layer_spec.k_norm_weight.contiguous(), name=f"layer_{layer_spec.layer_index}_k_norm"
        )
        cos_dt = self.pk.attach_input(cos.contiguous(), name=f"layer_{layer_spec.layer_index}_cos")
        sin_dt = self.pk.attach_input(sin.contiguous(), name=f"layer_{layer_spec.layer_index}_sin")
        k_cache_dt = self.pk.attach_input(
            self.k_cache, name=f"layer_{layer_spec.layer_index}_k_cache"
        )
        v_cache_dt = self.pk.attach_input(
            self.v_cache, name=f"layer_{layer_spec.layer_index}_v_cache"
        )
        attn_out_dt = self.pk.attach_input(
            self.attn_out, name=f"layer_{layer_spec.layer_index}_attn_out"
        )
        lse_dt = None
        attn_out_tmp_dt = None
        if self.use_split_kv_attention:
            import mirage

            lse_dt = self.pk.new_tensor(
                dims=(
                    1,
                    self.num_kv_cache_chunks * int(layer_spec.q_heads) // self.attn_kv_heads,
                    self.attn_kv_heads,
                ),
                strides=(
                    self.num_kv_cache_chunks * int(layer_spec.q_heads),
                    1,
                    self.num_kv_cache_chunks * int(layer_spec.q_heads) // self.attn_kv_heads,
                ),
                dtype=mirage.float32,
                name=f"layer_{layer_spec.layer_index}_attn_lse",
                io_category="cuda_tensor",
            )
            attn_out_tmp_dt = self.pk.new_tensor(
                dims=(
                    1,
                    self.num_kv_cache_chunks
                    * int(layer_spec.q_heads)
                    // self.attn_kv_heads
                    * int(layer_spec.head_dim),
                    self.attn_kv_heads,
                ),
                strides=(
                    self.num_kv_cache_chunks * int(layer_spec.q_heads),
                    1,
                    self.num_kv_cache_chunks
                    * int(layer_spec.q_heads)
                    // self.attn_kv_heads
                    * int(layer_spec.head_dim),
                ),
                name=f"layer_{layer_spec.layer_index}_attn_out_tmp",
                io_category="cuda_tensor",
            )
        o_proj_weight_dt = self.pk.attach_input(
            layer_spec.o_proj_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_o_proj_weight",
        )
        o_proj_dt = self.pk.attach_input(self.o_proj, name=f"layer_{layer_spec.layer_index}_o_proj")
        post_attn_norm_weight_dt = self.pk.attach_input(
            layer_spec.post_attention_layernorm_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_post_attn_norm_weight",
        )
        post_attn_norm_dt = self.pk.attach_input(
            self.post_attn_norm, name=f"layer_{layer_spec.layer_index}_post_attn_norm"
        )
        identity_weight_dt = self.pk.attach_input(
            identity_weight, name=f"layer_{layer_spec.layer_index}_identity"
        )
        post_attn_dt = self.pk.attach_input(
            self.post_attn, name=f"layer_{layer_spec.layer_index}_post_attn"
        )
        pre_ffn_norm_weight_dt = self.pk.attach_input(
            layer_spec.pre_feedforward_layernorm_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_pre_ffn_norm_weight",
        )
        ffn_normed_dt = self.pk.attach_input(
            self.ffn_normed, name=f"layer_{layer_spec.layer_index}_ffn_normed"
        )
        ffn_gate_up_weight_dt = self.pk.attach_input(
            layer_spec.ffn_gate_up_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_ffn_gate_up_weight",
        )
        ffn_gate_up_dt = self.pk.attach_input(
            self.ffn_gate_up, name=f"layer_{layer_spec.layer_index}_ffn_gate_up"
        )
        ffn_gate_up_moe_dt = self.pk.attach_input(
            self.ffn_gate_up_storage, name=f"layer_{layer_spec.layer_index}_ffn_gate_up_moe"
        )
        ffn_act_dt = self.pk.attach_input(
            self.ffn_act, name=f"layer_{layer_spec.layer_index}_ffn_act"
        )
        ffn_routing_indices_dt = self.pk.attach_input(
            self.ffn_routing_indices, name=f"layer_{layer_spec.layer_index}_ffn_routing_indices"
        )
        ffn_routing_mask_dt = self.pk.attach_input(
            self.ffn_routing_mask, name=f"layer_{layer_spec.layer_index}_ffn_routing_mask"
        )
        ffn_topk_weight_dt = self.pk.attach_input(
            self.ffn_topk_weight, name=f"layer_{layer_spec.layer_index}_ffn_topk_weight"
        )
        zero_residual_dt = self.pk.attach_input(
            self.zero_residual, name=f"layer_{layer_spec.layer_index}_zero_residual"
        )
        ffn_w2_weight_dt = self.pk.attach_input(
            self.ffn_w2_weight, name=f"layer_{layer_spec.layer_index}_ffn_w2_weight"
        )
        ffn_w2_out_dt = self.pk.attach_input(
            self.ffn_w2_out, name=f"layer_{layer_spec.layer_index}_ffn_w2_out"
        )
        ffn_down_dt = self.pk.attach_input(
            self.ffn_down, name=f"layer_{layer_spec.layer_index}_ffn_down"
        )
        post_ffn_norm_weight_dt = self.pk.attach_input(
            layer_spec.post_feedforward_layernorm_1_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_post_ffn_norm_weight",
        )
        ffn_norm_dt = self.pk.attach_input(
            self.ffn_norm, name=f"layer_{layer_spec.layer_index}_ffn_norm"
        )
        router_norm_weight_dt = self.pk.attach_input(
            router_norm_weight, name=f"layer_{layer_spec.layer_index}_router_norm_weight"
        )
        router_in_dt = self.pk.attach_input(
            self.router_in, name=f"layer_{layer_spec.layer_index}_router_in"
        )
        router_weight_dt = self.pk.attach_input(
            router_weight_scaled.contiguous(), name=f"layer_{layer_spec.layer_index}_router_weight"
        )
        router_logits_dt = self.pk.attach_input(
            self.router_logits, name=f"layer_{layer_spec.layer_index}_router_logits"
        )
        moe_in_norm_weight_dt = self.pk.attach_input(
            layer_spec.pre_feedforward_layernorm_2_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_moe_in_norm_weight",
        )
        moe_in_dt = self.pk.attach_input(self.moe_in, name=f"layer_{layer_spec.layer_index}_moe_in")
        topk_weight_dt = self.pk.attach_input(
            self.topk_weight, name=f"layer_{layer_spec.layer_index}_topk_weight"
        )
        routing_indices_dt = self.pk.attach_input(
            self.routing_indices, name=f"layer_{layer_spec.layer_index}_routing_indices"
        )
        routing_mask_dt = self.pk.attach_input(
            self.routing_mask, name=f"layer_{layer_spec.layer_index}_routing_mask"
        )
        moe_w13_weight_dt = self.pk.attach_input(
            layer_spec.moe_w13_stacked_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_moe_w13_weight",
        )
        moe_act_dt = self.pk.attach_input(
            self.moe_act, name=f"layer_{layer_spec.layer_index}_moe_act"
        )
        moe_w2_weight_dt = self.pk.attach_input(
            layer_spec.moe_w2_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_moe_w2_weight",
        )
        moe_w2_out_dt = self.pk.attach_input(
            self.moe_w2_out, name=f"layer_{layer_spec.layer_index}_moe_w2_out"
        )
        moe_out_dt = self.pk.attach_input(
            self.moe_out, name=f"layer_{layer_spec.layer_index}_moe_out"
        )
        post_moe_norm_weight_dt = self.pk.attach_input(
            layer_spec.post_feedforward_layernorm_2_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_post_moe_norm_weight",
        )
        moe_norm_dt = self.pk.attach_input(
            self.moe_norm, name=f"layer_{layer_spec.layer_index}_moe_norm"
        )
        ffn_moe_add_dt = self.pk.attach_input(
            self.ffn_moe_add, name=f"layer_{layer_spec.layer_index}_ffn_moe_add"
        )
        post_merge_norm_weight_dt = self.pk.attach_input(
            layer_spec.post_feedforward_layernorm_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_post_merge_norm_weight",
        )
        ffn_moe_norm_dt = self.pk.attach_input(
            self.ffn_moe_norm, name=f"layer_{layer_spec.layer_index}_ffn_moe_norm"
        )
        hidden_pre_scale_dt = self.pk.attach_input(
            self.hidden_pre_scale, name=f"layer_{layer_spec.layer_index}_hidden_pre_scale"
        )
        scaled_post_attn_dt = self.pk.attach_input(
            self.scaled_post_attn, name=f"layer_{layer_spec.layer_index}_scaled_post_attn"
        )
        layer_scale_weight_dt = self.pk.attach_input(
            layer_scale_weight.contiguous(),
            name=f"layer_{layer_spec.layer_index}_layer_scale_weight",
        )
        hidden_out_dt = self.pk.attach_input(
            self.hidden_out, name=f"layer_{layer_spec.layer_index}_hidden_out"
        )

        self.pk.rmsnorm_layer(
            input=hidden_dt,
            weight=attn_norm_dt,
            output=rmsnorm_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.pk.linear_layer(
            input=rmsnorm_out_dt,
            weight=qkv_weight_dt,
            output=qkv_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        if self.use_single_batch_extend_attention:
            self.pk.single_batch_extend_attention_layer(
                input=qkv_out_dt,
                k_cache=k_cache_dt,
                v_cache=v_cache_dt,
                q_norm=q_norm_dt,
                k_norm=k_norm_dt,
                cos_pos_embed=cos_dt,
                sin_pos_embed=sin_dt,
                output=attn_out_dt,
                grid_dim=(1, self.attn_kv_heads, 1),
                block_dim=(256, 1, 1),
            )
        elif self.use_split_kv_attention:
            assert lse_dt is not None
            assert attn_out_tmp_dt is not None
            self.pk.paged_attention_split_kv_layer(
                input=qkv_out_dt,
                k_cache=k_cache_dt,
                v_cache=v_cache_dt,
                q_norm=q_norm_dt,
                k_norm=k_norm_dt,
                cos_pos_embed=cos_dt,
                sin_pos_embed=sin_dt,
                lse=lse_dt,
                output=attn_out_tmp_dt,
                attention_params=(int(layer_spec.q_heads), self.num_kv_cache_chunks),
                grid_dim=(1, self.attn_kv_heads, self.num_kv_cache_chunks),
                block_dim=(256, 1, 1),
            )
            self.pk.paged_attention_split_kv_merge_layer(
                lse=lse_dt,
                output_tmp=attn_out_tmp_dt,
                output=attn_out_dt,
                attention_params=(int(layer_spec.q_heads), int(layer_spec.head_dim)),
                grid_dim=(1, self.attn_kv_heads, 1),
                block_dim=(256, 1, 1),
            )
        else:
            self.pk.paged_attention_layer(
                input=qkv_out_dt,
                k_cache=k_cache_dt,
                v_cache=v_cache_dt,
                q_norm=q_norm_dt,
                k_norm=k_norm_dt,
                cos_pos_embed=cos_dt,
                sin_pos_embed=sin_dt,
                output=attn_out_dt,
                grid_dim=(1, self.attn_kv_heads, 1),
                block_dim=(128, 1, 1),
            )
        self.pk.linear_layer(
            input=attn_out_dt,
            weight=o_proj_weight_dt,
            output=o_proj_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        if disable_post_attn_tail:
            self.pk.linear_with_residual_layer(
                input=o_proj_dt,
                weight=identity_weight_dt,
                residual=zero_residual_dt,
                output=hidden_out_dt,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        else:
            self.pk.rmsnorm_layer(
                input=o_proj_dt,
                weight=post_attn_norm_weight_dt,
                output=post_attn_norm_dt,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.pk.linear_with_residual_layer(
                input=post_attn_norm_dt,
                weight=identity_weight_dt,
                residual=hidden_dt,
                output=post_attn_dt,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
            self.pk.rmsnorm_layer(
                input=post_attn_dt,
                weight=pre_ffn_norm_weight_dt,
                output=ffn_normed_dt,
                block_dim=(128, 1, 1),
                grid_dim=(1, 1, 1),
            )
            self.pk.linear_layer(
                input=ffn_normed_dt,
                weight=ffn_gate_up_weight_dt,
                output=ffn_gate_up_dt,
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
            if disable_ffn_tail:
                self.pk.linear_with_residual_layer(
                    input=post_attn_dt,
                    weight=identity_weight_dt,
                    residual=zero_residual_dt,
                    output=hidden_out_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
            else:
                self.pk.moe_gelu_mul_layer(
                    input=ffn_gate_up_moe_dt,
                    output=ffn_act_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.pk.moe_w2_linear_layer(
                    input=ffn_act_dt,
                    weight=ffn_w2_weight_dt,
                    moe_routing_indices=ffn_routing_indices_dt,
                    moe_mask=ffn_routing_mask_dt,
                    output=ffn_w2_out_dt,
                    grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=False),
                    block_dim=(128, 1, 1),
                )
                self.pk.moe_mul_sum_add_layer(
                    input=ffn_w2_out_dt,
                    weight=ffn_topk_weight_dt,
                    residual=zero_residual_dt,
                    output=ffn_down_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
                self.pk.rmsnorm_layer(
                    input=ffn_down_dt,
                    weight=post_ffn_norm_weight_dt,
                    output=ffn_norm_dt,
                    grid_dim=(1, 1, 1),
                    block_dim=(128, 1, 1),
                )
                if disable_moe_tail:
                    self.pk.linear_with_residual_layer(
                        input=ffn_norm_dt,
                        weight=identity_weight_dt,
                        residual=zero_residual_dt,
                        output=hidden_out_dt,
                        grid_dim=(1, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                else:
                    self.pk.rmsnorm_layer(
                        input=post_attn_dt,
                        weight=router_norm_weight_dt,
                        output=router_in_dt,
                        grid_dim=(1, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                    self.pk.linear_layer(
                        input=router_in_dt,
                        weight=router_weight_dt,
                        output=router_logits_dt,
                        grid_dim=(1, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                    self.pk.moe_topk_softmax_routing_layer(
                        input=router_logits_dt,
                        output=(topk_weight_dt, routing_indices_dt, routing_mask_dt),
                        grid_dim=(1, 1, 1),
                        block_dim=(128, 1, 1),
                    )
                    if disable_moe_exec:
                        self.pk.linear_with_residual_layer(
                            input=ffn_norm_dt,
                            weight=identity_weight_dt,
                            residual=zero_residual_dt,
                            output=hidden_out_dt,
                            grid_dim=(1, 1, 1),
                            block_dim=(128, 1, 1),
                        )
                    else:
                        self.pk.rmsnorm_layer(
                            input=post_attn_dt,
                            weight=moe_in_norm_weight_dt,
                            output=moe_in_dt,
                            grid_dim=(1, 1, 1),
                            block_dim=(128, 1, 1),
                        )
                        self.pk.moe_w13_gelu_mul_layer(
                            input=moe_in_dt,
                            weight=moe_w13_weight_dt,
                            moe_routing_indices=routing_indices_dt,
                            moe_mask=routing_mask_dt,
                            output=moe_act_dt,
                            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=True),
                            block_dim=(128, 1, 1),
                        )
                        self.pk.moe_w2_linear_layer(
                            input=moe_act_dt,
                            weight=moe_w2_weight_dt,
                            moe_routing_indices=routing_indices_dt,
                            moe_mask=routing_mask_dt,
                            output=moe_w2_out_dt,
                            grid_dim=_moe_expert_grid_dim(self.pk, w13_linear=False),
                            block_dim=(128, 1, 1),
                        )
                        self.pk.moe_mul_sum_add_layer(
                            input=moe_w2_out_dt,
                            weight=topk_weight_dt,
                            residual=zero_residual_dt,
                            output=moe_out_dt,
                            grid_dim=(1, 1, 1),
                            block_dim=(128, 1, 1),
                        )
                        self.pk.rmsnorm_layer(
                            input=moe_out_dt,
                            weight=post_moe_norm_weight_dt,
                            output=moe_norm_dt,
                            grid_dim=(1, 1, 1),
                            block_dim=(128, 1, 1),
                        )
                        if disable_moe_merge:
                            self.pk.linear_with_residual_layer(
                                input=ffn_norm_dt,
                                weight=identity_weight_dt,
                                residual=zero_residual_dt,
                                output=hidden_out_dt,
                                grid_dim=(1, 1, 1),
                                block_dim=(128, 1, 1),
                            )
                        else:
                            self.pk.linear_with_residual_layer(
                                input=ffn_norm_dt,
                                weight=identity_weight_dt,
                                residual=moe_norm_dt,
                                output=ffn_moe_add_dt,
                                grid_dim=(1, 1, 1),
                                block_dim=(128, 1, 1),
                            )
                            if disable_post_merge_norm:
                                self.pk.linear_with_residual_layer(
                                    input=ffn_moe_add_dt,
                                    weight=identity_weight_dt,
                                    residual=zero_residual_dt,
                                    output=hidden_out_dt,
                                    grid_dim=(1, 1, 1),
                                    block_dim=(128, 1, 1),
                                )
                            else:
                                self.pk.rmsnorm_layer(
                                    input=ffn_moe_add_dt,
                                    weight=post_merge_norm_weight_dt,
                                    output=ffn_moe_norm_dt,
                                    grid_dim=(1, 1, 1),
                                    block_dim=(128, 1, 1),
                                )
                                if disable_final_residual:
                                    self.pk.linear_layer(
                                        input=ffn_moe_norm_dt,
                                        weight=layer_scale_weight_dt,
                                        output=hidden_pre_scale_dt,
                                        grid_dim=(1, 1, 1),
                                        block_dim=(128, 1, 1),
                                    )
                                else:
                                    self.pk.linear_layer(
                                        input=ffn_moe_norm_dt,
                                        weight=layer_scale_weight_dt,
                                        output=hidden_pre_scale_dt,
                                        grid_dim=(1, 1, 1),
                                        block_dim=(128, 1, 1),
                                    )
                                    self.pk.linear_layer(
                                        input=post_attn_dt,
                                        weight=layer_scale_weight_dt,
                                        output=scaled_post_attn_dt,
                                        grid_dim=(1, 1, 1),
                                        block_dim=(128, 1, 1),
                                    )
                                if disable_final_scale:
                                    self.pk.linear_with_residual_layer(
                                        input=ffn_moe_norm_dt,
                                        weight=identity_weight_dt,
                                        residual=zero_residual_dt
                                        if disable_final_residual
                                        else post_attn_dt,
                                        output=hidden_out_dt,
                                        grid_dim=(1, 1, 1),
                                        block_dim=(128, 1, 1),
                                    )
                                else:
                                    self.pk.linear_with_residual_layer(
                                        input=hidden_pre_scale_dt,
                                        weight=identity_weight_dt,
                                        residual=zero_residual_dt
                                        if disable_final_residual
                                        else scaled_post_attn_dt,
                                        output=hidden_out_dt,
                                        grid_dim=(1, 1, 1),
                                        block_dim=(128, 1, 1),
                                    )
        _constructor_probe("pre_compile")
        compile_persistent_kernel_with_patches(self.pk)
        if hasattr(self.pk, "init_request_func") and self.pk.init_request_func is not None:
            self.pk.init_request_func()
        _constructor_probe("post_compile")

    def supports(self, kv_cache: torch.Tensor, *, total_tokens: Optional[int] = None) -> bool:
        if (
            tuple(int(dim) for dim in kv_cache.shape) != self.kv_cache_shape
            or kv_cache.dtype != self.kv_cache_dtype
        ):
            return False
        if total_tokens is not None and int(total_tokens) > self.max_seq_length:
            return False
        return True

    def __call__(
        self,
        *,
        hidden: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_loc: torch.Tensor,
        cu_num_pages: torch.Tensor,
        last_page_len: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.numel() != self.hidden_size:
            raise ValueError(
                f"Expected hidden tensor with {self.hidden_size} elements, got {tuple(hidden.shape)}."
            )
        if not self.supports(kv_cache):
            raise ValueError(
                f"KV cache shape changed from {self.kv_cache_shape} to {tuple(kv_cache.shape)}."
            )
        if os.getenv("AD_MPK_SINGLE_PK_SYNC_AT_CALL", "0") == "1":
            torch.cuda.synchronize()

        self.hidden_in.copy_(hidden.reshape(1, self.hidden_size).contiguous())
        external_page_count = int(cu_num_pages[1].item())
        if external_page_count <= 0:
            raise ValueError(f"Expected at least one external KV page, got {external_page_count}.")

        external_last_page_len = int(last_page_len[0].item())
        total_tokens = (external_page_count - 1) * self.external_page_size + external_last_page_len
        if total_tokens <= 0:
            raise ValueError(f"Expected positive live KV token count, got {total_tokens}.")

        _mpk_debug(
            "single-pk call-start "
            f"layer={self.layer_spec.layer_index} total_tokens={total_tokens} "
            f"external_page_count={external_page_count} internal_page_size={self.page_size}"
        )
        internal_page_count = (total_tokens + self.page_size - 1) // self.page_size
        if internal_page_count > self.max_num_pages:
            raise ValueError(
                f"Internal KV page count {internal_page_count} exceeds allocated max {self.max_num_pages}."
            )

        used_page_ids = (
            cache_loc[:external_page_count].to(device=hidden.device, dtype=torch.long).contiguous()
        )
        used_k = (
            kv_cache.index_select(0, used_page_ids)[:, 0]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(external_page_count * self.external_page_size, self.kv_heads, self.head_dim)
        )
        used_v = (
            kv_cache.index_select(0, used_page_ids)[:, 1]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(external_page_count * self.external_page_size, self.kv_heads, self.head_dim)
        )
        if self.attn_kv_expand_factor > 1:
            used_k = used_k.repeat_interleave(self.attn_kv_expand_factor, dim=1)
            used_v = used_v.repeat_interleave(self.attn_kv_expand_factor, dim=1)

        self.k_cache.zero_()
        self.v_cache.zero_()
        self.k_cache.view(self.max_num_pages * self.page_size, self.attn_kv_heads, self.head_dim)[
            :total_tokens
        ].copy_(used_k[:total_tokens])
        self.v_cache.view(self.max_num_pages * self.page_size, self.attn_kv_heads, self.head_dim)[
            :total_tokens
        ].copy_(used_v[:total_tokens])

        paged_kv_indptr = torch.tensor(
            [0, internal_page_count], device=hidden.device, dtype=torch.int32
        )
        paged_kv_indices = torch.arange(
            internal_page_count, device=hidden.device, dtype=torch.int32
        )
        paged_kv_last_page_len = torch.tensor(
            [total_tokens - (internal_page_count - 1) * self.page_size],
            device=hidden.device,
            dtype=torch.int32,
        )
        _reset_pk_runtime_state(
            self.pk,
            active_tokens=1,
            qo_indptr=torch.tensor([0, 1], device=hidden.device, dtype=torch.int32),
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            reinit_request_resources=False,
        )
        _mpk_debug(
            "single-pk launch-start "
            f"layer={self.layer_spec.layer_index} internal_page_count={internal_page_count} "
            f"last_page_len={int(paged_kv_last_page_len[0].item())}"
        )
        launch_start = time.perf_counter()
        self.pk()
        _mpk_debug(
            "single-pk launch-returned "
            f"layer={self.layer_spec.layer_index} launch_s={time.perf_counter() - launch_start:.3f}"
        )
        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        _mpk_debug(
            "single-pk sync-done "
            f"layer={self.layer_spec.layer_index} sync_s={time.perf_counter() - sync_start:.3f}"
        )
        return self.hidden_out


def _make_synthetic_gemma_runtime_layer_spec(
    *,
    hidden_size: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    topk: int,
    num_experts: int,
    intermediate: int,
    ffn_intermediate: int,
    device: torch.device,
    weight_denom: float = 16.0,
    qkv_shared_kv: bool = False,
) -> _GemmaRuntimeLayerSpec:
    qkv_rows = (
        (q_heads + kv_heads) * head_dim if qkv_shared_kv else (q_heads + 2 * kv_heads) * head_dim
    )
    moe_up_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device=device, dtype=torch.bfloat16)
        / weight_denom
    )
    moe_gate_weight = (
        torch.randn((num_experts, intermediate, hidden_size), device=device, dtype=torch.bfloat16)
        / weight_denom
    )
    return _GemmaRuntimeLayerSpec(
        layer_index=0,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        topk=topk,
        sliding_window=None,
        qkv_weight=torch.randn((qkv_rows, hidden_size), device=device, dtype=torch.bfloat16)
        / weight_denom,
        qkv_shared_kv=qkv_shared_kv,
        input_layernorm_weight=torch.ones((hidden_size,), device=device, dtype=torch.bfloat16),
        q_norm_weight=torch.ones((head_dim,), device=device, dtype=torch.bfloat16),
        k_norm_weight=torch.ones((head_dim,), device=device, dtype=torch.bfloat16),
        v_norm_weight=torch.ones((head_dim,), device=device, dtype=torch.bfloat16),
        o_proj_weight=torch.randn((hidden_size, hidden_size), device=device, dtype=torch.bfloat16)
        / weight_denom,
        post_attention_layernorm_weight=torch.ones(
            (hidden_size,), device=device, dtype=torch.bfloat16
        ),
        pre_feedforward_layernorm_weight=torch.ones(
            (hidden_size,), device=device, dtype=torch.bfloat16
        ),
        ffn_gate_up_weight=torch.randn(
            (2 * ffn_intermediate, hidden_size), device=device, dtype=torch.bfloat16
        )
        / weight_denom,
        ffn_down_weight=torch.randn(
            (hidden_size, ffn_intermediate), device=device, dtype=torch.bfloat16
        )
        / weight_denom,
        post_feedforward_layernorm_1_weight=torch.ones(
            (hidden_size,), device=device, dtype=torch.bfloat16
        ),
        router_proj_weight=torch.randn(
            (num_experts, hidden_size), device=device, dtype=torch.bfloat16
        )
        / weight_denom,
        router_root_size=(
            0.9 + 0.2 * torch.rand((hidden_size,), device=device, dtype=torch.float32)
        ).to(torch.bfloat16),
        router_scale=(
            0.9 + 0.2 * torch.rand((hidden_size,), device=device, dtype=torch.float32)
        ).to(torch.bfloat16),
        pre_feedforward_layernorm_2_weight=torch.ones(
            (hidden_size,), device=device, dtype=torch.bfloat16
        ),
        moe_w13_stacked_weight=torch.cat((moe_up_weight, moe_gate_weight), dim=1).contiguous(),
        moe_gate_weight=moe_gate_weight,
        moe_up_weight=moe_up_weight,
        moe_w2_weight=torch.randn(
            (num_experts, hidden_size, intermediate), device=device, dtype=torch.bfloat16
        )
        / weight_denom,
        post_feedforward_layernorm_2_weight=torch.ones(
            (hidden_size,), device=device, dtype=torch.bfloat16
        ),
        post_feedforward_layernorm_weight=torch.ones(
            (hidden_size,), device=device, dtype=torch.bfloat16
        ),
        layer_scalar=torch.ones((hidden_size,), device=device, dtype=torch.bfloat16),
    )


def run_mirage_gemma_layer_single_pk_executor_external_page32_smoke(
    *,
    seed: int = 0,
    gemma_like_shapes: bool = False,
    external_page_size: int = 32,
) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    if gemma_like_shapes:
        hidden_size = 2816
        q_heads = 16
        kv_heads = 8
        head_dim = 256
        topk = 8
        num_experts = 128
        intermediate = 704
        ffn_intermediate = 2112
        weight_denom = 64.0
        hidden_denom = 32.0
        kv_base = 0.05
        kv_noise = 0.02
    else:
        hidden_size = 512
        q_heads = 4
        kv_heads = 1
        head_dim = 128
        topk = 8
        num_experts = 128
        intermediate = 64
        ffn_intermediate = 256
        weight_denom = 16.0
        hidden_denom = 8.0
        kv_base = 0.2
        kv_noise = 0.1

    layer_spec = _make_synthetic_gemma_runtime_layer_spec(
        hidden_size=hidden_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        topk=topk,
        num_experts=num_experts,
        intermediate=intermediate,
        ffn_intermediate=ffn_intermediate,
        device=device,
        weight_denom=weight_denom,
    )
    identity_weight = torch.eye(hidden_size, device=device, dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device=device, dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device=device, dtype=torch.bfloat16)

    external_max_num_pages = 8
    total_tokens = 9
    external_page_count = (total_tokens + external_page_size - 1) // external_page_size
    external_last_page_len = total_tokens - (external_page_count - 1) * external_page_size

    kv_cache = torch.zeros(
        (external_max_num_pages, 2, kv_heads, external_page_size, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    used_k = kv_base + kv_noise * torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    used_v = kv_base + kv_noise * torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    kv_cache[0, 0, :, :total_tokens, :].copy_(used_k.permute(1, 0, 2))
    kv_cache[0, 1, :, :total_tokens, :].copy_(used_v.permute(1, 0, 2))

    executor = _MirageGemmaLayerSinglePkExecutor(
        layer_spec=layer_spec,
        kv_cache=kv_cache,
        cos=cos,
        sin=sin,
        identity_weight=identity_weight,
    )
    hidden = torch.randn((1, hidden_size), device=device, dtype=torch.bfloat16) / hidden_denom
    external_last_page_len = total_tokens - (external_page_count - 1) * external_page_size
    cache_loc = torch.arange(external_page_count, device=device, dtype=torch.int32)
    cu_num_pages = torch.tensor([0, external_page_count], device=device, dtype=torch.int32)
    last_page_len = torch.tensor([external_last_page_len], device=device, dtype=torch.int32)

    hidden_out = executor(
        hidden=hidden,
        kv_cache=kv_cache,
        cache_loc=cache_loc,
        cu_num_pages=cu_num_pages,
        last_page_len=last_page_len,
    )
    torch.cuda.synchronize()
    return {
        "hidden_out_absmax": float(hidden_out.float().abs().max().item()),
        "hidden_out_norm": float(hidden_out.float().norm().item()),
        "hidden_out_has_nan": float(torch.isnan(hidden_out).any().item()),
        "external_page_size": float(external_page_size),
        "external_page_count": float(external_page_count),
        "external_last_page_len": float(external_last_page_len),
    }


def run_mirage_gemma_global_layer_single_pk_smoke(
    *,
    seed: int = 0,
    external_page_size: int = 32,
    total_tokens: int = 20,
) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden_size = 2816
    q_heads = 16
    kv_heads = 2
    head_dim = 512
    topk = 8
    num_experts = 128
    intermediate = 704
    ffn_intermediate = 2112
    weight_denom = 64.0
    hidden_denom = 32.0
    kv_base = 0.05
    kv_noise = 0.02

    layer_spec = _make_synthetic_gemma_runtime_layer_spec(
        hidden_size=hidden_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        topk=topk,
        num_experts=num_experts,
        intermediate=intermediate,
        ffn_intermediate=ffn_intermediate,
        device=device,
        weight_denom=weight_denom,
        qkv_shared_kv=True,
    )
    identity_weight = torch.eye(hidden_size, device=device, dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device=device, dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device=device, dtype=torch.bfloat16)

    external_max_num_pages = 64
    external_page_count = (total_tokens + external_page_size - 1) // external_page_size
    external_last_page_len = total_tokens - (external_page_count - 1) * external_page_size

    kv_cache = torch.zeros(
        (external_max_num_pages, 2, kv_heads, external_page_size, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    used_k = kv_base + kv_noise * torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    used_v = kv_base + kv_noise * torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    for page_idx in range(external_page_count):
        page_start = page_idx * external_page_size
        page_end = min(page_start + external_page_size, total_tokens)
        page_tokens = page_end - page_start
        kv_cache[page_idx, 0, :, :page_tokens, :].copy_(
            used_k[page_start:page_end].permute(1, 0, 2)
        )
        kv_cache[page_idx, 1, :, :page_tokens, :].copy_(
            used_v[page_start:page_end].permute(1, 0, 2)
        )

    executor = _MirageGemmaLayerSinglePkExecutor(
        layer_spec=layer_spec,
        kv_cache=kv_cache,
        cos=cos,
        sin=sin,
        identity_weight=identity_weight,
        max_seq_length=256,
    )
    hidden = torch.randn((1, hidden_size), device=device, dtype=torch.bfloat16) / hidden_denom
    cache_loc = torch.arange(external_page_count, device=device, dtype=torch.int32)
    cu_num_pages = torch.tensor([0, external_page_count], device=device, dtype=torch.int32)
    last_page_len = torch.tensor([external_last_page_len], device=device, dtype=torch.int32)

    hidden_out = executor(
        hidden=hidden,
        kv_cache=kv_cache,
        cache_loc=cache_loc,
        cu_num_pages=cu_num_pages,
        last_page_len=last_page_len,
    )
    torch.cuda.synchronize()
    return {
        "hidden_out_absmax": float(hidden_out.float().abs().max().item()),
        "hidden_out_norm": float(hidden_out.float().norm().item()),
        "hidden_out_has_nan": float(torch.isnan(hidden_out).any().item()),
        "external_page_size": float(external_page_size),
        "external_page_count": float(external_page_count),
        "external_last_page_len": float(external_last_page_len),
        "total_tokens": float(total_tokens),
    }


def run_mirage_gemma_global_attention_single_pk_smoke(
    *,
    seed: int = 0,
    external_page_size: int = 32,
    total_tokens: int = 20,
    include_o_proj: bool = False,
) -> Dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Mirage kernel correctness tests.")

    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden_size = 2816
    q_heads = 16
    kv_heads = 2
    head_dim = 512
    page_size = 64
    max_num_pages = 4
    weight_denom = 64.0
    hidden_denom = 32.0
    kv_base = 0.05
    kv_noise = 0.02

    pk = create_test_persistent_kernel(
        max_seq_length=256,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=max_num_pages,
        page_size=page_size,
        use_cutlass_kernel=False,
    )

    qkv_rows = (q_heads + 2 * kv_heads) * head_dim
    hidden_in = torch.randn((1, hidden_size), device=device, dtype=torch.bfloat16) / hidden_denom
    attn_norm_weight = torch.ones((hidden_size,), device=device, dtype=torch.bfloat16)
    q_weight = (
        torch.randn((q_heads * head_dim, hidden_size), device=device, dtype=torch.bfloat16)
        / weight_denom
    )
    shared_kv_weight = (
        torch.randn((kv_heads * head_dim, hidden_size), device=device, dtype=torch.bfloat16)
        / weight_denom
    )
    q_per_kv = q_heads // kv_heads
    q_weight_grouped = q_weight.view(kv_heads, q_per_kv, head_dim, hidden_size)
    shared_kv_weight_grouped = shared_kv_weight.view(kv_heads, head_dim, hidden_size)
    grouped_rows = []
    for kv_idx in range(kv_heads):
        grouped_rows.append(q_weight_grouped[kv_idx].reshape(q_per_kv * head_dim, hidden_size))
        grouped_rows.append(shared_kv_weight_grouped[kv_idx])
        grouped_rows.append(shared_kv_weight_grouped[kv_idx])
    qkv_weight = torch.cat(grouped_rows, dim=0).contiguous()
    q_norm_weight = torch.ones((head_dim,), device=device, dtype=torch.bfloat16)
    k_norm_weight = torch.ones((head_dim,), device=device, dtype=torch.bfloat16)
    cos = torch.ones((513, head_dim), device=device, dtype=torch.bfloat16)
    sin = torch.zeros((513, head_dim), device=device, dtype=torch.bfloat16)
    qkv_out = torch.zeros((1, qkv_rows), device=device, dtype=torch.bfloat16)
    rmsnorm_out = torch.zeros((1, hidden_size), device=device, dtype=torch.bfloat16)
    attn_out = torch.zeros((1, q_heads * head_dim), device=device, dtype=torch.bfloat16)
    k_cache = torch.zeros(
        (max_num_pages, page_size, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    used_k = kv_base + kv_noise * torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    used_v = kv_base + kv_noise * torch.randn(
        (total_tokens, kv_heads, head_dim), device=device, dtype=torch.bfloat16
    )
    external_page_count = (total_tokens + external_page_size - 1) // external_page_size
    for page_idx in range(external_page_count):
        page_start = page_idx * external_page_size
        page_end = min(page_start + external_page_size, total_tokens)
        page_tokens = page_end - page_start
        k_cache[page_idx, :page_tokens].copy_(used_k[page_start:page_end])
        v_cache[page_idx, :page_tokens].copy_(used_v[page_start:page_end])

    hidden_dt = pk.attach_input(hidden_in, name="global_attn_hidden")
    attn_norm_dt = pk.attach_input(attn_norm_weight, name="global_attn_norm_weight")
    qkv_weight_dt = pk.attach_input(qkv_weight, name="global_attn_qkv_weight")
    rmsnorm_out_dt = pk.attach_input(rmsnorm_out, name="global_attn_rmsnorm_out")
    qkv_out_dt = pk.attach_input(qkv_out, name="global_attn_qkv_out")
    q_norm_dt = pk.attach_input(q_norm_weight, name="global_attn_q_norm")
    k_norm_dt = pk.attach_input(k_norm_weight, name="global_attn_k_norm")
    cos_dt = pk.attach_input(cos, name="global_attn_cos")
    sin_dt = pk.attach_input(sin, name="global_attn_sin")
    k_cache_dt = pk.attach_input(k_cache, name="global_attn_k_cache")
    v_cache_dt = pk.attach_input(v_cache, name="global_attn_v_cache")
    attn_out_dt = pk.attach_input(attn_out, name="global_attn_out")

    pk.rmsnorm_layer(
        input=hidden_dt,
        weight=attn_norm_dt,
        output=rmsnorm_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out_dt,
        weight=qkv_weight_dt,
        output=qkv_out_dt,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.single_batch_extend_attention_layer(
        input=qkv_out_dt,
        k_cache=k_cache_dt,
        v_cache=v_cache_dt,
        q_norm=q_norm_dt,
        k_norm=k_norm_dt,
        cos_pos_embed=cos_dt,
        sin_pos_embed=sin_dt,
        output=attn_out_dt,
        grid_dim=(1, kv_heads, 1),
        block_dim=(256, 1, 1),
    )

    o_proj = None
    if include_o_proj:
        o_proj = torch.zeros((1, hidden_size), device=device, dtype=torch.bfloat16)
        o_proj_weight = (
            torch.randn((hidden_size, q_heads * head_dim), device=device, dtype=torch.bfloat16)
            / weight_denom
        )
        o_proj_weight_dt = pk.attach_input(o_proj_weight, name="global_attn_o_proj_weight")
        o_proj_dt = pk.attach_input(o_proj, name="global_attn_o_proj")
        pk.linear_layer(
            input=attn_out_dt,
            weight=o_proj_weight_dt,
            output=o_proj_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )

    _mpk_debug(
        "global-attn-single-pk compile-start "
        f"include_o_proj={include_o_proj} q_heads={q_heads} kv_heads={kv_heads} head_dim={head_dim}"
    )
    compile_start = time.perf_counter()
    compile_persistent_kernel_with_patches(pk)
    _mpk_debug(
        "global-attn-single-pk compile-finished "
        f"include_o_proj={include_o_proj} compile_s={time.perf_counter() - compile_start:.3f}"
    )
    internal_page_count = (total_tokens + page_size - 1) // page_size
    _reset_pk_runtime_state(
        pk,
        active_tokens=1,
        qo_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, internal_page_count], device=device, dtype=torch.int32),
        paged_kv_indices=torch.arange(internal_page_count, device=device, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor(
            [total_tokens - (internal_page_count - 1) * page_size],
            device=device,
            dtype=torch.int32,
        ),
        reinit_request_resources=False,
    )
    _mpk_debug(f"global-attn-single-pk launch-start include_o_proj={include_o_proj}")
    launch_start = time.perf_counter()
    pk()
    _mpk_debug(
        "global-attn-single-pk launch-returned "
        f"include_o_proj={include_o_proj} launch_s={time.perf_counter() - launch_start:.3f}"
    )
    sync_start = time.perf_counter()
    torch.cuda.synchronize()
    _mpk_debug(
        "global-attn-single-pk sync-done "
        f"include_o_proj={include_o_proj} sync_s={time.perf_counter() - sync_start:.3f}"
    )
    result = {
        "attn_out_absmax": float(attn_out.float().abs().max().item()),
        "attn_out_norm": float(attn_out.float().norm().item()),
        "attn_has_nan": float(torch.isnan(attn_out).any().item()),
        "include_o_proj": float(include_o_proj),
    }
    if include_o_proj and o_proj is not None:
        result.update(
            {
                "o_proj_absmax": float(o_proj.float().abs().max().item()),
                "o_proj_norm": float(o_proj.float().norm().item()),
                "o_proj_has_nan": float(torch.isnan(o_proj).any().item()),
            }
        )
    return result


class _GemmaMirageRuntime:
    def __init__(
        self,
        source_model: GraphModule,
        translation_plan: Dict[str, Any],
        max_batch_size: Optional[int] = None,
        max_seq_length: Optional[int] = None,
    ) -> None:
        self.source_model = source_model
        self.translation_plan = translation_plan
        self._max_batch_size = max_batch_size
        self._max_seq_length = max_seq_length
        self.layer_specs = _build_gemma_runtime_specs(source_model, translation_plan)
        self.layer_blocks = [
            _GemmaDecodeLayerBlock(self, layer_spec) for layer_spec in self.layer_specs
        ]
        self._force_torch_executors = os.getenv("AD_MPK_FORCE_TORCH_EXECUTORS", "0") == "1"
        self._disable_fused_dense_ffn = os.getenv("AD_MPK_DISABLE_FUSED_DENSE_FFN", "0") == "1"
        self._disable_fused_moe = os.getenv("AD_MPK_DISABLE_FUSED_MOE", "0") == "1"
        if self._max_batch_size is not None and self._disable_fused_dense_ffn:
            print(
                "WARNING: AD_MPK_DISABLE_FUSED_DENSE_FFN=1 with max_batch_size set. "
                "The unfused FFN path uses _gelu_mul which recompiles per shape. "
                "Use the fused dense FFN path (default) for one-time compilation."
            )
        node_map = {node.name: node for node in source_model.graph.nodes}

        embed_node = node_map["model_language_model_embed_tokens_embedding"]
        self.embed_weight = _lookup_tensor_attr(
            source_model, _node_arg_getattr_target(embed_node, 0)
        )

        embed_scale_to_node = node_map["model_language_model_embed_tokens_to"]
        self.embed_scale = _lookup_tensor_attr(
            source_model, _node_arg_getattr_target(embed_scale_to_node, 0)
        )

        gather_tokens_node = next(
            node for node in source_model.graph.nodes if "gather_tokens" in _node_target_name(node)
        )
        final_norm_node = gather_tokens_node.args[0]
        if not isinstance(final_norm_node, Node):
            raise ValueError("Expected gather_tokens input to be final norm node")
        self.final_norm_weight = _lookup_tensor_attr(
            source_model, _node_arg_getattr_target(final_norm_node, 1)
        )

        self.local_cos = _find_tensor_attr_by_substrings(
            source_model, "rotary_emb_local", "ad_cos_cached"
        )
        self.local_sin = _find_tensor_attr_by_substrings(
            source_model, "rotary_emb_local", "ad_sin_cached"
        )
        try:
            self.global_cos = _find_tensor_attr_by_substrings(
                source_model, "rotary_emb_global", "ad_cos_cached"
            )
        except AttributeError:
            self.global_cos = self.local_cos
        try:
            self.global_sin = _find_tensor_attr_by_substrings(
                source_model, "rotary_emb_global", "ad_sin_cached"
            )
        except AttributeError:
            self.global_sin = self.local_sin
        self._linear_cache: Dict[tuple[int, int, int], _MirageLinearExecutor] = {}
        self._dense_ffn_cache: Dict[tuple[int, int, int], _MirageDenseFfnExecutor] = {}
        self._router_cache: Dict[tuple[int, int, int, int], _MirageRouterExecutor] = {}
        self._moe_w13_cache: Dict[tuple[int, int, int, int, int], _MirageMoeW13Executor] = {}
        self._moe_w2_cache: Dict[tuple[int, int, int, int, int], _MirageMoeW2ReduceExecutor] = {}
        self._split_moe_cache: Dict[tuple[int, int, int, int, int], _MirageSplitMoeExecutor] = {}
        self._fused_moe_cache: Dict[tuple[int, int, int, int, int], _MirageFusedMoeExecutor] = {}
        self._gelu_cache: Dict[tuple[int, ...], _MirageGeluMulExecutor] = {}
        self._matmul_cache: Dict[tuple[int, int, int], _MirageMatmulExecutor] = {}
        self._single_pk_identity_cache: Dict[int, torch.Tensor] = {}
        self._single_pk_layer_cache: Dict[int, _MirageGemmaLayerSinglePkExecutor] = {}

    def prewarm_decode_executors(self) -> None:
        if self._force_torch_executors:
            _mpk_debug("skip prewarm because torch executor mode is enabled")
            return

        prewarm_start = time.perf_counter()
        # When _max_batch_size is set, cache keys don't include capacity,
        # so prewarm keys track (dims_only) to avoid duplicate compiles.
        warmed_linear_keys: set[tuple] = set()
        warmed_dense_ffn_keys: set[tuple] = set()
        warmed_router_keys: set[tuple] = set()
        warmed_split_moe_keys: set[tuple] = set()
        warmed_fused_moe_keys: set[tuple] = set()

        for layer_spec in self.layer_specs:
            device = layer_spec.qkv_weight.device
            hidden_size = int(layer_spec.input_layernorm_weight.numel())
            qkv_out_dim = int(layer_spec.qkv_weight.shape[0])
            o_proj_in_dim = int(layer_spec.o_proj_weight.shape[1])
            o_proj_out_dim = int(layer_spec.o_proj_weight.shape[0])
            dense_intermediate = int(layer_spec.ffn_down_weight.shape[1])
            num_experts = int(layer_spec.moe_w2_weight.shape[0])
            moe_intermediate = int(layer_spec.moe_w2_weight.shape[2])
            topk = int(layer_spec.topk)

            hidden = torch.zeros((1, hidden_size), device=device, dtype=torch.bfloat16)
            attn_out_hidden = torch.zeros((1, o_proj_in_dim), device=device, dtype=torch.bfloat16)
            router_hidden = torch.zeros((1, hidden_size), device=device, dtype=torch.bfloat16)
            router_hidden[0, 0] = 1.0
            moe_hidden = torch.zeros((1, hidden_size), device=device, dtype=torch.bfloat16)
            moe_hidden[0, 0] = 1.0

            # Key structure: (dims_only) when dynamic, (capacity, dims) when legacy
            if self._max_batch_size is not None:
                linear_key_qkv = (hidden_size, qkv_out_dim)
                linear_key_oproj = (o_proj_in_dim, o_proj_out_dim)
                dense_key = (hidden_size, dense_intermediate)
                router_key = (hidden_size, num_experts, topk)
                moe_key = (hidden_size, moe_intermediate, num_experts, topk)
            else:
                linear_key_qkv = (1, hidden_size, qkv_out_dim)
                linear_key_oproj = (1, o_proj_in_dim, o_proj_out_dim)
                dense_key = (1, hidden_size, dense_intermediate)
                router_key = (1, hidden_size, num_experts, topk)
                moe_key = (1, hidden_size, moe_intermediate, num_experts, topk)

            if linear_key_qkv not in warmed_linear_keys:
                self._linear(
                    name=f"prewarm_layer_{layer_spec.layer_index}_qkv",
                    input_tensor=hidden,
                    weight_tensor=layer_spec.qkv_weight,
                )
                warmed_linear_keys.add(linear_key_qkv)

            if linear_key_oproj not in warmed_linear_keys:
                self._linear(
                    name=f"prewarm_layer_{layer_spec.layer_index}_o_proj",
                    input_tensor=attn_out_hidden,
                    weight_tensor=layer_spec.o_proj_weight,
                )
                warmed_linear_keys.add(linear_key_oproj)

            if dense_key not in warmed_dense_ffn_keys:
                if self._disable_fused_dense_ffn:
                    ffn_normed = _rms_norm(hidden, layer_spec.pre_feedforward_layernorm_weight)
                    gate_up = self._linear(
                        name=f"prewarm_layer_{layer_spec.layer_index}_ffn_gate_up_unfused",
                        input_tensor=ffn_normed,
                        weight_tensor=layer_spec.ffn_gate_up_weight,
                    )
                    gate, up = torch.chunk(gate_up, 2, dim=-1)
                    act = self._gelu_mul(gate, up)
                    self._ffn_down_decode(
                        act_tensor=act,
                        weight_tensor=layer_spec.ffn_down_weight,
                    )
                else:
                    self._dense_ffn(
                        post_attn_tensor=hidden,
                        norm_weight_tensor=layer_spec.pre_feedforward_layernorm_weight,
                        gate_up_weight_tensor=layer_spec.ffn_gate_up_weight,
                        down_weight_tensor=layer_spec.ffn_down_weight,
                    )
                warmed_dense_ffn_keys.add(dense_key)

            topk_weight = routing_indices = routing_mask = None
            if router_key not in warmed_router_keys:
                topk_weight, routing_indices, routing_mask = self._router(
                    hidden_tensor=router_hidden,
                    weight_tensor=layer_spec.router_proj_weight,
                    topk=topk,
                )
                warmed_router_keys.add(router_key)

            if self._disable_fused_moe:
                already_warmed = moe_key in warmed_split_moe_keys
            else:
                already_warmed = moe_key in warmed_fused_moe_keys
            if not already_warmed:
                if topk_weight is None or routing_indices is None or routing_mask is None:
                    topk_weight, routing_indices, routing_mask = self._router(
                        hidden_tensor=router_hidden,
                        weight_tensor=layer_spec.router_proj_weight,
                        topk=topk,
                    )
                if self._disable_fused_moe:
                    self._split_moe(
                        hidden_tensor=moe_hidden,
                        w13_weight_tensor=layer_spec.moe_w13_stacked_weight,
                        routing_indices=routing_indices,
                        routing_mask=routing_mask,
                        w2_weight_tensor=layer_spec.moe_w2_weight,
                        topk_weight_tensor=topk_weight,
                    )
                    warmed_split_moe_keys.add(moe_key)
                else:
                    self._fused_moe(
                        hidden_tensor=moe_hidden,
                        w13_weight_tensor=layer_spec.moe_w13_stacked_weight,
                        routing_indices=routing_indices,
                        routing_mask=routing_mask,
                        w2_weight_tensor=layer_spec.moe_w2_weight,
                        topk_weight_tensor=topk_weight,
                    )
                    warmed_fused_moe_keys.add(moe_key)

        torch.cuda.synchronize()
        _mpk_debug(
            "prewarm decode executors finished "
            f"linear={len(warmed_linear_keys)} dense_ffn={len(warmed_dense_ffn_keys)} "
            f"router={len(warmed_router_keys)} split_moe={len(warmed_split_moe_keys)} "
            f"fused_moe={len(warmed_fused_moe_keys)} "
            f"elapsed_s={time.perf_counter() - prewarm_start:.3f}"
        )

    def _single_pk_identity(self, hidden_size: int, device: torch.device) -> torch.Tensor:
        identity = self._single_pk_identity_cache.get(hidden_size)
        if identity is None:
            identity = torch.eye(hidden_size, device=device, dtype=torch.bfloat16)
            self._single_pk_identity_cache[hidden_size] = identity
        return identity

    def _get_single_pk_layer_executor(
        self,
        *,
        layer_spec: _GemmaRuntimeLayerSpec,
        kv_cache: torch.Tensor,
        total_tokens: int,
    ) -> _MirageGemmaLayerSinglePkExecutor:
        executor = self._single_pk_layer_cache.get(layer_spec.layer_index)
        if executor is not None and executor.supports(kv_cache, total_tokens=total_tokens):
            _mpk_debug(
                "single-pk cache-hit "
                f"layer={layer_spec.layer_index} total_tokens={int(total_tokens)} "
                f"max_seq_length={int(executor.max_seq_length)}"
            )
            return executor

        cos = self.local_cos if layer_spec.head_dim == 256 else self.global_cos
        sin = self.local_sin if layer_spec.head_dim == 256 else self.global_sin
        if int(cos.shape[1]) != int(layer_spec.head_dim) or int(sin.shape[1]) != int(
            layer_spec.head_dim
        ):
            raise ValueError(
                f"RoPE cache width mismatch for layer {layer_spec.layer_index}: "
                f"expected {int(layer_spec.head_dim)}, got cos={tuple(int(dim) for dim in cos.shape)} "
                f"sin={tuple(int(dim) for dim in sin.shape)}."
            )
        identity = self._single_pk_identity(
            int(layer_spec.o_proj_weight.shape[0]), layer_spec.o_proj_weight.device
        )
        min_capacity = (
            256 if (layer_spec.qkv_shared_kv and int(layer_spec.head_dim) == 512) else 128
        )
        # When max_seq_length is configured, compile at the full envelope to avoid
        # recompilation as sequences grow during serving.
        if self._max_seq_length is not None:
            page_size = self._single_pk_page_size()
            requested_capacity = max(
                min_capacity,
                ((self._max_seq_length + page_size - 1) // page_size) * page_size,
            )
        else:
            requested_capacity = max(
                min_capacity,
                (
                    (int(total_tokens) + self._single_pk_page_size() - 1)
                    // self._single_pk_page_size()
                )
                * self._single_pk_page_size(),
            )
        _mpk_debug(
            "single-pk build-start "
            f"layer={layer_spec.layer_index} total_tokens={int(total_tokens)} "
            f"requested_capacity={requested_capacity} kv_cache_shape={tuple(int(dim) for dim in kv_cache.shape)} "
            f"q_heads={int(layer_spec.q_heads)} kv_heads={int(layer_spec.kv_heads)} "
            f"head_dim={int(layer_spec.head_dim)} shared_kv={bool(layer_spec.qkv_shared_kv)}"
        )
        build_start = time.perf_counter()
        executor = _MirageGemmaLayerSinglePkExecutor(
            layer_spec=layer_spec,
            kv_cache=kv_cache,
            cos=cos,
            sin=sin,
            identity_weight=identity,
            max_seq_length=requested_capacity,
        )
        _mpk_debug(
            "single-pk build-finished "
            f"layer={layer_spec.layer_index} build_s={time.perf_counter() - build_start:.3f}"
        )
        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        _mpk_debug(
            "single-pk build-sync-done "
            f"layer={layer_spec.layer_index} sync_s={time.perf_counter() - sync_start:.3f}"
        )
        if os.getenv("AD_MPK_SINGLE_PK_SELFTEST", "0") == "1":
            zero_hidden = torch.zeros(
                (1, int(layer_spec.o_proj_weight.shape[0])),
                device=layer_spec.o_proj_weight.device,
                dtype=torch.bfloat16,
            )
            executor(
                hidden=zero_hidden,
                kv_cache=kv_cache,
                cache_loc=torch.zeros((1,), device=kv_cache.device, dtype=torch.int32),
                cu_num_pages=torch.tensor([0, 1], device=kv_cache.device, dtype=torch.int32),
                last_page_len=torch.tensor([1], device=kv_cache.device, dtype=torch.int32),
            )
            selftest_sync_start = time.perf_counter()
            torch.cuda.synchronize()
            _mpk_debug(
                "single-pk selftest-sync-done "
                f"layer={layer_spec.layer_index} sync_s={time.perf_counter() - selftest_sync_start:.3f}"
            )
        self._single_pk_layer_cache[layer_spec.layer_index] = executor
        return executor

    def _single_pk_page_size(self) -> int:
        return 64

    def _linear(
        self,
        *,
        name: str,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self._force_torch_executors:
            return (input_tensor.float() @ weight_tensor.float().transpose(0, 1)).to(torch.bfloat16)

        actual_batch = int(input_tensor.shape[0])
        in_dim = int(weight_tensor.shape[1])
        out_dim = int(weight_tensor.shape[0])

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (in_dim, out_dim), compiled at max_batch
            max_cap = self._max_batch_size
            if actual_batch > max_cap:
                outputs = []
                for start in range(0, actual_batch, max_cap):
                    end = min(start + max_cap, actual_batch)
                    outputs.append(
                        self._linear(
                            name=f"{name}_chunk_{start}_{end}",
                            input_tensor=input_tensor[start:end].contiguous(),
                            weight_tensor=weight_tensor,
                        )
                    )
                return torch.cat(outputs, dim=0)

            key = (in_dim, out_dim)
            executor = self._linear_cache.get(key)
            if executor is None:
                _mpk_debug(
                    f"compile linear executor max_batch={max_cap} in_dim={in_dim} out_dim={out_dim} name={name}"
                )
                build_start = time.perf_counter()
                executor = _MirageLinearExecutor(
                    capacity=max_cap,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    name=f"linear_{max_cap}_{in_dim}_{out_dim}",
                )
                self._linear_cache[key] = executor
                _mpk_profile(
                    "build linear executor "
                    f"max_batch={max_cap} in_dim={in_dim} out_dim={out_dim} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(input_tensor, weight_tensor, actual_batch=actual_batch)

        # Legacy path: one executor per exact (capacity, in_dim, out_dim)
        if actual_batch > _MIRAGE_LINEAR_MAX_CAPACITY:
            outputs = []
            for start in range(0, actual_batch, _MIRAGE_LINEAR_MAX_CAPACITY):
                end = min(start + actual_batch, _MIRAGE_LINEAR_MAX_CAPACITY)
                outputs.append(
                    self._linear(
                        name=f"{name}_chunk_{start}_{end}",
                        input_tensor=input_tensor[start:end].contiguous(),
                        weight_tensor=weight_tensor,
                    )
                )
            return torch.cat(outputs, dim=0)

        key = (actual_batch, in_dim, out_dim)
        executor = self._linear_cache.get(key)
        if executor is None:
            _mpk_debug(
                f"compile linear executor capacity={actual_batch} in_dim={in_dim} out_dim={out_dim} name={name}"
            )
            build_start = time.perf_counter()
            executor = _MirageLinearExecutor(
                capacity=actual_batch,
                in_dim=in_dim,
                out_dim=out_dim,
                name=f"linear_{actual_batch}_{in_dim}_{out_dim}",
            )
            self._linear_cache[key] = executor
            _mpk_profile(
                "build linear executor "
                f"capacity={actual_batch} in_dim={in_dim} out_dim={out_dim} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(input_tensor, weight_tensor)

    def _router(
        self,
        *,
        hidden_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._force_torch_executors:
            logits = hidden_tensor.float() @ weight_tensor.float().transpose(0, 1)
            topk_weight, topk_indices = torch.topk(logits, k=topk, dim=-1)
            topk_weight = torch.softmax(topk_weight, dim=-1)
            capacity = int(hidden_tensor.shape[0])
            num_experts = int(weight_tensor.shape[0])
            routing_indices = torch.zeros(
                (num_experts, capacity), device=hidden_tensor.device, dtype=torch.int32
            )
            for token_idx in range(capacity):
                for rank_idx in range(topk):
                    expert_idx = int(topk_indices[token_idx, rank_idx].item())
                    routing_indices[expert_idx, token_idx] = rank_idx + 1
            routing_mask = torch.zeros(
                (num_experts + 1,), device=hidden_tensor.device, dtype=torch.int32
            )
            active_experts = torch.nonzero(routing_indices.any(dim=1), as_tuple=False).flatten()
            routing_mask[: active_experts.numel()] = active_experts.to(torch.int32)
            routing_mask[num_experts] = int(active_experts.numel())
            return (
                topk_weight.to(torch.float32),
                routing_indices,
                routing_mask,
            )

        capacity = int(hidden_tensor.shape[0])
        num_experts = int(weight_tensor.shape[0])

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (hidden, num_experts, topk)
            max_cap = self._max_batch_size
            actual_batch = capacity
            key = (int(hidden_tensor.shape[1]), num_experts, topk)
            executor = self._router_cache.get(key)
            if executor is None:
                _mpk_debug(
                    "compile router executor "
                    f"max_batch={max_cap} hidden={int(hidden_tensor.shape[1])} "
                    f"num_experts={num_experts} topk={topk}"
                )
                build_start = time.perf_counter()
                executor = _MirageRouterExecutor(
                    capacity=max_cap,
                    hidden_size=int(hidden_tensor.shape[1]),
                    num_experts=num_experts,
                    topk=topk,
                )
                self._router_cache[key] = executor
                _mpk_profile(
                    "build router executor "
                    f"max_batch={max_cap} hidden={int(hidden_tensor.shape[1])} "
                    f"num_experts={num_experts} topk={topk} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(hidden_tensor, weight_tensor, actual_batch=actual_batch)

        # Legacy path: one executor per exact (capacity, hidden, num_experts, topk)
        key = (capacity, int(hidden_tensor.shape[1]), num_experts, topk)
        executor = self._router_cache.get(key)
        if executor is None:
            _mpk_debug(
                "compile router executor "
                f"capacity={capacity} hidden={int(hidden_tensor.shape[1])} "
                f"num_experts={num_experts} topk={topk}"
            )
            build_start = time.perf_counter()
            executor = _MirageRouterExecutor(
                capacity=capacity,
                hidden_size=int(hidden_tensor.shape[1]),
                num_experts=num_experts,
                topk=topk,
            )
            self._router_cache[key] = executor
            _mpk_profile(
                "build router executor "
                f"capacity={capacity} hidden={int(hidden_tensor.shape[1])} "
                f"num_experts={num_experts} topk={topk} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(hidden_tensor, weight_tensor)

    def _dense_ffn(
        self,
        *,
        post_attn_tensor: torch.Tensor,
        norm_weight_tensor: torch.Tensor,
        gate_up_weight_tensor: torch.Tensor,
        down_weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self._force_torch_executors:
            normed = _rms_norm(post_attn_tensor, norm_weight_tensor)
            gate_up = normed.float() @ gate_up_weight_tensor.float().transpose(0, 1)
            gate, up = torch.chunk(gate_up, 2, dim=-1)
            return ((F.gelu(gate) * up) @ down_weight_tensor.float().transpose(0, 1)).to(
                torch.bfloat16
            )

        capacity = int(post_attn_tensor.shape[0])
        hidden_size = int(post_attn_tensor.shape[1])
        intermediate = int(down_weight_tensor.shape[1])

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (hidden_size, intermediate)
            max_cap = self._max_batch_size
            actual_batch = capacity
            key = (hidden_size, intermediate)
            executor = self._dense_ffn_cache.get(key)
            if executor is None:
                _mpk_debug(
                    "compile dense_ffn executor "
                    f"max_batch={max_cap} hidden={hidden_size} intermediate={intermediate}"
                )
                build_start = time.perf_counter()
                executor = _MirageDenseFfnExecutor(
                    capacity=max_cap,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate,
                )
                self._dense_ffn_cache[key] = executor
                _mpk_profile(
                    "build dense_ffn executor "
                    f"max_batch={max_cap} hidden={hidden_size} intermediate={intermediate} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(
                post_attn_tensor,
                norm_weight_tensor,
                gate_up_weight_tensor,
                down_weight_tensor,
                actual_batch=actual_batch,
            )

        # Legacy path: one executor per exact (capacity, hidden_size, intermediate)
        key = (capacity, hidden_size, intermediate)
        executor = self._dense_ffn_cache.get(key)
        if executor is None:
            _mpk_debug(
                "compile dense_ffn executor "
                f"capacity={capacity} hidden={hidden_size} intermediate={intermediate}"
            )
            build_start = time.perf_counter()
            executor = _MirageDenseFfnExecutor(
                capacity=capacity,
                hidden_size=hidden_size,
                intermediate_size=intermediate,
            )
            self._dense_ffn_cache[key] = executor
            _mpk_profile(
                "build dense_ffn executor "
                f"capacity={capacity} hidden={hidden_size} intermediate={intermediate} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(
            post_attn_tensor,
            norm_weight_tensor,
            gate_up_weight_tensor,
            down_weight_tensor,
        )

    def _moe_w13(
        self,
        *,
        hidden_tensor: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._force_torch_executors:
            capacity = int(hidden_tensor.shape[0])
            gate_out = torch.zeros(
                (capacity, topk, int(gate_weight.shape[1])),
                device=hidden_tensor.device,
                dtype=torch.bfloat16,
            )
            up_out = torch.zeros_like(gate_out)
            for token_idx in range(capacity):
                selected_experts = _extract_ranked_experts(
                    routing_indices[:, token_idx : token_idx + 1], topk
                )
                for route_idx, expert_index in enumerate(selected_experts):
                    gate_out[token_idx, route_idx] = (
                        hidden_tensor[token_idx].float()
                        @ gate_weight[expert_index].float().transpose(0, 1)
                    ).to(torch.bfloat16)
                    up_out[token_idx, route_idx] = (
                        hidden_tensor[token_idx].float()
                        @ up_weight[expert_index].float().transpose(0, 1)
                    ).to(torch.bfloat16)
            return gate_out, up_out

        capacity = int(hidden_tensor.shape[0])

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (hidden, intermediate, num_experts, topk)
            max_cap = self._max_batch_size
            actual_batch = capacity
            key = (
                int(hidden_tensor.shape[1]),
                int(gate_weight.shape[1]),
                int(gate_weight.shape[0]),
                topk,
            )
            executor = self._moe_w13_cache.get(key)
            if executor is None:
                _mpk_debug(
                    "compile moe_w13 executor "
                    f"max_batch={max_cap} hidden={int(hidden_tensor.shape[1])} "
                    f"intermediate={int(gate_weight.shape[1])} num_experts={int(gate_weight.shape[0])} "
                    f"topk={topk}"
                )
                build_start = time.perf_counter()
                executor = _MirageMoeW13Executor(
                    capacity=max_cap,
                    hidden_size=int(hidden_tensor.shape[1]),
                    intermediate_size=int(gate_weight.shape[1]),
                    num_experts=int(gate_weight.shape[0]),
                    topk=topk,
                )
                self._moe_w13_cache[key] = executor
                _mpk_profile(
                    "build moe_w13 executor "
                    f"max_batch={max_cap} hidden={int(hidden_tensor.shape[1])} "
                    f"intermediate={int(gate_weight.shape[1])} "
                    f"num_experts={int(gate_weight.shape[0])} topk={topk} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(
                hidden_tensor,
                gate_weight,
                up_weight,
                routing_indices,
                routing_mask,
                actual_batch=actual_batch,
            )

        # Legacy path: one executor per exact (capacity, hidden, intermediate, num_experts, topk)
        key = (
            capacity,
            int(hidden_tensor.shape[1]),
            int(gate_weight.shape[1]),
            int(gate_weight.shape[0]),
            topk,
        )
        executor = self._moe_w13_cache.get(key)
        if executor is None:
            _mpk_debug(
                "compile moe_w13 executor "
                f"capacity={capacity} hidden={int(hidden_tensor.shape[1])} "
                f"intermediate={int(gate_weight.shape[1])} num_experts={int(gate_weight.shape[0])} "
                f"topk={topk}"
            )
            build_start = time.perf_counter()
            executor = _MirageMoeW13Executor(
                capacity=capacity,
                hidden_size=int(hidden_tensor.shape[1]),
                intermediate_size=int(gate_weight.shape[1]),
                num_experts=int(gate_weight.shape[0]),
                topk=topk,
            )
            self._moe_w13_cache[key] = executor
            _mpk_profile(
                "build moe_w13 executor "
                f"capacity={capacity} hidden={int(hidden_tensor.shape[1])} "
                f"intermediate={int(gate_weight.shape[1])} "
                f"num_experts={int(gate_weight.shape[0])} topk={topk} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(hidden_tensor, gate_weight, up_weight, routing_indices, routing_mask)

    def _fused_moe(
        self,
        *,
        hidden_tensor: torch.Tensor,
        w13_weight_tensor: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        w2_weight_tensor: torch.Tensor,
        topk_weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self._force_torch_executors:
            capacity = int(hidden_tensor.shape[0])
            topk = int(topk_weight_tensor.shape[1])
            hidden_size = int(w2_weight_tensor.shape[1])
            out = torch.zeros(
                (capacity, hidden_size), device=hidden_tensor.device, dtype=torch.float32
            )
            for token_idx in range(capacity):
                selected_experts = _extract_ranked_experts(
                    routing_indices[:, token_idx : token_idx + 1], topk
                )
                for route_idx, expert_index in enumerate(selected_experts):
                    fused = hidden_tensor[token_idx].float() @ w13_weight_tensor[
                        expert_index
                    ].float().transpose(0, 1)
                    up, gate = torch.chunk(fused, 2, dim=-1)
                    act = F.gelu(gate) * up
                    expert_out = act @ w2_weight_tensor[expert_index].float().transpose(0, 1)
                    out[token_idx] += expert_out * topk_weight_tensor[token_idx, route_idx].float()
            return out.to(torch.bfloat16)

        capacity = int(hidden_tensor.shape[0])
        hidden_size = int(hidden_tensor.shape[1])
        intermediate = int(w2_weight_tensor.shape[2])
        num_experts = int(w2_weight_tensor.shape[0])
        topk = int(topk_weight_tensor.shape[1])

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (hidden, intermediate, num_experts, topk)
            max_cap = self._max_batch_size
            actual_batch = capacity
            key = (hidden_size, intermediate, num_experts, topk)
            executor = self._fused_moe_cache.get(key)
            if executor is None:
                _mpk_debug(
                    "compile fused_moe executor "
                    f"max_batch={max_cap} hidden={hidden_size} intermediate={intermediate} "
                    f"num_experts={num_experts} topk={topk}"
                )
                build_start = time.perf_counter()
                executor = _MirageFusedMoeExecutor(
                    capacity=max_cap,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate,
                    num_experts=num_experts,
                    topk=topk,
                )
                self._fused_moe_cache[key] = executor
                _mpk_profile(
                    "build fused_moe executor "
                    f"max_batch={max_cap} hidden={hidden_size} intermediate={intermediate} "
                    f"num_experts={num_experts} topk={topk} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(
                hidden_tensor,
                w13_weight_tensor,
                routing_indices,
                routing_mask,
                w2_weight_tensor,
                topk_weight_tensor,
                actual_batch=actual_batch,
            )

        # Legacy path: one executor per exact (capacity, hidden, intermediate, num_experts, topk)
        key = (capacity, hidden_size, intermediate, num_experts, topk)
        executor = self._fused_moe_cache.get(key)
        if executor is None:
            _mpk_debug(
                "compile fused_moe executor "
                f"capacity={capacity} hidden={hidden_size} intermediate={intermediate} "
                f"num_experts={num_experts} topk={topk}"
            )
            build_start = time.perf_counter()
            executor = _MirageFusedMoeExecutor(
                capacity=capacity,
                hidden_size=hidden_size,
                intermediate_size=intermediate,
                num_experts=num_experts,
                topk=topk,
            )
            self._fused_moe_cache[key] = executor
            _mpk_profile(
                "build fused_moe executor "
                f"capacity={capacity} hidden={hidden_size} intermediate={intermediate} "
                f"num_experts={num_experts} topk={topk} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(
            hidden_tensor,
            w13_weight_tensor,
            routing_indices,
            routing_mask,
            w2_weight_tensor,
            topk_weight_tensor,
        )

    def _split_moe(
        self,
        *,
        hidden_tensor: torch.Tensor,
        w13_weight_tensor: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        w2_weight_tensor: torch.Tensor,
        topk_weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if self._force_torch_executors:
            capacity = int(hidden_tensor.shape[0])
            topk = int(topk_weight_tensor.shape[1])
            hidden_size = int(w2_weight_tensor.shape[1])
            out = torch.zeros(
                (capacity, hidden_size), device=hidden_tensor.device, dtype=torch.float32
            )
            for token_idx in range(capacity):
                selected_experts = _extract_ranked_experts(
                    routing_indices[:, token_idx : token_idx + 1], topk
                )
                for route_idx, expert_index in enumerate(selected_experts):
                    fused = hidden_tensor[token_idx].float() @ w13_weight_tensor[
                        expert_index
                    ].float().transpose(0, 1)
                    up, gate = torch.chunk(fused, 2, dim=-1)
                    act = F.gelu(gate) * up
                    expert_out = act @ w2_weight_tensor[expert_index].float().transpose(0, 1)
                    out[token_idx] += expert_out * topk_weight_tensor[token_idx, route_idx].float()
            return out.to(torch.bfloat16)

        capacity = int(hidden_tensor.shape[0])
        hidden_size = int(hidden_tensor.shape[1])
        intermediate = int(w2_weight_tensor.shape[2])
        num_experts = int(w2_weight_tensor.shape[0])
        topk = int(topk_weight_tensor.shape[1])

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (hidden, intermediate, num_experts, topk)
            max_cap = self._max_batch_size
            actual_batch = capacity
            key = (hidden_size, intermediate, num_experts, topk)
            executor = self._split_moe_cache.get(key)
            if executor is None:
                _mpk_debug(
                    "compile split_moe executor "
                    f"max_batch={max_cap} hidden={hidden_size} intermediate={intermediate} "
                    f"num_experts={num_experts} topk={topk}"
                )
                build_start = time.perf_counter()
                executor = _MirageSplitMoeExecutor(
                    capacity=max_cap,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate,
                    num_experts=num_experts,
                    topk=topk,
                )
                self._split_moe_cache[key] = executor
                _mpk_profile(
                    "build split_moe executor "
                    f"max_batch={max_cap} hidden={hidden_size} intermediate={intermediate} "
                    f"num_experts={num_experts} topk={topk} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(
                hidden_tensor,
                w13_weight_tensor,
                routing_indices,
                routing_mask,
                w2_weight_tensor,
                topk_weight_tensor,
                actual_batch=actual_batch,
            )

        # Legacy path: one executor per exact (capacity, hidden, intermediate, num_experts, topk)
        key = (capacity, hidden_size, intermediate, num_experts, topk)
        executor = self._split_moe_cache.get(key)
        if executor is None:
            _mpk_debug(
                "compile split_moe executor "
                f"capacity={capacity} hidden={hidden_size} intermediate={intermediate} "
                f"num_experts={num_experts} topk={topk}"
            )
            build_start = time.perf_counter()
            executor = _MirageSplitMoeExecutor(
                capacity=capacity,
                hidden_size=hidden_size,
                intermediate_size=intermediate,
                num_experts=num_experts,
                topk=topk,
            )
            self._split_moe_cache[key] = executor
            _mpk_profile(
                "build split_moe executor "
                f"capacity={capacity} hidden={hidden_size} intermediate={intermediate} "
                f"num_experts={num_experts} topk={topk} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(
            hidden_tensor,
            w13_weight_tensor,
            routing_indices,
            routing_mask,
            w2_weight_tensor,
            topk_weight_tensor,
        )

    def _moe_w2_reduce(
        self,
        *,
        act_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        routing_indices: torch.Tensor,
        routing_mask: torch.Tensor,
        topk_weight: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if self._force_torch_executors:
            capacity, topk, intermediate = [int(dim) for dim in act_tensor.shape]
            hidden_size = int(weight_tensor.shape[1])
            out = residual.float().clone()
            for token_idx in range(capacity):
                for expert_idx in range(int(weight_tensor.shape[0])):
                    rank = int(routing_indices[expert_idx, token_idx].item())
                    if rank == 0:
                        continue
                    route_idx = rank - 1
                    if route_idx >= topk:
                        continue
                    expert_out = act_tensor[token_idx, route_idx].float() @ weight_tensor[
                        expert_idx
                    ].float().transpose(0, 1)
                    out[token_idx, :hidden_size] += (
                        expert_out * topk_weight[token_idx, route_idx].float()
                    )
            return out.to(torch.bfloat16)

        capacity, topk, intermediate = [int(dim) for dim in act_tensor.shape]

        if self._max_batch_size is not None:
            # Dynamic batch path: one executor per (topk, hidden, intermediate, num_experts)
            max_cap = self._max_batch_size
            actual_batch = capacity
            if actual_batch > max_cap:
                # Chunking fallback for actual_batch > max_batch
                outputs = []
                for start in range(0, actual_batch, max_cap):
                    end = min(start + max_cap, actual_batch)
                    chunk_capacity = end - start
                    chunk_mask = torch.tensor(
                        [0, chunk_capacity],
                        device=routing_mask.device,
                        dtype=torch.int32,
                    )
                    outputs.append(
                        self._moe_w2_reduce(
                            act_tensor=act_tensor[start:end].contiguous(),
                            weight_tensor=weight_tensor,
                            routing_indices=routing_indices[:, start:end].contiguous(),
                            routing_mask=chunk_mask,
                            topk_weight=topk_weight[start:end].contiguous(),
                            residual=residual[start:end].contiguous(),
                        )
                    )
                return torch.cat(outputs, dim=0)

            key = (
                topk,
                int(weight_tensor.shape[1]),
                intermediate,
                int(weight_tensor.shape[0]),
            )
            executor = self._moe_w2_cache.get(key)
            if executor is None:
                _mpk_debug(
                    "compile moe_w2 executor "
                    f"max_batch={max_cap} topk={topk} hidden={int(weight_tensor.shape[1])} "
                    f"intermediate={intermediate} num_experts={int(weight_tensor.shape[0])}"
                )
                build_start = time.perf_counter()
                executor = _MirageMoeW2ReduceExecutor(
                    capacity=max_cap,
                    topk=topk,
                    hidden_size=int(weight_tensor.shape[1]),
                    intermediate_size=intermediate,
                    num_experts=int(weight_tensor.shape[0]),
                )
                self._moe_w2_cache[key] = executor
                _mpk_profile(
                    "build moe_w2 executor "
                    f"max_batch={max_cap} topk={topk} hidden={int(weight_tensor.shape[1])} "
                    f"intermediate={intermediate} num_experts={int(weight_tensor.shape[0])} "
                    f"elapsed_s={time.perf_counter() - build_start:.6f}"
                )
            return executor(
                act_tensor,
                weight_tensor,
                routing_indices,
                routing_mask,
                topk_weight,
                residual,
                actual_batch=actual_batch,
            )

        # Legacy path: one executor per exact (capacity, topk, hidden, intermediate, num_experts)
        if (
            capacity > _MIRAGE_MOE_W2_MAX_CAPACITY
            and topk == 1
            and int(weight_tensor.shape[0]) == 1
            and int(routing_indices.shape[0]) == 1
            and int(topk_weight.shape[1]) == 1
            and int(routing_mask.numel()) == 2
        ):
            outputs = []
            for start in range(0, capacity, _MIRAGE_MOE_W2_MAX_CAPACITY):
                end = min(start + _MIRAGE_MOE_W2_MAX_CAPACITY, capacity)
                chunk_capacity = end - start
                chunk_mask = torch.tensor(
                    [0, chunk_capacity],
                    device=routing_mask.device,
                    dtype=torch.int32,
                )
                outputs.append(
                    self._moe_w2_reduce(
                        act_tensor=act_tensor[start:end].contiguous(),
                        weight_tensor=weight_tensor,
                        routing_indices=routing_indices[:, start:end].contiguous(),
                        routing_mask=chunk_mask,
                        topk_weight=topk_weight[start:end].contiguous(),
                        residual=residual[start:end].contiguous(),
                    )
                )
            return torch.cat(outputs, dim=0)

        key = (
            capacity,
            topk,
            int(weight_tensor.shape[1]),
            intermediate,
            int(weight_tensor.shape[0]),
        )
        executor = self._moe_w2_cache.get(key)
        if executor is None:
            _mpk_debug(
                "compile moe_w2 executor "
                f"capacity={capacity} topk={topk} hidden={int(weight_tensor.shape[1])} "
                f"intermediate={intermediate} num_experts={int(weight_tensor.shape[0])}"
            )
            build_start = time.perf_counter()
            executor = _MirageMoeW2ReduceExecutor(
                capacity=capacity,
                topk=topk,
                hidden_size=int(weight_tensor.shape[1]),
                intermediate_size=intermediate,
                num_experts=int(weight_tensor.shape[0]),
            )
            self._moe_w2_cache[key] = executor
            _mpk_profile(
                "build moe_w2 executor "
                f"capacity={capacity} topk={topk} hidden={int(weight_tensor.shape[1])} "
                f"intermediate={intermediate} num_experts={int(weight_tensor.shape[0])} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(
            act_tensor,
            weight_tensor,
            routing_indices,
            routing_mask,
            topk_weight,
            residual,
        )

    def _gelu_mul(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if self._force_torch_executors:
            return (F.gelu(gate.float()) * up.float()).to(torch.bfloat16)

        # TODO(dynamic-batch): _MirageGeluMulExecutor uses the Mirage graph API
        # (not PK) and does not support update_dynamic_dims. When _max_batch_size
        # is set, callers should use the fused dense FFN path
        # (_MirageDenseFfnExecutor) which handles gelu*mul inside a single PK
        # with full dynamic batch support. The unfused path
        # (_disable_fused_dense_ffn=True) will still recompile per shape.

        key = tuple(int(dim) for dim in gate.shape)
        executor = self._gelu_cache.get(key)
        if executor is None:
            _mpk_debug(f"compile gelu executor shape={key}")
            build_start = time.perf_counter()
            executor = _MirageGeluMulExecutor(shape=key)
            self._gelu_cache[key] = executor
            _mpk_profile(
                f"build gelu executor shape={key} elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(gate.contiguous(), up.contiguous())

    def _matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self._force_torch_executors:
            return (a.float() @ b.float()).to(torch.bfloat16)

        # TODO(dynamic-batch): _MirageMatmulExecutor uses the Mirage graph API
        # (not PK) and does not support update_dynamic_dims. Not in the main
        # Gemma4 decode path — _ffn_down_decode routes through _moe_w2_reduce
        # which has full dynamic batch support.

        key = (int(a.shape[0]), int(a.shape[1]), int(b.shape[1]))
        executor = self._matmul_cache.get(key)
        if executor is None:
            _mpk_debug(f"compile matmul executor shape_a={tuple(a.shape)} shape_b={tuple(b.shape)}")
            build_start = time.perf_counter()
            executor = _MirageMatmulExecutor(m=key[0], k=key[1], n=key[2])
            self._matmul_cache[key] = executor
            _mpk_profile(
                "build matmul executor "
                f"shape_a={tuple(a.shape)} shape_b={tuple(b.shape)} "
                f"elapsed_s={time.perf_counter() - build_start:.6f}"
            )
        return executor(a.contiguous(), b.contiguous())

    def _ffn_down_decode(
        self,
        *,
        act_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the decode FFN down projection through the stable ``moe_w2`` path.

        The generic Mirage matmul path is currently unstable for the single-token
        Gemma decode specialization. Decode runtime is now generate-only, so we
        intentionally use the proven ``topk=1 / num_experts=1`` ``moe_w2``
        reduction path here instead.
        """

        capacity = int(act_tensor.shape[0])
        hidden_size = int(weight_tensor.shape[0])
        device = act_tensor.device

        routing_indices = torch.ones((1, capacity), device=device, dtype=torch.int32)
        routing_mask = torch.tensor([0, capacity], device=device, dtype=torch.int32)
        topk_weight = torch.ones((capacity, 1), device=device, dtype=torch.float32)
        residual = torch.zeros((capacity, hidden_size), device=device, dtype=torch.bfloat16)

        return self._moe_w2_reduce(
            act_tensor=act_tensor.view(capacity, 1, -1).contiguous(),
            weight_tensor=weight_tensor.unsqueeze(0).contiguous(),
            routing_indices=routing_indices,
            routing_mask=routing_mask,
            topk_weight=topk_weight,
            residual=residual,
        )

    def _rope_inputs(
        self,
        *,
        position_ids: torch.Tensor,
        head_dim: int,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = batch_size * seq_len
        if head_dim == 256:
            cos = self.local_cos[position_ids][..., : head_dim // 2]
            sin = self.local_sin[position_ids][..., : head_dim // 2]
        elif head_dim == 512:
            cos = self.global_cos[position_ids][..., : head_dim // 2]
            sin = self.global_sin[position_ids][..., : head_dim // 2]
        else:
            raise ValueError(f"Unsupported Gemma head_dim for rotary cache: {head_dim}")

        cos_sin_cache = (
            torch.cat((cos, sin), dim=-1).reshape(num_tokens, head_dim).to(torch.float32)
        )
        rope_positions = torch.arange(num_tokens, device=position_ids.device, dtype=torch.int32)
        return rope_positions, cos_sin_cache

    def __call__(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_gather_indices: torch.Tensor,
        batch_info_host: torch.Tensor,
        cu_seqlen_host: torch.Tensor,
        cu_num_pages: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        last_page_len: torch.Tensor,
        last_page_len_host: torch.Tensor,
        seq_len_with_cache_host: torch.Tensor,
        cu_seqlen: torch.Tensor,
        seq_len_with_cache: torch.Tensor,
        *kv_caches: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = [int(dim) for dim in input_ids.shape]
        hidden = F.embedding(input_ids, self.embed_weight).to(torch.bfloat16)
        hidden = hidden * self.embed_scale.to(dtype=hidden.dtype)
        debug_tensors = {} if os.getenv("AD_MPK_COMPARE_LAYER0_DEBUG", "0") == "1" else None

        triton_batch_indices, triton_positions = (
            torch.ops.auto_deploy.triton_paged_prepare_metadata.default(
                position_ids,
                batch_info_host,
                cu_seqlen,
                seq_len_with_cache,
            )
        )
        batch_info = __import__(
            "tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface",
            fromlist=["BatchInfo"],
        ).BatchInfo(batch_info_host)
        num_prefill, num_prefill_tokens, num_decode = batch_info.get_absorbed_info()

        for layer_block, kv_cache in zip(self.layer_blocks, kv_caches):
            layer_spec = layer_block.layer_spec
            _mpk_debug(
                "enter layer "
                f"{layer_spec.layer_index} batch={batch_size} seq={seq_len} "
                f"prefill={num_prefill} prefill_tokens={num_prefill_tokens} decode={num_decode}"
            )
            if _should_use_single_pk_layer(
                batch_size=batch_size,
                seq_len=seq_len,
                num_prefill=num_prefill,
            ):
                external_page_count = int(cu_num_pages[1].item())
                if external_page_count <= 0:
                    raise ValueError(
                        f"Expected at least one external KV page, got {external_page_count}."
                    )
                external_last_page_len = int(last_page_len[0].item())
                total_tokens = (external_page_count - 1) * int(
                    kv_cache.shape[3]
                ) + external_last_page_len
                if total_tokens <= 0:
                    raise ValueError(f"Expected positive live KV token count, got {total_tokens}.")
                if os.getenv("AD_MPK_SINGLE_PK_DEBUG_META", "0") == "1":
                    _mpk_debug(
                        "single-pk meta "
                        f"layer={layer_spec.layer_index} "
                        f"kv_cache_shape={tuple(int(dim) for dim in kv_cache.shape)} "
                        f"cache_loc_head={cache_loc[:8].tolist()} "
                        f"cu_num_pages={cu_num_pages.tolist()} "
                        f"last_page_len={last_page_len.tolist()}"
                    )
                if os.getenv("AD_MPK_SINGLE_PK_PRECALL_SYNC", "0") == "1":
                    torch.cuda.synchronize()
                layer_executor = self._get_single_pk_layer_executor(
                    layer_spec=layer_spec,
                    kv_cache=kv_cache,
                    total_tokens=total_tokens,
                )
                hidden = layer_executor(
                    hidden=hidden.reshape(1, -1),
                    kv_cache=kv_cache,
                    cache_loc=cache_loc,
                    cu_num_pages=cu_num_pages,
                    last_page_len=last_page_len,
                ).view(batch_size, seq_len, -1)
                _mpk_debug(f"exit layer {layer_spec.layer_index} [single-pk]")
                continue
            layer_outputs = _mpk_run_profiled(
                f"layer_{layer_spec.layer_index}.total",
                lambda: layer_block(
                    hidden=hidden,
                    position_ids=position_ids,
                    batch_info_host=batch_info_host,
                    cu_seqlen_host=cu_seqlen_host,
                    cu_num_pages=cu_num_pages,
                    cu_num_pages_host=cu_num_pages_host,
                    cache_loc=cache_loc,
                    last_page_len=last_page_len,
                    last_page_len_host=last_page_len_host,
                    seq_len_with_cache_host=seq_len_with_cache_host,
                    triton_batch_indices=triton_batch_indices,
                    triton_positions=triton_positions,
                    kv_cache=kv_cache,
                    capture_debug=debug_tensors is not None and layer_spec.layer_index == 0,
                ),
            )
            hidden = layer_outputs.hidden
            if debug_tensors is not None and layer_spec.layer_index == 0:
                debug_tensors.update(layer_outputs.debug_tensors)
            _mpk_debug(f"exit layer {layer_spec.layer_index}")

        hidden = _mpk_run_profiled(
            "final.rmsnorm",
            lambda: _rms_norm(hidden, self.final_norm_weight),
        )
        gathered = _mpk_run_profiled(
            "final.gather_tokens",
            lambda: torch.ops.auto_deploy.gather_tokens.default(
                hidden,
                token_gather_indices,
                batch_info_host,
            ),
        )
        logits = _mpk_run_profiled(
            "final.lm_head",
            lambda: gathered @ self.embed_weight.transpose(0, 1),
        )
        logits = torch.tanh(logits / 30.0) * 30.0
        logits = logits.view(*gathered.shape[:-1], logits.shape[-1])
        _mpk_debug(f"return logits shape={tuple(logits.shape)}")
        output = {"logits": logits}
        if debug_tensors is not None:
            output["_debug_tensors"] = debug_tensors
        return output


def build_gemma_mirage_runtime_callable(
    translation_plan: Dict[str, Any],
    source_model: Optional[GraphModule] = None,
    max_batch_size: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> Callable[..., Any]:
    """Build the live Gemma MPK runtime callable.

    The runtime path is intentionally strict: once selected, generate-only
    execution must go through the Mirage-backed callable rather than an eager
    fallback. The default decode path is blockwise hybrid execution; the
    single-PK per-layer path remains an explicit opt-in.

    Args:
        max_batch_size: If set, executors compile once at this envelope and use
            pk.update_dynamic_dims() for smaller actual batches at runtime.
            If None, falls back to per-shape compilation (legacy behavior).
        max_seq_length: If set, the single-PK layer executor compiles at this
            envelope so growing sequences don't trigger recompilation.
    """
    if source_model is None:
        layer_lowerings = translation_plan.get("layer_lowerings", [])
        num_gap_steps = 0
        num_partial_steps = 0
        for layer in layer_lowerings:
            for step in layer.get("mpk_steps", []):
                status = str(step.get("status", ""))
                if status == GemmaLoweringStatus.GAP.value:
                    num_gap_steps += 1
                elif status == GemmaLoweringStatus.PARTIAL.value:
                    num_partial_steps += 1

        def _missing_model_runtime_callable(*args, **kwargs):
            raise NotImplementedError(
                "Live Mirage execution for the full Gemma MPK path requires the source GraphModule. "
                f"Current plan has {num_gap_steps} gap steps and {num_partial_steps} partial steps."
            )

        return _missing_model_runtime_callable
    return _GemmaMirageRuntime(
        source_model,
        translation_plan,
        max_batch_size=max_batch_size,
        max_seq_length=max_seq_length,
    )


def create_test_persistent_kernel(
    *,
    max_seq_length: int = 16,
    max_num_batched_requests: int = 2,
    max_num_batched_tokens: int = 4,
    max_num_pages: int = 8,
    page_size: int = 2,
    use_cutlass_kernel: bool = True,
    target_cc_override: Optional[int] = None,
    enable_profiler: bool = False,
    schedule_mode: str = "dynamic",
):
    """Create a tiny Mirage ``PersistentKernel`` suitable for task-registration smoke tests."""

    PersistentKernel = _require_mirage()
    meta_tensors = {
        "step": torch.zeros((1,), dtype=torch.int32, device="cuda"),
        "tokens": torch.zeros((1, max_seq_length), dtype=torch.int64, device="cuda"),
        "input_tokens": torch.zeros((max_num_batched_tokens, 1), dtype=torch.int64, device="cuda"),
        "output_tokens": torch.zeros((max_num_batched_tokens, 1), dtype=torch.int64, device="cuda"),
        "num_new_tokens": torch.ones((1,), dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.zeros((1,), dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.zeros(
            (max_num_batched_requests + 1,), dtype=torch.int32, device="cuda"
        ),
        "paged_kv_indptr_buffer": torch.zeros(
            (max_num_batched_requests + 1,), dtype=torch.int32, device="cuda"
        ),
        "paged_kv_indices_buffer": torch.zeros((max_num_pages,), dtype=torch.int32, device="cuda"),
        "paged_kv_last_page_len_buffer": torch.ones(
            (max_num_batched_requests,), dtype=torch.int32, device="cuda"
        ),
    }
    profiler_tensor = (
        torch.zeros((1,), dtype=torch.int32, device="cuda") if enable_profiler else None
    )
    is_static = schedule_mode == "static"
    pk_kwargs: Dict[str, Any] = {
        "mode": "offline",
        "world_size": 1,
        "mpi_rank": 0,
        "num_workers": 1,
        "num_local_schedulers": 0 if is_static else 1,
        "num_remote_schedulers": 0,
        "max_seq_length": max_seq_length,
        "max_num_batched_requests": max_num_batched_requests,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_pages": max_num_pages,
        "page_size": page_size,
        "meta_tensors": meta_tensors,
        "profiler_tensor": profiler_tensor,
        "trace_name": "gemma_mpk_bridge_smoke",
        "spec_decode_config": None,
        "use_cutlass_kernel": use_cutlass_kernel,
    }
    if is_static:
        pk_kwargs["schedule_mode"] = "static"
    pk = PersistentKernel(**pk_kwargs)
    if target_cc_override is not None:
        pk.target_cc = target_cc_override
    return pk


def _build_test_tensor_registry(pk) -> Dict[str, Any]:
    registry: Dict[str, Any] = {}

    registry["hidden_in"] = pk.attach_input(
        torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_hidden_in"
    )
    registry["weight_norm"] = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_weight_norm"
    )
    registry["qkv_weight"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_qkv_weight"
    )
    registry["qkv_out"] = pk.new_tensor((4, 16), name="bridge_qkv_out")

    registry["k_cache"] = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_k_cache"
    )
    registry["v_cache"] = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_v_cache"
    )
    registry["q_norm"] = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_q_norm"
    )
    registry["k_norm"] = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_k_norm"
    )
    registry["cos"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_cos"
    )
    registry["sin"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_sin"
    )
    registry["attn_out"] = pk.new_tensor((4, 16), name="bridge_attn_out")

    registry["proj_weight"] = pk.attach_input(
        torch.zeros((8, 16), dtype=torch.bfloat16, device="cuda"), name="bridge_proj_weight"
    )
    registry["post_attn_residual"] = pk.new_tensor((4, 8), name="bridge_post_attn_residual")

    registry["ffn_gate_up_weight"] = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_ffn_gate_up_weight"
    )
    registry["ffn_gate_up_out"] = pk.new_tensor((4, 16), name="bridge_ffn_gate_up_out")

    registry["router_weight"] = pk.attach_input(
        torch.zeros((8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_router_weight"
    )
    registry["router_logits"] = pk.new_tensor((4, 8), name="bridge_router_logits")
    registry["topk_weight"] = pk.attach_input(
        torch.zeros((4, 2), dtype=torch.float32, device="cuda"), name="bridge_topk_weight"
    )
    registry["routing_indices"] = pk.new_tensor((8, 4), name="bridge_routing_indices")
    registry["routing_mask"] = pk.new_tensor((9,), name="bridge_routing_mask")

    registry["moe_weight_w13"] = pk.attach_input(
        torch.zeros((8, 16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w13"
    )
    registry["moe_w13_out"] = pk.new_tensor((4, 2, 16), name="bridge_moe_w13_out")
    registry["moe_act_out"] = pk.new_tensor((4, 2, 8), name="bridge_moe_act_out")
    registry["moe_weight_w2"] = pk.attach_input(
        torch.zeros((8, 8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w2"
    )
    registry["moe_w2_out"] = pk.new_tensor((4, 2, 8), name="bridge_moe_w2_out")
    registry["hidden_out"] = pk.new_tensor((4, 8), name="bridge_hidden_out")
    return registry


def exercise_layer_plan_against_mirage(
    layer_plan: GemmaLayerLoweringPlan,
    *,
    execute_gap_steps: bool = False,
) -> Dict[str, Any]:
    """Execute the Mirage-resolved subset of a planned Gemma layer on a test kernel."""

    pk = create_test_persistent_kernel()
    tensors = _build_test_tensor_registry(pk)
    bindings = resolve_layer_plan_against_mirage(layer_plan)

    executed_steps: list[str] = []
    skipped_steps: list[str] = []

    for step, binding in zip(layer_plan.mpk_steps, bindings):
        is_gap = step.status == GemmaLoweringStatus.GAP
        if not binding.resolved or (is_gap and not execute_gap_steps):
            skipped_steps.append(step.name)
            continue

        if step.name == "attn_rmsnorm_linear":
            _compose_rmsnorm_linear_layer(
                pk,
                input_tensor=tensors["hidden_in"],
                weight_norm_tensor=tensors["weight_norm"],
                weight_linear_tensor=tensors["qkv_weight"],
                output_tensor=tensors["qkv_out"],
                intermediate_name="bridge_attn_rmsnorm_out",
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "paged_attention":
            pk.paged_attention_layer(
                input=tensors["qkv_out"],
                k_cache=tensors["k_cache"],
                v_cache=tensors["v_cache"],
                q_norm=tensors["q_norm"],
                k_norm=tensors["k_norm"],
                cos_pos_embed=tensors["cos"],
                sin_pos_embed=tensors["sin"],
                output=tensors["attn_out"],
                grid_dim=(2, 2, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "attn_out_proj":
            pk.linear_with_residual_layer(
                input=tensors["attn_out"],
                weight=tensors["proj_weight"],
                residual=tensors["hidden_in"],
                output=tensors["post_attn_residual"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "dense_ffn_gate_up":
            _compose_rmsnorm_linear_layer(
                pk,
                input_tensor=tensors["post_attn_residual"],
                weight_norm_tensor=tensors["weight_norm"],
                weight_linear_tensor=tensors["ffn_gate_up_weight"],
                output_tensor=tensors["ffn_gate_up_out"],
                intermediate_name="bridge_ffn_rmsnorm_out",
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "router_projection":
            pk.linear_layer(
                input=tensors["post_attn_residual"],
                weight=tensors["router_weight"],
                output=tensors["router_logits"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "router_topk_softmax":
            pk.moe_topk_softmax_routing_layer(
                input=tensors["router_logits"],
                output=(
                    tensors["topk_weight"],
                    tensors["routing_indices"],
                    tensors["routing_mask"],
                ),
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_w13_linear":
            pk.moe_w13_linear_layer(
                input=tensors["post_attn_residual"],
                weight=tensors["moe_weight_w13"],
                moe_routing_indices=tensors["routing_indices"],
                moe_mask=tensors["routing_mask"],
                output=tensors["moe_w13_out"],
                grid_dim=_moe_expert_grid_dim(pk, w13_linear=True),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_activation":
            pk.moe_silu_mul_layer(
                input=tensors["moe_w13_out"],
                output=tensors["moe_act_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_w2_linear":
            pk.moe_w2_linear_layer(
                input=tensors["moe_act_out"],
                weight=tensors["moe_weight_w2"],
                moe_routing_indices=tensors["routing_indices"],
                moe_mask=tensors["routing_mask"],
                output=tensors["moe_w2_out"],
                grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
                block_dim=(128, 1, 1),
            )
        elif step.name == "moe_reduce":
            pk.moe_mul_sum_add_layer(
                input=tensors["moe_w2_out"],
                weight=tensors["topk_weight"],
                residual=tensors["post_attn_residual"],
                output=tensors["hidden_out"],
                grid_dim=(1, 1, 1),
                block_dim=(128, 1, 1),
            )
        else:
            skipped_steps.append(step.name)
            continue

        executed_steps.append(step.name)

    task_graph = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    return {
        "executed_steps": executed_steps,
        "skipped_steps": skipped_steps,
        "generated_json_len": len(task_graph["json_file"]),
        "generated_cuda_len": len(task_graph["cuda_code"]),
    }


def exercise_mirage_task_registration() -> Dict[str, Any]:
    """Register a representative subset of Gemma-relevant Mirage tasks and generate the task graph."""

    pk = create_test_persistent_kernel()

    # Attention / projection path.
    rmsnorm_input = pk.attach_input(
        torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_rmsnorm_input"
    )
    weight_norm = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_weight_norm"
    )
    qkv_weight = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_qkv_weight"
    )
    qkv_out = pk.new_tensor((4, 16), name="bridge_qkv_out")
    pk.rmsnorm_linear_layer(
        input=rmsnorm_input,
        weight_norm=weight_norm,
        weight_linear=qkv_weight,
        output=qkv_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    k_cache = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_k_cache"
    )
    v_cache = pk.attach_input(
        torch.zeros((8, 2, 2, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_v_cache"
    )
    q_norm = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_q_norm"
    )
    k_norm = pk.attach_input(
        torch.ones((8,), dtype=torch.bfloat16, device="cuda"), name="bridge_k_norm"
    )
    cos = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_cos"
    )
    sin = pk.attach_input(
        torch.zeros((16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_sin"
    )
    attn_out = pk.new_tensor((4, 16), name="bridge_attn_out")
    pk.paged_attention_layer(
        input=qkv_out,
        k_cache=k_cache,
        v_cache=v_cache,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_pos_embed=cos,
        sin_pos_embed=sin,
        output=attn_out,
        grid_dim=(2, 2, 1),
        block_dim=(128, 1, 1),
    )

    proj_weight = pk.attach_input(
        torch.zeros((8, 16), dtype=torch.bfloat16, device="cuda"), name="bridge_proj_weight"
    )
    residual = pk.attach_input(
        torch.zeros((4, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_residual"
    )
    proj_out = pk.new_tensor((4, 8), name="bridge_proj_out")
    pk.linear_with_residual_layer(
        input=attn_out,
        weight=proj_weight,
        residual=residual,
        output=proj_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    # Routing + MoE path.
    router_weight = pk.attach_input(
        torch.zeros((8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_router_weight"
    )
    router_logits = pk.new_tensor((4, 8), name="bridge_router_logits")
    pk.linear_layer(
        input=proj_out,
        weight=router_weight,
        output=router_logits,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    topk_weight = pk.attach_input(
        torch.zeros((4, 2), dtype=torch.float32, device="cuda"), name="bridge_topk_weight"
    )
    routing_indices = pk.new_tensor((8, 4), name="bridge_routing_indices")
    routing_mask = pk.new_tensor((9,), name="bridge_routing_mask")
    pk.moe_topk_softmax_routing_layer(
        input=router_logits,
        output=(topk_weight, routing_indices, routing_mask),
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    moe_weight_w13 = pk.attach_input(
        torch.zeros((8, 16, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w13"
    )
    moe_w13_out = pk.new_tensor((4, 2, 16), name="bridge_moe_w13_out")
    pk.moe_w13_linear_layer(
        input=proj_out,
        weight=moe_weight_w13,
        moe_routing_indices=routing_indices,
        moe_mask=routing_mask,
        output=moe_w13_out,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=True),
        block_dim=(128, 1, 1),
    )

    moe_act_out = pk.new_tensor((4, 2, 8), name="bridge_moe_act_out")
    pk.moe_silu_mul_layer(
        input=moe_w13_out,
        output=moe_act_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    moe_weight_w2 = pk.attach_input(
        torch.zeros((8, 8, 8), dtype=torch.bfloat16, device="cuda"), name="bridge_moe_w2"
    )
    moe_w2_out = pk.new_tensor((4, 2, 8), name="bridge_moe_w2_out")
    pk.moe_w2_linear_layer(
        input=moe_act_out,
        weight=moe_weight_w2,
        moe_routing_indices=routing_indices,
        moe_mask=routing_mask,
        output=moe_w2_out,
        grid_dim=_moe_expert_grid_dim(pk, w13_linear=False),
        block_dim=(128, 1, 1),
    )

    moe_final = pk.new_tensor((4, 8), name="bridge_moe_final")
    pk.moe_mul_sum_add_layer(
        input=moe_w2_out,
        weight=topk_weight,
        residual=proj_out,
        output=moe_final,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )

    task_graph = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    return {
        "generated_json_len": len(task_graph["json_file"]),
        "generated_cuda_len": len(task_graph["cuda_code"]),
        "registered_tasks": [
            "rmsnorm_linear_layer",
            "paged_attention_layer",
            "linear_with_residual_layer",
            "linear_layer",
            "moe_topk_softmax_routing_layer",
            "moe_w13_linear_layer",
            "moe_silu_mul_layer",
            "moe_w2_linear_layer",
            "moe_mul_sum_add_layer",
        ],
    }


def compile_supported_rmsnorm_linear_smoke(
    *,
    output_dir: str = "./mirage_compile_supported_smoke",
    launch: bool = True,
) -> Dict[str, Any]:
    """Compile a tiny live Mirage kernel using supported task composition.

    This intentionally avoids Mirage's current ``rmsnorm_linear`` codegen path
    and instead composes ``rmsnorm_layer`` followed by ``linear_layer``.
    """

    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
        target_cc_override=80,
    )
    hidden_in = pk.attach_input(
        torch.zeros((1, 1024), dtype=torch.bfloat16, device="cuda"), name="bridge_compile_hidden_in"
    )
    weight_norm = pk.attach_input(
        torch.ones((1024,), dtype=torch.bfloat16, device="cuda"), name="bridge_compile_weight_norm"
    )
    rmsnorm_out = pk.new_tensor((1, 1024), name="bridge_compile_rmsnorm_out")
    qkv_weight = pk.attach_input(
        torch.zeros((1024, 1024), dtype=torch.bfloat16, device="cuda"),
        name="bridge_compile_qkv_weight",
    )
    qkv_out = pk.new_tensor((1, 1024), name="bridge_compile_qkv_out")

    pk.rmsnorm_layer(
        input=hidden_in,
        weight=weight_norm,
        output=rmsnorm_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_layer(
        input=rmsnorm_out,
        weight=qkv_weight,
        output=qkv_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk, output_dir=output_dir)
    launched = False
    if launch:
        pk()
        launched = True

    return {
        "compiled": pk._is_compiled,
        "launched": launched,
        "output_dir": output_dir,
    }


def compile_supported_attention_block_smoke(
    *,
    output_dir: str = "./mirage_compile_attention_smoke",
    launch: bool = True,
) -> Dict[str, Any]:
    """Compile a tiny live Mirage attention block on supported task grains."""

    pk = create_test_persistent_kernel(
        max_seq_length=128,
        max_num_batched_requests=1,
        max_num_batched_tokens=1,
        max_num_pages=8,
        page_size=64,
        use_cutlass_kernel=False,
        target_cc_override=80,
    )
    qkv_in = pk.attach_input(
        torch.zeros((1, 192), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_qkv_in",
    )
    k_cache = pk.attach_input(
        torch.zeros((8, 64, 1, 64), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_k_cache",
    )
    v_cache = pk.attach_input(
        torch.zeros((8, 64, 1, 64), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_v_cache",
    )
    q_norm = pk.attach_input(
        torch.ones((64,), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_q_norm",
    )
    k_norm = pk.attach_input(
        torch.ones((64,), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_k_norm",
    )
    cos = pk.attach_input(
        torch.zeros((128, 64), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_cos",
    )
    sin = pk.attach_input(
        torch.zeros((128, 64), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_sin",
    )
    attn_out = pk.new_tensor((1, 64), name="bridge_attn_out")
    residual = pk.attach_input(
        torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_residual",
    )
    proj_weight = pk.attach_input(
        torch.zeros((64, 64), dtype=torch.bfloat16, device="cuda"),
        name="bridge_attn_proj_weight",
    )
    block_out = pk.new_tensor((1, 64), name="bridge_attn_block_out")

    pk.paged_attention_layer(
        input=qkv_in,
        k_cache=k_cache,
        v_cache=v_cache,
        q_norm=q_norm,
        k_norm=k_norm,
        cos_pos_embed=cos,
        sin_pos_embed=sin,
        output=attn_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    pk.linear_with_residual_layer(
        input=attn_out,
        weight=proj_weight,
        residual=residual,
        output=block_out,
        grid_dim=(1, 1, 1),
        block_dim=(128, 1, 1),
    )
    compile_persistent_kernel_with_patches(pk, output_dir=output_dir)
    launched = False
    if launch:
        pk()
        launched = True

    return {
        "compiled": pk._is_compiled,
        "launched": launched,
        "output_dir": output_dir,
    }
