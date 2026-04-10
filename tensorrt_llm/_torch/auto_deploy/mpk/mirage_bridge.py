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

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .types import GemmaLayerLoweringPlan, GemmaLoweringStatus

_COMPOSED_STEP_METHODS: Dict[str, tuple[str, ...]] = {
    "attn_rmsnorm_linear": ("rmsnorm_layer", "linear_layer"),
    "dense_ffn_gate_up": ("rmsnorm_layer", "linear_layer"),
}

_MIRAGE_RUNTIME_EXTENSION_CACHE: Dict[str, Any] = {}

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
    if "mirage" not in sys.modules:
        mirage_pkg = types.ModuleType("mirage")
        mirage_pkg.__path__ = [str(package_dir)]
        sys.modules["mirage"] = mirage_pkg

    if "mirage.mpk" not in sys.modules:
        mirage_mpk_pkg = types.ModuleType("mirage.mpk")
        mirage_mpk_pkg.__path__ = [str(package_dir / "mpk")]
        sys.modules["mirage.mpk"] = mirage_mpk_pkg


def _require_mirage():
    try:
        from mirage.mpk.persistent_kernel import PersistentKernel
    except ImportError as exc:  # pragma: no cover - exercised only in Mirage-enabled envs
        if isinstance(exc, ModuleNotFoundError) and exc.name == "z3":
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
    try:
        persistent_kernel_mod = importlib.import_module("mirage.mpk.persistent_kernel")
        kernel_mod = importlib.import_module("mirage.kernel")
    except ImportError as exc:  # pragma: no cover - exercised only in Mirage-enabled envs
        raise RuntimeError(
            "Mirage compile helpers are not importable. Ensure mirage.mpk.persistent_kernel "
            "and mirage.kernel are available on PYTHONPATH."
        ) from exc
    return persistent_kernel_mod, kernel_mod


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


def patch_generated_mirage_cuda_source(cuda_code: str) -> str:
    """Patch known Mirage-generated CUDA source compatibility gaps.

    Current Mirage task registration can emit calls to ``norm_linear_task_impl``
    on the Ampere runtime path without including the corresponding task header in
    ``task_header.cuh``. Until that is fixed upstream, patch the generated CUDA
    source locally before invoking ``nvcc``.
    """

    if "norm_linear_task_impl" not in cuda_code:
        return cuda_code

    anchor = '#include "persistent_kernel.cuh"\n'
    compat_headers = (
        '#include "persistent_kernel.cuh"\n'
        '#include "tasks/ampere/norm_linear.cuh"\n'
        '#include "tasks/ampere/norm_linear_new.cuh"\n'
    )
    if anchor in cuda_code and "norm_linear_new.cuh" not in cuda_code:
        return cuda_code.replace(anchor, compat_headers, 1)
    return cuda_code


def compile_persistent_kernel_with_patches(
    pk,
    *,
    output_dir: Optional[str] = None,
):
    """Compile a Mirage ``PersistentKernel`` via a bridge-side patched codegen path."""

    persistent_kernel_mod, kernel_mod = _require_mirage_compile_support()
    hard_code = getattr(persistent_kernel_mod, "HARD_CODE")
    get_compile_command = getattr(persistent_kernel_mod, "get_compile_command")
    get_key_paths = getattr(kernel_mod, "get_key_paths")

    if pk._is_compiled:
        return pk

    if pk.mode in {"online_notoken", "online", "multi_turn"}:
        tempdir = "./permanent_output_dir/"
    else:
        tempdir_obj = tempfile.TemporaryDirectory()
        tempdir = tempdir_obj.name

    os.makedirs(tempdir, exist_ok=True)
    results = pk.kn_graph.generate_task_graph(num_gpus=pk.world_size, my_gpu_id=pk.mpi_rank)

    cuda_code = patch_generated_mirage_cuda_source(results["cuda_code"] + hard_code)
    cuda_code_path = os.path.join(tempdir, "test.cu")
    so_path = os.path.join(tempdir, "test" + sysconfig.get_config_var("EXT_SUFFIX"))
    json_file_path = os.path.join(tempdir, "task_graph.json")

    with open(json_file_path, "w", encoding="utf-8") as f:
        f.write(results["json_file"])
    with open(cuda_code_path, "w", encoding="utf-8") as f:
        f.write(cuda_code)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(cuda_code_path, os.path.join(output_dir, f"test_rank{pk.mpi_rank}.cu"))
        shutil.copy(json_file_path, os.path.join(output_dir, f"task_graph_rank{pk.mpi_rank}.json"))

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

    mirage_root, include_path, deps_path = get_key_paths()
    cc_cmd = get_compile_command(
        mpk=pk,
        target_cc=pk.target_cc,
        cc=cc,
        file_name=cuda_code_path,
        py_include_dir=py_include_dir,
        mirage_home_path=os.environ.get("MIRAGE_HOME", mirage_root),
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
    subprocess.check_call(cc_cmd)

    spec = importlib.util.spec_from_file_location("__mirage_launcher", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Mirage launcher module from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pk.init_func = getattr(mod, "init_func")
    pk.launch_func = getattr(mod, "launch_func")
    pk.init_request_func = getattr(mod, "init_request_func")
    pk.finalize_func = getattr(mod, "finalize_func")

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

    input_fp32 = input_tensor.float()
    weight_fp32 = weight.float()
    variance = input_fp32.square().mean(dim=-1, keepdim=True)
    normalized = input_fp32 * torch.rsqrt(variance + eps)
    return normalized * weight_fp32


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
    out = torch.empty_like(linear_input)

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


def build_gemma_mirage_runtime_callable(
    translation_plan: Dict[str, Any],
) -> Callable[..., Any]:
    """Build the live Gemma MPK runtime callable.

    The current implementation is intentionally strict: once the MPK path is
    selected, execution must go through a Mirage-backed callable rather than an
    eager fallback. Until the full-model Mirage emission path exists, this
    callable raises with an explicit summary of the remaining lowering gaps.
    """

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

    def _runtime_callable(*args, **kwargs):
        raise NotImplementedError(
            "Live Mirage execution for the full Gemma MPK path is not implemented yet. "
            f"Current plan has {num_gap_steps} gap steps and {num_partial_steps} partial steps."
        )

    return _runtime_callable


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
    pk = PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=1,
        num_local_schedulers=1,
        num_remote_schedulers=0,
        max_seq_length=max_seq_length,
        max_num_batched_requests=max_num_batched_requests,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_pages=max_num_pages,
        page_size=page_size,
        meta_tensors=meta_tensors,
        profiler_tensor=profiler_tensor,
        trace_name="gemma_mpk_bridge_smoke",
        spec_decode_config=None,
        use_cutlass_kernel=use_cutlass_kernel,
    )
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
    registry["topk_weight"] = pk.new_tensor((4, 2), name="bridge_topk_weight")
    registry["routing_indices"] = pk.new_tensor((2, 4), name="bridge_routing_indices")
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
                grid_dim=(1, 1, 1),
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
                grid_dim=(1, 1, 1),
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

    topk_weight = pk.new_tensor((4, 2), name="bridge_topk_weight")
    routing_indices = pk.new_tensor((2, 4), name="bridge_routing_indices")
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
        grid_dim=(1, 1, 1),
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
        grid_dim=(1, 1, 1),
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
