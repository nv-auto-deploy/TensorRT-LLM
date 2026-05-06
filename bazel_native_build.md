<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bazel Native Build Seeds

This branch starts native Bazel modeling for TensorRT-LLM C++ and CUDA code.
It is not a CI target-selection change and it does not model wheel packaging,
Triton backend packaging, or the full CMake target graph.

## Scope

- Bzlmod setup through `MODULE.bazel`.
- A local system dependency repository for CUDA.
- Initial `cc_library` and `cuda_library` targets under `//cpp`.
- Host and CUDA smoke checks.

## Useful Targets

```bash
bazel test //cpp:timestamp_utils_smoke_test
bazel test //cpp:common_host_smoke_test
bazel build //cpp:global_timer_kernel_cuda
```

The broader CUDA seed is available as:

```bash
bazel build //cpp:cuda_kernels
```

## Dependency Roots

The local CUDA repository discovers dependencies from these environment
variables when set, then falls back to `/usr/local/cuda`:

- `CUDA_HOME` or `CUDA_PATH`

The repository rule validates required headers and key version macros early so
missing or mismatched local dependencies fail during Bazel analysis rather than
deep inside a compile action.

## External Dependency Adapters

This PR keeps adapters at dependency boundaries. External packages may still be
provided by a system install, git submodule, CMake project, or prebuilt library,
but TensorRT-LLM Bazel targets should depend on a crisp Bazel contract such as a
headers target or shared-library target.

The first adapter is the local CUDA repository. Future adapters for packages
such as TensorRT, NCCL, MPI, CUTLASS, NIXL, or Triton should be added with the
native TensorRT-LLM target that consumes them. They should not wrap the
TensorRT-LLM CMake, wheel, or Triton backend packaging flows as a substitute for
modeling TensorRT-LLM source ownership in Bazel.

## Not Yet Modeled

- Jenkins/test-db parsing and CI impact selection.
- Generated pytest selector targets.
- Wheel packaging.
- Triton backend packaging.
- Full TensorRT-LLM CMake target parity.
