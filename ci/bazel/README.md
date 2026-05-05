<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TensorRT-LLM Bazel Spike

This directory contains the current Bazel spike for TensorRT-LLM CI impact selection and pytest selector
execution. It is not a full Bazel migration of TensorRT-LLM C++, CUDA, or Python builds, and it does not
replace Jenkins.

The implemented scope is:

- Generate a shadow-mode CI target graph manifest from Jenkins L0 stage metadata and test-db YAML files.
- Run selected pytest selectors through Bazel test targets.
- Seed manual AutoDeploy H100 one-GPU and four-GPU targets for validation.

## Current Layout

Repository-level Bazel setup:

- [`.bazelversion`](../../.bazelversion): Bazel version used by Bazelisk.
- [`.bazelrc`](../../.bazelrc): local output roots, cache paths, Bzlmod, and default test output settings.
- [`MODULE.bazel`](../../MODULE.bazel): module dependencies and Python 3.12/pip setup.
- [`MODULE.bazel.lock`](../../MODULE.bazel.lock): Bzlmod dependency lock file. Keep this committed with
  `MODULE.bazel` so module resolution remains reproducible.
- [`requirements_bazel_lock.txt`](../../requirements_bazel_lock.txt): lock file used by `rules_python`.

CI target graph:

- [`scripts/ci_target_graph/BUILD.bazel`](../../scripts/ci_target_graph/BUILD.bazel):
  - `//scripts/ci_target_graph:ci_target_graph_lib`
  - `//scripts/ci_target_graph:generate`
- [`tests/unittest/tools/BUILD.bazel`](../../tests/unittest/tools/BUILD.bazel):
  - `//tests/unittest/tools:ci_target_graph_test`

Pytest selector harness:

- [`ci/bazel/defs.bzl`](defs.bzl): defines `trtllm_pytest_case`.
- [`ci/bazel/pytest_selector_test.sh`](pytest_selector_test.sh): shell wrapper for Bazel `sh_test` targets.
- [`tools/bazel/pytest_selector_runner.py`](../../tools/bazel/pytest_selector_runner.py): resolves the repo root,
  normalizes selectors, and invokes pytest.
- [`ci/bazel/BUILD.bazel`](BUILD.bazel):
  - `//ci/bazel:autodeploy_h100_seed`
  - `//ci/bazel:autodeploy_h100_generated`
- [`ci/bazel/autodeploy/generated/BUILD.bazel`](autodeploy/generated/BUILD.bazel):
  generated pre-merge AutoDeploy H100 targets from the manifest.
- [`ci/bazel/autodeploy/BUILD.bazel`](autodeploy/BUILD.bazel):
  - `//ci/bazel/autodeploy:autodeploy_h100_seed`
  - `//ci/bazel/autodeploy:autodeploy_h100_generated`
  - `//ci/bazel/autodeploy:autodeploy_h100_4gpu`
  - AutoDeploy leaf seed tests such as `:llama31_trtllm_1gpu_h100`.

Platforms and constraints:

- [`platforms/BUILD.bazel`](../../platforms/BUILD.bazel): concrete host, H100, and B200 platform labels.
- [`platforms/gpu/BUILD.bazel`](../../platforms/gpu/BUILD.bazel): GPU type constraints.
- [`platforms/gpu_count/BUILD.bazel`](../../platforms/gpu_count/BUILD.bazel): one- and four-GPU constraints.
- [`platforms/backend/BUILD.bazel`](../../platforms/backend/BUILD.bazel): backend constraint values.

H100 one-GPU and four-GPU AutoDeploy compatibility has been validated for this spike. Do not infer B200
validation from the presence of B200 platform labels.

## Setup

Install Bazelisk, then run Bazel from the repository root:

```bash
bazelisk version
bazelisk test //tests/unittest/tools:ci_target_graph_test
```

Bazelisk reads [`.bazelversion`](../../.bazelversion), currently `8.6.0`. If your environment exposes the
binary as `bazel`, the same commands work with `bazel` instead of `bazelisk`.

## Quick Validation

These commands do not run GPU pytest workloads:

```bash
python3 -m scripts.ci_target_graph.validate --repo-root . --output /tmp/trtllm-ci-target-graph.json
bazel test //tests/unittest/tools:ci_target_graph_test
bazel build //scripts/ci_target_graph:generate
bazel query '//ci/bazel/...'
```

## Generate The Manifest

Write the shadow CI target graph manifest to a temporary file:

```bash
bazel run //scripts/ci_target_graph:generate -- \
    --repo-root "$PWD" \
    --output /tmp/trtllm-ci-target-graph.json
```

Omit `--output` to print the JSON manifest to stdout.

## Generate AutoDeploy Bazel Targets

Regenerate the checked-in AutoDeploy H100 BUILD package from the manifest:

```bash
python3 -m scripts.ci_target_graph.generate_bazel_autodeploy \
    --repo-root . \
    --output ci/bazel/autodeploy/generated/BUILD.bazel
```

The generated package is intentionally narrow: pre-merge AutoDeploy entries
from `l0_h100.yml` and `l0_dgx_h100.yml` whose GPU constraint matches H100.
Generated targets preserve manifest tags, pytest arguments, isolation and
timeout metadata, and hardware compatibility constraints. Runtime details such
as model family, backend, CUDA/model-cache needs, and Triton runtime evidence
remain query-visible tags instead of Bazel platform constraints.

## Select Impacted Targets

Run the Phase 4 changed-file impact selector from the repository root:

```bash
python3 scripts/ci_target_graph/select_impacted.py \
    --base upstream/main \
    --platform //platforms:h100_4gpu
```

The selector maps changed files in currently modeled source areas to Bazel owner
labels, queries reverse-dependent pytest targets under `//ci/bazel/...`, applies
optional manual/tag filtering, then cquery-filters the candidates for the
requested platform. Unknown inputs and CI/build/dependency policy changes use a
conservative broad fallback rather than skipping tests.

For CI sidecars, write both machine-readable outputs:

```bash
python3 scripts/ci_target_graph/select_impacted.py \
    --base upstream/main \
    --platform //platforms:h100_4gpu \
    --json-output /tmp/trtllm-impacted-targets.json \
    --targets-output /tmp/trtllm-impacted-targets.txt
```

Add structured runtime filters when a caller wants metadata-guarded tag
selection:

```bash
python3 scripts/ci_target_graph/select_impacted.py \
    --base upstream/main \
    --platform //platforms:h100_4gpu \
    --model-family llama \
    --backend autodeploy \
    --runtime-requirement triton
```

`--model-family`, `--backend`, and `--runtime-requirement` fall back
conservatively if candidate targets have incomplete or missing runtime metadata.
Use raw `--include-tag`, such as `--include-tag 'model:llama'`, as the escape
hatch for direct tag regex filtering.

## Query Examples

List Bazel targets under the spike packages:

```bash
bazel query '//ci/bazel/...'
```

List pytest selector test targets:

```bash
bazel query 'kind("sh_test rule", //ci/bazel/...)'
```

List generated Llama AutoDeploy selectors:

```bash
bazel query 'attr("tags", "model:llama", attr("tags", "backend:autodeploy", //ci/bazel/...))'
```

List selectors with incomplete runtime metadata:

```bash
bazel query 'attr("tags", "metadata:runtime_incomplete", //ci/bazel/...)'
```

List selectors with clear Triton runtime evidence:

```bash
bazel query 'attr("tags", "requires:triton", //ci/bazel/...)'
```

Inspect compatibility under an H100 one-GPU target platform:

```bash
bazel cquery //ci/bazel/autodeploy:all --platforms=//platforms:h100_1gpu
```

Use `//platforms:h100_4gpu` when inspecting four-GPU targets.

## Dry-Run A Seed Target

GPU seed targets are tagged `manual`, so wildcard test patterns intentionally skip them. Use explicit leaf
or suite labels.

Dry-run an H100 one-GPU leaf target:

```bash
bazel test //ci/bazel/autodeploy:llama31_trtllm_1gpu_h100 \
    --platforms=//platforms:h100_1gpu \
    --test_arg=--dry-run \
    --test_env=TRTLLM_BAZEL_REPO_ROOT="$PWD" \
    --test_output=streamed \
    --cache_test_results=no
```

Dry-run the four-GPU suite:

```bash
bazel test //ci/bazel/autodeploy:autodeploy_h100_4gpu \
    --platforms=//platforms:h100_4gpu \
    --test_arg=--dry-run \
    --test_env=TRTLLM_BAZEL_REPO_ROOT="$PWD" \
    --test_output=streamed \
    --cache_test_results=no
```

The top-level seed suite `//ci/bazel:autodeploy_h100_seed` includes the AutoDeploy seed suite. Run it
explicitly; do not expect `bazel test //ci/bazel/...` to execute manual GPU targets.

## Real GPU Execution

Real pytest execution requires a prepared TensorRT-LLM environment outside Bazel. At minimum, provide:

- `TRTLLM_BAZEL_REPO_ROOT`: absolute path to this checkout.
- `LLM_MODELS_ROOT`: model cache root required by the selected integration tests.
- `CUDA_VISIBLE_DEVICES` or an equivalent GPU allocation matching the target, such as one H100 for
  `//platforms:h100_1gpu` or four H100s for `//platforms:h100_4gpu`.

Example shape for a one-GPU run:

```bash
bazel test //ci/bazel/autodeploy:llama31_trtllm_1gpu_h100 \
    --platforms=//platforms:h100_1gpu \
    --test_env=TRTLLM_BAZEL_REPO_ROOT="$PWD" \
    --test_env=LLM_MODELS_ROOT="$LLM_MODELS_ROOT" \
    --test_env=CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    --test_output=streamed \
    --cache_test_results=no
```

## Add A Pytest Selector

Add new selector targets with `trtllm_pytest_case` in the relevant package, for example:

```starlark
load("//ci/bazel:defs.bzl", "trtllm_pytest_case")

trtllm_pytest_case(
    name = "my_autodeploy_case_h100",
    selector = "accuracy/test_llm_api_autodeploy.py::TestClass::test_case[params]",
    tags = [
        "backend:autodeploy",
        "gpu:h100",
        "gpu_count:1",
        "model:llama",
        "requires:cuda",
        "requires:model_cache",
        "metadata:runtime_complete",
    ],
    target_compatible_with = [
        "//platforms/gpu:h100",
        "//platforms/gpu_count:one",
    ],
)
```

Selectors relative to `tests/integration/test_lists/test-db` entries can omit the
`tests/integration/defs/` prefix. The runner normalizes such selectors to `tests/integration/defs/...`
when that path exists in the checkout.

By default, `trtllm_pytest_case` adds the `manual` tag. Keep GPU and model-cache tests manual unless there is
a deliberate reason for wildcard Bazel test patterns to include them.

## Known Limitations And Non-Goals

- This is a CI/impact-selection spike, not a production Bazel build for TensorRT-LLM.
- Bazel does not build TensorRT-LLM C++, CUDA, wheels, or Python packages in this spike.
- Bazel does not provision GPUs, model weights, CUDA libraries, or the Python runtime environment needed by
  real integration tests.
- The manifest generator is shadow-mode metadata. Jenkins remains the CI executor.
- Manual GPU targets must be invoked by explicit label; wildcard Bazel test commands skip them by design.
- H100 one-GPU and four-GPU AutoDeploy seed compatibility is the validated path. B200 is represented as a
  platform constraint only; do not claim B200 test validation from this spike.
