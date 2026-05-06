<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI Target Graph

This package generates shadow-mode Bazel-style metadata for TensorRT-LLM CI.
The repository now has a narrow Bazel spike with root Bazel files, platform
constraints, BUILD packages, and generated pytest selector targets. This
package remains the metadata source for that spike: it makes the current
Jenkins/test-db pytest selectors visible as explicit JSON targets so
impact-selection ideas can be compared against the existing CI without changing
what Jenkins runs.

## Generate

From the repository root:

```bash
python3 -m scripts.ci_target_graph.generate \
  --repo-root . \
  --output /tmp/trtllm-target-manifest.json
```

If `--output` is omitted, the JSON manifest is written to stdout:

```bash
python3 -m scripts.ci_target_graph.generate --repo-root .
```

The manifest uses `schema_version: 2`; its practical JSON Schema lives in
`manifest_schema_v2.json`.

## Validate

Regenerate the manifest and validate it against the checked-in schema:

```bash
python3 -m scripts.ci_target_graph.validate \
  --repo-root . \
  --output /tmp/trtllm-ci-target-graph.json
```

This command does not run GPU tests. It reports schema errors as failures and
prints a conservative warning for manifest targets with no Jenkins stage
candidate. A missing candidate means the YAML selector could not be matched to
an active Jenkins L0 stage config before pytest shard assignment; impact
selection should treat those entries as fallback inputs until the Jenkins
mapping is modeled more completely.

Use `--fail-on-missing-jenkins-stage` when a CI job wants to make those fallback
entries blocking.

Use `--fail-on-incomplete-runtime-metadata` when a CI job wants unknown or
ambiguous runtime metadata to become blocking instead of warning-only.

## Select Impacted Bazel Targets

Select Bazel pytest targets affected by a source diff:

```bash
python3 scripts/ci_target_graph/select_impacted.py \
  --base upstream/main \
  --platform //platforms:h100_4gpu
```

The selector reads `git diff --name-status -z --find-renames` by default. It
can also be driven without git, which is useful for shadow-mode CI plumbing and
unit tests:

```bash
python3 scripts/ci_target_graph/select_impacted.py \
  --base upstream/main \
  --head HEAD \
  --changed-file tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py \
  --platform //platforms:h100_4gpu \
  --bazel-binary /tmp/codex-bazelisk \
  --output both \
  --json-output /tmp/trtllm-impacted-targets.json \
  --targets-output /tmp/trtllm-impacted-targets.txt
```

Modeled source ownership is intentionally narrow:

- `tensorrt_llm/_torch/auto_deploy/**/*.py` maps to
  `//tensorrt_llm/_torch/auto_deploy:runtime`.
- `tests/integration/defs/accuracy/*.py` maps to
  `//tests/integration/defs/accuracy:accuracy_tests`.
- `scripts/ci_target_graph/*.py` maps to
  `//scripts/ci_target_graph:ci_target_graph_lib`.
- `cpp/tensorrt_llm/nanobind/**` maps to
  `//cpp:tensorrt_llm_bindings`.
- `cpp/tensorrt_llm/kernels/**` maps to
  `//cpp:cuda_kernels`.
- `cpp/tensorrt_llm/plugins/**` maps to
  `//cpp:nvinfer_plugin_tensorrt_llm`.
- `triton_backend/inflight_batcher_llm/**` maps to
  `//triton_backend:triton_tensorrt_llm_backend`.

Unknown paths, invalid paths, Bazel query failures, unmodeled native/build
areas, and build or CI policy changes use a conservative broad fallback over
`//ci/bazel/...`. Broad fallback is also used for Bazel metadata, dependency
lock files, Jenkins policy, GitHub policy, platform definitions, and test-db or
waive-list changes.

The JSON output has `schema_version: 1` and records the diff refs, changed-file
evidence, fallback reasons, owner labels, candidate targets, selected targets,
smoke targets, and warnings. Manual targets are included by default so current
manual GPU pytest selectors are not silently dropped; use
`--manual-policy exclude` or `--manual-policy only` when a caller needs a
different policy.

## Native Label Status

The native/package labels modeled by AutoDeploy Bazel deps are intentionally
uneven in build depth. AutoDeploy seed and generated targets use metadata/query
edges for CI impact selection; expensive manual/local wrapper targets are kept
out of those deps.

- `//:tensorrt_llm_wheel_metadata` is the wheel-related query and CI
  impact-selection edge. It is a metadata-only filegroup over wheel inputs and
  `//cpp:wheel_native_artifacts_metadata`.
- `//:tensorrt_llm_wheel` is the manual/local wrapper around the existing wheel
  flow. It is intentionally tagged local, non-hermetic, and expensive, and
  should not be used as an AutoDeploy CI impact-selection dependency.
- `//triton_backend:triton_tensorrt_llm_backend` is a source-coverage
  `filegroup` over the checked-in Triton backend CMake, script, client, C++
  source, linker script, test, and fixture inputs. It can be built and queried
  by Bazel as file metadata, but it does not produce `libtriton_tensorrtllm.so`.
- `//cpp:cuda_kernels` is a narrow real CUDA seed through the
  `global_timer_kernel_cuda` target. Treat it as one CUDA build proof, not
  complete CUDA or TensorRT-LLM native parity.
- `//cpp:cuda_kernels_metadata`, `//cpp:nvinfer_plugin_tensorrt_llm_metadata`,
  and `//cpp:tensorrt_llm_bindings_metadata` are metadata-only filegroups used
  by AutoDeploy seed and generated Bazel deps.
- Full `//cpp:tensorrt_llm_bindings` remains blocked until Bazel models the
  nanobind module stack: nanobind, Python libraries, Torch and `torch_python`,
  CUDA driver libraries, the shared TensorRT-LLM library target, `th_common`,
  `pg_utils`, and optional NVSHMEM or transfer-agent pieces.
- The full Triton backend shared library remains blocked until Bazel models the
  dependencies visible in `triton_backend/inflight_batcher_llm`: Triton
  common/core/backend repositories, CUDA Toolkit and runtime libraries, cuDNN,
  cuBLAS/cuBLASLt, CUDA driver/NVML libraries, MPI, NCCL, TensorRT 10,
  TensorRT-LLM core and plugin shared libraries, `executorWorker`,
  nlohmann/json, and GoogleTest for backend tests.

Structured runtime filters narrow selected Bazel targets using manifest runtime
metadata instead of raw tag regexes:

```bash
python3 scripts/ci_target_graph/select_impacted.py \
  --base upstream/main \
  --platform //platforms:h100_1gpu \
  --model-family llama \
  --backend autodeploy

python3 scripts/ci_target_graph/select_impacted.py \
  --base upstream/main \
  --runtime-requirement triton
```

These flags map to `model:<value>`, `backend:<value>`, and
`requires:<value>` tags. They first check candidate targets for incomplete or
missing runtime metadata and use conservative broad fallback if the metadata is
not complete. Use raw `--include-tag`, such as `--include-tag 'model:llama'`,
as the escape hatch when direct tag regex filtering is wanted.

## Inputs

The generator reads:

- `jenkins/L0_Test.groovy` for active stage-to-test-db mappings.
- `tests/integration/test_lists/test-db/*.yml`
- `tests/integration/test_lists/test-db/*.yaml`

Each emitted target has `kind: "pytest_selector"` and records the raw selector,
parsed pytest path, all parsed pytest paths, remaining pytest arguments,
timeout minutes, isolation marker, test-db context, matching Jenkins stage
candidates, test-db constraints, runtime metadata, tags, and conservative
component hints.

`source.jenkins_stages` is intentionally scoped as `pre_shard_candidates`.
It records Jenkins stage configs that match the YAML entry before pytest shard
assignment, so it is not exact per-shard membership.

## Conservative By Design

This is metadata extraction, not dependency inference. Component hints are path
prefixes such as `tests/integration/defs/accuracy` or `tests/unittest/_torch`.
When a selector has multiple pytest paths, hints are the union of those path
prefixes. They are useful for shadow analysis, but they are not exact
dependency edges.

The parser also keeps unfamiliar pytest arguments rather than rejecting them.
It detects the local `TIMEOUT (N)` convention as minutes, detects `ISOLATION`,
and leaves unusual selectors in the manifest instead of failing generation.

The Bazel spike consumes this manifest for a narrow generated AutoDeploy H100
subpackage. Native/package labels provide reverse-dependency selection edges,
with `//:tensorrt_llm_wheel_metadata` serving as the wheel-related CI edge and
`//:tensorrt_llm_wheel` remaining a manual/local wrapper. C++ CI edges use
metadata-only labels such as `//cpp:cuda_kernels_metadata`; explicit native
build labels such as `//cpp:cuda_kernels` remain opt-in. Most labels are still
metadata or narrow local seeds rather than production native builds. The
separate `//cpp:timestamp_utils_smoke_test` target is the first narrow host C++
Bazel build proof, not CUDA, TensorRT, native extension, or wheel build parity.
Most Python, C++, CUDA, model/data inputs, and generated artifacts are still
intentionally unmodeled. Unknown or unmodeled areas should fall back
conservatively rather than being skipped.

## Runtime Metadata

Schema v2 adds an explicit per-target `runtime` object:

```json
{
  "model_families": ["llama"],
  "backend": "autodeploy",
  "gpu_types": ["h100"],
  "gpu_count": 1,
  "requirements": ["cuda", "model_cache"],
  "metadata_complete": true,
  "missing": []
}
```

Runtime tags are query-visible and derived from that object:

- `model:<family>` or `model:unknown`
- `backend:<name>`
- `gpu:<type>` and `gpu_count:<n>`
- `requires:<requirement>`
- `metadata:runtime_complete` or `metadata:runtime_incomplete`

Model-family inference is conservative. Clear selector evidence covers Llama,
Nemotron, GLM, Gemma, Qwen, Mistral, and DeepSeek. Unknown selectors keep
`model:unknown`; selectors with multiple family signals keep the discovered
families but are marked `metadata:runtime_incomplete`.

Examples:

```bash
jq '.targets[] | select(.runtime.metadata_complete) | .target_id' \
  /tmp/trtllm-ci-target-graph.json

jq '.targets[] | select(.tags[]? == "requires:triton") | .target_id' \
  /tmp/trtllm-ci-target-graph.json
```

Bazel reverse-dependency queries can inspect generated selectors that depend on
the Phase 6 metadata/query labels:

```bash
bazel query 'rdeps(//ci/bazel/..., //:tensorrt_llm_wheel_metadata)'
bazel query 'rdeps(//ci/bazel/..., //cpp:tensorrt_llm_bindings_metadata)'
bazel query 'rdeps(//ci/bazel/..., //cpp:cuda_kernels_metadata)'
bazel query 'rdeps(//ci/bazel/..., //triton_backend:triton_tensorrt_llm_backend)'
```

Use `//:tensorrt_llm_wheel_metadata` for CI reverse-dependency queries.
Use `//cpp:cuda_kernels_metadata` for CI reverse-dependency queries and
`//cpp:cuda_kernels` for explicit CUDA build proof runs. `//:tensorrt_llm_wheel`
is for manual/local wheel runs.
