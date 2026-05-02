<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Bazel Implementation Test Plan

This plan tests the recommended hybrid Bazel direction for TensorRT-LLM CI:

```text
Keep Jenkins/Slurm execution
+ add explicit dependency/test metadata
+ validate Bazel-style selection in shadow mode
+ gradually move from generated metadata to real Bazel targets
```

The goal is not to Bazel-build the whole TensorRT-LLM repository immediately. The goal is to prove,
with actual Bazel installed, that Bazel can model the selected CI target graph accurately enough to
support impact selection.

## Current Baseline

The repo now has a pure-Python shadow target graph generator:

- `scripts/ci_target_graph/generate.py`
- `scripts/ci_target_graph/selector_parser.py`
- `scripts/ci_target_graph/manifest_schema_v1.json`
- `tests/unittest/tools/test_ci_target_graph.py`

The generator reads:

- `jenkins/L0_Test.groovy`
- `tests/integration/test_lists/test-db/*.yml`
- `tests/integration/test_lists/test-db/*.yaml`

It emits a JSON manifest with explicit `pytest_selector` targets, constraints, tags, conservative
component hints, and `source.jenkins_stage_scope = "pre_shard_candidates"`.

Known good local validation before introducing Bazel:

```bash
python3 -m py_compile \
  scripts/ci_target_graph/__init__.py \
  scripts/ci_target_graph/selector_parser.py \
  scripts/ci_target_graph/generate.py \
  tests/unittest/tools/test_ci_target_graph.py

pytest -q -p no:cacheprovider \
  tests/unittest/tools/test_ci_target_graph.py \
  tests/unittest/tools/test_test_to_stage_mapping.py

python3 -m scripts.ci_target_graph.generate \
  --repo-root . \
  --output /tmp/trtllm-target-manifest-final.json

python3 scripts/check_test_list.py --validate
```

## Phase 1: Install Bazel Reproducibly

Use Bazelisk rather than a bare Bazel binary. Bazelisk is the recommended Bazel installer and can
honor a checked-in `.bazelversion`, which keeps developer and CI behavior aligned.

Install into `/tmp` first so the experiment does not alter the rest of the environment:

```bash
mkdir -p /tmp/trtllm-bazel/bin
cd /tmp/trtllm-bazel/bin

# Option A: download Bazelisk release binary for linux-amd64.
# Option B: use npm if available: npm install -g @bazel/bazelisk
# Option C: use Go if available: go install github.com/bazelbuild/bazelisk@latest

ln -sf /tmp/trtllm-bazel/bin/bazelisk /tmp/trtllm-bazel/bin/bazel
export PATH=/tmp/trtllm-bazel/bin:$PATH

bazel --version
```

Acceptance:

- `bazel --version` succeeds.
- The Bazel binary is Bazelisk-managed.
- No global system Bazel installation is required.

## Phase 2: Keep Bazel Outputs Off Lustre

Use `/tmp` for Bazel output and cache roots. Bazel documentation recommends avoiding NFS-like
network filesystems for output roots because latency hurts builds.

Suggested local `.bazelrc` entries for the spike:

```text
startup --output_user_root=/tmp/trtllm-bazel/output_user_root

common --enable_bzlmod
common --repository_cache=/tmp/trtllm-bazel/repository_cache

build --disk_cache=/tmp/trtllm-bazel/disk_cache
build --guard_against_concurrent_changes

test --test_output=errors
test --cache_test_results=yes
```

Acceptance:

- `bazel info output_base` resolves under `/tmp/trtllm-bazel`.
- Repeated `bazel test` runs reuse local cache where possible.

## Phase 3: Add Minimal Bzlmod Scaffolding

Use Bzlmod with `MODULE.bazel`, not legacy `WORKSPACE`. Current Bazel external dependency
documentation describes `MODULE.bazel` as the root module entry point.

Minimal files to add in a Bazel spike branch:

```text
.bazelversion
.bazelrc
BUILD.bazel
MODULE.bazel
requirements_bazel_lock.txt
ci/bazel/BUILD.bazel
ci/bazel/defs.bzl
ci/bazel/pytest_selector_test.sh
ci/bazel/autodeploy/BUILD.bazel
scripts/ci_target_graph/BUILD.bazel
tests/unittest/tools/BUILD.bazel
tools/bazel/BUILD.bazel
tools/bazel/pytest_selector_runner.py
```

Initial Bazel dependencies:

- `rules_python`
- `platforms`
- a small Bazel-only Python lock containing `pytest`, its generated-repo transitive pins, and
  `PyYAML`

Keep this dependency set separate from TensorRT-LLM runtime dependencies. This first Bazel layer is
for testing the CI graph tooling, not for building the full project.

Acceptance:

```bash
bazel mod graph
bazel query //scripts/ci_target_graph:all
bazel query //tests/unittest/tools:all
```

## Phase 4: Bazel-Test the Shadow Graph Package

Create first-class Bazel targets for only the new tooling:

```text
//scripts/ci_target_graph:ci_target_graph_lib
//scripts/ci_target_graph:generate
//tests/unittest/tools:ci_target_graph_test
```

These are the currently implemented graph-tooling targets in the spike branch.

The Bazel test should run the same unit test that direct pytest runs today:

```bash
bazel test //tests/unittest/tools:ci_target_graph_test
```

Then verify the Bazel-run generator:

```bash
bazel run //scripts/ci_target_graph:generate -- \
  --repo-root "$PWD" \
  --output /tmp/trtllm-bazel-manifest.json
```

Compare with direct Python output:

```bash
python3 -m scripts.ci_target_graph.generate \
  --repo-root . \
  --output /tmp/trtllm-python-manifest.json

diff -u /tmp/trtllm-python-manifest.json /tmp/trtllm-bazel-manifest.json
```

Acceptance:

- Bazel test passes.
- Bazel-run generator produces byte-identical JSON to direct Python generation.
- Manifest still has the expected AutoDeploy DGX H100 and multi-path selector behavior.

## Phase 5: Define Platform and Constraint Vocabulary

Add a small platform model for CI scheduling facts:

```text
//platforms/gpu:h100
//platforms/gpu:b200
//platforms/gpu:none

//platforms/gpu_count:one
//platforms/gpu_count:four

//platforms/backend:pytorch
//platforms/backend:autodeploy
//platforms/backend:tensorrt
//platforms/backend:triton
//platforms/backend:cpp
```

Example platform labels:

```text
//platforms:h100_1gpu
//platforms:h100_4gpu
//platforms:b200_1gpu
//platforms:b200_4gpu
//platforms:host_cpu
```

The top-level platform labels also carry the standard `@platforms//os:linux` and
`@platforms//cpu:x86_64` constraints so Python toolchains can resolve under custom CI platforms.

Use `target_compatible_with` for hard platform compatibility and tags for CI policy:

```text
tags = [
  "backend:autodeploy",
  "gpu:h100",
  "gpu_count:4",
  "requires:cuda",
  "requires:model_cache",
  "manual",
]
```

Use `manual` on GPU tests so wildcard commands such as `bazel test //...` do not accidentally run
scarce GPU jobs.

Acceptance:

```bash
bazel query 'kind(rule, //platforms:*)'
bazel cquery 'kind(".*_test rule", //ci/bazel/...)' --platforms=//platforms:h100_4gpu
```

## Phase 6: Generate Bazel Test Targets From Manifest Seed

Create a Starlark macro that wraps pytest selectors:

```python
trtllm_pytest_case(
    name = "autodeploy_llama31_trtllm_no_cp_4gpu",
    selector = "accuracy/test_llm_api_autodeploy.py::TestLlama3_1_8B::test_auto_dtype[trtllm-False-4]",
    tags = [
        "backend:autodeploy",
        "gpu:h100",
        "gpu_count:4",
        "requires:cuda",
        "requires:model_cache",
        "manual",
    ],
    target_compatible_with = [
        "//platforms/gpu:h100",
        "//platforms/gpu_count:four",
    ],
)
```

Start with a generated BUILD file for the AutoDeploy Llama 3.1 H100 seed:

```text
//ci/bazel/autodeploy:llama31_trtllm_1gpu_h100
//ci/bazel/autodeploy:llama31_trtllm_4gpu_h100
//ci/bazel/autodeploy:llama31_attention_dp_4gpu_h100
```

The spike also includes seed suites:

```text
//ci/bazel:autodeploy_h100_seed
//ci/bazel/autodeploy:autodeploy_h100_seed
//ci/bazel/autodeploy:autodeploy_h100_4gpu
```

Execution wrapper behavior:

- Print the resolved repository root and pytest command in dry-run mode.
- In real mode, call pytest with the normalized selector and repeated `--pytest-arg` values.
- Require explicit environment for GPU/model tests.

Acceptance:

```bash
bazel query 'tests(//ci/bazel/autodeploy:all)'
bazel test --test_tag_filters=manual //ci/bazel/autodeploy:llama31_trtllm_1gpu_h100 --test_output=streamed
```

GPU test acceptance requires a prepared environment:

```bash
export LLM_MODELS_ROOT=/path/to/models
export CUDA_VISIBLE_DEVICES=0
```

For 4-GPU tests:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Phase 7: Query-Based Impact Selection Tests

Prove Bazel can answer the selection questions that Jenkins currently cannot.

Basic target inventory:

```bash
bazel query 'kind(".*_test rule", //ci/bazel/...)'
bazel query 'tests(//ci/bazel/autodeploy:all)'
```

Reverse dependency checks:

```bash
bazel query 'rdeps(//ci/bazel/..., //scripts/ci_target_graph:ci_target_graph_lib)'
```

Configured/platform-aware checks:

```bash
bazel cquery 'kind(".*_test rule", //ci/bazel/...)' \
  --platforms=//platforms:h100_4gpu

bazel cquery 'kind(".*_test rule", //ci/bazel/...)' \
  --platforms=//platforms:h100_1gpu
```

Impact-selection tag queries:

```bash
bazel query 'attr(tags, "backend:autodeploy", attr(tags, "gpu:h100", //ci/bazel/...))'

bazel cquery 'attr(tags, "gpu_count:4", kind(".*_test rule", //ci/bazel/...))' \
  --platforms=//platforms:h100_4gpu
```

Execution should use explicit target or suite labels from the selected set. Do not rely on wildcard
`bazel test //ci/bazel/...` execution for manual GPU targets; `manual` intentionally excludes them
from wildcard expansion.

Acceptance:

- H100 4-GPU platform selects H100 4-GPU compatible targets.
- H100 1-GPU platform does not select 4-GPU-only targets.
- `auto_trigger:others` targets do not select DeepSeek/GptOss-specific stages.
- Multi-path selectors contribute all component hints to generated metadata.

## Phase 8: Cache and Incrementality Tests

Local disk cache:

```bash
bazel clean
time bazel test //tests/unittest/tools:ci_target_graph_test
time bazel test //tests/unittest/tools:ci_target_graph_test
```

Expected:

- First run downloads/analyzes/builds.
- Second run is faster and reuses cached work.

Profile:

```bash
bazel test //tests/unittest/tools:ci_target_graph_test \
  --profile=/tmp/trtllm-bazel/profile.json

bazel analyze-profile /tmp/trtllm-bazel/profile.json
```

Remote cache comes later. A remote cache should be tested only after local actions are known to be
hermetic enough not to poison shared cache state.

## Implemented Validation As Of This Spike

The current spike has passed these non-GPU validation checks:

- `/tmp/trtllm-bazel/bin/bazelisk --version` reports Bazel 8.6.0.
- `/tmp/trtllm-bazel/bin/bazelisk test //tests/unittest/tools:ci_target_graph_test` passes.
- Bazel-run `//scripts/ci_target_graph:generate` and direct `python3 -m scripts.ci_target_graph.generate`
  produce byte-identical manifests under `diff -u`.
- Manual-tag query confirms the AutoDeploy seed suites and leaf seed tests carry `manual`.
- Seed dry-run normalizes test-db-relative selectors to `tests/integration/defs/...` without running
  GPU pytest.
- H100 one-/four-GPU compatibility checks pass: H100 4-GPU selects compatible seed targets, and
  H100 1-GPU rejects H100 4-GPU-only targets.
- `python3 scripts/check_test_list.py --validate` passes.

## Phase 9: Shadow-Mode CI Comparison

Add a non-blocking Jenkins sidecar step:

```text
1. Generate Bazel manifest.
2. Generate Bazel selected target list for the PR diff.
3. Record selected stages/tests as artifacts.
4. Run existing Jenkins stages unchanged.
5. Compare failures against Bazel-predicted impacted set.
```

Metrics:

- selected target count
- selected Jenkins stage candidates
- skipped target count
- failures inside selected set
- failures outside selected set
- false-negative rate
- queue-time savings estimate
- flake rerun reduction estimate

Acceptance before enforcement:

- At least several weeks of shadow data.
- No unexplained false negatives in areas covered by the modeled target graph.
- Clear fallback behavior for unknown metadata.
- Manual override remains available through existing `/bot run` controls.

## Non-Goals for the First Bazel Test

Do not attempt these in the first pass:

- Full TensorRT-LLM C++/CUDA Bazel migration.
- Full Python import graph inference.
- Replacing `trt-test-db`.
- Replacing Jenkins, Slurm, or Kubernetes execution.
- Enforcing Bazel-selected tests as the only required CI gate.
- Remote execution.
- Shared remote cache writes.

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Bazel target graph diverges from Jenkins/test-db semantics | Keep shadow-mode comparison and preserve existing CI as source of truth. |
| GPU tests run accidentally under wildcard patterns | Use `manual` tags and explicit target invocation. |
| Model/cache paths are non-hermetic | Treat model paths as declared external runtime requirements, not cacheable Bazel inputs. |
| Split stages are mistaken for exact shard membership | Keep `jenkins_stage_scope = "pre_shard_candidates"` until shard-aware modeling exists. |
| Python dependencies become too broad | Use a small Bazel-only lock for graph tooling first. |
| Remote cache gets poisoned | Delay remote cache writes until actions are proven hermetic locally. |

## Final Gate for the Spike

The Bazel spike is successful when all of this is true:

```text
bazel test //tests/unittest/tools:ci_target_graph_test
  passes

bazel run //scripts/ci_target_graph:generate
  produces the same manifest as direct Python

bazel query / cquery
  select the expected AutoDeploy H100 seed targets

manual GPU targets
  only run when explicitly requested

shadow-mode metrics
  can be collected without changing Jenkins behavior
```

Only after this should we discuss wiring Bazel-generated impact selection into Jenkins as an
optional selector.

## References

- Bazelisk installation and `.bazelversion` behavior:
  https://preview.bazel.build/install/bazelisk
- Bzlmod and `MODULE.bazel`:
  https://bazel.build/external/overview
- Platforms and constraints:
  https://bazel.build/extending/platforms
- `target_compatible_with` behavior:
  https://preview.bazel.build/versions/9.1.0/reference/be/common-definitions
- `manual` tag behavior:
  https://bazel.build/reference/be/common-definitions#common.tags
- `test_suite` behavior:
  https://bazel.build/reference/be/general#test_suite
- Query and reverse dependency examples:
  https://preview.bazel.build/versions/8.6.0/query/guide
- Configured query:
  https://preview.bazel.build/query/cquery
- `rules_python` Bzlmod `pip.parse`:
  https://rules-python.readthedocs.io/en/1.5.0/pypi/download.html
- Bazel remote cache concepts:
  https://bazel.build/remote/caching
