<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CI Target Graph

This package generates shadow-mode Bazel-style metadata for TensorRT-LLM CI.
It does not add Bazel root files, BUILD files, toolchains, or CI enforcement.
The first step is only to make the current Jenkins/test-db pytest selectors
visible as explicit JSON targets so impact-selection ideas can be compared
against the existing CI without changing what Jenkins runs.

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

The manifest uses `schema_version: 1`; its practical JSON Schema lives in
`manifest_schema_v1.json`.

## Inputs

The generator reads:

- `jenkins/L0_Test.groovy` for active stage-to-test-db mappings.
- `tests/integration/test_lists/test-db/*.yml`
- `tests/integration/test_lists/test-db/*.yaml`

Each emitted target has `kind: "pytest_selector"` and records the raw selector,
parsed pytest path, all parsed pytest paths, remaining pytest arguments,
timeout minutes, isolation marker, test-db context, matching Jenkins stage
candidates, test-db constraints, tags, and conservative component hints.

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

No Bazel files are added yet because the repo does not have a declared target
model for Python, C++, CUDA, GPU platforms, model/data inputs, and generated
artifacts. This shadow manifest is the low-risk first step toward that model.
