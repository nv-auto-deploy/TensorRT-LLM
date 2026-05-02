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

The manifest uses `schema_version: 1`; its practical JSON Schema lives in
`manifest_schema_v1.json`.

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

The Bazel spike consumes this manifest for a narrow generated AutoDeploy H100
subpackage, but most Python, C++, CUDA, model/data inputs, and generated
artifacts are still intentionally unmodeled. Unknown or unmodeled areas should
fall back conservatively rather than being skipped.
