<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hidden-State Capture Handoff

## Context

The current Qwen3.5 MTP branch has a workaround in
`tensorrt_llm/_torch/auto_deploy/utils/node_utils.py` for target graphs whose
live residual-chain seed is `inputs_embeds` rather than an `input_ids` ->
`aten.embedding` path.

This should probably be treated as a base `gramnarayan/qwen3-mtp` bug, not as a
VLM-only issue. Qwen3.5 MTP targets can be exported through `inputs_embeds`, and
the hidden-state capture transform should robustly find the residual chain in
that graph shape.

## Recommended Fix

Port the cleaner approach from `gramnarayan/mistral4-eagle-v2` into the base
Qwen3.5 MTP branch:

1. In `identify_regions_between_residuals(gm)`, collect all placeholders and
   the output node.
2. If the graph has an `aten.embedding` node, use the placeholder feeding that
   embedding as the seed and include the embedding node as the next boundary.
3. Otherwise, find the placeholder named `inputs_embeds` and use that as the
   seed.
4. If neither an embedding path nor an `inputs_embeds` placeholder exists,
   return a minimal `[first_placeholder, output_node]` boundary list and log a
   debug message with the placeholder names.

This is better than the current Qwen-specific workaround because it does not
assume that `inputs_embeds` is the first placeholder. Export may leave an unused
`input_ids` placeholder before the live `inputs_embeds` placeholder, and the
hidden-state capture path should still select the real residual-chain seed.

## Suggested Tests

Add or keep tests that cover both graph shapes:

1. Standard text graph: `input_ids` feeds an `aten.embedding` op, and layer
   region detection starts from that embedding path.
2. Qwen/Eagle target graph: no live `aten.embedding` node exists, an
   `inputs_embeds` placeholder is present, and hidden-state capture still finds
   the expected residual-add nodes.
3. Defensive case: if neither path exists, the function returns the minimal
   boundary list instead of asserting on the wrong placeholder.

The likely test home is the existing Qwen3.5 MTP/unit coverage near
`tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe.py`, especially
the MTP target export and hidden-state capture tests.

## Why This Belongs In Base Qwen3-MTP

The VLM branch will need this behavior for `target_model.language_model`, but
the underlying issue is not caused by the VLM wrapper. It is caused by target
exports that use precomputed embeddings. Fixing it in `gramnarayan/qwen3-mtp`
lets the VLM branch inherit the robust capture behavior instead of carrying a
second local workaround.
