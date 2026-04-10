# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# April 10 Status

## Goal

The target remains:

```text
AutoDeploy Gemma4MoE FX graph
  -> canonical Gemma MPK translation
  -> live Mirage-backed runtime execution
  -> valid/coherent end-to-end generation
```

## What Is Implemented

- `lower_to_mpk` runs at AutoDeploy `compile` stage, before `compile_model`.
- The Gemma analyzer can recover layer structure from the real post-KV-cache graph.
- The translator emits a structured MPK plan with graph info, buffer plan, and per-layer lowerings.
- The transformed graph can be rewritten to call an opaque runtime boundary:

```text
%gemma_mpk_runtime = gemma_mpk_runtime(...)
output %gemma_mpk_runtime
```

- Small Mirage building blocks were validated against the installed Mirage Python surface.
- Layer-level tests exist for:
  - Mirage task binding / task-graph generation
  - plan-driven numerical correctness using a torch reference executor

## What Is Not Done Yet

The current runtime path is still:

```text
gemma_mpk_runtime(...)
  -> eager_fallback(original GraphModule)
```

This means:

- end-to-end coherent generation currently proves the MPK integration point and graph replacement shape
- it does **not** prove full-model live Mirage execution

The eager fallback was added as a scaffold to validate:

- graph replacement at the compile-stage hook
- runtime call signature / input plumbing
- downstream compile compatibility
- optional CUDA-graph interaction

It is not the desired end state.

## End-to-End Status

### Full 30-layer transformed path

This run completes successfully and produces coherent output:

```bash
bash -ic 'f1 && PYTHONPATH=$PWD AD_DUMP_GRAPHS_DIR=$PWD/mpk_gemma4_graphs_full python3 examples/auto_deploy/build_and_run_ad.py --args.model google/gemma-4-26B-A4B-it --args.yaml-extra examples/auto_deploy/model_registry/configs/gemma4_moe.yaml --yaml-extra examples/auto_deploy/model_registry/configs/gemma4_chat_prompt.yaml --args.transforms.lower-to-mpk.dry-run-only=false 2>&1 | tee gemma_moe_mpk_wrapper_dump_chat.log'
```

What that proves:

- `lower_to_mpk` runs
- the graph contains `gemma_mpk_runtime(...)`
- compile completes
- generation completes
- output is coherent

What it does **not** prove:

- live Mirage execution of the full model

### 5-layer fast-iteration path

The reduced-layer config is valid and useful for quicker iteration:

```yaml
model_kwargs:
  text_config:
    num_hidden_layers: 5
```

But it is not a quality target. The 5-layer model runs end to end, but the output quality is not a meaningful correctness bar for Gemma.

## Test Status

### Truly Mirage-backed

- small-block / layer Mirage integration tests that:
  - resolve planned MPK methods against installed Mirage
  - instantiate real `PersistentKernel` objects
  - register tasks
  - generate Mirage task graphs

### Not full Mirage execution

- layer numerical correctness tests currently use the plan-driven torch reference executor in `mirage_bridge.py`
- full end-to-end generation currently goes through `eager_fallback`

So today:

- Mirage API compatibility: yes
- Mirage small-block execution scaffolding: yes
- full-model live Mirage execution: no

## Practical Conclusion

As of April 10:

- the MPK integration point in AutoDeploy is real
- the Gemma translation/planning path is real
- the transformed graph shape is real
- coherent generation on the transformed path is real
- but the backend under that path is still fallback-backed

## Next Step

Given that CUDA graph is optional for Mirage, the next implementation target should be:

```text
FX graph
  -> lower_to_mpk
  -> gemma_mpk_runtime(...)
  -> direct Mirage-backed callable
```

not:

```text
FX graph
  -> lower_to_mpk
  -> gemma_mpk_runtime(...)
  -> eager fallback
```

The next milestone is therefore:

1. remove the fallback assumption from the runtime path
2. bypass or disable `compile_model` / CUDA-graph capture for Mirage if needed
3. execute a real translated Gemma region through Mirage
4. expand until the full decode path runs through Mirage
