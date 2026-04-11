# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mirage Onboarding Skill

This document is a practical onboarding guide for bringing a new AutoDeploy model onto the
Mirage/MPK path.

It is intentionally based on the evidence gathered during the Gemma4MoE bring-up in this repo.
The main goal is to help a new engineer or agent avoid rediscovering the same architectural
constraints, working patterns, and failure modes.

## Who This Is For

Use this guide if you are:

- onboarding a new model to AutoDeploy + Mirage/MPK
- trying to understand where Mirage fits in the AutoDeploy stack
- building a decode-oriented runtime path, not a full generic compiler backend

## The Core Mental Model

Mirage/MPK is best understood here as a **decode-oriented persistent-kernel runtime**, not as a
generic primitive-op backend.

AutoDeploy remains the frontend and overall orchestration layer:

```text
HF model
  -> AutoDeploy export / transforms
  -> decode-semantic understanding
  -> Mirage runtime callable
  -> logits
```

For v1, the intended execution policy is:

```text
decode-only batch
  -> Mirage candidate

prefill / extend / mixed-prefill-heavy batch
  -> normal AutoDeploy path
```

Important:

- Do not start by trying to lower arbitrary FX primitives into Mirage.
- Do not block on CUDA graph integration.
- Do not try to make Mirage own prefill first.

## What Mirage Really Exposes

The most important architectural finding is:

- Mirage is flexible in graph assembly.
- Mirage is not semantically arbitrary at the kernel/task level.
- Its real ABI is a catalog of semantic task families plus explicit runtime buffers.

In practice, Mirage/MPK already thinks in terms close to:

- `rmsnorm_layer`
- `rmsnorm_linear_layer`
- `paged_attention_layer`
- `linear_layer`
- `linear_with_residual_layer`
- `silu_mul_layer`
- `moe_topk_softmax_routing_layer`
- `moe_w13_linear_layer`
- `moe_silu_mul_layer`
- `moe_w2_linear_layer`
- `moe_mul_sum_add_layer`

That is why the right onboarding target is:

```text
AutoDeploy decode graph
  -> canonical decode-semantic region
  -> Mirage task/buffer instantiation
```

and not:

```text
AutoDeploy FX primitives
  -> generic Mirage TBGraph lowering
```

## What AutoDeploy Already Gives You

By the time the graph has gone through KV-cache and cached-attention transforms, AutoDeploy often
already exposes the decode semantics you want.

For Gemma4MoE, the important graph stage was around:

- `insert_cached_attention`
- `initialize_cache`
- `resize_kv_cache`
- just before `compile_model`

Typical decode-relevant graph inputs already become explicit there:

- `batch_info_host`
- `cu_seqlen_host`
- `cu_num_pages`
- `cu_num_pages_host`
- `cache_loc`
- `last_page_len`
- `last_page_len_host`
- `seq_len_with_cache_host`
- `cu_seqlen`
- `seq_len_with_cache`
- per-layer KV cache buffers

That means a new model bring-up should usually start from the **post-KV-cache graph**, not the raw
export graph.

## Recommended v1 Goal

For a new model, the recommended first goal is:

- live Mirage execution for **decode-only batches**
- coherent decode generation
- no dependence on CUDA graph
- keep non-decode-only execution on the existing AutoDeploy path

This is the cleanest target because AutoDeploy already treats decode as a distinct runtime regime.

## Current Support Shape

What is already proven in this repo:

- compile-stage `lower_to_mpk` integration exists
- MPK translation planning exists
- a strict no-eager-fallback runtime wrapper exists
- live Mirage-backed layer/block execution has been validated for important sub-blocks
- live Gemma4MoE bring-up has completed prefill and entered decode on the Mirage path
- AutoDeploy `compile_model` is already skipped when Mirage runtime mode is active

What is not fully complete yet:

- full end-to-end coherent generation through live Mirage for the whole Gemma4MoE stack
- mixed-batch decode-subset offload inside a single mixed iteration
- a stable generic solution for every decode-time FFN specialization

Known current live blocker during Gemma decode:

- single-token decode FFN-down generic Mirage matmul path
- current symptom: `CUBLAS_STATUS_INVALID_VALUE` on the generic matmul executor

Do not overgeneralize from that blocker. It is a decode specialization issue, not evidence that the
overall integration shape is wrong.

## Where Mirage Hooks Into AutoDeploy

The intended hook point is:

```text
AutoDeploy transforms
  -> lower_to_mpk
  -> Mirage runtime wrapper
  -> skip compile_model
  -> direct runtime execution
```

In code, the important pieces are:

- [lower_to_mpk.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/lower_to_mpk.py)
- [runtime_wrapper.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/runtime_wrapper.py)
- [mirage_bridge.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)

Important runtime behavior already in place:

- if `mpk_runtime_mode == "mirage_runtime"`, `compile_model` is skipped
- this keeps the Mirage v1 path independent from CUDA graph capture

## How AutoDeploy CUDA Graph Works Today

This matters because it explains why decode-first Mirage is a natural fit.

The `torch-cudagraph` backend in AutoDeploy has two modes:

- **monolithic CUDA graph** for decode-only batches
- **piecewise CUDA graph** for prefill/mixed batches when enabled

So the current system already treats decode as the place where a full-model execution shortcut is
most natural.

For Mirage onboarding, the recommendation is:

- do not try to combine Mirage and CUDA graph first
- get decode working directly through Mirage
- revisit CUDA graph later only if it is still useful

## New Model Onboarding Workflow

Follow this order.

### 1. Pick a decode-first target

Do not start with:

- full-model generic support
- mixed-batch decode-subset offload
- prefill acceleration
- CUDA graph integration

Start with:

- decode-only batches
- one model family
- one realistic layer path

### 2. Dump the graph after KV-cache transforms

Use graph dumps to confirm the graph is already decode-semantic enough.

Recommended command pattern:

```bash
bash -ic 'f1 && PYTHONPATH=$PWD AD_DUMP_GRAPHS_DIR=$PWD/mpk_model_graphs python3 examples/auto_deploy/build_and_run_ad.py --args.model <hf-model> --args.yaml-extra <model-yaml> 2>&1 | tee model_mpk.log'
```

Look for graph stages around:

- cached attention insertion
- cache initialization
- just before compile

### 3. Confirm the graph already exposes decode semantics

You want to see explicit decode/runtime structure such as:

- paged KV metadata
- cached attention ops
- token gather indices
- explicit batch metadata
- FFN or MoE structure already grouped in useful ways

If the graph is still too primitive, stop and add normalization first. Do not force Mirage to
consume frontend noise.

### 4. Build a model-specific analyzer

The analyzer should recover:

- per-layer semantic structure
- key weights and runtime tensors
- head sizes / KV layout / MoE structure
- the exact logical boundaries you intend to lower

For Gemma4MoE, this lives in:

- [gemma_analyzer.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/gemma_analyzer.py)

### 5. Lower into Mirage-compatible semantic steps

Do not mirror every FX node.

Instead, lower into semantic phases close to Mirage task vocabulary, for example:

- QKV projection
- attention prep
- paged attention
- output projection / residual path
- router
- expert path
- FFN path

The planner/lowering side in this repo is organized around:

- [types.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/types.py)
- [translator.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/translator.py)
- [buffer_planner.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/buffer_planner.py)
- [gemma_layer_lowering.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/gemma_layer_lowering.py)

### 6. Build a live runtime path only after small blocks are correct

The runtime bridge belongs in:

- [mirage_bridge.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)

Do not wire the full model first.

First prove correctness for:

- block-level live kernels
- layer-level live execution
- runtime buffer contracts

Only then install the runtime wrapper.

## Files You Will Likely Touch

### Analyzer / translation side

- [gemma_analyzer.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/gemma_analyzer.py)
  - recover model-specific structure from the post-transform graph
- [translator.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/translator.py)
  - turn analyzer output into a plan
- [types.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/types.py)
  - define plan and lowering data structures
- [buffer_planner.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/buffer_planner.py)
  - make runtime buffers explicit
- [gemma_layer_lowering.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/gemma_layer_lowering.py)
  - map canonical ops to Mirage-relevant steps

### Runtime side

- [mirage_bridge.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)
  - live Mirage object construction and execution
- [runtime_wrapper.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/runtime_wrapper.py)
  - strict wrapper boundary, no silent eager fallback
- [lower_to_mpk.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/lower_to_mpk.py)
  - compile-stage hook that installs the Mirage runtime path

### Runtime metadata and batching behavior

- [ad_executor.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py)
  - builds `batch_info` and the runtime metadata used by the graph
- [attention_interface.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py)
  - source of truth for flattened layouts, gather behavior, and decode-vs-mixed metadata handling

### Tests

- [test_gemma4_mpk_layer.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tests/unittest/auto_deploy/singlegpu/transformations/library/test_gemma4_mpk_layer.py)
- [test_mpk_runtime_contract.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tests/unittest/auto_deploy/singlegpu/transformations/library/test_mpk_runtime_contract.py)

## Best Practices

These were learned the hard way during the Gemma bring-up.

### Prefer decode-first bring-up

If a new model has both prefill and decode paths, start with decode.

### Use semantic boundaries, not raw FX boundaries

Normalize into meaningful decode phases before lowering into Mirage.

### Keep runtime metadata explicit

Do not hide:

- batch geometry
- token gather behavior
- KV metadata
- routing metadata

Mirage likes explicit buffers and runtime state.

### Validate live numerical correctness block-by-block

Smoke tests are not enough.

For each live block you onboard, prove numerical correctness before composing it into a layer.

### Move from working to working

The safest order is:

- one live block
- one live composed sublayer
- one live full layer
- one live decode loop
- full generation

### Separate compile issues from runtime issues

Mirage compile latency is often dominated by `nvcc`, but runtime stalls can still be independent.
Always profile before assuming.

### Inspect generated Mirage artifacts when needed

The generated:

- `test.cu`
- `task_graph_rank0.json`

are often the fastest way to confirm whether the runtime wiring is what you think it is.

### Match AutoDeploy’s runtime contract exactly

A surprisingly small wrapper mismatch can break sampling or downstream logic.

Example from Gemma bring-up:

- returning a bare tensor instead of `{"logits": ...}` caused AutoDeploy to treat the output like a
  tuple and collapse the logits rank incorrectly

### Treat single-token decode as a special case

A kernel family that works for prefill-shaped batches may still fail for single-token decode
specializations.

## Known Pitfalls

Current known pitfalls from the Gemma path:

- some Mirage task families are specialization-sensitive
- not every mathematically equivalent kernel family is interchangeable at runtime
- generic Mirage paths can fail on decode-only shape specializations
- wrapper output shape/rank mistakes show up late in sampling, not necessarily at the runtime call
  site
- compile latency can be significant enough to hide the real next failure unless you inspect worker
  processes directly

## Testing Ladder

Follow this order for a new model.

### 1. Analyzer / plan test

Pass means:

- layer structure is recovered
- key weights and semantic steps are identified

### 2. Mirage task binding / task-graph generation test

Pass means:

- the planned steps map to real Mirage APIs
- task graph generation succeeds

### 3. Live single-block correctness test

Pass means:

- Mirage block output matches torch reference closely
- repeat launches are stable

### 4. Live full-layer correctness test

Pass means:

- layer output matches the reference path
- no hidden launch/reuse instability remains

### 5. Decode-only forward test

Pass means:

- the runtime produces logits with the correct shape and contract

### 6. Decode generation loop test

Pass means:

- prefill completes
- decode begins
- tokens are sampled without contract/rank errors

### 7. End-to-end coherence test

Pass means:

- generated text is valid and coherent for the target model

Only after this ladder is green should you consider:

- mixed-batch decode-subset offload
- prefill support through Mirage
- CUDA graph integration

## Mixed-Batch Guidance

Important distinction:

- decode-only Mirage execution is the primary v1 target
- mixed-batch **decode-subset-only** offload is a possible v2 extension

Why:

- AutoDeploy already exposes the metadata needed to identify decode tokens inside mixed batches
- but the current Mirage runtime shape is still one runtime path over the full step inputs
- selective decode-subset execution inside a mixed step requires explicit partitioning and output
  stitching

So for a new model:

- do not start by trying to Mirage only the decode rows of a mixed iteration
- get decode-only working first

## Concrete Commands

### Graph dumping

```bash
bash -ic 'f1 && PYTHONPATH=$PWD AD_DUMP_GRAPHS_DIR=$PWD/mpk_model_graphs python3 examples/auto_deploy/build_and_run_ad.py --args.model <hf-model> --args.yaml-extra <yaml> 2>&1 | tee model_mpk.log'
```

### Fast iteration with reduced layers

Use a model-specific yaml override like:

```yaml
model_kwargs:
  text_config:
    num_hidden_layers: 5
```

Then run:

```bash
bash -ic 'f1 && PYTHONPATH=$PWD python3 examples/auto_deploy/build_and_run_ad.py --args.model <hf-model> --args.yaml-extra <base-yaml> --args.yaml-extra <reduced-layer-yaml> 2>&1 | tee model_mpk_fast.log'
```

### Live Mirage runtime run

```bash
bash -ic 'f1 && PYTHONPATH=$PWD AD_DUMP_GRAPHS_DIR=$PWD/mpk_model_graphs_live AD_MPK_DEBUG=1 python3 examples/auto_deploy/build_and_run_ad.py --args.model <hf-model> --args.yaml-extra <base-yaml> --args.transforms.lower-to-mpk.dry-run-only=false 2>&1 | tee model_mpk_live.log'
```

### Layer/runtime tests

```bash
bash -ic 'f1 && PYTHONPATH=$PWD python3 -m pytest -q tests/unittest/auto_deploy/singlegpu/transformations/library/test_mpk_runtime_contract.py'
```

```bash
bash -ic 'f1 && PYTHONPATH=$PWD:/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python python3 -m pytest -q tests/unittest/auto_deploy/singlegpu/transformations/library/test_gemma4_mpk_layer.py'
```

### Useful log grep

```bash
bash -ic 'f1 && rg -n "AD_MPK_DEBUG|Traceback|RequestError|Processed requests|return logits shape|Destroying process group" model_mpk_live.log | tail -n 200'
```

### Useful process inspection

```bash
bash -ic 'f1 && ps -ef | rg "build_and_run_ad.py|mpi4py.futures.server|nvcc|ptxas"'
```

```bash
bash -ic 'f1 && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sort'
```

## What Not To Do First

Avoid these as an initial bring-up strategy:

- lowering raw primitive FX nodes directly into Mirage
- making Mirage own prefill before decode works
- trying to support mixed-batch decode-subset offload first
- coupling bring-up to CUDA graph capture
- trusting smoke tests without numerical validation
- overfitting the canonical IR to today’s exact MPK task decomposition

## Suggested Success Criteria For a New Model

A new model onboarding should be considered successful when:

- the model’s post-KV-cache graph can be understood semantically
- the analyzer and planner recover meaningful decode layers
- live Mirage block and layer correctness tests pass
- decode-only execution runs through the live Mirage path
- generation works without fallback
- generated text is coherent

That is the right v1 bar.
