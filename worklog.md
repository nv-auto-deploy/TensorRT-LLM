# Worklog: AutoDeploy + MPK / Mirage Exploration

## Context

We explored whether AutoDeploy can be used as the frontend for an MPK / Mirage-based decode megakernel path, using Gemma4MoE as the driving example.

The initial idea was:

```text
AutoDeploy FX graph
-> normalize into decode-specific semantic ops
-> lower into MPK / Mirage builder calls
-> compile megakernel
-> execute through a stable runtime boundary
```

We also considered whether xDSL could still be useful as an intermediate representation, not in its current elementwise form, but as a decode-semantic dialect.

## High-Level Conclusion

There is a feasible path.

The most important clarification is that the first realistic step is not:

- lowering arbitrary generic FX directly into MPK

Instead, the right first step is:

- start from the AutoDeploy graph after the important decode-oriented transforms have already happened
- identify the decode semantics that are already explicit
- define a semantic decode region / dialect around those
- lower that semantic region into MPK

In short:

- AutoDeploy looks like a viable frontend
- MPK looks like a viable backend/runtime
- the missing middle layer is a decode-semantic normalization / lowering stage

## Earlier Architectural Findings

Before looking at the Gemma4MoE dumps, we established:

- AutoDeploy already has the compiler-lane shape we want:
  - FX to normalized IR
  - transform / fuse / rewrite
  - opaque custom-op boundary
  - backend codegen
- MPK already has a Python-driven persistent-kernel builder and runtime
- the hard problem is the semantic bridge, not custom-op registration

We also concluded that if xDSL is used, it should probably not mirror the current elementwise fusion dialect. Instead, it should represent decode-semantic operations such as:

- embedding
- norm + linear
- paged cached attention
- residual / norm fused boundaries
- MoE routing
- MoE expert execution / combine
- LM head / token selection

## Gemma4MoE Graph Dumps Reviewed

We inspected the graph dumps under:

- [gemma4moe_graph](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph)

Relevant stages included:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt)
- [076_visualize_visualize_namespace.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/076_visualize_visualize_namespace.txt)
- [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt)

## Key Findings From The Graph Dumps

### 1. The graph is already decode-semantic by cache-init / visualize time

By the time we reach `insert_cached_attention`, the graph is no longer a generic transformer graph. It already exposes decode-relevant runtime structure.

The graph contains explicit runtime inputs such as:

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
- per-layer KV cache buffers `r0_kv_cache` through `r29_kv_cache`

This is visible directly in:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt)

### 2. Cached attention is already explicit

Instead of generic attention, the graph contains:

- `auto_deploy.triton_paged_prepare_metadata.default(...)`
- `auto_deploy.triton_paged_mha_with_cache.default(...)`

This means the attention part of the decode graph has already crossed into a paged-KV, runtime-metadata-aware representation.

### 3. MoE semantics are also already explicit

The Gemma4MoE graph includes:

- `auto_deploy.triton_fused_topk_softmax(...)`
- `auto_deploy.trtllm_moe_fused(...)`

This is a very important result: the graph is not only decode-specific, but also exposes fused MoE routing / execution structure that we can target semantically.

### 4. A lot of the surrounding math is already fused into meaningful blocks

The graph includes many `auto_deploy.mlir_fused_*` regions that appear to cover:

- residual + norm combinations
- pre/post-attention normalization boundaries
- pre/post-feedforward normalization boundaries
- other elementwise glue around major decode blocks

This suggests that a semantic decode-region builder does not need to begin from tiny unfused primitives.

### 5. The compile-stage graph still preserves the semantic structure

The final compile-stage dump still shows the same decode-oriented inputs and operations in the monolithic graph, for example in:

- [081_compile_compile_model.txt#L10](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt#L10)

That confirms that the relevant semantics are still present late in the transform pipeline.

## Mixed-Batch Decode Findings

We discussed uncertainty around mixed-batch decode and whether it is compatible with an MPK lowering.

Current understanding:

- AutoDeploy already represents mixed batch decode through runtime metadata, not by changing the graph topology per request mix
- this metadata includes sequence counts, token counts, page layout, and slot mappings

In executor code, the runtime builds metadata such as:

- `batch_info = [num_prefill, num_prefill_tokens, num_extend, num_extend_tokens, num_decode, num_decode_tokens]`
- `cache_loc`
- `cu_num_pages`
- `slot_idx`

This happens in:

- [ad_executor.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py#L835)

Implication:

- mixed-batch decode looks compatible with an MPK path
- the right model is a stable decode graph plus explicit metadata tensors
- this is aligned with how MPK expects runtime state to be supplied

## CUDA Graph Capture Discussion

We discussed whether the CUDA-graph capture point is the right place to convert to an MPK kernel.

### Short answer

Conceptually: yes, it is the right neighborhood.

Operationally: not exactly at the current CUDA-graph wrapper boundary.

### Why

The current `compile_model` / piecewise CUDA-graph path is already splitting the model at dynamic boundaries and wrapping static segments for capture.

Relevant code:

- [compile_model.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/compile_model.py)
- [torch_cudagraph.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/compile/backends/torch_cudagraph.py#L240)

The piecewise CUDA-graph backend explicitly:

- splits at dynamic op boundaries
- captures static segments
- wraps dynamic ops separately

That is useful for CUDA graph replay, but it is not the ideal compiler boundary for MPK, because MPK wants to own a larger fused decode region, including the dynamic decode-specific pieces.

### Recommended insertion point

The recommended MPK lowering point is:

- after cache and runtime metadata have been made explicit
- before `compile_model` performs the piecewise CUDA-graph splitting / wrapping

In practice, this suggests using the graph around stages:

- `066` to `076`

as the main source for the first semantic lowering.

## Recommended First Semantic Region For Gemma4MoE

A realistic first semantic decode region for Gemma4MoE would include operations like:

- `prepare_paged_attention_metadata`
- `qkv_projection`
- `q_norm`
- `k_norm`
- `v_norm` if needed semantically
- `rope`
- `paged_attention_with_cache`
- `o_projection`
- `post_attention_residual_norm`
- `moe_route`
- `moe_fused`
- `post_moe_residual_norm`

Region inputs should keep runtime state explicit:

- token inputs
- position inputs
- mixed-batch metadata tensors
- page metadata tensors
- KV cache tensors / handles
- gather/scatter or token selection metadata where needed

## Practical Interpretation

The decode semantics do not need to be invented from scratch.

They are already present in the AutoDeploy graph dumps in a fairly strong form:

- cached paged attention
- explicit decode metadata
- explicit KV resources
- explicit MoE fused execution

So the first compiler problem is not:

- "how do we run MPK?"

## Apr 10: Live Mirage Debugging Status

This round focused on moving from planning and dry-run translation toward real
live Mirage execution, while debugging in increasing scope:

```text
single live block
-> repeated live block
-> small composed live block
-> larger composed live block
```

The main rule for this phase was:

- only keep and trust milestones that are numerically validated
- do not count eager fallback or graph-shape-only success as backend success

## What Was Proven Live

### 1. Standalone live Mirage kernels can be numerically correct

We added and validated real numerical checks for:

- direct Mirage `norm_linear`
- direct Mirage `paged_attention`
- direct Mirage `linear_with_residual`

These are implemented in:

- [mirage_bridge.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)

and exercised from:

- [test_gemma4_mpk_layer.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tests/unittest/auto_deploy/singlegpu/transformations/library/test_gemma4_mpk_layer.py)

### 2. A compiled Mirage `PersistentKernel` `linear_with_residual` block works cleanly

The strongest stable live foothold we now have is:

- a real compiled `PersistentKernel`
- with a `linear_with_residual_layer`
- numerically validated against torch reference
- stable across repeated launches in the same process

This is captured by:

- `run_mirage_linear_with_residual_pk_forward_correctness(...)`

and the passing test:

- `test_mirage_linear_with_residual_pk_matches_reference_across_repeats`

This is the current "working to working" baseline.

## Main Bugs Found

### 1. Direct wrapped `linear_with_residual` was not safely reusable in-process

Observation:

- the direct extension-wrapper path for `linear_with_residual`
- could be correct on first launch
- but often became non-finite or wildly incorrect on later launches

Implication:

- this direct wrapper is not trustworthy as the execution substrate for larger
  composed live blocks
- we should prefer the compiled Mirage `PersistentKernel` launcher path for
  reusable execution

### 2. Repeated Mirage launches require explicit runtime/request-state reset

Observation:

- several composed live paths were correct once but failed on the second launch
- this was not just a numerical drift issue; it was runtime-state reuse

Evidence from Mirage runtime code:

- `PersistentKernel` exposes `init_request_func()`
- the runtime tracks mutable per-request state such as:
  - `step`
  - `tokens`
  - `input_tokens`
  - `output_tokens`
  - `prompt_lengths`
  - `qo_indptr_buffer`
  - `paged_kv_indptr_buffer`
  - `paged_kv_indices_buffer`
  - `paged_kv_last_page_len_buffer`

Relevant code:

- [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L1562)
- [persistent_kernel.cuh](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/include/mirage/persistent_kernel/persistent_kernel.cuh#L196)
- [persistent_kernel.cuh](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/include/mirage/persistent_kernel/persistent_kernel.cuh#L246)

Fix:

- add explicit bridge-side runtime reset before repeated `PersistentKernel`
  launches
- zero and reseed the relevant meta tensors
- call Mirage `init_request_func()` after reset

This is now implemented by:

- `_reset_pk_runtime_state(...)`
  in [mirage_bridge.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)

Result:

- repeated PK `linear_with_residual` launches became stable
- repeated PK/hybrid attention-block probes became deterministic instead of
  "first good / second broken"

### 3. Mirage request prep semantics matter for token count

Observation from runtime code:

- Mirage decides how many active tokens to process from request state
- if `prompt_length - step > 0`, it takes a prefill-like path
- otherwise it falls back to decode-like single-token processing

Implication:

- multi-token tests cannot rely on buffer shapes alone
- the runtime metadata must explicitly describe the intended active-token count

Fix:

- `_reset_pk_runtime_state(...)` now seeds:
  - `prompt_lengths`
  - `tokens`
  - request metadata buffers

This made the runtime behavior more interpretable, even when the larger block
still remained numerically wrong.

## Current Live Scope Status

### Stable and trusted

- direct Mirage `norm_linear`
- direct Mirage `paged_attention`
- compiled PK `linear_with_residual`

### Stable after reset, but still numerically off

- composed live attention block probes
- hybrid live attention sublayer probes

These now fail in deterministic ways rather than random repeat corruption.
That is progress, but they are not yet promotion-worthy as passing tests.

### Not yet solved

- full live Gemma attention sublayer through Mirage
- full live Gemma block through Mirage
- full live Gemma4MoE execution through Mirage

## Best Practices For Interfacing With MPK / Mirage

These are the main lessons so far.

### 1. Prefer the compiled `PersistentKernel` path over ad hoc direct wrappers

Direct extension wrappers are useful for narrow experiments, but they are not
the safest reusable execution substrate.

Best practice:

- use direct wrappers only for single-kernel bring-up and local numerical probes
- move reusable execution onto compiled `PersistentKernel` launchers as early as
  possible

### 2. Always treat Mirage runtime state as mutable and per-request

Do not assume a compiled kernel is stateless between launches.

Best practice:

- reset request/runtime meta tensors before repeated launches in tests
- reseed:
  - `step`
  - `tokens`
  - `input_tokens`
  - `output_tokens`
  - `num_new_tokens`
  - `prompt_lengths`
  - `qo_indptr_buffer`
  - `paged_kv_indptr_buffer`
  - `paged_kv_indices_buffer`
  - `paged_kv_last_page_len_buffer`
- call `init_request_func()` when available

### 3. Encode active-token count in metadata, not only in tensor shapes

Mirage request prep logic uses runtime metadata to determine batching behavior.

Best practice:

- when testing multi-token or prefill-like paths, explicitly set
  `prompt_lengths` and related metadata to match the intended live request
  shape

### 4. Synchronize after PK launches when validating numerics

Mirage launches are async from Python.

Best practice:

- call `torch.cuda.synchronize()` before reading results in numerical tests
- otherwise later host reads may see stale or partially completed state

### 5. Grow scope only from numerically validated working pieces

This debugging round reinforced the value of:

- first proving one live kernel
- then one repeated live block
- then one small composed live block

instead of jumping directly to end-to-end full-model execution

## Recommended Next Debugging Order

The next best order remains:

1. keep PK `linear_with_residual` as the trusted reusable live baseline
2. get a single-token live attention sublayer fully clean and testable
3. only then move to multi-token / bigger attention blocks
4. then revisit full layer
5. only after that attempt full-model live Mirage execution

This preserves the current "working to working" debugging discipline.

It is:

- "how do we canonicalize AutoDeploy's post-cache-init decode graph into a stable semantic region that MPK can own?"

## Suggested Next Step

The next concrete step is to define the first Gemma4MoE decode-semantic IR sketch:

- exact semantic ops
- exact region inputs / outputs
- mapping from current FX ops in the late-stage dump into those semantic ops
- what remains outside the MPK region for v1

That work should use the late post-cache-init graph as the source of truth, not the raw exported FX graph.

## 2026-04-10 Mirage Block Debugging Checkpoint

This round made two meaningful live-Mirage advances:

- the split-dense Gemma-style MoE block is now a real passing correctness test
- the attention-side live tests remain passing, so we now have both sides of a
  synthetic layer validated independently

### What turned out to be real bugs or bad assumptions

#### 1. MoE expert launch geometry was wrong

The biggest correctness miss on the MoE path was not primarily kernel math.
It was launch coverage.

Evidence:

- Mirage MoE kernels shard activated experts by `task_metadata.expert_offset`
- one logical MoE op needs multiple `grid_dim.x` launches to cover the full
  expert stride
- our earlier helpers were launching `grid_dim=(1,1,1)`, which only exercised
  shard 0

Fix:

- add `_moe_expert_grid_dim(...)`
- use full expert-stride launch shapes for:
  - `moe_w13_linear_layer`
  - `moe_w2_linear_layer`

## 2026-04-10 Full Live-Layer Bringup Learnings

This round closed an important live-Mirage blocker in the synthetic Gemma full
layer path.

### 1. The original full-layer stall was not just "compile is slow"

We first measured the standalone compile costs directly:

- generic Mirage `gelu + mul`: about `11.8 - 12.0 s`
- small MPK `PersistentKernel` compile: about `22.3 s`

Then we added stage-by-stage timing to the synthetic full-layer helper in:

- [mirage_bridge.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)

and recorded the flow in:

- [mpk_workflow_and_compile_profile.md](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/mpk_workflow_and_compile_profile.md)

The key finding was:

- front-half stages were expensive but finite
- the long tail after `ffn_phase2_compile_s` was a runtime stall in
  `torch.cuda.synchronize()`
- so the full-layer issue was not purely compile time

### 2. The real failing stage was FFN down projection

By progressively shrinking the reproducer, we found:

- generic Mirage `gelu * mul` activation completed successfully
- PK compile for FFN down completed successfully
- the process hung at the PK launch/sync for FFN down

Then we reduced it even further:

- plain PK `linear_layer` with shape `1 x 64 -> 1 x 512`
- with random input
- and with different `grid_dim.x` choices

This still hung at launch.

Conclusion:

- the problem is not specific to the generic activation handoff
- the problematic piece is the PK `linear_layer` specialization for the FFN
  down-projection shape itself

### 3. Generic Mirage matmul was not a drop-in replacement

We tested whether generic Mirage `KNGraph.matmul` could cover the FFN-down
projection live.

Result:

- it did not hang in the same way
- but it failed with `CUBLAS_STATUS_INVALID_VALUE`

Conclusion:

- generic Mirage `matmul` is not a ready-made substitute for this shape/path in
  our current bridge setup

### 4. A working live workaround exists: express FFN down as top-k=1 `moe_w2`

The most important new result is that FFN down-projection can be executed live
and correctly by re-expressing it as:

```text
topk = 1
num_experts = 1
moe_w2_linear_layer
-> moe_mul_sum_add_layer
```

This works because the FFN down math matches the same tensor contraction shape:

```text
(batch, 1, intermediate) x (1, hidden, intermediate) -> (batch, 1, hidden)
```

Using the standard MPK routing encoding:

- `routing_indices[0, 0] = 1`
- `routing_mask[0] = 0`
- `routing_mask[num_experts] = 1`
- `topk_weight = 1`
- zero residual

Result:

- live execution succeeded
- numerical correctness was good
- this held both for random input and for generic activation-produced input

Helper added:

- `run_mirage_ffn_down_via_moe_w2_forward_correctness(...)`

### 5. The synthetic full live layer now completes end to end

After swapping FFN phase 2 from broken PK `linear_layer` to the top-k=1
`moe_w2` workaround, the stage profile completed through all stages:

- `attn_pk`
- `ffn_phase1`
- `ffn_activation`
- `ffn_phase2`
- `router_pk`
- `expert_pk`
- `moe_activation`
- `phase3`

Representative stage timings from the successful profile:

- `attn_pk_compile_s ~= 26.9 s`
- `ffn_phase1_compile_s ~= 26.8 s`
- `ffn_activation_compile_s ~= 12.0 s`
- `ffn_phase2_compile_s ~= 30.9 s`
- `router_pk_compile_s ~= 31.2 s`
- `expert_pk_compile_s ~= 32.2 s`
- `moe_activation_compile_s ~= 11.8 s`
- `phase3_compile_s ~= 31.7 s`

All corresponding launch times were small, roughly `0.02 - 0.15 s`.

### 6. The full live layer is now numerically sane enough to test

The full synthetic live layer now returns finite, bounded errors instead of
hanging.

Observed result snapshot:

- `post_attn_max_abs ~= 0.050`
- `post_attn_mean_abs ~= 0.0086`
- `ffn_down_max_abs ~= 0.0064`
- `ffn_down_mean_abs ~= 0.00165`
- `topk_weight_max_abs ~= 0.00195`
- `topk_weight_mean_abs ~= 0.00074`
- `routing_overlap_count = 8`
- `moe_act_max_abs ~= 0.143`
- `moe_act_mean_abs ~= 0.0071`
- `w2_max_abs ~= 0.0527`
- `w2_mean_abs ~= 0.0076`
- `hidden_out_max_abs ~= 0.0216`
- `hidden_out_mean_abs ~= 0.0058`

Interpretation:

- FFN down is no longer the dominant error or a runtime blocker
- the remaining larger error bars are now on:
  - attention approximation
  - MoE activation
  - MoE W2

That is a much healthier debugging position.

### 7. New practical best practice

For Gemma-style GELU FFN lowering in the current Mirage bridge:

- do not use PK `linear_layer` directly for the `1 x 64 -> 1 x 512` FFN down
  projection
- prefer the proven top-k=1 `moe_w2_linear + moe_mul_sum_add` composition

This is now the bridge-side best practice until the underlying PK
`linear_layer` specialization is fixed for that shape.

### 8. Planner / translation implication

The dry-run plan previously marked:

- `dense_ffn_activation`

as a hard backend `GAP`.

That is no longer fully accurate.

Current better status:

- `dense_ffn_gate_up` is supported
- `dense_ffn_activation` is partially supported through a bridge-side lowering:
  - generic `gelu * mul`
  - top-k=1 `moe_w2_linear`
  - `moe_mul_sum_add`

So this step should be treated as:

- not a native one-op MPK task match
- but no longer a complete execution blocker either
  - `moe_w13_linear_layer`
  - `moe_w2_linear_layer`

Effect:

- `w2` and final block-output error dropped from clearly broken to acceptably
  small for the split-dense live block

#### 2. Mirage SM90 MoE mask layout needed patching

The generated SM90 MoE code was using `make_shape(num_experts)` for the expert
mask layout even though the runtime contract stores the sentinel/count in
`num_experts + 1`.

Fix:

- patch the generated CUDA source in `patch_generated_mirage_cuda_source(...)`
  so the SM90 MoE mask layout uses `num_experts + 1`

This did not solve the whole block by itself, but it removed one real
codegen/runtime mismatch.

#### 3. Some earlier "routing" and "activation" error measurements were
not isolating the right thing

Two comparisons were misleading:

- reading `router_logits` after `moe_topk_softmax_routing_layer`
  - Mirage's fused top-k kernel mutates/zeros the input buffer as part of the
    workflow, so post-kernel `router_logits` is not a valid pre-topk reference
- comparing activation output only against the full float reference path
  - that mixes in upstream bf16 matmul drift instead of isolating the live
    activation kernel itself

Best practice:

- compare routing outputs against the float reference from the same semantic
  stage, but do not expect the raw logits input buffer to survive unchanged
- compare activation both to the semantic float reference and to a stage-local
  reference built from the live upstream tensors when isolating kernel quality

### What is now genuinely passing

Focused Mirage tests that now pass:

- split-dense projection helper
- split-dense MoE block helper
- hybrid attention sublayer
- PK attention block
- PK attention sublayer

Verified commands:

- `bash -ic 'f1 && PYTHONPATH=$PWD:/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python python3 -m pytest -q tests/unittest/auto_deploy/singlegpu/transformations/library/test_gemma4_mpk_layer.py -k "split_dense_projection or split_dense_block"'`
  - result: `2 passed`
- `bash -ic 'f1 && PYTHONPATH=$PWD:/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python python3 -m pytest -q tests/unittest/auto_deploy/singlegpu/transformations/library/test_gemma4_mpk_layer.py -k "hybrid_attention_sublayer_single_token or attention_block_pk or attention_sublayer_pk"'`
  - result: `3 passed`

### First full live-layer attempt: current blocker

The first synthetic full live layer helper was added:

- `run_mirage_gemma_full_layer_split_dense_forward_correctness(...)`

Current shape:

- live attention sublayer
- live dense FFN branch
- live split-dense MoE branch
- final `hidden_out = ffn_down + moe_reduce`

What blocked first:

- Mirage rejected the FFN GELU inputs due to non-canonical stride expectations
  on size-1 leading dims

Fix:

- explicitly materialize packed destination tensors before calling the Mirage
  GELU-mul kernel

What blocks now:

- the generic Mirage GELU-mul compile path inside the full-layer composition is
  extremely slow / CPU-heavy
- this currently looks like a compile-cost issue rather than the earlier fast
  correctness failures

Important interpretation:

- the full-layer path is not numerically validated yet
- but the blocker has moved from obvious wrong math to runtime/compile behavior
  in the activation helper path

### Current recommended next step

Use the now-proven live attention block and live split-dense MoE block as the
stable floor, then make the full-layer helper practical by removing or
replacing the expensive generic activation compile path.

Most likely directions:

1. reuse a cached/proven Mirage activation kernel shape instead of triggering
   fresh generic graph compilation inside the full-layer helper
2. if needed, replace the activation helper with a compiled PK-friendly path
   that Mirage handles more predictably
3. only after the synthetic live full-layer helper finishes and is numerically
   bounded should we wire the same structure into the real runtime path

## 2026-04-11 Gemma4MoE Decode-Only Mirage Runtime Status

### Final status

Gemma4MoE now works end to end on the decode-only Mirage/MPK path and produces
coherent text on the full 30-layer configuration.

The strongest proof run is:

- `gemma_moe_mpk_live30_after_moe_block.log`

Prompt:

- chat-formatted `How big is the universe?`

Observed output start:

- `Because we cannot see everything in existence, astronomers distinguish between two different concepts...`

This means the intended v1 boundary is now real:

- generate-only / decode-only batches -> Mirage runtime path
- non-generate-only batches -> original AutoDeploy path
- `compile_model` / cudagraph remains bypassed for Mirage mode

### What bug fixes actually mattered

#### 1. Attention K-path analyzer bug

The analyzer was incorrectly binding `k_norm` to the same node as `q_norm`.

Evidence:

- `layer_0_q_norm` and `layer_0_v_norm` were near exact
- `layer_0_k_norm` and `layer_0_k_rope` were catastrophically wrong

Fix:

- special-case `getitem(flashinfer_rope, 0/1)` in the analyzer so q and k trace
  back to the correct norm sources

Effect:

- attention-side drift collapsed
- first-token compare moved from clearly wrong to mostly downstream-MoE error

#### 2. MoE stacked weight order bug

The fused MoE stack is in TRT-LLM `w3_w1` order:

- first half = up projection
- second half = gate projection

The runtime had been interpreting the first half as gate and the second as up.

Fix:

- unpack the stacked tensor as `up_weight, gate_weight = chunk(...)`

Effect:

- `layer_0_moe_out`, `layer_0_moe_norm`, and downstream hidden-state error
  dropped dramatically in the torch-shadow compare
- first-token logits moved to:
  - `top1_match=True`
  - `top5_overlap=5`
  - much smaller mean error

#### 3. Use the proven MoE block composition in the runtime

The runtime originally used a per-expert loop of separate dense linears.

The more reliable path was the already-proven split-dense MoE composition:

- router
- `moe_w13_linear`
- GELU gate/up activation
- `moe_w2_reduce`

Fix:

- switch the live runtime to the same semantic MoE block structure that the
  synthetic passing helper already used

Effect:

- real-Mirage first-token compare improved to:
  - `max_abs=7.343750`
  - `mean_abs=0.197819`
  - `top1_match=True`
  - `top5_overlap=5`
- layer-0 router indices matched exactly
- MoE output and hidden-state drift became small enough for coherent
  end-to-end generation

### Runtime policy that now works

The current wrapper policy is:

- if `BatchInfo.is_generate_only()`:
  - run the live Mirage runtime
- else:
  - run the original AutoDeploy graph

This is the right v1 shape for Gemma4MoE:

- Mirage is a decode-only acceleration path
- prefill / mixed non-generate-only work stays on the normal AutoDeploy path
- cudagraph integration is not required for Mirage correctness

### Validation summary

Verified runtime-contract tests:

- `bash -ic 'f1 && python3 -m pytest -q tests/unittest/auto_deploy/singlegpu/transformations/library/test_mpk_runtime_contract.py'`
  - result: `5 passed`

Verified first-token debug/compare shape after the final fixes:

- real Mirage:
  - logits compare:
    - `max_abs=7.343750`
    - `mean_abs=0.197819`
    - `top1_match=True`
    - `top5_overlap=5`
  - layer 0:
    - router indices exact
    - MoE output numerically close
    - final hidden-state drift small enough for coherent decode

Verified full end-to-end decode-only run:

- full 30-layer Gemma4MoE
- coherent answer on a real chat prompt

### Remaining caveats

This path is correct enough for coherent generation, but not yet optimized.

Known non-goals / remaining work:

- first-use compile cost is still very high
- per-token execution still involves many Python-orchestrated kernel calls
- mixed-step decode-subset offload is not implemented
- cudagraph integration for Mirage is still intentionally out of scope

### Recommended next steps

1. Preserve the decode-only boundary and avoid expanding scope to mixed-step
   decode-subset offload yet.
2. Add one lightweight repeatable regression driver for coherent decode-only
   generation.
3. Focus next on compile/execution efficiency:
   - persistent compile caching
   - fewer per-layer/per-token kernel objects
   - more fused / prebuilt executors

## 2026-04-13 Gemma4MoE Single-PK Global Attention Status

This round finally separated the remaining single-PK issue into a clean
Mirage-local pass/fail split.

### 1. The raw global `single_batch_extend` kernel now passes on the current Mirage checkout

We verified the installed Mirage path:

- `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage`

and confirmed Python imports from:

- `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/__init__.py`

The exact Mirage-local repro:

- `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/gemma_global_single_batch_extend_exact_repro.py`

now succeeds on the current checkout for:

- `q_heads = 16`
- `kv_heads = 2`
- `head_dim = 512`
- grouped fused `Q + K + V`
- `grid_dim = (1, 2, 1)`
- `block_dim = (256, 1, 1)`

Observed outcome:

- `compile_ok = true`
- `launch_ok = true`
- `sync_ok = true`

So the old conclusion "raw single_batch_extend is still broken on latest
Mirage" is now stale.

### 2. The remaining standalone failure is the next composition step

We then created a stricter Mirage-local repro for:

- `rmsnorm_layer`
- `linear_layer`
- `single_batch_extend_attention_layer`

File:

- `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/gemma_global_rmsnorm_linear_single_batch_extend_repro.py`

This one reproduces the remaining failure cleanly:

1. `compile-finished`
2. `launch-start`
3. `launch-returned`
4. then no `sync-done` before timeout

That means the current blocker is not the raw global attention kernel anymore.
It is the standalone Mirage-local composition:

```text
rmsnorm -> linear -> single_batch_extend
```

for the real Gemma global geometry.

### 3. TensorRT-LLM-side helpers agree with the Mirage-local split

Against the same current Mirage checkout:

- the pure Mirage-local exact attention repro passes
- the TensorRT-LLM-side `run_mirage_gemma_global_attention_single_pk_smoke(...)`
  still times out
- the TensorRT-LLM-side `run_mirage_gemma_global_layer_single_pk_smoke(...)`
  still times out

This is consistent with the standalone Mirage-local composition repro above.

### 4. Bridge-side contract fixes that were necessary, but not sufficient

Before reaching this split, several real integration bugs were fixed:

- `single_batch_extend` grid now uses KV-head count
- `single_batch_extend` block now matches Mirage demos:
  - `block_dim = (256, 1, 1)`
- shared-KV global layers now expand to grouped fused `Q + K + V`
- grouped rows are ordered per KV head, not packed as all-Q/all-K/all-V
- internal page metadata in the synthetic helper now uses real internal page
  count and last-page length

Those fixes were all correct and required. But they were not enough to solve the
composition failure.

### 5. New practical rule for Mirage bring-up

Do not stop at either of these extremes:

- "raw kernel passes, so Mirage is fine"
- "repo runtime fails, so the raw kernel must be broken"

Add a standalone composition rung in between:

1. raw Mirage kernel repro
2. Mirage-local semantic composition repro
3. repo-side helper
4. layer runtime
5. model runtime

For the Gemma global layer, the working split is now:

1. raw `single_batch_extend_attention_layer`
   - passes
2. `rmsnorm -> linear -> single_batch_extend`
   - fails as a standalone Mirage-local repro
3. TensorRT-LLM global-attention single-PK helper
   - fails
4. TensorRT-LLM full global-layer single-PK helper
   - fails

This is the cleanest escalation point we have had so far for Mirage-side fixes.
