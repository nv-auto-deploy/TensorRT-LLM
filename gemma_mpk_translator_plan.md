## Gemma4MoE MPK Translator Plan

This document is the implementation plan for a first Gemma4MoE AutoDeploy -> MPK translator using the no-MLIR-fusion graphs as the reference surface.

Primary references:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4_no_mlir_fusion/066_cache_init_insert_cached_attention.txt)
- [078_compile_multi_stream_moe.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4_no_mlir_fusion/078_compile_multi_stream_moe.txt)
- [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4_no_mlir_fusion/081_compile_compile_model.txt)

## Objective

Deliver a first working MPK path for Gemma4MoE decode by:

- lowering from the canonicalized no-MLIR-fusion AutoDeploy FX graph
- translating repeated layer structure into MPK builder calls
- compiling an MPK artifact
- integrating it as a compile-stage backend path before `compile_model`

## Non-Goals for v1

- generic model support
- generic FX-to-MPK translation
- support for `mlir_fused_*`
- quantized linear variants beyond the exact graph surface in the reference dumps
- perfect reuse of all existing AutoDeploy piecewise/cudagraph infrastructure

## Phase 1: Lock Down the Source Contract

### Goal

Codify the exact graph surface the translator supports.

### Tasks

1. Enumerate supported source op families from the no-MLIR-fusion graph:
   - `triton_paged_prepare_metadata`
   - `flashinfer_rms_norm`
   - `torch_linear_simple`
   - `flashinfer_rope`
   - `triton_paged_mha_with_cache`
   - `triton_fused_topk_softmax`
   - `trtllm_moe_fused`
   - `gather_tokens`
2. Record unsupported variants explicitly.
3. Detect the two layer shape schemas:
   - regular layers `0..28`
   - final layer `29`

### Deliverable

- a source-contract doc or module-level constant table in code

## Phase 2: Add the Compile-Stage Hook

### Goal

Create a new AutoDeploy transform hook for MPK lowering.

### Tasks

1. Add a new transform, e.g. `lower_to_mpk`.
2. Place it in `stage: compile`.
3. Ensure it runs before `compile_model`.
4. Add config gating so it only runs for MPK mode/backend.
5. Ensure `mlir_elementwise_fusion` remains disabled or bypassed for this path.

### Deliverable

- transform skeleton with config entry

## Phase 3: Build the Graph Analyzer

### Goal

Extract per-layer structure and graph-level resources from FX.

### Tasks

1. Implement `GemmaGraphInfo`.
2. Implement graph-level extraction:
   - metadata-prep node
   - input placeholders
   - cache placeholders
   - final logits/gather tail
3. Implement layer boundary detection using:
   - one cached attention node per layer
   - one MoE node per layer
4. Collect per-layer anchor nodes:
   - qkv linear
   - q/k/v norm
   - rope
   - cached attention
   - o_proj
   - ffn gate/up
   - down proj
   - router proj
   - top-k
   - moe fused
5. Record schema variant per layer.

### Deliverable

- `analyze(gm) -> GemmaGraphInfo`

## Phase 4: Implement MPK Buffer Planning

### Goal

Define the concrete MPK buffers and attached tensors for one translated Gemma graph.

### Tasks

1. Define graph-level buffers:
   - tokens/input/output
   - metadata tensors
   - final outputs
2. Define per-layer buffers:
   - qkv packed
   - q/k/v
   - attention output
   - o_proj output
   - ffn intermediates
   - router intermediates
   - MoE intermediates
3. Decide v1 allocation policy:
   - conservative per-layer dedicated buffers
4. Map AutoDeploy cache placeholders to MPK cache attachment strategy.

### Deliverable

- `buffer_planner.py`

## Phase 5: Implement Layer Lowering

### Goal

Lower one `GemmaLayerInfo` into MPK builder calls.

### Tasks

1. Lower qkv projection:
   - `torch_linear_simple` -> MPK linear setup
2. Lower q/k/v prep:
   - `flashinfer_rms_norm`
   - `flashinfer_rope`
3. Lower cached attention:
   - `triton_paged_mha_with_cache` -> `paged_attention_layer`
4. Lower o_proj and residual handoff:
   - use `linear_layer` or `linear_with_residual_layer`
   - fill remaining arithmetic explicitly if needed
5. Lower dense FFN branch:
   - gate/up linear
   - `silu_mul`
   - down linear
6. Lower router branch:
   - router prep arithmetic
   - router projection
   - top-k routing
7. Lower MoE prep and execution:
   - pre-MoE norm
   - `trtllm_moe_fused` expansion into MPK MoE tasks
8. Lower post-FFN merge / next-layer handoff explicitly.

### Deliverable

- `gemma_layer_lowering.py`

## Phase 6: Implement Graph-Level MPK Emission

### Goal

Turn `GemmaGraphInfo` into a complete MPK artifact.

### Tasks

1. Initialize `MPKMetadata`.
2. Create `MPK` / `PersistentKernel`.
3. Attach graph-level resources.
4. Emit entry embedding/norm handling.
5. Emit all layer lowerings in order.
6. Emit final norm / gather / lm-head path, or keep a small eager tail for v1 if needed.
7. Compile the MPK artifact.

### Deliverable

- `translator.py`

## Phase 7: Runtime Integration

### Goal

Expose the compiled MPK artifact back into AutoDeploy as a callable runtime unit.

### Options

1. Wrap MPK in a module and replace the full graph/module body.
2. Replace a large region with an opaque custom-op-like call.

### Recommendation

For v1, use a wrapper module that:

- owns the compiled MPK artifact
- receives the same runtime buffers AutoDeploy executor already prepares
- delegates to MPK for the translated region

### Deliverable

- `runtime_wrapper.py`

## Phase 8: Validation

### Goal

Prove the path works end to end for Gemma4MoE decode.

### Validation ladder

1. Analyzer-only validation
   - does the analyzer recover all 30 layers?
2. Dry-run lowering validation
   - can we emit a complete MPK plan without compiling?
3. Compile validation
   - can MPK compile the emitted artifact?
4. Runtime smoke test
   - can one decode step run?
5. Numerical spot checks
   - compare selected intermediate outputs and final outputs
6. Mixed-batch decode smoke test
   - validate metadata and cache integration

## Suggested File Layout

```text
tensorrt_llm/_torch/auto_deploy/mpk/
  __init__.py
  translator.py
  gemma_analyzer.py
  gemma_layer_lowering.py
  buffer_planner.py
  runtime_wrapper.py
  types.py

tensorrt_llm/_torch/auto_deploy/transform/library/
  lower_to_mpk.py
```

## Immediate Implementation Order

This is the recommended coding order.

### Step 1

Create the transform skeleton and config gating.

### Step 2

Implement `GemmaGraphInfo` and analyzer-only extraction.

### Step 3

Add debug dumps:

- print detected layers
- print key node mapping per layer
- print layer schema variant

### Step 4

Implement MPK metadata/buffer planner.

### Step 5

Lower just one regular layer through attention only:

- qkv
- q/k/v prep
- paged attention
- o_proj

### Step 6

Add dense FFN branch lowering.

### Step 7

Add router + MoE lowering.

### Step 8

Handle final layer shape variant.

### Step 9

Integrate final output / lm-head path.

### Step 10

Wire runtime wrapper and end-to-end execution.

## Risks and Mitigations

### Risk 1: MPK cache representation mismatch

Mitigation:

- isolate cache mapping behind one adapter layer in the buffer planner

### Risk 2: Post-attn / post-ffn merge logic has no exact MPK task

Mitigation:

- keep explicit translator-managed arithmetic/norm sequencing in v1

### Risk 3: Graph shape drift across AutoDeploy transforms

Mitigation:

- fail closed
- tie v1 to the no-MLIR-fusion Gemma4MoE source contract
- keep analyzer assertions precise

### Risk 4: Final layer differs from regular layers

Mitigation:

- explicit schema-variant handling from the start

## Success Criteria

v1 is successful if:

1. The translator can analyze the no-MLIR-fusion Gemma4MoE graph and recover all layers correctly.
2. It can emit a compilable MPK artifact for decode.
3. That artifact can run one decode step with the AutoDeploy runtime buffers.
4. The resulting outputs are numerically reasonable against the existing AutoDeploy path.

## Summary

The implementation should proceed in this order:

```text
transform skeleton
-> graph analyzer
-> buffer planner
-> one-layer attention lowering
-> FFN lowering
-> MoE lowering
-> final layer support
-> runtime wrapper
-> validation
```

The key to keeping v1 tractable is to stay strictly scoped to the no-MLIR-fusion Gemma4MoE graph surface and not generalize prematurely.
