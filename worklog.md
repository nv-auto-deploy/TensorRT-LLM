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

It is:

- "how do we canonicalize AutoDeploy's post-cache-init decode graph into a stable semantic region that MPK can own?"

## Suggested Next Step

The next concrete step is to define the first Gemma4MoE decode-semantic IR sketch:

- exact semantic ops
- exact region inputs / outputs
- mapping from current FX ops in the late-stage dump into those semantic ops
- what remains outside the MPK region for v1

That work should use the late post-cache-init graph as the source of truth, not the raw exported FX graph.
