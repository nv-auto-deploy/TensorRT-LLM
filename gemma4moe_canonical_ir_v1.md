## Gemma4MoE Canonical Decode IR v1

This note turns the earlier Gemma4MoE-to-MPK mapping into a concrete v1 canonical IR/schema proposal for AutoDeploy.

The goal is not to define a universal IR for every model up front. The goal is to define a first decode-oriented semantic region format that:

- matches the current Gemma4MoE decode graphs after KV/cache rewrite
- preserves the runtime metadata that mixed-batch decode and CUDA graph style execution need
- is close enough to MPK task vocabulary that lowering is mostly structural

## Scope

This v1 schema is for:

- decode-oriented execution
- post-KV-cache AutoDeploy graphs
- Gemma4MoE as the driving model
- single-GPU first, with explicit hooks for future multi-GPU/allreduce

It is intentionally not trying to cover:

- arbitrary FX graphs
- generic prefill semantics
- every possible model family
- every existing fused op in AutoDeploy

## Design Principles

The schema should:

1. Be semantic rather than primitive.
2. Keep runtime state explicit.
3. Preserve stable bucket-style input/output/buffer contracts.
4. Be close to MPK task vocabulary.
5. Allow a later xDSL dialect if desired, but not depend on xDSL-specific machinery.

## Where This Fits

The intended pipeline is:

`AutoDeploy FX after KV-cache rewrite -> canonical decode region IR -> MPK task/buffer instantiation -> MPK compile -> opaque runtime call`

This means the canonical IR is not the final execution format. It is the normalization boundary between AutoDeploy FX and the MPK builder/runtime.

## Evidence From Current Graphs

The core graph evidence comes from:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt)
- [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt)

Important observations:

- External graph inputs are already decode-runtime friendly:
  - `input_ids`
  - `position_ids`
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
  - `r0_kv_cache` ... `r29_kv_cache`
  from [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt#L8)
- Per decode block, the structure is very regular:
  - input hidden state
  - fused QKV projection
  - split Q / K / V
  - q norm / k norm / v norm
  - RoPE
  - paged cached attention
  - attention output projection
  - fused post-attn residual + norm handoff
  - dense FFN path
  - router path
  - fused MoE execution
  - fused post-FFN residual + norm handoff
  from [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L76)

That regularity is enough to define a v1 schema.

## Canonical Region Boundary

The basic canonical unit should be:

- one decode transformer block region

For Gemma4MoE, one canonical block begins at the block input hidden state and ends at the hidden state handed to the next block.

Conceptually:

`block_input_hidden -> attention subregion -> FFN/MoE subregion -> block_output_hidden`

This is better than using tiny fusion regions because:

- attention needs explicit cache/runtime metadata
- MoE needs explicit routing state and expert weights
- MPK already thinks in terms of semantic task sequences

## Region Inputs

Each canonical block region should carry explicit inputs in four groups.

### 1. Activation inputs

- `hidden_in`
  - shape: `[num_tokens, hidden_size]` or bucketized decode equivalent
- optionally `residual_in` if we choose not to model it as implicit block state

For the current graph, the model starts from `[batch, token, hidden]` and often flattens to `[num_tokens, hidden]` before routing/MoE. The canonical IR should standardize on a logical token-major activation view even if the underlying FX graph still uses shaped views.

### 2. Runtime metadata inputs

- `token_ids`
- `position_ids`
- `batch_layout`
  - canonical wrapper for `batch_info_host`
- `qo_indptr`
  - canonical wrapper for `cu_seqlen` / `cu_seqlen_host`
- `paged_kv_indptr`
  - canonical wrapper for `cu_num_pages` / `cu_num_pages_host`
- `paged_kv_indices`
  - canonical wrapper for `cache_loc`
- `paged_kv_last_page_len`
  - canonical wrapper for `last_page_len` / `last_page_len_host`
- `seq_len_with_cache`
  - canonical wrapper for `seq_len_with_cache` / `seq_len_with_cache_host`
- `prepared_attn_metadata`
  - optional derived metadata object if we want to keep `triton_paged_prepare_metadata` as a preprocessing op outside the block

The key is that the canonical IR should not preserve the raw current naming. It should rename this to an execution-oriented metadata schema that a backend can translate to MPK metadata tensors.

### 3. Cache/state inputs

- `kv_cache_layer`
  - one logical per-layer cache handle for the current block

For v1, I recommend modeling this as one abstract cache input in the canonical IR, even though MPK may currently want `k_cache` and `v_cache` separately. The canonical layer should not overfit the current MPK representation.

### 4. Weight/config inputs

- `qkv_proj_weight`
- `q_norm_weight`
- `k_norm_weight`
- `v_norm_weight`
- `o_proj_weight`
- `post_attn_norm_weight`
- `pre_ffn_norm_weight`
- `ffn_gate_up_proj_weight`
- `ffn_down_proj_weight`
- `router_root_size`
- `router_scale`
- `router_proj_weight`
- `pre_moe_norm_weight`
- `fused_moe_w13_weight`
- `fused_moe_w2_weight`
- `post_ffn_norm_1_weight`
- `post_ffn_norm_2_weight`
- `post_ffn_norm_weight`
- `layer_scalar`
- `next_block_input_norm_weight`
- `rope_tables`
  - canonical view of cos/sin data

We may later group these into structured attributes instead of passing every item as a separate op operand, but semantically they are all part of the block contract.

## Region Outputs

Each canonical block should produce:

- `hidden_out`
  - next block hidden state
- `block_state_out`
  - optional alias if we want to explicitly carry both residual and normalized state
- `kv_cache_layer_out`
  - if the cache update is modeled as an effectful output rather than an in-place side effect

For v1 I would keep the canonical form simple:

- main value output: `hidden_out`
- cache modeled as an effectful resource input/output

## Region Temporary Buffers

The canonical IR should make room for named temporaries even if the frontend does not materialize them directly.

Suggested canonical temporaries:

- `qkv_proj_out`
- `q_tensor`
- `k_tensor`
- `v_tensor`
- `q_rope`
- `k_rope`
- `attn_out`
- `attn_proj_out`
- `ffn_gate`
- `ffn_up`
- `ffn_gated`
- `ffn_down`
- `router_logits`
- `router_scores`
- `router_indices`
- `moe_in`
- `moe_w13_out`
- `moe_act_out`
- `moe_w2_out`
- `moe_out`

These do not all need to appear in the persisted IR if a pass is using higher-level ops, but they are useful as the conceptual storage interface for MPK lowering.

## Canonical Semantic Op Set v1

For Gemma4MoE decode, I recommend the following first semantic op set.

### 1. `decode_embed_scale_norm`

Purpose:

- consume input token ids at the model entry
- perform embedding, scale, and initial norm handoff into block 0

This is outside the per-block repeating region, but it is worth naming because it is the true graph entry in [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt#L53).

### 2. `attn_qkv_proj_split`

Purpose:

- project hidden input into fused QKV
- split into Q / K / V logical tensors

Inputs:

- `hidden_in`
- `qkv_proj_weight`

Outputs:

- `q_tensor`
- `k_tensor`
- `v_tensor`

### 3. `attn_qkv_norm_rope`

Purpose:

- apply q/k/v norm as needed
- apply RoPE to Q and K

Inputs:

- `q_tensor`
- `k_tensor`
- `v_tensor`
- `q_norm_weight`
- `k_norm_weight`
- `v_norm_weight`
- `position_ids` or canonical prepared positions
- `rope_tables`

Outputs:

- `q_ready`
- `k_ready`
- `v_ready`

Notes:

- Even if V norm is only a model-specific detail, keeping it in this semantic op prevents the canonical IR from fragmenting into tiny primitive ops.

### 4. `paged_cached_attention`

Purpose:

- perform decode attention against paged KV cache
- update/use per-layer cache state

Inputs:

- `q_ready`
- `k_ready`
- `v_ready`
- `attn_runtime_metadata`
- `kv_cache_layer`

Outputs:

- `attn_out`
- updated cache effect on `kv_cache_layer`

Notes:

- This should be the central semantic op for decode.
- This is the strongest current fit to MPK `paged_attention_layer`.

### 5. `attn_out_proj_residual_norm`

Purpose:

- project attention output back to hidden size
- apply residual combination
- produce both:
  - the value feeding later block residual state
  - the normalized value feeding FFN/MoE

Inputs:

- `attn_out`
- `o_proj_weight`
- `hidden_in` or `residual_in`
- `post_attn_norm_weight`
- `pre_ffn_norm_weight`

Outputs:

- `post_attn_residual`
- `ffn_input`

Notes:

- This directly reflects the paired outputs seen from `auto_deploy.mlir_fused_f27ca72ba0322556(...)` in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L107).

### 6. `ffn_gate_up_swiglu_down`

Purpose:

- compute the dense FFN side path
- used as one contributor to the post-FFN merge logic

Inputs:

- `ffn_input`
- `ffn_gate_up_proj_weight`
- `ffn_down_proj_weight`

Outputs:

- `ffn_down`

Notes:

- In the current graph this appears as:
  - fused gate/up projection
  - split
  - swiglu-like fused activation/mul
  - down projection

### 7. `moe_router_prep`

Purpose:

- prepare router input from residual-side hidden state
- include any model-specific scaling/root normalization needed before logits

Inputs:

- `post_attn_residual`
- `router_root_size`
- `router_scale`

Outputs:

- `router_input`

Notes:

- This captures the current `to.float -> mlir_fused -> type_as -> mlir_fused` routing-prep sequence without leaking those primitives.

### 8. `moe_route_topk`

Purpose:

- compute router logits
- select top-k experts and scores

Inputs:

- `router_input`
- `router_proj_weight`

Outputs:

- `router_scores`
- `router_indices`

### 9. `moe_pre_norm`

Purpose:

- produce the MoE expert input activation

Inputs:

- `post_attn_residual`
- `pre_moe_norm_weight`

Outputs:

- `moe_input`

Notes:

- This corresponds to the norm right before `trtllm_moe_fused` in the current graph.

### 10. `moe_experts`

Purpose:

- execute the expert path

Inputs:

- `moe_input`
- `router_indices`
- `router_scores`
- `fused_moe_w13_weight`
- `fused_moe_w2_weight`

Outputs:

- `moe_out`

Notes:

- In the canonical IR this can stay as one semantic op initially.
- During MPK lowering it can expand into:
  - `moe_topk_softmax_routing_layer`
  - `moe_w13_linear_layer`
  - `moe_silu_mul_layer`
  - `moe_w2_linear_layer`
  - `moe_mul_sum_add_layer`

This keeps the canonical IR stable while allowing backend-specific expansion.

### 11. `block_output_merge_norm`

Purpose:

- combine dense FFN output and MoE output
- apply post-FFN residual/norm logic
- produce the next block input state

Inputs:

- `ffn_down`
- `moe_out`
- `post_attn_residual`
- `post_ffn_norm_1_weight`
- `post_ffn_norm_2_weight`
- `post_ffn_norm_weight`
- `layer_scalar`
- `next_block_input_norm_weight`

Outputs:

- `hidden_out`
- optionally `next_block_normed_hidden`

Notes:

- This directly corresponds to the multi-output fused op `auto_deploy.mlir_fused_37fee4ff1d3c3673(...)` seen in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L133).

## Canonical Block Summary

One canonical Gemma4MoE decode block is therefore:

1. `attn_qkv_proj_split`
2. `attn_qkv_norm_rope`
3. `paged_cached_attention`
4. `attn_out_proj_residual_norm`
5. `ffn_gate_up_swiglu_down`
6. `moe_router_prep`
7. `moe_route_topk`
8. `moe_pre_norm`
9. `moe_experts`
10. `block_output_merge_norm`

This is intentionally semantic and compact.

## FX Pattern Folding Rules

This section describes which current AutoDeploy FX subgraphs should fold into each canonical op.

### `decode_embed_scale_norm`

Fold:

- `aten.embedding.default`
- scalar embed scale cast
- initial fused input-layernorm helper

Evidence:

- [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt#L53)

### `attn_qkv_proj_split`

Fold:

- `auto_deploy.torch_linear_simple.default(..., fused_weight_X, None)` producing `8192`
- `_insert_fused_gemm.<locals>.split_output`
- `aten.view.default` into `[B, T, H, D]`-style Q/K/V views

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L76)

### `attn_qkv_norm_rope`

Fold:

- q norm fused helper
- k norm fused helper
- v norm fused helper
- RoPE table indexing / dtype conversion / slicing
- `auto_deploy.flashinfer_rope`

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L85)

### `paged_cached_attention`

Fold:

- optional metadata-prep helper if kept in-block
- `auto_deploy.triton_paged_mha_with_cache.default`
- reshape of attention output back to hidden projection input

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L103)

### `attn_out_proj_residual_norm`

Fold:

- attention output projection linear
- fused op that returns:
  - post-attention residual stream
  - pre-FFN normalized stream

Evidence:

- `torch_linear_simple_default_1`
- `auto_deploy.mlir_fused_f27ca72ba0322556(...)`
from [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L105)

### `ffn_gate_up_swiglu_down`

Fold:

- FFN gate/up linear
- split gate/up outputs
- fused gate activation/multiply op
- FFN down projection

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L109)

### `moe_router_prep`

Fold:

- flatten residual-side hidden
- cast to fp32
- fused router normalization/prep helper
- type restore
- router root/scale helper

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L115)

### `moe_route_topk`

Fold:

- router projection linear
- `auto_deploy.triton_fused_topk_softmax`

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L125)

### `moe_pre_norm`

Fold:

- pre-feedforward-layernorm-2 style helper that feeds the MoE input

Evidence:

- `auto_deploy.mlir_fused_97b5c9bd79b34f2a(...)`
from [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L129)

### `moe_experts`

Fold:

- `auto_deploy.trtllm_moe_fused(...)`
- reshape back to `[B, T, hidden]`

Evidence:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L132)

### `block_output_merge_norm`

Fold:

- final fused block merge op that combines dense FFN output, MoE output, residual state, scalar, and next-layer norm handoff

Evidence:

- `auto_deploy.mlir_fused_37fee4ff1d3c3673(...)`
from [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt#L133)

## Proposed Canonical IR Object Model

This does not need to be xDSL-specific yet, but conceptually each region should have:

- region attributes
  - `block_index`
  - `hidden_size`
  - `num_q_heads`
  - `num_kv_heads`
  - `head_dim`
  - `num_experts`
  - `top_k`
  - `page_size`
- region resources
  - metadata bundle
  - cache bundle
  - weight bundle
- ordered semantic ops
- effect annotations
  - reads cache
  - writes cache
  - reads metadata

## Suggested Resource Bundles

To keep the op list clean, I recommend three structured bundles in the IR.

### `DecodeMetadata`

Contains:

- token ids
- positions
- batch layout
- qo indptr
- paged kv indptr
- paged kv indices
- paged kv last page lengths
- seq lengths with cache

### `LayerCache`

Contains:

- logical per-layer KV cache resource

### `LayerWeights`

Contains:

- all weights for one block

This is nicer than forcing every semantic op to carry many low-level operands.

## Lowering Notes to MPK

This v1 canonical IR is meant to lower into MPK approximately as follows:

- `attn_qkv_proj_split`
  - `linear_layer` or a dedicated fused qkv helper later
- `attn_qkv_norm_rope`
  - feed norm/rope information into MPK attention task setup
- `paged_cached_attention`
  - `paged_attention_layer`
- `attn_out_proj_residual_norm`
  - `linear_with_residual_layer` plus auxiliary norm handling, or a short sequence of MPK tasks
- `ffn_gate_up_swiglu_down`
  - `linear_layer` + `silu_mul_layer` + `linear_layer`
- `moe_route_topk`
  - `linear_layer` + `moe_topk_softmax_routing_layer`
- `moe_experts`
  - `moe_w13_linear_layer`
  - `moe_silu_mul_layer`
  - `moe_w2_linear_layer`
  - `moe_mul_sum_add_layer`

The canonical IR does not need to match MPK 1:1. It only needs to be close enough that the lowering is deterministic and semantically clean.

## Why This Is a Good v1

This v1 schema is useful because:

- it is based on the current real Gemma4MoE graphs, not guesswork
- it absorbs current fused helper noise into stable semantics
- it preserves explicit runtime metadata and cache state
- it gives AutoDeploy one normalization target for decode
- it gives MPK a natural frontend contract

## Recommended Next Step

The next implementation step after this schema is:

1. add a block-pattern recognizer over the post-KV-cache FX graph
2. emit a serialized canonical block description for one or more Gemma4MoE layers
3. prototype one lowering path:
   - `attn_qkv_proj_split`
   - `attn_qkv_norm_rope`
   - `paged_cached_attention`
   - `attn_out_proj_residual_norm`

That would validate the most important interface first before taking on the whole MoE path.
