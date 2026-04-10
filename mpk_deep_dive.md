## MPK Deep Dive

This note captures:

- the current understanding of Mirage/MPK building blocks and execution flow
- what is generic versus what is specialized in MPK
- the likely integration shape for AutoDeploy
- a first concrete mapping table from Gemma4MoE decode graphs to MPK task vocabulary

## Executive Summary

The main conclusion from the code dive is:

- MPK is not primarily a generic "lower any graph into a megakernel" compiler.
- MPK is better understood as a persistent decode runtime with:
  - a small graph IR layer
  - a semantic task registry
  - CUDA code emission for registered task variants
  - a task/event runtime that executes the resulting graph

This matters because the best AutoDeploy integration target is not a primitive math IR. The better target is a decode-semantic IR or canonical region that can be instantiated as MPK tasks plus explicit runtime buffers.

In short:

`AutoDeploy decode graph -> canonical decode-semantic region -> MPK task/buffer instantiation -> generated persistent task graph`

## Layering in Mirage/MPK

### 1. `TBGraph`: threadblock-local graph

The smallest graph abstraction is `TBGraph`, defined in [threadblock.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/threadblock.py).

It exposes a compact threadblock-local DSL over `STensor`s:

- `matmul`
- `add`, `mul`, `sub`, `div`
- `exp`, `silu`, `gelu`, `relu`
- `reduction`, `reduction_max`
- `rms_norm`
- `concat`
- `forloop_accum*`

This is a local compute description, not the full runtime contract.

### 2. `KNGraph`: kernel/global graph

`KNGraph`, exposed in [kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/kernel.py), is the kernel/global graph over `DTensor`s.

It owns:

- input tensor creation
- output marking
- attaching external/runtime buffers
- customized ops
- task registration
- final task-graph generation

Key APIs:

- `attach_torch_tensor`
- `attach_cuda_tensor`
- `attach_nvshmem_tensor`
- `customized`
- `register_task`
- `generate_task_graph`

The Cython bridge is very thin and simply forwards these calls into native code, in [core.pyx](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/_cython/core.pyx#L944).

### 3. `PersistentKernel`: semantic builder + runtime front-end

`PersistentKernel`, in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L262), is the main Python-side MPK builder/runtime interface.

It owns:

- serving/runtime configuration
- meta tensors
- tensor attachment/allocation
- semantic task construction
- compilation and launch wiring

The important point is that `PersistentKernel` is not exposing a fully generic graph lowering API. It mostly exposes semantic decode-serving tasks such as:

- `embed_layer`
- `rmsnorm_layer`
- `rmsnorm_linear_layer`
- `attention_layer`
- `paged_attention_layer`
- `linear_layer`
- `linear_with_residual_layer`
- `allreduce_layer`
- `silu_mul_layer`
- `moe_topk_softmax_routing_layer`
- `moe_w13_linear_layer`
- `moe_silu_mul_layer`
- `moe_w2_linear_layer`
- `moe_mul_sum_add_layer`
- `argmax_layer`

### 4. `MPK` / `MPKMetadata`: runtime envelope

`MPKMetadata` and `MPK` in [mpk.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/mpk.py) package the runtime state around `PersistentKernel`.

The required meta tensors are a major clue about the actual MPK contract. Important runtime tensors include:

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

This is already very close to a decode-serving ABI with explicit metadata/bucket buffers.

## How Tasks Are Actually Built

The common pattern inside `PersistentKernel` is:

1. Assert tensor ranks and shapes.
2. Derive semantic parameters from those shapes.
3. Create a `TBGraph` and wire tensor mappings.
4. Call `self.kn_graph.customized([...], tb_graph)`.
5. Call `self.kn_graph.register_task(tb_graph, "<task_name>", params)`.

This pattern is visible in:

- `rmsnorm_linear_layer`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L434)
- `attention_layer`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L455)
- `paged_attention_layer`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L592)
- `moe_*` layers, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L871)

That means the final semantic commitment is not the raw `TBGraph`. The final semantic commitment is:

- task name
- tensor roles
- derived params

## What `register_task` Really Means

Native `Graph::register_task`, in [graph.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/graph.cc#L439), dispatches on task names like:

- `embedding`
- `rmsnorm`
- `rmsnorm_linear`
- `attention`
- `paged_attention`
- `single_batch_extend_attention`
- `linear`
- `linear_with_residual`
- `silu_mul`
- `identity`
- `moe_*`
- `argmax_*`
- `reduction`
- `find_ngram_*`
- `target_verify_greedy`
- `nvshmem_*`

For each task it records:

- input/output arity
- runtime task enum
- a variant id

So `register_task` is not just tagging an op. It is choosing a concrete runtime kernel family.

## What `TaskRegister` Does

The native `TaskRegister`, in [task_register.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/task_register.cc), is where specialization really happens.

Each `register_*_task(...)` function:

- inspects the `TBGraph`
- extracts concrete tensor shapes/strides
- uses semantic params like number of heads, page size, rotary flags, qk-norm flags
- emits a C++ code snippet that directly calls a runtime kernel implementation
- interns that snippet as a task variant

Examples:

- `register_rmsnorm_linear_task` emits `kernel::norm_linear_task_impl<...>(...)`, using batch size, output size, reduction size, and output stride, in [task_register.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/task_register.cc#L121).
- `register_paged_attention_task` emits `kernel::multitoken_paged_attention_task_impl<...>(...)` and wires in runtime metadata buffers such as `qo_indptr_buffer`, `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, and `paged_kv_last_page_len_buffer`, in [task_register.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/task_register.cc#L226).
- `register_linear_task` emits `kernel::linear_kernel<...>(...)`, in [task_register.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/task_register.cc#L498).
- Hopper-specific linear tasks explicitly instantiate TMA-based variants, in [task_register.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/task_register.cc#L786).

This is the most important architectural finding:

- MPK is flexible in graph assembly.
- MPK is not semantically arbitrary at the kernel/task level.
- Its true ABI is the catalog of registered task families and their metadata/buffer contracts.

## How the Full Task Graph Gets Produced

Once tasks are registered, `Graph::generate_task_graph`, in [runtime.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/runtime.cc#L1306), constructs:

- runtime task descriptors
- dependency events
- first-task lists
- IO config
- emitted CUDA source
- a JSON task graph

This is more than simple kernel codegen. It is building a scheduled runtime graph with event dependencies.

## How Compilation Works

`PersistentKernel.compile()`, in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L1403):

- calls `kn_graph.generate_task_graph(...)`
- writes `test.cu`
- writes `task_graph.json`
- invokes `nvcc`
- loads the produced extension/launcher

So the output is not merely one hand-authored CUDA kernel. It is a generated persistent task runtime artifact.

## Model Builders Are Structured and Explicit

The model builders make the intended usage style very clear.

For example, the Qwen3 builder explicitly allocates and/or attaches named intermediate buffers such as:

- `embed_out`
- `rmsnorm_out`
- `attn_in`
- `attn_out`
- `attn_proj_out`
- `all_reduce_buf`
- `attn_allreduce_out`
- `mlp_mid`
- `silu_mul_out`
- `mlp_out`
- `mlp_final`
- argmax intermediates

in [builder.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/models/qwen3/builder.py#L114).

This confirms that MPK is comfortable with explicit buffer choreography and named stable intermediates.

## What Looks Generic vs Specialized

### Generic-ish pieces

- tensor and graph plumbing
- IO attachment
- named runtime buffer management
- task/event graph scheduling
- compile/load/launch flow
- some architecture selection across SM90 and SM100

### Specialized pieces

- decode-serving task vocabulary
- runtime metadata contracts for attention and token/state buffers
- expected tensor ranks/layouts
- architecture-specific implementations
- model-builder choreography of buffers and tasks

## Implication for AutoDeploy

The right integration target is likely:

- not a primitive op IR
- not direct lowering of arbitrary FX nodes one by one
- but a decode-semantic canonical form

That canonical form should carry:

- semantic ops
- explicit runtime metadata inputs
- explicit KV/cache/state buffers
- stable bucketized temporary buffers

Then the backend can instantiate MPK objects/tasks from that canonical form.

The right mental model is:

`FX after KV-cache rewrite -> canonical decode region -> MPK task/buffer instantiation`

not:

`raw FX -> arbitrary Mirage primitive graph`

## Gemma4MoE: Findings From AutoDeploy Decode Graph Dumps

The most useful graph points were:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt)
- [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt)

High-confidence observations:

- By `066`, the graph is already decode-shaped.
- Runtime metadata is explicit:
  - `batch_info_host`
  - `cu_seqlen_host`
  - `cu_num_pages`
  - `cu_num_pages_host`
  - `cache_loc`
  - `last_page_len`
  - `last_page_len_host`
  - `seq_len_with_cache_host`
- Per-layer KV cache tensors are explicit:
  - `r0_kv_cache` through `r29_kv_cache`
- Cached attention is already represented as:
  - `auto_deploy.triton_paged_mha_with_cache.default(...)`
- Q/K norm and RoPE are explicit before paged attention:
  - `auto_deploy.mlir_fused_*` for q/k norm
  - `auto_deploy.flashinfer_rope(...)`
- MoE routing and execution are explicit:
  - `auto_deploy.triton_fused_topk_softmax(...)`
  - `auto_deploy.trtllm_moe_fused(...)`

This is strong evidence that AutoDeploy already has most of the decode semantics we need. The missing step is canonicalization and naming, not inventing semantics from scratch.

## Gemma4MoE -> MPK Mapping Table

The table below is a first-pass mapping based on the graph dumps and current MPK task vocabulary.

| AutoDeploy / Gemma4MoE pattern | Evidence in graph dumps | Closest MPK building block | Mapping confidence | Notes / gaps |
|---|---|---|---|---|
| Explicit runtime decode metadata (`batch_info_host`, `cu_seqlen_host`, `cu_num_pages`, `cache_loc`, `last_page_len`, `seq_len_with_cache_host`) | [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) around paged attention calls | `MPKMetadata` meta tensors: `qo_indptr_buffer`, `paged_kv_*`, token buffers, `step`, etc. in [mpk.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/mpk.py) | High | Needs a canonical translation layer from AD metadata schema to MPK metadata schema. This is a schema mapping problem, not a kernel gap. |
| Per-layer KV cache buffers (`r0_kv_cache` ... `r29_kv_cache`) | [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `paged_attention_layer(...)` expects explicit `k_cache` and `v_cache` tensors, in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L592) | Medium | AD appears to use a packed per-layer KV object, while MPK currently models separate `k_cache` and `v_cache`. Likely needs cache unpacking or a small MPK adapter task/buffer convention. |
| QKV projection / attention input projection | The graph feeding q/k/v views into norm + RoPE before paged attention | `linear_layer(...)` or `splitk_linear_layer(...)` in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L1045) | High | Canonical op should probably be `qkv_proj` or `attn_in_proj`, then lower either to one fused projection buffer or a sequence of MPK linear tasks. |
| Q norm / K norm | `auto_deploy.mlir_fused_*` q/k norm ops, e.g. lines near 85/92 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `attention_layer(...)` / `paged_attention_layer(...)` support `q_norm` and `k_norm` inputs via flags/params, in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L455) and [task_register.cc](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/src/kernel/task_register.cc#L226) | High | Good semantic fit. Better to canonicalize as part of attention-prep than keep them as separate primitive ops. |
| RoPE application | `auto_deploy.flashinfer_rope(...)`, e.g. lines near 100 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `attention_layer(...)` / `paged_attention_layer(...)` accept `cos_pos_embed` / `sin_pos_embed` and a rotary flag, in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L455) | High | Strong fit. Canonical op should carry RoPE semantically, not as an unrelated call. |
| Cached paged decode attention | `auto_deploy.triton_paged_mha_with_cache.default(...)`, e.g. lines near 103 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) and near 4997 in [081_compile_compile_model.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/081_compile_compile_model.txt) | `paged_attention_layer(...)` and split-k variants in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L592) | High | This is the cleanest and most important direct mapping. Metadata schema translation is the main work. |
| Attention output projection (`o_proj`) | `auto_deploy.torch_linear_simple.default(..., self_attn_o_proj_weight, None)`, e.g. lines near 105 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `linear_layer(...)` or `linear_with_residual_layer(...)` in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L1045) | High | If residual add is already folded around the projection boundary, `linear_with_residual_layer` may be a better canonical target. |
| Post-attention residual add / projection + residual | Implicit in the graph around attention output and subsequent block inputs | `linear_with_residual_layer(...)`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L1071) | Medium | Need a cleaner canonicalization pass to identify whether Gemma4MoE post-attention residual is best represented as a separate residual op or a fused `linear_with_residual`. |
| Dense RMSNorm + linear patterns outside attention | Graph shows several `mlir_fused_*` plus linear sequences | `rmsnorm_layer(...)` and `rmsnorm_linear_layer(...)`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L416) | Medium | Need exact subgraph matching to know where Gemma4MoE lines up with MPK's fused RMSNorm+linear assumptions. |
| Router logits for MoE | `torch_linear_simple.default(... x 128)` feeding `triton_fused_topk_softmax`, e.g. line near 126 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `linear_layer(...)` for router logits + `moe_topk_softmax_routing_layer(...)`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L892) | High | Good fit if canonicalized as explicit `moe_route_logits -> moe_route_select`. |
| Top-k routing + routing indices | `auto_deploy.triton_fused_topk_softmax(...)`, e.g. lines near 126-128 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `moe_topk_softmax_routing_layer(...)`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L892) | High | Likely one of the easiest MoE mappings. Need to align exact routing mask/index layout. |
| Fused MoE execution | `auto_deploy.trtllm_moe_fused(...)`, e.g. lines near 132 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | Sequence of `moe_w13_linear_layer(...) -> moe_silu_mul_layer(...) -> moe_w2_linear_layer(...) -> moe_mul_sum_add_layer(...)`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L918) | Medium | Semantic fit is good, but AD has this as one fused op today. Lowering will probably need to explode this one op into multiple MPK MoE tasks or add a higher-level MPK MoE composite builder. |
| Expert weights in stacked layout | `fused_moe_w3_w1_stacked_* : 128x1408x2816`, `fused_moe_w2_stacked_* : 128x2816x704`, e.g. line near 132 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `moe_w13_linear_layer(...)` expects `(num_experts, 2*intermediate_size, hidden_size)` and `moe_w2_linear_layer(...)` expects `(num_experts, hidden_size, intermediate_size)`, in [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L918) | Medium | Layouts look broadly compatible in spirit, but exact dimension order and grouped experts-per-token conventions need validation. |
| Final combine with residual after MoE | Result of `trtllm_moe_fused` reshaped back to hidden size, e.g. lines near 132-133 in [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/066_cache_init_insert_cached_attention.txt) | `moe_mul_sum_add_layer(...)`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L992) | Medium | Very likely the right semantic target, but needs careful confirmation against Gemma residual placement. |
| LM head / token selection | Not the focus of the decode-block dump, but MPK has final-token tasks | `linear_layer`, `argmax_layer`, `argmax_partial_layer`, `argmax_reduce_layer`, [persistent_kernel.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/persistent_kernel.py#L1199) | Medium | This is downstream of the block mapping and can come later. |
| Mixed-batch decode representation | AD graph uses stable graph plus runtime metadata tensors rather than changing graph topology | MPK already uses explicit batched metadata tensors in `MPKMetadata`, [mpk.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python/mirage/mpk/mpk.py#L13) | High | Strong architectural fit. The issue is metadata schema translation, not conceptual mismatch. |

## Recommended Canonical Decode-Semantic Ops For Gemma4MoE

Based on the graph dumps and the current MPK task vocabulary, a first canonical region for one Gemma4MoE decode block could look like:

1. `attn_qkv_proj`
2. `attn_qk_norm_rope`
3. `paged_cached_attention`
4. `attn_out_proj_residual`
5. `ffn_or_pre_moe_norm`
6. `moe_route_logits`
7. `moe_topk_route`
8. `moe_expert_w13`
9. `moe_silu_mul`
10. `moe_expert_w2`
11. `moe_weighted_sum_residual`

Explicit region inputs should include:

- hidden state
- layer weights
- q/k norm weights
- KV cache handles/buffers
- batch metadata
- paged KV metadata
- position embedding tables or equivalent RoPE inputs

Explicit region outputs should include:

- next hidden state
- updated cache view / cache side effects
- optional debug/intermediate handles only if needed for staging

## High-Confidence First Implementation Strategy

The first practical integration path looks like:

1. Start from the AutoDeploy graph after KV-cache rewrite.
2. Canonicalize Gemma4MoE decode blocks into the semantic ops listed above.
3. Translate runtime metadata into the MPK metadata schema.
4. Instantiate MPK buffers/tasks rather than lowering primitive nodes one by one.
5. Let MPK compile and own the decode task graph artifact.

## Main Gaps To Resolve

The main unresolved items are:

1. Exact metadata schema mapping between AutoDeploy and MPK for mixed decode/paged cache execution.
2. KV cache representation mismatch:
   - AutoDeploy appears to use packed per-layer KV objects.
   - MPK currently expects explicit `k_cache` and `v_cache`.
3. Whether `auto_deploy.trtllm_moe_fused` should:
   - be exploded during canonicalization into several semantic MoE ops, or
   - map into a new MPK composite helper that emits the existing sequence of MoE tasks.
4. Where residual placement in Gemma4MoE best aligns with:
   - separate residual ops
   - `linear_with_residual`
   - `moe_mul_sum_add`

## Bottom Line

The deep dive supports the following design direction:

- Canonicalize AutoDeploy decode graphs into semantic decode regions.
- Instantiate those regions as MPK tasks/buffers.
- Treat MPK as a semantic persistent task runtime, not as a primitive op backend.

For Gemma4MoE specifically, the strongest direct matches are:

- paged cached attention
- q/k norm + RoPE carried semantically into attention
- router top-k softmax
- staged MoE task sequence
- stable bucket/buffer style runtime metadata

So the path forward looks real. The main work is semantic normalization and buffer/metadata schema alignment.
