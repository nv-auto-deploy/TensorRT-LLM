## Gemma4MoE MPK Translator Design

This document describes the design of a first translator from AutoDeploy's no-MLIR-fusion Gemma4MoE FX graph into MPK builder/runtime objects.

Primary reference graphs:

- [066_cache_init_insert_cached_attention.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4_no_mlir_fusion/066_cache_init_insert_cached_attention.txt)
- [078_compile_multi_stream_moe.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4_no_mlir_fusion/078_compile_multi_stream_moe.txt)

## Goal

Translate the post-transform AutoDeploy Gemma4MoE graph directly into MPK, without depending on:

- `mlir_fused_*`
- xDSL IR materialization
- a generic arbitrary-FX recognizer

The translator is intentionally scoped to:

- Gemma4MoE only
- single-GPU first
- the no-MLIR-fusion graph shape
- decode-focused execution

## Why the No-MLIR-Fusion Graph Is Better

The no-MLIR-fusion graph already exposes the semantics we care about in a much cleaner form.

Compared to the `mlir_fused_*` version, the main op families in [078_compile_multi_stream_moe.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4_no_mlir_fusion/078_compile_multi_stream_moe.txt) are:

- `auto_deploy.flashinfer_rms_norm`
- `auto_deploy.torch_linear_simple`
- `auto_deploy.flashinfer_rope`
- `auto_deploy.triton_paged_prepare_metadata`
- `auto_deploy.triton_paged_mha_with_cache`
- `auto_deploy.triton_fused_topk_softmax`
- `auto_deploy.trtllm_moe_fused`

That is a much better translation surface for MPK than the `mlir_fused_*` variant.

## High-Level Architecture

The translator should have three layers:

```text
1. FX graph scanner
   -> collects graph-level resources and per-layer anchors

2. Layer translators
   -> translate one Gemma layer at a time into MPK builder calls

3. MPK artifact builder
   -> owns metadata tensors, weight/cache attachment, intermediate buffers,
      and final compile/load wiring
```

In shorthand:

```text
Gemma FX GraphModule
  -> GemmaGraphInfo
  -> [GemmaLayerPlan x N]
  -> MPKBuilderEmitter
  -> compiled MPK artifact
```

## Translator Input Contract

The translator assumes an FX graph after:

- RoPE optimization
- fused MoE insertion
- cached attention insertion
- cache initialization

and before:

- `compile_model`

It must not require `mlir_elementwise_fusion`.

## Source Op Surface

### Primary translation targets

| Op family | Role |
|---|---|
| `auto_deploy.triton_paged_prepare_metadata` | graph-level metadata prep |
| `auto_deploy.flashinfer_rms_norm` | norm stages used throughout the block |
| `auto_deploy.torch_linear_simple` | all linear projections |
| `auto_deploy.flashinfer_rope` | Q/K rotary preparation |
| `auto_deploy.triton_paged_mha_with_cache` | decode attention |
| `auto_deploy.triton_fused_topk_softmax` | MoE routing |
| `auto_deploy.trtllm_moe_fused` | MoE execution |
| `auto_deploy.gather_tokens` | final decode gather before lm head |

### Secondary structural helpers

These are interpreted but not treated as semantic ops:

- `getitem`
- `aten.reshape.default`
- `aten.view.default`
- `aten.to.dtype`
- `aten.to`
- `aten.type_as.default`
- `aten.arange`
- `aten.index.Tensor`
- `aten.slice`
- `aten.cat`
- `aten.add.Tensor`
- `aten.mul.Tensor`
- `aten.div.Tensor`

## Translation Philosophy

The translator should be:

- direct
- deterministic
- graph-shape aware
- intentionally narrow

It should **not** attempt to rediscover generic semantics from arbitrary FX.
Instead, it should leverage the fact that AutoDeploy has already canonicalized the important parts for Gemma4MoE.

## Core Translator Objects

### `GemmaMpkTranslator`

Top-level driver.

Responsibilities:

- validate graph compatibility
- collect graph-level resources
- construct per-layer plans
- emit MPK graph/buffers/tasks
- return an executable MPK wrapper object

Suggested shape:

```python
class GemmaMpkTranslator:
    def __init__(self, gm, cached_seq_interface, shared_config):
        ...

    def analyze(self) -> GemmaGraphInfo:
        ...

    def lower(self) -> GemmaMpkArtifact:
        ...
```

### `GemmaGraphInfo`

Holds graph-level information extracted from FX.

Suggested fields:

- `num_layers`
- `hidden_size`
- `layer_infos`
- `metadata_nodes`
- `cache_nodes`
- `entry_nodes`
- `final_nodes`
- `layer_shape_variants`

### `GemmaLayerInfo`

Holds references to the key nodes for one layer.

Suggested fields:

- `layer_idx`
- `input_norm_node`
- `qkv_linear_node`
- `q_split_node`
- `k_split_node`
- `v_split_node`
- `q_norm_node`
- `k_norm_node`
- `v_norm_node`
- `rope_node`
- `cached_attn_node`
- `o_proj_node`
- `post_attn_norm_node`
- `pre_ffn_norm_node`
- `gate_up_linear_node`
- `swiglu_mid_node`
- `down_proj_node`
- `router_prep_node`
- `router_proj_node`
- `topk_node`
- `pre_moe_norm_node`
- `trtllm_moe_node`
- `post_ffn_norm_nodes`
- `next_input_norm_node`

This object is not an IR; it is a translation aid.

### `GemmaMpkArtifact`

Represents the lowered MPK object plus runtime-facing metadata.

Suggested fields:

- `mpk`
- `persistent_kernel`
- `buffer_table`
- `layer_runtime_info`
- `compile_cache_key`

## MPK Builder Ownership Model

The translator should explicitly manage:

### Graph-level resources

- token/input buffers
- position inputs
- metadata tensors / pointers
- final output buffers

### Per-layer resources

- layer weights
- layer KV cache handle(s)
- intermediate buffers

### Shared temporary buffers

Depending on v1 simplicity, we can choose either:

1. conservative per-layer dedicated intermediates
2. reused rolling buffers across layers

For v1, I recommend conservative dedicated buffers first for correctness and easier debugging.

## Graph Analysis Phase

The `analyze()` phase should perform:

### 1. Validate graph-level prerequisites

Required graph features:

- one `triton_paged_prepare_metadata`
- repeated `triton_paged_mha_with_cache`
- repeated `trtllm_moe_fused`
- repeated routing path with `triton_fused_topk_softmax`

### 2. Detect layer boundaries

The graph is already mostly linear and repeated. So layer boundaries should be anchored primarily by:

- one `triton_paged_mha_with_cache` per layer
- one `trtllm_moe_fused` per layer

If both exist, they define the layer body robustly.

### 3. Collect shape variants

The graph has at least two attention shape families:

- layers `0..28`
- layer `29`

Observed difference:

- regular layers use `16 q heads, 8 kv heads, head_dim 256`
- final layer uses `16 q heads, 2 kv heads, head_dim 512`

The analyzer should record this explicitly so the emitter does not assume one universal layer shape.

## Layer Translation Shape

For a regular layer, the source pattern looks approximately like:

```text
input_rmsnorm
  -> qkv linear
  -> split q/k/v
  -> q_norm / k_norm / v_norm
  -> flashinfer_rope
  -> triton_paged_mha_with_cache
  -> o_proj linear
  -> post-attn rmsnorm + residual handoff
  -> pre-ffn rmsnorm
  -> gate/up linear
  -> swiglu mid
  -> down proj linear
  -> router prep math
  -> router linear
  -> triton_fused_topk_softmax
  -> pre-moe rmsnorm
  -> trtllm_moe_fused
  -> post-ffn merge / norm / next-layer handoff
```

The MPK lowering shape should be:

```text
1. qkv projection setup
2. q/k/v prep
3. paged attention
4. o_proj + residual handoff
5. dense FFN branch
6. router branch
7. MoE input prep
8. MoE task expansion
9. post-FFN merge / next-layer handoff
```

## Detailed Lowering Rules

### 1. Metadata prep

Source:

- `triton_paged_prepare_metadata`

Lowering:

- translate its outputs into MPK metadata tensor bindings
- populate:
  - `qo_indptr_buffer`
  - paged-kv-related metadata
  - any derived request metadata needed by MPK runtime

This should happen once per graph, not once per layer.

### 2. Input embedding / entry norm

Source:

- embedding
- embed scale
- first `flashinfer_rms_norm`

Lowering:

- either:
  - use `embed_layer` + `rmsnorm_layer`
  - or keep a small eager prelude outside the main MPK block for v1

Recommendation:

- for v1, allow a small eager prelude if it simplifies the first implementation

### 3. QKV projection

Source:

- `torch_linear_simple` producing fused QKV
- split helper + views

Lowering:

- use MPK `linear_layer`
- allocate logical `q`, `k`, `v` intermediate buffers
- route split outputs into those buffers

### 4. Q/K/V norm and RoPE

Source:

- `flashinfer_rms_norm` on q/k/v views
- `flashinfer_rope`

Lowering:

- map norm weights explicitly
- feed q/k inputs through MPK attention prep path
- keep v_norm result as direct V input to attention

### 5. Cached attention

Source:

- `triton_paged_mha_with_cache`

Lowering:

- direct map to `paged_attention_layer`

This is the strongest direct MPK match in the graph.

### 6. Attention output projection and handoff

Source:

- `torch_linear_simple` for `o_proj`
- surrounding add/norm/residual nodes

Lowering:

- use `linear_layer` or `linear_with_residual_layer`
- explicitly materialize:
  - attention residual stream
  - FFN-normalized stream

### 7. Dense FFN branch

Source:

- gate/up `torch_linear_simple`
- swiglu middle
- down projection `torch_linear_simple`

Lowering:

- `linear_layer`
- `silu_mul_layer`
- `linear_layer`

### 8. Router branch

Source:

- router prep arithmetic
- router projection `torch_linear_simple`
- `triton_fused_topk_softmax`

Lowering:

- preserve router-prep arithmetic as a small explicit pre-step in the translator
- use MPK:
  - `linear_layer`
  - `moe_topk_softmax_routing_layer`

### 9. MoE input prep

Source:

- `flashinfer_rms_norm` before `trtllm_moe_fused`

Lowering:

- explicit MPK prep buffer
- reuse normalized hidden as MoE input activation

### 10. MoE execution

Source:

- `trtllm_moe_fused`

Lowering:

- expand into MPK tasks:
  - `moe_w13_linear_layer`
  - `moe_silu_mul_layer`
  - `moe_w2_linear_layer`
  - `moe_mul_sum_add_layer`

This is the major backend expansion point.

### 11. Post-FFN merge / next layer handoff

Source:

- post-ffn norms
- add/mul logic
- next layer input norm

Lowering:

- for v1, keep this as explicit translator-managed arithmetic/norm sequencing if no exact MPK task exists
- emit the next layer input activation buffer

This is a place where we should not overreach. A correct explicit sequence is fine in v1.

## Special Case: Final Layer

Layer `29` differs in attention shape:

- `16` q heads
- `2` kv heads
- `512` head dim

The translator must support two layer schemas:

- `RegularLayerSchema`
- `FinalLayerSchema`

These can share most logic but should compute projection and attention dims independently.

## Final Logits Path

The tail of the graph contains:

- final norm
- `gather_tokens`
- LM head linear
- division

Recommendation for v1:

- keep the logits path out of the first MPK lowering if needed
- or lower it as a small postlude after the block stack

Priority should remain on the repeated decode layers first.

## Failure Policy

The translator should fail closed.

If the graph differs from the expected Gemma4MoE surface:

- log a precise incompatibility reason
- skip MPK lowering
- fall back to existing AutoDeploy path

Examples:

- missing `triton_paged_mha_with_cache`
- unexpected number of MoE layers
- unsupported layer shape variant
- unsupported quantized linear op family replacing `torch_linear_simple`

## Where This Should Live

Suggested new code area:

```text
tensorrt_llm/_torch/auto_deploy/mpk/
  translator.py
  gemma_analyzer.py
  gemma_layer_lowering.py
  buffer_planner.py
  runtime_wrapper.py
```

Suggested top-level transform:

```text
tensorrt_llm/_torch/auto_deploy/transform/library/lower_to_mpk.py
```

## Minimal v1 End-to-End Shape

```text
FX GraphModule
  -> GemmaMpkTranslator.analyze()
  -> GemmaGraphInfo
  -> GemmaMpkTranslator.lower()
  -> MPK builder calls + named buffers
  -> MPK compile()
  -> runtime wrapper module
```

## Summary

The translator is intentionally:

- Gemma-specific
- no-MLIR-fusion specific
- direct FX-to-MPK
- narrow in source op surface

It should rely on the semantic custom ops AutoDeploy already exposes, and only interpret the remaining arithmetic and shape helpers needed to connect those ops into a coherent MPK artifact.
