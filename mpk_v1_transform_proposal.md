## MPK v1 Transform Proposal

This note captures the concrete proposal for a first AutoDeploy -> MPK integration path for Gemma4MoE.

The core decision is:

- do **not** use `mlir_fused_*` as the source for MPK lowering
- instead, lower from the post-transform AutoDeploy FX graph **before** MLIR/Triton-style elementwise fusion artifacts become the frontend contract

## Main Decision

For MPK v1, the intended source graph should be:

- after AutoDeploy has already canonicalized major decode semantics
- after cache/metadata have been made explicit
- before `compile_model`
- and without relying on `mlir_fused_*` nodes

In shorthand:

```text
AutoDeploy post-transform canonical FX
  (rope optimized, MoE fused, cached attention inserted)
    -> MPK lowering
      -> compile_model / runtime wrapping
```

## Why Not Use `mlir_fused_*`

`mlir_fused_*` is not a good frontend contract for MPK because:

- it comes from the Triton-oriented MLIR elementwise fusion pipeline
- it is backend-shaped rather than decode-semantic
- if MPK is going to build a large persistent decode artifact anyway, these small fused kernels are not the right abstraction boundary

The MLIR pass explicitly:

1. decomposes high-level ops into primitives
2. discovers fusible primitive subgraphs
3. generates Triton kernels
4. replaces them with `mlir_fused_*`

as described in [mlir_elementwise_fusion.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/mlir_elementwise_fusion.py#L16).

That is useful for the Triton path, but it is not the right source for a large MPK lowering path.

## Why Not Lower Earlier Than Cache Insertion

We should not lower before cached attention is inserted because MPK wants:

- explicit decode metadata
- explicit cache resources
- paged attention style execution inputs

AutoDeploy already creates that representation in `insert_cached_attention`, in [kvcache.py](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/transform/library/kvcache.py#L151).

So for MPK, the useful source graph is **after**:

- RoPE optimization
- fused MoE insertion
- cached attention insertion

## Recommended Insertion Point

The recommended insertion point is:

- `stage: compile`
- before `compile_model`

This is aligned with the existing transform ordering in [default.yaml](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/tensorrt_llm/_torch/auto_deploy/config/default.yaml#L277).

### Proposed ordering for MPK path

```text
post_load_fusion:
  fuse_rmsnorm / rope / fused_moe / existing semantic rewrites
  skip mlir_elementwise_fusion for MPK

cache_init:
  insert_cached_attention
  initialize_cache

compile:
  lower_to_mpk
  compile_model
```

## Config / Pipeline Changes

For the MPK backend path, the intended behavior should be:

1. `mlir_elementwise_fusion` stays disabled or bypassed.
2. Existing semantic canonicalization passes remain enabled.
3. A new transform is added in `compile` stage:
   - example name: `lower_to_mpk`
4. `lower_to_mpk` runs before `compile_model`.

Conceptually:

```yaml
mlir_elementwise_fusion:
  enabled: false   # for MPK path

insert_cached_attention:
  enabled: true

multi_stream_moe:
  enabled: existing behavior

lower_to_mpk:
  stage: compile
  enabled: true

compile_model:
  enabled: true
```

## Source Graph Contract for Gemma4MoE v1

The concrete source graph reference is:

- [078_compile_multi_stream_moe.txt](/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/dev_feat1/TensorRT-LLM/gemma4moe_graph/078_compile_multi_stream_moe.txt)

This graph is already highly canonicalized around AutoDeploy custom ops.

### Supported source op families

For Gemma4MoE v1, the translator should only understand the following op families:

| Source FX op family | Role in graph | MPK v1 handling |
|---|---|---|
| `auto_deploy.triton_paged_prepare_metadata` | attention runtime metadata prep | translate to MPK metadata setup |
| `auto_deploy.torch_linear_simple` | canonical linear op | translate to MPK linear tasks / buffers |
| `auto_deploy.flashinfer_rope` | canonical RoPE op | translate into MPK attention prep / rotary inputs |
| `auto_deploy.triton_paged_mha_with_cache` | canonical cached paged attention | translate to `paged_attention_layer` |
| `auto_deploy.triton_fused_topk_softmax` | canonical MoE routing | translate to routing buffers and MPK top-k routing task |
| `auto_deploy.trtllm_moe_fused` | canonical fused MoE execution | expand into MPK MoE task sequence |

### Important non-target ops

These should **not** be treated as primary translation targets:

| Op family | Why not a primary target |
|---|---|
| `getitem` | tuple unpacking / graph plumbing |
| `aten.view.default` | layout/view bookkeeping |
| `aten.reshape.default` | shape adaptation |
| `aten.to.dtype`, `aten.to`, `aten.to.device` | dtype/materialization detail |
| `aten.slice`, `aten.cat`, `aten.index.Tensor` | operand assembly for already-canonicalized semantic ops |
| `aten.sym_size.int`, `aten.arange` | shape/runtime helper scaffolding |

These can still be interpreted where necessary to recover operands, but they should not define the translation boundary.

## What Remains Model-Specific

Even after avoiding `mlir_fused_*` as the main frontend contract, some helper interpretation still remains.

For Gemma4MoE v1, we still need to interpret:

- the packed Q/K/V linear shape and split structure around `torch_linear_simple`
- the residual/norm handoff structure around attention output
- the FFN gate/up split and swiglu middle
- router prep around routing input

But the crucial point is:

- those are now small structured interpretation problems around already-canonical AutoDeploy ops
- not a dependency on Triton-generated `mlir_fused_*` kernels

## Per-Layer Translation Shape

One repeated Gemma4MoE layer in `078` is structurally:

```text
torch_linear_simple(qkv)
-> split q/k/v
-> flashinfer_rope inputs assembled from q/k norm helpers
-> triton_paged_mha_with_cache
-> torch_linear_simple(o_proj)
-> post-attn handoff helper
-> torch_linear_simple(ffn gate/up)
-> swiglu helper
-> torch_linear_simple(ffn down)
-> router prep
-> torch_linear_simple(router proj)
-> triton_fused_topk_softmax
-> pre-moe norm
-> trtllm_moe_fused
-> final block merge / next-layer handoff
```

For MPK v1, this should lower approximately as:

```text
1. QKV projection setup
2. attention prep (norm/rope operands)
3. MPK paged_attention_layer
4. o_proj + residual/norm handoff
5. dense FFN branch
6. router branch
7. MoE input prep
8. MPK MoE expansion
9. block output merge
```

## Concrete MPK Lowering Intent

### Attention side

| Source AutoDeploy op(s) | MPK target |
|---|---|
| `torch_linear_simple` for fused qkv | `linear_layer` or equivalent projection buffer setup |
| q/k/v prep around RoPE inputs | attention prep logic feeding MPK |
| `flashinfer_rope` | rotary inputs into attention lowering |
| `triton_paged_mha_with_cache` | `paged_attention_layer` |
| `torch_linear_simple` for `o_proj` | `linear_layer` / `linear_with_residual_layer` |

### FFN / MoE side

| Source AutoDeploy op(s) | MPK target |
|---|---|
| `torch_linear_simple` for gate/up | `linear_layer` |
| swiglu helper region | `silu_mul_layer` |
| `torch_linear_simple` for down proj | `linear_layer` |
| router proj + `triton_fused_topk_softmax` | `linear_layer` + `moe_topk_softmax_routing_layer` |
| `trtllm_moe_fused` | `moe_w13_linear_layer` -> `moe_silu_mul_layer` -> `moe_w2_linear_layer` -> `moe_mul_sum_add_layer` |

## Practical v1 Scope

The translator should be explicitly scoped to:

- Gemma4MoE only
- the exact post-transform pattern family seen in `078`
- repeated layers `0..28`
- the special final layer shape variant `29`

It should **not** initially try to support:

- arbitrary models
- arbitrary helper ids
- generic MLIR fused kernels
- logits/head/gather path beyond what is needed for end-to-end smoke testing

## Proposed New Transform

### Name

- `lower_to_mpk`

### Stage

- `compile`

### Input

- post-transform `GraphModule`
- after cached attention insertion
- after fused MoE insertion
- before `compile_model`
- with MLIR elementwise fusion disabled for this path

### Output

One of:

1. an opaque MPK-backed runtime call inserted into FX, or
2. a wrapper module that owns compiled MPK artifacts and is invoked by the executor

For v1, option 2 is probably simpler operationally.

## Why This Is the Right v1

This plan is good because:

- it uses the semantic AutoDeploy custom-op vocabulary that already exists
- it avoids coupling MPK to Triton-specific fusion artifacts
- it preserves explicit cache and metadata contracts
- it keeps the translation surface narrow and testable
- it gives a clear path to first end-to-end Gemma4MoE decode support

## Summary

The recommended MPK v1 path is:

```text
disable mlir_elementwise_fusion for MPK
keep existing semantic canonicalization
lower from post-cache-init canonical FX
insert MPK lowering before compile_model
scope translator to the exact Gemma4MoE op families seen in 078
```

This gives the cleanest possible frontend contract for a first MPK backend without depending on `mlir_fused_*`.
