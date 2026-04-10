## Gemma4MoE to MPK Mapping Rationale

This note explains the rationale behind the proposed Gemma4MoE canonical decode blocks and why their boundaries were chosen.

The key question is:

- are these semantic block boundaries chosen because MPK already has a close enough semantic unit?
- or are they chosen because they are natural/stable decode boundaries even if MPK lowers them into multiple tasks?

The answer is:

- some boundaries are chosen because MPK already has a close semantic grain
- some are chosen because they are stable decode-phase boundaries
- some are chosen to absorb frontend/FX noise that should not leak into the canonical IR

So the boundary rule is not:

- "create a block only when MPK already has an exact same op"

The real rule is:

- choose a boundary when:
  - the graph is performing one coherent decode function
  - the interface across that boundary is stable
  - MPK can consume it either directly or by a short deterministic expansion

In shorthand:

```text
Canonical boundary
  = stable decode meaning
  + stable tensor/resource interface
  + reasonably direct MPK lowering
```

## Why We Need a Canonical Layer At All

The current FX graph already contains the right decode semantics, but they are mixed with:

- helper-specific fusion ops
- `getitem` plumbing
- `view` / `reshape`
- dtype conversions
- backend-specific op names

If we lowered this directly, we would couple the frontend to graph noise.

The canonical layer is meant to preserve:

- attention phases
- routing phases
- expert execution phases
- block handoff phases

while hiding:

- split/getitem/view clutter
- helper-fusion artifacts
- backend-specific implementation detail

## Big Picture

Current FX block shape is roughly:

```text
hidden
  -> fused qkv linear
  -> split q/k/v
  -> q norm, k norm, v norm
  -> rope
  -> paged attention with cache/meta
  -> o_proj
  -> fused residual + norm handoff
  -> dense ffn path
  -> router prep
  -> router topk
  -> moe fused
  -> fused output merge + next-layer handoff
```

Canonicalized shape is:

```text
hidden
  -> attn_qkv_proj_split
  -> attn_qkv_norm_rope
  -> paged_cached_attention
  -> attn_out_proj_residual_norm
  -> ffn_gate_up_swiglu_down
  -> moe_router_prep
  -> moe_route_topk
  -> moe_pre_norm
  -> moe_experts
  -> block_output_merge_norm
```

The point is not to mirror MPK exactly. The point is to normalize the frontend into stable decode semantics that lower naturally to MPK.

## Three Kinds of Boundaries

There are really three kinds of boundary decisions.

### 1. Direct MPK match

These blocks are chosen because MPK already has a very close semantic unit.

Examples:

- paged attention
- routing top-k
- MoE expert substeps
- some residual/projected linear patterns

### 2. Natural decode handoff boundary

These are chosen because the block transition itself is stable and meaningful, even if MPK needs several tasks underneath.

Examples:

- `attn_qkv_proj_split`
- `attn_out_proj_residual_norm`
- `block_output_merge_norm`

### 3. Frontend noise cleanup boundary

These are chosen because the current FX graph has clutter that should be normalized before lowering.

Examples:

- `attn_qkv_norm_rope`
- `moe_router_prep`
- `ffn_gate_up_swiglu_down`

## Boundary-by-Boundary Rationale

### `attn_qkv_proj_split`

Current FX shape:

```text
hidden
  -> fused qkv linear
  -> split_output
  -> getitem q/k/v
  -> view q/k/v
```

ASCII sketch:

```text
hidden
  |
  v
+-------------------+
| fused qkv linear  |
+-------------------+
          |
          v
     [QKV packed]
      /   |   \
     /    |    \
    v     v     v
   Q      K      V
```

Closest MPK semantic grain:

- not an exact single MPK task
- closest lowering target is a projection task plus logical Q/K/V roles

Why this boundary exists:

- this is the stable point where one hidden activation becomes three attention roles
- `split`, `getitem`, and `view` are plumbing, not semantics

Why it is not smaller:

- splitting into `linear`, `split`, and `view` leaks frontend mechanics

Why it is not larger:

- if we merged norm/rope into it, we would lose a reusable and separately matchable attention-prep phase

Conclusion:

- this boundary is mainly a stable decode handoff boundary, not a direct MPK-op match

### `attn_qkv_norm_rope`

Current FX shape:

```text
Q -> q_norm
K -> k_norm
V -> v_norm
Q,K + rope tables + positions -> rope
```

ASCII sketch:

```text
Q ----> q_norm ---\
                   \
K ----> k_norm ---- +--> rope --> Q_ready, K_ready
                   /
V ----> v_norm ---/
```

Closest MPK semantic grain:

- MPK attention tasks already encode q/k norm and rotary semantics as part of attention setup

Why this boundary exists:

- this is one coherent attention-preparation phase
- it gathers model-specific prep that should not leak into lower layers

Why it is not smaller:

- primitive separation would expose dtype casts, slices, concat, and helper-specific detail

Why it is not larger:

- merging it into full attention would make attention harder to reason about and harder to match independently

Conclusion:

- this boundary is partly motivated by MPK semantic grain and partly by frontend normalization

### `paged_cached_attention`

Current FX shape:

```text
Q_ready, K_ready, V_ready, metadata, cache
  -> triton_paged_mha_with_cache
```

ASCII sketch:

```text
Q_ready ---\
K_ready ----+--> paged_cached_attention --> attn_out
V_ready ---/
meta ------/
cache ----/
```

Closest MPK semantic grain:

- `paged_attention_layer`

Why this boundary exists:

- this is the core decode semantic unit
- the runtime metadata and cache interface are already explicit

Why it is not smaller:

- the current op is already semantic

Why it is not larger:

- attention output projection/residual is a separate block transition

Conclusion:

- this is the strongest direct MPK match in the whole mapping

### `attn_out_proj_residual_norm`

Current FX shape:

```text
attn_out
  -> o_proj
  -> fused post-attn residual/norm handoff
```

ASCII sketch:

```text
attn_out
   |
   v
 o_proj
   |
   +-------- residual combine --------+
   |                                  |
   v                                  v
post_attn_residual                ffn_input
```

Closest MPK semantic grain:

- partial fit to `linear_with_residual_layer`
- norm-related parts may need extra lowering logic

Why this boundary exists:

- this is the stable handoff from attention phase into FFN/MoE phase
- the fused helper appears to produce two meaningful streams:
  - residual-carrying state
  - FFN-ready normalized state

Why it is not smaller:

- splitting projection from residual/norm would lose the fact that this is one stable block transition

Why it is not larger:

- FFN/MoE is the next logical phase and should stay separate

Conclusion:

- this is mainly a stable block-transition boundary with a partial MPK match

### `ffn_gate_up_swiglu_down`

Current FX shape:

```text
ffn_input
  -> fused gate/up linear
  -> split gate/up
  -> activation * multiply
  -> down projection
```

ASCII sketch:

```text
ffn_input
   |
   v
+------------------+
| gate/up linear   |
+------------------+
      /      \
     v        v
   gate      up
     \        /
      \      /
       v    v
      swiglu
         |
         v
      down_proj
         |
         v
      ffn_down
```

Closest MPK semantic grain:

- close to `linear_layer` + `silu_mul_layer` + `linear_layer`

Why this boundary exists:

- this is one coherent dense FFN branch

Why it is not smaller:

- smaller boundaries would expose temporary packing, split/getitem noise, and activation plumbing

Why it is not larger:

- router/MoE logic is semantically separate

Conclusion:

- this is a natural decode-phase boundary with a deterministic MPK expansion

### `moe_router_prep`

Current FX shape:

```text
post_attn_residual
  -> flatten
  -> cast fp32
  -> fused router prep math
  -> cast back
```

ASCII sketch:

```text
post_attn_residual
   |
   v
router_prep
   |
   v
router_input
```

Closest MPK semantic grain:

- no exact single MPK task

Why this boundary exists:

- Gemma-specific router conditioning should be normalized away from raw primitive ops
- this keeps routing decision logic separate from routing input preparation

Why it is not smaller:

- primitive separation would preserve model-specific clutter in the canonical IR

Why it is not larger:

- combining it with route-topk would make it harder to support models with different prep but the same routing semantics

Conclusion:

- this is mostly a frontend normalization boundary

### `moe_route_topk`

Current FX shape:

```text
router_input
  -> router linear
  -> triton_fused_topk_softmax
```

ASCII sketch:

```text
router_input
   |
   v
router_logits
   |
   v
topk_route
   |
   +--> router_scores
   |
   +--> router_indices
```

Closest MPK semantic grain:

- close to `linear_layer` + `moe_topk_softmax_routing_layer`

Why this boundary exists:

- this is the stable expert-selection decision boundary

Why it is not smaller:

- separating logits from top-k has little semantic value for the frontend IR

Why it is not larger:

- expert execution should remain separate from routing

Conclusion:

- this is a near-direct MPK-aligned semantic boundary

### `moe_pre_norm`

Current FX shape:

```text
post_attn_residual
  -> pre-moe norm helper
```

ASCII sketch:

```text
post_attn_residual
   |
   v
moe_pre_norm
   |
   v
moe_input
```

Closest MPK semantic grain:

- no exact single MPK task, but a stable expert-input preparation stage

Why this boundary exists:

- it isolates the activation that the experts actually consume

Why it is not smaller:

- splitting helper internals is not useful semantically

Why it is not larger:

- it is useful to keep expert-input activation separate from routing and separate from expert execution

Conclusion:

- this is mainly a frontend semantic normalization boundary

### `moe_experts`

Current FX shape:

```text
moe_input + routing + expert weights
  -> trtllm_moe_fused
```

ASCII sketch:

```text
moe_input ----+
              |
router -------+--> moe_experts --> moe_out
weights ------+
```

Closest MPK semantic grain:

- not one exact MPK task
- best viewed as a deterministic MPK expansion:
  - `moe_topk_softmax_routing_layer`
  - `moe_w13_linear_layer`
  - `moe_silu_mul_layer`
  - `moe_w2_linear_layer`
  - `moe_mul_sum_add_layer`

Why this boundary exists:

- frontend semantic IR should keep "expert execution" as one coherent phase
- backend can expand it later

Why it is not smaller:

- forcing the canonical IR to match every internal MPK MoE task would overfit the frontend to one backend decomposition

Why it is not larger:

- final block merge is a separate handoff boundary

Conclusion:

- this is a semantic frontend block with a deterministic MPK expansion underneath

### `block_output_merge_norm`

Current FX shape:

```text
ffn_down + moe_out + residual/state
  -> fused merge/norm/next-layer handoff
```

ASCII sketch:

```text
ffn_down ---------\
                   \
moe_out ----------- +--> block_output_merge_norm --> hidden_out
                    /
post_attn_residual-/
```

Closest MPK semantic grain:

- partial fit to residual/merge-style task sequences
- no exact one-op MPK equivalent today

Why this boundary exists:

- this is the true end-of-block semantic boundary
- it produces the next block input state

Why it is not smaller:

- smaller boundaries would expose model-specific fusion debris

Why it is not larger:

- crossing a block boundary is exactly what we want to avoid

Conclusion:

- this is a stable block handoff boundary first, backend match second

## Mapping Table

| Canonical Block | Current FX Span | Closest MPK Semantic Grain | Why This Boundary Exists | Why It Is Not Smaller | Why It Is Not Larger | Confidence |
|---|---|---|---|---|---|---|
| `attn_qkv_proj_split` | fused qkv linear + split + getitem + view | `linear_layer` plus logical q/k/v tensor roles | Stable point where one hidden state becomes Q/K/V roles | `linear`/`split`/`view` are plumbing, not semantics | Merging with norm/rope hides a useful attention-prep phase | High |
| `attn_qkv_norm_rope` | q/k/v norm helpers + rope prep + rope | MPK attention tasks already encode q/k norm and rotary semantics | One coherent attention-preparation phase | Primitive separation exposes helper clutter | Merging into full attention hides a useful normalization stage | High |
| `paged_cached_attention` | paged MHA with cache/meta | `paged_attention_layer` | Core decode semantic unit with explicit metadata/cache | Already semantic | Attention output projection/residual is a separate phase | Very high |
| `attn_out_proj_residual_norm` | o_proj + fused post-attn residual/norm handoff | partial fit to `linear_with_residual_layer` | Stable handoff from attention to FFN/MoE | Splitting projection from residual/norm loses block-transition meaning | FFN/MoE is a separate phase | Medium |
| `ffn_gate_up_swiglu_down` | gate/up linear + split + activation/mul + down proj | `linear_layer` + `silu_mul_layer` + `linear_layer` | One coherent dense FFN branch | Smaller boundaries expose activation/split plumbing | Router/MoE is separate | High |
| `moe_router_prep` | flatten + casts + fused router prep | no exact MPK op | Normalizes Gemma-specific routing prep | Smaller boundaries preserve model-specific clutter | Folding into route-topk mixes prep with selection | Medium |
| `moe_route_topk` | router linear + top-k softmax | `linear_layer` + `moe_topk_softmax_routing_layer` | Stable expert-selection decision boundary | Splitting logits and top-k adds little frontend value | Expert execution should remain separate | Very high |
| `moe_pre_norm` | pre-MoE norm helper | no exact MPK op | Isolates expert-input activation | Helper internals are not semantic | Keeps expert input separate from routing and execution | Medium |
| `moe_experts` | fused MoE execution | deterministic expansion to MPK MoE tasks | One coherent expert-execution phase | Frontend should not overfit to MPK’s exact internal decomposition | Final block merge is a separate handoff | High |
| `block_output_merge_norm` | fused final merge/norm/next-layer handoff | partial fit to residual/merge-style task sequences | True end-of-block boundary producing next block input | Smaller boundaries expose fusion debris | Crossing block boundaries is undesirable | High |

## Current FX -> Canonical Op -> MPK Lowering

| Current FX Pattern | Canonical Op | MPK Lowering |
|---|---|---|
| `torch_linear_simple(..., fused_weight)` -> `split_output` -> `getitem` -> `view` into Q/K/V | `attn_qkv_proj_split` | `linear_layer` or fused projection setup that materializes logical Q/K/V tensors |
| q-norm helper + k-norm helper + v-norm helper + rope table prep + `flashinfer_rope` | `attn_qkv_norm_rope` | feed norm weights and rotary inputs into attention-task setup; either folded into MPK attention instantiation or emitted as a short prep sequence |
| `triton_paged_mha_with_cache.default(...)` with batch/page/cache metadata | `paged_cached_attention` | `paged_attention_layer` |
| attention output reshape + `o_proj` + fused post-attn residual/norm handoff | `attn_out_proj_residual_norm` | `linear_with_residual_layer` plus any extra norm/residual handling needed around it |
| FFN fused gate/up GEMM -> split gate/up -> fused activation*multiply -> down proj | `ffn_gate_up_swiglu_down` | `linear_layer` -> `silu_mul_layer` -> `linear_layer` |
| flatten residual stream -> fp32 cast -> router prep helper -> cast back | `moe_router_prep` | likely frontend-side normalization logic before MPK routing tasks; no exact one-op MPK equivalent today |
| router projection GEMM -> `triton_fused_topk_softmax` | `moe_route_topk` | `linear_layer` -> `moe_topk_softmax_routing_layer` |
| pre-MoE norm helper on residual stream | `moe_pre_norm` | norm-style prep before expert tasks; likely a short prep sequence rather than one exact MPK task |
| `trtllm_moe_fused(...)` | `moe_experts` | `moe_w13_linear_layer` -> `moe_silu_mul_layer` -> `moe_w2_linear_layer` -> `moe_mul_sum_add_layer` |
| dense FFN result + MoE result + residual/state + scalar + next-layer norm handoff fused helper | `block_output_merge_norm` | short residual/merge/norm sequence; no exact one-op MPK equivalent yet |
| embedding + embed scale + initial fused layernorm handoff | `decode_embed_scale_norm` | MPK embed/norm entry setup, likely `embed_layer` plus entry normalization handling |

## Direct Match vs Expansion vs Frontend Normalization

Another useful way to classify the blocks:

### Direct MPK match

- `paged_cached_attention`
- `moe_route_topk`

### Deterministic MPK expansion

- `ffn_gate_up_swiglu_down`
- `moe_experts`
- `attn_out_proj_residual_norm` (partially)

### Frontend normalization boundary

- `attn_qkv_proj_split`
- `attn_qkv_norm_rope`
- `moe_router_prep`
- `moe_pre_norm`
- `block_output_merge_norm`

## Why the Canonical IR Should Not Equal MPK Exactly

If the canonical IR matched MPK task decomposition exactly, it would become backend-shaped instead of model/decode-shaped.

That would push us toward something like:

```text
linear
paged_attention
linear_with_residual
linear
silu_mul
linear
linear
moe_topk_softmax
...
```

That is too low-level for the normalization layer.

The intended layering is:

```text
FX graph
   ->
canonical semantic block
   ->
MPK task expansion
```

not:

```text
FX graph
   ->
MPK internal task list directly
```

## Final Principle

The final principle behind the mapping is:

```text
Choose a canonical boundary when:
1. it represents stable decode meaning
2. it has a stable tensor/resource interface
3. MPK can consume it either:
   - directly, or
   - through a short deterministic expansion
```

That is why some blocks exist because MPK already has a close semantic unit, while others exist because they are the right frontend semantic wrappers for a clean lowering.
