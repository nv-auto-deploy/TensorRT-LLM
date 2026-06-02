<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash Algorithm Summary

This note is for readers who already understand speculative decoding and have a high-level
mental model of MTP/Eagle. It focuses on the operational differences that matter for an
AutoDeploy implementation.

## One-Sentence Summary

DFlash is a target-conditioned speculative decoding method that predicts a block of draft
tokens in one parallel masked-token draft pass, using a persistent drafter-side context KV cache
derived from accepted target hidden states.

## Current Status & Open Blocker (handoff)

> **Handoff to a CI-resourced cluster.** Full restart instructions: `debug/dflash_restart_handoff.md`.
> Executable plan (per-step gates): `~/.claude/plans/ethereal-frolicking-steele.md`.
>
> **Done:** Spike A — flash_attn contract validated (the op wraps the classic **`flash_attn`
> 2.7.4.post1** package, *not* the separate `flash-attn-4==4.0.0b11` pin). Step 1 ops written:
> `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/dflash_attention.py`. Qwen3-8B target +
> `z-lab/Qwen3-8B-DFlash-b16` draft downloaded + symlinked under `~/dev/model-symlinks/`.
>
> **OPEN BLOCKER (resolve first on the CI cluster):** the PyTorch DFlash *oracle* on the **public**
> `z-lab` checkpoint gives **~0 acceptance** (0.017 accepted/iter vs the unwaived unit test's ≥1.0).
> Both public z-lab drafts store **separate `q_proj`/`k_proj`/`v_proj` + shared embed/lm_head
> (58 tensors)** while TRT-LLM uses a fused `qkv_proj`. Suspected public-vs-internal checkpoint
> divergence or a load/config mismatch (qkv packing / `target_layer_ids` capture / `mask_token_id`).
> Next: obtain the **internal** `Qwen3-8B-DFlash-b16` the CI test (`TestQwen3_8B::test_dflash`) loads
> from `llm_models_root()`, or debug the public path, until the PyTorch reference shows real
> acceptance — it is the oracle the whole AD port validates against (Step 0 + rung-8 parity).

## The Usual Eagle/MTP Mental Model

In Eagle/MTP, the drafter is advanced with accepted tokens so that its internal state matches
the accepted prefix. After the target verifies a speculative block, the drafter effectively does
a small prefix advance over the accepted tokens and then drafts the next tokens.

At a high level:

```text
accepted tokens + target hidden states
  -> run drafter layer(s)
  -> update drafter KV/state cache
  -> produce next draft tokens
```

The important point is that the drafter's persistent state is produced by running the drafter
model autoregressively. The loop is seeded by accepted target-verified tokens plus target
hidden states, and then each draft step also consumes the drafter's own generated tokens and
hidden states as it extends its draft-side state.

## DFlash Mental Model

DFlash also needs the drafter to represent the accepted prefix, but it does not replay accepted
tokens through the full draft transformer to build that state. Instead, it directly converts
target hidden states from accepted tokens into the drafter's K/V space.

At a high level:

```text
accepted target hidden states
  -> DFlash fc
  -> DFlash hidden_norm
  -> DFlash draft-layer K/V projections
  -> append to DFlash context KV cache
```

The next draft pass then uses a query block:

```text
[last_accepted_token, MASK, MASK, ..., MASK]
```

The mask positions are future-token query slots. They all start from the same learned mask
embedding, but they have different positions and attend to the accumulated DFlash context KV
cache. The logits from those mask positions produce the next `K` draft tokens in one draft
forward.

## What The DFlash Context KV Cache Is

The DFlash context KV cache is not the target model KV cache. It is a persistent drafter-side
cache whose rows are derived from target hidden states, then projected into each DFlash draft
layer's K/V representation.

Conceptually:

```text
target hidden states for accepted prefix
  -> DFlash adapter/projection
  -> per-draft-layer K/V rows
  -> persistent DFlash ctx KV
```

This cache persists across decode iterations for the lifetime of the request. It advances only
by accepted target-verified tokens. Rejected draft tokens are not appended.

This is close in purpose to Eagle/MTP's drafter KV cache:

```text
Eagle/MTP drafter cache:
  accepted prefix represented by running the drafter over tokens

DFlash ctx KV cache:
  accepted prefix represented by projecting target hidden states into drafter K/V space
```

## How A DFlash Step Works

For draft length `K`:

```text
1. Target verifies:
   target([last_accepted, previous_draft_1, ..., previous_draft_K])

2. Verification decides how many draft tokens were accepted.

3. DFlash updates its context KV:
   captured hidden states for accepted target positions
     -> fc + hidden_norm
     -> draft-layer K/V projection
     -> append to DFlash ctx KV

4. DFlash drafts the next block:
   query block = [last_accepted_or_bonus, MASK, ..., MASK]
   each draft layer attends over:
     persistent DFlash ctx K/V + current query-block K/V

5. Logits at mask positions produce:
   [next_draft_1, ..., next_draft_K]
```

## Why The MASK Tokens Exist

The masks are future-position query slots. Without them, `draft([last])` would naturally produce
one next-token distribution. With `[last, MASK, ..., MASK]`, the drafter produces separate hidden
states and logits for multiple future positions in one forward pass.

The masks are not the main source of information. The main conditioning signal is the DFlash
context KV cache. The mask slots ask, "what token should go at this future position, given the
accepted-history memory I can attend to?"

## How DFlash Differs From Eagle

The key difference is how the accepted prefix becomes drafter state.

```text
Eagle:
  target hidden states + accepted token inputs
  -> run draft layer(s) autoregressively
  -> drafter KV/state is updated by the draft model's forward pass

DFlash:
  accepted target hidden states
  -> project directly into draft-layer K/V space
  -> DFlash ctx KV is updated without replaying accepted tokens through the full drafter
```

Another difference is the drafting shape:

```text
Eagle-style drafting:
  draft token 1
  draft token 2 conditioned on token 1
  draft token 3 conditioned on token 1, token 2

DFlash-style drafting:
  predict token 1, token 2, token 3 from mask positions in one block forward
```

So DFlash is still speculative decoding: the target model remains the verifier and the final
output follows the target model. The speedup opportunity comes from producing multiple draft
tokens in a single parallel masked draft pass.

## AutoDeploy Implication

For AutoDeploy, DFlash should be represented as a sibling of Eagle/MTP rather than as an Eagle
variant.

Shared pieces can include:

- target hidden-state capture,
- target-logit verification,
- accepted-token bookkeeping,
- output packaging (`new_tokens`, `new_tokens_lens`, `next_draft_tokens`, `next_new_tokens`).

DFlash-specific pieces should include:

- a persistent DFlash context K/V resource,
- an update phase that appends accepted target-derived K/V rows,
- a DFlash draft attention op that reads that resource,
- a masked block drafting path instead of Eagle's serial draft loop.

A clean implementation split would be:

```text
DFlash wrapper/eager orchestration:
  verifies target outputs
  selects accepted target hidden states
  updates DFlash ctx KV

DFlash draft-layer attention op:
  reads DFlash ctx KV
  attends from [last, MASK, ..., MASK] query slots
  does not commit persistent cache updates
```

That keeps the cache commit rule explicit: only target-verified accepted tokens become persistent
DFlash context.

## AutoDeploy First-Pass Implementation Sketch

This section is intentionally scoped to "support DFlash at all." The first pass uses a separate
`DFlashWrapper.forward()` / `modeling_dflash.py` path, with cleanup and unification with Eagle/MTP
deferred.

> **Converged Design Decisions (post-review) — these supersede any conflicting detail in the
> first-pass sketch below.** They were settled through several review rounds (incl. an independent
> reviewer). The executable, step-by-step plan with file:line references lives in the plan file
> `~/.claude/plans/ethereal-frolicking-steele.md`; this section records *what we decided and why*.

### Attention op + routing

- **Distinct source op `auto_deploy::dflash_attention(q, k, v, ctx_len, scale, …)` whose eager body
  delegates to `torch_attention(q, k, v, is_causal=False)`.** *Why:* a distinct op lets us route by
  **op-type** (the existing MLA/SSM idiom) — a dedicated `insert_cached_dflash_attention` transform
  matches it and the default `insert_cached_attention` (which matches `torch_attention`) naturally
  no-ops on the draft GM. This needs **no per-GM backend gate and no `kvcache.py` edit**. Delegating
  the body to `torch_attention` keeps the source/export-mode math the **well-tested non-causal SDPA**
  (good for a future prefill-math parity test). DFlash drafter attention *is* plain non-causal MHA —
  only the runtime cache semantics differ, and those live in the cached op + descriptor.
- **Cached op `auto_deploy::dflash_attention_with_kvcache(q, k, v, slot_idx, ctx_len, ctx_k_cache,
  ctx_v_cache, scale)`** wraps `flash_attn_with_kvcache(..., causal=False)` (same kernel as the
  PyTorch path), `mutates_args` on the ctx caches.
- **Why not reuse `torch_attention` + per-GM backend selection (the original sketch's idea):** it
  would introduce a per-GM backend-routing mechanism, and AD's single KVCacheManager is **not** built
  to mix two *paged* backends across graph modules — `KVPagedResourceHandler.__eq__` is spec-only
  (`head_dim/dtype/kv_factor/kv_layout/sliding_window`), so two different paged backends with matching
  specs would silently merge into one pool and mis-execute. We deliberately keep DFlash special-purpose
  via op-type routing. (Long-term, general per-GM backends should be solved with one KVCacheManager
  per GM — a documented TODO, out of v1.)

### `block_size` vs `max_draft_len` (terminology — the original sketch wrongly equated these)

- **`block_size`** — the drafter's *intrinsic, trained* parallel-prediction width, **read from the
  draft `config.json`** (`16` for the `…-b16` drafts, `10` for the Llama draft). It is the width of the
  query block `[last, MASK×(block_size-1)]` and the size of the ctx-cache slack. The masked block pass
  is **non-causal** — every mask attends to every other mask (confirmed by Spike A) — so the drafter
  **must run at its full trained `block_size`**; you cannot shorten the block without changing the math.
- **`max_draft_len`** — the runtime *speculation depth* (a serving knob, e.g. `4`). We consume the
  block's mask outputs at positions `1..max_draft_len` as draft tokens and discard the rest;
  `tokens_per_gen_step = max_draft_len + 1` is the target *verification* width.
- **Constraint `max_draft_len + 1 ≤ block_size`** (1 last-token slot + `max_draft_len` mask slots). AD
  **validates this and raises** when resolving the draft config (Step 8) — better than the PyTorch
  worker, which silently `clamp`s out-of-range gather indices (`dflash.py:481`).

So `block_size` sizes the query block + ctx slack (the drafter's *capacity*); `max_draft_len` sizes
what's consumed/verified (the *operating point*).

### Context K/V cache + metadata

- **Unpaged dense ctx K/V, one resource per draft attention node** (not the sketch's single 5D
  `[slot, L, …]` tensor), shaped `[max_state_slots, max_ctx + block_size, nkv, hd]` where **`block_size`
  is the draft config's trained query-block width** (≥ `max_draft_len+1`; see terminology above).
  *Why:* `flash_attn_with_kvcache` **appends** the query-block K/V
  in place at `cache_seqlens`, so we need `+block_size` slack; and unpaged dense resources **bypass
  the KVCacheManager** (exactly Eagle's paged-target + unpaged-`hidden_states_cache` pattern), avoiding
  the paged-mixing hazard above. `max_ctx = max_seq_len`, so the cap never fires (no clamp/divergence;
  AD exposes no `max_ctx` knob).
- **`ctx_len` is a declared input of the draft graph**, not a SequenceInfo field and not a persistent
  counter. *Why:* it equals the committed sequence length (`ctx_len = input_pos + num_accepted`), and
  making it a forward parameter (→ export placeholder, carried by the source op, *retrieved* by
  cached-attention insertion) avoids any `SequenceInfo` surgery. The wrapper computes it each step and
  passes it in; `slot_idx` is the standard SequenceInfo arg.

### Drafter state-building (precompute)

- **Port `precompute_context_kv` 1:1 as eager pure-PyTorch in the wrapper** (Triton idea dropped;
  routing it through a separate exported GraphModule is a *profiling-gated follow-up*). *Why:* the
  PyTorch precompute is already built on hand-written cross-layer fusions (one stacked `_fused_kv_weight`
  GEMM, one fused k-norm, one fused RoPE), so eager == the reference exactly; the only fusions left on
  the table are compiler glue-fusions the PyTorch backend also doesn't capture, and the precompute runs
  on a handful of accepted tokens (negligible). Eager also keeps the algorithm auditable and out of the
  graph-compile machinery. The TRT-LLM/AD draft **model** uses a **fused `qkv_proj`** (sliced to build
  the stacked `_fused_kv_weight` for precompute), built **eagerly** at load (never lazily in forward →
  cuda-graph-safe). ⚠ But the public z-lab **checkpoint** stores **separate `q_proj`/`k_proj`/`v_proj`**
  (+ shared embed/lm_head from target; 58 tensors), so the weight loader must **pack separate→fused
  qkv** at load (standard QKV packing). See the Status/Blocker note.

### Runtime / bring-up

- **Target attention backend = `trtllm`; no host sync between target and draft**
  (`sync_before_hidden_state_capture=False`). *Why:* the Eagle `torch.cuda.synchronize()` exists only
  for FlashInfer's internal-stream behavior; with trtllm everything is on one stream and the eager
  commit→draft-read ordering holds.
- **Phased compile-backend bring-up:** `torch-simple` + overlap-off → `torch-simple` + overlap-on →
  `torch-cudagraph`. *Why:* correctness first. Note the **whole wrapper forward is cuda-graph-captured**
  (`compile_model.run_per_gm: false`), so under cudagraph the commit (precompute + masked scatter +
  `ctx_len` advance) must be alloc-free / sync-free / fixed-shape — hence cudagraph is a deliberate
  phase-3 validation, not free. FlashInfer + cudagraph for DFlash is a separate follow-up.

### Factory / export preservation

- **`DFlashWrapper` is a sibling of `EagleWrapper`** (own class; does not reuse Eagle's serial draft
  loop or hidden-state ordering). Output is `EagleWrapperOutput`-shaped; **`Eagle3OneModelSampler` is
  reused** (gated on wrapper-vs-PyTorch output parity).
- **Hidden states are concatenated in `target_layer_ids` order** (not Eagle's lexicographic
  resource-name sort), because `fc` was trained for that order.
- **Export-preservation refactor:** each draft model declares `submodules_to_preserve() -> list[str]`
  and a generic `DraftModelExportInfo.post_process` preserves them (`set_submodule` + keepalive
  sentinel). DFlash declares `[fc, hidden_norm, per-layer qkv_proj + k_norm, rotary]` (+ eager fused-KV
  buffers). Eagle keeps a thin tail for its accessor rebinds / `d2t` / `dtype`. **Target models do not
  declare anything — embeddings (via `expose_graph_module_accessor`) are enough**, so DFlash reuses
  `TargetModelExportInfo` unchanged. Build on the `gramnarayan/qwen3-vlm-mtp` helpers
  (`insert_keepalive_sentinel`, `expose_graph_module_accessor`) and plan to rebase on that branch.
  *(The full "share everything except the drafting algorithm" spec-dec base is a follow-up PR.)*

### Reference model + scope

- **Reference pair = Qwen3-8B + `z-lab/Qwen3-8B-DFlash-b16`** (via HF ids; draft `block_size=16`,
  `target_layer_ids=[1,9,17,25,33]`, `mask_token_id=151669`, 5 draft layers, `model_type=qwen3`).
  *Why:* its PyTorch DFlash GSM8K test is **unwaived** — a trustworthy oracle (reference accuracy
  ≈ **87.11**); `Qwen/Qwen3-8B` is an AD-supported target (support matrix + `modeling_qwen3.py`); and
  the draft is Qwen3 regardless of target, so target+draft are **uniform Qwen3**. (We moved off the
  Llama pair, whose PyTorch DFlash test is CI-waived, for a trustworthy oracle.) **v1 = `world_size ==
  1`** (multi-GPU is a later milestone).

### De-risking order (front-loaded spikes)

- **Spike A (day 1, no AD code):** standalone `flash_attn_with_kvcache(causal=False)` contract probe.
- **Spike B (before the wrapper):** tiny toy-export probe confirming the `ctx_len`-as-graph-input
  threading (placeholder → carried by source op → *retrieved* by insertion).
- Each implementation step then has a concrete gate test (cached-op parity modeled on
  `test_torch_attention_op.py`; `precompute_context_kv` near-exact parity; black-box wrapper-vs-PyTorch
  output parity; E2E accept-rate ≈ the Step-0 reference). See the plan file for the full per-step gate
  table.

### Current PyTorch DFlash Attention Path

The current PyTorch DFlash draft path does not call the normal `Attention` module forward for the
DFlash cross-attention operation. In `DFlashForCausalLM.dflash_forward(...)`, it manually runs the
drafter layer stack:

```text
noise/query hidden states
  -> layer input norm
  -> qkv_proj
  -> q/k norm + RoPE handling
  -> flash_attn.flash_attn_with_kvcache(..., causal=False)
  -> o_proj
  -> post-attn norm + MLP
```

The important call is:

```text
flash_attn_with_kvcache(
  q=query_q,
  k_cache=dflash_ctx_k_cache_for_layer,
  v_cache=dflash_ctx_v_cache_for_layer,
  k=query_block_k,
  v=query_block_v,
  cache_seqlens=dflash_ctx_len,
  cache_batch_idx=dflash_slot_idx,
  causal=False,
)
```

So the "bidirectional" part is not a TRTLLM attention mode here. It is the `causal=False` flag on
FlashAttention's K/V-cache kernel. The query block can attend over the accepted-prefix context and
the whole current `[last, MASK, ..., MASK]` query block.

One subtlety: `flash_attn_with_kvcache` appends the current query-block K/V into the passed
`k_cache` / `v_cache` at `cache_seqlens`. PyTorch DFlash handles this by allocating each request's
ctx cache with `block_size` slack beyond the persistent context length, where `block_size` is the
drafter's trained query-block width read from its config (`getattr(draft_model,'block_size')`,
e.g. 16 / 10 — *not* `max_draft_len+1`, which is only the fallback). See the terminology note above. The transient query-block K/V lives in that slack
region and is overwritten on the next step; the persistent `ctx_len` only advances by accepted
target tokens.

### Do We Have An AutoDeploy Equivalent Today?

Not exactly.

AutoDeploy already has several attention custom ops:

- `auto_deploy::trtllm_attention_mha_with_cache` wraps TRTLLM `thop.attention` and uses
  `KVPagedResourceHandler` / `KVCacheManager` metadata.
- `auto_deploy::flashinfer_attention_mha_with_cache` wraps FlashInfer paged attention and uses
  FlashInfer planning metadata.
- `auto_deploy::torch_cached_attention_with_cache` is a torch reference cached-attention path.
- `auto_deploy::torch_attention` is the export-time source op that later gets replaced by one of
  the cached attention backends.

None of those is the PyTorch DFlash operation, because DFlash wants:

- dense per-request K/V caches, not necessarily paged TRTLLM/FlashInfer caches,
- explicit `cache_seqlens` and `cache_batch_idx`,
- in-place append of the current query-block K/V into temporary slack,
- `causal=False` for the block-masked draft pass,
- no commit of current query-block K/V into persistent state.

The first-pass AD implementation adds a **distinct source op** `auto_deploy::dflash_attention` (whose
eager body delegates to the well-tested `torch_attention(is_causal=False)`) plus a cached op
`auto_deploy::dflash_attention_with_kvcache` that wraps `flash_attn_with_kvcache(..., causal=False)`
directly. Routing is by **op-type** — a dedicated `insert_cached_dflash_attention` transform lowers
the distinct source op, exactly as MLA/SSM register their own source ops and transforms — so DFlash
gets the same underlying kernel as the PyTorch branch through a parallel, self-contained lowering path
rather than by overloading the normal target attention-backend replacement.

### AutoDeploy First-Pass Plan

The first AutoDeploy DFlash implementation is a direct, separate DFlash path. It reuses Eagle/MTP
infrastructure where the contract is identical, but it does not try to make DFlash look like
Eagle/MTP internally. The accepted-prefix state, attention lowering, and cache commit rules are
different enough that they get their own wrapper, cached op, and resource contract.

#### 1. Add A DFlash Modeling And Export Path

Create a `modeling_dflash.py` path with a `DFlashWrapper.forward()` and a DFlash draft module. The
export path should trace through the DFlash-specific modules: shared embeddings/lm head as needed,
`fc`, `hidden_norm`, draft layers, mask-token embedding, and non-causal `dflash_attention` sites
(the op's body is non-causal SDPA via `torch_attention`).

The prefill/source version is an export scaffold. It should build a valid FX graph with the same
module boundaries and replacement sites that cached mode will use. It does not need to be a
full-fidelity DFlash correctness oracle in v1; dummy or zero hidden-state inputs are acceptable as
long as the graph contains the expected DFlash draft attention sites and all required modules are
present.

#### 2. Add A DFlash Cached Attention Op

AutoDeploy needs DFlash-specific custom ops + an attention descriptor — not a new CUDA kernel.

- **Source op (distinct):** `auto_deploy::dflash_attention(q, k, v, ctx_len, scale, …)`. Its eager
  body delegates to `torch_attention(q, k, v, is_causal=False)`, so the source/export-mode math is the
  canonical, well-tested non-causal SDPA. It is a *distinct* op (not `torch_attention` itself) so that
  routing can be done purely by op-type (see #3). It carries `ctx_len` as an argument even though the
  SDPA math ignores it, so the `ctx_len` graph input is not pruned and flows through to the cached op
  (see #5).
- **Cached op:** `auto_deploy::dflash_attention_with_kvcache(q, k, v, slot_idx, ctx_len, ctx_k_cache,
  ctx_v_cache, scale, out=None)`, wrapping the same FlashAttention kernel as the PyTorch branch:

```text
flash_attn_with_kvcache(
  q,
  ctx_k_cache,
  ctx_v_cache,
  k=k,                       # current query-block K to append
  v=v,                       # current query-block V to append
  cache_seqlens=ctx_len,
  cache_batch_idx=slot_idx,
  causal=False,
)
```

The cached op must declare `ctx_k_cache` / `ctx_v_cache` as **mutated** (`mutates_args`), because
`flash_attn_with_kvcache` appends the current query-block K/V into the cache slack region. The DFlash
wrapper controls persistence by writing only accepted target-derived K/V rows before the op runs; the
query-block append is temporary scratch, not committed state.

The DFlash `AttentionDescriptor` maps `source op: auto_deploy::dflash_attention` →
`cached op: auto_deploy::dflash_attention_with_kvcache`. It sanity-checks that the source attention is
DFlash-shaped (especially `is_causal=False`, expected layout, no unsupported mask/dropout args), so a
causal node never gets DFlash-lowered. `get_cache_initializers` returns the slack-sized unpaged ctx
K/V resource handlers (see #4), and `get_standard_metadata_args = [slot_idx, ctx_len]`. Register it
with `@AttentionRegistry.register("dflash")`.

#### 3. Route Target And DFlash Attention By Op-Type

User-facing `attn_backend` stays target-facing (`trtllm` for the reference config). "TRTLLM attention
with DFlash" means TRTLLM attention for the target graph and DFlash cached attention for the draft
graph:

```text
target model graph:
  causal target attention -> replaced per user attn_backend (e.g. trtllm)

DFlash draft graph:
  non-causal dflash_attention -> replaced with auto_deploy::dflash_attention_with_kvcache
```

Routing is by **op-type**, not by a per-graph backend switch — the same idiom MLA/SSM already use
(distinct source op + dedicated `insert_cached_*` transform):

- the draft layer emits the distinct `dflash_attention` op;
- a dedicated `insert_cached_dflash_attention` transform (registered in `config/default.yaml`,
  backend `dflash`) matches only `dflash_attention` nodes and lowers them;
- the default `insert_cached_attention` matches only `torch_attention`, of which the draft GM has
  none, so it no-ops there naturally.

This needs **no `is_dflash_draft` routing marker, no per-GM backend gate, and no `kvcache.py` edit**.
(`sub_gm.is_draft = True` is still set, as Eagle does, so sharding/collective transforms skip the
draft GM.)

**Why not the earlier idea of reusing `torch_attention` + a per-GM backend switch:** that would
introduce a per-GM backend-routing mechanism, edging toward a general "any backend per GM" framework.
AD's single KVCacheManager is not built to mix two *paged* backends across graph modules —
`KVPagedResourceHandler.__eq__` is spec-only (`head_dim/dtype/kv_factor/kv_layout/sliding_window`) and
both trtllm and flashinfer default to `HND`, so two different paged backends with matching specs would
silently merge into one pool and mis-execute. DFlash sidesteps the whole hazard by being **unpaged**
(see #4) and routed by op-type. The general per-GM-backend solution (one KVCacheManager per GM) is a
documented follow-up, out of v1.

#### 4. Register DFlash Ctx K/V As Explicit Unpaged AD Resources

Register DFlash ctx K/V as explicit `CachedSequenceInterface` resources — **one K and one V resource
per draft attention node** (AD mints a resource per attention node, so there is no separate
`num_dflash_layers` dimension):

```text
ctx_k_cache (per draft layer): [max_state_slots, max_ctx + block_size, num_kv_heads, head_dim]
ctx_v_cache (per draft layer): [max_state_slots, max_ctx + block_size, num_kv_heads, head_dim]
block_size = draft config's trained query-block width (e.g. 16 for b16; >= max_draft_len+1)
max_ctx    = max_seq_len         # so the cap can never fire; no clamp/divergence
```

These are **unpaged dense** resources that bypass the paged `KVCacheManager` entirely — the same
pattern Eagle already ships (paged target KV + unpaged `hidden_states_cache`). Only
`KVPagedResourceHandler` resources are manager-owned; unpaged ones are allocated locally. This is what
keeps the target's paged trtllm cache and the draft's dense dflash cache from ever contending for a
manager pool. (`UnpagedResourceHandler` has no `+block_size` slack, so DFlash uses its own slack-sized
handler; the `+block_size` slack is where `flash_attn_with_kvcache` appends the transient query-block
K/V.)

This is the main difference from the PyTorch backend, whose `DFlashWorker` buffer/slot/length
mechanics (`_ctx_len`/`clamp`, `_free_slots`/`_req_to_slot`, `_lazy_init`) are **PyExecutor-worker
plumbing AD does not run** — AD replaces them with `SequenceInfo.slot_idx`, resource handlers, and a
derived context length:

```text
PyTorch backend:           DFlashWorker owns dense buffers + ctx-length counters + slot dict.
AutoDeploy v1:             CachedSequenceInterface owns the dense ctx K/V resources; slot_idx comes
                           from SequenceInfo; the context length is derived (see #5); the compiled
                           graph receives stable resource args for CUDA-graph capture.
```

Resource contract: slot-indexed dense layout; stable request-slot lifetime; `block_size` slack after
the persistent context; mutation annotation on the cached op; no persistent commit of rejected draft
tokens or of the temporary query-block K/V; room for a manager-backed/paged layout later as a separate
project. **Do not add a persistent `dflash_ctx_len` resource** — the length is a derived, declared
graph input (see #5).

#### 5. Derive DFlash Attention Metadata

The DFlash attention op needs the dense ctx K/V resource, `cache_seqlens` per request,
`cache_batch_idx` per request, query positions, and `causal=False`. The context length equals the
committed sequence length, so:

```text
cache_batch_idx = SequenceInfo.slot_idx            # standard SequenceInfo arg
cache_seqlens   = input_pos + num_accepted_tokens  # == drafter ctx_len
write positions = input_pos + arange(num_accepted_tokens)
query positions = cache_seqlens + arange(block_size)
```

For extend/spec-verify, `input_pos` is the first target-verify position for the iteration; the wrapper
writes accepted target-derived K/V rows starting at `input_pos`, and passing
`input_pos + num_accepted` as `cache_seqlens` makes the query-block append land after the freshly
committed rows.

**`ctx_len`/`cache_seqlens` is a declared input of the draft graph — not a `SequenceInfo` field and
not a persistent counter.** It is a parameter of the DFlash draft model's `forward` (→ an export
placeholder), carried by the source `dflash_attention` op, and *retrieved* by cached-attention
insertion (`get_standard_metadata_args = [slot_idx, ctx_len]`: `slot_idx` is *added* from
`SequenceInfo`; `ctx_len` is *retrieved* from the existing placeholder). The wrapper computes it each
step and passes it into the draft GM. This avoids adding any new field to `SequenceInfo`, and because
`max_ctx = max_seq_len` the value can never exceed the buffer (no clamp, no divergence from the target
sequence length).

Implement metadata derivation next to the DFlash wrapper/op first. Then move pieces outward as the
shape stabilizes:

```text
v1:
  derive metadata adjacent to the DFlash wrapper/op

prepare_metadata_host():
  move host-side batch-layout and request-invariant work here

prepare_metadata():
  move graph-local derived tensors here

per-layer loop:
  keep only layer views, q/k/v projection, qk norm, RoPE, and the attention call here
```

This staging keeps the first implementation simple and CUDA-graph-compatible, while leaving a clear
path to shift work into host-side prepare, per-graph prepare, or per-layer operations later.

#### 6. Capture Target Hidden States And Commit Accepted K/V

DFlash uses target hidden states from configured target layers, then converts accepted target
positions into draft-space K/V via the **ported, fused `precompute_context_kv`**:

```text
accepted target hidden states  (gathered in target_layer_ids order)
  -> DFlash fc
  -> DFlash hidden_norm
  -> single fused-KV GEMM across all draft layers (sliced from qkv_proj) + fused k-norm + fused RoPE
  -> scatter into the per-layer ctx_{k,v}_cache resources
```

Hidden-state capture reuses the existing AD capture infrastructure, but the projection/materialization
is DFlash-specific — do not reuse Eagle/MTP normalization assumptions. The collect helper must
concatenate captured layers in **`target_layer_ids` order** (not a lexicographic resource-name sort),
because `fc` was trained for that order. The TRT-LLM/AD draft model uses a fused `qkv_proj` (sliced to
build the stacked fused-KV buffers, **eagerly at load**, never lazily in `forward`). ⚠ The public
z-lab checkpoint stores **separate `q_proj`/`k_proj`/`v_proj`** (fused at load via standard QKV
packing) + shared embed/lm_head — the AD weight loader must do the same packing. In cached mode the wrapper gathers only accepted target positions and writes only
those rows into the ctx K/V resources.

#### 7. Implement The Cached-Mode DFlash Wrapper

The cached-mode wrapper should execute this loop:

```text
1. Run target verification.
2. Compute accepted token counts.
3. Gather accepted target hidden states.
4. Project accepted hidden states through DFlash fc + hidden_norm.
5. Precompute per-layer accepted K/V rows, including q/k norm and RoPE behavior required by the
   drafter.
6. Scatter accepted K/V rows into dflash_ctx_{k,v}_cache at positions derived from input_pos.
7. Build [last_accepted_or_bonus, MASK, ..., MASK] embeddings and query positions.
8. Run DFlash draft layers with auto_deploy::dflash_attention_with_kvcache.
9. Return next_draft_tokens, next_new_tokens, and the normal speculative-output bookkeeping.
```

Rejected draft tokens never enter persistent DFlash ctx K/V. Current query-block K/V is written by
the attention kernel into slack inside the cache resource, but the persistent length does not
advance for those rows.

**Prefill / context-only requests:** there is no `_store_prefill_context` to port — the AD rule is to
commit all captured prompt hidden states into ctx K/V at rows `input_pos + arange(seq_len)` (chunked
prefill appends by absolute position). v1 returns **zero draft tokens** for pure-context batches
(matching the PyTorch worker); drafting begins on the first generate/extend step.

#### 8. Build The Factory And Config Wiring

`DFlashOneModelFactory` mirrors `EagleOneModelFactory`: build target + draft submodules, export them
separately, and expose the shared/algorithm-specific modules. `get_export_infos` returns
`[TargetModelExportInfo, DraftModelExportInfo]`. There are **no separate "graph routing hints"** —
routing is by op-type (see #3).

- **Target:** reuse `TargetModelExportInfo` unchanged. DFlash's target only needs its embedding
  preserved (to embed `input_ids` for the query block; lm_head/final_norm conditionally), exactly what
  it already does via `expose_graph_module_accessor`. Target models declare nothing extra.
- **Draft (export-preservation refactor):** each draft model declares `submodules_to_preserve() ->
  list[str]`, and a generic `DraftModelExportInfo.post_process` preserves them (`set_submodule` +
  keepalive sentinel). DFlash declares `[fc, hidden_norm, per-layer qkv_proj + k_norm, rotary]` (+ the
  eagerly-built fused-KV buffers, registered as buffers on a preserved module). Eagle keeps a thin
  tail for its accessor rebinds / `d2t` / `dtype`. Build on the `gramnarayan/qwen3-vlm-mtp` helpers
  (`insert_keepalive_sentinel`, `expose_graph_module_accessor`) and rebase onto that branch.
- **Config:** `llm_args` accepts `DFlashDecodingConfig` → `model_factory="dflash_one_model"`; relax
  the PyTorch-only guard for the AD path only (`DFlashDecodingConfig.supports_backend` allows
  `"autodeploy"`); set target `attn_backend="trtllm"` and `sync_before_hidden_state_capture=False`;
  resolve `target_layer_ids` / `mask_token_id` / `block_size` from the draft config and **validate
  `max_draft_len + 1 ≤ block_size`** (raise otherwise — PyTorch only silently clamps); validate
  `world_size == 1` for
  v1. Add `is_dflash()` to the `Eagle3OneModelSampler` selection (reused once wrapper output parity
  holds).

The DFlash wrapper/modeling code stays separate for v1 so the cache and attention contracts remain
easy to audit; the full "share-everything-but-the-drafting-algorithm" spec-dec base is a follow-up PR.

#### 9. Validate With An Incremental Test Ladder

Two **front-loaded spikes** de-risk the novel mechanisms before the wrapper/factory is built:

- **Spike A (standalone, no AD):** a `flash_attn_with_kvcache(causal=False)` contract probe — confirm
  it attends only `ctx_len` rows, appends the query block at `ctx_len`, and matches a hand-rolled
  non-causal SDPA over `[ctx ‖ query]`.
- **Spike B (toy export):** a 1-layer draft whose `forward` takes `ctx_len`; confirm `ctx_len` becomes
  a placeholder and is *retrieved* (not `activate_arg`'d) by the dedicated transform.

Then, wiring before accuracy:

- layer/module tests (`fc`, `hidden_norm`, fused-KV projection shape, mask-token embedding,
  query-position construction);
- cached-op micro-parity — model it on
  `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_torch_attention_op.py`: build ctx
  K/V + query block + `slot_idx`/`ctx_len`, call `dflash_attention_with_kvcache`, compare to
  `torch_attention` over `[ctx ‖ query]`, assert the in-place append at `ctx_len`;
- `precompute_context_kv` parity vs the PyTorch reference on identical inputs — same GEMM/RoPE/k-norm
  kernels, so expect **near-exact** parity (not just "close");
- hidden-state capture test (target captures `target_layer_ids`; draft GM skipped; collect order =
  `target_layer_ids`);
- reduced no-weight smoke (tiny target + DFlash draft, `skip_loading_weights=True`, reduced layers);
- isolated draft export — assert `is_draft = True` and the graph contains `dflash_attention` nodes,
  preserved modules present;
- replacement/resource test — `dflash_attention` → `dflash_attention_with_kvcache`, ctx K/V resources
  + metadata wired;
- black-box wrapper-vs-PyTorch output parity (`new_tokens` / `new_tokens_lens` / `next_draft_tokens`):
  the wrapper introduces no new math, so this (plus the E2E accept-rate) is the meaningful gate — not
  tests contorted around private wrapper state;
- (checkpoint-gated) module/weight-loading audit, full-size draft build through ADEngine, then the
  GSM8K accuracy/acceptance test.

The final accuracy goal mirrors the PyTorch DFlash GSM8K style: target accuracy stays correct and
average accepted length shows DFlash is contributing. The reference pair is Qwen3-8B +
`z-lab/Qwen3-8B-DFlash-b16` (gsm8k reference ≈ 87.11, unwaived PyTorch test). Every step has a concrete
gate test — see the plan file for the full per-step gate table.

### Test Shape

Use the existing Eagle draft tests as the closest template — the small single-GPU
`skip_loading_weights=True` / torch-export coverage in
`tests/unittest/auto_deploy/singlegpu/models/test_eagle.py`, and the spec-dec smoke in
`tests/unittest/auto_deploy/singlegpu/smoke/test_ad_speculative_decoding.py` (already targets
`meta-llama/Meta-Llama-3.1-8B-Instruct`). The detailed DFlash ladder is in #9 above and — with
per-step gates and the two front-loaded spikes — in the plan file. New tests live in a new
`tests/unittest/auto_deploy/singlegpu/models/test_dflash.py` plus a cached-op micro-test under the
custom-ops attention tests.

A roadmap **prefill-math** test is worth noting: if accurate prefill-only hidden states become useful,
compare the source-mode `dflash_attention` representation against an explicit PyTorch reference that
concatenates target-derived ctx K/V and query-block K/V.

Existing PyTorch DFlash coverage is also a useful reference point once AD has enough plumbing to
run end-to-end:

- `tests/unittest/_torch/speculative/hw_agnostic/test_dflash.py` for algorithm-level/unit coverage;
- PyTorch backend GSM8K DFlash entries in `tests/integration/defs/accuracy/test_llm_api_pytorch.py`;
- expected accuracy/acceptance references in `tests/integration/defs/accuracy/references/gsm8k.yaml`.

### Cleanup / Unification Roadmap

**This is a follow-up PR, not v1.** v1 ships DFlash as an Eagle *sibling* plus the minimal
`submodules_to_preserve()` export-preservation refactor (#8). After the first DFlash path works, the
next cleanup should separate "generic speculative decoding orchestration" from algorithm-specific
drafter behavior. The goal is a manageable shared layer, not one wrapper that pretends Eagle, MTP,
DFlash, and future algorithms are the same internally.

A useful target shape is:

```text
modeling_spec_dec.py
  shared target/draft wrapper utilities
  shared target verification and acceptance bookkeeping
  shared hidden-state capture plumbing
  shared output packaging

modeling_eagle.py
  Eagle/MTP draft algorithm
  Eagle/MTP hidden-state processing
  Eagle/MTP draft-cache update behavior

modeling_dflash.py
  DFlash draft algorithm
  DFlash hidden-state processing
  DFlash ctx K/V update behavior
```

The shared spec-dec layer can own the flow that is genuinely common:

```text
1. Run the target model.
2. Sample/verify target tokens.
3. Collect target hidden states when the algorithm requests them.
4. Process hidden states when the algorithm provides a processor.
5. Optionally run SA enhancement on the accepted suffix.
6. Build the drafter inputs from accepted tokens, the golden token, and processed hidden states.
7. Call the algorithm-specific draft step.
8. Package the algorithm output into EagleWrapperOutput-compatible fields.
```

The shared module should expose common submodules and accessors that wrappers need:

- target embedding,
- draft embedding,
- draft LM head,
- final normalization when needed,
- `fc` / projection modules when shared wrapper code needs to preserve or expose them,
- any common helpers for filtering kwargs into target or draft graph modules.

The algorithm-specific layer should own the parts where the contracts diverge:

- which target layers to capture,
- whether captured hidden states are raw, target-final-normalized, Eagle3-FC processed, or
  DFlash-projected,
- how accepted tokens update persistent draft state,
- whether the drafter runs autoregressively, as an MTP stack, or as a DFlash masked block,
- which cache resources are persistent and which writes are provisional,
- what data is needed to produce the next draft token block.

This should be broad enough to cover more than Eagle/MTP/DFlash:

- **SA enhancement** fits naturally as a shared optional stage after acceptance and before drafter
  input construction. It can enhance or override the accepted-suffix continuation without changing
  the neural drafter interface. This is the mode we should plan for in AD: support SA as an
  enhancer for neural/algorithmic drafters, matching how PyTorch composes `SADraftEnhancer` with
  MTP/Eagle/PARD-style paths. Standalone SA does not need to be part of the near-term AD cleanup
  roadmap.
- **External drafter / DraftTarget** should fit as an algorithm whose `draft_next_block()` calls a
  separate draft model or draft engine rather than an in-wrapper draft layer stack. This means the
  shared layer must not assume the draft algorithm is a submodule of the target wrapper, or that
  draft state lives in the same graph resources as the target model. It should pass the target
  verification result, accepted prefix state, and optional hidden states across a clean algorithm
  boundary.
- **NGram / user-provided drafter** are non-neural or user-owned variants of the same interface:
  they may not need hidden states, target embeddings, or draft K/V at all. The shared layer should
  make hidden-state collection and processing opt-in.
- **Tree and parallel draft algorithms** should fit if the shared layer delegates drafting behind an
  algorithm interface. Eagle static/dynamic tree drafting, PARD, DFlash, and Medusa-style heads can
  all produce more structure than a simple serial draft chain. The shared output packaging should
  allow linear blocks now but avoid assuming the draft algorithm always emits exactly one serial
  chain forever.
- **Medusa** is less clear from this DFlash pass, but the framework should leave room for it. Medusa
  looks like a target-attached multi-head/tree drafter: it may not require a separate draft model in
  the same way Eagle/DFlash do, and its "draft step" may be closer to reading extra heads from the
  target forward. That still fits if the generic layer only requires an algorithm to produce draft
  tokens plus verification metadata, rather than requiring a particular drafter module shape.

For DFlash specifically, this means the generic wrapper can run target verification and compute
accepted lengths, but DFlash code should still own:

```text
accepted hidden states
  -> fc + hidden_norm
  -> per-layer K/V projection
  -> scatter into dflash_ctx_{k,v}_cache
  -> masked-block draft attention
```

There are two reasonable implementation styles:

- composition: a generic `SpecDecWrapper` calls an algorithm object with hooks such as
  `collect_hidden_states()`, `process_hidden_states()`, `update_state_from_accepted()`, and
  `draft_next_block()`;
- inheritance: `DFlashWrapper` and `EagleWrapper` inherit a base wrapper and override those hooks.

Composition is probably cleaner once DFlash exists, because it keeps algorithm state and resources
in the algorithm object instead of growing a base class with many optional methods. Inheritance is
acceptable if the first cleanup only extracts small shared utilities and does not force a rigid
abstract interface too early.

The main holes to watch during cleanup are:

- output contract: `EagleWrapperOutput` may need a neutral name if it becomes the generic
  spec-dec output type;
- hidden-state contract: the shared layer should not assume Eagle3 or MTP normalization rules;
- cache ownership: the shared layer should not assume every draft algorithm uses a normal
  autoregressive KV cache;
- SA integration: SA enhancement should sit after acceptance and before drafter input construction,
  but it should remain optional and algorithm-aware;
- external drafter boundary: the shared layer should allow draft tokens to come from a separate
  model/engine or user drafter, not only from a graph submodule;
- tree/block outputs: the output type should not overfit to one linear draft chain if Medusa/PARD
  style trees become AD targets;
- CUDA graph compatibility: hook boundaries must produce static shapes and stable resource
  arguments;
- test naming: generic tests should cover shared orchestration, while algorithm tests should cover
  Eagle/MTP/DFlash-specific state updates and drafter loops.
