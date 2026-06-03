<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash → AutoDeploy — Restart / Handoff

Self-contained brief to resume the DFlash-for-AutoDeploy port on a **CI-resourced cluster** with a
fresh Claude instance. Read this first, then the two governing docs.

## 0. Governing docs (read in this order)
1. **This file** — status, the open blocker, environment, working contract.
2. **Design summary (authoritative spec):** `debug/dflash_algorithm_summary.md` (in the repo → travels
   with the worktree/branch). This is the *workshop artifact* — see the working contract below.
3. **Executable plan (per-step gates):** `~/.claude/plans/ethereal-frolicking-steele.md`. If `~/.claude`
   did not transfer to the new cluster, this file + the summary contain enough to reconstruct it.
4. **Memory (if `~/.claude` transferred):** `…/memory/project_dflash_autodeploy_onboarding.md` and
   `feedback_dflash_summary_source_of_truth.md`.

## 1. Working contract (MUST follow)
- The **design summary is the source of truth.** Use it *and* the plan as implementation guides.
- **Flag — and pause on — any deviation** from the summary (what the summary says → what you found →
  why it forces a change → options). The user re-workshops the summary on deviations rather than you
  coding around them. (This session already hit: flash_attn version, `block_size` vs `max_draft_len`,
  the checkpoint format, and the oracle blocker — all flagged + folded in.)
- **Reviews** at logical-step boundaries: fan out 4 subagent types — *architect / thorough /
  test-coverage / test-cleanliness* — plus a **Codex** second-opinion (same 4 types) via MCP; collect,
  find patterns, then address. Details in `~/dev/TensorRT-LLM/CLAUDE.local.md`.
- **Testing/exec:** run pytest with `-sv` + `tee` to a log; use `ad-run-agent` for `build_and_run_ad.py`;
  check `nvidia-smi` and set `CUDA_VISIBLE_DEVICES` to free GPUs before GPU runs. Debug logging = plain
  `print`s (no env gating), removed before commit. Throwaway probes live in `debug/spikes/`.

## 1b. Git state (2026-06-02)
- **Committed:** Step 1 tests + Step 2/5 descriptor/transform/resource as `2288b896e6` (post-rebase
  hash); Step 1 ops + docs + spikes as `bfa71ec09e`.
- **REBASED** `gramnarayan/dflash` onto `origin/gramnarayan/qwen3-vlm-mtp` (base now `c23ae34068`)
  — zero conflicts. The export helpers (`insert_keepalive_sentinel`, `expose_graph_module_accessor`
  in `models/hf.py`) are now available for the modeling/export-preservation phase. All 26 DFlash tests
  pass on the new base.
- **Future cleanup:** when `qwen3-vlm-mtp` merges into `main`, `git rebase --onto main
  <old-vlm-mtp-base> gramnarayan/dflash` will auto-drop the vlm-mtp commits (patch-id equivalence);
  if vlm-mtp is *squash*-merged the auto-drop won't match by patch-id but content-identical changes
  resolve trivially. Safety backup ref: `gramnarayan/dflash-prerebase-backup`.

## 2. Environment
- **Worktree:** `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/dflash`
  (also bind-mounted at `~/dev/dflash`). Branch `gramnarayan/dflash`. **Rebase target:**
  `gramnarayan/qwen3-vlm-mtp` (cleans up graph-export into `insert_keepalive_sentinel` /
  `expose_graph_module_accessor` in `models/hf.py` — build the export-preservation refactor on these).
- **venv:** `.venv/bin/python` in the worktree.
- **flash_attn:** the op wraps the classic **`flash_attn` 2.7.4.post1** package (`from flash_attn import
  flash_attn_with_kvcache`) — NOT the separate `flash-attn-4==4.0.0b11` dist (both are installed).
  Signature: `(q, k_cache, v_cache, k=, v=, cache_seqlens=, cache_batch_idx=, softmax_scale=, causal=)`,
  bshd layout, appends query-block K/V in place at `cache_seqlens`. (Verified by Spike A.)
- **Models:** `HF_HOME=/lustre/…/autodeploy_data/hf_home`. Symlinks in `~/dev/model-symlinks/`:
  `Qwen3-8B` (target) + `Qwen3-8B-DFlash-b16` (draft) and `LLaMA3.1-8B-Instruct-DFlash-UltraChat`.
  If the new cluster doesn't share that HF_HOME, re-`hf download Qwen/Qwen3-8B` +
  `z-lab/Qwen3-8B-DFlash-b16` and re-symlink. **`LLM_MODELS_ROOT` here is a *fake* shape-only dir** —
  use HF ids / the symlinks instead.
- Upstream tracking issue (lean): NVIDIA/TensorRT-LLM#14843.

## 3. Done this session
- **Spike A** ✅ — `debug/spikes/spike_a_flash_attn_contract.py` (PASS): validated the
  `flash_attn_with_kvcache(causal=False)` contract (non-causal SDPA over `[ctx ‖ query-block]`,
  in-place append at `cache_seqlens`, ignores garbage past `ctx_len+block`).
- **Step 1 ops written** (not yet unit-tested): `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/
  dflash_attention.py` — source op `auto_deploy::dflash_attention` (delegates to
  `torch_attention(is_causal=False)`, carries `ctx_len`) + cached op
  `auto_deploy::dflash_attention_with_kvcache` (wraps flash_attn, `mutates_args` on ctx caches) +
  `register_fake` for both. Auto-imported via `custom_ops/__init__.py` `pkgutil.walk_packages`.
- **Checkpoints** downloaded + symlinked (see Environment).
- **PyTorch smoke** run: `debug/spikes/pt_dflash_qwen3_8b_smoke.py` → **the open blocker (§4).**

## 4. ✅ RESOLVED (2026-06-02, CI cluster `/home/scratch.gramnarayan_coreai`) — oracle works
The PyTorch DFlash oracle **works on the internal CI checkpoint**. Run on a free H100 with
`LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models`:
- `pytest …/test_dflash.py::test_dflash_qwen3_8b[True]` → **PASSED** (127s).
- Real acceptance (script `debug/spikes/pt_dflash_qwen3_8b_internal.py`, overlap off, internal ckpt):
  per-request accepted/iter = **1.909 / 0.984 / 1.081**, **mean = 1.325** (max 4.0 at `max_draft_len=4`).
  Coherent text. Drafter is functional.
- **Root cause of the old ~0 acceptance:** NOT structural. The internal `Qwen3-8B-DFlash-b16` is
  shape-identical to the public z-lab one (58 tensors, separate `q/k/v_proj`, no embed/lm_head — the
  loader's separate→fused qkv packing is fine). The divergence was **public HF-id download vs internal
  `llm_models_root()` path** — i.e. the public `z-lab/Qwen3-8B-DFlash-b16` *weights* the old cluster had
  were stale/different (no public copy on this cluster to diff). Moot now: use the internal checkpoint,
  which is the CI oracle the AD port must match. (A public→TRT-LLM conversion/validation, if ever needed,
  is a separate task.)
- Repro: `CUDA_VISIBLE_DEVICES=<free> TRITON_CACHE_DIR=/home/scratch.gramnarayan_coreai/.triton/cache
  LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models python debug/spikes/pt_dflash_qwen3_8b_internal.py`

### (historical) original blocker description
The PyTorch DFlash *oracle* on the **public** `z-lab/Qwen3-8B-DFlash-b16` gave **~0 acceptance**
(0.017 accepted/iter; avg_decoded≈1.0) — vs the **unwaived** unit test `TestQwen3_8B::test_dflash`'s
threshold `mean_accepted ≥ 1.0`. Target text was coherent; spec decoding *ran* but proposals were
~always rejected → the **drafter was non-functional as loaded** (with the public download).
- The smoke replicates `tests/unittest/_torch/speculative/hw_agnostic/test_dflash.py::_make_llm_config`
  exactly; the ONLY difference is it uses **public HF ids** vs the CI's `{llm_models_root()}/…` paths.
- Both public z-lab drafts (Qwen3-8B & Llama) are identical in shape: **58 tensors, separate
  `q_proj`/`k_proj`/`v_proj`, no `embed`/`lm_head` (shared from target)**, `fc`+`hidden_norm`+5 layers.
  TRT-LLM's DFlash model uses a **fused `qkv_proj`** (`modeling_speculative.py:97,1107`).
- **Hypotheses:** (a) the CI test loads an **internal, possibly converted** `Qwen3-8B-DFlash-b16` that
  differs from the public one; (b) a load/config mismatch on the public checkpoint — q/k/v→qkv packing,
  `target_layer_ids` capture not propagating, or `mask_token_id`.
- **Next actions (CI cluster):**
  1. Point at the **internal** `Qwen3-8B-DFlash-b16` the CI uses (the *real* `llm_models_root()`), and
     re-run the smoke / `TestQwen3_8B::test_dflash`. If it accepts → public-vs-internal divergence
     confirmed; use the internal checkpoint (and add a public→TRT-LLM conversion as a separate task).
  2. If only the public checkpoint is available, **debug** with the `ad-debug-agent`: re-run with
     verbose load logging (missing/unexpected keys; the DFlash init `logger.info` of
     `target_layer_ids`/`block_size`/`mask_token_id`); confirm qkv packing + hidden-state capture.
  - Repro: `CUDA_VISIBLE_DEVICES=<free> HF_HOME=… .venv/bin/python debug/spikes/pt_dflash_qwen3_8b_smoke.py`
- **Why it matters:** this PyTorch path is the acceptance **oracle** for the whole AD port (Step 0
  baseline + rung-8 wrapper parity). No AD acceptance number is trustworthy until it works.

## 5. Locked design decisions (condensed — full detail in the summary)
- **Attention:** distinct `auto_deploy::dflash_attention` source op (delegates to
  `torch_attention(is_causal=False)`) → cached `dflash_attention_with_kvcache` (wraps
  `flash_attn_with_kvcache(causal=False)`). Routed by **op-type** via a dedicated
  `insert_cached_dflash_attention` transform — NO per-GM backend gate, NO `kvcache.py` edit.
- **Cache:** unpaged dense slack-sized ctx K/V, **one resource per draft attention node**
  `[max_slots, max_ctx+block_size, nkv, hd]`, `max_ctx=max_seq_len`; **bypasses KVCacheManager**
  (Eagle `hidden_states_cache` precedent). Per-GM *paged* mixing is unsafe today → follow-up.
- **`ctx_len` (= `input_pos + num_accepted`)** is a **declared draft-graph input** (forward placeholder
  → carried by the source op → *retrieved* by insertion), NOT a `SequenceInfo` field. `slot_idx` is the
  standard SequenceInfo arg.
- **`block_size` vs `max_draft_len`:** `block_size` = drafter's **intrinsic trained query-block width
  read from the draft config** (b16=16, Llama=10) — sizes the query block + ctx slack; non-causal block
  ⇒ runs full-width. `max_draft_len` (=4) = how many mask outputs (pos 1..max_draft_len) are consumed;
  `tokens_per_gen_step=max_draft_len+1` = target-verify width. **Validate `max_draft_len+1 ≤ block_size`
  and raise** (PyTorch only silently clamps).
- **Precompute:** port `precompute_context_kv` 1:1 eagerly (Triton dropped; 3rd-GraphModule = profiling
  follow-up). Model uses fused `qkv_proj`; **checkpoint has separate q/k/v_proj** → loader must pack
  separate→fused; eager fused-KV buffers (built at load, never lazily in forward).
- **Runtime:** target `attn_backend=trtllm`, `sync_before_hidden_state_capture=False` (no host sync).
  Phased bring-up: `torch-simple`(overlap off→on) → `torch-cudagraph`. v1 `world_size==1`.
  **CUDA-graph target = single monolithic graph** (AD `CapturedGraph`, the `torch-cudagraph` default,
  `piecewise_enabled=False`). In monolithic capture the draft attention runs *inside* the one captured
  graph with `out=None` (normal return path); intermediate addresses are frozen by the capture itself,
  so the cached op's `out=` param is **inert/unused** here. We do NOT need piecewise. The `out=` param
  is kept anyway (fixed to the AD-canonical `return out.new_empty(0)`, see [[reference_ad_cached_op_out_convention]])
  so piecewise works for free if ever wanted — zero cost in single-graph mode. Eagle precedent: Eagle
  captures its draft loop monolithically; `CapturedGraph.refresh_args_static` handles the
  extend→generate batch-mode switch inside the loop.
- **Export:** generic `submodules_to_preserve()` (modules-only) + thin Eagle tail; target reuses
  `TargetModelExportInfo` (embeddings enough). Build on the `qwen3-vlm-mtp` helpers.
- **Sampler:** reuse `Eagle3OneModelSampler` (add `is_dflash()` to selection) — only after rung-8
  wrapper output parity. **DFlashWrapper is an Eagle *sibling*** (own class).
- **Reference pair:** `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16` (unwaived test, ref acc ≈ 87.11).
  Full shared spec-dec base = follow-up PR.

## 5b. OPEN QUESTIONS — revisit at FINAL CODE REVIEW (do not lose these)
- **DFlash verify vs draft width — RESOLVED (2026-06-02, user-confirmed).** Two *distinct* widths:
  - **Target-verify width = `tokens_per_gen_step = max_draft_len + 1`** (the number of tokens the
    target re-verifies per gen step). Confirmed by `_torch/pyexecutor/model_engine.py:4041`
    ("tokens_per_gen_step (PARD: 2K, DFlash: K+1)") and the summary §5.
  - **Draft-forward (query-block) width = `block_size`** — whatever the DFlash *head was trained
    with* (b16=16, Llama=10); the query block is `[last_accepted, MASK, ..., MASK]` and may be
    **mask-padded** out to block_size. `model_engine.py:436-439` / `mamba_cache_manager.py:1860`
    note DFlash sizes per-request draft scratch by query tokens per gen ("K drafts + K mask fillers").
  - Net for the **single monolithic CUDA graph**: capture the *draft* forward at the fixed
    `block_size` query width and the *target-verify* forward at fixed `max_draft_len+1`. The summary's
    `max_draft_len+1 ≤ block_size` validation stands. Still worth a final-review glance at the exact
    PyTorch ref widths, but the design framing is correct as written — no rework expected.
- **Compare against the other AD attention ops** (`trtllm`/`flashinfer`/`triton`/`torch_backend`
  cached ops + descriptors) at review time: signature/arg-ordering conventions, `get_constants`,
  metadata wiring, and especially how each handles its fixed-shape capture width — to make sure the
  DFlash op + descriptor follow the same patterns and the verify-width choice is consistent.
- **REMINDER:** surface both bullets above (and re-read [[reference_ad_cached_op_out_convention]])
  when doing the final code review.

## 6. Task list (statuses at handoff)
1. ✅ Spike A: flash_attn contract probe.
2. ✅ Step 0: checkpoints present (internal) ✅; **PyTorch reference acceptance = mean 1.325 accepted/iter, test PASSES (§4 resolved 2026-06-02).**
3. ✅ Step 1: ops written + **unit tests done** (2026-06-02):
   `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_dflash_attention_op.py` — 15 tests
   PASS. Source-op == `torch_attention(is_causal=False)` parity (MHA+GQA, scale variants), `ctx_len`
   inert, non-causal sanity, fake shapes; cached-op SDPA parity over `[ctx‖block]` (GQA ratios),
   in-place query-block append at `ctx_len`, persistent-context preservation, default-scale, `out=`
   CUDA-graph buffer path, fake shape. **Fixed during testing:** cached op's `out=` path returned
   `out` (an input) → `torch._library` aliasing violation; changed to the AD-canonical convention
   (write into `out`, `return out.new_empty(0)`; fake likewise) matching trtllm/flashinfer cached ops.
   `out=` is the runtime's pre-allocated graph-stable output buffer (needed for the `torch-cudagraph`
   phase), injected as a trailing kwarg — NOT droppable.
   **Step-1 review done (2026-06-02):** 4-subagent fan-out (architect/thorough/coverage/cleanliness);
   Codex 5th-opinion timed out mid-survey (no unique findings; it did corroborate verify=K+1, §5b).
   Addressed: (a) cached-op `scale` made positional-required (canonical convention vs trtllm/flashinfer);
   (b) tests expanded 15→**25**, added independent hand-rolled SDPA reference (breaks shared-ref
   dependency), batch=1, ctx_len=0, max-slack append, uniform ctx_len, bf16, production block_size=16,
   slack-tail-untouched guard, append-still-happens-on-out= assertion; autouse seed+empty_cache fixture;
   deduped shape constants. pre-commit clean. Parked for Step 2 (architect flags): the descriptor must
   match the **distinct** `auto_deploy::dflash_attention` packet via `get_source_attention_op`, and
   source `slot_idx`/`ctx_len` as standard-or-extra metadata.
4. ⏳ Step 2 + Spike B: **Spike B ✅ (2026-06-02)** — `debug/spikes/spike_b_ctx_len_export.py` PASSES:
   toy 1-layer draft `forward(q,k,v,ctx_len)` → `dflash_attention`, `torch.export` keeps `ctx_len` as a
   placeholder named "ctx_len" carried as the op's arg[3], survives DCE; replicated
   `_add_or_retrieve_input` (interface.py:797) → RETRIEVE for ctx_len / ADD for slot_idx; control
   (no-carry) confirms the carry is load-bearing.
   **Step 2 + 5 DONE (2026-06-02):** `DFlashCtxKVResourceHandler` (unpaged dense
   `[max_slots, max_seq_len+block_size, n_kv, hd]`) + `@AttentionRegistry.register("dflash")
   DFlashAttention` descriptor (`get_source_attention_op==auto_deploy::dflash_attention`,
   `get_cached==dflash_attention_with_kvcache`, `get_standard_metadata_args=[slot_idx, ctx_len]`,
   bsnd, `get_constants=[scale]`) in `custom_ops/attention/dflash_attention.py`;
   `InsertCachedDFlashAttention(_InsertCachedOperator)` in `transform/library/kvcache.py`; default.yaml
   `insert_cached_dflash_attention {stage: cache_init, backend: dflash}`. Verified the descriptor's
   produced arg order EXACTLY matches the cached-op signature. **Structural gate test**
   `tests/.../transformations/library/test_dflash_cache.py` PASSES (rewrite: src→cached, slot_idx
   ADDED, ctx_len RETRIEVED, 2 slack-sized caches, scale const). **TEST-UPGRADE (user-requested):** the
   toy-module structural test is a placeholder; once Step 3 `modeling_dflash.py` exists, replace it with
   a small test that exports the **real** prefill-version DFlash draft model and runs the transform over
   its actual `dflash_attention` sites (keep the toy test as a fast regression guard).
5. ☐ Step 5: slack-sized ctx K/V resource handler + `ctx_len` metadata wiring.
6. ⏳ Step 4: **`precompute_context_kv` DONE (2026-06-02)** — `DFlashModel.precompute_context_kv`
   (eager, `@torch.no_grad`): captured target hidden (in target_layer_ids order) → `fc` →
   `hidden_norm` → per-layer `k_proj`/`v_proj` → `k_norm` (K only) → RoPE (K only) → returns per-layer
   `k`/`v` `[N, L, n_kv, hd]`. **Critical fidelity detail (verified vs z-lab `dflash.py`):** the context
   path is asymmetric to the query path — context does NOT go through each layer's `input_layernorm`
   (only the query stream does); `k_norm` is per-token RMSNorm so applying it to context-K alone
   matches the oracle's post-`cat` norm. Test `test_precompute_context_kv` PASSES (shape; V is raw
   `v_proj`; K has k_norm+RoPE; pos-0 RoPE-identity row matches `k_norm(raw_k)`). **Re fused-KV (clarified):** the *traced* query-block forward keeps separate q/k/v_proj in source and
   the AD pipeline FUSES them via `fuse_gemms`/`fuse_gemms_mixed_children` (default.yaml) — canonical
   AD path, no deviation. The fused-KV gap exists ONLY in `precompute_context_kv`, which is *eager*
   (plain method, not exported) so it bypasses `fuse_gemms` and runs per-layer GEMMs. Correct fix is
   NOT manual fused-KV buffers (the oracle did that because its precompute is eager in PyExecutor) but
   to export precompute as a **3rd GraphModule** so AD fuses it automatically — the summary's
   "3rd-GraphModule = profiling follow-up". v1 eager per-layer is fine (identical math, simple). **Still TODO in Step 4:** the export-preservation
   `DraftModelExportInfo.post_process` (keepalive for fc/hidden_norm; this lives with the factory, Step 7).
7. ⏳ Step 3: **draft module `models/custom/modeling_dflash.py` DONE (2026-06-02)** — exportable draft
   model (`DFlashModel`/`DFlashDrafterForCausalLM` + attention/layer/MLP/RMSNorm/RoPE), module names
   match the 58-tensor checkpoint, query-block forward `(inputs_embeds, position_ids, ctx_len)` emits
   `auto_deploy::dflash_attention` **per layer** with `ctx_len` threaded. Key design: the traced
   forward processes the **query block only**; context K/V comes from the eager `precompute_context_kv`
   (Step 4) via the cache — NOT in this forward.
   **STANDALONE + sharding-IR (user-directed):** the Qwen3 building blocks are *copied* from
   `modeling_qwen3.py` (NOT imported — no cross-model coupling; copy-paste even if identical is the AD
   convention, except MTP/Eagle which intentionally reuse the *target's* layers). Uses the sharding IR
   ops (`torch_linear_simple` colwise/rowwise + `tp_min_local_shape`, `view` `tp_scaled_dim=2`,
   `torch_rope_with_explicit_cos_sin`, `all_reduce`) so the exported graph carries TP hints. Mirrors the
   PyTorch oracle, which reuses Qwen3's qkv/q-k-norm/RoPE/MLP and swaps only the attention call — here
   the swap is to `dflash_attention`. v1 = Qwen3 family; non-Qwen DFlash bases get their own standalone
   modeling later. TODO: TP-sharding of the dense ctx K/V resource (v1 world_size==1, so deferred). Tests `tests/.../models/test_dflash_model.py` (3) PASS,
   incl. **model-based Step-2 gate** `test_transform_over_real_model` (the user-requested upgrade: run
   `insert_cached_dflash_attention` over the REAL exported model — all N layers lower, slot_idx shared,
   ctx_len retrieved, 2 caches/layer). **Still TODO:** `DFlashWrapper` dual-mode forward
   (prefill-only / kv-cache), `precompute_context_kv` + fused-KV buffers (Step 4), `DFlashOneModelFactory`
   + `DraftModelExportInfo.post_process` (keepalive for fc/hidden_norm + shared embed/lm_head) + config
   wiring (Step 7). NOTE: `submodules_to_preserve()` does NOT exist — use `SubModuleExportInfo.post_process`
   + `insert_keepalive_sentinel`/`expose_graph_module_accessor`; custom drafts are built by a
   `@ModelFactoryRegistry.register("dflash_one_model")` factory, NOT the `_MODEL_MODULES` architecture map.
8. ☐ Step 6: hidden-state capture wiring (`target_layer_ids` order).
9. ⏳ Step 7: **factory + wrapper + export infos DONE (2026-06-02)** —
   `models/dflash.py`: `DFlashOneModelFactory` (registered `"dflash_one_model"`, mirrors
   `EagleOneModelFactory`: target via `target_model_factory`, draft via `AutoConfig` +
   `DFlashDrafterForCausalLM`, wrapped in `DFlashWrapper`; validates `max_draft_len+1 ≤ block_size`,
   resolves block_size/mask_token_id from the draft config) + `DFlashDraftModelExportInfo`
   (keepalive sentinels for `fc`/`hidden_norm`; `ctx_len` declared dynamic input) + reuses the generic
   `TargetModelExportInfo`. `DFlashWrapper`/`DFlashWrapperConfig` in `modeling_dflash.py`
   (__init__ + target-embed/lm_head accessors + dual-mode dispatch). Registered via `models/__init__.py`.
   Test `tests/.../models/test_dflash_factory.py` (4) PASS: factory registered, **builds the wrapper
   through the factory** (block_size=16, mask_token_id=151669 resolved), export-infos properties,
   block_size guard raises. (Mirrors the Eagle factory tests, per user guidance — build-through-factory
   + check properties.) **Still TODO (E2E-coupled, validated at build_and_run_ad):** the wrapper's
   `_forward_prefill_only` (export path) + `_forward_with_kv_cache` (inference draft loop + capture +
   scatter + sampler) bodies (currently documented NotImplementedError stubs); draft weight loading
   (separate→fused qkv packing in `_load_checkpoint`); AD `llm_args` wiring (DFlash branch in
   `validate_supported_speculative_config`/`setup_hidden_state_capture`/`validate_speculative_model_factory`
   + `DFlashDecodingConfig.supports_backend("autodeploy")`); sampler `is_dflash()` on Eagle3OneModelSampler.
   **Config wiring DONE (2026-06-02, commit dcd6a88c63):** AD `llm_args` DFlash branches in
   `validate_supported_speculative_config` (allow + `world_size<=1`), `setup_hidden_state_capture`
   (capture = `target_layer_ids`, required for v1), and factory resolution via new
   `_required_one_model_factory()` (DFlash → `dflash_one_model`). `DFlashDecodingConfig.supports_backend`
   now allows `"_autodeploy"` (note the leading underscore — the internal backend string). Tests in
   `shim/test_llm_config.py::TestSpeculativeConfigValidation` (3 DFlash) PASS.
   **Wrapper prefill-only forward DONE (2026-06-02, commit c54b8797eb):**
   `DFlashWrapper._forward_prefill_only` (export path, mirrors EagleWrapper): target → bonus →
   query block `[bonus, MASK…]` (embed via target) → draft (`inputs_embeds`, `position_ids`,
   `ctx_len=0`) emits dflash_attention per layer → target lm_head → `DFlashWrapperOutput`. Fixed
   `_draft_dtype` to read the draft param dtype (was defaulting to bf16 → fp16 mismatch). Tests
   `models/test_dflash_wrapper.py` (2) PASS on a tiny target+draft (no 8B build).
   **Draft dtype (resolved):** driven by `config.torch_dtype` (overridable via
   `speculative_model_kwargs={"torch_dtype": ...}`, Llama+Eagle convention). `DFlashModel.dtype =
   config.torch_dtype`; `_build_model` builds the draft in that dtype; `_draft_dtype` reads it. (Earlier
   runtime param-reading band-aid removed.)
   **Draft weight loading DONE (2026-06-02, commit 32b5721d7e):**
   `DFlashOneModelFactory._load_draft_weights` — DIRECT state_dict load (checkpoint keys → `model.`
   prefix), `strict=True`. NO separate→fused qkv packing (our standalone model keeps separate q/k/v,
   AD fuses the GEMMs in the exported graph via `fuse_gemms`). `test_draft_weights_load_strict` loads
   the REAL z-lab checkpoint strict — a fidelity check that the standalone modeling matches the
   58-tensor checkpoint EXACTLY (PASSES). `_load_checkpoint` loads both target + draft.
   **kv-cache inference forward + sampler wiring DONE — first cut (2026-06-02, commit 59db527579):**
   `DFlashWrapper._forward_with_kv_cache` implemented: target verify (cumprod, Eagle-mirrored) →
   `_scatter_context_kv` (precompute_context_kv → per-seq scatter of accepted ctx K/V into the per-layer
   `ctx_k/v_cache` resources at input_pos) → single non-autoregressive draft pass over `[last_accepted,
   MASK...]` (`ctx_len=input_pos+num_accepted`, `query_positions=ctx_len+arange`) → lm_head → next draft
   tokens. Helpers `_filter_kwargs_for_submodule`/`_collect_hidden_states` (copied from Eagle) +
   `_collect_ctx_cache_pairs`. `DFlashWrapperOutput` gained `next_draft_tokens`/`next_new_tokens` (sampler
   needs all 4). `ad_executor.py` selects `Eagle3OneModelSampler` for `is_dflash()`. Unit tests still pass
   (helpers tested; full path is E2E).
   **E2E DEBUG IN PROGRESS (the remaining work):** smoke `debug/spikes/ad_dflash_qwen3_8b_smoke.py`
   (AD LLM, torch-simple, Qwen3-8B + b16). The runtime contract has UNVALIDATED parts to debug from the
   smoke log: (i) per-seq scatter positions (prefill = whole prompt; extend = accepted prefix) vs how AD
   lays out tokens; (ii) draft-GM arg flow — does it accept `inputs_embeds [num_seq, block_size, H]` +
   our overridden `position_ids`/`ctx_len`, and do slot_idx/ctx caches arrive via named_args?
   (iii) `input_pos` semantics; (iv) per-seq `.item()` scatter is torch-simple-only (CUDA-graph-safe
   fixed-shape scatter is a follow-up). Iterate the smoke → acceptance vs the **1.325 oracle**.
   **E2E smoke status (2026-06-02):** `debug/spikes/ad_dflash_qwen3_8b_smoke.py` (committed) now passes
   config validation. **Current blocker: executor-init `Cannot copy out of meta tensor; use
   to_empty()`** — the AD executor builds the model on `meta` then materializes; my draft build
   (`DFlashOneModelFactory._build_model` does `draft_model.to(device=device, dtype=...)`) or the load
   flow leaves a meta tensor moved via `.to()`. This is BEFORE the forward (build/load path). Likely fix:
   ensure the draft is materialized with `to_empty()` + then load weights, or align with how
   EagleOneModelFactory/AutoModelForCausalLMFactory handle the meta→device transition (they build on
   meta for export then load real weights). Debug from `/tmp/ad_dflash_smoke2.log`.
   **Context-KV PARITY TEST (user-requested, HIGH VALUE, independent of the E2E bug):** add a unit test
   comparing AD `DFlashModel.precompute_context_kv(raw_captured, positions)` vs the PyTorch backend
   `modeling_speculative.py::DFlashForCausalLM.precompute_context_kv(projected_hidden, positions)`
   (signature `[N, hidden] -> ([N,L,nkv,hd], [N,L,nkv,hd])`, fused-KV GEMM + k_norm + RoPE). Plan: load
   the SAME z-lab checkpoint into both; AD takes raw captured (does fc+hidden_norm internally), PyTorch
   takes `pt.hidden_norm(pt.fc(raw))`; assert K and V close. Validates the per-layer (AD) vs fused (PT)
   precompute produce identical K/V — the context-KV fidelity core. Caveat: constructing the PyTorch
   `DFlashForCausalLM` standalone needs its init/load_weights path wired (it's a PyTorch-backend model);
   sort that out (or hook the z-lab HF reference model's attention to expose k_ctx/v_ctx as an alt ref).
   **Test inventory (all PASS, 40 DFlash unit tests across 6 files):** op (25), transform toy (1),
   model (4: eager/export/transform/precompute), factory (5: register/build/export-infos/block_size/
   strict-weight-load), wrapper (2: prefill/kv-stub), config-resolution (3).
10. ☐ E2E: phased bring-up + GSM8K acceptance on Qwen3-8B (ref ≈ 87.11).

## 7. Key reference code
- PyTorch DFlash (oracle to match; AD does NOT run it): `_torch/speculative/dflash.py` (worker
  mechanics — NOT ported), `_torch/models/modeling_speculative.py:1181-1372` (`dflash_forward`,
  `precompute_context_kv` `:978-1135`, fused-KV `:1082-1135`, flash call `:1344-1353`),
  `llmapi/llm_args.py:1745` (`DFlashDecodingConfig`).
- AD Eagle path to mirror: `_torch/auto_deploy/models/custom/modeling_eagle.py`, `models/eagle.py`,
  `transform/library/kvcache.py`, `custom_ops/attention_interface.py`,
  `transform/library/hidden_states.py`, `shim/ad_executor.py`, `llm_args.py`.
- New: `_torch/auto_deploy/custom_ops/attention/dflash_attention.py`.

## 8. First moves on the CI cluster
1. Re-establish env (venv, HF_HOME/symlinks or re-download; confirm `flash_attn` imports).
2. **Resolve the §4 oracle blocker** (internal checkpoint or debug) — gate everything else on a working
   PyTorch reference acceptance.
3. Finish Step 1 (cached-op + source-op unit tests, model on `test_torch_attention_op.py`); run the
   4-subagent + Codex review.
4. Spike B (ctx_len-threading export probe) → Step 2 → 5 → 4 → 3 → 6 → 7 → E2E, flagging deviations.
