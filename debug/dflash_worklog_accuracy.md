<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash worklog — E2E accuracy / zero-acceptance

**GOAL:** make `build_and_run_ad` output a coherent "Paris" answer for "The capital of France is"
**with DFlash spec-dec ON** (ws1, trtllm runtime). Reference: plain Qwen3-8B (spec OFF) is coherent;
PyTorch DFlash oracle = mean 1.325 accepted/iter. Repro:
`GPU=2 SPEC=1 REGSTYLE=0 LOG=... bash debug/spikes/robust_bnr.sh` (robust runner retries the flaky
MPI-startup hang; always `python -u`).

Per-issue worklog. Issue: the AD DFlash E2E smoke RUNS (no crash) but produces garbage + 0 accept.

## ============ RESUME / HANDOFF SUMMARY (read this first) ============
**Where we are (2026-06-03) — ✅ DFLASH E2E WORKS on BOTH Qwen3-8B and Llama-3.1-8B:**
- Qwen3-8B + DFlash: coherent "…the capital is **Paris**…", **mean 2.097 accepted/iter** (31 steps,
  `qwen_accept_measure.log`) — ABOVE the 1.325 oracle.
- Llama-3.1-8B + DFlash: coherent "The capital of France is **Paris**.", mean 2.0/iter
  (`llama_verify_bakedfix.log`).
- ROOT CAUSE of the old target-garbage / "zero acceptance": the `resize_kv_cache` pass's large sample
  forward corrupts a target weight in place (Qwen3→lm_head, Llama→a body weight). It is resize-driven
  and target-AGNOSTIC, NOT Qwen3-specific. "Acceptance 0" was purely a symptom of the corrupted verify
  logits — once the target is coherent the draft accepts normally.
- FIX (TEMPORARY, in code): `DFlashOneModelFactory.get_cache_config_updates()` forces
  `free_gpu_memory_fraction=0.0` → `needs_resize()`==False → resize pass deterministically SKIPPED.
  The lm_head re-home/restore workaround was REMOVED. Debug probes in kvcache.py removed.
- The detailed evidence is in the dated sections below (CONFIRMED FIX → ACCEPTANCE → LLAMA VERIFIED).

**REMAINING TODOs (priority order):**
1. PROPER fix (task #9): make `resize_kv_cache`'s sample forward not corrupt model weights, so KV can
   be sized for capacity again (remove the free_gpu_memory_fraction=0.0 override). Likely tied to
   export `clone=False` weight sharing / `expose_graph_module_accessor`. Exact orphaning mechanism not
   yet isolated.
2. Remove remaining debug scaffolding before PR: `DFLASH_SKIP_CTX` bisection + `DFDBG_ACCEPT` print in
   `models/custom/modeling_dflash.py`; spike env-branches (NORESIZE/NOCAP/DFLASH_SKIP_CTX) are in
   debug/ and won't ship.
3. Context-KV parity unit test (task #4) using HF ref `z-lab/Qwen3-4B-DFlash-b16/modeling_dflash.py`.
4. Run the real accuracy test (test_llm_api_autodeploy.py) for DFlash end-to-end.

--- (historical notes below; superseded where they conflict with the summary above) ---

**Repro:** `GPU=<n> SPEC=1 REGSTYLE=0 ATTN=trtllm LOG=debug/logs/x.log bash debug/spikes/robust_bnr.sh`
(robust_bnr.sh retries the flaky MPI-startup hang; always python -u). SPEC=0 = plain target control.
Models: internal `Qwen3/Qwen3-8B` + `Qwen3-8B-DFlash-b16`. Env: H100 (NOT H20 — H20 ws1 crashes).

**Fixed this session (committed or in working tree):**
1. meta-tensor build/load (load_or_random_init materializes+loads draft; dtype on meta path).
2. SymInt-in-allocate (draft seq dim STATIC in DFlashDraftModelExportInfo).
3. precompute_context_kv lost after export (post_process re-attaches fc/hidden_norm/rotary/layers).
4. Rebased onto origin/gramnarayan/qwen3-vlm-mtp.
5. lm_head corruption (the big one — see ROOT CAUSE below). WORKAROUND in place.
6. Removed stale #13135 (trtllm+torch-simple) workaround in the Eagle3 accuracy test (user: fixed).

**ROOT CAUSE of the target garbage = lm_head corruption by `resize_kv_cache`:**
`resize_kv_cache` (kvcache.py:724) -> `resize_kv_cache_manager` (interface.py:1223) does
`shutdown()` + `torch.cuda.empty_cache()` + reallocates the KV pool; the new pool reuses the memory
block the (untied, separately-preserved) target `lm_head.weight` occupied -> overwrites it with
garbage (std 1.644 vs 0.026) -> ~50x logits -> wrong tokens from token 1. Proven by a per-transform
probe (lm_head correct through end of cache-init; only resize after). NOT compile (torch-simple
`compile()` is a no-op), NOT the draft (DFLASH_SKIP_CTX still corrupts), NOT gather_logits.
lm_head is NOT tied (config tie=False; separate ckpt weight std 0.026; modeling `_tied_weights_keys`
is declared but inert). The embedding (preserved the same way but NOT used by the target graph)
survives; the graph-used lm_head does not.

**WORKAROUND currently in code (debug/spikes + factory/wrapper):**
- `DFlashOneModelFactory._stash_target_head`: re-home lm_head into a fresh param-owned block at load
  + stash a backup. `DFlashWrapper._maybe_restore_target_lm_head`: restore once when corruption is
  detected (post-resize). (Run J with every-forward restore gave coherent Paris; cleaned once-restore
  + re-home is NOT yet re-confirmed due to flaky launches.)

**IMPORTANT CORRECTION (2026-06-03, open gap):** the "resize corrupts untied lm_head" framing is
INCOMPLETE. Eagle on untied Llama-3.1-8B (SAME TargetModelExportInfo + expose_graph_module_accessor +
resize_kv_cache) WORKS (accuracy test passes), and PLAIN Qwen3-8B (also goes through resize) WORKS.
Both `modeling_qwen3.py` and `modeling_llama3.py` declare the SAME `_tied_weights_keys` -> not the
difference. So the corruption is **DFlash-specific**, not a generic untied/resize bug. Also: the probe
never cleanly captured lm_head AFTER resize (the probe run crashed AT resize on the draft-scatter OOB),
so "resize is the corruptor" is INFERRED (correct before resize, garbage at runtime, resize is the
only transform between), not directly observed.
DFlash-specific suspects (differ from Eagle/plain): (a) DFlash's large UNPAGED ctx-K/V caches (5×k+v,
slack-sized) that resize frees+reallocs after empty_cache -> could grab lm_head's block; (b) the draft
SHARES the target lm_head (load_lm_head_from_target) -> extra ref/aliasing.
DECISIVE EXPERIMENT (next): probe lm_head std immediately BEFORE and AFTER `resize_kv_cache` with
`DFLASH_SKIP_CTX=1` (skips the draft so the resize forward doesn't crash) -> confirm/refute resize and
isolate the DFlash-specific cause. Then the proper fix follows.

**PROPER FIX (TODO — NOT a workaround):** fix `expose_graph_module_accessor` (models/hf.py) to be
tie-aware: for tied models preserve `embed_tokens` + re-establish the lm_head tie; for UNTIED models
(Qwen3-8B) ensure the preserved lm_head holds its OWN storage so a generic post-load realloc
(`resize_kv_cache` `empty_cache`) cannot reclaim it. Open sub-question: the exact orphaning mechanism
(why the preserved lm_head's block is on the allocator free list at resize) — needs storage/data_ptr
tracking through resize. Related: export `clone=False` (shared param storage) + the preservation.

**TODOs (priority order):**
- [ ] PROPER lm_head preservation fix (above) — replace the re-home/restore workaround.
- [ ] Draft acceptance = 0 (task #7): use HF reference `z-lab/Qwen3-4B-DFlash-b16/modeling_dflash.py`
      as oracle. Build parity tests (task #4): precompute_context_kv K/V vs ref k_ctx/v_ctx; draft
      query-block logits. Check the capture layer offset (ref uses `hidden_states[layer_id+1]`).
- [ ] Secondary: `resize_kv_cache` big forward hit an illegal-memory-access when draft NOT skipped
      -> likely draft-scatter OOB at large batch (`set_max_num_tokens_sample`).
- [ ] Clean up debug prints + the DFLASH_NO_RESTORE/DFLASH_SKIP_CTX scaffolding before PR.
- [ ] Remove the per-forward/once restore once the proper preservation fix lands.

**Key files:** `models/dflash.py` (factory: _stash_target_head, load_or_random_init, post_process),
`models/custom/modeling_dflash.py` (wrapper: _maybe_restore_target_lm_head, _forward_with_kv_cache),
`models/hf.py` (expose_graph_module_accessor — proper-fix target), `transform/library/kvcache.py`
(resize_kv_cache), `shim/interface.py` (resize_kv_cache_manager).
**Spikes:** `debug/spikes/ad_dflash_qwen3_8b_bnr.py` (main E2E, env-toggled SPEC/REGSTYLE/ATTN/NOCAP),
`robust_bnr.sh` (retry runner), `dflash_build_load_repro.py`, `dflash_export_precompute_repro.py`.
## ====================================================================


## HYPOTHESIS / EXPERIMENT LEDGER (live)
| # | Hypothesis | Experiment | Verdict |
|---|------------|------------|---------|
| H0 | H20 box breaks ws1 target | run plain Qwen3-8B ws1 on H100 | ✅ confirmed env; plain target coherent on H100 |
| H-overlap | overlap scheduler corrupts spec | `disable_overlap_scheduler=True` | ❌ not the cause (still garbage); keep it off |
| H-capture-op | capture op not passthrough | read `cached_residual_add` | ❌ ruled out — verified `ret=t1+t2; cache.copy_(ret); return ret` |
| H2-scatter/draft | draft `flash_attn` / ctx-scatter corrupts target memory (OOB) | `DFLASH_SKIP_CTX=1` (skip scatter+draft, zero drafts) | ❌ **RULED OUT** — identical garbage + identical accepted tokens `[1731,466,23045,23045…]` |
| H1-targetverify | spec 5-pos verify + KV rewind / positions corrupt the target | (next) skip-ctx + disable capture transform → isolate capture vs KV | ⏳ leading hypothesis |
| H1a-capture-graph | capture transform changes target graph beyond the passthrough op | skip-ctx + `detect_hidden_states_for_capture.enabled=false` | ⚠ BLOCKED — args `transforms` override didn't disable it (still `matches=5`); DFlash spec config force-enables capture |
| H1b-trtllm-specattn | trtllm spec-dec attention corrupts the target verify | `attn_backend=flashinfer` | ❌ RULED OUT — identical garbage on flashinfer |
| H3-lmhead-bad | target lm_head is wrong/random → off-scale logits | weight-stat dump of target GM | ✅ CONFIRMED the *symptom* (lm_head random at runtime); root **mechanism** still open (see run log) |
| H3a-not-loaded | lm_head simply not loaded from ckpt | force-copy ckpt lm_head into target param at load (`_ensure_target_head_loaded`) | ❌ it IS loaded at load (0.026); gets overwritten LATER |
| H3b-overwrite-after-load | a post-load transform overwrites lm_head in place | staged std prints + data_ptr compare | ✅ confirmed: 0.026 at end of load, 1.644 at forward, SAME dptr |
| H3c-two-lmheads | target out.logits uses a DIFFERENT lm_head param than the accessor/draft | restore `named_parameters[lm_head.weight]`; observe draft vs target | ✅ confirmed: restoring it changed the DRAFT tokens but NOT the target's → 2 distinct lm_head weights |
| H3d-graph-lmhead-location | the graph's lm_head weight isn't a normal named_parameter (lifted constant/buffer?) | dump all vocab-shaped params/buffers/state_dict at runtime | ⏳ running (`bnr_lmdump.log`) |

## CORRECTION (supersedes earlier notes)
Earlier I mis-decoded the accepted tokens. **Correct decode (verified with the tokenizer):**
`1731="Info"`, `466="ain"`, `23045=" streams"`. So the DFlash output is **garbage from the FIRST
generated token** (a pure prefill, no spec-verify yet) — NOT "coherent first 2 tokens then collapse".
The plain target's first token would be ~"Okay"(32313)/`<think>`. This means the bug is in the
target's **prefill** inside the wrapper, independent of attention backend / draft / spec-verify.

## TARGET-LM_HEAD SUB-INVESTIGATION — run log (all H100, SPEC=1, bnr spike, `python -u`)
| run | log | change | observation |
|-----|-----|--------|-------------|
| A | `bnr_dflash_weights.log` | dump all target weights at runtime | embedding std 0.022 (loaded ✓), all body/norm loaded ✓; **`lm_head.weight` std=1.644 absmax=154 (RANDOM)** at runtime. So only lm_head is wrong. |
| B | (HF ref, no AD) | plain HF Qwen3-8B forward on the prompt | logits std **3.636**; ckpt has `tie_word_embeddings=False` + SEPARATE `lm_head.weight` std **0.026**. So correct lm_head→std 3.6; DFlash's std-1.6 head→logits std 180 (≈50×). |
| C | `bnr_dflash_lmheadfix.log` | `_ensure_target_head_loaded` (copy ckpt lm_head into target param at load) | STILL garbage; runtime lm_head STILL 1.644 → the copy didn't stick. |
| D | `bnr_headfix_dbg.log` | `[HEADFIX]` diag at load | at LOAD: `lm_head.weight` before=**0.026** (target factory already loaded it!), after-copy=0.026. So correct at load, wrong at runtime. |
| E | `bnr_headfix_ptr.log` | add data_ptr | load-time dptr **==** runtime dptr, data 0.026→1.644 → **overwritten in place AFTER load**. |
| F | `bnr_stage.log` | std at each step of `load_or_random_init` | lm_head=0.026 at ALL stages (after target load / ensure / draft `_to_maybe_random` / draft load). → overwrite is OUTSIDE the factory load — a later transform (move_to_device / post_load_fusion / cache_init / compile). |
| G | `bnr_lmhead_restore.log` | stash correct lm_head at load; restore `named_parameters[lm_head.weight]` at 1st forward | **DRAFT tokens changed** (6418… vs 13268…) but **TARGET accepted tokens UNCHANGED** (1731,466,23045…). → the param I restored is used by the DRAFT (`get_output_embeddings`/`apply_lm_head`); the target's `out.logits` uses a **DIFFERENT** lm_head. **Two lm_head weights exist.** |
| H | `bnr_lmhead_all.log` | restore ALL `named_parameters` with shape==lm_head & std>0.5 | `[LMRESTORE] restored []` (nothing matched at restore-time) AND target still garbage. → the random lm_head the target graph uses is **not a `named_parameter`** at restore-time (a lifted constant/buffer?), or only becomes random during the forward. |
| I | `bnr_lmdump.log` | dump all vocab-shaped params+buffers+state_dict @ runtime | ⏳ running — to locate the actual random lm_head tensor the graph uses. |

## ✅ MILESTONE (2026-06-03) — TARGET FIXED, coherent Paris with spec-dec ON
Restoring `lm_head.weight` on **every** forward (not once) fixed the target. `bnr_restore_every.log`:
output = "…Okay, the user is asking for the capital of France… The most common answer I remember is
**Paris**…" — coherent, matches the base model. Accepted tokens decode sanely (`<think>`,`\n`,`Okay`,
`,`,` the`…). **The lm_head corruption WAS the entire target garbage bug.**

Why "every forward": `_maybe_restore` (once-guarded) fired on the **cache-init validation forward**
(np=4, the FIRST `_forward_with_kv_cache`), where lm_head was still 0.026 (LMDUMP). lm_head is then
overwritten in place (same dptr) before the REAL generation forwards (a post-validation/compile
transform). So the once-restore was too early; restoring each forward keeps it correct.
LMDUMP also proved there is exactly ONE lm_head-shaped param — so H3c "two lm_heads" was wrong; the
earlier "draft changed but target didn't" (run G) was the once-guard firing on the validation forward
only.

**Run J** `bnr_restore_every.log`: restore lm_head every forward → coherent Paris. acceptance still
all `lens=[1]` (0 accepted) → TARGET done, DRAFT path is the remaining work.

## REMAINING WORK (draft acceptance: 0 → ~1.325 oracle)
Target now coherent; drafts are always rejected. step0 draft = `[142895,142895,142895,142895]` (all
same token), later steps varied but rejected. Investigate the DFlash draft path: precompute_context_kv
correctness, ctx-K/V scatter positions, query-block construction, the draft attention op, and whether
the draft uses the (now-restored) lm_head. The context-KV parity unit test (task #4) is the right
next gate. NOTE: the lm_head "restore every forward" is a WORKAROUND (copies ~1.2GB/forward); the
proper fix = find & fix the post-load transform that corrupts the preserved lm_head in place (follow-up).

## ROOT-CAUSE HUNT: which transform corrupts lm_head (2026-06-03)
Per-transform probe in `optimizer.py` (`[LMPROBE]` after each transform), restore disabled
(`DFLASH_NO_RESTORE=1`). Finding (`bnr_lmprobe.log`):
- lm_head = **0.026 (correct), same dptr, through the END of cache-init (`initialize_cache`)** —
  across load_weights + all post_load_fusion + insert_cached_* + initialize_cache.
- After `initialize_cache` the name-based probe found **no `lm_head.weight` named_parameter** →
  a post-cache-init transform RESTRUCTURES mod (lm_head no longer a plain named_param). Remaining
  transforms: `resize_kv_cache`, then compile stage (`compile_model`). Re-running with a by-shape
  probe to name the exact transform + see if it's an in-place overwrite vs param swap (`probe3.log`).
- **`compile_model` is the prime suspect**: even for torch-simple it runs CAPTURE forwards
  (`cm.info.reset(); set_capture_batch; mod(**named_args, cache_seq_interface=cm)`), executing the
  full wrapper forward during compile.
- `gather_logits_before_lm_head` ran (post_load_fusion) with lm_head STILL 0.026 → NOT the weight
  corruptor. (But it inserts a `gather_tokens` before lm_head; must confirm it keeps ALL K+1 verify
  positions for spec-dec — separate concern.) Eagle's UNIT tests use skip_loading_weights (random),
  so they would not catch a real-weight lm_head corruption — consistent with this being unnoticed.

## HF REFERENCE ORACLE (for draft testing — task #4/#7)
`z-lab/Qwen3-4B-DFlash-b16/modeling_dflash.py` is the authoritative DFlash impl. Key facts (vs our
AD port) for parity tests:
- NO precompute method — context K/V computed in attention: `k_ctx=k_proj(target_hidden)`, concat
  with query-block K/V, reshape, **per-head k_norm**, transpose, **RoPE jointly over [ctx‖noise]**
  (RoPE asymmetric: Q only last q_len positions, K all). is_causal=False.
- `target_hidden = fc(concat(selected hidden states))`; `extract_context_feature` indexes
  `hidden_states[layer_id + 1]` (**+1 offset for the embedding layer**) — verify our capture order/offset.
- lm_head NOT tied (separate param); embeddings + lm_head taken from the target.
- Plan: load the SAME checkpoint into HF-ref + our AD `DFlashModel`; compare (a) precompute_context_kv
  K/V vs ref k_ctx/v_ctx, (b) draft query-block logits. Pins down the draft acceptance=0 bug.

## ✅ ROOT CAUSE OF lm_head CORRUPTION (2026-06-03) — `resize_kv_cache`
Per-transform probe (`optimizer.py`, `[LMPROBE]`) proved lm_head is **0.026 (correct, same dptr)
through the END of cache-init (`initialize_cache`)**. The ONLY transform after that (before runtime)
is **`resize_kv_cache`** (`transform/library/kvcache.py:724`). It calls
`cm.resize_kv_cache_manager()` (`shim/interface.py:1223`):
```
self._kv_cache_manager.shutdown()                      # free all cache views
_, free_mem, *_ = get_mem_info(empty_cache=True)       # torch.cuda.empty_cache() -> releases blocks to CUDA
self._create_kv_cache_manager(max_tokens=optimal)      # reallocate a bigger KV pool, reusing freed memory
```
→ the reallocated KV pool **reuses the memory block the lm_head weight occupied**, overwriting it with
uninitialized garbage (std 1.644). Ruled out: **torch-simple `compile()` is a NO-OP** (`return self.model`);
**draft** (`DFLASH_SKIP_CTX` still corrupts); **gather_logits** (lm_head still 0.026 after it; Eagle/MTP
don't disable it). It's one-time (build phase), so **restore-once-after-build** suffices (every-forward
is overkill; the once-guard failed only because it fired on the PRE-resize validation forward).
- WHY lm_head (not embed, both preserved): lm_head's storage is orphaned/freed before resize's
  `empty_cache()` — tied to the preserved lm_head (export clone=False + `expose_graph_module_accessor`
  + `gather_logits`). embed is preserved the same way but not reused. Exact orphaning mechanism = the
  remaining sub-question for the PROPER fix (keep the preserved param's allocation live across resize,
  or exclude model params from the resize free/realloc).
- SECONDARY bug surfaced: `resize_kv_cache`'s big forward (`set_max_num_tokens_sample`) hit an illegal
  memory access when the draft was NOT skipped → likely a draft-scatter OOB at large batch
  (`bnr_lmprobe`/`probe3.log`). Separate from lm_head; track under draft acceptance (#7).

## PARALLEL EXPERIMENTS IN FLIGHT (2026-06-03) — both DFLASH_NO_RESTORE=1 + DFLASH_SKIP_CTX=1
Added `[RESIZEPROBE]` in `resize_kv_cache` (kvcache.py): prints target lm_head std+dptr BEFORE the
resize forward, AFTER it, and AFTER `resize_kv_cache_manager`. Re-home gated off under NO_RESTORE.
The bnr spike is now env-parameterized (TARGET_MODEL / DRAFT_MODEL / TGT_LAYER_IDS).
- **EXP-B (GPU0, Qwen3+DFlash):** `exp_qwen3_resize.log` — confirm Qwen3 target lm_head flips
  0.026->garbage AT resize (and pinpoint forward vs resize_kv_cache_manager).
- **EXP-A (GPU1, Llama+DFlash):** `exp_llama_dflash.log` — Llama-3.1-8B target + Qwen3-arch Llama
  DFlash draft (`LLaMA3.1-8B-Instruct-DFlash-UltraChat`, layer_ids 1,8,15,22,29). Does the LLAMA
  target lm_head survive resize? If yes -> corruption is **Qwen3-target-specific** (modeling_qwen3
  under the DFlash preservation path), not DFlash-generic. NOTE the Llama "DFlash" draft is itself
  model_type=qwen3 (Qwen3-arch draft), so our DFlashModel builds it; only the TARGET changes.
- Both run python directly (NOT robust_bnr.sh, whose pkill would kill the sibling).

## ✅ SHARPER ROOT CAUSE (2026-06-03, exp B Qwen3) — it's the resize FORWARD, not the realloc
`[RESIZEPROBE]` (Qwen3, DFLASH_NO_RESTORE=1 + DFLASH_SKIP_CTX=1):
```
before resize forward          : lm_head std=0.02602  (correct)
after  resize forward          : lm_head std=0.9492   (CORRUPTED, SAME dptr)
after  resize_kv_cache_manager : lm_head std=0.9492   (unchanged)
```
=> The corruption happens DURING the resize **forward** (`mod(**named_args)` at kvcache.py ~line 750,
run at `set_max_num_tokens_sample` = large batch), NOT in `resize_kv_cache_manager`. It is **in-place**
(same dptr) and happens with the **draft SKIPPED** -> it's the **target's own forward** overwriting its
own lm_head. Mechanism = **aliasing/orphaning**: lm_head's storage block is on the allocator free list
(despite being a "live" preserved param), so the large forward's activation allocations reuse it. This
is the preservation flaw (expose_graph_module_accessor) — confirms the user's instinct. (Earlier
"resize_kv_cache_manager realloc" conclusion was WRONG; corrected here.)
Next: Llama comparison (exp A) — if Llama's resize forward keeps lm_head 0.026, the orphaning is
Qwen3-target-specific.

## EXP-A RESULT (Llama+DFlash) — reframes the root cause (2026-06-03)
Llama-3.1-8B target + Qwen3-arch Llama DFlash draft, same NO_RESTORE+SKIP_CTX:
```
[RESIZEPROBE] before/after resize forward + after manager: lm_head std=0.01427 (UNCHANGED, same dptr)
```
**Llama target lm_head is INTACT through resize — BUT Llama output is ALSO GARBAGE**
(`": OOiterm,ist and it,oreOO..."`). So:
- NOT cleanly Qwen3-specific. Both Qwen3 and Llama DFlash produce garbage; Qwen3's victim is lm_head,
  Llama's victim is some OTHER weight (lm_head survives) — or the body.
- REFRAMED ROOT CAUSE: a **memory-orphaning bug in the DFlash path** — the large-batch resize forward
  (`set_max_num_tokens_sample`) reuses **orphaned model-weight storage blocks**; WHICH weight is
  clobbered is model/layout-dependent. The lm_head-restore workaround is INCIDENTAL (Qwen3 only).
- DFlash-vs-Eagle difference (both Llama target, same TargetModelExportInfo): DFlash allocates LARGE
  UNPAGED ctx-K/V caches (5×(k+v), slack-sized) + my post_process re-attaches the draft's layers; one
  of these leaves model-weight blocks freed/orphaned so the resize forward's activations reuse them.
- NEXT: dump ALL target weights' std around the resize forward for Llama (find the clobbered weight),
  and investigate why DFlash orphans model-weight storage (vs Eagle). PROPER FIX must stop the
  orphaning generally, not restore one weight.

## ✅✅ CONFIRMED FIX (2026-06-03): `resize_kv_cache` is the SOLE cause
Test: `NORESIZE=1` (kv_cache_config `free_gpu_memory_fraction=0.0` -> `needs_resize()`==False ->
`resize_kv_cache` SKIPPED, no resize forward/realloc) + `DFLASH_NO_RESTORE=1` (lm_head re-home/restore
DISABLED). Result (`qwen_noresize.log`): **COHERENT Paris**, RESIZEPROBE count=0, resize transform
"skipped". So with resize disabled the target lm_head is NOT corrupted **without any lm_head workaround**
-> `resize_kv_cache` is the entire cause.
**TEMPORARY FIX (user-directed):** set `free_gpu_memory_fraction=0.0` for DFlash (disables the resize
pass; trades away KV-capacity tuning -> uses the estimation-mode pool). Remove the lm_head
re-home/restore workaround. Bake into `DFlashOneModelFactory.get_cache_config_updates()` with a TODO.
**PROPER FIX (follow-up):** make `resize_kv_cache`'s large sample forward not corrupt model weights
(the exact mechanism — overflow of the estimation paged pool vs storage orphaning — was not isolated;
the small-resize test crashed at MPI launch). Likely tied to the DFlash unmanaged ctx caches reducing
the estimation paged-pool budget and/or the spec extra-tokens, so the `set_max_num_tokens_sample`
forward overruns the pool into adjacent model-weight memory. Plain Qwen3 + Eagle-Llama don't trip it.
**HYPOTHESIS:** free_gpu_memory_fraction=0.0 should ALSO fix Llama+DFlash (Llama's victim was a
different weight) -> tell the Llama fork to test NORESIZE.

### ✅✅✅ BAKED-FIX VERIFICATION (2026-06-03) — clean run, NO env toggles
Edits landed:
  - `DFlashOneModelFactory.get_cache_config_updates()` now forces `free_gpu_memory_fraction=0.0`
    (TEMPORARY/TODO comment pointing here).
  - REMOVED the lm_head workaround entirely: `_stash_target_head` (dflash.py) +
    `_maybe_restore_target_lm_head` and its Phase-1 call (modeling_dflash.py).
  - REMOVED debug scaffolding from `kvcache.py` (`[RESIZEPROBE]` `_lmprobe`, `DFLASH_SMALL_RESIZE`
    gate) and the now-unused `import torch` in dflash.py.
Run: `CUDA_VISIBLE_DEVICES=0 SPEC=1 REGSTYLE=0 ATTN=trtllm python -u debug/spikes/ad_dflash_qwen3_8b_bnr.py`
  — **NO NORESIZE, NO NO_RESTORE env** (relies solely on the factory override).
Result (`debug/logs/qwen_verify_bakedfix.log`):
  - `resize_kv_cache [SUMMARY] skipped | time: 0.000s` (line 940) — pass skipped because the factory
    override drove `needs_resize()`==False. (The `free_gpu_memory_fraction: 0.9` at line 159 is just
    the pre-override llm_args echo, NOT the runtime decision.)
  - Output: **COHERENT** — "The capital of France is … the capital is … Paris." No errors.
=> The factory cache-config override alone fixes the target corruption with zero lm_head workaround.
   User hypothesis confirmed verbatim: "if only the resize is the problem, we should be good." ✓

### 🎉 ACCEPTANCE IS WORKING (2026-06-03) — "zero acceptance" was a TARGET symptom, not a draft bug
The `[DFDBG]` per-step trace from the baked-fix verify run (`qwen_verify_bakedfix.log`) shows the
draft IS being accepted. `lens=[N]` is the accepted-token count per decode step:
  step0..7 lens = 1,1,1,3,3,5,3,2  -> mean **2.375 accepted/iter** over the first 8 steps
  (step5 lens=5 = all 4 drafted tokens + the bonus token accepted = full block hit).
This is ABOVE the PyTorch oracle (1.325). => the previously-observed "acceptance 0" (task #7) was
purely a consequence of the corrupted target lm_head producing garbage verify logits — once the
target is coherent, the draft path accepts normally. Draft path (scatter/precompute_context_kv/
non-causal block attn/lm_head) is functionally correct E2E.
NOTE: the old DFDBG print was capped at `_n < 8` (hence only 8 lines for a 64-token output) — NOT a
generation stop. Replaced with an env-gated `DFDBG_ACCEPT=1` that prints `lens` every step so we can
compute the full-run mean. Clean measurement run in flight (`qwen_accept_measure.log`).

### ✅ FULL-RUN ACCEPTANCE (2026-06-03) — `qwen_accept_measure.log`, DFDBG_ACCEPT=1, no other toggles
**steps=31, sum_accepted=65, mean = 2.097 accepted/iter** (oracle=1.325). Coherent output.
=> DFlash Qwen3-8B is functionally complete E2E with acceptance ABOVE the oracle. (Higher than the
   1.325 oracle is expected: single highly-predictable factual prompt + greedy, vs the oracle's
   multi-prompt dataset average.) TASK #7 RESOLVED.

## CURRENT OPEN QUESTION
The target graph's `out.logits` is computed from a **second, random lm_head weight** that is distinct
from `target_model.get_output_embeddings()` / `named_parameters()["lm_head.weight"]` (which the draft
uses and which I can fix). Need to find WHERE the graph's lm_head weight lives (run I) and restore
THAT — or fix the post-load transform that corrupts it. Likely tied to the export's
`expose_graph_module_accessor` `set_submodule("lm_head", …)` creating a duplicate vs the graph's
get_attr param, combined with the unconditional `_tied_weights_keys` in `modeling_qwen3.py`
(Qwen3-8B is untied).

NOTE: `attn_backend` env-toggled in the bnr spike (`ATTN=flashinfer`). Base Qwen3-8B (SPEC=0) on
single H100 = COHERENT ("…the most common answer is Paris…") — base loads lm_head correctly; only
the DFlash wrapped/exported target is affected.

## Symptom (2026-06-02, smoke5)
`debug/spikes/ad_dflash_qwen3_8b_smoke.py`, Qwen3-8B + b16, torch-simple, prompt "The capital of
France is", max_tokens=64, temperature=0:
```
[0] avg_decoded_tokens_per_iter=1.000  accepted/iter=0.000
     gen: '.ap destination streams streams streams streams streams streams streams ...'
```
- accepted/iter = 0 → no draft tokens accepted.
- Text is INCOHERENT and repetitive ("streams streams ...") → suggests the TARGET generation is
  broken, not merely bad drafting. (If only drafting were broken, the target would still emit
  coherent text at 0 acceptance.)

## Oracle to match
PyTorch DFlash on internal b16 ckpt: mean 1.325 accepted/iter, coherent text (see
`dflash_restart_handoff.md` §4).

## Plan / hypotheses
1. CONTROL: run plain AD Qwen3-8B (no DFlash spec) same settings. If garbage too → target setup
   bug (not DFlash). If coherent → bug is in `DFlashWrapper._forward_with_kv_cache` (target verify
   path / kwargs / positions / sampling-verify).
2. If wrapper bug: inspect target forward kwargs (`_filter_kwargs_for_submodule`), gather/squeeze of
   logits, the cumprod-verify, new_tokens packing, input_pos/positions semantics.
3. Draft path (0 acceptance) is secondary until the target emits coherent text.

## Attempts
- **Attempt 1 (2026-06-02) — plain AD Qwen3-8B control — INCONCLUSIVE.**
  `debug/spikes/ad_qwen3_8b_control.py` (no DFlash, same target+settings) FAILS at the cache-init
  validation forward (`kvcache.py:758` → GM forward → `custom_ops/linear/linear.py:68 simple`) with
  `CUBLAS_STATUS_EXECUTION_FAILED` / illegal memory access — on a CLEAN GPU, reproducible. Odd: the
  DFlash run PASSED this same stage and generated. So the control can't serve as a clean baseline
  (likely env/config-flaky, or a torch-2.11-nv cublas quirk at a padded warmup batch). Pivoting to
  direct comparison of the DFlash target-verify path vs EagleWrapper instead of relying on control.
  TODO: revisit whether `cuda_graph_config(enable_padding)` warmup batch triggers the control cublas.
- **Attempt 2 (2026-06-02) — compare DFlash target-verify vs EagleWrapper — target code matches.**
  `DFlashWrapper._forward_with_kv_cache` Phase 1 (target forward + `maybe_gather_and_squeeze(out.logits)`)
  is IDENTICAL to `EagleWrapper._forward_with_kv_cache` (modeling_eagle.py:1013-1018). So the target
  forward/logits handling is not the bug. Since accepted/iter=0 (every emitted token = the target's
  own greedy token) AND the text is incoherent, the TARGET is generating garbage autoregressively.
  **Leading hypothesis: hidden-state capture wiring.** DFlash enables capture (`target_layer_ids`);
  the control did NOT (capture `disabled` in control log line 170). Capture rewrites the target
  graph with `residual_add_for_capture` ops + a `cached_residual_add` resource (Step 6 in the
  handoff was never completed). If that rewrite isn't a faithful passthrough, it corrupts the
  target residual stream → garbage. NEXT: verify the capture op is transparent to the target
  (compare target greedy tokens with capture ON vs OFF), and audit Step-6 capture wiring
  (`transform/library/hidden_states.py`, `detect_hidden_states_for_capture`, target_layer_ids order).
  Secondary suspect: my post_process replaced the draft GM's `model.layers` with the eager ModuleList
  — validate the draft query-block output is still numerically correct (lightweight test checked
  shape only), though this would only affect drafting, not the target's first token.
- **Attempt 3 (2026-06-03) — H100 + rebase: CLEAN BASELINE established.**
  Moved to 8×H100 80GB and rebased onto updated `origin/gramnarayan/qwen3-vlm-mtp`.
  - plain Qwen3-8B, ws1, trtllm runtime → **COHERENT** ("…capital is Paris…") — the prior ws1 CUBLAS
    crash was **H20-specific**, NOT DFlash/onboarding. Log `bnr_qwen3_8b_ws1_regstyle.log`.
  - DFlash, ws1, trtllm runtime, torch-simple → **STILL GARBAGE**: `"Info ain streams streams streams
    streams…"`. Log `bnr_dflash_h100_ws1.log`. accepted/iter still ~0 (every token = target greedy).
  => The garbage is a **genuine DFlash bug**, now reproducible cleanly (no H20 confound) on H100.
  The target is coherent standalone but garbages inside DFlashWrapper → DFlash-specific code is
  corrupting the target forward. Hypotheses unchanged: (A) hidden-state capture rewrite of the target
  graph not transparent; (B) wrapper target-forward kwargs (`_filter_kwargs_for_submodule`) dropping
  needed args; (C) something in the target submodule export. NEXT: instrument
  `_forward_with_kv_cache` to print the prefill bonus token (expect " Paris") and confirm whether the
  TARGET logits are already wrong, then bisect capture-on vs capture-off.
- **Attempt 4 (2026-06-03) — overlap scheduler OFF — did NOT fix it.**
  Added `disable_overlap_scheduler=True` (the AD spec-dec smoke sets this for every spec case).
  Confirmed applied (`disable_overlap_scheduler: true` in log). Output still garbage. So overlap is
  necessary-but-not-sufficient; not the root cause. (Keep it off regardless — required for spec.)
- **Attempt 5 (2026-06-03) — per-step instrumentation: KEY DATA.**
  Printed real decode steps (`bnr_dflash_dbg2.log`). Verify/draft-feedback bookkeeping is CORRECT
  (step1 verifies `[accepted, draft0..draft3]`; cumprod-accept works; ctx_len advances +1/accept).
  Accepted tokens = `[1731, 466, 23045, 23045, …]`. **CORRECTION (later):** these decode to
  `"Info"/"ain"/" streams"` — garbage from token 1, NOT "Okay," then collapse. So the target's
  pos-0 prediction (the accepted token) is wrong **from the prefill**, deterministically. This is the
  lm_head bug (see SUB-INVESTIGATION run log above), not a per-step KV-rewind drift.
  NOTE: always run with `python -u` (stdout is block-buffered to file; hides progress + DFDBG prints).

### ✅ LLAMA+DFLASH VERIFIED (2026-06-03) — baked fix is UNIVERSAL, not Qwen3-specific
Clean run (`llama_verify_bakedfix.log`): TARGET=Llama-3.1-8B-Instruct,
DRAFT=LLaMA3.1-8B-Instruct-DFlash-UltraChat, TGT_LAYER_IDS=1,8,15,22,29, baked factory fix only
(no NORESIZE/NO_RESTORE env). resize_kv_cache SKIPPED. Output: **"The capital of France is Paris."**
(coherent, correct). Acceptance lens=1,5,1,1 -> mean 2.0/iter (terse Instruct answer -> EOS at 8 tok).
=> The `free_gpu_memory_fraction=0.0` factory override fixes BOTH Qwen3 and Llama targets. Task #8
   answered: lm_head/weight corruption is resize-driven and target-agnostic (Qwen3 victim=lm_head,
   Llama victim=a body weight) — NOT Qwen3-specific. The two parallel tracks converged on the same
   root cause; Llama fork agent stopped (its DFLASH_SMALL_RESIZE experiment was confounded by the
   kvcache.py cleanup that removed that gate).

## ============ GSM8K ACCURACY COMPARISON (2026-06-03) ============
Harness: `debug/spikes/gsm8k_ad_compare.py` (AutoDeployLLM + tensorrt_llm.evaluate.GSM8K, greedy,
200 samples, local dataset). PyT reference via `trtllm-eval --backend pytorch`. Goal (user): AD
DFlash on==off accuracy (lossless under greedy) + reasonable acceptance; AD≈PyT validates the backend.

| config | GSM8K acc (200, greedy) |
|--------|--------------------------|
| AD Llama-3.1-8B  DFlash OFF | **75.50** |
| PyT Llama-3.1-8B (ref)      | 75.75 (flex 78.5 / strict 73.0) |
| AD Qwen3-8B      DFlash OFF | **88.25** |
| PyT Qwen3-8B (ref)          | 86.25 (flex 87.0 / strict 85.5) |
| AD Llama  DFlash ON         | (re-running, see below) |
| AD Qwen3  DFlash ON         | (re-running, see below) |

=> AD DFlash-OFF matches PyT for both models -> the AutoDeploy backend itself is correct.

**OOM on first DFlash-ON attempt — temp-fix side effect (important for task #9):** the
`free_gpu_memory_fraction=0.0` factory override DISABLES the memory-aware `resize_kv_cache`, so the KV
pool is sized at the FULL estimate (max_batch_size=128 x max_seq_len=8192 ~ 131GB of KV) instead of
shrinking to fit free memory -> "tried to allocate 32 GiB, 20 GiB free" OOM at init. The DFlash-OFF
runs survived because resize was still active for them (factory forces 0.0 only for DFlash). Workaround
for the eval: bound memory explicitly for DFlash (max_batch_size=16, max_tokens=32768, cuda_graph
batch_sizes<=16). Greedy accuracy is batch/pool-invariant so on-vs-off parity is preserved. The PROPER
fix (task #9) — make the resize sample forward not corrupt weights — would also restore memory-aware
KV sizing and remove this footgun.

### ✅ GSM8K FINAL TABLE (2026-06-03) — DFlash on==off accuracy CONFIRMED (lossless)
| config | GSM8K acc (200, greedy) | draft accept_rate | ≈accepted_draft/step (rate×4) |
|--------|-------------------------|-------------------|-------------------------------|
| AD Llama-3.1-8B  DFlash OFF | 75.50 | — | — |
| AD Llama-3.1-8B  DFlash ON  | **75.75** | **46.38%** (12518/26988) | **1.86** (+1 bonus = 2.86 tok/step) |
| PyT Llama-3.1-8B (ref)      | 75.75 | — | — |
| AD Qwen3-8B      DFlash OFF | 88.25 | — | — |
| AD Qwen3-8B      DFlash ON  | **87.25** | **58.36%** (18528/31748) | **2.33** (+1 bonus = 3.33 tok/step) |
| PyT Qwen3-8B (ref)          | 86.25 | — | — |

CONCLUSIONS:
- **DFlash on==off accuracy**: Llama 75.50 vs 75.75 (Δ0.25), Qwen3 88.25 vs 87.25 (Δ1.0) — within
  sampling noise. Spec-dec is lossless under greedy as expected. ✓ (user's primary requirement)
- **AD ≈ PyT** (DFlash off): Llama 75.50 vs 75.75; Qwen3 88.25 vs 86.25 — AutoDeploy backend correct. ✓
- **Acceptance reasonable**: ~1.86 (Llama) / ~2.33 (Qwen3) accepted draft tokens/step, both ABOVE the
  1.325 PyT oracle. ✓
- accept_rate is the trustworthy metric (a ratio). NOTE: specDecodingStats numDraftTokens/
  numAcceptedTokens are BATCH-SUMMED per stat entry, so an "accepted_per_iter = sum/niter" formula is
  WRONG (gave an impossible 28.5); derive per-step from rate*max_draft_len instead. Harness fixed.
- Memory: DFlash-ON needed explicit bounds (max_batch_size=16, max_tokens=32768) because the temp fix
  disables resize auto-sizing (see prior OOM note). Off used defaults (batch 128 / seq 8192).

TASK #10 DONE. Next roadmap: (#11) enable overlap scheduler, (#12) monolithic cudagraph — re-verify
accuracy+acceptance unchanged at each step.

### ✅ TASK #11 — OVERLAP SCHEDULER VERIFIED (2026-06-03)
Hypothesis confirmed: the earlier "overlap-on garbage" was the resize/target-corruption bug, not an
overlap incompatibility. With the target fix, enabling overlap (just DON'T set
disable_overlap_scheduler; OVERLAP=1 in the harness) works. GSM8K 200-sample, DFlash ON:
| metric | overlap OFF | overlap ON |
|--------|-------------|------------|
| Llama acc        | 75.75  | 75.25  |
| Llama accept_rate| 46.38% | 47.03% |
| Qwen3 acc        | 87.25  | 87.50  |
| Qwen3 accept_rate| 58.36% | 57.45% |
=> accuracy + acceptance effectively unchanged (all within sampling noise). Overlap scheduler OK.
Next: task #12 monolithic cudagraph (compile_backend torch-simple -> torch-cudagraph).
