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

## 4. OPEN BLOCKER — resolve FIRST on the CI cluster
The PyTorch DFlash *oracle* on the **public** `z-lab/Qwen3-8B-DFlash-b16` gives **~0 acceptance**
(0.017 accepted/iter; avg_decoded≈1.0) — vs the **unwaived** unit test `TestQwen3_8B::test_dflash`'s
threshold `mean_accepted ≥ 1.0`. Target text is coherent; spec decoding *runs* but proposals are
~always rejected → the **drafter is non-functional as loaded**.
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
- **Export:** generic `submodules_to_preserve()` (modules-only) + thin Eagle tail; target reuses
  `TargetModelExportInfo` (embeddings enough). Build on the `qwen3-vlm-mtp` helpers.
- **Sampler:** reuse `Eagle3OneModelSampler` (add `is_dflash()` to selection) — only after rung-8
  wrapper output parity. **DFlashWrapper is an Eagle *sibling*** (own class).
- **Reference pair:** `Qwen/Qwen3-8B` + `z-lab/Qwen3-8B-DFlash-b16` (unwaived test, ref acc ≈ 87.11).
  Full shared spec-dec base = follow-up PR.

## 6. Task list (statuses at handoff)
1. ✅ Spike A: flash_attn contract probe.
2. ⏳ Step 0: checkpoints downloaded ✅; **PyTorch reference acceptance = BLOCKED (§4).**
3. ⏳ Step 1: ops written; **cached-op unit test** (model on `test_torch_attention_op.py`) + source-op
   == `torch_attention` test still TODO.
4. ☐ Step 2 + Spike B: dedicated `insert_cached_dflash_attention` transform + ctx_len-threading export probe.
5. ☐ Step 5: slack-sized ctx K/V resource handler + `ctx_len` metadata wiring.
6. ☐ Step 4: eager `precompute_context_kv` + fused-KV buffers + `submodules_to_preserve()` refactor.
7. ☐ Step 3: `DFlashWrapper` + draft module (`modeling_dflash.py`).
8. ☐ Step 6: hidden-state capture wiring (`target_layer_ids` order).
9. ☐ Step 7: `DFlashOneModelFactory` + config + sampler wiring (incl. qkv packing + block_size validation).
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
