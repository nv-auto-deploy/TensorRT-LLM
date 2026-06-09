<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash worklog — LLAMA FORK (Llama-3.1-8B + DFlash)

This is the **Llama fork** worklog. ONLY the Llama-track process writes here. The Qwen3 track writes
to `debug/dflash_worklog_accuracy.md`. Shared root context lives in `debug/dflash_restart_handoff.md`
and the accuracy worklog's RESUME/HANDOFF section.

## Goal
Get Llama-3.1-8B + DFlash working E2E (coherent output, ideally nonzero draft acceptance) via
`build_and_run_ad`, ws1, on H100 (use **GPU 1**).

## Constraints (IMPORTANT)
- **Do NOT edit core code** (`tensorrt_llm/...`) — the Qwen3 track owns core edits to avoid conflicts.
  Use ONLY the env-parameterized spike + existing toggles. Coordinate any needed core change by
  writing a request in this worklog (don't make it yourself).
- Use **GPU 1** (Qwen track uses GPU 0).
- Always `python -u`; tee to `debug/logs/llama_*.log`.

## Repro (env-parameterized spike already supports this)
```
cd /home/scratch.gramnarayan_coreai/dev/TensorRT-LLM
export TRITON_CACHE_DIR=/home/scratch.gramnarayan_coreai/.triton/cache
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models
CUDA_VISIBLE_DEVICES=1 SPEC=1 REGSTYLE=0 ATTN=trtllm \
  TARGET_MODEL=$LLM_MODELS_ROOT/llama-3.1-model/Llama-3.1-8B-Instruct \
  DRAFT_MODEL=$LLM_MODELS_ROOT/LLaMA3.1-8B-Instruct-DFlash-UltraChat \
  TGT_LAYER_IDS=1,8,15,22,29 \
  python -u debug/spikes/ad_dflash_qwen3_8b_bnr.py > debug/logs/llama_run.log 2>&1
```
Env toggles available in the spike / code: `SPEC` (1=DFlash, 0=plain target), `DFLASH_NO_RESTORE=1`
(disable the lm_head restore + re-home, observe natural behavior), `DFLASH_SKIP_CTX=1` (skip the draft
scatter+draft pass), `ATTN=flashinfer|trtllm`, `REGSTYLE=1` (torch-cudagraph + registry-style config).
The shared `[RESIZEPROBE]` (in resize_kv_cache) prints target lm_head std before/after the resize fwd.
NOTE: a direct `python` run can intermittently hang at MPI-pool startup — if no progress (no
`Loading weight file`) in ~2 min, kill (`pkill -9 -f ad_dflash_qwen3_8b_bnr`) and relaunch.

## Known state (from the Qwen3 track, 2026-06-03)
- Llama+DFlash with NO_RESTORE+SKIP_CTX: target **lm_head is INTACT** through resize (std 0.0143
  unchanged) but the **output is still GARBAGE** (`garbage tokens like ": OOiterm, <gibberish>, oreOO..."`). So Llama's bug
  is NOT lm_head — a DIFFERENT weight (or the body) is clobbered, OR a target-forward issue.
- Suspected shared root: a memory-orphaning bug in the DFlash path — the large-batch resize forward
  (`set_max_num_tokens_sample`) reuses orphaned model-weight storage blocks; victim weight is
  model-dependent (Qwen3->lm_head; Llama->?).
- The Llama "DFlash" draft (`LLaMA3.1-8B-Instruct-DFlash-UltraChat`) is itself model_type=qwen3
  (Qwen3-arch draft, block_size=10, 5 layers, target_layer_ids=[1,8,15,22,29], mask_token_id=128002).
  Our DFlashModel builds it; only the TARGET is Llama.

## Suggested first experiments (no core edits)
1. Llama plain (SPEC=0) baseline — confirm the Llama target alone is coherent in AD.
2. Llama+DFlash with the restore ENABLED (default, no NO_RESTORE) — does it help? (Expect NO, since
   Llama's lm_head is fine; this confirms the victim isn't lm_head.)
3. Llama+DFlash full (no SKIP_CTX) — full behavior + acceptance + any crash.
4. If the body is clobbered: request (in this worklog) a "dump all target weight std around the resize
   forward" probe from the Qwen track, OR run with REGSTYLE=1 (torch-cudagraph) to see if the
   compile/forward path differs.

## Attempts

### Setup confirmed (2026-06-03)
- GPU 1 free (all 8 H100 idle at start). Using CUDA_VISIBLE_DEVICES=1.
- Paths exist: target `llama-3.1-model/Llama-3.1-8B-Instruct`, draft
  `LLaMA3.1-8B-Instruct-DFlash-UltraChat` (model_type=qwen3, vocab=128256, block_size=10,
  5 layers, target_layer_ids=[1,8,15,22,29], tie_word_embeddings=False, mask_token_id=128002).
- IMPORTANT (read core, no edit): `_maybe_restore_target_lm_head` only restores when target
  lm_head std > 0.5. For Llama the lm_head stays ~0.0143 (per accuracy worklog EXP-A), so the
  restore is a **no-op for Llama** regardless of NO_RESTORE. => Experiment 2 (restore on/off)
  will be IDENTICAL for Llama by construction; the victim is NOT lm_head, so the restore can't help.

### Exp1 — Llama plain (SPEC=0) baseline — attempt 1: CUBLAS crash at resize validation forward
Config: `SPEC=0 REGSTYLE=0 ATTN=trtllm` Llama target, bnr spike (torch-simple, max_batch_size=4,
cuda_graph batch_sizes=[1,2,4], kv max_tokens=2048). Log `debug/logs/llama_exp1_plain_spec0.log`.
- Crashed at `kvcache.py:763` `mod(**cm.named_args)` (the resize_kv_cache **validation forward** at
  `set_max_num_tokens_sample`), inside `torch_swiglu_mlp` -> `F.linear` (layer 0 MLP gate_proj),
  with `CUBLAS_STATUS_EXECUTION_FAILED` then a follow-on illegal-memory-access at teardown.
- This is the SAME crash the Qwen3 track hit in their accuracy-worklog "Attempt 1" (plain control
  also dies at the cache-init/resize validation forward in F.linear, flagged flaky/env). So this is
  NOT Llama-DFlash-specific — it's the plain-target resize-validation-forward flakiness.
- Action: retrying (per worklog guidance the crash is intermittent).

### Exp1 — Llama plain (SPEC=0) baseline — attempt 2: SAME CUBLAS crash (deterministic, not flaky)
Log `llama_exp1_plain_spec0_try2.log`. Identical crash at `kvcache.py:763` resize validation forward
in `torch_swiglu_mlp` F.linear -> CUBLAS_STATUS_EXECUTION_FAILED. So with the **bnr default config**
(torch-simple, max_seq_len=2048, max_batch_size=4) the plain Llama target crashes deterministically
in the large-batch resize forward (`set_max_num_tokens_sample`). NOT flaky here.

### ✅ Exp1 — Llama plain (SPEC=0) baseline — REGSTYLE=1: COHERENT (baseline established)
Config: `SPEC=0 REGSTYLE=1 ATTN=trtllm` (torch-cudagraph, max_seq_len=512, max_batch_size=128).
Log `llama_exp1_plain_spec0_regstyle.log`. Output:
> ": The capital of France is Paris."
=> **Plain Llama target in AD is COHERENT.** The CUBLAS crash is specific to the bnr *default* config
(torch-simple / max_seq_len=2048 / mbs=4) large-batch resize forward, NOT to the Llama target itself.
Use **REGSTYLE=1** as the stable config for the DFlash experiments (its resize forward survives).
Relevant toggle found (read-only): `DFLASH_SMALL_RESIZE=1` in `kvcache.py:765` skips
`set_max_num_tokens_sample()` (keeps the resize forward at the small/default batch) — an alternative
way to dodge the large-batch CUBLAS crash without core edits.

### Exp3 — Llama+DFlash full (SPEC=1) REGSTYLE=1 — crash: restore probe `.item()` during graph capture
Config: `SPEC=1 REGSTYLE=1 ATTN=trtllm` (torch-cudagraph). Log `llama_exp3_dflash_full_regstyle.log`.
- RESIZEPROBE: lm_head **0.01427 before AND after the resize forward, same dptr** (confirms EXP-A:
  Llama lm_head is NOT the victim; resize forward is clean for Llama).
- Then crashes during `compile_model` torch-cudagraph **graph capture** (batch_size=128) at
  `modeling_dflash.py:479` `_maybe_restore_target_lm_head` -> `param.data.float().std().item()`:
  `cudaErrorStreamCaptureUnsupported` (the `.item()` CPU-sync is illegal inside CUDA-graph capture).
  => the lm_head-restore WORKAROUND is incompatible with the torch-cudagraph (REGSTYLE=1) path.
  For Llama the restore is a no-op anyway (lm_head fine), so disable it with `DFLASH_NO_RESTORE=1`.
- This is a WORKAROUND-vs-cudagraph bug, not a Llama correctness signal. (REQUEST-TO-QWEN candidate:
  guard the restore probe to skip `.item()`/std-check during `torch.cuda.is_current_stream_capturing()`,
  or move the corruption check off the capture stream.)

### ✅ Exp3 — Llama+DFlash FULL (SPEC=1) torch-simple + NO_RESTORE — runs, GARBAGE (reproduces EXP-A)
Config: `SPEC=1 REGSTYLE=0 ATTN=trtllm DFLASH_NO_RESTORE=1` (bnr default torch-simple, mbs=4,
max_seq_len=2048). Log `llama_exp3_dflash_torchsimple_norestore.log`. RAN to completion, no crash.
- RESIZEPROBE: lm_head **0.01427 before AND after resize forward + after manager, same dptr** —
  Llama lm_head intact through resize (confirms NOT the victim).
- Output GARBAGE: `garbage tokens like ": OOiterm, <gibberish>, oreOO ..."`. step0: `new_tokens_2d=[[20066,0,0,0,0]]`,
  `lens=[1]` (0 draft accepted), `draft=[[40215,...]]`, `ctx_len=[40]`.
- Tokenizer decode (Llama-3.1-8B-Instruct): accepted **20066='OO'** (GARBAGE from token 0),
  draft 40215='.ms'. (Coherent 'Paris'=12366/60704, 'The'=578/791.) => garbage from the FIRST
  prefill token, same as Qwen3 — but lm_head is fine, so a DIFFERENT target weight/body is wrong.

### ★ Exp4 — Llama+DFlash SPEC=1 + DFLASH_SKIP_CTX=1 + NO_RESTORE — **COHERENT** (root cause located)
Config: add `DFLASH_SKIP_CTX=1` (skip the ctx-scatter + draft pass; dummy drafts). Log
`llama_exp4_skipctx.log`. Output **COHERENT**: `": The capital of France is Paris."`; step0
`new_tokens_2d=[[791,...]]` last_acc=**791='The'** (CORRECT), `ctx_len=[0]`, `draft=[[0,0,0,0]]`.
=> **For Llama, skipping the draft scatter/draft pass makes the target COHERENT.** This is the
OPPOSITE of Qwen3 (where SKIP_CTX still produced garbage). So **Llama's corruptor is the DFlash
ctx-scatter / draft pass**, NOT a generic resize/lm_head bug.
- CONFOUND to resolve: in exp4 `resize_kv_cache` was **SKIPPED** (`needs_resize()`=false, summary
  "skipped"), so exp4 also skipped the build-time large-batch resize forward. So exp4-coherent could
  be due to (a) no draft pass at runtime, OR (b) no large-batch resize forward at build. In exp3 the
  resize forward RAN with SPEC=1 (= the FULL wrapper forward incl. scatter+draft at LARGE batch).
  Hypothesis: the **large-batch resize forward running the draft scatter/draft pass** orphans/clobbers
  a target body weight (the DFlash UNPAGED ctx-K/V caches + draft activations at big batch reuse a
  freed model-weight block). DECISIVE NEXT: full run (no SKIP_CTX) + `DFLASH_SMALL_RESIZE=1` (resize
  runs but at SMALL batch) — if coherent, it's the large-batch resize draft pass; if garbage, the
  draft pass corrupts at runtime too.
