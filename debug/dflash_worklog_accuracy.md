<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash worklog — E2E accuracy / zero-acceptance

Per-issue worklog. Issue: the AD DFlash E2E smoke now RUNS (no crash) but produces garbage and
accepts nothing.

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
