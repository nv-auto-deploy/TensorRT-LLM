<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash worklog — export-preservation of `precompute_context_kv`

Per-issue worklog. Issue: after the draft is exported to a GraphModule, the eager
`DFlashModel.precompute_context_kv` method (and the per-layer projection nn.Modules it needs) are
gone, so `_scatter_context_kv` → `self.draft_model.model.precompute_context_kv(...)` raises
`AttributeError: 'Module' object has no attribute 'precompute_context_kv'`.

## Lightweight test (loop target)
`debug/spikes/dflash_export_precompute_repro.py` — builds ONLY the draft (small, ~2GB; no 8B
target), loads weights, exports it via the real `torch_export_to_gm` + the DFlash dynamic-shape
lookup, applies the `DFlashDraftModelExportInfo.post_process`, then asserts
`gm.model.precompute_context_kv(hs, positions)` works AND matches the pre-export eager model's
output. Fast (~30s). Loop on THIS until green, then re-run the full E2E smoke.

## Prior context (resolved upstream of this issue)
- meta-tensor blocker — FIXED: `load_or_random_init` now materializes+loads the draft (mirrors
  Eagle delegating to its draft sub-factory); `_build_model` applies config dtype on the meta path.
- SymInt-in-allocate — FIXED: draft seq dim made STATIC in `DFlashDraftModelExportInfo` (the draft
  always runs at fixed `block_size`), so `block_size = q_fake.shape[1]` is concrete.

## Design decision (user-directed)
Preserve everything precompute needs IN `post_process` (set_submodule + keepalive, loop over
layers), accept minor duplication with good names — NOT a separate eager precompute model. Then
the existing call site `self.draft_model.model.precompute_context_kv` keeps working unchanged.

## Attempts
- **Attempt 1 (2026-06-02) — re-attach + rebind in post_process — ✅ lightweight test GREEN.**
  `DFlashDraftModelExportInfo.post_process` now, after export: `inner_gm = sub_gm.get_submodule("model")`,
  loops `set_submodule("model.{fc,hidden_norm,rotary_emb,layers}", eager_module)`, keepalive sentinels
  on `model.fc.weight`/`model.hidden_norm.weight` (precompute-only; layer projections kept by graph
  usage; rotary has no persistent weight), then rebinds
  `inner_gm.precompute_context_kv = types.MethodType(DFlashModel.precompute_context_kv, inner_gm)`.
  Lightweight test: (1) post-export precompute MATCHES eager exactly (re-attached modules share the
  graph's params, clone=False), (2) query-block forward still runs after replacing `model.layers`
  with the eager ModuleList. No second weight copy. Next: re-run full E2E smoke.

## RESOLVED (2026-06-02)
Full E2E smoke now runs end-to-end with NO crash: export → cache_init → compile → generate all
complete. `precompute_context_kv` AttributeError gone. Three blockers cleared in sequence this
session: meta-tensor, SymInt-in-allocate, precompute-export. **New issue (separate worklog
`dflash_worklog_accuracy.md`):** output is garbage (`".ap destination streams streams..."`) and
accepted/iter=0.000 — a correctness bug in the forward, likely the target path. Issue closed here.
