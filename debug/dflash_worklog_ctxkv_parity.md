<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DFlash AD — context-KV parity test: SCOPING / HANDOFF (read this first)

## STATUS: DONE ✅ (2026-06-09)
Test landed: `tests/unittest/auto_deploy/singlegpu/models/test_dflash_context_kv_parity.py`
(`test_precompute_context_kv_matches_hf`, params `fp32-tight` + `bf16-production`). Both PASS.
- Builds AD `DFlashModel` via the factory (mirrors `test_dflash_factory.py::test_draft_weights_load_strict`)
  and the HF vanilla draft via `AutoModel.from_pretrained(Qwen3-8B-DFlash-b16, trust_remote_code=True)`,
  same checkpoint into both. HF reference replicates the context-only slice of
  `Qwen3DFlashAttention.forward` using HF's OWN `fc/hidden_norm/k_proj/v_proj/k_norm/rotary_emb`
  + remote-module `apply_rotary_pos_emb` (imported via `type(hf).__module__`).
- **RESULT — the port is correct; no context-KV math bug.**
  - **fp32:** V bit-exact (maxabs 0.0); K matches to ~1e-4. Tight check PASSES.
  - **bf16:** V STILL bit-exact (no norm/RoPE). K: maxabs ~8.6e-2, **mean-abs ~6e-4**, only ~0.09%
    of elements over a 2e-2 tol; huge maxrel is the classic near-zero-K RoPE artifact. At |K|~10,
    one bf16 ULP ≈ 0.08 ⇒ this is ~1 ULP of rounding in `k_norm + RoPE`, symmetric, NOT a systematic
    shift. bf16 tol set to rtol 3e-2 / atol 1.6e-1 + a mean-abs guard (`< 1e-2 * mean|K_ref|`) so a
    real (systematic) regression still fails while ULP outliers pass.
- **The ~1.5pt AD-vs-PyT acceptance "gap" is NOT real** — verified on full GSM8K (user-confirmed
  2026-06-09); it's within run-to-run / bf16 variance, not a regression. This parity test
  independently corroborates that: the context-KV path is fp32-exact and bf16-V-exact, so there was
  never a context-KV math error to explain a gap. No further chase needed; the test stands as a
  guard against future context-KV regressions.

(Original scoping below kept for provenance.)


**Goal:** a unit test that numerically compares the AutoDeploy DFlash drafter's
`precompute_context_kv` against a reference, feeding identical inputs and asserting the per-layer
K/V match. Validates the AD context path and is the tool to localize the small AD-vs-PyT acceptance
gap (Llama 47.9% vs 49.4%, Qwen3 57.9% vs 56.4% on full GSM8K — see
debug/dflash_worklog_accuracy.md "AD vs PyTorch-NATIVE DFlash").

**REFERENCE CHOICE (user pref):** compare against the **HF vanilla modeling code** — more principled
(no cache-aware machinery), and it's the source the AD port was written from. It exists LOCALLY (see
HF REFERENCE section). Fall back to the PyTorch-backend oracle (modeling_speculative.py) only if the
HF path is fiddly to stand up — but HF is the goal.

## HF REFERENCE (PRIMARY oracle — local, principled)
Files (loaded via `trust_remote_code`, `auto_map: AutoModel -> dflash.DFlashDraftModel`):
- `$LLM_MODELS_ROOT/Qwen3-8B-DFlash-b16/dflash.py`  (the auto_map class) + `modeling_dflash.py`
  (near-identical sibling) + `utils.py`. Also `Qwen3-4B-DFlash-b16/` has the same.
- NOTE: the **Llama** DFlash checkpoint (`LLaMA3.1-8B-Instruct-DFlash-UltraChat/`) has `dflash.py` +
  `utils.py` but NO `modeling_dflash.py`. => use **Qwen3-8B-DFlash-b16** for the HF-parity test.
- Internet IS reachable (HF hub HTTP 200) as a backup, but the code is already local.

HF has **NO standalone `precompute_context_kv`** — the context K/V is computed INLINE inside
`Qwen3DFlashAttention.forward` (dflash.py ~L58-82 / modeling_dflash.py ~L67-91). The relevant
context-only path (with `target_hidden` already = `hidden_norm(fc(raw))`, done in
`DFlashDraftModel.forward` L177/207):
```
k_ctx = self.k_proj(target_hidden)                       # [bsz, N, nkv*hd]
v_ctx = self.v_proj(target_hidden)
k = k_ctx.view(bsz, N, -1, head_dim); k = self.k_norm(k).transpose(1,2)   # k_norm per head_dim
q, k = apply_rotary_pos_emb(q, k, cos, sin)              # cos/sin from self.rotary_emb at positions
v = v_ctx.view(bsz, N, -1, head_dim)                     # V: no norm/rope
```
(In the real forward k_ctx is cat'd with the query `k_noise` before k_norm/RoPE, but k_norm is
per-token/per-head_dim and RoPE is per-position, so the context slice is identical to computing it
alone — exactly what AD's `precompute_context_kv` does.) So the REFERENCE for the test = replicate
this context-only path using the HF model's `fc/hidden_norm/layers[li].self_attn.{k_proj,v_proj,
k_norm}/rotary_emb` + HF `apply_rotary_pos_emb`, stack per-layer -> `[N, L, nkv, hd]`.

The ONE thing the test really probes: AD's RoPE (`torch.ops.auto_deploy.torch_rope_with_explicit_
cos_sin`) + AD k_norm vs HF `apply_rotary_pos_emb` + HF `Qwen3RMSNorm`. Same weights (same ckpt) =>
any delta is kernel/dtype, and is the prime suspect for the acceptance gap.

## The two AD-side / fallback implementations (same math, ported)

| | AutoDeploy (under test) | Oracle (reference) |
|--|--|--|
| file | `tensorrt_llm/_torch/auto_deploy/models/custom/modeling_dflash.py` | `tensorrt_llm/_torch/models/modeling_speculative.py` |
| class / method | `DFlashModel` (L258) `.precompute_context_kv` (**L309**) | `DFlashForCausalLM` (L811) `.precompute_context_kv` (**L989**) |
| input | `captured_hidden [N, num_capture*hidden]` (**RAW**, fc+hidden_norm applied INSIDE) | `projected_hidden [N, hidden]` (**ALREADY fc+hidden_norm'd** by caller) |
| also | `position_ids [N]` | `positions [N]` |
| output | `(k, v)` each `[N, L, nkv, hd]` | `(k, v)` each `[N, L, nkv, hd]` — SAME layout |
| K/V proj | per-layer **separate** `k_proj`/`v_proj` (nn.Linear), looped | **single fused-KV GEMM** (`_fused_kv_weight`, layout `[L0_K|L0_V|L1_K|...]`) |
| k_norm | per-layer `layer.self_attn.k_norm` (RMSNorm), K only | fused stacked RMSNorm (`_k_norm_stacked`, shared eps), K only |
| RoPE (K only) | `torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(k,k,cos,sin,2)`; cos/sin from `self.rotary_emb` | `_fused_rope_inplace` (flashinfer-style cos_sin cache, fp32); positions repeat_interleaved by L |
| V | split only (no norm/rope) | split only |

Pipeline both sides: `fc -> hidden_norm -> (per-layer) k_proj/v_proj -> k_norm(K) -> RoPE(K)`.
NOT through each layer's `input_layernorm` (that is query-stream only). Math is identical IF weights
match; the **separate-vs-fused GEMM** and **per-layer-vs-stacked k_norm** are layout-only (same
result). The **RoPE kernels DIFFER** (auto_deploy op vs flashinfer cache) and the **k_norm
implementation differs** — these are the most likely sources of any small numerical delta and the
prime suspects for the ~1.5pt acceptance gap.

## KEY GOTCHA — input convention differs
AD takes RAW `captured_hidden` and does `ctx = hidden_norm(fc(captured_hidden))` internally (L329).
Oracle expects the input ALREADY projected. So in the test:
- AD call:     `k_ad, v_ad = ad.precompute_context_kv(raw, positions)`
- Oracle call: `proj = ad.hidden_norm(ad.fc(raw))`  (== what PyT caller does, speculative/dflash.py:320-321)
              `k_ref, v_ref = oracle.precompute_context_kv(proj, positions)`
(Use the SAME fc+hidden_norm weights for `proj` — load both models from the same checkpoint so fc /
hidden_norm / k_proj / v_proj / k_norm / rope weights are identical.)

## Proposed test design (DOABLE — single eager call each), use Qwen3-8B-DFlash-b16
1. Load the DFlash draft checkpoint into BOTH model objects (same weights):
   - REFERENCE (HF, primary): `AutoModel.from_pretrained(<Qwen3-8B-DFlash-b16>,
     trust_remote_code=True)` → `dflash.DFlashDraftModel` (or import the local `dflash.py` directly).
     Then compute the context K/V via the inline path documented in HF REFERENCE above (per-layer
     k_proj/v_proj on `hidden_norm(fc(raw))`, k_norm, HF `apply_rotary_pos_emb`). Write a small
     helper `ref_precompute_context_kv(hf_model, raw, positions) -> (k,v) [N,L,nkv,hd]`.
   - AD draft: reuse `tests/unittest/auto_deploy/singlegpu/models/test_dflash_factory.py` fixtures
     (it already builds the AD draft via `DFlashOneModelFactory` / `DFlashModel`). Confirm how it
     materializes + load_state_dict's the draft, and grab the `DFlashModel` to call
     `precompute_context_kv` directly.
   - FALLBACK only if HF stand-up is fiddly: `DFlashForCausalLM(draft_config)` from
     modeling_speculative.py (has a real `precompute_context_kv` at L989, input already projected) —
     see `tests/unittest/_torch/speculative/hw_agnostic/test_dflash.py` for how it's wired (heavier,
     via `LLM`).
2. Inputs: random `captured_hidden [N, num_capture*hidden]` (bf16 or fp32) + `positions`
   (e.g. `arange(N)` or random within max_pos). N small (e.g. 8–32).
3. Compute `k_ad,v_ad` and `k_ref,v_ref` as above; `torch.testing.assert_close`.
   - tol: bf16 ~ rtol/atol 1e-2; for a TIGHT check load weights in fp32.
   - Compare V first (no norm/rope → should match nearly exactly = isolates GEMM/load correctness),
     then K (norm+rope → where divergence, if any, will show).
4. If K mismatches but V matches → the delta is in k_norm and/or RoPE. Drill: compare pre-RoPE K
   (norm only) vs post-RoPE K separately to attribute it. That attribution is the real payoff.

## Construction details already confirmed
- AD `DFlashModel.__init__`: `num_capture_layers = len(target_layer_ids)` (L274),
  `fc_in = hidden_size * num_capture_layers` (L277). `self.fc`, `self.hidden_norm`, `self.layers`
  (each `.self_attn` has `k_proj/v_proj/k_norm/num_kv_heads/head_dim`), `self.rotary_emb`.
- Oracle `DFlashForCausalLM`: `self.fc` (L956), `self.hidden_norm` (L966), `_build_fused_kv_buffers()`
  (builds `_fused_kv_weight`, `_k_norm_stacked`, `_k_norm_eps`, `_num_attn_layers`, `_num_kv_heads`,
  `_head_dim`), `target_layer_ids`/`block_size` from dflash_config (L864-865).
- Both consume the same HF DFlash checkpoint:
  - Qwen3:  target `Qwen3/Qwen3-8B`,            draft `Qwen3-8B-DFlash-b16`            (target_layer_ids 1,9,17,25,33)
  - Llama:  target `llama-3.1-model/Llama-3.1-8B-Instruct`, draft `LLaMA3.1-8B-Instruct-DFlash-UltraChat` (1,8,15,22,29)
  under `LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models`.

## Test placement
`tests/unittest/auto_deploy/singlegpu/...` (mirror test_dflash_factory.py location). Mark single-GPU /
hw-agnostic as appropriate. Needs the checkpoint → likely gated like other model tests
(LLM_MODELS_ROOT). Keep N small so it's fast.

## Open questions to resolve at impl time
1. Lightest way to instantiate the oracle `DFlashForCausalLM` draft with loaded weights (without a
   full `LLM`). Check modeling_speculative draft-build + test_dflash.py.
2. Does the AD factory expose the draft `DFlashModel` post-load conveniently (test_dflash_factory.py
   should show this), or do we build `DFlashModel(config)` + load_state_dict directly?
3. RoPE: ensure both use the SAME rope_theta/positions; the cos/sin path differs (auto_deploy op vs
   flashinfer fp32 cache) — if that's the only mismatch, it both validates the port AND likely
   explains the acceptance delta.
4. dtype: run fp32 for a tight numerical check; also confirm bf16 (production) stays within a
   sensible tol.

## Context (so this doc stands alone)
DFlash AD is fully working E2E (accuracy matches PyT, cudagraph lossless, overlap fine — see
debug/dflash_worklog_accuracy.md). The ONLY open numerical question is the ~1.5pt acceptance delta
vs PyT-native DFlash; this parity test is how we'd chase it. Other open items: task #9 (proper
resize fix so we can drop the explicit free_gpu_memory_fraction=0.0), and PR cleanup (strip
DFLASH_SKIP_CTX/DFDBG_ACCEPT scaffolding in modeling_dflash.py; add a real AD DFlash accuracy test).
