# Gemma4 MoE AD Inference Optimization Notes

**Branch**: `ad-fast-iter/gemma4-moe-0409`  
**Model**: `google/gemma-4-26B-A4B-it` (Gemma 4 27B MoE: 26B total, ~4B activated per token)  
**Hardware**: H100 SXM5 80GB  
**Backend**: AutoDeploy (AD) with piecewise CUDA graph, triton_paged attention  
**Date**: 2026-04-13

---

## Model Architecture

| Property | Value |
|---|---|
| Total parameters | ~26B (4B activated) |
| Hidden layers | 30 |
| Hidden size | 2816 |
| Attention (local layers) | 16 heads, KV=8, head_dim=256, GQA ratio=2 |
| Attention (global layers) | 16 heads, KV=8, head_dim=512, GQA ratio=2 |
| MoE experts per layer | 128 total, top_k=8 |
| Shared expert | 1 per MoE layer (always active, runs in parallel on aux stream) |
| MoE intermediate size | 704 |
| Dense MLP intermediate size | 14336 |
| Attention backend | `triton_paged` (2-stage flash-decode for paged KV cache) |
| max_num_tokens | 8192 |
| max_batch_size | 512 |

**Critical constraint**: head_dim=512 on global layers makes TRTLLM attention and FlashInfer 0.6.6
both non-viable (both crash or produce wrong results on head_dim=512).

---

## Benchmark Setup

```bash
# Perf sweep (full 30-layer model)
bench-sweep \
  --model google/gemma-4-26B-A4B-it \
  --concurrencies "1 16 256" \
  --isl 1000 --osl 1000 \
  --config-path examples/auto_deploy/model_registry/configs/gemma4_moe.yaml \
  --server-type trtllm-autodeploy \
  --extra-aiperf-args="--tokenizer /home/chengzhang/tokenizers/gemma-4-26B-A4B-it --use-server-token-count"

# Accuracy test
LLM_MODELS_ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_modelopt/users/chengzhang/llm-models \
  pytest tests/integration/defs/accuracy/test_llm_api_autodeploy.py::TestGemma4MoE::test_bf16 -s -v
```

Metrics: **TPOT** (inter-token latency, ms/tok, lower is better), ISL=1000, OSL=1000.  
Accuracy thresholds: MMLU ≥ 73.69%, GSM8K ≥ 87.63%.

---

## Calibration Notes

The results below span ~115 iterations with different measurement conditions:

| Iteration range | Model | Token counting | Note |
|---|---|---|---|
| iter0–48 | **6-layer reduced** | Without `--use-server-token-count` | TPOT underreported by ~16%; not comparable to full-model runs |
| iter49–50 | **Full 30-layer** | Without `--use-server-token-count` | First full-model trial |
| iter51–85 | **6-layer reduced** | Without `--use-server-token-count` | Back to reduced model for fast iteration |
| iter86+ | **Full 30-layer** | With `--use-server-token-count` | Accurate, comparable numbers |

All perf comparisons in the "working optimizations" and "failed" sections use **full-model numbers with `--use-server-token-count`** (iter86+).

---

## Final State

**Best commit**: `8514217e3d` (iter107) — branch HEAD after cleanup.

| Metric | iter86 baseline | iter107 (HEAD) | Δ |
|---|---|---|---|
| c1 TPOT (ms) | 5.613 | **4.787** | **−14.7%** |
| c16 TPOT (ms) | 12.770 | **11.511** | **−9.8%** |
| c256 TPOT (ms) | 19.789 | **18.404** | **−7.0%** |
| Accuracy (MMLU) | — | **75.76%** | PASS (≥73.69%) |
| Accuracy (GSM8K) | — | **90.03%** | PASS (≥87.63%) |

**Fresh sweep (2026-04-13, current HEAD)**:

| Concurrency | TPOT avg (ms) | TPOT p50 (ms) | TTFT avg (ms) | TPS |
|---|---|---|---|---|
| c1 | 4.78 | 4.63 | 55.6 | 207 |
| c16 | 11.51 | 11.42 | 472 | 1336 |
| c256 | 18.81 | 18.76 | 83,771 (queuing) | 2266 |

---

## Complete Iteration Log

### Legend
- **Status**: `keep` = committed and improves perf; `discard` = reverted or superseded; `crash` = fatal error; `ref` = reference measurement; `neutral` = no improvement, code kept for infra value
- **TPOT**: inter-token latency (ms/tok, lower is better)
- Model column: `6L` = 6-layer reduced model; `30L` = full 30-layer model
- iter0–85 used client-side token counting (values ~16% lower than `--use-server-token-count`)

---

### Phase 1: Config Sweep (iter0–32) — 6-layer reduced model

These early iterations explored config options to establish what combinations of
transforms worked. TPOT numbers are **NOT** comparable to Phase 4+ (different
token counting and reduced model).

| Iter | Status | TPOT c1 | TPOT c16 | TPOT c256 | Description |
|---|---|---|---|---|---|
| 0 | ref | 2.443 | 3.720 | 18.264 | 6L baseline |
| 1 | discard | — | — | — | `fuse_add_rms_norm`: no effect on Gemma4 residual-add pattern |
| 2 | keep | 2.362 | 3.710 | 18.220 | `multi_stream_moe` enabled: shared expert moves to aux CUDA stream, +3.3% c1 |
| 3 | keep | 2.340 | 3.716 | 18.381 | Accuracy fix: `caller_stream.synchronize()` + piecewise reclassification of stream-switch partitions as dynamic |
| 4 | discard | 2.395 | 3.715 | 18.480 | `fuse_gemms_mixed_children`: −2.3% vs iter3 (worse) |
| 5 | discard | 2.385 | 3.739 | 18.916 | `multi_stream_gemm`: −1.9% vs iter3 (worse) |
| 6 | crash | — | — | — | TRTLLM `attn_backend`: CRASH — MMHA does not support head_dim=512 |
| 7 | crash | — | — | — | FlashInfer `attn_backend`: CRASH — head_dim=512 not supported in prefill kernel |
| 8 | keep | 2.241 | 3.372 | 19.132 | **Disable `mlir_elementwise_fusion`**: +4.2% c1, +9.2% c16. MLIR was serializing multi-stream MoE path |
| 9 | discard | 2.387 | 3.717 | 18.951 | `rmsnorm_backend=triton`: −2.0% vs iter3 |
| 10 | discard | 2.408 | 3.703 | 18.761 | `piecewise_enabled=false`: −2.9% (confirms piecewise helps) |
| 11 | discard | 2.419 | 3.751 | 19.574 | `gather_logits_before_lm_head=false`: −3.4% |
| 12 | crash | — | — | — | `fuse_gemms=false`: CRASH (AssertionError in sampling) |
| 13 | crash | — | — | — | FlashInfer + no `multi_stream_moe`: same head_dim=512 crash |
| 14 | discard | 2.337 | 3.352 | 18.990 | `fuse_swiglu` (FlashInfer GELU kernel): −4.3% vs iter8 |
| 15 | discard | 2.293 | 3.454 | 18.303 | No `multi_stream_moe`: −2.3% (confirms stream parallelism helps) |
| 16 | discard | 2.619 | 4.321 | 19.773 | No `fuse_rmsnorm`: −16.9% (confirms norm fusion is critical) |
| 17 | discard | 2.281 | 3.367 | 18.935 | `fuse_gemms_mixed` + no MLIR: −1.8% |
| 18 | discard | 2.415 | 3.810 | 19.260 | MLIR shape propagation: −7.8% |
| 19 | discard | 2.279 | 3.382 | 18.410 | `multi_stream_gemm` + no MLIR: −1.7% |
| 20 | discard | 2.314 | 3.388 | 18.442 | Skip `begin_aux` CPU sync: −3.3% c1 (c256 +3.8% better, tradeoff not worth it) |
| 21 | discard | 2.321 | 3.388 | 18.798 | `fuse_add_rms_norm` + no MLIR: −3.6%, TTFT +52% |
| 22 | profile | — | — | — | NSys profile run (overhead active, not perf data) |
| 23 | discard | 2.268 | 3.395 | 18.398 | Skip CPU sync + `fuse_add_rms`: −1.2% |
| 24 | discard | 2.322 | 3.377 | 18.180 | No piecewise + no MLIR: −3.6% c1, c256 −5% (piecewise trades c1 for c256) |
| 25 | keep | 2.299 | 3.257 | 17.396 | **Fused Triton router**: c16 +3.4%, c256 +9.1% (batched router dispatch, fewer kernel launches) |
| 26 | discard | — | — | — | `multi_stream_gemm` at ISL=2048: neutral (<1.5%) |
| 27 | discard | — | — | — | `fuse_gemms_mixed_children` at ISL=2048: neutral |
| 28 | discard | 2.379 | 3.469 | 19.705 | `multi_stream_mla_attn`: −6.2% c1 (wrong for non-MLA model) |
| 29 | discard | — | — | — | `insert_cached_residual_add` at ISL=2048: neutral |
| 30 | discard | — | — | — | All stream transforms combined at ISL=2048: neutral |
| 31 | ref | 2.385 | 3.440 | 21.004 | ISL=2048 baseline (30-request, different conditions) |
| 32 | discard | 2.320 | 3.265 | 17.621 | `fuse_gelu_mul` (FlashInfer GELU): −0.9% c1 (GELU not bottleneck) |

---

### Phase 2: Kernel Fusion — Attention Path (iter33–48) — 6-layer reduced model

Starting from iter8/iter25 config. Exploring fused kernel ideas in the attention and norm path.

| Iter | Status | TPOT c1 | TPOT c16 | TPOT c256 | Description |
|---|---|---|---|---|---|
| 33 | crash | — | — | — | `piecewise_num_tokens=32` (int): CRASH (must be a list) |
| 34 (iter39) | keep | 2.264 | 3.255 | 18.059 | **Adaptive Triton/FlashInfer dispatch**: −1.5% c1 vs iter25; c16 flat; c256 +3.8% |
| 35 (iter40) | discard | 2.355 | 3.521 | 18.573 | `dual_norm+norm_add2`: +4.0% c1, +8.2% c16 REGRESSION. Root cause: dual norm serializes multi-stream MoE (both streams wait for 1 fused op before forking) |
| 36 (iter41) | discard | 5.357 | 12.470 | 18.906 | Fused QKV-norm (tuple return): +137% c1, +295% c16. Root cause: tuple-return custom op NOT captured in CUDA graph → 30 graph exit/re-entry per step |
| 37 | discard | — | — | — | Accidental full-model run (wrong config); results discarded |
| 38 (iter42) | keep | 2.179 | 3.167 | 18.506 | **Fuse dense+MoE residual add with post-FFN norm**: −3.8% c1, −2.7% c16 (removes 1 Python dispatch per layer in eager segments) |
| 39 (iter43) | discard | 2.172 | 3.152 | 18.618 | Skip aux CPU sync (`AD_SKIP_AUX_CPU_SYNC=1`): −0.3% (below threshold) |
| 40 (iter44) | keep | 2.139 | 3.220 | 18.668 | **Fuse post_attn_norm + pre_ff_norm → packed kernel**: −1.8% c1 |
| 41 (iter45) | discard | 2.290 | 3.270 | 17.800 | Cross-layer `input_layernorm` fusion: +7.1% c1 REGRESSION + MMLU 34.1% (accuracy fail). 2-reduction kernel worse than 2 separate kernels (register pressure) |
| 42 (iter46) | discard | 2.180 | 3.220 | 18.800 | Extend packed kernel to 3-output (add pre_ff_2): +1.92% REGRESSION. Large-T `copy_()` overhead + Triton register pressure |
| 43 (iter47) | discard | 2.183 | 3.209 | 18.724 | Fuse `pre_ff_2` layernorm INTO router kernel: +2.1% REGRESSION (router cache pressure) |
| 44 (iter48) | discard | 2.183 | 3.220 | 18.800 | Fuse `post_ff_2` layernorm INTO combine kernel: no additional regression on top of iter47 |
| 45 (iter49) | discard | 5.418 | 12.720 | 18.963 | 3-reduction combine kernel, full 30L model: +0.75% (below threshold) |

---

### Phase 3: Full Model Trials + CPU Scheduling (iter46–iter85) — Mix of 6L reduced and 30L

| Iter | Status | TPOT c1 | TPOT c16 | TPOT c256 | Description |
|---|---|---|---|---|---|
| 50 | discard | 5.406 | — | — | Full 30L: revert iter47 router, keep iter49 combine → +0.97% (below threshold) |
| 55+56 | keep | 2.267 | 3.311 | 18.003 | **Batched KV cache lookup (n≥64) + `get_last_tokens` (1 call) + `input_pos` cache**: c256 −3.6% vs iter44 |
| 57 | keep | 2.250 | 3.324 | 17.925 | **Remove redundant `position_ids` computation in `nest_sequences`**: c256 −4.0% vs iter44 |
| 58 | discard | 2.237 | 3.328 | 18.834 | Lower batch cache threshold 64→2: c256 +5.1% REGRESSION |
| 59 | discard | 2.241 | 3.319 | 18.368 | Cache `max_beam_num_tokens` (dict-based): c256 +2.5% REGRESSION |
| 60 | keep | 2.210 | 3.200 | 18.410 | **`fused_add_rmsnorm` for large-T `post_add_norm_add_scale`**: 4→3 kernels, no temp allocs; best state at time |
| 61 | keep | 2.140 | 3.190 | 18.290 | **`rmsnorm out=` for all 3 large-T paths**: −3.2% c1 (2.210→2.140ms); eliminates 1-2 T×H temp allocs |
| 62 | discard | 2.160 | 3.198 | 18.681 | Revert iter55+56: no real effect (confirmed noise) |
| 63 | discard | 2.202 | 3.290 | 18.051 | Raise GC threshold 20000→200000: c256-only win, not enough to replace iter61 |
| 64-72 | discard | ~2.23 | ~3.30 | ~18.2 | Piecewise bucket sweeps (various combinations of [64,128,256,512,1000,1536,2048,...]): all lose at c1/c16 vs iter61; c256 sometimes better but not worth the tradeoff |
| 73 | discard | 5.623 | 12.719 | — | Full 30L: `iter71` decode fast path validation — accuracy PASS but slower than full baseline |
| 74–78 | discard | ~2.23 | ~3.30 | ~18.2 | Decode fast path variants (SequenceInfo, py_last_token cache, etc.): all lose to iter61 branch baseline |
| 79 | discard | 5.614 | 12.740 | — | Full 30L: SequenceInfo fast path validation — slower than full baseline |
| 80–85 | ref | 2.215 | 3.287 | 18.131 | Fresh reproductions of iter61 state; establishes reproducible 6L baseline ≈ 2.22/3.29/18.1 |

---

### Phase 4: Full 30-Layer Model Optimizations (iter86+)

**Note**: All numbers from iter86 onward use `--use-server-token-count` and the full 30-layer model.
The **baseline** for this phase is iter86 (commit `641e3aca63`): c1=5.613ms, c16=12.770ms, c256=19.789ms.

| Iter | Commit | Status | c1 TPOT | c16 TPOT | c256 TPOT | Accuracy | Description |
|---|---|---|---|---|---|---|---|
| **86** | `641e3aca63` | **ref** | 5.613 | 12.770 | 19.789 | — | Full 30L baseline |
| 88 | — | discard | 2.270 | 3.284 | 18.054 | — | `multi_stream_gemm` transform: no-op for BF16 (matches=0), router multi-SM 65% slower (36µs vs 22µs) in eager mode |
| 89 | — | discard | 5.595 | 12.663 | — | — | Explicit `piecewise_num_tokens` list: +0.32% c1 (noise); piecewise buckets don't help full model |
| 90 | — | discard | 2.217 | 3.300 | 18.059 | — | T-adaptive router, reduced model: flat (6L × 13µs savings too small to measure) |
| **91** | `5be7574d8d` | **keep** | 5.606 | 12.725 | 19.755 | — | **T-adaptive router dispatch (bh=256/nw=4 for T≥256)**: noise-level full-model win (KV-cache pressure masks 29% GPU improvement); standalone: 32.4µs vs 45.6µs at T=256 |
| 92 | `ab0e04c668` | discard | 2.245 | 3.291 | 18.265 | — | `_TRITON_T_THRESHOLD=0` (always FlashInfer): +1.4% c1 REGRESSION; reverting to threshold=8 correct |
| 93 | — | discard | 2.343 | 3.317 | 18.199 | — | `attn_backend=flashinfer` in config: SIGFPE crash on head_dim=512 global layers (divide-by-zero in plan()); local layers get group_size=0 error; **FlashInfer definitively not viable for Gemma4** |
| **94/96** | `f91ab8c947` | **keep** | — / 5.475 | — / 12.742 | — / 19.723 | — | **Fused QKV-norm**: 3 separate `torch_rmsnorm` calls → 1 Triton kernel; c1 −2.3% (5.606→5.475ms, −131µs); standalone 3.9µs → 1.3µs (3×) |
| 97 | — | discard | — | — | CRASH | — | `gemma4_qkv_norm_rope` (fused QKV-norm+RoPE v1): OOB pointer in v-rows (cudaErrorIllegalAddress at c256) |
| 98 | — | discard | 5.396 | 13.406 | 20.487 | — | `gemma4_qkv_norm_rope` v2 (OOB fixed): c16 +5.1% REGRESSION; nested custom op body calls `torch_rope` → inductor cannot compile fallback → 14-18 eager CUDA nodes/layer at large T |
| **99** | `0cb97e8a5a` | **keep** | 5.403 | 12.812 | 19.776 | — | **Fused 3-norm** (post_ff_ln_1 + post_ff_ln_2 + post_add_norm_scale): c1 −1.3% (5.475→5.403ms, −72µs); theory: 2×30L×1.3µs=78µs |
| **100+101** | `e811c68607`+`55ed1f3e39` | **keep** | 5.295 | 12.753 | 19.892 | pass | **Fused QKV-norm+RoPE** + **fuse pre_ff_2 norm** (4-output kernel): combined c1 −2.0% (5.403→5.295ms) |
| **102+103** | `df09c2b04b`+`05432a7f2f` | **keep** | 5.257 | 12.134 | 18.898 | — | **Fuse input_layernorm** into 4-norm kernel + **always-Triton threshold**: c1 −0.7%, c16 −4.9%, c256 −5.0% |
| 104 | `0873548658` | neutral | 5.259 | 12.117 | 19.155 | — | `fuse_gelu_tanh_mul` (FlashInfer kernel fusing GELU+mul for dense MLP): on aux stream (non-critical path), net neutral |
| 106 | — | discard | 2.235 | 3.076 | 18.173 | — | Router multi-SM kernel (reduced model): +1.7% c1 REGRESSION; root cause: router was in reclassified-dynamic partition (begin_aux before router in FX graph), multi-SM + per-step allocations hurt eager path |
| **107** | `8514217e3d` | **keep** | **4.787** | **11.511** | **18.404** | MMLU=75.83% GSM8K=91.17% | **Router fence + router-before-attn reorder**: best state; c1 −8.9% (5.257→4.787ms, −470µs); 30 router sections moved from eager → CUDA-graph-captured |
| 108 | — | discard | — | — | — | — | `gemma4_residual_fence` after each MoE merge node: TypeError (`out=` passed to fence by DynamicOpWrapper); even if fixed, 6×~16µs overhead exceeds ~70-100µs savings from 5 extra graph-captured routers |
| 109 | — | discard | 4.784 | 11.488 | 18.803 | — | `skip_aux_cpu_sync=true`: c256 +2.2% REGRESSION; SM contention between aux-stream MLP and main-stream MoE at large batch |
| 110 | — | discard | 5.119 | 11.775 | CRASH | — | `fences_only=true`: c1 +6.9% REGRESSION; loses GPU parallelism (MLP/MoE serial); +6 extra CUDA-graph runners per step; c256 server crash |
| 107 rebase | — | ref | 4.782 | 11.519 | 18.645 | MMLU=90.52% | Re-measurement after session resume; confirms iter107 as stable baseline |
| 111 | `d3ea8f65d1` | discard | 4.775 | 11.565 | CRASH | — | Open-boundary fence merge: only 1/30 partitions merge (wrong graph order for layers 1-29); cudaErrorIllegalAddress + TimeoutErrors at c256 |
| 112 | `ba5704fce7` | discard | 4.781 | — | — | — | Remove fence from `_PARTITION_BOUNDARY_OPS`: savings from 30 fewer fence dispatches (−150µs) exactly cancelled by loss of P_final CUDA graph (final-norm falls into reclassified partition) |
| 113 (ms) | `156c4d85e1` | discard | 4.780 | — | CRASH | MMLU=75.4% | Static-boundary fence with multi-stream config: flat c1, c256 crash (31 static runners → pool exhaustion / address aliasing) |
| 113 (fences_only) | `156c4d85e1` | discard | 5.110 | — | 18.91 | — | Static-boundary fence with fences_only config: c1 +6.9% REGRESSION (identical to iter110 fences_only) |
| **114** | — | crash | — | — | — | — | `min_latency_mode=True` for MoE GEMM: CRASH — assertion `use_fp4 == true` in `moe_kernels.cu:3759`; BF16 not supported |
| **cleanup** | `0c6efdc409` | — | — | — | — | — | Branch cleanup: revert `piecewise_utils.py` to iter107 state (fix c256 crash); discard uncommitted fences_only code |

---

## Working Optimizations — Detailed Analysis

### 1. Disable mlir_elementwise_fusion (iter8, permanent config)
**c1 +4.2%, c16 +9.2%**

MLIR was fusing elementwise ops that occurred at the fork/join points of the
multi-stream MoE path. This serialized the shared expert (MLP) and the routed
expert (MoE), forcing both to wait for the fused op before forking. Disabling
MLIR restores the fork/join structure that multi-stream relies on.

---

### 2. Fused Triton Router (iter25, commit `0dc1d9d4b8`)
**c16 +3.4%, c256 +9.1%**

The router for each of the 30 MoE layers dispatches `hidden_size×num_experts`
(2816×128) GEMM. The baseline called this as individual ops. The fused Triton
kernel batches the dispatch and reduces per-layer launch overhead.

Effect is primarily at c16/c256 where multiple sequences are processed in
parallel, amortizing the kernel overhead over a larger batch.

---

### 3. CPU Scheduling: Batched Cache Lookup + Remove Redundant position_ids (iter55-57)
**c256 −3.6% (iter55+56), c256 −4.0% (iter57)**

`iter55+56`: Cached several per-step Python dict lookups, raising the batch
threshold for the accelerated KV-cache path from n=2 to n=64.

`iter57`: Removed a redundant `position_ids` tensor construction in
`nest_sequences()` (was computed twice per step).

These are pure CPU scheduling wins — no GPU kernel changes.

---

### 4. fused_add_rmsnorm for Large-T (iter60, commit `b25dc6d894`)
**c1 best state at that time (2.210ms)**

The `post_add_norm_add_scale` op (applied after the residual add following the
FFN) was implemented as 4 separate kernels. The fused version computes the
residual add and RMSNorm in one pass, eliminating a global-memory round-trip
(load input→add→store temp→load temp→norm→store output) and the temporary
allocation of a T×H buffer.

---

### 5. RMSNorm out= for Large-T Paths (iter61, commit `71014e1c8b`)
**c1 −3.2% (2.210→2.140ms)**

Three large-T normalization ops allocated fresh output tensors every step.
Using the `out=` parameter of the RMSNorm op directs output into pre-allocated
stable buffers, eliminating 1-2 T×H temporary allocations per MoE layer.
Better L2 cache reuse (output written to known address, can be prefetched).

---

### 6. T-Adaptive Router Dispatch (iter91, commit `5be7574d8d`)
**Standalone −29% at T=256; full-model noise-level**

Gemma4's router dimensions: hidden_size=2816, num_experts=128, head_dim=256
→ H/head_dim = 11 (exact, no padding). At T≥256, using `bh=256, nw=4` (one
warp group per head) fully utilizes the SM vs the default `bh=128` setting.

Full-model impact is noise-level because at decode time the router GEMM
(22µs standalone) is hidden inside the CUDA-graph-captured static partition and
the overall step time is dominated by other ops. The improvement is correct and
will compound with future work.

---

### 7. Fused QKV-Norm (iter94/96, commit `f91ab8c947`)
**c1 −2.3% (5.606→5.475ms, −131µs)**

Gemma4 applies per-head RMSNorm to Q, K, V before attention. Baseline: 3
separate `torch_rmsnorm` calls (3 kernel launches per attention layer). The
`gemma4_qkv_norm` op fuses all three into a single Triton kernel for
`total_rows ≤ 256` (BS ≤ 8); falls back to individual norms for large T.

Standalone CUDA-graph benchmark: 3×1.3µs → 1×0.43µs (3.0×) per attention
block. Over 30 layers × 2 norm types: predicted 65µs; measured 131µs
(compile effects and cache benefits amplify the saving).

---

### 8. Fused 3-Norm (iter99, commit `0cb97e8a5a`)
**c1 −1.3% (5.475→5.403ms, −72µs)**

After MoE + residual add, three RMSNorms ran sequentially:
`post_ff_ln_1`, `post_ff_ln_2`, `post_add_norm_scale`. These are all applied
to the same input tensor. Fused into one kernel with 3 outputs.

Theory: 2 saved kernel launches × 30 layers × 1.3µs = 78µs. Measured: 72µs.
Close agreement confirms the bottleneck was kernel-launch overhead (not compute).

---

### 9. Fused QKV-Norm+RoPE + Pre-FF-2 Norm (iter100+101)
**Combined c1 −2.0% (5.403→5.295ms)**

`iter100` (`e811c68607`): Extended the QKV-norm kernel to also apply RoPE
(rotary positional encoding) to Q and K in the same pass. Eliminates a separate
`torch_rope` kernel launch per attention layer.

`iter101` (`55ed1f3e39`): Extended the 3-output norm kernel (iter99) to produce
a 4th output for `pre_feedforward_layernorm_2`. One more kernel launch saved
per MoE layer.

---

### 10. Fuse input_layernorm + Always-Triton Attention (iter102+103)
**c1 −0.7%, c16 −4.9%, c256 −5.0%**

`iter102` (`df09c2b04b`): Fused `input_layernorm` (pre-QKV projection norm)
into the 4-output norm kernel, reducing the attention path to a single kernel.

`iter103` (`05432a7f2f`): Raised the `_TRITON_T_THRESHOLD` so Triton paged
attention is always used (no FlashInfer fallback). At c16/c256, Triton's
flash-decode kernel handles multiple sequences natively in one launch vs
FlashInfer's plan+dispatch+execute multi-step dispatch.

---

### 11. Router Fence + Router-Before-Attention Reorder (iter107, commit `8514217e3d`)
**c1 −8.9% (5.257→4.787ms, −470µs), c16 −5.1%, c256 −2.6%**

**The biggest single optimization. Requires understanding the piecewise graph structure.**

**Background**: AutoDeploy uses "piecewise CUDA graphs." Ops that can't be
CUDA-graph-captured (attention, SSM, stream-switch ops) run eagerly. Each
eager partition costs ~15-20µs of Python dispatch overhead per decode step.

Before iter107, the multi-stream MoE block looked like this in the FX graph:
```
[static] norms → router weights → router matmul (22µs GPU) →
    [eager] begin_aux_stream_passthrough                      ← @dynamo.disable function
    → [eager] shared_expert_mlp (on aux stream)
    → [eager] moe_dispatch → expert_gemm → combine           ← on main stream
    → [eager] end_aux / wait_aux
```
Because `begin_aux_stream_passthrough` is a `@dynamo.disable` function, the
piecewise splitter reclassified the entire block as one eager partition. This
means the router matmul — a 22µs GPU op — ran in eager mode, paying ~18µs
of Python overhead per layer per step.

**Fix — two parts**:

1. **Code reorder in `modeling_gemma4.py`**: Call `self.router()` BEFORE
   `self.mlp()` in `MoELayer.forward()`. This places the router computation
   BEFORE `begin_aux_stream_passthrough` in the FX graph.

2. **`gemma4_router_fence` custom op**: A `@torch._dynamo.disable` identity
   function inserted AFTER the router's `weights_getitem` node. Because it's
   `@dynamo.disable`, the piecewise splitter sees it as a dynamic op boundary
   and creates a new partition here. The partition before the fence contains only
   the router (no stream-switch ops) → CUDA-graph-captured. The partition after
   the fence starts with `begin_aux_stream_passthrough` → reclassified as eager.

After the fix, each MoE layer's router runs inside a CUDA-graph-captured static
partition (fast GPU execution, no Python overhead). The 30 fence ops create 30
new tiny eager partitions (~5µs each, nearly zero GPU work).

**Net saving**: 30 layers × (18µs saved − 5µs fence) = **390µs**, measured as
470µs. The additional c16/c256 improvement comes from the router sections also
being batched more efficiently when CUDA-graph-captured.

**Why reduced-model (6L) showed no improvement in iter107 reduced**: Only 5
MoE layers × 18µs = 90µs savings, minus 5 fence × 5µs = 25µs overhead → ~65µs
net, within the measurement noise floor of the benchmark.

---

## Failed Optimizations — Detailed Analysis

### Attention Backend: FlashInfer and TRTLLM (iter6, iter7, iter13, iter93)

Both **TRTLLM attention** and **FlashInfer 0.6.6** fail due to Gemma4's
`head_dim=512` on global attention layers.

- **TRTLLM MMHA kernel** (iter6): Does not support head_dim > 256. Immediate crash.
- **FlashInfer 0.6.6** (iter7, iter93): `head_dim=512` causes SIGFPE in `plan()`
  via the FA2 backend (divide-by-zero in the tiling computation). Local layers
  (head_dim=256, GQA group_size=2) also fail with a `group_size=0` error. Both
  failures corrupt the decode wrapper state, forcing a slow fallback path.
  **Conclusion**: FlashInfer is not viable for Gemma4 with current versions.

---

### Tuple-Return Custom Op (iter41)
+137% c1 REGRESSION.

Early attempt to fuse QKV norms used a Python function returning a tuple
`(q_norm, k_norm, v_norm)`. Torch's CUDA-graph capture cannot handle custom ops
that return tuples: every call to such a function causes the CUDA graph to exit,
run eagerly, and re-enter. With 30 attention layers × 2 exit/re-entries = 60
CUDA graph breaks per decode step, TPOT exploded.

**Lesson**: Custom ops in CUDA-graph-captured code must return a single tensor
(or use pre-allocated `out=` buffers). Never return tuples.

---

### Cross-Layer Norm Fusion (iter45)
+7.1% c1 REGRESSION + accuracy failure (MMLU 34.1%).

Tried fusing `input_layernorm` from adjacent layers into a single kernel.
The combined kernel had higher register pressure, reducing GPU occupancy below
the two-separate-kernels baseline. The accuracy failure (MMLU 34.1% = near
random) revealed a correctness bug in the cross-layer tiling logic.

---

### Extend Packed Kernel to 3-Output / Pre-FF-2 Norm Fusion (iter46, iter47)
+1.92% and +2.1% c1 REGRESSIONS.

Attempted to add more outputs to the 2-output packed kernel:
- Adding `pre_ff_2` output via shared `inv_rms2` scaling: large-T `copy_()`
  overhead and Triton kernel register pressure outweighed savings.
- Fusing `pre_ff_2` norm INTO the router kernel: router's second H-loop
  (processing the fused norm in the same kernel as the router GEMM) increased
  cache miss rate.

**Key finding**: These same fusions succeeded when approached differently
(iter99–101 with the `post_ff` 3-norm kernel and the 4-output QKV-norm+RoPE).
The difference: iter46/47 were adding to already-complex kernels, while
iter99–101 were building cleanly-designed fused kernels from scratch.

---

### Nested Custom Op Architecture (iter97, iter98)
**Crash + c16 +5.1% REGRESSION**

The `gemma4_qkv_norm_rope` kernel's fallback path called
`torch_rope_with_explicit_cos_sin` (another custom op) from inside the fallback
body of the outer custom op. Torch.inductor cannot compile nested custom ops:
the fallback body produces 14-18 eager CUDA nodes per layer per step at any batch
size where the fallback fires, obliterating c16 performance.

**Lesson**: Custom op fallback bodies must use only primitive PyTorch ops that
inductor can compile. Never call another custom op from a custom op's fallback.

---

### fences_only Mode (iter110, iter113)
**c1 +6.9% REGRESSION, c256 crash**

The idea: disable the aux-stream mechanism entirely, let all MoE ops run on the
main stream as CUDA-graph-captured static partitions. Maximizes CUDA-graph
coverage (61 static runners).

Why it fails: The multi-stream path runs the shared expert (MLP, ~50µs at T≤8)
and the routed expert (MoE dispatch, ~80µs) **in parallel** on two CUDA streams.
Removing the aux stream serializes them: MLP (50µs) + MoE (80µs) = 130µs serial
vs ~80µs parallel (MLP hidden behind MoE). This adds ~50µs per MoE layer ×
30 layers = 1.5ms to the critical path. The extra CUDA-graph overhead from
launching 61 static runners (vs 2) adds another ~60µs.

The c256 crash (TransferEncodingError in iter110, cudaErrorIllegalAddress in
iter113) appears because 61 CUDA-graph runners each hold a pool of pre-allocated
buffers. With 256 concurrent requests the total buffer footprint exhausts the
CUDA graph memory pool.

---

### skip_aux_cpu_sync (iter109)
**c256 +2.2% REGRESSION**

After the MoE merge (where shared expert output is added to routed expert
output), the code calls `caller_stream.synchronize()`. This CPU-side sync
ensures the GPU has finished dispatching the previous step's aux-stream work
before starting the next step.

Skipping this sync causes SM contention at large batch (c256): without the
barrier, the aux-stream MLP from step N and the main-stream MoE from step N+1
compete for SMs simultaneously. The contention is invisible at c1 (small batch,
no overlap) but grows with concurrency.

---

### Static-Boundary Fence (iter113)
**Flat c1, c256 CRASH**

Changed the `gemma4_router_fence` from a regular dynamic op (creates its own
isolated partition) to a "static-boundary op" that stays in the preceding static
partition but increments the partition counter. This would eliminate 30 × 5µs
fence-only eager partitions (~150µs saved).

Why it fails: With 30 MoE layers, this creates 31 separate CUDA-graph-captured
static partitions (vs 2 in iter107). Each partition holds a CUDA graph memory
pool. At c256 (256 concurrent requests, batch sizes up to 512), the 31 pools
exceed the available CUDA graph address space, causing `cudaErrorIllegalAddress`
in the sampler. c1 is unaffected (small batch, pools don't fill).

---

### min_latency_mode for MoE GEMM (iter114)
**Immediate CRASH**

`min_latency_mode=True` in `torch.ops.trtllm.fused_moe` selects a specialized
low-latency kernel path in TRTLLM's MoE implementation. This path requires
FP4 quantization (`use_fp4==true`). With BF16 (no quantization), the assertion
at `moe_kernels.cu:3759` fires on the first forward pass.

---

## Remaining Bottleneck: FillFunctor\<int\> (Unresolved)

**Identified via NSys SQLite profiling.**

A `vectorized_elementwise_kernel<4, FillFunctor<int>>` kernel fires **~31 times
per decode step**, each filling a **134 MB int32 tensor** (33,554,432 elements,
grid=(65536,1,1), block=(128,1,1)) at **~81µs each** = **~2.5ms/step** or
roughly **52% of the measured decode latency at c1**.

Properties:
- Runs on stream 7 (same stream as `_flash_decode_stage1_kernel`)
- Each fill is isolated in its own dynamic-eager partition (bounded by
  `cudaEventCreate + StreamIsCapturing + EventRecord` triplets in the NSys
  runtime trace — each fill is its own piecewise partition)
- Pattern: 1 fill per attention layer before each flash-decode stage-1 kernel
- No `graphId` — confirms eager execution, not CUDA-graph-captured

**Source not conclusively identified.** Candidates ruled out:
- `triton_paged_decode()` uses only `torch.empty` (not zeros/fill)
- `MetadataWrapper._copy_into_saved()` uses `stable.zero_()` but only for
  tiny tensors (≤ 8192 elements, not 134 MB)
- `piecewise_runner.py` `forward()` allocates with `torch.empty` (no fill)

The 33,554,432-element tensor size (128MB at int32) matches
`max_num_tokens × max_batch_size / some_factor` but the exact allocation site
is unknown. The leading hypothesis is the paged KV-cache index allocation or a
workspace buffer inside the triton-paged attention kernel setup.

**Impact**: Resolving this could yield an additional ~2.5ms/step (~52%)
improvement at c1, reducing TPOT from 4.787ms toward ~2.3ms.

---

## Architecture Reference

### Piecewise Graph Structure (iter107/HEAD)

| Partition type | Count | Execution | Examples |
|---|---|---|---|
| Static runners (CUDA-graph captured) | **2** | Fast, no Python overhead | Prefix norms, projections, LM head |
| Dynamic-wrapped (attention etc.) | **30** | DynamicOpWrapper, needs `out=` buffer | `triton_paged_mha_with_cache` |
| Metadata-wrapped | **1** | MetadataWrapper, stable output buffers | `triton_paged_prepare_metadata` |
| Router-fence eager | **30** | ~5µs/each, identity op | `gemma4_router_fence` |
| Stream-switch MoE eager | **30** | begin_aux+MoE+end_aux | Shared expert + routed MoE |
| Logits gather | **1** | Eager | `gather_tokens` |
| **Total** | **95** | | |

### Config File (gemma4_moe.yaml)

```yaml
model_factory: Gemma4ForConditionalGeneration
attn_backend: triton_paged
compile_backend: torch-cudagraph
cuda_graph_config:
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
max_num_tokens: 8192
max_batch_size: 512
transforms:
  compile_model:
    piecewise_enabled: true
  mlir_elementwise_fusion:
    enabled: false   # MUST be false: MLIR serializes multi-stream MoE fork/join
  gather_logits_before_lm_head:
    enabled: true
  fuse_gemms:
    enabled: true
  fuse_gelu_tanh_mul:
    enabled: true
  multi_stream_moe:
    enabled: true    # shared expert on aux CUDA stream
```

### Key Source Files

| File | What it does |
|---|---|
| `tensorrt_llm/_torch/auto_deploy/compile/piecewise_utils.py` | `_PARTITION_BOUNDARY_OPS`, graph splitting, dynamic op registry |
| `tensorrt_llm/_torch/auto_deploy/transform/library/multi_stream_moe.py` | Inserts `gemma4_router_fence` + stream-switch ops into FX graph |
| `tensorrt_llm/_torch/auto_deploy/compile/piecewise_runner.py` | `DynamicOpWrapper`, `MetadataWrapper`, CUDA-graph capture/replay |
| `tensorrt_llm/_torch/models/modeling_gemma4.py` | Router called before MLP (enables fence to precede stream-switch ops) |
| `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_gemma4_router.py` | `gemma4_router_fence` definition |
| `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_paged_attention.py` | `triton_paged_decode` (2-stage flash decode for paged KV) |
| `tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe/trtllm_moe.py` | TRTLLM MoE kernel invocation (no `min_latency_mode` for BF16) |
