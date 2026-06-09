# MLA Sharding Investigation — TPLA upper-bound benchmark

> **Status: investigation scratch — NOT FOR MERGE.**
> Branch: `gk/mla-sharding-investigation` (off `upstream/main`).
> Tracking issue: [NVIDIA/TensorRT-LLM#15162](https://github.com/NVIDIA/TensorRT-LLM/issues/15162).
> This branch carries a deliberately **numerically-garbage** perf hack. Do not
> open a PR from it as-is.

## TL;DR for the next agent

We want to know **how much decode throughput/latency we could gain if MLA stopped
replicating its latent KV cache across TP ranks.** This branch adds a one-file,
env-gated hack that shrinks the per-rank cached/read MLA latent by a factor `G`
(`TPLA_LATENT_DIV=G`). Outputs are garbage, but per-rank KV **bandwidth and
footprint** match Tensor-Parallel Latent Attention (TPLA) with `G` latent groups.

**Your job:** run the `bench-sweep` arms in [§3](#3-the-experiment) on the
8×B200 node, collect decode inter-token latency (ITL) + output token throughput
vs `G` across context lengths, and report whether the ceiling justifies a real
TPLA implementation. **First, read [§4](#4-critical-caveat-flash-mla-is-specialized-to-head_dim576) — the default MLA decode kernel may reject the
reduced latent; there is a documented fallback.**

---

## 1. The problem (why this is worth investigating)

MLA caches a single compressed latent (`kv_lora_rank + qk_rope_head_dim`, e.g.
`512 + 64 = 576`/token for DeepSeek-V3). That latent is **shared across all query
heads** (one KV "head"). Under tensor parallelism we shard the *query heads*, but
every rank still needs the **full** latent to reconstruct its heads' K/V — so the
latent KV cache is **replicated on every TP rank**, and the per-rank latent
**read** does not shrink with TP size.

Per-rank decode read volume:

```
V(P) ~= [ q/kv_b/o weights + MoE weights + activations ] / P   +   KV(1)
                         ^ scales with TP size P                  ^ CONSTANT
```

Every term scales with `P` except the latent KV read `KV(1)`. That single
non-scaling term is an **Amdahl floor** on MLA decode and is the bandwidth wall
for long-context / high-concurrency serving.

### The axis law (what is even possible)

Decode output for head `h` is
`o_h = softmax_s( q̃_h · c_s / √d ) · (c·W_UV_h)`, where `q̃_h = W_UKᵀ q_nope_h`
spans the *full* latent and `c_s` is the shared latent per token. Which index you
cut decides feasibility:

- **Latent feature axis (TPLA):** partial scores must be summed *before* softmax
  ⇒ exact version needs an all-reduce of the full `[batch, heads, kv_len]` score
  matrix per step (prohibitive). Independent per-shard softmax is **approximate**
  (needs Hadamard/PCA reparam to recover accuracy). ← what this hack emulates.
- **Sequence / key-set axis (CP):** softmax over disjoint key sets composes
  **exactly** via the flash-attention log-sum-exp merge; per-rank read scales
  `1/P`; merge comm is independent of sequence length. Exact, no retrain.
- **Head axis:** one KV head ⇒ nothing to split ⇒ forces replication.
- **Request / batch axis (DP):** trivially exact; replicate weights, each rank
  owns disjoint requests + its own cache. (This is what `deepseek_v2_ep.yaml`
  already does via `enable_attention_dp: true`.)

So the exact, no-retrain levers are **CP** and **DP**; TPLA is the **approximate**
lever that keeps the head-TP topology. This benchmark measures the TPLA ceiling.

---

## 2. What this branch contains (the hack)

**One file changed:** `tensorrt_llm/_torch/auto_deploy/custom_ops/mla/trtllm_mla.py`.

A helper `_tpla_latent_div()` reads `TPLA_LATENT_DIV` (default `1` = no-op) and is
applied at the three places that all key off tensor shapes:

1. `get_cache_initializers` → paged cache allocated for `kv_lora_rank//G + qk_rope`
   instead of the full latent (smaller per-rank cache).
2. `get_constants` → passes the reduced `kv_lora_rank` to the op.
3. `_mla_with_cache_impl` → truncates the (full, replicated) `compressed_kv` and
   `kv_b_proj_weight` to the per-rank latent slice so cache write + decode read +
   absorbed BMMs all operate on `kv_lora_rank//G`.

**Why garbage numerics are fine for this measurement:** the aiperf client runs
with `ignore_eos: true`, so the number of decode steps is fixed regardless of the
(wrong) tokens produced — the **timing** is representative. We are measuring the
memory-traffic ceiling, not correctness. (`G=1` is bit-identical to today.)

**What is intentionally NOT modeled** (so this is an *upper bound*, slightly
optimistic on compute): no Hadamard/PCA reparam, no cross-group reduce; and heads
stay sharded across all `world_size` ranks (real TPLA shards heads across `k/G`,
giving `G×` more heads/rank — extra compute that is hidden in memory-bound decode).

Enable it:

```bash
export TPLA_LATENT_DIV=2   # or 4, 8 ; unset / 1 = stock MLA
```

---

## 3. The experiment

Tooling: the internal **`bench-sweep`** driver (launches
`trtllm-serve --backend _autodeploy --extra_llm_api_options <cfg>`, sweeps
concurrency with aiperf, writes `profile_export_aiperf.csv` per arm). It accepts
`--env-vars`, which is how we inject `TPLA_LATENT_DIV`.

### Model + config

- **Model:** an MLA model. DeepSeek-V2-Lite is ideal (small, fits 8×B200). Numerics
  don't matter, so `skip_loading_weights: true` works with just a HF `config.json`
  (no checkpoint download needed). DeepSeek-V3 / Kimi-K2 also fine if available.
- **Configs:** base on the registry MLA configs in
  `examples/auto_deploy/model_registry/configs/` (`deepseek_v2_ep.yaml`,
  `deepseek-r1.yaml`, `kimi_k2.yaml`).
  - **`dp_ref` arm** = `deepseek_v2_ep.yaml` as-is (`enable_attention_dp: true`)
    — production reference (no latent duplication; the alternative we want to beat).
  - **`tp_*` arms** = a TP variant with **`enable_attention_dp: false`** so MLA is
    head-TP sharded and the latent cache *is* replicated (the regime TPLA improves).
    `attn_backend: trtllm` (→ `trtllm_mla`); `compile_backend: torch-cudagraph`.
  - **VERIFY** in the server log that the `tp_*` arms actually take the head-TP MLA
    path (replicated latent cache), not attention-DP. If AutoDeploy's defaults route
    DeepSeek to attention-DP regardless, force the TP path before trusting `tp_g1`.

### Arms

| tag | `enable_attention_dp` | `TPLA_LATENT_DIV` | meaning |
|---|---|---|---|
| `dp_ref` | true  | 1 | production reference (DP attention + EP) |
| `tp_g1`  | false | 1 | TPLA baseline: full replicated latent (head-TP MLA) |
| `tp_g2`  | false | 2 | latent read shrunk ×2 |
| `tp_g4`  | false | 4 | latent read shrunk ×4 |
| `tp_g8`  | false | 8 | latent read shrunk ×8 |

### Driver

```bash
source ~/utils/github/set_github_token.sh   # if needed
WT=/path/to/this/worktree
RES=$WT/_bench/results/mla_tpla
MODEL=$LLM_MODELS_ROOT/DeepSeek-V2-Lite       # or a fake DeepSeek dir

run_arm () {  # $1=tag  $2=config  $3=G
  bench-sweep \
    --model "$MODEL" \
    --config-path "$2" \
    --server-type trtllm-autodeploy \
    --world-size 8 \
    --concurrencies "1 4 16 64 256" \
    --isl "$ISL" --osl 1024 --min-requests 64 --rounds-per-concurrency 1 \
    --tag "${1}_isl${ISL}" --result-base-dir "$RES" \
    --env-vars "TPLA_LATENT_DIV=$3 PYTHONPATH=$WT"
}

# Sweep context length to find where the KV read overtakes the weight read.
for ISL in 1024 4096 8192 16384; do
  run_arm dp_ref  configs/deepseek_dp.yaml  1
  run_arm tp_g1   configs/deepseek_tp.yaml  1
  run_arm tp_g2   configs/deepseek_tp.yaml  2
  run_arm tp_g4   configs/deepseek_tp.yaml  4
  run_arm tp_g8   configs/deepseek_tp.yaml  8
done
```

### Metrics & expected result

From each `profile_export_aiperf.csv`: **Inter-Token Latency (ms)** = decode-step
time (the clean bandwidth signal) and **Output Token Throughput (tokens/sec)**.

Expected: `tp_g1` worst (latent replicated ×8); `tp_g2/4/8` improve **as ISL
grows** (KV read becomes the dominant decode term) and saturate. Strategic readout:
**does `tp_g8` reach or beat `dp_ref`?** If yes, TPLA could make head-TP competitive
with attention-DP while avoiding DP's load imbalance + all-to-all.

---

## 4. CRITICAL caveat: flash-MLA is specialized to `head_dim=576`

The `trtllm_mla` **decode** path uses flash-MLA (SM90) / trtllm-gen MLA, which is
hard-specialized to the DeepSeek MLA shape (`kv_lora_rank=512`, `head_dim=576`).
Shrinking the latent (`G>1`) very likely makes that kernel **reject the shape** —
on B200 (SM100) too, since the specialized kernel is shape-, not arch-, gated.

If the `tp_g2/4/8` arms crash in the decode kernel:

1. **Fallback to a flexible MLA backend** for the `tp_*` arms: set
   `attn_backend` to `triton` (→ `triton_mla`) or `torch` (→ `torch_backend_mla`),
   which accept arbitrary `kv_lora_rank`. Absolute numbers are lower than flash-MLA,
   but the **relative** `G`-scaling on one backend is still a valid bandwidth-ceiling
   signal — run `tp_g1..g8` all on the same flexible backend.
2. The flexible backends need the **same 3-spot pattern** as in `trtllm_mla.py`
   (`get_cache_initializers` head_dim, `get_constants` kv_lora_rank,
   `_mla_with_cache_impl`/equivalent slice). Port `_tpla_latent_div()` there
   (`tensorrt_llm/_torch/auto_deploy/custom_ops/mla/triton_mla.py` /
   `torch_backend_mla.py`) if they don't already route through the shared helper.

Check which kernel ran in the server log under `$RES/*<tag>*/logs/server_log*`.

---

## 5. Next steps (if the ceiling is worth it)

- **Real TPLA (approximate, drop-in):** 2D `(head_tp × latent_group)` sharding in
  `_process_mla_sharding` (`transform/library/sharding.py`) + the cache slice +
  a **PCA/Hadamard load-time reparam** of `kv_b_proj`/`kv_a_layernorm` (a load hook
  next to `_kv_b_proj_dequant_load_hook` in the custom DeepSeek modeling) to recover
  accuracy. Validate `G=1` bit-exact vs head-TP MLA, then accuracy after reparam.
- **Exact alternatives (no approx, no retrain):** sequence/context parallelism
  (LSE-merge; shards the latent read `1/P` exactly) and attention-DP (already in
  `deepseek_v2_ep.yaml`). Compare against the TPLA ceiling — if CP gets most of the
  win exactly, it may be the better target.

---

## 6. References

- TPLA: *Tensor-Parallel Latent Attention*, arXiv:2508.15881 — per-shard local
  attention + output all-reduce; Hadamard/PCA reparam; ~1.79× (DeepSeek-V3) /
  1.93× (Kimi-K2) at 32K.
- GLA: *Hardware-Efficient Attention for Fast Decoding* (Zadouri, Strauss, Dao),
  arXiv:2505.21487 — the trained latent-head-sharded architecture TPLA approximates.
- Hack lives in `tensorrt_llm/_torch/auto_deploy/custom_ops/mla/trtllm_mla.py`
  (`_tpla_latent_div`, env `TPLA_LATENT_DIV`).
