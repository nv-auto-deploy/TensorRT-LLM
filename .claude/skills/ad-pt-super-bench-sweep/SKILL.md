---
name: ad-pt-super-bench-sweep
description: >
  Run a head-to-head serving-throughput sweep (trtllm-serve + aiperf) for Nemotron
  Super V3 with MTP + attention-DP, comparing the AutoDeploy backend (optionally with
  vs without the ADP request balancer) against the PyTorch backend, over a variable-ISL
  dataset (SPEED-Bench) across concurrency 1..256, then plot the OTPS-vs-per-user-TPS
  pareto. Use when the user wants to benchmark SuperV3 AD vs PT, measure the attention-DP
  MoE all-to-all / balancer effect, or produce a serving pareto. Triggers on: "super
  benchmark sweep", "AD vs PT pareto", "run the super sweep", "balancer benchmark",
  "speed-bench serving sweep", "combine the PT line".
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# SuperV3-MTP AutoDeploy-vs-PyTorch serving benchmark sweep

Drive `trtllm-serve` + `aiperf` to compare backends on Nemotron Super V3 (MTP +
attention-DP) under realistic variable-ISL serving load, and produce a throughput pareto.

**Read `references/gotchas.md` first.** Most failures here are environment issues
(launch/parallelism, a broken `nvidia-cutlass-dsl`, lease changes) — not the benchmark.

## Input

- **TensorRT-LLM source dir** (required) — the repo whose code is under test (AD changes
  live here). The `trtllm-serve`/`aiperf` binaries are resolved from `PATH`.
- **Model dir** (required) — HF checkpoint (e.g. NVFP4 SuperV3). Also used as the aiperf
  `--tokenizer`.
- **Dataset** (required) — SPEED-Bench `single_turn` JSONL (variable ISL). Synthetic
  fixed ISL hides the attention-DP effect (see gotchas).
- **Backends/configs to compare** (default: AD+balancer, AD−balancer, PT) — each is a
  `(label, backend, config-yaml)`. Starter configs in `references/configs/`.
- **Output dir** (required) — for per-config summary JSONs, aiperf artifacts, and the plot.
- **Concurrencies** (optional, default `1 2 4 8 16 32 64 128 256`).

## Output

- Per-config `OUT/<label>_serve.json` — list of `{concurrency, otps, itl_ms, ttft_ms,
  user_tps, req_per_s}`.
- `OUT/pareto.png` — OTPS vs per-user-TPS, one line per config, points annotated by
  concurrency.
- A comparison table (printed) of OTPS per concurrency with % deltas vs the first
  (baseline) series, plus TTFT.

## Workflow

1. **Pre-flight (per distinct config).** Catch env breakage before a multi-hour sweep:
   ```
   scripts/validate_build.sh _autodeploy <ad_yaml> 31180 <model> <out>
   scripts/validate_build.sh pytorch     <pt_yaml> 31180 <model> <out>
   ```
   Must print `VALIDATE: READY`. If it dies, the script prints the key errors — cross-ref
   `references/gotchas.md` (CuTe/cutlass-dsl → ensure `FLASHINFER_USE_CUDA_NORM=1`;
   "Rank 0 to 0" → `world_size`/`tensor_parallel_size` in the yaml; "No such file" →
   stale binary path).

2. **Run each config sweep** (each launches a fresh server, sweeps c, kills it):
   ```
   scripts/serve_sweep.sh ad_balancer   _autodeploy <ad_yaml> 31180 <model> <dataset> <out>
   # AD without balancer: toggle the source flag, run, restore:
   sed -i 's/enable_balance=True/enable_balance=False/' \
     tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
   scripts/serve_sweep.sh ad_nobalancer _autodeploy <ad_yaml> 31190 <model> <dataset> <out>
   git checkout -- tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py
   # PyTorch backend:
   scripts/serve_sweep.sh pt            pytorch     <pt_yaml> 31200 <model> <dataset> <out>
   ```
   Run them sequentially (one server at a time on the GPUs). Each is ~20-30 min after a
   ~10 min build. Launch as background tasks and watch the first c=1 point land to confirm
   health before walking away.

   ### Running the PyTorch backend (PT BE) specifically

   PT is launched by the *same* `serve_sweep.sh` (same aiperf SPEED-Bench sweep, same
   metrics) — only the backend flag and config differ:
   ```
   scripts/serve_sweep.sh pt pytorch references/configs/pt_super_v3_mtp_serve.yaml \
       31200 <model> <dataset> <out>
   ```
   Under the hood this runs (no `mpirun`, no `--tp_size`):
   ```
   FLASHINFER_USE_CUDA_NORM=1 trtllm-serve <model> --backend pytorch \
       --extra_llm_api_options pt_super_v3_mtp_serve.yaml --trust_remote_code \
       --host 0.0.0.0 --port 31200
   ```
   Key PT-vs-AD differences:
   - **Parallelism comes from the config**, not the launcher: PT reads
     `tensor_parallel_size: 4` + `moe_expert_parallel_size: 4` (AD instead reads
     `world_size: 4`). Neither backend uses `mpirun`.
   - **No AutoDeploy `transforms:` block** — PT uses its own MTP/attention-DP
     implementation. The PT config is plain `TorchLlmArgs` (`enable_attention_dp: true`,
     `speculative_config: {decoding_type: MTP, num_nextn_predict_layers: 6,
     mtp_eagle_one_model: true}`, `cuda_graph_config`, `kv_cache_config`).
   - **No balancer toggle** — `enable_balance` is AutoDeploy-only; PT is a single config.
   - Batching is matched to the AD config (`max_batch_size: 256`, `max_num_tokens: 4096`)
     so the AD-vs-PT comparison is apples-to-apples.
   - Validate first: `scripts/validate_build.sh pytorch <pt_yaml> 31200 <model> <out>`
     → expect `VALIDATE: READY`.

3. **Plot + compare.** Baseline first:
   ```
   scripts/plot_pareto.py <out>/pareto.png \
     pt:<out>/pt_serve.json \
     AD-nobal:<out>/ad_nobalancer_serve.json \
     AD+bal:<out>/ad_balancer_serve.json
   ```

4. **Report** the pareto + table. State the balancer Δ (AD+bal vs AD−bal) and the AD-vs-PT
   gap per concurrency. Be honest about noise (single runs; repeat c=256 if a delta is
   marginal) and about any non-comparable conditions.

5. **Restore state**: `git checkout` the balancer flag, confirm `enable_balance=True`,
   and note that `FLASHINFER_USE_CUDA_NORM` / any `cute.experimental` patch are
   environment-only (not committed).

## Notes

- Metrics: **OTPS** = `output_token_throughput` (system tok/s); **per-user TPS** =
  `1000 / inter_token_latency_ms`; **TTFT** = `time_to_first_token`. All from aiperf's
  `profile_export_aiperf.json`.
- The balancer is a source flag (`enable_balance` in `ad_executor.py`), not a config knob.
- Output length is model-determined (`--use-server-token-count`) for the variable-load
  comparison; do not force `--osl` unless you specifically want fixed-length output.
- Absolute throughput is not comparable across different OSL regimes (variable SPEED-Bench
  vs fixed `--osl`); only compare runs that used the same recipe.
