---
name: ad-nano-perf-bench
description: >
  Run AutoDeploy (and PyTorch-backend) performance sweeps for the Nemotron-3 Nano-30B-A3B
  model with the `sweep` CLI, extract per-concurrency throughput/latency metrics, capture
  nsys traces, and produce throughput-vs-latency plots. Use when the user wants to benchmark
  Nano perf, run a concurrency sweep, compare AD vs PT backend, measure user-TPS / ITL,
  profile the decode loop, or quantify the perf impact of a branch. Triggers on: "nano perf",
  "nano sweep", "run a sweep", "bench nano", "AD vs PT throughput", "user tps", "ITL sweep",
  "trace nano", "TP2/TP4 nano", "perf impact of <branch> on nano".
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# AutoDeploy Nano Perf Benchmark

Run, parse, trace, and plot Nemotron-3 Nano-30B-A3B performance sweeps. This skill encodes
the working recipe (model paths, `sweep` flags, env setup, nsys workflow, metric extraction,
plotting) plus the failure modes that waste the most time.

## Input

- **Backend** (required) — `trtllm-autodeploy` (AD) or `trtllm-pytorch` (PT baseline).
- **Parallelism** — TP world size (2 or 4 typical). For AD use `--world-size N`. For PT the
  `sweep` CLI rejects `--world-size`; put `tensor_parallel_size`/`moe_expert_parallel_size`
  in the YAML instead (see Gotcha #5).
- **Concurrencies** — e.g. `"1 2 4 8 16 32 64 128 256"`. Single point: `"1"`.
- **ISL/OSL** — input/output seq len, default `1000/1000`.
- **Config YAML** — the AD/PT extra-llm-api-options file (see Configs).
- **Optional**: nsys trace (yes/no), branch to evaluate, output result dir.

## Output

- Per-concurrency table: `conc, per-user TPS, ITL (ms), aggregate out-TPS, TTFT (ms), req latency (ms)`.
- A `*_summary.csv` and a two-panel PNG (throughput Pareto + per-user-TPS/ITL vs concurrency).
- Optional `.nsys-rep` traces and a kernel/NVTX breakdown for gap analysis.

## Known-good facts (B200 cluster, NVFP4, ISL/OSL 1000/1000)

These are sanity anchors — if a fresh run is wildly off these, suspect the environment, not the model.

| setup | per-user TPS @ c=1 | ITL @ c=1 |
|---|---|---|
| AD `nano_v3.yaml`, TP2 | ~402 | ~2.49 ms |
| AD `nano_v3.yaml`, TP4 | ~405 | ~2.47 ms |
| PT backend, TP2, minimal yaml | ~426 | ~2.35 ms |
| AD **minimal** yaml (no transforms) | ~266 | ~3.76 ms |

Key insight: AD's perf comes almost entirely from the `nano_v3.yaml` transforms
(piecewise compile, mlir fusion, multi-stream MoE, trtllm_gen MoE, SYMM_MEM sharding,
flashinfer_ssm, fuse_mamba_a_log, gather_logits). Stripping them drops AD to ~266 tps/u,
*below* PT. TP4 aggregate throughput saturates ~13.5–14k tok/s by c=64 (the curve knee).

## Model paths (IMPORTANT)

Use the **CI mirror** — its `tokenizer.json` is a real file:

```
/scratch/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
```

Do **not** use `.../coreai_comparch_autodeploy/llm-models/...` — that mirror's `tokenizer.json`
is an unresolved git-LFS pointer (133 bytes) and the server dies with
`Exception: expected value at line 1 column 1`. FP8 variant exists at the same CI path with
`-FP8` suffix. Set `HF_HUB_OFFLINE=1` to avoid HF 401s on the `nvidia/...` repo id.

## Configs

- **Registry config** (full AD perf): `examples/auto_deploy/model_registry/configs/nano_v3.yaml`.
  Sets `max_batch_size: 384`, `max_seq_len: 65536`, `attn_backend: trtllm`, a
  `cuda_graph_config.batch_sizes` list, `SYMM_MEM` sharding with a manual `tp_plan`
  (mamba `in_proj`, attn q/k/v/o, MoE up/down, MoLE fc1/fc2_latent), and transforms
  `multi_stream_moe`, `gather_logits_before_lm_head`, `fuse_mamba_a_log`,
  `insert_cached_ssm_attention: flashinfer_ssm`, `compile_model.piecewise_enabled`,
  `fuse_nvfp4_moe: trtllm_gen`, `mlir_elementwise_fusion`.
- **msl=2020 variant** (matches nightly dashboard): copy `nano_v3.yaml`, set `max_seq_len: 2020`.
  Verified perf-neutral vs 65536 at c=1 (the dashboard's lower number is *not* from msl).
- **PT-backend minimal yaml** (what the nightly runs):
  ```yaml
  tensor_parallel_size: 2
  moe_expert_parallel_size: 2
  max_seq_len: 2020
  max_batch_size: 1
  cuda_graph_config: {enable_padding: true, max_batch_size: 1}
  kv_cache_config: {dtype: auto, enable_block_reuse: false, free_gpu_memory_fraction: 0.9}
  ```
- The nightly dashboard runs `--backend pytorch`, **not** AutoDeploy. A bare minimal yaml on
  AD strips all perf transforms (~266 tps/u) — don't confuse the two.

## Environment setup

If `python3 -c "import tensorrt_llm; from tensorrt_llm._torch.auto_deploy import LLM"` already
works, skip this. After a container/lease reset it usually won't — see References for the full
recovery sequence. The minimal env exports that matter for CUDA-13 builds:

```bash
export CUDA_HOME=/home/egeva/.local/lib/python3.12/site-packages/nvidia/cu13   # if pip-CUDA
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1
```

**Verify the active package first** — the editable install can point at a *different* checkout:
```bash
pip show tensorrt_llm | grep -iE "Editable|Location"   # "Editable project location" = the live tree
```
If it points somewhere other than the checkout you think you're testing, your sweep runs the
wrong code. See Gotcha #7.

## Workflow

1. **[Verify env]** Confirm `import tensorrt_llm` + AD LLM import works and `which sweep` resolves.
   Confirm the editable-install location is the checkout you intend to benchmark (Gotcha #7).
   Check GPUs are free: `nvidia-smi --query-gpu=index,memory.used --format=csv`. Kill orphans
   (`pkill -9 -f trtllm-serve; pkill -9 -f nsys`) and confirm memory drops to ~0.

2. **[Pick config]** Use `nano_v3.yaml` for AD perf numbers, or build a variant
   (msl=2020 / PT-minimal). Confirm the chosen yaml exists.

3. **[Run sweep]** Launch **detached** with `setsid nohup` so it survives session exits
   (a plain background task dies on `/exit`). Use `scripts/run_sweep.sh`:
   ```bash
   scripts/run_sweep.sh <backend> <world_size> "<concurrencies>" <config_yaml> <result_dir> <port>
   ```
   It writes `<result_dir>/sweep.log`. For one server across all concurrencies (default),
   each point gets `concurrency * rounds` prompts. Monitor the log for
   `Benchmark completed for concurrency=N` and `All benchmarks completed successfully!`.
   Transient `Failed to parse JSON string` lines from aiperf are benign (it retries) — only
   act if a concurrency produces no CSV.

4. **[Extract metrics]** Once all points land:
   ```bash
   python3 scripts/extract_metrics.py <result_dir> "<concurrencies>" > <result_dir>/summary.csv
   ```
   Sanity-check monotonicity: per-user TPS should fall and ITL rise with concurrency. A point
   that breaks the trend (e.g. aggregate out-TPS far below neighbors, TTFT spike into seconds)
   is usually an aiperf/scheduling hiccup — **re-run that single concurrency** before plotting.

5. **[Plot]**
   ```bash
   python3 scripts/plot_sweep.py <result_dir>/summary.csv <result_dir>/plot.png
   ```
   Two panels: throughput Pareto (per-user TPS vs aggregate out-TPS, annotated with c=) and
   per-user-TPS + ITL vs concurrency.

6. **[Optional — nsys trace]** For gap/decode analysis, see "nsys tracing" below.

7. **[Report]** Present the table + plot, name the curve knee, flag any re-run points, and state
   the result dir.

## nsys tracing (the recipe that works)

The `sweep --profile` path and `--capture-range=cudaProfilerApi` both proved flaky
(empty traces / hangs / CUPTI conflicts). The reliable method is **manual `nsys launch` +
`nsys start`/`nsys stop`**:

```bash
# 1. launch server under nsys (profiling OFF until 'nsys start')
nsys launch -t cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
  --session-new=ad_session -e TLLM_LLMAPI_ENABLE_NVTX=1 \
  trtllm-serve <MODEL> --host 0.0.0.0 --port <P> --trust_remote_code \
  --backend _autodeploy --extra_llm_api_options <yaml> > server.log 2>&1 &
# 2. wait for "Application startup complete" in server.log
# 3. warm up with one short aiperf request (no profiling)
# 4. start capture, drive steady-state traffic, stop:
nsys start --session=ad_session -o <out>            # note: -o lives on `start`, not `launch`
aiperf profile --model <MODEL> --url http://0.0.0.0:<P> --endpoint-type chat --ui-type None \
  --streaming --concurrency 1 --request-count 3 --warmup-request-count 1 \
  --isl 1000 --osl 1000 --no-server-metrics --use-server-token-count \
  --extra-inputs ignore_eos:true --artifact-dir <dir>/aiperf
nsys stop --session=ad_session                       # writes <out>.nsys-rep, then exits
pkill -9 -f trtllm-serve; pkill -9 -f "nsys --start-agent"   # clean up
```

For the PT backend use `--backend pytorch` and a PT yaml; same `nsys launch`/`start`/`stop`.

Extract a kernel + NVTX breakdown (re-export with `--force-export=true` if a stale `.sqlite` exists):
```bash
nsys stats --report cuda_gpu_kern_sum   --format csv <trace>.nsys-rep | grep -E "^(Time|[0-9])" > kern.csv
nsys stats --report nvtx_pushpop_sum    --format csv <trace>.nsys-rep | grep -E "^(Time|[0-9])"
```
AD decode-loop NVTX worth knowing: `ad_prepare_inputs`, `ad_nest_sequences` (15× `ad_stage_*`
host-staging calls + `ad_rescatter_input_ids_` + `ad_host_prepare_for_attention_forward`),
`ad_run_forward`. At c=1 the AD-vs-PT ITL gap is host-overhead in these ranges, **not** GPU
kernel time (PT actually runs more kernel work per token). The overlap scheduler hides most of
`ad_rescatter_input_ids_`, so disabling it makes things *worse* (266 tps/u) — don't.

## Gotchas (ranked by time wasted)

1. **Editable install points at the wrong checkout.** `import tensorrt_llm` follows the
   `__editable__*.pth` finder, which may map to a sibling dir (e.g. `TensorRT-LLM1`), not the
   tree you `git checkout`ed. Always confirm `pip show tensorrt_llm` "Editable project location"
   before trusting a result. Python/C++ mismatch shows as
   `operator trtllm::inplace_slice_copy does not exist` (the live tree's Python expects a C++ op
   the installed libs don't export → that tree needs its own build/whl).
2. **Wrong model mirror** → LFS-pointer tokenizer → `expected value at line 1 column 1`. Use the
   CI mirror.
3. **Sweep dies on session `/exit`** if started as a plain background task. Use `setsid nohup`.
   Recover orphaned zombie servers by checking `nvidia-smi` (workers gone = 0 MiB) and re-running.
4. **CUDA toolkit mismatch** on pip-CUDA installs: `nvcc` major.minor must equal `CUDART_VERSION`
   or flashinfer/FMHA JIT fails (`#error CUDA compiler and CUDA toolkit headers are incompatible`,
   `Unsupported .version 9.3`, or NVRTC `could not open source file "cuda/std/type_traits"`).
   Fixes: align all `nvidia-cuda-{nvcc,crt,nvrtc,runtime}` + `nvidia-nvvm` to one version;
   `FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`; repoint the build's
   `cpp/build/.../fmha/cuda` symlink to a real CUDA include tree that contains both `cuda.h` and
   `cccl/cuda/std/`. (Full sequence in References.)
5. **PT backend + `--world-size`** → `Error: --world-size is only supported for trtllm-autodeploy`.
   Put `tensor_parallel_size` / `moe_expert_parallel_size` in the PT yaml instead.
6. **Anomalous sweep point** (out-TPS far below neighbors, multi-second TTFT) from short
   measurement windows or aiperf JSON hiccups → re-run that single concurrency. Per-user TPS and
   ITL are the robust metrics; aggregate out-TPS at small N is noisy.
7. **Branch perf comparisons**: identify the merge-base
   (`git merge-base HEAD origin/main`) and whether the branch's commits are pure-Python. If the
   only diff is Python (e.g. `ad_executor.py`), you can swap that one file between arms with **no
   rebuild** (editable install uses the working tree). If the branch carries C++ deltas, it needs
   its own build/whl. A pure-Python config-plumbing change is a **no-op unless the yaml exercises
   the plumbed fields** (or `ad_config` defaults differ from the old hardcoded values) — confirm
   the eval config actually sets them before sweeping.

## References

- [references/environment_recovery.md](references/environment_recovery.md) — full container/lease
  reset recovery: pip install order, CUDA-13 lib symlinks, FMHA NVRTC include fix, version pins.
