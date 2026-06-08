# Gotchas — SuperV3-MTP AD-vs-PT serving sweep

Hard-won setup issues. Most are *environment* problems that masquerade as code/benchmark
failures. Check these before concluding anything about the benchmark itself.

## Launch / parallelism

- **AutoDeploy: NO `mpirun`, set `world_size` in the yaml.** Launch plain
  `trtllm-serve <model> --backend _autodeploy --extra_llm_api_options <yaml>` (no
  `mpirun -n N`, no `--gpus_per_node`). AD spawns its own `world_size` workers from the
  yaml. AD **rejects `--tp_size`** ("AutoDeploy only supports parallelization via the
  `world_size` argument").
  - Under `mpirun -n 4`, ranks 1-3 come up as MPI workers that never receive `world_size`
    → `ValueError: Rank should be an integer between 0 and 0, but got 2` in
    `base_worker.to_mapping()`.
  - If `world_size` is missing from the yaml, `moe_ep_size` collapses to 1 →
    `DistConfig` validation fails (`moe grid (1) != tp_size (4)`).
  - Success signature in the server log: `dist_config=world_size=4 ... moe_ep_size=4
    enable_attention_dp=True` on all ranks.
- **PyTorch: set `tensor_parallel_size` (+ `moe_expert_parallel_size`) in the yaml**, also
  no `mpirun`.
- **Resolve `trtllm-serve`/`aiperf`/`nvidia-smi` via `command -v`, never hardcode
  `/usr/local/bin/...`.** Paths move across containers/leases; a stale absolute path fails
  with `No such file or directory` and a confusing "server died".

## nvidia-cutlass-dsl / CuTe (build-time crashes)

Some nodes ship a broken `nvidia-cutlass-dsl` (seen with 4.5.0). Two distinct failures,
both during model build (kvcache-init warmup forward, via flashinfer `rmsnorm_cute`):
1. `NotImplementedError: CuTe Experimental module is only supported on Cuda toolkit 13.1
   and above!` — `cute/experimental/__init__.py` is an unconditional-`raise` stub (the
   <13.1 variant) imported incidentally by `cutlass_dsl.get_version()`'s
   `pkgutil.walk_packages` (version hashing), even on CUDA 13.1+ nodes.
2. After patching that: `TypeError: mlir_global_dtors() got an unexpected keyword argument
   'data'` → `DSLRuntimeError: 🧊 ICE 🧊` — a Python↔MLIR binding mismatch in the package.

**Fix (robust): set `FLASHINFER_USE_CUDA_NORM=1`** (the scripts do this). It switches
flashinfer's rmsnorm to the CUDA-JIT implementation, avoiding `nvidia-cutlass-dsl`
entirely (rmsnorm is the only DSL kernel in this model path). It's a documented flashinfer
fallback and identical for both backends, so it does not confound AD-vs-PT.
**Clean fix:** reinstall a working `nvidia-cutlass-dsl` (re-installing the tensorrt_llm
wheel does NOT touch this separate package).

## Workload / aiperf

- **Use a variable-ISL dataset** (SPEED-Bench: `speed_bench_1k.jsonl`, single `text`
  field, ISL ~11-998 words). Synthetic fixed `--isl/--osl` gives every DP rank identical
  prefill load → the ADP balancer has no straggler to fix → its effect is muted to noise.
  Even so (see below) the balancer benefit has been hard to reproduce.
- **aiperf recipe** (what the script uses): `--endpoint-type chat --streaming
  --input-file <dataset> --custom-dataset-type single_turn --tokenizer <model>
  --tokenizer-trust-remote-code --use-server-token-count --no-server-metrics
  --num-warmup-requests 1 --request-count $((5*c))`. Output length is model-determined
  (don't force `--osl` for the variable-load comparison).
- **request-count = 5×c** for steady state. Smaller counts (e.g. 3×c) are noisy; single
  runs still have run-to-run variance — repeat key points (esp. c=256) before trusting a
  few-percent delta.

## ADP balancer toggle (AD only)

The balancer is a *source* flag, not a config knob:
`tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py` →
`AttentionDpConfig(enable_balance=True, batching_wait_iters=10, timeout_iters=16)`.
To benchmark with/without: `sed -i 's/enable_balance=True/enable_balance=False/'` the file,
run, then `git checkout -- <file>` to restore. The server reads the source at launch.
Empirically (SPEED-Bench serving, this PR): the balancer is **neutral-to-negative on
throughput** (best ~+2% @ c=4; −5 to −7% @ c=8/16/256) and only helps TTFT at high c.
It did **not** reproduce the historical "+39% @ c=128" (likely measured vs the older
`.item()` budget baseline).

## Orchestration hygiene

- **Self-`pkill` bug**: `pkill -f "trtllm-serve"` in a command whose own text contains
  "trtllm-serve" matches and kills the launching shell. Put pattern-kills in a separate
  step, or kill by PID/`fuser -k <port>/tcp`.
- **Background launches**: run the sweep scripts as files (`bash serve_sweep.sh ...`),
  ideally via the harness background, not inline `setsid` with heredocs (flaky here).
- **Lease changes mid-run** kill the server (GPUs go to 0) and may leave
  `ad_executor.py` at `enable_balance=False` (the driver flips it for the no-balancer
  run and restores at the end). Always restore with `git checkout` and re-validate.
- **Restore state when done**: `git checkout -- ad_executor.py`,
  confirm `enable_balance=True`, and remember the `FLASHINFER_USE_CUDA_NORM` env and any
  `cute.experimental` patch are environment-only (not in git).

## Model / dataset paths used (this cluster)

- Model (NVFP4 SuperV3): `/lustre/share/coreai_dlalgo_ci/artifacts/model/nvidia_nvidia-nemotron-3-super-120b-a12b-nvfp4/hf/hf-4f0cf9d_orig`
- SPEED-Bench: `/lustre/fsw/coreai_comparch_autodeploy/egeva/datasets/SPEED-Bench/speed_bench_1k.jsonl`
- 120B TP4 build takes ~10 min per server; a full c=1..256 sweep ~20-30 min per config.
