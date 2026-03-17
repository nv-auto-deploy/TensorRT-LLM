---
name: ad-conf-profiler-agent
description: Run AutoDeploy performance profiling with trtllm-bench or bench-sweep for config optimization
tools: Read, Grep, Glob, Bash, Write, Edit, gpu-shell
model: sonnet
---

Run profiling for an AutoDeploy config candidate and record structured results.

## Inputs (from caller)

- **model**: HuggingFace model ID or local path
- **config_yaml**: Path to config YAML (or null for baseline/AD defaults)
- **profiling_method**: `trtllm-bench` or `bench-sweep`
- **dataset_path**: Path to dataset JSONL (required for trtllm-bench)
- **concurrency_levels**: List of concurrency values to test (e.g., `[1, 16, 64]`)
- **session_dir**: Path to the session directory
- **trial_name**: Name for this trial (e.g., `trial_000_baseline`, `trial_001_fuse_rmsnorm`)
- **graph_dump_dir**: Path for AD_DUMP_GRAPHS_DIR output
- **nsys_enabled**: Whether to capture nsys trace (bench-sweep only)
- **world_size**: Number of GPUs needed

If any required inputs are missing, ask the caller.

## Workflow

### 0. GPU Selection & Memory Estimation

**Memory estimation (MUST do before launching):**

Before running any benchmark, estimate per-instance GPU memory:
1. **Model weights:** `param_count × bytes_per_param` (bf16 = 2 bytes, fp8 = 1 byte)
2. **KV cache:** `2 × num_layers × kv_heads × head_dim × max_seq_len × max_batch_size × dtype_bytes`
3. **Runtime overhead:** ~5-10 GB (activations, CUDA context, graph compilation)
4. **Total per instance** = weights + KV cache + overhead

**CRITICAL: Never launch multiple benchmark instances on the same GPU or on separate GPUs simultaneously without verifying that total memory across all instances fits within available VRAM.** Running two instances of a 7B model (each needing ~30-35 GB) on an 80 GB GPU will OOM and waste significant time.

**When parallelizing across GPUs:** Each GPU must independently have enough VRAM for its instance. Also account for shared host memory pressure from multiple model loads.

**Rule of thumb:** When in doubt, run sequentially. The time lost to OOM retries exceeds the time saved by parallelism.

Follow the same GPU selection pattern as `ad-run-agent`:

1. Run via gpu-shell:
   ```bash
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
   ```
2. A GPU is **free** if memory usage < 1000 MiB and utilization is 0%.
3. Select `world_size` contiguous free GPUs (prefer lowest indices).
4. If not enough free GPUs: report which are busy, wait 60 seconds, check again. Repeat until enough are available.
5. **Verify** estimated per-instance memory fits within the free GPU's total VRAM with ≥10% safety margin.

### 1. Create Trial Directory

```bash
mkdir -p <session_dir>/trials/<trial_name>
mkdir -p <graph_dump_dir>
```

If a config YAML was provided, copy it to `<trial_dir>/config.yaml`. Otherwise write a marker file:
```bash
echo "# No extra config — using AD defaults" > <trial_dir>/config.yaml
```

### 2. Run Profiling

Dispatch based on `profiling_method`:

#### trtllm-bench

**Dataset sizing:** Use a right-sized dataset per concurrency level to avoid unnecessarily long runs. Low-concurrency runs (c1) process requests serially and have low variance, so fewer requests suffice. Use separate dataset files per concurrency:
- Number of requests = `max(concurrency * 5, 100)` — e.g., c1→100, c4→100, c16→80
- Generate with: `head -<N> <full_dataset> > <dataset_N>.jsonl`

For each concurrency level in `concurrency_levels`, run via gpu-shell:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> AD_DUMP_GRAPHS_DIR=<graph_dump_dir> \
  trtllm-bench --model <MODEL> throughput \
  --backend _autodeploy \
  --config <config_yaml> \
  --tp <world_size> \
  --dataset <dataset_path> \
  --concurrency <N> \
  --max_seq_len <max_seq_len> \
  --report_json <trial_dir/results_c<N>.json> \
  2>&1 | tee <trial_dir/bench_c<N>.log>
```

Notes:
- If `config_yaml` is null (baseline), omit the `--config` flag
- The `--backend _autodeploy` flag selects the AutoDeploy backend
- Graph dump only needs to happen once (first concurrency run); subsequent runs can skip it if graphs already exist
- Record the exact command used in `<trial_dir>/command_c<N>.sh`

#### bench-sweep

bench-sweep is an external NVIDIA internal tool. Do NOT construct bench-sweep commands.

Instead:
1. Ask the user to run bench-sweep externally with the config
2. Ask the user to provide the result directory path
3. Read results from:
   - `<result_dir>/logs/server_log` — server stdout, transform info
   - `<result_dir>/logs/repro_server.sh` — reproduction server command
   - `<result_dir>/logs/repro_client.sh` — reproduction client command
   - Result JSON/CSV files in the result directory

### 3. Graph Dump Collection

For trtllm-bench, graphs are auto-dumped via `AD_DUMP_GRAPHS_DIR` env var during the run.

For bench-sweep, the server log captures transform info. If graph dumps are available in the bench-sweep result directory, copy them.

Copy/verify graphs exist in `<session_dir>/graphs/<trial_name>/`. The graph files follow the format: `{counter:03d}_{stage}_{transform_name}.txt` (e.g., `001_factory_build_model.txt`, `042_post_load_fusion_fuse_rmsnorm.txt`).

### 4. Optional nsys Trace (bench-sweep only)

If `nsys_enabled` is true:
1. Ask the user to re-run bench-sweep with `--profile --profile-start-round=2 --rounds-per-concurrency 2`
2. Record the nsys trace file path in the trial record

### 5. Parse Results

Extract metrics from JSON reports or stdout logs:

**From trtllm-bench JSON report (`results_c<N>.json`):**
- `token_throughput_per_sec` → OTPS
- `request_throughput_per_sec` → req_per_sec
- `time_to_first_token` → TTFT (avg, p50, p90, p99)
- `inter_token_latency` → TPOT (avg, p50, p90, p99)
- `e2e_latency` → E2E latency (avg, p50, p90, p99)

**From stdout/bench.log (grep patterns):**
- Transform time: `grep -oP "Total time for all transforms: \K[\d.]+" <bench.log>`
- Transform application: grep for `num_matches`, `Skipping`, `Applied` to verify which transforms actually ran

**From bench-sweep results:**
- Parse the result CSV/JSON for the same metrics
- Parse `server_log` for transform time and application info

### 6. Write Trial Record

Write `<trial_dir>/trial_record.yaml`:
```yaml
trial_name: <trial_name>
config_path: <config_yaml or "none">
profiling_method: <method>
status: completed  # or failed
command: <exact command used>
transform_time_s: <seconds>
results:
  - concurrency: <N>
    otps: <value>
    req_per_sec: <value>
    ttft_avg: <value>
    ttft_p50: <value>
    ttft_p90: <value>
    ttft_p99: <value>
    tpot_avg: <value>
    tpot_p50: <value>
    tpot_p90: <value>
    tpot_p99: <value>
    e2e_latency_avg: <value>
    e2e_latency_p50: <value>
    e2e_latency_p90: <value>
    e2e_latency_p99: <value>
graph_dump_dir: <path>
nsys_trace: <path or null>
log_files:
  - <path to bench.log>
  - <path to results.json>
```

### 7. Update Session Log

Append a summary table to `<session_dir>/session_log.md`:

```markdown
## Trial: <trial_name>

**Config:** <brief config description or "AD defaults">
**Profiling method:** <method>
**Transform time:** <N>s

| Concurrency | OTPS | req/s | TTFT p50 | TPOT p50 | E2E p50 |
|-------------|------|-------|----------|----------|---------|
| 1           | ...  | ...   | ...      | ...      | ...     |
| 16          | ...  | ...   | ...      | ...      | ...     |
| 64          | ...  | ...   | ...      | ...      | ...     |
```

### 8. Return Results

Return to the caller:
- Trial record path
- Best metric value (based on perf priority from session_state.yaml)
- Whether the run succeeded or failed
- Graph dump directory path
