---
name: ad-conf-generate
description: Automates finding the best AutoDeploy configuration for a target model by iterating between profiling and config generation. Records all trials throughout the process.
---

# AutoDeploy Configuration Generator

**Input:** HuggingFace model ID (or local path). **Output:** Optimized config YAML + performance report with all trial results.

Iterates: Profile → Analyze graph dumps → Generate candidate configs → Profile again → Verify winner.

---

## Phase 0 — Model Analysis & Setup

### Step 0.1 — Ask for model

Ask the user for the model name (HuggingFace ID or local directory path).

### Step 0.2 — GPU inventory

Run via gpu-shell:
```bash
nvidia-smi --query-gpu=index,memory.total,memory.used,name --format=csv,noheader
```
Record GPU type, count, and total VRAM.

### Step 0.3 — Analyze model

Read the model's `config.json` (from HF cache or local path). Extract:
- `model_type`, `architectures`
- Parameter count estimate (from `hidden_size`, `num_hidden_layers`, `intermediate_size`, `vocab_size`)
- Attention type: MHA (`num_key_value_heads == num_attention_heads`), GQA (`num_key_value_heads < num_attention_heads`), or MLA (if `q_lora_rank` or similar exists)
- MLP type: dense or MoE (check for `num_local_experts`, `num_experts`)
- Expert count (if MoE)
- `max_position_embeddings`

### Step 0.4 — Check existing configs

1. Read `examples/auto_deploy/model_registry/models.yaml` — look up the model, read any referenced `yaml_extra` files
2. Check `examples/auto_deploy/model_registry/configs/` for model-specific YAML
3. Check `examples/configs/curated/` for curated deployment configs

### Step 0.5 — Ask about baseline

Ask the user if they have an existing config they want to use as baseline. If yes, get the path.

### Step 0.6 — Create session directory

Create `$PWD/ad-conf-sessions/<MODEL_SHORT>_<YYYYMMDD_HHMMSS>/` with:
- `session_log.md` — running markdown log of all activity
- `session_state.yaml` — machine-readable state tracking
- `trials/` — per-trial subdirectories
- `graphs/` — AD_DUMP_GRAPHS_DIR output per trial

Initialize `session_state.yaml`:
```yaml
model: <HF_ID>
model_short: <MODEL_SHORT>
gpu_type: <GPU_NAME>
gpu_count: <N>
gpu_vram_gb: <VRAM>
iteration_count: 0
profiler_calls: 0
best_trial: null
perf_priority: null
profiling_method: null
tried_configs: []
```

Initialize `session_log.md`:
```markdown
# AD Config Generation Session: <MODEL_SHORT>

**Model:** <HF_ID>
**GPU:** <GPU_COUNT>x <GPU_NAME> (<VRAM> GB each)
**Started:** <YYYY-MM-DD HH:MM:SS>

---
```

---

## Phase 1 — Requirement Collection

Ask the user for:

1. **`max_seq_len`** — default from model's `max_position_embeddings` or 2048
2. **`max_batch_size`** — default 64
3. **Typical concurrency levels** — default `[1, 16, 64]`
4. **Performance priority** — one of: `OTPS` (output tokens per second), `req_per_sec`, `TTFT`, `TPOT`, `E2E_latency`
5. **Profiling method** — `trtllm-bench` or `bench-sweep`
   - If bench-sweep selected: ask whether nsys trace capture is desired (only available with bench-sweep)
6. **Profiling approach** — single method throughout, OR two-stage (trtllm-bench for fast screening, bench-sweep for final serving eval of winner)
7. If model estimated size > single GPU VRAM: ask about `skip_loading_weights: true` for fast iteration
8. If trtllm-bench: need dataset JSONL path, or offer to generate with `trtllm-bench --model <MODEL> prepare-dataset --stdout --num-requests 512 --output-sequence-length <OSL> --input-sequence-length <ISL> > <path>`. Then create per-concurrency datasets with right-sized request counts: `max(concurrency * 5, 100)` requests per concurrency level (e.g., c1→100, c4→100, c16→80). This avoids wasting time on low-concurrency runs where 512 requests at c1 takes ~68 min vs ~13 min with 100 requests.
9. If existing config found in Step 0.4 and user didn't provide one in Step 0.5: ask whether to use it as baseline

Record all answers in `session_state.yaml` and `session_log.md`.

---

## Step 2 — Config Generator (invoke `ad-conf-generator-agent`)

### First call (baseline establishment)

- If user has an existing config and opted to use as baseline → use it
- Otherwise → run with no extra config (AD defaults)
- Run `build_and_run_ad.py` via gpu-shell to record baseline generation output (needed for Step 4 verification):
  ```bash
  CUDA_VISIBLE_DEVICES=<GPUs> python examples/auto_deploy/build_and_run_ad.py \
    --model <MODEL> [--args.yaml-extra <baseline.yaml>] \
    2>&1 | tee <session_dir>/trials/trial_000_baseline/build_run.log
  ```
- Record baseline output in `trials/trial_000_baseline/`
- **Go to Step 3** (profile baseline)

### Subsequent calls (from Step 3)

Invoke the `ad-conf-generator-agent` subagent with:
- Session directory path
- Graph dump path (from latest profiler run)
- Profiler logs and trial records
- Model architecture info (from Phase 0)
- Performance priority
- List of tried configs (from `session_state.yaml`)
- nsys trace path (if available from bench-sweep)

The generator agent:
1. **Preprocesses**: Reads all `trials/*/trial_record.yaml`, checks profiler logs for actual transform application (`num_matches`, `Skipping`, `Applied` patterns)
2. **Launches 4 subagents in parallel**: fusion-analyst, sharding-analyst, ops-analyst, attn-analyst
3. **Combines results** and generates up to 3 candidate YAML files in `trials/candidates/`

**Wrap-up:**
- If candidates found → **go to Step 3** (profile candidates)
- If no candidates (all combos exhausted) → **go to Step 5** (report)

---

## Step 3 — Profiler (invoke `ad-conf-profiler-agent`)

**Memory check before profiling:** Before launching any profiling run, estimate per-instance GPU memory:
- Model weights (param_count × bytes_per_param) + KV cache + ~5-10 GB runtime overhead
- **Never run multiple benchmark instances simultaneously** unless you have verified each GPU has sufficient VRAM independently. OOM kills (signal 9) waste 20-30+ minutes per failed run.
- When in doubt, run benchmarks sequentially on a single GPU.

Invoke the `ad-conf-profiler-agent` subagent for each config to profile. Pass it:
- Model HF ID
- Config YAML path (or null for baseline)
- Profiling method (`trtllm-bench` or `bench-sweep`)
- Dataset path (if trtllm-bench)
- Concurrency levels
- Session directory path
- Trial name (e.g., `trial_001_fuse_rmsnorm`)
- AD_DUMP_GRAPHS_DIR path (e.g., `<session_dir>/graphs/<trial_name>/`)
- nsys flag (bench-sweep only)

The profiler agent:
1. Runs the profiling tool with `AD_DUMP_GRAPHS_DIR` set
2. Copies graph dumps to `session_dir/graphs/<trial_name>/`
3. Parses results: OTPS, req_per_sec, TTFT, TPOT, E2E latency, transform time
4. Writes `<trial_dir>/trial_record.yaml`
5. Updates `session_log.md` with trial summary

**Wrap-up logic:**
- **First time** (just profiled baseline) → **go to Step 2** (generate initial candidates based on graph dumps)
- **Not first time** → compare best new performance vs best so far:
  - **Improving** → **go to Step 4** (verify winner)
  - **Not improving** AND profiler called < 2 times → **go to Step 2** (try generating more candidates)
  - **Not improving** AND profiler called >= 2 times → **go to Step 5** (report)

Update `session_state.yaml` with `profiler_calls` count and `best_trial` after each profiler run.

---

## Step 4 — Verify (invoke `ad-conf-verify-agent`)

Invoke the `ad-conf-verify-agent` subagent with:
- Model HF ID
- Winner config YAML path
- Baseline build_and_run_ad.py output (from Step 2 first call)
- Session directory path

The verify agent:
1. Runs `build_and_run_ad.py --model <MODEL> --args.yaml-extra <winner.yaml>`
2. Compares generation output with baseline — checks for garbled, repetitive, or empty output
3. Reports PASS or FAIL

**Wrap-up:**
- **PASS** → copy winner config to `session_dir/winner_config.yaml`, **go to Step 5**
- **FAIL** → discard this config. If there are more candidates to try → **go to Step 2**. Otherwise → **go to Step 5** with best passing config.

---

## Step 5 — Final Report

Generate `session_dir/FINAL_REPORT.md` containing:

1. **Model overview** — name, architecture, param count, attention type, MLP type
2. **Hardware** — GPU type, count, VRAM
3. **Performance priority** optimized for
4. **All trials table**:
   | Trial | Config Summary | OTPS | TTFT (p50) | TPOT (p50) | req/s | Transform Time |
   |-------|---------------|------|------------|------------|-------|----------------|
   | baseline | AD defaults | ... | ... | ... | ... | ... |
   | trial_001 | fuse_rmsnorm | ... | ... | ... | ... | ... |
5. **Before/after comparison** — baseline vs winner with % delta for each metric
6. **Winner config YAML** — full content
7. **Reproduction commands** — exact commands to reproduce the winning run
8. **Session directory path** — for accessing full logs

Print the report to chat AND write to `session_dir/FINAL_REPORT.md`.

---

## Control Flow Summary

```
Phase 0 (model analysis) → Phase 1 (requirements) → Step 2 (baseline: build_and_run_ad.py + config)
    ↓
Step 3 (profile baseline)
    ↓
Step 2 (generate initial candidates)  ←──────────────────────────┐
    ↙           ↘                                                │
candidates    no candidates → Step 5 (report)                    │
    ↓                                                            │
Step 3 (profile candidates)                                      │
    ↓                                                            │
improving? ──yes──→ Step 4 (verify) → Step 5 (report)           │
    ↓ no                                                         │
profiler called < 2 times? ──yes──→ Step 2 (generate more) ─────┘
    ↓ no
Step 5 (report)
```

---

## Session Directory Structure

```
ad-conf-sessions/<model_short>_<timestamp>/
├── session_state.yaml          # Machine-readable state
├── session_log.md              # Human-readable running log
├── trials/
│   ├── trial_000_baseline/
│   │   ├── config.yaml         # Config used (or "none" marker)
│   │   ├── trial_record.yaml   # Structured metrics
│   │   ├── bench.log           # Full profiler stdout
│   │   ├── build_run.log       # build_and_run_ad.py output (baseline only)
│   │   └── results.json        # trtllm-bench JSON (if applicable)
│   ├── trial_001_<name>/
│   │   └── ...
│   └── candidates/
│       ├── candidate_001.yaml
│       ├── candidate_002.yaml
│       ├── candidate_003.yaml
│       └── manifest.yaml       # Candidate descriptions
├── graphs/
│   ├── baseline/               # AD_DUMP_GRAPHS_DIR from baseline
│   └── trial_001/              # AD_DUMP_GRAPHS_DIR from trial 001
├── verify/
│   └── verify.log
├── winner_config.yaml          # Final winning config
└── FINAL_REPORT.md             # Performance report
```

---

## Key Files Referenced

| File | Role |
|------|------|
| `tensorrt_llm/_torch/auto_deploy/config/default.yaml` | All 60+ transforms with defaults |
| `tensorrt_llm/_torch/auto_deploy/llm_args.py` | Config schema (LlmArgs fields) |
| `examples/auto_deploy/build_and_run_ad.py` | Entry point for verify agent |
| `tensorrt_llm/_torch/auto_deploy/transform/optimizer.py:78` | Transform time log line: `"Total time for all transforms: {total_time:.2f}s"` |
| `tensorrt_llm/_torch/auto_deploy/utils/graph_writer.py` | AD_DUMP_GRAPHS_DIR impl, file format: `{counter:03d}_{stage}_{transform_name}.txt` |
| `examples/auto_deploy/model_registry/models.yaml` | Model registry |
| `examples/auto_deploy/model_registry/configs/` | Existing model configs |
| `tensorrt_llm/commands/bench.py` | trtllm-bench entry point |
| `tensorrt_llm/bench/benchmark/throughput.py` | Throughput benchmark CLI options |
