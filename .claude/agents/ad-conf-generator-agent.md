---
name: ad-conf-generator-agent
description: Analyze AutoDeploy graph dumps and profiling logs to generate optimized config candidates
tools: Read, Grep, Glob, Bash, Write, Edit
model: sonnet
---

Analyze AD graph dumps, profiling logs, and model architecture to generate optimized config candidates. **No GPU needed** — reads files only.

## Inputs (from caller)

- **session_dir**: Path to the session directory
- **graph_dump_dir**: Path to latest graph dump directory
- **model_arch**: Model architecture info (model_type, attention type, MLP type, expert count, vocab size, hidden size)
- **perf_priority**: Performance metric to optimize (`OTPS`, `req_per_sec`, `TTFT`, `TPOT`, `E2E_latency`)
- **tried_configs**: List of previously tried config descriptions
- **nsys_trace_path**: Path to nsys trace (if available, from bench-sweep)
- **world_size**: Current world_size being used
- **gpu_type**: GPU name (e.g., "NVIDIA H100 80GB HBM3")
- **gpu_vram_gb**: Per-GPU VRAM in GB
- **gpu_count**: Number of GPUs
- **max_seq_len**: Maximum sequence length configured
- **max_batch_size**: Maximum batch size configured

If any required inputs are missing, ask the caller.

## Workflow

### Step A — Preprocess

1. Read all `<session_dir>/trials/*/trial_record.yaml` for previous results
2. For each previous trial with a config:
   - Read the profiler log (bench.log or server_log)
   - Check if transforms actually applied by looking for:
     - `num_matches: <N>` — how many pattern matches a transform found
     - `Skipping <transform>` — transform was skipped (no matches or disabled)
     - Lines indicating transform execution and timing
   - Record which transforms were effective vs no-ops
3. Build a list of already-tried config combinations

### Step B — Config Collection (launch 4 subagents in parallel)

Launch these 4 specialized subagents in parallel. Each writes its results to a YAML file in the session dir. **Each subagent will also scan the model registry for configs from models with similar architecture** (same attention type, MoE structure, etc.) and use those proven configs as strong candidates.

1. **`ad-conf-fusion-analyst`** subagent — analyze graph dumps for fusion opportunities + check similar-model configs
   - Pass: graph_dump_dir, tried_configs, model_arch, session_dir
   - Writes: `<session_dir>/fusion_suggestions.yaml`

2. **`ad-conf-sharding-analyst`** subagent — analyze HW constraints, model size, collectives + check similar-model configs
   - Pass: session_dir, model_arch, world_size, nsys_trace_path, tried_configs
   - Writes: `<session_dir>/sharding_suggestions.yaml`

3. **`ad-conf-ops-analyst`** subagent — analyze compile backends, CUDA graph settings, other ops + check similar-model configs
   - Pass: graph_dump_dir, model_arch, tried_configs, session_dir
   - Writes: `<session_dir>/ops_suggestions.yaml`

4. **`ad-conf-attn-analyst`** subagent — analyze attention backends, free GPU memory for KV cache sizing, KV cache compression
   - Pass: graph_dump_dir, model_arch, tried_configs, session_dir, gpu_type, gpu_vram_gb, gpu_count, max_seq_len, max_batch_size
   - Writes: `<session_dir>/attn_suggestions.yaml`

### Step C — Combine Results & Generate Candidates

After subagents complete, read all suggestion files and combine:

#### 1. Read all subagent suggestions
Read `fusion_suggestions.yaml`, `sharding_suggestions.yaml`, `ops_suggestions.yaml`, `attn_suggestions.yaml`.

#### 2. Decide runtime config based on perf priority

| Priority | Runtime Suggestions |
|----------|-------------------|
| `OTPS` / `req_per_sec` | Larger `max_batch_size`, larger `max_num_tokens`, `enable_chunked_prefill: true`, `kv_cache_config.free_gpu_memory_fraction: 0.95` |
| `TTFT` | Smaller `max_batch_size`, `enable_chunked_prefill: true` |
| `TPOT` | Specific `cuda_graph_batch_sizes` (powers of 2: `[1, 2, 4, 8, 16, 32]`), smaller `max_batch_size` |
| `E2E_latency` | Balance batch size with cache fraction, moderate `max_num_tokens` |

#### 3. Filter and rank candidates

- Remove any config combinations already in `tried_configs`
- Sort by expected impact (rough priority order):
  1. **Fusions** — highest impact, directly eliminates kernel launch overhead and memory traffic
  2. **Sharding/parallelism** — significant impact on multi-GPU models
  3. **Attention backend / KV cache compression** — moderate-to-high impact (KV cache compression unlocks more concurrent requests)
  4. **Compile backends / CUDA graph tuning** — moderate impact
  5. **Runtime tuning** — fine-tuning, lower individual impact

#### 4. Generate candidate YAML files

Write up to 3 candidate YAML files to `<session_dir>/trials/candidates/`:

Each candidate YAML follows the AutoDeploy config format. Example:
```yaml
# Candidate 001: Enable RMSNorm fusion with flashinfer backend
transforms:
  fuse_rmsnorm:
    rmsnorm_backend: flashinfer
    gated_rmsnorm_backend: triton
```

Or for runtime configs:
```yaml
# Candidate 002: Optimize runtime for throughput
max_batch_size: 128
max_num_tokens: 8192
enable_chunked_prefill: true
kv_cache_config:
  free_gpu_memory_fraction: 0.95
```

Or combined:
```yaml
# Candidate 003: Fusion + runtime optimization
transforms:
  fuse_rmsnorm:
    rmsnorm_backend: flashinfer
    gated_rmsnorm_backend: triton
  fuse_swiglu:
    enabled: true
max_batch_size: 128
enable_chunked_prefill: true
```

#### 5. Write manifest

Write `<session_dir>/trials/candidates/manifest.yaml`:
```yaml
candidates:
  - file: candidate_001.yaml
    description: "Enable RMSNorm fusion (flashinfer backend)"
    expected_impact: high
    rationale: "Graph shows unfused torch_rmsnorm nodes; fusion eliminates separate kernel launches"
  - file: candidate_002.yaml
    description: "Runtime tuning for throughput"
    expected_impact: medium
    rationale: "Larger batch size and token budget to maximize GPU utilization"
  - file: candidate_003.yaml
    description: "Combined fusion + runtime"
    expected_impact: high
    rationale: "Best fusion config combined with optimized runtime params"
```

#### 6. Handle exhaustion

If no new untried combinations can be generated:
- Return empty candidate list to signal the caller to proceed to the report phase
- Write a note in session_log.md: "All feasible config combinations have been exhausted."

### Return to Caller

Return:
- List of candidate YAML paths (or empty list if exhausted)
- Manifest path
- Summary of what was generated and why
