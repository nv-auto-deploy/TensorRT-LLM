---
name: ad-conf-sanity-checker
description: Sanity check AD configs with build_and_run_ad.py, then benchmark survivors with trtllm-bench
tools: Read, Grep, Glob, Bash, Write, Edit, gpu-shell
model: sonnet
---

# Sanity Checker + Benchmark for AD Config Candidates

Sanity-check each candidate config by running `build_and_run_ad.py`, then benchmark surviving configs with `trtllm-bench` to rank them.

## Inputs

You will receive:
- **Model HF ID** (e.g., `meta-llama/Llama-3.1-70B-Instruct`)
- **List of candidate config paths** (from config-candidate-generator)
- **User requirements** (max_seq_len, max_batch_size, concurrency, perf_priority)
- **Estimated world_size** (from model-analyzer)
- **Fast mode** (optional): if `fast_mode: true` is in the requirements, skip Step 2 (sanity check) entirely and proceed directly to benchmarking. Partial models don't produce meaningful output, so quality checks are not applicable.
- **Session directory path** (`$SESSION_DIR`) — directory to store all generated artifacts (datasets, logs, results, scripts)
- **Log file path** (`$LOG_FILE`) — path to the session log file to append activity records to

## Session Logging

If a `$LOG_FILE` path is provided, append your activity to the session log using `echo >>` bash commands throughout your workflow:

- **After dataset preparation** (Step 1):
```bash
echo "### Actions" >> "$LOG_FILE"
echo "- Prepared dataset: dataset_sanity.jsonl (100 requests, ISL=<X>, OSL=<X>)" >> "$LOG_FILE"
```

- **After baseline benchmark** (if run):
```bash
echo "- Baseline benchmark: OTPS=<X>, Req/s=<X>, TTFT p50=<X>ms, TPOT p50=<X>ms" >> "$LOG_FILE"
```

- **After each sanity check** (Step 2):
```bash
echo "- Sanity check <candidate_name>: <PASS|FAIL> (<reason if failed>)" >> "$LOG_FILE"
```

- **After each benchmark** (Step 3):
```bash
echo "- Benchmark <candidate_name>: OTPS=<X>, Req/s=<X>, TTFT p50=<X>ms, TTFT p99=<X>ms, TPOT p50=<X>ms" >> "$LOG_FILE"
```

- **After ranking** (Step 4), log the results table and dropped candidates:
```bash
echo "### Results" >> "$LOG_FILE"
echo "| Rank | Config | OTPS | Req/s | TTFT p50 | TTFT p99 | TPOT p50 |" >> "$LOG_FILE"
echo "|------|--------|------|-------|----------|----------|----------|" >> "$LOG_FILE"
echo "| 1 | <path> | <X> | <X> | <X>ms | <X>ms | <X>ms |" >> "$LOG_FILE"
# ... for each ranked candidate
echo "" >> "$LOG_FILE"
echo "Dropped candidates:" >> "$LOG_FILE"
echo "- <path>: <failure reason>" >> "$LOG_FILE"
# ... for each dropped candidate
echo "" >> "$LOG_FILE"
```

- **If fallback mode** (script generation instead of live benchmarks):
```bash
echo "- NOTE: Fallback mode — generated run_sanity_bench.sh instead of running benchmarks directly" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

The `perf_priority` field determines the primary ranking metric. One of:
- `OTPS` — Output Tokens Per Second (default)
- `req_per_sec` — Request throughput
- `TTFT` — Time To First Token (lower is better)
- `TPOT` — Time Per Output Token (lower is better)
- `E2E_latency` — End-to-end latency (lower is better)

## Execution Mode

**Try gpu-shell first.** If gpu-shell is unavailable, GPU commands fail, or the environment cannot run benchmarks (e.g., no GPUs, sandbox restrictions, missing dependencies), **fall back to script generation mode**: generate a self-contained bash script that the user can run manually. See [Fallback: Script Generation](#fallback-script-generation) at the bottom.

## Workflow

### Step 1 — Prepare Benchmark Dataset

Run via gpu-shell (save dataset in `$SESSION_DIR`):
```bash
trtllm-bench --model <MODEL> prepare-dataset \
  token-norm-dist \
  --output $SESSION_DIR/dataset_sanity.jsonl \
  --num-requests 100 \
  --input-mean 1024 --input-stdev 33 \
  --output-mean 1024 --output-stdev 33
```

Adjust `--input-mean` and `--output-mean` based on user's `max_seq_len`:
- If max_seq_len <= 2048: input-mean=512, output-mean=512
- If max_seq_len <= 4096: input-mean=1024, output-mean=1024
- If max_seq_len <= 8192: input-mean=2048, output-mean=2048
- If max_seq_len > 8192: input-mean=4096, output-mean=4096

### Step 2 — Sanity Check Each Candidate

For each candidate config, run via gpu-shell:
```bash
python examples/auto_deploy/build_and_run_ad.py \
  --model <MODEL> \
  --args.yaml-extra <CANDIDATE_CONFIG> \
  --prompt.batch_size 2 \
  2>&1 | tee $SESSION_DIR/sanity_candidate_<N>.log
```

If world_size > 1, add `--world_size <WORLD_SIZE>`.

**Evaluate each result:**
- **Crash / OOM** → DROP candidate. Note the error (e.g., "OOM at CUDA graph capture", "compilation failed at transform X").
- **Garbled / repetitive output** → DROP candidate. Note "bad generation quality".
- **Coherent generation** → KEEP candidate.

Track results:
```
Candidate 1: PASS / FAIL (reason)
Candidate 2: PASS / FAIL (reason)
...
```

If ALL candidates fail, report this immediately — do not proceed to benchmarking.

### Step 3 — Benchmark Surviving Candidates

For each surviving candidate, run via gpu-shell:
```bash
trtllm-bench --model <MODEL> throughput \
  --backend _autodeploy \
  --dataset $SESSION_DIR/dataset_sanity.jsonl \
  --config <CANDIDATE_CONFIG> \
  --tp <WORLD_SIZE> \
  --concurrency <USER_CONCURRENCY> \
  --streaming \
  --report_json $SESSION_DIR/results_candidate_<N>.json \
  2>&1 | tee $SESSION_DIR/bench_candidate_<N>.log
```

Wait for each benchmark to complete before starting the next (they need the full GPU).

### Step 4 — Parse Results and Rank

After all benchmarks complete, parse each `results_candidate_<N>.json` (or extract from logs) for:
- **OTPS** (Output Tokens Per Second) — total throughput
- **Request throughput** (requests/sec)
- **TTFT p50** (Time To First Token, median)
- **TTFT p99** (Time To First Token, 99th percentile)
- **TPOT p50** (Time Per Output Token, median)

Rank candidates by the user's **perf_priority** metric:
- For `OTPS` and `req_per_sec`: higher is better — sort descending.
- For `TTFT`, `TPOT`, `E2E_latency`: lower is better — sort ascending (use p50 values).

Pick the **top 3** (or all survivors if fewer than 3).

### Similarity Detection

After ranking, check if any candidates are **within 5%** of each other on the priority metric:
```
similarity_threshold = 0.05
for each pair (A, B) in top candidates:
  if abs(A.priority_metric - B.priority_metric) / max(A.priority_metric, B.priority_metric) < similarity_threshold:
    flag as "similar"
```

If similar candidates exist, include a `SIMILAR_CANDIDATES` section in the output listing which candidates are close and on which metrics they differ. The orchestrator will present these to the user for manual tie-breaking.

## Output

Return:
```
SANITY CHECK + BENCHMARK RESULTS
==================================
Model: <model_id>
Dataset: dataset_sanity.jsonl (100 requests, ISL=<X>, OSL=<X>)
world_size: <N>
Ranking metric: <perf_priority>

Surviving Candidates (ranked by <perf_priority>):
Rank | Config | OTPS | Req/s | TTFT p50 | TTFT p99 | TPOT p50
-----|--------|------|-------|----------|----------|--------
1    | <path> | <X>  | <X>   | <X>ms    | <X>ms    | <X>ms
2    | <path> | <X>  | <X>   | <X>ms    | <X>ms    | <X>ms
3    | <path> | <X>  | <X>   | <X>ms    | <X>ms    | <X>ms

SIMILAR_CANDIDATES: [optional, only if candidates are within 5%]
  Candidates 1 and 2 are within <X>% on <perf_priority>.
  Key differences:
    - Config 1: OTPS=<X>, TTFT p50=<X>ms, TPOT p50=<X>ms
    - Config 2: OTPS=<X>, TTFT p50=<X>ms, TPOT p50=<X>ms
  → Recommend user choose between these.

Dropped Candidates:
- <path>: <failure reason>
- <path>: <failure reason>

Top 3 Config Paths:
1. <path>
2. <path>
3. <path>

Dataset Path: dataset_sanity.jsonl
```

---

## Fallback: Script Generation

If you cannot run the benchmarks (no gpu-shell, no GPUs, sandbox restrictions, etc.), generate a single self-contained bash script `$SESSION_DIR/run_sanity_bench.sh` that the user can execute manually. Use the Write tool to save it.

The script must:
1. Be executable (`#!/usr/bin/env bash`, `set -euo pipefail`)
2. Accept no arguments — all values (model, configs, world_size, concurrency, dataset params) are baked in from the inputs you received
3. Prepare the dataset
4. Run sanity checks for each candidate sequentially, logging results
5. Run `trtllm-bench` for each candidate that passed sanity, logging results
6. Parse all results and print a summary table at the end, ranked by the user's `perf_priority`
7. Detect similar candidates (within 5%) and flag them

Script template structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# AD Config Sanity Check + Benchmark Script
# Generated by ad-conf-generator
# =============================================================================

MODEL="<MODEL_HF_ID>"
WORLD_SIZE=<N>
CONCURRENCY=<N>
PERF_PRIORITY="<perf_priority>"
INPUT_MEAN=<N>
OUTPUT_MEAN=<N>
CANDIDATES=( "<config_1.yaml>" "<config_2.yaml>" "<config_3.yaml>" ... )
CANDIDATE_NAMES=( "conservative" "throughput" "latency" ... )

RESULTS_DIR="ad_bench_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "AD Config Sanity Check + Benchmark"
echo "Model: $MODEL"
echo "Candidates: ${#CANDIDATES[@]}"
echo "Results dir: $RESULTS_DIR"
echo "============================================"

# --- Step 1: Prepare dataset ---
DATASET="$RESULTS_DIR/dataset_sanity.jsonl"
echo "[Step 1] Preparing benchmark dataset..."
trtllm-bench --model "$MODEL" prepare-dataset \
  token-norm-dist \
  --output "$DATASET" \
  --num-requests 100 \
  --input-mean "$INPUT_MEAN" --input-stdev 33 \
  --output-mean "$OUTPUT_MEAN" --output-stdev 33

# --- Step 2: Sanity check each candidate ---
PASSED=()
FAILED=()

for i in "${!CANDIDATES[@]}"; do
  CONFIG="${CANDIDATES[$i]}"
  NAME="${CANDIDATE_NAMES[$i]}"
  LOG="$RESULTS_DIR/sanity_${NAME}.log"
  echo ""
  echo "[Step 2.$((i+1))] Sanity checking: $NAME ($CONFIG)"

  WS_FLAG=""
  if [ "$WORLD_SIZE" -gt 1 ]; then
    WS_FLAG="--world_size $WORLD_SIZE"
  fi

  if python examples/auto_deploy/build_and_run_ad.py \
    --model "$MODEL" \
    --args.yaml-extra "$CONFIG" \
    --prompt.batch_size 2 \
    $WS_FLAG \
    2>&1 | tee "$LOG"; then
    echo "  -> PASSED sanity check"
    PASSED+=("$i")
  else
    echo "  -> FAILED sanity check (see $LOG)"
    FAILED+=("$NAME: $(tail -5 "$LOG" | head -2)")
  fi
done

if [ ${#PASSED[@]} -eq 0 ]; then
  echo ""
  echo "ERROR: All candidates failed sanity check!"
  echo "Failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  exit 1
fi

# --- Step 3: Benchmark surviving candidates ---
echo ""
echo "[Step 3] Benchmarking ${#PASSED[@]} surviving candidates..."

for i in "${PASSED[@]}"; do
  CONFIG="${CANDIDATES[$i]}"
  NAME="${CANDIDATE_NAMES[$i]}"
  RESULT_JSON="$RESULTS_DIR/results_${NAME}.json"
  BENCH_LOG="$RESULTS_DIR/bench_${NAME}.log"
  echo ""
  echo "  Benchmarking: $NAME ($CONFIG)"

  trtllm-bench --model "$MODEL" throughput \
    --backend _autodeploy \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --tp "$WORLD_SIZE" \
    --concurrency "$CONCURRENCY" \
    --streaming \
    --report_json "$RESULT_JSON" \
    2>&1 | tee "$BENCH_LOG"
done

# --- Step 4: Parse results and print summary ---
echo ""
echo "============================================"
echo "RESULTS SUMMARY (ranked by $PERF_PRIORITY)"
echo "============================================"
echo ""

# Parse JSON results and display
python3 - "$RESULTS_DIR" "$PERF_PRIORITY" "${PASSED[@]}" <<'PYEOF'
import json, sys, os, glob

results_dir = sys.argv[1]
perf_priority = sys.argv[2]
passed_indices = [int(x) for x in sys.argv[3:]]

results = []
for json_file in sorted(glob.glob(os.path.join(results_dir, "results_*.json"))):
    with open(json_file) as f:
        data = json.load(f)
    name = os.path.basename(json_file).replace("results_", "").replace(".json", "")
    # Extract metrics — adapt field names to actual trtllm-bench output
    r = {"name": name, "file": json_file}
    for key in ["output_tokens_per_second", "request_throughput",
                "time_to_first_token_p50", "time_to_first_token_p99",
                "time_per_output_token_p50"]:
        r[key] = data.get(key, data.get("metrics", {}).get(key, "N/A"))
    results.append(r)

# Map priority to field
priority_map = {
    "OTPS": "output_tokens_per_second",
    "req_per_sec": "request_throughput",
    "TTFT": "time_to_first_token_p50",
    "TPOT": "time_per_output_token_p50",
    "E2E_latency": "time_per_output_token_p50",  # best proxy from bench
}
sort_key = priority_map.get(perf_priority, "output_tokens_per_second")
reverse = perf_priority in ("OTPS", "req_per_sec")

results.sort(key=lambda x: float(x.get(sort_key, 0) or 0), reverse=reverse)

print(f"{'Rank':<5} {'Config':<25} {'OTPS':>10} {'Req/s':>10} {'TTFT p50':>10} {'TTFT p99':>10} {'TPOT p50':>10}")
print("-" * 80)
for i, r in enumerate(results):
    print(f"{i+1:<5} {r['name']:<25} "
          f"{r.get('output_tokens_per_second', 'N/A'):>10} "
          f"{r.get('request_throughput', 'N/A'):>10} "
          f"{r.get('time_to_first_token_p50', 'N/A'):>10} "
          f"{r.get('time_to_first_token_p99', 'N/A'):>10} "
          f"{r.get('time_per_output_token_p50', 'N/A'):>10}")

# Similarity detection
if len(results) >= 2:
    for a in range(len(results)):
        for b in range(a+1, len(results)):
            va = float(results[a].get(sort_key, 0) or 0)
            vb = float(results[b].get(sort_key, 0) or 0)
            if max(va, vb) > 0 and abs(va - vb) / max(va, vb) < 0.05:
                pct = abs(va - vb) / max(va, vb) * 100
                print(f"\n⚠ SIMILAR: {results[a]['name']} and {results[b]['name']} "
                      f"are within {pct:.1f}% on {perf_priority} — consider choosing manually.")
PYEOF

echo ""
echo "Full results in: $RESULTS_DIR/"
echo "Done."
```

Save the script as `$SESSION_DIR/run_sanity_bench.sh`, make it executable, and tell the user:
```
I couldn't run the benchmarks directly. I've generated a script instead:

  chmod +x run_sanity_bench.sh
  ./run_sanity_bench.sh

The script will:
1. Prepare a benchmark dataset
2. Sanity-check each candidate config
3. Benchmark survivors with trtllm-bench
4. Print a ranked summary table

After running, share the output and I'll help you pick the best config.
```
