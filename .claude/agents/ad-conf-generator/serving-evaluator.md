---
name: ad-conf-serving-evaluator
description: Run trtllm-serve with top config candidates and pick the final winner using serving benchmarks
tools: Read, Grep, Glob, Bash, Write, Edit, gpu-shell
model: sonnet
---

# Serving Evaluator for AD Config Candidates

Run `trtllm-serve` with each of the top config candidates and benchmark with the serving benchmark script to pick the final winner.

## Inputs

You will receive:
- **Model HF ID** (e.g., `meta-llama/Llama-3.1-70B-Instruct`)
- **Top 3 config paths** (from sanity-checker, ranked by user's priority metric)
- **User concurrency target** (e.g., 32)
- **Dataset path** (e.g., `dataset_sanity.jsonl`)
- **perf_priority** — the user's chosen ranking metric: `OTPS`, `req_per_sec`, `TTFT`, `TPOT`, or `E2E_latency`
- **Session directory path** (`$SESSION_DIR`) — directory to store all generated artifacts (serve logs, benchmark logs, scripts)
- **Log file path** (`$LOG_FILE`) — path to the session log file to append activity records to

## Session Logging

If a `$LOG_FILE` path is provided, append your activity to the session log using `echo >>` bash commands throughout your workflow:

- **Per candidate** (Steps 1-3), log server start, wait time, and benchmark results:
```bash
echo "### Actions" >> "$LOG_FILE"
echo "- Candidate <N> (<config_name>):" >> "$LOG_FILE"
echo "  - Server started on port <PORT>" >> "$LOG_FILE"
echo "  - Server ready after <X>s" >> "$LOG_FILE"
echo "  - Benchmark results: OTPS=<X>, Req/s=<X>, TTFT p50=<X>ms, TTFT p99=<X>ms, TPOT p50=<X>ms, E2E p50=<X>ms, E2E p99=<X>ms" >> "$LOG_FILE"
echo "  - Server killed" >> "$LOG_FILE"
```

- **For failed candidates**:
```bash
echo "  - FAILED: <reason (e.g., server timeout, benchmark crash)>" >> "$LOG_FILE"
```

- **After comparison** (Step 4), log the comparison table:
```bash
echo "### Results" >> "$LOG_FILE"
echo "| Rank | Config | OTPS | Req/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 | E2E p99 |" >> "$LOG_FILE"
echo "|------|--------|------|-------|----------|----------|----------|---------|---------|" >> "$LOG_FILE"
echo "| 1 | <path> | <X> | <X> | <X>ms | <X>ms | <X>ms | <X>ms | <X>ms |" >> "$LOG_FILE"
# ... for each candidate
echo "" >> "$LOG_FILE"
```

- **If fallback mode** (script generation instead of live benchmarks):
```bash
echo "- NOTE: Fallback mode — generated run_serving_eval.sh instead of running serving benchmarks directly" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

## Execution Mode

**Try gpu-shell first.** If gpu-shell is unavailable, GPU commands fail, or the environment cannot run serving benchmarks, **fall back to script generation mode**: generate a self-contained bash script that the user can run manually. See [Fallback: Script Generation](#fallback-script-generation) at the bottom.

## Workflow

For each of the top 3 candidates, perform Steps 1-3 sequentially. Use a different port for each to avoid conflicts (8001, 8002, 8003).

### Step 1 — Start the Server

Run via gpu-shell (background):
```bash
trtllm-serve <MODEL> --backend _autodeploy \
  --extra_llm_api_options <CANDIDATE_CONFIG> \
  --port <PORT> \
  2>&1 | tee $SESSION_DIR/serve_candidate_<N>.log &
```

Wait for server readiness by polling:
```bash
# Poll until server is ready (max 10 minutes)
for i in $(seq 1 120); do
  if curl -s http://localhost:<PORT>/health | grep -q "ok\|healthy\|200"; then
    echo "Server ready"
    break
  fi
  sleep 5
done
```

If the server fails to start within 10 minutes, kill it and skip this candidate.

### Step 2 — Run Serving Benchmark

Run via gpu-shell:
```bash
python tensorrt_llm/serve/scripts/benchmark_serving.py \
  --backend openai \
  --model <MODEL> \
  --tokenizer <MODEL> \
  --dataset-name trtllm_custom \
  --dataset-path <DATASET_PATH> \
  --num-prompts 100 \
  --max-concurrency <USER_CONCURRENCY> \
  --port <PORT> \
  2>&1 | tee $SESSION_DIR/serving_bench_candidate_<N>.log
```

### Step 3 — Kill Server and Collect Results

Kill the server process:
```bash
# Find and kill the trtllm-serve process on this port
kill $(lsof -ti:<PORT>) 2>/dev/null || pkill -f "trtllm-serve.*--port <PORT>"
sleep 5
```

Parse the benchmark output from `$SESSION_DIR/serving_bench_candidate_<N>.log` for:
- **OTPS** (Output Tokens Per Second)
- **Request throughput** (requests/sec)
- **TTFT p50 / p99** (Time To First Token)
- **E2E latency p50 / p99** (End-to-End latency)
- **TPOT p50** (Time Per Output Token)

### Step 4 — Compare and Pick Winner

After all 3 candidates are evaluated, rank by the user's **perf_priority** metric:
- For `OTPS` and `req_per_sec`: higher is better — sort descending.
- For `TTFT`: sort by TTFT p50 ascending (lower is better).
- For `TPOT`: sort by TPOT p50 ascending (lower is better).
- For `E2E_latency`: sort by E2E p50 ascending (lower is better).

**Similarity detection:** Check if the top candidates are **within 5%** of each other on the priority metric:
```
similarity_threshold = 0.05
for each pair (A, B) in candidates:
  if abs(A.priority_metric - B.priority_metric) / max(A.priority_metric, B.priority_metric) < similarity_threshold:
    flag as "similar"
```

If similar candidates exist, do NOT auto-pick a winner. Instead, include a `SIMILAR_CANDIDATES` section in the output so the orchestrator can present them to the user for manual tie-breaking. List the tradeoffs clearly (e.g., "Config A has 2% higher OTPS but Config B has 15% lower TTFT p99").

If there is a clear winner (>5% gap on the priority metric), pick it automatically.

### Step 5 — Save Final Config

Copy the winning config to the session directory:
```
$SESSION_DIR/{MODEL_SHORT_NAME}_final_config.yaml
```

## Output

Return:
```
SERVING EVALUATION RESULTS
============================
Model: <model_id>
Concurrency: <user_concurrency>
Dataset: <dataset_path> (100 requests)
Ranking metric: <perf_priority>

Comparison Table (ranked by <perf_priority>):
Rank | Config | OTPS | Req/s | TTFT p50 | TTFT p99 | TPOT p50 | E2E p50 | E2E p99
-----|--------|------|-------|----------|----------|----------|---------|--------
1    | <path> | <X>  | <X>   | <X>ms    | <X>ms    | <X>ms    | <X>ms   | <X>ms
2    | <path> | <X>  | <X>   | <X>ms    | <X>ms    | <X>ms    | <X>ms   | <X>ms
3    | <path> | <X>  | <X>   | <X>ms    | <X>ms    | <X>ms    | <X>ms   | <X>ms

SIMILAR_CANDIDATES: [optional, only if candidates are within 5% on priority metric]
  Candidates 1 and 2 are within <X>% on <perf_priority>.
  Tradeoffs:
    - Config 1: OTPS=<X>, TTFT p50=<X>ms, TPOT p50=<X>ms, E2E p50=<X>ms
    - Config 2: OTPS=<X>, TTFT p50=<X>ms, TPOT p50=<X>ms, E2E p50=<X>ms
    - Config 1 has <X>% higher OTPS but Config 2 has <X>% lower TTFT p99
  → User should choose between these.
  WINNER: PENDING_USER_CHOICE

WINNER: <config_path>  [or PENDING_USER_CHOICE if similar]
  Reason: <brief explanation — "clear winner by X% on <priority>" or "user decision needed">

Final Config: <MODEL_SHORT_NAME>_final_config.yaml  [only if auto-picked]

Usage:
  # Serve with trtllm-serve
  trtllm-serve <MODEL> --backend _autodeploy --extra_llm_api_options <FINAL_CONFIG> --port 8000

  # Quick test with build_and_run_ad.py
  python examples/auto_deploy/build_and_run_ad.py --model <MODEL> --args.yaml-extra <FINAL_CONFIG>
```

## Error Handling

- If a server fails to start: skip that candidate, note the failure reason.
- If a benchmark times out or crashes: skip that candidate, note the reason.
- If only 1 candidate survives: that's the winner by default.
- If no candidates survive serving evaluation: fall back to the best trtllm-bench result from the sanity-checker phase and report that the serving evaluation could not be completed.

---

## Fallback: Script Generation

If you cannot run serving benchmarks (no gpu-shell, no GPUs, sandbox restrictions, etc.), generate a self-contained bash script `$SESSION_DIR/run_serving_eval.sh` that the user can execute manually. Use the Write tool to save it.

The script must:
1. Be executable (`#!/usr/bin/env bash`, `set -euo pipefail`)
2. Accept no arguments — all values baked in from inputs
3. For each candidate: start server, wait for health, run benchmark, kill server
4. Parse all results and print a comparison table ranked by the user's `perf_priority`
5. Detect similar candidates (within 5%) and flag them for user choice

Script template structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# AD Config Serving Evaluation Script
# Generated by ad-conf-generator
# =============================================================================

MODEL="<MODEL_HF_ID>"
CONCURRENCY=<N>
PERF_PRIORITY="<perf_priority>"
DATASET="<dataset_path>"
CANDIDATES=( "<config_1.yaml>" "<config_2.yaml>" "<config_3.yaml>" )
CANDIDATE_NAMES=( "candidate_1" "candidate_2" "candidate_3" )
PORTS=( 8001 8002 8003 )

RESULTS_DIR="ad_serving_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "AD Config Serving Evaluation"
echo "Model: $MODEL"
echo "Concurrency: $CONCURRENCY"
echo "Priority: $PERF_PRIORITY"
echo "Candidates: ${#CANDIDATES[@]}"
echo "Results dir: $RESULTS_DIR"
echo "============================================"

cleanup_server() {
  local port=$1
  kill $(lsof -ti:"$port") 2>/dev/null || pkill -f "trtllm-serve.*--port $port" 2>/dev/null || true
  sleep 3
}

wait_for_server() {
  local port=$1
  local max_wait=600  # 10 minutes
  local elapsed=0
  echo "  Waiting for server on port $port..."
  while [ $elapsed -lt $max_wait ]; do
    if curl -s "http://localhost:$port/health" 2>/dev/null | grep -qE "ok|healthy|200"; then
      echo "  Server ready after ${elapsed}s"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "  Server failed to start within ${max_wait}s"
  return 1
}

PASSED_NAMES=()
PASSED_LOGS=()

for i in "${!CANDIDATES[@]}"; do
  CONFIG="${CANDIDATES[$i]}"
  NAME="${CANDIDATE_NAMES[$i]}"
  PORT="${PORTS[$i]}"
  SERVE_LOG="$RESULTS_DIR/serve_${NAME}.log"
  BENCH_LOG="$RESULTS_DIR/serving_bench_${NAME}.log"

  echo ""
  echo "[Candidate $((i+1))/${#CANDIDATES[@]}] $NAME"
  echo "  Config: $CONFIG"
  echo "  Port: $PORT"

  # Ensure port is free
  cleanup_server "$PORT"

  # Start server
  echo "  Starting trtllm-serve..."
  trtllm-serve "$MODEL" --backend _autodeploy \
    --extra_llm_api_options "$CONFIG" \
    --port "$PORT" \
    2>&1 | tee "$SERVE_LOG" &
  SERVER_PID=$!

  # Wait for server
  if ! wait_for_server "$PORT"; then
    echo "  SKIPPING $NAME — server failed to start"
    cleanup_server "$PORT"
    continue
  fi

  # Run benchmark
  echo "  Running serving benchmark..."
  if python tensorrt_llm/serve/scripts/benchmark_serving.py \
    --backend openai \
    --model "$MODEL" \
    --tokenizer "$MODEL" \
    --dataset-name trtllm_custom \
    --dataset-path "$DATASET" \
    --num-prompts 100 \
    --max-concurrency "$CONCURRENCY" \
    --port "$PORT" \
    2>&1 | tee "$BENCH_LOG"; then
    echo "  Benchmark completed"
    PASSED_NAMES+=("$NAME")
    PASSED_LOGS+=("$BENCH_LOG")
  else
    echo "  Benchmark FAILED (see $BENCH_LOG)"
  fi

  # Kill server
  echo "  Stopping server..."
  cleanup_server "$PORT"
done

# --- Summary ---
echo ""
echo "============================================"
echo "SERVING EVALUATION SUMMARY"
echo "Ranked by: $PERF_PRIORITY"
echo "============================================"
echo ""

if [ ${#PASSED_NAMES[@]} -eq 0 ]; then
  echo "ERROR: All candidates failed serving evaluation!"
  echo "Check logs in: $RESULTS_DIR/"
  exit 1
fi

echo "Completed benchmarks: ${#PASSED_NAMES[@]}"
echo ""
echo "Benchmark logs:"
for i in "${!PASSED_NAMES[@]}"; do
  echo "  ${PASSED_NAMES[$i]}: ${PASSED_LOGS[$i]}"
done

echo ""
echo "Review the logs above and compare metrics:"
echo "  - OTPS (Output Tokens Per Second)"
echo "  - Req/s (Request throughput)"
echo "  - TTFT p50/p99 (Time To First Token)"
echo "  - TPOT p50 (Time Per Output Token)"
echo "  - E2E latency p50/p99"
echo ""
echo "Full results in: $RESULTS_DIR/"
echo "Done."
```

Save the script as `$SESSION_DIR/run_serving_eval.sh`, make it executable, and tell the user:
```
I couldn't run the serving benchmarks directly. I've generated a script instead:

  chmod +x run_serving_eval.sh
  ./run_serving_eval.sh

The script will:
1. Start trtllm-serve for each candidate config (ports 8001-8003)
2. Run the serving benchmark against each server
3. Shut down each server after benchmarking
4. Print a summary of completed benchmarks

After running, share the output and I'll help you compare results and pick the winner.
```
