---
name: ad-conf-generator
description: >
  Generate and evaluate optimal AutoDeploy configs for a user's model.
  Asks for model, precision, max_seq_len, max_batch_size, concurrency,
  then runs 4 agents: model analysis, config candidate generation,
  sanity check + trtllm-bench, and final trtllm-serve evaluation.
---

# AutoDeploy Config Generator

Generate optimal AutoDeploy (AD) YAML configurations for a given model by analyzing the model, generating config candidates, sanity-checking them, benchmarking with `trtllm-bench`, and evaluating with `trtllm-serve`.

## Session Logging

At the very start of the session (before Phase 0), create a **session directory** to store all artifacts (session log, candidate configs, benchmark results, final config). Derive `MODEL_SHORT_NAME` from the model ID (e.g., `meta-llama/Llama-3.1-70B-Instruct` → `llama3.1_70b`).

```bash
MODEL_SHORT_NAME="<derived_short_name>"  # e.g., llama3.1_70b
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="ad_conf_gen_${MODEL_SHORT_NAME}_${TIMESTAMP}"
mkdir -p "$SESSION_DIR"
LOG_FILE="$SESSION_DIR/session_log.md"
echo "# AD Config Generator Session Log" > "$LOG_FILE"
echo "**Session ID:** $SESSION_DIR" >> "$LOG_FILE"
echo "**Started:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "**Status:** in_progress" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

All generated artifacts (candidate configs, benchmark logs, result JSONs, datasets, final config, fallback scripts) should be saved under `$SESSION_DIR`. Pass both `$SESSION_DIR` and `$LOG_FILE` to every agent so they store outputs in the session directory.

**Phase 0 recording:** After collecting user requirements, append:
```bash
echo "## User Requirements" >> "$LOG_FILE"
echo "| Parameter | Value |" >> "$LOG_FILE"
echo "|-----------|-------|" >> "$LOG_FILE"
echo "| Model | $MODEL |" >> "$LOG_FILE"
echo "| Precision | $PRECISION |" >> "$LOG_FILE"
echo "| max_seq_len | $MAX_SEQ_LEN |" >> "$LOG_FILE"
echo "| max_batch_size | $MAX_BATCH_SIZE |" >> "$LOG_FILE"
echo "| Concurrency | $CONCURRENCY |" >> "$LOG_FILE"
echo "| Performance priority | $PERF_PRIORITY |" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "## Phase 0 — User Interaction" >> "$LOG_FILE"
echo "- User provided model: $MODEL" >> "$LOG_FILE"
echo "- Precision: $PRECISION" >> "$LOG_FILE"
echo "- Performance priority: $PERF_PRIORITY" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

**Phase transitions:** Before launching each agent (Phases 1-4), append a phase header:
```bash
echo "---" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "## Phase N — <Phase Name>" >> "$LOG_FILE"
echo "**Started:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

After each agent returns, append end time and status:
```bash
echo "**Ended:** $(date '+%Y-%m-%d %H:%M:%S') | **Status:** success" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

(Use `**Status:** failed — <reason>` if the agent failed.)

**Pass `$SESSION_DIR` and `$LOG_FILE` to every agent:** Add both paths to each agent's inputs (Phases 1-4) so agents store outputs in the session directory and append activity to the log.

**Phase 4.5 tie-breaking:** If tie-breaking is needed, append:
```bash
echo "## Phase 4.5 — Tie-Breaking" >> "$LOG_FILE"
echo "- Close candidates: <list>" >> "$LOG_FILE"
echo "- User chose: <chosen config>" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

**Phase 5 summary:** After generating the final report, append:
```bash
echo "---" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
echo "## Session Summary" >> "$LOG_FILE"
echo "**Ended:** $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "**Duration:** <calculated duration>" >> "$LOG_FILE"
echo "**Winner:** <winning config path>" >> "$LOG_FILE"
echo "**Improvement over baseline:** <e.g., OTPS +54%, TTFT -29%>" >> "$LOG_FILE"
echo "**Final config:** <final config path>" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
# Update status from in_progress to completed
sed -i 's/\*\*Status:\*\* in_progress/**Status:** completed/' "$LOG_FILE"
```

## Phase 0 — Collect User Requirements

Before launching any agents, gather the following from the user. If the user has already provided some of these, confirm the rest:

| Parameter | Description | Example |
|-----------|-------------|---------|
| **Model** | HuggingFace model ID or local path | `meta-llama/Llama-3.1-70B-Instruct` |
| **Precision** | bf16, fp8, or fp4 (can be deduced from model name/config) | `fp8` |
| **max_seq_len** | Target max sequence length (ISL + OSL) | `4096` |
| **max_batch_size** | Maximum batch size for serving | `64` |
| **Concurrency** | Typical number of concurrent requests | `32` |
| **Performance priority** | Which metric matters most for ranking configs | `OTPS` |
| **Benchmark mode** | `trtllm-bench` only (faster) or `trtllm-bench` + `bench-sweep` (comprehensive) | `trtllm-bench` |

Ask the user which performance metric they want to prioritize for ranking configs. Present these options:
1. **OTPS** — Output Tokens Per Second (total throughput) — *default*
2. **Req/s** — Request throughput (requests per second)
3. **TTFT** — Time To First Token (responsiveness / prefill speed)
4. **TPOT** — Time Per Output Token (streaming / decode speed)
5. **E2E latency** — End-to-end request latency

If the user doesn't specify, default to **OTPS**.

Ask the user which benchmark tests to run for candidate screening:
1. **trtllm-bench only** — Faster screening (~N minutes per config). Good for narrowing down candidates quickly. *default*
2. **trtllm-bench + bench-sweep** — More comprehensive (adds serving benchmark per config). Takes significantly longer.

**Note:** Regardless of the user's choice, the **final winning config** will always be evaluated with `bench-sweep` (serving benchmark) in Phase 4 to get production-representative metrics.

Store these as a requirements dict to pass to each agent:
```python
requirements = {
    "model": "<HF_MODEL_ID>",
    "precision": "<bf16|fp8|fp4>",
    "max_seq_len": <int>,
    "max_batch_size": <int>,
    "concurrency": <int>,
    "perf_priority": "<OTPS|req_per_sec|TTFT|TPOT|E2E_latency>",
    "bench_mode": "<trtllm-bench|trtllm-bench+bench-sweep>",
}
```

If the user only provides a model name, deduce precision from the model name (e.g., `-FP8` suffix → fp8) and ask for the remaining parameters with sensible defaults.

## Phase 1 — Model Analysis

Launch the `ad-conf-model-analyzer` agent (`.claude/agents/ad-conf-generator/model-analyzer.md`).

**Pass to agent:**
- HF model ID
- User-specified precision
- All user requirements
- Session directory path (`$SESSION_DIR`) and log file path (`$LOG_FILE`)

**Expect back:** A structured model analysis report containing model architecture details, param count, GPU requirements, estimated minimum `world_size`, and (if the model is large) a recommended `num_hidden_layers` for fast mode with the minimum safe layer count.

### Fast Mode Decision (after model analysis)

If the model analysis report indicates the model is **fast mode eligible** (estimated size >100GB), present the analysis results to the user and offer fast mode:

> "Based on the model analysis, this model is ~X GB (Y B parameters at Z precision). It qualifies for **fast mode**, which loads only N out of L layers. This speeds up benchmarking significantly but skips output quality validation (sanity checks). Benchmark metrics (throughput, latency) are still valid for comparing configs against each other.
>
> Would you like to use fast mode? [yes/no]"

Include key details from the model analysis (architecture type, layer diversity, why N was chosen) so the user can make an informed decision.

**If the user chooses fast mode:**
- Add `"fast_mode": true` and `"num_hidden_layers": N` to the requirements dict
- The config-candidate-generator will add `num_hidden_layers: N` to each generated config
- The sanity-checker will **skip** the output quality check (Step 2) and go directly to benchmarking
- Benchmark results are relative (valid for comparing configs) but not absolute

**If the user declines fast mode:**
- Proceed normally with full model loading
- All phases run as usual including sanity checks

**Choosing N (num_hidden_layers):**
- N must include all architecturally distinct layer types. The model-analyzer report includes the recommended N.
- Example: Nemotron Super V3 has attention layers starting at layer 8 — so N must be >= 8 to capture all layer types.
- Default heuristic: `N = max(8, first_layer_with_all_unique_types + 2)`

## Phase 1.5 — Baseline Performance + Time Estimation

Before generating any optimized candidates, establish a **vanilla baseline**. This serves two purposes:
1. Reference performance numbers for comparison
2. **Time estimation** for planning the rest of the session

### Step 1: Run vanilla baseline with trtllm-bench

Run a single `trtllm-bench` throughput benchmark with vanilla (default) AD settings:
- **No extra config** — just the model with default AD settings (no `--args.yaml-extra`)
- **Standardized dataset** — ISL=1024, OSL=1024, 100 requests
- **Fixed concurrency** — 16 (standardized for comparable baseline)

**Time the entire run** (model load + transform + warmup + benchmark):
```bash
BASELINE_START=$(date +%s)
# ... run trtllm-bench ...
BASELINE_END=$(date +%s)
BASELINE_DURATION_SEC=$((BASELINE_END - BASELINE_START))
BASELINE_DURATION_MIN=$((BASELINE_DURATION_SEC / 60))
```

### Step 2: Report time estimation and get user approval on candidate count

After the vanilla baseline completes, **immediately report to the user**:

> **Vanilla baseline complete.** Here are the results:
> (show baseline metrics table)
>
> **Time for one benchmark run:** ~X minutes (Y minutes transform/load + Z minutes benchmark)
>
> Phase 2 will generate N candidate configs. Here's the time estimate:
>
> | Candidates | trtllm-bench screening | + bench-sweep final eval | Total estimate |
> |------------|----------------------|-------------------------|---------------|
> | 3 configs  | ~{3*X} min           | ~{X*2} min              | ~{3*X + X*2} min |
> | 5 configs  | ~{5*X} min           | ~{X*2} min              | ~{5*X + X*2} min |
>
> **Your priority metric:** {perf_priority}
> **Benchmark mode:** {bench_mode}
>
> How many candidates would you like to test? (default: 3-5 based on model complexity)
> Would you like to proceed?

**Important rules for time communication:**
- **ALWAYS** show the per-run time and total estimated time before starting candidate benchmarks
- If total estimated time exceeds **1 hour**, explicitly warn the user: "This will take approximately X hours."
- If total estimated time exceeds **2 hours**, suggest reducing candidate count or using fast mode
- Let the user override the candidate count (they may want fewer for speed or more for coverage)
- Store `BASELINE_DURATION_SEC` in the requirements dict so Phase 3 can use it for progress updates

### Step 3: bench-sweep baseline (deferred)

The `bench-sweep` serving baseline is **not** run here. It will be run in Phase 4 alongside the winning config(s), so the serving comparison is apples-to-apples. This saves time during screening.

The trtllm-bench baseline results are carried through all phases and included in the final report as a reference row.

## Phase 2 — Config Candidate Generation

Launch the `ad-conf-candidate-generator` agent (`.claude/agents/ad-conf-generator/config-candidate-generator.md`).

**Pass to agent:**
- Model analysis report from Phase 1
- All user requirements
- Session directory path (`$SESSION_DIR`) and log file path (`$LOG_FILE`)

**Expect back:** 1-5 candidate config YAML file paths (count depends on model complexity) with descriptions of each candidate's tradeoff profile.

## Phase 3 — Sanity Check + Benchmark

Launch the `ad-conf-sanity-checker` agent (`.claude/agents/ad-conf-generator/sanity-checker.md`).

**Pass to agent:**
- HF model ID
- List of candidate config paths from Phase 2
- User requirements (max_seq_len, max_batch_size, concurrency, bench_mode)
- Estimated world_size from Phase 1
- Baseline duration (`BASELINE_DURATION_SEC`) from Phase 1.5 for progress reporting
- Session directory path (`$SESSION_DIR`) and log file path (`$LOG_FILE`)

The agent should use `BASELINE_DURATION_SEC` to report estimated remaining time as each candidate benchmark starts:
> "Starting candidate 2/5... Estimated ~X minutes remaining (based on ~Y min/run)"

**Expect back:** Baseline performance numbers + ranked list of top 3 configs with `trtllm-bench` metrics (OTPS, request throughput, TTFT, TPOT), plus dropped configs with failure reasons. If multiple candidates have similar performance (within 5% of each other on the priority metric), the agent will flag this and you should present the close candidates to the user for manual selection before proceeding to Phase 4.

**After Phase 3 returns**, present the baseline vs. candidates comparison to the user immediately:
> "Here's how your candidates compare to the vanilla baseline:"
> (show table with baseline row highlighted)

## Phase 4 — Serving Evaluation (bench-sweep)

The final evaluation **always** uses `bench-sweep` to get production-representative serving metrics with multi-concurrency sweeps. This applies regardless of the user's `bench_mode` choice in Phase 0 (which only controls Phase 3 screening).

Launch the `ad-conf-serving-evaluator` agent (`.claude/agents/ad-conf-generator/serving-evaluator.md`).

**Pass to agent:**
- HF model ID
- Top 3 config paths from Phase 3 (or fewer if tie-breaking already happened)
- User concurrency target
- Dataset path generated in Phase 3
- Baseline duration (`BASELINE_DURATION_SEC`) for time estimates
- Session directory path (`$SESSION_DIR`) and log file path (`$LOG_FILE`)

The agent runs `bench-sweep` for each config with a concurrency sweep (e.g., `"1 4 8 16 32 64"`). Before starting, report the estimated time:
> "Running bench-sweep serving evaluation for N configs. Estimated ~X minutes per config (based on baseline timing). Total: ~Y minutes."

**Expect back:** Final config recommendation with full comparison table across all configs. If multiple candidates have similar serving performance (within 5% on the priority metric), the agent will flag this for user decision.

## Phase 4.5 — User Tie-Breaking (if needed)

If the serving-evaluator reports that top candidates are within 5% of each other on the user's priority metric, present the close candidates side by side with all metrics and ask the user to pick their preferred config. Explain the tradeoffs (e.g., "Config A has 2% higher OTPS but Config B has 15% lower TTFT p99").

## Phase 5 — Final Report

Present to the user:

1. **Winner config** — the recommended YAML with a brief explanation of why it won (or note that the user chose it in a tie-break)
2. **Baseline vs. Winner** — show the improvement over vanilla baseline:
   - e.g., "OTPS: 1200 → 1850 (+54%), TTFT p50: 45ms → 32ms (-29%)"
3. **Full comparison table** — all evaluated configs + baseline with key metrics side by side:
   - OTPS (output tokens per second)
   - Request throughput (req/s)
   - TTFT p50 / p99
   - TPOT p50
   - E2E latency p50 / p99
   - The **baseline** row should be clearly labeled and shown first as the reference point
4. **Dropped configs** — configs that failed sanity check and why
4. **Final config file path** — saved as `$SESSION_DIR/{model_short_name}_final_config.yaml`
5. **Usage instructions** — how to use the winning config:
   ```bash
   # With trtllm-serve
   trtllm-serve <MODEL> --backend _autodeploy --extra_llm_api_options <FINAL_CONFIG> --port 8000

   # With build_and_run_ad.py
   python examples/auto_deploy/build_and_run_ad.py --model <MODEL> --args.yaml-extra <FINAL_CONFIG>
   ```

## Error Handling

- If Phase 1 fails (model not found, GPU unavailable), report error and stop.
- If Phase 2 produces no candidates, fall back to the closest existing config from `examples/auto_deploy/model_registry/configs/`.
- If all candidates fail sanity check in Phase 3, report failures and suggest the user adjust requirements (e.g., lower max_batch_size, increase world_size).
- If Phase 4 serving evaluation fails for all configs, fall back to the best `trtllm-bench` result from Phase 3.

## Fallback: Script Generation Mode

If the agents **cannot run benchmarks** (no gpu-shell, no GPUs, sandbox restrictions, dependency issues), they will automatically fall back to generating runnable bash scripts instead of executing benchmarks directly.

**Phase 3 fallback** → generates `$SESSION_DIR/run_sanity_bench.sh`:
- Prepares dataset, sanity-checks each candidate, benchmarks survivors, prints ranked summary

**Phase 4 fallback** → generates `$SESSION_DIR/run_serving_eval.sh`:
- Starts trtllm-serve per candidate, runs serving benchmarks, shuts down servers, prints summary

When scripts are generated instead of live results:
1. Present the scripts to the user with instructions to run them
2. Tell the user to share the output after running
3. Once the user provides results, parse them and continue with Phase 4.5 (tie-breaking) and Phase 5 (final report) as normal
