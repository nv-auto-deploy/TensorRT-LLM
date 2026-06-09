#!/usr/bin/env bash
# GSM8K AD-backend matrix: {llama,qwen3} x {dflash off,on}, one GPU each, in parallel.
# Robust against the flaky MPI-startup hang: each run is its own process group (setsid) so a
# retry kills the WHOLE group (incl. MPI worker grandchildren) -> no orphaned GPU memory.
set -u
export TRITON_CACHE_DIR=/home/scratch.gramnarayan_coreai/.triton/cache
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models
N="${NUM_SAMPLES:-200}"
LOGDIR=debug/logs
mkdir -p "$LOGDIR"

run_one() {  # $1=gpu $2=model $3=dflash
  local gpu="$1" model="$2" dflash="$3"
  local log="$LOGDIR/gsm8k_${model}_$([ "$dflash" = 1 ] && echo on || echo off).log"
  for attempt in 1 2 3 4 5; do
    echo "=== attempt $attempt (gpu=$gpu) ===" > "$log"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$gpu MODEL=$model DFLASH=$dflash NUM_SAMPLES=$N ATTN=trtllm \
      python -u debug/spikes/gsm8k_ad_compare.py >> '$log' 2>&1" &
    local pid=$! pgid
    pgid=$(ps -o pgid= -p $pid | tr -d ' ')
    local prog=0
    for i in $(seq 1 24); do
      sleep 10
      grep -qE "stage=pattern_matcher|Loading weight|lm_eval|>>> RESULT" "$log" 2>/dev/null && { prog=1; break; }
      kill -0 $pid 2>/dev/null || break
    done
    if [ "$prog" = 1 ]; then
      echo "[launcher $model/$dflash] progressed (gpu=$gpu)"; wait $pid; break
    fi
    echo "[launcher $model/$dflash] hang -> kill group $pgid, retry"
    kill -9 -"$pgid" 2>/dev/null; sleep 4
  done
}

run_one 0 llama 0 &
run_one 1 llama 1 &
run_one 2 qwen3 0 &
run_one 3 qwen3 1 &
wait
echo "======================== MATRIX RESULTS ========================"
grep -hE ">>> RESULT" "$LOGDIR"/gsm8k_llama_off.log "$LOGDIR"/gsm8k_llama_on.log \
  "$LOGDIR"/gsm8k_qwen3_off.log "$LOGDIR"/gsm8k_qwen3_on.log 2>/dev/null
echo "================================================================"
