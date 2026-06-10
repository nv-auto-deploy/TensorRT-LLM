#!/usr/bin/env bash
# Full 1319-sample GSM8K matrix (single GPU 0, sequential). Resumable: a run whose log already has a
# final result line is skipped. AD runs retry on the flaky MPI-startup hang (setsid -> own group).
#   AD DFlash (overlap ON, DFLASH ON): {torch-simple, cudagraph} x {llama, qwen3}  -> compile backend
#     is the only variable, to isolate the cudagraph effect on accuracy.
#   PyTorch backend, default params (no spec): {llama, qwen3}                        -> reference ceiling.
set -u
export TRITON_CACHE_DIR=/home/scratch.gramnarayan_coreai/.triton/cache
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models
GPU=0
N=1319
DS=$LLM_MODELS_ROOT/datasets/openai/gsm8k
LOGDIR=debug/logs/full_matrix; mkdir -p "$LOGDIR"
declare -A LLAMA=( [path]=$LLM_MODELS_ROOT/llama-3.1-model/Llama-3.1-8B-Instruct )
declare -A QWEN=( [path]=$LLM_MODELS_ROOT/Qwen3/Qwen3-8B )

ad_run() {  # $1=model(llama|qwen3) $2=cudagraph(0|1)
  local model="$1" cg="$2"
  local tag="ad_${model}_$([ "$cg" = 1 ] && echo cudagraph || echo torchsimple)"
  local log="$LOGDIR/${tag}.log"
  grep -q ">>> RESULT.*GSM8K_accuracy" "$log" 2>/dev/null && { echo "[$tag] already done, skip"; return; }
  for attempt in 1 2 3 4; do
    echo "=== attempt $attempt ===" > "$log"
    setsid bash -c "CUDA_VISIBLE_DEVICES=$GPU MODEL=$model DFLASH=1 OVERLAP=1 CUDAGRAPH=$cg NUM_SAMPLES=$N ATTN=trtllm python -u debug/spikes/gsm8k_ad_compare.py >> '$log' 2>&1" &
    sleep 3
    while pgrep -f "MODEL=$model DFLASH=1 OVERLAP=1 CUDAGRAPH=$cg NUM_SAMPLES=$N" >/dev/null; do
      sleep 15
      grep -qE ">>> RESULT.*GSM8K_accuracy|Traceback|OutOfMemory|AcceleratorError|incompatible function" "$log" 2>/dev/null && break
    done
    grep -q ">>> RESULT.*GSM8K_accuracy" "$log" && { echo "[$tag] done"; return; }
    if grep -qE "No CUDA|MpiPoolSession" "$log" && ! grep -q "stage=compile" "$log"; then echo "[$tag] startup hang, retry"; sleep 4; continue; fi
    echo "[$tag] failed (see log)"; return
  done
}

pyt_run() {  # $1=model $2=hf-path
  local model="$1" path="$2" log="$LOGDIR/pyt_${model}.log"
  grep -q "average accuracy" "$log" 2>/dev/null && { echo "[pyt_$model] already done, skip"; return; }
  CUDA_VISIBLE_DEVICES=$GPU trtllm-eval --model "$path" --backend pytorch gsm8k \
    --dataset_path "$DS" --num_samples "$N" --random_seed 0 > "$log" 2>&1
  grep -q "average accuracy" "$log" && echo "[pyt_$model] done" || echo "[pyt_$model] failed (see log)"
}

ad_run llama 0
ad_run llama 1
ad_run qwen3 0
ad_run qwen3 1
pyt_run llama "${LLAMA[path]}"
pyt_run qwen3 "${QWEN[path]}"

echo ""
echo "================== FULL 1319-SAMPLE GSM8K MATRIX =================="
printf "%-34s | %-8s | %s\n" "config" "acc" "accept_rate"
printf -- "-----------------------------------+----------+------------\n"
for f in ad_llama_torchsimple ad_llama_cudagraph ad_qwen3_torchsimple ad_qwen3_cudagraph; do
  log="$LOGDIR/${f}.log"
  acc=$(grep -oE "GSM8K_accuracy=[0-9.]+" "$log" 2>/dev/null | tail -1 | cut -d= -f2)
  ar=$(grep -oE "accept_rate=[0-9.]+%" "$log" 2>/dev/null | tail -1 | cut -d= -f2)
  printf "%-34s | %-8s | %s\n" "$f" "${acc:-FAIL}" "${ar:-}"
done
for m in llama qwen3; do
  log="$LOGDIR/pyt_${m}.log"
  acc=$(grep -oE "average accuracy: [0-9.]+" "$log" 2>/dev/null | tail -1 | awk '{print $3}')
  printf "%-34s | %-8s | %s\n" "pyt_${m}_default" "${acc:-FAIL}" "(no spec)"
done
echo "=================================================================="
