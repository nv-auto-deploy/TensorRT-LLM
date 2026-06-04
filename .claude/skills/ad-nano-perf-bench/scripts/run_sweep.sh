#!/bin/bash
# Run a detached Nano perf sweep that survives session exits.
#
# Usage:
#   run_sweep.sh <backend> <world_size> "<concurrencies>" <config_yaml> <result_dir> <port> [model_path]
#
#   backend       : trtllm-autodeploy | trtllm-pytorch
#   world_size    : TP size (only passed for trtllm-autodeploy; PT must set TP in yaml)
#   concurrencies : space-separated, e.g. "1 2 4 8 16 32 64 128 256"
#   config_yaml   : path to extra-llm-api-options yaml
#   result_dir    : output dir (created)
#   port          : server port
#   model_path    : optional; defaults to the CI mirror NVFP4 Nano-30B
#
# Writes <result_dir>/sweep.log. Monitor it for "All benchmarks completed successfully!".
set -u

BACKEND="${1:?backend}"
WORLD="${2:?world_size}"
CONC="${3:?concurrencies}"
CFG="${4:?config_yaml}"
RD="${5:?result_dir}"
PORT="${6:?port}"
MODEL="${7:-/scratch/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"

export HF_HUB_OFFLINE=1
mkdir -p "$RD"

# --world-size is only valid for trtllm-autodeploy; PT carries TP/EP in the yaml.
WS_ARG=""
if [ "$BACKEND" = "trtllm-autodeploy" ]; then
  WS_ARG="--world-size $WORLD"
fi

echo "RESULT_DIR=$RD"
echo "backend=$BACKEND world=$WORLD conc='$CONC' cfg=$CFG port=$PORT" | tee "$RD/run_params.txt"

setsid nohup sweep \
  --model "$MODEL" \
  --config-path "$CFG" \
  --server-type "$BACKEND" \
  $WS_ARG \
  --concurrencies "$CONC" \
  --isl 1000 --osl 1000 \
  --result-base-dir "$RD" \
  --tag nano_bench \
  --port "$PORT" > "$RD/sweep.log" 2>&1 &

sleep 3
if pgrep -f "sweep --model.*$PORT" >/dev/null 2>&1 || pgrep -f "sweep --model" >/dev/null 2>&1; then
  echo "launched detached; log: $RD/sweep.log"
else
  echo "WARNING: sweep process not detected — check $RD/sweep.log"
fi
