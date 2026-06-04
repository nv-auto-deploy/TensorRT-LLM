#!/bin/bash
# Capture an nsys trace of a Nano server steady-state decode loop using the reliable
# nsys launch + start/stop workflow (the sweep --profile / capture-range paths are flaky).
#
# Usage:
#   trace_nano.sh <backend> <config_yaml> <result_dir> <port> [model_path]
#     backend : _autodeploy | pytorch   (note: trtllm-serve --backend value, not sweep server-type)
#
# Produces <result_dir>/trace.nsys-rep. Run extract on it with:
#   nsys stats --report cuda_gpu_kern_sum --format csv <result_dir>/trace.nsys-rep
#   nsys stats --report nvtx_pushpop_sum --format csv <result_dir>/trace.nsys-rep
set -u

BACKEND="${1:?backend (_autodeploy|pytorch)}"
CFG="${2:?config_yaml}"
RD="${3:?result_dir}"
PORT="${4:?port}"
MODEL="${5:-/scratch/fsw/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4}"

export HF_HUB_OFFLINE=1
mkdir -p "$RD"
SESSION="nano_trace_$$"

# 1. launch under nsys (profiling OFF until 'nsys start')
nsys launch -t cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true \
  --session-new="$SESSION" -e TLLM_LLMAPI_ENABLE_NVTX=1 \
  trtllm-serve "$MODEL" --host 0.0.0.0 --port "$PORT" --trust_remote_code \
  --backend "$BACKEND" --extra_llm_api_options "$CFG" > "$RD/server.log" 2>&1 &
echo "launched server under nsys session=$SESSION; waiting for ready..."

# 2. wait for readiness
for _ in $(seq 1 360); do
  grep -q "Application startup complete" "$RD/server.log" 2>/dev/null && break
  sleep 5
done
grep -q "Application startup complete" "$RD/server.log" || { echo "server did not become ready"; exit 1; }
echo "server ready"

# 3. warm up (no profiling)
aiperf profile --model "$MODEL" --url "http://0.0.0.0:$PORT" --endpoint-type chat --ui-type None \
  --streaming --concurrency 1 --request-count 1 --warmup-request-count 0 \
  --isl 1000 --osl 200 --no-server-metrics --use-server-token-count \
  --extra-inputs ignore_eos:true --artifact-dir "$RD/warmup" >/dev/null 2>&1 || true

# 4. start capture, drive steady-state traffic, stop
nsys start --session="$SESSION" -o "$RD/trace"
aiperf profile --model "$MODEL" --url "http://0.0.0.0:$PORT" --endpoint-type chat --ui-type None \
  --streaming --concurrency 1 --request-count 3 --warmup-request-count 1 \
  --isl 1000 --osl 1000 --no-server-metrics --use-server-token-count \
  --extra-inputs ignore_eos:true --artifact-dir "$RD/aiperf" 2>&1 | tail -3
nsys stop --session="$SESSION"

# 5. clean up
pkill -9 -f "trtllm-serve.*$PORT" 2>/dev/null
pkill -9 -f "nsys --start-agent.*$SESSION" 2>/dev/null
ls -lh "$RD"/trace.nsys-rep 2>/dev/null && echo "trace written: $RD/trace.nsys-rep"
