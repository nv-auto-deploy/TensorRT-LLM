#!/bin/bash
# Pre-flight: launch ONE trtllm-serve and confirm it reaches READY (model builds + serves)
# before committing to a long sweep. Catches env breakages (CuTe/cutlass-dsl, world_size
# propagation, missing binaries) in ~10 min instead of mid-sweep.
#
# Usage: validate_build.sh BACKEND CONFIG_YAML PORT MODEL OUT_DIR
set -uo pipefail
BACKEND="$1"; CONFIG="$2"; PORT="$3"; MODEL="$4"; OUT="${5:-/tmp}"
mkdir -p "$OUT"; LOG="$OUT/validate_${BACKEND}.log"
TRTLLM_SERVE="$(command -v trtllm-serve)"
[ -n "$TRTLLM_SERVE" ] || { echo "FATAL: trtllm-serve not on PATH"; exit 2; }
export FLASHINFER_USE_CUDA_NORM=1
: > "$LOG"
"$TRTLLM_SERVE" "$MODEL" --backend "$BACKEND" --extra_llm_api_options "$CONFIG" \
  --trust_remote_code --host 0.0.0.0 --port "$PORT" >> "$LOG" 2>&1 &
SP=$!
RC=1
for i in $(seq 1 900); do
  if curl -sf "http://0.0.0.0:${PORT}/v1/models" >/dev/null 2>&1; then echo "VALIDATE: READY after ${i}s"; RC=0; break; fi
  if ! kill -0 $SP 2>/dev/null; then echo "VALIDATE: SERVER DIED at ${i}s"; break; fi
  sleep 1
done
pkill -KILL -f "trtllm-serve.*--port $PORT" 2>/dev/null
pkill -KILL -f "trtllm-llmapi-launch" 2>/dev/null
fuser -k "${PORT}/tcp" 2>/dev/null
if [ $RC -ne 0 ]; then
  echo "VALIDATE: FAILED — last errors:"
  grep -iE "Error|Traceback|ICE|CuTe|Rank should|moe_ep_size=1|NotImplemented|No such file" "$LOG" \
    | grep -viE "RequestsDep|incompat|deprecat" | tail -8
fi
exit $RC
