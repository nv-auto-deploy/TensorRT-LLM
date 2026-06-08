#!/bin/bash
# Serving throughput sweep for one backend+config: launch trtllm-serve, drive aiperf
# over a variable-ISL dataset (SPEED-Bench) across concurrencies, and parse
# OTPS / ITL / TTFT into a summary JSON.
#
# Usage:
#   serve_sweep.sh LABEL BACKEND CONFIG_YAML PORT MODEL DATASET OUT_DIR [CONCS]
#     LABEL       e.g. ad_balancer | ad_nobalancer | pt
#     BACKEND     _autodeploy | pytorch
#     CONFIG_YAML extra_llm_api_options yaml (AD: must contain `world_size: <N>`;
#                 PT: must contain `tensor_parallel_size: <N>`). NO mpirun is used —
#                 each backend spawns its own workers from the config.
#     PORT        server port (use a fresh one per run)
#     MODEL       HF model dir (also used as aiperf --tokenizer)
#     DATASET     SPEED-Bench single_turn JSONL (variable ISL)
#     OUT_DIR     output dir for logs + summary json + aiperf artifacts
#     CONCS       optional space-separated concurrencies (default "1 2 4 8 16 32 64 128 256")
#
# Writes: $OUT_DIR/${LABEL}_serve.json  and  $OUT_DIR/serve_${LABEL}.log
set -uo pipefail

LABEL="$1"; BACKEND="$2"; CONFIG="$3"; PORT="$4"; MODEL="$5"; DATASET="$6"; OUT="$7"
CONCS="${8:-1 2 4 8 16 32 64 128 256}"
mkdir -p "$OUT"
LOG="$OUT/serve_${LABEL}.log"
SUMMARY="$OUT/${LABEL}_serve.json"

# Resolve binaries from PATH — do NOT hardcode /usr/local/bin (it moves across
# containers/leases; a stale absolute path silently fails with "No such file").
TRTLLM_SERVE="$(command -v trtllm-serve)"
AIPERF="$(command -v aiperf)"
[ -n "$TRTLLM_SERVE" ] || { echo "FATAL: trtllm-serve not on PATH"; exit 2; }
[ -n "$AIPERF" ] || { echo "FATAL: aiperf not on PATH"; exit 2; }
[ -f "$DATASET" ] || { echo "FATAL: dataset not found: $DATASET"; exit 2; }

cleanup() {
  set +e
  pkill -INT  -f "trtllm-serve.*--port $PORT" 2>/dev/null
  sleep 3
  pkill -KILL -f "trtllm-serve.*--port $PORT" 2>/dev/null
  pkill -KILL -f "trtllm-llmapi-launch"       2>/dev/null
  fuser -k "${PORT}/tcp" 2>/dev/null
}
trap cleanup EXIT

# Sidestep a broken nvidia-cutlass-dsl cute.experimental stub on some nodes by using
# flashinfer's CUDA-JIT rmsnorm instead of the CuTe-DSL kernel (see references/gotchas.md).
export FLASHINFER_USE_CUDA_NORM=1

echo "[$LABEL] launching $BACKEND server on port $PORT at $(date)"
# NO mpirun / --tp_size: AutoDeploy reads `world_size` from the yaml and spawns its own
# workers; PyTorch reads `tensor_parallel_size`. Launching under `mpirun -n N` instead
# breaks AD (workers never receive world_size -> "Rank should be an integer between 0 and 0").
"$TRTLLM_SERVE" "$MODEL" \
  --backend "$BACKEND" --extra_llm_api_options "$CONFIG" \
  --trust_remote_code --host 0.0.0.0 --port "$PORT" \
  > "$LOG" 2>&1 &
SERVER_PID=$!

# Wait for readiness (model build can take ~10 min for 120B TP4); bail if the server dies.
READY=0
for i in $(seq 1 1200); do
  if curl -sf "http://0.0.0.0:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; echo "[$LABEL] ready after ${i}s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[$LABEL] SERVER DIED during build"; tail -40 "$LOG"; exit 1; fi
  sleep 1
done
[ "$READY" = 1 ] || { echo "[$LABEL] server ready timeout"; tail -40 "$LOG"; exit 1; }

echo "{\"label\": \"$LABEL\", \"backend\": \"$BACKEND\", \"results\": [" > "$SUMMARY"
FIRST=1
for c in $CONCS; do
  N=$(( c * 5 )); [ $N -lt 20 ] && N=20; [ $N -gt 768 ] && N=768   # ~5x concurrency for steady state, bounded
  DIR="$OUT/aiperf_${LABEL}_c${c}"
  echo "[$LABEL] === c=$c, $N reqs === $(date +%H:%M:%S)"
  # SPEED-Bench variable-ISL prompts; output length is model-determined (--use-server-token-count).
  "$AIPERF" profile --model "$MODEL" --url "0.0.0.0:${PORT}" --endpoint-type chat --ui-type None \
    --streaming --concurrency "$c" --request-count "$N" \
    --num-warmup-requests 1 --request-timeout-seconds 1800 \
    --use-server-token-count --no-server-metrics \
    --input-file "$DATASET" --custom-dataset-type single_turn \
    --tokenizer "$MODEL" --tokenizer-trust-remote-code \
    --artifact-dir "$DIR" >> "$LOG" 2>&1
  python3 - "$c" "$DIR/profile_export_aiperf.json" "$SUMMARY" "$FIRST" <<'PY'
import json, sys
c, jpath, summ, first = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
def g(d, k, s="avg"):
    v = d.get(k, {})
    return v.get(s) if isinstance(v, dict) else None
try:
    d = json.load(open(jpath))
    itl = g(d, "inter_token_latency")  # ms
    row = {"concurrency": c, "otps": g(d, "output_token_throughput"),
           "itl_ms": itl, "ttft_ms": g(d, "time_to_first_token"),
           "req_per_s": g(d, "request_throughput"),
           "user_tps": (1000.0/itl) if itl else None}
except Exception as e:
    row = {"concurrency": c, "error": str(e)}
with open(summ, "a") as f:
    f.write(("" if first == "1" else ",\n") + json.dumps(row))
print(f"  c={c}: OTPS={row.get('otps')}  userTPS={row.get('user_tps')}  TTFT={row.get('ttft_ms')}")
PY
  FIRST=0
done
echo "" >> "$SUMMARY"; echo "]}" >> "$SUMMARY"
echo "[$LABEL] DONE; summary -> $SUMMARY"
