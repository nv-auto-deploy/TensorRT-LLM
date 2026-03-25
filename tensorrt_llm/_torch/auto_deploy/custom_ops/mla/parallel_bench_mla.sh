#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# parallel_bench_mla.sh — Run MLA kernel benchmarks across all 8 GPUs in parallel.
#
# Usage:
#   bash parallel_bench_mla.sh <mode> [extra_args...]
#
# Modes:
#   sweep        Split the SEQ_BLOCK×warps×stages parameter sweep across 8 GPUs
#   head_block   Benchmark HEAD_BLOCK=2,4,8,16,32 on 5 GPUs simultaneously
#   correctness  Run correctness check on all 8 GPUs at once
#   benchmark    Run standard benchmark with best params on all 8 GPUs
#
# Examples:
#   bash parallel_bench_mla.sh sweep
#   bash parallel_bench_mla.sh head_block
#   bash parallel_bench_mla.sh correctness
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
SWEEP="${SCRIPT_DIR}/sweep_triton_mla.py"
LOG_DIR="/tmp/mla_parallel_bench_$(date +%s)"
mkdir -p "${LOG_DIR}"

MODE="${1:-benchmark}"
shift || true
EXTRA_ARGS="$*"

echo "=== MLA Parallel Benchmark ==="
echo "Mode: ${MODE}  Extra: ${EXTRA_ARGS}"
echo "Log dir: ${LOG_DIR}"
echo ""

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
run_gpu() {
    local gpu_id="$1"
    local label="$2"
    local args="$3"
    local log="${LOG_DIR}/gpu${gpu_id}_${label}.log"
    echo "  GPU ${gpu_id}: ${label} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" python "${SWEEP}" ${args} >"${log}" 2>&1 &
    echo $!  # return PID
}

wait_all() {
    echo ""
    echo "Waiting for all jobs..."
    wait
    echo "All done."
}

print_logs() {
    for f in "${LOG_DIR}"/*.log; do
        echo ""
        echo "========================================="
        echo "$(basename "${f}")"
        echo "========================================="
        cat "${f}"
    done
}

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
case "${MODE}" in

sweep)
    # Split the 150-config grid into 8 GPU processes.
    # Each GPU sweeps a disjoint slice of SEQ_BLOCK values.
    # GPU 0-1: SEQ_BLOCK 4,8     (small)
    # GPU 2-3: SEQ_BLOCK 16,32   (medium)
    # GPU 4-5: SEQ_BLOCK 64      (large)
    # GPU 6-7: SEQ_BLOCK 128     (xlarge)
    WARPS_1="1,2,4"
    WARPS_2="8,16"
    STAGES_ALL="1,2,3,4,5"

    run_gpu 0 "sb4_8_w1"  "--sweep --sweep-seq-blocks 4,8   --sweep-num-warps ${WARPS_1} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu0.json --rep 30"
    run_gpu 1 "sb4_8_w2"  "--sweep --sweep-seq-blocks 4,8   --sweep-num-warps ${WARPS_2} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu1.json --rep 30"
    run_gpu 2 "sb16_32_w1" "--sweep --sweep-seq-blocks 16,32 --sweep-num-warps ${WARPS_1} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu2.json --rep 30"
    run_gpu 3 "sb16_32_w2" "--sweep --sweep-seq-blocks 16,32 --sweep-num-warps ${WARPS_2} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu3.json --rep 30"
    run_gpu 4 "sb64_w1"   "--sweep --sweep-seq-blocks 64    --sweep-num-warps ${WARPS_1} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu4.json --rep 30"
    run_gpu 5 "sb64_w2"   "--sweep --sweep-seq-blocks 64    --sweep-num-warps ${WARPS_2} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu5.json --rep 30"
    run_gpu 6 "sb128_w1"  "--sweep --sweep-seq-blocks 128   --sweep-num-warps ${WARPS_1} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu6.json --rep 30"
    run_gpu 7 "sb128_w2"  "--sweep --sweep-seq-blocks 128   --sweep-num-warps ${WARPS_2} --sweep-num-stages ${STAGES_ALL} --output ${LOG_DIR}/gpu7.json --rep 30"

    wait_all

    # Merge and find best per shape across all JSON files
    python3 - <<'PYEOF'
import json, math, glob, sys

results = []
for f in sorted(glob.glob("${LOG_DIR}/gpu*.json")):
    with open(f) as fh:
        results.extend(json.load(fh))

shape_ids = list(dict.fromkeys(r["shape_id"] for r in results))
print("\n=== Best config per shape (merged from all GPUs) ===")
for sid in shape_ids:
    cands = [r for r in results if r["shape_id"] == sid and not math.isnan(r.get("kernel_us", float("nan")))]
    if cands:
        best = min(cands, key=lambda r: r["kernel_us"])
        print(f"  {sid:>4}  kernel={best['kernel_us']:>8.1f} µs  "
              f"SEQ_BLOCK={best['SEQ_BLOCK']:>3}  num_warps={best['num_warps']:>2}  "
              f"num_stages={best['num_stages']:>1}")
PYEOF
    ;;

head_block)
    # Test HEAD_BLOCK = 2, 4, 8, 16, 32 on 5 GPUs simultaneously.
    # GPUs 5-7 run the original kernel with best sweep params for comparison.
    echo "Benchmarking HEAD_BLOCK variants in parallel..."

    run_gpu 0 "hb2"   "--head-block 2  --rep 100"
    run_gpu 1 "hb4"   "--head-block 4  --rep 100"
    run_gpu 2 "hb8"   "--head-block 8  --rep 100"
    run_gpu 3 "hb16"  "--head-block 16 --rep 100"
    run_gpu 4 "hb32"  "--head-block 32 --rep 100"
    # Original kernel with best params on remaining GPUs for sanity check
    run_gpu 5 "orig_decode" "--seq-block 128 --num-warps 8 --num-stages 4 --rep 100"
    run_gpu 6 "orig_prefill" "--seq-block 16  --num-warps 1 --num-stages 5 --rep 100"

    wait_all
    print_logs
    ;;

correctness)
    # Run correctness check on all GPUs simultaneously to verify all variants
    echo "Running correctness checks in parallel..."

    for gpu in 0 1 2 3 4 5 6 7; do
        run_gpu "${gpu}" "correctness" "--correctness ${EXTRA_ARGS}"
    done

    wait_all
    echo ""
    echo "=== Correctness Summary ==="
    for f in "${LOG_DIR}"/*.log; do
        status=$(grep -c "Overall: PASS" "${f}" 2>/dev/null || echo 0)
        echo "  $(basename ${f}): $([ "${status}" -gt 0 ] && echo PASS || echo FAIL)"
    done
    ;;

benchmark)
    # Run standard benchmark (with best params) on all 8 GPUs
    echo "Running benchmark with best params on all 8 GPUs..."

    for gpu in 0 1 2 3 4 5 6 7; do
        run_gpu "${gpu}" "bench" "--seq-block 128 --num-warps 8 --num-stages 4 ${EXTRA_ARGS}"
    done

    wait_all
    # Print first GPU's output as representative
    echo ""
    echo "=== Results from GPU 0 ==="
    cat "${LOG_DIR}/gpu0_bench.log"
    ;;

*)
    echo "Unknown mode: ${MODE}"
    echo "Usage: bash parallel_bench_mla.sh [sweep|head_block|correctness|benchmark]"
    exit 1
    ;;

esac

echo ""
echo "All logs in: ${LOG_DIR}"
