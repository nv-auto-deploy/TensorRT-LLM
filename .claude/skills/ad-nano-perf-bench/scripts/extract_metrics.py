#!/usr/bin/env python3
"""Extract per-concurrency perf metrics from a sweep result dir into a CSV.

Usage:
    extract_metrics.py <result_dir> "<concurrencies>" [isl] [osl]

Reads <result_dir>/latest_run/isl_<isl>_osl_<osl>_conc_<c>/profile_export_aiperf.csv
for each concurrency and emits a tidy CSV to stdout:
    conc,tps_per_user,itl_ms,out_tps,ttft_ms,req_lat_ms

Flags points that break monotonicity (per-user TPS should fall, ITL should rise) to stderr.
"""

import os
import sys


def field(csv_path, metric_prefix):
    """Return the avg-column value for the first row whose Metric starts with prefix."""
    if not os.path.isfile(csv_path):
        return None
    with open(csv_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if parts and parts[0].strip().startswith(metric_prefix):
                # aiperf CSV: Metric,avg,min,max,... (statistics rows) OR Metric,Value (singletons)
                try:
                    return float(parts[1])
                except (IndexError, ValueError):
                    return None
    return None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    rd = sys.argv[1]
    concs = [int(x) for x in sys.argv[2].split()]
    isl = sys.argv[3] if len(sys.argv) > 3 else "1000"
    osl = sys.argv[4] if len(sys.argv) > 4 else "1000"

    rows = []
    print("conc,tps_per_user,itl_ms,out_tps,ttft_ms,req_lat_ms")
    for c in concs:
        d = os.path.join(rd, "latest_run", f"isl_{isl}_osl_{osl}_conc_{c}")
        csvp = os.path.join(d, "profile_export_aiperf.csv")
        tpu = field(csvp, "Output Token Throughput Per User")
        itl = field(csvp, "Inter Token Latency")
        otps = field(csvp, "Output Token Throughput (tokens")
        ttft = field(csvp, "Time to First Token")
        rlat = field(csvp, "Request Latency")
        if tpu is None:
            print(f"# c={c} MISSING ({csvp})", file=sys.stderr)
            continue
        rows.append((c, tpu, itl, otps, ttft, rlat))
        print(f"{c},{tpu},{itl},{otps},{ttft},{rlat}")

    # monotonicity sanity check
    for i in range(1, len(rows)):
        c, tpu, itl = rows[i][0], rows[i][1], rows[i][2]
        pc, ptpu, pitl = rows[i - 1][0], rows[i - 1][1], rows[i - 1][2]
        if tpu is not None and ptpu is not None and tpu > ptpu + 1.0:
            print(
                f"# WARN c={c}: per-user TPS {tpu:.1f} > c={pc} {ptpu:.1f} (non-monotonic) — re-run?",
                file=sys.stderr,
            )
        if itl is not None and pitl is not None and itl < pitl - 0.05:
            print(
                f"# WARN c={c}: ITL {itl:.2f} < c={pc} {pitl:.2f} (non-monotonic) — re-run?",
                file=sys.stderr,
            )
        # aggregate out-tps should generally rise until saturation; a big dip is suspicious
        otps, potps = rows[i][3], rows[i - 1][3]
        if otps is not None and potps is not None and otps < potps * 0.6:
            print(
                f"# WARN c={c}: aggregate out-TPS {otps:.0f} << c={pc} {potps:.0f} — likely bad point, re-run.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
