#!/usr/bin/env python3
"""Plot a Nano perf sweep: throughput Pareto + per-user-TPS/ITL vs concurrency.

Usage:
    plot_sweep.py <summary.csv> <out.png> [title] [--overlay other.csv:label ...]

summary.csv columns: conc,tps_per_user,itl_ms,out_tps,ttft_ms,req_lat_ms
Pass extra `--overlay path:label` args to compare multiple runs (e.g. AD vs PT, branch vs base)
on the same axes.
"""

import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    conc, tpu, itl, otps = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                conc.append(int(r["conc"]))
                tpu.append(float(r["tps_per_user"]))
                itl.append(float(r["itl_ms"]))
                otps.append(float(r["out_tps"]))
            except (ValueError, KeyError):
                continue
    return conc, tpu, itl, otps


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    csv_path, out_png = sys.argv[1], sys.argv[2]
    title = (
        sys.argv[3]
        if len(sys.argv) > 3 and not sys.argv[3].startswith("--")
        else "Nano-30B-A3B NVFP4 sweep (ISL/OSL 1000/1000)"
    )

    series = [(csv_path, "this run")]
    for a in sys.argv[3:]:
        if a.startswith("--overlay"):
            continue
        if ":" in a and a.endswith(".csv") is False and "/" in a.split(":")[0]:
            p, lab = a.split(":", 1)
            series.append((p, lab))
    # also accept "path:label" tokens after a literal --overlay
    for i, a in enumerate(sys.argv):
        if a == "--overlay" and i + 1 < len(sys.argv):
            tok = sys.argv[i + 1]
            if ":" in tok:
                p, lab = tok.rsplit(":", 1)
                series.append((p, lab))

    colors = ["#0071c5", "#d62728", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (path, label) in enumerate(series):
        conc, tpu, itl, otps = load(path)
        if not conc:
            continue
        col = colors[idx % len(colors)]
        ax = axes[0]
        ax.plot(otps, tpu, "o-", color=col, lw=2, ms=7, label=label)
        for c, x, y in zip(conc, otps, tpu):
            ax.annotate(f"c={c}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax2 = axes[1]
        ax2.plot(conc, tpu, "o-", color=col, lw=2, ms=7, label=f"{label} TPS/u")
        ax2.plot(conc, itl, "s--", color=col, lw=1.5, ms=5, alpha=0.6, label=f"{label} ITL")

    axes[0].set_xlabel("Aggregate Output Throughput (tokens/sec)")
    axes[0].set_ylabel("Per-User Throughput (tokens/sec/user)")
    axes[0].set_title(f"Throughput Pareto\n{title}")
    axes[0].grid(True, alpha=0.3)
    if len(series) > 1:
        axes[0].legend()

    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("Per-User TPS  /  ITL (ms)")
    axes[1].set_xscale("log", base=2)
    conc0 = load(series[0][0])[0]
    if conc0:
        axes[1].set_xticks(conc0)
        axes[1].set_xticklabels([str(c) for c in conc0])
    axes[1].set_title("Per-User TPS (solid) & ITL (dashed) vs Concurrency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
