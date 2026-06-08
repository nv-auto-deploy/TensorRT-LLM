#!/usr/bin/env python3
"""Plot an OTPS-vs-per-user-TPS pareto from N serve_sweep summary JSONs and print a
comparison table.

Usage:
  plot_pareto.py OUT_PNG LABEL1:summary1.json LABEL2:summary2.json [LABEL3:...] ...

Each summary JSON is produced by serve_sweep.sh:
  {"label":..., "results":[{"concurrency","otps","itl_ms","ttft_ms","user_tps"}, ...]}

The pareto plots per-user TPS (x) vs system OTPS (y); up-and-to-the-right is better.
The first series is treated as the baseline for the % comparison table.
"""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["tab:gray", "tab:orange", "tab:green", "tab:blue", "tab:red", "tab:purple"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

if len(sys.argv) < 3:
    print(__doc__); sys.exit(2)

out_png = sys.argv[1]
series = []  # (label, {c: row})
for arg in sys.argv[2:]:
    label, path = arg.split(":", 1)
    d = json.load(open(path))
    rows = {r["concurrency"]: r for r in d["results"] if r.get("otps")}
    series.append((label, rows))

fig, ax = plt.subplots(figsize=(9, 6))
for i, (label, rows) in enumerate(series):
    cs = sorted(rows)
    xs = [rows[c]["user_tps"] for c in cs]
    ys = [rows[c]["otps"] for c in cs]
    ax.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
            linewidth=2, markersize=7, label=label)
    for c in cs:
        ax.annotate(f"c{c}", (rows[c]["user_tps"], rows[c]["otps"]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7,
                    color=COLORS[i % len(COLORS)])
ax.set_xlabel("per-user output throughput (tokens/s/user)")
ax.set_ylabel("system output throughput (tokens/s)")
ax.set_title("SuperV3-MTP attn-DP — serving pareto (OTPS vs per-user TPS)")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
fig.savefig(out_png, dpi=140)
print(f"saved {out_png}\n")

# Comparison table vs the first series (baseline).
base_label, base = series[0]
all_cs = sorted(set().union(*[set(r) for _, r in series]))
hdr = f"{'c':>4} " + " ".join(f"{lab[:12]+' OTPS':>17}" for lab, _ in series) + \
      "".join(f" | {lab[:10]+' vs '+base_label[:8]+'%':>22}" for lab, _ in series[1:])
print(hdr)
for c in all_cs:
    cells = [f"{c:>4}"]
    for lab, rows in series:
        v = rows.get(c, {}).get("otps")
        cells.append(f"{v:>17.0f}" if v else f"{'NA':>17}")
    b = base.get(c, {}).get("otps")
    for lab, rows in series[1:]:
        v = rows.get(c, {}).get("otps")
        pct = f"{100*(v-b)/b:>+21.1f}%" if (v and b) else f"{'-':>22}"
        cells.append(" | " + pct)
    print("".join(cells))
