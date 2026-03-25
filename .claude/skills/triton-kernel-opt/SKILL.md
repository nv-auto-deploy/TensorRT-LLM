---
name: triton-kernel-opt
description: Iterative Triton kernel performance optimization loop. Given a Triton kernel file, systematically benchmarks, optimizes, and git-commits every iteration with a running doc. Produces a per-shape lookup table of best configs and structurally improved kernels.
---

# Triton Kernel Optimization Loop

**Input:** Path to a Triton kernel file (containing one or more `@triton.jit` kernels) + optional list of target models/shapes.
**Output:** Optimized kernel(s) + per-shape config lookup table + running optimization doc + **50+ git-committed iterations** (parameter sweeps + structural changes).

> **Before starting:** Check whether a `CLAUDE.local.md` or project-specific instructions override anything here. Read `CODING_GUIDELINES.md` if you haven't already.

---

## Decision Tree

```
Does the file contain multiple @triton.jit kernels?
  YES → Profile first to find the dominant kernel, optimize in order of wall-clock impact.
  NO  → Single kernel path.

Is the kernel already equipped with @triton.autotune?
  YES → Phase 1 is still useful — expand the autotune config space beyond defaults.
  NO  → Full Phase 1 sweep needed.

Is this a serving hot path?
  YES → Prefer manual lookup table over @triton.autotune (compile-time overhead is paid every cold start).
  NO  → @triton.autotune is acceptable.
```

---

## Phase 0 — Setup & Research

### Step 0.1 — Read the kernel

1. Read the kernel file end-to-end (`Read` tool).
2. For each `@triton.jit` kernel, document in the running doc:
   - **What it computes** — mathematical operation, input/output tensors, grid dimensions, work per program.
   - **Current launch parameters** — `num_warps`, `num_stages`, `BLOCK_*` sizes, any `tl.constexpr` flags.
   - **Tiling strategy** — how the problem is split across programs (1D/2D/3D grid, how M/N/K map to blocks).
   - **Memory access pattern** — are loads/stores coalesced? stride patterns? any atomic ops?
   - **Arithmetic intensity** (estimated) — FLOP count per byte of memory traffic. This determines whether the kernel is **memory-bound** or **compute-bound** and dictates which optimization strategies matter most.
3. For each Python launcher, document:
   - Input/output allocations.
   - Surrounding torch ops (these are out of scope for Triton optimization but may be fusion candidates).
   - Grid formula.

> **Key classification:** Before any benchmarking, classify each kernel as memory-bound, compute-bound, or launch-overhead-bound. This shapes which optimization categories to prioritize. See PATTERNS.md#bottleneck-classification.

### Step 0.2 — Identify callers and target shapes

1. `Grep` for the kernel launcher and custom op name across the repo to find all callers.
2. Trace through model files, pattern matchers, and transform libraries to find which models hit this path.
3. For each model found, extract exact dimensions from the HuggingFace config:
   - Use `WebFetch` on `https://huggingface.co/<org>/<model>/blob/main/config.json`.
   - Identify the dimensions that feed this kernel (e.g., `hidden_size`, `num_attention_heads`, `head_dim`, `intermediate_size`, etc.).
4. Build a **benchmark shape matrix** covering:
   - All identified models (one group per model, label A, B, C ...).
   - Multiple token counts / batch sizes per model representing real inference scenarios:
     - Decode: T=1, T=4
     - Small batch: T=8, T=32
     - Prefill: T=128, T=512, T=2048 (include only shapes realistic for this kernel)
   - Assign each config a short ID (A1, A2 ... B1, B2 ...) for table references.
5. If no models are found in the codebase, use representative synthetic shapes covering small/medium/large regimes.

### Step 0.3 — Characterize the baseline bottleneck

Before optimizing, understand *what* is bottlenecking the kernel. Run a quick profiling pass:

1. **Roofline position** — Using the arithmetic intensity from Step 0.1 and the GPU's spec sheet (memory bandwidth, peak TFLOPS), estimate where the kernel sits relative to the roofline. This sets the theoretical performance ceiling.
2. **Occupancy estimate** — Use `triton.compiler.CompiledKernel` or `ncu` if available to check register usage and occupancy. Low occupancy (< 25%) often indicates register pressure or excessive shared memory use.
3. **Launch overhead test** — For very small shapes, launch overhead may dominate. Time the Python-side grid computation + kernel dispatch alone (with an empty kernel body) vs. full kernel.

Document these findings in the running doc's "Baseline Bottleneck" section. Use them throughout the optimization loop to sanity-check whether each idea targets the actual bottleneck.

### Step 0.4 — Create the running doc

Create `<kernel_dir>/<kernel_name>_opt.md`. Use the template in TEMPLATES.md#running-doc. It must contain:

1. **Kernel Overview** — mathematical description, scope notes.
2. **Bottleneck Classification** — memory-bound / compute-bound / launch-overhead-bound + roofline estimate.
3. **Target Models & Benchmark Shapes** — model table + shape matrix with IDs.
4. **Current Best Summary** — updated every iteration; shows best latency per shape ID.
5. **Optimization Iterations** — one subsection per iteration using the template in TEMPLATES.md#iteration-entry.
6. **Optimization Ideas Backlog** — categorized checklist, initialized in Step 0.5.
7. **Final Best Configuration** — filled at end.
8. **Appendix: Reproduce** — GPU name, PyTorch version, Triton version, dtype, benchmark command.

### Step 0.5 — Build the optimization backlog

Using the bottleneck classification and the catalog in PATTERNS.md#optimization-catalog, build a prioritized, kernel-specific checklist of ideas. For each idea record:
- **Why it might help** for *this* kernel specifically.
- **Expected impact**: High / Medium / Low.
- **Which shapes benefit most** (small, large, all).
- **Correctness risk**: Yes (requires testing) / No (parameter-only).

Organize by category (Memory Access, Tiling, Compute, Fusion, Parallelism). Put the highest expected-impact ideas first within each category.

> **Do NOT just copy the catalog verbatim.** Drop ideas that clearly don't apply to this kernel and add kernel-specific ideas not in the catalog.

### Step 0.6 — Write the benchmark script

Create `<kernel_dir>/sweep_<kernel_name>.py` using TEMPLATES.md#benchmark-script as the base. The script must:

1. Benchmark **each `@triton.jit` kernel in isolation** (kernel-only timing, excluding Python overhead).
2. Benchmark the **full end-to-end launcher** (Python + kernel) to show Triton's share of total cost.
3. Test all shapes from the shape matrix.
4. Use `triton.testing.do_bench(fn, warmup=25, rep=100)` (returns ms; convert to µs for display).
5. Print results as a table: shape ID | key dims | per-kernel µs | total Triton µs | E2E µs | Triton % of E2E.
6. Accept CLI flags to override kernel parameters: `--num-warps`, `--num-stages`, and any `BLOCK_*` sizes.
7. Support `--sweep` mode: iterate over a parameter grid (defined in the script), run all shapes, emit `sweep_results_<timestamp>.json` with full results.
8. Support `--correctness` mode: compare kernel output against a reference (e.g., the `torch_*` op or a pure-torch equivalent) across all shapes and report max absolute/relative error.

### Step 0.7 — Branch and baseline commit

1. If not already on a feature branch:
   ```bash
   git checkout -b <username>/<kernel_name>_opt
   ```
2. Run the benchmark script in `--correctness` mode. Record baseline correctness.
3. Run the benchmark script normally. Fill in the baseline table in the running doc.
4. Record environment: GPU model, `torch.__version__`, `triton.__version__`, dtype.
5. Commit:
   ```
   [None][perf] <kernel_name> opt iter 0: baseline + benchmark script
   ```
   Use `git commit -s`. Run pre-commit; re-stage if hooks modify files.

---

## Phase 1 — Parameter Sweep

**Goal:** Find the best `(num_warps, num_stages, BLOCK_*)` per shape for each kernel. Different shapes often have different optimal configs — never assume one-size-fits-all.

### Step 1.1 — Correctness baseline

Before sweeping, verify correctness once with `--correctness` mode. If any shape fails, **stop and fix** before sweeping — parameter changes can mask bugs.

### Step 1.2 — Design the sweep space

For each kernel, define the sweep grid:

| Parameter | Values to try |
|-----------|---------------|
| `num_warps` | 1, 2, 4, 8, 16 |
| `num_stages` | 1, 2, 3, 4, 5 |
| `BLOCK_M` (if present) | Powers of 2 from max(16, dim/8) to min(512, dim*2) |
| `BLOCK_N` (if present) | Same range |
| `BLOCK_K` (if present) | 16, 32, 64, 128 |

Prune obviously bad combos:
- `num_warps=16` is rarely useful for blocks < 64 elements.
- `num_stages=5` with `BLOCK_K=128` may exceed shared memory on older GPUs — skip.
- Any combo where `BLOCK_* > problem_dim` (wastes threads on masking).

Estimate total combinations. If > 500 per shape, further prune or use a coarser grid.

### Step 1.3 — Run the sweep

```bash
python sweep_<kernel_name>.py --sweep --output sweep_results.json
```

Or run manually in a loop if `--sweep` mode is not yet implemented.

### Step 1.4 — Analyze results and build the lookup table

1. For each shape ID, find the `(num_warps, num_stages, BLOCK_*)` combo with lowest kernel latency.
2. Group shapes that share the same optimal config.
3. Identify the **key dimensions** that predict which config wins (often just 1–2 dimensions like total token count or M*N product). These become the lookup keys.
4. Build `KERNEL_CONFIGS` dict (see TEMPLATES.md#lookup-table).
5. Document the "winner" table in the running doc (one row per shape ID, columns: shape dims | best config | latency | vs current defaults).

### Step 1.5 — Integrate the lookup table

In the kernel launcher, replace hard-coded `num_warps=X` etc. with a call to a `_get_kernel_config(...)` function. Use the mechanism appropriate for the hot-path constraints:

- **Serving hot path** → manual lookup table function (zero overhead).
- **Offline / batch path** → `@triton.autotune` with the top-N configs as candidates (reduce from full sweep to ≤ 8 configs to keep compile time sane).

### Step 1.6 — Commit each kernel's sweep results separately

One commit per kernel + one commit for the lookup table integration. Example messages:
```
[None][perf] <kernel_name> opt iter N: sweep <kernel_func_name>, best configs found
[None][perf] <kernel_name> opt iter N+1: integrate per-shape lookup table
```

Update the running doc in the same commit as the code.

---

## Phase 2 — Structural Optimization (Main Loop)

**Goal:** Try fundamentally different kernel implementations. Parameter tuning (Phase 1) optimizes *around* a structure; structural changes replace the structure itself.

**Minimum committed iterations: 50** (Phase 1 + Phase 2 combined). Do not stop early.

### Step 2.1 — Prioritize the backlog

Sort the backlog by:
1. **Expected impact × ease** — high-impact, low-effort ideas first.
2. **Bottleneck alignment** — prefer ideas that directly attack the identified bottleneck.
3. **Independence** — try orthogonal ideas early so results are interpretable before combining.

### Step 2.2 — Iteration loop

**Repeat until stopping criteria are met (Step 2.5):**

For each idea from the backlog:

1. **Implement** the change. Keep it focused — one structural idea per iteration.
2. **Test correctness:**
   ```bash
   python sweep_<kernel_name>.py --correctness
   ```
   Or run the existing test suite:
   ```bash
   pytest tests/unittest/ -k "<kernel_name>" -x
   ```
   If no tests exist, use the benchmark script's `--correctness` mode as the gate.
3. **Benchmark** (only if correctness passes). Run all shapes.
4. **Update running doc** with the iteration entry (use TEMPLATES.md#iteration-entry).
5. **Git commit** — commit even if the idea failed or broke correctness. Mark it clearly.
   ```
   [None][perf] <kernel_name> opt iter N: <one-line description> [REVERT/FAILED/+X%]
   ```
6. **Decide:**
   - **Helped all shapes** → keep; layer next idea on top.
   - **Helped some shapes, hurt others** → analyze why; consider shape-conditional dispatch; keep if net-positive.
   - **No measurable change** → revert code; document in running doc; note "no effect".
   - **Broke correctness** → revert code; document root cause; check whether the idea is salvageable with a fix.
   - **Partially correct** → investigate precision issue; may be acceptable (e.g., ±1 ULP) or require accumulator upcast.

> **After each iteration, explicitly state:** "Iteration N complete. Minimum target: 50. Remaining: 50-N iterations needed."

### Step 2.3 — When the backlog runs dry

If you have exhausted the backlog but haven't hit the minimum target, proceed in order:

1. **Combine orthogonal wins** — layer two independently-validated improvements. Re-verify they compose correctly.
2. **Tune structural changes** — if a structural change helped, sweep its new parameters (e.g., new BLOCK sizes created by a tiling change).
3. **Profile the new kernel** — the bottleneck may have shifted. Repeat the roofline analysis on the current code.
4. **Try the opposite extreme** — if larger tiles helped, try even larger; if more stages helped, try fewer.
5. **Revisit failed ideas** — an idea that didn't help on the original code may work on the current (structurally different) code.
6. **Consult Appendix A** (PATTERNS.md#optimization-catalog) for any category you haven't explored.
7. Generate new ideas from the new kernel structure and add them to the backlog.

### Step 2.4 — Multi-kernel files

If the file has multiple kernels:
1. Profile all shapes to determine which kernel dominates per shape. A kernel may dominate for large shapes but be irrelevant for small ones.
2. Optimize the dominant kernel for your primary use case first.
3. After 20+ iterations on kernel 1, re-profile — the bottleneck may have shifted to kernel 2.
4. Optimize kernel 2.
5. Check for **cross-kernel opportunities**: changing data layout between the two kernels, or fusing them entirely.

### Step 2.5 — Stopping criteria

Stop when **all** of these hold:
- **≥ 50 committed iterations** (or the user-specified target).
- **Last 5 ideas produced < 1% change** on all shapes.
- **Ideas from ≥ 4 different categories** have been tried (see PATTERNS.md#optimization-catalog).
- **The running doc backlog has no unchecked ideas** with plausible benefit for the current code.

If only some shapes have plateaued, continue optimizing for the remaining shapes.

---

## Phase 3 — Combine & Finalize

### Step 3.1 — Cherry-pick and compose best changes

Review the running doc's history. Identify:
- The subset of structural changes that each helped independently.
- The best config per shape from Phase 1.
- Build a composition plan. Note that improvements don't always compose additively — verify each combination actually produces the expected combined speedup.

### Step 3.2 — Implement config selection

Choose the appropriate mechanism (see TEMPLATES.md#config-selection for code):

| Hot-path type | Mechanism | Trade-off |
|---------------|-----------|-----------|
| Serving inference | Manual lookup table (`_get_kernel_config`) | Zero overhead; needs manual maintenance |
| Offline / batch | `@triton.autotune` with 4–8 top configs | Auto-tunes per shape; adds first-call overhead |
| Smooth variation | Heuristic function | Simple; may miss narrow optima |

For `@triton.autotune`, limit candidates to the top-N from the Phase 1 sweep (not the full grid) to keep compile time reasonable.

### Step 3.3 — Final validation

1. Run full correctness suite:
   ```bash
   python sweep_<kernel_name>.py --correctness
   pytest tests/unittest/ -k "<kernel_name>" -v
   ```
2. Run benchmark on all shapes. Fill "Final Best Configuration" in the running doc.
3. Compute total speedup vs. baseline per shape. Summarize as a table.
4. Check: does the final kernel still make sense? No dead code, no regression on any shape vs. Phase 1 best.

### Step 3.4 — Final commit

```
[None][perf] <kernel_name> opt final: iter N total, X-Y% improvement across shapes
```

---

## Rules & Conventions

### Git commits
- **Every iteration gets its own commit**, including failed/reverted experiments — the history is part of the output.
- Format: `[None][perf] <kernel_name> opt iter <N>: <description> [optional: REVERT/FAILED/+X%]`
- `git commit -s` (DCO sign-off). Do not add co-authors.
- Run pre-commit hooks; re-stage if files are modified.

### Running doc
- Update the running doc **in the same commit** as the code change, every iteration.
- Keep the "Current Best Summary" table at the top of the iterations section up to date — it should always show the best result seen *so far* per shape.
- The doc is the audit trail; make it legible to someone reading it without the git history.

### Iteration counting
- Parameter sweeps count: each distinct `(num_warps, num_stages, BLOCK_*)` configuration tested and committed is one iteration.
- Structural changes count: each code change to the kernel body or launcher is one iteration.
- Combined "sweep then integrate" counts as 2 iterations.

### Correctness
- Structural changes: always run correctness check before benchmarking.
- Parameter-only sweeps (`num_warps`, `num_stages` on unchanged kernel): verify once at Phase 1 start, then skip per-combo checks.
- `BLOCK_*` size changes: require correctness check (masking logic may change).

### Benchmarking
- `triton.testing.do_bench(fn, warmup=25, rep=100)` — returns ms, convert to µs.
- Always record: GPU model, `torch.__version__`, `triton.__version__`, dtype, batch config.
- For kernel-isolation timing, wrap only the `triton_kernel[grid](...)` call, not the Python launcher.
- Do not rely on wall-clock `time.time()` for GPU kernels — it does not account for async execution.

### When configs diverge by shape
- Do NOT force one config for all shapes.
- Build a lookup table keyed on the 1–2 dimensions that actually predict which config wins.
- Implement selection in the Python launcher, not in the Triton kernel itself (avoid runtime branching inside `@triton.jit`).

---

## Supporting Files

- **TEMPLATES.md** — Code templates: running doc, benchmark script, iteration entry, lookup table, config selection.
- **PATTERNS.md** — Bottleneck classification guide + optimization catalog with kernel-specific examples.
- **CHECKLIST.md** — Per-phase validation checklists.
