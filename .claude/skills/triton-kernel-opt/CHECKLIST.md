# Triton Kernel Optimization: Per-Phase Checklists

---

## Phase 0 — Setup Checklist

### Kernel Understanding
- [ ] Read the full kernel file; every `@triton.jit` function documented.
- [ ] Arithmetic intensity estimated (FLOP/byte); kernel classified as memory-bound / compute-bound / launch-overhead-bound.
- [ ] Surrounding torch ops (out-of-scope) identified and listed.
- [ ] All callers found via Grep; all models that hit this path identified.
- [ ] Real HuggingFace configs fetched; exact dimensions extracted.
- [ ] Shape matrix covers decode (T=1), small batch, and prefill regimes.

### Running Doc
- [ ] `<kernel_name>_opt.md` created with all required sections.
- [ ] Bottleneck classification section filled.
- [ ] Shape matrix with IDs (A1, A2 ... B1, B2 ...) documented.
- [ ] Optimization backlog populated and prioritized (not just copied from catalog).

### Benchmark Script
- [ ] `sweep_<kernel_name>.py` created with all required modes.
- [ ] `--correctness` mode runs and produces a clear PASS/FAIL per shape.
- [ ] `--sweep` mode runs and emits a JSON file.
- [ ] Kernel-only timing uses `triton.testing.do_bench`, not `time.time()`.
- [ ] All shapes from the shape matrix are included.
- [ ] CLI overrides for `num_warps`, `num_stages`, `BLOCK_*` work.

### Baseline Commit
- [ ] On a dedicated git branch (not `main`).
- [ ] Correctness verified against reference at baseline.
- [ ] Baseline table in the running doc filled with actual measurements.
- [ ] GPU, torch, triton versions recorded.
- [ ] Baseline commit made with `git commit -s`.

---

## Phase 1 — Parameter Sweep Checklist

- [ ] Correctness verified once at the start (before any config changes).
- [ ] Sweep grid defined (num_warps, num_stages, BLOCK_*); obviously bad combos pruned.
- [ ] Sweep run successfully; results in JSON.
- [ ] Best config per shape identified and documented in the running doc.
- [ ] Key dimensions that predict config divergence identified (the lookup key).
- [ ] Lookup table or autotune config list implemented in the launcher.
- [ ] Config selection tested: correct config is chosen at runtime for each shape.
- [ ] Correctness re-verified after lookup table integration (BLOCK_* changes affect masking).
- [ ] Running doc "Current Best Summary" updated.
- [ ] Commits made: one per kernel's sweep + one for lookup table integration.
- [ ] Pre-commit hooks ran; files re-staged if modified.

---

## Phase 2 — Structural Optimization Checklist

Per iteration:
- [ ] One focused change per commit (not multiple independent changes bundled together).
- [ ] Correctness checked after every structural change (not just parameter changes).
- [ ] Benchmark run on all shapes (not a subset).
- [ ] Running doc updated (iteration entry + Current Best Summary table) in the same commit.
- [ ] Commit made with `git commit -s`, even if the change was reverted.
- [ ] Iteration count stated explicitly after the commit ("Iteration N complete. Remaining: M").

Overall:
- [ ] Ideas attempted from at least 4 different categories (A.1 through A.6).
- [ ] When backlog is exhausted, "when backlog runs dry" protocol followed before stopping.
- [ ] Minimum 50 committed iterations reached (or user-specified target).
- [ ] Last 5 ideas showed < 1% change on all shapes before stopping.

---

## Phase 3 — Finalization Checklist

- [ ] Best structural changes identified from the running doc history.
- [ ] Composition plan documented: which changes are being combined and why.
- [ ] Combined kernel benchmarked; improvements confirmed to compose (not cancel).
- [ ] Final config selection mechanism chosen (lookup table / autotune / heuristic) and justified.
- [ ] Correctness suite passes fully:
  - [ ] `python sweep_<kernel_name>.py --correctness`
  - [ ] `pytest tests/unittest/ -k "<kernel_name>" -v`
- [ ] "Final Best Configuration" section in running doc filled.
- [ ] Per-shape speedup table in running doc filled.
- [ ] Final commit made: `[None][perf] <kernel_name> opt final: ...`
- [ ] No dead code or commented-out experiments left in the kernel file.
- [ ] NVIDIA copyright year updated on all modified files.
- [ ] `pre-commit` ran cleanly on final commit.

---

## Common Mistakes to Avoid

### Benchmarking mistakes
- **Using `time.time()` for GPU timing** — async execution means this measures dispatch time, not execution time. Always use `triton.testing.do_bench`.
- **Benchmarking without warmup** — first run includes JIT compilation time. Always use `warmup ≥ 25`.
- **Only timing one shape** — report all shapes; different shapes may have opposite trends.
- **Comparing stale results** — always re-run baseline in the same session before comparing (GPU clock states, thermal throttling, memory pressure differ between sessions).

### Optimization mistakes
- **Treating num_warps/num_stages as independent of BLOCK_**** — they interact. When changing BLOCK sizes, re-sweep warps and stages.
- **Ignoring SMEM limits** — `BLOCK_M × BLOCK_N × dtype_bytes × num_stages` must fit in shared memory (~96–228KB depending on GPU). If SMEM > limit, the kernel silently falls back to reduced occupancy or fails.
- **Fusing when the bottleneck is compute** — fusion mainly helps memory-bound kernels. For compute-bound kernels, fusion may hurt by increasing register pressure.
- **Forgetting to re-check correctness after `BLOCK_*` changes** — masking logic like `mask = offs < N` depends on BLOCK size and can be subtly wrong at boundaries.
- **Setting `num_stages > num_loop_iterations`** — if the inner loop has fewer iterations than `num_stages`, the pipeline is wasted.

### Commit mistakes
- **Bundling multiple experiments into one commit** — each iteration must be a separate commit for the history to be useful.
- **Skipping commits for failed experiments** — failed experiments are data. Commit them with `[FAILED]` or `[REVERT]` in the message.
- **Forgetting `git commit -s`** — DCO sign-off is required.
- **Not updating the running doc in the same commit as the code** — the doc and code must stay in sync.
