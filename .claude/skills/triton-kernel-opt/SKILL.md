---
name: triton-kernel-opt
description: Iterative Triton kernel performance optimization loop. Given a Triton kernel, systematically benchmarks, optimizes, and logs every iteration via git commits and a running doc. Produces a per-shape lookup table of best configs and structurally improved kernels.
---

# Triton Kernel Optimization Loop

**Input:** Path to a Triton kernel file + list of target models/shapes.
**Output:** Optimized kernel(s) + per-shape config lookup table + running doc with full history + 100+ git-committed iterations.

---

## Phase 0 — Setup & Research

### Step 0.1 — Understand the kernel

1. Read the kernel file end-to-end.
2. Document in the running doc:
   - What each kernel does (inputs, outputs, grid, work per program).
   - What the surrounding launcher code does (Python-side allocations, torch ops, etc.).
   - What is in scope for optimization (Triton kernels) vs out of scope (e.g., torch.bmm).

### Step 0.2 — Identify target models and shapes

1. Search the codebase for all callers of the kernel / custom op:
   - `Grep` for the op name across the repo.
   - Trace through pattern matchers, transforms, and model files to find which models hit this code path.
2. For each model found, look up the **real HuggingFace config** to get exact dimensions:
   - `hidden_size`, `intermediate_size`, `num_local_experts`, `num_experts_per_tok`, etc.
   - Use `WebFetch` on `https://huggingface.co/<org>/<model>/blob/main/config.json` if needed.
3. Build a **benchmark shape matrix** covering:
   - All identified models.
   - Multiple token counts representing real scenarios: T=1 (single-token decode), T=8 (small batch), T=32 (medium), T=128 (large batch), T=512 (prefill).
   - Assign each config a short ID (e.g., A1, A2, ... B1, B2, ...).

### Step 0.3 — Create the running doc

Create `<kernel_dir>/<kernel_name>_opt.md` with these sections:

1. **Kernel Overview** — what each kernel does, scope of optimization.
2. **Target Models & Benchmark Shapes** — model table + shape matrix.
3. **Optimization Iterations** — baseline + per-iteration tables (initially empty).
4. **Optimization Ideas Backlog** — checklist of ideas to try.
5. **Final Best Configuration** — to be filled at end.
6. **Appendix: How to Reproduce** — benchmark command.

### Step 0.4 — Write the benchmark script

Create `<kernel_dir>/bench_<kernel_name>.py` that:

1. Benchmarks **each Triton kernel in isolation** (not just end-to-end).
2. Also benchmarks **end-to-end** (including non-Triton ops like torch.bmm) to show Triton's share of total cost.
3. Tests all shapes from the matrix.
4. Uses `triton.testing.do_bench` with `warmup=25, rep=100` (ms) for reliable timing.
5. Prints results as a formatted table with: shape ID, dimensions, per-kernel latency (us), total Triton latency, E2E latency, Triton % of E2E.
6. Accepts optional CLI args to override kernel parameters (num_warps, num_stages, BLOCK sizes) for sweep iterations.

### Step 0.5 — Create a new git branch

```bash
git checkout -b <user>/<kernel_name>_opt
```

### Step 0.6 — Collect baseline

1. Run the benchmark script.
2. Fill in the baseline table in the running doc.
3. Record environment info: GPU name, PyTorch version, Triton version, dtype.
4. Commit: `[None][perf] <kernel_name> opt iter 0: baseline measurements`

---

## Phase 1 — Parameter Sweep (num_warps, num_stages, BLOCK sizes)

**Goal:** Find the best (num_warps, num_stages) per shape for each kernel. This is NOT one-size-fits-all — different shapes may have different optimal configs.

### Step 1.1 — Design the sweep

For each kernel, sweep over:
- `num_warps` in {1, 2, 4, 8, 16}
- `num_stages` in {1, 2, 3, 4, 5}
- Optionally: `BLOCK_*` sizes (powers of 2 near the dimension, or multiples)

That's 25 combos per kernel. For 2 kernels = 50 combos.

### Step 1.2 — Run the sweep

Write a sweep script or extend the benchmark script to accept `--num_warps` and `--num_stages` overrides. For each combo:

1. Modify the kernel launch parameters (or pass them as args).
2. Run the benchmark for all shapes.
3. Record results.

**Important:** Run correctness tests after each structural change. For pure parameter sweeps (num_warps/num_stages), correctness is preserved so test once at the start.

### Step 1.3 — Build the lookup table

From sweep results, build a per-shape config:

```python
KERNEL_CONFIGS = {
    # (E_range, T_range, H, I): (num_warps, num_stages, BLOCK_SIZE)
    "default": (4, 3, None),
    (128, 1, 2880, 2880): (8, 2, 4096),
    (128, 512, 2880, 2880): (4, 4, 4096),
    ...
}
```

### Step 1.4 — Commit each meaningful change

For the sweep phase, group commits logically:
- One commit per kernel's full sweep + best-config selection.
- One commit for integrating the lookup table into the kernel launcher.
- Update the running doc in each commit.

Commit message format:
```
[None][perf] <kernel_name> opt iter N: <short description>
```

---

## Phase 2 — Structural Optimization (Kernel 1: Activation/Fusion)

**Goal:** Try fundamentally different kernel implementations, not just parameter tuning.

### Ideas to try (iterate through these, one commit per change):

**Memory access patterns:**
- [ ] Coalesced loads: load contiguous chunk then deinterleave in registers (vs current stride-2 access).
- [ ] Vectorized loads: use `tl.load` with explicit vector widths (e.g., load float4).
- [ ] Prefetch / software pipelining via `num_stages`.

**Tiling strategies:**
- [ ] 2D tiling: split I dimension into multiple blocks per row (grid = (E*T, num_I_blocks)).
- [ ] Different BLOCK_I sizes: try smaller blocks (512, 1024, 2048) instead of next_power_of_2.
- [ ] Multiple elements per thread.

**Compute optimizations:**
- [ ] Fast sigmoid approximation (if accuracy allows).
- [ ] Fuse the `.contiguous()` call (avoid the copy before kernel launch).
- [ ] Mixed precision: compute in fp16 where safe, only upcast for sigmoid.

**Entirely new implementations:**
- [ ] Rewrite as a 2D kernel (rows × columns).
- [ ] Non-interleaved layout: reshape gate_up to separate gate and up tensors before kernel.
- [ ] Persistent kernel: multiple rows per program.

### Per-iteration workflow

For each idea:
1. Implement the change.
2. Run correctness tests: `pytest tests/unittest/auto_deploy/singlegpu/custom_ops/moe/test_triton_moe_dense_mlp.py -x`
3. If correct, run benchmark.
4. Update running doc with: iteration number, what changed, results, delta vs baseline.
5. Git commit.
6. If incorrect, fix or revert, still commit with a note that it failed correctness.

---

## Phase 3 — Structural Optimization (Kernel 2: Weighted Sum)

**Goal:** The weighted sum kernel is often the bigger bottleneck at low token counts. Try new parallelization strategies.

### Ideas to try:

**Parallelism:**
- [ ] 2D grid: `(T, ceil(H/BLOCK_H))` — split H across multiple programs, then reduce.
- [ ] Parallelize over experts: instead of sequential loop, launch E programs per token and atomicAdd or tree-reduce.
- [ ] Hybrid: parallelize over both tokens and H-blocks.

**Memory access:**
- [ ] Reorder expert_out to `[T, E, H]` to improve locality per token.
- [ ] Vectorized loads for expert output rows.
- [ ] Prefetch next expert's data while computing current one.

**Loop optimizations:**
- [ ] Unroll expert loop for known E values (e.g., E=32, E=128).
- [ ] Process multiple experts per iteration (e.g., load 2 or 4 experts at once).
- [ ] Fuse routing weight broadcast with expert output load.

**Entirely new implementations:**
- [ ] Reduce-style kernel: treat as a reduction over E dimension.
- [ ] Use shared memory for routing weights (small: T×E scalars).
- [ ] Persistent kernel processing multiple tokens.

---

## Phase 4 — Combine & Finalize

### Step 4.1 — Cherry-pick best changes

After 100+ iterations, review the running doc to identify:
- Best parameter configs per shape (from Phase 1).
- Best structural changes (from Phases 2-3).
- Combine them into the final kernel.

### Step 4.2 — Implement @triton.autotune (if applicable)

If multiple configs are needed for different shapes, consider using `@triton.autotune`:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_I": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_I": 4096}, num_warps=8, num_stages=3),
        ...
    ],
    key=["I_SIZE"],
)
```

Or implement a manual lookup table in the launcher if autotune overhead is unacceptable.

### Step 4.3 — Final validation

1. Run full correctness test suite.
2. Run benchmark on all shapes.
3. Fill in "Final Best Configuration" in the running doc.
4. Summarize total speedup vs baseline per shape.

### Step 4.4 — Final commit

```
[None][perf] <kernel_name> opt final: best config after N iterations, X-Y% improvement
```

---

## Rules & Conventions

### Git commits
- **Every code change gets a commit.** Even failed experiments (mark them as such).
- Commit message format: `[None][perf] <kernel_name> opt iter <N>: <description>`
- Use `git commit -s` (DCO sign-off required).
- Run `pre-commit` hooks; re-stage if files are modified.

### Running doc updates
- Update the running doc **in the same commit** as the code change.
- Each iteration gets a row in the results table + a brief description of what changed.
- At the top of section 3, maintain a **summary table** showing best result so far per shape.

### Correctness
- Run correctness tests after **every structural change** (new kernel implementation, different tiling, etc.).
- For pure parameter sweeps (num_warps, num_stages), verify correctness once at the start — these don't change the algorithm.
- If a change breaks correctness, note it in the running doc and revert. Still commit with a note.

### Benchmark reproducibility
- Always record: GPU, PyTorch version, Triton version, dtype.
- Use `triton.testing.do_bench(warmup=25, rep=100)` for consistency.
- All benchmark scripts live alongside the kernel file for easy reproduction.

### When multiple configs are optimal
- Do NOT force a single config for all shapes.
- Build a lookup table keyed by shape characteristics (E, T, H, I ranges).
- Implement the lookup in the Python launcher, not in the kernel itself.

---

## Template: Running Doc Iteration Entry

```markdown
### Iteration N — <short title>

**What changed:**
- <bullet points describing the code change>

**Correctness:** PASS / FAIL (test name)

| ID | K1 (us) | K2 (us) | Total (us) | Delta vs baseline | Delta vs prev best |
|----|---------|---------|------------|-------------------|-------------------|
| A1 | ... | ... | ... | ... | ... |
...

**Analysis:** <what worked, what didn't, next idea>

**Commit:** `<short hash>`
```
