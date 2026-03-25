# Triton Kernel Optimization: Code Templates

---

## Running Doc (`<kernel_name>_opt.md`)

```markdown
# <KernelName> Optimization Log

**File:** `path/to/<kernel_name>.py`
**Started:** <date>
**GPU:** <GPU model>
**PyTorch:** <version>  **Triton:** <version>  **Dtype:** <dtype>

---

## Kernel Overview

### `<kernel_func_1>` (primary)

**What it computes:** <one-paragraph mathematical description>

**Signature:**
```python
@triton.jit
def <kernel_func_1>(ptr_input, ptr_output, ..., M, N, BLOCK_M: tl.constexpr, ...):
```

**Grid:** `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`
**Work per program:** `BLOCK_M × BLOCK_N` elements
**Current params:** `num_warps=4, num_stages=2, BLOCK_M=128, BLOCK_N=64`

**Memory access pattern:** <describe stride patterns, coalescing>
**Estimated arithmetic intensity:** ~X FLOP/byte → **memory-bound / compute-bound**
**Surrounding torch ops (out of scope):** <list any ops in the Python launcher before/after the kernel>

### `<kernel_func_2>` (secondary — if present)

<same structure>

---

## Bottleneck Classification

| Kernel | AI (FLOP/byte) | Ridge point (GPU) | Classification |
|--------|---------------|-------------------|----------------|
| `<kernel_func_1>` | X | Y | memory-bound / compute-bound |
| `<kernel_func_2>` | X | Y | ... |

**Primary bottleneck:** <memory bandwidth / compute / launch overhead>
**Theoretical peak (memory-bound):** X µs for shape A1 @ Y GB/s
**Occupancy at baseline:** <register count, warps/SM>

---

## Target Models & Benchmark Shapes

### Models

| ID | Model | Key dims |
|----|-------|----------|
| A | <model_name> | hidden=X, heads=Y, head_dim=Z, ... |
| B | <model_name> | ... |

### Shape Matrix

| ID | Model | Batch | Tokens | <dim1> | <dim2> | Notes |
|----|-------|-------|--------|--------|--------|-------|
| A1 | A | 1 | 1 | ... | ... | decode |
| A2 | A | 1 | 8 | ... | ... | small batch |
| A3 | A | 1 | 128 | ... | ... | prefill |
| B1 | B | 1 | 1 | ... | ... | decode |
| ... | | | | | | |

---

## Current Best Summary

*Updated after every iteration. Shows the best result seen so far for each shape.*

| Shape ID | Best Kernel µs | Best E2E µs | Config | Iter achieved | vs Baseline |
|----------|---------------|-------------|--------|---------------|-------------|
| A1 | — | — | baseline | 0 | — |
| A2 | — | — | baseline | 0 | — |
| ... | | | | | |

---

## Optimization Iterations

### Iteration 0 — Baseline

**Environment:** GPU=<X>, torch=<Y>, triton=<Z>, dtype=<D>

| Shape ID | Kernel µs | E2E µs | Triton % of E2E |
|----------|-----------|--------|-----------------|
| A1 | | | |
| A2 | | | |
| ... | | | |

**Commit:** `<hash>`

---

<!-- Subsequent iterations added here using the iteration entry template below -->

---

## Optimization Ideas Backlog

### A.1 Memory Access
- [ ] Coalesced loads — **Why:** <kernel-specific reason> | **Impact:** High | **Correctness risk:** No
- [ ] Vectorized loads — **Why:** ... | **Impact:** Medium | **Correctness risk:** No
- ...

### A.2 Tiling
- [ ] Larger BLOCK_M — **Why:** ... | **Impact:** Medium | **Correctness risk:** No (if masking correct)
- ...

### A.3 Compute
- ...

### A.4 Fusion
- [ ] Fuse with <adjacent_op> — **Why:** saves one global mem round-trip | **Impact:** High | **Correctness risk:** Yes
- ...

### A.5 Parallelism
- ...

### A.6 Layout
- ...

---

## Final Best Configuration

*Filled at end of Phase 3.*

### Per-shape best configs

| Shape ID | num_warps | num_stages | BLOCK_M | BLOCK_N | ... | Kernel µs | vs Baseline |
|----------|-----------|------------|---------|---------|-----|-----------|-------------|
| A1 | | | | | | | |
| ... | | | | | | | |

### Config selection mechanism

<autotune / manual lookup table / heuristic — with code excerpt>

### Summary speedup

| Shape ID | Baseline µs | Final µs | Speedup |
|----------|------------|----------|---------|
| A1 | | | Xx |
| ... | | | |

---

## Appendix: How to Reproduce

```bash
# Environment
GPU: <GPU>
torch: <version>
triton: <version>

# Run benchmark
cd <kernel_dir>
python sweep_<kernel_name>.py

# Run correctness check
python sweep_<kernel_name>.py --correctness

# Run full parameter sweep
python sweep_<kernel_name>.py --sweep --output results.json

# Run tests
pytest tests/unittest/ -k "<kernel_name>" -v
```
```

---

## Iteration Entry Template

Use this for every iteration section in the running doc.

```markdown
### Iteration N — <short descriptive title>

**Category:** A.X — <category name>
**What changed:**
- <bullet: specific code change>
- <bullet: why this was expected to help>

**Correctness:** PASS / FAIL / SKIP (parameter-only)
*Test:* `python sweep_<kernel_name>.py --correctness` → <result>

**Results:**

| Shape ID | Kernel µs | E2E µs | Δ vs baseline | Δ vs prev best |
|----------|-----------|--------|---------------|----------------|
| A1 | | | | |
| A2 | | | | |
| ... | | | | |

**Outcome:** KEEP / REVERT / PARTIAL-KEEP (shape-conditional)

**Analysis:**
<2–4 sentences: what happened, why, what to try next>

**Commit:** `<hash>` — `[None][perf] <kernel_name> opt iter N: <description>`
```

---

## Benchmark Script

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmark and sweep script for <kernel_name>.

Usage:
  python sweep_<kernel_name>.py                        # Run all shapes, print table
  python sweep_<kernel_name>.py --correctness          # Correctness check vs reference
  python sweep_<kernel_name>.py --num-warps 8          # Override a parameter
  python sweep_<kernel_name>.py --sweep                # Full parameter sweep → JSON
  python sweep_<kernel_name>.py --sweep --output out.json
"""

import argparse
import json
import time
from itertools import product
from typing import Any

import torch
import triton
import triton.testing

# ---- Import the kernel under test ----
# from tensorrt_llm._torch.auto_deploy.custom_ops.<category>.<op_name> import (
#     triton_<kernel_name>_launcher,
#     torch_<kernel_name>,   # reference implementation
# )

DTYPE = torch.float16
DEVICE = "cuda"

# ---- Shape matrix ----
# Each entry: (shape_id, *dims, description)
SHAPES = [
    # fmt: off
    # Model A (e.g., Llama-3-8B)
    ("A1", dict(M=1,   N=4096, K=4096), "decode"),
    ("A2", dict(M=8,   N=4096, K=4096), "small batch"),
    ("A3", dict(M=128, N=4096, K=4096), "prefill"),
    ("A4", dict(M=512, N=4096, K=4096), "long prefill"),
    # Model B (e.g., DeepSeek-V3)
    ("B1", dict(M=1,   N=7168, K=7168), "decode"),
    ("B2", dict(M=8,   N=7168, K=7168), "small batch"),
    ("B3", dict(M=128, N=7168, K=7168), "prefill"),
    # fmt: on
]

# ---- Sweep parameter grid ----
SWEEP_GRID = {
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3, 4, 5],
    "BLOCK_M": [32, 64, 128, 256],
    "BLOCK_N": [32, 64, 128, 256],
}


def make_inputs(shape: dict[str, int], dtype=DTYPE, device=DEVICE) -> dict[str, torch.Tensor]:
    """Create random input tensors for the given shape."""
    M, N, K = shape["M"], shape["N"], shape["K"]
    return {
        "x": torch.randn(M, K, dtype=dtype, device=device),
        # Add more inputs as needed
    }


def run_kernel(inputs: dict[str, torch.Tensor], **kernel_kwargs) -> torch.Tensor:
    """Run the Triton kernel launcher with given inputs and config."""
    # Replace with actual launcher call
    # return triton_<kernel_name>_launcher(inputs["x"], ..., **kernel_kwargs)
    raise NotImplementedError("Replace with actual kernel call")


def run_reference(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run the reference implementation (torch_* op or pure-torch equivalent)."""
    # Replace with actual reference
    # return torch_<kernel_name>(inputs["x"], ...)
    raise NotImplementedError("Replace with actual reference call")


def bench_kernel_only(inputs, warmup=25, rep=100, **kwargs) -> float:
    """Benchmark only the Triton kernel dispatch (not Python overhead). Returns µs."""
    ms = triton.testing.do_bench(
        lambda: run_kernel(inputs, **kwargs), warmup=warmup, rep=rep
    )
    return ms * 1e3  # ms → µs


def bench_e2e(inputs, warmup=25, rep=100, **kwargs) -> float:
    """Benchmark the full Python launcher + kernel. Returns µs."""
    # If the launcher does extra allocations, include them here
    ms = triton.testing.do_bench(
        lambda: run_kernel(inputs, **kwargs), warmup=warmup, rep=rep
    )
    return ms * 1e3


def check_correctness(shapes, default_kwargs) -> bool:
    """Compare kernel output against reference for all shapes. Returns True if all pass."""
    all_pass = True
    print("\n=== Correctness Check ===")
    print(f"{'Shape':>8} {'Max Abs Err':>14} {'Max Rel Err':>14} {'Status':>8}")
    print("-" * 50)
    for shape_id, shape, _ in shapes:
        inputs = make_inputs(shape)
        ref = run_reference(inputs)
        out = run_kernel(inputs, **default_kwargs)
        abs_err = (out - ref).abs().max().item()
        rel_err = ((out - ref).abs() / (ref.abs() + 1e-8)).max().item()
        status = "PASS" if abs_err < 1e-2 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{shape_id:>8} {abs_err:>14.6f} {rel_err:>14.6f} {status:>8}")
    print()
    return all_pass


def run_benchmark(shapes, default_kwargs, label="") -> list[dict]:
    """Run benchmark on all shapes. Returns list of result dicts."""
    results = []
    for shape_id, shape, desc in shapes:
        inputs = make_inputs(shape)
        k_us = bench_kernel_only(inputs, **default_kwargs)
        e2e_us = bench_e2e(inputs, **default_kwargs)
        triton_pct = 100.0 * k_us / e2e_us if e2e_us > 0 else 0.0
        results.append({
            "shape_id": shape_id,
            "shape": shape,
            "desc": desc,
            "kernel_us": k_us,
            "e2e_us": e2e_us,
            "triton_pct": triton_pct,
            "config": default_kwargs,
            "label": label,
        })
    return results


def print_results_table(results: list[dict], baseline: list[dict] | None = None):
    baseline_map = {r["shape_id"]: r["kernel_us"] for r in baseline} if baseline else {}
    header = f"{'ID':>4} {'M':>6} {'N':>6} {'K':>6} {'Kernel µs':>10} {'E2E µs':>10} {'Triton%':>8}"
    if baseline_map:
        header += f" {'vs Base':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        s = r["shape"]
        line = (
            f"{r['shape_id']:>4} {s.get('M', 0):>6} {s.get('N', 0):>6} {s.get('K', 0):>6}"
            f" {r['kernel_us']:>10.1f} {r['e2e_us']:>10.1f} {r['triton_pct']:>7.1f}%"
        )
        if baseline_map and r["shape_id"] in baseline_map:
            base = baseline_map[r["shape_id"]]
            delta_pct = 100.0 * (r["kernel_us"] - base) / base
            sign = "+" if delta_pct >= 0 else ""
            line += f" {sign}{delta_pct:.1f}%"
        print(line)
    print()


def run_sweep(shapes, output_path: str):
    """Run full parameter sweep and save results to JSON."""
    keys = list(SWEEP_GRID.keys())
    combos = list(product(*[SWEEP_GRID[k] for k in keys]))
    total = len(combos) * len(shapes)
    print(f"Sweep: {len(combos)} configs × {len(shapes)} shapes = {total} runs")

    all_results = []
    for i, combo in enumerate(combos):
        kwargs = dict(zip(keys, combo))
        try:
            results = run_benchmark(shapes, kwargs)
            all_results.extend(results)
        except Exception as e:
            # Record failed config (e.g., SMEM exceeded)
            for shape_id, shape, desc in shapes:
                all_results.append({
                    "shape_id": shape_id,
                    "shape": shape,
                    "config": kwargs,
                    "error": str(e),
                })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(combos)} configs done...")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Print best config per shape
    print("\n=== Best config per shape ===")
    shape_ids = {r["shape_id"] for r in all_results if "kernel_us" in r}
    for sid in sorted(shape_ids):
        candidates = [r for r in all_results if r["shape_id"] == sid and "kernel_us" in r]
        best = min(candidates, key=lambda r: r["kernel_us"])
        print(f"  {sid}: {best['kernel_us']:.1f} µs  config={best['config']}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark sweep_<kernel_name>")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--output", default=f"sweep_results_{int(time.time())}.json")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--block-m", type=int, default=128, dest="BLOCK_M")
    parser.add_argument("--block-n", type=int, default=64, dest="BLOCK_N")
    args = parser.parse_args()

    default_kwargs = {
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "BLOCK_M": args.BLOCK_M,
        "BLOCK_N": args.BLOCK_N,
    }

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}  triton: {triton.__version__}  dtype: {DTYPE}")
    print(f"Config: {default_kwargs}\n")

    if args.correctness:
        ok = check_correctness(SHAPES, default_kwargs)
        exit(0 if ok else 1)

    if args.sweep:
        run_sweep(SHAPES, args.output)
        return

    results = run_benchmark(SHAPES, default_kwargs)
    print_results_table(results)


if __name__ == "__main__":
    main()
```

---

## Lookup Table

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class KernelConfig:
    num_warps: int
    num_stages: int
    BLOCK_M: int
    BLOCK_N: int
    # Add more fields as needed


# Per-shape best configs derived from sweep results.
# Keys are tuples of the 1-2 dimensions that actually predict config divergence.
# Run `python sweep_<kernel_name>.py --sweep` to regenerate.
_KERNEL_CONFIGS: dict[tuple, KernelConfig] = {
    # Small shapes: low token count (T <= 8)
    "small": KernelConfig(num_warps=2, num_stages=2, BLOCK_M=32, BLOCK_N=64),
    # Medium shapes: T in [32, 256]
    "medium": KernelConfig(num_warps=4, num_stages=3, BLOCK_M=128, BLOCK_N=64),
    # Large shapes: T >= 512
    "large": KernelConfig(num_warps=8, num_stages=4, BLOCK_M=128, BLOCK_N=128),
    # Fallback
    "default": KernelConfig(num_warps=4, num_stages=3, BLOCK_M=128, BLOCK_N=64),
}


def _get_kernel_config(M: int, N: int) -> KernelConfig:
    """Select best kernel config based on problem dimensions.

    Keyed by total element count (a proxy for which config regime we're in).
    Adjust the thresholds based on your sweep results.
    """
    total = M * N
    if total <= 8 * 4096:
        return _KERNEL_CONFIGS["small"]
    elif total <= 256 * 4096:
        return _KERNEL_CONFIGS["medium"]
    else:
        return _KERNEL_CONFIGS["large"]
```

---

## Config Selection — `@triton.autotune`

Use when compile-time autotuning overhead is acceptable (offline / batch paths, not serving hot paths).

```python
import triton

# Restrict to the top-N configs found in the sweep — do NOT use the full sweep grid.
# Fewer configs = faster first-call compile time.
_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=2, num_stages=2),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64},  num_warps=8, num_stages=4),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["M", "N"],  # dimensions that determine which config is best
)
@triton.jit
def my_kernel(
    ptr_a, ptr_b, ptr_c,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    ...
```

---

## Config Selection — Manual Launcher Integration

For serving hot paths where the first-call autotune overhead is unacceptable.

```python
def my_op_launcher(
    x: torch.Tensor,
    weight: torch.Tensor,
    # ...
) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    output = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # Get best config for this problem size
    cfg = _get_kernel_config(M, N)

    grid = (triton.cdiv(M, cfg.BLOCK_M), triton.cdiv(N, cfg.BLOCK_N))
    my_kernel[grid](
        x, weight, output,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        output.stride(0), output.stride(1),
        BLOCK_M=cfg.BLOCK_M,
        BLOCK_N=cfg.BLOCK_N,
        BLOCK_K=64,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return output
```
