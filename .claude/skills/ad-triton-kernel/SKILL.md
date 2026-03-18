---
name: ad-triton-kernel
description: Write a Triton kernel backend for an existing AutoDeploy torch_* custom op, with full registration, dispatch wiring, and correctness/performance tests.
---

# AutoDeploy Triton Kernel Authoring

**Input:** An existing `torch_*` custom op name or operation description.
**Output:** A Triton kernel + `triton_*` custom op registration + graph transform dispatch entry + correctness tests.

**Precondition:** A `torch_*` canonical op MUST already exist for the target operation. If it does not, **stop and tell the user** — the torch reference op must be created first (that is a separate task). This skill only adds the Triton backend.

## Background: How AD Custom Ops Work

AutoDeploy uses a **two-stage dispatch** pattern for custom ops:

1. **Canonical `torch_*` ops** — Pure PyTorch reference implementations registered as `torch.ops.auto_deploy.torch_<op_name>`. These are the IR nodes that AD graph transforms recognize. Custom models always call these.
2. **Backend ops** (`triton_*`, `flashinfer_*`, `trtllm_*`) — Optimized implementations that graph transforms swap in at deployment time. The user never calls these directly.

The file layout follows a consistent pattern:
```
tensorrt_llm/_torch/auto_deploy/custom_ops/
├── <op_category>/                    # e.g., normalization/, activation/, etc.
│   ├── __init__.py                   # __all__ list of module names
│   ├── <op_name>.py                  # torch_* + flashinfer_* + triton_* custom op wrappers
│   └── triton_<op_name>.py           # @triton.jit kernel + Python launcher
└── __init__.py                       # Auto-imports all submodules (triggers registration)
```

Graph transforms that wire up backend selection live in:
```
tensorrt_llm/_torch/auto_deploy/transform/library/<op_name>.py
```

**Reference files to study before starting:**
- Triton kernel: `custom_ops/normalization/triton_rms_norm.py`
- Custom op wrappers (all backends): `custom_ops/normalization/rms_norm.py`
- Graph transform with dispatch: `transform/library/rms_norm.py`

## Phase 0 — Locate and Validate the `torch_*` Reference

### Step 1 — Find the torch_* op

1. **If the user names a `torch_*` op** (e.g., `torch_rmsnorm`): find it in `custom_ops/` and read the implementation.
2. **If the user names a PyTorch op** (e.g., "softmax", "SiLU"): search `custom_ops/README.md` for a matching `torch_*` canonical op.
3. **If the user provides a model or layer**: identify the hotspot op, then search as above.

**If no `torch_*` canonical op exists for the target operation: STOP.** Report to the user that the torch reference must be created first. Do not proceed.

### Step 2 — Read and document the torch reference

Read the existing `torch_*` custom op implementation in full. Document:
- **Signature**: all input tensors, scalar parameters, return type(s)
- **Semantics**: what the op computes (mathematically)
- **Dtypes**: what input dtypes are expected (bf16, fp16, fp32)
- **Shapes**: what tensor shapes are expected (e.g., `[B, S, H]`, `[M, N]`)
- **Edge cases**: masking, optional args, multi-output

### Step 3 — Check for existing Triton backend

Verify that a `triton_*` op does not already exist:
- Search `custom_ops/README.md` for `triton_<op_name>`
- Search the op's `.py` file for `triton_` registrations

**If a Triton backend already exists: STOP.** Report to the user.

Report the signature, semantics, and confirmation that no Triton backend exists to the user before proceeding.

## Phase 1 — Write the Triton Kernel

Create `custom_ops/<category>/triton_<op_name>.py`.

### Kernel structure

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (full license header)

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def <op_name>_kernel(
    # --- Pointer arguments (one per tensor) ---
    input_ptr,
    output_ptr,
    # --- Scalar arguments ---
    stride_row: tl.constexpr,   # strides for pointer arithmetic
    N_COLS: tl.constexpr,       # problem size dimensions
    BLOCK_SIZE: tl.constexpr,   # tile size (power of 2)
    # ... other constexpr params
):
    """Triton kernel for <op_name>.

    Grid: (num_rows,) — one program per row.
    """
    # 1. Identify which row this program handles
    row_idx = tl.program_id(0)

    # 2. Compute column offsets within the tile
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N_COLS  # bounds check

    # 3. Load input data
    input_row_ptr = input_ptr + row_idx * stride_row
    x = tl.load(input_row_ptr + col_offsets, mask=mask, other=0.0)

    # 4. Compute — upcast to float32 for numerical stability
    xf = x.to(tl.float32)
    # ... actual computation ...
    result = xf  # placeholder

    # 5. Cast back and store
    out = result.to(x.dtype)
    output_row_ptr = output_ptr + row_idx * stride_row
    tl.store(output_row_ptr + col_offsets, out, mask=mask)


def <op_name>(input: Tensor, ...) -> Tensor:
    """Python launcher for the Triton kernel.

    Handles grid computation, block size selection, output allocation,
    and kernel launch.
    """
    # Compute grid dimensions
    *batch_dims, N = input.shape
    num_rows = input.numel() // N
    stride_row = input.stride(-2)

    # Choose block size (must be power of 2, >= N)
    BLOCK_SIZE = triton.next_power_of_2(N)

    # Allocate output
    out = torch.empty_like(input)

    # Launch kernel
    grid = (num_rows,)
    <op_name>_kernel[grid](
        input,
        out,
        stride_row=stride_row,
        N_COLS=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,     # tune: 1, 2, 4, 8
        num_stages=3,    # tune: 1, 2, 3, 4
    )

    return out
```

### Triton kernel checklist

- [ ] **Grid design**: One program per independent work unit (typically one row). Use `tl.program_id(0)` for 1D grids. Use 2D grids (`tl.program_id(0)`, `tl.program_id(1)`) for tiled matmuls or 2D reductions.
- [ ] **Block size**: Must be `tl.constexpr` and a power of 2. Use `triton.next_power_of_2(N)` in the launcher.
- [ ] **Masking**: Always use `mask=offsets < N_COLS` on `tl.load`/`tl.store` to avoid out-of-bounds access when `N` is not a power of 2.
- [ ] **Numerical precision**: Upcast to `tl.float32` for reductions (sum, mean, variance). Cast back to input dtype before store.
- [ ] **Strides**: Use tensor strides for pointer arithmetic, not hardcoded shapes. This handles non-contiguous inputs.
- [ ] **`other=0.0`**: Set on `tl.load` with mask to avoid undefined values in masked-off lanes (important for reductions).
- [ ] **Reductions**: Use `tl.sum`, `tl.max`, etc. with appropriate axis. For row-wise reduction, axis=0 on a 1D `tl.arange` block.
- [ ] **No Python control flow inside `@triton.jit`**: All branching must use `tl.where` or constexpr parameters, not Python `if`.
- [ ] **Warp/stage tuning**: Start with `num_warps=4, num_stages=3`. Profile and tune later.

### Common Triton patterns

**Elementwise** (e.g., activation functions):
```python
# Grid: one program per BLOCK_SIZE chunk of elements
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(input_ptr + offsets, mask=mask)
out = tl.sigmoid(x) * x  # SiLU example
tl.store(output_ptr + offsets, out, mask=mask)
```

**Row-wise reduction** (e.g., softmax, layer norm):
```python
# Grid: one program per row
row_idx = tl.program_id(0)
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < N_COLS
x = tl.load(input_ptr + row_idx * stride + offsets, mask=mask, other=-float('inf'))
max_val = tl.max(x, axis=0)
x = tl.exp(x - max_val)
sum_val = tl.sum(x, axis=0)
out = x / sum_val
```

**Fused elementwise + reduction** (e.g., RMSNorm):
```python
# Load, reduce for variance, normalize, scale — all in one kernel
x = tl.load(...)
xf = x.to(tl.float32)
var = tl.sum(xf * xf, 0) * (1.0 / N_COLS)
out = xf / tl.sqrt(var + eps)
out = (w * out).to(x.dtype)
tl.store(...)
```

## Phase 2 — Register the `triton_*` Custom Op

In the same file as the `torch_*` op (e.g., `custom_ops/<category>/<op_name>.py`), add:

```python
from .triton_<op_name> import <op_name> as _triton_<op_name>

@torch.library.custom_op("auto_deploy::triton_<op_name>", mutates_args=())
def triton_<op_name>(input: torch.Tensor, ...) -> torch.Tensor:
    """Triton backend for <op_name>."""
    return _triton_<op_name>(input, ...)

@triton_<op_name>.register_fake
def _(input: torch.Tensor, ...) -> torch.Tensor:
    """Fake implementation for torch.export tracing."""
    return torch.empty_like(input)
```

**Requirements:**
- The `triton_*` custom op MUST have the **exact same signature** (parameter names, types, order) as the `torch_*` op. This is critical for the graph transform to swap them.
- The `.register_fake` can be identical to the `torch_*` version.
- Import the Triton launcher function (NOT the `@triton.jit` kernel) from the `triton_<op_name>.py` module.

Update `custom_ops/<category>/__init__.py` to include the new module in `__all__`.

## Phase 3 — Wire Up Graph Transform Dispatch (If Applicable)

**This phase is needed only if you want AD to automatically select the Triton backend via configuration.** If the op is always dispatched as Triton (no backend selection), skip this phase — the Triton op can be used directly.

### Option A: Simple backend dictionary (like RMSNorm)

If the op has a corresponding graph transform in `transform/library/`, add the Triton backend to its dispatch dictionary:

```python
# In transform/library/<op_name>.py
_BACKEND_OPS = {
    "flashinfer": torch.ops.auto_deploy.<flashinfer_op>,
    "triton": torch.ops.auto_deploy.triton_<op_name>,  # ADD THIS
    "torch": torch.ops.auto_deploy.torch_<op_name>,
}
```

Also update the `Literal` type in the config class to include `"triton"`.

### Option B: Registry-based dispatch (like Attention)

If the op uses a registry pattern (e.g., `AttentionRegistry`), register the Triton backend descriptor:

```python
@AttentionRegistry.register("triton")
class TritonAttentionDescriptor(AttentionDescriptor):
    ...
```

### Option C: New transform (for a new op category)

If no graph transform exists for this op yet, create `transform/library/<op_name>.py` following the `rms_norm.py` pattern:

1. **Pattern match stage** (`MatchXxxPattern`): Match raw PyTorch subgraphs → replace with `torch_*` canonical op
2. **Fusion stage** (`FuseXxx`): Replace `torch_*` op with backend-specific op based on config

Register both transforms:
```python
@TransformRegistry.register("match_<op_name>_pattern")
class MatchXxxPattern(BaseTransform):
    ...

@TransformRegistry.register("fuse_<op_name>")
class FuseXxx(BaseTransform):
    ...
```

Add the transforms to the appropriate pipeline stages (check existing transform ordering in the AD pipeline configuration).

## Phase 4 — Correctness Tests

Create `tests/unittest/auto_deploy/singlegpu/custom_ops/<category>/test_triton_<op_name>.py`.

**Structure:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ... (full license header)

import pytest
import torch

# Import the torch reference and triton backend
from tensorrt_llm._torch.auto_deploy.custom_ops.<category>.<op_file> import (
    torch_<op_name>,
    triton_<op_name>,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 128), (4, 512), (8, 1024), (32, 4096)])
def test_triton_<op_name>_matches_torch(shape, dtype):
    """Triton kernel output matches torch reference within tolerance."""
    torch.manual_seed(42)
    input = torch.randn(*shape, device="cuda", dtype=dtype)
    # ... other inputs (weights, etc.)

    expected = torch_<op_name>(input, ...)
    actual = triton_<op_name>(input, ...)

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", [(1, 128), (4, 513), (8, 1023)])  # include non-power-of-2
def test_triton_<op_name>_non_power_of_2(shape):
    """Triton kernel handles non-power-of-2 dimensions correctly."""
    input = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    expected = torch_<op_name>(input, ...)
    actual = triton_<op_name>(input, ...)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_triton_<op_name>_large():
    """Triton kernel works on large realistic shapes."""
    input = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16)
    expected = torch_<op_name>(input, ...)
    actual = triton_<op_name>(input, ...)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
```

**Test checklist:**
- [ ] Multiple dtypes (bf16, fp16, fp32)
- [ ] Multiple shapes including non-power-of-2 dimensions
- [ ] Large realistic shapes (e.g., hidden_size=4096, 8192)
- [ ] Edge cases: single-element, single-row, very wide rows
- [ ] All optional parameters exercised
- [ ] Gradient correctness (if the op needs to support autograd)

Also add the `"triton"` variant to existing transform fusion tests if they exist (e.g., `test_fuse_<op_name>.py`).

Run tests:
```bash
pytest tests/unittest/auto_deploy/singlegpu/custom_ops/<category>/test_triton_<op_name>.py -v
```

## Phase 5 — Performance Validation

After correctness is verified, benchmark the Triton kernel against the torch reference.

```python
import torch
import triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg='provider',
        line_vals=['torch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        ylabel='GB/s',
        plot_name='<op_name>-performance',
        args={'M': 2048},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    # ... setup
    if provider == 'torch':
        fn = lambda: torch_<op_name>(x, ...)
    else:
        fn = lambda: triton_<op_name>(x, ...)
    ms = triton.testing.do_bench(fn)
    gbps = 2 * x.numel() * x.element_size() / ms * 1e-6  # read + write
    return gbps
```

The Triton kernel should be **at least as fast** as the torch reference for typical shapes. If it is slower, investigate:
- Block size tuning (`BLOCK_SIZE`, `num_warps`, `num_stages`)
- Memory access patterns (coalescing)
- Occupancy (too few or too many warps)

Report the benchmark results to the user.

## Phase 6 — Summary Report

Print (not file) after completion:

1. **Op overview**: what the operation computes, signature
2. **Files created/modified** (with paths)
3. **Test results** (name | PASS/FAIL)
4. **Performance** (torch vs triton throughput for key shapes)
5. **Known limitations** (e.g., max dimension constraints, dtype restrictions)
6. **Graph transform status** (was dispatch wiring added? which config controls it?)

## Key Gotchas

- **Signature parity**: The `triton_*` and `torch_*` custom ops MUST have identical signatures (names, types, order). Graph transforms swap them by replacing the op target while keeping args unchanged.
- **`register_fake` is mandatory**: Without it, `torch.export` tracing will fail. The fake must return tensors with correct shape and dtype but does NOT run on GPU.
- **`mutates_args`**: Must accurately reflect which args are modified in-place. Get this wrong and `torch.compile`/`torch.export` will produce incorrect results.
- **Power-of-2 block sizes**: Triton requires `BLOCK_SIZE` to be a power of 2. Always use `triton.next_power_of_2(N)` and mask loads/stores.
- **Contiguous tensors**: Triton kernels assume contiguous memory layout. Either assert contiguity in the launcher or call `.contiguous()` before launching.
- **Float32 accumulation**: Always upcast to `tl.float32` for reductions to avoid catastrophic numerical error in bf16/fp16.
- **No Python `if` inside `@triton.jit`**: Use `tl.where` for conditional computation. Python `if` on `tl.constexpr` params is OK for compile-time branching.
- **NVIDIA copyright header**: Required on ALL new files.
- **Auto-import**: New modules in `custom_ops/` are auto-imported by the top-level `__init__.py` (via `pkgutil.walk_packages`). You do NOT need to add explicit imports there — just ensure your file is a `.py` file in the right subdirectory. You DO need to update the subdirectory's `__init__.py` `__all__` list.
