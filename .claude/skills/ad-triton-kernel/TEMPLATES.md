# Code Templates

Code templates for the ad-triton-kernel skill. Each section is a self-contained template referenced from SKILL.md phases.

## Triton Kernel and Launcher

File: `custom_ops/<category>/triton_<op_name>.py`

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

## Custom Op Registration

Add to the same file as the `torch_*` op (e.g., `custom_ops/<category>/<op_name>.py`):

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

## Graph Transform

File: `transform/library/<op_name>.py` (for Option C — new transform)

```python
@TransformRegistry.register("match_<op_name>_pattern")
class MatchXxxPattern(BaseTransform):
    ...

@TransformRegistry.register("fuse_<op_name>")
class FuseXxx(BaseTransform):
    ...
```

Follow the `rms_norm.py` pattern:
1. **Pattern match stage** (`MatchXxxPattern`): Match raw PyTorch subgraphs → replace with `torch_*` canonical op
2. **Fusion stage** (`FuseXxx`): Replace `torch_*` op with backend-specific op based on config

## Correctness Tests

File: `tests/unittest/auto_deploy/singlegpu/custom_ops/<category>/test_triton_<op_name>.py`

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

## Performance Benchmark

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
