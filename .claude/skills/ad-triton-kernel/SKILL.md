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

## Decision Tree

Before starting, classify the op to determine which phases apply:

```
Is this a simple op (elementwise / row-reduction / fused)?
  YES → Standard flow: Phase 0 → 1 → 2 → 3 (if dispatch needed) → 4 → 5 → 6
  NO  → Is this an attention op with prefill/decode phases?
          YES → Standard flow BUT use dual-phase pattern in Phase 1
                (see PATTERNS.md#dual-phase-kernels-for-attention-ops)
          NO  → Assess complexity, discuss with user before proceeding

Does the op already have a graph transform in transform/library/?
  YES → Phase 3 Option A or B (add backend to existing dispatch)
  NO  → Does it need configurable backend selection?
          YES → Phase 3 Option C (create new transform)
          NO  → Skip Phase 3
```

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

Use the kernel + launcher template from **TEMPLATES.md#triton-kernel-and-launcher**.

Choose the appropriate Triton pattern from **PATTERNS.md**:
- Elementwise ops → **PATTERNS.md#elementwise**
- Row-wise reduction ops → **PATTERNS.md#row-wise-reduction**
- Fused elementwise + reduction → **PATTERNS.md#fused-elementwise--reduction**
- Attention with prefill/decode → **PATTERNS.md#dual-phase-kernels-for-attention-ops**

After writing, verify against **CHECKLIST.md#triton-kernel**.

## Phase 2 — Register the `triton_*` Custom Op

In the same file as the `torch_*` op (e.g., `custom_ops/<category>/<op_name>.py`), add the registration using the template from **TEMPLATES.md#custom-op-registration**.

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

If no graph transform exists for this op yet, create `transform/library/<op_name>.py` following the `rms_norm.py` pattern. See **TEMPLATES.md#graph-transform** for the template.

Add the transforms to the appropriate pipeline stages (check existing transform ordering in the AD pipeline configuration).

## Phase 4 — Correctness Tests

Create `tests/unittest/auto_deploy/singlegpu/custom_ops/<category>/test_triton_<op_name>.py`.

Use the test template from **TEMPLATES.md#correctness-tests**.

Also add the `"triton"` variant to existing transform fusion tests if they exist (e.g., `test_fuse_<op_name>.py`).

Verify against **CHECKLIST.md#correctness-tests**.

Run tests:
```bash
pytest tests/unittest/auto_deploy/singlegpu/custom_ops/<category>/test_triton_<op_name>.py -v
```

## Phase 5 — Performance Validation

After correctness is verified, benchmark the Triton kernel against the torch reference using the template from **TEMPLATES.md#performance-benchmark**.

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
