# Checklists

Pre-merge checklists for the ad-triton-kernel skill. Run through these after completing the relevant phases.

## Triton Kernel

Verify after Phase 1 (writing the kernel):

- [ ] **Grid design**: One program per independent work unit (typically one row). Use `tl.program_id(0)` for 1D grids. Use 2D grids (`tl.program_id(0)`, `tl.program_id(1)`) for tiled matmuls or 2D reductions.
- [ ] **Block size**: Must be `tl.constexpr` and a power of 2. Use `triton.next_power_of_2(N)` in the launcher.
- [ ] **Masking**: Always use `mask=offsets < N_COLS` on `tl.load`/`tl.store` to avoid out-of-bounds access when `N` is not a power of 2.
- [ ] **Numerical precision**: Upcast to `tl.float32` for reductions (sum, mean, variance). Cast back to input dtype before store.
- [ ] **Strides**: Use tensor strides for pointer arithmetic, not hardcoded shapes. This handles non-contiguous inputs.
- [ ] **`other=0.0`**: Set on `tl.load` with mask to avoid undefined values in masked-off lanes (important for reductions).
- [ ] **Reductions**: Use `tl.sum`, `tl.max`, etc. with appropriate axis. For row-wise reduction, axis=0 on a 1D `tl.arange` block.
- [ ] **No Python control flow inside `@triton.jit`**: All branching must use `tl.where` or constexpr parameters, not Python `if`.
- [ ] **Warp/stage tuning**: Start with `num_warps=4, num_stages=3`. Profile and tune later.

## Correctness Tests

Verify after Phase 4 (writing tests):

- [ ] Multiple dtypes (bf16, fp16, fp32)
- [ ] Multiple shapes including non-power-of-2 dimensions
- [ ] Large realistic shapes (e.g., hidden_size=4096, 8192)
- [ ] Edge cases: single-element, single-row, very wide rows
- [ ] All optional parameters exercised
- [ ] Gradient correctness (if the op needs to support autograd)

## Dual-Phase Attention Kernels

Additional checks for attention ops with prefill/decode (see PATTERNS.md):

- [ ] **No PyTorch fallback**: Both prefill and decode must use Triton (or equivalent optimized kernel)
- [ ] **Shared kernel**: Use per-token metadata to parameterize, not separate kernel functions
- [ ] **Vectorized cache update**: No Python `for` loops in the prefill cache update path
- [ ] **Causal masking via kv_len**: Each token's kv_len naturally encodes the causal boundary
- [ ] **Weight absorption correctness**: Verify the mathematical equivalence of absorption vs expansion
- [ ] **Chunked prefill**: Handle the case where `input_pos > 0` (prefilling into an existing cache)

## Pre-Merge Summary

Final checks before reporting to user:

- [ ] NVIDIA copyright header on ALL new files
- [ ] `triton_*` and `torch_*` signatures are identical (names, types, order)
- [ ] `register_fake` is present on the `triton_*` custom op
- [ ] `mutates_args` accurately reflects in-place mutations
- [ ] `__init__.py` `__all__` list updated for the op category
- [ ] All tests pass
- [ ] Performance is at least on par with the torch reference
