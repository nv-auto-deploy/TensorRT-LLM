# Gemma4 Megakernel A: HazyResearch-Style Persistent Kernel for Decode Attention

## Context

The current Triton Kernel A (`triton_gemma_kernel_a_decode.py`) fuses the entire attention
half of a Gemma4 layer into one op, but launches on a **single SM** (`grid=(num_tokens,)`).
For batch_size=1 decode, this means 131 of 132 SMs sit idle. The kernel is **slower** than
the decomposed baseline (~8 separate kernels using cuBLAS + Triton) because memory-bandwidth-bound
matvecs on 1 SM get 1/132 of total bandwidth.

| Path | Estimated Latency | Why |
|------|------------------|-----|
| Decomposed (8 kernels) | 100-150 us | Full GPU utilization, ~40 us launch overhead |
| Current Triton Kernel A | ~3000 us | 1 SM, 1/132 bandwidth |
| **Target: Megakernel A** | **28-40 us** | All 132 SMs, zero launch overhead, fused elementwise |

The HazyResearch Megakernels project demonstrates that a persistent CUDA kernel across all SMs
with an instruction stream can achieve near-bandwidth-optimal matvec for decode. We adapt this
for Gemma4's attention sublayer, handling paged attention and per-head norms that HazyResearch
doesn't support.

---

## Phase 0: ThunderKittens Scaffold + Persistent Kernel Proof (Est. 2-3 days)

### Goal
Get a minimal persistent kernel compiling and launching on H100 — verifying that
ThunderKittens works with Gemma4's head_dim=256 tiles and that inter-SM barriers function.

### New Files
```
tensorrt_llm/_torch/auto_deploy/custom_ops/attention/megakernel/
  Makefile                      # nvcc + pybind11, sm_90a
  gemma4_config.cuh             # Compile-time constants (HIDDEN=2816, HEAD_DIM=256, etc.)
  megakernel_framework.cuh      # Instruction struct, global barrier, SM dispatch loop
  gemma4_globals.cuh            # Runtime globals struct (weight pointers, cache, barriers)
  noop_opcode.cuh               # No-op instruction handler (for scaffold testing)
  megakernel_host.cu            # pybind11 module: launch + instruction builder
  test_scaffold.py              # Python: build, load, launch noop, verify barrier sync
```

### ThunderKittens Integration
- Git submodule or vendored headers from `HazyResearch/ThunderKittens` branch
  `bvm-single-ctrl-pre-new-warps` into `megakernel/thunderkittens/`
- Build flags: `-I./thunderkittens/include -std=c++20 --expt-relaxed-constexpr`
- Verify: `kittens::rt_bf<1,16>` (1 row x 256 cols bf16 register tile) loads from
  shared memory without register spill

### Key Design Decisions
- **Instruction encoding**: 32 x int32 (128 bytes), matching HazyResearch format
- **Warp roles**: 20 warps = controller(1) + loader(1) + launcher(1) + storer(1) + consumer(16)
- **Shared memory**: 12 pages x 16KB = 192KB (within H100's 228KB limit)
- **Launch config**: `<<<132, 640, 192*1024>>>` (132 SMs, 20 warps, 192KB smem)

### Go/No-Go
- [ ] Kernel launches on 132 SMs without errors
- [ ] Global barrier (atomicAdd + spin-wait) synchronizes all SMs correctly
- [ ] ThunderKittens bf16 register tile (head_dim=256) loads without spill
- [ ] Compile time < 3 minutes

---

## Phase 1: QKV MatVec Opcode (Est. 1-2 weeks)

### Goal
Implement `RMS_QKV_PerHeadNorm_RoPE_PagedAppend` — the most complex and most
performance-critical opcode. This single instruction replaces 5 decomposed kernels.

### Fused Operations
```
Input (already RMS-normed from prior layer):
  [1, 2816] bf16

1. QKV MatVec: [1,2816] x [8192,2816]^T → [1,8192]     ← 46 MB weights, dominates
2. Split → Q[16,256], K[8,256], V[8,256]
3. Per-head RMSNorm on Q, K, V (3 separate norms)        ← Gemma4-specific
4. RoPE on Q and K                                        ← Standard rotation
5. Write K,V to paged KV cache                            ← Page-table indirection
6. Output Q to scratch for attention                       ← Global memory write
```

### SM Work Distribution (132 SMs)
- **MatVec**: Each SM handles ~62 output columns of [8192 x 2816] weight
  - Load input vector (5.6KB) from L2 cache (broadcast)
  - Stream weight slice (62 x 2816 x 2B = 349KB) via TMA pipeline (3 stages)
  - Accumulate dot products in fp32 registers, store bf16 results
- **Post-MatVec** (on same SM, in registers — zero extra memory traffic):
  - If output row is in K range: apply K-head RMSNorm + RoPE + write to paged cache
  - If output row is in V range: apply V-head RMSNorm + write to paged cache
  - If output row is in Q range: apply Q-head RMSNorm + RoPE + write to Q scratch

### New Files
```
megakernel/opcodes/rms_qkv_norm_rope_paged.cuh    # Opcode handler
megakernel/tiles/matvec_pipeline.cuh               # TMA-pipelined GEMV (adapt from HazyResearch)
megakernel/tiles/rmsnorm_inline.cuh                # In-register RMSNorm for head_dim=256
megakernel/tiles/rope_gemma4.cuh                   # RoPE (standard + proportional variants)
megakernel/tiles/paged_cache_write.cuh             # Page-table-indexed cache writes
megakernel/instructions.py                         # Python instruction encoder
```

### Paged Cache Write (vs HazyResearch's contiguous cache)
```cuda
// HazyResearch: TMA store to contiguous cache
tma::store_async(kv_cache_gl, smem_tile, {layer, position, head, 0});

// Gemma4: Compute physical page address, use cp.async or direct store
int page_idx = position / PAGE_SIZE;
int offset_in_page = position % PAGE_SIZE;
int phys_page = __ldg(&page_table[page_start + page_idx]);
bf16* dst = &kv_cache[phys_page * stride_block + kv * stride_kv
                      + head * stride_head + offset_in_page * stride_tok];
// Store from registers (head_dim=256 = 512 bytes = 16 stores of 32B)
```
This is fine because KV write is 8KB total (tiny vs 46MB weight load).

### Testing
**File**: `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_megakernel_qkv_opcode.py`
- Reference: `aten.linear` + `rms_norm` + `flashinfer_rope` + `update_paged_kv_cache`
- Tolerances: atol=1e-2, rtol=1e-2 (outputs), atol=0/rtol=2e-7 (cache)
- Shape sweep: batch={1,4}, past_length={128,1024}, page_size={16,32}

### Go/No-Go
- [ ] QKV output matches reference within tolerances
- [ ] Paged cache entries match reference exactly
- [ ] **Opcode latency < 18 us** (batch=1) — bandwidth limit is 46MB/3.35TB/s = 13.7 us

---

## Phase 2: Paged Attention Opcodes (Est. 1-2 weeks)

### Goal
Implement `PagedPartialAttention` and `AttentionReduction` opcodes.

### PagedPartialAttention
- Each SM processes one (kv_head, page_range) pair
- Loads Q from scratch, iterates over assigned KV cache pages
- Online softmax: track running max `m_i` and sum `l_i`
- Output: partial `O[gqa_ratio, head_dim]` + LSE per Q-head

### SM Assignment (dynamic based on sequence length)
| seq_len | pages (ps=16) | SMs per head | Total SMs | Idle SMs |
|---------|---------------|--------------|-----------|----------|
| 128     | 8             | 1            | 8         | 124      |
| 1024    | 64            | 8            | 64        | 68       |
| 4096    | 256           | 32           | 132*      | 0        |

*Capped at 132 SMs, some heads get more partials than others.

### Paged Cache Read (key adaptation from HazyResearch)
HazyResearch uses TMA descriptors for contiguous KV blocks. For paged cache:
- Use `cp.async.cg.shared.global` with computed physical page addresses
- Or regular loads with software prefetch
- KV data per page: `PAGE_SIZE * HEAD_DIM * 2B` = 16 * 256 * 2 = 8KB — small enough
  that TMA vs cp.async difference is negligible

### AttentionReduction
- LSE-corrected weighted combination of partial outputs
- `O_final = Σ (O_partial_i * exp(LSE_i - LSE_max)) / Σ exp(LSE_i - LSE_max)`
- 16 SMs (one per Q-head), only when num_partials > 1
- Skip entirely for short sequences (1 partial per head)

### New Files
```
megakernel/opcodes/paged_partial_attention.cuh
megakernel/opcodes/attention_reduction.cuh
megakernel/tiles/online_softmax.cuh
megakernel/tiles/paged_cache_read.cuh
```

### Testing
**File**: `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_megakernel_attention_opcodes.py`
- Reference: `torch.ops.auto_deploy.triton_paged_mha_with_cache`
- Tolerances: atol=2e-2, rtol=2e-2 (attention has more numerical variance)
- Sweep: seq_len={128, 512, 2048, 8192}, batch={1, 4}

### Go/No-Go
- [ ] Attention output matches paged MHA reference within tolerances
- [ ] Handles last-page partial fill correctly
- [ ] Reduction produces identical results to single-pass reference
- [ ] **Attention latency < 15 us** for seq=2048, batch=1

---

## Phase 3: O-Projection Opcode + Full Kernel A Assembly (Est. 1-2 weeks)

### Goal
Implement `OProj_PostNorm_Residual_PreNorm` and wire all 4 instructions into the
complete Kernel A custom op.

### OProj_PostNorm_Residual_PreNorm Opcode
```
Input: attention output [1, 4096] from attention scratch

1. O-proj MatVec: [1,4096] x [2816,4096]^T → [1,2816]    ← 23 MB weights
2. RMSNorm on O-proj output (post_attention_layernorm)     ← Fused in epilogue
3. Residual add: post_attn = residual + normed_o_proj      ← Fused in epilogue
4. RMSNorm on post_attn (pre_feedforward_layernorm)        ← Fused in epilogue
5. Output: post_attn_residual [2816] + pre_ffn_normed [2816]
```

Steps 2-4 are **fused into the matvec epilogue** — the SM that computes output row `h`
also applies the norms and residual for that element. Zero extra global memory traffic.

### Full Instruction Stream (per layer)
```python
instructions = [
    RmsQkvNormRopePagedAppend(sm_range=(0, 131), ...),    # All SMs
    # --- global barrier ---
    PagedPartialAttention(sm_range=(0, N-1), ...),         # N SMs
    # --- global barrier ---
    AttentionReduction(sm_range=(0, 15), ...),             # 16 SMs (skip if N<=8)
    # --- global barrier ---
    OProjPostNormResidualPreNorm(sm_range=(0, 131), ...),  # All SMs
]
```

### New Files
```
megakernel/opcodes/oproj_postnorm_residual_prenorm.cuh
megakernel/kernel_a_launcher.py        # Builds instruction queue from model config + runtime metadata
megakernel/kernel_a_custom_op.py       # @torch.library.custom_op with same signature as Triton Kernel A
```

### Custom Op Registration
```python
@torch.library.custom_op(
    "auto_deploy::megakernel_gemma_kernel_a_decode",
    mutates_args=("kv_cache",)
)
def megakernel_gemma_kernel_a_decode(
    residual_in, attn_normed_in, qkv_weight, o_proj_weight,
    q_norm_weight, k_norm_weight, v_norm_weight,
    post_attention_layernorm_weight, pre_feedforward_layernorm_weight,
    position_ids, cos_sin_cache,
    batch_info_host, cu_seqlen_host, cu_num_pages, cu_num_pages_host,
    cache_loc, last_page_len, last_page_len_host,
    seq_len_with_cache_host, triton_batch_indices, triton_positions,
    kv_cache, scale=None, sliding_window=None, eps=1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```
**Same signature as existing Triton op** — drop-in replacement.

### Testing
**File**: `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_megakernel_gemma_kernel_a.py`
- Reference: `_run_reference_kernel_a` from `triton_gemma_kernel_a_decode.py`
- Full shape sweep matching existing benchmark cases:
  - batch={1,4}, past={128,1024}, page_size={16,32,64}
- Tolerances: atol=1e-2, rtol=1e-2

**Benchmark extension**: Add megakernel to `benchmark_gemma_kernel_a_decode.py`:
```
case  bs  past  page  megakernel_us  decomp_us  triton_us  mk/decomp
```

### Go/No-Go
- [ ] Full Kernel A matches reference for all sweep cases
- [ ] KV cache mutations match exactly
- [ ] **Total latency: 28-40 us** (batch=1, seq=2048)
- [ ] Stable over 1000 iterations (no hangs/deadlocks)
- [ ] 3-4x faster than decomposed baseline

---

## Phase 4: AutoDeploy Custom Op Integration (Est. 3-5 days)

### Goal
Register the megakernel as a standard AutoDeploy custom op and wire it into the
Gemma4 model via an FX graph transformation. **No MPK lowering pipeline needed** —
the megakernel is a drop-in replacement for the decomposed attention subgraph.

### Approach: FX Graph Transform (same pattern as existing attention op replacements)
The megakernel custom op has the **exact same signature** as the existing Triton Kernel A.
Integration is a simple FX graph transformation that pattern-matches the attention
subgraph (input_norm → QKV → per-head norms → RoPE → paged_attn → o_proj → residual →
pre_ffn_norm) and replaces it with a single `megakernel_gemma_kernel_a_decode` call.

### New Files
```
tensorrt_llm/_torch/auto_deploy/transformations/library/
  replace_attention_with_megakernel.py    # FX transform: pattern match → replace
```

### Modified Files
1. **`tensorrt_llm/_torch/auto_deploy/custom_ops/attention/__init__.py`**
   - Import the megakernel custom op module

2. **`examples/auto_deploy/model_registry/configs/gemma4_moe.yaml`**
   - Add optional `megakernel_attention: true` config flag

3. **Transform registration** (in the appropriate pipeline config):
   - Gated by env var `AD_USE_MEGAKERNEL_ATTN=1` or YAML config flag
   - Runs during the `compile` stage, before `compile_model`
   - Decode-only: only replaces attention in generate-only code path

### Build: JIT via torch.utils.cpp_extension
```python
def _load_megakernel_a_extension():
    from torch.utils.cpp_extension import load
    return load(
        name="gemma4_megakernel_a",
        sources=[str(src_dir / "megakernel_host.cu")],
        extra_include_paths=[str(src_dir / "thunderkittens/include")],
        extra_cuda_cflags=["-O3", "-std=c++20", "--expt-relaxed-constexpr",
                           "-gencode=arch=compute_90a,code=sm_90a"],
    )
```

### Why Not MPK Lowering?
The MPK pipeline (analyzer → planner → Mirage bridge) orchestrates *multiple*
persistent kernels with scheduling. For a single custom op that replaces the
attention sublayer end-to-end, that machinery is unnecessary. A direct custom op
+ FX transform is simpler, faster to iterate, and follows the standard AutoDeploy
pattern for fused ops.

### Testing
- Full-model decode: compare logits megakernel path vs standard path
- Verify prefill still works (megakernel is decode-only, transform only matches decode)
- Per-layer timing comparison via nsys

### Go/No-Go
- [ ] Full-model decode logits match (top-1 token agreement)
- [ ] No regression in prefill or mixed-batch paths
- [ ] Per-layer attention latency: 28-40 us (megakernel) vs 100-150 us (decomposed)

---

## Phase 5: CMake Build + Production Hardening (Est. 1 week)

### Goal
Move from JIT to AOT build, add error handling, prepare for main branch.

### New Files (CMake path)
```
cpp/tensorrt_llm/kernels/gemma4MegakernelA/
  CMakeLists.txt                # Pattern: llama4MinLatencyKernels/CMakeLists.txt
  thunderkittens/               # Vendored TK headers
  *.cuh, *.cu                   # All CUDA sources

cpp/tensorrt_llm/thop/gemma4MegakernelAOp.cpp   # Torch op wrapper
```

### CMakeLists.txt Pattern
```cmake
file(GLOB_RECURSE SRC_CU *.cu)
add_library(gemma4_megakernel_a OBJECT ${SRC_CU})
set_property(TARGET gemma4_megakernel_a PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET gemma4_megakernel_a PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_cuda_architectures(gemma4_megakernel_a 90)
target_include_directories(gemma4_megakernel_a PRIVATE thunderkittens/include)
```

### Hardening
- GPU arch check (sm_90a only, error on other archs)
- Instruction queue bounds checking (debug mode)
- CUDA error checking after launch (debug mode)
- Memory: ensure scratch buffers are pre-allocated and reused

---

## Performance Budget (Kernel A, batch=1, seq=2048)

| Instruction | Data | Theoretical | Target | Notes |
|-------------|------|-------------|--------|-------|
| QKV GEMV | 46 MB weights | 13.7 us | 15-18 us | All 132 SMs, TMA pipeline |
| + norms/RoPE/cache | negligible | ~0 | fused | In matvec epilogue |
| Barrier 1 | | | 0.5-1 us | atomicAdd spin-wait |
| Attention | 1 MB KV cache | 0.3 us | 5-8 us | 8-64 SMs, page indirection |
| Barrier 2 | | | 0.5-1 us | |
| Reduction | tiny | ~0 | 1-2 us | 16 SMs, may skip |
| Barrier 3 | | | 0.5-1 us | |
| O-proj GEMV | 23 MB weights | 6.9 us | 8-12 us | All 132 SMs |
| + norms/residual | negligible | ~0 | fused | In matvec epilogue |
| **Total** | **69 MB** | **20.6 us** | **28-40 us** | **~35% overhead over BW limit** |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| head_dim=256 register pressure | Occupancy drop | Profile early in Phase 0; reduce consumer warps from 16→8 if needed |
| head_dim=256 smem tile size | Fewer pipeline stages | Use 2-stage pipeline instead of 3; tile head_dim into 4×64 chunks |
| Compile time | Slow iteration | Separate .cu files per opcode; use ccache; precompile TK templates |
| Paged attention TMA incompatibility | Must use slower loads | cp.async with software addressing (KV is 1% of traffic — negligible) |
| Global barrier tail latency | SM starvation | `__nanosleep(20)` in spin-wait; balance work per SM |
| Numerical drift (fp32 accum) | Test failures | All reductions in fp32 (matches existing Triton kernel pattern) |

---

## Key Reference Files

| File | Role |
|------|------|
| `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/triton_gemma_kernel_a_decode.py` | Existing Triton kernel — op signature to match, reference impl `_run_reference_kernel_a` |
| `tensorrt_llm/_torch/auto_deploy/custom_ops/attention/benchmark_gemma_kernel_a_decode.py` | Benchmark harness — extend with megakernel path |
| `tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_triton_gemma_kernel_a_decode.py` | Test patterns — shape sweeps, tolerances, metadata setup |
| `tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py` | JIT build pattern reference (`_load_mirage_runtime_extension`) |
| `cpp/tensorrt_llm/kernels/llama4MinLatencyKernels/CMakeLists.txt` | CMake pattern for Phase 5 |
| `cpp/tensorrt_llm/thop/llama4MinLatency.cpp` | Thop wrapper pattern for Phase 5 |
