# Triton Kernel Patterns

Common Triton kernel patterns referenced from SKILL.md Phase 1. Choose the pattern that matches your op's computation structure.

## Elementwise

For ops that apply a function independently to each element (e.g., activation functions like SiLU, GELU).

**Grid**: one program per `BLOCK_SIZE` chunk of the flattened tensor.

```python
@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    out = tl.sigmoid(x) * x  # SiLU example
    tl.store(output_ptr + offsets, out, mask=mask)
```

**Key points:**
- Flatten all dimensions into a single axis for simplicity
- Grid size = `ceil(n_elements / BLOCK_SIZE)`
- No reductions, no shared state between programs

## Row-wise Reduction

For ops that reduce across the last dimension of each row (e.g., softmax, layer norm).

**Grid**: one program per row.

```python
@triton.jit
def row_reduce_kernel(input_ptr, output_ptr, stride, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS
    x = tl.load(input_ptr + row_idx * stride + offsets, mask=mask, other=-float('inf'))

    # Online softmax example
    max_val = tl.max(x, axis=0)
    x = tl.exp(x - max_val)
    sum_val = tl.sum(x, axis=0)
    out = x / sum_val

    tl.store(output_ptr + row_idx * stride + offsets, out, mask=mask)
```

**Key points:**
- Use `other=-float('inf')` for max reductions, `other=0.0` for sum reductions
- Always upcast to `tl.float32` before reduction for numerical stability
- `BLOCK_SIZE` must be >= `N_COLS` (entire row fits in one tile)

## Fused Elementwise + Reduction

For ops that combine a reduction with per-element scaling (e.g., RMSNorm, LayerNorm).

**Grid**: one program per row.

```python
@triton.jit
def fused_kernel(input_ptr, weight_ptr, output_ptr, stride, N_COLS: tl.constexpr,
                 BLOCK_SIZE: tl.constexpr, eps: tl.constexpr):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS

    # Load and upcast
    x = tl.load(input_ptr + row_idx * stride + offsets, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # Reduce (RMSNorm variance)
    var = tl.sum(xf * xf, 0) * (1.0 / N_COLS)

    # Normalize + scale
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    tl.store(output_ptr + row_idx * stride + offsets, out, mask=mask)
```

**Key points:**
- Single kernel pass: load → reduce → elementwise → store
- Weight tensor is shared across rows (broadcast load)
- Cast back to input dtype only at the final store

## Dual-Phase Kernels for Attention Ops

Some ops — especially attention variants like MLA — require **separate handling for prefill and decode phases**. The standard skill flow (Phase 1-5) covers simple ops where one Triton kernel replaces the entire `torch_*` op. This section covers the more complex pattern.

### When you need dual-phase kernels

An op needs separate prefill/decode handling when:
- The **decode** path processes 1 token per sequence against a full KV cache (streaming pattern)
- The **prefill** path processes many tokens per sequence (batch attention pattern)
- The two phases use fundamentally different algorithms (e.g., expansion vs absorption)

**Falling back to PyTorch for prefill is NOT acceptable** if the PyTorch reference has Python `for` loops over sequences. FlashInfer backends use optimized CUDA kernels for both phases — the Triton backend must match.

### The absorption pattern for MLA-like attention

For MLA (Multi-head Latent Attention), the key insight is that **weight absorption** works for both prefill and decode:

```
# Mathematically equivalent:
# Expansion: score = q_nope . (compressed_kv @ W_kn^T) + q_pe . kpe
# Absorption: score = (q_nope @ W_kn) . compressed_kv + q_pe . kpe
#             i.e.    q_absorbed . compressed_kv + q_pe . kpe
```

This means a **single Triton kernel** can handle both phases — the only difference is the per-token causal KV length.

### Shared kernel with per-token metadata

The key design pattern: **parameterize the kernel by per-token metadata, not by phase**.

```python
@triton.jit
def _mla_attention_kernel(
    q_absorbed_ptr,     # [num_tokens, N, kv_lora_rank]
    q_pe_ptr,           # [num_tokens, N, qk_rope_head_dim]
    mla_cache_ptr,      # [max_batch, max_seq, cache_dim]
    token_slot_ptr,     # [num_tokens] - which cache slot per token
    token_kv_len_ptr,   # [num_tokens] - causal KV length per token
    out_ptr,            # [num_tokens, N, kv_lora_rank]
    ...
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    slot_idx = tl.load(token_slot_ptr + token_id)
    kv_len = tl.load(token_kv_len_ptr + token_id)
    # ... online softmax over [0, kv_len) positions in cache
```

**For decode**: `kv_len = input_pos + 1`, `token_slot = slot_idx` (per-batch-element)

**For prefill**: Per-token metadata is built from per-sequence metadata:
```python
# Vectorized (no Python loops):
token_slots = slot_idx.repeat_interleave(seq_lengths)        # [total_tokens]
base_positions = input_pos.repeat_interleave(seq_lengths)     # [total_tokens]
# Within-sequence offsets: [0,1,...,sl0-1, 0,1,...,sl1-1, ...]
cum = torch.zeros(num_seq + 1, device=device, dtype=torch.long)
cum[1:] = seq_lengths.cumsum(0)
base_in_dense = cum[:-1].repeat_interleave(seq_lengths)
within_offsets = torch.arange(total_tokens, device=device) - base_in_dense
token_cache_pos = base_positions + within_offsets
token_kv_len = (token_cache_pos + 1).to(torch.int32)  # causal
```

### Launcher structure for dual-phase ops

Each phase gets its own Python launcher that:
1. Handles cache update (vectorized for prefill, simple indexing for decode)
2. Computes weight absorption: `q_absorbed = q_nope @ W_kn` (PyTorch einsum)
3. Builds per-token metadata
4. Launches the shared Triton kernel
5. Projects output: `out = weighted_kv @ W_v^T` (PyTorch einsum)

```python
def _triton_mla_decode(...):
    # Cache update (1 token per batch element)
    mla_cache[slot_idx, input_pos, :kv_lora_rank] = compressed_kv
    mla_cache[slot_idx, input_pos, kv_lora_rank:] = kpe
    # Weight absorption -> q_absorbed
    # kv_len = input_pos + 1
    # Launch kernel with grid=(batch, heads)
    # Value projection -> output

def _triton_mla_prefill(...):
    # Vectorized cache update (all tokens at once)
    mla_cache[token_slots, token_cache_pos, :kv_lora_rank] = compressed_kv
    mla_cache[token_slots, token_cache_pos, kv_lora_rank:] = kpe
    # Weight absorption -> q_absorbed
    # Per-token kv_len from token_cache_pos + 1
    # Launch kernel with grid=(total_tokens, heads)
    # Value projection -> output
```

### Tuning differences between phases

While the kernel is shared, the optimal launch parameters differ:

| Parameter | Decode | Prefill |
|-----------|--------|---------|
| Grid | `(batch_size, heads)` | `(total_tokens, heads)` |
| `SEQ_BLOCK` | 8 (short iteration) | 16 (longer sequences) |
| `num_warps` | 2 | 4 |

### Reference implementation

- Shared kernel: `custom_ops/mla/triton_mla.py::_mla_attention_kernel`
- Decode launcher: `custom_ops/mla/triton_mla.py::_triton_mla_decode`
- Prefill launcher: `custom_ops/mla/triton_mla.py::_triton_mla_prefill`
- FlashInfer comparison: `custom_ops/mla/flashinfer_mla.py` (uses FlashInfer CUDA kernels for both phases)
