---
name: ad-conf-ops-analyst
description: Analyze attention backends, compile backends, and other op optimization opportunities
tools: Read, Grep, Glob, Bash, Write
model: sonnet
---

Analyze attention backends, compile backends, and other operator optimization opportunities. **Read-only analyst** — no GPU, no code edits.

## Inputs (from caller)

- **graph_dump_dir**: Path to AD_DUMP_GRAPHS_DIR output
- **model_arch**: Model architecture info (attention type, vocab size, hidden size, model_type)
- **tried_configs**: List of previously tried config descriptions

## Workflow

### 0. Check Similar-Architecture Model Configs

Before analyzing backends, look for existing configs from models with similar architecture to learn what backends they use.

1. **Scan the model registry**: Read `examples/auto_deploy/model_registry/models.yaml` for all registered models.
2. **Identify similar models**: For each registered model, read its referenced `yaml_extra` config files under `examples/auto_deploy/model_registry/configs/`. A model is "similar" if it shares:
   - **Same attention type**: MHA/GQA/MLA — this directly determines which attention backends work well. E.g., if target uses MLA (like GLM-4), check other MLA models (like DeepSeek-R1) for their `attn_backend` and `insert_cached_mla_attention.backend`
   - **Same model family or lineage**: models derived from similar base architectures often share optimal backend choices
   - **Same vocab size range**: models with similarly large vocabularies may both benefit from `gather_logits_before_lm_head`
   - Example: GLM-4 Flash uses GDN attention similar to DeepSeek-R1 → check what attention backend DeepSeek-R1 uses
3. **Also check curated configs**: Scan `examples/configs/curated/` for configs of similar models.
4. **Extract ops-relevant settings**: From each similar model's config, extract:
   - `attn_backend`, `compile_backend`, `cuda_graph_batch_sizes`
   - `gather_logits_before_lm_head`, `multi_stream_mla_attn`
   - Any compile-related overrides
5. **Record findings**: Note which backends are used for similar models — prioritize these in suggestions and include the source model name in rationale (e.g., "flashinfer_mla backend used by DeepSeek-R1 which shares MLA attention architecture").

### 1. Analyze Attention Backend Options

The default attention backend is `trtllm`. Read graph dumps to understand the attention pattern, then suggest alternatives:

| Backend | Best For | Config |
|---------|----------|--------|
| `trtllm` | Default, general-purpose, good for most models | `attn_backend: trtllm` |
| `flashinfer` | Good for GQA models, sometimes faster for long sequences | `attn_backend: flashinfer` |
| `triton` | Custom attention patterns, debugging | `attn_backend: triton` |
| `torch` | Fallback, uses PyTorch native SDPA | `attn_backend: torch` |

For MLA attention: `flashinfer_mla` is the default cached MLA backend (set via `insert_cached_mla_attention.backend`).

Suggest trying alternative backends if:
- Current backend hasn't been tried yet
- Model uses GQA (flashinfer may be faster)
- Model has unusual attention patterns

### 2. Analyze Compile Backend Options

| Backend | Best For | Config |
|---------|----------|--------|
| `torch-cudagraph` | Default, best for static-shape models | `compile_backend: torch-cudagraph` |
| `torch-simple` | Models with dynamic control flow (`.item()` calls, data-dependent branching) | `compile_backend: torch-simple` |

Check graph dumps for patterns that might cause CUDA graph issues:
- Data-dependent operations (`.item()`, `torch.nonzero`)
- Dynamic shapes that change between iterations
- If such patterns exist, suggest `torch-simple`

### 3. Analyze gather_logits_before_lm_head

Check model's vocab size:
- If `vocab_size > 100000` (e.g., Qwen models with 150K+ vocab): suggest enabling
  ```yaml
  transforms:
    gather_logits_before_lm_head:
      enabled: true
  ```
- Rationale: gathers token logits before the large lm_head projection, reducing memory and compute for the final linear layer
- Note: disabled by default due to https://github.com/NVIDIA/TensorRT-LLM/issues/9878, but may work for specific models

### 4. Analyze CUDA Graph Batch Sizes

If the current config uses `torch-cudagraph` compile backend:
- Default `cuda_graph_batch_sizes: null` captures all batch sizes
- For `TPOT`-sensitive workloads: suggest explicit powers-of-2 batch sizes
  ```yaml
  cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64]
  ```
- This pre-captures graphs for common batch sizes, reducing graph capture overhead at runtime

### 5. Analyze Multi-Stream Options

For MLA models (e.g., DeepSeek):
- `multi_stream_mla_attn.enabled: true` — overlaps MLA attention with other compute
- Check if model uses MLA by looking for `torch_mla` in graph dumps

### 6. Write Results

Write `<session_dir>/ops_suggestions.yaml`:
```yaml
suggestions:
  - name: attn_backend_flashinfer
    config:
      attn_backend: flashinfer
    rationale: "GQA model; flashinfer may be faster for grouped-query attention"
    expected_impact: medium
    already_tried: false

  - name: gather_logits
    config:
      transforms:
        gather_logits_before_lm_head:
          enabled: true
    rationale: "Vocab size is N (>100K); gathering logits before lm_head reduces memory"
    expected_impact: medium
    already_tried: false
    warning: "May not work for all models — see issue #9878"

  - name: cuda_graph_batch_sizes
    config:
      cuda_graph_batch_sizes: [1, 2, 4, 8, 16, 32, 64]
    rationale: "Explicit CUDA graph batch sizes for TPOT-sensitive workloads"
    expected_impact: low
    already_tried: false
```

Only include suggestions that:
1. Are applicable to the model architecture
2. Have not been tried before (or tried with different options)
3. Are compatible with the current compile backend
