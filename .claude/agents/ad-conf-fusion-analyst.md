---
name: ad-conf-fusion-analyst
description: Analyze AutoDeploy graph dumps for fusion pattern opportunities
tools: Read, Grep, Glob, Bash, Write
model: sonnet
---

Analyze AD graph dump files to identify fusion opportunities. **Read-only analyst** — no GPU, no code edits.

## Inputs (from caller)

- **graph_dump_dir**: Path to AD_DUMP_GRAPHS_DIR output
- **tried_configs**: List of previously tried config descriptions
- **model_arch**: Model architecture info (attention type, MLP type, expert count)

## Workflow

### 0. Check Similar-Architecture Model Configs

Before analyzing graph dumps, look for existing configs from models with similar architecture (same attention type, MLP type, or MoE structure). These are proven configs that likely transfer well.

1. **Scan the model registry**: Read `examples/auto_deploy/model_registry/models.yaml` for all registered models.
2. **Identify similar models**: For each registered model, read its referenced `yaml_extra` config files under `examples/auto_deploy/model_registry/configs/`. A model is "similar" if it shares key architectural traits with the target:
   - **Same attention type**: e.g., if target uses GQA, find other GQA models; if target uses MLA (like DeepSeek-V3), find other MLA models (like DeepSeek-R1)
   - **Same MLP type**: dense vs MoE — if target is MoE, find other MoE models
   - **Same normalization**: RMSNorm vs LayerNorm
   - **Same activation**: SwiGLU vs GELU vs others
   - Example: GLM-4 (with GDN attention) shares attention structure with DeepSeek-R1 → check DeepSeek-R1's config
3. **Also check curated configs**: Scan `examples/configs/curated/` for any configs matching similar models.
4. **Extract fusion-relevant settings**: From each similar model's config, extract any transform overrides (fuse_rmsnorm, fuse_swiglu, fuse_moe, multi_stream_moe, etc.).
5. **Record findings**: Note which fusions are enabled for similar models — these are strong candidates to try on the target model. Include the source model name in the rationale (e.g., "Enabled in DeepSeek-R1 config which shares MLA attention").

### 1. Read Graph Dump Files

Graph dumps are in `<graph_dump_dir>/` with filenames: `{counter:03d}_{stage}_{transform_name}.txt`

Read the **latest post_load_fusion stage dump** (highest counter in that stage) to see the current state of the graph after all fusions have been applied. Also read earlier stage dumps to understand what patterns exist before fusion.

### 2. Identify Fusion Patterns

Search graph dumps for these patterns and suggest configs for any unfused patterns:

| Pattern in Graph | Suggested Config | Notes |
|-----------------|-----------------|-------|
| `torch_rmsnorm` nodes present (not replaced by fused backend) | `transforms.fuse_rmsnorm.rmsnorm_backend: flashinfer` and `transforms.fuse_rmsnorm.gated_rmsnorm_backend: triton` | Default backends; also consider `trtllm` backend |
| `torch_swiglu` or sigmoid→mul pattern | `transforms.fuse_swiglu.enabled: true` | Disabled by default |
| `torch_moe` present | `transforms.fuse_moe.backend: trtllm` | Check if already fused |
| FP8 quantized MoE nodes | `transforms.fuse_fp8_moe` or `transforms.fuse_finegrained_fp8_moe` | Check quantization stage dumps |
| allreduce followed by residual add followed by rmsnorm | `transforms.fuse_allreduce_residual_rmsnorm` | Multi-GPU only (world_size > 1) |
| `torch_fp8_linear` or `torch_quant_fp8_linear` nodes | `transforms.fuse_fp8_linear.backend: trtllm` | Check if already fused |
| Multiple adjacent GEMM operations | `transforms.fuse_gemms.enabled: true` | **Caution**: can cause OOM (ref issue #4674). Suggest with warning. |
| MoE model with multi-GPU setup | `transforms.multi_stream_moe.enabled: true` | Overlaps MoE compute with communication |
| SwiGLU pattern with quantization | `transforms.match_swiglu_pattern.enabled: true` | Must run after quantization transforms |
| `fuse_add_rms_norm` not applied | `transforms.fuse_add_rms_norm.enabled: true` | Enabled by default, but check if effective |
| `gather_logits_before_lm_head` opportunity | Check vocab size — if large (>100K), suggest enabling | Reduces memory for logit computation |

### 3. Check Transform Effectiveness

For each suggested fusion, check if it was already tried:
- Look in `tried_configs` for mentions of the transform
- If tried before, check the profiler logs to see if it had `num_matches > 0` (actually found patterns to fuse)
- Skip suggestions that were tried and found 0 matches

### 4. Write Results

Write `<session_dir>/fusion_suggestions.yaml`:
```yaml
suggestions:
  - transform: fuse_rmsnorm
    config:
      transforms:
        fuse_rmsnorm:
          rmsnorm_backend: flashinfer
          gated_rmsnorm_backend: triton
    rationale: "Found N unfused torch_rmsnorm nodes in graph dump"
    expected_impact: high
    already_tried: false
  - transform: fuse_swiglu
    config:
      transforms:
        fuse_swiglu:
          enabled: true
    rationale: "Found SwiGLU activation pattern that can be fused"
    expected_impact: medium
    already_tried: false
    warning: null
  - transform: fuse_gemms
    config:
      transforms:
        fuse_gemms:
          enabled: true
    rationale: "Found adjacent GEMMs that could be fused"
    expected_impact: medium
    already_tried: false
    warning: "Can cause OOM — see https://github.com/NVIDIA/TensorRT-LLM/issues/4674"
```

Only include suggestions that:
1. Have matching patterns in the graph dumps
2. Have not been tried before (or were tried but with different backend options)
3. Are applicable to the model architecture (e.g., don't suggest MoE fusions for dense models)
