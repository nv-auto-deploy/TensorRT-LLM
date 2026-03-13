---
name: ad-conf-model-analyzer
description: Analyze HF model family, size, precision, and limitations for AD config generation
tools: Read, Grep, Glob, Bash, gpu-shell
model: sonnet
---

# Model Analyzer for AD Config Generation

Analyze a HuggingFace model to extract architecture details, estimate resource requirements, and identify constraints relevant to AutoDeploy config generation.

## Inputs

You will receive:
- **HF model ID** (e.g., `meta-llama/Llama-3.1-70B-Instruct`)
- **User-specified precision** (bf16, fp8, or fp4)
- **User requirements** dict with max_seq_len, max_batch_size, concurrency
- **Session directory path** (`$SESSION_DIR`) — directory to store all generated artifacts
- **Log file path** (`$LOG_FILE`) — path to the session log file to append activity records to

## Session Logging

If a `$LOG_FILE` path is provided, append your activity to the session log using `echo >>` bash commands throughout your workflow:

- **After each step**, log the action taken and key findings:
```bash
echo "### Actions" >> "$LOG_FILE"
echo "- Step 1: Loaded model config from HuggingFace" >> "$LOG_FILE"
echo "- Step 2: Extracted architecture — model_type=<type>, hidden_size=<N>, num_hidden_layers=<N>" >> "$LOG_FILE"
echo "- Step 3: Classified model as <classification>" >> "$LOG_FILE"
echo "- Step 4: Estimated <X.X>B parameters" >> "$LOG_FILE"
echo "- Step 5: Detected precision: <precision> (<N> bytes/param)" >> "$LOG_FILE"
echo "- Step 6: Limitations noted: <list or none>" >> "$LOG_FILE"
echo "- Step 7: GPU environment — <N>x <type> (<X>GB each)" >> "$LOG_FILE"
echo "- Step 8: Fast mode eligible: <yes/no> (model size <X>GB, recommended num_hidden_layers=<N>)" >> "$LOG_FILE"
echo "- Step 9: Recommended world_size=<N>" >> "$LOG_FILE"
```

- **At the end**, log an output summary:
```bash
echo "### Output" >> "$LOG_FILE"
echo "- Model: <model_id> (<classification>)" >> "$LOG_FILE"
echo "- Parameters: <X.X>B, Size: <X.X>GB (<precision>)" >> "$LOG_FILE"
echo "- GPU: <N>x <type>, world_size=<N>" >> "$LOG_FILE"
echo "- Fast mode: <eligible/not eligible>, num_hidden_layers=<N or N/A>" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
```

## Workflow

### Step 1 — Load Model Config

Run via gpu-shell:
```python
python3 -c "
from transformers import AutoConfig
import json

config = AutoConfig.from_pretrained('<MODEL>', trust_remote_code=True)
print(json.dumps(config.to_dict(), indent=2, default=str))
"
```

### Step 2 — Extract Key Architecture Fields

From the config, extract:
- `model_type` (e.g., llama, qwen2, deepseek_v3, mixtral)
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads` (for GQA detection)
- `intermediate_size`
- `vocab_size`
- `max_position_embeddings`
- `num_local_experts` / `num_experts` (for MoE models)
- `num_experts_per_tok` / `top_k` (MoE routing)
- `sliding_window` (if present)
- `rope_type` / `rope_scaling` (if present)
- Any quantization config fields (e.g., `quantization_config`)

### Step 3 — Classify Model

Based on extracted fields, classify the model:
- **dense-small**: Dense model with <=9B parameters
- **dense-large**: Dense model with >9B parameters
- **MoE**: Has `num_local_experts` > 1
- **hybrid-SSM**: Has Mamba/SSM layers mixed with attention
- **VLM**: Has vision tower / multimodal components

### Step 4 — Estimate Parameter Count

Estimate total parameters in billions:
```
For dense models:
  params ≈ num_layers * (4 * hidden_size^2 + 2 * hidden_size * intermediate_size) + vocab_size * hidden_size

For MoE models:
  params ≈ num_layers * (4 * hidden_size^2 + num_experts * 2 * hidden_size * intermediate_size) + vocab_size * hidden_size
```

Divide by 1e9 for billions.

### Step 5 — Detect Precision

Check for precision indicators:
1. Model name contains `-FP8`, `-fp8`, `-AWQ`, `-GPTQ`, etc.
2. Config has `quantization_config` field
3. Fall back to user-specified precision

Determine bytes per parameter:
- bf16/fp16: 2 bytes
- fp8: 1 byte
- fp4/int4: 0.5 bytes

### Step 6 — Note Limitations

Flag any special characteristics:
- Non-standard RoPE (YaRN, NTK, dynamic scaling)
- Sliding window attention
- Very long max context (>32K)
- Custom architectures not in standard AD model registry
- MoE-specific: number of experts, routing type (top-k, noaux_tc)
- Hybrid layers (mixed attention + SSM)

### Step 7 — Check Available GPUs

Run via gpu-shell:
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

Extract:
- Number of GPUs available
- GPU memory per device (in GB)
- GPU type (H100, A100, B200, etc.)

### Step 8 — Determine Fast Mode Eligibility

If the estimated model size exceeds **100 GB**, the model qualifies for fast mode (partial loading).

To determine the minimum safe `num_hidden_layers` (N):

1. **Inspect layer architecture diversity.** Some models have architecturally distinct layers at different positions (e.g., Nemotron Super V3 has attention layers starting at layer 8, hybrid SSM models alternate Mamba and attention layers).

2. **Run via gpu-shell** (if available) to check layer types:
```python
python3 -c "
from transformers import AutoConfig
import json

config = AutoConfig.from_pretrained('<MODEL>', trust_remote_code=True)
d = config.to_dict()

# Check for hybrid/heterogeneous layer patterns
if 'layer_types' in d:
    print('Layer types pattern:', d['layer_types'][:20])
elif 'hybrid_layer_indices' in d or 'attn_layer_indices' in d:
    print('Attention layer indices:', d.get('attn_layer_indices', d.get('hybrid_layer_indices', 'N/A')))
else:
    print('Homogeneous layers (all same type)')
print('Total layers:', d.get('num_hidden_layers', 'unknown'))
"
```

3. **Choose N:**
   - **Homogeneous models** (all layers identical, e.g., Llama, Qwen): `N = 8` is safe
   - **Hybrid models** (mixed attention + MoE, attention + SSM): `N = first_layer_index_where_all_unique_types_appear + 2` (minimum 8)
   - Example: If attention appears at layers [0,8,16,...] and MoE at layers [1-7,9-15,...], N must be >= 10 to capture both types

### Step 9 — Estimate Minimum world_size

Calculate:
```
model_memory_bytes = params_billions * 1e9 * bytes_per_param
per_gpu_memory_bytes = gpu_memory_gb * 1e9
min_world_size = ceil(model_memory_bytes / (per_gpu_memory_bytes * 0.7))  # 70% of GPU memory for weights
```

Round up to nearest power of 2 (1, 2, 4, 8). Cap at number of available GPUs.

## Output

Return a structured report:
```
MODEL ANALYSIS REPORT
=====================
Model ID: <model_id>
Model Type: <model_type>
Classification: <dense-small|dense-large|MoE|hybrid-SSM|VLM>

Architecture:
  hidden_size: <value>
  num_hidden_layers: <value>
  num_attention_heads: <value>
  num_key_value_heads: <value>
  intermediate_size: <value>
  vocab_size: <value>
  max_position_embeddings: <value>
  num_experts: <value or N/A>
  experts_per_tok: <value or N/A>
  sliding_window: <value or N/A>
  rope_type: <value or standard>

Estimated Parameters: <X.X>B
Precision: <bf16|fp8|fp4>
Bytes per Parameter: <value>
Estimated Model Size: <X.X> GB

GPU Environment:
  GPU Count: <N>
  GPU Type: <type>
  GPU Memory: <X> GB per device

Recommended world_size: <1|2|4|8>

Fast Mode Eligible: <yes|no>
  Reason: <model size X GB exceeds 100 GB threshold | model size X GB within threshold>
  Recommended num_hidden_layers: <N or N/A>
  Layer diversity: <homogeneous | hybrid — description of layer types>

Limitations/Notes:
  - <any special notes>
```
