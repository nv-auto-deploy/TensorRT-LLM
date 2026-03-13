# Porting Models to Sharding-Aware Custom Ops

## Purpose

This document provides step-by-step instructions for porting an existing
AutoDeploy custom model (`models/custom/modeling_*.py`) to use explicit
sharding hint ops (`models/custom/new_sharding/modeling_*.py`).

The result is a model whose FX graph is a complete, self-contained specification
of "how this model should be sharded." The `apply_sharding_hints` transform
reads the hints together with a runtime `DistConfig` to apply deterministic,
node-local sharding -- no pattern matching, no heuristics.

## Reference Examples

Study these before porting a new model:

| Original | Sharded | Layer types |
|----------|---------|-------------|
| `models/custom/modeling_nemotron_h.py` | `new_sharding/modeling_nemotron_h.py` | Mamba SSM, Attention, SwiGLU MLP, MoE |
| `models/custom/modeling_qwen3_5_moe.py` | `new_sharding/modeling_qwen3_5_moe.py` | GatedDeltaNet, Gated Attention, SwiGLU MLP, MoE |

______________________________________________________________________

## Available Sharding-Aware Custom Ops

| Op | Sharding hints | When to use |
|----|---------------|-------------|
| `torch.ops.auto_deploy.torch_linear_simple` | `tp_mode: str`, `output_sizes: List[int]`, `tp_min_local_shape: int` | Replace every `nn.Linear` / `self.proj(x)` call |
| `torch.ops.auto_deploy.view` | `tp_scaled_dim: int` | Replace `.view()` / `.reshape()` where a dimension contains a concrete head count that scales with TP |
| `torch.ops.auto_deploy.split_with_sizes` | `shardable: bool` | Replace `torch.split` / `torch.split_with_sizes` after a colwise-sharded projection |
| `torch.ops.auto_deploy.all_reduce` | *(none)* | Insert after every rowwise projection. Identity when `world_size=1`; real `dist.all_reduce` when sharded |
| `torch.ops.auto_deploy.torch_causal_conv1d` | `shardable: bool`, `output_sizes: List[int]` | Already used in model code; add sharding hints |
| `torch.ops.auto_deploy.torch_ssm` | `shardable: bool` | Already used in Mamba models; add sharding hint |
| `torch.ops.auto_deploy.torch_gated_delta_rule` | `shardable: bool` | Already used in GatedDeltaNet models; add sharding hint |
| `torch.ops.auto_deploy.torch_rmsnorm_gated` | `tp_mode: str` | Gated RMSNorm whose weight scales with TP (e.g., Mamba norm) |
| `torch.ops.auto_deploy.torch_mla` | `shardable: bool` | MLA attention op; when shardable, `_apply_hint_mla` shards `kv_b_proj_weight` (arg\[4\]) colwise |
| `torch.ops.auto_deploy.torch_moe` | *(none)* | Already used in MoE models; `apply_sharding_hints` handles EP/TP automatically |

### Hint parameter reference

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `tp_mode` | `str` | `"none"` | `"colwise"` = shard weight dim 0. `"rowwise"` = shard weight dim 1. `"none"` = no sharding. |
| `output_sizes` | `Optional[List[int]]` | `None` | Fused weight group sizes for proportional column sharding (e.g., `[key_dim, key_dim, value_dim]`). |
| `tp_min_local_shape` | `int` | `1` | Minimum output size per rank. Used for GQA where `num_kv_heads < tp_size` (set to `head_dim`). |
| `tp_scaled_dim` | `int` | `-1` | Index of the shape dimension that scales with TP. `-1` means no scaling. `apply_sharding_hints` replaces `shape[tp_scaled_dim]` with `-1` (inferred). |
| `shardable` | `bool` | `False` | When True, `apply_sharding_hints` shards the op's weights/parameters along the head dimension. |

______________________________________________________________________

## Step-by-Step Porting Procedure

### Step 1: Copy the source file

```
cp models/custom/modeling_foo.py models/custom/new_sharding/modeling_foo.py
```

### Step 2: Update the module docstring and add imports

Add at the top of the file:

```python
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 -- register all ops
```

Add sharding control flags (one per layer family):

```python
SHARD_ATTENTION = True
SHARD_MLP = True
SHARD_SSM = True  # or SHARD_GDN, etc.
```

These flags let you toggle sharding per layer type during debugging.
Use `_s = SHARD_ATTENTION` as a local alias in each forward method.

### Step 3: Replace linear projections

For every `self.proj(x)` or `nn.Linear` call in the forward methods:

```python
# BEFORE
output = self.proj(hidden_states)

# AFTER
output = torch.ops.auto_deploy.torch_linear_simple(
    hidden_states,
    self.proj.weight,
    self.proj.bias,              # None if no bias
    tp_mode="colwise" if _s else "none",  # or "rowwise"
)
```

**Rules for choosing `tp_mode`:**

- **Opening projections** (Q, K, V, gate, up, in_proj) -> `"colwise"`
- **Closing projections** (O, down, out_proj) -> `"rowwise"`
- **Tiny projections** (shared_expert_gate with output_dim=1) -> `"none"` (cannot shard)
- **Latent projections** (MLA q_a_proj, kv_a_proj) -> keep replicated or use `"gather"`

**Fused weights** -- when a single linear produces concatenated outputs that are
later split (e.g., QKV fused, or Mamba in_proj = \[gate | conv_input | dt\]):

```python
output = torch.ops.auto_deploy.torch_linear_simple(
    x, self.in_proj.weight, self.in_proj.bias,
    tp_mode="colwise" if _s else "none",
    output_sizes=[gate_dim, conv_dim, dt_dim] if _s else None,
)
```

`output_sizes` tells the sharder to split the output dimension proportionally
across ranks, preserving the group structure.

**GQA (num_kv_heads \< num_q_heads)** -- K/V projections need:

```python
key = torch.ops.auto_deploy.torch_linear_simple(
    x, self.k_proj.weight, self.k_proj.bias,
    tp_mode="colwise" if _s else "none",
    tp_min_local_shape=self.head_dim,
)
```

`tp_min_local_shape=head_dim` ensures each rank keeps at least one full head,
enabling partial replication when `num_kv_heads < tp_size`.

### Step 4: Replace split / chunk operations

After a colwise-sharded fused projection, replace `torch.split` with:

```python
# BEFORE
gate, up = torch.split(projected, [gate_dim, up_dim], dim=-1)

# AFTER
gate, up = torch.ops.auto_deploy.split_with_sizes(
    projected, [gate_dim, up_dim], dim=-1, shardable=_s,
)
```

When `shardable=True`, `apply_sharding_hints` divides all split sizes by `tp_size`.

### Step 5: Replace view / reshape with concrete head counts

**Critical rule**: during `torch.export`, every `-1` in `.view()` / `.reshape()`
gets concretized to a concrete integer. After TP sharding changes tensor sizes,
these concrete values become wrong. Any reshape dimension that scales with TP
**must** use `auto_deploy::view`.

```python
# BEFORE
key = key_proj_output.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

# AFTER
key = torch.ops.auto_deploy.view(
    key_proj_output,
    [bsz, seq_len, self.num_kv_heads, self.head_dim],
    tp_scaled_dim=2 if _s else -1,
)
```

`tp_scaled_dim=2` tells the sharder to replace `shape[2]` with `-1` (inferred)
so the view adapts to the sharded tensor size.

**When NOT to use `auto_deploy::view`:**

- Flattening to 2D: `x.reshape(-1, x.shape[-1])` -- safe, no head count
- Flattening heads back: `x.reshape(bsz, seq_len, -1)` -- safe IF the input
  tensor already has the correct sharded shape (check that no concrete head
  count is baked in upstream)

**When you MUST use `auto_deploy::view`:**

- Any reshape with a concrete `num_heads`, `num_kv_heads`, `num_v_heads` at
  position 2 (or any other position that scales with TP)
- Reshapes after norm that restore a 4D `[B, S, H, D]` shape

### Step 6: Insert all_reduce

After every rowwise projection, add:

```python
output = torch.ops.auto_deploy.torch_linear_simple(
    x, self.out_proj.weight, self.out_proj.bias,
    tp_mode="rowwise" if _s else "none",
)
if _s:
    output = torch.ops.auto_deploy.all_reduce(output)
```

**MoE shared expert exception**: when the shared expert MLP is inside a MoE
block, defer all_reduce to the merge point:

```python
class SharedExpertMLP(nn.Module):
    def __init__(self, ..., add_all_reduce=True):
        self.add_all_reduce = add_all_reduce
        ...

    def forward(self, x):
        # gate/up colwise, down rowwise (no all_reduce here if deferred)
        ...
        if self.tp_sharded and self.add_all_reduce:
            down = torch.ops.auto_deploy.all_reduce(down)
        return down

class MoEBlock(nn.Module):
    def __init__(self, ...):
        self.shared_expert = SharedExpertMLP(..., add_all_reduce=False)

    def forward(self, x):
        routed_out = torch.ops.auto_deploy.torch_moe(...)
        shared_out = self.shared_expert(x)
        out = routed_out + shared_out
        if SHARD_MLP:
            out = torch.ops.auto_deploy.all_reduce(out)
        return out
```

### Step 7: Handle special layer-specific ops

**Conv1d** (Mamba, GatedDeltaNet) -- add hints:

```python
conv_out = torch.ops.auto_deploy.torch_causal_conv1d(
    x, self.conv1d.weight, self.conv1d.bias,
    self.conv1d.stride[0], self.conv1d.padding[0],
    self.conv1d.dilation[0], self.conv1d.groups,
    self.conv1d.padding_mode,
    shardable=_s,
    output_sizes=[key_dim, key_dim, value_dim] if _s else None,
)
```

**SSM** (Mamba) -- add `shardable`:

```python
y = torch.ops.auto_deploy.torch_ssm(
    ..., shardable=_s,
)
```

**GatedDeltaNet** -- add `shardable`:

```python
out = torch.ops.auto_deploy.torch_gated_delta_rule(
    query, key, value, g, beta, shardable=_s,
)
```

**Gated RMSNorm** (Mamba) -- use custom op with `tp_mode`:

```python
out = torch.ops.auto_deploy.torch_rmsnorm_gated(
    x, self.norm.weight, gate, self.norm.eps, group_size,
    tp_mode="colwise" if _s else "none",
)
```

Note: if the norm weight has constant size (e.g., `head_v_dim` that does not
scale with TP), keep it as a plain PyTorch module -- no custom op needed.

### Step 8: Handle MoE

`torch_moe` is sharded automatically by `apply_sharding_hints` (EP partitioning,
TP weight sharding, expert ID localization). No changes needed to the `torch_moe`
call itself.

The routed expert weights (gate_proj, up_proj, down_proj per expert) are passed
as lists and sharded by the `_apply_hint_moe` handler.

### Step 9: Register the new model

In `models/custom/__init__.py`, add an override import:

```python
from .new_sharding.modeling_foo import FooForCausalLM  # noqa: F811
```

The `# noqa: F811` suppresses the redefinition warning. The last import wins,
so the new_sharding version replaces the original.

### Step 10: Create YAML config

Create a YAML config file at `examples/auto_deploy/new_sharding/<model_family>/<model>_sharding_poc.yaml`.
This is REQUIRED -- without it the model cannot be tested.

Use this template (adapted from the working Qwen3.5 config at
`examples/auto_deploy/new_sharding/qwen/qwen3_5_moe_sharding_poc.yaml`):

```yaml
model: org/Model-Name
args:
  world_size: 2
  runtime: trtllm
  compile_backend: torch-cudagraph
  max_seq_len: 512
  max_num_tokens: 512
  max_batch_size: 8
  enable_chunked_prefill: true
  model_factory: AutoModelForCausalLM
  kv_cache_config:
    enable_block_reuse: false
    free_gpu_memory_fraction: 0.95
    tokens_per_block: 128
  skip_loading_weights: false
  model_kwargs:
    torch_dtype: bfloat16
  transforms:
    export_to_gm:
      num_moe_experts_for_export: 2    # REQUIRED for models with many experts (>64)
    detect_sharding:
      stage: sharding
      enabled: false                   # MUST disable legacy sharding
    sharding_transform_executor:
      stage: sharding
      enabled: false                   # MUST disable legacy executor
    apply_sharding_hints:
      stage: sharding
      enabled: true                    # Enable new hint-driven sharding
      run_shape_prop: true
      allreduce_strategy: NCCL
    gather_logits_before_lm_head:
      enabled: true
```

Key points:

- `detect_sharding` and `sharding_transform_executor` MUST be disabled
- `apply_sharding_hints` MUST be enabled with `run_shape_prop: true`
- Add `num_moe_experts_for_export: 2` under `export_to_gm` for MoE models
- Add model-specific transforms (e.g., `fuse_mamba_a_log`, `insert_cached_ssm_attention`)
  as needed -- check the model's existing config for these

Run with:

```bash
export HF_HOME=/path/to/hf/cache
cd examples/auto_deploy
python build_and_run_ad.py --yaml-extra new_sharding/<family>/<model>_sharding_poc.yaml
```

______________________________________________________________________

## Layer-Specific Sharding Patterns

### Attention (standard or gated)

```
q_proj  -> colwise (+ tp_min_local_shape for GQA)
k_proj  -> colwise (+ tp_min_local_shape for GQA)
v_proj  -> colwise (+ tp_min_local_shape for GQA)
view    -> tp_scaled_dim=2 (head count dimension)
o_proj  -> rowwise
          + all_reduce
```

If Q projection is fused with a gate (e.g., Qwen3.5 `q_proj` outputs `2 * num_heads * head_dim`):
the weight is interleaved per-head `[q_h0, g_h0, q_h1, g_h1, ...]`, so plain
colwise sharding is correct -- no `output_sizes` needed.

### SwiGLU MLP

```
gate_proj -> colwise
up_proj   -> colwise
down_proj -> rowwise
              + all_reduce
```

### Mamba / SSM

```
in_proj   -> colwise + output_sizes=[gate, conv_input, dt]
split     -> shardable (sizes scale with TP)
conv1d    -> shardable + output_sizes=[hidden, B, C]
split     -> shardable
view      -> tp_scaled_dim=2 (head count)
torch_ssm -> shardable (A, D, dt_bias sharded by handler)
norm      -> tp_mode="colwise" (if weight scales with TP)
out_proj  -> rowwise + all_reduce
```

### GatedDeltaNet

```
in_proj_qkv -> colwise + output_sizes=[key_dim, key_dim, value_dim]
in_proj_z   -> colwise
in_proj_b   -> colwise
in_proj_a   -> colwise
conv1d      -> shardable + output_sizes=[key_dim, key_dim, value_dim]
split       -> shardable
view        -> tp_scaled_dim=2 (Q, K, V, Z head reshapes)
torch_gated_delta_rule -> shardable (A_log, dt_bias sharded by handler)
norm        -> replicated (constant head_v_dim, plain PyTorch module)
view        -> tp_scaled_dim=2 (post-norm reshape back to [B,S,H,D])
out_proj    -> rowwise + all_reduce
```

### MoE + Shared Expert

```
router      -> replicated (not sharded)
torch_moe   -> automatic (EP/TP by apply_sharding_hints)
shared_expert:
  gate_proj -> colwise
  up_proj   -> colwise
  down_proj -> rowwise (NO all_reduce here)
shared_expert_gate -> replicated (output dim=1)
merge: routed + shared -> all_reduce
```

### MLA (DeepSeek)

MLA uses `torch_mla` which absorbs `kv_b_proj_weight` internally. Do NOT
decompose `torch_mla` into explicit `kv_b_proj` + `torch_attention` -- the
decomposition introduces `expand` ops with concrete `num_heads` that break
after TP sharding.

Instead, keep `torch_mla` and pass `shardable=True`. The `_apply_hint_mla`
handler shards `kv_b_proj_weight` (arg\[4\]) colwise along the head dimension.

```
q_a_proj    -> tp_mode="none" (replicated latent projection)
q_a_layernorm -> unchanged
q_b_proj    -> tp_mode="colwise" (shard by num_heads)
kv_a_proj   -> tp_mode="none" (replicated latent projection)
kv_a_layernorm -> unchanged
torch_mla   -> shardable=True (kv_b_proj_weight sharded by _apply_hint_mla)
view        -> tp_scaled_dim=2 for num_heads (Q reshape only)
o_proj      -> tp_mode="rowwise" + all_reduce
```

The Q split into `q_nope` and `q_pe` is on the LAST dim (head_dim), not the
head count dim, so it does NOT need `auto_deploy::split_with_sizes`.

______________________________________________________________________

## Common Pitfalls

1. **Forgetting `auto_deploy::view` for reshapes.** During `torch.export`, every
   `-1` in `.view()` / `.reshape()` is resolved to a concrete number. After
   sharding changes tensor sizes, these concrete values cause shape mismatches.
   Replace every reshape that has a head count at any position with `auto_deploy::view`.

1. **Sharding tiny projections.** Projections with output_dim=1 (like
   `shared_expert_gate`) cannot be sharded. Use `tp_mode="none"`.

1. **Wrong all_reduce count in MoE.** Use exactly ONE `all_reduce` at the merge
   point of routed + shared expert outputs. Do NOT add separate all_reduces for
   the shared expert and the routed expert path.

1. **Cross-layer parameter contamination.** When implementing `_apply_hint_*`
   handlers that use `get_source_nodes()` to find parameter ancestors, use the
   `allowed_ops` parameter to restrict traversal to elementwise ops only.
   Without this, the traversal crosses layer boundaries through residual
   connections and shards parameters from earlier layers.

1. **Missing `num_moe_experts_for_export: 2` in YAML.** Models with many experts
   (e.g., 128 or 256) hang during `torch.export` without this setting.

1. **Do NOT decompose custom ops that absorb weights.** Some custom ops like
   `torch_mla` take weight tensors as arguments and perform the linear projection
   internally. Do NOT decompose these into explicit `torch_linear_simple` +
   downstream ops -- the decomposition introduces `expand`/`view` operations
   with concrete `num_heads` that get baked into the FX graph and break after
   TP sharding. Instead, add `shardable=True` to the op and let the
   corresponding `_apply_hint_*` handler shard the weight.

1. **Interleaved vs contiguous fused weights.** If a fused weight is interleaved
   per-head-group (e.g., Qwen3Next `in_proj_qkvz`), plain colwise sharding
   works -- no `output_sizes` needed. If the weight is contiguously concatenated
   (e.g., Qwen3.5 `in_proj_qkv = [all_Q | all_K | all_V]`), you MUST provide
   `output_sizes` for proportional splitting.

______________________________________________________________________

## Validation Checklist

1. **world_size=1**: Run unsharded. `apply_sharding_hints` skips when `world_size < 2`.
   Verify output matches the original model.
1. **world_size=2**: Basic TP sharding. Check for shape mismatches, assertion
   errors, or garbage output.
1. **world_size=8**: Full TP. Verify coherent output and no OOM.
1. **Compare with legacy**: Run the same model on `main` branch with
   `detect_sharding` + manual TP plan. Compare output quality.
1. **Check node count**: The `apply_sharding_hints` log prints
   `N nodes processed`. Verify this matches your expectation (count all
   shardable ops in the model).
