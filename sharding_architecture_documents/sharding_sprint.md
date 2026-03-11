# Nemotron-H New Sharding PoC Sprint

**Status:** In Progress
**Branch:** `gk/nemotron-h-new-sharding-poc`
**Started:** 2026-03-09
**Target:** Runnable Nemotron Nano 31B with hint-driven sharding (no pattern matching)

---

## Progress

| Stage | Status | Notes |
|-------|--------|-------|
| New custom ops (view, split_with_sizes, all_reduce) | Done | |
| Add tp_mode to conv1d, ssm, norm (incl. triton variant) | Done | |
| Model rewrite | Done | |
| apply_sharding_hints scaffold | Done | |
| Stage I: Replicated validation | Done | Model loads, exports, compiles. Runtime SSM cache error is pre-existing. |
| Stage II: TP linear + auxiliaries | Done | Implemented in apply_sharding_hints |
| Stage III: all_reduce | Done | Implemented in apply_sharding_hints |
| DistConfig refactor | Done | Decoupled auto_deploy from tensorrt_llm.mapping.Mapping |
| ShardableOp enum + dispatch | Done | Clean dispatch pattern in apply_sharding_hints |
| Stage IV: MoE EP | Done | Expert partitioning, ID localization, explicit all_reduce at merge point |
| View -1 inference | Done | Replace tp_scaled_dim with -1 for robust partial replication |
| min_local_shape for GQA | Done | K/V projections handle num_kv_heads < tp_size |
| MoE all_reduce placement fix | Done | Single all_reduce at shared+routed merge, not per-branch |
| Quantized linear ops in ShardableOp | Done | All FP8/NVFP4/INT4 ops classified as LINEAR |
| Robust linear handler | Done | get_source_nodes for quantized weight traversal |
| FP8 validation | In Progress | Testing FP8 checkpoint loading + sharding |

---

## Design Decisions

- Only 5 types of shardable custom ops: **linear**, **view**, **split_with_sizes**, **all_reduce**, plus hint-annotated versions of **conv1d**, **ssm**, and **norm** for Mamba2
- No custom sharding op for BMM (always translated to torch_moe) or generic collectives (only all_reduce needed)
- `torch_moe` sharding is fully determined by `DistConfig` -- `apply_sharding_hints` injects `a2a_size`/`a2a_rank` (0 = replicated, >0 = EP-aware routing)
- Auxiliary ops (view, split_with_sizes) use only `tp_scaled_dim` or `tp_scale_sizes` -- no redundant `tp_mode`
- MoE latent projections (`fc1_latent_proj`, `fc2_latent_proj`) are **replicated**, not sharded

## Sharding-Aware Custom Ops

| Op | Hint params | Status |
|----|------------|--------|
| `auto_deploy::torch_linear_simple` | `tp_mode`, `output_sizes`, `tp_min_local_shape` | Exists |
| `auto_deploy::view` | `tp_scaled_dim: int = -1` | New |
| `auto_deploy::split_with_sizes` | `tp_scale_sizes: bool = False` | New |
| `auto_deploy::all_reduce` | (none -- identity when unsharded, real dist op when sharded) | New |
| `auto_deploy::torch_causal_conv1d` | `tp_mode: str = "none"` | Add hint |
| `auto_deploy::torch_ssm` | `tp_mode: str = "none"` | Add hint |
| `auto_deploy::torch_rmsnorm_gated` | `tp_mode: str = "none"` | Add hint |

## Key Files

- Model: `tensorrt_llm/_torch/auto_deploy/models/custom/new_sharding/modeling_nemotron_h.py`
- New ops: `tensorrt_llm/_torch/auto_deploy/custom_ops/sharding_ops.py`
- DistConfig: `tensorrt_llm/_torch/auto_deploy/utils/dist_config.py`
- ShardableOp enum: `tensorrt_llm/_torch/auto_deploy/utils/node_utils.py`
- Transform: end of `tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`
- Config: `examples/auto_deploy/nemotron_sharding_poc.yaml`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-09 | Sprint started. Branch created. |
| 2026-03-10 | Stage I complete. Fixed aliasing in view/split ops, added tp_mode to triton_rmsnorm_gated. |
| 2026-03-10 | Stage II+III complete. Full TP sharding + all_reduce in apply_sharding_hints. |
| 2026-03-10 | DistConfig refactor. New DistConfig class replaces Mapping in sharding/collectives transforms. |
| 2026-03-10 | ShardableOp enum + dispatch dict refactor. Cleaner node classification and action dispatch. |
| 2026-03-10 | MoE EP sharding implemented. Expert ID localization + weight partitioning. |
| 2026-03-10 | Fixed MoE all_reduce: single explicit node at merge point, not per-branch. |
| 2026-03-10 | Fixed view sharding: use -1 for inferred dim, handles partial replication. |
| 2026-03-10 | Added min_local_shape to K/V projections for GQA (num_kv_heads < tp_size). |
| 2026-03-10 | BF16 accuracy test PASSED on 2 GPUs. 4 GPU test pending (running). |
| 2026-03-10 | Added all quantized linear ops to ShardableOp.LINEAR dispatch. |
| 2026-03-10 | Robust linear handler with get_source_nodes for quantized weight chains. |
