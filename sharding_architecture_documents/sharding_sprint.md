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
| Stage II: TP linear + auxiliaries | Pending | |
| Stage III: all_reduce | Pending | |
| Stage IV: MoE EP | Pending | |

---

## Design Decisions

- Only 5 types of shardable custom ops: **linear**, **view**, **split_with_sizes**, **all_reduce**, plus hint-annotated versions of **conv1d**, **ssm**, and **norm** for Mamba2
- No custom sharding op for BMM (always translated to torch_moe) or generic collectives (only all_reduce needed)
- `torch_moe` sharding is fully determined by `Mapping` -- `apply_sharding_hints` injects `a2a_size`/`a2a_rank` (0 = replicated, >0 = EP-aware routing)
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
- Transform: end of `tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`

---

## Changelog

| Date | Change |
|------|--------|
| 2026-03-09 | Sprint started. Branch created. |
| 2026-03-10 | Stage I complete. Fixed aliasing in view/split ops, added tp_mode to triton_rmsnorm_gated. |
