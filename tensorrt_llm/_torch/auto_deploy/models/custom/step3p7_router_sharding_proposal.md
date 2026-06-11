<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Step-3.7 Router Sharding Proposal

## Current Path

`origin/jonghyun/step37-perf` keeps the Step-3.7 fp32 router gate replicated:

```python
router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())
```

This means every TP rank computes full router logits with shape `[T, 288]`, then the fused router
kernel performs top-8 over all 288 experts locally. There is no distributed router collective and
selected expert IDs are already global.

The CuteDSL and optimized Triton router integrations should preserve this contract:

- `router_logits`: `[T, 288]`, fp32
- `router_bias`: `[288]`, fp32 or bf16, promoted to fp32 in the kernel
- `top_k`: 8
- `routed_scaling_factor`: 3.0
- output weights: bf16
- output expert IDs: int64 global expert IDs

## Sharded Router Alternative

A separate optimization could column-shard the router gate across TP ranks:

```python
router_logits_local = torch_linear_simple(..., tp_mode="colwise", layer_type="moe")
```

For TP8, each rank would produce `[T, 36]`. Correct global routing would then require:

1. Local top-8 over the rank's 36 experts.
1. Convert local expert IDs to global IDs with `rank * 36 + local_id`.
1. Gather each rank's 8 candidates.
1. Final top-8 over the gathered 64 candidates.
1. Normalize and scale the unbiased probabilities for those final experts.

This is exact because any global top-8 expert must be in the local top-8 of its owning shard.

## Performance Assessment

The sharded alternative saves router GEMM compute. For hidden size 4096 and 288 experts, TP8
reduces each rank's router projection from 288 output columns to 36 output columns.

It also adds costs that the current replicated path does not pay:

- At least one candidate all-gather per MoE layer.
- Extra local-topk and finalize-topk kernel launches unless fused with the collective path.
- Synchronization latency on the routed critical path.
- Candidate traffic. A straightforward representation gathers 8 candidates per rank containing
  global index, biased score, and unbiased probability. That is about `T * 8 * 16` bytes produced
  per rank and `T * 64 * 16` bytes materialized after gather.

Decode is the riskiest case for sharding: `T` is usually small, so the current `[T, 288]` router
GEMM is tiny and often launch-bound, while the added all-gather latency is paid for every MoE layer.
This is likely slower unless the collective is hidden or fused into a broader communication point.

Prefill is less clear. For large `T`, the saved fp32 router projection work may offset the
candidate gather and extra kernels, but this needs end-to-end measurement because the routed expert
work and existing collectives can dominate.

## Recommendation

Do not include router-gate sharding in the kernel integration commits. Keep the integrated kernels
on the existing `[T, 288]` replicated-router contract. Treat `[T, 36]` router sharding as a separate
benchmark-driven experiment with its own correctness tests and latency breakdown for prefill and
decode.
