<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash AutoDeploy decode — perf-agent results

**Date:** 2026-06-29
**Workload:** DSV4-Flash, AutoDeploy backend, TP8 / B200, decode (cudagraph), bs=1
**Source of truth:** Locally captured Nsight Systems decode-cudagraph traces
(`0629_decode_cg_10layers_original` vs `0629_decode_cg_10layers`; `0629_decode_cg_original` vs `0629_decode_cg`)

---

## Headline: ~10× faster decode

| metric | ORIGINAL | OPTIMIZED | **speedup** |
|---|---|---|---|
| **10-layer** decode GPU-kernel-time (whole trace) | 7,329 ms | **757 ms** | **9.68×** |
| **2-layer** decode GPU-kernel-time | 1,481 ms | **145 ms** | **10.19×** |
| kernel launches (10L) | 777,099 | 353,661 | 2.2× fewer |
| kernel launches (2L) | 104,880 | 35,760 | 2.9× fewer |
| distinct kernels (10L) | 134 | 126 | — |

The **9.68× / 10.19×** GPU-time reductions match the observed wall-clock
(96 ms → 11 ms at 10 layers; 20 ms → 2 ms at 2 layers). This is real GPU work
eliminated — not a host-gap artifact — delivered fully autonomously by the
perf-agent across **28 landed wins**, each verified for correctness (byte-exact
or numerically-checked) before landing.

---

## Where the 10× comes from

The agent recovered **6.57 s** of per-trace GPU time (10-layer). Attribution to the
landed optimizations, by kernel family:

| Δ saved (ms) | % of total | kernel family | landed optimization(s) |
|---:|---:|---|---|
| **5,651** | 86.0% | elementwise sea (605k → 285k inst) | **MoE MXFP4 weight-dequant removal** — selective decode dequant (idea_0002, +18.87%) then **trtllm-gen W4A16 from-routing runner** (idea_0023, +40.31%) |
| **541** | 8.2% | `at::native::` index / gather | the FAT `aten::index` MoE-dequant gather, removed with the dequant chain |
| **126** | 1.9% | reduce | **HC 20-iteration sinkhorn loop fused** into one kernel (idea_0028) |
| **105** | 1.6% | tf32 cutlass GEMM (→ 0) | HC fp32 compressor GEMM fused away (idea_0036) |
| **97** | 1.5% | nccl AllReduce (3040 → 1600 inst) | dropped the f32 index AllReduce (idea_0077) + collinear AllReduce fusion (idea_0047) |
| **59** | 0.9% | fp8 block-matmul | block-FP8 GEMM autotune + split-K decode GEMV (idea_0004 / 0025 / 0063) |

Top-18 kernel families account for 6,634 ms ≈ the full 6,572 ms saved.

### One-line story

**~86% of the 10× is the MoE MXFP4 fp32-dequant chain being deleted.** The original
decode dequantized **all 32 local experts every step** into fp32 and ran a reference
bmm — the giant 512M+256M `index` / elementwise pair that dominated the trace. The
agent first restricted dequant to only the routed slots, then replaced the whole
chain with the proven trtllm-gen W4A16 grouped-GEMV runner. That single lever
collapses the 6.1 s elementwise sea to 0.44 s.

The remaining ~14% is the tail the agent then mined: the HC (hierarchical-composition)
sinkhorn 20-iteration loop fused into one kernel, the HC fp32 compressor GEMM removed,
the LM-head gather-sharded and its bf16→fp32 recast hoisted, RoPE / hadamard / SwiGLU
chains fused, the redundant f32 AllReduce dropped, and the block-FP8 GEMMs re-tuned with
a split-K decode path.

---

## Verification

- **Method:** absolute GPU-kernel-time from `nsys stats --report cuda_gpu_kern_sum`,
  summed over all decode steps in each cudagraph trace, original vs optimized.
- The GPU-time speedup (9.68× / 10.19×) and the wall-clock speedup (8.7× / 10×) agree,
  confirming the win is GPU work removed. Kernel-launch count also dropped ~2–3×, so
  host/launch overhead fell as well (a secondary contributor on top of the GPU-time win).

### Why the per-idea agent metric showed small %s, not "10×"

The agent's fast-mode proxy gated on **relative op-type GPU-time shares** against a
**moving baseline** (each landed win became the next idea's baseline). So it reported
each idea's *incremental* delta (+40%, +19%, +8% …) and never multiplied them into a
cumulative — the ~10× is the *product* of the 28 increments. Relative shares also can't
express an absolute collapse: deleting the single largest chain just renormalizes the
percentages. The absolute start-vs-end measurement (these nsys traces) is the faithful
total, and it is **~10×**.
