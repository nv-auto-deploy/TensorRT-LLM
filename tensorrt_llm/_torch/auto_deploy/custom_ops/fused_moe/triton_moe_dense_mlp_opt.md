# Triton MoE Dense MLP — Kernel Optimization Log

> **Kernel file:** `triton_moe_dense_mlp.py`
> **Owner:** see git log
> **Started:** 2026-03-20
> **GPU target:** TBD (to be filled after first run)

______________________________________________________________________

## 1. Kernel Overview

The file contains **two Triton kernels** used in the AutoDeploy path for GPT-OSS-style
dense MoE. The end-to-end op (`triton_moe_dense_mlp`) performs:

```
1. gate_up = bmm(hidden, gate_up_w) + gate_up_b          # torch.bmm
2. Interleaved split: gate = gate_up[..., ::2], up = gate_up[..., 1::2]
3. Clamp: gate = clamp(gate, max=limit), up = clamp(up, min=-limit, max=limit)
4. GLU: glu = gate * sigmoid(gate * alpha)                 # ← Triton kernel 1
5. Fused multiply: act_out = (up + 1) * glu                # ← Triton kernel 1
6. down_out = bmm(act_out, down_w) + down_b               # torch.bmm
7. Weighted sum over experts                               # ← Triton kernel 2
```

**Scope of optimization:** Triton kernels only (steps 2-5 and step 7). The BMMs
(steps 1 & 6) are handled by `torch.bmm` and out of scope.

### Kernel 1: `_fused_glu_activation_kernel`

- **Input:** `gate_up` tensor `[E*T, 2*I]` (interleaved)
- **Output:** `act_out` tensor `[E*T, I]`
- **Grid:** `(E*T,)` — one program per row
- **Work per program:** read 2\*I elements (strided: even/odd), compute clamp+GLU+multiply, write I elements

### Kernel 2: `_weighted_expert_sum_kernel`

- **Input:** `expert_out` tensor `[E, T, H]`, `routing_weights` `[T, E]`
- **Output:** `output` tensor `[T, H]`
- **Grid:** `(T,)` — one program per token
- **Work per program:** loop over E experts, accumulate weighted sum over H elements

______________________________________________________________________

## 2. Target Models & Benchmark Shapes

### Models

| Model | H | I | E | top-k | Source |
|-------|---|---|---|-------|--------|
| GPT-OSS-120B | 2880 | 2880 | 128 | 4 | [HF config](https://huggingface.co/openai/gpt-oss-120b) |
| GPT-OSS-20B | 2880 | 2880 | 32 | 4 | [HF config](https://huggingface.co/openai/gpt-oss-20b) |

> **Note:** H == I == 2880 (not a power of 2). `BLOCK_I` / `BLOCK_H` will be 4096.
> The dense MoE pattern (BMM over all experts) is currently GPT-OSS specific.

### Benchmark Configs

| ID | Model | E | T | H | I | Scenario |
|----|-------|---|---|---|---|----------|
| A1 | GPT-OSS-120B | 128 | 1 | 2880 | 2880 | Single-token decode |
| A2 | GPT-OSS-120B | 128 | 8 | 2880 | 2880 | Small batch decode |
| A3 | GPT-OSS-120B | 128 | 32 | 2880 | 2880 | Medium batch decode |
| A4 | GPT-OSS-120B | 128 | 128 | 2880 | 2880 | Large batch decode |
| A5 | GPT-OSS-120B | 128 | 512 | 2880 | 2880 | Prefill |
| B1 | GPT-OSS-20B | 32 | 1 | 2880 | 2880 | Single-token decode |
| B2 | GPT-OSS-20B | 32 | 32 | 2880 | 2880 | Medium batch decode |
| B3 | GPT-OSS-20B | 32 | 512 | 2880 | 2880 | Prefill |

______________________________________________________________________

## 3. Optimization Iterations

### Baseline (v0) — Current Implementation

**Kernel 1 config:** `BLOCK_I = next_power_of_2(2880) = 4096`, `num_warps=4`, `num_stages=3`
**Kernel 2 config:** `BLOCK_H = next_power_of_2(2880) = 4096`, `num_warps=4`, `num_stages=3`

| ID | Kernel 1 (μs) | Kernel 2 (μs) | Total Triton (μs) | E2E (μs) | Triton % of E2E |
|----|--------------|--------------|-------------------|----------|-----------------|
| A1 | 7.7 | 108.6 | 116.3 | 2130.8 | 5.5% |
| A2 | 13.8 | 113.5 | 127.3 | 2192.7 | 5.8% |
| A3 | 33.4 | 121.3 | 154.7 | 2370.9 | 6.5% |
| A4 | 107.4 | 131.1 | 238.5 | 2879.3 | 8.3% |
| A5 | 402.4 | 172.0 | 574.4 | 6751.8 | 8.5% |
| B1 | 7.5 | 31.3 | 38.8 | 562.9 | 6.9% |
| B2 | 13.6 | 34.2 | 47.8 | 617.2 | 7.7% |
| B3 | 107.1 | 56.0 | 163.1 | 1693.1 | 9.6% |

> **GPU:** NVIDIA H100 80GB HBM3
> **Triton:** 3.5.1
> **PyTorch:** 2.10.0a0+b4e4ee81d3.nv25.12
> **Dtype:** bfloat16

______________________________________________________________________

### Iteration 1 — TBD

**What changed:**

```
(describe the optimization)
```

**Results:**

| ID | Kernel 1 (μs) | Kernel 2 (μs) | Total Triton (μs) | Δ vs baseline |
|----|--------------|--------------|-------------------|---------------|
| A1 | | | | |
| A2 | | | | |
| A3 | | | | |
| A4 | | | | |
| A5 | | | | |
| B1 | | | | |
| B2 | | | | |
| B3 | | | | |

**Analysis:**

```
(what worked, what didn't, next steps)
```

______________________________________________________________________

## 4. Optimization Ideas Backlog

- \[ \] **2D tiling for kernel 1:** Current 1D block (BLOCK_I=4096) may be too wide for I=2880. Try tiling rows into multiple blocks.
- \[ \] **Vectorized loads for kernel 1:** Interleaved access (stride-2) is unfriendly to coalescing. Explore load-then-deinterleave.
- \[ \] **num_warps / num_stages sweep:** Hardcoded 4 warps / 3 stages — autotune across configs.
- \[ \] **Kernel 2 parallelization:** For large H, split across multiple programs instead of one per token.
- \[ \] **Kernel 2 expert unrolling:** For small E (e.g., 32), unroll the expert loop.
- \[ \] **Memory layout:** Explore contiguous gate/up layout instead of interleaved.
- \[ \] **Fuse kernel 1 + BMM:** Longer-term, fuse activation into GEMM epilogue.

______________________________________________________________________

## 5. Final Best Configuration

> (To be filled after optimization loop completes)

| Kernel | Config | Notes |
|--------|--------|-------|
| `_fused_glu_activation_kernel` | | |
| `_weighted_expert_sum_kernel` | | |

______________________________________________________________________

## Appendix: How to Reproduce

```bash
cd tensorrt_llm/_torch/auto_deploy/custom_ops/fused_moe
python bench_triton_moe_dense_mlp.py
```
