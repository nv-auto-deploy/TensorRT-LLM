<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Flash TP4 TTFT optimization

This directory records the AutoDeploy TTFT experiment now integrated with the
DeepSeek-V4 decode optimizations on this branch. The separate perf-agent campaign
state is not modified by these files.

## Comparison workload

- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Parallelism: TP4 / EP4, MXFP4 MoE
- GPUs: 4, 5, 6, 7
- Concurrency: 1
- Actual input/output lengths: 1004 / 8 tokens
- AutoDeploy config: `config_tp4_piecewise_1024.yaml`
- SGLang reference: TP4 MXFP4, 135.89 ms average TTFT and 135.49 ms p50

The SGLang artifact uses the same TP4/MXFP4, concurrency, and approximately
1000-token input, but requests 1000 output tokens rather than 8. Since TTFT ends at
the first generated token, it is a useful target, but this is not a strictly
output-length-matched end-to-end comparison.

The 1004-token prompt must fit in one prefill chunk. With `max_num_tokens: 512`,
the cached sparse-attention reference processes the second 492-token chunk one token
at a time, making that setup unsuitable for a matched prefill comparison.

## Results

| Variant | TTFT avg | TTFT p50 | Speedup from base | Gap to SGLang p50 |
| --- | ---: | ---: | ---: | ---: |
| Original one-chunk AutoDeploy | 17,035.11 ms | 17,025.74 ms | 1.0x | 125.66x |
| Vectorized compressed-cache construction | 300.70 ms | 274.36 ms | 56.6x | 2.02x |
| Direct row reuse + vectorized paged writes, first sample | 244.29 ms | 195.46 ms | 69.7x | 1.44x |
| Direct row reuse + vectorized paged writes, repeat | 196.18 ms | 195.07 ms | 86.8x | 1.44x |
| Routed piecewise prefill graph, first run | **96.29 ms** | **96.39 ms** | **176.9x** | **0.71x** |
| Routed piecewise prefill graph, independent repeat | **98.41 ms** | **97.00 ms** | **173.1x** | **0.72x** |

The routed piecewise result is stable across independent five-request samples. The
first sample spans 95.50--96.88 ms with 0.52 ms standard deviation. The repeat has
four requests near 96--97 ms and one 104.28 ms sample; its p50 is 97.00 ms. Relative
to SGLang's 135.49 ms p50, AutoDeploy is 28.4--28.9% faster. Relative to the prior
195.07 ms AutoDeploy p50, it is 2.01--2.02x faster.

The first request at a previously unseen production shape performs Triton lazy
compilation. The two-request warmup took 44.46 seconds in the first run; steady
requests then completed normally. A production service must prewarm the 1024-token
bucket before accepting latency-sensitive traffic.

The original measurement node wrote benchmark artifacts to:

- `/tmp/llm_ttft_mnt1024_baseline`
- `/tmp/llm_ttft_bulk_candidate`
- `/tmp/llm_ttft_direct_reuse_candidate`
- `/tmp/llm_ttft_direct_reuse_candidate_rep2`
- `/tmp/llm_ttft_piecewise_routed_candidate`
- `/tmp/llm_ttft_piecewise_routed_candidate_rep2`

These node-local paths are evidence locations from the experiment and may not exist
after moving to a new server.

The SGLang comparison result is
`dsv4_flash/sglang/res_sglang_tp4_conc1/260612_1523/isl_1000_osl_1000_conc_1/profile_export_aiperf.json`
under Yeonbok Lee's AutoDeploy experiment directory.

## Root cause and optimization

The original cached prefill rebuilt every compressed R4/R128 row through scalar
Python-controlled paged gathers. Across the full model this issued roughly 120,000
scalar gathers and made sparse-cache priming dominate TTFT. Vectorizing that path
reduced TTFT from 17 seconds to about 195 ms, after which a trace of the exact
candidate exposed the second bottleneck.

The steady rank-0 Nsight trace measured a 225.49 ms forward under profiling overhead:

- 73.71 ms GPU-busy union versus 151.78 ms of inter-kernel/edge gaps.
- 5,661 kernels and about 5,664 CUDA launch calls.
- 215 device-to-host metadata copies, 249 host-to-device page-map copies, and
  exactly 464 stream synchronizations.
- Static regions between sparse-attention calls consumed 157.75 ms wall / 58.10 ms
  GPU; the 43 dynamic sparse-attention regions consumed 67.75 ms wall / 15.61 ms GPU.

AutoDeploy successfully captured 43 static piecewise graphs, but the executor's
outer decode-only CUDA-graph gate marked every context batch ineligible and entered
`BypassCapturedGraphs`. The inner dual-mode runner therefore executed the original
eager model. Allowing non-attention-DP, non-speculative context/mixed batches to
delegate bucket selection to the piecewise runner is the main final optimization.

The optimized path:

1. Resolves each sequence's page table once and writes contiguous raw compressor
   state with one indexed device operation instead of one `copy_` per page.
2. Reconstructs completed compressed rows in a tensor batch instead of looping over
   rows and compression offsets in Python.
3. Uses the fused compressor RMSNorm and preserves exact production R4/R128 output.
4. During initial prefill, builds compressed rows once from the source tensors,
   writes those exact rows to the persistent MHC cache, and passes the same rows to
   sparse attention. The old path compressed once after a paged-cache round trip and
   a second time for attention.
5. Writes the heterogeneous R4/R128 raw cache set with one device page-table kernel;
   the production-shape microbenchmark projects about 5.47 ms full-model savings.
6. Uses the already-staged `SequenceInfo` CPU mirrors instead of copying five metadata
   tensors from the GPU in each of 43 layers.
7. Routes context batches into the captured 1024-token piecewise graphs instead of
   forcing the captured graph backend into eager bypass.

An attempted live-row compaction (251 of 512 R4 rows and 8 of 16 R128 rows at ISL
1004) improved p50 by only about 1.4 ms and worsened the small-sample tail, so it was
rejected.

## Correctness and validation

- The standalone `build_and_run_ad.py` production path completed on TP4 and returned
  coherent output: `2 + 2 -> 4.`
- Real-weight `trtllm-serve` returned coherent output: `2 + 2 -> 4.`
- Input-dependent replay was checked with three additional prompts: `Paris`, `35`,
  and `Blue` were returned for capital, multiplication, and sky-color questions.
- Production-shape source-built and paged-cache R4/R128 rows are byte exact.
- Direct reused-row attention output is byte exact against the source path.
- Paged writes are exact for aligned, unaligned, shuffled-page, and 1004-token cases.
- The original checkpoint passed 158 focused tests. The follow-up multi-cache/schema
  suite passes 16/16, the piecewise-routing regression passes 4/4, and the complete
  executor test file passes 20/20.

The benchmark config deliberately avoids general chunked prefill. Before an upstream
production merge supports `max_num_tokens: 512`, add an end-to-end 512 + 492 cache
continuation followed by decode test.

## Reproduction

Start the server:

```bash
REPO_ROOT="${PWD}"
CUDA_VISIBLE_DEVICES=4,5,6,7 \
PYTHONPATH="${REPO_ROOT}" \
trtllm-serve deepseek-ai/DeepSeek-V4-Flash \
  --host 127.0.0.1 --port 8134 --trust_remote_code \
  --backend _autodeploy \
  --config "${REPO_ROOT}/ttft/config_tp4_piecewise_1024.yaml"
```

Run the matched benchmark:

```bash
aiperf profile \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --url 127.0.0.1:8134 --endpoint-type chat --streaming \
  --concurrency 1 --request-count 5 --num-warmup-requests 2 \
  --request-timeout-seconds 1800 --use-server-token-count \
  --no-server-metrics --isl 1000 --osl 8 \
  --extra-inputs '{"ignore_eos": true}' \
  --tokenizer <path-to-dsv4-tokenizer>
```

Run the standalone real-weight smoke test:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH="${PWD}" \
python examples/auto_deploy/build_and_run_ad.py \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --args.yaml-extra "${PWD}/ttft/config_tp4_piecewise_1024.yaml" \
  --prompt.batch-size=1 \
  --prompt.queries='["What is 2 + 2? Answer with only the number."]' \
  --prompt.sp-kwargs.max-tokens=8 --prompt.sp-kwargs.temperature=0
```

## Outcome and remaining scope

The steady TTFT comparison target is closed: AutoDeploy's 96--97 ms p50 beats the
135.49 ms SGLang reference by about 29%. The staged context-FMHA experiment remains
disabled because its cubin requires 128-token pages while this runtime uses 32-token
pages, and its eligibility contract does not preserve arbitrary explicit top-k order.

This result covers the single-chunk 1004-token workload. General 512 + 492 chunked
prefill continuation remains separate follow-up work and must retain exact cache
continuation and first-decode behavior.
