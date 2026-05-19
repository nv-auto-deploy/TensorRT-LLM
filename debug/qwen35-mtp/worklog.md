<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5 MTP AutoDeploy Worklog

## Baseline

- Base commit: `3ae0b706046c447783943923712a1310f0d844d5`
- Scope: CausalLM/text-only Qwen3.5 MTP first. Defer VLM/`Qwen3_5MoeForConditionalGeneration` support.
- Base target status: `Qwen/Qwen3.5-397B-A17B-FP8` runs through AutoDeploy without MTP using the registry path and `TRTLLM_DG_JIT_USE_NVCC=1`.
- Relevant existing local changes before this work:
  - `examples/auto_deploy/model_registry/models.yaml`
  - `triton_backend/tools/gpt/input_data.json`

## 2026-05-18: Full GSM8K FP8 Accuracy and Small-Path Feasibility

### Current Task

Run full GSM8K on the 8xH100 FP8 no-MTP and FP8 MTP paths, record the
accuracy comparison, then check whether the existing 4xH100 "small" no-MTP
path works locally as a possible CI-compatible target for a later MTP variant.

### Plan

- Run `TestQwen3_5_397B_MoE::test_fp8_gsm8k[8]` on GPUs 0-7 and save the log
  under `ad_runs/`.
- Run `TestQwen3_5_397B_MoE::test_fp8_mtp_gsm8k[8]` on GPUs 0-7 and save the
  log under `ad_runs/`.
- Compare the full GSM8K accuracy numbers and any speculative acceptance stats
  visible in the MTP run.
- Run the existing small no-MTP test path locally on 4xH100 and record whether
  it gets through model load/evaluation.

### Starting State

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`
- The test file currently has a temporary FP8 no-MTP GSM8K-only row and an FP8
  MTP GSM8K-only row. The no-MTP row is intended for local comparison and
  should be removed before keeping the CI-facing MTP test.

## 2026-05-15: Phase 0 - Checkpoint and Code Shape

### Current Task

Understand the Qwen3.5 MTP checkpoint shape and map it to the existing AutoDeploy Eagle/MTP implementation before adding code.

### Plan

- Use `ad-eagle-onboard` as the workflow.
- Compare Qwen3.5 MTP keys with NemotronH MTP and PyTorch Qwen3-Next MTP.
- Keep first implementation on the text model path because the registry already uses `model_factory: AutoModelForCausalLM`.
- Update the Eagle onboarding skill with explicit worklog guidance because the skill does not currently contain it.

### Findings

- HF config advertises top-level `architectures=["Qwen3_5MoeForConditionalGeneration"]`, `model_type="qwen3_5_moe"`, and nested text `model_type="qwen3_5_moe_text"`.
- The registry config for `qwen3.5_moe_400b.yaml` intentionally sets `model_factory: AutoModelForCausalLM` for text-only mode.
- MTP keys are present in the same checkpoint snapshot:
  - `mtp.fc.weight`
  - `mtp.pre_fc_norm_embedding.weight`
  - `mtp.pre_fc_norm_hidden.weight`
  - `mtp.norm.weight`
  - `mtp.layers.0.self_attn.*`
  - `mtp.layers.0.mlp.*`
- This matches the Qwen3-Next MTP naming pattern more closely than NemotronH's `mtp.* -> model.*` pattern.

### Status

- Done.
- Added worklog guidance to `ad-eagle-onboard` source and installed skill copies.

## 2026-05-15: Phase 1/2 - Text MTP Implementation and Unit Tests

### Current Task

Implement the CausalLM/text-only Qwen3.5 MTP path and prove the local layer/config contracts with unit tests.

### Plan

- Keep implementation scoped to the text model path first.
- Unwrap composite Qwen3.5 config to `text_config` inside the Eagle drafter factory.
- Add a Qwen3.5 MTP layer matching the PyTorch Qwen3-Next MTP contract.
- Make the drafter wrapper tolerant of unused HF kwargs such as `use_cache`.
- Add tests for defaults, checkpoint key remapping, strict synthetic MTP key loading, layer
  math, unsupported layer counts, text CausalLM hooks, and unused kwargs.

### Results

- Added `Qwen3_5MoeEagleLayer` and `build_qwen3_5_moe_eagle_layers()`.
- Registered `qwen3_5_moe_text` and `qwen3_5_moe` in `EagleConfig` and `get_eagle_layers()`.
- Added target hooks for `get_output_embeddings()` and `get_final_normalization()` on Qwen3.5 text/CausalLM paths.
- Changed the CausalLM factory registration for `Qwen3_5MoeConfig` to build `Qwen3_5MoeForCausalLM`, which unwraps the composite config to the text config.
- Added a registry overlay config `qwen3.5_moe_400b_mtp.yaml` and `qwen3_5_moe_400b_fp8_mtp` registry id.
- Verified the real local HF snapshot builds on `meta`:
  - target factory: `Qwen3_5MoeForCausalLM Qwen3_5MoeTextConfig True`
  - drafter factory: `EagleDrafterForCausalLM qwen3_5_moe_text Qwen3_5MoeEagleLayer True True True`
- Checked the PyTorch backend Qwen3Next MTP path:
  - `Qwen3NextForCausalLM.post_load_weights()` wires the last decoder layer's
    `next_layer_layernorm` to `model.norm`.
  - `SpecDecOneEngineForCausalLM.forward()` passes that normalized target hidden state
    to `MTPWorker`.
  - AutoDeploy therefore keeps `normalize_target_hidden_state=True` for Qwen3.5 MTP
    because its capture point is pre-final-normalization.
- Started an overly broad full real-weight AutoDeploy MTP smoke, then stopped it after
  narrowing the validation goal. It had reached export/pattern-matching, but this was not
  the right test for isolated MTP-head support at this stage.

### Tests

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py tensorrt_llm/_torch/auto_deploy/models/custom/modeling_eagle.py tensorrt_llm/_torch/auto_deploy/models/eagle.py`
- `python -m py_compile tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py tensorrt_llm/_torch/auto_deploy/models/custom/modeling_eagle.py tensorrt_llm/_torch/auto_deploy/models/eagle.py tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
- `pytest -sv tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Result: 6 passed.
  - Log: `/tmp/qwen35_mtp_unit.log`.
- `pytest -sv tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe.py::test_causal_lm_position_embeddings tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe.py::test_export_text_model_with_position_ids`
  - Result: 2 passed.
  - Log: `/tmp/qwen35_existing_causallm.log`.

### Status

- Unit phase passed.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## GSM8K Accuracy Sample: No-MTP TRTLLM Torch-CUDAGraph

### Task

Complete the missing no-MTP TRTLLM `torch-cudagraph` GSM8K control so the
TRTLLM 2x2 matrix is available:

- no-MTP vs MTP
- `torch-simple` vs `torch-cudagraph`

### Plan

- Use registry config `qwen3_5_moe_400b_fp8` with TRTLLM attention, batch size
  `4`, multi-stream GEMM/MoE disabled, chunked prefill disabled, CUDA graph
  batch sizes `[1, 4]`, and overlap enabled.
- Save the log, JSON summary, and sample outputs under `ad_runs/`.

### Result

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8 QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_GSM8K_NUM_SAMPLES=128 QWEN35_GSM8K_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_no_mtp_trtllm_torch_cudagraph_samples QWEN35_GSM8K_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_no_mtp_trtllm_torch_cudagraph.json python debug/qwen35-mtp/qwen35_gsm8k_probe.py`
- Log: `ad_runs/gsm8k_no_mtp_trtllm_torch_cudagraph.log`
- JSON: `ad_runs/gsm8k_no_mtp_trtllm_torch_cudagraph.json`
- Samples: `ad_runs/gsm8k_no_mtp_trtllm_torch_cudagraph_samples/`
- Result:
  - samples: `128`
  - accuracy: `64.453125`
  - `disable_overlap_scheduler`: `false`
  - speculative stats: not applicable; no drafting configured
  - `num_stats`: `1001`
- The run completed and all ranks destroyed the process group cleanly.
- It emitted the same repeated `storeContextBlocks: Can not find sequence`
  warnings during response fetching as the other successful GSM8K rows.

### 2x2 TRTLLM GSM8K Matrix

- No-MTP TRTLLM `torch-simple`: `64.0625`
- No-MTP TRTLLM `torch-cudagraph`: `64.453125`
- MTP TRTLLM `torch-simple`: `66.40625`, acceptance
  `8439/9068 = 0.9306352007057785`
- MTP TRTLLM `torch-cudagraph`: `64.0625`, acceptance
  `8539/9154 = 0.9328162551889885`

### Interpretation

- Qwen3.5 MTP runs end-to-end in the intended performant configuration:
  TRTLLM attention + `torch-cudagraph` + overlap scheduler enabled.
- The MTP CUDA graph row has nontrivially high acceptance (`0.9328`) and lands
  in the same 128-sample GSM8K accuracy band as the no-MTP TRTLLM rows.
- The suspicious outlier is the MTP TRTLLM `torch-simple` row being higher by
  about three samples on this slice. That does not look like evidence that the
  CUDA graph MTP path is uniquely hurting accuracy.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## GSM8K Accuracy Sample: No-MTP TRTLLM Torch-Simple

### Task

Run the no-MTP TRTLLM `torch-simple` GSM8K control for the same-kernel
simple-vs-CUDA-graph comparison.

### Plan

- Use registry config `qwen3_5_moe_400b_fp8` with TRTLLM attention, batch size
  `4`, multi-stream GEMM/MoE disabled, chunked prefill disabled, and overlap
  enabled.
- Save the log, JSON summary, and sample outputs under `ad_runs/`.

### Result

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8 QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_GSM8K_NUM_SAMPLES=128 QWEN35_GSM8K_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_no_mtp_trtllm_torch_simple_samples QWEN35_GSM8K_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_no_mtp_trtllm_torch_simple.json python debug/qwen35-mtp/qwen35_gsm8k_probe.py`
- Log: `ad_runs/gsm8k_no_mtp_trtllm_torch_simple.log`
- JSON: `ad_runs/gsm8k_no_mtp_trtllm_torch_simple.json`
- Samples: `ad_runs/gsm8k_no_mtp_trtllm_torch_simple_samples/`
- Result:
  - samples: `128`
  - accuracy: `64.0625`
  - `disable_overlap_scheduler`: `false`
  - speculative stats: not applicable; no drafting configured
  - `num_stats`: `1001`
- The run completed and all ranks destroyed the process group cleanly.
- It emitted the same repeated `storeContextBlocks: Can not find sequence`
  warnings during response fetching as the other successful GSM8K rows.
- Interim interpretation:
  - MTP TRTLLM `torch-simple`: `66.40625`
  - MTP TRTLLM `torch-cudagraph`: `64.0625`
  - No-MTP TRTLLM `torch-simple`: `64.0625`
  - No-MTP TRTLLM `torch-cudagraph`: still missing after the earlier
    interrupted row.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Interrupted Baseline: No-MTP TRTLLM Torch-CUDAGraph

### Task

Start an apples-to-apples no-MTP TRTLLM `torch-cudagraph` GSM8K baseline.

### Result

- Command was launched with registry config `qwen3_5_moe_400b_fp8`,
  `QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph`, TRTLLM attention, batch size
  `4`, and `disable_overlap_scheduler=False` from the probe.
- Log: `ad_runs/gsm8k_no_mtp_trtllm_torch_cudagraph.log`
- JSON: not produced.
- The command was interrupted early during model build/factory setup and did
  not reach weight load or evaluation.
- GPUs were checked afterward with `nvidia-smi`; no resident memory remained on
  GPUs 0-7.
- Next plan: run the no-MTP TRTLLM `torch-simple` row, then rerun the no-MTP
  TRTLLM `torch-cudagraph` row, so the simple-vs-cudagraph comparison is
  available for the same TRTLLM kernels.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## GSM8K Accuracy Sample: TRTLLM Torch-Simple MTP

### Task

Run the same 128-sample GSM8K probe with MTP, TRTLLM attention, and
`torch-simple` before moving to the target performant row
TRTLLM attention + CUDA graph + overlap scheduler.

### Plan

- Use the Qwen3.5 MTP registry config with batch size `4`, multi-stream GEMM/MoE
  disabled, chunked prefill disabled, iter stats enabled, and TRTLLM attention.
- Save the log, JSON summary, and sample outputs under `ad_runs/`.

### Result

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_GSM8K_NUM_SAMPLES=128 QWEN35_GSM8K_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_mtp_trtllm_torch_simple_samples QWEN35_GSM8K_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_mtp_trtllm_torch_simple.json python debug/qwen35-mtp/qwen35_gsm8k_probe.py`
- Log: `ad_runs/gsm8k_mtp_trtllm_torch_simple.log`
- JSON: `ad_runs/gsm8k_mtp_trtllm_torch_simple.json`
- Samples: `ad_runs/gsm8k_mtp_trtllm_torch_simple_samples/`
- Result:
  - samples: `128`
  - accuracy: `66.40625`
  - `spec_iters`: `1001`
  - `total_drafted`: `9068`
  - `total_accepted`: `8439`
  - `acceptance_rate`: `0.9306352007057785`
  - `num_stats`: `1001`
- The run completed and all ranks destroyed the process group cleanly.
- As in the FlashInfer GSM8K rows, it emitted repeated
  `storeContextBlocks: Can not find sequence` warnings during response
  fetching. The warnings did not abort the run.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Probe Config Update: Explicit Overlap Scheduler

### Task

Make the temporary smoke and GSM8K probe scripts explicit about the overlap
scheduler state before running the target CUDA graph row.

### Plan

- Set `disable_overlap_scheduler=False` in the probe `llm_args`.
- Rely on the JSON `llm_args` echo in each probe output to prove that overlap is
  enabled for subsequent rows.

### Result

- Updated:
  - `debug/qwen35-mtp/qwen35_gsm8k_probe.py`
  - `debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- This is a probe-only change; it does not touch production code.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## GSM8K Accuracy Sample: TRTLLM Torch-CUDAGraph MTP With Overlap

### Task

Run the target performant configuration for this experiment: Qwen3.5 MTP with
TRTLLM attention, `torch-cudagraph`, and overlap scheduler enabled.

### Plan

- Use the Qwen3.5 MTP registry config with batch size `4`, multi-stream GEMM/MoE
  disabled, chunked prefill disabled, and TRTLLM attention.
- Make overlap explicit in the probe config with
  `disable_overlap_scheduler=False`.
- Save the log, JSON summary, and sample outputs under `ad_runs/`.

### Result

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_GSM8K_NUM_SAMPLES=128 QWEN35_GSM8K_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_mtp_trtllm_torch_cudagraph_samples QWEN35_GSM8K_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_mtp_trtllm_torch_cudagraph.json python debug/qwen35-mtp/qwen35_gsm8k_probe.py`
- Log: `ad_runs/gsm8k_mtp_trtllm_torch_cudagraph.log`
- JSON: `ad_runs/gsm8k_mtp_trtllm_torch_cudagraph.json`
- Samples: `ad_runs/gsm8k_mtp_trtllm_torch_cudagraph_samples/`
- Result:
  - samples: `128`
  - accuracy: `64.0625`
  - `disable_overlap_scheduler`: `false`
  - `spec_iters`: `1001`
  - `total_drafted`: `9154`
  - `total_accepted`: `8539`
  - `acceptance_rate`: `0.9328162551889885`
  - `num_stats`: `1001`
- The run completed and all ranks destroyed the process group cleanly.
- It emitted the same repeated `storeContextBlocks: Can not find sequence`
  warnings during response fetching as the previous successful GSM8K rows.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next useful validation: a cheap checkpoint-loading smoke for the real HF snapshot that
  verifies MTP keys are consumed, without running full AutoDeploy generation.

## 2026-05-15: Phase 3 - Real Snapshot MTP Key Audit

### Current Task

Check the real local `Qwen/Qwen3.5-397B-A17B-FP8` snapshot key layout against the
AutoDeploy drafter state dict without loading the full model or running generation.

### Plan

- Read `model.safetensors.index.json` from the local HF cache.
- Build the Qwen3.5 Eagle drafter on `meta`.
- Apply the AutoDeploy `_checkpoint_conversion_mapping` to all real `mtp.*` keys.
- Compare the remapped key set with the drafter's MTP state dict keys.

### Results

- Snapshot:
  `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d`
- Real checkpoint MTP keys: 3096.
- Expected drafter MTP state-dict keys: 1553.
- Missing required drafter keys after remapping: 0.
- Unexpected remapped real-checkpoint keys: 1543.
- All unexpected keys end with `weight_scale_inv`; there were no unexpected non-scale
  MTP keys. These scale tensors belong to the FP8 checkpoint/quantization path rather
  than the plain pre-transform module state dict.

### Tests

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/models/custom/modeling_qwen3_5_moe.py tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
- `pytest -sv tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Result: 6 passed.
  - Log: `/tmp/qwen35_mtp_unit.log`.

### Status

- Done.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Qwen3.5 MTP Draft Sharding: Hidden-Size Inference

### Task

Debug the 8-GPU MTP CUDA-graph batch-32 failure where draft `k_proj` is column
sharded but its downstream view remains `[batch, seq, 2, 256]`, producing an
invalid shape after TP.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Previous full run log:
  `/tmp/qwen35_ad_mtp_cudagraph_b32_after_typed_parent.log`.
- Symptom: sharding log reported `Simple: 3, row-col: 1 (attention: 0...)`,
  so q/k/v were direct column shards rather than a grouped MHA shard. The later
  shape-prop failure hit:
  `shape '[..., 2, 256]' is invalid for input of size ...`.

### Plan

Inspect whether Qwen3.5 MTP draft layer detection classifies the exported draft
attention as MHA. If not, patch the layer-boundary inference so q/k/v/o enter
the grouped `_process_column_sharding()` path and the attention views get
updated.

### Result

- A small real export of `Qwen3_5MoeEagleLayer` showed zero layer subgraphs.
- Root cause: `infer_draft_embedding_size()` inferred hidden size from the last
  linear output. In Qwen3.5 MoE MTP, the exported draft ends with
  `shared_expert_gate` shape `[1, hidden]`, so it inferred `embd=1`.
- Updated draft hidden-size inference to prefer an explicit fused MTP prologue
  shape `[h, 2h]` when present. This keeps the existing final-linear fallback
  for non-prologue Eagle/Nemotron paths.
- Re-inspecting the small real export now reports:
  - layer 0: `LayerType.MHA`, opening q/k/v, terminating o_proj, head size 16
  - layer 1: `LayerType.MLP`, opening shared gate/up, terminating down
  - unprocessed: fc prologue, router gate, scalar shared-expert gate
- Added unit coverage for Qwen3.5 gated attention view updates and for the MTP
  prologue winning over a trailing scalar gate.

### Tests

- Focused sharding tests:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_qwen3_5_gated_attention_uses_grouped_column_sharding tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_draft_embedding_inference_prefers_mtp_prologue_over_trailing_gate`
  - Result: 2 passed.
  - Log: `/tmp/qwen35_draft_embedding_sharding_unit.log`.
- Full sharding utility unit file:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
  - Result: 16 passed.
  - Log: `/tmp/qwen35_node_utils_sharding_full.log`.
- Small real exported Qwen3.5 MTP draft graph inspection:
  - Result: MHA and MLP layer subgraphs now detected as expected.

### Status

- Unit-level fix is done.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: rerun the 8-GPU Qwen3.5 MTP CUDA-graph smoke with the batch-32
  registry config and check whether it reaches cache initialization/generation.

## Draft Layer Boundary Follow-Up

### Task

Finish the draft manual-sharding fix after the first prologue skip did not
cover the real exported draft graph.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Full 8-GPU MTP run with the first shape-based prologue skip:
  `/tmp/qwen35_ad_mtp_cudagraph_b32_after_shape_prologue_fix.log`.
- Result: failed again in draft manual sharding:
  `Start linear node (index 0) not found in opening linear nodes -
  start_linear node: model_layers_fc_torch_linear_simple`.

### Plan

1. Preserve the target/draft shared manual TP plan so draft KV layers use the
   same sharding/block-offset policy as target KV layers.
2. Fix the generic layer-boundary finder so a candidate linear that is upstream
   of a layer, but is not itself an opening projection for that layer, remains
   unprocessed and layer detection advances.
3. Add a focused CPU regression for that boundary behavior plus keep the
   Qwen3.5 MTP unit coverage green.
4. Rerun the full 8-GPU MTP registry smoke with batch size 32.

### Result

- Replaced the hard assertion in `get_layer_after_linear_node()` with the
  generic skip behavior described by the function docstring: if the candidate
  start linear is not in the backward-discovered opening set, append that
  candidate index to `terminating_indices`, return an empty unknown subgraph,
  and let the caller advance. This keeps prologue/internal linears in the
  unprocessed set without treating them as decoder-layer starts.
- Added
  `test_layer_boundary_skips_linear_that_is_not_an_opening_node`.
- The first attempt at this regression used missing shape metadata and did not
  exercise the intended path; after setting `lin_node_shape`, the test covers
  the intended non-opening candidate case.
- Started full 8-GPU rerun:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --use-registry --registry-config-id qwen3_5_moe_400b_fp8_mtp`
  - Log: `/tmp/qwen35_ad_mtp_cudagraph_b32_after_boundary_skip.log`.
  - Result: got past the old `Start linear node` assertion, then failed in
    draft manual sharding with
    `'list' object has no attribute 'opening_nodes'`.
  - Why: after skipping the MTP `fc` prologue, some directly matched manual
    `colwise` nodes had ambiguous parent layer-subgraph matches. The manual
    sharding code left the ambiguous list in `layer_subgraph` and then passed
    that list into `_process_column_sharding()`.
- Follow-up fix: for manual `colwise`, if there is exactly one
  `LayerSubgraph`, keep using grouped column sharding. If the parent layer is
  ambiguous, apply a direct column `WeightShardingInfo` to the matched linear
  node instead of crashing.
- Started another full 8-GPU rerun:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --use-registry --registry-config-id qwen3_5_moe_400b_fp8_mtp`
  - Log: `/tmp/qwen35_ad_mtp_cudagraph_b32_after_ambiguous_colwise.log`.
  - Result: got through the ambiguous-list crash but failed in post-sharding
    shape propagation:
    `shape '[s44, s29, 2, 256]' is invalid for input of size 64*s29*s44`
    at the draft `k_proj(...).view(...)`.
  - Why: the direct-column fallback sharded `k_proj` without using the grouped
    attention layer sharding helper, so the downstream view was not updated to
    the local KV width.
- Follow-up fix: when a linear has multiple candidate parent layer subgraphs,
  prefer a unique non-`UNKNOWN` parent before falling back to direct column
  sharding. This should keep draft q/k/v/o on the grouped MHA path and preserve
  view updates.

### Tests

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/utils/node_utils.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
- Escalated focused regression set:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_layer_boundary_skips_linear_that_is_not_an_opening_node tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_draft_mtp_prologue_does_not_break_manual_attention_sharding tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_config_sharding_skips_linear_without_weight_name tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Result: 9 passed.
  - Log: `/tmp/qwen35_generic_prologue_skip_unit.log`.
- `python -m py_compile tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py tensorrt_llm/_torch/auto_deploy/utils/node_utils.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
- Escalated focused regression set after ambiguous-colwise fallback:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_layer_boundary_skips_linear_that_is_not_an_opening_node tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_draft_mtp_prologue_does_not_break_manual_attention_sharding tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_config_sharding_skips_linear_without_weight_name tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Result: 9 passed.
  - Log: `/tmp/qwen35_ambiguous_colwise_fallback_unit.log`.
- `python -m py_compile tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`
- Escalated focused regression set after typed parent selection:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_layer_boundary_skips_linear_that_is_not_an_opening_node tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_draft_mtp_prologue_does_not_break_manual_attention_sharding tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_config_sharding_skips_linear_without_weight_name tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Result: 9 passed.
  - Log: `/tmp/qwen35_typed_parent_selection_unit.log`.

### Status

- Unit coverage passed.
- Full 8-GPU validation needs rerun after typed parent selection.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Qwen3.5 MTP Registry Smoke Shape

### Task

Create a separate registry config for Qwen3.5 MTP and rerun the real-weight
AutoDeploy smoke path with a smaller serving shape.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Previous MTP registry entry inherited the baseline Qwen3.5 config:
  `max_batch_size=256`, `max_num_tokens=16000`, `free_gpu_memory_fraction=0.8`,
  and multistream GEMM/MoE enabled.
- The first full MTP CUDA-graph run failed during resize/warmup with CUDA OOM
  in fused allreduce workspace allocation after weight load. Cache stats before
  failure were `total=197/196, kv=16/16, ssm=90/90, conv=90/90, other=1`.

### Plan

Use a dedicated MTP overlay that keeps the base model's sharding/quantization
settings but reduces capacity and disables multistream transforms that have
previously been problematic for MTP.

### Result

- Added `examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b_mtp.yaml`.
- Added a `qwen3_5_moe_400b_fp8_mtp` registry entry for
  `Qwen/Qwen3.5-397B-A17B-FP8`.
- First overlay used `max_batch_size=64`, `max_num_tokens=8192`,
  `free_gpu_memory_fraction=0.7`, `max_draft_len=1`, and disabled
  `multi_stream_gemm`/`multi_stream_moe`.
- The 64-batch run loaded weights successfully but failed before cache
  allocation with:
  `KV cache layer r15_kv_cache (idx=15) has block_offset_multiplier 2 != reference 30`.
  This points to non-uniform KV layout between target and draft KV layers,
  not to the batch-size/OOM issue.
- Investigation found a local sharding change that skipped manual TP sharding
  for draft graphs. That likely left the Qwen MTP draft attention unsharded
  while target attention KV was TP-sharded, forcing the draft KV layer into a
  separate cache pool.
- Removed the draft/manual-sharding skip so the Qwen manual TP plan can shard
  draft `q_proj`, `k_proj`, `v_proj`, and `o_proj`.
- Lowered the MTP overlay to `max_batch_size=32` before the next expensive
  run, matching the suspected practical smoke shape.

### Tests

- 64-batch full run:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --use-registry --registry-config-id qwen3_5_moe_400b_fp8_mtp`
  - Result: failed with non-uniform KV block offset multiplier before cache
    allocation.
  - Log: `/tmp/qwen35_ad_mtp_cudagraph_smaller_config.log`.
- Focused unit tests after the draft manual-sharding fix:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Sandbox run hit the known OpenMPI init failure.
  - Escalated run passed: `18 passed`.
  - Log: `/tmp/qwen35_manual_draft_sharding_unit.log`.

### Status

- In progress.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: rerun the Qwen3.5 MTP CUDA-graph smoke with the 32-batch MTP
  overlay and verify whether the draft KV layer now shares the target KV cache
  layout.

## Full Qwen3.5 FP8 MTP CUDA-Graph Run After FLA State Fix

### Task

Run the real 8-GPU Qwen3.5-397B-A17B-FP8 MTP AutoDeploy path with the registry
config and CUDA graph enabled, to check whether the FLA extend-state changes
clear the previous target-verification/capture issue.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- GPUs were free before launch.
- Command used the registry MTP config with `TRTLLM_DG_JIT_USE_NVCC=1`.

### Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home \
TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 \
python examples/auto_deploy/build_and_run_ad.py \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --use-registry \
  --registry-config-id qwen3_5_moe_400b_fp8_mtp \
2>&1 | tee /tmp/qwen35_ad_mtp_cudagraph_after_fla_fix.log
```

### Result

- The run reached export, hidden-state detection, sharding, weight loading,
  post-load fusion, and `cache_init`.
- `detect_hidden_states_for_capture` succeeded on all ranks with `matches=1`.
- The run failed in `resize_kv_cache` during its warmup forward, before CUDA
  graph capture. Rank 0 raised `cudaErrorMemoryAllocation` while creating the
  custom allreduce IPC workspace from
  `trtllm_fused_allreduce_residual_rmsnorm -> trtllm_allreduce ->
  get_allreduce_workspace -> IpcMemory.open_ipc_memory`.
- This is earlier than the prior CUDA-graph/FLA host-sync issue and does not
  directly exercise graph capture. It appears to be a memory-capacity issue in
  the default full-size registry config after loading target + draft + cache
  resources, not an FLA extend-request correctness failure.
- After the failure, the worker processes temporarily held about 80.4 GiB on
  each H100 with 0% utilization. Open MPI then aborted and released all GPUs.

### Tests

- Log: `/tmp/qwen35_ad_mtp_cudagraph_after_fla_fix.log`.
- `nvidia-smi` after MPI abort showed all 8 GPUs back to `0MiB` used.

### Status

- Blocked for the full registry CUDA-graph E2E run by VRAM pressure before
  capture.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: run a GSM8K accuracy matrix comparing MTP vs no-MTP under
  `torch-simple` and `torch-cudagraph`, using matching target/model settings
  and recording raw accuracy plus acceptance stats for MTP.

## Reverted Layer-Wise TRTLLM Block-Offset Remapping

### Task

Review and remove the layer-specific TRTLLM block-offset remapping experiment.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- I had added a speculative helper that could remap `kv_cache_block_offsets`
  when target and draft KV views had different `stride(0) // stride(1)`
  multipliers, and relaxed the uniform-KV check when `spec_config` was present.

### Result

- Reverted that change completely.
- Rationale: one-model MTP should keep target and drafter KV layers in the same
  managed cache layout with matching dtype/sharding, so the blockwise offsets
  should stay uniform. A multiplier mismatch should remain an error because it
  indicates cache-manager/layout drift that must be fixed at the source, not
  hidden by per-layer offset remapping.
- The only remaining TRTLLM spec-dec block-offset behavior is the pre-existing
  scratch copy used for CUDA graph compatibility; it does not alter offset
  values.

### Tests

- No test run for this revert yet.
- Verified by diff inspection that no block-offset remapping changes remain in
  `trtllm_attention.py`, `shim/interface.py`, or related tests.

### Status

- Done.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## 2026-05-15: Phase 4 - Heavy E2E Smoke and Acceptance Prep

### Current Task

Run the real 8-GPU `Qwen/Qwen3.5-397B-A17B-FP8` AutoDeploy MTP path end to end
before adding acceptance-rate or GSM8K diagnostics.

### Plan

- Start with a minimal generation smoke using the new registry MTP overlay.
- Keep `TRTLLM_DG_JIT_USE_NVCC=1` and `--use-registry`.
- If generation succeeds, add a temporary probe inspired by
  `tests/integration/defs/examples/test_ad_speculative_decoding.py` and
  `tests/integration/defs/accuracy/test_llm_api_autodeploy.py` to print
  acceptance stats without enforcing thresholds.

### Results

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --registry-config-id qwen3_5_moe_400b_fp8_mtp --use-registry --args.max-batch-size=1 --args.cuda-graph-config.max-batch-size=1 --args.cuda-graph-config.batch-sizes='[1]' --args.transforms.compile-model.cuda-graph-batch-sizes='[1]' --prompt.batch-size=1 --prompt.sp-kwargs.max-tokens=8 --prompt.queries '["Hello, my name is"]'`
- Log: `/tmp/qwen35_ad_mtp_e2e.log`.
- The run successfully built the Qwen3.5 target and drafter factories, exported the
  combined target/draft graph, matched Qwen MoE routing, and applied fine-grained
  FP8 linear/MoE transforms.
- It failed during executor initialization in
  `detect_hidden_states_for_capture`:
  `get_weight_shape()` indexed `extract_weight_nodes(node).weights[0]` for a
  linear-like node that had no recoverable weight node after FP8 graph rewrites.

### Follow-up Fix

- Make `get_weight_shape()` return `None` when a linear-like node has no
  extractable weight node.
- Make `get_all_layer_subgraphs()` filter out unshapeable linear-like nodes and
  treat such nodes as traversal boundaries during shape-based layer detection.
- Add a CPU unit test covering a linear node whose weight is not a registered
  graph attribute.

### Tests After Fix

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/utils/node_utils.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
- `pytest -sv tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_get_weight_shape_returns_none_for_unregistered_weight`
  - Result: 7 passed.
  - Log: `/tmp/qwen35_mtp_post_node_utils_unit.log`.

### Status

- E2E generation has not passed yet.
- The first retest cleared `detect_hidden_states_for_capture` on all ranks
  (`matches=1`) and reached manual sharding.
- The retest then failed in `detect_sharding_from_config()` because
  `extract_weight_name()` returned `False` for the preserved `lm_head` linear,
  and the config matcher passed that boolean into `re.match()`.

### Follow-up Fix 2

- Root cause hypothesis: the preserved target `lm_head` weight has an auxiliary
  `torch._assert` user to keep it alive in the exported graph. The shared
  weight-node mapper followed only one downstream user from a parameter; if it
  followed the auxiliary assert path, the real linear consumer never got tagged
  with the weight node.
- Updated `_precompute_weight_node_mapping()` to traverse each direct parameter
  user independently so auxiliary preservation users cannot hide a compute
  consumer.
- Added a CPU regression test where a parameter feeds both a `torch._assert`
  node and an `aten.linear` node.

### Tests After Follow-up Fix 2

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/utils/node_utils.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
- `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_get_weight_shape_returns_none_for_unregistered_weight tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_extract_weight_name_with_auxiliary_parameter_user`
  - Result: 2 passed.
  - Log: `/tmp/qwen35_node_utils_aux_user.log`.

### Status

- E2E generation has not passed yet.
- Retest after the auxiliary-parameter-user mapping fix still failed in the
  same manual-sharding location. The real exported `lm_head` linear still did
  not have an extractable weight name.

### Follow-up Fix 3

- Updated config-driven TP sharding to respect `extract_weight_name()`'s
  `Union[str, bool]` contract and skip linear nodes whose weight name cannot be
  extracted instead of passing `False` into `re.match()`.
- Added a CPU regression test with one shapeable registered-weight linear and
  one runtime-weight linear to verify config sharding skips the unidentifiable
  linear without crashing.

### Tests After Follow-up Fix 3

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
- `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_get_weight_shape_returns_none_for_unregistered_weight tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_extract_weight_name_with_auxiliary_parameter_user tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_config_sharding_skips_linear_without_weight_name`
  - Result: 3 passed.
  - Log: `/tmp/qwen35_node_utils_sharding_skip.log`.

### Status

- E2E generation has not passed yet.
- Retest after the config-sharding skip fix cleared the previous `False`/regex
  crash. Target manual sharding applied 60 TP shards.
- The next failure happened when the same target-authored manual TP config was
  applied to the draft graph. The draft graph starts with the MTP `fc` before
  the decoder/MoE block, and layer-boundary recovery asserted:
  `Start linear node (index 0) not found in opening linear nodes`.

### Follow-up Fix 4

- Skip manual TP config for draft GraphModules, matching the existing behavior
  that already skips factory TP config for drafters. Manual config is authored
  for the target graph, and the draft graph has extra Eagle/MTP structure before
  the decoder block.

### Tests After Follow-up Fix 4

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py`

### Status

- E2E generation has not passed yet.
- Retest after skipping manual TP config on draft graphs.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### Retest After Follow-up Fix 4

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --registry-config-id qwen3_5_moe_400b_fp8_mtp --use-registry --args.max-batch-size=1 --args.cuda-graph-config.max-batch-size=1 --args.cuda-graph-config.batch-sizes='[1]' --args.transforms.compile-model.cuda-graph-batch-sizes='[1]' --prompt.batch-size=1 --prompt.sp-kwargs.max-tokens=8 --prompt.queries '["Hello, my name is"]'`
- Log: `/tmp/qwen35_ad_mtp_e2e_after_skip_draft_manual.log`.
- Result: failed during full checkpoint loading after reaching all ranks'
  `load_weights` phase. The failure was a large set of `weight_scale_inv`
  size mismatches, for example checkpoint shape `[96, 32]` versus current
  local TP shape `[12, 32]` for linear-attention projections.
- Root cause hypothesis: quantized linear weights register load hooks on their
  owning submodules, so they run after parent-level checkpoint remap hooks.
  Quantized scale buffers used the top-level GraphModule hook path, which can
  run before Qwen's `model.language_model.*` to exported-text `model.*` remap.
  The scale hook then silently missed the checkpoint keys, leaving unsharded
  scale tensors for `load_state_dict()`.

### Follow-up Fix 5

- Changed `QuantizationShardingMixin.quantization_cb()` to register scale load
  hooks on the owning submodule, using the local weight parameter name. This
  mirrors the main sharded-weight hook path and the existing fused-MoE blocked
  scale hook path.
- Added a CPU regression test where a parent hook remaps a checkpoint scale key
  and the child quantized-scale hook must run after that remap.

### Tests After Follow-up Fix 5

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
- First sandboxed `pytest` attempt hit the known OpenMPI local-network sandbox
  failure before collection.
- Escalated rerun:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_quant_scale_load_hook_runs_after_parent_remap tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_config_sharding_skips_linear_without_weight_name tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_extract_weight_name_with_auxiliary_parameter_user`
  - Result: 3 passed.
  - Log: `/tmp/qwen35_quant_scale_hook_unit.log`.

### Status

- E2E generation has not passed yet.
- Next task: rerun the 8-GPU generation smoke after Follow-up Fix 5.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### Retest After Follow-up Fix 5

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --registry-config-id qwen3_5_moe_400b_fp8_mtp --use-registry --args.max-batch-size=1 --args.cuda-graph-config.max-batch-size=1 --args.cuda-graph-config.batch-sizes='[1]' --args.transforms.compile-model.cuda-graph-batch-sizes='[1]' --prompt.batch-size=1 --prompt.sp-kwargs.max-tokens=8 --prompt.queries '["Hello, my name is"]'`
- Log: `/tmp/qwen35_ad_mtp_e2e_after_scale_hook.log`.
- Result: cleared the previous `weight_scale_inv` load failure. All ranks
  reported `Checkpoint loading completed`.
- New failure: cache initialization asserted
  `Mismatched Conv spec layer count: expected 45, got 0`. The graph had 45
  cached causal-conv base resources, but the selected `cuda_causal_conv`
  backend did not register matching speculative intermediate conv resources.

### Follow-up Fix 6

- Added `transforms.insert_cached_causal_conv.backend: triton_causal_conv` to
  the Qwen3.5 MTP overlay, matching the existing SuperV3 MTP guidance that
  Triton SSM/causal-conv are required for speculative state caching.

### Status

- E2E generation has not passed yet.
- Next task: rerun the 8-GPU generation smoke after Follow-up Fix 6.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### Retest After Follow-up Fix 6

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --registry-config-id qwen3_5_moe_400b_fp8_mtp --use-registry --args.max-batch-size=1 --args.cuda-graph-config.max-batch-size=1 --args.cuda-graph-config.batch-sizes='[1]' --args.transforms.compile-model.cuda-graph-batch-sizes='[1]' --prompt.batch-size=1 --prompt.sp-kwargs.max-tokens=8 --prompt.queries '["Hello, my name is"]'`
- Log: `/tmp/qwen35_ad_mtp_e2e_triton_conv.log`.
- Result: cleared the previous speculative conv-state assertion and loaded all
  checkpoint shards on all ranks. Cache initialization then failed while
  assigning KV views:
  `KV cache layer r15_kv_cache (idx=15) has block_offset_multiplier 2 != reference 30`.
- Root cause hypothesis: Qwen3.5 has 15 full-attention target layers managed in
  one interleaved TRTLLM KV pool (`15 layers * K/V = 30` stride multiplier), and
  the appended MTP draft attention layer is allocated in a separate one-layer
  pool (`K/V = 2`). AutoDeploy's TRTLLM attention op passes a layer-specific pool
  pointer, but its block-offset table was still using one global multiplier.

### Follow-up Fix 7

- Allowed speculative AutoDeploy graphs to bind managed KV views with differing
  page-stride multipliers. Non-speculative TRTLLM attention still rejects mixed
  multipliers.
- Added layer-specific block-offset remapping in the AutoDeploy TRTLLM attention
  planner: when a draft layer's KV view has a different stride multiplier, the
  op reconstructs raw page ids from the target-multiplied block table and writes
  a scratch table using the draft layer multiplier.
- Added focused regressions for both pieces:
  - CSI allows a speculative draft KV pool with multiplier 2 beside a target
    interleaved pool with multiplier 30.
  - TRTLLM attention planner remaps target-multiplied offsets `[0, 30, ...]` to
    draft offsets `[0, 2, ...]`.

### Tests After Follow-up Fix 7

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/shim/interface.py tensorrt_llm/_torch/auto_deploy/custom_ops/attention/trtllm_attention.py tests/unittest/auto_deploy/singlegpu/shim/test_cached_sequence_interface.py tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_trtllm_attention_op.py`
- First sandboxed pytest attempt hit the known OpenMPI local-network sandbox
  failure before collection.
- Escalated rerun:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/shim/test_cached_sequence_interface.py::test_assign_kv_cache_views_allows_speculative_draft_pool_multiplier tests/unittest/auto_deploy/singlegpu/custom_ops/attention/test_trtllm_attention_op.py::test_layer_block_offsets_remaps_speculative_draft_pool_multiplier`
  - Result: 2 passed.

### Status

- E2E generation has not passed yet.
- Next task: rerun the 8-GPU generation smoke after Follow-up Fix 7.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### Retest After Follow-up Fix 7

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --registry-config-id qwen3_5_moe_400b_fp8_mtp --use-registry --args.max-batch-size=1 --args.cuda-graph-config.max-batch-size=1 --args.cuda-graph-config.batch-sizes='[1]' --args.transforms.compile-model.cuda-graph-batch-sizes='[1]' --prompt.batch-size=1 --prompt.sp-kwargs.max-tokens=8 --prompt.queries '["Hello, my name is"]'`
- Log: `/tmp/qwen35_ad_mtp_e2e_after_kv_multiplier.log`.
- Result: cleared the previous KV multiplier failure. All ranks completed cache
  initialization with `kv=16/16`, then resized the KV cache and reached
  `compile_model` CUDA graph warmup/capture.
- New failure: CUDA graph capture failed in the target Qwen3.5 FLA
  gated-delta path:
  `torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing`.
  The direct stack goes through
  `fla_cached_gated_delta_rule -> chunk_gated_delta_rule -> chunk_local_cumsum -> prepare_chunk_indices`,
  where `prepare_chunk_indices(...).tolist()` is executed while the stream is
  capturing.
- Root cause hypothesis: the MTP integration now reaches full graph execution,
  and the observed failure stack is in Qwen3.5's target hybrid gated-delta
  operator during CUDA graph capture. This does not prove that a no-MTP target
  run would fail the same way; an apples-to-apples no-MTP CUDA-graph rerun is
  needed to separate a target-only capture issue from an MTP-exposed capture
  issue.

### Status

- CUDA-graph E2E generation has not passed yet.
- Next task: run the same registry entry with `compile_backend=torch-simple` to
  validate the MTP path without CUDA graph capture, then run an apples-to-apples
  no-MTP CUDA-graph baseline before assigning the capture failure to the target
  model or the MTP path.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### Eager E2E Smoke

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --registry-config-id qwen3_5_moe_400b_fp8_mtp --use-registry --args.compile-backend=torch-simple --args.max-batch-size=1 --args.cuda-graph-config.max-batch-size=1 --args.cuda-graph-config.batch-sizes='[1]' --prompt.batch-size=1 --prompt.sp-kwargs.max-tokens=8 --prompt.queries '["Hello, my name is"]'`
- Log: `/tmp/qwen35_ad_mtp_e2e_torch_simple.log`.
- Result: passed. All ranks completed transforms and prompt generation.
- Output began:
  `<think>\n: Thinking Process:\n\n1.  **`
- Why I think this worked: the run used the same MTP registry entry and real
  HF checkpoint, but `compile_backend=torch-simple` avoided the observed CUDA
  graph capture failure. This validates
  that the Qwen3.5 CausalLM + MTP drafter builds, loads weights, initializes
  caches, and runs generation end to end in eager AutoDeploy.

### Status

- Eager E2E generation passed.
- CUDA-graph E2E is still blocked by a capture failure whose stack is in target
  FLA gated-delta code; no-MTP CUDA-graph parity has not been rechecked yet.
- Next task: run a stats/acceptance probe with longer prompts and
  `enable_iter_perf_stats` / `enable_iter_req_stats` enabled.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### Acceptance Stats Probe

- Added a temporary probe:
  `debug/qwen35-mtp/qwen35_mtp_stats_probe.py`.
- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- Log: `/tmp/qwen35_mtp_stats_probe.log`.
- JSON result: `/tmp/qwen35_mtp_stats_probe.json`.
- Result: passed.
- Speculative summary from `llm.get_stats()`:
  - `spec_iters`: 37
  - `total_drafted`: 95
  - `total_accepted`: 73
  - `acceptance_rate`: 0.7684210526315789
  - `num_stats`: 38
- Representative iter stats:
  - iter 2: drafted 1, accepted 1, acceptanceLength 2.0
  - iter 3: drafted 4, accepted 4, acceptanceLength 2.0
  - iter 5: drafted 4, accepted 3, acceptanceLength 1.75
  - iter 38: drafted 1, accepted 1
- Why I think this worked: the same real Qwen3.5 FP8 snapshot and the same MTP
  registry entry were used as the eager smoke, but with longer generation and
  iter stats enabled. The public script log did not print `specDecodingStats`,
  but `LLM.get_stats()` returned nonzero drafted and accepted token counts.

### Status

- Eager E2E generation passed.
- Speculative acceptance is nontrivially above zero in the eager path.
- CUDA-graph E2E remains blocked by a capture failure whose stack is in target
  FLA gated-delta code; no-MTP CUDA-graph parity still needs an apples-to-apples
  baseline.
- Next task: run a one-sample GSM8K probe with accuracy threshold validation
  disabled and collect both the raw score and speculative stats.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### One-Sample GSM8K Probe

- Added a temporary probe:
  `debug/qwen35-mtp/qwen35_mtp_gsm8k_probe.py`.
- First sandboxed run failed in OpenMPI init:
  `No network interfaces were found for out-of-band communications`.
- First escalated rerun failed before model construction because the temporary
  script imported `tests.integration...`; fixed the script to add
  `tests/integration` to `sys.path` and import
  `defs.accuracy.accuracy_core.GSM8K`.
- Final command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python debug/qwen35-mtp/qwen35_mtp_gsm8k_probe.py`
- Log: `/tmp/qwen35_mtp_gsm8k_probe.log`.
- JSON result: `/tmp/qwen35_mtp_gsm8k_probe.json`.
- lm-eval sample file:
  `/tmp/qwen35_mtp_gsm8k_lm_eval/samples_gsm8k.json`.
- Result: passed end to end.
- Build/runtime observations:
  - The run used the real `Qwen/Qwen3.5-397B-A17B-FP8` snapshot and
    `qwen3_5_moe_400b_fp8_mtp` registry entry.
  - All ranks completed graph transforms; total transform time was about
    710.5s.
  - Weight load allocated about 53.23GB per rank before cache resize.
  - Cache initialization reported `total=152/106`, `kv=16/16`,
    `conv=90/90`, `state_other=45`, matching the target hybrid layers plus
    draft path.
- GSM8K outcome:
  - Score: `0.0` on the single sample.
  - Effective sample count: 1.
  - Prompt/question: combined money question from one GSM8K sample.
  - Expected final answer: `400`.
  - Generated response in the sample file:
    `<name> has $100 -  = 100`
  - The response is incorrect and very short; this is a generation quality issue
    to investigate separately from the fact that MTP executed.
- Speculative summary from `llm.get_stats()`:
  - `spec_iters`: 8
  - `total_drafted`: 8
  - `total_accepted`: 7
  - `acceptance_rate`: 0.875
  - `num_stats`: 9
  - Iter 2 through 8 accepted 1/1 draft token each; iter 9 accepted 0/1.

### Status

- Eager E2E generation passed.
- One-sample GSM8K ran end to end with nonzero MTP acceptance.
- The one-sample GSM8K answer quality was bad (`0.0`, generated `100` instead
  of `400`).
- CUDA-graph E2E remains blocked by a capture failure whose stack is in target
  FLA gated-delta code; no-MTP CUDA-graph parity still needs an apples-to-apples
  baseline.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## No-MTP CUDA Graph Baseline and MTP Overlay Update

### Task

Check whether Qwen3.5 AutoDeploy CUDA graph capture fails even without MTP, then
remove known-risk multi-stream transforms from the MTP overlay before rerunning
the speculative path.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- The prior MTP CUDA-graph run failed during capture inside the target hybrid
  FLA gated-delta stack.
- The previous worklog wording was too strong; target-only CUDA graph parity had
  not been proven.

### No-MTP Baseline

- First attempted the clean default registry command:
  `TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --use-registry`
- Sandboxed run failed in OpenMPI init:
  `No network interfaces were found for out-of-band communications`.
- Escalated rerun reached CLI config validation but stopped because this branch
  now has two registry entries for the same HF model:
  `qwen3_5_moe_400b_fp8` and `qwen3_5_moe_400b_fp8_mtp`.
- Final command:
  `TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --use-registry --registry-config-id qwen3_5_moe_400b_fp8`
- Log: `/tmp/qwen35_ad_nomtp_default_registry.log`.
- Result: passed end to end.
- Important observations:
  - The run used the base non-MTP registry entry.
  - `compile_backend: torch-cudagraph`.
  - CUDA graph capture completed for all default configured batch sizes, from
    256 down to 1.
  - Prompt generation completed for all 10 example prompts.
- Conclusion: Qwen3.5 target-only AutoDeploy CUDA graph capture works with the
  default registry config. The earlier MTP CUDA-graph failure is therefore not a
  blanket target-model CUDA graph incompatibility; it is specific to the MTP /
  speculative path or to config differences in that path.

### MTP Overlay Change

- Updated `examples/auto_deploy/model_registry/configs/qwen3.5_moe_400b_mtp.yaml`
  to disable:
  - `transforms.multi_stream_gemm.enabled: false`
  - `transforms.multi_stream_moe.enabled: false`
- Why: multi-stream GEMM/MoE has caused problems in prior MTP work, and the MTP
  overlay should avoid that extra variable before the next CUDA-graph rerun.

### Status

- No-MTP CUDA graph baseline passed.
- MTP overlay now disables multi-stream GEMM and MoE.
- Next task: rerun the MTP registry entry with CUDA graph after the overlay
  change and compare the failure/success point.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Minimal FLA CUDA Graph Reproducer

### Task

Build a small smoke test that exercises the FLA gated-delta path under CUDA graph
capture without running the full Qwen3.5 model.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- The prior full MTP CUDA-graph failure was in the FLA prefill path:
  `fla_cached_gated_delta_rule -> chunk_gated_delta_rule -> chunk_local_cumsum
  -> prepare_chunk_indices`, where `prepare_chunk_indices` calls `.tolist()` on
  chunk counts derived from CUDA `cu_seqlens`.
- Plan:
  1. Add a focused prefill-only CUDA graph capture test to the existing
     `fla_cached_gated_delta_rule` unit tests.
  2. Run only that test on one GPU with `pytest -sv` and tee the log.
  3. If it reproduces, use it as the fast future failure case before trying a
     tiny Qwen3.5 `linear_attention` model-level smoke.

### Result

- Added `test_prefill_cuda_graph_capture` to
  `tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py`.
- The test is intentionally focused on prefill-only
  `torch.ops.auto_deploy.fla_cached_gated_delta_rule`; decode-only would not
  call `chunk_gated_delta_rule` or `prepare_chunk_indices`.
- Warmup and capture use distinct `cu_seqlen` tensor objects so FLA's
  identity-based `tensor_cache` cannot hide the capture-time path.
- Commands:
  - `python -m py_compile tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py`
  - `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py::test_prefill_cuda_graph_capture 2>&1 | tee /tmp/qwen35_fla_cached_gdr_cudagraph_smoke.log`
- The sandboxed pytest command hit the known OpenMPI init failure before the
  test body, so it was rerun outside the sandbox with `set -o pipefail`.
- Log: `/tmp/qwen35_fla_cached_gdr_cudagraph_smoke.log`.
- Result: reproduced the CUDA graph failure in the focused op-only test.
- Failure point:
  `fla_cached_gated_delta_rule -> chunk_gated_delta_rule ->
  chunk_local_cumsum -> prepare_chunk_indices`, at
  `triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()`.
- Error:
  `torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing`.
- Why this is useful: it isolates the same FLA prefill capture failure without
  exporting/loading/running the full Qwen3.5 model.
- New commit: none; changes remain uncommitted.

## Qwen3Next GDN MTP Target-Verify Reference

### Task

Document the PyTorch backend Qwen3Next gated-delta MTP target-verify path so the
AutoDeploy FLA cached gated-delta fix has a concrete reference.

### Result

- Added note:
  [qwen3next_gdn_mtp_target_verify.md](qwen3next_gdn_mtp_target_verify.md).
- Key conclusion: Qwen3Next does not absorb MTP target-verify tokens into a
  prefill path. It treats the verification region as
  `num_decodes * (max_total_draft_tokens + 1)`, writes candidate conv/GDN states
  into intermediate speculative buffers with state updates disabled, and lets
  the cache manager commit only the accepted candidate state after sampling.
- Implication for AutoDeploy FLA cached GDN: using
  `BatchInfo.get_absorbed_info()` is the wrong abstraction for the MTP-capable
  path. The op likely needs natural `num_prefill`, `num_extend`, `num_decode`
  handling plus an explicit extend/target-verify branch analogous to
  `triton_backend_mamba`.

## FLA Extend/Target-Verify Path

### Task

Replace the FLA gated-delta op's speculative extend behavior with a
target-verify path that avoids absorbing extend tokens into prefill and records
intermediate recurrent states for later acceptance.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Problem: `fla_cached_gated_delta_rule()` used `BatchInfo.get_absorbed_info()`,
  so extend requests were executed through the FLA chunk/prefill path. That path
  has host-side work (`.tolist()` through chunk-index preparation) and fails
  during CUDA graph capture.
- Cache-manager contract: `MambaCacheManager.update_mamba_states()` reads
  intermediate states from compact indices `intermediate_state_indices[:num_gens]`.
  Therefore the op must write intermediate candidate states at compact
  per-generation indices, while using real state-slot indices only to load the
  live starting states.

### Plan

1. Split FLA handling into natural prefill, extend, and decode regions from
   `BatchInfo.get_num_sequences()` and `BatchInfo.get_num_tokens()`.
2. For speculative extend, gather live `delta_cache[slot_idx_extend]`, call the
   fused recurrent GDN update with state updates disabled, and write
   intermediate states at `torch.arange(num_extend)`.
3. Register the base delta state as an `SSMResourceHandler` and the intermediate
   delta state as a `SpecSSMResourceHandler` so the hybrid cache manager can
   promote the accepted candidate through `update_mamba_states()`.
4. Verify with an extend-only output/state reference test and an extend CUDA
   graph capture test before rerunning the whole FLA op test file.

### Result

- Updated `fla_cached_gated_delta_rule()` to use natural prefill, extend, and
  decode counts instead of `get_absorbed_info()`.
- True prefill still uses `chunk_gated_delta_rule()` and commits final states to
  the real `delta_cache` slots.
- Speculative extend/target-verify now:
  - reshapes tokens as `[num_extend, tokens_per_extend, ...]`;
  - gathers live recurrent state from `delta_cache[slot_idx_extend]`;
  - uses compact `torch.arange(num_extend)` indices for the intermediate
    buffer, matching `MambaCacheManager.update_mamba_states()`;
  - runs `fused_recurrent_gated_delta_rule_update()` with
    `disable_state_update=True`;
  - writes per-step candidate states to `intermediate_delta_cache` without
    mutating `delta_cache`.
- Non-speculative extend falls back to the fused sigmoid recurrent update and
  mutates the real `delta_cache`.
- Marked both `delta_cache` and `intermediate_delta_cache` in the custom op
  `mutates_args`.
- Changed FLA GDN cache resources from generic local state to managed recurrent
  state:
  - `delta_cache`: `SSMResourceHandler`
  - `intermediate_delta_cache`: `SpecSSMResourceHandler`
- Rationale: the delta state is not merely scratch storage; it has the same
  lifecycle as Mamba SSM state for MTP target verification. It must be managed
  so accepted candidate states can be promoted by the existing mamba cache
  manager path after sampling.

### Tests

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/custom_ops/fla/fla_backend_gated_delta.py tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py`
- Sandbox pytest hit the known OpenMPI init failure before collection:
  `No network interfaces were found for out-of-band communications`.
- Escalated focused tests:
  `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py::test_extend_only_matches_prefill_reference tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py::test_extend_cuda_graph_capture`
  - Result: 3 passed.
  - Log: `/tmp/qwen35_fla_cached_gdr_extend_intermediate_tests.log`.
- Full FLA op and cache-binding validation after `mutates_args` fix:
  `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py tests/unittest/auto_deploy/singlegpu/shim/test_cached_sequence_interface.py::test_intermediate_state_resources_bind_via_managed_state_path`
  - Result: 10 passed.
  - Log: `/tmp/qwen35_fla_after_mutates_and_binding.log`.

### Status

- Done for the focused op-level fix.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: rerun the Qwen3.5 MTP CUDA-graph E2E path to see whether the FLA
  target-verify fix clears the previous capture failure.

## Qwen3.5 MTP Runtime Overlay and Draft Sharding

### Task

Create a separate MTP registry overlay for Qwen3.5 that reduces serving shape
and disables known-risk multi-stream transforms, then rerun the 8-GPU CUDA-graph
MTP smoke.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- The full MTP registry path had reached cache initialization, but failed either
  from memory pressure with the full non-MTP serving shape or from mixed target /
  draft KV cache block-offset multipliers when the draft attention stayed
  unsharded.
- Goal: keep target and drafter KV resources in the same managed layout by
  matching sharding/dtype, not by adding layer-specific block-offset remapping.

### Plan

1. Add an MTP-specific registry config overlay on top of the existing Qwen3.5
   400B config.
2. Reduce `max_batch_size` to 32, reduce `max_num_tokens`, lower cache free
   memory fraction, and disable multi-stream GEMM/MoE.
3. Let the draft attention receive the same manual TP treatment as the target,
   while avoiding target-only manual-plan assumptions that do not apply to the
   MTP `fc` prologue.
4. Validate with focused unit tests before rerunning the full 8-GPU path.

### Result

- Added `qwen3_5_moe_400b_fp8_mtp` as a separate registry config ID and
  `qwen3.5_moe_400b_mtp.yaml` as an overlay.
- The overlay currently uses:
  - `max_batch_size: 32`
  - `cuda_graph_config.max_batch_size: 32`
  - `cuda_graph_config.batch_sizes: [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]`
  - `max_num_tokens: 8192`
  - `kv_cache_config.free_gpu_memory_fraction: 0.7`
  - `speculative_config.decoding_type: MTP`
  - `speculative_config.max_draft_len: 1`
  - `speculative_config.mtp_eagle_one_model: true`
  - `transforms.insert_cached_causal_conv.backend: triton_causal_conv`
  - `transforms.multi_stream_gemm.enabled: false`
  - `transforms.multi_stream_moe.enabled: false`
- Full 8-GPU rerun with the 32-batch overlay:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python examples/auto_deploy/build_and_run_ad.py --model Qwen/Qwen3.5-397B-A17B-FP8 --use-registry --registry-config-id qwen3_5_moe_400b_fp8_mtp`
  - Log: `/tmp/qwen35_ad_mtp_cudagraph_b32_after_draft_shard.log`.
  - Result: failed during draft manual sharding before cache initialization.
  - Error: `Start linear node (index 0) not found in opening linear nodes -
    start_linear node: model_layers_fc_torch_linear_simple`.
  - Why: applying the target-authored manual TP plan to the draft graph is
    directionally needed for matching attention/KV sharding, but the generic
    layer-boundary detector was treating the MTP `fc` prologue (`2h -> h`) as
    the start of a decoder layer.
- Follow-up fix: `get_all_layer_subgraphs()` now skips draft prologue linears
  during layer-boundary detection when they project into the hidden width but do
  not directly feed attention. The skipped prologue remains in the unprocessed
  set and is not sharded by the Qwen manual TP plan. The draft q/k/v/o attention
  projections can then be matched and sharded by the same manual config.
- Added a CPU regression test that builds a draft-like FX graph
  `fc -> q/k/v -> torch_attention -> o` and verifies manual sharding applies
  only to q/k/v/o.

### Tests

- `python -m py_compile tensorrt_llm/_torch/auto_deploy/utils/node_utils.py tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py`
- Sandboxed focused pytest hit the known OpenMPI local socket init failure
  before collection.
- Escalated focused test:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_draft_mtp_prologue_does_not_break_manual_attention_sharding`
  - Result: 1 passed.
  - Log: `/tmp/qwen35_draft_prologue_sharding_unit.log`.
- Escalated focused regression set:
  `pytest -sv tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_draft_mtp_prologue_does_not_break_manual_attention_sharding tests/unittest/auto_deploy/singlegpu/utils/test_node_utils_sharding.py::test_config_sharding_skips_linear_without_weight_name tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe_mtp.py`
  - Result: 8 passed.
  - Log: `/tmp/qwen35_draft_prologue_and_mtp_unit.log`.

### Status

- Focused draft-sharding fix passed unit coverage.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: rerun the 8-GPU Qwen3.5 MTP CUDA-graph smoke with the 32-batch
  overlay and check whether cache initialization now uses uniform KV block
  offsets.

## FLA Intermediate State Semantics

### Task

Strengthen the op-level test so it proves every intermediate SSM state written
by FLA extend matches the state obtained by running the same sequence prefix.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Existing test checked only the final intermediate state for each extend
  sequence. That did not explicitly prove the first intermediate state was the
  one-step update result, or that all candidate acceptance points were stored
  correctly.

### Plan

Update `test_extend_only_matches_prefill_reference` to compare
`intermediate_delta_cache[seq_idx, step]` against a reference
`chunk_gated_delta_rule()` run over the prefix ending at `step`.

### Result

- Strengthened `test_extend_only_matches_prefill_reference` so every
  intermediate candidate state is checked, including the one-token update case.
- The per-step state reference uses `fused_recurrent_gated_delta_rule_fwd()`,
  matching the fused recurrent update family used by the extend path. A first
  attempt compared candidate states against the chunk prefill kernel and hit a
  small kernel-numerics mismatch (`0.007` absolute vs `0.005` tolerance), so the
  final test keeps the output reference on chunk GDN while comparing internal
  recurrent states against the recurrent reference.

### Tests

- `python -m py_compile tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py`
- Sandboxed focused pytest hit the known OpenMPI init failure before collection.
- Escalated focused test:
  `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py::test_extend_only_matches_prefill_reference`
  - Result: 2 passed.
  - Log: `/tmp/qwen35_fla_intermediate_prefix_test.log`.
- Full FLA op file:
  `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py`
  - Result: 9 passed.
  - Log: `/tmp/qwen35_fla_full_after_prefix_test.log`.

### Status

- Done.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Full Qwen3.5 FP8 MTP CUDA-Graph B32 Smoke

### Task

Rerun the real 8-GPU Qwen3.5-397B-A17B-FP8 MTP AutoDeploy path with the
dedicated batch-32 registry overlay after the draft-sharding hidden-size fix.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Previous full MTP CUDA-graph attempts had either failed from memory pressure
  or from draft manual-sharding issues before generation.
- The active suspicion was that the overlay may need `max_batch_size: 32`, and
  that any remaining cache-manager/dummy-slot issue should show up in the full
  runtime path.

### Command

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home \
TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 \
python examples/auto_deploy/build_and_run_ad.py \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --use-registry \
  --registry-config-id qwen3_5_moe_400b_fp8_mtp \
2>&1 | tee /tmp/qwen35_ad_mtp_cudagraph_b32_after_embd_fix.log
```

### Result

- Passed end to end.
- The run used the MTP overlay with:
  - `max_batch_size: 32`
  - `max_num_tokens: 8192`
  - `max_draft_len: 1`
  - `multi_stream_gemm.enabled: false`
  - `multi_stream_moe.enabled: false`
  - `triton_causal_conv` state cache backend
- Sharding reached the desired shape:
  - target graph: `Simple: 0, row-col: 60 (attention: 15, delta: 45...)`
  - draft graph: `Simple: 0, row-col: 1 (attention: 1...)`
  - the previous draft `k_proj` view shape-prop failure did not recur.
- Weight load completed on all ranks. After load, per-rank allocation was about
  53.14 GiB; after CUDA-graph capture, allocation was about 57.41 GiB.
- CUDA graph capture completed for all configured batch sizes:
  `[32, 16, 8, 7, 6, 5, 4, 3, 2, 1]`.
- The script generated all 10 example prompts and destroyed the process group
  cleanly.
- No cache-manager or dummy-slot failure was observed in this B32 smoke.

### Tests

- Log: `/tmp/qwen35_ad_mtp_cudagraph_b32_after_embd_fix.log`.

### Status

- Full B32 CUDA-graph smoke passed.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: run a CUDA-graph stats probe and verify nonzero speculative
  drafting/acceptance on the successful path.

## CUDA-Graph MTP Acceptance Stats Probe

### Task

Run the stats probe under CUDA graph, not eager, to verify that the successful
CUDA-graph MTP path produces nonzero drafted and accepted token counts.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Prior eager stats probe passed with `73/95` accepted drafted tokens
  (`acceptance_rate ~= 0.768`), but the CUDA-graph path had not yet reported
  `LLM.get_stats()` values.

### Attempt 1

- Updated `debug/qwen35-mtp/qwen35_mtp_stats_probe.py` to accept:
  - `QWEN35_MTP_COMPILE_BACKEND`
  - `QWEN35_MTP_MAX_BATCH_SIZE`
  - `QWEN35_MTP_STATS_OUT`
- Command used `QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph` and
  `QWEN35_MTP_MAX_BATCH_SIZE=4`.
- The run started, reached export/pattern matching/quantization transforms, but
  the tool session was interrupted before completion.
- After the interruption, no GPU processes remained. The intended `/tmp` log and
  JSON result were not present, so this attempt is treated as inconclusive
  rather than pass or fail.

### Status

- CUDA-graph stats probe rerun passed.
- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_MTP_STATS_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/debug/qwen35-mtp/logs/qwen35_mtp_stats_probe_cudagraph.json python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- Log:
  `debug/qwen35-mtp/logs/qwen35_mtp_stats_probe_cudagraph.log`.
- JSON:
  `debug/qwen35-mtp/logs/qwen35_mtp_stats_probe_cudagraph.json`.
- Result:
  - `spec_iters`: 34
  - `total_drafted`: 131
  - `total_accepted`: 123
  - `acceptance_rate`: 0.9389312977099237
  - `num_stats`: 35
- CUDA graph capture completed for batch sizes `[4, 1]`, prompt generation
  completed, and all ranks destroyed the process group cleanly.
- The sharding logs again showed target `Simple: 0, row-col: 60` and draft
  `Simple: 0, row-col: 1`, so this is the fixed shared-sharding path.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.
- Next task: run a FlashInfer + `torch-simple` smoke/stats probe to separate the
  MTP/FLA path from TRTLLM attention block-offset handling.

## FlashInfer + Torch-Simple MTP Stats Probe

### Task

Run the same MTP stats probe with `attn_backend=flashinfer` and
`torch-simple`. This avoids the TRTLLM attention block-offset path while still
exercising Qwen3.5 MTP, FLA/GDN state updates, checkpoint loading, sharding, and
the speculative decoding loop.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Prior CUDA-graph/TRTLLM-attention stats probe passed with nonzero MTP stats:
  `123/131` accepted drafted tokens (`acceptance_rate ~= 0.939`).
- The question being isolated here is whether the speculative path is healthy
  independent of TRTLLM attention's same-block-offset requirement.

### Result

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=flashinfer QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_MTP_STATS_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/debug/qwen35-mtp/logs/qwen35_mtp_stats_probe_flashinfer_torch_simple.json python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- Log:
  `debug/qwen35-mtp/logs/qwen35_mtp_stats_probe_flashinfer_torch_simple.log`.
- JSON:
  `debug/qwen35-mtp/logs/qwen35_mtp_stats_probe_flashinfer_torch_simple.json`.
- Result:
  - `spec_iters`: 34
  - `total_drafted`: 130
  - `total_accepted`: 125
  - `acceptance_rate`: 0.9615384615384616
  - `num_stats`: 35
- The run completed all four prompts and all ranks destroyed the process group
  cleanly.
- Sharding again reached target `Simple: 0, row-col: 60` and draft
  `Simple: 0, row-col: 1`, matching the intended target/draft sharding contract.
- This supports the current interpretation: the previous TRTLLM block-offset
  assertion was a useful signal of incorrect draft sharding, not a reason to
  weaken the same-offset requirement. With the sharding fix, the MTP path works
  with TRTLLM attention under CUDA graph and with FlashInfer under eager
  `torch-simple`.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## Controlled Output/Acceptance Comparison Matrix

### Task

Run a local `ad_runs/` comparison matrix before moving to accuracy:

1. No MTP, FlashInfer attention, `torch-simple`.
2. MTP, FlashInfer attention, `torch-simple` baseline.
3. MTP, TRTLLM attention, `torch-simple`.
4. MTP, TRTLLM attention, `torch-cudagraph`.

The goal is to verify that MTP preserves output relative to no-MTP and that
acceptance rate/output behavior is stable across attention backend and compile
backend.

### Plan

- Save all logs and JSON artifacts under `ad_runs/`.
- Use the same probe script for every case, with the same prompt set, sampling
  parameters (`temperature=0`, `max_tokens=64`), serving shape
  (`max_batch_size=4`, `max_num_tokens=8192`, `free_gpu_memory_fraction=0.7`),
  chunked prefill disabled, and multi-stream GEMM/MoE disabled.
- For no-MTP, use registry config `qwen3_5_moe_400b_fp8` plus the same runtime
  overrides; for MTP, use `qwen3_5_moe_400b_fp8_mtp`.
- Compare decoded output strings exactly first. If not exact, record the first
  differing prompt and inspect whether the mismatch is backend numerical drift
  or a speculative decoding issue.
- Compare MTP acceptance summaries across the three MTP runs.

### Starting Point

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`.
- Updated `debug/qwen35-mtp/qwen35_mtp_stats_probe.py` so artifacts record the
  registry config ID, attention backend, compile backend, max batch size, token
  budget, and runtime overrides.
- `python -m py_compile debug/qwen35-mtp/qwen35_mtp_stats_probe.py` passed.

### No-MTP FlashInfer Torch-Simple Baseline

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8 QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=flashinfer QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_MTP_STATS_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/no_mtp_flashinfer_torch_simple.json python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- First attempt reached generation but failed while writing the artifact because
  the probe assumed `specDecodingStats` was a dict. In no-MTP runs it can be
  `None`. This was a probe bug, not a model/runtime failure.
- Fixed `_summarize_acceptance()` to treat missing speculative stats as an
  empty dict. `python -m py_compile debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
  passed after the fix.
- Rerun completed successfully:
  - Log: `ad_runs/no_mtp_flashinfer_torch_simple.log`
  - JSON: `ad_runs/no_mtp_flashinfer_torch_simple.json`
  - `num_stats`: 65
  - Speculative stats: not applicable; no MTP/drafting configured.
- Runtime details:
  - `registry_config_id=qwen3_5_moe_400b_fp8`
  - `compile_backend=torch-simple`
  - `attn_backend=flashinfer`
  - `max_batch_size=4`
  - `max_num_tokens=8192`
  - chunked prefill disabled
  - multi-stream GEMM/MoE disabled
- Sharding reached target-only manual sharding:
  `Applied 61 TP shards from config. Simple: 1, row-col: 60 (attention: 15, delta: 45...)`.
- The run generated all four prompts and all ranks destroyed the process group
  cleanly.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### MTP FlashInfer Torch-Simple

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=flashinfer QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_MTP_STATS_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/mtp_flashinfer_torch_simple.json python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- Log: `ad_runs/mtp_flashinfer_torch_simple.log`
- JSON: `ad_runs/mtp_flashinfer_torch_simple.json`
- Result:
  - `num_stats`: 35
  - `spec_iters`: 34
  - `total_drafted`: 130
  - `total_accepted`: 125
  - `acceptance_rate`: 0.9615384615384616
- Runtime details:
  - `registry_config_id=qwen3_5_moe_400b_fp8_mtp`
  - `compile_backend=torch-simple`
  - `attn_backend=flashinfer`
  - `max_batch_size=4`
  - `max_num_tokens=8192`
  - chunked prefill disabled
  - multi-stream GEMM/MoE disabled
- Sharding reached the intended target/draft split:
  - target: `Simple: 0, row-col: 60 (attention: 15, delta: 45...)`
  - draft: `Simple: 0, row-col: 1 (attention: 1...)`
- The run generated all four prompts and all ranks destroyed the process group
  cleanly.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

### MTP TRTLLM Torch-Simple

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_MTP_STATS_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/mtp_trtllm_torch_simple.json python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- Log: `ad_runs/mtp_trtllm_torch_simple.log`
- JSON: `ad_runs/mtp_trtllm_torch_simple.json`
- Result:
  - `num_stats`: 36
  - `spec_iters`: 35
  - `total_drafted`: 132
  - `total_accepted`: 125
  - `acceptance_rate`: 0.946969696969697
- Runtime details:
  - `registry_config_id=qwen3_5_moe_400b_fp8_mtp`
  - `compile_backend=torch-simple`
  - `attn_backend=trtllm`
  - `max_batch_size=4`
  - `max_num_tokens=8192`
  - chunked prefill disabled
  - multi-stream GEMM/MoE disabled
- Sharding reached the intended target/draft split:
  - target: `Simple: 0, row-col: 60 (attention: 15, delta: 45...)`
  - draft: `Simple: 0, row-col: 1 (attention: 1...)`
- The run generated all four prompts and all ranks destroyed the process group
  cleanly.

### MTP TRTLLM Torch-Cudagraph

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_MTP_STATS_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/mtp_trtllm_torch_cudagraph.json python debug/qwen35-mtp/qwen35_mtp_stats_probe.py`
- Log: `ad_runs/mtp_trtllm_torch_cudagraph.log`
- JSON: `ad_runs/mtp_trtllm_torch_cudagraph.json`
- Result:
  - `num_stats`: 35
  - `spec_iters`: 34
  - `total_drafted`: 131
  - `total_accepted`: 123
  - `acceptance_rate`: 0.9389312977099237
- Runtime details:
  - `registry_config_id=qwen3_5_moe_400b_fp8_mtp`
  - `compile_backend=torch-cudagraph`
  - `attn_backend=trtllm`
  - `max_batch_size=4`
  - `max_num_tokens=8192`
  - chunked prefill disabled
  - multi-stream GEMM/MoE disabled
- CUDA graph capture completed for batch sizes `[4, 1]`. The run generated all
  four prompts and all ranks destroyed the process group cleanly.

### Matrix Comparison

- No-MTP FlashInfer `torch-simple` is an output-only baseline. Acceptance rate
  is not applicable because there is no drafting or acceptance path.
- MTP acceptance summaries are close across the three MTP rows:
  - FlashInfer `torch-simple`: `125/130 = 0.9615384615384616`
  - TRTLLM `torch-simple`: `125/132 = 0.946969696969697`
  - TRTLLM `torch-cudagraph`: `123/131 = 0.9389312977099237`
- Exact decoded strings are not fully invariant:
  - No-MTP FlashInfer vs MTP FlashInfer: prompts 0, 2, and 3 match exactly;
    prompt 1 differs in wording/formatting.
  - MTP FlashInfer vs MTP TRTLLM `torch-simple`: prompts 0, 1, and 2 match
    exactly; prompt 3 differs in wording.
  - MTP TRTLLM `torch-simple` vs MTP TRTLLM `torch-cudagraph`: prompts 0 and 1
    match exactly; prompts 2 and 3 differ in wording.
  - MTP FlashInfer vs MTP TRTLLM `torch-cudagraph`: prompts 0 and 1 match
    exactly; prompts 2 and 3 differ in wording.
- Interpretation: the four smoke rows prove that Qwen3.5 MTP runs end-to-end
  with FlashInfer/torch-simple, TRTLLM/torch-simple, and TRTLLM/cudagraph, and
  that acceptance is nontrivially above zero. The exact-output mismatch needs
  investigation before treating this as accuracy-equivalent to no-MTP.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## GSM8K Accuracy Sample: FlashInfer Torch-Simple

### Task

Run a GSM8K accuracy probe with thresholds disabled, comparing no-MTP and MTP
under the same FlashInfer `torch-simple` runtime config.

### Plan

- Add a temporary `debug/qwen35-mtp/qwen35_gsm8k_probe.py` probe that uses the
  local lm-eval GSM8K evaluator directly and records observed accuracy plus
  speculative-decoding stats without asserting either threshold.
- Use a bounded `128` sample run first rather than the full `1319` sample set.
- Save logs, JSON summaries, and lm-eval samples under `ad_runs/`.

### No-MTP FlashInfer Torch-Simple

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8 QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=flashinfer QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_GSM8K_NUM_SAMPLES=128 QWEN35_GSM8K_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_no_mtp_flashinfer_torch_simple_samples QWEN35_GSM8K_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_no_mtp_flashinfer_torch_simple.json python debug/qwen35-mtp/qwen35_gsm8k_probe.py`
- Log: `ad_runs/gsm8k_no_mtp_flashinfer_torch_simple.log`
- JSON: `ad_runs/gsm8k_no_mtp_flashinfer_torch_simple.json`
- Samples: `ad_runs/gsm8k_no_mtp_flashinfer_torch_simple_samples/`
- Result:
  - samples: `128`
  - accuracy: `62.109375`
  - speculative stats: not applicable; no drafting configured
  - `num_stats`: `1001`
- The run completed and all ranks destroyed the process group cleanly.

### MTP FlashInfer Torch-Simple

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-simple QWEN35_MTP_ATTN_BACKEND=flashinfer QWEN35_MTP_MAX_BATCH_SIZE=4 QWEN35_GSM8K_NUM_SAMPLES=128 QWEN35_GSM8K_OUTPUT_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_mtp_flashinfer_torch_simple_samples QWEN35_GSM8K_OUT=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/gramnarayan/dev/qwen3-mtp/ad_runs/gsm8k_mtp_flashinfer_torch_simple.json python debug/qwen35-mtp/qwen35_gsm8k_probe.py`
- Log: `ad_runs/gsm8k_mtp_flashinfer_torch_simple.log`
- JSON: `ad_runs/gsm8k_mtp_flashinfer_torch_simple.json`
- Samples: `ad_runs/gsm8k_mtp_flashinfer_torch_simple_samples/`
- Result:
  - samples: `128`
  - accuracy: `62.109375`
  - `spec_iters`: `1001`
  - `total_drafted`: `9133`
  - `total_accepted`: `8480`
  - `acceptance_rate`: `0.9285010401839483`
  - `num_stats`: `1001`
- The run completed and all ranks destroyed the process group cleanly.

### Result

- On this 128-sample GSM8K slice, MTP and no-MTP match exactly on observed
  accuracy: `62.109375`.
- MTP acceptance is nontrivially high: `8480/9133 = 0.9285010401839483`.
- Both rows emitted repeated `storeContextBlocks: Can not find sequence`
  warnings from the KV cache manager during response fetching. They did not
  abort either run, but the warning is worth tracking separately.
- Current commit remains `3ae0b706046c447783943923712a1310f0d844d5`; changes are
  uncommitted in the working tree.

## 2026-05-18: Full GSM8K FP8 Accuracy and Small-Path Feasibility

### Task

Run full GSM8K on the real Qwen3.5-397B-A17B-FP8 snapshot for no-MTP and MTP,
record accuracy/acceptance, then check whether the existing 35B "small" path can
run locally without MTP.

### Starting State

- Commit: `3ae0b706046c447783943923712a1310f0d844d5`
- Snapshot:
  `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d`
- Added temporary FP8 GSM8K test coverage in
  `tests/integration/defs/accuracy/test_llm_api_autodeploy.py`:
  - no-MTP comparison: `test_fp8_gsm8k[8]`
  - MTP candidate: `test_fp8_mtp_gsm8k[8]`
- Both tests set `TRTLLM_ACCURACY_NO_REFERENCE=1` so the observed score can be
  recorded before selecting CI thresholds.

### No-MTP FP8 Full GSM8K, Default Registry Shape

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home QWEN35_FP8_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 pytest -sv tests/integration/defs/accuracy/test_llm_api_autodeploy.py::TestQwen3_5_397B_MoE::test_fp8_gsm8k[8] 2>&1 | tee ad_runs/full_gsm8k_fp8_no_mtp_pytest.log`
- Log: `ad_runs/full_gsm8k_fp8_no_mtp_pytest.log`
- Result:
  - Reached AutoDeploy build, full FP8 weight load, CUDA graph capture, and
    GSM8K request submission.
  - OOMed during the first forward in
    `trtllm_finegrained_fp8_linear_aux` / `fp8_quantize_1x128`.
  - The default no-MTP registry shape captured batch sizes up to `256` with
    `max_num_tokens=16000`, leaving only about `65 MiB` free on GPU 0 before a
    `64 MiB` allocation.
  - No accuracy result was produced.
- Interpretation:
  - This failure is a serving-shape memory issue, not evidence about MTP
    correctness or GSM8K accuracy.
  - A quick unit-test run on GPU 1 was unlikely to be the deciding factor:
    GPU 0 was the OOM rank and all GPUs were clear after cleanup.

### Next Plan

- Rerun no-MTP full GSM8K with the same small runtime envelope used for MTP:
  `max_batch_size=32`, `max_num_tokens=8192`, CUDA graph batches up to `32`,
  block reuse disabled, `free_gpu_memory_fraction=0.7`, and multi-stream
  GEMM/MoE disabled.
- If the no-MTP small-envelope run succeeds, run the MTP FP8 full GSM8K test and
  compare observed accuracy plus acceptance rate.

### No-MTP FP8 Full GSM8K, Small MTP-Like Serving Shape

- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home QWEN35_FP8_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 pytest -sv tests/integration/defs/accuracy/test_llm_api_autodeploy.py::TestQwen3_5_397B_MoE::test_fp8_gsm8k[8] 2>&1 | tee ad_runs/full_gsm8k_fp8_no_mtp_smallshape_pytest.log`
- Log: `ad_runs/full_gsm8k_fp8_no_mtp_smallshape_pytest.log`
- Runtime shape:
  - `max_batch_size=32`
  - `max_num_tokens=8192`
  - CUDA graph batch sizes: `[1, 2, 3, 4, 5, 6, 7, 8, 16, 32]`
  - `enable_chunked_prefill=false`
  - `kv_cache_config.free_gpu_memory_fraction=0.7`
  - `multi_stream_gemm` and `multi_stream_moe` disabled
- Result:
  - `PASSED`
  - Full GSM8K `1319` samples completed.
  - Flexible exact match: `87.7938`
  - Strict exact match: `86.6566`
  - Average accuracy: `87.23`
  - TRTLLM execution time: `620.645` seconds
  - Total pytest duration: `1196.44` seconds
- Interpretation:
  - The 397B FP8 no-MTP baseline is viable on 8xH100 with the MTP-like serving
    envelope.
  - The previous no-MTP failure was caused by the default large serving shape,
    not by the FP8 weights or core Qwen3.5 graph.
  - Next comparison point is the full MTP FP8 GSM8K run under the corresponding
    registry MTP config.

### MTP FP8 Full GSM8K, Small MTP Serving Shape

- Commit: `562a273de79bc9466ad885e3a13d08f5117018e2`
- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home QWEN35_FP8_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 pytest -sv tests/integration/defs/accuracy/test_llm_api_autodeploy.py::TestQwen3_5_397B_MoE::test_fp8_mtp_gsm8k[8] 2>&1 | tee ad_runs/full_gsm8k_fp8_mtp_pytest.log`
- Log: `ad_runs/full_gsm8k_fp8_mtp_pytest.log`
- Runtime shape:
  - registry config: `qwen3_5_moe_400b_fp8_mtp`
  - `max_batch_size=32`
  - `max_num_tokens=8192`
  - CUDA graph batch sizes: `[1, 2, 3, 4, 5, 6, 7, 8, 16, 32]`
  - `enable_chunked_prefill=false`
  - `kv_cache_config.free_gpu_memory_fraction=0.7`
  - `multi_stream_gemm` and `multi_stream_moe` disabled
- Build evidence:
  - `EagleDrafterFactory` built a drafter for `model_type='qwen3_5_moe_text'`.
  - `detect_hidden_states_for_capture` matched one hidden-state capture point
    per rank.
  - Sharding used `TP=420, EP=60` for the target graph and `TP=4, EP=1` for the
    draft graph.
  - Full FP8 weights loaded without missing or unexpected weight failures.
  - CUDA graph capture completed for batch sizes `[32, 16, 8, 7, 6, 5, 4, 3, 2,
    1]`.
- Result:
  - `PASSED`
  - Full GSM8K `1319` samples completed.
  - Flexible exact match: `88.0212`
  - Strict exact match: `87.2631`
  - Average accuracy: `87.64`
  - TRTLLM execution time: `456.385` seconds
  - Total pytest duration: `1304.98` seconds
  - Acceptance-rate counters were not emitted by this pytest path.
- Comparison to no-MTP small-shape baseline:
  - No-MTP average accuracy: `87.23`
  - MTP average accuracy: `87.64`
  - Difference: `+0.41` points for MTP, within normal lm-eval noise for this
    run.
  - No-MTP TRTLLM execution time: `620.645` seconds
  - MTP TRTLLM execution time: `456.385` seconds
- Interpretation:
  - Full-size Qwen3.5 FP8 MTP runs end-to-end on 8xH100 with TRTLLM attention,
    CUDA graph capture, overlap scheduler, and the reduced MTP serving envelope.
  - Accuracy is aligned with the full no-MTP FP8 baseline.
  - The remaining missing evidence from this exact full pytest run is
    acceptance rate; earlier 128-sample MTP probe runs recorded nontrivial
    acceptance around `0.93`.

### MTP FP8 B32 Acceptance Stats Probe

- Commit: `562a273de79bc9466ad885e3a13d08f5117018e2`
- Command:
  `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home QWEN35_MTP_MODEL=Qwen/Qwen3.5-397B-A17B-FP8 QWEN35_MTP_REGISTRY_CONFIG_ID=qwen3_5_moe_400b_fp8_mtp QWEN35_MTP_COMPILE_BACKEND=torch-cudagraph QWEN35_MTP_ATTN_BACKEND=trtllm QWEN35_MTP_MAX_BATCH_SIZE=32 QWEN35_MTP_MAX_TOKENS=128 QWEN35_MTP_STATS_OUT=ad_runs/mtp_trtllm_torch_cudagraph_b32_stats.json TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 python debug/qwen35-mtp/qwen35_mtp_stats_probe.py 2>&1 | tee ad_runs/mtp_trtllm_torch_cudagraph_b32_stats.log`
- Logs:
  - `ad_runs/mtp_trtllm_torch_cudagraph_b32_stats.log`
  - `ad_runs/mtp_trtllm_torch_cudagraph_b32_stats.json`
- Runtime shape:
  - registry config: `qwen3_5_moe_400b_fp8_mtp`
  - model: `Qwen/Qwen3.5-397B-A17B-FP8`
  - `compile_backend=torch-cudagraph`
  - `attn_backend=trtllm`
  - `max_batch_size=32`
  - `max_tokens=128`
- Build evidence:
  - `EagleDrafterFactory` built a drafter for `model_type='qwen3_5_moe_text'`.
  - `detect_hidden_states_for_capture` matched one hidden-state capture point
    per rank.
  - Sharding used `TP=420, EP=60` for the target graph and `TP=4, EP=1` for the
    draft graph.
  - `insert_cached_gated_delta_rule` matched `45` GDN nodes per rank.
  - `initialize_cache` reported `total=197/196`, `kv=16/16`, `ssm=90/90`,
    `conv=90/90`, and `max_tokens=8192`.
  - CUDA graph compile completed after cache initialization.
- Result:
  - Generation completed for 4 prompts.
  - `spec_iters=71`
  - `total_drafted=272`
  - `total_accepted=241`
  - `acceptance_rate=0.8860294117647058`
  - `num_stats=72`
- Interpretation:
  - The full-size FP8 MTP serving envelope drafts and accepts nontrivially with
    TRTLLM attention and CUDA graphs.
  - The acceptance rate is lower than the earlier small-batch short probe
    (`~0.94`) but still very healthy for these open-ended prompts.

## PR 252 Review Comment Cleanup

- Commit before task: `04cb31c2dacc79f52ed7ae603565ebb2223070d4`
- Task:
  - Address review comments on PR 252, especially around MTP registry shape,
    FLA extend-state invariants, Qwen3.5 text/composite config handling,
    strict MTP checkpoint loading, accuracy-test cleanup, and clearer modeling
    test placement/names.
- Plan:
  - Keep model-specific Qwen3.5 MTP logic auditable in the Qwen3.5 modeling
    file.
  - Remove temporary no-MTP FP8 test scaffolding and add the full FP8 MTP GSM8K
    reference measured from the completed run.
  - Make the FLA extend path always require intermediate SSM state storage, then
    test extend intermediate states against prefill-prefix states through the AD
    custom op.
  - Consolidate MTP modeling tests into the existing Qwen3.5 MoE test file and
    rename them to describe contracts rather than repeat the model family name.
- Changes:
  - Updated `qwen3.5_moe_400b_mtp.yaml` to use `max_draft_len=6`,
    `max_batch_size=8`, CUDA graph batches up to 8, and inherited multi-stream
    settings.
  - Removed the non-MTP FP8 registry entry from `models.yaml`.
  - Moved Qwen3.5 composite-config resolution into the Qwen3.5 modeling code
    and exposed it through a generic `resolve_eagle_config()` dispatch helper
    so `models/eagle.py` stays model-family agnostic.
  - Kept the production `Qwen3_5MoeConfig` CausalLM registration because the
    text-only base registry still uses `AutoModelForCausalLM`; tests no longer
    depend on this redirect.
  - Left the internal `mlp` naming in the Qwen3.5 MTP block because it follows
    the checkpoint/module convention for the block's channel mixer, even though
    the implementation is MoE-backed.
  - Reinstated the BF16 no-MTP accuracy test shape, removed the temporary FP8
    no-MTP accuracy override, and added an FP8 MTP GSM8K reference of `87.64`.
  - Renamed the FLA extend helper to `make_extend_kernel_inputs()` and rewrote
    the extend test to compare custom-op extend intermediate states against
    repeated custom-op prefill prefixes.
  - Moved MTP modeling tests into
    `tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe.py` and
    renamed them around explicit contracts such as checkpoint mapping, layer
    reference parity, factory construction, export I/O, hidden-state capture,
    and strict HF checkpoint loading.
- Verification:
  - `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe.py -k 'mtp' 2>&1 | tee debug/qwen35-mtp/logs/test_qwen3_5_moe_mtp_after_comments.log`
    - Result after final resolver cleanup: `14 passed, 41 deselected, 3 warnings in 4.32s`.
  - `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/custom_ops/fla/test_fla_cached_gated_delta_rule.py 2>&1 | tee debug/qwen35-mtp/logs/test_fla_cached_gated_delta_rule_after_comments.log`
    - Result: `9 passed, 3 warnings in 24.94s`.
  - The first FLA rerun failed because the state tolerance variables were scoped
    to the wrong test; moving them into the extend test fixed it.
- Current status:
  - Review-comment cleanup is implemented and focused unit coverage passes.
  - The new `max_draft_len=6` plus inherited multi-stream MTP registry shape has
    not yet been rerun end-to-end after this cleanup.
  - No commit has been created yet; current HEAD remains
    `04cb31c2dacc79f52ed7ae603565ebb2223070d4`.

## Full-Size FP8 MTP GSM8K Validation After Review Cleanup

- Commit before task: `04cb31c2dacc79f52ed7ae603565ebb2223070d4`
- Task:
  - Validate the polished full-size FP8 MTP accuracy test path for
    Qwen/Qwen3.5-397B-A17B-FP8 after the PR review cleanup.
  - Stop treating short stats probes as the main evidence path; use them only
    for quick sanity checks.
- Plan:
  - Run the real pytest accuracy test with the full FP8 snapshot, MTP enabled
    through the model registry, TRTLLM attention, CUDA graphs, overlap
    scheduler defaults, and the updated MTP config (`max_batch_size=8`,
    `max_draft_len=6`).
  - Check whether explicitly enabled multi-stream MoE/GEMM transforms apply.
  - Record the exact GSM8K accuracy and compare it to the stored FP8 MTP
    reference.
- Commands:
  - First attempt:
    `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 pytest -sv tests/integration/defs/accuracy/test_llm_api_autodeploy.py::TestQwen3_5_397B_MoE::test_fp8_mtp_gsm8k --tb=short 2>&1 | tee ad_runs/test_fp8_mtp_gsm8k_after_comments.log`
  - Successful rerun:
    `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LLM_MODELS_ROOT=/home/gramnarayan/dev/model-symlinks HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home QWEN35_FP8_MODEL_PATH=/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/autodeploy_data/hf_home/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d TRTLLM_DG_JIT_USE_NVCC=1 PYTHONUNBUFFERED=1 pytest -sv tests/integration/defs/accuracy/test_llm_api_autodeploy.py::TestQwen3_5_397B_MoE::test_fp8_mtp_gsm8k --tb=short 2>&1 | tee ad_runs/test_fp8_mtp_gsm8k_after_comments_with_model_path.log`
- What happened:
  - The first attempt failed before model build because
    `Qwen/Qwen3.5-397B-A17B-FP8` was not present under `LLM_MODELS_ROOT`.
  - The rerun used `QWEN35_FP8_MODEL_PATH` to point at the real local HF
    snapshot and completed successfully.
  - The full run built both target and drafter:
    - target: `Qwen3_5MoeForCausalLM`
    - drafter: `model_type='qwen3_5_moe_text'`
  - Sharding used the expected full-size split:
    - target graph: `TP=420`, `EP=60`
    - draft graph: `TP=4`, `EP=1`
  - Cache initialization covered the GDN/SSM resources:
    - `total=197/196`, `kv=16/16`, `ssm=90/90`, `conv=90/90`,
      `max_tokens=8192`
  - Cache resize completed:
    - rank 0: `max_tokens=492064`, `paged=7.51GB`,
      `non_paged=1.69GB`, `total=9.20GB`
    - other ranks: `max_tokens=492064`, `paged=7.92GB`,
      `non_paged=1.69GB`, `total=9.61GB`
  - Multi-stream transforms were actually enabled in this run:
    - `multi_stream_moe`: `matches=61` per rank
    - `multi_stream_gemm`: `matches=61` per rank
    - `multi_stream_mla_attn`: disabled
  - CUDA graph capture covered batch sizes `8, 7, 6, 5, 4, 3, 2, 1`.
  - Total AutoDeploy transform time was about `731.35s`.
  - GSM8K generation/evaluation completed in about `691.604s`.
- Result:
  - Pytest result: `1 passed, 3 warnings in 1463.99s (0:24:23)`.
  - GSM8K flexible-extract exact match: `88.097`.
  - GSM8K strict-match exact match: `86.884`.
  - GSM8K average accuracy: `87.49`.
  - Reference accuracy: `87.640`.
  - Evaluated accuracy: `87.491`.
  - Hypothesis threshold: `84.437`; the test passed.
- Interpretation:
  - This is the strongest validation so far for the intended landing shape:
    full-size FP8 Qwen3.5 MTP, registry config, TRTLLM attention, CUDA graphs,
    overlap scheduler path, managed SSM/conv cache resources, and multi-stream
    MoE/GEMM enabled.
  - The score is effectively aligned with the stored FP8 MTP reference and with
    the prior full-size FP8 MTP run.
  - The local test needs `QWEN35_FP8_MODEL_PATH` because this machine has the
    FP8 snapshot in HF cache rather than under `LLM_MODELS_ROOT`; CI/model
    staging still needs to provide the same model path through its normal model
    root mechanism.
- Current status:
  - Full-size FP8 MTP GSM8K validation passed.
  - No commit has been created yet; current HEAD remains
    `04cb31c2dacc79f52ed7ae603565ebb2223070d4`.

## Limit Text-Only Qwen3.5 MTP Config Coercion To Accuracy Test Setup

- Commit before task: `04cb31c2dacc79f52ed7ae603565ebb2223070d4`
- Task:
  - Remove the production-path workaround that made the composite
    `Qwen3_5MoeConfig` resolve to `Qwen3_5MoeForCausalLM`.
  - Keep this PR scoped to LLM-only MTP support; future VLM+MTP support should
    be implemented explicitly rather than hidden behind a global config
    redirect.
- Plan:
  - Revert the global `AutoModelForCausalLMFactory` registration for
    `Qwen3_5MoeConfig`.
  - Remove composite-to-text conversion from the generic Eagle drafter config
    path.
  - In the FP8 MTP accuracy test only, create a temporary text-only view of the
    local HF snapshot so the test exercises the LLM path without changing
    production handling of composite Qwen3.5 configs.
- Changes:
  - Removed
    `AutoModelForCausalLMFactory.register_custom_model_cls("Qwen3_5MoeConfig", Qwen3_5MoeForCausalLM)`.
  - Removed `Qwen3_5MoeForCausalLM.__init__` accepting composite configs via
    `config.text_config`.
  - Removed `resolve_eagle_config()` and the Qwen3.5 composite-to-text drafter
    resolver from the production Eagle path.
  - Added `make_fp8_mtp_text_only_model_path()` in
    `tests/integration/defs/accuracy/test_llm_api_autodeploy.py`.
    - It creates a `tmp_path` snapshot view.
    - It symlinks the real FP8 snapshot files.
    - It writes a flattened `qwen3_5_moe_text` `config.json`.
    - It preserves the top-level quantization config and still uses the real
      snapshot path as the tokenizer path.
- Verification:
  - Sandboxed pytest initially failed during local MPI/socket initialization;
    reran outside the sandbox.
  - `CUDA_VISIBLE_DEVICES=0 pytest -sv tests/unittest/auto_deploy/singlegpu/models/test_qwen3_5_moe.py -k mtp 2>&1 | tee debug/qwen35-mtp/logs/test_qwen3_5_moe_mtp_text_only_test_setup.log`
    - Result: `14 passed, 41 deselected, 3 warnings in 4.33s`.
  - Temporary text-only snapshot sanity check:
    - `AutoConfig.from_pretrained(temp_snapshot)` returned
      `Qwen3_5MoeTextConfig`.
    - `model_type == "qwen3_5_moe_text"`.
    - `mtp_num_hidden_layers == 1`.
    - Quantization config and sharded safetensors index were present.
  - `git diff --check -- . ':!triton_backend/tools/gpt/input_data.json'`
    passed.
- Interpretation:
  - This removes the production-path hack and makes the current limitation
    explicit: the new full-size accuracy test is LLM-only by construction.
  - The future VLM+MTP work should preserve and compose the original VLM target
    factory instead of relying on text-only config coercion.
- Current status:
  - Test-local text-only setup is implemented and focused unit/sanity checks
    pass.
  - The full-size GSM8K accuracy pass should be rerun once more with this new
    test-local setup if we want final end-to-end evidence for the exact landing
    diff.
  - No commit has been created yet; current HEAD remains
    `04cb31c2dacc79f52ed7ae603565ebb2223070d4`.
