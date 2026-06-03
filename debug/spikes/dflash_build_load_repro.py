# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight repro of the DFlash meta-tensor build/load blocker.

Exercises ONLY the factory build + load path (no LLM/executor/export), isolating the
"Cannot copy out of meta tensor; use to_empty()" error seen at executor init. Builds the
DFlash wrapper on ``meta`` (cheap), then probes the draft materialize+load path — the small
5-layer draft only, so we never pay the 8B target weight load.

Run: CUDA_VISIBLE_DEVICES=<free> LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     python debug/spikes/dflash_build_load_repro.py
"""

import sys

import torch

sys.path.append("tests/unittest/utils")
from llm_data import llm_models_root  # noqa: E402

from tensorrt_llm._torch.auto_deploy.models.dflash import DFlashOneModelFactory  # noqa: E402
from tensorrt_llm.llmapi import DFlashDecodingConfig  # noqa: E402


def _param_devices(mod):
    return {p.device.type for p in mod.parameters()} | {b.device.type for b in mod.buffers()}


def main():
    root = llm_models_root()
    target = f"{root}/Qwen3/Qwen3-8B"
    draft = f"{root}/Qwen3-8B-DFlash-b16"
    spec = DFlashDecodingConfig(
        max_draft_len=4,
        speculative_model=draft,
        target_layer_ids=[1, 9, 17, 25, 33],
    )
    factory = DFlashOneModelFactory(
        model=target,
        speculative_config=spec,
        max_seq_len=2048,
    )

    print(">>> build_model('meta') ...")
    model = factory.build_model("meta")
    print(f"    draft param devices after build: {_param_devices(model.draft_model)}")
    print(
        f"    draft param dtypes after build:  {set(p.dtype for p in model.draft_model.parameters())}"
    )

    device = "cuda"

    # Exercise the REAL load path (what the load_weights transform calls). After the fix this must
    # materialize + load the draft (not leave it on meta).
    print(">>> factory.load_or_random_init(model, cuda) ...")
    factory.load_or_random_init(model, device)
    devs = _param_devices(model.draft_model)
    dt = {p.dtype for p in model.draft_model.parameters()}
    print(f"    draft param devices after load: {devs}")
    print(f"    draft param dtypes after load:  {dt}")
    assert devs == {"cuda"}, f"draft not fully materialized on cuda: {devs}"
    assert dt == {torch.bfloat16}, f"draft dtype mismatch (expect bf16): {dt}"
    print("OK: draft built + loaded on cuda in bf16, no meta-tensor error.")


if __name__ == "__main__":
    main()
