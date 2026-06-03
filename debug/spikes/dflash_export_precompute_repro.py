# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight repro for the DFlash export-preservation issue (precompute_context_kv).

Exports ONLY the small draft (no 8B target, no full LLM pipeline), applies
``DFlashDraftModelExportInfo.post_process``, then checks:
  (1) ``gm.model.precompute_context_kv(hs, positions)`` runs post-export and MATCHES the pre-export
      eager model's output (re-attached modules share weights), and
  (2) the exported query-block forward ``gm(inputs_embeds, position_ids, ctx_len)`` still runs after
      the layer re-attachment.

Loop on THIS until green, then re-run the full E2E smoke.

Run: CUDA_VISIBLE_DEVICES=<free> LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models \
     python debug/spikes/dflash_export_precompute_repro.py
"""

import sys

import torch
from transformers import AutoConfig

sys.path.append("tests/unittest/utils")
from llm_data import llm_models_root  # noqa: E402

from tensorrt_llm._torch.auto_deploy.export.export import torch_export_to_gm  # noqa: E402
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_dflash import (  # noqa: E402
    DFlashDrafterForCausalLM,
)
from tensorrt_llm._torch.auto_deploy.models.dflash import DFlashDraftModelExportInfo  # noqa: E402


def main():
    torch.manual_seed(0)
    root = llm_models_root()
    draft_path = f"{root}/Qwen3-8B-DFlash-b16"
    device = "cuda"

    cfg = AutoConfig.from_pretrained(draft_path, trust_remote_code=True)
    dtype = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None) or torch.bfloat16
    draft = DFlashDrafterForCausalLM(cfg).to(device=device, dtype=dtype).eval()

    # Load the real checkpoint (strict) so weights are meaningful.
    from glob import glob

    from safetensors.torch import load_file

    state = {}
    for f in sorted(glob(f"{draft_path}/*.safetensors")):
        state.update(load_file(f, device=device))
    draft.load_state_dict({f"model.{k}": v for k, v in state.items()}, strict=True)

    H = cfg.hidden_size
    block_size = cfg.block_size
    n_capture = len(cfg.dflash_config["target_layer_ids"])
    B = 2

    # --- Pre-export eager precompute snapshot ---
    N = 5  # accepted context tokens
    captured = torch.randn(N, n_capture * H, device=device, dtype=dtype)
    positions = torch.arange(N, device=device)
    with torch.no_grad():
        k_ref, v_ref = draft.model.precompute_context_kv(captured, positions)
    print(f"eager precompute: k {tuple(k_ref.shape)} v {tuple(v_ref.shape)} dtype={k_ref.dtype}")

    # --- Export the draft (batch dynamic, seq static = block_size) ---
    example = {
        "inputs_embeds": torch.randn(B, block_size, H, device=device, dtype=dtype),
        "position_ids": torch.arange(block_size, device=device).expand(B, block_size).contiguous(),
        "ctx_len": torch.zeros(B, dtype=torch.int32, device=device),
    }
    dyn = DFlashDraftModelExportInfo()._init_dynamic_shape_lookup()
    gm = torch_export_to_gm(draft, args=(), kwargs=example, dynamic_shapes=dyn, clone=False)
    print("export OK")

    # --- post_process: re-attach precompute modules + rebind method ---
    DFlashDraftModelExportInfo().post_process(draft, gm)
    print("post_process OK; has precompute:", hasattr(gm.model, "precompute_context_kv"))

    # (1) precompute post-export matches eager
    with torch.no_grad():
        k_pe, v_pe = gm.model.precompute_context_kv(captured, positions)
    assert torch.equal(k_pe, k_ref) and torch.equal(v_pe, v_ref), "precompute mismatch post-export"
    print("OK (1): post-export precompute matches eager exactly")

    # (2) the query-block forward still runs
    with torch.no_grad():
        out = gm(**example)
    norm_hs = out.norm_hidden_state if hasattr(out, "norm_hidden_state") else out
    print(f"OK (2): query-block forward runs -> {tuple(norm_hs.shape)}")
    print("ALL GREEN")


if __name__ == "__main__":
    main()
