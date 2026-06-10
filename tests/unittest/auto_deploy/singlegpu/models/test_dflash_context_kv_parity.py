# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Numerical parity test for the AutoDeploy DFlash drafter's ``precompute_context_kv``.

DFlash projects accepted *target* hidden states into the drafter's per-layer context K/V cache
(``fc -> hidden_norm -> per-layer k_proj/v_proj -> k_norm (K only) -> RoPE (K only)``). The AD
implementation lives in ``modeling_dflash.py::DFlashModel.precompute_context_kv``; this test feeds it
the same raw hidden states + positions as the **HuggingFace vanilla DFlash modeling code** (loaded
locally via ``trust_remote_code`` from the ``Qwen3-8B-DFlash-b16`` checkpoint) and asserts the
per-layer K/V match.

The HF reference has no standalone ``precompute_context_kv`` -- the context K/V is computed inline in
``Qwen3DFlashAttention.forward`` (the ``k_ctx``/``v_ctx`` slice). Because ``k_norm`` is per-head_dim
RMSNorm and RoPE is per-position, that context slice is identical whether computed alone or cat'd with
the query stream, so we replicate the context-only path here using the HF model's own
``fc``/``hidden_norm``/``k_proj``/``v_proj``/``k_norm``/``rotary_emb`` + HF ``apply_rotary_pos_emb``.

Both sides load the SAME checkpoint, so any delta is purely the kernel/dtype path: AD's
``torch_rope_with_explicit_cos_sin`` + ``torch_rmsnorm`` vs HF ``apply_rotary_pos_emb`` +
``Qwen3RMSNorm``. We run an fp32 (tight) and a bf16 (production-dtype) variant, and compare V (no
norm/RoPE -> isolates the GEMM/load) before K (where any norm/RoPE divergence shows).
"""

import importlib
from pathlib import Path

import pytest
import torch
from _model_test_utils import assert_rmse_close
from utils.llm_data import llm_models_root

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers factories + ops)
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_dflash import DFlashDrafterForCausalLM
from tensorrt_llm._torch.auto_deploy.models.dflash import DFlashOneModelFactory
from tensorrt_llm.llmapi import DFlashDecodingConfig


def _paths():
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM_MODELS_ROOT not set")
    target = Path(root) / "Qwen3" / "Qwen3-8B"
    draft = Path(root) / "Qwen3-8B-DFlash-b16"
    if not target.is_dir() or not draft.is_dir():
        pytest.skip("Qwen3-8B / Qwen3-8B-DFlash-b16 checkpoints not found")
    return str(target), str(draft)


def _make_factory(max_draft_len: int = 4) -> DFlashOneModelFactory:
    target, draft = _paths()
    spec = DFlashDecodingConfig(max_draft_len=max_draft_len, speculative_model=draft)
    return DFlashOneModelFactory(
        model=target, speculative_config=spec, skip_loading_weights=True, max_seq_len=64
    )


def _build_ad_draft(factory: DFlashOneModelFactory, dtype: torch.dtype, device: str):
    """Build the AD ``DFlashModel`` and load the real draft checkpoint in ``dtype``.

    Mirrors ``test_dflash_factory.py::test_draft_weights_load_strict``. Building the params in
    ``dtype`` and then loading the (bf16) checkpoint upcasts the weights exactly for the fp32 run, so
    both sides see bit-identical weights and the only difference under test is the math path.
    """
    draft_config = factory._build_draft_config()
    draft_model = DFlashDrafterForCausalLM(draft_config).to(device=device, dtype=dtype).eval()
    factory._load_draft_weights(draft_model, device)
    return draft_model.model  # DFlashModel -- owns precompute_context_kv


def _build_hf_ref(draft_path: str, dtype: torch.dtype, device: str):
    """Load the HF vanilla DFlash draft model from the local checkpoint via ``trust_remote_code``."""
    from transformers import AutoModel

    hf = AutoModel.from_pretrained(draft_path, trust_remote_code=True, torch_dtype=dtype)
    return hf.to(device=device).eval()


@torch.inference_mode()
def _hf_ref_context_kv(hf, raw: torch.Tensor, positions: torch.Tensor):
    """Replicate the HF context-only K/V path, returning per-layer ``(k, v)`` ``[N, L, nkv, hd]``.

    Faithful to ``Qwen3DFlashAttention.forward``'s context slice: ``k_ctx = k_proj(target_hidden)``
    with ``target_hidden = hidden_norm(fc(raw))``, ``k = k_norm(k.view(...))``, then HF's own
    ``apply_rotary_pos_emb`` (K branch). V is ``v_proj(target_hidden)`` reshaped (no norm/RoPE).
    """
    # HF's apply_rotary_pos_emb lives in the dynamically-loaded remote module -- use HF's own op.
    apply_rotary_pos_emb = importlib.import_module(type(hf).__module__).apply_rotary_pos_emb

    cfg = hf.config
    nkv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    n = raw.shape[0]

    ctx = hf.hidden_norm(hf.fc(raw)).unsqueeze(0)  # [1, N, hidden]
    cos, sin = hf.rotary_emb(ctx, positions.unsqueeze(0))  # [1, N, head_dim]

    k_layers, v_layers = [], []
    for layer in hf.layers:
        attn = layer.self_attn
        k = attn.k_proj(ctx).view(1, n, nkv, hd)
        k = attn.k_norm(k).transpose(
            1, 2
        )  # [1, nkv, N, hd]  (k_norm over head_dim, then [B,nkv,S,hd])
        _, k = apply_rotary_pos_emb(k, k, cos, sin)  # K branch; unsqueeze_dim=1
        v = attn.v_proj(ctx).view(1, n, nkv, hd).transpose(1, 2)  # [1, nkv, N, hd]
        # AD returns [N, nkv, hd] per layer -> match by dropping batch and moving seq to front.
        k_layers.append(k.squeeze(0).transpose(0, 1).contiguous())  # [N, nkv, hd]
        v_layers.append(v.squeeze(0).transpose(0, 1).contiguous())
    return torch.stack(k_layers, dim=1), torch.stack(v_layers, dim=1)  # [N, L, nkv, hd]


# Tolerances are RMSE-ratio (rmse(actual-expected)/rmse(expected)) via ``assert_rmse_close`` -- the
# AD modeling-test house convention for bf16 equivalence. RMSE is robust to the handful of near-zero-K
# RoPE entries whose per-element relative error blows up (a per-element atol loose enough to pass them
# would also mask a systematic shift), while still failing hard on any systematic divergence.
# Observed: fp32 V & K == 0 (exact); bf16 V == 0 (bit-exact: V's path is deterministic GEMMs + an
# fp32-reduction RMSNorm that rounds identically on both sides), bf16 K ~= 1.6e-3. The bf16 K tol of
# 1e-2 is a ~6x margin (far tighter than the helper's recommended 0.10 for attention+RoPE); the bf16 V
# tol of 1e-4 keeps the V path a tight isolation guard while tolerating any cross-GPU RMSNorm rounding.
@pytest.mark.parametrize(
    "dtype,v_tol,k_tol",
    [
        pytest.param(torch.float32, 1e-5, 1e-5, id="fp32-tight"),
        pytest.param(torch.bfloat16, 1e-4, 1e-2, id="bf16-production"),
    ],
)
@torch.inference_mode()
def test_precompute_context_kv_matches_hf(dtype, v_tol, k_tol):
    """AD ``precompute_context_kv`` matches the HF vanilla DFlash context K/V path.

    Same checkpoint loaded into both; identical raw hidden states + positions fed in. V is compared
    first (no norm/RoPE -> isolates the projection/load), then K (norm + RoPE).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    _, draft_path = _paths()
    device = "cuda"

    factory = _make_factory()
    ad = _build_ad_draft(factory, dtype, device)
    hf = _build_hf_ref(draft_path, dtype, device)

    cfg = ad.config
    n = 24
    fc_in = cfg.hidden_size * ad.num_capture_layers
    # Deterministic inputs (seeded); raw target hidden states + absolute context positions.
    gen = torch.Generator(device=device).manual_seed(0)
    raw = torch.randn(n, fc_in, device=device, dtype=dtype, generator=gen)
    positions = torch.arange(n, device=device, dtype=torch.long)

    k_ad, v_ad = ad.precompute_context_kv(raw, positions)
    k_ref, v_ref = _hf_ref_context_kv(hf, raw, positions)

    assert k_ad.shape == k_ref.shape, f"K shape {k_ad.shape} != {k_ref.shape}"
    assert v_ad.shape == v_ref.shape, f"V shape {v_ad.shape} != {v_ref.shape}"

    # V first: no norm/RoPE, so a mismatch here points at fc/hidden_norm/v_proj or weight loading.
    assert_rmse_close(v_ad, v_ref, v_tol, msg="context V mismatch (AD vs HF): ")
    # K: adds k_norm + RoPE -- the prime suspects for any small numerical delta.
    assert_rmse_close(k_ad, k_ref, k_tol, msg="context K mismatch (AD vs HF): ")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
