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
"""Tests for the AutoDeploy DFlash draft model (modeling_dflash, Step 3).

Covers:
  - eager query-block forward (shape + finite);
  - torch.export emits one ``auto_deploy::dflash_attention`` per draft layer with ``ctx_len`` carried;
  - the REAL-model Step-2 gate: export the actual draft model and run
    ``insert_cached_dflash_attention`` over its real attention sites (the model-based upgrade of the
    toy test in ``transformations/library/test_dflash_cache.py``).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers ops + the DFlash descriptor)
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_dflash import DFlashModel
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
    InsertCachedAttentionConfig,
    InsertCachedDFlashAttention,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

DEVICE = "cuda"
DTYPE = torch.float16
B, BLOCK = 2, 5
N_LAYERS = 3


def _tiny_config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=N_LAYERS,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        rope_theta=1.0e6,
        max_position_embeddings=64,
        attention_bias=False,
        hidden_act="silu",
        dflash_config={"target_layer_ids": [1, 9, 17, 25, 33]},
        block_size=16,
    )


def _make_model():
    torch.manual_seed(0)
    return DFlashModel(_tiny_config()).to(device=DEVICE, dtype=DTYPE).eval()


def _example_inputs():
    cfg = _tiny_config()
    inputs_embeds = torch.randn(B, BLOCK, cfg.hidden_size, device=DEVICE, dtype=DTYPE)
    position_ids = torch.arange(BLOCK, device=DEVICE).unsqueeze(0).expand(B, BLOCK).contiguous()
    ctx_len = torch.tensor([7, 4], device=DEVICE, dtype=torch.int32)
    return inputs_embeds, position_ids, ctx_len


@torch.inference_mode()
def test_eager_forward_shape():
    """Eager query-block forward returns [B, block, hidden] and is finite."""
    model = _make_model()
    inputs_embeds, position_ids, ctx_len = _example_inputs()
    out = model(inputs_embeds, position_ids, ctx_len)
    assert out.shape == (B, BLOCK, _tiny_config().hidden_size)
    assert torch.isfinite(out).all()


@torch.inference_mode()
def test_export_emits_dflash_attention_per_layer():
    """torch.export keeps one dflash_attention op per draft layer, each carrying the ctx_len placeholder."""
    model = _make_model()
    gm = torch.export.export(model, _example_inputs()).module()

    src_op = torch.ops.auto_deploy.dflash_attention
    dflash_nodes = [n for n in gm.graph.nodes if is_op(n, src_op)]
    assert len(dflash_nodes) == N_LAYERS

    # ctx_len is a single declared placeholder carried (as arg[3]) into every dflash_attention node.
    ctx_phs = gm.graph.find_nodes(op="placeholder", target="ctx_len")
    assert len(ctx_phs) == 1
    for n in dflash_nodes:
        assert n.args[3] is ctx_phs[0]


@torch.inference_mode()
def test_transform_over_real_model():
    """Model-based Step-2 gate: run insert_cached_dflash_attention over the REAL exported draft model.

    Every dflash_attention site (one per layer) must lower to dflash_attention_with_kvcache; slot_idx
    is added once and shared, ctx_len is the single retrieved placeholder, and each layer gets its own
    pair of dense ctx K/V caches.
    """
    pytest.importorskip("flash_attn")
    model = _make_model()
    gm = torch.export.export(model, _example_inputs()).module()

    kv_cache_config = KvCacheConfig(
        tokens_per_block=32, max_tokens=256, free_gpu_memory_fraction=0.0
    )
    cm = CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=B,
        max_num_tokens=B * 128,
        device=DEVICE,
        kv_cache_config=kv_cache_config,
    )

    src_op = torch.ops.auto_deploy.dflash_attention
    cached_op = torch.ops.auto_deploy.dflash_attention_with_kvcache
    assert sum(is_op(n, src_op) for n in gm.graph.nodes) == N_LAYERS

    transform = InsertCachedDFlashAttention(
        config=InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="dflash")
    )
    gm_out, info = transform._apply(gm, cm, MagicMock(), MagicMock())

    assert not info.skipped and info.num_matches == N_LAYERS
    assert sum(is_op(n, src_op) for n in gm_out.graph.nodes) == 0
    cached_nodes = [n for n in gm_out.graph.nodes if is_op(n, cached_op)]
    assert len(cached_nodes) == N_LAYERS

    # slot_idx added once and shared; ctx_len the single retrieved placeholder.
    slot_phs = gm_out.graph.find_nodes(op="placeholder", target="slot_idx")
    ctx_phs = gm_out.graph.find_nodes(op="placeholder", target="ctx_len")
    assert len(slot_phs) == 1 and len(ctx_phs) == 1
    for n in cached_nodes:
        # args: (q, k, v, slot_idx, ctx_len, ctx_k_cache, ctx_v_cache, scale)
        assert n.args[3] is slot_phs[0]
        assert n.args[4] is ctx_phs[0]
        assert n.args[5].op == "placeholder" and n.args[6].op == "placeholder"

    # each layer owns a distinct pair of ctx K/V cache inputs (2 per layer).
    cache_inputs = {n.args[5] for n in cached_nodes} | {n.args[6] for n in cached_nodes}
    assert len(cache_inputs) == 2 * N_LAYERS


@torch.inference_mode()
def test_precompute_context_kv():
    """precompute_context_kv: shape + the asymmetric context path (V raw, K normed+RoPE'd).

    Validates the oracle contract: context K/V = fc -> hidden_norm -> per-layer k/v_proj, with k_norm
    + RoPE on K only (V is the raw v_proj output, no norm/RoPE). NOT routed through input_layernorm.
    """
    cfg = _tiny_config()
    model = _make_model()
    n_ctx = 6
    num_capture = len(cfg.dflash_config["target_layer_ids"])
    captured = torch.randn(n_ctx, num_capture * cfg.hidden_size, device=DEVICE, dtype=DTYPE)
    positions = torch.arange(n_ctx, device=DEVICE, dtype=torch.long)

    k, v = model.precompute_context_kv(captured, positions)
    assert k.shape == (n_ctx, N_LAYERS, cfg.num_key_value_heads, cfg.head_dim)
    assert v.shape == (n_ctx, N_LAYERS, cfg.num_key_value_heads, cfg.head_dim)
    assert torch.isfinite(k).all() and torch.isfinite(v).all()

    # Independent check of the context projection (raw F-ops, no module reuse for the norm/proj path).
    ctx = model.hidden_norm(model.fc(captured))  # [N, hidden]
    for i, layer in enumerate(model.layers):
        attn = layer.self_attn
        # V is the raw v_proj output (NO k_norm, NO RoPE).
        ref_v = attn.v_proj(ctx).view(n_ctx, cfg.num_key_value_heads, cfg.head_dim)
        torch.testing.assert_close(v[:, i], ref_v, atol=0.0, rtol=0.0)
        # K has k_norm + RoPE applied => it must differ from the raw projection.
        raw_k = attn.k_proj(ctx).view(n_ctx, cfg.num_key_value_heads, cfg.head_dim)
        assert not torch.allclose(k[:, i], raw_k, atol=1e-2, rtol=1e-2)
        # K with k_norm but position 0 (RoPE identity at pos 0) should match k_norm(raw_k) on row 0.
        k_normed_row0 = attn.k_norm(raw_k)[0]
        torch.testing.assert_close(k[0, i], k_normed_row0, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
