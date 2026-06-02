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
"""Structural test for the ``insert_cached_dflash_attention`` transform (Step 2).

Exports a toy draft whose ``forward(q, k, v, ctx_len)`` emits the distinct source op
``auto_deploy::dflash_attention``, runs the transform, and asserts the graph REWRITE:

  - the source ``dflash_attention`` node is replaced by ``dflash_attention_with_kvcache``;
  - args are ``(q, k, v, slot_idx, ctx_len, ctx_k_cache, ctx_v_cache, scale)`` -- i.e.
    ``(*qkv, *standard_metadata, *caches, *constants)``;
  - ``slot_idx`` is ADDED as a new graph input (no placeholder existed -> activate_arg);
  - ``ctx_len`` is RETRIEVED (the *same* placeholder the toy forward declared, carried by the
    source op -- see ``debug/spikes/spike_b_ctx_len_export.py``);
  - two dense ctx K/V resources are registered with the slack-sized ``+block_size`` seq dim.

This is a STRUCTURAL gate: it does not execute the rewritten graph. End-to-end execution needs the
wrapper/factory ``ctx_len`` + cache-allocation plumbing (Steps 7-8); the op's runtime math is covered
by ``tests/.../custom_ops/attention/test_dflash_attention_op.py``.

TODO (upgrade once Step 3 ``modeling_dflash.py`` lands): replace the toy module with a small instance
of the real prefill-version DFlash draft model and run the transform over its actual
``dflash_attention`` sites (multiple draft layers, real ``ctx_len`` wiring). That exercises the genuine
exported graph shape rather than a hand-built single-op stub. Keep this toy test as the fast unit-level
regression guard for the rewrite mechanics.
"""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers ops + the DFlash descriptor)
from tensorrt_llm._torch.auto_deploy._compat import KvCacheConfig
from tensorrt_llm._torch.auto_deploy.shim.interface import CachedSequenceInterface
from tensorrt_llm._torch.auto_deploy.transform.interface import Stages
from tensorrt_llm._torch.auto_deploy.transform.library.kvcache import (
    InsertCachedAttentionConfig,
    InsertCachedDFlashAttention,
)
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

pytest.importorskip("flash_attn")

B, BLOCK, N_HEADS, N_KV, HEAD_DIM = 2, 5, 4, 2, 16


class ToyDFlashDraft(nn.Module):
    """1-layer toy draft: ctx_len is a declared forward input carried into the source op."""

    def forward(self, q, k, v, ctx_len):
        return torch.ops.auto_deploy.dflash_attention(q, k, v, ctx_len, None)


def _export_toy_gm():
    m = ToyDFlashDraft().eval().cuda()
    q = torch.randn(B, BLOCK, N_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    k = torch.randn(B, BLOCK, N_KV, HEAD_DIM, device="cuda", dtype=torch.float16)
    v = torch.randn(B, BLOCK, N_KV, HEAD_DIM, device="cuda", dtype=torch.float16)
    ctx_len = torch.tensor([7, 4], device="cuda", dtype=torch.int32)
    return torch.export.export(m, (q, k, v, ctx_len)).module()


def _make_cm():
    kv_cache_config = KvCacheConfig(
        tokens_per_block=32, max_tokens=256, free_gpu_memory_fraction=0.0
    )
    return CachedSequenceInterface(
        max_seq_len=128,
        max_batch_size=B,
        max_num_tokens=B * 128,
        device="cuda",
        kv_cache_config=kv_cache_config,
    )


def _placeholder_names(gm):
    return [n.target for n in gm.graph.nodes if n.op == "placeholder"]


@torch.inference_mode()
def test_insert_cached_dflash_attention_rewrite():
    gm = _export_toy_gm()
    cm = _make_cm()

    # The source op is present; the cached op is not, pre-transform.
    src_op = torch.ops.auto_deploy.dflash_attention
    cached_op = torch.ops.auto_deploy.dflash_attention_with_kvcache
    assert sum(is_op(n, src_op) for n in gm.graph.nodes) == 1
    assert sum(is_op(n, cached_op) for n in gm.graph.nodes) == 0

    ctx_len_ph_before = gm.graph.find_nodes(op="placeholder", target="ctx_len")
    assert len(ctx_len_ph_before) == 1
    assert "slot_idx" not in _placeholder_names(gm)  # not declared by the toy forward

    transform = InsertCachedDFlashAttention(
        config=InsertCachedAttentionConfig(stage=Stages.CACHE_INIT, backend="dflash")
    )
    gm_out, info = transform._apply(gm, cm, MagicMock(), MagicMock())

    # --- rewrite happened ---
    assert not info.skipped and info.num_matches == 1
    assert sum(is_op(n, src_op) for n in gm_out.graph.nodes) == 0
    cached_nodes = [n for n in gm_out.graph.nodes if is_op(n, cached_op)]
    assert len(cached_nodes) == 1
    cached = cached_nodes[0]

    # --- arg wiring: (q, k, v, slot_idx, ctx_len, ctx_k_cache, ctx_v_cache, scale) ---
    args = cached.args
    assert len(args) == 8, f"expected 8 positional args, got {len(args)}: {args}"
    q_n, k_n, v_n, slot_n, ctxlen_n, ck_n, cv_n, scale_c = args
    assert q_n.op == "placeholder" and q_n.target == "q"
    assert k_n.op == "placeholder" and k_n.target == "k"
    assert v_n.op == "placeholder" and v_n.target == "v"

    # slot_idx was ADDED as a new graph input.
    assert slot_n.op == "placeholder" and slot_n.target == "slot_idx"
    assert "slot_idx" in _placeholder_names(gm_out)

    # ctx_len was RETRIEVED: the exact same placeholder node the forward declared.
    assert ctxlen_n is ctx_len_ph_before[0]
    assert ctxlen_n.op == "placeholder" and ctxlen_n.target == "ctx_len"

    # two dense ctx K/V caches were inserted as graph inputs.
    assert ck_n.op == "placeholder" and cv_n.op == "placeholder"
    assert ck_n is not cv_n
    assert scale_c is None  # the toy used scale=None

    # --- the descriptor builds two dense ctx K/V handlers, slack-sized on the seq dim ---
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention.dflash_attention import (
        DFlashAttention,
        DFlashCtxKVResourceHandler,
    )

    # Re-export a fresh gm to get a clean (pre-rewrite) source node for the initializer check.
    gm2 = _export_toy_gm()
    src2 = next(n for n in gm2.graph.nodes if is_op(n, src_op))
    inits = DFlashAttention.get_cache_initializers(src2, cm.kv_cache_config)
    assert set(inits) == {"ctx_k_cache", "ctx_v_cache"}
    for h in inits.values():
        assert isinstance(h, DFlashCtxKVResourceHandler)
        assert h.block_size == BLOCK and h.num_kv_heads == N_KV and h.head_dim == HEAD_DIM
        allocated = h.allocate(cm.info)
        assert allocated.shape == (
            cm.info.max_num_state_slots,
            cm.info.max_seq_len + BLOCK,
            N_KV,
            HEAD_DIM,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
