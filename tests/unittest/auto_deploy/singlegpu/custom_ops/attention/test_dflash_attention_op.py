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
"""Unit tests for the DFlash draft-attention ops (Step 1).

Two ops, two contracts:

- ``auto_deploy::dflash_attention`` (source/export op): its eager body MUST equal canonical
  non-causal SDPA via ``auto_deploy::torch_attention(is_causal=False, layout="bsnd")``, and the
  carried ``ctx_len`` argument MUST NOT affect the math (it exists only so the graph input survives
  export and reaches the cached op).

- ``auto_deploy::dflash_attention_with_kvcache`` (cached/runtime op): wraps
  ``flash_attn_with_kvcache(causal=False)``. The query block attends non-causally over
  ``[ctx_k_cache[slot, :ctx_len] || query-block]`` per request (``cache_batch_idx=slot_idx``,
  ``cache_seqlens=ctx_len``), and the current query-block K/V are APPENDED in place into the cache
  slack at row ``ctx_len`` (so the ctx caches are declared mutated). Modeled on
  ``test_torch_attention_op.py`` and the validated ``debug/spikes/spike_a_flash_attn_contract.py``.

Parity is checked two ways: against the canonical ``torch_attention`` op (ties the cached op to the
same math as the source op) AND against an INDEPENDENT hand-rolled SDPA (so a bug shared by
``dflash_attention`` and ``torch_attention`` cannot hide).
"""

import math

import pytest
import torch
from torch._subclasses import FakeTensorMode

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers auto_deploy::* ops)

# The cached op hard-depends on the classic flash_attn package; skip the whole module if absent.
pytest.importorskip("flash_attn")

DEVICE = "cuda"
ATOL = 2e-2
RTOL = 2e-2

# Default cached-op scenario shapes. The production drafter block_size is 16; tests use a small
# block for speed and cover the production width via parametrization.
BLOCK = 5
HEAD_DIM = 64
MAX_CTX = 16
POOL = 8


@pytest.fixture(autouse=True)
def _seed_and_clean():
    """Deterministic, isolated state per test (mirrors test_torch_attention_op.py's setup)."""
    torch.cuda.empty_cache()
    torch.manual_seed(0)
    yield


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[L, n_kv, hd] -> [L, n_kv*n_rep, hd] (GQA expansion)."""
    L, n_kv, hd = x.shape
    return x[:, :, None, :].expand(L, n_kv, n_rep, hd).reshape(L, n_kv * n_rep, hd)


def _handrolled_noncausal_sdpa(
    q_b: torch.Tensor, k_full: torch.Tensor, v_full: torch.Tensor, scale: float
) -> torch.Tensor:
    """Independent non-causal SDPA for ONE request (no torch_attention, no flash).

    Mirrors the reference in ``spike_a_flash_attn_contract.py``. ``q_b`` is ``[block, n_heads, hd]``;
    ``k_full``/``v_full`` are ``[L, n_kv, hd]``. Returns ``[block, n_heads, hd]``.
    """
    n_rep = q_b.shape[1] // k_full.shape[1]
    k = _repeat_kv(k_full, n_rep).transpose(0, 1)  # [n_heads, L, hd]
    v = _repeat_kv(v_full, n_rep).transpose(0, 1)
    qb = q_b.transpose(0, 1)  # [n_heads, block, hd]
    scores = torch.matmul(qb.float(), k.float().transpose(-1, -2)) * scale  # non-causal: no mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v.float()).transpose(0, 1).to(q_b.dtype)  # [block, n_heads, hd]


# ============================================================================ #
#  Source op: auto_deploy::dflash_attention                                    #
# ============================================================================ #
class TestDFlashSourceOp:
    """The source op is honest non-causal SDPA; ``ctx_len`` is inert."""

    def _qkv(self, batch, block, n_heads, n_kv, head_dim, dtype=torch.float32):
        q = torch.randn(batch, block, n_heads, head_dim, device=DEVICE, dtype=dtype)
        k = torch.randn(batch, block, n_kv, head_dim, device=DEVICE, dtype=dtype)
        v = torch.randn(batch, block, n_kv, head_dim, device=DEVICE, dtype=dtype)
        return q, k, v

    @pytest.mark.parametrize("n_heads,n_kv", [(4, 4), (8, 2)])  # MHA and GQA
    @pytest.mark.parametrize("scale", [None, 0.3])
    def test_equals_torch_attention_non_causal(self, n_heads, n_kv, scale):
        """Source-op math == torch_attention(is_causal=False, layout='bsnd') exactly."""
        q, k, v = self._qkv(2, 5, n_heads, n_kv, 16)
        ctx_len = torch.tensor([7, 4], device=DEVICE, dtype=torch.int32)

        actual = torch.ops.auto_deploy.dflash_attention(q, k, v, ctx_len, scale)
        expected = torch.ops.auto_deploy.torch_attention(
            q, k, v, is_causal=False, scale=scale, layout="bsnd"
        )
        # Same underlying computation => bit-exact.
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_ctx_len_is_inert(self):
        """Different / absent ctx_len must not change the source-op output."""
        q, k, v = self._qkv(2, 5, 8, 2, 16)
        out_none = torch.ops.auto_deploy.dflash_attention(q, k, v, None, None)
        out_a = torch.ops.auto_deploy.dflash_attention(
            q, k, v, torch.tensor([7, 4], device=DEVICE, dtype=torch.int32), None
        )
        out_b = torch.ops.auto_deploy.dflash_attention(
            q, k, v, torch.tensor([1, 0], device=DEVICE, dtype=torch.int32), None
        )
        torch.testing.assert_close(out_none, out_a, atol=0.0, rtol=0.0)
        torch.testing.assert_close(out_none, out_b, atol=0.0, rtol=0.0)

    def test_is_non_causal(self):
        """Sanity: the source op is genuinely non-causal (differs from causal SDPA)."""
        q, k, v = self._qkv(1, 5, 4, 4, 16)
        non_causal = torch.ops.auto_deploy.dflash_attention(q, k, v, None, None)
        causal = torch.ops.auto_deploy.torch_attention(q, k, v, is_causal=True, layout="bsnd")
        assert not torch.allclose(non_causal, causal, atol=ATOL, rtol=RTOL)

    def test_output_shape_and_fake(self):
        """Output shape is [B, block, n_heads, v_head_dim]; register_fake yields the same shape."""
        q, k, v = self._qkv(3, 6, 8, 2, 16)
        out = torch.ops.auto_deploy.dflash_attention(q, k, v, None, None)
        assert out.shape == (3, 6, 8, 16)

        with FakeTensorMode():
            fq = torch.empty(3, 6, 8, 16, device=DEVICE, dtype=torch.float16)
            fk = torch.empty(3, 6, 2, 16, device=DEVICE, dtype=torch.float16)
            fv = torch.empty(3, 6, 2, 16, device=DEVICE, dtype=torch.float16)
            fout = torch.ops.auto_deploy.dflash_attention(fq, fk, fv, None, None)
            assert fout.shape == (3, 6, 8, 16)


# ============================================================================ #
#  Cached op: auto_deploy::dflash_attention_with_kvcache                       #
# ============================================================================ #
class TestDFlashCachedOp:
    """flash_attn-backed cached op: non-causal SDPA over [ctx || query-block] + in-place append."""

    def _build(
        self, slot_idx, ctx_len, n_heads, n_kv, block=BLOCK, head_dim=HEAD_DIM, dtype=torch.float16
    ):
        """Build q + query-block K/V + per-slot ctx caches.

        Caches are filled with large GARBAGE everywhere; real context written only in [0:ctx_len].
        Parity then implicitly requires that garbage (incl. rows past ctx_len+block) is ignored.
        """
        batch = len(slot_idx)
        max_seq = MAX_CTX + block
        ctx_k = torch.full((POOL, max_seq, n_kv, head_dim), 1e4, device=DEVICE, dtype=dtype)
        ctx_v = torch.full((POOL, max_seq, n_kv, head_dim), 1e4, device=DEVICE, dtype=dtype)
        for b in range(batch):
            s, n = int(slot_idx[b]), int(ctx_len[b])
            if n > 0:
                ctx_k[s, :n] = torch.randn(n, n_kv, head_dim, device=DEVICE, dtype=dtype)
                ctx_v[s, :n] = torch.randn(n, n_kv, head_dim, device=DEVICE, dtype=dtype)
        q = torch.randn(batch, block, n_heads, head_dim, device=DEVICE, dtype=dtype)
        qk = torch.randn(batch, block, n_kv, head_dim, device=DEVICE, dtype=dtype)
        qv = torch.randn(batch, block, n_kv, head_dim, device=DEVICE, dtype=dtype)
        slot = torch.tensor(slot_idx, device=DEVICE, dtype=torch.int32)
        ctxlen = torch.tensor(ctx_len, device=DEVICE, dtype=torch.int32)
        return q, qk, qv, slot, ctxlen, ctx_k, ctx_v

    def _reference(self, q, qk, qv, ctx_k, ctx_v, slot, ctxlen, scale, handrolled=False):
        """Per-request non-causal SDPA over [ctx_cache[slot, :ctx_len] || query-block].

        ``handrolled=False`` uses the canonical ``torch_attention`` op (ties the cached op to the
        same math as the source op). ``handrolled=True`` uses an INDEPENDENT implementation so a bug
        common to dflash/torch_attention cannot hide.
        """
        s_scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        outs = []
        for b in range(q.shape[0]):
            s, n = int(slot[b]), int(ctxlen[b])
            k_full = torch.cat([ctx_k[s, :n], qk[b]], dim=0)  # [n+block, n_kv, hd]
            v_full = torch.cat([ctx_v[s, :n], qv[b]], dim=0)
            if handrolled:
                outs.append(_handrolled_noncausal_sdpa(q[b], k_full, v_full, s_scale))
            else:
                ref = torch.ops.auto_deploy.torch_attention(
                    q[b : b + 1],
                    k_full.unsqueeze(0),
                    v_full.unsqueeze(0),
                    is_causal=False,
                    scale=scale,
                    layout="bsnd",
                )
                outs.append(ref[0])
        return torch.stack(outs, dim=0)

    # ---- parity --------------------------------------------------------------------------------- #
    @pytest.mark.parametrize("n_heads,n_kv", [(4, 4), (4, 2), (8, 1)])  # MHA + GQA ratios
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sdpa_parity_over_ctx_and_block(self, n_heads, n_kv, dtype):
        """Output == non-causal SDPA over [ctx || query-block], per request, with slot indirection."""
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build(
            [3, 1], [7, 4], n_heads, n_kv, dtype=dtype
        )
        ref = self._reference(q, qk, qv, ctx_k.clone(), ctx_v.clone(), slot, ctxlen, scale)

        out = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale
        )
        assert out.shape == (2, BLOCK, n_heads, HEAD_DIM)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_parity_independent_reference(self):
        """Cross-check against a hand-rolled SDPA (independent of torch_attention)."""
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build([3, 1], [7, 4], 8, 2)
        ref = self._reference(
            q, qk, qv, ctx_k.clone(), ctx_v.clone(), slot, ctxlen, scale, handrolled=True
        )
        out = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale
        )
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize(
        "slot_idx,ctx_len",
        [
            ([2], [5]),  # batch=1 (the common decode case)
            ([2], [0]),  # batch=1, empty persistent context (first draft step)
            ([3, 1], [0, 4]),  # mixed: one empty context, one populated
            ([3, 1], [MAX_CTX, MAX_CTX]),  # append lands exactly in the final slack rows
            ([2, 5], [6, 6]),  # uniform ctx_len across requests
        ],
    )
    def test_parity_edge_scenarios(self, slot_idx, ctx_len):
        """Parity across batch=1, ctx_len=0, max-slack, and uniform-ctx_len edge cases."""
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build(slot_idx, ctx_len, 8, 2)
        ref = self._reference(q, qk, qv, ctx_k.clone(), ctx_v.clone(), slot, ctxlen, scale)
        out = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale
        )
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_production_block_size(self):
        """Parity at the production drafter block width (block_size=16)."""
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build([3, 1], [7, 4], 8, 2, block=16)
        ref = self._reference(q, qk, qv, ctx_k.clone(), ctx_v.clone(), slot, ctxlen, scale)
        out = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale
        )
        assert out.shape == (2, 16, 8, HEAD_DIM)
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    def test_default_scale_matches_reference(self):
        """scale=None uses 1/sqrt(head_dim) for both the op and the reference."""
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build([3, 1], [7, 4], 8, 2)
        ref = self._reference(q, qk, qv, ctx_k.clone(), ctx_v.clone(), slot, ctxlen, None)
        out = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, None
        )
        torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)

    # ---- mutation contract ---------------------------------------------------------------------- #
    def test_query_block_appended_in_place(self):
        """The current query-block K/V are written into the cache slack at row ctx_len."""
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build([3, 1], [7, 4], 4, 2)
        pre = ctx_k[slot.long()].clone()

        torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale
        )

        for b in range(2):
            s, n = int(slot[b]), int(ctxlen[b])
            torch.testing.assert_close(ctx_k[s, n : n + BLOCK], qk[b], atol=0.0, rtol=0.0)
            torch.testing.assert_close(ctx_v[s, n : n + BLOCK], qv[b], atol=0.0, rtol=0.0)
            # And it really changed from the pre-call garbage.
            assert not torch.equal(ctx_k[s, n : n + BLOCK], pre[b, n : n + BLOCK])

    def test_persistent_context_and_tail_preserved(self):
        """The persistent [0:ctx_len] context AND the slack tail past ctx_len+block stay untouched."""
        scale = 1.0 / math.sqrt(HEAD_DIM)
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build([3, 1], [7, 4], 4, 2)
        pre_k, pre_v = ctx_k.clone(), ctx_v.clone()

        torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale
        )

        for b in range(2):
            s, n = int(slot[b]), int(ctxlen[b])
            # persistent context untouched
            torch.testing.assert_close(ctx_k[s, :n], pre_k[s, :n], atol=0.0, rtol=0.0)
            torch.testing.assert_close(ctx_v[s, :n], pre_v[s, :n], atol=0.0, rtol=0.0)
            # slack tail beyond the appended block untouched (op must not write past ctx_len+block)
            tail = slice(n + BLOCK, None)
            torch.testing.assert_close(ctx_k[s, tail], pre_k[s, tail], atol=0.0, rtol=0.0)
            torch.testing.assert_close(ctx_v[s, tail], pre_v[s, tail], atol=0.0, rtol=0.0)

    # ---- out= CUDA-graph buffer ----------------------------------------------------------------- #
    def test_out_buffer_path(self):
        """out= path writes the result into out and returns an empty tensor.

        Verifies the AD CUDA-graph convention: the op returns an EMPTY dummy (the real output is in
        out_buf), and the in-place query-block append STILL happens on the out= branch.
        """
        scale = 1.0 / math.sqrt(HEAD_DIM)
        # Reference: same call WITHOUT out=, on independent cache copies.
        q, qk, qv, slot, ctxlen, ctx_k, ctx_v = self._build([3, 1], [7, 4], 4, 2)
        expected = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k.clone(), ctx_v.clone(), scale
        )

        out_buf = torch.empty(2, BLOCK, 4, HEAD_DIM, device=DEVICE, dtype=torch.float16)
        returned = torch.ops.auto_deploy.dflash_attention_with_kvcache(
            q, qk, qv, slot, ctxlen, ctx_k, ctx_v, scale, out_buf
        )
        assert returned.numel() == 0  # op returns an empty dummy; real output is in out_buf
        torch.testing.assert_close(out_buf, expected, atol=ATOL, rtol=RTOL)
        # The append is an independent code path from the out.copy_ — verify it still ran.
        for b in range(2):
            s, n = int(slot[b]), int(ctxlen[b])
            torch.testing.assert_close(ctx_k[s, n : n + BLOCK], qk[b], atol=0.0, rtol=0.0)

    def test_fake_output_shape(self):
        """register_fake gives [B, block, n_heads, v_head_dim] under FakeTensorMode."""
        n_heads, n_kv, block = 4, 2, BLOCK
        max_seq = MAX_CTX + block
        with FakeTensorMode():
            q = torch.empty(2, block, n_heads, HEAD_DIM, device=DEVICE, dtype=torch.float16)
            qk = torch.empty(2, block, n_kv, HEAD_DIM, device=DEVICE, dtype=torch.float16)
            qv = torch.empty(2, block, n_kv, HEAD_DIM, device=DEVICE, dtype=torch.float16)
            slot = torch.empty(2, device=DEVICE, dtype=torch.int32)
            ctxlen = torch.empty(2, device=DEVICE, dtype=torch.int32)
            ctx_k = torch.empty(POOL, max_seq, n_kv, HEAD_DIM, device=DEVICE, dtype=torch.float16)
            ctx_v = torch.empty(POOL, max_seq, n_kv, HEAD_DIM, device=DEVICE, dtype=torch.float16)
            out = torch.ops.auto_deploy.dflash_attention_with_kvcache(
                q, qk, qv, slot, ctxlen, ctx_k, ctx_v, None
            )
            assert out.shape == (2, block, n_heads, HEAD_DIM)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-sv"]))
