# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spike A: flash_attn_with_kvcache(causal=False) contract probe (standalone, no AutoDeploy).

De-risks the DFlash cached-attention contract the AD op will wrap. Confirms, on the actually-installed
`flash_attn` package (the same one the PyTorch DFlash reference imports), that:

  (i)   the query block attends over [ctx[:ctx_len]  ||  query-block] and NOTHING past ctx_len+block
        (garbage rows in the slack beyond the appended block must not affect the output);
  (ii)  the kernel APPENDS the current query-block K/V in place at row cache_seqlens (== ctx_len);
  (iii) the output matches a hand-rolled non-causal SDPA over [ctx[:ctx_len] || query-block],
        per-request, with GQA and cache_batch_idx indirection.

This is a throwaway probe (not a committed test). The real cached-op unit test (Step 1 / rung 2) is
modeled on tests/.../custom_ops/attention/test_torch_attention_op.py.
"""

import inspect

import torch
from flash_attn import flash_attn_with_kvcache

torch.manual_seed(0)
DEV = "cuda"
DT = torch.float16

# ---- shapes -------------------------------------------------------------------------------------
B = 2  # requests
POOL = 8  # cache pool slots (cache_batch_idx selects into this)
NHEADS = 4
NKV = 2  # GQA (NHEADS // NKV = 2 groups)
HD = 64
BLOCK = 5  # query block = [last, MASK, MASK, MASK, MASK] = max_draft_len(4) + 1
MAX_CTX = 16  # persistent-context capacity
MAX_SEQ = MAX_CTX + BLOCK  # slack-sized seq dim (where flash appends the query block)

slot_idx = torch.tensor([3, 1], device=DEV, dtype=torch.int32)  # non-trivial pool indices
ctx_len = torch.tensor([7, 4], device=DEV, dtype=torch.int32)  # per-request accepted-context length
scale = 1.0 / (HD**0.5)

# ---- build caches: real context in [0:ctx_len], GARBAGE everywhere else (slack incl. append region)
k_cache = torch.full((POOL, MAX_SEQ, NKV, HD), 1e4, device=DEV, dtype=DT)
v_cache = torch.full((POOL, MAX_SEQ, NKV, HD), 1e4, device=DEV, dtype=DT)
ctx_k_ref, ctx_v_ref = {}, {}
for b in range(B):
    s = slot_idx[b].item()
    n = ctx_len[b].item()
    kk = torch.randn(n, NKV, HD, device=DEV, dtype=DT)
    vv = torch.randn(n, NKV, HD, device=DEV, dtype=DT)
    k_cache[s, :n] = kk
    v_cache[s, :n] = vv
    ctx_k_ref[b], ctx_v_ref[b] = kk, vv  # remember the real context for the reference

# snapshot the append region BEFORE the call to prove (ii) it gets overwritten by the query block
pre_append_k = k_cache[slot_idx.long(), :, :, :].clone()  # [B, MAX_SEQ, NKV, HD]

# ---- query block + its K/V to append ------------------------------------------------------------
q = torch.randn(B, BLOCK, NHEADS, HD, device=DEV, dtype=DT)
qk = torch.randn(B, BLOCK, NKV, HD, device=DEV, dtype=DT)
qv = torch.randn(B, BLOCK, NKV, HD, device=DEV, dtype=DT)

print("flash_attn_with_kvcache signature:", inspect.signature(flash_attn_with_kvcache))

# ---- the call under test ------------------------------------------------------------------------
out = flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=qk,
    v=qv,
    cache_seqlens=ctx_len,
    cache_batch_idx=slot_idx,
    softmax_scale=scale,
    causal=False,
)  # -> [B, BLOCK, NHEADS, HD]
print("out shape:", tuple(out.shape))


def repeat_kv(x, n_rep):  # [s, nkv, hd] -> [s, nkv*n_rep, hd]
    s, nkv, hd = x.shape
    return x[:, :, None, :].expand(s, nkv, n_rep, hd).reshape(s, nkv * n_rep, hd)


# ---- (iii) hand-rolled non-causal SDPA reference over [ctx || query-block] -----------------------
n_rep = NHEADS // NKV
ok_iii = True
for b in range(B):
    n = ctx_len[b].item()
    k_full = torch.cat([ctx_k_ref[b], qk[b]], dim=0)  # [n+BLOCK, NKV, HD]
    v_full = torch.cat([ctx_v_ref[b], qv[b]], dim=0)
    k_full = repeat_kv(k_full, n_rep).transpose(0, 1)  # [NHEADS, n+BLOCK, HD]
    v_full = repeat_kv(v_full, n_rep).transpose(0, 1)
    qb = q[b].transpose(0, 1)  # [NHEADS, BLOCK, HD]
    scores = (
        torch.matmul(qb.float(), k_full.float().transpose(-1, -2)) * scale
    )  # non-causal: no mask
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, v_full.float()).transpose(0, 1).to(DT)  # [BLOCK, NHEADS, HD]
    try:
        torch.testing.assert_close(out[b], ref, atol=2e-2, rtol=2e-2)
        print(f"  [iii] request {b}: SDPA-over-[ctx||query] parity  PASS")
    except AssertionError as e:
        ok_iii = False
        print(f"  [iii] request {b}: FAIL\n{e}")

# ---- (ii) append happened at cache_seqlens (== ctx_len), overwriting the prior garbage -----------
ok_ii = True
for b in range(B):
    s = slot_idx[b].item()
    n = ctx_len[b].item()
    appended = k_cache[s, n : n + BLOCK]
    try:
        torch.testing.assert_close(appended, qk[b], atol=0, rtol=0)
        # and prove it really changed from the pre-call garbage
        changed = not torch.equal(appended, pre_append_k[b, n : n + BLOCK])
        assert changed, "append region unchanged"
        print(f"  [ii]  request {b}: query-block K appended in place at row ctx_len={n}  PASS")
    except AssertionError as e:
        ok_ii = False
        print(f"  [ii]  request {b}: FAIL\n{e}")

# ---- (i) garbage beyond ctx_len+block was ignored ------------------------------------------------
# The reference in (iii) only used [0:ctx_len] + query block; rows in [ctx_len+block:] were 1e4
# garbage and untouched. Passing (iii) with that garbage present == the kernel ignored it.
ok_i = ok_iii
print(
    f"  [i]   garbage rows beyond ctx_len+block ignored (implied by (iii) parity): "
    f"{'PASS' if ok_i else 'FAIL'}"
)

print("\nSPIKE A RESULT:", "PASS" if (ok_i and ok_ii and ok_iii) else "FAIL")
