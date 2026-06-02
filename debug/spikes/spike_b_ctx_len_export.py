# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spike B: ctx_len-as-graph-input export probe (toy, no full AD pipeline).

De-risks the DFlash design decision that ``ctx_len`` is a **declared input of the draft graph**
(a ``forward`` parameter → export placeholder), **carried by the source op**
``auto_deploy::dflash_attention(q, k, v, ctx_len, scale)`` even though the SDPA math ignores it, and
later **retrieved** (not re-added) by the dedicated cached-attention insertion transform.

The real Step-2 transform retrieves metadata via ``_add_or_retrieve_input`` (interface.py:797):
  find_nodes(op="placeholder", target=name) -> 0 found  => activate_arg + add_graph_input  (slot_idx)
                                            -> 1 found  => return that placeholder           (ctx_len)

So this spike confirms, on a 1-layer toy draft exported with ``torch.export``:
  (i)   ``ctx_len`` survives as a placeholder named "ctx_len";
  (ii)  it is an arg of the ``dflash_attention`` call node (i.e. carried by the source op);
  (iii) the placeholder + op survive dead-code elimination (the carry is what keeps it un-pruned);
  (iv)  replicating ``_add_or_retrieve_input`` retrieves it (exactly 1 placeholder), while a
        SequenceInfo-style arg like ``slot_idx`` is NOT present (0 -> would be added).
  (v)   CONTROL: a twin model that takes ``ctx_len`` but does NOT pass it to the op leaves it dangling
        (0 uses), so after DCE it cannot be retrieved from any op node — showing the carry is load-bearing.

Throwaway probe (not a committed test). The committed Step-2 test exercises the real transform.
"""

import torch

import tensorrt_llm._torch.auto_deploy  # noqa: F401  (registers auto_deploy::* ops)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, BLOCK, NH, NKV, HD = 2, 5, 4, 2, 16


class ToyDraftCarry(torch.nn.Module):
    """1-layer toy draft: forward takes ctx_len and carries it into the source op."""

    def forward(self, q, k, v, ctx_len):
        return torch.ops.auto_deploy.dflash_attention(q, k, v, ctx_len, None)


class ToyDraftNoCarry(torch.nn.Module):
    """Control: takes ctx_len but does NOT pass it to the op (uses torch_attention directly)."""

    def forward(self, q, k, v, ctx_len):
        return torch.ops.auto_deploy.torch_attention(q, k, v, is_causal=False, layout="bsnd")


def _example_inputs():
    q = torch.randn(B, BLOCK, NH, HD, device=DEV)
    k = torch.randn(B, BLOCK, NKV, HD, device=DEV)
    v = torch.randn(B, BLOCK, NKV, HD, device=DEV)
    ctx_len = torch.tensor([7, 4], device=DEV, dtype=torch.int32)
    return (q, k, v, ctx_len)


def _placeholders(gm):
    return [n for n in gm.graph.nodes if n.op == "placeholder"]


def _dflash_nodes(gm):
    return [
        n for n in gm.graph.nodes if n.op == "call_function" and "dflash_attention" in str(n.target)
    ]


def _simulate_add_or_retrieve(gm, name):
    """Mirror _add_or_retrieve_input's decision (without a real CachedSequenceInterface)."""
    found = gm.graph.find_nodes(op="placeholder", target=name)
    if len(found) == 0:
        return "ADD"  # activate_arg + add_graph_input
    elif len(found) == 1:
        return "RETRIEVE"  # return the existing placeholder
    return f"ERROR(>1: {len(found)})"


def main():
    torch.manual_seed(0)
    ok = True

    # ---- carry model -------------------------------------------------------------------------- #
    ep = torch.export.export(ToyDraftCarry().to(DEV), _example_inputs())
    gm = ep.module()
    ph_names = [n.target for n in _placeholders(gm)]
    print(f"[carry] placeholders: {ph_names}")

    # (i) ctx_len placeholder present, named "ctx_len"
    ctx_phs = gm.graph.find_nodes(op="placeholder", target="ctx_len")
    cond_i = len(ctx_phs) == 1
    print(
        f"  (i)   ctx_len placeholder present (==1): {len(ctx_phs)}  {'PASS' if cond_i else 'FAIL'}"
    )
    ok &= cond_i

    # (ii) ctx_len is an arg of the dflash_attention node (carried by the source op)
    dnodes = _dflash_nodes(gm)
    cond_ii = len(dnodes) == 1 and (len(ctx_phs) == 1 and ctx_phs[0] in dnodes[0].all_input_nodes)
    print(
        f"  (ii)  ctx_len placeholder is an input of the dflash_attention node: "
        f"{'PASS' if cond_ii else 'FAIL'}  (dflash nodes={len(dnodes)}, "
        f"args={[str(a) for a in (dnodes[0].args if dnodes else [])]})"
    )
    ok &= cond_ii

    # (iii) survives DCE (the carry keeps it live)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    ctx_phs_after = gm.graph.find_nodes(op="placeholder", target="ctx_len")
    dnodes_after = _dflash_nodes(gm)
    cond_iii = len(ctx_phs_after) == 1 and len(dnodes_after) == 1
    print(
        f"  (iii) ctx_len + dflash node survive DCE: "
        f"ctx_len={len(ctx_phs_after)}, dflash={len(dnodes_after)}  {'PASS' if cond_iii else 'FAIL'}"
    )
    ok &= cond_iii

    # (iv) retrieve ctx_len (==1 -> RETRIEVE), slot_idx absent (==0 -> ADD)
    dec_ctx = _simulate_add_or_retrieve(gm, "ctx_len")
    dec_slot = _simulate_add_or_retrieve(gm, "slot_idx")
    cond_iv = dec_ctx == "RETRIEVE" and dec_slot == "ADD"
    print(
        f"  (iv)  _add_or_retrieve_input: ctx_len->{dec_ctx} (want RETRIEVE), "
        f"slot_idx->{dec_slot} (want ADD)  {'PASS' if cond_iv else 'FAIL'}"
    )
    ok &= cond_iv

    # ---- control model (no carry) ------------------------------------------------------------- #
    ep_nc = torch.export.export(ToyDraftNoCarry().to(DEV), _example_inputs())
    gm_nc = ep_nc.module()
    ctx_ph_nc = gm_nc.graph.find_nodes(op="placeholder", target="ctx_len")
    # Under raw torch.export an unused input is NOT pruned — it is retained but consumed only by the
    # export guard machinery (``_guards_fn``), never by an attention op. So the meaningful negative is
    # "not carried by any attention op": insertion would have no op-carried ctx_len to retrieve, and a
    # pipeline that DCEs non-output, op-unused inputs would drop it.
    user_targets = [str(u.target) for u in ctx_ph_nc[0].users] if ctx_ph_nc else []
    only_guard_user = all("guard" in t.lower() for t in user_targets)  # not a real data use
    dnodes_nc = [
        n for n in gm_nc.graph.nodes if n.op == "call_function" and "attention" in str(n.target)
    ]
    carried_nc = any((ctx_ph_nc and ctx_ph_nc[0] in n.all_input_nodes) for n in dnodes_nc)
    cond_v = (not carried_nc) and only_guard_user
    print(
        f"  (v)   CONTROL no-carry: carried-by-attn-op={carried_nc}, "
        f"ctx_len users={user_targets} (guard-only={only_guard_user})  "
        f"{'PASS' if cond_v else 'FAIL'}  (not op-carried => not meaningfully retrievable)"
    )
    ok &= cond_v

    print("\nSPIKE B RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
