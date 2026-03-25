# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Benchmark and parameter-sweep script for _mla_attention_kernel.

Benchmarks the Triton MLA attention kernel used by Mistral-Small-4-119B-2603
(kv_lora_rank=256, qk_rope_head_dim=64, num_heads=32).

Usage:
  # Run all shapes, print table
  python sweep_triton_mla.py

  # Correctness check vs pure-torch reference
  python sweep_triton_mla.py --correctness

  # Override kernel parameters
  python sweep_triton_mla.py --seq-block 32 --num-warps 4 --num-stages 3

  # Full parameter sweep -> JSON
  python sweep_triton_mla.py --sweep
  python sweep_triton_mla.py --sweep --output my_results.json

  # Sweep only a subset of parameters
  python sweep_triton_mla.py --sweep --sweep-seq-blocks 8,16,32,64
"""

import argparse
import json
import math
import sys
import time
from itertools import product
from pathlib import Path

import torch
import triton
import triton.testing

# ---------------------------------------------------------------------------
# Make sure the package is importable when run as a standalone script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tensorrt_llm._torch.auto_deploy.custom_ops.mla.triton_mla import (  # noqa: E402
    _mla_attention_kernel_multihead,
    _mla_attention_kernel_splitk,
    _mla_splitk_reduce,
)

# ---------------------------------------------------------------------------
# Mistral-Small-4-119B-2603 fixed dimensions
# ---------------------------------------------------------------------------
KV_LORA_RANK = 256  # kv_lora_rank
QK_ROPE_HEAD_DIM = 64  # qk_rope_head_dim
QK_NOPE_HEAD_DIM = 64  # qk_nope_head_dim
V_HEAD_DIM = 128  # v_head_dim
NUM_HEADS = 32  # num_attention_heads
CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 320

KV_BLOCK = triton.next_power_of_2(KV_LORA_RANK)  # 256
PE_BLOCK = triton.next_power_of_2(QK_ROPE_HEAD_DIM)  # 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 128
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)

MODEL_DTYPE = torch.bfloat16
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Benchmark shape matrix
# (shape_id, num_tokens, kv_len, description)
# For decode shapes: num_tokens = batch_size, S=1 per seq.
# For prefill shapes: num_tokens = total sequence length (single sequence).
# ---------------------------------------------------------------------------
SHAPES = [
    # ---- Model A: Mistral-Small-4 (Decode — short context) ----
    ("A1", 1, 64, "decode  B=1  kv=64"),
    ("A2", 1, 256, "decode  B=1  kv=256"),
    ("A3", 1, 512, "decode  B=1  kv=512"),
    ("A4", 1, 1024, "decode  B=1  kv=1024"),
    ("A5", 1, 2048, "decode  B=1  kv=2048"),
    # ---- Model A: Mistral-Small-4 (Decode — batched) ----
    ("A6", 8, 256, "decode  B=8  kv=256"),
    ("A7", 8, 512, "decode  B=8  kv=512"),
    ("A8", 16, 512, "decode  B=16 kv=512"),
    ("A9", 32, 256, "decode  B=32 kv=256"),
    ("A10", 32, 512, "decode  B=32 kv=512"),
    # ---- Model B: Mistral-Small-4 (Prefill) ----
    ("B1", 128, 128, "prefill T=128"),
    ("B2", 512, 512, "prefill T=512"),
    ("B3", 1024, 1024, "prefill T=1024"),
    ("B4", 2048, 2048, "prefill T=2048"),
]

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    "seq_block": [4, 8, 16, 32, 64, 128],
    "num_warps": [1, 2, 4, 8, 16],
    "num_stages": [1, 2, 3, 4, 5],
}

# Default params (baseline)
DEFAULT_DECODE_PARAMS = dict(SEQ_BLOCK=64, num_warps=8, num_stages=3)
DEFAULT_PREFILL_PARAMS = dict(SEQ_BLOCK=16, num_warps=4, num_stages=2)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def make_kernel_inputs(num_tokens: int, kv_len: int, max_seq_len: int = None):
    """Create random inputs for direct kernel benchmarking."""
    if max_seq_len is None:
        max_seq_len = max(kv_len, 64)
    # q_absorbed: [num_tokens, NUM_HEADS, KV_LORA_RANK] fp32
    q_absorbed = torch.randn(
        num_tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE, dtype=torch.float32
    )
    # q_pe: [num_tokens, NUM_HEADS, QK_ROPE_HEAD_DIM] fp32
    q_pe = torch.randn(num_tokens, NUM_HEADS, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=torch.float32)
    # mla_cache: [num_tokens, max_seq_len, CACHE_DIM] bf16 (one slot per token)
    mla_cache = torch.randn(num_tokens, max_seq_len, CACHE_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    # token_slot: each token maps to its own batch slot (0..num_tokens-1)
    token_slot = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)
    # kv_len per token: all attend to kv_len positions
    token_kv_len = torch.full((num_tokens,), kv_len, device=DEVICE, dtype=torch.int32)
    # output buffer
    out = torch.empty(num_tokens, NUM_HEADS, KV_LORA_RANK, device=DEVICE, dtype=torch.float32)
    return q_absorbed, q_pe, mla_cache, token_slot, token_kv_len, out, max_seq_len


def make_decode_launcher_inputs(batch_size: int, kv_len: int):
    """Create inputs for the _triton_mla_decode launcher."""
    max_seq = max(kv_len + 1, 64)
    q_nope = torch.randn(
        batch_size, 1, NUM_HEADS, QK_NOPE_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE
    )
    q_pe = torch.randn(batch_size, 1, NUM_HEADS, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    compressed_kv = torch.randn(batch_size, 1, KV_LORA_RANK, device=DEVICE, dtype=MODEL_DTYPE)
    kpe = torch.randn(batch_size, 1, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    kv_b_proj = torch.randn(
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK, device=DEVICE, dtype=MODEL_DTYPE
    )
    mla_cache = torch.randn(batch_size, max_seq, CACHE_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    slot_idx = torch.arange(batch_size, device=DEVICE, dtype=torch.int32)
    # Current position = kv_len - 1 (new token goes at kv_len)
    input_pos = torch.full((batch_size,), kv_len - 1, device=DEVICE, dtype=torch.int32)
    # Pre-fill cache with the kv_len - 1 positions
    mla_cache[:, : kv_len - 1, :] = torch.randn(
        batch_size, kv_len - 1, CACHE_DIM, device=DEVICE, dtype=MODEL_DTYPE
    )
    out = torch.empty(batch_size, NUM_HEADS, V_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    return q_nope, q_pe, compressed_kv, kpe, kv_b_proj, mla_cache, slot_idx, input_pos, out


def make_prefill_launcher_inputs(total_tokens: int, kv_len: int):
    """Create inputs for the _triton_mla_prefill launcher (single sequence)."""
    max_seq = max(kv_len + 1, 64)
    # Single sequence of total_tokens tokens
    q_nope = torch.randn(
        total_tokens, NUM_HEADS, QK_NOPE_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE
    )
    q_pe = torch.randn(total_tokens, NUM_HEADS, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    compressed_kv = torch.randn(total_tokens, KV_LORA_RANK, device=DEVICE, dtype=MODEL_DTYPE)
    kpe = torch.randn(total_tokens, 1, QK_ROPE_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    kv_b_proj = torch.randn(
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK, device=DEVICE, dtype=MODEL_DTYPE
    )
    mla_cache = torch.randn(1, max_seq, CACHE_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    slot_idx = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    input_pos = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    seq_len = torch.tensor([total_tokens], device=DEVICE, dtype=torch.int32)
    seq_start = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    out = torch.zeros(total_tokens, NUM_HEADS, V_HEAD_DIM, device=DEVICE, dtype=MODEL_DTYPE)
    return (
        q_nope,
        q_pe,
        compressed_kv,
        kpe,
        kv_b_proj,
        mla_cache,
        input_pos,
        slot_idx,
        seq_len,
        seq_start,
        out,
    )


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch, for correctness)
# ---------------------------------------------------------------------------


def reference_mla_attention(
    q_absorbed: torch.Tensor,  # [T, N, KV_LORA_RANK]
    q_pe: torch.Tensor,  # [T, N, QK_ROPE_HEAD_DIM]
    mla_cache: torch.Tensor,  # [T, max_seq, CACHE_DIM]  (one slot per token)
    token_kv_len: torch.Tensor,  # [T]
    scale: float,
) -> torch.Tensor:
    """Pure-torch reference for _mla_attention_kernel output.

    For each (token, head), attends to the first kv_len cache positions
    in that token's batch slot (slot_idx = token_id for our test tensors).
    Returns weighted_kv: [T, N, KV_LORA_RANK] fp32.
    """
    T, N, _ = q_absorbed.shape
    max_kv_len = token_kv_len.max().item()
    # Extract compressed_kv and kpe from the cache (use only token's own slot, index = token_id)
    # cache: [T, max_seq, CACHE_DIM]
    ckv = mla_cache[:, :max_kv_len, :KV_LORA_RANK].float()  # [T, S, KV_LORA_RANK]
    kpe = mla_cache[:, :max_kv_len, KV_LORA_RANK:].float()  # [T, S, QK_ROPE_HEAD_DIM]

    # scores_nope: [T, N, S]  = q_absorbed [T, N, K] @ ckv [T, S, K]^T
    scores_nope = torch.einsum("tnk,tsk->tns", q_absorbed.float(), ckv)
    # scores_pe:   [T, N, S]  = q_pe [T, N, P] @ kpe [T, S, P]^T
    scores_pe = torch.einsum("tnp,tsp->tns", q_pe.float(), kpe)
    scores = (scores_nope + scores_pe) * scale  # [T, N, S]

    # Causal mask: token t attends to positions [0, kv_len[t])
    kv_lens = token_kv_len.long()  # [T]
    seq_idx = torch.arange(max_kv_len, device=q_absorbed.device)  # [S]
    mask = seq_idx[None, None, :] < kv_lens[:, None, None]  # [T, 1, S]
    scores = scores.masked_fill(~mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)  # handle all-inf rows

    # weighted_kv: [T, N, K]
    weighted_kv = torch.einsum("tns,tsk->tnk", attn, ckv)
    return weighted_kv


# ---------------------------------------------------------------------------
# Correctness check
# ---------------------------------------------------------------------------


def run_correctness(shapes=SHAPES, HEAD_BLOCK=8, SEQ_BLOCK=64, num_warps=8, num_stages=3):
    """Compare _mla_attention_kernel_multihead output against pure-torch reference."""
    print("\n=== Correctness Check ===")
    print(
        f"Config: HEAD_BLOCK={HEAD_BLOCK}, SEQ_BLOCK={SEQ_BLOCK}, "
        f"num_warps={num_warps}, num_stages={num_stages}"
    )
    header = f"{'ID':>4}  {'shape':>24}  {'max_abs_err':>12}  {'max_rel_err':>12}  {'status':>6}"
    print(header)
    print("-" * len(header))
    all_pass = True
    for shape_id, num_tokens, kv_len, desc in shapes:
        q_absorbed, q_pe, mla_cache, token_slot, token_kv_len, out, max_seq_len = (
            make_kernel_inputs(num_tokens, kv_len)
        )
        # Use multihead kernel (the only active kernel since iter 16)
        num_head_groups = NUM_HEADS // HEAD_BLOCK
        grid = (num_tokens, num_head_groups)
        _mla_attention_kernel_multihead[grid](
            q_absorbed,
            q_pe,
            mla_cache,
            token_slot,
            token_kv_len,
            out,
            SCALE=SCALE,
            MAX_SEQ_LEN=max_seq_len,
            N_HEADS=NUM_HEADS,
            KV_LORA_RANK=KV_LORA_RANK,
            QK_ROPE_HEAD_DIM=QK_ROPE_HEAD_DIM,
            CACHE_DIM=CACHE_DIM,
            KV_BLOCK=KV_BLOCK,
            PE_BLOCK=PE_BLOCK,
            SEQ_BLOCK=SEQ_BLOCK,
            HEAD_BLOCK=HEAD_BLOCK,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        torch.cuda.synchronize()

        ref = reference_mla_attention(q_absorbed, q_pe, mla_cache, token_kv_len, SCALE)

        abs_err = (out - ref).abs().max().item()
        rel_err = ((out - ref).abs() / (ref.abs() + 1e-6)).max().item()
        status = "PASS" if abs_err < 1e-2 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{shape_id:>4}  {desc:>24}  {abs_err:>12.6f}  {rel_err:>12.6f}  {status:>6}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def bench_multihead_kernel(
    num_tokens: int,
    kv_len: int,
    HEAD_BLOCK: int,
    SEQ_BLOCK: int,
    num_warps: int,
    num_stages: int,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark _mla_attention_kernel_multihead in isolation. Returns µs.

    tl.dot requires SEQ_BLOCK >= 16; callers must enforce this constraint.
    """
    assert NUM_HEADS % HEAD_BLOCK == 0, (
        f"N_HEADS={NUM_HEADS} must be divisible by HEAD_BLOCK={HEAD_BLOCK}"
    )
    assert SEQ_BLOCK >= 16, f"SEQ_BLOCK={SEQ_BLOCK} must be >= 16 for tl.dot in multihead kernel"
    q_absorbed, q_pe, mla_cache, token_slot, token_kv_len, out, max_seq_len = make_kernel_inputs(
        num_tokens, kv_len
    )
    num_head_groups = NUM_HEADS // HEAD_BLOCK
    grid = (num_tokens, num_head_groups)

    fn = lambda: _mla_attention_kernel_multihead[grid](  # noqa: E731
        q_absorbed,
        q_pe,
        mla_cache,
        token_slot,
        token_kv_len,
        out,
        SCALE=SCALE,
        MAX_SEQ_LEN=max_seq_len,
        N_HEADS=NUM_HEADS,
        KV_LORA_RANK=KV_LORA_RANK,
        QK_ROPE_HEAD_DIM=QK_ROPE_HEAD_DIM,
        CACHE_DIM=CACHE_DIM,
        KV_BLOCK=KV_BLOCK,
        PE_BLOCK=PE_BLOCK,
        SEQ_BLOCK=SEQ_BLOCK,
        HEAD_BLOCK=HEAD_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3


def bench_splitk_kernel(
    num_tokens: int,
    kv_len: int,
    HEAD_BLOCK: int,
    SEQ_BLOCK: int,
    num_warps: int,
    num_stages: int,
    num_parts: int = 8,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Benchmark _mla_attention_kernel_splitk + _mla_splitk_reduce. Returns µs.

    Used for small-batch long-context shapes (num_tokens <= 4, kv_len >= 512).
    """
    assert NUM_HEADS % HEAD_BLOCK == 0
    assert SEQ_BLOCK >= 16
    q_absorbed, q_pe, mla_cache, token_slot, token_kv_len, _out, max_seq_len = make_kernel_inputs(
        num_tokens, kv_len
    )
    ws_acc = torch.empty(
        num_tokens, NUM_HEADS, num_parts, KV_BLOCK, device=DEVICE, dtype=torch.float32
    )
    ws_ml = torch.empty(num_tokens, NUM_HEADS, num_parts, 2, device=DEVICE, dtype=torch.float32)
    out = torch.empty(num_tokens, NUM_HEADS, KV_BLOCK, dtype=MODEL_DTYPE, device=DEVICE)
    num_hg = NUM_HEADS // HEAD_BLOCK
    grid_sk = (num_tokens, num_hg, num_parts)
    grid_r = (num_tokens, NUM_HEADS)

    def fn():
        _mla_attention_kernel_splitk[grid_sk](
            q_absorbed,
            q_pe,
            mla_cache,
            token_slot,
            token_kv_len,
            ws_acc,
            ws_ml,
            SCALE=SCALE,
            MAX_SEQ_LEN=max_seq_len,
            N_HEADS=NUM_HEADS,
            KV_LORA_RANK=KV_LORA_RANK,
            QK_ROPE_HEAD_DIM=QK_ROPE_HEAD_DIM,
            CACHE_DIM=CACHE_DIM,
            KV_BLOCK=KV_BLOCK,
            PE_BLOCK=PE_BLOCK,
            SEQ_BLOCK=SEQ_BLOCK,
            HEAD_BLOCK=HEAD_BLOCK,
            NUM_PARTS=num_parts,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _mla_splitk_reduce[grid_r](
            ws_acc,
            ws_ml,
            out,
            N_HEADS=NUM_HEADS,
            KV_LORA_RANK=KV_LORA_RANK,
            KV_BLOCK=KV_BLOCK,
            NUM_PARTS=num_parts,
            num_warps=4,
        )

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    shapes=SHAPES,
    decode_params: dict = None,
    prefill_params: dict = None,
    baseline_results: dict = None,
    warmup: int = 25,
    rep: int = 100,
) -> list:
    """Run kernel-only benchmark for all shapes using multihead kernel. Returns list of result dicts."""
    if decode_params is None:
        decode_params = DEFAULT_DECODE_PARAMS
    if prefill_params is None:
        prefill_params = DEFAULT_PREFILL_PARAMS

    results = []
    for shape_id, num_tokens, kv_len, desc in shapes:
        is_prefill = shape_id.startswith("B")
        params = prefill_params if is_prefill else decode_params
        hb = params.get("HEAD_BLOCK", 8)
        sb = max(params["SEQ_BLOCK"], 16)  # tl.dot requires SEQ_BLOCK >= 16
        # Use split-K for small-batch long-context decode (mirrors _triton_mla_decode dispatch)
        use_splitk = not is_prefill and num_tokens <= 4 and kv_len >= 512
        try:
            if use_splitk:
                # Adaptive SB for split-K: SB=64 maximizes partition fill for kv≤1536,
                # SB=128 better for kv>1536 (more computation per block amortizes overhead).
                sb_sk = 64 if kv_len <= 1536 else 128
                kernel_us = bench_splitk_kernel(
                    num_tokens,
                    kv_len,
                    HEAD_BLOCK=hb,
                    SEQ_BLOCK=sb_sk,
                    num_warps=params["num_warps"],
                    num_stages=params["num_stages"],
                    warmup=warmup,
                    rep=rep,
                )
            else:
                kernel_us = bench_multihead_kernel(
                    num_tokens,
                    kv_len,
                    HEAD_BLOCK=hb,
                    SEQ_BLOCK=sb,
                    num_warps=params["num_warps"],
                    num_stages=params["num_stages"],
                    warmup=warmup,
                    rep=rep,
                )
        except Exception as e:
            kernel_us = float("nan")
            print(f"  {shape_id} ERROR: {e}")

        delta_str = ""
        if baseline_results:
            base = baseline_results.get(shape_id)
            if base and not math.isnan(kernel_us):
                pct = 100.0 * (kernel_us - base) / base
                sign = "+" if pct >= 0 else ""
                delta_str = f"{sign}{pct:.1f}%"

        results.append(
            {
                "shape_id": shape_id,
                "num_tokens": num_tokens,
                "kv_len": kv_len,
                "desc": desc,
                "kernel_us": kernel_us,
                "params": params,
                "delta": delta_str,
            }
        )
    return results


def print_results_table(results: list, title: str = ""):
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
    cols = f"{'ID':>4}  {'T':>5}  {'kv_len':>6}  {'SEQ_BLK':>7}  {'warps':>5}  {'stgs':>4}"
    header = cols + f"  {'kernel µs':>10}  {'vs base':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        p = r["params"]
        print(
            f"{r['shape_id']:>4}  {r['num_tokens']:>5}  {r['kv_len']:>6}  "
            f"{p.get('SEQ_BLOCK', '?'):>7}  {p.get('num_warps', '?'):>5}  "
            f"{p.get('num_stages', '?'):>4}  {r['kernel_us']:>10.1f}  "
            f"{r['delta']:>8}"
        )
    print()


# ---------------------------------------------------------------------------
# Full parameter sweep
# ---------------------------------------------------------------------------


def run_sweep(
    shapes=SHAPES,
    output_path: str = None,
    seq_blocks=None,
    num_warps_list=None,
    num_stages_list=None,
    head_block: int = 8,
    warmup: int = 25,
    rep: int = 50,
):
    """Sweep SEQ_BLOCK × num_warps × num_stages for all shapes using multihead kernel."""
    sb_vals = seq_blocks or SWEEP_GRID["seq_block"]
    nw_vals = num_warps_list or SWEEP_GRID["num_warps"]
    ns_vals = num_stages_list or SWEEP_GRID["num_stages"]

    combos = list(product(sb_vals, nw_vals, ns_vals))
    total = len(combos) * len(shapes)
    print(f"Sweep: {len(combos)} configs × {len(shapes)} shapes = {total} runs")

    all_results = []
    for i, (sb, nw, ns) in enumerate(combos):
        for shape_id, num_tokens, kv_len, desc in shapes:
            try:
                # tl.dot requires SEQ_BLOCK >= 16
                if sb < 16:
                    raise ValueError(f"SEQ_BLOCK={sb} < 16 required by tl.dot")
                kernel_us = bench_multihead_kernel(
                    num_tokens,
                    kv_len,
                    HEAD_BLOCK=head_block,
                    SEQ_BLOCK=sb,
                    num_warps=nw,
                    num_stages=ns,
                    warmup=warmup,
                    rep=rep,
                )
                all_results.append(
                    {
                        "shape_id": shape_id,
                        "num_tokens": num_tokens,
                        "kv_len": kv_len,
                        "SEQ_BLOCK": sb,
                        "num_warps": nw,
                        "num_stages": ns,
                        "kernel_us": kernel_us,
                    }
                )
            except Exception as e:
                all_results.append(
                    {
                        "shape_id": shape_id,
                        "num_tokens": num_tokens,
                        "kv_len": kv_len,
                        "SEQ_BLOCK": sb,
                        "num_warps": nw,
                        "num_stages": ns,
                        "kernel_us": float("nan"),
                        "error": str(e),
                    }
                )
        if (i + 1) % 5 == 0 or (i + 1) == len(combos):
            print(f"  {i + 1}/{len(combos)} configs done...")

    # Print best per shape
    print("\n=== Best config per shape ===")
    shape_ids = list(dict.fromkeys(r["shape_id"] for r in all_results))
    best_configs = {}
    for sid in shape_ids:
        candidates = [
            r for r in all_results if r["shape_id"] == sid and not math.isnan(r["kernel_us"])
        ]
        if candidates:
            best = min(candidates, key=lambda r: r["kernel_us"])
            best_configs[sid] = best
            print(
                f"  {sid:>4}  kernel={best['kernel_us']:>8.1f} µs  "
                f"SEQ_BLOCK={best['SEQ_BLOCK']:>3}  num_warps={best['num_warps']:>2}  "
                f"num_stages={best['num_stages']:>1}"
            )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {output_path}")

    return all_results, best_configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark sweep for _mla_attention_kernel")
    parser.add_argument(
        "--correctness", action="store_true", help="Run correctness check vs pure-torch reference"
    )
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for sweep results (default: auto-timestamped)",
    )
    # Kernel parameter overrides
    parser.add_argument(
        "--seq-block", type=int, default=None, help="Override SEQ_BLOCK for decode shapes"
    )
    parser.add_argument(
        "--seq-block-prefill",
        type=int,
        default=None,
        help="Override SEQ_BLOCK for prefill shapes (default: same as --seq-block)",
    )
    parser.add_argument("--num-warps", type=int, default=None)
    parser.add_argument("--num-stages", type=int, default=None)
    # Sweep overrides
    parser.add_argument(
        "--sweep-seq-blocks",
        type=str,
        default=None,
        help="Comma-separated SEQ_BLOCK values for sweep, e.g. 8,16,32,64",
    )
    parser.add_argument("--sweep-num-warps", type=str, default=None)
    parser.add_argument("--sweep-num-stages", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    # Multi-GPU: select which GPU to use (sets CUDA_VISIBLE_DEVICES before torch init)
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU index to use (sets CUDA device; for multi-GPU parallel launches)",
    )
    # HEAD_BLOCK structural experiment
    parser.add_argument(
        "--head-block",
        type=int,
        default=None,
        help="If set, benchmark _mla_attention_kernel_multihead with this HEAD_BLOCK value",
    )
    args = parser.parse_args()

    if args.gpu_id is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        # Re-set DEVICE global after setting visibility
        globals()["DEVICE"] = "cuda"

    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print(f"triton: {triton.__version__}")
    print(f"dtype:  {MODEL_DTYPE}  device: {DEVICE}")
    print("Model:  Mistral-Small-4-119B-2603")
    print(
        f"Dims:   N_HEADS={NUM_HEADS}  KV_LORA_RANK={KV_LORA_RANK}  "
        f"QK_ROPE={QK_ROPE_HEAD_DIM}  V_HEAD={V_HEAD_DIM}  CACHE_DIM={CACHE_DIM}"
    )

    # Build params
    decode_params = dict(DEFAULT_DECODE_PARAMS)
    prefill_params = dict(DEFAULT_PREFILL_PARAMS)
    if args.seq_block is not None:
        decode_params["SEQ_BLOCK"] = args.seq_block
        prefill_params["SEQ_BLOCK"] = args.seq_block
    if args.seq_block_prefill is not None:
        prefill_params["SEQ_BLOCK"] = args.seq_block_prefill
    if args.num_warps is not None:
        decode_params["num_warps"] = args.num_warps
        prefill_params["num_warps"] = args.num_warps
    if args.num_stages is not None:
        decode_params["num_stages"] = args.num_stages
        prefill_params["num_stages"] = args.num_stages

    if args.correctness:
        hb = args.head_block if args.head_block is not None else 8
        ok = run_correctness(
            SHAPES,
            HEAD_BLOCK=hb,
            SEQ_BLOCK=decode_params["SEQ_BLOCK"],
            num_warps=decode_params["num_warps"],
            num_stages=decode_params["num_stages"],
        )
        sys.exit(0 if ok else 1)

    if args.sweep:
        output_path = args.output or f"sweep_results_{int(time.time())}.json"
        sb_list = (
            [int(x) for x in args.sweep_seq_blocks.split(",")] if args.sweep_seq_blocks else None
        )
        nw_list = (
            [int(x) for x in args.sweep_num_warps.split(",")] if args.sweep_num_warps else None
        )
        ns_list = (
            [int(x) for x in args.sweep_num_stages.split(",")] if args.sweep_num_stages else None
        )
        run_sweep(
            SHAPES,
            output_path=output_path,
            seq_blocks=sb_list,
            num_warps_list=nw_list,
            num_stages_list=ns_list,
            warmup=args.warmup,
            rep=args.rep,
        )
        return

    if args.head_block is not None:
        # Structural experiment: benchmark multihead kernel at given HEAD_BLOCK
        hb = args.head_block
        print(f"\n--- HEAD_BLOCK={hb} structural benchmark ---")
        baseline = {
            "A1": 20.9,
            "A2": 66.6,
            "A3": 127.1,
            "A4": 249.0,
            "A5": 491.1,
            "A6": 68.5,
            "A7": 130.2,
            "A8": 149.2,
            "A9": 130.3,
            "A10": 252.6,
            "B1": 396.4,
            "B2": 6107.9,
            "B3": 24183.3,
            "B4": 96220.1,
        }
        cols = f"{'ID':>4}  {'T':>5}  {'kv_len':>6}  {'HB':>4}  {'SB':>5}"
        header = cols + f"  {'kernel µs':>10}  {'vs orig':>9}"
        print(header)
        print("-" * len(header))
        for shape_id, num_tokens, kv_len, desc in SHAPES:
            is_prefill = shape_id.startswith("B")
            params = prefill_params if is_prefill else decode_params
            # multihead kernel requires SEQ_BLOCK >= 16 for tl.dot
            sb = max(params["SEQ_BLOCK"], 16)
            nw = params["num_warps"]
            ns = params["num_stages"]
            try:
                us = bench_multihead_kernel(
                    num_tokens,
                    kv_len,
                    hb,
                    sb,
                    nw,
                    ns,
                    warmup=args.warmup,
                    rep=args.rep,
                )
                base = baseline.get(shape_id, None)
                delta = f"{100.0 * (us - base) / base:+.1f}%" if base else ""
            except Exception as e:
                us, delta = float("nan"), f"ERR: {e}"
            print(
                f"{shape_id:>4}  {num_tokens:>5}  {kv_len:>6}  {hb:>4}  {sb:>5}"
                f"  {us:>10.1f}  {delta:>9}"
            )
        print()
        return

    # Default: run benchmark on all shapes
    results = run_benchmark(SHAPES, decode_params, prefill_params, warmup=args.warmup, rep=args.rep)
    print_results_table(
        results, title=f"Baseline benchmark  decode={decode_params}  prefill={prefill_params}"
    )


if __name__ == "__main__":
    main()
