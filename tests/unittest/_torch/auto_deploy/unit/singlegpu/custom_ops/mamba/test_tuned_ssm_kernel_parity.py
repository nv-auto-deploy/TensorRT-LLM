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

"""Parity tests: tuned_ssm_kernel vs flashinfer.mamba.selective_state_update.

These tests guard against behavioral divergence between the two dispatch paths:
- batch <= _TUNED_DECODE_THRESHOLD: flashinfer (no dt clamping)
- batch >  _TUNED_DECODE_THRESHOLD: tuned Triton kernel

The canonical failure mode is adding hardcoded dt_clamp to the tuned kernel
while flashinfer applies no clamping — this caused a catastrophic accuracy
regression (GSM8K: 1.4% vs reference 68.7%) that only showed at large batch.

Test strategy:
1. FlashInfer parity: run both on identical inputs with dt values OUTSIDE any
   plausible clamping range; assert outputs match within bf16 tolerance.
2. Cross-threshold consistency: batch=32 (flashinfer path) vs batch=33
   (tuned path) with the same per-sequence inputs; assert per-sequence outputs
   are identical within bf16 tolerance.
3. dt_clamp sensitivity: show that adding clamping to the tuned kernel while
   running without it on flashinfer would produce a large diff (regression
   detector self-test).
"""

import pytest
import torch

# Nemotron Nano v3 decode dimensions
NHEADS = 64
DIM = 64
DSTATE = 128
NGROUPS = 8

# The dispatch boundary: <= this → flashinfer, > this → tuned Triton kernel
_TUNED_DECODE_THRESHOLD = 32


def _make_inputs(batch, seed=42, device="cuda"):
    """Create reproducible random inputs for one decode step.

    dt is built in production layout: base shape [batch, nheads] expanded to
    [batch, nheads, dim] with stride 0 on the last axis.  FlashInfer requires
    dt.stride(1)==1, which this layout satisfies (stride = (nheads, 1, 0)).

    dt values are drawn from a wide range so some will be outside any narrow
    clamping window like [0.001, 0.1] after softplus — these are the cases
    that distinguish a clamping kernel from a non-clamping one.
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    state = torch.randn(batch, NHEADS, DIM, DSTATE, dtype=torch.float32, device=device, generator=g)
    x = torch.randn(batch, NHEADS, DIM, dtype=torch.bfloat16, device=device, generator=g)
    # Use a wide dt range: raw values [-5, 5] → softplus → [0.007, 5.007]
    # Values > 0.1 after softplus would be clamped to 0.1 by the old bug.
    # Match production layout (from mamba_backend_common._prepare_ssm_grouped_state_update_inputs):
    # Per-head scalars (A, D, dt_bias) are 1D [nheads] expanded to include head_dim,
    # so stride(0)==1 and stride(last)==0.  FlashInfer validates these strides.
    # dt is [batch, nheads] expanded to [batch, nheads, dim] with stride(1)==1.

    # Use a wide dt range: raw values [-5, 5] → softplus → [0.007, 5.007]
    # Values > 0.1 after softplus would be clamped to 0.1 by the old bug.
    dt_base = torch.rand(batch, NHEADS, dtype=torch.bfloat16, device=device, generator=g) * 10 - 5
    dt = dt_base.unsqueeze(-1).expand(-1, -1, DIM)  # strides: (nheads, 1, 0)

    # FlashInfer requires dt_bias to have the same dtype as dt (bfloat16)
    dt_bias_1d = torch.randn(NHEADS, dtype=torch.bfloat16, device=device, generator=g) * 0.1
    dt_bias = dt_bias_1d.unsqueeze(-1).expand(-1, DIM)  # strides: (1, 0)

    A_1d = -torch.rand(NHEADS, dtype=torch.float32, device=device, generator=g)
    A = A_1d[:, None, None].expand(-1, DIM, DSTATE)  # strides: (1, 0, 0)

    B = torch.randn(batch, NGROUPS, DSTATE, dtype=torch.bfloat16, device=device, generator=g)
    C = torch.randn(batch, NGROUPS, DSTATE, dtype=torch.bfloat16, device=device, generator=g)

    # FlashInfer requires D to have the same dtype as dt (bfloat16)
    D_1d = torch.randn(NHEADS, dtype=torch.bfloat16, device=device, generator=g)
    D = D_1d.unsqueeze(-1).expand(-1, DIM)  # strides: (1, 0)

    return state, x, dt, dt_bias, A, B, C, D


def _run_flashinfer(state, x, dt, A, B, C, D, dt_bias):
    """Run flashinfer.mamba.selective_state_update (no dt clamping)."""
    import flashinfer

    state_fi = state.clone()
    return flashinfer.mamba.selective_state_update(
        state_fi,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        z=None,
        dt_bias=dt_bias,
        dt_softplus=True,
    )


def _run_tuned(state, x, dt, A, B, C, D, dt_bias, dt_clamp_min=None, dt_clamp_max=None):
    """Run tuned_selective_state_update with optional dt clamping."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.mamba.tuned_ssm_kernel import (
        tuned_selective_state_update,
    )

    state_t = state.clone()
    out = torch.empty(state.shape[0], NHEADS, DIM, dtype=x.dtype, device=x.device)
    tuned_selective_state_update(
        state_t,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
        dt_clamp_min=dt_clamp_min,
        dt_clamp_max=dt_clamp_max,
    )
    return out


@pytest.mark.parametrize("batch", [33, 64, 128, 256])
def test_tuned_kernel_matches_flashinfer(batch):
    """Tuned kernel (no clamping) must match flashinfer within bf16 tolerance.

    This is the primary guard against dispatch-path behavioral divergence.
    Uses dt values outside any narrow clamping window to maximize sensitivity.
    """
    pytest.importorskip("flashinfer", reason="flashinfer not installed")
    assert batch > _TUNED_DECODE_THRESHOLD, "batch must be in tuned kernel path"

    state, x, dt, dt_bias, A, B, C, D = _make_inputs(batch)

    out_fi = _run_flashinfer(state, x, dt, A, B, C, D, dt_bias)
    out_tuned = _run_tuned(state, x, dt, A, B, C, D, dt_bias, dt_clamp_min=None, dt_clamp_max=None)

    max_diff = (out_fi.float() - out_tuned.float()).abs().max().item()
    mean_diff = (out_fi.float() - out_tuned.float()).abs().mean().item()

    assert max_diff < 0.5, (
        f"batch={batch}: tuned kernel diverges from flashinfer "
        f"(max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}). "
        "Check for dt_clamp or other behavioral differences."
    )


def test_cross_threshold_consistency():
    """Output at batch=32 (flashinfer) vs batch=33 (tuned) must match per-sequence.

    Runs 32 sequences through flashinfer and 33 sequences through the tuned
    kernel (including those same 32), then checks the outputs for sequences
    0..31 agree within bf16 tolerance.
    """
    pytest.importorskip("flashinfer", reason="flashinfer not installed")

    # Use batch=33 for the tuned path, take first 32 seqs for comparison
    state, x, dt, dt_bias, A, B, C, D = _make_inputs(batch=33)

    # Flashinfer: first 32 sequences
    out_fi = _run_flashinfer(state[:32], x[:32], dt[:32], A, B[:32], C[:32], D, dt_bias)

    # Tuned kernel: all 33 sequences, compare first 32
    out_tuned = _run_tuned(state, x, dt, A, B, C, D, dt_bias)

    max_diff = (out_fi.float() - out_tuned[:32].float()).abs().max().item()

    assert max_diff < 0.5, (
        f"Cross-threshold inconsistency: flashinfer(batch=32) vs tuned(batch=33)[:32] "
        f"max_diff={max_diff:.4f}. The two dispatch paths produce different results."
    )


def test_clamping_causes_regression():
    """Self-test: hardcoded dt_clamp on the tuned kernel DOES produce large diff vs flashinfer.

    This verifies that our test has the sensitivity to catch the original bug.
    If this test starts failing (i.e., clamping no longer matters), it means
    the test inputs need to be updated.
    """
    pytest.importorskip("flashinfer", reason="flashinfer not installed")

    # Use batch=64: in tuned kernel path
    batch = 64
    state, x, dt, dt_bias, A, B, C, D = _make_inputs(batch)

    out_fi = _run_flashinfer(state, x, dt, A, B, C, D, dt_bias)
    # Reproduce the original bug: hardcode clamp [0.001, 0.1] on tuned kernel
    out_clamped = _run_tuned(
        state, x, dt, A, B, C, D, dt_bias, dt_clamp_min=0.001, dt_clamp_max=0.1
    )

    max_diff_clamped = (out_fi.float() - out_clamped.float()).abs().max().item()

    # The clamped version SHOULD diverge from flashinfer significantly
    assert max_diff_clamped > 1.0, (
        f"Expected large diff when dt_clamp=[0.001,0.1] vs flashinfer, "
        f"but got max_diff={max_diff_clamped:.4f}. "
        "Update dt range in _make_inputs to ensure sensitivity."
    )
