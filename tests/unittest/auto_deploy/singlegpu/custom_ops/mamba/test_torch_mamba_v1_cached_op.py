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

"""Unit tests for cached Mamba v1 selective scan custom op.

Validates the prefill and decode paths of torch_cached_mamba_v1 against
the uncached reference (torch_mamba_v1_selective_scan), and tests the
critical prefill→decode continuity path used during autoregressive generation.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_mamba_v1  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_mamba_v1  # noqa: F401


def _random_v1_params(device, dtype, batch, seq_len, d_inner, d_state):
    """Create random Mamba v1 parameters."""
    hidden_states = torch.randn(batch, seq_len, d_inner, device=device, dtype=dtype)
    A = -torch.exp(torch.randn(d_inner, d_state, device=device, dtype=torch.float32))
    B = torch.randn(batch, seq_len, d_state, device=device, dtype=dtype)
    C = torch.randn(batch, seq_len, d_state, device=device, dtype=dtype)
    D = torch.ones(d_inner, device=device, dtype=torch.float32)
    dt = torch.nn.functional.softplus(
        torch.randn(batch, seq_len, d_inner, device=device, dtype=dtype)
    )
    return hidden_states, A, B, C, D, dt


@pytest.fixture
def env():
    device = "cuda"
    dtype = torch.bfloat16
    atol = 1e-2
    rtol = 1e-2
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    return {"device": device, "dtype": dtype, "atol": atol, "rtol": rtol}


class TestPrefillPath:
    """Test that the cached op's prefill path matches the uncached reference."""

    def test_single_sequence_prefill_output(self, env):
        """Single sequence: cached prefill output == uncached reference output."""
        device, dtype = env["device"], env["dtype"]
        batch, seq_len, d_inner, d_state = 1, 8, 16, 4
        hs, A, B, C, D, dt = _random_v1_params(device, dtype, batch, seq_len, d_inner, d_state)

        # Reference: uncached op
        y_ref = torch.ops.auto_deploy.torch_mamba_v1_selective_scan(hs, A, B, C, D, dt)

        # Cached op in prefill mode
        max_batch_size = 4
        ssm_cache = torch.zeros(max_batch_size, d_inner, 1, d_state, device=device, dtype=dtype)
        slot_idx = torch.tensor([0], device=device, dtype=torch.int32)
        seq_len_t = torch.tensor([seq_len], device=device, dtype=torch.int32)
        cu_seqlen = torch.tensor([0], device=device, dtype=torch.int32)
        use_initial = torch.zeros(1, device=device, dtype=torch.bool)
        batch_info = torch.tensor([1, seq_len, 0], device=device, dtype=torch.int32)

        y_cached = torch.ops.auto_deploy.torch_cached_mamba_v1(
            hs,
            A,
            B,
            C,
            D,
            dt,
            batch_info,
            seq_len_t,
            cu_seqlen,
            slot_idx,
            use_initial,
            ssm_cache,
        )

        torch.testing.assert_close(y_cached, y_ref, atol=env["atol"], rtol=env["rtol"])

    def test_flattened_multi_sequence_prefill(self, env):
        """Multiple sequences flattened into [1, total_len, D]: each segment matches reference."""
        device, dtype = env["device"], env["dtype"]
        d_inner, d_state = 16, 4
        lens = [5, 3, 7]
        total = sum(lens)

        # Generate flattened inputs
        hs, A, B, C, D, dt = _random_v1_params(device, dtype, 1, total, d_inner, d_state)

        # Cached op
        max_batch_size = 8
        ssm_cache = torch.zeros(max_batch_size, d_inner, 1, d_state, device=device, dtype=dtype)
        slot_idx = torch.tensor([2, 5, 0], device=device, dtype=torch.int32)
        seq_len_t = torch.tensor(lens, device=device, dtype=torch.int32)
        cu_seqlen = torch.tensor(
            [sum(lens[:i]) for i in range(len(lens))], device=device, dtype=torch.int32
        )
        use_initial = torch.zeros(len(lens), device=device, dtype=torch.bool)
        batch_info = torch.tensor([len(lens), total, 0], device=device, dtype=torch.int32)

        y_cached = torch.ops.auto_deploy.torch_cached_mamba_v1(
            hs,
            A,
            B,
            C,
            D,
            dt,
            batch_info,
            seq_len_t,
            cu_seqlen,
            slot_idx,
            use_initial,
            ssm_cache,
        )

        # Reference: run uncached op per-segment
        y_ref = torch.empty_like(hs)
        offset = 0
        for i, ln in enumerate(lens):
            hs_seg = hs[:, offset : offset + ln, :]
            B_seg = B[:, offset : offset + ln, :]
            C_seg = C[:, offset : offset + ln, :]
            dt_seg = dt[:, offset : offset + ln, :]
            y_seg = torch.ops.auto_deploy.torch_mamba_v1_selective_scan(
                hs_seg, A, B_seg, C_seg, D, dt_seg
            )
            y_ref[:, offset : offset + ln, :] = y_seg
            offset += ln

        torch.testing.assert_close(y_cached, y_ref, atol=env["atol"], rtol=env["rtol"])

    def test_prefill_stores_correct_final_state(self, env):
        """After prefill, the cached SSM state matches the state from running the reference scan."""
        device, dtype = env["device"], env["dtype"]
        d_inner, d_state = 16, 4
        batch, seq_len = 1, 10

        hs, A, B, C, D, dt = _random_v1_params(device, dtype, batch, seq_len, d_inner, d_state)

        # Compute reference final state using _mamba_v1_prefill directly
        from tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_mamba_v1 import (
            _mamba_v1_prefill,
        )

        _, ref_state = _mamba_v1_prefill(hs, A, B, C, D, dt)
        # ref_state: [1, D_inner, d_state]

        # Run cached op
        max_batch_size = 4
        ssm_cache = torch.zeros(max_batch_size, d_inner, 1, d_state, device=device, dtype=dtype)
        slot_idx = torch.tensor([2], device=device, dtype=torch.int32)
        seq_len_t = torch.tensor([seq_len], device=device, dtype=torch.int32)
        cu_seqlen = torch.tensor([0], device=device, dtype=torch.int32)
        use_initial = torch.zeros(1, device=device, dtype=torch.bool)
        batch_info = torch.tensor([1, seq_len, 0], device=device, dtype=torch.int32)

        torch.ops.auto_deploy.torch_cached_mamba_v1(
            hs,
            A,
            B,
            C,
            D,
            dt,
            batch_info,
            seq_len_t,
            cu_seqlen,
            slot_idx,
            use_initial,
            ssm_cache,
        )

        # Check cached state at slot 2 (stored in bfloat16, so allow wider tolerance)
        cached_state = ssm_cache[2].squeeze(1)  # [D_inner, d_state]
        torch.testing.assert_close(cached_state.float(), ref_state[0].float(), atol=0.05, rtol=0.05)


class TestDecodePath:
    """Test that the cached op's decode path produces correct single-step outputs."""

    def test_decode_matches_reference(self, env):
        """Single decode step with known state matches _mamba_v1_decode reference."""
        device, dtype = env["device"], env["dtype"]
        batch, d_inner, d_state = 3, 16, 4

        # Single-token inputs: [batch, 1, D]
        hs, A, B, C, D, dt = _random_v1_params(device, dtype, batch, 1, d_inner, d_state)

        # Pre-populate cache with known state
        max_batch_size = 6
        ssm_cache = torch.zeros(max_batch_size, d_inner, 1, d_state, device=device, dtype=dtype)
        slot_idx = torch.tensor([4, 1, 3], device=device, dtype=torch.int32)

        # Fill initial states
        init_states = torch.randn(batch, d_inner, d_state, device=device, dtype=torch.float32)
        for i in range(batch):
            ssm_cache[slot_idx[i].item(), :, 0, :] = init_states[i].to(dtype)

        # Reference decode
        from tensorrt_llm._torch.auto_deploy.custom_ops.mamba.torch_backend_mamba_v1 import (
            _mamba_v1_decode,
        )

        y_ref, state_ref = _mamba_v1_decode(
            hs.squeeze(1), A, B.squeeze(1), C.squeeze(1), D, dt.squeeze(1), init_states
        )

        # Cached op in decode mode
        seq_len_t = torch.ones(batch, device=device, dtype=torch.int32)
        cu_seqlen = torch.zeros(batch, device=device, dtype=torch.int32)
        use_initial = torch.zeros(batch, device=device, dtype=torch.bool)
        batch_info = torch.tensor([0, 0, batch], device=device, dtype=torch.int32)

        y_cached = torch.ops.auto_deploy.torch_cached_mamba_v1(
            hs,
            A,
            B,
            C,
            D,
            dt,
            batch_info,
            seq_len_t,
            cu_seqlen,
            slot_idx,
            use_initial,
            ssm_cache,
        )

        torch.testing.assert_close(y_cached.squeeze(1), y_ref, atol=env["atol"], rtol=env["rtol"])

        # Verify updated state in cache (stored in bfloat16, so allow wider tolerance)
        for i in range(batch):
            cached_i = ssm_cache[slot_idx[i].item(), :, 0, :].float()
            torch.testing.assert_close(cached_i, state_ref[i].float(), atol=0.05, rtol=0.05)


class TestPrefillThenDecode:
    """Critical test: prefill a sequence, then decode token-by-token.

    The concatenated output (prefill + decode tokens) should match running
    the uncached reference on the full sequence at once.
    """

    def test_prefill_then_decode_matches_full_reference(self, env):
        """Prefill N tokens, then decode M tokens one-by-one.

        Compare against running the uncached reference on all N+M tokens.
        """
        device, dtype = env["device"], env["dtype"]
        d_inner, d_state = 32, 8
        prefill_len = 10
        decode_steps = 5
        total_len = prefill_len + decode_steps

        # Generate full sequence parameters
        hs_full, A, B_full, C_full, D, dt_full = _random_v1_params(
            device, dtype, 1, total_len, d_inner, d_state
        )

        # === Reference: run uncached op on full sequence ===
        y_ref_full = torch.ops.auto_deploy.torch_mamba_v1_selective_scan(
            hs_full, A, B_full, C_full, D, dt_full
        )

        # === Cached: prefill first N tokens, then decode M tokens ===
        max_batch_size = 4
        ssm_cache = torch.zeros(max_batch_size, d_inner, 1, d_state, device=device, dtype=dtype)
        slot_idx = torch.tensor([1], device=device, dtype=torch.int32)

        # Step 1: Prefill
        hs_pf = hs_full[:, :prefill_len, :]
        B_pf = B_full[:, :prefill_len, :]
        C_pf = C_full[:, :prefill_len, :]
        dt_pf = dt_full[:, :prefill_len, :]

        seq_len_t = torch.tensor([prefill_len], device=device, dtype=torch.int32)
        cu_seqlen = torch.tensor([0], device=device, dtype=torch.int32)
        use_initial = torch.zeros(1, device=device, dtype=torch.bool)
        batch_info = torch.tensor([1, prefill_len, 0], device=device, dtype=torch.int32)

        y_prefill = torch.ops.auto_deploy.torch_cached_mamba_v1(
            hs_pf,
            A,
            B_pf,
            C_pf,
            D,
            dt_pf,
            batch_info,
            seq_len_t,
            cu_seqlen,
            slot_idx,
            use_initial,
            ssm_cache,
        )

        # Verify prefill output matches reference
        torch.testing.assert_close(
            y_prefill,
            y_ref_full[:, :prefill_len, :],
            atol=env["atol"],
            rtol=env["rtol"],
        )

        # Step 2: Decode token-by-token
        decode_outputs = []
        batch_info_decode = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)
        seq_len_decode = torch.tensor([1], device=device, dtype=torch.int32)
        cu_seqlen_decode = torch.tensor([0], device=device, dtype=torch.int32)
        use_initial_decode = torch.zeros(1, device=device, dtype=torch.bool)

        for t in range(decode_steps):
            idx = prefill_len + t
            hs_t = hs_full[:, idx : idx + 1, :]  # [1, 1, D]
            B_t = B_full[:, idx : idx + 1, :]  # [1, 1, N]
            C_t = C_full[:, idx : idx + 1, :]  # [1, 1, N]
            dt_t = dt_full[:, idx : idx + 1, :]  # [1, 1, D]

            y_t = torch.ops.auto_deploy.torch_cached_mamba_v1(
                hs_t,
                A,
                B_t,
                C_t,
                D,
                dt_t,
                batch_info_decode,
                seq_len_decode,
                cu_seqlen_decode,
                slot_idx,
                use_initial_decode,
                ssm_cache,
            )
            decode_outputs.append(y_t)  # [1, 1, D]

        y_decode = torch.cat(decode_outputs, dim=1)  # [1, decode_steps, D]

        # Compare decode outputs against reference
        # Wider tolerance: bfloat16 state accumulation drifts over decode steps
        y_ref_decode = y_ref_full[:, prefill_len:, :]
        torch.testing.assert_close(
            y_decode,
            y_ref_decode,
            atol=0.05,
            rtol=0.05,
        )

    def test_multi_batch_prefill_then_decode(self, env):
        """Multi-batch prefill (flattened) then multi-batch decode."""
        device, dtype = env["device"], env["dtype"]
        d_inner, d_state = 16, 4
        lens = [6, 4]
        decode_steps = 3
        total_lens = [ln + decode_steps for ln in lens]

        # Generate full params per sequence (we'll use batch=1 and manually split)
        all_hs, all_B, all_C, all_dt = [], [], [], []
        A = -torch.exp(torch.randn(d_inner, d_state, device=device, dtype=torch.float32))
        D = torch.ones(d_inner, device=device, dtype=torch.float32)

        for tl in total_lens:
            hs_i = torch.randn(1, tl, d_inner, device=device, dtype=dtype)
            B_i = torch.randn(1, tl, d_state, device=device, dtype=dtype)
            C_i = torch.randn(1, tl, d_state, device=device, dtype=dtype)
            dt_i = torch.nn.functional.softplus(
                torch.randn(1, tl, d_inner, device=device, dtype=dtype)
            )
            all_hs.append(hs_i)
            all_B.append(B_i)
            all_C.append(C_i)
            all_dt.append(dt_i)

        # Reference: run uncached op per-sequence on full length
        y_refs = []
        for i in range(len(lens)):
            y_ref_i = torch.ops.auto_deploy.torch_mamba_v1_selective_scan(
                all_hs[i], A, all_B[i], all_C[i], D, all_dt[i]
            )
            y_refs.append(y_ref_i)

        # === Cached: prefill ===
        max_batch_size = 8
        ssm_cache = torch.zeros(max_batch_size, d_inner, 1, d_state, device=device, dtype=dtype)
        slot_idx = torch.tensor([3, 6], device=device, dtype=torch.int32)

        # Flatten prefill tokens
        total_pf = sum(lens)
        hs_flat = torch.cat([all_hs[i][:, : lens[i], :] for i in range(len(lens))], dim=1)
        B_flat = torch.cat([all_B[i][:, : lens[i], :] for i in range(len(lens))], dim=1)
        C_flat = torch.cat([all_C[i][:, : lens[i], :] for i in range(len(lens))], dim=1)
        dt_flat = torch.cat([all_dt[i][:, : lens[i], :] for i in range(len(lens))], dim=1)

        seq_len_t = torch.tensor(lens, device=device, dtype=torch.int32)
        cu_seqlen = torch.tensor(
            [sum(lens[:j]) for j in range(len(lens))], device=device, dtype=torch.int32
        )
        use_initial = torch.zeros(len(lens), device=device, dtype=torch.bool)
        batch_info = torch.tensor([len(lens), total_pf, 0], device=device, dtype=torch.int32)

        y_prefill = torch.ops.auto_deploy.torch_cached_mamba_v1(
            hs_flat,
            A,
            B_flat,
            C_flat,
            D,
            dt_flat,
            batch_info,
            seq_len_t,
            cu_seqlen,
            slot_idx,
            use_initial,
            ssm_cache,
        )

        # Verify prefill outputs
        offset = 0
        for i, ln in enumerate(lens):
            torch.testing.assert_close(
                y_prefill[:, offset : offset + ln, :],
                y_refs[i][:, :ln, :],
                atol=env["atol"],
                rtol=env["rtol"],
            )
            offset += ln

        # === Decode token-by-token (batch of 2) ===
        num_seqs = len(lens)
        batch_info_decode = torch.tensor([0, 0, num_seqs], device=device, dtype=torch.int32)
        seq_len_decode = torch.ones(num_seqs, device=device, dtype=torch.int32)
        cu_seqlen_decode = torch.zeros(num_seqs, device=device, dtype=torch.int32)
        use_initial_decode = torch.zeros(num_seqs, device=device, dtype=torch.bool)

        for t in range(decode_steps):
            # Batch decode: stack single tokens from each sequence
            hs_batch = torch.cat(
                [all_hs[i][:, lens[i] + t : lens[i] + t + 1, :] for i in range(num_seqs)],
                dim=0,
            )  # [num_seqs, 1, D]
            B_batch = torch.cat(
                [all_B[i][:, lens[i] + t : lens[i] + t + 1, :] for i in range(num_seqs)],
                dim=0,
            )
            C_batch = torch.cat(
                [all_C[i][:, lens[i] + t : lens[i] + t + 1, :] for i in range(num_seqs)],
                dim=0,
            )
            dt_batch = torch.cat(
                [all_dt[i][:, lens[i] + t : lens[i] + t + 1, :] for i in range(num_seqs)],
                dim=0,
            )

            y_t = torch.ops.auto_deploy.torch_cached_mamba_v1(
                hs_batch,
                A,
                B_batch,
                C_batch,
                D,
                dt_batch,
                batch_info_decode,
                seq_len_decode,
                cu_seqlen_decode,
                slot_idx,
                use_initial_decode,
                ssm_cache,
            )

            # Compare each sequence's decode output
            for i in range(num_seqs):
                ref_idx = lens[i] + t
                torch.testing.assert_close(
                    y_t[i : i + 1],
                    y_refs[i][:, ref_idx : ref_idx + 1, :],
                    atol=env["atol"],
                    rtol=env["rtol"],
                )
