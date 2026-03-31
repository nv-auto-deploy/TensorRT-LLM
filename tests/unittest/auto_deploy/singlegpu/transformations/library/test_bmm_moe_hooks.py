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
"""Tests for BMM MoE checkpoint loading hooks."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.transform.library.fused_moe import (
    _bmm_down_to_stacked_hook,
    _bmm_gate_up_to_stacked_hook,
)


@pytest.fixture
def gate_up_stacked_weight():
    """Fixture for stacked gate_up weight in Llama4 format (E, H, 2*I)."""
    num_experts = 4
    hidden_size = 64
    intermediate_size = 32
    return torch.randn(num_experts, hidden_size, intermediate_size * 2)


@pytest.fixture
def down_stacked_weight():
    """Fixture for stacked down weight in Llama4 format (E, I, H)."""
    num_experts = 4
    hidden_size = 64
    intermediate_size = 32
    return torch.randn(num_experts, intermediate_size, hidden_size)


class TestBmmMoeGateUpToStackedHook:
    """Tests for _bmm_gate_up_to_stacked_hook."""

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size",
        [
            (4, 64, 32),
            (8, 128, 64),
            (2, 32, 16),
        ],
    )
    def test_splits_stacked_weights_into_w1_w3_stacked(
        self, num_experts, hidden_size, intermediate_size
    ):
        """Verify gate_up hook splits Llama4 (E,H,2I) into w1_stacked and w3_stacked (E,I,H)."""
        # Llama4 format: (E, H, 2*I)
        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        state_dict = {"gate_up_weight": stacked}

        _bmm_gate_up_to_stacked_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            w1_stacked_key="w1_stacked",
            w3_stacked_key="w3_stacked",
            intermediate_size=intermediate_size,
        )

        assert "w1_stacked" in state_dict
        assert "w3_stacked" in state_dict
        # After transpose: (E, I, H)
        assert state_dict["w1_stacked"].shape == (num_experts, intermediate_size, hidden_size)
        assert state_dict["w3_stacked"].shape == (num_experts, intermediate_size, hidden_size)

    def test_w1_w3_content_matches_original_stacked(self):
        """Verify split w1/w3 tensors match the original stacked content."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        state_dict = {"gate_up_weight": stacked.clone()}

        _bmm_gate_up_to_stacked_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            w1_stacked_key="w1_stacked",
            w3_stacked_key="w3_stacked",
            intermediate_size=intermediate_size,
        )

        # w1 is gate (first half): stacked[:, :, :I].transpose(1,2) → (E, I, H)
        expected_w1 = stacked[:, :, :intermediate_size].transpose(1, 2).contiguous()
        # w3 is up (second half): stacked[:, :, I:].transpose(1,2) → (E, I, H)
        expected_w3 = stacked[:, :, intermediate_size:].transpose(1, 2).contiguous()

        torch.testing.assert_close(state_dict["w1_stacked"], expected_w1)
        torch.testing.assert_close(state_dict["w3_stacked"], expected_w3)

    def test_handles_missing_source_key(self):
        """Verify hook does nothing when source key is missing."""
        state_dict = {}

        # Should not raise
        _bmm_gate_up_to_stacked_hook(
            state_dict,
            "",
            source_key="missing_key",
            w1_stacked_key="w1_stacked",
            w3_stacked_key="w3_stacked",
            intermediate_size=32,
        )

        assert len(state_dict) == 0

    @pytest.mark.parametrize("prefix", ["", "model.layers.0.moe."])
    def test_works_with_module_prefix(self, prefix):
        """Verify hook works correctly with module path prefix."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        state_dict = {f"{prefix}gate_up_weight": stacked}

        _bmm_gate_up_to_stacked_hook(
            state_dict,
            prefix,
            source_key="gate_up_weight",
            w1_stacked_key="w1_stacked",
            w3_stacked_key="w3_stacked",
            intermediate_size=intermediate_size,
        )

        assert f"{prefix}w1_stacked" in state_dict
        assert f"{prefix}w3_stacked" in state_dict

    def test_no_op_if_targets_already_present(self):
        """Verify hook is a no-op if both target keys already exist."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        existing_w1 = torch.zeros(num_experts, intermediate_size, hidden_size)
        existing_w3 = torch.zeros(num_experts, intermediate_size, hidden_size)
        state_dict = {
            "gate_up_weight": stacked,
            "w1_stacked": existing_w1,
            "w3_stacked": existing_w3,
        }

        _bmm_gate_up_to_stacked_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            w1_stacked_key="w1_stacked",
            w3_stacked_key="w3_stacked",
            intermediate_size=intermediate_size,
        )

        # Values should be unchanged
        torch.testing.assert_close(state_dict["w1_stacked"], existing_w1)
        torch.testing.assert_close(state_dict["w3_stacked"], existing_w3)


class TestBmmMoeDownToStackedHook:
    """Tests for _bmm_down_to_stacked_hook."""

    @pytest.mark.parametrize(
        "num_experts,hidden_size,intermediate_size",
        [
            (4, 64, 32),
            (8, 128, 64),
            (2, 32, 16),
        ],
    )
    def test_transposes_down_weight_to_w2_stacked(
        self, num_experts, hidden_size, intermediate_size
    ):
        """Verify down hook transposes Llama4 (E,I,H) to w2_stacked (E,H,I)."""
        # Llama4 format: (E, I, H)
        stacked = torch.randn(num_experts, intermediate_size, hidden_size)
        state_dict = {"down_weight": stacked}

        _bmm_down_to_stacked_hook(
            state_dict,
            "",
            source_key="down_weight",
            w2_stacked_key="w2_stacked",
        )

        assert "w2_stacked" in state_dict
        # After transpose: (E, H, I)
        assert state_dict["w2_stacked"].shape == (num_experts, hidden_size, intermediate_size)

    def test_w2_content_matches_original_stacked(self):
        """Verify w2_stacked matches transposed original stacked content."""
        num_experts = 2
        hidden_size = 32
        intermediate_size = 16

        stacked = torch.randn(num_experts, intermediate_size, hidden_size)
        state_dict = {"down_weight": stacked.clone()}

        _bmm_down_to_stacked_hook(
            state_dict,
            "",
            source_key="down_weight",
            w2_stacked_key="w2_stacked",
        )

        expected_w2 = stacked.transpose(1, 2).contiguous()
        torch.testing.assert_close(state_dict["w2_stacked"], expected_w2)

    def test_handles_missing_source_key(self):
        """Verify hook does nothing when source key is missing."""
        state_dict = {}

        _bmm_down_to_stacked_hook(
            state_dict,
            "",
            source_key="missing_key",
            w2_stacked_key="w2_stacked",
        )

        assert len(state_dict) == 0


class TestBmmMoeHooksIntegration:
    """Integration tests for BMM MoE hooks working together."""

    def test_full_checkpoint_loading_flow(self):
        """Test the full flow: gate_up + down → w1/w2/w3 stacked tensors."""
        num_experts = 4
        hidden_size = 64
        intermediate_size = 32

        # Simulate a checkpoint with Llama4-style stacked weights
        gate_up_stacked = torch.randn(num_experts, hidden_size, intermediate_size * 2)
        down_stacked = torch.randn(num_experts, intermediate_size, hidden_size)

        state_dict = {
            "gate_up_weight": gate_up_stacked.clone(),
            "down_weight": down_stacked.clone(),
        }

        # Step 1: Convert gate_up to w1_stacked and w3_stacked
        _bmm_gate_up_to_stacked_hook(
            state_dict,
            "",
            source_key="gate_up_weight",
            w1_stacked_key="w1_stacked",
            w3_stacked_key="w3_stacked",
            intermediate_size=intermediate_size,
        )

        # Step 2: Convert down to w2_stacked
        _bmm_down_to_stacked_hook(
            state_dict,
            "",
            source_key="down_weight",
            w2_stacked_key="w2_stacked",
        )

        # Verify: stacked weights present with correct shapes
        assert state_dict["w1_stacked"].shape == (num_experts, intermediate_size, hidden_size)
        assert state_dict["w2_stacked"].shape == (num_experts, hidden_size, intermediate_size)
        assert state_dict["w3_stacked"].shape == (num_experts, intermediate_size, hidden_size)
