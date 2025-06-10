"""Tests for basic graph sharding."""

from functools import partial
from typing import Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.transformations.library import column_row_shard
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

num_heads = 4
num_key_value_heads = 2


class MLP(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, 4 * in_features, bias=bias)
        self.linear2 = nn.Linear(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


class GQABlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = in_features // num_heads

        # Grouped-Query Attention projections
        self.q_proj = nn.Linear(
            in_features, num_heads * self.head_dim, bias=bias, dtype=torch.float16
        )
        self.k_proj = nn.Linear(
            in_features, num_key_value_heads * self.head_dim, bias=bias, dtype=torch.float16
        )
        self.v_proj = nn.Linear(
            in_features, num_key_value_heads * self.head_dim, bias=bias, dtype=torch.float16
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, out_features, bias=bias, dtype=torch.float16
        )

    #     # RoPE parameters
    #     self.rope_base = 10000
    #     self._init_rope()

    # def _init_rope(self):
    #     # Initialize RoPE frequency matrix
    #     inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
    #     self.register_buffer('inv_freq', inv_freq, persistent=False)

    # def _apply_rope(self, x, seq_len):
    #     # Apply Rotary Position Embedding
    #     batch_size, num_heads, seq_len, head_dim = x.shape

    #     # Create position indices
    #     position_ids = torch.arange(seq_len, device=x.device, dtype=torch.float16)

    #     # Compute frequencies
    #     freqs = torch.outer(position_ids, self.inv_freq)
    #     freqs = torch.cat([freqs, freqs], dim=-1)

    #     # Apply rotation
    #     cos_freqs = freqs.cos().view(1, 1, seq_len, head_dim)
    #     sin_freqs = freqs.sin().view(1, 1, seq_len, head_dim)

    #     # Split x into even and odd dimensions
    #     x1 = x[..., ::2]
    #     x2 = x[..., 1::2]

    #     # Apply RoPE rotation
    #     rotated_x = torch.cat([
    #         x1 * cos_freqs[..., ::2] - x2 * sin_freqs[..., 1::2],
    #         x1 * sin_freqs[..., ::2] + x2 * cos_freqs[..., 1::2]
    #     ], dim=-1)

    #     return rotated_x

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Apply RoPE to Q and K
        # q = self._apply_rope(q, seq_len)
        # k = self._apply_rope(k, seq_len)

        # Expand K and V to match Q's head count for Grouped-Query Attention
        num_groups = self.num_heads // self.num_key_value_heads
        k = k.repeat_interleave(num_groups, dim=1)
        v = v.repeat_interleave(num_groups, dim=1)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim**0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1).to(torch.float16)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


def _run_job(
    model_cls: nn.Module,
    dist_op_expected: str,
    bias: bool,
    rank: int,
    world_size: int,
) -> None:
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 32

    model = model_cls(num_features, num_features, bias=bias).to(device="cuda", dtype=torch.float16)
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    def _get_expected_num_params(num_p_og: int) -> int:
        num_update = 0
        if bias and dist_op_expected == "all_reduce":
            num_p_og -= num_features
            num_update = num_features * (rank == world_size - 1)

        num_params = num_p_og // world_size + num_update
        return num_params

    if model_cls == GQABlock:
        head_dim = num_features // num_heads
        min_local_size = head_dim
    else:
        min_local_size = 1

    def verify_local_weight_sizes(gm) -> bool:
        """Verify that all weight tensors have first dimension >= min_local_size after sharding."""
        for name, param in gm.named_parameters():
            # Only check parameters that have at least 1 dimension and are weight matrices
            if param.dim() >= 1 and "weight" in name:
                if param.shape[0] < min_local_size:
                    print(
                        f"Weight {name} has shape {param.shape}, dim {param.shape[0]} < min_local_size {min_local_size}"
                    )
                    return False
        return True

    # now run the test
    op_expected = getattr(torch.ops.dist, dist_op_expected)

    transform_func = partial(
        column_row_shard, rank=rank, world_size=world_size, min_local_shape=min_local_size
    )

    def combined_graph_check(gm) -> bool:
        # Check for expected distributed operations
        has_expected_dist_ops = any(is_op(n, op_expected) for n in gm.graph.nodes) == (
            world_size > 1
        )
        # Check weight size constraints
        weight_sizes_valid = verify_local_weight_sizes(gm)
        return has_expected_dist_ops and weight_sizes_valid

    run_test(
        model,
        x,
        transform=transform_func,
        check_transformed_graph=combined_graph_check,
        _get_expected_num_params=_get_expected_num_params,
    )


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "model_cls, dist_op_expected",
    (
        (MLP, "all_reduce"),
        (nn.Linear, "all_gather"),
        # GQABlock computes attention explicitly (no "attention nodes"), so
        # _insert_sharded_matmul heuristic will fail and it will fall back to
        # _simple_shard row-only strategy.
        (GQABlock, "all_gather"),
    ),
)
def test_sharding(model_cls: Type[nn.Module], dist_op_expected: str, bias: bool, device_count: int):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, model_cls, dist_op_expected, bias),
        size=device_count,
    )
