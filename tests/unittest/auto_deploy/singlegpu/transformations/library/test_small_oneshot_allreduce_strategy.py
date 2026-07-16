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
"""Unit tests for the small-message ONESHOT allreduce qualification.

Covers the static (transform-time) gate ``qualify_small_oneshot_allreduce`` /
``resolve_plain_allreduce_strategy`` and the per-call (runtime) numel gate
``resolve_oneshot_small_strategy``. CPU-only — no collectives are run here; the
multi-GPU numerics live in
tests/unittest/auto_deploy/multigpu/custom_ops/test_small_oneshot_allreduce.py.
"""

import torch
import torch.fx as fx

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  — registers auto_deploy ops
from tensorrt_llm._torch.auto_deploy.custom_ops.distributed.trtllm_dist import (
    ONESHOT_SMALL_STRATEGY,
    resolve_oneshot_small_strategy,
)
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    qualify_small_oneshot_allreduce,
    resolve_plain_allreduce_strategy,
)
from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig

TRTLLM_AR_OP = torch.ops.auto_deploy.trtllm_dist_all_reduce.default
TORCH_AR_OP = torch.ops.auto_deploy.torch_dist_all_reduce.default


def _dc(world_size=4, tp_size=4, strategy="NCCL"):
    return DistConfig(
        world_size=world_size,
        rank=0,
        tp_size=tp_size,
        moe_ep_size=tp_size,
        allreduce_strategy=strategy,
    )


def _node_with_val(val):
    """Build a graph node carrying *val* as its shape-prop meta."""
    g = fx.Graph()
    n = g.placeholder("x")
    if val is not None:
        n.meta["val"] = val
    return n


def test_qualified_grid_upgrades_to_oneshot_small():
    strategy = qualify_small_oneshot_allreduce(_dc(), TRTLLM_AR_OP, torch.bfloat16, 4096)
    assert strategy == ONESHOT_SMALL_STRATEGY


def test_explicit_non_nccl_strategy_is_preserved():
    for base in ("AUTO", "SYMM_MEM", "ONESHOT", "TWOSHOT"):
        strategy = qualify_small_oneshot_allreduce(
            _dc(strategy=base), TRTLLM_AR_OP, torch.bfloat16, 4096
        )
        assert strategy == base


def test_torch_backend_keeps_nccl():
    assert qualify_small_oneshot_allreduce(_dc(), TORCH_AR_OP, torch.bfloat16, 4096) == "NCCL"


def test_other_topologies_keep_nccl():
    assert (
        qualify_small_oneshot_allreduce(
            _dc(world_size=8, tp_size=8), TRTLLM_AR_OP, torch.bfloat16, 4096
        )
        == "NCCL"
    )
    assert (
        qualify_small_oneshot_allreduce(
            _dc(world_size=4, tp_size=2), TRTLLM_AR_OP, torch.bfloat16, 4096
        )
        == "NCCL"
    )


def test_other_dtype_or_hidden_keeps_nccl():
    assert qualify_small_oneshot_allreduce(_dc(), TRTLLM_AR_OP, torch.float16, 4096) == "NCCL"
    assert qualify_small_oneshot_allreduce(_dc(), TRTLLM_AR_OP, torch.bfloat16, 8192) == "NCCL"
    # non-int (e.g. symbolic) last dim cannot prove a static hidden size
    assert qualify_small_oneshot_allreduce(_dc(), TRTLLM_AR_OP, torch.bfloat16, None) == "NCCL"


def test_node_meta_resolution():
    qualified = _node_with_val(torch.empty(2, 1, 4096, dtype=torch.bfloat16, device="meta"))
    assert resolve_plain_allreduce_strategy(_dc(), qualified, TRTLLM_AR_OP) == (
        ONESHOT_SMALL_STRATEGY
    )

    wrong_hidden = _node_with_val(torch.empty(2, 1, 2048, dtype=torch.bfloat16, device="meta"))
    assert resolve_plain_allreduce_strategy(_dc(), wrong_hidden, TRTLLM_AR_OP) == "NCCL"

    wrong_dtype = _node_with_val(torch.empty(2, 1, 4096, dtype=torch.float32, device="meta"))
    assert resolve_plain_allreduce_strategy(_dc(), wrong_dtype, TRTLLM_AR_OP) == "NCCL"

    missing_meta = _node_with_val(None)
    assert resolve_plain_allreduce_strategy(_dc(), missing_meta, TRTLLM_AR_OP) == "NCCL"


def test_runtime_numel_gate():
    # one decode token at hidden 4096 → ONESHOT; anything larger → NCCL
    assert resolve_oneshot_small_strategy(4096) == "ONESHOT"
    assert resolve_oneshot_small_strategy(1) == "ONESHOT"
    assert resolve_oneshot_small_strategy(4097) == "NCCL"
    assert resolve_oneshot_small_strategy(2 * 4096) == "NCCL"  # batch-2 decode
    assert resolve_oneshot_small_strategy(512 * 4096) == "NCCL"  # prefill chunk
