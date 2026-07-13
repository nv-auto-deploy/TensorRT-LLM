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

"""Tests for the FuseHcSeamPair graph transform (DeepSeek-V4 HC seam pair)."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops import (
    deepseek_v4_hc_post as _hc_post_ops,  # noqa: F401 (registers ops)
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

HM, H, MIX = 4, 256, 24
SCALARS = (HM, 20, 1e-4, 1e-6, 1e-6, torch.bfloat16)


class HcSeamChainModel(torch.nn.Module):
    """Two fusible seam pairs (base + _y32) followed by the head pair (not fusible)."""

    def __init__(self, expose_partials: bool = False):
        super().__init__()
        self.expose_partials = expose_partials
        D = HM * H
        for i in (1, 2):
            setattr(self, f"fn{i}", torch.nn.Parameter(0.02 * torch.randn(MIX, D)))
            setattr(self, f"scale{i}", torch.nn.Parameter(torch.randn(3)))
            setattr(self, f"base{i}", torch.nn.Parameter(0.02 * torch.randn(MIX)))
            setattr(self, f"w{i}", torch.nn.Parameter(1.0 + 0.05 * torch.randn(H)))
        self.head_fn = torch.nn.Parameter(0.02 * torch.randn(HM, D))
        self.head_scale = torch.nn.Parameter(torch.ones(1))
        self.head_base = torch.nn.Parameter(torch.zeros(HM))

    def forward(self, x, residual, post0, comb0):
        out1, parts1 = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
            x, residual, post0, comb0, self.fn1
        )
        y1, post1, comb1 = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials(
            parts1, out1.flatten(2), self.fn1, self.scale1, self.base1, self.w1, *SCALARS
        )
        out2, parts2 = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
            y1, out1, post1, comb1, self.fn2
        )
        y2, y32, post2, comb2 = torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32(
            parts2, out2.flatten(2), self.fn2, self.scale2, self.base2, self.w2, *SCALARS
        )
        out3, parts3 = torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials(
            y2, out2, post2, comb2, self.head_fn
        )
        head = torch.ops.auto_deploy.deepseek_v4_hc_head_norm(
            parts3,
            out3.flatten(2),
            self.head_fn,
            self.head_scale,
            self.head_base,
            self.w1,
            1e-4,
            1e-6,
            1e-6,
        )
        if self.expose_partials:
            # A second consumer of the first pair's partials blocks its fusion.
            return head + y32, parts1
        return head + y32


def _count(gm, packet):
    return sum(is_op(n, packet) for n in gm.graph.nodes)


def _export_and_transform(model, inputs):
    gm = torch_export_to_gm(model, args=inputs, clone=True)
    gm_t = InferenceOptimizer(None, {"fuse_hc_seam_pair": {"stage": "post_load_fusion"}})(None, gm)
    return gm_t


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fuse_hc_seam_pair_rewrites_pairs():
    torch.manual_seed(0)
    model = HcSeamChainModel().cuda()
    B, S = 1, 1
    x = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(B, S, HM, H, device="cuda", dtype=torch.bfloat16)
    post0 = 2.0 * torch.sigmoid(torch.randn(B, S, HM, device="cuda", dtype=torch.float32))
    comb0 = torch.randn(B, S, HM, HM, device="cuda", dtype=torch.float32).softmax(dim=-1)
    inputs = (x, residual, post0, comb0)

    ref = model(*inputs)
    gm_t = _export_and_transform(model, inputs)

    # Two pairs fused (one base, one _y32); the head producer + consumer survive.
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_post_pre_combine) == 1
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_post_pre_combine_y32) == 1
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials) == 1
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials) == 0
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials_y32) == 0
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_head_norm) == 1

    # Numerics: the merged kernel carries the standing ~1-2 fp32 ULP seam
    # contract vs the pair (bf16-rounded head/y32 flips stay at the LSB).
    got = gm_t(*inputs)
    torch.testing.assert_close(got, ref, rtol=1.6e-2, atol=8e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fuse_hc_seam_pair_skips_multi_consumer_partials():
    torch.manual_seed(1)
    model = HcSeamChainModel(expose_partials=True).cuda()
    B, S = 1, 1
    x = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(B, S, HM, H, device="cuda", dtype=torch.bfloat16)
    post0 = 2.0 * torch.sigmoid(torch.randn(B, S, HM, device="cuda", dtype=torch.float32))
    comb0 = torch.randn(B, S, HM, HM, device="cuda", dtype=torch.float32).softmax(dim=-1)
    inputs = (x, residual, post0, comb0)

    gm_t = _export_and_transform(model, inputs)

    # parts1 is also a graph output -> pair 1 must NOT fuse; pair 2 still does.
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_post_pre_combine) == 0
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_post_pre_combine_y32) == 1
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_post_next_partials) == 2
    assert _count(gm_t, torch.ops.auto_deploy.deepseek_v4_hc_pre_mix_combine_partials) == 1
