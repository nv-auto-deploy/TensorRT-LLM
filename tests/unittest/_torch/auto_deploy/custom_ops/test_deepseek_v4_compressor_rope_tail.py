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

"""Unit tests for the DeepSeek-V4 compressor rope/quant tail fusion (idea_0014).

``_apply_compressed_rope_and_quantize`` (rotate=False, the main-compressor tail) was
rewritten to fp8-quantize the nope slice and then route the interleaved RoPE + concat
through the fused ``auto_deploy::deepseek_v4_fused_rope_concat`` op, dropping the eager
rope's elementwise muls + stack, the redundant intermediate ``cat``+re-``split``, and
the final ``cat``. These tests pin the rewrite against the *original* eager expression:

* rotate=False: the fp8(nope) half is byte-identical; the rope(pe) half differs only by
  the fused kernel's FMA folding (<=1 ULP), so we assert close, not equal.
* rotate=True (indexer RoPE->Hadamard): unchanged, so byte-identical (torch.equal).
"""

import pytest
import torch

# Side-effect import: registers auto_deploy::deepseek_v4_{fused_rope_concat,compress_pool,
# hadamard_fp4} and exposes the helper under test.
from tensorrt_llm._torch.auto_deploy.custom_ops.attention import deepseek_v4_sparse_attention as dsa
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fake_fp8_act_quant

DEV = "cuda"


def _old_tail(
    compressed: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    rotate: bool,
) -> torch.Tensor:
    """Verbatim copy of the pre-idea_0014 eager implementation."""
    nope_dim = compressed.shape[-1] - rope_dim
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    pe = dsa._apply_interleaved_rope_ref(pe, cos, sin)
    compressed = torch.cat((nope, pe), dim=-1)
    if rotate:
        return torch.ops.auto_deploy.deepseek_v4_hadamard_fp4(compressed, 32)
    nope, pe = torch.split(compressed, [nope_dim, rope_dim], dim=-1)
    nope = fake_fp8_act_quant(nope, block_size=64)
    return torch.cat((nope, pe), dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,rope_dim",
    [
        ((2, 512), 64),  # main-compressor decode row: [B, head_dim], nope=448 (7*64)
        ((1, 512), 64),  # batch-1 decode
        ((1, 257, 512), 64),  # context build: [B, max_compressed_len, head_dim]
        ((3, 40, 512), 64),  # batched context
        ((2, 130), 64),  # nope_dim=66 not %64 -> fake_fp8 is a pass-through
    ],
)
def test_rotate_false_matches_eager(shape, rope_dim):
    """rotate=False fused tail == cat((fake_fp8(nope), rope(pe))) to ~1 ULP."""
    torch.manual_seed(0)
    compressed = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    head_dim = shape[-1]
    dh = rope_dim // 2
    cos = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)
    sin = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)

    out = dsa._apply_compressed_rope_and_quantize(compressed, cos, sin, rope_dim, rotate=False)
    ref = _old_tail(compressed, cos, sin, rope_dim, rotate=False)

    assert out.shape == ref.shape == (*shape[:-1], head_dim)
    assert out.dtype == ref.dtype == torch.bfloat16

    nope_dim = head_dim - rope_dim
    # fp8(nope) half is computed identically (op order unchanged) -> byte-exact.
    torch.testing.assert_close(out[..., :nope_dim], ref[..., :nope_dim], atol=0.0, rtol=0.0)
    # rope(pe) half: fused FMA vs eager mul/sub -> <=1 ULP.
    torch.testing.assert_close(out[..., nope_dim:], ref[..., nope_dim:], atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(2, 512), (1, 512), (1, 257, 512), (3, 40, 512)])
def test_compressor_rms_norm_byte_identical(shape):
    """The main-compressor RMSNorm routed through triton_rms_norm must be byte-identical
    to the eager _rms_norm_ref for the head_dim=512 compressor shapes."""
    torch.manual_seed(2)
    head_dim = shape[-1]
    x = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(head_dim, device=DEV, dtype=torch.bfloat16)
    eps = 1e-6
    out = dsa._compressor_rms_norm(x, w, eps)
    ref = dsa._rms_norm_ref(x, w, eps)
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressor_rms_norm_falls_back_no_weight():
    """No-weight / mismatched-weight cases must fall back to the eager reference."""
    x = torch.randn(2, 512, device=DEV, dtype=torch.bfloat16)
    empty = torch.empty(0, device=DEV, dtype=torch.bfloat16)
    out = dsa._compressor_rms_norm(x, empty, 1e-6)
    ref = dsa._rms_norm_ref(x, empty, 1e-6)
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "shape,rope_dim",
    [
        ((4, 128), 64),  # indexer decode row (ratio-4 compressor, index_head_dim)
        ((1, 5, 128), 64),  # indexer context rows
    ],
)
def test_rotate_true_byte_identical(shape, rope_dim):
    """rotate=True (RoPE->Hadamard) path is untouched -> bit-for-bit identical."""
    torch.manual_seed(1)
    compressed = torch.randn(*shape, device=DEV, dtype=torch.bfloat16)
    dh = rope_dim // 2
    cos = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)
    sin = torch.randn(*shape[:-1], dh, device=DEV, dtype=torch.float32)

    out = dsa._apply_compressed_rope_and_quantize(compressed, cos, sin, rope_dim, rotate=True)
    ref = _old_tail(compressed, cos, sin, rope_dim, rotate=True)
    assert torch.equal(out, ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
