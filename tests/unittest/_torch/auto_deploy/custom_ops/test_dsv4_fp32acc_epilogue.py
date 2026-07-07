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

"""Correctness tests for the scoped DeepSeek-V4 Q norm and grouped epilogue fold."""

import operator

import pytest
import torch

# Register the custom ops (side-effect imports).
import tensorrt_llm._torch.auto_deploy.custom_ops.deepseek_v4_rope  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.normalization.rms_norm  # noqa: F401
import tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
    torch_fake_quant_grouped_finegrained_fp8_linear,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

DEVICE = "cuda"
Q_LORA = 1024
KV_HEAD = 512
NOPE = 448
PE = 64
FUSED_OUT = Q_LORA + KV_HEAD
DSV4_FLASH_HIDDEN = 4096
EPS = 1e-6


def _rand_acc(shape: tuple[int, ...], seed: int, scale: float = 8.0) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=DEVICE, dtype=torch.float32) * scale


def _fp8_weight(n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 400)
    weight = torch.randn((n, k), generator=generator, device=DEVICE, dtype=torch.float32)
    weight_fp8 = weight.clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = (
        torch.rand(
            ((n + 127) // 128, (k + 127) // 128),
            generator=generator,
            device=DEVICE,
            dtype=torch.float32,
        )
        * 0.02
        + 0.005
    )
    return weight_fp8, scale


def _assert_equal_up_to_splitk_atomic_wiggle(
    out: torch.Tensor, ref: torch.Tensor, max_frac: float = 2e-3
) -> None:
    """Allow only the pre-existing cross-launch split-K atomic-order variation."""
    if torch.equal(out, ref):
        return
    diff = out != ref
    mismatch_fraction = diff.float().mean().item()
    assert mismatch_fraction <= max_frac, (
        f"mismatch fraction {mismatch_fraction} exceeds atomic-wiggle allowance"
    )
    actual = out[diff].float()
    expected = ref[diff].float()
    assert torch.allclose(actual, expected, rtol=1.6e-2, atol=1e-20), (
        "mismatches exceed BF16 rounding-boundary flips: "
        f"{actual[:4].tolist()} vs {expected[:4].tolist()}"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rows", [1, 37])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_deepseek_v4_q_rmsnorm_matches_bf16_reference(
    seed: int, rows: int, weight_dtype: torch.dtype
) -> None:
    """The specialized op keeps both sides of the public contract explicitly BF16."""
    full = _rand_acc((1, rows, FUSED_OUT), seed).to(torch.bfloat16)
    q = full.narrow(-1, 0, Q_LORA)
    generator = torch.Generator(device=DEVICE).manual_seed(seed + 100)
    weight = (
        torch.rand(Q_LORA, generator=generator, device=DEVICE, dtype=torch.float32) * 2 - 0.5
    ).to(weight_dtype)

    ref = torch.ops.auto_deploy.torch_rmsnorm(q, weight, EPS)
    out = torch.ops.auto_deploy.deepseek_v4_q_rmsnorm(q, weight, EPS)

    assert q.dtype == torch.bfloat16
    assert ref.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, ref)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.float32])
def test_deepseek_v4_q_rmsnorm_rejects_non_bf16_input(input_dtype: torch.dtype) -> None:
    q = torch.randn((1, 1, Q_LORA), device=DEVICE, dtype=input_dtype)
    weight = torch.ones(Q_LORA, device=DEVICE, dtype=torch.float32)
    with pytest.raises(TypeError, match="input must be bfloat16"):
        torch.ops.auto_deploy.deepseek_v4_q_rmsnorm(q, weight, EPS)


def test_deepseek_v4_q_rmsnorm_rejects_non_dsv4_shape() -> None:
    q = torch.randn((1, 1, Q_LORA // 2), device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(Q_LORA // 2, device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="1024"):
        torch.ops.auto_deploy.deepseek_v4_q_rmsnorm(q, weight, EPS)


def test_deepseek_v4_q_rmsnorm_rejects_unsupported_weight_dtype() -> None:
    q = torch.randn((1, 1, Q_LORA), device=DEVICE, dtype=torch.bfloat16)
    weight = torch.ones(Q_LORA, device=DEVICE, dtype=torch.float16)
    with pytest.raises(TypeError, match="weight must be bfloat16 or float32"):
        torch.ops.auto_deploy.deepseek_v4_q_rmsnorm(q, weight, EPS)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("tokens", [1, 2])
def test_grouped_decode_single_accumulator_matches_per_group_chain(seed: int, tokens: int) -> None:
    """The shared accumulator preserves the prior per-group split-K result."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
        _safe_act_quant,
        _w8a8_block_fp8_matmul_triton,
    )

    num_groups = 2
    rank = 1024
    input_tensor = _rand_acc((1, tokens, num_groups, DSV4_FLASH_HIDDEN), seed, scale=1.0).to(
        torch.bfloat16
    )
    weight_fp8, weight_scale = _fp8_weight(num_groups * rank, DSV4_FLASH_HIDDEN, seed)

    out = torch_fake_quant_grouped_finegrained_fp8_linear(
        input_tensor, weight_fp8, None, [], [weight_scale], [], []
    )

    qinput, input_scales = _safe_act_quant(input_tensor.contiguous(), 128, "")
    m_tokens = qinput.numel() // (num_groups * DSV4_FLASH_HIDDEN)
    grouped_input = qinput.reshape(m_tokens, num_groups, DSV4_FLASH_HIDDEN)
    grouped_input_scale = input_scales.reshape(m_tokens, num_groups, input_scales.shape[-1])
    grouped_weight = weight_fp8.view(num_groups, rank, DSV4_FLASH_HIDDEN)
    grouped_weight_scale = weight_scale.view(
        num_groups, weight_scale.shape[0] // num_groups, weight_scale.shape[1]
    )
    parts = [
        _w8a8_block_fp8_matmul_triton(
            grouped_input[:, group, :].contiguous(),
            grouped_weight[group].contiguous(),
            grouped_input_scale[:, group, :].contiguous(),
            grouped_weight_scale[group].contiguous(),
            [128, 128],
            output_dtype=input_tensor.dtype,
        )
        for group in range(num_groups)
    ]
    ref = torch.stack(parts, dim=1).reshape(1, tokens, num_groups * rank)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    _assert_equal_up_to_splitk_atomic_wiggle(out, ref)


def _make_splitk_inputs(
    tokens: int = 2, out_features: int = 256
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import _safe_act_quant

    input_tensor = _rand_acc((tokens, DSV4_FLASH_HIDDEN), 71, scale=1.0).to(torch.bfloat16)
    qinput, input_scale = _safe_act_quant(input_tensor, 128, "")
    weight_fp8, weight_scale = _fp8_weight(out_features, DSV4_FLASH_HIDDEN, 72)
    return qinput, weight_fp8, input_scale, weight_scale


def _run_splitk_with_accumulator(
    c_out: torch.Tensor,
    qinput_override: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import (
        _w8a8_block_fp8_matmul_splitk,
    )

    tokens = 2
    out_features = 256
    qinput, weight_fp8, input_scale, weight_scale = _make_splitk_inputs(tokens, out_features)
    if qinput_override is not None:
        qinput = qinput_override
    return _w8a8_block_fp8_matmul_splitk(
        qinput,
        weight_fp8,
        input_scale,
        weight_scale,
        128,
        128,
        output_dtype,
        tokens,
        out_features,
        DSV4_FLASH_HIDDEN,
        C_out=c_out,
    )


def test_splitk_c_out_accepts_nonoverlapping_strided_slice() -> None:
    tokens = 2
    out_features = 256
    backing = torch.zeros((tokens, 2 * out_features), device=DEVICE, dtype=torch.float32)
    c_out = backing[:, 37 : 37 + out_features]

    out = _run_splitk_with_accumulator(c_out)

    assert out.data_ptr() == c_out.data_ptr()
    assert out.shape == (tokens, out_features)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("shape", "shape"),
        ("device", "device"),
        ("stride", "stride"),
        ("overlap", "overlap"),
    ],
)
def test_splitk_c_out_rejects_unsafe_views(case: str, match: str) -> None:
    tokens = 2
    out_features = 256
    aliased_qinput = None
    if case == "shape":
        c_out = torch.zeros((tokens + 1, out_features), device=DEVICE, dtype=torch.float32)
    elif case == "device":
        c_out = torch.zeros((tokens, out_features), device="cpu", dtype=torch.float32)
    elif case == "stride":
        c_out = torch.zeros((out_features, tokens), device=DEVICE, dtype=torch.float32).transpose(
            0, 1
        )
    else:
        backing = torch.zeros((tokens, DSV4_FLASH_HIDDEN // 4), device=DEVICE, dtype=torch.float32)
        c_out = backing[:, :out_features]
        aliased_qinput = backing.view(torch.float8_e4m3fn).reshape(tokens, DSV4_FLASH_HIDDEN)

    with pytest.raises(ValueError, match=match):
        _run_splitk_with_accumulator(c_out, aliased_qinput)


def test_splitk_c_out_rejects_non_fp32_dtype() -> None:
    c_out = torch.zeros((2, 256), device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="float32"):
        _run_splitk_with_accumulator(c_out)


def test_splitk_c_out_rejects_finish_cast_output_dtype() -> None:
    c_out = torch.zeros((2, 256), device=DEVICE, dtype=torch.float32)
    with pytest.raises(ValueError, match="output_dtype=torch.float32"):
        _run_splitk_with_accumulator(c_out, output_dtype=torch.bfloat16)


def _build_q_norm_graph(
    *,
    prequant: bool = False,
    hidden_size: int = DSV4_FLASH_HIDDEN,
    q_start: int = 0,
    q_width: int = Q_LORA,
    nope_width: int = NOPE,
    producer_dtype: torch.dtype = torch.bfloat16,
    with_bias: bool = False,
    with_foreign_consumer: bool = False,
) -> torch.fx.GraphModule:
    """Build the exact fused DSV4 Q/KV projection and its two consumer branches."""
    from torch.fx import Graph, GraphModule

    class Holder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "weight_fp8",
                torch.empty((FUSED_OUT, hidden_size), device=DEVICE, dtype=torch.float8_e4m3fn),
            )
            self.register_buffer(
                "weight_scale",
                torch.ones(
                    ((FUSED_OUT + 127) // 128, (hidden_size + 127) // 128),
                    device=DEVICE,
                    dtype=torch.float32,
                ),
            )
            self.register_buffer(
                "q_weight", torch.ones(q_width, device=DEVICE, dtype=torch.float32)
            )
            self.register_buffer(
                "kv_weight", torch.ones(KV_HEAD, device=DEVICE, dtype=torch.float32)
            )
            if with_bias:
                self.register_buffer(
                    "bias", torch.zeros(FUSED_OUT, device=DEVICE, dtype=producer_dtype)
                )

    holder = Holder()
    graph = Graph()
    input_node = graph.placeholder("input")
    cos = graph.placeholder("cos")
    sin = graph.placeholder("sin")
    weight_fp8 = graph.get_attr("weight_fp8")
    weight_scale = graph.get_attr("weight_scale")
    q_weight = graph.get_attr("q_weight")
    kv_weight = graph.get_attr("kv_weight")
    bias = graph.get_attr("bias") if with_bias else None
    if prequant:
        input_scale = graph.placeholder("input_scale")
        linear = graph.call_function(
            torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear_prequant.default,
            args=(input_node, input_scale, weight_fp8, bias, [weight_scale]),
        )
    else:
        linear = graph.call_function(
            torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear.default,
            args=(input_node, weight_fp8, bias, [], [weight_scale], [], []),
        )

    q_narrow = graph.call_function(torch.narrow, args=(linear, -1, q_start, q_width))
    q_contiguous = graph.call_method("contiguous", args=(q_narrow,))
    q_out = graph.call_function(
        torch.ops.auto_deploy.torch_rmsnorm.default,
        args=(q_contiguous, q_weight, EPS),
    )

    kv_narrow = graph.call_function(torch.narrow, args=(linear, -1, Q_LORA, KV_HEAD))
    kv_contiguous = graph.call_method("contiguous", args=(kv_narrow,))
    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        args=(kv_contiguous, [nope_width, KV_HEAD - nope_width], -1),
    )
    nope = graph.call_function(operator.getitem, args=(split, 0))
    pe = graph.call_function(operator.getitem, args=(split, 1))
    kv_out = graph.call_function(
        torch.ops.auto_deploy.deepseek_v4_kv_norm_rope_concat.default,
        args=(nope, pe, kv_weight, cos, sin, EPS, 64),
    )
    outputs = [q_out, kv_out]
    if with_foreign_consumer:
        outputs.append(graph.call_function(torch.ops.aten.silu.default, args=(linear,)))
    graph.output(tuple(outputs))

    from torch._subclasses.fake_tensor import FakeTensorMode

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        linear.meta["val"] = torch.empty((1, 1, FUSED_OUT), dtype=producer_dtype, device=DEVICE)
        q_contiguous.meta["val"] = torch.empty((1, 1, q_width), dtype=producer_dtype, device=DEVICE)
        q_out.meta["val"] = torch.empty((1, 1, q_width), dtype=producer_dtype, device=DEVICE)
        nope.meta["val"] = torch.empty((1, 1, nope_width), dtype=producer_dtype, device=DEVICE)
        pe.meta["val"] = torch.empty(
            (1, 1, KV_HEAD - nope_width), dtype=producer_dtype, device=DEVICE
        )
        kv_out.meta["val"] = torch.empty((1, 1, KV_HEAD), dtype=producer_dtype, device=DEVICE)
    return GraphModule(holder, graph)


def _apply_q_norm_transform(
    graph_module: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, object]:
    from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
    from tensorrt_llm._torch.auto_deploy.transform.library.fuse_quant import FuseDeepSeekV4QRMSNorm

    transform = FuseDeepSeekV4QRMSNorm(config=TransformConfig(stage="post_load_fusion"))
    return transform._apply(graph_module, None, None, None)


@pytest.mark.parametrize("prequant", [False, True])
def test_fuse_deepseek_v4_q_rmsnorm_matches_exact_dsv4_graph(prequant: bool) -> None:
    """Both producer forms match at the exact DSV4-Flash [1536, 4096] shape."""
    graph_module = _build_q_norm_graph(prequant=prequant)
    new_graph_module, info = _apply_q_norm_transform(graph_module)

    targets = [str(node.target) for node in new_graph_module.graph.nodes]
    assert info.num_matches == 1
    assert sum("deepseek_v4_q_rmsnorm" in target for target in targets) == 1
    assert sum("deepseek_v4_kv_norm_rope_concat" in target for target in targets) == 1
    assert not any("fp32acc" in target for target in targets)
    if prequant:
        assert any(
            target.endswith("torch_fake_quant_finegrained_fp8_linear_prequant.default")
            for target in targets
        )
    else:
        assert any(
            target.endswith("torch_fake_quant_finegrained_fp8_linear.default") for target in targets
        )


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("wrong_hidden", {"hidden_size": 7168}),
        ("wrong_q_start", {"q_start": 1}),
        ("wrong_q_width", {"q_width": Q_LORA // 2}),
        ("wrong_kv_split", {"nope_width": NOPE - 64}),
        ("wrong_dtype", {"producer_dtype": torch.float16}),
        ("bias", {"with_bias": True}),
        ("foreign_consumer", {"with_foreign_consumer": True}),
    ],
)
def test_fuse_deepseek_v4_q_rmsnorm_rejects_non_dsv4_graph(
    case: str, kwargs: dict[str, object]
) -> None:
    del case
    graph_module = _build_q_norm_graph(**kwargs)
    new_graph_module, info = _apply_q_norm_transform(graph_module)

    targets = [str(node.target) for node in new_graph_module.graph.nodes]
    assert info.num_matches == 0
    assert not any("deepseek_v4_q_rmsnorm" in target for target in targets)
    assert any(target.endswith("torch_rmsnorm.default") for target in targets)


def test_grouped_decode_cuda_graph_replays_reset_shared_accumulator() -> None:
    """Each replay zeros the captured shared accumulator before split-K atomics."""
    num_groups = 2
    rank = 256
    tokens = 1
    weight_fp8, weight_scale = _fp8_weight(num_groups * rank, DSV4_FLASH_HIDDEN, 91)
    static_input = _rand_acc((1, tokens, num_groups, DSV4_FLASH_HIDDEN), 92, scale=1.0).to(
        torch.bfloat16
    )

    def run(input_tensor: torch.Tensor) -> torch.Tensor:
        return torch_fake_quant_grouped_finegrained_fp8_linear(
            input_tensor, weight_fp8, None, [], [weight_scale], [], []
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            run(static_input)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = run(static_input)

    replay_outputs = []
    for seed in (93, 94):
        fresh_input = _rand_acc((1, tokens, num_groups, DSV4_FLASH_HIDDEN), seed, scale=1.0).to(
            torch.bfloat16
        )
        ref = run(fresh_input)
        static_input.copy_(fresh_input)
        graph.replay()
        torch.cuda.synchronize()
        _assert_equal_up_to_splitk_atomic_wiggle(static_out, ref)
        replay_outputs.append(static_out.clone())

    assert not torch.equal(replay_outputs[0], replay_outputs[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
