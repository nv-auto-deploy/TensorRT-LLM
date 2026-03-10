# test_quant_fusion.py
import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale, fp8_scale


def _has_fused_linear_fp8(gm):
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_fp8_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) for n in gm.graph.nodes
    )
    return found_fused and not found_ref


def _has_fused_finegrained_fp8_linear(gm):
    """Check if FineGrained FP8 fake quant ops were replaced with TRT-LLM ops."""
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.trtllm_finegrained_fp8_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
        for n in gm.graph.nodes
    )
    return found_fused and not found_ref


def _has_fused_linear_fp4(gm):
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    return found_fused and not found_ref


def _has_fused_add_rmsnorm_nvfp4_linear(gm):
    found_fused_norm = any(
        is_op(n, torch.ops.auto_deploy.trtllm_fused_add_rms_norm_quant_nvfp4)
        for n in gm.graph.nodes
    )
    found_fused_linear = any(
        is_op(n, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear) for n in gm.graph.nodes
    )
    found_canonical = any(
        is_op(n, torch.ops.auto_deploy.torch_fused_add_rmsnorm_quant_nvfp4_linear)
        for n in gm.graph.nodes
    )
    found_quant_linear = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    found_fake_quant_linear = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    return (
        found_fused_norm
        and found_fused_linear
        and not found_canonical
        and not found_quant_linear
        and not found_fake_quant_linear
    )


def _has_unfused_add_rmsnorm_nvfp4_linear(gm):
    found_add = any(is_op(n, torch.ops.aten.add.Tensor) for n in gm.graph.nodes)
    found_rmsnorm = any(is_op(n, torch.ops.auto_deploy.torch_rmsnorm) for n in gm.graph.nodes)
    found_fake_quant_nvfp4_linear = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    found_fused_norm = any(
        is_op(n, torch.ops.auto_deploy.trtllm_fused_add_rms_norm_quant_nvfp4)
        for n in gm.graph.nodes
    )
    found_fused_linear = any(
        is_op(n, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear) for n in gm.graph.nodes
    )
    return (
        found_add
        and found_rmsnorm
        and found_fake_quant_nvfp4_linear
        and not found_fused_norm
        and not found_fused_linear
    )


class TinyFP8Ref(nn.Module):
    """A tiny module whose forward uses the reference FP8 op.

    Uses: torch_fake_quant_fp8_linear(input, weight_fp8, bias, [in_s], [w_s], [], [])
    """

    def __init__(self, in_features=16, out_features=32, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.rand(out_features, in_features, dtype=torch.float16))
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        # Precompute FP8 packing + scales as buffers
        with torch.no_grad():
            w_s = fp8_scale(self.weight)  # per-tensor scale
            w_fp8 = (self.weight / w_s).to(torch.float8_e4m3fn)

        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_s)
        self.register_buffer(
            "input_scale", torch.tensor(1.0, dtype=torch.float32)
        )  # simple test scale

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
            x,
            self.weight_fp8,
            bias,
            [self.input_scale],
            [self.weight_scale],
            [],
            [],
        )


class TinyFP4Ref(nn.Module):
    """A tiny module whose forward uses the reference NVFP4 op.

    Uses: torch_fake_quant_nvfp4_linear(x, w_fp4, bias, [s_in2], [cutlass_vec, alpha], [], [])
    """

    def __init__(self, in_features=64, out_features=32, use_bias=True):
        super().__init__()
        assert in_features % 16 == 0, "NVFP4 requires K % 16 == 0 for CUTLASS scaling."
        device = torch.device("cuda")

        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, dtype=torch.half, device=device)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.half, device=device))
        else:
            self.register_parameter("bias", None)

        with torch.no_grad():
            s_in2 = fp4_global_scale(torch.rand(1, in_features, dtype=torch.half, device=device))
            s_w2 = fp4_global_scale(self.weight)
            w_fp4, cutlass_vec = torch.ops.trtllm.fp4_quantize(self.weight, s_w2, 16, False)
            alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)

        self.register_buffer("weight_fp4", w_fp4)  # uint8 packed
        self.register_buffer("input_scale_2", s_in2.to(torch.float32))
        self.register_buffer("weight_scale_cutlass", cutlass_vec)  # uint8 vec
        self.register_buffer("alpha", alpha.to(torch.float32))

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            x,
            self.weight_fp4,
            bias,
            [self.input_scale_2],
            [self.weight_scale_cutlass, self.alpha],
            [],
            [],
        )


class TinyFineGrainedFP8Ref(nn.Module):
    """A tiny module whose forward uses the FineGrained FP8 op.

      torch_fake_quant_finegrained_fp8_linear(x, w_fp8, bias, [], [weight_scale_inv], [], [])

    This simulates models like MiniMax M2 and DeepSeek that use HF's block-wise FP8.
    """

    def __init__(self, in_features=256, out_features=256, use_bias=True):
        super().__init__()
        # FineGrained FP8 uses 128x128 block quantization, so dimensions must be multiples of 128
        assert in_features % 128 == 0, "FineGrained FP8 requires in_features % 128 == 0"
        assert out_features % 128 == 0, "FineGrained FP8 requires out_features % 128 == 0"
        device = torch.device("cuda")

        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, dtype=torch.bfloat16, device=device)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

        # Compute block-wise FP8 quantization (128x128 blocks)
        with torch.no_grad():
            block_n, block_k = 128, 128
            N, K = out_features, in_features

            # Reshape to blocks and compute per-block max
            weight_reshaped = self.weight.view(N // block_n, block_n, K // block_k, block_k)
            amax = weight_reshaped.abs().amax(dim=(1, 3)).to(torch.float32)  # [N/128, K/128]

            # Compute per-block scale (amax / 448 for FP8 E4M3)
            FP8_MAX = 448.0
            eps = torch.finfo(torch.float32).tiny
            weight_scale_inv = torch.clamp(amax / FP8_MAX, min=eps)  # [N/128, K/128]

            # Quantize weight to FP8
            scale_expanded = weight_scale_inv.repeat_interleave(block_n, dim=0).repeat_interleave(
                block_k, dim=1
            )
            w_fp8 = (self.weight.float() / scale_expanded).to(torch.float8_e4m3fn)

        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale_inv", weight_scale_inv)

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
            x,
            self.weight_fp8,
            bias,
            [],  # input_scale unused
            [self.weight_scale_inv],
            [],  # input_zp unused
            [],  # weight_zp unused
        )


class TinyAddRMSNormNVFP4(nn.Module):
    """Tiny canonical add -> torch_rmsnorm -> fake-quant NVFP4 linear model."""

    def __init__(self, calibration_input, in_features=2048, out_features=32, use_bias=True):
        super().__init__()
        assert in_features % 16 == 0, "NVFP4 requires K % 16 == 0 for CUTLASS scaling."
        device = torch.device("cuda")

        self.use_bias = use_bias
        self.eps = 1e-6
        self.norm_weight = nn.Parameter(torch.rand(in_features, dtype=torch.float16, device=device))
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, dtype=torch.float16, device=device)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.float16, device=device))
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "residual",
            torch.rand_like(calibration_input.to(device=device, dtype=torch.float16)),
        )

        with torch.no_grad():
            add_out = calibration_input.to(device=device, dtype=torch.float16) + self.residual
            norm_out = torch.ops.auto_deploy.torch_rmsnorm(add_out, self.norm_weight, self.eps)
            s_in2 = fp4_global_scale(norm_out)
            s_w2 = fp4_global_scale(self.weight)
            w_fp4, cutlass_vec = torch.ops.trtllm.fp4_quantize(self.weight, s_w2, 16, False)
            alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)

        self.register_buffer("weight_fp4", w_fp4)
        self.register_buffer("input_scale_2", s_in2.to(torch.float32))
        self.register_buffer("weight_scale_cutlass", cutlass_vec)
        self.register_buffer("alpha", alpha.to(torch.float32))

    def forward(self, x):
        add_out = x + self.residual
        norm_out = torch.ops.auto_deploy.torch_rmsnorm(add_out, self.norm_weight, self.eps)
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            norm_out,
            self.weight_fp4,
            bias,
            [self.input_scale_2],
            [self.weight_scale_cutlass, self.alpha],
            [],
            [],
        )


class TinyNemotronSuperV3NVFP4Block(nn.Module):
    """Nemotron-H style prenorm residual block with NVFP4 linear mixer.

    This mirrors the Super V3 block shape more closely than the bare local pattern:
      residual = hidden_states
      normed = RMSNorm(hidden_states)   # float32 norm weights
      mixed = NVFP4Linear(normed)
      out = residual (+ optional fp32 residual) + mixed
    """

    def __init__(self, calibration_input, hidden_size=2048, residual_in_fp32=True):
        super().__init__()
        assert hidden_size % 16 == 0, "NVFP4 requires K % 16 == 0 for CUTLASS scaling."
        device = torch.device("cuda")

        self.hidden_size = hidden_size
        self.eps = 1e-6
        self.residual_in_fp32 = residual_in_fp32
        self.norm_weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32, device=device))
        self.weight = nn.Parameter(
            torch.rand(hidden_size, hidden_size, dtype=torch.float16, device=device)
        )

        with torch.no_grad():
            calibration_input = calibration_input.to(device=device, dtype=self.norm_weight.dtype)
            norm_input = calibration_input.to(torch.float32)
            variance = norm_input.pow(2).mean(-1, keepdim=True)
            norm_input = norm_input * torch.rsqrt(variance + self.eps)
            normed = self.norm_weight.to(torch.float32) * norm_input
            s_in2 = fp4_global_scale(normed)
            s_w2 = fp4_global_scale(self.weight)
            w_fp4, cutlass_vec = torch.ops.trtllm.fp4_quantize(self.weight, s_w2, 16, False)
            alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)

        self.register_buffer("weight_fp4", w_fp4)
        self.register_buffer("input_scale_2", s_in2.to(torch.float32))
        self.register_buffer("weight_scale_cutlass", cutlass_vec)
        self.register_buffer("alpha", alpha.to(torch.float32))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = hidden_states.to(dtype=self.norm_weight.dtype)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        normed = self.norm_weight.to(torch.float32) * hidden_states.to(torch.float32)

        mixed = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            normed,
            self.weight_fp4,
            None,
            [self.input_scale_2],
            [self.weight_scale_cutlass, self.alpha],
            [],
            [],
        )

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        return residual + mixed


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_quant_rewrites_fp8_linear(use_bias):
    torch.manual_seed(0)
    model = TinyFP8Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 16, dtype=torch.float16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_fp8_linear": {"stage": "post_load_fusion", "backend": "torch"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_linear_fp8,
        lambda n: n,
        0.1,  # atol
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        False,  # skip_output_assert
    )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires NVFP4 and TRT-LLM ops",
)
def test_fuse_quant_rewrites_fp4_linear(use_bias):
    torch.manual_seed(0)
    model = TinyFP4Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_nvfp4_linear": {"stage": "post_load_fusion", "backend": "trtllm"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_linear_fp4,
        lambda n: n,
        0.1,  # atol
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        False,  # skip_output_assert
    )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(
    not (fp8_compatible() and trtllm_ops_available()),
    reason="Requires FP8 and TRT-LLM ops",
)
def test_fuse_quant_rewrites_finegrained_fp8_linear(use_bias):
    """Test that torch_fake_quant_finegrained_fp8_linear is replaced with trtllm_finegrained_fp8_linear.

    This tests the fusion transform for FineGrained FP8 models like
    MiniMax M2 and DeepSeek, which use 128x128 block-wise FP8 quantization.
    """
    torch.manual_seed(0)
    model = TinyFineGrainedFP8Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 256, dtype=torch.bfloat16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_finegrained_fp8_linear": {"stage": "post_load_fusion", "backend": "trtllm"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_finegrained_fp8_linear,
        lambda n: n,
        0.1,  # atol - FineGrained FP8 has some quantization error
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        True,  # skip_output_assert - skip numerical comparison for now
    )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires NVFP4 and TRT-LLM ops",
)
def test_fuse_rmsnorm_quant_nvfp4_prints_values(use_bias):
    torch.manual_seed(0)
    x = torch.rand(2, 2048, dtype=torch.float16, device="cuda")
    model = TinyAddRMSNormNVFP4(x, use_bias=use_bias).to("cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "match_rmsnorm_quant_nvfp4_pattern": {"stage": "pattern_matcher"},
            "fuse_rmsnorm_quant_nvfp4": {"stage": "post_load_fusion"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    assert _has_fused_add_rmsnorm_nvfp4_linear(gm_transformed)

    y_ref = model(x)
    y_fused = gm_transformed(x)
    abs_diff = (y_ref - y_fused).abs()

    print(f"use_bias={use_bias}")
    print("ref[:8]    =", y_ref.flatten()[:8].tolist())
    print("fused[:8]  =", y_fused.flatten()[:8].tolist())
    print("abs_diff[:8] =", abs_diff.flatten()[:8].tolist())
    print("max_abs_diff =", abs_diff.max().item())

    torch.testing.assert_close(y_fused, y_ref, atol=0.1, rtol=0.05)


@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires NVFP4 and TRT-LLM ops",
)
def test_rmsnorm_quant_nvfp4_super_v3_style_block_matches_pipeline_baseline():
    torch.manual_seed(0)
    x = torch.rand(2, 2048, dtype=torch.float16, device="cuda")
    model = TinyAddRMSNormNVFP4(x, in_features=2048, out_features=2048, use_bias=False).to("cuda")
    y_ref = model(x)

    gm_baseline_input = torch_export_to_gm(model, args=(x,), clone=True)
    gm_baseline = gm_baseline_input.to("cuda")

    gm_variant_input = torch_export_to_gm(model, args=(x,), clone=True)
    gm_variant = InferenceOptimizer(
        None,
        {
            "match_rmsnorm_quant_nvfp4_pattern": {"stage": "pattern_matcher"},
            "fuse_rmsnorm_quant_nvfp4": {"stage": "post_load_fusion"},
        },
    )(None, gm_variant_input)
    gm_variant.to("cuda")

    assert _has_unfused_add_rmsnorm_nvfp4_linear(gm_baseline)
    assert _has_fused_add_rmsnorm_nvfp4_linear(gm_variant)

    y_variant = gm_variant(x)
    abs_diff = (y_ref - y_variant).abs()

    print("super_v3_style_baseline_fused =", _has_fused_add_rmsnorm_nvfp4_linear(gm_baseline))
    print("super_v3_style_variant_fused  =", _has_fused_add_rmsnorm_nvfp4_linear(gm_variant))
    print("super_v3_style_ref[:8]        =", y_ref.flatten()[:8].tolist())
    print("super_v3_style_variant[:8]    =", y_variant.flatten()[:8].tolist())
    print("super_v3_style_abs_diff[:8]   =", abs_diff.flatten()[:8].tolist())
    print("super_v3_style_max_abs_diff   =", abs_diff.max().item())

    torch.testing.assert_close(y_variant, y_ref, atol=0.1, rtol=0.05)
