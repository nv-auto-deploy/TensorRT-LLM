import os
import sys

import pytest
import torch
import torch.nn.functional as F
from _torch_test_utils import fp8_compatible

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.modules.fused_moe import FusedMoE  # noqa: F401

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
from helpers import reference_moe_torch


def setup_moe_test(dtype, num_experts):
    SEQ_LEN = 8
    HIDDEN_SIZE = 64
    INTERMEDIATE_SIZE = 32
    NUM_EXPERTS = num_experts
    TOP_K = 2

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5

    router_logits = torch.randn((SEQ_LEN, NUM_EXPERTS), dtype=torch.float32).cuda()
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    final_scales, selected_experts = torch.topk(routing_weights, TOP_K, dim=-1)
    final_scales = final_scales / final_scales.sum(dim=-1, keepdim=True)
    final_scales = final_scales.to(x.dtype)

    w1_weight, w2_weight, w3_weight = [], [], []
    weights = {}
    fused_w3_w1_stacked_weight = torch.empty(
        (NUM_EXPERTS, INTERMEDIATE_SIZE * 2, HIDDEN_SIZE), dtype=dtype
    ).cuda()
    fused_w2_weight = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda()

    for expert_id in range(NUM_EXPERTS):
        w1 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5
        w2 = torch.randn((HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype).cuda() * 0.5
        w3 = torch.randn((INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype).cuda() * 0.5

        weights[f"{expert_id}.w1.weight"] = w1
        weights[f"{expert_id}.w2.weight"] = w2
        weights[f"{expert_id}.w3.weight"] = w3

        w1_weight.append(w1)
        w2_weight.append(w2)
        w3_weight.append(w3)

        fused_w3_w1_stacked_weight.data[expert_id].copy_(torch.cat([w3, w1], dim=-2))
        fused_w2_weight.data[expert_id].copy_(w2)

    return (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_op_run(dtype):
    num_experts = 3
    (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    ) = setup_moe_test(dtype, num_experts)

    with torch.inference_mode():
        output_torch_moe = torch.ops.moe.torch_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
        )
        output_torch_fused_moe = torch.ops.moe.torch_fused_moe(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        output_trt_fused_moe = torch.ops.moe.trtllm_fused_moe(
            x,
            selected_experts,
            final_scales,
            fused_w3_w1_stacked_weight,
            fused_w2_weight,
        )
        ref_output = reference_moe_torch(x, selected_experts, final_scales, num_experts, weights)

    torch.cuda.synchronize()
    torch.testing.assert_close(output_trt_fused_moe, output_torch_fused_moe, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_trt_fused_moe, ref_output, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(output_torch_fused_moe, ref_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output_torch_moe, ref_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fp8_moe_op_run(dtype):
    num_experts = 3
    (
        x,
        selected_experts,
        final_scales,
        w1_weight,
        w2_weight,
        w3_weight,
        weights,
        fused_w3_w1_stacked_weight,
        fused_w2_weight,
    ) = setup_moe_test(dtype, num_experts)

    with torch.inference_mode():
        output_torch_moe = torch.ops.moe.torch_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
        )

    w1_input_scale, w2_input_scale, w3_input_scale = [], [], []
    w1_weight_scale, w2_weight_scale, w3_weight_scale = [], [], []
    for i in range(num_experts):
        inp_scale_val = torch.tensor(1.0).float().cuda()
        wt_scale_factor = 448 if dtype == torch.bfloat16 else 432  # float16 overflow with 448
        wt_scale_val = (torch.max(torch.abs(w1_weight[i])) / wt_scale_factor).float().to("cuda")
        w1_input_scale.append(inp_scale_val)
        w2_input_scale.append(inp_scale_val)
        w3_input_scale.append(inp_scale_val)
        w1_weight_scale.append(wt_scale_val)
        w2_weight_scale.append(wt_scale_val)
        w3_weight_scale.append(wt_scale_val)
        # Cast the expert weight tensors and fused weights to FP8.
        w1_weight[i] = (w1_weight[i] / w1_weight_scale[i]).to(torch.float8_e4m3fn)
        w2_weight[i] = (w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)
        w3_weight[i] = (w3_weight[i] / w3_weight_scale[i]).to(torch.float8_e4m3fn)
        fused_w3_w1_stacked_weight[i] = (fused_w3_w1_stacked_weight[i] / w1_weight_scale[i]).to(
            torch.float8_e4m3fn
        )
        fused_w2_weight[i] = (fused_w2_weight[i] / w2_weight_scale[i]).to(torch.float8_e4m3fn)

    with torch.inference_mode():
        output_torch_fp8_moe = torch.ops.moe.torch_fp8_moe(
            x,
            selected_experts,
            final_scales,
            w1_weight,
            w2_weight,
            w3_weight,
            w1_input_scale,
            w2_input_scale,
            w3_input_scale,
            w1_weight_scale,
            w2_weight_scale,
            w3_weight_scale,
        )
        ref_output = reference_moe_torch(x, selected_experts, final_scales, num_experts, weights)

    torch.cuda.synchronize()
    rtol = 0.5 if dtype == torch.bfloat16 else 1.5
    atol = 0.8 if dtype == torch.bfloat16 else 1
    torch.testing.assert_close(output_torch_fp8_moe, output_torch_moe, rtol=rtol, atol=atol)
    torch.testing.assert_close(output_torch_fp8_moe, ref_output, rtol=rtol, atol=atol)
