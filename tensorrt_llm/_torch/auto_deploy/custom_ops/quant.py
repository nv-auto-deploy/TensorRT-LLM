"""Definition of the quant module that can be used for PTQ."""

from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.autotuner import autotune

from ..distributed import common as dist
from ..distributed import trtllm as trtllm_dist
from .torch_libs.float8_python_api import addmm_float8_unwrapped

TRTLLM_FP4_OP_AVAILABLE = True

TRTLLM_NVFP4_SCALING_VECTOR_SIZE = 16


@torch.library.custom_op("quant::quant_fn", mutates_args=())
def quant_fn(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled_x = x / scale
    rounded_x = torch.round(scaled_x)
    rounded_x = scaled_x
    clamped_x = torch.clamp(rounded_x, -127, 128)
    y = clamped_x * scale
    return y


@quant_fn.register_fake
def quant_fn_fake(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class QuantModule(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x: torch.Tensor):
        return torch.ops.quant.quant_fn(x, self.scale)


FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6.0
FP4_GLOBAL_SCALE_MAX = FP8_MAX * FP4_MAX


def _to_fp8(x, scale):
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


@torch.library.custom_op("quant::fp8_linear", mutates_args=())
@torch.compile(dynamic=True)
def fp8_linear(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 linear op similar to torch.nn.linear.

    Args:
        input: unquantized input tensor
        weight_fp8: pre-quantized weight tensor, with dtype torch.float8_e4m3fn
        input_scale: a scalar tensor defined as amax / max value (448.0).
        weight_scale: a scalar tensor defined as amax / max value (448.0).

    Returns:
        The linear output with the original dtype as the input.
    """
    assert input.shape[-1] % 16 == 0
    assert weight_fp8.shape[-1] % 16 == 0

    input_shape = input.shape
    weight_shape = weight_fp8.shape

    # Cuda graph compatibility
    assert input_scale is not None
    input_fp8 = _to_fp8(input, input_scale)

    weight_fp8_t = weight_fp8.reshape(-1, weight_shape[-1]).t()
    output = addmm_float8_unwrapped(
        input_fp8.reshape(-1, input_shape[-1]),
        input_scale,
        weight_fp8_t,
        weight_scale,
        input.dtype,
        bias=bias,
        use_fast_accum=True,
    )

    return output.reshape(*input_shape[:-1], output.shape[-1])


@fp8_linear.register_fake
def fp8_linear_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp8.to(input.dtype), bias)


@torch.library.custom_op("quant::fused_fp8_linear_all_reduce", mutates_args=())
@torch.compile(dynamic=True)
def fused_fp8_linear_all_reduce(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.ops.quant.fp8_linear(input, weight_fp8, bias, input_scale, weight_scale)
    if trtllm_dist.is_trtllm_op_available():
        return trtllm_dist.trtllm_allreduce(out, op=dist.ReduceOp.SUM)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


@fused_fp8_linear_all_reduce.register_fake
def fused_fp8_linear_all_reduce_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.quant.fp8_linear(input, weight_fp8, bias, input_scale, weight_scale)


class FP8Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.weight.device
        weight_scale = torch.max(torch.abs(self.weight)).to(torch.float).to(device) / FP8_MAX
        self.weight = nn.Parameter((self.weight / weight_scale).to(torch.float8_e4m3fn))
        self.register_buffer(
            "input_scale", torch.tensor(1.0, device=self.weight.device, dtype=torch.float)
        )
        self.register_buffer("weight_scale", weight_scale)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(torch.half))

    def forward(self, x):
        return torch.ops.quant.fp8_linear(
            x, self.weight, self.bias, self.input_scale, self.weight_scale
        )


@torch.library.custom_op("quant::fp4_linear", mutates_args=())
@torch.compile(dynamic=True)
def fp4_linear(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 linear op similar to torch.nn.linear.

    Args:
        input: unquantized input tensor
        weight_fp4: pre-quantized weight tensor, with dtype torch.uint8 (1 uint8 == 2 elements)
        input_scale: a scalar tensor defined as per_tensor_amax / (FP8 max value (448.0) * FP4 max value (6.0)).
        weight_scale: a 1D tensor with shape (out_dim * in_dim / 16) padded to be multiple of (128 * 4).
            with value: per_block_amax / per_tensor_amax * FP8 max value (448.0)
        weight_scale_2: a scalar tensor defined as per_tensor_amax / (FP8 max value (448.0) * FP4 max value (6.0)).

    Returns:
        The linear output with the original dtype as the input.
    """
    assert TRTLLM_FP4_OP_AVAILABLE, "TRT-LLM FP4 operators are not available."

    input_shape = input.shape
    weight_shape = weight_fp4.shape

    n = weight_shape[0]
    k = input_shape[-1]
    assert k % 16 == 0
    assert weight_shape[-1] % 8 == 0
    assert weight_scale.numel() % (128 * 4) == 0

    input = input.reshape(-1, k)

    # FP4 compatibility
    assert input_scale is not None
    assert weight_scale is not None
    assert alpha is not None

    x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
        input, input_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE, False
    )
    with autotune():
        output = torch.ops.trtllm.nvfp4_gemm(
            x_fp4, weight_fp4, x_sf_block, weight_scale, alpha, False, input.dtype
        )

    if bias is not None:
        output = output + bias

    return output.reshape(*input_shape[:-1], n)


@fp4_linear.register_fake
def fp4_linear_fake(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp4.repeat(1, 2).to(input.dtype), bias)


@torch.library.custom_op("quant::fp8_bmm", mutates_args=())
@torch.compile(dynamic=True)
def fp8_bmm(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 BMM op similar to torch.bmm.

    Args:
        input: unquantized input tensor with shape (B, M, K)
        weight_fp8: pre-quantized weight tensor with shape (B, K, N), with dtype torch.float8_e4m3fn
        input_scale: a scalar tensor defined as amax / max value (448.0)
        weight_scale: a scalar tensor defined as amax / max value (448.0)

    Returns:
        The BMM output with shape (B, M, N) and the original dtype as the input.
    """
    # Ensure input is contiguous
    input = input.contiguous()
    input_fp8 = _to_fp8(input, input_scale)

    # Ensure weight is in the correct memory layout
    # For cuBLASLt, we need one matrix in row-major and one in column-major format
    # Since PyTorch uses row-major by default, we'll transpose the weight to get a different layout
    weight_fp8 = weight_fp8.transpose(-2, -1).contiguous().transpose(-2, -1)

    # Get dimensions
    B, M, K = input.shape
    B2, K2, N = weight_fp8.shape
    assert B == B2, f"Batch dimensions must match: {B} vs {B2}"
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    # Process each batch separately
    output = torch.empty((B, M, N), dtype=input.dtype, device=input.device)
    for b in range(B):
        # Allocate output for this batch to avoid in-place issues
        batch_output = output[b]

        # Use _scaled_mm with explicit out parameter
        torch._scaled_mm(
            input_fp8[b],
            weight_fp8[b],
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
            scale_result=None,
            out_dtype=input.dtype,
            use_fast_accum=True,
            out=batch_output,
        )

    return output


@fp8_bmm.register_fake
def fp8_bmm_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation of fp8_bmm for testing and tracing."""
    # Use standard bmm
    return torch.bmm(input, weight_fp8.to(input.dtype))


@torch.library.custom_op("quant::fp4_bmm", mutates_args=())
@torch.compile(dynamic=True)
def fp4_bmm(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """FP4 BMM op similar to torch.bmm.

    Args:
        input: unquantized input tensor with shape (B, M, K)
        weight_fp4: pre-quantized weight tensor with shape (B, N, K/2),
                    with dtype torch.uint8 (1 uint8 == 2 elements)
        input_scale: a scalar tensor for input quantization
        weight_scale: scale tensor for weight quantization
        alpha: a scalar tensor for scaling the result

    Returns:
        The BMM output with shape (B, M, N) and the original dtype as the input.
    """
    assert TRTLLM_FP4_OP_AVAILABLE, "TRT-LLM FP4 operators are not available."

    # Get dimensions
    B, M, K = input.shape
    B2, N, K2_div_2 = weight_fp4.shape
    assert B == B2, f"Batch dimensions must match: {B} vs {B2}"
    assert K // 2 == K2_div_2, f"Inner dimensions must match: {K // 2} vs {K2_div_2}"
    assert K % 16 == 0, "K must be divisible by 16"

    # Process each batch separately
    output = torch.empty((B, M, N), dtype=input.dtype, device=input.device)
    for b in range(B):
        # Quantize this batch's input to FP4
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            input[b], input_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE, False
        )

        # Perform GEMM - note that weight_fp4 is already in (N, K/2) format per batch
        with autotune():
            output[b] = torch.ops.trtllm.nvfp4_gemm(
                x_fp4, weight_fp4[b], x_sf_block, weight_scale, alpha, False, input.dtype
            )

    return output


@fp4_bmm.register_fake
def fp4_bmm_fake(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation of fp4_bmm for testing and tracing."""
    weight_fp4 = weight_fp4.transpose(1, 2)
    expanded_weight = weight_fp4.repeat(1, 1, 2).to(input.dtype)

    return torch.bmm(input, expanded_weight)


QUANT_LINEAR_OPS = [
    torch.ops.quant.fp8_linear,
    torch.ops.quant.fp4_linear,
]

QUANT_BMM_OPS = [
    torch.ops.quant.fp8_bmm,
    torch.ops.quant.fp4_bmm,
]

QUANT_OPS = QUANT_LINEAR_OPS + QUANT_BMM_OPS
