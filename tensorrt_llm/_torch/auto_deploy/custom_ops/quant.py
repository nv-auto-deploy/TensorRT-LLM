"""Definition of the quant module that can be used for PTQ."""

import warnings
from typing import Optional

import torch
from flashinfer import bmm_fp8
from torch import nn

from tensorrt_llm._torch.autotuner import autotune

from ..distributed import common as dist
from ..distributed import trtllm as trtllm_dist

TRTLLM_FP4_OP_AVAILABLE = True

TRTLLM_NVFP4_SCALING_VECTOR_SIZE = 16


@torch.library.custom_op("auto_deploy::torch_quant_fn", mutates_args=())
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
        return torch.ops.auto_deploy.torch_quant_fn(x, self.scale)


FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6.0
FP4_GLOBAL_SCALE_MAX = FP8_MAX * FP4_MAX


def _to_fp8(x, scale):
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


@torch.library.custom_op("auto_deploy::torch_quant_fp8_linear", mutates_args=())
def fp8_linear(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 linear op similar to torch.nn.linear using TensorRT-LLM FP8 operations.

    Args:
        input: unquantized input tensor
        weight_fp8: pre-quantized weight tensor, with dtype torch.float8_e4m3fn
        input_scale: (Optional) pre-computed scalar tensor for static quantization.
        weight_scale: scalar tensor for weight dequantization.

    Returns:
        The linear output with the original dtype as the input.
    """
    input_shape = input.shape
    input_dtype = input.dtype

    n = weight_fp8.shape[0]
    k = input_shape[-1]
    input = input.reshape(-1, k)

    assert n % 2 == 0
    assert k % 16 == 0

    # Use TensorRT-LLM FP8 per-tensor quantization
    assert input_scale is not None
    input_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(input, input_scale)

    # Use TensorRT-LLM FP8 scaled matrix multiply
    # Choose between CUDA core (for small M) and cuBLAS (for large M) implementations
    if input_fp8.shape[0] <= 8:
        # Use CUDA core for small M dimension (better for small batch sizes)
        output = torch.ops.trtllm.cuda_scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=input_dtype,
        )
        if bias is not None:
            output = output + bias
    else:
        # Use cuBLAS for large M dimension
        output = torch.ops.trtllm.cublas_scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=bias,
            out_dtype=input_dtype,
        )
    return output.reshape(*input_shape[:-1], n)


@fp8_linear.register_fake
def fp8_linear_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp8.to(input.dtype), bias)


@torch.library.custom_op("auto_deploy::torch_quant_fused_fp8_linear_all_reduce", mutates_args=())
@torch.compile(dynamic=True)
def fused_fp8_linear_all_reduce(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.ops.auto_deploy.torch_quant_fp8_linear(
        input, weight_fp8, bias, input_scale, weight_scale
    )
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
    return torch.ops.auto_deploy.torch_quant_fp8_linear(
        input, weight_fp8, bias, input_scale, weight_scale
    )


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
        return torch.ops.auto_deploy.torch_quant_fp8_linear(
            x, self.weight, self.bias, self.input_scale, self.weight_scale
        )


@torch.library.custom_op("auto_deploy::torch_quant_nvfp4_linear", mutates_args=())
@torch.compile(dynamic=True)
def nvfp4_linear(
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
            x_fp4, weight_fp4, x_sf_block, weight_scale, alpha, input.dtype
        )

    if bias is not None:
        output = output + bias

    return output.reshape(*input_shape[:-1], n)


@nvfp4_linear.register_fake
def fp4_linear_fake(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp4.repeat(1, 2).to(input.dtype), bias)


def is_column_major(tensor):
    rows, _ = tensor.shape[-2:]
    strides = tensor.stride()
    return strides[-2] == 1 and strides[-1] == rows


@torch.library.custom_op("auto_deploy::torch_quant_fp8_bmm", mutates_args=())
def fp8_bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 BMM op similar to torch.bmm.

    Args:
        input: unquantized input tensor with shape (B, M, K)
        mat2: weight tensor with shape (B, K, N), with dtype torch.float8_e4m3fn,
            or torch.float16, or torch.bfloat16
        input_scale: a scalar tensor - the inverse scale for input quantization
        weight_scale: a scalar tensor - the inverse scale for weight quantization

    Returns:
        The BMM output with shape (B, M, N) and the original dtype as the input.
    """
    # Ensure input is contiguous
    input = input.contiguous()
    original_input_dtype = input.dtype

    # Convert input to fp8 using provided scale
    if input.dtype in [torch.float16, torch.bfloat16]:
        input_fp8 = _to_fp8(input, input_scale)
    else:
        assert input.dtype == torch.float8_e4m3fn
        input_fp8 = input

    # Convert mat2 to fp8 using provided scale
    if mat2.dtype in [torch.float16, torch.bfloat16]:
        mat2_fp8 = _to_fp8(mat2, weight_scale)
    else:
        assert mat2.dtype == torch.float8_e4m3fn
        mat2_fp8 = mat2

    # Ensure mat2 is contiguous in column-major format only if needed
    # Check if the tensor is already contiguous when transposed (i.e., already column-major)
    if not is_column_major(mat2_fp8):
        warnings.warn(
            "mat2 is not in column-major format, transposing it, this will cause performance degradation."
        )
        mat2_fp8 = mat2_fp8.transpose(-2, -1).contiguous().transpose(-2, -1)

    # Get dimensions
    B, M, K = input.shape
    B2, K2, N = mat2_fp8.shape
    assert B == B2, f"Batch dimensions must match: {B} vs {B2}"
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    output = torch.empty((B, M, N), dtype=original_input_dtype, device=input.device)
    bmm_fp8(
        input_fp8, mat2_fp8, input_scale.float(), weight_scale.float(), original_input_dtype, output
    )

    return output


@fp8_bmm.register_fake
def fp8_bmm_fake(
    input: torch.Tensor,
    mat2: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation of fp8_bmm for testing and tracing."""
    # Use standard bmm
    return torch.bmm(input.to(torch.float), mat2.to(torch.float)).to(input.dtype)
