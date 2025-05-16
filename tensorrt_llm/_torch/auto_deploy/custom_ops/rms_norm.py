"""Custom operator for FlashInfer RMSNorm implementation."""

import flashinfer
import torch


@torch.library.custom_op("rms_norm::flashinfer", mutates_args=())
def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for FlashInfer RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using FlashInfer implementation.
    """
    return flashinfer.norm.rmsnorm(input, weight, eps)


@flashinfer_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Empty tensor with same shape as input.
    """
    return torch.empty_like(input)
