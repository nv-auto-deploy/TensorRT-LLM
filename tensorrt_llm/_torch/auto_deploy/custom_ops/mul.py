"""Custom ops for elementwise multiply (a * b)."""

import torch


@torch.library.custom_op("auto_deploy::torch_mul_simple", mutates_args=())
def mul_simple(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper for aten.mul to stabilize how a*b appears in export graphs."""
    output = torch.ops.aten.mul(a, b)
    return output


@mul_simple.register_fake
def mul_simple_fake(a, b):
    return torch.ops.aten.mul(a, b)
