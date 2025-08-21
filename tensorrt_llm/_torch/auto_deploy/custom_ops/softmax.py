"""Custom ops for softmax."""

import torch


@torch.library.custom_op("auto_deploy::torch_softmax_simple", mutates_args=())
def softmax_simple(
    a: torch.Tensor, dim: int = -1, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """A wrapper for softmax to control export representation.

    We use aten._softmax which matches torch.nn.functional.softmax behavior.
    """
    if dtype is None:
        return torch.ops.aten._softmax(a, dim, False)
    output = torch.ops.aten._softmax(a.to(dtype), dim, False)
    return output


@softmax_simple.register_fake
def softmax_simple_fake(a, dim: int = -1, dtype=None):
    if dtype is None:
        return torch.ops.aten._softmax(a, dim, False)
    return torch.ops.aten._softmax(a.to(dtype), dim, False)
