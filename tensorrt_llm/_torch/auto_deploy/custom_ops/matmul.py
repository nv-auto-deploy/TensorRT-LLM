"""Custom ops for matmul."""

import torch


@torch.library.custom_op("auto_deploy::torch_matmul_simple", mutates_args=())
def matmul_simple(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """A wrapper for matmul to control how it is exposed during export.

    Using a custom op helps avoid extra view/reshape noise and stabilizes
    how the operation appears in the exported graph.
    """
    output = torch.ops.aten.matmul(a, b)
    return output


@matmul_simple.register_fake
def matmul_simple_fake(a, b):
    """Fake implementation of matmul_simple for export shape/dtype propagation."""
    return torch.ops.aten.matmul(a, b)
