"""Custom ops required for implementing tensor parallelism."""

from typing import List, Optional

import torch

from ..distributed import common as dist


@torch.library.custom_op("auto_deploy::torch_all_gather", mutates_args=(), device_types="cuda")
def all_gather(
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """All gather followed by concat in dim = 0. This is the default nccl behavior."""
    tl = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tl, tensor)
    return torch.cat(tl, dim=dim)


@all_gather.register_fake
def all_gather_fake(tensor, dim=0):
    return torch.cat([torch.empty_like(tensor) for _ in range(dist.get_world_size())], dim=dim)


@torch.library.custom_op("auto_deploy::torch_all_reduce", mutates_args=(), device_types="cuda")
def all_reduce(t: torch.Tensor) -> torch.Tensor:
    """All_reduce across the ranks. Reduction op is SUM.

    NOTE: this op requires an extra memory copy and should ONLY be used for debugging + testing. For
    efficient all_reduce ops one should write/replace it with a fused op.
    """
    t_res = t.clone()
    dist.all_reduce(t_res, op=dist.ReduceOp.SUM)
    return t_res


@all_reduce.register_fake
def all_reduce_fake(tensor):
    return torch.empty_like(tensor)


@torch.library.custom_op(
    "auto_deploy::torch_fused_linear_all_reduce", mutates_args=(), device_types="cuda"
)
def fused_linear_all_reduce(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Fused linear followed by in-place all_reduce on the output."""
    output = torch.ops.aten.linear(input, weight, bias)
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output


@fused_linear_all_reduce.register_fake
def fused_linear_all_reduce_fake(input, weight, bias):
    return torch.ops.aten.linear(input, weight, bias)
