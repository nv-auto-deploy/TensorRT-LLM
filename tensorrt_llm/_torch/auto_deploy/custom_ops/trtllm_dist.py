"""TRT-LLM optimized dist ops."""

from typing import List, Optional

import torch

from ....functional import AllReduceParams
from ....mapping import Mapping
from ...distributed import AllReduce, allgather
from ...modules.linear import AllReduceFusionOp, AllReduceStrategy
from ..distributed.common import get_rank_world_size


@torch.library.custom_op("auto_deploy::trtllm_all_gather", mutates_args=(), device_types="cuda")
def all_gather(
    tensor: torch.Tensor, dim: int = 0, sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """TRT-LLM all gather followed by concat in dim = 0."""
    rank, world_size = get_rank_world_size()
    p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
    result = allgather(tensor, p_config, dim=dim, sizes=sizes)
    assert isinstance(result, torch.Tensor), "Expected tensor result from allgather"
    return result


@all_gather.register_fake
def all_gather_fake(tensor, dim=0, sizes=None):
    rank, world_size = get_rank_world_size()
    return torch.cat([torch.empty_like(tensor) for _ in range(world_size)], dim=dim)


@torch.library.custom_op("auto_deploy::trtllm_all_reduce", mutates_args=(), device_types="cuda")
def all_reduce(tensor: torch.Tensor, strategy: int = int(AllReduceStrategy.AUTO)) -> torch.Tensor:
    """TRT-LLM all_reduce across the ranks."""
    rank, world_size = get_rank_world_size()
    p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)
    torch_op = AllReduce(mapping=p_config, strategy=AllReduceStrategy(strategy))
    result = torch_op(tensor)
    assert isinstance(result, torch.Tensor), "Expected tensor result from allreduce"
    return result


@all_reduce.register_fake
def all_reduce_fake(tensor):
    return torch.empty_like(tensor)


@torch.library.custom_op(
    "auto_deploy::trtllm_fused_allreduce_residual_rmsnorm", mutates_args=(), device_types="cuda"
)
def trtllm_fused_allreduce_residual_rmsnorm(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    strategy: int = int(AllReduceStrategy.AUTO),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fusing allreduce, residual (add), and hf_rms_norm together."""
    rank, world_size = get_rank_world_size()
    p_config = Mapping(world_size=world_size, tp_size=world_size, rank=rank)

    # Use AllReduceParams like the old implementation
    all_reduce_params = AllReduceParams(
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        bias=None,
        residual=residual,
        norm_weight=norm_weight,
        eps=eps,
    )

    torch_op = AllReduce(mapping=p_config, strategy=AllReduceStrategy(strategy))
    output = torch_op(tensor, all_reduce_params=all_reduce_params)
    assert len(output) == 2, "Expected 2 outputs from trtllm_fused_allreduce_residual_rmsnorm"
    return output[0], output[1]


@trtllm_fused_allreduce_residual_rmsnorm.register_fake
def trtllm_fused_allreduce_residual_rmsnorm_fake(
    tensor: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    strategy: int = int(AllReduceStrategy.AUTO),
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(tensor), torch.empty_like(tensor)
