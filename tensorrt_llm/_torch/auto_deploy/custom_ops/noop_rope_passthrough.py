from typing import Tuple

import torch


@torch.library.custom_op("auto_deploy::noop_passthrough", mutates_args=())
def noop_passthrough(x: torch.Tensor) -> torch.Tensor:
    """Identity op used to make a region opaque while preserving exact shapes/dtypes."""
    # torch.library.custom_op forbids returning outputs that alias inputs.
    return x.clone()


@noop_passthrough.register_fake
def noop_passthrough_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("auto_deploy::noop_rope_cos_sin", mutates_args=())
def noop_rope_cos_sin(cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity op for RoPE cos/sin outputs (returns (cos, sin) unchanged)."""
    # torch.library.custom_op forbids returning outputs that alias inputs.
    return cos.clone(), sin.clone()


@noop_rope_cos_sin.register_fake
def noop_rope_cos_sin_fake(
    cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(cos), torch.empty_like(sin)
