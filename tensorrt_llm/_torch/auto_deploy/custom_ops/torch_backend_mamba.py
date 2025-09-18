"""Custom op collection for cached mamba mixer (linear attention) in pure PyTorch."""

from typing import List, Tuple

import torch

from .torch_mamba import _torch_ssm_transform_prefill


def _torch_cached_ssm_transform_decode(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    # NOTE: `torch` custom ops do not like `Tuple` inputs. Using `List` is the suggested WAR.
    time_step_limit: List[float],
    chunk_size: int,
    ssm_state_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    # retrieve some shape information
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    n_groups, ssm_state_size = B.shape[2:]

    # Note: there is no need to pad parameter matrices here, as there is just one new token
    # for batched generation
    dt = dt[:, 0, :][:, None, ...]
    dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], head_dim)
    # [num_heads] -> [num_heads, head_dim]
    dt_bias = dt_bias[..., None].expand(dt_bias.shape[0], head_dim)

    dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
    dt = torch.clamp(dt, time_step_limit[0], time_step_limit[1])
    A = A[..., None, None].expand(num_heads, head_dim, ssm_state_size).to(dtype=torch.float32)
    # [bsz, num_heads, head_dim, state_size]
    dA = torch.exp(dt[..., None] * A)

    # Discretize B
    # [bsz, n_groups * state_size] -> [bsz, n_groups, 1, state_size] ->
    # -> [bsz, n_groups, group to head repetition factor, state_size] -> [bsz, num_heads, state_size]
    B = B.reshape(batch_size, n_groups, -1)[..., None, :]
    B = B.expand(batch_size, n_groups, num_heads // n_groups, B.shape[-1]).contiguous()
    B = B.reshape(batch_size, -1, B.shape[-1])
    # [bsz, num_heads, head_dim, state_size]
    dB = dt[..., None] * B[..., None, :]

    # Discretize x into dB
    # [bsz, intermediate_size] -> [bsz, num_heads, head_dim]
    hidden_states = hidden_states.reshape(batch_size, -1, head_dim)
    dBx = dB * hidden_states[..., None]

    # State calculation
    updated_ssm_state = ssm_state_cache * dA + dBx

    # Subsequent output
    # [bsz, n_groups * state_size] -> [bsz, num_heads, state_size]
    C = C.reshape(batch_size, n_groups, -1)[..., None, :]
    C = C.expand(batch_size, n_groups, num_heads // n_groups, C.shape[-1]).contiguous()
    C = C.reshape(batch_size, -1, C.shape[-1])
    # [bsz, num_heads, head_dim]

    ssm_states = updated_ssm_state.to(dtype=C.dtype)  # Shape: [b, h, d, n]
    # Reshape ssm_states to merge the first two dimensions
    ssm_states_reshaped = ssm_states.view(
        batch_size * num_heads, head_dim, ssm_state_size
    )  # Shape: [b*h, d, n]
    C_reshaped = C.view(batch_size * num_heads, ssm_state_size, 1)  # Shape: [b*h, n, 1]
    y = torch.bmm(ssm_states_reshaped, C_reshaped)
    y = y.view(batch_size, num_heads, head_dim)

    # D skip connection
    # [num_heads] -> [num_heads, head_dim]
    D = D[..., None].expand(D.shape[0], head_dim)
    y = (y + hidden_states * D).to(y.dtype)

    # [bsz, num_heads, head_dim] -> [bsz, 1, num_heads, head_dim]
    y = y.reshape(batch_size, 1, num_heads, head_dim)
    return y, updated_ssm_state


def _update_ssm_state_cache(ssm_cache: torch.Tensor, ssm_state: torch.Tensor) -> None:
    ssm_cache.copy_(ssm_state)


@torch.library.custom_op("auto_deploy::torch_cached_ssm_transform", mutates_args={})
def _torch_cached_ssm_transform(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
    ssm_state_cache: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len = hidden_states.shape[:2]

    if seq_len == 1:
        y, updated_ssm_state = _torch_cached_ssm_transform_decode(
            hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size, ssm_state_cache
        )
    else:
        y, updated_ssm_state = _torch_ssm_transform_prefill(
            hidden_states, A, B, C, D, dt, dt_bias, time_step_limit, chunk_size
        )
    _update_ssm_state_cache(ssm_state_cache, updated_ssm_state)
    return y


@_torch_cached_ssm_transform.register_fake
def _torch_cached_ssm_transform_meta(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt: torch.Tensor,
    dt_bias: torch.Tensor,
    time_step_limit: List[float],
    chunk_size: int,
    ssm_state_cache: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)
