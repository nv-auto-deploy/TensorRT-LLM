"""Custom op collection for cached causal conv1d in pure PyTorch.

This mirrors the structure used by the cached Mamba/SSM ops:
- clean functional interface identical to the uncached op plus cache argument at the end
- prefill vs decode handled internally
- cache read/write handled internally
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _build_conv_state_from_sequence(input_bt_c: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Builds a convolution state of fixed window `kernel_size` from a sequence.

    input_bt_c: [B, T, C]
    Returns: [B, C, K]
    """
    # [B, T, C] -> [B, C, T]
    input_b_c_t = input_bt_c.transpose(1, 2)
    seq_len = input_b_c_t.shape[-1]
    if seq_len >= kernel_size:
        return input_b_c_t[..., -kernel_size:]
    pad_amount = kernel_size - seq_len
    return F.pad(input_b_c_t, (pad_amount, 0))


def _torch_causal_conv1d_prefill(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prefill path: compute full conv and produce initial cache state.

    Returns (y, conv_state) where y: [B, T, C_out] and conv_state: [B, C_in, K].
    """
    assert padding_mode == "zeros", "padding_mode must be zeros"

    batch_size, seq_len, _ = input.shape
    # Reuse the uncached op for the actual convolution
    y = torch.ops.auto_deploy.torch_causal_conv1d(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        padding_mode,
    )

    kernel_size = weight.shape[-1]
    conv_state = _build_conv_state_from_sequence(input, kernel_size)
    return y, conv_state


def _torch_causal_conv1d_decode(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
    conv_state_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode path: update cache with the latest token and compute the last output.

    Returns (y, updated_conv_state) where y: [B, 1, C_out] and updated state: [B, C_in, K].
    """
    assert padding_mode == "zeros", "padding_mode must be zeros"
    # For cached decode we currently support stride=1 and dilation=1 (standard causal conv)
    # This mirrors the original decode implementation assumptions for Bamba.
    assert stride == 1, "cached causal conv1d currently supports stride == 1 only"
    assert dilation == 1, "cached causal conv1d currently supports dilation == 1 only"

    batch_size, seq_len, _ = input.shape
    assert seq_len == 1, "decode path expects seq_len == 1"

    kernel_size = weight.shape[-1]
    assert conv_state_cache.shape[-1] == kernel_size, (
        "conv_state_cache's last dim must equal kernel_size"
    )

    # Update cache in-place: roll left and place the new element at the end
    updated_cache = conv_state_cache.roll(shifts=-1, dims=-1)
    # [B, T=1, C] -> [B, C]
    new_sample_bc = input.transpose(1, 2)[..., 0]
    updated_cache[:, :, -1] = new_sample_bc.to(updated_cache.dtype).to(updated_cache.device)

    # Compute output for the current step using the cache window.
    # Convert cache window [B, C_in, K] -> short sequence [B, K, C_in]
    window_seq = updated_cache.transpose(1, 2).to(device=weight.device, dtype=weight.dtype)
    # Reuse the uncached op with padding=0, stride=1, dilation=1 so output length == 1
    y = torch.ops.auto_deploy.torch_causal_conv1d(
        window_seq,
        weight,
        bias,
        1,  # stride
        0,  # padding
        1,  # dilation
        groups,
        padding_mode,
    )

    return y, updated_cache


def _update_conv_state_cache(conv_cache: torch.Tensor, new_state: torch.Tensor) -> None:
    conv_cache.copy_(new_state)


@torch.library.custom_op("auto_deploy::torch_cached_causal_conv1d", mutates_args={})
def _torch_cached_causal_conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    conv_state_cache: torch.Tensor = None,
) -> torch.Tensor:
    batch_size, seq_len, _ = input.shape

    if seq_len == 1:
        y, updated_state = _torch_causal_conv1d_decode(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            padding_mode,
            conv_state_cache,
        )
    else:
        y, updated_state = _torch_causal_conv1d_prefill(
            input, weight, bias, stride, padding, dilation, groups, padding_mode
        )

    _update_conv_state_cache(conv_state_cache, updated_state)
    return y


@_torch_cached_causal_conv1d.register_fake
def _torch_cached_causal_conv1d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
    conv_state_cache: torch.Tensor = None,
) -> torch.Tensor:
    return torch.empty_like(input)
