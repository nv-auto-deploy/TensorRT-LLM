"""Custom ops required for Qwen2.5-VL vision model export."""

from typing import Tuple

import torch


@torch.library.custom_op(
    "auto_deploy::qwen_vision_data_dependent_ops", mutates_args=(), device_types=["cuda", "cpu"]
)
def qwen_vision_data_dependent_ops(
    grid_thw: torch.Tensor,
    hidden_states: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom op that encapsulates all data-dependent operations for Qwen2.5-VL vision model.

    Args:
        grid_thw: Grid dimensions [num_images, 3] where each row is [t, h, w]
        hidden_states: Hidden states after patch embedding
        spatial_merge_size: Spatial merge size for the vision model
        window_size: Window size for windowed attention
        patch_size: Vision transformer patch size
        spatial_merge_unit: Spatial merge unit

    Returns:
        processed_hidden_states: Hidden states after window indexing
        pos_emb_cos: Cosine part of position embeddings
        pos_emb_sin: Sine part of position embeddings
        cu_window_seqlens: Cumulative window sequence lengths
        cu_seqlens: Cumulative sequence lengths for full attention
        reverse_indices: Indices to reverse the window ordering (for final step)
    """
    device = grid_thw.device

    # === ROT_POS_EMB CALCULATION ===
    pos_ids = []
    for t, h, w in grid_thw:
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

    pos_ids = torch.cat(pos_ids, dim=0)
    max_grid_size = grid_thw[:, 1:].max()

    # Rotary embedding calculation (matching original implementation)
    dim = 40  # head_dim // 2 for Qwen2.5-VL
    theta = 10000.0
    # Match original implementation exactly - no explicit device specification
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    # Move to device after calculation to match the flow
    inv_freq = inv_freq.to(device)
    torch.save(inv_freq, "inv_freq_patched.pt")
    seq = torch.arange(max_grid_size, device=device, dtype=torch.float)
    freqs = torch.outer(seq, inv_freq)
    rotary_pos_emb_full = freqs
    rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

    # === GET_WINDOW_INDEX CALCULATION ===
    window_index = []
    cu_window_seqlens = [0]
    window_index_id = 0
    vit_merger_window_size = window_size // spatial_merge_size // patch_size

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h, llm_grid_w = (
            grid_h.item() // spatial_merge_size,
            grid_w.item() // spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
            grid_t, llm_grid_h, llm_grid_w
        )
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * spatial_merge_unit + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()

    window_index = torch.cat(window_index, dim=0)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    # === CU_SEQLENS CALCULATION ===
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    # === REVERSE_INDICES CALCULATION ===
    reverse_indices = torch.argsort(window_index)

    # === ADVANCED INDEXING OPERATIONS ===
    # Process hidden_states with window indexing
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]  # ADVANCED INDEXING
    hidden_states = hidden_states.reshape(seq_len, -1)

    # Process rotary_pos_emb with window indexing
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]  # ADVANCED INDEXING
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

    pos_emb_cos = emb.cos()
    pos_emb_sin = emb.sin()

    return hidden_states, pos_emb_cos, pos_emb_sin, cu_window_seqlens, cu_seqlens, reverse_indices


@qwen_vision_data_dependent_ops.register_fake
def qwen_vision_data_dependent_ops_fake(
    grid_thw: torch.Tensor,
    hidden_states: torch.Tensor,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
    spatial_merge_unit: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fake implementation for symbolic tracing.
    Returns tensors with correct symbolic shapes but dummy values.
    """
    device = grid_thw.device
    dtype = grid_thw.dtype

    # Calculate total sequence length from hidden_states
    seq_len = hidden_states.shape[0]

    # Fake processed hidden_states: same shape as input
    processed_hidden_states = torch.zeros_like(hidden_states)

    # Fake position embeddings: separate cos and sin tensors with doubled embedding dim
    emb_dim = 80  # head_dim for Qwen2.5-VL
    pos_emb_cos = torch.zeros(seq_len, emb_dim, device=device, dtype=hidden_states.dtype)
    pos_emb_sin = torch.zeros(seq_len, emb_dim, device=device, dtype=hidden_states.dtype)

    # Fake cu_window_seqlens: varies based on windowing, but approximate
    num_windows = (seq_len // spatial_merge_unit // 16) + 2  # rough estimate
    cu_window_seqlens = torch.arange(num_windows + 1, device=device, dtype=dtype) * 16

    # Fake cu_seqlens: [num_images + 1]
    cu_seqlens = torch.cumsum(grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0], dim=0)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
    cu_seqlens = cu_seqlens.to(dtype=dtype)

    # Fake reverse_indices: identity mapping for fake implementation
    reverse_indices = torch.arange(seq_len, device=device, dtype=torch.long)

    return (
        processed_hidden_states,
        pos_emb_cos,
        pos_emb_sin,
        cu_window_seqlens,
        cu_seqlens,
        reverse_indices,
    )


@torch.library.custom_op(
    "auto_deploy::qwen_prepare_attention_mask", mutates_args=(), device_types=["cuda", "cpu"]
)
def qwen_prepare_attention_mask(
    hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, attn_implementation: str
) -> torch.Tensor:
    """
    Custom op for _prepare_attention_mask to handle data-dependent operations.

    Based on Qwen2_5_VisionTransformerPretrainedModel._prepare_attention_mask

    Returns a special marker tensor (empty tensor) when attention_mask should be None.
    """
    # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
    if attn_implementation == "flash_attention_2":
        # Return empty tensor as marker for None
        return torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)

    seq_length = hidden_states.shape[0]
    attention_mask = torch.full(
        [1, 1, seq_length, seq_length],
        torch.finfo(hidden_states.dtype).min,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    for i in range(1, len(cu_seqlens)):
        attention_mask[
            ..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]
        ] = 0
    return attention_mask


@qwen_prepare_attention_mask.register_fake
def qwen_prepare_attention_mask_fake(
    hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, attn_implementation: str
) -> torch.Tensor:
    """Fake implementation for symbolic tracing."""
    if attn_implementation == "flash_attention_2":
        # Return empty tensor as marker for None
        return torch.empty(0, device=hidden_states.device, dtype=hidden_states.dtype)

    seq_length = hidden_states.shape[0]
    attention_mask = torch.zeros(
        [1, 1, seq_length, seq_length],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    return attention_mask


@torch.library.custom_op(
    "auto_deploy::qwen_reverse_indexing", mutates_args=(), device_types=["cuda", "cpu"]
)
def qwen_reverse_indexing(
    hidden_states: torch.Tensor, reverse_indices: torch.Tensor
) -> torch.Tensor:
    """
    Custom op for reverse indexing operation to handle advanced indexing with symbolic indices.
    """
    return hidden_states[reverse_indices, :]


@qwen_reverse_indexing.register_fake
def qwen_reverse_indexing_fake(
    hidden_states: torch.Tensor, reverse_indices: torch.Tensor
) -> torch.Tensor:
    """Fake implementation for symbolic tracing."""
    return torch.zeros_like(hidden_states)
