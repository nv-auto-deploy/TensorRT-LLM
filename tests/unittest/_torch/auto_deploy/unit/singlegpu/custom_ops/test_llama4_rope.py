from typing import Tuple

import torch
from torch.testing import assert_close


# Version 1: explicit sine/cosine approach
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,  # (B, seq, head_dim)
    sin: torch.Tensor,  # (B, seq, head_dim)
    unsqueeze_dim: int = 1,  # dim of n_heads in q and k
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Version 2: complex multiplication approach
# Expect input of shape (B, seq, n_heads, head_dim)
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,  # (B, seq, head_dim//2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(
        xq.float().reshape(*xq.shape[:-1], -1, 2)
    )  # (B, seq, n_heads, head_dim//2)
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis[:, :, None, :]).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis[:, :, None, :]).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def test_apply_rotary_equivalence():
    batch_size = 2
    n_heads = 3
    seq_len = 7
    head_dim = 8
    base = 10000.0

    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)  # shape (B, seq)

    # Calculate inv_freq, cos, and sin for Version 1's rotary embedding.
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
    # (head_dim//2,)
    inv_freq_expanded = inv_freq[None, :, None].expand(batch_size, -1, 1)  # (B, head_dim//2, 1)
    position_ids_expanded = position_ids[:, None, :].float()  # (B, 1, seq)

    device = q.device

    freqs = (inv_freq_expanded.to(device) @ position_ids_expanded).transpose(
        1, 2
    )  # (B, seq, head_dim//2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (B, seq, head_dim)
    cos = emb.cos()
    sin = emb.sin()
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    # For Version 2: Compute the complex rotary factors.
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (B, seq, head_dim//2)

    # Apply Version 1's rotary embedding.
    out1_q, out1_k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    # Adapt inputs for Version 2.
    #
    # Version 2 groups consecutive elements in the last dimension into complex pairs:
    # i.e., it interprets the last dimension as
    # [p0, p1, p2, p3, p4, p5, ...] forming pairs (p0, p1), (p2, p3), etc.
    #
    # In contrast, Version 1 groups the head_dim into two halves: the first half and the second half.
    # To match the rotations, we must reorder the last dimension of q and k from
    # [0, 1, 2, 3, 4, 5, 6, 7] to [0, 4, 1, 5, 2, 6, 3, 7] (for head_dim = 8).

    order = torch.stack(
        [torch.arange(head_dim // 2), torch.arange(head_dim // 2) + head_dim // 2], dim=1
    ).reshape(-1)
    q_reordered = q[..., order]
    k_reordered = k[..., order]

    q_v2 = q_reordered.transpose(1, 2)
    k_v2 = k_reordered.transpose(1, 2)

    out2_q_v2, out2_k_v2 = apply_rotary_emb(q_v2, k_v2, freqs_cis)
    # First transpose back to (B, n_heads, seq, head_dim) and then invert the reordering
    # on the last dimension to recover the original half-first ordering.
    out2_q = out2_q_v2.transpose(1, 2)
    out2_k = out2_k_v2.transpose(1, 2)

    inv_order = torch.argsort(order)
    out2_q = out2_q[..., inv_order]
    out2_k = out2_k[..., inv_order]

    rtol = 1e-5
    atol = 1e-5
    assert_close(out1_q, out2_q, rtol=rtol, atol=atol)
    assert_close(out1_k, out2_k, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_apply_rotary_equivalence()
