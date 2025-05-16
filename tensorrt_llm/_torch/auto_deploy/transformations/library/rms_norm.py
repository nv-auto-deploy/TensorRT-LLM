"""Graph transform to optimize RMSNorm execution using FlashInfer."""

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch.fx import GraphModule

from ...utils.pattern_matcher_utils import register_pattern
from .._graph import canonicalize_graph


def _rms_norm_pattern(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Implements the RMSNorm pattern for pattern matching.

    Args:
        hidden_states: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor.
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def _rms_norm_replacement(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Replacement function that uses the FlashInfer RMSNorm implementation.

    Args:
        hidden_states: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using FlashInfer implementation.
    """
    return torch.ops.rms_norm.flashinfer(hidden_states, weight, eps)


def match_rms_norm_with_pm(gm: GraphModule) -> GraphModule:
    """Matches and replaces RMSNorm patterns in the graph with FlashInfer implementation.

    This function sets up pattern matching to identify RMSNorm operations in the graph
    and replaces them with optimized FlashInfer implementations.

    Args:
        gm: Input graph module to transform.

    Returns:
        Transformed graph module with optimized RMSNorm operations.
    """
    graph = gm.graph
    patterns = PatternMatcherPass()

    # Create dummy tensors for pattern matching
    bs = 1
    hidden_size = 512

    dummy = [
        torch.randn(bs, hidden_size, device="cuda", dtype=torch.float16),
        torch.randn(hidden_size, device="cuda", dtype=torch.float16),
        1e-6,
    ]
    register_pattern(
        search_fn=_rms_norm_pattern,
        replace_fn=_rms_norm_replacement,
        patterns=patterns,
        dummy_args=dummy,
        op_ignore_types={},
        scalar_workaround={"eps": 1e-6},
    )

    patterns.apply(graph)
    gm = canonicalize_graph(gm)
    return gm
