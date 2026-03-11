"""MLA (Multi-head Latent Attention) and DSA (DeepSeek Sparse Attention) custom ops.

Exports:
- TorchBackendMLAAttention: Attention descriptor for MLA (registered as "torch_mla")
- FlashInferMLAAttention: Attention descriptor for FlashInfer MLA (registered as "flashinfer_mla")
- TorchBackendDSAAttention: Attention descriptor for DSA (registered as "torch_dsa")
- FlashMLADSAAttention: Attention descriptor for FlashMLA DSA (registered as "flashmla_dsa")
- torch_mla: Source op for MLA attention
- torch_dsa: Source op for DSA attention (MLA + Indexer sparse masking)
- torch_backend_mla_with_cache: Cached backend op with FlashInfer-compatible cache
- torch_backend_dsa_with_cache: Cached backend op for DSA
- flashinfer_mla_with_cache: Cached backend op using FlashInfer MLA kernels
- flash_mla_dsa_with_cache: Cached backend op for DSA using FlashMLA paged kernels
"""

from .flashinfer_mla import FlashInferMLAAttention, flashinfer_mla_with_cache
from .flashmla_dsa import FlashMLADSAAttention, flash_mla_dsa_with_cache
from .torch_backend_dsa import TorchBackendDSAAttention, torch_backend_dsa_with_cache
from .torch_backend_mla import TorchBackendMLAAttention, torch_backend_mla_with_cache
from .torch_dsa import torch_dsa
from .torch_mla import torch_mla

__all__ = [
    "TorchBackendMLAAttention",
    "FlashInferMLAAttention",
    "TorchBackendDSAAttention",
    "FlashMLADSAAttention",
    "torch_mla",
    "torch_dsa",
    "torch_backend_mla_with_cache",
    "torch_backend_dsa_with_cache",
    "flashinfer_mla_with_cache",
    "flash_mla_dsa_with_cache",
]
