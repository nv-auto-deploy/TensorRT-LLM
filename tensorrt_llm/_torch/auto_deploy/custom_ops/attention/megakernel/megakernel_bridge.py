# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone decode runtime using the CUDA megakernel for attention.

Reuses the MPK layer spec infrastructure for weight extraction but does NOT
use Mirage for any computation. The attention sublayer runs through the
megakernel custom op; FFN/MoE uses standard torch ops.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def build_megakernel_runtime_callable(
    translation_plan: dict,
    source_model: GraphModule,
) -> "_MegakernelRuntime":
    """Build a decode runtime that uses the megakernel for attention.

    Reuses ``_build_gemma_runtime_specs`` from mirage_bridge for weight
    extraction, but all compute is done via the megakernel custom op
    (attention) and standard torch ops (FFN/MoE). No Mirage dependency
    at inference time.
    """
    # Import here to avoid circular dependency at module level
    import tensorrt_llm._torch.auto_deploy.custom_ops.attention.megakernel.kernel_a_custom_op  # noqa: F401

    return _MegakernelRuntime(source_model, translation_plan)


class _MegakernelRuntime:
    """Full decode runtime: megakernel attention + torch FFN/MoE."""

    def __init__(self, source_model: GraphModule, translation_plan: Dict[str, Any]):
        from tensorrt_llm._torch.auto_deploy.mpk.mirage_bridge import (
            _build_gemma_runtime_specs,
            _find_tensor_attr_by_substrings,
            _lookup_tensor_attr,
            _node_arg_getattr_target,
        )

        self.source_model = source_model
        self.plan = translation_plan

        # Extract per-layer weight specs (reuse MPK infrastructure)
        self.layer_specs = _build_gemma_runtime_specs(source_model, translation_plan)

        # Extract embedding, final norm, RoPE caches
        node_map = {node.name: node for node in source_model.graph.nodes}

        embed_node = node_map["model_language_model_embed_tokens_embedding"]
        self.embed_weight = _lookup_tensor_attr(
            source_model, _node_arg_getattr_target(embed_node, 0)
        )
        embed_scale_node = node_map["model_language_model_embed_tokens_to"]
        self.embed_scale = _lookup_tensor_attr(
            source_model, _node_arg_getattr_target(embed_scale_node, 0)
        )

        gather_node = next(n for n in source_model.graph.nodes if "gather_tokens" in (n.name or ""))
        final_norm_node = gather_node.args[0]
        self.final_norm_weight = _lookup_tensor_attr(
            source_model, _node_arg_getattr_target(final_norm_node, 1)
        )

        self.local_cos = _find_tensor_attr_by_substrings(
            source_model, "rotary_emb_local", "ad_cos_cached"
        )
        self.local_sin = _find_tensor_attr_by_substrings(
            source_model, "rotary_emb_local", "ad_sin_cached"
        )

        # Build cos_sin_cache in the layout the megakernel expects: [max_pos, head_dim]
        # Share one cache per RoPE type (local vs global) to avoid OOM.
        self._cos_sin_caches = {}
        _shared_cs_cache = {}  # keyed by (head_dim, is_sliding)

        def _reshape_cos_sin(t, half):
            """Normalize cos/sin to [max_pos, half] regardless of input shape."""
            if t.ndim == 3:
                return t[:, 0, :half]
            if t.ndim == 2:
                return t[:, :half]
            raise ValueError(f"Unexpected cos/sin shape: {t.shape}")

        for spec in self.layer_specs:
            hd = spec.head_dim
            half = hd // 2
            is_sliding = spec.sliding_window is not None
            cache_key = (hd, is_sliding)

            if cache_key not in _shared_cs_cache:
                if is_sliding:
                    cos = _reshape_cos_sin(self.local_cos, half)
                    sin = _reshape_cos_sin(self.local_sin, half)
                else:
                    try:
                        gcos = _find_tensor_attr_by_substrings(
                            source_model, "rotary_emb_global", "ad_cos_cached"
                        )
                        gsin = _find_tensor_attr_by_substrings(
                            source_model, "rotary_emb_global", "ad_sin_cached"
                        )
                        cos = _reshape_cos_sin(gcos, half)
                        sin = _reshape_cos_sin(gsin, half)
                    except AttributeError:
                        cos = _reshape_cos_sin(self.local_cos, half)
                        sin = _reshape_cos_sin(self.local_sin, half)
                _shared_cs_cache[cache_key] = torch.cat(
                    [cos.float(), sin.float()], dim=-1
                ).contiguous()

            self._cos_sin_caches[spec.layer_index] = _shared_cs_cache[cache_key]

        num_eligible = sum(
            1 for s in self.layer_specs if s.head_dim == 256 and s.sliding_window is not None
        )
        ad_logger.info(
            f"MegakernelRuntime: {len(self.layer_specs)} layers, "
            f"{num_eligible} sliding-attn megakernel-eligible"
        )

    def _get_input_names(self) -> list[str]:
        if not hasattr(self, "_input_names_cache"):
            self._input_names_cache = [
                n.name for n in self.source_model.graph.nodes if n.op == "placeholder"
            ]
        return self._input_names_cache

    def prewarm_decode_executors(self):
        from .launcher import load_megakernel_module

        ad_logger.info("Prewarming megakernel CUDA extension...")
        load_megakernel_module()
        ad_logger.info("Megakernel CUDA extension ready.")

    def __call__(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        token_gather_indices: torch.Tensor,
        batch_info_host: torch.Tensor,
        cu_seqlen_host: torch.Tensor,
        cu_num_pages: torch.Tensor,
        cu_num_pages_host: torch.Tensor,
        cache_loc: torch.Tensor,
        last_page_len: torch.Tensor,
        last_page_len_host: torch.Tensor,
        seq_len_with_cache_host: torch.Tensor,
        cu_seqlen: torch.Tensor,
        seq_len_with_cache: torch.Tensor,
        *kv_caches: torch.Tensor,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        hidden = F.embedding(input_ids, self.embed_weight).to(torch.bfloat16)
        hidden = hidden * self.embed_scale.to(dtype=hidden.dtype)

        triton_batch_indices, triton_positions = (
            torch.ops.auto_deploy.triton_paged_prepare_metadata.default(
                position_ids,
                batch_info_host,
                cu_seqlen,
                seq_len_with_cache,
            )
        )

        for layer_spec, kv_cache in zip(self.layer_specs, kv_caches):
            hidden = self._run_layer(
                hidden,
                layer_spec,
                kv_cache,
                position_ids=position_ids,
                batch_info_host=batch_info_host,
                cu_seqlen_host=cu_seqlen_host,
                cu_num_pages=cu_num_pages,
                cu_num_pages_host=cu_num_pages_host,
                cache_loc=cache_loc,
                last_page_len=last_page_len,
                last_page_len_host=last_page_len_host,
                seq_len_with_cache_host=seq_len_with_cache_host,
                triton_batch_indices=triton_batch_indices,
                triton_positions=triton_positions,
            )

        hidden = _rms_norm(hidden, self.final_norm_weight)
        gathered = torch.ops.auto_deploy.gather_tokens.default(
            hidden,
            token_gather_indices,
            batch_info_host,
        )
        logits = gathered @ self.embed_weight.transpose(0, 1)
        logits = torch.tanh(logits / 30.0) * 30.0
        logits = logits.view(*gathered.shape[:-1], logits.shape[-1])
        return {"logits": logits}

    def _run_layer(self, hidden, spec, kv_cache, **metadata):
        """Run one decoder layer: megakernel attention + torch FFN/MoE."""
        eps = 1e-6
        use_megakernel = spec.head_dim == 256 and spec.sliding_window is not None

        if use_megakernel:
            # ── Megakernel attention ──
            residual = hidden.reshape(-1, hidden.shape[-1])
            attn_normed = _rms_norm(hidden, spec.input_layernorm_weight, eps).reshape(
                -1, hidden.shape[-1]
            )
            cos_sin = self._cos_sin_caches[spec.layer_index]

            post_attn, pre_ffn = torch.ops.auto_deploy.megakernel_gemma_kernel_a_decode(
                residual,
                attn_normed,
                spec.qkv_weight,
                spec.o_proj_weight,
                spec.q_norm_weight.float(),
                spec.k_norm_weight.float(),
                spec.v_norm_weight.float(),
                spec.post_attention_layernorm_weight.float(),
                spec.pre_feedforward_layernorm_weight.float(),
                metadata["position_ids"],
                cos_sin,
                metadata["batch_info_host"],
                metadata["cu_seqlen_host"],
                metadata["cu_num_pages"],
                metadata["cu_num_pages_host"],
                metadata["cache_loc"],
                metadata["last_page_len"],
                metadata["last_page_len_host"],
                metadata["seq_len_with_cache_host"],
                metadata["triton_batch_indices"],
                metadata["triton_positions"],
                kv_cache,
                None,  # scale
                spec.sliding_window,
                eps,
            )
            post_attn = post_attn.reshape(hidden.shape)
            pre_ffn = pre_ffn.reshape(hidden.shape)
        else:
            # ── Fallback: standard torch attention for non-sliding layers ──
            residual = hidden
            attn_normed = _rms_norm(hidden, spec.input_layernorm_weight, eps)
            qkv = F.linear(attn_normed, spec.qkv_weight)
            q_width = spec.q_heads * spec.head_dim
            kv_width = spec.kv_heads * spec.head_dim
            q = qkv[..., :q_width].reshape(*qkv.shape[:-1], spec.q_heads, spec.head_dim)
            k = qkv[..., q_width : q_width + kv_width].reshape(
                *qkv.shape[:-1], spec.kv_heads, spec.head_dim
            )
            if spec.qkv_shared_kv:
                v = k.clone()
            else:
                v = qkv[..., q_width + kv_width :].reshape(
                    *qkv.shape[:-1], spec.kv_heads, spec.head_dim
                )
            q = _rms_norm(q, spec.q_norm_weight.float(), eps)
            k = _rms_norm(k, spec.k_norm_weight.float(), eps)
            v = _rms_norm(v, spec.v_norm_weight.float(), eps)

            # RoPE + paged attention via existing triton ops
            q_rope, k_rope = torch.ops.auto_deploy.flashinfer_rope.default(
                q,
                k,
                metadata["position_ids"],
                self._cos_sin_caches[spec.layer_index],
                True,
            )
            attn_out = torch.ops.auto_deploy.triton_paged_mha_with_cache.default(
                q_rope,
                k_rope,
                v,
                metadata["batch_info_host"],
                metadata["cu_seqlen_host"],
                metadata["cu_num_pages"],
                metadata["cu_num_pages_host"],
                metadata["cache_loc"],
                metadata["last_page_len"],
                metadata["last_page_len_host"],
                metadata["seq_len_with_cache_host"],
                metadata["triton_batch_indices"],
                metadata["triton_positions"],
                kv_cache,
                None,
                spec.sliding_window,
            )
            o_proj = F.linear(attn_out.reshape(*attn_out.shape[:-2], -1), spec.o_proj_weight)
            attn_branch = _rms_norm(o_proj, spec.post_attention_layernorm_weight.float(), eps)
            post_attn = residual + attn_branch
            pre_ffn = _rms_norm(post_attn, spec.pre_feedforward_layernorm_weight.float(), eps)

        # ── FFN: Dense MLP ──
        ffn_out = F.linear(pre_ffn, spec.ffn_gate_up_weight)
        gate, up = ffn_out.chunk(2, dim=-1)
        ffn_act = F.gelu(gate, approximate="tanh") * up
        ffn_down = F.linear(ffn_act, spec.ffn_down_weight)
        ffn_normed = _rms_norm(ffn_down, spec.post_feedforward_layernorm_1_weight.float(), eps)

        # ── MoE ──
        hs_flat = post_attn.reshape(-1, post_attn.shape[-1])
        # Router: RMSNorm(hidden) * root_size * scale → linear(proj_weight) → softmax → topk
        router_normed = _rms_norm(hs_flat, spec.router_root_size.float(), eps)
        router_input = router_normed * spec.router_scale.float().view(1, -1)
        router_logits = F.linear(
            router_input.to(spec.router_proj_weight.dtype), spec.router_proj_weight
        )
        top_k_weights, top_k_indices = torch.topk(
            router_logits.softmax(dim=-1), k=spec.topk, dim=-1
        )
        moe_input = _rms_norm(hs_flat, spec.pre_feedforward_layernorm_2_weight.float(), eps)

        # fused_moe API: (input, token_selected_experts, token_final_scales,
        #   fc1_expert_weights, fc1_expert_biases, fc2_expert_weights, fc2_expert_biases,
        #   output_dtype, quant_scales, ..., activation_type=5)
        # activation_type: 2=Gelu, 5=GeluGatedTanh (Gemma4 uses gelu_pytorch_tanh)
        empty_bias = None
        moe_results = torch.ops.trtllm.fused_moe(
            moe_input.to(spec.moe_w13_stacked_weight.dtype),
            top_k_indices.to(torch.int32),
            top_k_weights.float(),
            spec.moe_w13_stacked_weight,
            empty_bias,
            spec.moe_w2_weight,
            empty_bias,
            moe_input.dtype,
            [],  # quant_scales
            activation_type=5,  # GeluGatedTanh
        )
        moe_out = moe_results[0]
        moe_normed = _rms_norm(
            moe_out.reshape(post_attn.shape),
            spec.post_feedforward_layernorm_2_weight.float(),
            eps,
        )

        # ── Combine ──
        ffn_moe = ffn_normed + moe_normed
        ffn_moe_normed = _rms_norm(ffn_moe, spec.post_feedforward_layernorm_weight.float(), eps)
        hidden = post_attn + ffn_moe_normed
        hidden = hidden * spec.layer_scalar

        return hidden
