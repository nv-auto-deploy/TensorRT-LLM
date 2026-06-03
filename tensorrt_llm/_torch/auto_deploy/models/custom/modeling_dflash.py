# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""AutoDeploy custom DFlash draft model (prefill / export representation).

DFlash drafts a whole query block ``[last_accepted, MASK, ..., MASK]`` (width ``block_size``) in one
**non-causal** pass over a persistent drafter-side context K/V cache plus the current query block.
This module is the *exportable draft model*: its ``forward`` processes **only the query block** and
emits ``auto_deploy::dflash_attention(q, k, v, ctx_len, scale)`` per draft layer (the source op a
dedicated ``insert_cached_dflash_attention`` transform lowers to the cached, kv-cache-backed op).

Split of concerns (mirrors the PyTorch oracle ``modeling_speculative.py::dflash_forward``):
  - **context K/V** (from accepted target hidden states) is produced separately and *eagerly* by the
    wrapper's ``precompute_context_kv`` (Step 4) and scattered into the ctx K/V cache resources; it is
    NOT computed in this traced forward.
  - **this forward** projects only the query-block Q/K/V (q/k norm + RoPE) and calls
    ``dflash_attention``. At runtime the cached op reads ``ctx_k/v_cache[slot, :ctx_len]`` and appends
    the query-block K/V at ``ctx_len`` (non-causal over ``[ctx || block]``).

This file is **standalone**: the Qwen3 building blocks (RMSNorm, RoPE, MLP, the attention
projection/norm/RoPE/o_proj structure) are *copied* from ``modeling_qwen3.py`` rather than imported,
so DFlash does not couple to another model's definitions. Only the attention call differs (it emits
``dflash_attention`` and threads ``ctx_len``) plus the DFlash-specific ``fc`` / ``hidden_norm``. The
copied blocks use the **sharding IR** (``torch_linear_simple`` / ``view`` with ``tp_scaled_dim`` /
``torch_rope_with_explicit_cos_sin`` / ``all_reduce``) so the exported graph carries TP sharding hints.

Module/parameter names match the z-lab DFlash checkpoint (58 tensors: ``fc``, ``hidden_norm``,
``layers.{i}.self_attn.{q,k,v,o}_proj`` + ``{q,k}_norm``, ``layers.{i}.mlp.{gate,up,down}_proj``,
``layers.{i}.{input_layernorm,post_attention_layernorm}``, ``norm``) so the loader maps weights
directly (separate q/k/v_proj kept separate; AD fuses downstream).

v1 targets the **Qwen3** DFlash family (e.g. Qwen3-8B-DFlash-b16). DFlash drafts for non-Qwen bases
(gpt-oss, gemma, ...) would get their own standalone modeling — a later generalization.

NOTE (Step 3 scope): draft *nn.Module* classes + the exportable query-block forward.
``precompute_context_kv`` + fused-KV buffers (Step 4) and ``DFlashWrapper`` / ``DFlashOneModelFactory``
/ export-preservation ``post_process`` (Step 7) are added in later steps.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from ... import custom_ops  # noqa: F401  -- register all auto_deploy ops
from ...shim.interface import CachedSequenceInterface
from .rotary_utils import RotaryEmbeddingBase, build_rope_cos_sin_cache


class DFlashRMSNorm(nn.Module):
    """RMS Normalization (AutoDeploy torch_rmsnorm reference op). Copied from Qwen3RMSNorm."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )


class DFlashRotaryEmbedding(RotaryEmbeddingBase):
    """RoPE table generator. Copied from Qwen3RotaryEmbedding (small inv_freq buffer only)."""

    def __init__(self, dim: int, max_position_embeddings: int = 32768, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = build_rope_cos_sin_cache(self.inv_freq, self.max_position_embeddings, x)
        return cos[position_ids], sin[position_ids]


class DFlashMLP(nn.Module):
    """SwiGLU MLP. Copied from Qwen3MLP (gate/up colwise, down rowwise + all_reduce)."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ops.auto_deploy.torch_linear_simple(
            x, self.gate_proj.weight, self.gate_proj.bias, tp_mode="colwise", layer_type="mlp"
        )
        up = torch.ops.auto_deploy.torch_linear_simple(
            x, self.up_proj.weight, self.up_proj.bias, tp_mode="colwise", layer_type="mlp"
        )
        down = torch.ops.auto_deploy.torch_linear_simple(
            self.act_fn(gate) * up,
            self.down_proj.weight,
            self.down_proj.bias,
            tp_mode="rowwise",
            layer_type="mlp",
        )
        down = torch.ops.auto_deploy.all_reduce(down, layer_type="mlp")
        return down


class DFlashAttention(nn.Module):
    """Per-layer query-block attention emitting ``dflash_attention``.

    Copied from Qwen3Attention (per-head Q/K RMSNorm, sharding-IR projections), with the single
    DFlash change: the attention call is ``auto_deploy::dflash_attention(q, k, v, ctx_len, scale)``
    (non-causal; ``ctx_len`` carried inert so the cached lowering retrieves it) instead of the standard
    causal ``torch_attention``. Context K/V is supplied at runtime via the ctx cache (filled by
    ``precompute_context_kv``); this forward computes only the query-block Q/K/V.

    Sharding strategy (same as Qwen3): q/k/v_proj colwise (+ tp_min_local_shape for GQA),
    view tp_scaled_dim=2, o_proj rowwise + all_reduce.
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.scaling = self.head_dim ** (-0.5)
        attn_bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attn_bias)

        # Per-head Q/K normalization (Qwen3-style, over head_dim).
        self.q_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = DFlashRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, block, hidden]
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) [B, block, head_dim]
        ctx_len: torch.Tensor,  # [B] int32, persistent context length (inert at export)
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.q_proj.weight,
            self.q_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.k_proj.weight,
            self.k_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.v_proj.weight,
            self.v_proj.bias,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        q = torch.ops.auto_deploy.view(
            q, [bsz, q_len, self.num_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )
        k = torch.ops.auto_deploy.view(
            k, [bsz, q_len, self.num_kv_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )
        v = torch.ops.auto_deploy.view(
            v, [bsz, q_len, self.num_kv_heads, self.head_dim], tp_scaled_dim=2, layer_type="mha"
        )

        # Per-head Q/K norm (on head_dim), then RoPE (BSND, unsqueeze_dim=2).
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = position_embeddings  # [B, block, head_dim]
        q, k = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 2)

        # DFlash source op: non-causal block attention; cached lowering reads ctx K/V from cache and
        # appends this block's K/V at ctx_len. (Replaces Qwen3's causal torch_attention.)
        attn_output = torch.ops.auto_deploy.dflash_attention(q, k, v, ctx_len, self.scaling)

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output, self.o_proj.weight, self.o_proj.bias, tp_mode="rowwise", layer_type="mha"
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")
        return attn_output


class DFlashDecoderLayer(nn.Module):
    """Draft decoder layer. Copied from Qwen3DecoderLayer with DFlashAttention + ctx_len threading."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DFlashAttention(config, layer_idx=layer_idx)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        ctx_len: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings, ctx_len)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashModel(nn.Module):
    """DFlash draft transformer over the query block.

    Owns ``fc`` + ``hidden_norm`` (used by the wrapper's eager precompute_context_kv (Step 4), NOT in
    this traced forward), the draft layers, the final ``norm``, and the rotary table generator.
    ``embed_tokens`` / ``lm_head`` are NOT owned (shared from the target).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        target_layer_ids = getattr(config, "dflash_config", {}).get("target_layer_ids", []) or []
        self.num_capture_layers = len(target_layer_ids)

        # ``fc`` fuses concatenated multi-layer target hidden states -> draft hidden size.
        fc_in = config.hidden_size * max(self.num_capture_layers, 1)
        self.fc = nn.Linear(fc_in, config.hidden_size, bias=False)
        self.hidden_norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = DFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = (getattr(config, "rope_scaling", None) or {}).get("rope_theta", 10000.0)
        self.rotary_emb = DFlashRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,  # [B, block, hidden] -- embedded query block (target embed)
        position_ids: torch.Tensor,  # [B, block] -- query-block absolute positions
        ctx_len: torch.Tensor,  # [B] int32 -- persistent context length per request
    ) -> torch.Tensor:
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, ctx_len)
        return self.norm(hidden_states)

    @torch.no_grad()
    def precompute_context_kv(
        self,
        captured_hidden: torch.Tensor,  # [N, num_capture_layers * hidden] (target_layer_ids order)
        position_ids: torch.Tensor,  # [N] absolute positions of the accepted context tokens
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project accepted target hidden states into per-layer drafter context K/V (eager).

        Ports the oracle ``modeling_speculative.py::precompute_context_kv``: the context path is
        ``fc -> hidden_norm -> per-layer k_proj/v_proj -> k_norm (K only) -> RoPE (K only)``. It does
        NOT go through each layer's ``input_layernorm`` (that is only on the query stream); ``k_norm``
        is per-token RMSNorm so applying it to the context K alone matches the oracle's post-cat norm.
        Returns per-layer ``k``/``v`` ``[N, num_layers, n_kv, head_dim]`` (no batch dim); the wrapper
        (Step 7) scatters ``k[:, i]``/``v[:, i]`` into draft layer ``i``'s ctx K/V cache at ``positions``.

        Eager (not traced/exported). Runs the projections via ``nn.Linear`` directly (not the sharding
        ops). Per-layer separate ``k_proj``/``v_proj`` are used (our standalone model keeps q/k/v
        separate, AD fuses downstream); the oracle's single fused-KV GEMM is a deferred perf
        optimization (identical math). TP-sharding of this path is a follow-up (v1 world_size==1).
        """
        n = captured_hidden.shape[0]
        ctx = self.hidden_norm(self.fc(captured_hidden)).unsqueeze(0)  # [1, N, hidden]
        cos, sin = self.rotary_emb(ctx, position_ids.unsqueeze(0))  # [1, N, head_dim]

        k_layers, v_layers = [], []
        for layer in self.layers:
            attn = layer.self_attn
            k = attn.k_proj(ctx).view(1, n, attn.num_kv_heads, attn.head_dim)
            k = attn.k_norm(k)
            # RoPE on K only (bsnd, unsqueeze_dim=2); the op rotates both args identically.
            k, _ = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(k, k, cos, sin, 2)
            v = attn.v_proj(ctx).view(1, n, attn.num_kv_heads, attn.head_dim)
            k_layers.append(k.squeeze(0))  # [N, n_kv, head_dim]
            v_layers.append(v.squeeze(0))
        return torch.stack(k_layers, dim=1), torch.stack(v_layers, dim=1)  # [N, L, n_kv, head_dim]


@dataclass
class DFlashDraftOutput(ModelOutput):
    """Draft output: post-final-norm hidden states for the query block."""

    norm_hidden_state: Optional[torch.Tensor] = None


class DFlashDrafterForCausalLM(nn.Module):
    """Thin draft wrapper around ``DFlashModel`` (lm_head is the target's; applied by the wrapper).

    ``base_model_prefix = "model"`` so checkpoint keys map under ``model.``; kept a plain ``nn.Module``
    for v1 (HF ``PreTrainedModel`` wrapping + config_class are wired with the factory, Step 7).
    """

    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DFlashModel(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        ctx_len: torch.Tensor,
    ) -> DFlashDraftOutput:
        norm_hidden_state = self.model(inputs_embeds, position_ids, ctx_len)
        return DFlashDraftOutput(norm_hidden_state=norm_hidden_state)


@dataclass
class DFlashWrapperConfig:
    """Config for DFlashWrapper (mirrors EagleWrapperConfig)."""

    max_draft_len: int
    block_size: int
    mask_token_id: int
    # DFlash shares embed_tokens + lm_head from the target (the draft owns neither).
    load_embedding_from_target: bool = True
    load_lm_head_from_target: bool = True
    sync_before_hidden_state_capture: bool = False


class DFlashWrapper(nn.Module):
    """Combined target + DFlash draft for one-model speculative decoding (mirrors EagleWrapper).

    Holds the target model and the DFlash draft model as submodules. ``forward`` is dual-mode: at
    export time (no ``cache_seq_interface``) it runs the prefill-only path; at inference it runs the
    kv-cache path driven by the ``CachedSequenceInterface``. Embedding + lm_head are taken from the
    target (DFlash shares both).

    NOTE: ``__init__`` + accessors + ``get_export_infos`` wiring are complete and tested via the
    factory-build test. The two forward bodies are the E2E-coupled runtime glue (Step 7 continuation):
    they need the draft loop + hidden-state capture + sampler + cache scatter, and are validated at E2E
    (build_and_run_ad). They are documented stubs below.
    """

    def __init__(
        self, config: DFlashWrapperConfig, target_model: nn.Module, draft_model: nn.Module
    ):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_draft_len = config.max_draft_len
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.load_embedding_from_target = config.load_embedding_from_target
        self.load_lm_head_from_target = config.load_lm_head_from_target
        self.sync_before_hidden_state_capture = config.sync_before_hidden_state_capture

    @property
    def _draft_inner_model(self) -> "DFlashModel":
        return self.draft_model.model

    @property
    def _draft_dtype(self) -> torch.dtype:
        return getattr(self._draft_inner_model, "dtype", None) or torch.bfloat16

    def apply_draft_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed query-block token ids (incl. the mask token), from the target's embedding."""
        embeds = self.target_model.get_input_embeddings()(input_ids)
        return embeds.to(self._draft_dtype)

    def apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project draft hidden states to logits via the target's lm_head."""
        logits = self.target_model.get_output_embeddings()(hidden_states)
        return logits.to(self._draft_dtype)

    def forward(self, cache_seq_interface: Optional[CachedSequenceInterface] = None, **kwargs):
        if cache_seq_interface is not None:
            return self._forward_with_kv_cache(cache_seq_interface)
        return self._forward_prefill_only(**kwargs)

    def _forward_prefill_only(self, input_ids: torch.Tensor, position_ids: torch.Tensor, **kwargs):
        """Export-time prefill path (TODO Step 7 continuation).

        Algorithm (mirrors the cached path's single-iteration shape so export sees the right kwargs):
          1. target forward over (embed(input_ids), position_ids) -> logits -> bonus token.
          2. build the query block [bonus, MASK x (block_size-1)] -> embed via target.
          3. precompute_context_kv(captured_hidden, ctx_positions) -> (eager; scatter handled in the
             cached path). At export the context is empty (ctx_len=0), so the draft attends the block.
          4. draft_model(inputs_embeds=block_embeds, position_ids=query_positions, ctx_len=zeros) ->
             emits dflash_attention per layer.
          5. apply_lm_head -> draft tokens; package speculative output.
        """
        raise NotImplementedError(
            "DFlashWrapper._forward_prefill_only is the Step-7 export-path continuation; "
            "the model builds + exports its draft today via DFlashModel (see test_dflash_model.py)."
        )

    def _forward_with_kv_cache(self, cache_seq_interface: CachedSequenceInterface):
        """Inference path (TODO Step 7 continuation; validated at E2E).

        Loop (mirrors modeling_speculative.py::dflash_forward + the worker):
          1. target verification over the K+1 verify window (tokens_per_gen_step = max_draft_len+1).
          2. compute accepted token counts; gather accepted target hidden states (target_layer_ids order).
          3. precompute_context_kv -> scatter accepted K/V rows into each layer's ctx K/V cache at
             input_pos.
          4. build query block [last_accepted/bonus, MASK, ...] + query_positions = ctx_len + arange.
          5. run the (cached) draft GM -> dflash_attention_with_kvcache per layer.
          6. apply_lm_head -> next_draft_tokens; return speculative bookkeeping.
        """
        raise NotImplementedError(
            "DFlashWrapper._forward_with_kv_cache is the Step-7 runtime continuation, validated at E2E."
        )
