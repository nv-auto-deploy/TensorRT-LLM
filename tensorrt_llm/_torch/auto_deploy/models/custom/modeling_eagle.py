# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Eagle3 model implementation for AutoDeploy.

Eagle3 is a speculative decoding draft model that predicts next tokens based on
hidden states from a target model (e.g., Llama-3.1-8B-Instruct).

This file contains model definitions used for executing Eagle3 speculative decoding in AutoDeploy.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.utils import ModelOutput

from tensorrt_llm._torch.speculative.eagle3 import Eagle3ResourceManager
from tensorrt_llm.llmapi.llm_args import EagleDecodingConfig


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = getattr(config, "rope_theta", 10000.0)
        self.config = config

        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if (
            not hasattr(config, "rope_type")
            or config.rope_type is None
            or config.rope_type == "default"
        ):
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == "ntk":
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            self.base = self.base * (
                (self.factor * max_position_embeddings / orig_max_position_embeddings)
                - (self.factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
            )

            self.max_seq_len_cached = orig_max_position_embeddings
        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


class EagleRMSNorm(nn.Module):
    """RMSNorm implementation that uses the torch_rmsnorm custom op."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = torch.ops.auto_deploy.torch_rmsnorm(
            hidden_states, self.weight, self.variance_epsilon
        )
        return result


class EagleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Eagle3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.is_causal = True

        # Note: Eagle3Attention expects 2 * hidden_size input, which is the concatenation of the hidden states
        # and the input embeddings.

        self.q_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            2 * config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        # Projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to [Batch, Seq, Heads, Dim]
        query_states = query_states.view(bsz, q_len, -1, self.head_dim)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=2
        )

        attn_output = torch.ops.auto_deploy.torch_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=self.is_causal,
            layout="bsnd",
        )

        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Eagle3DecoderLayer(nn.Module):
    """Eagle decoder layer with modified attention and hidden state normalization."""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.dtype = config.torch_dtype
        self.self_attn = Eagle3Attention(config, layer_idx=layer_idx)
        self.hidden_norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = EagleMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embeds: torch.Tensor,
        position_embeds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embeddings could have a different dtype if they come from the target model.
        embeds = embeds.to(self.dtype)

        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)

        embeds = self.input_layernorm(embeds)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeds,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Eagle3Model(nn.Module):
    """Core Eagle model architecture."""

    def __init__(self, config):
        super().__init__()

        self.dtype = config.torch_dtype

        load_embedding_from_target = getattr(config, "load_embedding_from_target", False)
        self.embed_tokens = (
            None
            if load_embedding_from_target
            else nn.Embedding(config.vocab_size, config.hidden_size)
        )

        if config.draft_vocab_size is not None and config.draft_vocab_size != config.vocab_size:
            # Vocab mappings for draft <-> target token conversion
            # Needed to convert draft outputs to target inputs for Eagle3.
            # Since we reuse the target model's embedding in the drafter, we need
            # to do this conversion after every draft iteration.
            self.d2t = nn.Parameter(
                torch.empty((config.draft_vocab_size,), dtype=torch.int32),
                requires_grad=False,
            )

        # Hidden size compression for target hidden states.
        # Assumption: No feedforward fusion needed if we have just one capture layer (valid for MTPEagle)
        self.fc = (
            nn.Linear(
                config.hidden_size * config.num_capture_layers,
                config.hidden_size,
                bias=getattr(config, "bias", False),
                dtype=self.dtype,
            )
            if config.num_capture_layers > 1
            else None
        )

        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )

        self.rotary_emb = LlamaRotaryEmbedding(
            config=config, dim=self.head_dim, device=torch.device("cuda")
        )

        if config.num_hidden_layers > 1:
            self.midlayer = nn.ModuleList(
                [Eagle3DecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
            )
        else:
            self.midlayer = Eagle3DecoderLayer(config, layer_idx=0)

        self.num_hidden_layers = config.num_hidden_layers

    # Assumption: The hidden states are already fused if necessary
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeds = (cos, sin)

        if self.num_hidden_layers > 1:
            for layer in self.midlayer:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    embeds=inputs_embeds,
                    position_embeds=position_embeds,
                )
        else:
            hidden_states = self.midlayer(
                hidden_states=hidden_states,
                embeds=inputs_embeds,
                position_embeds=position_embeds,
            )

        return hidden_states


@dataclass
class Eagle3DraftOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None
    norm_hidden_state: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class Eagle3DrafterForCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper for EagleModel.

    This wrapper makes EagleModel compatible with AutoDeploy's model loading
    and inference pipeline.
    """

    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["Eagle3DecoderLayer"]

    # Checkpoint conversion mapping: Eagle checkpoints have keys like "fc.weight"
    # but the wrapper model expects "model.fc.weight" (due to self.model = Eagle3Model).
    # This mapping tells the factory to add "model." prefix when loading weights.
    # Used by AutoModelForCausalLMFactory._remap_param_names_load_hook()

    _checkpoint_conversion_mapping = {
        "^(?!lm_head|norm)": "model.",  # Prepend "model." to all keys EXCEPT lm_head and norm
    }

    def __init__(self, config):
        super().__init__(config)

        self.load_embedding_from_target = getattr(config, "load_embedding_from_target", False)
        self.load_lm_head_from_target = getattr(config, "load_lm_head_from_target", False)

        self.model = Eagle3Model(config)
        self.norm = EagleRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = (
            None
            if self.load_lm_head_from_target
            else nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        )

        eagle_config = getattr(config, "eagle_config", {})
        self._return_hidden_post_norm = eagle_config.get("return_hidden_post_norm", False)

    def forward(
        self,
        inputs_embeds: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Eagle3DraftOutput:
        """
        Kwargs:
            hidden_states: Hidden states from the target model. Required.

        Raises:
            ValueError: If hidden_states is not provided in kwargs.
        """
        batch_size, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None:
            raise ValueError("hidden_states must be provided.")

        hidden_states = self.model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, hidden_states=hidden_states
        )

        norm_hidden_state = self.norm(hidden_states)

        last_hidden_state = norm_hidden_state if self._return_hidden_post_norm else hidden_states

        return Eagle3DraftOutput(
            norm_hidden_state=norm_hidden_state,
            last_hidden_state=last_hidden_state,
        )


@dataclass
class EagleWrapperOutput(ModelOutput):
    """Output format compatible with Eagle3OneModelSampler/MTPSampler.

    This output format allows the one-model speculative decoding flow to bypass
    logits-based sampling in the sampler. The EagleWrapper performs all sampling
    and verification internally, returning pre-computed tokens.
    """

    # logits: [batch_size, 1, vocab_size].  Used for compatibility.
    logits: Optional[torch.Tensor] = None

    # new_tokens: [batch_size, max_draft_len + 1]. Accepted tokens from verification.
    # This is a 2D tensor where each row contains the accepted tokens for a request,
    # padded if fewer tokens were accepted.
    new_tokens: Optional[torch.Tensor] = None

    # new_tokens_lens: [batch_size]. Number of newly accepted tokens per request in this iteration.
    new_tokens_lens: Optional[torch.Tensor] = None

    # next_draft_tokens: [batch_size, max_draft_len]. Draft tokens for the next iteration.
    # These are the tokens predicted by the draft model, already converted via d2t.
    next_draft_tokens: Optional[torch.Tensor] = None

    # next_new_tokens: [batch_size, max_draft_len + 1]. Input tokens for the next iteration.
    # Format: [last_accepted_token, draft_token_0, draft_token_1, ...]
    next_new_tokens: Optional[torch.Tensor] = None


@dataclass
class EagleWrapperConfig:
    max_draft_len: int
    load_embedding_from_target: bool
    load_lm_head_from_target: bool


class EagleWrapper(nn.Module):
    def __init__(self, config, target_model, draft_model, resource_manager):
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.resource_manager = resource_manager
        self.max_draft_len = config.max_draft_len
        self.load_embedding_from_target = config.load_embedding_from_target
        self.load_lm_head_from_target = config.load_lm_head_from_target

    def apply_eagle3_fc(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the fc layer that fuses hidden states from multiple target layers."""
        draft_model = self.draft_model.model
        hidden_states = hidden_states.to(draft_model.dtype)

        fc = getattr(draft_model, "fc", None)
        if fc is not None:
            hidden_states = fc(hidden_states)
        return hidden_states

    def apply_d2t(self, draft_output_ids: torch.Tensor) -> torch.Tensor:
        """Apply draft-to-target token mapping if available."""
        d2t = getattr(self.draft_model.model, "d2t", None)
        if d2t is not None:
            draft_output_ids = d2t.data[draft_output_ids] + draft_output_ids
        return draft_output_ids

    def _filter_kwargs_for_submodule(
        self,
        kwargs: dict,
        submodule_name: str,
        cache_prefixes: tuple[str, ...] = ("k_cache", "v_cache", "hidden_states_cache"),
    ) -> dict:
        filtered = {}
        for key, value in kwargs.items():
            is_cache_kwarg = any(key.startswith(prefix) for prefix in cache_prefixes)

            if is_cache_kwarg:
                if submodule_name in key:
                    filtered[key] = value
            else:
                filtered[key] = value

        return filtered

    def _debug_print_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        indent: str = "  ",
        max_elements: int = 20,
    ) -> None:
        """Print debug info for a single tensor.

        Args:
            name: Name/label for the tensor
            tensor: The tensor to print
            indent: Indentation prefix
            max_elements: Only print values if tensor has <= this many elements
        """
        print(f"{indent}{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
        if tensor.numel() <= max_elements:
            print(f"{indent}  values={tensor.tolist()}")

    def _debug_print_kwargs(
        self,
        title: str,
        kwargs: dict,
        old_kwargs: dict | None = None,
        indent: str = "  ",
        max_elements: int = 20,
    ) -> None:
        """Print debug info for kwargs dict, optionally showing old->new comparison.

        Args:
            title: Header title for the debug output
            kwargs: The kwargs dict to print
            old_kwargs: Optional old kwargs dict for comparison (prints old -> new)
            indent: Indentation prefix
            max_elements: Only print tensor values if <= this many elements
        """
        print(f"\n{title}")
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                if old_kwargs is not None and key in old_kwargs:
                    old_val = old_kwargs[key]
                    # Print old->new comparison if both tensors are small enough
                    if (
                        isinstance(old_val, torch.Tensor)
                        and old_val.numel() <= max_elements
                        and val.numel() <= max_elements
                    ):
                        print(f"{indent}{key}: {old_val.tolist()} -> {val.tolist()}")
                    else:
                        self._debug_print_tensor(key, val, indent, max_elements)
                else:
                    self._debug_print_tensor(key, val, indent, max_elements)
            else:
                if old_kwargs is not None and key in old_kwargs:
                    print(f"{indent}{key}: {old_kwargs[key]} -> {val}")
                else:
                    print(f"{indent}{key}: {val}")

    def _debug_print_draft_inputs(
        self,
        title: str,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        kwargs: dict | None = None,
        old_kwargs: dict | None = None,
        output: Any = None,
    ) -> None:
        """Print debug info for a draft model forward call.

        Args:
            title: Header title for the debug output
            input_ids: Draft input token IDs (packed format)
            position_ids: Position IDs (packed format)
            hidden_states: Hidden states (packed format), optional
            kwargs: Optional kwargs dict to print
            old_kwargs: Optional old kwargs dict for comparison (prints old -> new)
            output: Optional draft model output to print
        """
        print("\n" + "=" * 60)
        print(f"[EagleWrapper] {title}")
        print("=" * 60)
        self._debug_print_tensor("input_ids (packed)", input_ids)
        self._debug_print_tensor("position_ids (packed)", position_ids)
        if hidden_states is not None:
            print(
                f"  hidden_states (packed): shape={hidden_states.shape}, "
                f"dtype={hidden_states.dtype}"
            )

        if kwargs is not None:
            self._debug_print_kwargs("  kwargs:", kwargs, old_kwargs, indent="    ")

        if output is not None:
            print(f"  output: {output}")
            print(f"  norm_hidden_state: {output.norm_hidden_state.shape}")
            print(f"  last_hidden_state: {output.last_hidden_state.shape}")

        print("=" * 60 + "\n")

    def _recompute_metadata_from_position_ids(
        self,
        kwargs: dict,
        packed_position_ids: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> dict:
        """Recompute all position-related metadata from packed position_ids and seq_lens.

        This is the shared helper used by both _increment_position_metadata and
        _truncate_and_recompute_metadata to derive all metadata from position_ids.

        Args:
            kwargs: Current kwargs dict (already filtered for submodule). Used to get
                page_size from k_cache and to copy non-metadata keys.
            packed_position_ids: Position IDs in packed format [1, total_tokens].
            seq_lens: Number of tokens per sequence [num_seq]. Can be a tensor or list.

        Returns:
            New kwargs dict with all metadata recomputed from position_ids.
        """
        device = packed_position_ids.device

        # Convert seq_lens to tensor if needed
        if not isinstance(seq_lens, torch.Tensor):
            seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        else:
            seq_lens_tensor = seq_lens.int().to(device)

        num_seq = len(seq_lens_tensor)

        # Get page_size from k_cache shape: [num_pages, page_size, num_kv_heads, head_dim]
        page_size = None
        for key, val in kwargs.items():
            if key.startswith("k_cache") and isinstance(val, torch.Tensor):
                page_size = val.shape[1]
                break

        if page_size is None:
            raise ValueError("Could not determine page_size from k_cache in kwargs")

        # Compute cu_seqlen: cumulative sequence lengths [0, seq_len[0], seq_len[0]+seq_len[1], ...]
        cu_seqlen = torch.zeros(num_seq + 1, dtype=torch.int32, device=device)
        cu_seqlen[1:] = torch.cumsum(seq_lens_tensor, dim=0)
        cu_seqlen_host = cu_seqlen.cpu()

        # Compute batch_info: [num_prefill, num_prefill_tokens, num_decode]
        # seq_len > 1 is "prefill" (multi-token), seq_len == 1 is "decode"
        num_prefill = (seq_lens_tensor > 1).sum().item()
        num_prefill_tokens = seq_lens_tensor[seq_lens_tensor > 1].sum().item()
        num_decode = (seq_lens_tensor <= 1).sum().item()
        batch_info_host = torch.tensor(
            [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32, device="cpu"
        )

        # Compute seq_len_with_cache for each sequence
        # seq_len_with_cache = last_position + 1 = position_ids[last_token_of_seq] + 1
        # Last token of sequence i is at packed index cu_seqlen[i+1] - 1
        last_token_indices = cu_seqlen[1:] - 1  # [num_seq]
        last_positions = packed_position_ids[0, last_token_indices]  # [num_seq]
        seq_len_with_cache = (last_positions + 1).int()
        seq_len_with_cache_host = seq_len_with_cache.cpu()

        # Compute last_page_len: (seq_len_with_cache - 1) % page_size + 1
        last_page_len = ((seq_len_with_cache - 1) % page_size + 1).int()
        last_page_len_host = last_page_len.cpu()

        # Build new kwargs with recomputed metadata
        metadata_updates = {
            "batch_info_host": batch_info_host,
            "cu_seqlen_host": cu_seqlen_host,
            "cu_seqlen": cu_seqlen,
            "seq_len_with_cache": seq_len_with_cache,
            "seq_len_with_cache_host": seq_len_with_cache_host,
            "last_page_len": last_page_len,
            "last_page_len_host": last_page_len_host,
        }

        new_kwargs = {}
        for key, val in kwargs.items():
            if key in metadata_updates:
                new_kwargs[key] = metadata_updates[key]
            else:
                # Pass through unchanged (caches, cache_loc, pages_per_seq, cu_num_pages, etc.)
                new_kwargs[key] = val

        return new_kwargs

    def _prepare_next_draft_metadata(
        self, kwargs: dict, curr_position_ids: torch.Tensor, curr_cu_seq_len: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        # TODO: Need to handle the case where the increment wraps around a page boundary.

        last_token_indices = curr_cu_seq_len[1:] - 1  # [num_sequences]
        last_positions = curr_position_ids[0, last_token_indices].unsqueeze(0)  # [num_sequences]
        new_position_ids = last_positions + 1

        # Each draft iteration is a standard decode step.
        seq_lens = torch.ones(
            new_position_ids.shape[1], dtype=torch.int32, device=curr_position_ids.device
        )

        # Recompute all metadata from the new packed position_ids
        new_kwargs = self._recompute_metadata_from_position_ids(kwargs, new_position_ids, seq_lens)

        return new_position_ids, new_kwargs

    def _extract_draft_iteration_outputs(
        self,
        draft_idx: int,
        draft_output: Any,
        draft_output_logits: torch.Tensor,
        draft_kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract logits and hidden state from draft model output.

        This helper encapsulates the iteration-dependent indexing logic:
        - iter 0 (prefill-like): Variable sequence lengths, use cu_seqlen to find last positions
        - iter 1+ (decode-like): Each sequence has exactly 1 token, simple indexing

        Args:
            draft_idx: Current draft iteration index (0 = first/prefill-like, 1+ = decode-like)
            draft_output: Output from draft model forward
            draft_output_logits: Logits from lm_head, packed [1, total_tokens, vocab_size]
            draft_kwargs: kwargs used for this iteration (contains cu_seqlen for iter 0)

        Returns:
            Tuple of:
            - latest_draft_logits: [num_seq, vocab_size] - logits at last position of each sequence
            - last_draft_hidden_state: [num_seq, hidden_size] - hidden state at last position
        """
        if draft_idx == 0:
            # First call: use cu_seqlen to find last token positions
            # cu_seqlen = [0, len0, len0+len1, ...], so last token of seq i is at cu_seqlen[i+1]-1
            cu_seqlen = draft_kwargs["cu_seqlen"]
            last_token_indices = cu_seqlen[1:] - 1  # [num_sequences]
            latest_draft_logits = draft_output_logits[0, last_token_indices, :]  # [num_seq, vocab]
            last_draft_hidden_state = draft_output.last_hidden_state[
                0, last_token_indices, :
            ]  # [num_seq, hidden]
        else:
            # Subsequent calls: decode_cu_seqlen = [0, 1, 2, ..., num_sequences]
            # Last token of seq i is at position i (since each seq has 1 token)
            # Simplifies to taking all tokens: output[0, :, :] = [num_seq, ...]
            latest_draft_logits = draft_output_logits[0, :, :]  # [num_seq, vocab]
            last_draft_hidden_state = draft_output.last_hidden_state[0, :, :]  # [num_seq, hidden]

        return latest_draft_logits, last_draft_hidden_state

    def _truncate_and_recompute_metadata(
        self,
        kwargs: dict,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Convert padded input_ids/position_ids to packed format and recompute metadata.

        This function takes padded tensors and num_accepted_tokens, converts to packed
        format, and recomputes all derived metadata using the shared helper.

        Use case: After speculative decoding verification, we may have processed
        max_draft_len + 1 tokens but only num_accepted_tokens were actually accepted.
        This function packs to the accepted length and fixes up all metadata.

        Args:
            kwargs: Current kwargs dict (already filtered for the submodule).
            input_ids: Current input_ids tensor of shape [batch, seq_len] (padded).
            position_ids: Current position_ids tensor of shape [batch, seq_len] (padded).
            num_accepted_tokens: [batch_size] tensor with number of accepted tokens per sequence.

        Returns:
            Tuple of (packed_input_ids, packed_position_ids, new_kwargs) where:
            - packed_input_ids is in packed format [1, total_tokens]
            - packed_position_ids is in packed format [1, total_tokens]
            - new_kwargs has all metadata recomputed from packed position_ids
        """
        self._debug_print_draft_inputs(
            "_truncate_and_recompute_metadata INPUT",
            input_ids,
            position_ids,
            kwargs=kwargs,
        )
        print(f"  num_accepted_tokens: {num_accepted_tokens.tolist()}")

        # Convert to packed format directly using _pack_padded
        # _pack_padded extracts only the first seq_lens[i] elements from each row
        seq_lens = num_accepted_tokens.tolist()
        packed_input_ids = self._pack_padded(input_ids, seq_lens)  # [1, total_tokens]
        packed_position_ids = self._pack_padded(position_ids, seq_lens)  # [1, total_tokens]

        # Recompute all metadata from packed position_ids using shared helper
        new_kwargs = self._recompute_metadata_from_position_ids(
            kwargs, packed_position_ids, num_accepted_tokens
        )

        # Debug print the output with old->new comparison
        self._debug_print_draft_inputs(
            "_truncate_and_recompute_metadata OUTPUT",
            packed_input_ids,
            packed_position_ids,
            kwargs=new_kwargs,
            old_kwargs=kwargs,
        )

        return packed_input_ids, packed_position_ids, new_kwargs

    def apply_draft_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply embedding to input_ids for the draft model."""
        if self.load_embedding_from_target:
            return self.target_model.get_input_embeddings()(input_ids)
        else:
            return self.draft_model.model.embed_tokens(input_ids)

    def apply_lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply lm_head to get logits from hidden states."""
        if self.load_lm_head_from_target:
            return self.target_model.get_output_embeddings()(hidden_states)
        else:
            return self.draft_model.lm_head(hidden_states)

    def sample_greedy(self, logits: torch.Tensor) -> torch.Tensor:
        ret = torch.argmax(logits, dim=-1)
        return ret

    def prepare_first_drafter_inputs(
        self,
        target_output_ids: torch.Tensor,
        target_position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        kwargs: dict,
        cu_seqlen: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict, list[int]]:
        """Prepare packed inputs for the first draft iteration.

        Takes padded target outputs and flat hidden states, converts everything to
        packed format for the draft model kernels.

        NOTE: This method is NOT CUDA graph compatible due to data-dependent
        tensor slicing. Will need to be reworked for CUDA graph support.

        Args:
            target_output_ids: [num_sequences, max_seq_len] - output token ids from target sampling/verification
            target_position_ids: [num_sequences, max_seq_len] - position ids from target model
            hidden_states: [total_tokens, hidden_size] - hidden states from target model (flat)
            num_accepted_tokens: [num_sequences] - number of accepted tokens per sequence
            kwargs: kwargs dict (already filtered for the submodule) containing metadata to truncate
            cu_seqlen: [num_sequences + 1] - cumulative sequence lengths for indexing into flat hidden_states

        Returns:
            Tuple of:
            - packed_output_ids: [1, total_tokens] - packed format
            - packed_position_ids: [1, total_tokens] - packed format
            - packed_hidden_states: [1, total_tokens, hidden_size] - packed format
            - truncated_kwargs: dict with recomputed metadata for the packed sequences
            - seq_lens: list[int] - actual lengths per sequence
        """
        num_sequences = target_output_ids.shape[0]

        print("In prepare_first_drafter_inputs()")
        print(
            f"num_sequences: {num_sequences}, num_accepted_tokens.shape: {num_accepted_tokens.shape}"
        )
        print(f"num_accepted_tokens: {num_accepted_tokens}")

        if num_sequences != num_accepted_tokens.shape[0]:
            raise ValueError(
                f"num_sequences ({num_sequences}) != num_accepted_tokens.shape[0] ({num_accepted_tokens.shape[0]})"
            )

        # Convert input_ids and position_ids to packed format and recompute metadata
        # _truncate_and_recompute_metadata now returns packed tensors directly
        packed_output_ids, packed_position_ids, truncated_kwargs = (
            self._truncate_and_recompute_metadata(
                kwargs, target_output_ids, target_position_ids, num_accepted_tokens
            )
        )

        # Compute seq_lens from num_accepted_tokens (actual lengths per sequence)
        seq_lens = num_accepted_tokens.tolist()

        # Truncate hidden_states per-sequence using cu_seqlen
        # Each sequence i has tokens at hidden_states[cu_seqlen[i]:cu_seqlen[i+1]]
        # We want the first num_accepted_tokens[i] hidden states from each
        truncated_hidden_list = []
        for i in range(num_sequences):
            start = int(cu_seqlen[i].item())
            n_accepted = int(num_accepted_tokens[i].item())
            truncated_hidden_list.append(hidden_states[start : start + n_accepted, :])

        # Concatenate truncated hidden states - already in packed format [total_tokens, hidden_size]
        # Add batch dimension to get [1, total_tokens, hidden_size]
        packed_hidden_states = torch.cat(truncated_hidden_list, dim=0).unsqueeze(0)

        return (
            packed_output_ids,
            packed_position_ids,
            packed_hidden_states,
            truncated_kwargs,
            seq_lens,
        )

    def sample_prefill(
        self,
        input_ids: torch.Tensor,
        target_logits: torch.Tensor,
        num_prefill: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample new tokens for prefill sequences (no verification needed).

        For prefill, all input tokens are the prompt (already "accepted").
        We just sample a new token from the last logit position.

        Args:
            input_ids: [batch_size, seq_len] - input token ids
            target_logits: [batch_size, seq_len, vocab_size] - logits from target model
            num_prefill: Number of prefill sequences (first num_prefill in batch)

        Returns:
            output_ids: [num_prefill, seq_len] - input_ids[1:] + sampled token (shifted for draft)
            num_newly_accepted_tokens: [num_prefill] - always 0 for prefill (no verification)
            num_accepted_tokens: [num_prefill] - always seq_len (all prompt tokens accepted)
            last_logits: [num_prefill, 1, vocab_size] - the last logit used for sampling,
                formatted as [batch, 1, vocab] for easy concatenation with draft logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if num_prefill > batch_size:
            raise ValueError(f"num_prefill ({num_prefill}) > batch_size ({batch_size})")

        # Only process prefill sequences
        prefill_input_ids = input_ids[:num_prefill]
        prefill_logits = target_logits[:num_prefill]

        # Sample new token from the last position's logits for each sequence
        # last_logits: [num_prefill, vocab_size]
        last_logits = prefill_logits[:, -1, :]
        # sampled_tokens: [num_prefill]
        sampled_tokens = self.sample_greedy(last_logits)

        # Construct output_ids: [input_ids[1:], sampled_token]
        # This shifts the input by 1 (drops first token) and appends the sampled token
        # Shape: [num_prefill, seq_len]
        output_ids = torch.cat(
            [prefill_input_ids[:, 1:], sampled_tokens.unsqueeze(1)],
            dim=1,
        )

        # For prefill, no tokens are "newly accepted" (all were already the prompt)
        num_newly_accepted_tokens = torch.zeros(num_prefill, dtype=torch.long, device=device)

        # For prefill, all tokens are accepted (the full prompt)
        num_accepted_tokens = torch.full((num_prefill,), seq_len, dtype=torch.long, device=device)

        # Return last_logits with shape [num_prefill, 1, vocab_size] for easy concatenation
        # This is the logit that was used to sample the first output token
        last_logits_3d = last_logits.unsqueeze(1)

        return output_ids, num_newly_accepted_tokens, num_accepted_tokens, last_logits_3d

    def _unflatten_to_padded_3d(
        self,
        flat_tensor: torch.Tensor,
        seq_lens: list[int],
        cu_seqlen: torch.Tensor,
    ) -> torch.Tensor:
        """Unflatten flat 2D tensor to padded 3D using sequence lengths.

        Args:
            flat_tensor: [total_tokens, hidden_size] flat tensor
            seq_lens: list of actual sequence lengths for each sequence
            cu_seqlen: cumulative sequence lengths [0, len0, len0+len1, ...]

        Returns:
            Padded 3D tensor [num_sequences, max_seq_len, hidden_size]
        """
        num_sequences = len(seq_lens)
        max_seq_len = max(seq_lens)
        hidden_size = flat_tensor.shape[-1]

        padded = torch.zeros(
            (num_sequences, max_seq_len, hidden_size),
            device=flat_tensor.device,
            dtype=flat_tensor.dtype,
        )

        for i in range(num_sequences):
            start = int(cu_seqlen[i].item())
            end = int(cu_seqlen[i + 1].item())
            padded[i, : seq_lens[i], :] = flat_tensor[start:end, :]

        return padded

    def _unpack_to_padded(
        self,
        packed_tensor: torch.Tensor,
        cu_seqlen: torch.Tensor,
        seq_lens: list[int],
        num_sequences: int,
        pad_value: int = 0,
    ) -> torch.Tensor:
        """Unpack packed [1, total_tokens, ...] tensor to padded [num_sequences, max_len, ...].

        Args:
            packed_tensor: Packed tensor of shape [1, total_tokens] or [1, total_tokens, extra_dims...]
            cu_seqlen: Cumulative sequence lengths [0, len0, len0+len1, ...]
            seq_lens: List of actual sequence lengths for each sequence
            num_sequences: Number of sequences
            pad_value: Value to use for padding (default 0)

        Returns:
            Padded tensor of shape [num_sequences, max_len] or [num_sequences, max_len, extra_dims...]
        """
        max_len = max(seq_lens)
        extra_dims = packed_tensor.shape[2:]  # e.g., vocab_size for logits

        padded = torch.full(
            (num_sequences, max_len) + extra_dims,
            pad_value,
            dtype=packed_tensor.dtype,
            device=packed_tensor.device,
        )

        for i in range(num_sequences):
            start = (
                int(cu_seqlen[i].item())
                if isinstance(cu_seqlen, torch.Tensor)
                else int(cu_seqlen[i])
            )
            length = seq_lens[i]
            padded[i, :length] = packed_tensor[0, start : start + length]

        print(f"[EagleWrapper._unpack_to_padded] packed_tensor: {packed_tensor}")
        print(f"[EagleWrapper._unpack_to_padded] packed_tensor.shape: {packed_tensor.shape}")
        print(f"[EagleWrapper._unpack_to_padded] cu_seqlen: {cu_seqlen}")
        print(f"[EagleWrapper._unpack_to_padded] num_sequences: {num_sequences}")

        print(f"[EagleWrapper._unpack_to_padded] padded: {padded}")
        print(f"[EagleWrapper._unpack_to_padded] padded.shape: {padded.shape}")

        return padded

    @torch.compiler.disable
    def _pack_padded(
        self,
        padded_tensor: torch.Tensor,
        seq_lens: list[int],
    ) -> torch.Tensor:
        """Convert padded [num_seq, max_len, ...] to packed [1, total_tokens, ...].

        Preserves trailing dimensions (e.g., hidden_size for 3D tensors).

        Note: @torch.compiler.disable is used because this function is called with
        different tensor dtypes (int32/int64 for input_ids, float16 for hidden_states),
        which causes excessive recompilation. The function is simple enough that
        compilation provides minimal benefit anyway.

        Args:
            padded_tensor: Padded tensor of shape [num_seq, max_len] or [num_seq, max_len, hidden_size]
            seq_lens: List of actual (non-padded) lengths for each sequence

        Returns:
            Packed tensor of shape [1, total_tokens] or [1, total_tokens, hidden_size]
        """
        num_sequences = padded_tensor.shape[0]

        # Collect non-padding tokens from each sequence
        packed_list = []
        for i in range(num_sequences):
            packed_list.append(padded_tensor[i, : seq_lens[i]])

        # Concatenate along sequence dimension
        packed_flat = torch.cat(packed_list, dim=0)

        # Add "batch dimension" of 1
        return packed_flat.unsqueeze(0)

    def sample_and_verify(
        self, input_ids, target_logits: torch.Tensor, num_previously_accepted: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            target_logits: [batch_size, seq_len, vocab_size]
            num_previously_accepted: [batch_size]. Number of input tokens accepted so far for each batch.

        Returns:
            output_ids: [batch_size, seq_len] (result of greedy sampling on input ids)
            num_newly_accepted_tokens: [batch_size]. Number of newly accepted tokens in each batch.
            num_accepted_tokens: [batch_size]. Number of tokens accepted in each batch, including previously accepted.
                So num_accepted_tokens[i] = num_previously_accepted + num_newly_accepted_tokens.
            last_logits_3d: [batch_size, 1, vocab_size]. The logit used to sample the bonus token.

        How it works:
        - Get input ids that were not previously accepted.
        - Get the corresponding target logits to these input ids (target_logit[j-1] corresponds to input_ids[j])
        - Sample a token from the logits for each batch and compare to input_ids to get the newly accepted tokens.
        - The output_ids consist of the previously accepted tokens, the newly accepted tokens,
        and a newly sampled token after the last accepted token.
        """

        batch_size, seq_len = input_ids.shape

        # During export/tracing with meta tensors, we cannot perform actual verification
        # because operations like .item(), .all(), and data-dependent indexing fail on meta tensors.
        # Return simulated values with correct shapes instead.
        if input_ids.device.type == "meta":
            # Simulate accepting all tokens (optimistic case for shape tracing)
            draft_input_ids = input_ids.clone()
            num_newly_accepted_tokens = (
                torch.full((batch_size,), seq_len, dtype=torch.long, device=input_ids.device)
                - num_previously_accepted
            )
            num_accepted_tokens = torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=input_ids.device
            )
            last_logits_3d = target_logits[:, -1:, :]  # [batch_size, 1, vocab_size]
            return draft_input_ids, num_newly_accepted_tokens, num_accepted_tokens, last_logits_3d

        # First, check that num_previously_accepted is <= seq_len for each batch
        # Additionally, num_previously_accepted should be >= 1 for each batch,
        # which corresponds to having some context tokens (context tokens are always accepted).
        assert (num_previously_accepted >= 1).all(), (
            "num_previously_accepted must be >= 1. Please provide non-empty context in each batch."
        )
        assert (num_previously_accepted <= seq_len).all(), (
            "num_previously_accepted must be <= seq_len for each batch"
        )

        # We get input tokens that were not yet accepted.
        unchecked_input_ids = [
            input_ids[i, num_previously_accepted[i] : seq_len].unsqueeze(0)
            for i in range(batch_size)
        ]

        # We get the corresponding target logits for the unchecked input tokens.
        # logit j-1 corresponds to input j.
        # Note that because of our check that num_previously_accepted is >= 1
        # We also get the last output token for each batch, because we may need to append it
        # at the end.
        unchecked_target_logits = [
            target_logits[i, (num_previously_accepted[i] - 1) : seq_len, :].unsqueeze(0)
            for i in range(batch_size)
        ]

        unchecked_output_ids = [self.sample_greedy(x) for x in unchecked_target_logits]

        # corresponding_output_ids: [batch_size, seq_len - 1]. The output ids that correspond to the unchecked input ids
        # Omits the last index because that corresponds to a freshly sampled output model.
        corresponding_output_ids = [output_id[:, :-1] for output_id in unchecked_output_ids]

        # After sample_greedy, corresponding_output_ids should have same shape as unchecked_input_ids
        assert [x.shape for x in unchecked_input_ids] == [
            x.shape for x in corresponding_output_ids
        ], "unchecked_input_ids and corresponding_output_ids must have the same shape"

        matches = [
            (corresponding_output_ids[i] == unchecked_input_ids[i]).int() for i in range(batch_size)
        ]

        # Compute num_newly_accepted_tokens per batch (handles different sizes across batches)
        num_newly_accepted_tokens = []
        for i in range(batch_size):
            if matches[i].numel() == 0:
                # No unchecked tokens for this batch (num_previously_accepted == seq_len)
                num_newly_accepted_tokens.append(
                    torch.tensor(0, dtype=torch.long, device=input_ids.device)
                )
            else:
                # prefix_matches[j] is 1 if first j+1 tokens all matched
                prefix_matches = matches[i].cumprod(dim=-1)
                num_newly_accepted_tokens.append(prefix_matches.sum().long())
        num_newly_accepted_tokens = torch.stack(num_newly_accepted_tokens)

        # num_accepted_tokens: [batch_size]. The total number of accepted tokens in each batch,
        # including previously accepted tokens.
        num_accepted_tokens = num_previously_accepted + num_newly_accepted_tokens

        assert (num_accepted_tokens <= seq_len).all(), (
            "num_accepted_tokens must be <= seq_len for each batch"
        )

        # Construct draft_input_ids for the draft model
        # For each sequence:
        # 1. Take previously accepted tokens (skipping the first one)
        # 2. Append newly accepted tokens directly from input_ids.
        # 3. Append the sampled token for last accepted position: unchecked_output_ids[0][num_newly_accepted]
        # 4. Fill the rest with zeros (padding)
        # Total real tokens: (num_previously_accepted - 1) + num_newly_accepted + 1 = num_accepted_tokens

        draft_input_ids = torch.zeros(
            (batch_size, seq_len), dtype=input_ids.dtype, device=input_ids.device
        )

        for i in range(batch_size):
            # 1. Previously accepted tokens (skip the first one in keeping with Eagle convention)
            # Note that this potentially includes context tokens, but is structured this way because we
            # want the output to contain the entire prefix of accepted tokens because the drafters have no KV cache.
            prev_accepted = input_ids[i, 1 : num_previously_accepted[i]]

            # 2. Newly accepted input tokens
            newly_accepted = input_ids[
                i,
                num_previously_accepted[i] : num_previously_accepted[i]
                + num_newly_accepted_tokens[i],
            ]

            # 3. The sampled output token for the last accepted position
            # unchecked_output_ids[i][j] is the sampled token for position (num_previously_accepted + j)
            # We want the token for position num_accepted_tokens, which is index num_newly_accepted_tokens
            next_token = unchecked_output_ids[i][0][num_newly_accepted_tokens[i]].unsqueeze(0)

            # Concatenate all parts
            draft_prefix = torch.cat([prev_accepted, newly_accepted, next_token])

            # Sanity check: draft_prefix length should equal num_accepted_tokens
            assert draft_prefix.shape[0] == num_accepted_tokens[i], (
                f"draft_prefix length {draft_prefix.shape[0]} != num_accepted_tokens {num_accepted_tokens[i]}"
            )

            # Fill into draft_input_ids (rest remains zeros as padding)
            draft_input_ids[i, : num_accepted_tokens[i]] = draft_prefix

        # Construct last_logits_3d: [batch_size, 1, vocab_size]
        # This is the logit used to sample the bonus token for each sequence.
        # The bonus token is sampled from unchecked_target_logits[i][0][num_newly_accepted_tokens[i]]
        last_logits_list = []
        for i in range(batch_size):
            # unchecked_target_logits[i] has shape [1, num_unchecked + 1, vocab_size]
            # Index num_newly_accepted_tokens[i] gives the logit for the bonus token
            bonus_logit = unchecked_target_logits[i][0, num_newly_accepted_tokens[i], :].unsqueeze(
                0
            )
            last_logits_list.append(bonus_logit)
        last_logits_3d = torch.stack(last_logits_list, dim=0)  # [batch_size, 1, vocab_size]

        return draft_input_ids, num_newly_accepted_tokens, num_accepted_tokens, last_logits_3d

    def forward(self, input_ids, position_ids, **kwargs):
        """Dispatch to appropriate forward implementation based on kwargs.

        If num_previously_accepted is provided, use the prefill-only (no KV cache) implementation.
        Otherwise, use the KV cache implementation (for generation with cached attention).
        """
        num_previously_accepted = kwargs.get("num_previously_accepted", None)

        if num_previously_accepted is not None:
            return self._forward_prefill_only(input_ids, position_ids, **kwargs)
        else:
            return self._forward_with_kv_cache(input_ids, position_ids, **kwargs)

    def _forward_with_kv_cache(self, input_ids, position_ids, **kwargs):
        """Forward pass with KV cache support (for generation with cached attention).

        Uses batch_info to distinguish prefill from decode/spec-dec sequences:
        - Prefill (input_pos == 0): All tokens are prompt, use sample_prefill
        - Decode/Spec-dec (input_pos > 0): TODO - needs verification logic

        TODO: Add drafting loop for speculative decoding.
        """
        print(f"[EagleWrapper._forward_with_kv_cache] input_ids: {input_ids}")
        print(f"[EagleWrapper._forward_with_kv_cache] input_ids.shape: {input_ids.shape}")

        # Extract batch_info from kwargs: [num_multi_token, num_multi_token_total, num_single_token]
        # NOTE: batch_info counts multi-token requests as "prefill" for attention kernels.
        # For spec dec, multi-token requests can be EITHER:
        #   - True context/prefill (prompt processing, position_ids starts at 0)
        #   - Spec dec decode (verification, position_ids starts > 0)
        # We use position_ids to distinguish these cases.
        batch_info_host = kwargs.get("batch_info_host", None)
        if batch_info_host is None:
            raise ValueError("batch_info_host must be provided in kwargs. ")

        num_multi_token, _num_multi_token_total, num_single_token = batch_info_host.tolist()

        # With spec dec enabled, all requests are multi-token (context or spec dec verification)
        # Single-token decode doesn't happen in one-model spec dec flow
        assert num_single_token == 0, (
            f"Single-token requests not expected in spec dec flow. Got num_single_token={num_single_token}"
        )

        # Total number of sequences = num_multi_token (since num_single_token == 0)
        num_sequences = num_multi_token

        # Get cu_seqlen_host for sequence boundaries
        # NOTE: Data is in PACKED format: input_ids is [1, total_tokens], not [num_sequences, seq_len]
        # cu_seqlen provides boundaries: [0, len0, len0+len1, ...]
        cu_seqlen_host = kwargs.get("cu_seqlen_host", None)
        if cu_seqlen_host is None:
            raise ValueError(
                "cu_seqlen_host must be provided in kwargs for packed sequence format."
            )

        # Compute sequence lengths and start positions from cu_seqlen
        seq_lens = []
        seq_starts = []
        is_tensor = isinstance(cu_seqlen_host, torch.Tensor)
        for i in range(num_sequences):
            start = int(cu_seqlen_host[i].item()) if is_tensor else int(cu_seqlen_host[i])
            end = int(cu_seqlen_host[i + 1].item()) if is_tensor else int(cu_seqlen_host[i + 1])
            seq_lens.append(end - start)
            seq_starts.append(start)

        # Count true context requests vs spec dec decode requests
        # True context: first position_id of sequence == 0 (sequence starts from beginning)
        # Spec dec decode: first position_id > 0 (continuing an existing sequence)
        # NOTE: position_ids is [1, total_tokens], so we gather first positions using seq_starts
        # Use tensor indexing to be torch.compile compatible
        seq_starts_tensor = torch.tensor(seq_starts, device=position_ids.device, dtype=torch.long)
        first_positions = position_ids[0, seq_starts_tensor]  # [num_sequences]

        # Build context and spec_dec index lists from the gathered first positions
        # TODO: We should probably pass this in through batch_info_host instead of computing it here
        context_indices = []
        spec_dec_indices = []
        first_positions_cpu = first_positions.tolist()
        for i in range(num_sequences):
            if first_positions_cpu[i] == 0:
                context_indices.append(i)
            else:
                spec_dec_indices.append(i)

        num_context = len(context_indices)
        num_spec_dec = len(spec_dec_indices)

        print(
            f"[EagleWrapper] num_sequences={num_sequences}, "
            f"num_context={num_context}, num_spec_dec={num_spec_dec}"
        )
        print(f"[EagleWrapper] seq_lens={seq_lens}, seq_starts={seq_starts}")
        print(
            f"[EagleWrapper] context_indices={context_indices}, spec_dec_indices={spec_dec_indices}"
        )

        # Compute embeddings using the target embedding layer
        inputs_embeds = self.target_model.get_input_embeddings()(input_ids)

        # Filter kwargs to only include target_model entries (exclude draft_model caches)
        target_kwargs = self._filter_kwargs_for_submodule(kwargs, "target_model")

        # Run target model with KV cache
        # target_logits: [batch_size, seq_len, vocab_size]
        target_logits = self.target_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids, **target_kwargs
        ).logits

        # Capture hidden states after target model forward pass.
        # In one-model flow, hidden_states_cache_* buffers are passed via kwargs.
        # We collect them and use store_hidden_states() to copy to resource_manager.hidden_states.
        assert isinstance(self.resource_manager, ADHiddenStateManager), (
            f"Expected resource_manager to be ADHiddenStateManager, got {type(self.resource_manager)}"
        )

        # Collect hidden_states_cache buffers from kwargs (e.g., hidden_states_cache_0, _1, _2)
        # Sort by name to ensure correct order: hidden_states_cache_0, _1, _2, etc.
        hidden_state_items = sorted(
            [
                (name, tensor)
                for name, tensor in kwargs.items()
                if name.startswith("hidden_states_cache")
            ],
            key=lambda x: x[0],
        )

        # Debug: Print the order of hidden_states_cache buffers
        hidden_state_names = [name for name, _ in hidden_state_items]
        print(
            f"[EagleWrapper._forward_with_kv_cache] Hidden state buffer names (in order): "
            f"{hidden_state_names}"
        )
        for name, tensor in hidden_state_items:
            print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")

        hidden_state_buffers = [tensor for _, tensor in hidden_state_items]

        # In Eagle3 one-model flow, we must have hidden state buffers (one per captured layer)
        assert len(hidden_state_buffers) > 0, (
            "No hidden_states_cache buffers found in kwargs. "
            "Eagle3 one-model flow requires hidden state capture."
        )

        # Store hidden states using cu_seqlen to get total token count
        # cu_seqlen_host[num_sequences] gives total tokens across all sequences
        if isinstance(cu_seqlen_host, torch.Tensor):
            num_tokens = int(cu_seqlen_host[num_sequences].item())
        else:
            num_tokens = cu_seqlen_host[num_sequences]
        print(f"[EagleWrapper._forward_with_kv_cache] num_tokens from cu_seqlen_host: {num_tokens}")
        self.resource_manager.store_hidden_states(hidden_state_buffers, num_tokens)

        # Get device from input_ids
        device = input_ids.device

        print(
            "[EagleWrapper._forward_with_kv_cache] Done with KV cache storage, starting sampling of target logits."
        )
        print(f"num_context: {num_context}, num_spec_dec: {num_spec_dec}")

        # ============================================================
        # Unpack packed tensors to padded format for sample_prefill / sample_and_verify
        # These methods expect [batch_size, seq_len] format, not packed [1, total_tokens]
        # ============================================================
        padded_input_ids = self._unpack_to_padded(
            input_ids, cu_seqlen_host, seq_lens, num_sequences
        )
        padded_logits = self._unpack_to_padded(
            target_logits, cu_seqlen_host, seq_lens, num_sequences
        )
        padded_position_ids = self._unpack_to_padded(
            position_ids, cu_seqlen_host, seq_lens, num_sequences
        )

        print("[EagleWrapper] Unpacked to padded format:")
        print(f"  padded_input_ids: {padded_input_ids.shape}")
        print(f"  padded_logits: {padded_logits.shape}")
        print(f"  padded_position_ids: {padded_position_ids.shape}")

        # Get vocab_size for allocating result tensors
        vocab_size = target_logits.shape[-1]
        max_seq_len = max(seq_lens)

        # ============================================================
        # Process context and spec_dec sequences separately, then merge
        # ============================================================

        # Allocate result tensors
        output_ids = torch.zeros(
            (num_sequences, max_seq_len), dtype=padded_input_ids.dtype, device=device
        )
        num_accepted_tokens = torch.zeros(num_sequences, dtype=torch.long, device=device)
        num_newly_accepted_tokens = torch.zeros(num_sequences, dtype=torch.long, device=device)
        last_logits_3d = torch.zeros(
            (num_sequences, 1, vocab_size), dtype=target_logits.dtype, device=device
        )

        # Process CONTEXT (prefill) sequences
        if num_context > 0:
            # Gather context sequences
            ctx_input_ids = padded_input_ids[context_indices]
            ctx_logits = padded_logits[context_indices]

            print(f"[EagleWrapper] Processing {num_context} CONTEXT sequences")
            print(f"  ctx_input_ids: {ctx_input_ids.shape}")
            print(f"  ctx_logits: {ctx_logits.shape}")

            ctx_output_ids, ctx_newly_accepted, ctx_accepted, ctx_last_logits = self.sample_prefill(
                ctx_input_ids, ctx_logits, num_context
            )

            print("[EagleWrapper] CONTEXT results:")
            print(f"  ctx_output_ids: {ctx_output_ids.shape}")
            print(f"  ctx_accepted: {ctx_accepted}")
            print(f"  ctx_newly_accepted: {ctx_newly_accepted}")

            # Scatter results back to original indices
            for local_i, global_i in enumerate(context_indices):
                output_ids[global_i] = ctx_output_ids[local_i]
                num_accepted_tokens[global_i] = ctx_accepted[local_i]
                num_newly_accepted_tokens[global_i] = ctx_newly_accepted[local_i]
                last_logits_3d[global_i] = ctx_last_logits[local_i]

        # Process SPEC_DEC (verification) sequences
        if num_spec_dec > 0:
            # Gather spec_dec sequences
            sd_input_ids = padded_input_ids[spec_dec_indices]
            sd_logits = padded_logits[spec_dec_indices]

            print(f"[EagleWrapper] Processing {num_spec_dec} SPEC_DEC sequences")
            print(f"  sd_input_ids: {sd_input_ids.shape}")
            print(f"  sd_logits: {sd_logits.shape}")

            # For spec dec decode, num_previously_accepted = 1 means the first input token
            # (last_accepted from previous iteration) is "golden" and the rest are draft tokens to verify.
            num_previously_accepted = torch.ones(num_spec_dec, dtype=torch.long, device=device)
            sd_output_ids, sd_newly_accepted, sd_accepted, sd_last_logits = self.sample_and_verify(
                sd_input_ids, sd_logits, num_previously_accepted
            )

            print("[EagleWrapper] SPEC_DEC results:")
            print(f"  sd_output_ids: {sd_output_ids.shape}")
            print(f"  sd_accepted: {sd_accepted}")
            print(f"  sd_newly_accepted: {sd_newly_accepted}")

            # Scatter results back to original indices
            for local_i, global_i in enumerate(spec_dec_indices):
                output_ids[global_i] = sd_output_ids[local_i]
                num_accepted_tokens[global_i] = sd_accepted[local_i]
                num_newly_accepted_tokens[global_i] = sd_newly_accepted[local_i]
                last_logits_3d[global_i] = sd_last_logits[local_i]

        # ============================================================
        # DEBUG: Print merged results and stop to inspect (only for mixed batches)
        # ============================================================
        if num_context > 0 and num_spec_dec > 0:
            print("\n" + "=" * 60)
            print("[EagleWrapper] STEP 1 COMPLETE - Mixed batch merged results:")
            print("=" * 60)
            print(f"output_ids shape: {output_ids.shape}")
            for i in range(num_sequences):
                seq_type = "CONTEXT" if i in context_indices else "SPEC_DEC"
                print(f"  Sequence {i} ({seq_type}):")
                print(f"    output_ids[{i}]: {output_ids[i, : seq_lens[i]].tolist()}")
                print(f"    num_accepted_tokens[{i}]: {num_accepted_tokens[i].item()}")
                print(f"    num_newly_accepted_tokens[{i}]: {num_newly_accepted_tokens[i].item()}")
            print(f"last_logits_3d shape: {last_logits_3d.shape}")
            print("=" * 60 + "\n")

        # Get the hidden states from resource_manager and apply eagle3_fc
        # The hidden states were stored after target model forward (line ~886)
        # resource_manager.hidden_states is [max_tokens, hidden_size * num_capture_layers]
        # hidden_states is flat: [num_tokens, hidden_size] after applying fc
        # This flat buffer is used directly for iter 0 (prefill), no need to copy to a separate buffer
        hidden_states = self.resource_manager.hidden_states[:num_tokens, :]
        print(f"[DEBUG] hidden_states after slicing from resource_manager: {hidden_states.shape}")
        hidden_states = self.apply_eagle3_fc(hidden_states)
        print(f"[DEBUG] hidden_states after apply_eagle3_fc: {hidden_states.shape}")

        # Filter kwargs for draft_model before truncation (exclude target_model caches)
        draft_kwargs_untruncated = self._filter_kwargs_for_submodule(kwargs, "draft_model")

        # Truncate output_ids, position_ids, hidden_states, and kwargs to accepted token lengths for drafting
        # Note: output_ids is already in padded [num_sequences, max_seq_len] format from sample_prefill/verify
        # We pass padded_position_ids (also [num_sequences, max_seq_len]) and cu_seqlen for hidden state slicing
        print(
            f"[DEBUG] Before truncation - hidden_states: {hidden_states.shape}, "
            f"num_accepted_tokens: {num_accepted_tokens}"
        )

        # Get cu_seqlen for hidden state per-sequence slicing
        cu_seqlen = kwargs.get("cu_seqlen")
        if cu_seqlen is None:
            cu_seqlen = cu_seqlen_host.to(device)

        # prepare_first_drafter_inputs now returns packed format directly
        (
            draft_input_ids,  # [1, total_tokens] - packed
            draft_position_ids,  # [1, total_tokens] - packed
            draft_hidden_states,  # [1, total_tokens, hidden_size] - packed
            draft_kwargs,  # metadata recomputed for packed sequences
            draft_seq_lens,
        ) = self.prepare_first_drafter_inputs(
            output_ids,
            padded_position_ids,
            hidden_states,
            num_accepted_tokens,
            draft_kwargs_untruncated,
            cu_seqlen,
        )

        print(
            f"[DEBUG] After prepare_first_drafter_inputs - draft_hidden_states shape: {draft_hidden_states.shape}"
        )
        print(f"[DEBUG] draft_seq_lens: {draft_seq_lens}")
        print("About to start draft loop")
        print(
            f"draft_input_ids: {draft_input_ids.shape}, draft_position_ids: {draft_position_ids.shape}"
        )

        draft_tokens: list[torch.Tensor] = []  # List of tensors, each [num_sequences]

        for draft_idx in range(self.max_draft_len):
            self._debug_print_draft_inputs(
                f"draft_model.forward() iter {draft_idx}",
                draft_input_ids,
                draft_position_ids,
                draft_hidden_states,
                kwargs=draft_kwargs,
            )

            inputs_embeds = self.apply_draft_embedding(draft_input_ids)
            draft_output = self.draft_model(
                inputs_embeds=inputs_embeds,
                position_ids=draft_position_ids,
                hidden_states=draft_hidden_states,
                **draft_kwargs,
            )

            # draft_output.norm_hidden_state is packed 3D: [1, total_tokens, hidden_size]
            draft_output_logits = self.apply_lm_head(draft_output.norm_hidden_state)

            # Extract logits and hidden state for last position in each sequence
            latest_draft_logits, last_draft_hidden_state = self._extract_draft_iteration_outputs(
                draft_idx, draft_output, draft_output_logits, draft_kwargs
            )

            # Sample draft token from logits
            draft_token = self.sample_greedy(latest_draft_logits)
            draft_token = self.apply_d2t(draft_token)

            # Store the draft token tensor for this iteration
            # draft_token has shape [num_sequences]
            draft_tokens.append(draft_token)

            print(
                f"[EagleWrapper] Draft iteration {draft_idx}: sampled token {draft_token.tolist()}"
            )

            if draft_idx < self.max_draft_len - 1:
                draft_position_ids, draft_kwargs = self._prepare_next_draft_metadata(
                    draft_kwargs,
                    curr_position_ids=draft_position_ids,
                    curr_cu_seq_len=draft_kwargs["cu_seqlen"],
                )

                # Prepare tensor inputs for next iteration
                # last_draft_token: [num_sequences] -> [1, num_sequences]
                draft_input_ids = draft_token.unsqueeze(0)
                # last_draft_hidden_state: [num_sequences, hidden_size] -> [1, num_sequences, hidden_size]
                draft_hidden_states = last_draft_hidden_state.unsqueeze(0)

        # ============================================================
        # End of Drafting Loop
        # ============================================================

        print(f"[EagleWrapper] Draft tokens: {draft_tokens}")

        # For one-model spec dec, we only return target logits (not draft logits)
        # The sampler doesn't need draft logits for greedy verification
        output_logits = last_logits_3d  # [batch, 1, vocab_size]
        print(f"[EagleWrapper] output_logits shape: {output_logits.shape}")

        # Get sequence lengths from output_ids
        _, seq_len_out = output_ids.shape

        # Convert new_tokens from output_ids to 2D tensor format expected by MTPSampler
        # For MTPSampler: new_tokens[batch, step] contains the accepted token at that step
        # new_tokens_lens tells us how many tokens were accepted
        #
        # Processing is split by request type:
        # - CONTEXT requests (num_context): sampled token at output_ids[i, seq_len-1]
        # - SPEC_DEC requests (num_spec_dec): verified tokens from sample_and_verify
        new_tokens_2d = torch.zeros(
            (num_sequences, self.max_draft_len + 1),
            dtype=torch.int32,
            device=device,
        )
        new_tokens_lens = torch.zeros(num_sequences, dtype=torch.int32, device=device)

        # Convert draft_tokens list to tensor: [batch_size, max_draft_len]
        # draft_tokens is a list of max_draft_len tensors, each of shape [num_sequences]
        assert len(draft_tokens) == self.max_draft_len, (
            f"Expected {self.max_draft_len} draft tokens, got {len(draft_tokens)}"
        )
        assert all(t.shape == (num_sequences,) for t in draft_tokens), (
            f"All draft tokens should have shape [{num_sequences}]"
        )
        # Stack along dim=1 to get [num_sequences, max_draft_len]
        next_draft_tokens = torch.stack(draft_tokens, dim=1).to(dtype=torch.int32, device=device)

        # Prepare next_new_tokens: [batch_size, max_draft_len + 1]
        # Format: [last_accepted_token, draft_token_0, draft_token_1, ...]
        next_new_tokens = torch.zeros(
            (num_sequences, self.max_draft_len + 1),
            dtype=torch.int32,
            device=device,
        )

        # Process CONTEXT (prefill) requests
        # sample_prefill returns output_ids = [input_ids[1:], sampled_token]
        # The sampled token is at the last position
        # Use context_indices to map local index to global sequence index
        for local_i in range(num_context):
            global_i = context_indices[local_i]
            sampled_token = output_ids[global_i, seq_len_out - 1].to(torch.int32)
            new_tokens_2d[global_i, 0] = sampled_token
            new_tokens_lens[global_i] = 1
            # next_new_tokens: [sampled_token, draft_tokens...]
            next_new_tokens[global_i, 0] = sampled_token
            next_new_tokens[global_i, 1:] = next_draft_tokens[global_i]

        # Process SPEC_DEC (verification) requests
        # With num_previously_accepted = 1, sample_and_verify returns:
        # - output_ids: [verified_draft_0, ..., verified_draft_N-1, bonus_token, padding...]
        # - num_newly_accepted_tokens: N (count of verified INPUT draft tokens)
        # - num_accepted_tokens: N + 1 (count of OUTPUT tokens = verified + bonus)
        # We use num_accepted_tokens to slice output_ids since it gives us the output token count.
        # Use spec_dec_indices to map local index to global sequence index
        for local_i in range(num_spec_dec):
            global_i = spec_dec_indices[local_i]
            n_output = num_accepted_tokens[global_i].item()  # verified drafts + bonus
            assert n_output > 0, f"n_output must be > 0 for spec_dec request {global_i}"
            new_tokens_2d[global_i, :n_output] = output_ids[global_i, :n_output].to(torch.int32)
            new_tokens_lens[global_i] = n_output
            # next_new_tokens: [last_output_token (bonus), draft_tokens...]
            next_new_tokens[global_i, 0] = new_tokens_2d[global_i, n_output - 1]
            next_new_tokens[global_i, 1:] = next_draft_tokens[global_i]

        print(f"[EagleWrapper] output_ids: {output_ids}")
        print(
            f"[EagleWrapper] num_newly_accepted_tokens (verified inputs): {num_newly_accepted_tokens}"
        )
        print(f"[EagleWrapper] num_accepted_tokens (output count): {num_accepted_tokens}")
        print(f"[EagleWrapper] new_tokens_lens (for sampler): {new_tokens_lens}")
        print(f"[EagleWrapper] new_tokens_2d: {new_tokens_2d}")
        print(f"[EagleWrapper] next_draft_tokens: {next_draft_tokens}")
        print(f"[EagleWrapper] next_new_tokens: {next_new_tokens}")
        print(f"[EagleWrapper] new_tokens_2d shape: {new_tokens_2d.shape}")
        print(f"[EagleWrapper] next_draft_tokens shape: {next_draft_tokens.shape}")
        print(f"[EagleWrapper] next_new_tokens shape: {next_new_tokens.shape}")

        return EagleWrapperOutput(
            logits=output_logits,
            new_tokens=new_tokens_2d,
            new_tokens_lens=new_tokens_lens,  # Use corrected value, not num_newly_accepted_tokens
            next_draft_tokens=next_draft_tokens,
            next_new_tokens=next_new_tokens,
        )

    def _forward_prefill_only(self, input_ids, position_ids, **kwargs):
        """Forward pass without KV cache (prefill-only mode).

        This is the original implementation that recomputes all attention
        from scratch on every forward call.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        num_previously_accepted = kwargs.get("num_previously_accepted", None)
        if num_previously_accepted is None:
            raise ValueError("num_previously_accepted must be provided for prefill-only mode.")

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)

        # Compute embeddings using the target embedding layer
        # These embeddings will be passed to both target and draft models
        inputs_embeds = self.apply_draft_embedding(input_ids)

        # target_logits: [batch_size, seq_len, vocab_size]
        # Pass embeddings to target model instead of input_ids
        target_logits = self.target_model(
            inputs_embeds=inputs_embeds, position_ids=position_ids
        ).logits

        # output_ids: [batch_size, seq_len]. Contains a prefix of accepted tokens from the target model,
        # a generated token from the target model, and some padding to fill out the tensor.
        # num_accepted_tokens: [batch_size]. The number of accepted tokens in each batch.
        # num_newly_accepted_tokens: [batch_size]. The number of newly accepted tokens in each batch.

        output_ids, num_newly_accepted_tokens, num_accepted_tokens, _ = self.sample_and_verify(
            input_ids, target_logits, num_previously_accepted
        )

        # Get hidden states from the resource manager
        # resource_manager.hidden_states is [max_tokens, hidden_size * num_capture_layers] (flattened)
        # We slice to get [batch_size * seq_len, hidden_size * num_capture_layers]
        hidden_states = self.resource_manager.hidden_states[: (batch_size * seq_len), :]

        # Apply eagle3 fc to reduce hidden size.
        # Note: Since we are in prefill-only mode, this is extremely wasteful - we will apply the eagle3 fc layer
        # to hidden states that we have applied it to previously. But, this is generally the case in prefill-only mode.
        # Input: [batch_size * seq_len, hidden_size * num_capture_layers]
        # Output: [batch_size * seq_len, hidden_size]
        hidden_states = self.apply_eagle3_fc(hidden_states)

        # Reshape from [batch_size * seq_len, hidden_size] to [batch_size, seq_len, hidden_size]
        hidden_size = hidden_states.shape[-1]
        hidden_states = hidden_states.view(batch_size, seq_len, hidden_size)

        # Create a working buffer for the drafting loop in [batch, seq + draft_len, hidden] format.
        # This is separate from resource_manager.hidden_states which remains in flattened format.
        all_hidden_states = torch.zeros(
            (batch_size, seq_len + self.max_draft_len, hidden_size),
            device=device,
            dtype=hidden_states.dtype,
        )
        # Copy the initial hidden states from target model
        all_hidden_states[:, :seq_len, :] = hidden_states

        # Construct our inputs for the drafting loop.
        # We want tensors that will be able to hold all the tokens we draft.

        dummy_input_ids = torch.zeros(
            (batch_size, self.max_draft_len), device=device, dtype=output_ids.dtype
        )

        # draft_input_ids: [batch_size, seq_len + self.max_draft_len]
        draft_input_ids = torch.cat((output_ids, dummy_input_ids), dim=1)

        draft_position_ids = 1 + torch.arange(
            self.max_draft_len, device=device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)

        draft_position_ids = draft_position_ids + position_ids[:, -1:].expand(
            -1, self.max_draft_len
        )

        # draft_position_ids: [batch_size, seq_len + self.max_draft_len]
        # These position ids will work throughout the drafting loop.
        draft_position_ids = torch.cat((position_ids, draft_position_ids), dim=1)

        # The number of tokens currently in the draft input ids. Possibly includes padding.
        curr_num_tokens = seq_len

        # [batch_size]
        # The number of valid tokens currently in the draft input ids (does not include padding).
        curr_valid_tokens = num_accepted_tokens.clone()

        batch_indices = torch.arange(batch_size, device=device)

        for _ in range(self.max_draft_len):
            # Get the input ids, position ids, and hidden states for the current tokens.
            # size of tensor is constant for the current iteration and constant across dimensions (curr_num_tokens)
            # These tensors may correspond to padding tokens, but due to the causality of the draft model,
            # we can extract the draft tokens and hidden states corresponding to the valid tokens.

            input_ids = draft_input_ids[:, :curr_num_tokens]
            position_ids = draft_position_ids[:, :curr_num_tokens]
            hidden_states = all_hidden_states[:, :curr_num_tokens, :]

            inputs_embeds = self.apply_draft_embedding(input_ids)
            draft_output = self.draft_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                hidden_states=hidden_states,
            )

            draft_output_logits = self.apply_lm_head(draft_output.norm_hidden_state)

            # get the output logits for the latest valid token in each batch
            # It is at curr_valid_tokens-1 due to 0-indexing.
            latest_draft_logits = draft_output_logits[batch_indices, curr_valid_tokens - 1, :]

            # draft_output_tokens: [batch_size, 1]
            draft_output_tokens = self.sample_greedy(latest_draft_logits)

            # if the lm_head outputs tokens from the draft vocab, we need to convert them to tokens
            # from the target vocab before the next iteration.
            draft_output_tokens = self.apply_d2t(draft_output_tokens)

            # insert the draft output tokens into the draft input ids.
            draft_input_ids[batch_indices, curr_valid_tokens] = draft_output_tokens

            # Similarly, we want the hidden state for the latest drafted token in each batch.
            # This is a draft hidden state for the token that was just created from the latest valid token.

            # [batch_size, seq_len + self.max_draft_len, hidden_size]
            all_hidden_states[batch_indices, curr_valid_tokens, :] = draft_output.last_hidden_state[
                batch_indices, curr_valid_tokens - 1, :
            ]

            curr_valid_tokens = curr_valid_tokens + 1
            curr_num_tokens = curr_num_tokens + 1

        # Return the full draft_input_ids tensor for each batch element.
        # The valid prefix within each tensor has length:
        # num_previously_accepted[i] + num_newly_accepted_tokens[i] + max_draft_len
        # Callers should use this to slice out the valid tokens if needed.
        new_tokens = [draft_input_ids[i] for i in range(batch_size)]

        return EagleWrapperOutput(
            new_tokens=new_tokens,
            new_tokens_lens=num_newly_accepted_tokens,
        )


class ADHiddenStateManager(Eagle3ResourceManager):
    """AutoDeploy-specific hidden state manager for Eagle3 speculative decoding.

    Stores hidden states for use by the draft model in EagleWrapper.forward().
    This class extends Eagle3ResourceManager with functionality tailored to
    how AutoDeploy captures hidden states.
    """

    def __init__(
        self,
        config: EagleDecodingConfig,
        dtype: torch.dtype,
        hidden_size: int,
        max_num_requests: int,
        max_seq_len: int,
        max_num_tokens: int,
    ):
        print(f"DEBUG ADHiddenStateManager __init__: config: {config}")
        super().__init__(config, dtype, hidden_size, max_num_requests, max_seq_len, max_num_tokens)

        self.hidden_state_write_indices: torch.Tensor = torch.empty(
            max_num_tokens, dtype=torch.long, device="cuda"
        )

    @classmethod
    def build_from_target_engine(
        cls,
        engine,
        config: EagleDecodingConfig,
        max_num_requests: int,
    ) -> "ADHiddenStateManager":
        hidden_state_buffer = cls._get_hidden_state_buffers(engine.cache_seq_interface)[0]
        dtype = hidden_state_buffer.dtype
        hidden_size = hidden_state_buffer.shape[1]

        return cls(
            config=config,
            dtype=dtype,
            hidden_size=hidden_size,
            max_num_requests=max_num_requests,
            max_seq_len=engine.llm_args.max_seq_len,
            max_num_tokens=engine.llm_args.max_num_tokens,
        )

    @classmethod
    def build_from_target_factory(
        cls,
        target_factory,
        config: EagleDecodingConfig,
        max_num_requests: int,
        max_num_tokens: int,
    ) -> "ADHiddenStateManager":
        hidden_size = target_factory.hidden_size
        if hidden_size is None:
            raise ValueError(
                "Cannot determine hidden_size from target_factory. "
                "Ensure the factory implements the hidden_size property."
            )

        dtype = target_factory.dtype
        assert dtype is not None, "dtype must be available in target factory."

        return cls(
            config=config,
            dtype=dtype,
            hidden_size=hidden_size,
            max_num_requests=max_num_requests,
            max_seq_len=target_factory.max_seq_len,
            max_num_tokens=max_num_tokens,
        )

    @staticmethod
    def _get_hidden_state_buffers(
        cache_seq_interface,
    ) -> List[torch.Tensor]:
        hidden_state_buffers = []
        for name, tensor in cache_seq_interface.named_args.items():
            if "hidden_states_cache" in name:
                hidden_state_buffers.append(tensor)

        if not hidden_state_buffers:
            raise ValueError(
                "No hidden_state_buffers found in cache_seq_interface. Check if we are actually running Eagle3."
            )
        return hidden_state_buffers

    def prepare_hidden_states_capture(self, ordered_requests, cache_seq_interface) -> None:
        """Prepare the hidden states for capture by establishing indices that the hidden states will be written to."""
        seq_lens = cache_seq_interface.info.seq_len
        num_tokens = sum(seq_lens)

        start_idx = 0
        hidden_states_write_indices = []
        for request, seq_len in zip(ordered_requests, seq_lens):
            request_id = request.request_id
            slot_id = self.slot_manager.get_slot(request_id)
            self.start_indices[slot_id] = start_idx
            hidden_states_write_indices.extend(range(start_idx, start_idx + seq_len))
            start_idx += max(seq_len, self.max_total_draft_tokens + 1)
            assert start_idx < self.hidden_states.shape[0], (
                f"start_idx {start_idx} exceeds hidden_states capacity {self.hidden_states.shape[0]}"
            )

        if len(hidden_states_write_indices) != num_tokens:
            raise ValueError(
                f"len(hidden_state_write_indices) ({len(hidden_states_write_indices)}) != num_tokens \
                ({num_tokens}). Check whether ordered_requests matches up with seq_lens."
            )

        hidden_state_write_indices_host = torch.tensor(
            hidden_states_write_indices, dtype=torch.long
        )

        self.hidden_state_write_indices[:num_tokens].copy_(
            hidden_state_write_indices_host, non_blocking=True
        )

    def store_hidden_states(
        self, hidden_state_buffers: List[torch.Tensor], num_tokens: int
    ) -> None:
        """Store hidden states from buffers into self.hidden_states.

        This method takes a list of hidden state tensors (one per captured layer)
        and copies them into the resource manager's hidden_states buffer using
        the write indices set up by prepare_hidden_states_capture().

        Args:
            hidden_state_buffers: List of tensors, each of shape [max_num_tokens, hidden_size].
                One tensor per captured layer.
            num_tokens: Number of tokens to copy from each buffer.
        """
        if not hidden_state_buffers:
            return

        hidden_states = [buffer[:num_tokens] for buffer in hidden_state_buffers]
        hidden_states = torch.cat(hidden_states, dim=1)
        hidden_states = hidden_states.to(dtype=self.dtype)

        # Use write indices to copy to the correct locations in self.hidden_states
        token_idx = self.hidden_state_write_indices[:num_tokens]
        self.hidden_states[:, : hidden_states.shape[1]].index_copy_(0, token_idx, hidden_states)

    def capture_hidden_states(self, cache_seq_interface) -> None:
        full_hidden_states = self._get_hidden_state_buffers(cache_seq_interface)
        if not full_hidden_states:
            return

        num_tokens = sum(cache_seq_interface.info.seq_len)
        self.store_hidden_states(full_hidden_states, num_tokens)
