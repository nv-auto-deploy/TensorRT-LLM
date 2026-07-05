# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Slimmed-down PyTorch StepFun Step-3.7-Flash text model for AutoDeploy export (prefill only).

Source:
https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8 (and bf16 sibling Step-3.7-Flash)

Step-3.7-Flash is a vision-language model. This file ports ONLY the text decoder
(``Step3p5``-style ``model_type="step3p5"``); the vision tower is intentionally not
exported (AutoDeploy onboards the text generation path).

Key text-architecture features:
* Per-layer attention type: ``full_attention`` and ``sliding_attention`` alternate
  (1 full + 3 sliding per group). Full-attention layers use 64 Q heads; sliding-attention
  layers use 96 Q heads. Both use 8 KV heads (GQA) and head_dim=128.
* Head-wise attention gate (``g_proj``): the attention output of each head is multiplied by
  ``sigmoid(g_proj(hidden_states))`` before the output projection.
* Per-head QK RMSNorm over head_dim (Qwen3-style).
* Per-layer-type partial RoPE: full-attention layers rotate the first half of head_dim
  (partial_rotary_factor=0.5, rope_theta=5e6, llama3 rope-scaling); sliding-attention layers
  rotate the full head_dim (partial_rotary_factor=1.0, rope_theta=1e4, no scaling).
* Dense SwiGLU MLP on the first ``len(layers) - len(moe_layers)`` layers (layers 0-2);
  the remaining layers are MoE (288 routed experts, top-8, sigmoid routing with a per-expert
  bias used for *selection only*, fp32 gate, scaling 3.0) plus a dense shared expert.
* Gemma-style ``(1 + weight)`` RMSNorm convention for ALL norms (absorbed into the weight at
  load time via a pre-hook so the graph uses plain ``torch_rmsnorm``).

Differences from the HF reference (modeling_step3p7.py):
* Vision tower, multimodal merging, KV cache, training paths, dropout, and the MTP/spec
  ``mtp_block`` layers (45-47) are all removed — prefill text decode only.
* Uses AD canonical ops: torch_rmsnorm, torch_attention, torch_moe, plus the fused
  ``step3p7_partial_rope`` (one FlashInfer launch per layer over full head_dim, prefused
  fp32 cos/sin cache). No repeat_kv (torch_attention handles GQA natively).
* Stacked checkpoint MoE expert weights are split into per-expert Linear modules via a
  load-state-dict pre-hook for torch_moe dispatch.
* The SwiGLU activation clamp (``swiglu_limits``) present on routed experts of the last two
  MoE layers is NOT applied (the clamp limits are large numerical guards; see note in the
  MoE block). It is still applied on the dense shared-expert path where it is a plain MLP.

Tensor-parallel sharding (sharding-IR hints):
* Every shardable projection uses ``torch.ops.auto_deploy.torch_linear_simple`` with explicit
  ``tp_mode`` / ``layer_type`` hints, head reshapes use ``torch.ops.auto_deploy.view`` with
  ``tp_scaled_dim``, and rowwise outputs are followed by ``torch.ops.auto_deploy.all_reduce``.
  The exported graph fully specifies sharding; ``apply_sharding_hints`` applies it.
* MHA: q/k/v/g colwise (k/v use ``tp_min_local_shape=head_dim`` for GQA; the head-wise gate
  ``g_proj`` is a per-head column shard, ``tp_min_local_shape=1``), o_proj rowwise + all_reduce.
* MoE: routed experts via ``torch_moe(layer_type="moe")`` (EP/TP handled by the sharder); the
  shared expert is a colwise/rowwise MLP with no internal all_reduce — a single all_reduce at
  the ``routed + shared`` merge point covers both. Dense MLP layers reduce internally.
* The router gate is TP-replicated (opaque ``step3p7_router_gemv`` custom op: bf16 weight
  read + fp32 accumulate at batch=1 decode, reference fp32 GEMM elsewhere).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from ... import custom_ops  # noqa: F401 -- register all sharding-aware ops
from ..._compat import ActivationType
from ..hf import AutoModelForCausalLMFactory
from .rotary_utils import RotaryEmbeddingBase

# ---------------------------------------------------------------------------
# Bundled config
# ---------------------------------------------------------------------------


class Step3p7Config(PretrainedConfig):
    """Minimal flat text config for Step-3.7-Flash.

    Real deployments load the model's ``trust_remote_code`` config (the VLM wrapper
    ``Step3p7Config`` with a nested ``text_config``); AutoDeploy passes that object straight to
    ``_from_config`` and the model reads ``config.text_config`` via ``_get_text_config``. This
    bundled class is the resolvable config the model registers under, used for standalone
    construction and the offline sharding-IR equivalence harness (which builds a tiny instance and
    overrides the universal dims). Its defaults are intentionally small and tensor-parallel
    friendly so a 4-layer / 4-head tiny model shards cleanly; production values come from the
    checkpoint config.
    """

    model_type = "step3p5"

    def __init__(
        self,
        vocab_size: int = 128896,
        hidden_size: int = 64,
        head_dim: int = 16,
        num_attention_heads: int = 4,
        num_attention_groups: int = 4,
        attention_other_setting: Optional[dict] = None,
        intermediate_size: int = 64,
        num_hidden_layers: int = 4,
        layer_types: Optional[list] = None,
        moe_layers_enum: tuple = (2, 3),
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_intermediate_size: int = 16,
        share_expert_dim: int = 16,
        moe_router_scaling_factor: float = 3.0,
        rms_norm_eps: float = 1e-5,
        sliding_window: int = 4,
        max_position_embeddings: int = 256,
        rope_theta=(5e6, 1e4, 5e6, 1e4),
        partial_rotary_factors=(0.5, 1.0, 0.5, 1.0),
        rope_scaling: Optional[dict] = None,
        yarn_only_types=("full_attention",),
        swiglu_limits: Optional[list] = None,
        swiglu_limits_shared: Optional[list] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.attention_other_setting = attention_other_setting or {
            "num_attention_heads": num_attention_heads,
            "num_attention_groups": num_attention_groups,
            "head_dim": head_dim,
        }
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_types = layer_types or [
            "full_attention" if i % 2 == 0 else "sliding_attention"
            for i in range(num_hidden_layers)
        ]
        self.moe_layers_enum = moe_layers_enum
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = list(rope_theta)
        self.partial_rotary_factors = list(partial_rotary_factors)
        self.rope_scaling = rope_scaling or {
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": max_position_embeddings,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
        }
        self.yarn_only_types = list(yarn_only_types)
        self.swiglu_limits = swiglu_limits
        self.swiglu_limits_shared = swiglu_limits_shared
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Step3p7ModelOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None


@dataclass
class Step3p7CausalLMOutput(ModelOutput):
    logits: Optional[torch.FloatTensor] = None


# ---------------------------------------------------------------------------
# Config access helper
# ---------------------------------------------------------------------------


def _get_text_config(config):
    """Return the text sub-config (Step-3.7 wraps the LLM in a VLM ``Step3p7Config``)."""
    return getattr(config, "text_config", config)


# ---------------------------------------------------------------------------
# Load-state-dict pre-hooks (run on the full ForCausalLM)
# ---------------------------------------------------------------------------


def _step3p7_norm_weight_load_hook(state_dict, prefix, *args, **kwargs):
    """Absorb Step's ``(1 + weight)`` RMSNorm convention into the weight at load time.

    HF Step stores all norm weights as a bias around zero and applies ``(1 + weight)``.
    Adding 1.0 here lets the forward use the standard ``torch_rmsnorm(x, weight, eps)``
    without an extra add node in the exported graph (matches the Gemma onboarding pattern).
    """
    for key in list(state_dict.keys()):
        if key.endswith("layernorm.weight") or key.endswith("norm.weight"):
            state_dict[key] = state_dict[key] + 1.0


def _step3p7_moe_split_load_hook(state_dict, prefix, *args, **kwargs):
    """Split stacked routed-expert tensors into per-expert Linear tensors.

    The checkpoint stores routed experts as stacked tensors per projection:
      * ``...moe.gate_proj.weight``  [E, moe_intermediate, hidden]
      * ``...moe.up_proj.weight``    [E, moe_intermediate, hidden]
      * ``...moe.down_proj.weight``  [E, hidden, moe_intermediate]
    plus, for the FP8 checkpoint, block-wise dequant scales (one per projection):
      * ``...moe.{proj}.weight_scale_inv``  [E, ceil(out/128), ceil(in/128)]
    The custom model keeps per-expert ``nn.Linear`` modules for ``torch_moe`` dispatch, so split
    every stacked ``[E, ...]`` tensor into per-expert tensors:
      * ``...moe.experts.{e}.{gate,up,down}_proj.weight`` (and matching ``.weight_scale_inv``).
    The FP8 ``quantize_finegrained_fp8_moe`` transform then consumes the per-expert FP8 weight +
    ``weight_scale_inv`` exactly as it does for the (per-expert) DeepSeek-V3 checkpoint.
    """
    for key in list(state_dict.keys()):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for leaf in ("weight", "weight_scale_inv"):
                suffix = f".moe.{proj}.{leaf}"
                if key.endswith(suffix) and state_dict[key].dim() == 3:
                    stacked = state_dict.pop(key)
                    base = key[: -len(suffix)]
                    for e in range(stacked.shape[0]):
                        state_dict[f"{base}.moe.experts.{e}.{proj}.{leaf}"] = stacked[e]
                    break


# ---------------------------------------------------------------------------
# RMSNorm (using AD canonical op; (1 + weight) absorbed at load time)
# ---------------------------------------------------------------------------


class Step3p7RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.torch_rmsnorm(x, self.weight, self.variance_epsilon)


# ---------------------------------------------------------------------------
# Rotary Embedding (per-layer-type: partial rotation + optional llama3 scaling)
# ---------------------------------------------------------------------------


def _compute_step3p7_inv_freq(
    head_dim: int,
    partial_rotary_factor: float,
    base: float,
    rope_scaling: Optional[dict],
) -> torch.Tensor:
    """Inverse frequencies for Step RoPE (default or llama3-scaled)."""
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    if not rope_scaling:
        return inv_freq

    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    assert rope_type == "llama3", f"Step-3.7 only supports llama3 rope-scaling, got {rope_type!r}"

    # Faithful copy of transformers _compute_llama3_parameters scaling math.
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    return inv_freq_llama


def _build_step3p7_fused_cos_sin_cache(
    inv_freq: torch.Tensor, max_position_embeddings: int
) -> torch.Tensor:
    """FlashInfer-layout fused RoPE cache ``[cos | sin]``, shape [max_pos, rotary_dim], fp32.

    Same math (and thus bit-identical values) as the cache ``optimize_rope`` used to
    materialize from the graph-computed cos/sin tables: fp32 ``outer(arange, inv_freq)``
    angles, scale 1.0 (llama3 scaling is folded into ``inv_freq`` itself, and the llama3
    attention-scaling factor is 1.0 for this model).
    """
    positions = torch.arange(max_position_embeddings, dtype=inv_freq.dtype, device=inv_freq.device)
    freqs = torch.outer(positions, inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(torch.float32)


class Step3p7RotaryEmbedding(RotaryEmbeddingBase):
    """RoPE cache holder for one attention type (partial rotation, optional llama3 scaling).

    Registers the small ``inv_freq`` buffer plus the prefused fp32 ``cos_sin_cache`` consumed
    directly by ``step3p7_partial_rope`` (FlashInfer layout, rotary_dim wide). The cache is
    built once at construction instead of graph-computed per forward, so no per-step table
    build and no ``optimize_rope`` rewrite are needed.
    """

    def __init__(
        self,
        head_dim: int,
        partial_rotary_factor: float,
        base: float,
        max_position_embeddings: int,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        inv_freq = _compute_step3p7_inv_freq(head_dim, partial_rotary_factor, base, rope_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "cos_sin_cache",
            _build_step3p7_fused_cos_sin_cache(inv_freq, max_position_embeddings),
            persistent=False,
        )

    def _apply(self, fn):
        """Keep ``cos_sin_cache`` fp32 across blanket dtype casts.

        ``HFQuantConfigReader.post_process_model`` runs ``model.to(bf16)`` on the freshly
        built model, which would downcast the cache (FlashInfer requires fp32). Rebuild from
        ``inv_freq`` — which the base class just re-pinned to fp32 — rather than re-floating
        the downcast values, so the cache keeps full fp32 precision.
        """
        super()._apply(fn)
        cache = getattr(self, "cos_sin_cache", None)
        if isinstance(cache, torch.Tensor) and cache.dtype != torch.float32:
            self.cos_sin_cache = _build_step3p7_fused_cos_sin_cache(
                self.inv_freq, self.max_position_embeddings
            )
        return self


@torch.library.custom_op("auto_deploy::step3p7_partial_rope", mutates_args=())
def step3p7_partial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partial-rotary RoPE over full [B, S, N, head_dim] q/k in one FlashInfer launch.

    ``rotary_dim = cos_sin_cache.shape[-1]`` may be < head_dim (full-attention layers) or
    == head_dim (sliding-attention layers). The kernel rotates ``[..., :rotary_dim]`` with
    neox pairing (== the ``rotate_half`` reference) in fp32 and copies the pass-through
    remainder into the output, replacing the previous slice → rope-op → 2×``torch.cat``
    chain (three device kernels per layer plus an extra HBM round trip of both halves)
    with a single kernel. N may be TP-sharded; the op is shape-agnostic per head.

    - position_ids: [B, S] (or flat [B*S]) integer positions indexing ``cos_sin_cache`` rows
    - cos_sin_cache: [max_pos, rotary_dim] fp32 in fused ``[cos | sin]`` layout
    """
    import flashinfer

    q_3d = q.flatten(0, -3)
    k_3d = k.flatten(0, -3)
    q_rope = torch.empty_like(q_3d)
    k_rope = torch.empty_like(k_3d)
    pos = position_ids.reshape(-1)
    if pos.dtype != torch.int32:
        pos = pos.to(torch.int32)
    flashinfer.rope._apply_rope_pos_ids_cos_sin_cache(
        q=q_3d,
        k=k_3d,
        q_rope=q_rope,
        k_rope=k_rope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos,
        interleave=False,  # neox pairing, matches the rotate_half reference
    )
    return q_rope.view(q.shape), k_rope.view(k.shape)


@step3p7_partial_rope.register_fake
def _step3p7_partial_rope_fake(q, k, position_ids, cos_sin_cache):
    return torch.empty_like(q), torch.empty_like(k)


# ---------------------------------------------------------------------------
# Dense SwiGLU MLP (dense layers + shared expert)
# ---------------------------------------------------------------------------


class Step3p7MLP(nn.Module):
    """SwiGLU MLP with an optional post-activation clamp (Step ``swiglu_limit``).

    Sharding: gate/up colwise, down rowwise. ``apply_all_reduce`` controls whether the rowwise
    output is reduced here (True for a standalone dense MLP) or left partial for a downstream
    merge-point all_reduce (False when used as a MoE shared expert).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        swiglu_limit: Optional[float] = None,
        layer_type: str = "mlp",
        apply_all_reduce: bool = True,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.limit = swiglu_limit
        self.layer_type = layer_type
        self.apply_all_reduce = apply_all_reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.ops.auto_deploy.torch_linear_simple(
            x, self.gate_proj.weight, None, tp_mode="colwise", layer_type=self.layer_type
        )
        up = torch.ops.auto_deploy.torch_linear_simple(
            x, self.up_proj.weight, None, tp_mode="colwise", layer_type=self.layer_type
        )
        gate = self.act_fn(gate)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        down = torch.ops.auto_deploy.torch_linear_simple(
            gate * up, self.down_proj.weight, None, tp_mode="rowwise", layer_type=self.layer_type
        )
        if self.apply_all_reduce:
            down = torch.ops.auto_deploy.all_reduce(down, layer_type=self.layer_type)
        return down


# ---------------------------------------------------------------------------
# Fused MoE routing op (sigmoid + per-expert bias + top-k + renormalize)
# ---------------------------------------------------------------------------
#
# Step's router does the following per token (see ``Step3p7MoE.forward``):
#     probs   = sigmoid(router_logits)               # un-biased gate probs
#     scores  = probs + router_bias                  # selection-only bias
#     idx     = topk(scores, k)                       # pick experts by score
#     weights = gather(probs, idx)                    # weights use UN-biased probs
#     weights = weights / (weights.sum(-1) + 1e-20)   # renormalize
#     weights = weights * routed_scaling_factor       # scale
#     weights = weights.to(hidden.dtype)              # cast
#
# At TP8/batch=1 decode the router gate is replicated, so this runs on a tiny
# ``[1, 288]`` fp32 tensor for each of the 42 MoE layers. As separate torch ops
# that is ~7 launch-bound kernels per layer on the routed critical path (the
# shared expert is overlapped on an aux stream by ``multi_stream_moe``). This
# custom op fuses that router into one optimized Triton launch.


_STEP3P7_ROUTER_NUM_EXPERTS = 288
_STEP3P7_ROUTER_TOP_K = 8
_STEP3P7_ROUTER_SCALING_FACTOR = 3.0


def _step3p7_router_production_contract(
    router_logits: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    out_dtype: torch.dtype,
) -> bool:
    """Return whether the specialized Step-3.7 router kernels can handle this call."""
    return (
        router_logits.is_cuda
        and router_bias.is_cuda
        and router_logits.dtype == torch.float32
        and router_bias.dtype in (torch.float32, torch.bfloat16)
        and out_dtype == torch.bfloat16
        and router_logits.ndim == 2
        and router_logits.shape[1] == _STEP3P7_ROUTER_NUM_EXPERTS
        and router_bias.ndim == 1
        and router_bias.shape[0] == _STEP3P7_ROUTER_NUM_EXPERTS
        and top_k == _STEP3P7_ROUTER_TOP_K
        and routed_scaling_factor == _STEP3P7_ROUTER_SCALING_FACTOR
    )


def _require_step3p7_router_optimized_triton(
    router_logits: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    out_dtype: torch.dtype,
) -> None:
    if not _step3p7_router_production_contract(
        router_logits,
        router_bias,
        top_k,
        routed_scaling_factor,
        out_dtype,
    ):
        raise ValueError(
            "Step-3.7 optimized Triton router only supports CUDA fp32 logits and fp32/bf16 "
            f"bias with {_STEP3P7_ROUTER_NUM_EXPERTS} experts, "
            f"top-{_STEP3P7_ROUTER_TOP_K}, scaling={_STEP3P7_ROUTER_SCALING_FACTOR}, "
            "and bf16 output; got "
            f"logits_shape={tuple(router_logits.shape)}, logits_dtype={router_logits.dtype}, "
            f"logits_device={router_logits.device}, bias_shape={tuple(router_bias.shape)}, "
            f"bias_dtype={router_bias.dtype}, bias_device={router_bias.device}, top_k={top_k}, "
            f"scaling={routed_scaling_factor}, out_dtype={out_dtype}"
        )


@triton.jit
def _step3p7_router_rcp_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _step3p7_router_fast_sigmoid(x):
    x_half = x * 0.5
    tanh_x = tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [x_half],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tanh_x * 0.5 + 0.5


@triton.jit
def _step3p7_router_topk_optimized_triton_kernel(
    logits_ptr,
    bias_ptr,
    indices_ptr,
    weights_ptr,
    scaling_factor,
):
    """Triton r4-v20 router: hard-coded Step-3.7 top-8 over 288 experts."""
    K: tl.constexpr = 8
    BLOCK_LO: tl.constexpr = 256
    BLOCK_HI: tl.constexpr = 32

    pid = tl.program_id(0)
    e_lo = tl.arange(0, BLOCK_LO)
    e_hi = tl.arange(0, BLOCK_HI)
    e_hi_global = e_hi + BLOCK_LO

    base = pid * 288
    logits_lo = tl.load(logits_ptr + base + e_lo).to(tl.float32)
    logits_hi = tl.load(logits_ptr + base + e_hi_global).to(tl.float32)
    bias_lo = tl.load(bias_ptr + e_lo).to(tl.float32)
    bias_hi = tl.load(bias_ptr + e_hi_global).to(tl.float32)

    scores_lo = _step3p7_router_fast_sigmoid(logits_lo) + bias_lo
    scores_hi = _step3p7_router_fast_sigmoid(logits_hi) + bias_hi

    score_lo_i32 = scores_lo.to(tl.int32, bitcast=True)
    score_hi_i32 = scores_hi.to(tl.int32, bitcast=True)
    packed_lo = (score_lo_i32 & -512) | e_lo
    packed_hi = (score_hi_i32 & -512) | e_hi_global

    k_offsets = tl.arange(0, K)
    top_packed = tl.zeros([K], dtype=tl.int32)

    ZERO_I32: tl.constexpr = 0
    for k in tl.static_range(K):
        max_lo = tl.max(packed_lo, axis=0)
        max_hi = tl.max(packed_hi, axis=0)
        max_packed = tl.maximum(max_lo, max_hi)
        max_idx = max_packed & 0x1FF
        top_packed = tl.where(k_offsets == k, max_packed, top_packed)
        if k < K - 1:
            packed_lo = tl.where(e_lo == max_idx, ZERO_I32, packed_lo)
            packed_hi = tl.where(e_hi_global == max_idx, ZERO_I32, packed_hi)

    top_indices = top_packed & 0x1FF
    top_score_i32 = top_packed & -512
    top_score_f32 = top_score_i32.to(tl.float32, bitcast=True)
    top_bias = tl.load(bias_ptr + top_indices).to(tl.float32)
    top_probs = top_score_f32 - top_bias

    sum_probs = tl.sum(top_probs, axis=0)
    inv_sum_x_sf = scaling_factor * _step3p7_router_rcp_approx(sum_probs + 1.0e-20)
    out_weights = top_probs * inv_sum_x_sf

    tl.store(indices_ptr + pid * K + k_offsets, top_indices.to(tl.int64))
    tl.store(weights_ptr + pid * K + k_offsets, out_weights.to(tl.bfloat16))


def _run_step3p7_router_topk_optimized_triton(
    router_logits: torch.Tensor,
    router_bias: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    routed_scaling_factor: float,
) -> None:
    grid = (router_logits.shape[0],)
    _step3p7_router_topk_optimized_triton_kernel[grid](
        router_logits,
        router_bias,
        selected_experts,
        routing_weights,
        float(routed_scaling_factor),
        num_warps=1,
        num_stages=1,
    )


@torch.library.custom_op("auto_deploy::step3p7_fused_router_topk", mutates_args=())
def step3p7_fused_router_topk(
    router_logits: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid + per-expert-bias top-k MoE routing.

    This implementation is intentionally optimized-Triton-only with no fallback path.

    Returns:
        routing_weights: ``(T, top_k)`` tensor in ``out_dtype``.
        selected_experts: ``(T, top_k)`` ``int64`` tensor of expert ids.
    """
    assert router_logits.ndim == 2, "router_logits must be 2-D (T, E)"
    num_tokens = router_logits.shape[0]
    routing_weights = torch.empty((num_tokens, top_k), dtype=out_dtype, device=router_logits.device)
    selected_experts = torch.empty(
        (num_tokens, top_k), dtype=torch.int64, device=router_logits.device
    )

    _require_step3p7_router_optimized_triton(
        router_logits,
        router_bias,
        top_k,
        routed_scaling_factor,
        out_dtype,
    )
    _run_step3p7_router_topk_optimized_triton(
        router_logits,
        router_bias,
        selected_experts,
        routing_weights,
        routed_scaling_factor,
    )
    return routing_weights, selected_experts


@step3p7_fused_router_topk.register_fake
def _step3p7_fused_router_topk_fake(
    router_logits: torch.Tensor,
    router_bias: torch.Tensor,
    top_k: int,
    routed_scaling_factor: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = router_logits.shape[0]
    routing_weights = router_logits.new_empty((num_tokens, top_k), dtype=out_dtype)
    selected_experts = router_logits.new_empty((num_tokens, top_k), dtype=torch.int64)
    return routing_weights, selected_experts


# ---------------------------------------------------------------------------
# Router gate GEMV (bf16 weight read, fp32 accumulate, fp32 logits)
# ---------------------------------------------------------------------------
#
# The router gate weight ships as bf16 in the checkpoint; ``need_fp32_gate`` requires fp32
# GEMM *accumulation*, not fp32 *operands*: every bf16 x bf16 product is exactly
# representable in fp32 (8-bit x 8-bit mantissas), so reading bf16 operands and accumulating
# in fp32 yields the same products as an fp32-materialized GEMM and differs only in
# summation order (~1e-7 relative on the logits). At TP8/batch=1 decode the fp32 gate GEMV
# was a ``[1, 4096] x [288, 4096]^T`` cuBLAS gemvx behind a per-call ``hidden.float()`` cast,
# x42 MoE layers per step; it re-read a 4.7MB fp32 weight every call. The Triton GEMV below
# instead keeps the gate weight bf16 (half the read bytes, half the weight memory), consumes
# the bf16 hidden state directly (dropping the cast kernel), and stores fp32 logits for the
# fused router top-k. One program per expert row, mask-free (K must divide by BLOCK_K).
# Off the decode hot path (prefill, multi-token decode, offline harnesses) the op upcasts
# both operands and runs the reference fp32 GEMM: ``weight.float()`` reproduces the
# fp32-materialized master bit-exactly, so those paths match the pre-optimization graph.


_STEP3P7_ROUTER_GEMV_BLOCK_K = 4096


@triton.jit
def _step3p7_router_gemv_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``out[n] = dot(x, w[n, :])`` in fp32; one program per BLOCK_N expert rows."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + offs_n, acc)


def _step3p7_router_gemv_production_contract(hidden: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the batch=1 decode Triton GEMV can handle this call."""
    return (
        hidden.is_cuda
        and weight.is_cuda
        and hidden.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden.ndim == 2
        and weight.ndim == 2
        and hidden.shape[0] == 1
        and hidden.shape[1] == weight.shape[1]
        and weight.shape[1] % _STEP3P7_ROUTER_GEMV_BLOCK_K == 0
    )


@torch.library.custom_op("auto_deploy::step3p7_router_gemv", mutates_args=())
def step3p7_router_gemv(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 router logits ``hidden @ weight^T`` with a bf16 weight read at batch=1 decode.

    Args:
        hidden: ``[T, hidden]`` flattened token activations (bf16 in production).
        weight: ``[E, hidden]`` router gate weight (bf16 in production).

    ``T == 1`` (the cudagraph decode hot path) runs the Triton GEMV: bf16 loads, fp32
    accumulate, fp32 store — matching the fp32 reference GEMM up to summation order (see
    section comment). Any other shape/dtype/device (prefill, multi-token decode batches,
    offline harnesses) upcasts both operands and computes the reference fp32 GEMM,
    bit-identical to the pre-optimization fp32-materialized graph.
    """
    if _step3p7_router_gemv_production_contract(hidden, weight):
        hidden = hidden.contiguous()
        weight = weight.contiguous()
        num_experts, k = weight.shape
        out = torch.empty((1, num_experts), dtype=torch.float32, device=hidden.device)
        _step3p7_router_gemv_kernel[(num_experts,)](
            hidden,
            weight,
            out,
            K=k,
            BLOCK_N=1,
            BLOCK_K=_STEP3P7_ROUTER_GEMV_BLOCK_K,
            num_warps=4,
        )
        return out
    return F.linear(hidden.float(), weight.float())


@step3p7_router_gemv.register_fake
def _step3p7_router_gemv_fake(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden.new_empty((hidden.shape[0], weight.shape[0]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fused head-wise attention gate (sigmoid + per-head broadcast multiply)
# ---------------------------------------------------------------------------
#
# Step applies a per-head gate to the attention output before o_proj:
#     gate = sigmoid(g_proj(hidden_states))      # [B, S, N]
#     attn = attn * gate.unsqueeze(-1)           # [B, S, N, D]
# As separate torch ops that is a sigmoid launch plus a broadcast-multiply launch
# per attention layer (x45 layers) on tiny [.., N, D] tensors at batch=1 decode --
# two launch-bound elementwise kernels that no generic matcher (silu_mul / swiglu /
# rmsnorm) touches. This custom op fuses the sigmoid + per-head broadcast-multiply
# into one Triton launch (one kernel per layer instead of two). The gate-proj GEMV
# is left as a separate (sharded) torch_linear_simple so its per-head column-shard
# hint (tp_min_local_shape=1) is preserved; this op is a transparent elementwise
# pass-through that keeps the [.., N/tp, D] head partition for the downstream
# view -> o_proj -> all_reduce chain.


@triton.jit
def _step3p7_head_gate_kernel(
    attn_ptr,
    gate_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """``out[row, :] = attn[row, :] * sigmoid(gate[row])``; one program per (B*S*N) head row."""
    row = tl.program_id(0)
    logit = tl.load(gate_ptr + row).to(tl.float32)
    # Match PyTorch's bf16 sigmoid: accurate fp32 sigmoid, then round to bf16 BEFORE the
    # multiply, so the fused result mirrors `attn * gate.sigmoid()` to within bf16 rounding.
    g = 1.0 / (1.0 + tl.exp(-logit))
    g = g.to(tl.bfloat16).to(tl.float32)
    d = tl.arange(0, BLOCK_D)
    mask = d < D
    a = tl.load(attn_ptr + row * D + d, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * D + d, (a * g).to(tl.bfloat16), mask=mask)


@torch.library.custom_op("auto_deploy::step3p7_head_gate", mutates_args=())
def step3p7_head_gate(attn_output: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """Fused head-wise attention gate: ``attn_output * sigmoid(gate_logits)[..., None]``.

    Args:
        attn_output: ``[..., N, D]`` attention output (one row of size ``D`` per head).
        gate_logits: ``[..., N]`` per-head gate pre-activation (the ``g_proj`` output).

    Returns a tensor with the same shape and dtype as ``attn_output``.
    """
    assert attn_output.dim() >= 2, "attn_output must be at least 2-D [..., N, D]"
    assert attn_output.shape[:-1] == gate_logits.shape, (
        f"gate_logits {tuple(gate_logits.shape)} must equal attn_output[..., N] "
        f"{tuple(attn_output.shape[:-1])}"
    )
    # Production decode/prefill is bf16 on CUDA -> fused Triton. Any other dtype/device
    # (e.g. the offline sharding-IR equivalence harness) falls back to the reference ops.
    if not (
        attn_output.is_cuda
        and attn_output.dtype == torch.bfloat16
        and gate_logits.dtype == torch.bfloat16
    ):
        return attn_output * gate_logits.sigmoid().unsqueeze(-1)

    attn_output = attn_output.contiguous()
    gate_logits = gate_logits.contiguous()
    D = attn_output.shape[-1]
    out = torch.empty_like(attn_output)
    n_rows = gate_logits.numel()
    BLOCK_D = triton.next_power_of_2(D)
    _step3p7_head_gate_kernel[(n_rows,)](
        attn_output, gate_logits, out, D, BLOCK_D, num_warps=1, num_stages=1
    )
    return out


@step3p7_head_gate.register_fake
def _step3p7_head_gate_fake(attn_output: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(attn_output)


# ---------------------------------------------------------------------------
# Sparse MoE block (routed experts + shared expert)
# ---------------------------------------------------------------------------


class Step3p7MoE(nn.Module):
    """Routed MoE with sigmoid routing + per-expert bias (selection only).

    The dense shared expert is a sibling of this module on the decoder layer (matching the HF
    hierarchy and the checkpoint layout ``model.layers.N.share_expert.*``), so it is NOT part of
    this module.

    Routing (HF ``router_bias_func``):
      1. ``probs = sigmoid(fp32 router logits)``
      2. select top-k experts by ``probs + router_bias``
      3. gather the *un-biased* ``probs`` for the selected experts
      4. renormalize the gathered weights and scale by ``moe_router_scaling_factor``

    NOTE on ``swiglu_limit``: the routed experts of the last two MoE layers carry a SwiGLU
    activation clamp in the HF reference. ``torch_moe`` has no clamp parameter, so the routed
    clamp is not applied here (the limits are large guards that rarely activate). The clamp IS
    applied on the dense shared-expert path (a plain MLP, see ``Step3p7DecoderLayer``).
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.routed_scaling_factor = getattr(config, "moe_router_scaling_factor", 1.0)

        # bf16 by construction, matching the checkpoint storage: the batch=1 decode GEMV
        # (``step3p7_router_gemv``) reads the weight as bf16 with fp32 accumulation, which
        # matches the fp32-materialized GEMM up to summation order while halving the
        # per-call weight read; fp32 GEMM *accumulation* is what config.need_fp32_gate
        # requires. Off-hot-path shapes upcast per call inside the op.
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=torch.bfloat16)
        self.register_buffer(
            "router_bias", torch.zeros(self.num_experts, dtype=torch.float32), persistent=True
        )

        self.experts = nn.ModuleList(
            [
                Step3p7MLP(self.hidden_size, config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def _apply(self, fn, recurse=True):
        """Pin the router gate weight to bf16 across blanket dtype casts.

        Blanket post-processing casts (e.g. the HF fp8 quant reader's ``model.to(bf16)``,
        ``HFQuantConfigReader.post_process_model``) must not change the gate dtype: the
        exported graph relies on a bf16 gate weight for the batch=1 decode Triton GEMV
        (``step3p7_router_gemv``). Re-pinning here is lossless: real values only arrive
        later, when the bf16 checkpoint weight is copied into the parameter by
        ``load_state_dict``.
        """
        ret = super()._apply(fn, recurse)
        weight = self.gate.weight
        if weight is not None and weight.is_floating_point() and weight.dtype != torch.bfloat16:
            self.gate.weight = nn.Parameter(
                weight.data.to(torch.bfloat16), requires_grad=weight.requires_grad
            )
        return ret

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # fp32 router logits (config.need_fp32_gate): batch=1 decode reads the bf16 gate
        # weight via a Triton GEMV (fp32 accumulate/output, drops the hidden cast); all
        # other shapes upcast and run the reference fp32 GEMM (see step3p7_router_gemv).
        router_logits = torch.ops.auto_deploy.step3p7_router_gemv(hidden_flat, self.gate.weight)

        # Fused sigmoid + per-expert-bias top-k routing in one optimized Triton launch
        # (replaces the 7 separate torch ops; numerics-faithful, see custom op above).
        routing_weights, selected_experts = torch.ops.auto_deploy.step3p7_fused_router_topk(
            router_logits,
            self.router_bias,
            self.top_k,
            self.routed_scaling_factor,
            hidden_flat.dtype,
        )

        routed = torch.ops.auto_deploy.torch_moe(
            hidden_flat,
            selected_experts,
            routing_weights,
            w1_weight=[e.gate_proj.weight for e in self.experts],
            w2_weight=[e.down_proj.weight for e in self.experts],
            w3_weight=[e.up_proj.weight for e in self.experts],
            is_gated_mlp=True,
            act_fn=int(ActivationType.Silu),
            layer_type="moe",
        )
        return routed.view(bsz, seq_len, hidden_dim)


# ---------------------------------------------------------------------------
# Attention (GQA + per-head QK norm + head-wise gate + partial RoPE)
# ---------------------------------------------------------------------------


class Step3p7Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.attention_type = config.layer_types[layer_idx]
        is_sliding = self.attention_type == "sliding_attention"

        if is_sliding:
            other = config.attention_other_setting
            self.num_heads = other["num_attention_heads"]
            self.num_kv_heads = other["num_attention_groups"]
            self.sliding_window = config.sliding_window
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups
            self.sliding_window = None

        self.scaling = self.head_dim ** (-0.5)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        # Head-wise attention gate: one sigmoid scalar per head, applied to the attention output
        # before o_proj. Sharded as a per-head column shard (tp_min_local_shape=1) so it follows
        # the same head partition as q/k/v under tensor parallelism.
        self.g_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)

        # Per-head QK RMSNorm over head_dim.
        self.q_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p7RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.q_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        k = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.k_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=self.head_dim,
            layer_type="mha",
        )
        v = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.v_proj.weight,
            None,
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

        # Per-head QK norm over head_dim.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Fused partial-rotary RoPE: one FlashInfer kernel rotates [..., :rotary_dim] and
        # copies the pass-through remainder (no slice + torch.cat per projection).
        position_ids, cos_sin_cache = position_embeddings
        q, k = torch.ops.auto_deploy.step3p7_partial_rope(q, k, position_ids, cos_sin_cache)

        attn_output = torch.ops.auto_deploy.torch_attention(
            q,
            k,
            v,
            None,  # attn_mask
            0.0,  # dropout_p
            True,  # is_causal
            self.scaling,  # scale
            None,  # sinks
            self.sliding_window,  # sliding_window
            None,  # logit_cap
            "bsnd",  # layout
        )  # [B, S, N, head_dim]

        # Head-wise gate: scale each head's output by sigmoid(per-head gate). g_proj is a per-head
        # column shard (tp_min_local_shape=1), so its [B, S, N] output is sharded over the same
        # head partition as the attention output. The sigmoid + per-head broadcast-multiply are
        # fused into one Triton launch (step3p7_head_gate), dropping one launch-bound elementwise
        # kernel per attention layer at batch=1 decode. The gate-proj GEMV stays a separate sharded
        # linear so its per-head column-shard hint is preserved (the fused op is a transparent
        # elementwise pass-through over the [.., N/tp, D] head partition).
        gate_logits = torch.ops.auto_deploy.torch_linear_simple(
            hidden_states,
            self.g_proj.weight,
            None,
            tp_mode="colwise",
            tp_min_local_shape=1,
            layer_type="mha",
        )  # [B, S, N]
        attn_output = torch.ops.auto_deploy.step3p7_head_gate(attn_output, gate_logits)

        attn_output = torch.ops.auto_deploy.view(
            attn_output,
            [bsz, q_len, self.num_heads * self.head_dim],
            tp_scaled_dim=2,
            layer_type="mha",
        )
        attn_output = torch.ops.auto_deploy.torch_linear_simple(
            attn_output, self.o_proj.weight, None, tp_mode="rowwise", layer_type="mha"
        )
        attn_output = torch.ops.auto_deploy.all_reduce(attn_output, layer_type="mha")
        return attn_output


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class Step3p7DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int, is_moe_layer: bool):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = Step3p7Attention(config, layer_idx)

        _, shared_swiglu_limit = _layer_swiglu_limits(config, layer_idx)
        self.is_moe_layer = is_moe_layer
        if is_moe_layer:
            self.moe = Step3p7MoE(config)
            # Shared expert is a sibling of ``moe`` (checkpoint key model.layers.N.share_expert.*).
            # No internal all_reduce: the single merge-point all_reduce (routed + shared) reduces it.
            self.share_expert = Step3p7MLP(
                config.hidden_size,
                config.share_expert_dim,
                swiglu_limit=shared_swiglu_limit,
                layer_type="moe",
                apply_all_reduce=False,
            )
        else:
            self.mlp = Step3p7MLP(
                config.hidden_size, config.intermediate_size, swiglu_limit=shared_swiglu_limit
            )

        self.input_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Step3p7RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        full_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        sliding_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        position_embeddings = (
            sliding_position_embeddings
            if self.attention_type == "sliding_attention"
            else full_position_embeddings
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.is_moe_layer:
            # Single all_reduce at the routed + shared merge point (both are left partial above).
            hidden_states = self.moe(hidden_states) + self.share_expert(hidden_states)
            hidden_states = torch.ops.auto_deploy.all_reduce(hidden_states, layer_type="moe")
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _moe_layer_indices(config) -> List[int]:
    moe_layers_enum = getattr(config, "moe_layers_enum", None)
    if moe_layers_enum is None:
        return list(range(1, config.num_hidden_layers))
    if isinstance(moe_layers_enum, str):
        return [int(i) for i in moe_layers_enum.split(",") if i.strip()]
    return [int(i) for i in moe_layers_enum]


def _layer_swiglu_limits(config, layer_idx: int) -> Tuple[Optional[float], Optional[float]]:
    """Return (routed-expert limit, shared/dense limit) for a layer, or None when disabled."""

    def _val(values):
        if not values or layer_idx >= len(values):
            return None
        v = values[layer_idx]
        return float(v) if v else None

    return _val(getattr(config, "swiglu_limits", None)), _val(
        getattr(config, "swiglu_limits_shared", None)
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Step3p7PreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    _no_split_modules = ["Step3p7DecoderLayer"]
    supports_gradient_checkpointing = False


class Step3p7TextModel(Step3p7PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        text_config = _get_text_config(config)
        self.config = config

        moe_layers = set(_moe_layer_indices(text_config))
        self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Step3p7DecoderLayer(text_config, idx, is_moe_layer=idx in moe_layers)
                for idx in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = Step3p7RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

        self.full_rotary_emb, self.sliding_rotary_emb = _build_rotary_embeddings(text_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Step3p7ModelOutput:
        assert position_ids is not None, "position_ids is required for AD export"

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.to(self.norm.weight.dtype)

        # One flattened int32 position tensor + the per-type prefused fp32 caches feed the
        # fused partial-rope op in every layer (no per-layer cos/sin gather or table build).
        pos_flat = position_ids.reshape(-1).to(torch.int32)
        full_pe = (pos_flat, self.full_rotary_emb.cos_sin_cache)
        sliding_pe = (pos_flat, self.sliding_rotary_emb.cos_sin_cache)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, full_pe, sliding_pe)

        hidden_states = self.norm(hidden_states)
        return Step3p7ModelOutput(last_hidden_state=hidden_states)


def _build_rotary_embeddings(text_config):
    """Build the two RoPE tables (full-attention and sliding-attention) from per-layer config.

    ``rope_theta`` and ``partial_rotary_factors`` are per-layer lists in the checkpoint config,
    but they are constant within each attention type, so we read the value from a representative
    layer of each type. llama3 rope-scaling applies only to ``yarn_only_types`` (full attention).
    """
    layer_types = text_config.layer_types
    rope_theta = text_config.rope_theta
    partial_rotary_factors = getattr(text_config, "partial_rotary_factors", None)
    rope_scaling = getattr(text_config, "rope_scaling", None)
    yarn_only_types = getattr(text_config, "yarn_only_types", None)
    head_dim = text_config.head_dim
    max_pos = text_config.max_position_embeddings

    def _theta(idx):
        return rope_theta[idx] if isinstance(rope_theta, (list, tuple)) else rope_theta

    def _partial(idx):
        if partial_rotary_factors is not None:
            return partial_rotary_factors[idx]
        return getattr(text_config, "partial_rotary_factor", 1.0)

    def _scaling(layer_type):
        if rope_scaling is None:
            return None
        if yarn_only_types is not None and layer_type not in yarn_only_types:
            return None
        return rope_scaling

    def _rep_index(layer_type):
        return next(i for i, t in enumerate(layer_types) if t == layer_type)

    embeds = {}
    for layer_type in ("full_attention", "sliding_attention"):
        idx = _rep_index(layer_type)
        embeds[layer_type] = Step3p7RotaryEmbedding(
            head_dim=head_dim,
            partial_rotary_factor=_partial(idx),
            base=_theta(idx),
            max_position_embeddings=max_pos,
            rope_scaling=_scaling(layer_type),
        )
    return embeds["full_attention"], embeds["sliding_attention"]


class Step3p7ForCausalLM(Step3p7PreTrainedModel, GenerationMixin):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        text_config = _get_text_config(config)
        self.model = Step3p7TextModel(config)
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        # Load-time checkpoint adapters: absorb (1 + weight) RMSNorm convention and split stacked
        # MoE expert weights into per-expert Linear modules.
        self._register_load_state_dict_pre_hook(_step3p7_norm_weight_load_hook)
        self._register_load_state_dict_pre_hook(_step3p7_moe_split_load_hook)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Step3p7CausalLMOutput:
        assert position_ids is not None, "position_ids is required for AD export"
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        return Step3p7CausalLMOutput(logits=logits)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

AutoModelForCausalLMFactory.register_custom_model_cls("Step3p7Config", Step3p7ForCausalLM)
