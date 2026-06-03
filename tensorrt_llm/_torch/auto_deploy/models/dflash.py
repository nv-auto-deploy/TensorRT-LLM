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
"""DFlash one-model speculative-decoding factory for AutoDeploy.

Composes a target factory + a DFlash draft model into a ``DFlashWrapper`` (mirrors
``eagle.py::EagleOneModelFactory``). Registered as ``"dflash_one_model"``; the AD ``llm_args``
resolves a ``DFlashDecodingConfig`` to this factory (with the original ``model_factory`` becoming the
``target_model_factory``).

The generic ``TargetModelExportInfo`` is reused (it is target-agnostic export plumbing, per the design
summary §8); the draft export info is DFlash-specific (preserves ``fc`` + ``hidden_norm`` across export
via keepalive sentinels — they are used by the eager ``precompute_context_kv``, not the traced graph).
"""

import types
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch.nn as nn
from accelerate import init_empty_weights
from torch._prims_common import DeviceLikeType
from torch.export import Dim
from torch.fx import GraphModule
from transformers import AutoConfig

from .custom.modeling_dflash import (
    DFlashDrafterForCausalLM,
    DFlashModel,
    DFlashWrapper,
    DFlashWrapperConfig,
)
from .eagle import TargetModelExportInfo  # reuse generic target export-info plumbing (summary §8)
from .factory import DynamicShape, ModelFactory, ModelFactoryRegistry, SubModuleExportInfo
from .hf import insert_keepalive_sentinel


class DFlashDraftModelExportInfo(SubModuleExportInfo):
    """Export info for the DFlash draft model inside ``DFlashWrapper``.

    DFlash shares ``embed_tokens``/``lm_head`` from the target (the draft owns neither), so nothing is
    preserved for those. It DOES own ``fc`` + ``hidden_norm``, which are used by the eager
    ``precompute_context_kv`` (not the traced query-block forward), so they would be dropped by export
    graph-cleanup without keepalive sentinels. ``ctx_len`` is declared as a dynamic graph input so the
    cached-attention insertion can retrieve its placeholder.
    """

    def __init__(self):
        super().__init__("draft_model")

    def _init_dynamic_shape_lookup(self) -> Dict[str, DynamicShape]:
        batch_size_dyn = Dim.DYNAMIC
        # NOTE (DFlash-specific, differs from Eagle's draft): the DFlash draft ALWAYS runs at the
        # fixed trained ``block_size`` query width (non-causal fixed-width block -- never variable),
        # so the seq dim is STATIC, not dynamic. This is load-bearing: the ctx-K/V resource handler
        # reads ``block_size`` from ``q.shape[1]`` in ``get_cache_initializers`` to size the cache
        # slack; a dynamic seq dim makes that a SymInt and breaks ``torch.empty`` at allocation.
        return {
            "inputs_embeds": {0: batch_size_dyn},  # [B, block_size, H]; seq (dim 1) static
            "position_ids": {0: batch_size_dyn},  # [B, block_size]; seq (dim 1) static
            "ctx_len": {0: batch_size_dyn},  # [B] int32, one entry per request
        }

    def post_process(self, sub_mod: nn.Module, sub_gm: GraphModule):
        inner_model = sub_mod.model  # eager DFlashModel (full module structure + the method)
        sub_gm.is_draft = True
        inner_gm = sub_gm.get_submodule("model")  # the exported inner query-block GraphModule

        # The eager ``precompute_context_kv`` (Step 4/6) runs OUTSIDE the traced query-block graph, so
        # export drops BOTH the method and the nn.Modules it calls. Re-attach everything precompute
        # needs onto the inner GM (the proven Eagle ``set_submodule`` + keepalive pattern) and rebind
        # the method, so ``self.draft_model.model.precompute_context_kv(...)`` keeps working unchanged
        # post-export. ``fc``/``hidden_norm`` are precompute-ONLY (not in the traced graph) and so also
        # need keepalive sentinels to survive graph cleanup; ``rotary_emb`` (no persistent weight) and
        # the per-layer projections (already kept alive by the query-block attention's graph usage) do
        # not. Re-attached modules share their parameters with the graph (export clone=False), so this
        # is not a second weight copy.
        for name in ("fc", "hidden_norm", "rotary_emb", "layers"):
            mod = getattr(inner_model, name, None)
            if mod is not None:
                sub_gm.set_submodule(f"model.{name}", mod)
        for weight_attr in ("model.fc.weight", "model.hidden_norm.weight"):
            insert_keepalive_sentinel(sub_gm, weight_attr)

        # Rebind the eager precompute method onto the inner GM (``self`` -> ``inner_gm``); it now finds
        # fc/hidden_norm/rotary_emb/layers as attributes.
        inner_gm.precompute_context_kv = types.MethodType(
            DFlashModel.precompute_context_kv, inner_gm
        )


@ModelFactoryRegistry.register("dflash_one_model")
class DFlashOneModelFactory(ModelFactory):
    """Factory composing target + DFlash draft for one-model DFlash speculative decoding.

    Mirrors ``EagleOneModelFactory``: builds a target factory from ``target_model_factory`` and a
    DFlash draft model from ``speculative_config.speculative_model``, and wraps them in a
    ``DFlashWrapper``.
    """

    def __init__(
        self,
        model: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[str] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        skip_loading_weights: bool = False,
        max_seq_len: int = 512,
        speculative_config: Any = None,
        speculative_model_kwargs: Optional[Dict[str, Any]] = None,
        target_model_factory: str = "AutoModelForCausalLM",
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        if speculative_config is None:
            raise ValueError("speculative_config is required for DFlashOneModelFactory.")
        self.speculative_config = speculative_config
        self.sync_before_hidden_state_capture = kwargs.get(
            "sync_before_hidden_state_capture", False
        )
        if speculative_config.speculative_model is None:
            raise ValueError("speculative_config.speculative_model must be set.")
        self.draft_model_path = str(speculative_config.speculative_model)
        self.speculative_model_kwargs = speculative_model_kwargs or {}

        target_factory_cls = ModelFactoryRegistry.get(target_model_factory)
        self.target_factory = target_factory_cls(
            model=model,
            model_kwargs=model_kwargs,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            skip_loading_weights=skip_loading_weights,
            max_seq_len=max_seq_len,
        )

    @property
    def max_seq_len(self) -> int:
        return self.target_factory.max_seq_len

    @property
    def vocab_size_padded(self) -> Optional[int]:
        return self.target_factory.vocab_size_padded

    def _build_draft_config(self):
        draft_config = AutoConfig.from_pretrained(
            self.draft_model_path, trust_remote_code=True, **self.speculative_model_kwargs
        )
        return draft_config

    def _build_model(self, device: str) -> nn.Module:
        target_model = self.target_factory.build_model(device)

        draft_config = self._build_draft_config()
        # Build the draft in its config dtype (torch_dtype, overridable via speculative_model_kwargs)
        # so it matches the target; mirrors the Llama+Eagle dtype convention.
        draft_dtype = getattr(draft_config, "torch_dtype", None) or getattr(
            draft_config, "dtype", None
        )
        with (init_empty_weights if device == "meta" else nullcontext)():
            draft_model = DFlashDrafterForCausalLM(draft_config)
        if device == "meta":
            if hasattr(draft_model, "post_init"):
                draft_model.post_init()
            # HF's _from_config builds meta params already in config.dtype; our standalone module
            # builds in float32. Apply a dtype-only conversion (safe on meta -- no data copy) so the
            # draft's compute dtype matches both the non-meta path below and the bf16 checkpoint.
            if draft_dtype is not None:
                draft_model.to(dtype=draft_dtype)
        elif draft_dtype is not None:
            draft_model.to(device=device, dtype=draft_dtype)
        else:
            draft_model.to(device)
        draft_model.eval()

        dflash_cfg = getattr(draft_config, "dflash_config", {}) or {}
        block_size = getattr(draft_config, "block_size", None)
        if block_size is None:
            raise ValueError("DFlash draft config must define block_size.")
        # Validate the verify width fits the trained query block (PyTorch only silently clamps).
        verify_width = self.speculative_config.max_draft_len + 1
        if verify_width > block_size:
            raise ValueError(
                f"DFlash requires max_draft_len + 1 ({verify_width}) <= block_size ({block_size}); "
                f"got max_draft_len={self.speculative_config.max_draft_len}, block_size={block_size}."
            )
        mask_token_id = self.speculative_config.mask_token_id or dflash_cfg.get("mask_token_id")
        if mask_token_id is None:
            raise ValueError(
                "mask_token_id must be set on DFlashDecodingConfig or the draft config's dflash_config."
            )

        wrapper_config = DFlashWrapperConfig(
            max_draft_len=self.speculative_config.max_draft_len,
            block_size=block_size,
            mask_token_id=mask_token_id,
            sync_before_hidden_state_capture=self.sync_before_hidden_state_capture,
        )
        return DFlashWrapper(
            config=wrapper_config, target_model=target_model, draft_model=draft_model
        )

    def _load_draft_weights(self, draft_model: nn.Module, device: DeviceLikeType) -> None:
        """Load the DFlash draft checkpoint directly into ``draft_model``.

        Our standalone draft keeps SEPARATE q/k/v_proj (matching the z-lab checkpoint), so this is a
        direct state_dict load — no separate->fused qkv packing (AD fuses the q/k/v GEMMs downstream in
        the exported graph). Checkpoint keys (``fc.weight``, ``hidden_norm.weight``, ``layers.N...``,
        ``norm.weight``) map under the ``model.`` prefix. ``strict=True`` doubles as a fidelity check
        that the standalone modeling matches the checkpoint exactly (all 58 tensors).
        """
        import glob
        import os

        from safetensors.torch import load_file

        files = sorted(glob.glob(os.path.join(self.draft_model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(
                f"No *.safetensors found for the DFlash draft at {self.draft_model_path}."
            )
        state: Dict[str, Any] = {}
        for f in files:
            state.update(load_file(f, device=str(device)))
        # The draft wraps DFlashModel under ``model.``; checkpoint keys are unprefixed.
        remapped = {f"model.{k}": v for k, v in state.items()}
        draft_model.load_state_dict(remapped, strict=True)

    def _load_checkpoint(
        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
    ):
        """Load both the target checkpoint (via the target factory) and the DFlash draft checkpoint."""
        assert isinstance(model, DFlashWrapper), f"Expected DFlashWrapper, got {type(model)}"
        self.target_factory._load_checkpoint(model.target_model, device, disable_preload)
        self._load_draft_weights(model.draft_model, device)

    def load_or_random_init(
        self, model: nn.Module, device: DeviceLikeType, disable_preload: bool = False
    ):
        assert isinstance(model, DFlashWrapper), f"Expected DFlashWrapper, got {type(model)}"
        # Mirror EagleOneModelFactory.load_or_random_init: initialize BOTH submodels. The target has
        # its own factory; the DFlash draft has none (it is built inline in ``_build_model``), so we
        # run the base ``load_or_random_init`` logic for it here -- materialize any (meta) params on
        # ``device`` via ``_to_maybe_random``, then load the real draft checkpoint over them. Without
        # this, the draft stayed on ``meta`` and the subsequent ``move_to_device`` in the load_weights
        # transform raised "Cannot copy out of meta tensor; use to_empty()".
        self.target_factory.load_or_random_init(model.target_model, device, disable_preload)
        self._to_maybe_random(model.draft_model, device)
        if not self.skip_loading_weights:
            self.prefetch_checkpoint(force=True)
            self._load_draft_weights(model.draft_model, device)

    def get_export_infos(self, model: nn.Module) -> List[SubModuleExportInfo]:
        target_export_info = next(
            iter(self.target_factory.get_export_infos(model.target_model)), None
        )
        target_submodule_name = "target_model"
        if target_export_info is not None and target_export_info.submodule_name:
            target_submodule_name = f"target_model.{target_export_info.submodule_name}"

        return [
            TargetModelExportInfo(
                # DFlash takes lm_head from the target.
                load_lm_head_from_target=True,
                submodule_name=target_submodule_name,
                target_export_info=target_export_info,
            ),
            DFlashDraftModelExportInfo(),
        ]

    def get_sharding_config(self) -> Dict[str, Any]:
        return self.target_factory.get_sharding_config()

    def get_quant_config(self) -> Dict[str, Any]:
        return self.target_factory.get_quant_config()

    def get_cache_config_updates(self) -> Dict[str, Any]:
        return self.target_factory.get_cache_config_updates()

    def init_tokenizer(self) -> Optional[Any]:
        return self.target_factory.init_tokenizer()

    def init_processor(self) -> Optional[Any]:
        return self.target_factory.init_processor()

    def prefetch_checkpoint(self, force: bool = False, skip_loading_weights: Optional[bool] = None):
        self.target_factory.prefetch_checkpoint(force, skip_loading_weights)
        super().prefetch_checkpoint(force, skip_loading_weights)
