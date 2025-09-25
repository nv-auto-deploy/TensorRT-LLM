import contextlib
import types
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModelForCausalLM

from tensorrt_llm._torch.auto_deploy.models.patches.bamba import _bamba_mixer_torch_forward


def rmsnorm_patch(self, hidden_states, gate=None):
    return _rmsnorm_ref(
        x=hidden_states,
        weight=self.weight,
        z=gate,
        eps=self.variance_epsilon,
        group_size=self.group_size,
    )


# Forked from:
# https://github.com/state-spaces/mamba/blob/6b32be06d026e170b3fdaf3ae6282c5a6ff57b06/mamba_ssm/ops/triton/layernorm_gated.py
# NOTES:
# 1. At time of writing (09/25/2025), the nano nemotron v2 modeling code expects `mamba_ssm`
#    to be installed so as to be able to make use of its grouped gated RMS norm operation.
#    We therefore replace it with one that uses einops + pytorch.
# 2. Arguments / code paths unused by nemotron H have been removed for clarity.
def _rmsnorm_ref(x, weight, z=None, eps=1e-5, group_size=None):
    dtype = x.dtype
    weight = weight.float()
    # Always gate before normalizing.
    if z is not None:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = x * rstd * weight
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
    return out.to(dtype)


# The original implementation looks at `cache_position[0]` to decide what to do which does not
# play well with export. Plus, we do not want it to be updated anyway.
def _nemotron_h_model_update_mamba_mask(self, attention_mask, cache_position):
    return None


def _nemotron_h_model_update_causal_mask(self, attention_mask, input_tensor, cache_position):
    # Force attention to use causal mode without explicit masks
    return None


def _nemotron_h_block_forward(
    self,
    hidden_states,
    cache_params=None,
    cache_position: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    device = hidden_states.device
    with contextlib.ExitStack() as stack:
        if device.type == "cuda":
            stack.enter_context(torch.cuda.stream(torch.cuda.default_stream(device)))
        # * Use torch.cuda.stream() to avoid NaN issues when using multiple GPUs
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(
                hidden_states, cache_params=cache_params, cache_position=cache_position
            )
        elif self.block_type == "attention":
            hidden_states = self.mixer(hidden_states, cache_position=cache_position)
            hidden_states = hidden_states[0]
        elif self.block_type == "mlp":
            hidden_states = self.mixer(hidden_states)
        else:
            raise ValueError(f"Invalid block_type: {self.block_type}")

        hidden_states = residual + hidden_states
        return hidden_states


_from_config_original = AutoModelForCausalLM.from_config

CUSTOM_MODULE_PATCHES: Dict[str, List[Tuple[str, Callable]]] = {
    "MambaRMSNormGated": [("forward", rmsnorm_patch)],
    "NemotronHMamba2Mixer": [("forward", _bamba_mixer_torch_forward)],
    "NemotronHModel": [
        ("_update_causal_mask", _nemotron_h_model_update_causal_mask),
        ("_update_mamba_mask", _nemotron_h_model_update_mamba_mask),
    ],
    "NemotronHBlock": [("forward", _nemotron_h_block_forward)],
}


def get_model_from_config_patched(config, **kwargs):
    model = _from_config_original(config, **kwargs)
    # Patch modules
    for _, module in model.named_modules():
        if (module_name := type(module).__name__) in CUSTOM_MODULE_PATCHES.keys():
            patches = CUSTOM_MODULE_PATCHES[module_name]
            for method_name, method_patch in patches:
                setattr(module, method_name, types.MethodType(method_patch, module))

    return model


# TODO: figure out how this can be incorporated into the export patch system
AutoModelForCausalLM.from_config = get_model_from_config_patched
