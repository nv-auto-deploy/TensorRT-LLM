import importlib
import logging

_logger = logging.getLogger(__name__)

# Import each custom model individually so that models with transitive TRT-LLM
# dependencies (e.g., NemotronH needing mamba layernorm_gated) don't prevent
# other models from loading in standalone mode.
#
# NOTE: deepseek, nemotron_h, qwen3, qwen3_5_moe entries are the sharding-IR
# variants (see PR for #13429). The legacy non-IR versions of these four files
# were removed in the same change; the canonical name now points to the
# IR-aware implementation that uses ``apply_sharding_hints`` for TP/EP/BMM.
_MODEL_MODULES = {
    "modeling_deepseek": ["DeepSeekV3ForCausalLM"],
    "modeling_gemma3n": ["Gemma3nForCausalLM", "Gemma3nForConditionalGeneration"],
    "modeling_gemma4": ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"],
    "modeling_glm4_moe_lite": ["Glm4MoeLiteForCausalLM"],
    "modeling_kimi_k2": ["KimiK2ForCausalLM", "KimiK25ForConditionalGeneration"],
    "modeling_llama4": ["Llama4ForCausalLM", "Llama4ForConditionalGeneration"],
    "modeling_minimax_m2": ["MiniMaxM2ForCausalLM"],
    "modeling_mistral3": ["Mistral3ForConditionalGenerationAD", "Mistral4ForCausalLM"],
    "modeling_nemotron_flash": ["NemotronFlashForCausalLM", "NemotronFlashPreTrainedTokenizerFast"],
    "modeling_nemotron_h": ["NemotronHForCausalLM"],
    "modeling_qwen3": ["Qwen3ForCausalLM"],
    "modeling_qwen3_5_moe": ["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
    "modeling_qwen3_moe": ["Qwen3MoeForCausalLM"],
    "modeling_starcoder2": ["Starcoder2ForCausalLM"],
}

__all__ = []
for _module_name, _names in _MODEL_MODULES.items():
    try:
        _mod = importlib.import_module(f".{_module_name}", __name__)
        for _name in _names:
            globals()[_name] = getattr(_mod, _name)
            if _name not in __all__:
                __all__.append(_name)
    except (ImportError, ModuleNotFoundError, ValueError) as _exc:
        _logger.debug("Skipping custom model %s: %s", _module_name, _exc)

__all__ = tuple(__all__)
